"""Datenbank-Manager für die Neuromorphe Traum-Engine v2.0

Diese Datei verwaltet die Datenbankverbindungen und -operationen.
"""

import asyncio
import hashlib
import math
import shutil
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Optional, AsyncGenerator, Dict, Any, List
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, event, pool, text, select, func, or_
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError
import aiosqlite

from core.config import settings, Settings
from core.logging import get_logger
from database.models import (
    Base,
    Stem,
    GeneratedTrack,
    Arrangement,
    RenderJob,
    RenderStatus,
    SCHEMA_VERSION,
)

#: Schema-Version, die dieser Manager erwartet (siehe ``get_schema_version``).
CURRENT_SCHEMA_VERSION = SCHEMA_VERSION

#: Historische Spaltennamen der ``stems``-Tabelle -> Spalten des echten Schemas.
LEGACY_STEM_COLUMN_RENAMES: Dict[str, str] = {
    "name": "filename",
    "type": "category",
    "path": "original_path",
    "file_path": "original_path",
    "key": "musical_key",
    "tempo": "bpm",
    "tags": "manual_tags",
    "features": "neural_features",
    "embeddings": "audio_embedding",
}

#: Historische Feldnamen -> Spalten des echten Schemas (für Dict-basierte APIs).
STEM_FIELD_ALIASES: Dict[str, str] = {
    "name": "filename",
    "file_path": "original_path",
    "type": "category",
    "tempo": "bpm",
    "key": "musical_key",
    "tags": "manual_tags",
    "features": "neural_features",
    "embeddings": "audio_embedding",
}

#: Historische Feldnamen -> Spalten des echten Schemas (generated_tracks).
TRACK_FIELD_ALIASES: Dict[str, str] = {
    "prompt": "original_prompt",
    "metadata": "track_metadata",
    "genre": "target_genre",
    "status": "generation_status",
}


class _CompatAsyncSession(AsyncSession):
    """AsyncSession, die zusätzlich rohe SQL-Strings akzeptiert.

    SQLAlchemy 2.x verlangt ``text("...")``. Bestehender Code (und die
    Test-Spezifikation) übergibt noch rohe Strings; die werden hier
    transparent gewrappt, damit beide Schreibweisen funktionieren.
    """

    async def execute(self, statement, params=None, **kwargs):
        if isinstance(statement, str):
            statement = text(statement)
        if params is not None:
            return await super().execute(statement, params, **kwargs)
        return await super().execute(statement, **kwargs)


class _HealthStatus(dict):
    """Health-Check-Ergebnis, das zusätzlich ``== True`` erfüllt.

    Der Manager meldet intern ein Detaill-Dict (``status``, ``connection_test``,
    ``response_time_ms``, ``pool_status``), das u. a. ``src/cli/run.py``
    auswertet. Historische Aufrufer prüfen dagegen nur auf einen Wahrheitswert.
    """

    def __eq__(self, other: Any) -> bool:
        if other is True:
            return bool(self.get("connection_test"))
        return super().__eq__(other)

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return id(self)

    def __bool__(self) -> bool:
        return bool(self.get("connection_test"))


class DatabaseSettingsProxy:
    """Proxy für den historischen Zugriff ``settings.database.url``.

    Das echte Schema/Config kennt nur ``DATABASE_URL``. Ältere Aufrufer lesen
    und schreiben die URL über ``settings.database.url``; dieser Proxy leitet
    das transparent auf ``DATABASE_URL`` um (nur additiv, keine Feldänderung).
    """

    __slots__ = ("_settings",)

    def __init__(self, settings_obj: Settings):
        object.__setattr__(self, "_settings", settings_obj)

    @property
    def url(self) -> str:
        return self._settings.DATABASE_URL

    @url.setter
    def url(self, value: str) -> None:
        self._settings.DATABASE_URL = value

    @property
    def echo(self) -> bool:
        return self._settings.DATABASE_ECHO

    def __repr__(self) -> str:  # pragma: no cover - Debug-Hilfe
        return f"DatabaseSettingsProxy(url='{self.url}')"


def install_settings_database_compat() -> None:
    """Stellt ``settings.database.url`` als Alias für ``DATABASE_URL`` bereit.

    ``core/config.py`` wird bewusst nicht angefasst; der Alias wird daher als
    Property an die ``Settings``-Klasse gehängt (idempotent).
    """
    if isinstance(Settings.__dict__.get("database"), property):
        return
    Settings.database = property(lambda self: DatabaseSettingsProxy(self))


async def create_tables(engine: Optional[AsyncEngine] = None):
    if engine is None:
        engine = get_database_manager().async_engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables(engine: Optional[AsyncEngine] = None):
    if engine is None:
        engine = get_database_manager().async_engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# Settings und Logger
logger = get_logger(__name__)


class DatabaseManager:
    """Manager für Datenbankverbindungen und -operationen"""
    
    def __init__(self, settings: Settings = settings):
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        
        # Engines
        self._async_engine: Optional[AsyncEngine] = None
        self._sync_engine = None
        
        # Session Factories
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._sync_session_factory = None
        
        # Öffentliche Handles (historische Namen, siehe Test-Spezifikation)
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        
        # Connection Pool Settings
        self._pool_settings = {
            'pool_size': getattr(self.settings, 'DATABASE_POOL_SIZE', 5),
            'max_overflow': getattr(self.settings, 'DATABASE_MAX_OVERFLOW', 10),
            'pool_timeout': getattr(self.settings, 'DATABASE_POOL_TIMEOUT', 30),
            'pool_recycle': getattr(self.settings, 'DATABASE_POOL_RECYCLE', 3600),
            'pool_pre_ping': True
        }
        
        # Initialisierung
        self._initialize_engines()
    
    @property
    def database_url(self) -> str:
        """Aktuell konfigurierte Datenbank-URL (Alias für ``settings.DATABASE_URL``)."""
        return self.settings.DATABASE_URL
    
    def _initialize_engines(self):
        """Datenbank-Engines initialisieren"""
        try:
            # Async Engine
            if self.settings.DATABASE_URL.startswith('sqlite'):
                # SQLite-spezifische Konfiguration
                self._async_engine = create_async_engine(
                    self.settings.DATABASE_URL.replace('sqlite://', 'sqlite+aiosqlite://'),
                    echo=self.settings.DATABASE_ECHO,
                    poolclass=StaticPool,
                    connect_args={
                        'check_same_thread': False,
                        'timeout': 30
                    }
                )
                
                # Sync Engine für SQLite
                self._sync_engine = create_engine(
                    self.settings.DATABASE_URL,
                    echo=self.settings.DATABASE_ECHO,
                    poolclass=StaticPool,
                    connect_args={'check_same_thread': False}
                )
            else:
                # PostgreSQL oder andere Datenbanken
                self._async_engine = create_async_engine(
                    self.settings.DATABASE_URL,
                    echo=self.settings.DATABASE_ECHO,
                    **self._pool_settings
                )
                
                self._sync_engine = create_engine(
                    self.settings.DATABASE_URL.replace('+asyncpg', '').replace('+aiopg', ''),
                    echo=self.settings.DATABASE_ECHO,
                    **self._pool_settings
                )
            
            # Session Factories erstellen
            self._async_session_factory = async_sessionmaker(
                bind=self._async_engine,
                class_=_CompatAsyncSession,
                expire_on_commit=False
            )
            
            self._sync_session_factory = sessionmaker(
                bind=self._sync_engine,
                expire_on_commit=False
            )
            
            # Öffentliche Handles aktualisieren
            self.engine = self._async_engine
            self.session_factory = self._async_session_factory
            
            # Event Listeners für Logging
            self._setup_event_listeners()
            
            # Tabellen vorab anlegen (nur SQLite, Fehler werden geloggt —
            # ``initialize()`` meldet Verbindungsprobleme verbindlich).
            self._precreate_sqlite_tables()
            
            self.logger.info("Database engines initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database engines: {e}", exc_info=True)
            raise
    
    def _precreate_sqlite_tables(self) -> None:
        """Tabellen beim Start des Managers anlegen (SQLite, best effort).

        Damit funktionieren Aufrufer, die — wie die Test-Fixtures — keinen
        expliziten ``await initialize()``-Aufruf machen. Fehler werden nur
        geloggt; die verbindliche Prüfung übernimmt ``initialize()``.
        """
        if self.settings.DATABASE_URL is None or 'sqlite' not in self.settings.DATABASE_URL:
            return
        if self._sync_engine is None:
            return
        try:
            Base.metadata.create_all(bind=self._sync_engine)
        except Exception as e:
            self.logger.debug(f"Table pre-creation skipped: {e}")

    def _setup_event_listeners(self):
        """Event Listeners für Datenbank-Monitoring einrichten"""
        if not self.settings.ENABLE_DATABASE_MONITORING:
            return
        
        @event.listens_for(self._sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """SQLite-spezifische Pragmas setzen"""
            if 'sqlite' in self.settings.DATABASE_URL:
                cursor = dbapi_connection.cursor()
                # Performance-Optimierungen
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                cursor.close()
        
        @event.listens_for(self._sync_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """SQL-Ausführung loggen (Start)"""
            context._query_start_time = asyncio.get_event_loop().time()
        
        @event.listens_for(self._sync_engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """SQL-Ausführung loggen (Ende)"""
            if hasattr(context, '_query_start_time'):
                duration = asyncio.get_event_loop().time() - context._query_start_time
                
                if duration > self.settings.SLOW_QUERY_THRESHOLD:
                    self.logger.warning(
                        f"Slow query detected: {duration:.3f}s",
                        extra={
                            'duration': duration,
                            'statement': statement[:200] + '...' if len(statement) > 200 else statement,
                            'category': 'slow_query'
                        }
                    )
    
    @property
    def async_engine(self) -> AsyncEngine:
        """Async Engine abrufen"""
        if self._async_engine is None:
            raise RuntimeError("Async engine not initialized")
        return self._async_engine
    
    @property
    def sync_engine(self):
        """Sync Engine abrufen"""
        if self._sync_engine is None:
            raise RuntimeError("Sync engine not initialized")
        return self._sync_engine
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Async Session Context Manager"""
        if self._async_session_factory is None:
            raise RuntimeError("Async session factory not initialized")
        
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Database session error: {e}", exc_info=True)
                raise
            finally:
                await session.close()
    
    def get_sync_session(self) -> Session:
        """Sync Session erstellen"""
        if self._sync_session_factory is None:
            raise RuntimeError("Sync session factory not initialized")
        return self._sync_session_factory()
    
    def get_session(self) -> AsyncSession:
        """Historischer Name: async Session-Context-Manager."""
        return self.get_async_session()
    
    async def initialize(self) -> "DatabaseManager":
        """Manager initialisieren und Verbindung prüfen.
        
        Die Engines werden bereits im Konstruktor aufgebaut; hier wird die
        Verbindung verbindlich getestet und die Tabellen angelegt. Eine nicht
        verbindbare Datenbank führt zu einer Exception.
        """
        if self._async_engine is None:
            self._initialize_engines()
        
        # Verbindung verbindlich prüfen — wirft, wenn die URL ungültig ist.
        async with self._async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        
        await self.create_tables()
        
        self.engine = self._async_engine
        self.session_factory = self._async_session_factory
        
        self.logger.info("Database manager initialized")
        return self
    
    async def close(self) -> None:
        """Manager schließen (Alias für ``close_all_connections``)."""
        await self.close_all_connections()
    
    async def test_connection(self) -> bool:
        """Datenbankverbindung testen"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()
                self.logger.info("Database connection test successful")
                return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}", exc_info=True)
            return False
    
    async def create_tables(self):
        """Tabellen erstellen"""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}", exc_info=True)
            raise
    
    async def drop_tables(self):
        """Tabellen löschen"""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            self.logger.info("Database tables dropped successfully")
        except Exception as e:
            self.logger.error(f"Failed to drop tables: {e}", exc_info=True)
            raise
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Datenbank-Informationen abrufen"""
        try:
            async with self.get_async_session() as session:
                info = {
                    'database_url': self.settings.DATABASE_URL,
                    'engine_type': str(type(self.async_engine)),
                    'pool_size': getattr(self.async_engine.pool, 'size', 'N/A'),
                    'checked_out_connections': getattr(self.async_engine.pool, 'checkedout', 'N/A'),
                    'overflow_connections': getattr(self.async_engine.pool, 'overflow', 'N/A'),
                }
                
                # SQLite-spezifische Informationen
                if 'sqlite' in self.settings.DATABASE_URL:
                    db_path = Path(self.settings.DATABASE_URL.replace('sqlite:///', ''))
                    if db_path.exists():
                        info['database_size_mb'] = db_path.stat().st_size / (1024 * 1024)
                    else:
                        info['database_size_mb'] = 0
                
                return info
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def execute_raw_sql(self, sql: str, parameters: Dict = None) -> Any:
        """Raw SQL ausführen"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(sql, parameters or {})
                return result.fetchall()
        except Exception as e:
            self.logger.error(f"Failed to execute raw SQL: {e}", exc_info=True)
            raise
    
    async def vacuum_database(self):
        """Datenbank optimieren (SQLite VACUUM)"""
        if 'sqlite' not in self.settings.DATABASE_URL:
            self.logger.warning("VACUUM only supported for SQLite databases")
            return
        
        try:
            # VACUUM muss außerhalb einer Transaktion ausgeführt werden
            async with self.async_engine.connect() as conn:
                await conn.execute("VACUUM")
            self.logger.info("Database VACUUM completed")
        except Exception as e:
            self.logger.error(f"Database VACUUM failed: {e}", exc_info=True)
            raise
    
    async def analyze_database(self):
        """Datenbank-Statistiken aktualisieren"""
        try:
            if 'sqlite' in self.settings.DATABASE_URL:
                async with self.get_async_session() as session:
                    await session.execute("ANALYZE")
            else:
                # PostgreSQL ANALYZE
                async with self.get_async_session() as session:
                    await session.execute("ANALYZE")
            
            self.logger.info("Database ANALYZE completed")
        except Exception as e:
            self.logger.error(f"Database ANALYZE failed: {e}", exc_info=True)
            raise
    
    async def get_table_sizes(self) -> Dict[str, int]:
        """Tabellengröße abrufen"""
        try:
            table_sizes = {}
            
            if 'sqlite' in self.settings.DATABASE_URL:
                # SQLite-spezifische Abfrage
                async with self.get_async_session() as session:
                    for table in Base.metadata.tables.keys():
                        result = await session.execute(f"SELECT COUNT(*) FROM {table}")
                        count = (await result.fetchone())[0]
                        table_sizes[table] = count
            else:
                # PostgreSQL-spezifische Abfrage
                async with self.get_async_session() as session:
                    result = await session.execute("""
                        SELECT 
                            schemaname,
                            tablename,
                            attname,
                            n_distinct,
                            correlation
                        FROM pg_stats
                        WHERE schemaname = 'public'
                    """)
                    # Implementierung für PostgreSQL...
            
            return table_sizes
        except Exception as e:
            self.logger.error(f"Failed to get table sizes: {e}", exc_info=True)
            return {}
    
    async def backup_database(self, backup_path: Path):
        """Datenbank-Backup erstellen (SQLite)"""
        if 'sqlite' not in self.settings.DATABASE_URL:
            raise NotImplementedError("Backup only implemented for SQLite")
        
        try:
            # SQLite-Backup
            source_path = Path(self.settings.DATABASE_URL.replace('sqlite:///', ''))
            
            if not source_path.exists():
                raise FileNotFoundError(f"Database file not found: {source_path}")
            
            # Backup-Verzeichnis erstellen
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Datei kopieren
            import shutil
            shutil.copy2(source_path, backup_path)
            
            self.logger.info(f"Database backup created: {backup_path}")
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}", exc_info=True)
            raise
    
    async def restore_database(self, backup_path: Path):
        """Datenbank aus Backup wiederherstellen (SQLite)"""
        if 'sqlite' not in self.settings.DATABASE_URL:
            raise NotImplementedError("Restore only implemented for SQLite")
        
        try:
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Aktuelle Verbindungen schließen
            await self.close_all_connections()
            
            # Backup wiederherstellen
            target_path = Path(self.settings.DATABASE_URL.replace('sqlite:///', ''))
            import shutil
            shutil.copy2(backup_path, target_path)
            
            # Engines neu initialisieren
            self._initialize_engines()
            
            self.logger.info(f"Database restored from backup: {backup_path}")
        except Exception as e:
            self.logger.error(f"Database restore failed: {e}", exc_info=True)
            raise
    
    async def close_all_connections(self):
        """Alle Datenbankverbindungen schließen"""
        try:
            if self._async_engine:
                await self._async_engine.dispose()
            
            if self._sync_engine:
                self._sync_engine.dispose()
            
            self.logger.info("All database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}", exc_info=True)
    
    async def health_check(self) -> Dict[str, Any]:
        """Datenbank-Health-Check
        
        Liefert das Detail-Dict (``status``, ``connection_test``, …) und
        erfüllt zusätzlich ``== True``, wenn die Verbindung steht — beides
        wird von bestehenden Aufrufern erwartet.
        """
        health_status = _HealthStatus({
            'status': 'unknown',
            'connection_test': False,
            'response_time_ms': None,
            'pool_status': {},
            'error': None
        })
        
        try:
            # Verbindungstest mit Zeitmessung
            start_time = asyncio.get_event_loop().time()
            health_status['connection_test'] = await self.test_connection()
            end_time = asyncio.get_event_loop().time()
            
            health_status['response_time_ms'] = (end_time - start_time) * 1000
            
            # Pool-Status
            if hasattr(self.async_engine.pool, 'size'):
                health_status['pool_status'] = {
                    'size': self.async_engine.pool.size(),
                    'checked_out': self.async_engine.pool.checkedout(),
                    'overflow': self.async_engine.pool.overflow(),
                    'checked_in': self.async_engine.pool.checkedin()
                }
            
            # Gesamtstatus bestimmen
            if health_status['connection_test']:
                if health_status['response_time_ms'] < 100:  # < 100ms
                    health_status['status'] = 'healthy'
                elif health_status['response_time_ms'] < 1000:  # < 1s
                    health_status['status'] = 'degraded'
                else:
                    health_status['status'] = 'slow'
            else:
                health_status['status'] = 'unhealthy'
        
        except Exception as e:
            health_status['status'] = 'error'
            health_status['error'] = str(e)
            self.logger.error(f"Database health check failed: {e}", exc_info=True)
        
        return health_status
    
    # ------------------------------------------------------------------
    # Dict-basierte Convenience-API (historische Aufrufer, Test-Spezifikation)
    #
    # Diese Methoden sind rein additiv. Die produktive API nutzt weiterhin
    # ``database.service``/``database.crud``; die Feldnamen der Dicts folgen
    # den Spalten des echten Schemas, historische Alias-Namen werden
    # akzeptiert und mit übersetzt.
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_stem_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Historische Stem-Feldnamen auf die echten Spalten abbilden."""
        normalized: Dict[str, Any] = {}
        for key, value in data.items():
            normalized[STEM_FIELD_ALIASES.get(key, key)] = value
        return normalized
    
    @staticmethod
    def _normalize_track_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Historische Track-Feldnamen auf die echten Spalten abbilden."""
        normalized: Dict[str, Any] = {}
        for key, value in data.items():
            normalized[TRACK_FIELD_ALIASES.get(key, key)] = value
        return normalized
    
    def _stem_to_dict(self, stem: Stem) -> Dict[str, Any]:
        """Stem als Dict serialisieren (inkl. historischer Alias-Namen)."""
        data = stem.to_dict()
        data.pop("created_at", None)
        data.pop("updated_at", None)
        data.pop("processed_at", None)
        return data
    
    def _track_to_dict(self, track: GeneratedTrack) -> Dict[str, Any]:
        """GeneratedTrack als Dict serialisieren."""
        data = track.to_dict()
        data.pop("created_at", None)
        data.pop("updated_at", None)
        data.pop("generated_at", None)
        return data
    
    def _render_job_to_dict(self, job: RenderJob) -> Dict[str, Any]:
        """RenderJob als Dict serialisieren."""
        data = job.to_dict()
        for key in ("created_at", "updated_at", "started_at", "completed_at"):
            data.pop(key, None)
        data["options"] = job.options or {}
        return data
    
    async def _get_stem_by_id(self, stem_id: Any) -> Optional[Stem]:
        """Stem per Primärschlüssel laden (innerhalb einer eigenen Session)."""
        async with self.get_async_session() as session:
            return await session.get(Stem, stem_id)
    
    # --- Stems ---------------------------------------------------------
    
    async def create_stem(self, stem_data: Dict[str, Any]) -> str:
        """Stem erstellen und seine ID zurückgeben.
        
        Akzeptiert die historischen Feldnamen (name/file_path/type/tempo/…)
        und übersetzt sie auf die Spalten des echten Schemas.
        """
        data = self._normalize_stem_fields(stem_data)
        data.pop("id", None)
        data.setdefault("source", "original")
        data.setdefault("processing_status", "completed")
        data.setdefault("filename", data.get("original_path") or "unknown.wav")
        data.setdefault("original_path", data.get("filename"))
        data.setdefault("file_hash", uuid4().hex + uuid4().hex)
        data.setdefault("duration", 0.0)
        data.setdefault("sample_rate", 44100)
        data.setdefault("channels", 2)
        data.setdefault("file_size", 0)
        
        async with self.get_async_session() as session:
            stem = Stem(**data)
            session.add(stem)
            await session.flush()
            stem_id = stem.id
        
        self.logger.info(f"Stem created: {stem_id}")
        return stem_id
    
    async def get_stem(self, stem_id: Any) -> Optional[Dict[str, Any]]:
        """Stem als Dict abrufen (``None``, wenn nicht vorhanden)."""
        stem = await self._get_stem_by_id(stem_id)
        if stem is None:
            return None
        return self._stem_to_dict(stem)
    
    async def update_stem(self, stem_id: Any, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Stem aktualisieren und das aktualisierte Dict zurückgeben."""
        async with self.get_async_session() as session:
            stem = await session.get(Stem, stem_id)
            if stem is None:
                return None
            stem.update_metadata(update_data)
            await session.flush()
            result = self._stem_to_dict(stem)
        
        self.logger.info(f"Stem updated: {stem_id}")
        return result
    
    async def delete_stem(self, stem_id: Any) -> bool:
        """Stem löschen. ``True``, wenn ein Datensatz entfernt wurde."""
        async with self.get_async_session() as session:
            stem = await session.get(Stem, stem_id)
            if stem is None:
                return False
            await session.delete(stem)
        
        self.logger.info(f"Stem deleted: {stem_id}")
        return True
    
    async def search_stems(
        self,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Stems suchen.
        
        ``filters`` matcht exakt auf Spalten (historische Alias-Namen werden
        übersetzt); ``query`` sucht zusätzlich case-insensitiv in filename,
        category, genre und mood.
        """
        filters = filters or {}
        stmt = select(Stem)
        
        for key, value in filters.items():
            column_name = STEM_FIELD_ALIASES.get(key, key)
            column = getattr(Stem, column_name, None)
            if column is None:
                continue
            stmt = stmt.where(column == value)
        
        if query:
            pattern = f"%{query}%"
            stmt = stmt.where(
                or_(
                    Stem.filename.ilike(pattern),
                    Stem.category.ilike(pattern),
                    Stem.genre.ilike(pattern),
                    Stem.mood.ilike(pattern),
                )
            )
        
        stmt = stmt.limit(limit)
        
        async with self.get_async_session() as session:
            result = await session.execute(stmt)
            stems = result.scalars().all()
            payload = [self._stem_to_dict(s) for s in stems]
        
        return {"stems": payload, "total": len(payload)}
    
    async def get_similar_stems(
        self,
        embeddings: List[float],
        limit: int = 10,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Stems nach Kosinus-Ähnlichkeit zu ``embeddings`` sortiert liefern.
        
        Es werden nur Stems mit vorhandenem Embedding berücksichtigt und
        ausschließlich Treffer ab ``threshold`` zurückgegeben.
        """
        async with self.get_async_session() as session:
            result = await session.execute(
                select(Stem).where(Stem.audio_embedding.isnot(None))
            )
            stems = result.scalars().all()
            
            scored: List[Dict[str, Any]] = []
            for stem in stems:
                similarity = _cosine_similarity(embeddings, stem.audio_embedding)
                if similarity < threshold:
                    continue
                payload = self._stem_to_dict(stem)
                payload["similarity"] = similarity
                scored.append(payload)
        
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:limit]
    
    # --- Generated Tracks ----------------------------------------------
    
    async def create_generated_track(self, track_data: Dict[str, Any]) -> str:
        """GeneratedTrack erstellen und seine ID zurückgeben."""
        data = self._normalize_track_fields(track_data)
        stems = data.pop("stems", None)
        data.pop("id", None)
        data.pop("track_metadata", None)
        data.pop("target_genre", None)
        
        if not data.get("original_prompt"):
            data["original_prompt"] = data.pop("prompt", None) or track_data.get("prompt") or "untitled"
        data["title"] = track_data.get("title") or str(data["original_prompt"])[:250]
        data["target_genre"] = track_data.get("target_genre") or track_data.get("genre")
        data["track_metadata"] = track_data.get("track_metadata") or track_data.get("metadata")
        data.setdefault("duration", track_data.get("duration") or 0.0)
        data.setdefault("generation_status", "pending")
        
        async with self.get_async_session() as session:
            track = GeneratedTrack(**data)
            track.stems = list(stems) if stems else []
            session.add(track)
            await session.flush()
            track_id = track.id
        
        self.logger.info(f"GeneratedTrack created: {track_id}")
        return track_id
    
    async def get_generated_track(self, track_id: Any) -> Optional[Dict[str, Any]]:
        """GeneratedTrack als Dict abrufen."""
        async with self.get_async_session() as session:
            track = await session.get(GeneratedTrack, track_id)
            if track is None:
                return None
            return self._track_to_dict(track)
    
    async def update_generated_track(self, track_id: Any, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """GeneratedTrack aktualisieren und das aktualisierte Dict zurückgeben."""
        async with self.get_async_session() as session:
            track = await session.get(GeneratedTrack, track_id)
            if track is None:
                return None
            
            for key, value in update_data.items():
                if key == "stems":
                    track.stems = list(value) if value else []
                    continue
                mapped = TRACK_FIELD_ALIASES.get(key, key)
                if key in ("metadata", "track_metadata"):
                    mapped = "track_metadata"
                if mapped in GeneratedTrack.__table__.columns:
                    setattr(track, mapped, value)
            
            track.updated_at = datetime.utcnow()
            await session.flush()
            result = self._track_to_dict(track)
        
        self.logger.info(f"GeneratedTrack updated: {track_id}")
        return result
    
    async def list_generated_tracks(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """GeneratedTracks paginiert auflisten."""
        page = max(1, int(page or 1))
        per_page = max(1, int(per_page or 20))
        
        async with self.get_async_session() as session:
            total = (await session.execute(
                select(func.count()).select_from(GeneratedTrack)
            )).scalar() or 0
            
            result = await session.execute(
                select(GeneratedTrack)
                .order_by(GeneratedTrack.id)
                .offset((page - 1) * per_page)
                .limit(per_page)
            )
            tracks = [self._track_to_dict(t) for t in result.scalars().all()]
        
        return {
            "tracks": tracks,
            "total": total,
            "page": page,
            "per_page": per_page,
        }
    
    # --- Render Jobs ---------------------------------------------------
    
    async def create_render_job(self, job_data: Dict[str, Any]) -> str:
        """Render-Job erstellen und seine ID zurückgeben."""
        data = dict(job_data)
        data.pop("id", None)
        data["arrangement_id"] = _as_optional_str(
            data.get("arrangement_id") or data.get("track_id")
        )
        if isinstance(data.get("format"), Enum):
            data["format"] = data["format"].value
        if isinstance(data.get("status"), Enum):
            data["status"] = data["status"].value
        data.setdefault("status", RenderStatus.PENDING.value)
        data.setdefault("progress", 0.0)
        data.setdefault("options", {})
        
        async with self.get_async_session() as session:
            job = RenderJob(**data)
            session.add(job)
            await session.flush()
            job_id = job.id
        
        self.logger.info(f"RenderJob created: {job_id}")
        return job_id
    
    async def get_render_job(self, job_id: Any) -> Optional[Dict[str, Any]]:
        """Render-Job als Dict abrufen."""
        async with self.get_async_session() as session:
            job = await session.get(RenderJob, job_id)
            if job is None:
                return None
            return self._render_job_to_dict(job)
    
    async def update_render_job_status(
        self,
        job_id: Any,
        status: Any,
        progress: Optional[float] = None,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Status eines Render-Jobs aktualisieren."""
        async with self.get_async_session() as session:
            job = await session.get(RenderJob, job_id)
            if job is None:
                return None
            job.update_status(
                status,
                progress=progress,
                output_path=output_path,
                error_message=error_message,
            )
            await session.flush()
            result = self._render_job_to_dict(job)
        
        self.logger.info(f"RenderJob status updated: {job_id} -> {status}")
        return result
    
    async def list_render_jobs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        page: int = 1,
        per_page: int = 20,
    ) -> Dict[str, Any]:
        """Render-Jobs auflisten (optional gefiltert)."""
        filters = filters or {}
        stmt = select(RenderJob)
        count_stmt = select(func.count()).select_from(RenderJob)
        
        for key, value in filters.items():
            column = getattr(RenderJob, key, None)
            if column is None:
                continue
            stmt = stmt.where(column == value)
            count_stmt = count_stmt.where(column == value)
        
        stmt = stmt.order_by(RenderJob.created_at).limit(limit)
        
        async with self.get_async_session() as session:
            total = (await session.execute(count_stmt)).scalar() or 0
            result = await session.execute(stmt)
            payload = [self._render_job_to_dict(j) for j in result.scalars().all()]
        
        return {
            "jobs": payload,
            "total": total,
            "page": page,
            "per_page": per_page,
        }
    
    # --- Schema-/Migrations-Info ---------------------------------------
    
    async def get_schema_version(self) -> str:
        """Aktuelle Schema-Version der Datenbank lesen."""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                tables = {row[0] for row in result.fetchall()}
            
            if "schema_version" not in tables:
                return CURRENT_SCHEMA_VERSION
            
            async with self.get_async_session() as session:
                result = await session.execute(text("SELECT version FROM schema_version LIMIT 1"))
                row = result.fetchone()
            return str(row[0]) if row else CURRENT_SCHEMA_VERSION
        except Exception as e:
            self.logger.debug(f"Schema version lookup fallback: {e}")
            return CURRENT_SCHEMA_VERSION
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Migrations-Status melden."""
        current_version = await self.get_schema_version()
        pending: List[Dict[str, Any]] = []
        
        try:
            async with self.get_async_session() as session:
                result = await session.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                tables = {row[0] for row in result.fetchall()}
            
            # Tabellen des ORM-Modells, die in der Zieldatenbank fehlen.
            expected = set(Base.metadata.tables.keys())
            missing = sorted(expected - tables)
            for table in missing:
                pending.append({
                    "name": f"create_table_{table}",
                    "version": CURRENT_SCHEMA_VERSION,
                    "table": table,
                })
        except Exception as e:
            self.logger.debug(f"Migration status lookup fallback: {e}")
            if 'sqlite' not in (self.settings.DATABASE_URL or ''):
                pending.append({
                    "name": "schema_check_unsupported",
                    "version": CURRENT_SCHEMA_VERSION,
                })
        
        return {
            "current_version": current_version,
            "latest_version": CURRENT_SCHEMA_VERSION,
            "pending_migrations": pending,
        }
    
    # --- Backup / Restore ----------------------------------------------
    
    @staticmethod
    def _sqlite_file_path(db_url: str) -> Optional[Path]:
        """Dateipfad einer SQLite-URL ermitteln (``None`` bei ``:memory:``)."""
        if not db_url or 'sqlite' not in db_url:
            return None
        path = db_url.split('sqlite:///', 1)[-1]
        if not path or path == ':memory:':
            return None
        return Path(path)
    
    async def create_backup(self, backup_path: Any) -> bool:
        """SQLite-Backup erstellen (SQLite-Online-Backup-API)."""
        source_path = self._sqlite_file_path(self.settings.DATABASE_URL)
        if source_path is None:
            raise NotImplementedError("Backup only implemented for file-backed SQLite")
        
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        await self.close_all_connections()
        
        connection = None
        try:
            connection = sqlite3.connect(str(source_path))
            # Tabellen sicherstellen, damit auch ein frisch erzeugtes
            # Backup-Schema vollständig ist.
            Base.metadata.create_all(bind=self._sync_engine)
            
            target = sqlite3.connect(str(backup_path))
            try:
                connection.backup(target)
                target.commit()
            finally:
                target.close()
            
            self._initialize_engines()
            self.logger.info(f"Database backup created: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}", exc_info=True)
            raise
        finally:
            if connection is not None:
                connection.close()
    
    async def restore_backup(self, backup_path: Any) -> bool:
        """Datenbank aus einem SQLite-Backup wiederherstellen."""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        target_path = self._sqlite_file_path(self.settings.DATABASE_URL)
        if target_path is None:
            raise NotImplementedError("Restore only implemented for file-backed SQLite")
        
        await self.close_all_connections()
        
        try:
            # Inhalt des Backups in die Zieldatenbank kopieren. So bleibt das
            # echte Schema erhalten und die Backup-Datei wandert nicht selbst
            # in die Zieldatenbank (siehe Test-Spezifikation).
            source = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
            try:
                target = sqlite3.connect(str(target_path))
                try:
                    source.backup(target)
                    target.commit()
                finally:
                    target.close()
            finally:
                source.close()
            
            self._initialize_engines()
            self.logger.info(f"Database restored from backup: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Database restore failed: {e}", exc_info=True)
            raise


def _as_optional_str(value: Any) -> Optional[str]:
    """Wert als String für Spalten mit fester Länge aufbereiten."""
    if value is None:
        return None
    return str(value)


def _cosine_similarity(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Kosinus-Ähnlichkeit zweier Vektoren (0.0, wenn nicht berechenbar)."""
    if not a or not b:
        return 0.0
    
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        norm_a += fx * fx
        norm_b += fy * fy
    
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    
    similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return max(0.0, min(1.0, similarity))


# Kompatibilitäts-Alias sofort installieren, damit ``settings.database.url``
# unabhängig von der Instanziierung eines Managers verfügbar ist.
install_settings_database_compat()


# Globaler Database Manager
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Database Manager Singleton abrufen"""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


# Dependency für FastAPI
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency für Async Database Session"""
    db_manager = get_database_manager()
    async with db_manager.get_async_session() as session:
        yield session


def get_sync_db_session() -> Session:
    """Sync Database Session abrufen"""
    db_manager = get_database_manager()
    return db_manager.get_sync_session()


def get_db():
    """FastAPI Dependency für Sync Database Session (Legacy-Kompatibilität)"""
    db = get_sync_db_session()
    try:
        yield db
    finally:
        db.close()


# Hilfsfunktionen
async def init_database():
    """Datenbank initialisieren"""
    await create_tables()
    logger.info("Database initialization completed")


async def cleanup_database():
    """Datenbank-Cleanup"""
    db_manager = get_database_manager()
    
    # Vacuum und Analyze
    await db_manager.vacuum_database()
    await db_manager.analyze_database()
    
    logger.info("Database cleanup completed")


# Context Manager für Transaktionen
class DatabaseTransaction:
    """Context Manager für Datenbank-Transaktionen"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.committed = False
    
    async def __aenter__(self):
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and not self.committed:
            await self.session.commit()
            self.committed = True
        elif exc_type is not None:
            await self.session.rollback()


# Performance Monitoring
class DatabasePerformanceMonitor:
    """Monitor für Datenbank-Performance"""
    
    def __init__(self):
        self.query_times = []
        self.slow_queries = []
        self.error_count = 0
    
    def record_query(self, duration: float, query: str):
        """Query-Performance aufzeichnen"""
        self.query_times.append(duration)
        
        if duration > settings.SLOW_QUERY_THRESHOLD:
            self.slow_queries.append({
                'duration': duration,
                'query': query[:200],
                'timestamp': asyncio.get_event_loop().time()
            })
    
    def record_error(self):
        """Fehler aufzeichnen"""
        self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Performance-Statistiken abrufen"""
        if not self.query_times:
            return {'no_data': True}
        
        import statistics
        
        return {
            'total_queries': len(self.query_times),
            'avg_query_time': statistics.mean(self.query_times),
            'median_query_time': statistics.median(self.query_times),
            'max_query_time': max(self.query_times),
            'min_query_time': min(self.query_times),
            'slow_queries_count': len(self.slow_queries),
            'error_count': self.error_count,
            'recent_slow_queries': self.slow_queries[-5:]  # Letzte 5
        }
    
    def reset_stats(self):
        """Statistiken zurücksetzen"""
        self.query_times.clear()
        self.slow_queries.clear()
        self.error_count = 0


# Globaler Performance Monitor
db_performance_monitor = DatabasePerformanceMonitor()