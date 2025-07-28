"""Datenbank-Manager für die Neuromorphe Traum-Engine v2.0

Diese Datei verwaltet die Datenbankverbindungen und -operationen.
"""

import asyncio
from typing import Optional, AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, pool
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

from ..core.config import settings
from ..core.logging import get_logger
from .models import Base

# Settings und Logger
settings = settings
logger = get_logger(__name__)


class DatabaseManager:
    """Manager für Datenbankverbindungen und -operationen"""
    
    def __init__(self):
        self.settings = settings
        self.logger = get_logger(self.__class__.__name__)
        
        # Engines
        self._async_engine: Optional[AsyncEngine] = None
        self._sync_engine = None
        
        # Session Factories
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._sync_session_factory = None
        
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
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._sync_session_factory = sessionmaker(
                bind=self._sync_engine,
                expire_on_commit=False
            )
            
            # Event Listeners für Logging
            self._setup_event_listeners()
            
            self.logger.info("Database engines initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database engines: {e}", exc_info=True)
            raise
    
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
        """Datenbank-Health-Check"""
        health_status = {
            'status': 'unknown',
            'connection_test': False,
            'response_time_ms': None,
            'pool_status': {},
            'error': None
        }
        
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
async def create_tables():
    """Tabellen erstellen (Legacy-Kompatibilität)"""
    db_manager = get_database_manager()
    await db_manager.create_tables()
    logger.info("Database tables created")


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