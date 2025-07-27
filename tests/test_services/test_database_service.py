"""Tests für Database-Service"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

from src.services.database_service import (
    DatabaseService, DatabaseConnection, DatabaseTransaction,
    QueryBuilder, DatabaseMigration, DatabaseBackup,
    ConnectionPool, QueryOptimizer, IndexManager,
    DatabaseMetrics, DatabaseHealth
)
from src.core.config import DatabaseConfig
from src.core.exceptions import (
    DatabaseError, ConnectionError, TransactionError,
    ValidationError, ConfigurationError
)
from src.database.models import Stem, Arrangement, RenderJob
from src.schemas.stem import StemCreate, StemUpdate, StemResponse
from src.schemas.arrangement import ArrangementCreate, ArrangementResponse
from src.schemas.render import RenderJobCreate, RenderJobResponse


class TestDatabaseService:
    """Tests für Database-Service"""
    
    @pytest.fixture
    def database_config(self):
        """Datenbank-Konfiguration für Tests"""
        return DatabaseConfig(
            host="localhost",
            port=5432,
            database="neuromorphe_test",
            username="test_user",
            password="test_password",
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False,
            ssl_mode="disable",
            connection_timeout=10,
            query_timeout=30,
            enable_migrations=True,
            backup_enabled=True,
            backup_interval=3600
        )
    
    @pytest.fixture
    def mock_connection_pool(self):
        """Mock Connection Pool"""
        pool = Mock()
        pool.acquire = AsyncMock()
        pool.release = AsyncMock()
        pool.close = AsyncMock()
        pool.get_stats = Mock()
        return pool
    
    @pytest.fixture
    def database_service(self, database_config, mock_connection_pool):
        """Database-Service für Tests"""
        service = DatabaseService(database_config)
        service.connection_pool = mock_connection_pool
        return service
    
    @pytest.mark.unit
    def test_database_service_initialization(self, database_config):
        """Test: Database-Service-Initialisierung"""
        service = DatabaseService(database_config)
        
        assert service.config == database_config
        assert service.host == "localhost"
        assert service.port == 5432
        assert service.database == "neuromorphe_test"
        assert isinstance(service.query_builder, QueryBuilder)
        assert isinstance(service.migration_manager, DatabaseMigration)
    
    @pytest.mark.unit
    def test_database_service_invalid_config(self):
        """Test: Database-Service mit ungültiger Konfiguration"""
        invalid_config = DatabaseConfig(
            host="",  # Leer
            port=0,   # Ungültig
            database="",
            username="",
            password=""
        )
        
        with pytest.raises(ConfigurationError):
            DatabaseService(invalid_config)
    
    @pytest.mark.unit
    async def test_connect_to_database(self, database_service):
        """Test: Datenbankverbindung herstellen"""
        # Mock erfolgreiche Verbindung
        mock_connection = Mock()
        database_service.connection_pool.acquire.return_value = mock_connection
        
        connection = await database_service.connect()
        
        assert connection == mock_connection
        database_service.connection_pool.acquire.assert_called_once()
    
    @pytest.mark.unit
    async def test_connect_to_database_failure(self, database_service):
        """Test: Datenbankverbindung fehlgeschlagen"""
        # Mock Verbindungsfehler
        database_service.connection_pool.acquire.side_effect = ConnectionError(
            "Could not connect to database"
        )
        
        with pytest.raises(ConnectionError):
            await database_service.connect()
    
    @pytest.mark.unit
    async def test_disconnect_from_database(self, database_service):
        """Test: Datenbankverbindung trennen"""
        mock_connection = Mock()
        
        await database_service.disconnect(mock_connection)
        
        database_service.connection_pool.release.assert_called_once_with(mock_connection)
    
    @pytest.mark.unit
    async def test_execute_query(self, database_service):
        """Test: Query ausführen"""
        query = "SELECT * FROM stems WHERE id = $1"
        params = ["stem_123"]
        
        # Mock Connection und Result
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            {"id": "stem_123", "name": "Test Stem", "duration": 30.0}
        ]
        
        database_service.connection_pool.acquire.return_value = mock_connection
        mock_connection.execute.return_value = mock_result
        
        result = await database_service.execute_query(query, params)
        
        assert len(result) == 1
        assert result[0]["id"] == "stem_123"
        mock_connection.execute.assert_called_once_with(query, *params)
    
    @pytest.mark.unit
    async def test_execute_query_with_timeout(self, database_service):
        """Test: Query mit Timeout ausführen"""
        query = "SELECT * FROM stems"
        
        # Mock langsame Query
        mock_connection = Mock()
        database_service.connection_pool.acquire.return_value = mock_connection
        
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(2)  # 2 Sekunden Verzögerung
            return Mock()
        
        mock_connection.execute = slow_execute
        
        with pytest.raises(DatabaseError):
            await database_service.execute_query(query, timeout=1)
    
    @pytest.mark.unit
    async def test_create_stem(self, database_service):
        """Test: Stem erstellen"""
        stem_data = StemCreate(
            name="Test Stem",
            file_path="/audio/test.wav",
            duration=30.0,
            sample_rate=48000,
            tags=["test", "audio"],
            metadata={"artist": "Test Artist"}
        )
        
        # Mock Datenbank-Antwort
        mock_stem = Stem(
            id="stem_123",
            name="Test Stem",
            file_path="/audio/test.wav",
            duration=30.0,
            sample_rate=48000,
            created_at=datetime.now()
        )
        
        with patch.object(database_service, '_insert_record') as mock_insert:
            mock_insert.return_value = mock_stem
            
            stem = await database_service.create_stem(stem_data)
            
            assert isinstance(stem, Stem)
            assert stem.name == "Test Stem"
            assert stem.duration == 30.0
            mock_insert.assert_called_once()
    
    @pytest.mark.unit
    async def test_get_stem(self, database_service):
        """Test: Stem abrufen"""
        stem_id = "stem_123"
        
        # Mock Datenbank-Antwort
        mock_stem = Stem(
            id=stem_id,
            name="Test Stem",
            file_path="/audio/test.wav",
            duration=30.0,
            sample_rate=48000
        )
        
        with patch.object(database_service, '_get_record_by_id') as mock_get:
            mock_get.return_value = mock_stem
            
            stem = await database_service.get_stem(stem_id)
            
            assert isinstance(stem, Stem)
            assert stem.id == stem_id
            mock_get.assert_called_once_with(Stem, stem_id)
    
    @pytest.mark.unit
    async def test_get_stem_not_found(self, database_service):
        """Test: Stem nicht gefunden"""
        stem_id = "nonexistent_stem"
        
        with patch.object(database_service, '_get_record_by_id') as mock_get:
            mock_get.return_value = None
            
            stem = await database_service.get_stem(stem_id)
            
            assert stem is None
    
    @pytest.mark.unit
    async def test_update_stem(self, database_service):
        """Test: Stem aktualisieren"""
        stem_id = "stem_123"
        update_data = StemUpdate(
            name="Updated Stem",
            tags=["updated", "test"]
        )
        
        # Mock existierender Stem
        existing_stem = Stem(
            id=stem_id,
            name="Test Stem",
            file_path="/audio/test.wav",
            duration=30.0
        )
        
        # Mock aktualisierter Stem
        updated_stem = Stem(
            id=stem_id,
            name="Updated Stem",
            file_path="/audio/test.wav",
            duration=30.0
        )
        
        with patch.object(database_service, '_get_record_by_id') as mock_get:
            mock_get.return_value = existing_stem
            
            with patch.object(database_service, '_update_record') as mock_update:
                mock_update.return_value = updated_stem
                
                stem = await database_service.update_stem(stem_id, update_data)
                
                assert stem.name == "Updated Stem"
                mock_update.assert_called_once()
    
    @pytest.mark.unit
    async def test_delete_stem(self, database_service):
        """Test: Stem löschen"""
        stem_id = "stem_123"
        
        # Mock existierender Stem
        existing_stem = Stem(
            id=stem_id,
            name="Test Stem",
            file_path="/audio/test.wav"
        )
        
        with patch.object(database_service, '_get_record_by_id') as mock_get:
            mock_get.return_value = existing_stem
            
            with patch.object(database_service, '_delete_record') as mock_delete:
                mock_delete.return_value = True
                
                success = await database_service.delete_stem(stem_id)
                
                assert success == True
                mock_delete.assert_called_once_with(Stem, stem_id)
    
    @pytest.mark.unit
    async def test_list_stems(self, database_service):
        """Test: Stems auflisten"""
        # Mock Stems
        mock_stems = [
            Stem(
                id="stem_1",
                name="Stem 1",
                duration=30.0,
                created_at=datetime.now() - timedelta(hours=1)
            ),
            Stem(
                id="stem_2",
                name="Stem 2",
                duration=45.0,
                created_at=datetime.now() - timedelta(minutes=30)
            )
        ]
        
        with patch.object(database_service, '_list_records') as mock_list:
            mock_list.return_value = mock_stems
            
            stems = await database_service.list_stems(
                limit=10,
                offset=0,
                filters={"duration_min": 20.0}
            )
            
            assert len(stems) == 2
            assert all(isinstance(stem, Stem) for stem in stems)
            mock_list.assert_called_once()
    
    @pytest.mark.unit
    async def test_search_stems(self, database_service):
        """Test: Stems suchen"""
        search_query = "test audio"
        
        # Mock Suchergebnisse
        mock_results = [
            Stem(
                id="stem_1",
                name="Test Audio Stem",
                tags=["test", "audio"],
                duration=30.0
            )
        ]
        
        with patch.object(database_service, '_search_records') as mock_search:
            mock_search.return_value = mock_results
            
            results = await database_service.search_stems(
                query=search_query,
                limit=10
            )
            
            assert len(results) == 1
            assert "test" in results[0].name.lower()
            mock_search.assert_called_once()


class TestDatabaseTransaction:
    """Tests für Database-Transaction"""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock Datenbankverbindung"""
        connection = Mock()
        connection.begin = AsyncMock()
        connection.commit = AsyncMock()
        connection.rollback = AsyncMock()
        connection.execute = AsyncMock()
        return connection
    
    @pytest.fixture
    def database_transaction(self, mock_connection):
        """Database-Transaction für Tests"""
        return DatabaseTransaction(mock_connection)
    
    @pytest.mark.unit
    async def test_transaction_commit(self, database_transaction, mock_connection):
        """Test: Transaktion erfolgreich committen"""
        async with database_transaction:
            # Simuliere Datenbankoperationen
            await mock_connection.execute("INSERT INTO stems ...")
            await mock_connection.execute("UPDATE arrangements ...")
        
        # Transaction sollte gestartet und committed worden sein
        mock_connection.begin.assert_called_once()
        mock_connection.commit.assert_called_once()
        mock_connection.rollback.assert_not_called()
    
    @pytest.mark.unit
    async def test_transaction_rollback(self, database_transaction, mock_connection):
        """Test: Transaktion bei Fehler rollback"""
        with pytest.raises(DatabaseError):
            async with database_transaction:
                # Simuliere Datenbankoperationen
                await mock_connection.execute("INSERT INTO stems ...")
                
                # Simuliere Fehler
                raise DatabaseError("Something went wrong")
        
        # Transaction sollte gestartet und rollback gemacht worden sein
        mock_connection.begin.assert_called_once()
        mock_connection.rollback.assert_called_once()
        mock_connection.commit.assert_not_called()
    
    @pytest.mark.unit
    async def test_nested_transactions(self, mock_connection):
        """Test: Verschachtelte Transaktionen"""
        outer_transaction = DatabaseTransaction(mock_connection)
        inner_transaction = DatabaseTransaction(mock_connection)
        
        async with outer_transaction:
            await mock_connection.execute("INSERT INTO stems ...")
            
            async with inner_transaction:
                await mock_connection.execute("INSERT INTO arrangements ...")
        
        # Nur äußere Transaktion sollte begin/commit aufrufen
        assert mock_connection.begin.call_count == 1
        assert mock_connection.commit.call_count == 1


class TestQueryBuilder:
    """Tests für Query-Builder"""
    
    @pytest.fixture
    def query_builder(self):
        """Query-Builder für Tests"""
        return QueryBuilder()
    
    @pytest.mark.unit
    def test_select_query(self, query_builder):
        """Test: SELECT-Query erstellen"""
        query = (
            query_builder
            .select(["id", "name", "duration"])
            .from_table("stems")
            .where("duration > $1")
            .order_by("created_at DESC")
            .limit(10)
            .build()
        )
        
        expected = (
            "SELECT id, name, duration FROM stems "
            "WHERE duration > $1 "
            "ORDER BY created_at DESC "
            "LIMIT 10"
        )
        
        assert query.strip() == expected.strip()
    
    @pytest.mark.unit
    def test_insert_query(self, query_builder):
        """Test: INSERT-Query erstellen"""
        query = (
            query_builder
            .insert_into("stems")
            .values({
                "id": "$1",
                "name": "$2",
                "duration": "$3",
                "created_at": "NOW()"
            })
            .returning(["id", "created_at"])
            .build()
        )
        
        expected = (
            "INSERT INTO stems (id, name, duration, created_at) "
            "VALUES ($1, $2, $3, NOW()) "
            "RETURNING id, created_at"
        )
        
        assert query.strip() == expected.strip()
    
    @pytest.mark.unit
    def test_update_query(self, query_builder):
        """Test: UPDATE-Query erstellen"""
        query = (
            query_builder
            .update("stems")
            .set({
                "name": "$1",
                "updated_at": "NOW()"
            })
            .where("id = $2")
            .returning(["id", "updated_at"])
            .build()
        )
        
        expected = (
            "UPDATE stems SET name = $1, updated_at = NOW() "
            "WHERE id = $2 "
            "RETURNING id, updated_at"
        )
        
        assert query.strip() == expected.strip()
    
    @pytest.mark.unit
    def test_delete_query(self, query_builder):
        """Test: DELETE-Query erstellen"""
        query = (
            query_builder
            .delete_from("stems")
            .where("id = $1")
            .build()
        )
        
        expected = "DELETE FROM stems WHERE id = $1"
        
        assert query.strip() == expected.strip()
    
    @pytest.mark.unit
    def test_complex_query_with_joins(self, query_builder):
        """Test: Komplexe Query mit JOINs"""
        query = (
            query_builder
            .select([
                "s.id",
                "s.name",
                "a.name as arrangement_name"
            ])
            .from_table("stems s")
            .join("arrangement_stems as ON s.id = as.stem_id")
            .join("arrangements a ON as.arrangement_id = a.id")
            .where("s.duration > $1")
            .where("a.created_at > $2")
            .order_by("s.created_at DESC")
            .build()
        )
        
        assert "JOIN" in query
        assert "WHERE" in query
        assert "ORDER BY" in query
    
    @pytest.mark.unit
    def test_query_with_subquery(self, query_builder):
        """Test: Query mit Subquery"""
        subquery = (
            QueryBuilder()
            .select(["arrangement_id"])
            .from_table("arrangement_stems")
            .where("stem_id = $1")
            .build()
        )
        
        query = (
            query_builder
            .select(["id", "name"])
            .from_table("arrangements")
            .where(f"id IN ({subquery})")
            .build()
        )
        
        assert "SELECT" in query
        assert "IN (" in query
        assert "arrangement_stems" in query


class TestConnectionPool:
    """Tests für Connection-Pool"""
    
    @pytest.fixture
    def pool_config(self):
        """Pool-Konfiguration für Tests"""
        return {
            "min_size": 5,
            "max_size": 20,
            "timeout": 30,
            "recycle": 3600
        }
    
    @pytest.fixture
    def connection_pool(self, pool_config):
        """Connection-Pool für Tests"""
        return ConnectionPool(pool_config)
    
    @pytest.mark.unit
    async def test_pool_initialization(self, pool_config):
        """Test: Connection-Pool-Initialisierung"""
        pool = ConnectionPool(pool_config)
        
        assert pool.min_size == 5
        assert pool.max_size == 20
        assert pool.timeout == 30
        assert pool.active_connections == 0
    
    @pytest.mark.unit
    async def test_acquire_connection(self, connection_pool):
        """Test: Verbindung aus Pool abrufen"""
        # Mock Connection-Erstellung
        with patch.object(connection_pool, '_create_connection') as mock_create:
            mock_connection = Mock()
            mock_create.return_value = mock_connection
            
            connection = await connection_pool.acquire()
            
            assert connection == mock_connection
            assert connection_pool.active_connections == 1
    
    @pytest.mark.unit
    async def test_release_connection(self, connection_pool):
        """Test: Verbindung an Pool zurückgeben"""
        # Mock Connection
        mock_connection = Mock()
        mock_connection.is_closed = False
        
        # Simuliere aktive Verbindung
        connection_pool.active_connections = 1
        connection_pool._active_connections.add(mock_connection)
        
        await connection_pool.release(mock_connection)
        
        assert connection_pool.active_connections == 0
        assert mock_connection not in connection_pool._active_connections
    
    @pytest.mark.unit
    async def test_pool_max_size_limit(self, connection_pool):
        """Test: Maximale Pool-Größe"""
        # Simuliere Pool am Limit
        connection_pool.active_connections = connection_pool.max_size
        
        with patch.object(connection_pool, '_create_connection') as mock_create:
            # Sollte keine neue Verbindung erstellen
            with pytest.raises(ConnectionError):
                await connection_pool.acquire(timeout=0.1)
            
            mock_create.assert_not_called()
    
    @pytest.mark.unit
    async def test_pool_statistics(self, connection_pool):
        """Test: Pool-Statistiken"""
        stats = connection_pool.get_statistics()
        
        assert "active_connections" in stats
        assert "idle_connections" in stats
        assert "total_connections" in stats
        assert "max_size" in stats
        assert "min_size" in stats
        
        assert stats["max_size"] == 20
        assert stats["min_size"] == 5


class TestDatabaseMigration:
    """Tests für Database-Migration"""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock Datenbankverbindung"""
        connection = Mock()
        connection.execute = AsyncMock()
        connection.fetchall = AsyncMock()
        return connection
    
    @pytest.fixture
    def migration_manager(self, mock_connection):
        """Migration-Manager für Tests"""
        return DatabaseMigration(mock_connection)
    
    @pytest.mark.unit
    async def test_get_current_version(self, migration_manager, mock_connection):
        """Test: Aktuelle Datenbankversion abrufen"""
        # Mock Versions-Tabelle
        mock_connection.fetchall.return_value = [
            {"version": "1.2.3", "applied_at": datetime.now()}
        ]
        
        version = await migration_manager.get_current_version()
        
        assert version == "1.2.3"
    
    @pytest.mark.unit
    async def test_apply_migration(self, migration_manager, mock_connection):
        """Test: Migration anwenden"""
        migration = {
            "version": "1.3.0",
            "description": "Add new indexes",
            "up_sql": "CREATE INDEX idx_stems_duration ON stems(duration);",
            "down_sql": "DROP INDEX idx_stems_duration;"
        }
        
        success = await migration_manager.apply_migration(migration)
        
        assert success == True
        # SQL sollte ausgeführt worden sein
        mock_connection.execute.assert_called()
    
    @pytest.mark.unit
    async def test_rollback_migration(self, migration_manager, mock_connection):
        """Test: Migration rückgängig machen"""
        migration = {
            "version": "1.3.0",
            "description": "Add new indexes",
            "down_sql": "DROP INDEX idx_stems_duration;"
        }
        
        success = await migration_manager.rollback_migration(migration)
        
        assert success == True
        mock_connection.execute.assert_called()
    
    @pytest.mark.unit
    async def test_get_pending_migrations(self, migration_manager):
        """Test: Ausstehende Migrationen abrufen"""
        # Mock aktuelle Version
        with patch.object(migration_manager, 'get_current_version') as mock_version:
            mock_version.return_value = "1.2.0"
            
            # Mock verfügbare Migrationen
            available_migrations = [
                {"version": "1.2.1", "description": "Fix indexes"},
                {"version": "1.3.0", "description": "Add new tables"},
                {"version": "1.1.0", "description": "Old migration"}
            ]
            
            with patch.object(migration_manager, 'get_available_migrations') as mock_available:
                mock_available.return_value = available_migrations
                
                pending = await migration_manager.get_pending_migrations()
                
                # Nur Migrationen nach 1.2.0 sollten ausstehend sein
                assert len(pending) == 2
                assert pending[0]["version"] == "1.2.1"
                assert pending[1]["version"] == "1.3.0"


class TestDatabaseServiceIntegration:
    """Integrationstests für Database-Service"""
    
    @pytest.mark.integration
    async def test_full_crud_workflow(self):
        """Test: Vollständiger CRUD-Workflow"""
        config = DatabaseConfig(
            host="localhost",
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        # Mock Connection Pool
        mock_pool = Mock()
        mock_connection = Mock()
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()
        
        service = DatabaseService(config)
        service.connection_pool = mock_pool
        
        # 1. Stem erstellen
        stem_data = StemCreate(
            name="Integration Test Stem",
            file_path="/test/audio.wav",
            duration=45.0,
            sample_rate=48000
        )
        
        mock_stem = Stem(
            id="stem_integration",
            name="Integration Test Stem",
            file_path="/test/audio.wav",
            duration=45.0,
            sample_rate=48000,
            created_at=datetime.now()
        )
        
        with patch.object(service, '_insert_record') as mock_insert:
            mock_insert.return_value = mock_stem
            
            created_stem = await service.create_stem(stem_data)
            assert created_stem.name == "Integration Test Stem"
        
        # 2. Stem abrufen
        with patch.object(service, '_get_record_by_id') as mock_get:
            mock_get.return_value = mock_stem
            
            retrieved_stem = await service.get_stem("stem_integration")
            assert retrieved_stem.id == "stem_integration"
        
        # 3. Stem aktualisieren
        update_data = StemUpdate(name="Updated Integration Stem")
        
        updated_stem = Stem(
            id="stem_integration",
            name="Updated Integration Stem",
            file_path="/test/audio.wav",
            duration=45.0
        )
        
        with patch.object(service, '_get_record_by_id') as mock_get:
            mock_get.return_value = mock_stem
            
            with patch.object(service, '_update_record') as mock_update:
                mock_update.return_value = updated_stem
                
                updated = await service.update_stem("stem_integration", update_data)
                assert updated.name == "Updated Integration Stem"
        
        # 4. Stem löschen
        with patch.object(service, '_get_record_by_id') as mock_get:
            mock_get.return_value = updated_stem
            
            with patch.object(service, '_delete_record') as mock_delete:
                mock_delete.return_value = True
                
                deleted = await service.delete_stem("stem_integration")
                assert deleted == True
    
    @pytest.mark.performance
    async def test_database_service_performance(self):
        """Test: Database-Service-Performance"""
        import time
        
        config = DatabaseConfig(
            host="localhost",
            database="test_db",
            pool_size=20
        )
        
        mock_pool = Mock()
        mock_connection = Mock()
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()
        
        service = DatabaseService(config)
        service.connection_pool = mock_pool
        
        # Performance-Test: Viele Stems erstellen
        stems_data = [
            StemCreate(
                name=f"Performance Test Stem {i}",
                file_path=f"/test/audio_{i}.wav",
                duration=30.0 + i,
                sample_rate=48000
            )
            for i in range(100)
        ]
        
        with patch.object(service, '_insert_record') as mock_insert:
            mock_insert.side_effect = [
                Stem(
                    id=f"stem_{i}",
                    name=f"Performance Test Stem {i}",
                    duration=30.0 + i
                )
                for i in range(100)
            ]
            
            start_time = time.time()
            
            # Stems parallel erstellen
            tasks = [
                service.create_stem(stem_data)
                for stem_data in stems_data
            ]
            
            created_stems = await asyncio.gather(*tasks)
            
            creation_time = time.time() - start_time
            
            assert len(created_stems) == 100
            # Sollte unter 5 Sekunden dauern
            assert creation_time < 5.0
        
        # Performance-Test: Stems auflisten
        mock_stems = [
            Stem(id=f"stem_{i}", name=f"Stem {i}", duration=30.0)
            for i in range(1000)
        ]
        
        with patch.object(service, '_list_records') as mock_list:
            mock_list.return_value = mock_stems
            
            start_time = time.time()
            stems = await service.list_stems(limit=1000)
            list_time = time.time() - start_time
            
            assert len(stems) == 1000
            # Sollte unter 1 Sekunde dauern
            assert list_time < 1.0