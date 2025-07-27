"""Tests für Cache-Service"""

import pytest
import asyncio
import time
import json
import pickle
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from src.services.cache_service import (
    CacheService, MemoryCache, RedisCache, FileCache,
    CacheManager, CacheStrategy, CacheEntry, CacheStats,
    CacheSerializer, CacheCompression, CacheEviction,
    LRUCache, LFUCache, TTLCache, CacheCluster,
    CacheMetrics, CacheMonitor, CacheBackend
)
from src.core.config import CacheConfig
from src.core.exceptions import (
    CacheError, CacheConnectionError, CacheSerializationError,
    CacheEvictionError, CacheConfigurationError
)
from src.schemas.cache import (
    CacheKey, CacheValue, CacheMetadata, CacheOperation,
    CacheHit, CacheMiss, CacheStatistics
)


class TestCacheService:
    """Tests für Cache-Service"""
    
    @pytest.fixture
    def cache_config(self):
        """Cache-Konfiguration für Tests"""
        return CacheConfig(
            backend="memory",
            host="localhost",
            port=6379,
            database=0,
            password=None,
            max_memory="100MB",
            max_entries=1000,
            default_ttl=3600,
            compression=True,
            serialization="json",
            eviction_policy="lru",
            cluster_enabled=False,
            cluster_nodes=[],
            monitoring_enabled=True,
            metrics_interval=60,
            health_check_interval=30,
            connection_timeout=5,
            operation_timeout=1,
            retry_attempts=3,
            retry_delay=0.1
        )
    
    @pytest.fixture
    def cache_service(self, cache_config):
        """Cache-Service für Tests"""
        return CacheService(cache_config)
    
    @pytest.mark.unit
    def test_cache_service_initialization(self, cache_config):
        """Test: Cache-Service-Initialisierung"""
        service = CacheService(cache_config)
        
        assert service.config == cache_config
        assert service.backend == "memory"
        assert service.max_entries == 1000
        assert service.default_ttl == 3600
        assert isinstance(service.cache, MemoryCache)
        assert isinstance(service.manager, CacheManager)
        assert isinstance(service.stats, CacheStats)
    
    @pytest.mark.unit
    def test_cache_service_invalid_config(self):
        """Test: Cache-Service mit ungültiger Konfiguration"""
        invalid_config = CacheConfig(
            backend="invalid_backend",
            max_memory="invalid_size",
            max_entries=-1,
            default_ttl=-1
        )
        
        with pytest.raises(CacheConfigurationError):
            CacheService(invalid_config)
    
    @pytest.mark.unit
    async def test_cache_set_and_get(self, cache_service):
        """Test: Cache Set und Get"""
        key = "test_key"
        value = {"data": "test_value", "number": 42}
        
        # Wert setzen
        await cache_service.set(key, value)
        
        # Wert abrufen
        retrieved_value = await cache_service.get(key)
        
        assert retrieved_value == value
        assert cache_service.stats.hits == 1
        assert cache_service.stats.sets == 1
    
    @pytest.mark.unit
    async def test_cache_get_nonexistent_key(self, cache_service):
        """Test: Cache Get für nicht existierenden Key"""
        key = "nonexistent_key"
        
        # Nicht existierenden Wert abrufen
        retrieved_value = await cache_service.get(key)
        
        assert retrieved_value is None
        assert cache_service.stats.misses == 1
    
    @pytest.mark.unit
    async def test_cache_set_with_ttl(self, cache_service):
        """Test: Cache Set mit TTL"""
        key = "ttl_key"
        value = "ttl_value"
        ttl = 1  # 1 Sekunde
        
        # Wert mit TTL setzen
        await cache_service.set(key, value, ttl=ttl)
        
        # Sofort abrufen - sollte existieren
        retrieved_value = await cache_service.get(key)
        assert retrieved_value == value
        
        # Warten bis TTL abläuft
        await asyncio.sleep(1.1)
        
        # Erneut abrufen - sollte abgelaufen sein
        expired_value = await cache_service.get(key)
        assert expired_value is None
    
    @pytest.mark.unit
    async def test_cache_delete(self, cache_service):
        """Test: Cache Delete"""
        key = "delete_key"
        value = "delete_value"
        
        # Wert setzen
        await cache_service.set(key, value)
        
        # Existenz prüfen
        assert await cache_service.exists(key) == True
        
        # Wert löschen
        deleted = await cache_service.delete(key)
        assert deleted == True
        
        # Existenz erneut prüfen
        assert await cache_service.exists(key) == False
        
        # Nicht existierenden Key löschen
        deleted_again = await cache_service.delete(key)
        assert deleted_again == False
    
    @pytest.mark.unit
    async def test_cache_clear(self, cache_service):
        """Test: Cache Clear"""
        # Mehrere Werte setzen
        for i in range(5):
            await cache_service.set(f"key_{i}", f"value_{i}")
        
        # Cache-Größe prüfen
        assert cache_service.size() == 5
        
        # Cache leeren
        await cache_service.clear()
        
        # Cache sollte leer sein
        assert cache_service.size() == 0
        
        # Alle Keys sollten weg sein
        for i in range(5):
            assert await cache_service.exists(f"key_{i}") == False
    
    @pytest.mark.unit
    async def test_cache_keys_pattern(self, cache_service):
        """Test: Cache Keys mit Pattern"""
        # Verschiedene Keys setzen
        await cache_service.set("user:123", "user_data_123")
        await cache_service.set("user:456", "user_data_456")
        await cache_service.set("session:abc", "session_data_abc")
        await cache_service.set("config:main", "config_data")
        
        # Alle Keys abrufen
        all_keys = await cache_service.keys()
        assert len(all_keys) == 4
        
        # User-Keys mit Pattern abrufen
        user_keys = await cache_service.keys("user:*")
        assert len(user_keys) == 2
        assert "user:123" in user_keys
        assert "user:456" in user_keys
        
        # Session-Keys mit Pattern abrufen
        session_keys = await cache_service.keys("session:*")
        assert len(session_keys) == 1
        assert "session:abc" in session_keys
    
    @pytest.mark.unit
    async def test_cache_increment_decrement(self, cache_service):
        """Test: Cache Increment und Decrement"""
        key = "counter_key"
        
        # Increment auf nicht existierenden Key
        result = await cache_service.increment(key)
        assert result == 1
        
        # Weitere Increments
        result = await cache_service.increment(key, delta=5)
        assert result == 6
        
        # Decrement
        result = await cache_service.decrement(key, delta=2)
        assert result == 4
        
        # Aktuellen Wert prüfen
        current_value = await cache_service.get(key)
        assert current_value == 4
    
    @pytest.mark.unit
    async def test_cache_batch_operations(self, cache_service):
        """Test: Cache Batch-Operationen"""
        # Batch Set
        batch_data = {
            "batch_key_1": "batch_value_1",
            "batch_key_2": "batch_value_2",
            "batch_key_3": "batch_value_3"
        }
        
        await cache_service.set_many(batch_data)
        
        # Batch Get
        keys = list(batch_data.keys())
        retrieved_data = await cache_service.get_many(keys)
        
        assert len(retrieved_data) == 3
        for key, value in batch_data.items():
            assert retrieved_data[key] == value
        
        # Batch Delete
        deleted_count = await cache_service.delete_many(keys)
        assert deleted_count == 3
        
        # Prüfen dass alle gelöscht wurden
        for key in keys:
            assert await cache_service.exists(key) == False


class TestMemoryCache:
    """Tests für Memory-Cache"""
    
    @pytest.fixture
    def memory_cache(self):
        """Memory-Cache für Tests"""
        return MemoryCache(
            max_entries=100,
            default_ttl=3600,
            eviction_policy="lru"
        )
    
    @pytest.mark.unit
    async def test_memory_cache_basic_operations(self, memory_cache):
        """Test: Memory-Cache Grundoperationen"""
        # Set und Get
        await memory_cache.set("key1", "value1")
        value = await memory_cache.get("key1")
        assert value == "value1"
        
        # Exists
        assert await memory_cache.exists("key1") == True
        assert await memory_cache.exists("nonexistent") == False
        
        # Delete
        deleted = await memory_cache.delete("key1")
        assert deleted == True
        assert await memory_cache.exists("key1") == False
    
    @pytest.mark.unit
    async def test_memory_cache_ttl_expiration(self, memory_cache):
        """Test: Memory-Cache TTL-Ablauf"""
        # Kurze TTL setzen
        await memory_cache.set("ttl_key", "ttl_value", ttl=0.5)
        
        # Sofort abrufen
        value = await memory_cache.get("ttl_key")
        assert value == "ttl_value"
        
        # Warten bis Ablauf
        await asyncio.sleep(0.6)
        
        # Sollte abgelaufen sein
        expired_value = await memory_cache.get("ttl_key")
        assert expired_value is None
    
    @pytest.mark.unit
    async def test_memory_cache_lru_eviction(self):
        """Test: Memory-Cache LRU-Eviction"""
        # Kleiner Cache für Eviction-Test
        small_cache = MemoryCache(
            max_entries=3,
            eviction_policy="lru"
        )
        
        # Cache füllen
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")
        
        # Alle sollten existieren
        assert await small_cache.exists("key1") == True
        assert await small_cache.exists("key2") == True
        assert await small_cache.exists("key3") == True
        
        # key1 zugreifen (macht es "recently used")
        await small_cache.get("key1")
        
        # Neuen Key hinzufügen - sollte key2 evicten (least recently used)
        await small_cache.set("key4", "value4")
        
        assert await small_cache.exists("key1") == True  # Recently used
        assert await small_cache.exists("key2") == False # Evicted
        assert await small_cache.exists("key3") == True
        assert await small_cache.exists("key4") == True  # Neu
    
    @pytest.mark.unit
    def test_memory_cache_size_and_stats(self, memory_cache):
        """Test: Memory-Cache Größe und Statistiken"""
        # Initial leer
        assert memory_cache.size() == 0
        
        # Einträge hinzufügen
        asyncio.run(memory_cache.set("key1", "value1"))
        asyncio.run(memory_cache.set("key2", "value2"))
        
        assert memory_cache.size() == 2
        
        # Statistiken prüfen
        stats = memory_cache.get_stats()
        assert stats.total_entries == 2
        assert stats.memory_usage > 0
    
    @pytest.mark.unit
    async def test_memory_cache_clear(self, memory_cache):
        """Test: Memory-Cache Clear"""
        # Einträge hinzufügen
        await memory_cache.set("key1", "value1")
        await memory_cache.set("key2", "value2")
        
        assert memory_cache.size() == 2
        
        # Cache leeren
        await memory_cache.clear()
        
        assert memory_cache.size() == 0
        assert await memory_cache.exists("key1") == False
        assert await memory_cache.exists("key2") == False


class TestRedisCache:
    """Tests für Redis-Cache"""
    
    @pytest.fixture
    def redis_config(self):
        """Redis-Konfiguration für Tests"""
        return {
            "host": "localhost",
            "port": 6379,
            "database": 1,  # Test-Database
            "password": None,
            "connection_timeout": 5,
            "operation_timeout": 1
        }
    
    @pytest.fixture
    def redis_cache(self, redis_config):
        """Redis-Cache für Tests (mit Mock)"""
        with patch('redis.asyncio.Redis') as mock_redis:
            # Mock Redis-Client
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(**redis_config)
            cache._client = mock_client
            
            return cache, mock_client
    
    @pytest.mark.unit
    async def test_redis_cache_connection(self, redis_cache):
        """Test: Redis-Cache-Verbindung"""
        cache, mock_client = redis_cache
        
        # Mock erfolgreiche Verbindung
        mock_client.ping.return_value = True
        
        # Verbindung testen
        connected = await cache.connect()
        assert connected == True
        
        mock_client.ping.assert_called_once()
    
    @pytest.mark.unit
    async def test_redis_cache_connection_failure(self, redis_cache):
        """Test: Redis-Cache-Verbindungsfehler"""
        cache, mock_client = redis_cache
        
        # Mock Verbindungsfehler
        mock_client.ping.side_effect = Exception("Connection failed")
        
        # Verbindung sollte fehlschlagen
        with pytest.raises(CacheConnectionError):
            await cache.connect()
    
    @pytest.mark.unit
    async def test_redis_cache_set_and_get(self, redis_cache):
        """Test: Redis-Cache Set und Get"""
        cache, mock_client = redis_cache
        
        key = "redis_key"
        value = {"data": "redis_value"}
        serialized_value = json.dumps(value)
        
        # Mock Set-Operation
        mock_client.set.return_value = True
        
        # Mock Get-Operation
        mock_client.get.return_value = serialized_value
        
        # Set
        await cache.set(key, value)
        mock_client.set.assert_called_once()
        
        # Get
        retrieved_value = await cache.get(key)
        assert retrieved_value == value
        mock_client.get.assert_called_once_with(key)
    
    @pytest.mark.unit
    async def test_redis_cache_ttl_operations(self, redis_cache):
        """Test: Redis-Cache TTL-Operationen"""
        cache, mock_client = redis_cache
        
        key = "ttl_key"
        value = "ttl_value"
        ttl = 3600
        
        # Mock TTL-Set
        mock_client.setex.return_value = True
        
        # Set mit TTL
        await cache.set(key, value, ttl=ttl)
        mock_client.setex.assert_called_once()
        
        # Mock TTL-Get
        mock_client.ttl.return_value = 1800  # Verbleibende Zeit
        
        # TTL abrufen
        remaining_ttl = await cache.get_ttl(key)
        assert remaining_ttl == 1800
    
    @pytest.mark.unit
    async def test_redis_cache_batch_operations(self, redis_cache):
        """Test: Redis-Cache Batch-Operationen"""
        cache, mock_client = redis_cache
        
        # Mock Pipeline
        mock_pipeline = AsyncMock()
        mock_client.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True, True, True]
        
        # Batch Set
        batch_data = {
            "batch_key_1": "batch_value_1",
            "batch_key_2": "batch_value_2",
            "batch_key_3": "batch_value_3"
        }
        
        await cache.set_many(batch_data)
        
        # Pipeline sollte verwendet worden sein
        mock_client.pipeline.assert_called_once()
        mock_pipeline.execute.assert_called_once()
    
    @pytest.mark.unit
    async def test_redis_cache_pattern_operations(self, redis_cache):
        """Test: Redis-Cache Pattern-Operationen"""
        cache, mock_client = redis_cache
        
        # Mock Keys mit Pattern
        mock_client.keys.return_value = ["user:123", "user:456"]
        
        # Keys mit Pattern abrufen
        keys = await cache.keys("user:*")
        
        assert len(keys) == 2
        assert "user:123" in keys
        assert "user:456" in keys
        
        mock_client.keys.assert_called_once_with("user:*")


class TestCacheManager:
    """Tests für Cache-Manager"""
    
    @pytest.fixture
    def cache_manager(self):
        """Cache-Manager für Tests"""
        return CacheManager(
            primary_cache=MemoryCache(max_entries=100),
            secondary_cache=None,
            write_through=True,
            write_behind=False,
            read_through=True
        )
    
    @pytest.mark.unit
    async def test_cache_manager_write_through(self, cache_manager):
        """Test: Cache-Manager Write-Through"""
        key = "manager_key"
        value = "manager_value"
        
        # Mock Secondary Cache
        secondary_cache = AsyncMock()
        cache_manager.secondary_cache = secondary_cache
        
        # Write-Through Set
        await cache_manager.set(key, value)
        
        # Primary Cache sollte Wert haben
        primary_value = await cache_manager.primary_cache.get(key)
        assert primary_value == value
        
        # Secondary Cache sollte auch aufgerufen worden sein
        secondary_cache.set.assert_called_once_with(key, value, ttl=None)
    
    @pytest.mark.unit
    async def test_cache_manager_read_through(self, cache_manager):
        """Test: Cache-Manager Read-Through"""
        key = "read_through_key"
        value = "read_through_value"
        
        # Mock Secondary Cache mit Wert
        secondary_cache = AsyncMock()
        secondary_cache.get.return_value = value
        cache_manager.secondary_cache = secondary_cache
        
        # Primary Cache hat keinen Wert
        assert await cache_manager.primary_cache.get(key) is None
        
        # Read-Through Get
        retrieved_value = await cache_manager.get(key)
        
        # Sollte Wert aus Secondary Cache zurückgeben
        assert retrieved_value == value
        
        # Und in Primary Cache speichern
        primary_value = await cache_manager.primary_cache.get(key)
        assert primary_value == value
    
    @pytest.mark.unit
    async def test_cache_manager_fallback(self, cache_manager):
        """Test: Cache-Manager Fallback"""
        key = "fallback_key"
        
        # Mock Primary Cache Fehler
        cache_manager.primary_cache = AsyncMock()
        cache_manager.primary_cache.get.side_effect = Exception("Primary cache error")
        
        # Mock Secondary Cache mit Wert
        secondary_cache = AsyncMock()
        secondary_cache.get.return_value = "fallback_value"
        cache_manager.secondary_cache = secondary_cache
        
        # Sollte auf Secondary Cache zurückfallen
        value = await cache_manager.get(key)
        assert value == "fallback_value"
    
    @pytest.mark.unit
    def test_cache_manager_stats_aggregation(self, cache_manager):
        """Test: Cache-Manager Statistik-Aggregation"""
        # Mock Primary Cache Stats
        primary_stats = CacheStats()
        primary_stats.hits = 10
        primary_stats.misses = 5
        primary_stats.sets = 8
        cache_manager.primary_cache.get_stats = Mock(return_value=primary_stats)
        
        # Mock Secondary Cache Stats
        secondary_stats = CacheStats()
        secondary_stats.hits = 3
        secondary_stats.misses = 2
        secondary_stats.sets = 4
        secondary_cache = Mock()
        secondary_cache.get_stats.return_value = secondary_stats
        cache_manager.secondary_cache = secondary_cache
        
        # Aggregierte Stats abrufen
        total_stats = cache_manager.get_aggregated_stats()
        
        assert total_stats.hits == 13  # 10 + 3
        assert total_stats.misses == 7  # 5 + 2
        assert total_stats.sets == 12  # 8 + 4


class TestCacheServiceIntegration:
    """Integrationstests für Cache-Service"""
    
    @pytest.mark.integration
    async def test_full_cache_workflow(self):
        """Test: Vollständiger Cache-Workflow"""
        config = CacheConfig(
            backend="memory",
            max_entries=1000,
            default_ttl=3600,
            eviction_policy="lru"
        )
        
        service = CacheService(config)
        
        # 1. Grundlegende Operationen
        await service.set("user:123", {"name": "John", "age": 30})
        user_data = await service.get("user:123")
        assert user_data["name"] == "John"
        
        # 2. TTL-Test
        await service.set("temp:data", "temporary", ttl=1)
        temp_data = await service.get("temp:data")
        assert temp_data == "temporary"
        
        await asyncio.sleep(1.1)
        expired_data = await service.get("temp:data")
        assert expired_data is None
        
        # 3. Batch-Operationen
        batch_data = {
            "item:1": "value1",
            "item:2": "value2",
            "item:3": "value3"
        }
        await service.set_many(batch_data)
        
        retrieved_batch = await service.get_many(["item:1", "item:2", "item:3"])
        assert len(retrieved_batch) == 3
        assert retrieved_batch["item:1"] == "value1"
        
        # 4. Pattern-Suche
        item_keys = await service.keys("item:*")
        assert len(item_keys) == 3
        
        # 5. Statistiken
        stats = service.get_stats()
        assert stats.hits > 0
        assert stats.sets > 0
        
        # 6. Cache leeren
        await service.clear()
        assert service.size() == 0
    
    @pytest.mark.performance
    async def test_cache_service_performance(self):
        """Test: Cache-Service-Performance"""
        import time
        
        config = CacheConfig(
            backend="memory",
            max_entries=10000,
            eviction_policy="lru"
        )
        
        service = CacheService(config)
        
        # Performance-Test: Viele Set-Operationen
        start_time = time.time()
        
        for i in range(1000):
            await service.set(f"perf_key_{i}", f"perf_value_{i}")
        
        set_time = time.time() - start_time
        
        # Performance-Test: Viele Get-Operationen
        start_time = time.time()
        
        for i in range(1000):
            value = await service.get(f"perf_key_{i}")
            assert value == f"perf_value_{i}"
        
        get_time = time.time() - start_time
        
        # Performance-Assertions
        assert set_time < 1.0  # Unter 1 Sekunde für 1000 Sets
        assert get_time < 0.5  # Unter 0.5 Sekunden für 1000 Gets
        
        # Durchschnittliche Operationszeit
        avg_set_time = set_time / 1000
        avg_get_time = get_time / 1000
        
        assert avg_set_time < 0.001  # Unter 1ms pro Set
        assert avg_get_time < 0.0005  # Unter 0.5ms pro Get
        
        # Performance-Test: Batch-Operationen
        batch_data = {f"batch_key_{i}": f"batch_value_{i}" for i in range(100)}
        
        start_time = time.time()
        await service.set_many(batch_data)
        batch_set_time = time.time() - start_time
        
        start_time = time.time()
        retrieved_batch = await service.get_many(list(batch_data.keys()))
        batch_get_time = time.time() - start_time
        
        # Batch-Operationen sollten effizienter sein
        assert batch_set_time < set_time / 5  # Mindestens 5x schneller
        assert batch_get_time < get_time / 5  # Mindestens 5x schneller
        
        # Alle Batch-Werte sollten korrekt sein
        assert len(retrieved_batch) == 100
        for key, value in batch_data.items():
            assert retrieved_batch[key] == value
    
    @pytest.mark.integration
    async def test_cache_eviction_policies(self):
        """Test: Cache-Eviction-Policies"""
        # LRU-Test
        lru_config = CacheConfig(
            backend="memory",
            max_entries=3,
            eviction_policy="lru"
        )
        
        lru_service = CacheService(lru_config)
        
        # Cache füllen
        await lru_service.set("lru_1", "value1")
        await lru_service.set("lru_2", "value2")
        await lru_service.set("lru_3", "value3")
        
        # lru_1 zugreifen (macht es recently used)
        await lru_service.get("lru_1")
        
        # Neuen Eintrag hinzufügen - sollte lru_2 evicten
        await lru_service.set("lru_4", "value4")
        
        assert await lru_service.exists("lru_1") == True  # Recently used
        assert await lru_service.exists("lru_2") == False # Evicted
        assert await lru_service.exists("lru_3") == True
        assert await lru_service.exists("lru_4") == True  # Neu
        
        # TTL-Test
        ttl_config = CacheConfig(
            backend="memory",
            max_entries=1000,
            eviction_policy="ttl"
        )
        
        ttl_service = CacheService(ttl_config)
        
        # Einträge mit verschiedenen TTLs
        await ttl_service.set("ttl_short", "short_value", ttl=0.5)
        await ttl_service.set("ttl_long", "long_value", ttl=2.0)
        
        # Kurz warten
        await asyncio.sleep(0.6)
        
        # Kurzer TTL sollte abgelaufen sein
        assert await ttl_service.get("ttl_short") is None
        assert await ttl_service.get("ttl_long") == "long_value"
        
        # Länger warten
        await asyncio.sleep(1.5)
        
        # Beide sollten abgelaufen sein
        assert await ttl_service.get("ttl_long") is None