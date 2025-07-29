"""Tests für Core-Utils"""

import pytest
import tempfile
import os
import json
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from core.utils import (
    # File utilities
    ensure_directory, safe_file_path, get_file_size, get_file_extension,
    copy_file, move_file, delete_file, list_files, find_files,
    
    # Audio utilities
    validate_audio_format, get_audio_duration, normalize_audio_path,
    convert_sample_rate, get_audio_info, create_silence,
    
    # Data utilities
    deep_merge, flatten_dict, unflatten_dict, sanitize_dict,
    serialize_data, deserialize_data, hash_data, compare_data,
    
    # String utilities
    sanitize_filename, generate_id, slugify, truncate_string,
    format_duration, format_file_size, validate_email,
    
    # Time utilities
    get_timestamp, format_timestamp, parse_timestamp,
    time_ago, duration_to_seconds, seconds_to_duration,
    
    # Async utilities
    run_async, gather_with_concurrency, retry_async,
    timeout_async, debounce, throttle,
    
    # Validation utilities
    validate_schema, validate_range, validate_type,
    validate_required_fields, ValidationResult,
    
    # Performance utilities
    Timer, MemoryProfiler, performance_monitor,
    cache_result, memoize, rate_limit,
    
    # Error utilities
    safe_execute, error_handler, exception_to_dict,
    log_exception, create_error_response
)


class TestFileUtilities:
    """Tests für File-Utilities"""
    
    @pytest.mark.unit
    def test_ensure_directory(self):
        """Test: Verzeichnis erstellen"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test", "nested", "directory")
            
            # Verzeichnis sollte nicht existieren
            assert not os.path.exists(test_dir)
            
            # Verzeichnis erstellen
            ensure_directory(test_dir)
            
            # Verzeichnis sollte jetzt existieren
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)
    
    @pytest.mark.unit
    def test_safe_file_path(self):
        """Test: Sichere Dateipfade"""
        # Normale Pfade
        assert safe_file_path("test.txt") == "test.txt"
        assert safe_file_path("folder/test.txt") == "folder/test.txt"
        
        # Gefährliche Pfade
        assert ".." not in safe_file_path("../test.txt")
        assert ".." not in safe_file_path("folder/../test.txt")
        
        # Absolute Pfade
        abs_path = safe_file_path("/absolute/path/test.txt")
        assert not abs_path.startswith("/")
    
    @pytest.mark.unit
    def test_get_file_size(self):
        """Test: Dateigröße ermitteln"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("Hello, World!" * 100)  # ~1300 Bytes
            temp_file_path = temp_file.name
        
        try:
            size = get_file_size(temp_file_path)
            assert size > 1000
            assert size < 2000
        finally:
            os.unlink(temp_file_path)
    
    @pytest.mark.unit
    def test_get_file_extension(self):
        """Test: Dateierweiterung ermitteln"""
        assert get_file_extension("test.txt") == ".txt"
        assert get_file_extension("audio.wav") == ".wav"
        assert get_file_extension("archive.tar.gz") == ".gz"
        assert get_file_extension("no_extension") == ""
        assert get_file_extension(".hidden") == ""
    
    @pytest.mark.unit
    def test_copy_file(self):
        """Test: Datei kopieren"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Quelldatei erstellen
            source_path = os.path.join(temp_dir, "source.txt")
            with open(source_path, 'w') as f:
                f.write("Test content")
            
            # Datei kopieren
            dest_path = os.path.join(temp_dir, "destination.txt")
            copy_file(source_path, dest_path)
            
            # Beide Dateien sollten existieren
            assert os.path.exists(source_path)
            assert os.path.exists(dest_path)
            
            # Inhalt sollte gleich sein
            with open(dest_path, 'r') as f:
                assert f.read() == "Test content"
    
    @pytest.mark.unit
    def test_list_files(self):
        """Test: Dateien auflisten"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test-Dateien erstellen
            files = ["test1.txt", "test2.wav", "test3.mp3", "other.log"]
            for file in files:
                with open(os.path.join(temp_dir, file), 'w') as f:
                    f.write("test")
            
            # Alle Dateien
            all_files = list_files(temp_dir)
            assert len(all_files) == 4
            
            # Nur Audio-Dateien
            audio_files = list_files(temp_dir, extensions=[".wav", ".mp3"])
            assert len(audio_files) == 2
            
            # Mit Pattern
            test_files = list_files(temp_dir, pattern="test*")
            assert len(test_files) == 3
    
    @pytest.mark.unit
    def test_find_files(self):
        """Test: Dateien rekursiv finden"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Verschachtelte Struktur erstellen
            sub_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(sub_dir)
            
            files = [
                os.path.join(temp_dir, "root.txt"),
                os.path.join(sub_dir, "nested.txt"),
                os.path.join(sub_dir, "audio.wav")
            ]
            
            for file in files:
                with open(file, 'w') as f:
                    f.write("test")
            
            # Alle .txt Dateien finden
            txt_files = find_files(temp_dir, pattern="*.txt")
            assert len(txt_files) == 2
            
            # Alle Dateien finden
            all_files = find_files(temp_dir)
            assert len(all_files) == 3


class TestAudioUtilities:
    """Tests für Audio-Utilities"""
    
    @pytest.mark.unit
    def test_validate_audio_format(self):
        """Test: Audio-Format validieren"""
        # Gültige Formate
        assert validate_audio_format("test.wav") == True
        assert validate_audio_format("test.mp3") == True
        assert validate_audio_format("test.flac") == True
        
        # Ungültige Formate
        assert validate_audio_format("test.txt") == False
        assert validate_audio_format("test.jpg") == False
        assert validate_audio_format("test") == False
    
    @pytest.mark.unit
    def test_normalize_audio_path(self):
        """Test: Audio-Pfad normalisieren"""
        # Normale Pfade
        assert normalize_audio_path("audio.wav") == "audio.wav"
        
        # Pfade mit Leerzeichen
        normalized = normalize_audio_path("my audio file.wav")
        assert " " not in normalized
        
        # Pfade mit Sonderzeichen
        normalized = normalize_audio_path("äöü-audio.wav")
        assert "ä" not in normalized
        assert "ö" not in normalized
        assert "ü" not in normalized
    
    @pytest.mark.unit
    def test_get_audio_info(self):
        """Test: Audio-Info abrufen (Mock)"""
        with patch('librosa.load') as mock_load:
            mock_load.return_value = ([0.1, 0.2, 0.3], 44100)
            
            info = get_audio_info("test.wav")
            
            assert info["sample_rate"] == 44100
            assert info["duration"] > 0
            assert info["samples"] == 3
    
    @pytest.mark.unit
    def test_create_silence(self):
        """Test: Stille erstellen"""
        silence = create_silence(duration=1.0, sample_rate=44100)
        
        assert len(silence) == 44100  # 1 Sekunde bei 44.1kHz
        assert all(sample == 0.0 for sample in silence)


class TestDataUtilities:
    """Tests für Data-Utilities"""
    
    @pytest.mark.unit
    def test_deep_merge(self):
        """Test: Deep-Merge von Dictionaries"""
        dict1 = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [1, 2]
        }
        
        dict2 = {
            "a": 10,
            "b": {"d": 30, "f": 4},
            "e": [3, 4],
            "g": 5
        }
        
        merged = deep_merge(dict1, dict2)
        
        assert merged["a"] == 10  # Überschrieben
        assert merged["b"]["c"] == 2  # Erhalten
        assert merged["b"]["d"] == 30  # Überschrieben
        assert merged["b"]["f"] == 4  # Hinzugefügt
        assert merged["e"] == [1, 2, 3, 4]  # Listen zusammengeführt
        assert merged["g"] == 5  # Hinzugefügt
    
    @pytest.mark.unit
    def test_flatten_dict(self):
        """Test: Dictionary flach machen"""
        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        flattened = flatten_dict(nested)
        
        assert flattened["a"] == 1
        assert flattened["b.c"] == 2
        assert flattened["b.d.e"] == 3
    
    @pytest.mark.unit
    def test_unflatten_dict(self):
        """Test: Dictionary entflachen"""
        flattened = {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3
        }
        
        unflattened = unflatten_dict(flattened)
        
        assert unflattened["a"] == 1
        assert unflattened["b"]["c"] == 2
        assert unflattened["b"]["d"]["e"] == 3
    
    @pytest.mark.unit
    def test_sanitize_dict(self):
        """Test: Dictionary bereinigen"""
        dirty = {
            "valid_key": "valid_value",
            "password": "secret123",
            "api_key": "abc123",
            "token": "xyz789",
            "normal_data": {"nested": "value"}
        }
        
        clean = sanitize_dict(dirty, sensitive_keys=["password", "api_key", "token"])
        
        assert clean["valid_key"] == "valid_value"
        assert clean["password"] == "***"
        assert clean["api_key"] == "***"
        assert clean["token"] == "***"
        assert clean["normal_data"]["nested"] == "value"
    
    @pytest.mark.unit
    def test_serialize_deserialize_data(self):
        """Test: Daten serialisieren/deserialisieren"""
        data = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "datetime": datetime.now()
        }
        
        # Serialisieren
        serialized = serialize_data(data)
        assert isinstance(serialized, str)
        
        # Deserialisieren
        deserialized = deserialize_data(serialized)
        
        assert deserialized["string"] == "test"
        assert deserialized["number"] == 42
        assert deserialized["list"] == [1, 2, 3]
        assert deserialized["dict"]["nested"] == True
        # Datetime wird als String serialisiert
        assert isinstance(deserialized["datetime"], str)
    
    @pytest.mark.unit
    def test_hash_data(self):
        """Test: Daten hashen"""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}  # Gleiche Daten, andere Reihenfolge
        data3 = {"a": 1, "b": 3}  # Andere Daten
        
        hash1 = hash_data(data1)
        hash2 = hash_data(data2)
        hash3 = hash_data(data3)
        
        assert hash1 == hash2  # Reihenfolge sollte egal sein
        assert hash1 != hash3  # Verschiedene Daten, verschiedene Hashes
        assert len(hash1) == 64  # SHA-256 Hash
    
    @pytest.mark.unit
    def test_compare_data(self):
        """Test: Daten vergleichen"""
        data1 = {"a": 1, "b": [1, 2, 3]}
        data2 = {"a": 1, "b": [1, 2, 3]}
        data3 = {"a": 1, "b": [1, 2, 4]}
        
        assert compare_data(data1, data2) == True
        assert compare_data(data1, data3) == False
        
        # Mit Toleranz für Floats
        float_data1 = {"value": 1.0000001}
        float_data2 = {"value": 1.0000002}
        
        assert compare_data(float_data1, float_data2, tolerance=1e-6) == True
        assert compare_data(float_data1, float_data2, tolerance=1e-8) == False


class TestStringUtilities:
    """Tests für String-Utilities"""
    
    @pytest.mark.unit
    def test_sanitize_filename(self):
        """Test: Dateinamen bereinigen"""
        # Normale Namen
        assert sanitize_filename("normal_file.txt") == "normal_file.txt"
        
        # Namen mit Sonderzeichen
        assert "/" not in sanitize_filename("path/to/file.txt")
        assert "\\" not in sanitize_filename("path\\to\\file.txt")
        assert ":" not in sanitize_filename("file:name.txt")
        
        # Leerzeichen
        sanitized = sanitize_filename("file with spaces.txt")
        assert " " not in sanitized or sanitized.replace(" ", "_") != sanitized
    
    @pytest.mark.unit
    def test_generate_id(self):
        """Test: ID generieren"""
        # Standard-ID
        id1 = generate_id()
        id2 = generate_id()
        
        assert len(id1) > 0
        assert id1 != id2  # Sollten unterschiedlich sein
        
        # ID mit Präfix
        prefixed_id = generate_id(prefix="test")
        assert prefixed_id.startswith("test")
        
        # ID mit bestimmter Länge
        long_id = generate_id(length=32)
        # Länge kann je nach Implementierung variieren
        assert len(long_id) >= 16
    
    @pytest.mark.unit
    def test_slugify(self):
        """Test: String zu Slug konvertieren"""
        assert slugify("Hello World") == "hello-world"
        assert slugify("Test with CAPS") == "test-with-caps"
        assert slugify("Special!@#$%Characters") == "special-characters"
        assert slugify("  Multiple   Spaces  ") == "multiple-spaces"
        assert slugify("äöü-umlauts") == "aou-umlauts"
    
    @pytest.mark.unit
    def test_truncate_string(self):
        """Test: String kürzen"""
        long_string = "This is a very long string that needs to be truncated"
        
        # Normale Kürzung
        truncated = truncate_string(long_string, 20)
        assert len(truncated) <= 23  # 20 + "..."
        assert truncated.endswith("...")
        
        # Kürzung ohne Ellipsis
        truncated_no_ellipsis = truncate_string(long_string, 20, ellipsis="")
        assert len(truncated_no_ellipsis) == 20
        assert not truncated_no_ellipsis.endswith("...")
        
        # String kürzer als Limit
        short_string = "Short"
        assert truncate_string(short_string, 20) == "Short"
    
    @pytest.mark.unit
    def test_format_duration(self):
        """Test: Dauer formatieren"""
        assert format_duration(65) == "1:05"
        assert format_duration(3661) == "1:01:01"
        assert format_duration(30) == "0:30"
        assert format_duration(0) == "0:00"
    
    @pytest.mark.unit
    def test_format_file_size(self):
        """Test: Dateigröße formatieren"""
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(500) == "500 B"
    
    @pytest.mark.unit
    def test_validate_email(self):
        """Test: E-Mail validieren"""
        # Gültige E-Mails
        assert validate_email("test@example.com") == True
        assert validate_email("user.name@domain.co.uk") == True
        assert validate_email("test+tag@example.org") == True
        
        # Ungültige E-Mails
        assert validate_email("invalid-email") == False
        assert validate_email("@example.com") == False
        assert validate_email("test@") == False
        assert validate_email("") == False


class TestTimeUtilities:
    """Tests für Time-Utilities"""
    
    @pytest.mark.unit
    def test_get_timestamp(self):
        """Test: Timestamp abrufen"""
        timestamp = get_timestamp()
        assert isinstance(timestamp, (int, float))
        assert timestamp > 0
    
    @pytest.mark.unit
    def test_format_timestamp(self):
        """Test: Timestamp formatieren"""
        timestamp = 1640995200  # 2022-01-01 00:00:00 UTC
        
        # ISO-Format
        iso_formatted = format_timestamp(timestamp, format="iso")
        assert "2022-01-01" in iso_formatted
        
        # Custom-Format
        custom_formatted = format_timestamp(timestamp, format="%Y-%m-%d")
        assert custom_formatted == "2022-01-01"
    
    @pytest.mark.unit
    def test_parse_timestamp(self):
        """Test: Timestamp parsen"""
        # ISO-String
        iso_string = "2022-01-01T00:00:00Z"
        timestamp = parse_timestamp(iso_string)
        assert isinstance(timestamp, (int, float))
        
        # Custom-Format
        date_string = "2022-01-01"
        timestamp = parse_timestamp(date_string, format="%Y-%m-%d")
        assert isinstance(timestamp, (int, float))
    
    @pytest.mark.unit
    def test_time_ago(self):
        """Test: Zeit-Differenz formatieren"""
        now = datetime.now()
        
        # 1 Minute ago
        one_minute_ago = now - timedelta(minutes=1)
        assert "minute" in time_ago(one_minute_ago)
        
        # 1 Stunde ago
        one_hour_ago = now - timedelta(hours=1)
        assert "hour" in time_ago(one_hour_ago)
        
        # 1 Tag ago
        one_day_ago = now - timedelta(days=1)
        assert "day" in time_ago(one_day_ago)
    
    @pytest.mark.unit
    def test_duration_conversion(self):
        """Test: Dauer-Konvertierung"""
        # Sekunden zu Dauer
        duration = seconds_to_duration(3661)
        assert duration["hours"] == 1
        assert duration["minutes"] == 1
        assert duration["seconds"] == 1
        
        # Dauer zu Sekunden
        seconds = duration_to_seconds(hours=1, minutes=1, seconds=1)
        assert seconds == 3661
        
        # String-Format
        duration_str = seconds_to_duration(3661, format="string")
        assert "1h" in duration_str and "1m" in duration_str and "1s" in duration_str


class TestAsyncUtilities:
    """Tests für Async-Utilities"""
    
    @pytest.mark.asyncio
    async def test_run_async(self):
        """Test: Async-Funktion ausführen"""
        async def async_function(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        result = await run_async(async_function, 5)
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_gather_with_concurrency(self):
        """Test: Async-Funktionen mit Concurrency-Limit"""
        async def slow_function(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        tasks = [slow_function(i) for i in range(10)]
        results = await gather_with_concurrency(tasks, max_concurrency=3)
        
        assert len(results) == 10
        assert results[0] == 0
        assert results[5] == 10
    
    @pytest.mark.asyncio
    async def test_retry_async(self):
        """Test: Async-Retry"""
        call_count = 0
        
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await retry_async(failing_function, max_retries=3, delay=0.01)
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_timeout_async(self):
        """Test: Async-Timeout"""
        async def slow_function():
            await asyncio.sleep(1.0)
            return "completed"
        
        # Sollte Timeout auslösen
        with pytest.raises(asyncio.TimeoutError):
            await timeout_async(slow_function(), timeout=0.1)
        
        # Sollte erfolgreich sein
        async def fast_function():
            await asyncio.sleep(0.01)
            return "completed"
        
        result = await timeout_async(fast_function(), timeout=0.1)
        assert result == "completed"
    
    @pytest.mark.unit
    def test_debounce(self):
        """Test: Debounce-Decorator"""
        call_count = 0
        
        @debounce(delay=0.1)
        def debounced_function():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # Mehrere schnelle Aufrufe
        debounced_function()
        debounced_function()
        debounced_function()
        
        # Kurz warten
        time.sleep(0.05)
        assert call_count == 0  # Noch nicht aufgerufen
        
        # Länger warten
        time.sleep(0.1)
        assert call_count == 1  # Nur einmal aufgerufen
    
    @pytest.mark.unit
    def test_throttle(self):
        """Test: Throttle-Decorator"""
        call_count = 0
        
        @throttle(interval=0.1)
        def throttled_function():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # Mehrere schnelle Aufrufe
        result1 = throttled_function()
        result2 = throttled_function()
        result3 = throttled_function()
        
        assert result1 == 1  # Erster Aufruf erfolgreich
        assert result2 is None  # Gedrosselt
        assert result3 is None  # Gedrosselt
        
        # Nach Intervall sollte wieder funktionieren
        time.sleep(0.15)
        result4 = throttled_function()
        assert result4 == 2


class TestValidationUtilities:
    """Tests für Validation-Utilities"""
    
    @pytest.mark.unit
    def test_validate_schema(self):
        """Test: Schema-Validierung"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        # Gültige Daten
        valid_data = {"name": "John", "age": 30}
        result = validate_schema(valid_data, schema)
        assert result.is_valid == True
        assert len(result.errors) == 0
        
        # Ungültige Daten
        invalid_data = {"age": -5}
        result = validate_schema(invalid_data, schema)
        assert result.is_valid == False
        assert len(result.errors) > 0
    
    @pytest.mark.unit
    def test_validate_range(self):
        """Test: Range-Validierung"""
        # Gültige Werte
        assert validate_range(5, min_val=0, max_val=10) == True
        assert validate_range(0, min_val=0, max_val=10) == True
        assert validate_range(10, min_val=0, max_val=10) == True
        
        # Ungültige Werte
        assert validate_range(-1, min_val=0, max_val=10) == False
        assert validate_range(11, min_val=0, max_val=10) == False
        
        # Nur Minimum
        assert validate_range(5, min_val=0) == True
        assert validate_range(-1, min_val=0) == False
        
        # Nur Maximum
        assert validate_range(5, max_val=10) == True
        assert validate_range(15, max_val=10) == False
    
    @pytest.mark.unit
    def test_validate_type(self):
        """Test: Typ-Validierung"""
        # Korrekte Typen
        assert validate_type("hello", str) == True
        assert validate_type(42, int) == True
        assert validate_type(3.14, float) == True
        assert validate_type([1, 2, 3], list) == True
        assert validate_type({"key": "value"}, dict) == True
        
        # Falsche Typen
        assert validate_type("hello", int) == False
        assert validate_type(42, str) == False
        
        # Multiple Typen
        assert validate_type(42, (int, float)) == True
        assert validate_type(3.14, (int, float)) == True
        assert validate_type("hello", (int, float)) == False
    
    @pytest.mark.unit
    def test_validate_required_fields(self):
        """Test: Required-Fields-Validierung"""
        data = {
            "name": "John",
            "email": "john@example.com",
            "age": 30
        }
        
        # Alle erforderlichen Felder vorhanden
        required_fields = ["name", "email"]
        result = validate_required_fields(data, required_fields)
        assert result.is_valid == True
        assert len(result.missing_fields) == 0
        
        # Fehlende Felder
        required_fields = ["name", "email", "phone"]
        result = validate_required_fields(data, required_fields)
        assert result.is_valid == False
        assert "phone" in result.missing_fields
    
    @pytest.mark.unit
    def test_validation_result(self):
        """Test: ValidationResult-Klasse"""
        # Erfolgreiche Validierung
        success_result = ValidationResult(is_valid=True)
        assert success_result.is_valid == True
        assert len(success_result.errors) == 0
        assert success_result.summary() == "Validation successful"
        
        # Fehlgeschlagene Validierung
        error_result = ValidationResult(
            is_valid=False,
            errors=["Field 'name' is required", "Field 'age' must be positive"]
        )
        assert error_result.is_valid == False
        assert len(error_result.errors) == 2
        assert "2 validation errors" in error_result.summary()


class TestPerformanceUtilities:
    """Tests für Performance-Utilities"""
    
    @pytest.mark.unit
    def test_timer(self):
        """Test: Timer-Klasse"""
        timer = Timer()
        
        timer.start()
        time.sleep(0.1)
        elapsed = timer.stop()
        
        assert elapsed >= 0.1
        assert elapsed < 0.2  # Sollte nicht viel länger dauern
        
        # Context-Manager
        with Timer() as timer:
            time.sleep(0.05)
        
        assert timer.elapsed >= 0.05
    
    @pytest.mark.unit
    def test_memory_profiler(self):
        """Test: Memory-Profiler"""
        profiler = MemoryProfiler()
        
        profiler.start()
        
        # Speicher allokieren
        large_list = [i for i in range(100000)]
        
        memory_usage = profiler.stop()
        
        assert memory_usage["peak_memory"] > 0
        assert memory_usage["memory_delta"] >= 0
        
        # Context-Manager
        with MemoryProfiler() as profiler:
            another_list = [i for i in range(50000)]
        
        assert profiler.peak_memory > 0
    
    @pytest.mark.unit
    def test_performance_monitor(self):
        """Test: Performance-Monitor-Decorator"""
        @performance_monitor()
        def monitored_function(n):
            return sum(range(n))
        
        result = monitored_function(1000)
        
        assert result == sum(range(1000))
        # Performance-Daten sollten geloggt werden
    
    @pytest.mark.unit
    def test_cache_result(self):
        """Test: Cache-Result-Decorator"""
        call_count = 0
        
        @cache_result(max_size=100, ttl=60)
        def cached_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # Erste Aufrufe
        result1 = cached_function(5)
        result2 = cached_function(5)  # Sollte aus Cache kommen
        result3 = cached_function(10)  # Neuer Wert
        
        assert result1 == 10
        assert result2 == 10
        assert result3 == 20
        assert call_count == 2  # Nur 2 echte Aufrufe
    
    @pytest.mark.unit
    def test_memoize(self):
        """Test: Memoize-Decorator"""
        call_count = 0
        
        @memoize
        def fibonacci(n):
            nonlocal call_count
            call_count += 1
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        result = fibonacci(10)
        
        assert result == 55  # 10. Fibonacci-Zahl
        # Durch Memoization sollten deutlich weniger Aufrufe nötig sein
        assert call_count <= 11  # Maximal n+1 Aufrufe
    
    @pytest.mark.unit
    def test_rate_limit(self):
        """Test: Rate-Limit-Decorator"""
        call_count = 0
        
        @rate_limit(max_calls=2, time_window=1.0)
        def rate_limited_function():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # Erste 2 Aufrufe sollten funktionieren
        result1 = rate_limited_function()
        result2 = rate_limited_function()
        
        assert result1 == 1
        assert result2 == 2
        
        # 3. Aufruf sollte blockiert werden
        with pytest.raises(Exception):  # Rate limit exceeded
            rate_limited_function()


class TestErrorUtilities:
    """Tests für Error-Utilities"""
    
    @pytest.mark.unit
    def test_safe_execute(self):
        """Test: Safe-Execute"""
        # Erfolgreiche Ausführung
        def successful_function():
            return "success"
        
        result = safe_execute(successful_function)
        assert result.success == True
        assert result.result == "success"
        assert result.error is None
        
        # Fehlgeschlagene Ausführung
        def failing_function():
            raise ValueError("Test error")
        
        result = safe_execute(failing_function)
        assert result.success == False
        assert result.result is None
        assert isinstance(result.error, ValueError)
        
        # Mit Default-Wert
        result = safe_execute(failing_function, default="default_value")
        assert result.success == False
        assert result.result == "default_value"
    
    @pytest.mark.unit
    def test_error_handler(self):
        """Test: Error-Handler-Decorator"""
        @error_handler(default_return="error_occurred")
        def function_with_error():
            raise ValueError("Test error")
        
        result = function_with_error()
        assert result == "error_occurred"
        
        # Mit Custom-Handler
        def custom_handler(error):
            return f"Handled: {str(error)}"
        
        @error_handler(handler=custom_handler)
        def another_function():
            raise ValueError("Another error")
        
        result = another_function()
        assert result == "Handled: Another error"
    
    @pytest.mark.unit
    def test_exception_to_dict(self):
        """Test: Exception zu Dictionary"""
        try:
            raise ValueError("Test exception")
        except Exception as e:
            error_dict = exception_to_dict(e)
        
        assert error_dict["type"] == "ValueError"
        assert error_dict["message"] == "Test exception"
        assert "traceback" in error_dict
        assert "timestamp" in error_dict
    
    @pytest.mark.unit
    def test_log_exception(self):
        """Test: Exception loggen"""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            try:
                raise ValueError("Test exception")
            except Exception as e:
                log_exception(e, context={"operation": "test"})
            
            mock_logger.error.assert_called_once()
    
    @pytest.mark.unit
    def test_create_error_response(self):
        """Test: Error-Response erstellen"""
        error_response = create_error_response(
            message="Test error",
            error_code="TEST_001",
            status_code=400,
            details={"field": "value"}
        )
        
        assert error_response["error"]["message"] == "Test error"
        assert error_response["error"]["code"] == "TEST_001"
        assert error_response["status_code"] == 400
        assert error_response["error"]["details"]["field"] == "value"
        assert "timestamp" in error_response["error"]


class TestUtilsIntegration:
    """Integrationstests für Utils"""
    
    @pytest.mark.integration
    def test_file_processing_workflow(self):
        """Test: Vollständiger File-Processing-Workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Verzeichnisstruktur erstellen
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            
            ensure_directory(input_dir)
            ensure_directory(output_dir)
            
            # Test-Dateien erstellen
            test_files = ["audio1.wav", "audio2.mp3", "document.txt"]
            for file in test_files:
                file_path = os.path.join(input_dir, file)
                with open(file_path, 'w') as f:
                    f.write(f"Content of {file}")
            
            # Audio-Dateien finden
            audio_files = find_files(input_dir, pattern="*.wav") + find_files(input_dir, pattern="*.mp3")
            assert len(audio_files) == 2
            
            # Dateien verarbeiten und kopieren
            for audio_file in audio_files:
                if validate_audio_format(audio_file):
                    filename = os.path.basename(audio_file)
                    safe_filename = sanitize_filename(filename)
                    output_path = os.path.join(output_dir, safe_filename)
                    copy_file(audio_file, output_path)
            
            # Ergebnis prüfen
            output_files = list_files(output_dir)
            assert len(output_files) == 2
    
    @pytest.mark.performance
    def test_utils_performance(self):
        """Test: Performance der Utils"""
        # Große Datenmengen testen
        large_data = {f"key_{i}": f"value_{i}" for i in range(10000)}
        
        # Serialisierung
        start_time = time.time()
        serialized = serialize_data(large_data)
        serialization_time = time.time() - start_time
        
        # Deserialisierung
        start_time = time.time()
        deserialized = deserialize_data(serialized)
        deserialization_time = time.time() - start_time
        
        assert len(deserialized) == 10000
        assert serialization_time < 1.0  # Sollte unter 1 Sekunde dauern
        assert deserialization_time < 1.0
        
        # Hash-Performance
        start_time = time.time()
        data_hash = hash_data(large_data)
        hash_time = time.time() - start_time
        
        assert len(data_hash) == 64  # SHA-256
        assert hash_time < 0.5  # Sollte unter 0.5 Sekunden dauern