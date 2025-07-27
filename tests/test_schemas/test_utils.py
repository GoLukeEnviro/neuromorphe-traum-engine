"""Tests für Utility-Schemas"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock
from enum import Enum
from pathlib import Path

from src.schemas.utils import (
    BaseModel, TimestampMixin, UUIDMixin, MetadataMixin,
    PaginationMixin, ValidationMixin, CacheMixin,
    FileInfo, AudioInfo, ProcessingInfo, ErrorInfo,
    PerformanceMetrics, SystemMetrics, ResourceUsage,
    TaskInfo, JobInfo, QueueInfo, StatusInfo,
    ConfigInfo, EnvironmentInfo, VersionInfo,
    LogEntry, AuditEntry, SecurityEntry,
    FilterCriteria, SortCriteria, SearchCriteria,
    ExportOptions, ImportOptions, BackupOptions,
    NotificationSettings, UserPreferences, SystemSettings
)
from src.core.exceptions import ValidationError


class TestBaseModel:
    """Tests für BaseModel-Schema"""
    
    @pytest.mark.unit
    def test_base_model_creation(self):
        """Test: BaseModel-Erstellung"""
        # Erstelle eine einfache Klasse, die von BaseModel erbt
        class TestModel(BaseModel):
            name: str
            value: int
            active: bool = True
        
        model = TestModel(name="test", value=42)
        
        assert model.name == "test"
        assert model.value == 42
        assert model.active == True
    
    @pytest.mark.unit
    def test_base_model_serialization(self):
        """Test: BaseModel-Serialisierung"""
        class TestModel(BaseModel):
            name: str
            data: Dict[str, Any]
        
        model = TestModel(
            name="test_model",
            data={"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        )
        
        # Zu Dict
        model_dict = model.dict()
        assert model_dict["name"] == "test_model"
        assert model_dict["data"]["key1"] == "value1"
        
        # JSON-Serialisierung
        model_json = model.json()
        assert "test_model" in model_json
        assert "key1" in model_json
    
    @pytest.mark.unit
    def test_base_model_validation(self):
        """Test: BaseModel-Validierung"""
        class TestModel(BaseModel):
            email: str
            age: int
            
            @validator('email')
            def validate_email(cls, v):
                if '@' not in v:
                    raise ValueError('Invalid email format')
                return v
            
            @validator('age')
            def validate_age(cls, v):
                if v < 0 or v > 150:
                    raise ValueError('Age must be between 0 and 150')
                return v
        
        # Gültige Daten
        valid_model = TestModel(email="test@example.com", age=25)
        assert valid_model.email == "test@example.com"
        
        # Ungültige Email
        with pytest.raises(ValidationError):
            TestModel(email="invalid_email", age=25)
        
        # Ungültiges Alter
        with pytest.raises(ValidationError):
            TestModel(email="test@example.com", age=200)


class TestTimestampMixin:
    """Tests für TimestampMixin-Schema"""
    
    @pytest.mark.unit
    def test_timestamp_mixin_basic(self):
        """Test: Grundlegendes TimestampMixin"""
        class TestModel(BaseModel, TimestampMixin):
            name: str
        
        model = TestModel(name="test")
        
        assert hasattr(model, 'created_at')
        assert hasattr(model, 'updated_at')
        assert isinstance(model.created_at, datetime)
        assert isinstance(model.updated_at, datetime)
        assert model.created_at == model.updated_at
    
    @pytest.mark.unit
    def test_timestamp_mixin_update(self):
        """Test: TimestampMixin-Update"""
        class TestModel(BaseModel, TimestampMixin):
            name: str
            
            def update_name(self, new_name: str):
                self.name = new_name
                self.touch()  # Aktualisiert updated_at
        
        model = TestModel(name="original")
        original_updated_at = model.updated_at
        
        # Kurz warten, dann aktualisieren
        import time
        time.sleep(0.01)
        
        model.update_name("updated")
        
        assert model.name == "updated"
        assert model.updated_at > original_updated_at
        assert model.created_at < model.updated_at


class TestUUIDMixin:
    """Tests für UUIDMixin-Schema"""
    
    @pytest.mark.unit
    def test_uuid_mixin_basic(self):
        """Test: Grundlegendes UUIDMixin"""
        class TestModel(BaseModel, UUIDMixin):
            name: str
        
        model = TestModel(name="test")
        
        assert hasattr(model, 'id')
        assert isinstance(model.id, str)
        assert len(model.id) == 36  # UUID4-Format
        assert model.id.count('-') == 4
    
    @pytest.mark.unit
    def test_uuid_mixin_uniqueness(self):
        """Test: UUIDMixin-Eindeutigkeit"""
        class TestModel(BaseModel, UUIDMixin):
            name: str
        
        model1 = TestModel(name="test1")
        model2 = TestModel(name="test2")
        
        assert model1.id != model2.id
        
        # Viele Modelle erstellen und Eindeutigkeit prüfen
        ids = set()
        for i in range(1000):
            model = TestModel(name=f"test_{i}")
            assert model.id not in ids
            ids.add(model.id)


class TestMetadataMixin:
    """Tests für MetadataMixin-Schema"""
    
    @pytest.mark.unit
    def test_metadata_mixin_basic(self):
        """Test: Grundlegendes MetadataMixin"""
        class TestModel(BaseModel, MetadataMixin):
            name: str
        
        model = TestModel(name="test")
        
        assert hasattr(model, 'metadata')
        assert isinstance(model.metadata, dict)
        assert len(model.metadata) == 0
    
    @pytest.mark.unit
    def test_metadata_mixin_operations(self):
        """Test: MetadataMixin-Operationen"""
        class TestModel(BaseModel, MetadataMixin):
            name: str
        
        model = TestModel(name="test")
        
        # Metadaten hinzufügen
        model.set_metadata("key1", "value1")
        model.set_metadata("key2", 123)
        model.set_metadata("key3", {"nested": "data"})
        
        assert model.get_metadata("key1") == "value1"
        assert model.get_metadata("key2") == 123
        assert model.get_metadata("key3")["nested"] == "data"
        assert model.get_metadata("nonexistent") is None
        assert model.get_metadata("nonexistent", "default") == "default"
        
        # Metadaten entfernen
        model.remove_metadata("key2")
        assert model.get_metadata("key2") is None
        
        # Alle Metadaten löschen
        model.clear_metadata()
        assert len(model.metadata) == 0


class TestPaginationMixin:
    """Tests für PaginationMixin-Schema"""
    
    @pytest.mark.unit
    def test_pagination_mixin_basic(self):
        """Test: Grundlegendes PaginationMixin"""
        class TestModel(BaseModel, PaginationMixin):
            items: List[str]
        
        model = TestModel(
            items=["item1", "item2", "item3"],
            page=1,
            page_size=10,
            total_count=25
        )
        
        assert model.page == 1
        assert model.page_size == 10
        assert model.total_count == 25
        assert model.total_pages == 3
        assert model.has_next == True
        assert model.has_previous == False
    
    @pytest.mark.unit
    def test_pagination_mixin_calculations(self):
        """Test: PaginationMixin-Berechnungen"""
        class TestModel(BaseModel, PaginationMixin):
            items: List[str]
        
        # Mittlere Seite
        model = TestModel(
            items=["item1", "item2"],
            page=3,
            page_size=5,
            total_count=22
        )
        
        assert model.total_pages == 5  # ceil(22/5)
        assert model.has_next == True
        assert model.has_previous == True
        assert model.offset == 10  # (3-1) * 5
        
        # Letzte Seite
        last_page_model = TestModel(
            items=["item1", "item2"],
            page=5,
            page_size=5,
            total_count=22
        )
        
        assert last_page_model.has_next == False
        assert last_page_model.has_previous == True


class TestFileInfo:
    """Tests für FileInfo-Schema"""
    
    @pytest.mark.unit
    def test_file_info_basic(self):
        """Test: Grundlegende FileInfo"""
        file_data = {
            "filename": "test_audio.wav",
            "file_path": "/uploads/test_audio.wav",
            "file_size": 10485760,  # 10 MB
            "mime_type": "audio/wav",
            "checksum": "abc123def456",
            "created_at": datetime.now()
        }
        
        file_info = FileInfo(**file_data)
        
        assert file_info.filename == "test_audio.wav"
        assert file_info.file_size == 10485760
        assert file_info.mime_type == "audio/wav"
        assert file_info.checksum == "abc123def456"
    
    @pytest.mark.unit
    def test_file_info_with_metadata(self):
        """Test: FileInfo mit Metadaten"""
        file_data = {
            "filename": "complex_audio.flac",
            "file_path": "/storage/audio/complex_audio.flac",
            "file_size": 52428800,  # 50 MB
            "mime_type": "audio/flac",
            "metadata": {
                "duration": 180.5,
                "sample_rate": 44100,
                "bit_depth": 24,
                "channels": 2,
                "codec": "FLAC",
                "bitrate": 1411
            },
            "tags": ["high_quality", "lossless", "stereo"]
        }
        
        file_info = FileInfo(**file_data)
        
        assert file_info.metadata["duration"] == 180.5
        assert file_info.metadata["sample_rate"] == 44100
        assert "high_quality" in file_info.tags
        assert len(file_info.tags) == 3
    
    @pytest.mark.unit
    def test_file_info_validation(self):
        """Test: FileInfo-Validierung"""
        # Ungültige Dateigröße
        with pytest.raises(ValidationError):
            FileInfo(
                filename="test.wav",
                file_path="/test.wav",
                file_size=-1,  # Muss >= 0 sein
                mime_type="audio/wav"
            )
        
        # Ungültiger MIME-Type
        with pytest.raises(ValidationError):
            FileInfo(
                filename="test.txt",
                file_path="/test.txt",
                file_size=1000,
                mime_type="text/plain"  # Nur Audio-MIME-Types erlaubt
            )


class TestAudioInfo:
    """Tests für AudioInfo-Schema"""
    
    @pytest.mark.unit
    def test_audio_info_basic(self):
        """Test: Grundlegende AudioInfo"""
        audio_data = {
            "duration": 180.5,
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 2,
            "format": "WAV",
            "bitrate": 1411
        }
        
        audio_info = AudioInfo(**audio_data)
        
        assert audio_info.duration == 180.5
        assert audio_info.sample_rate == 44100
        assert audio_info.bit_depth == 16
        assert audio_info.channels == 2
        assert audio_info.format == "WAV"
    
    @pytest.mark.unit
    def test_audio_info_detailed(self):
        """Test: Detaillierte AudioInfo"""
        audio_data = {
            "duration": 240.0,
            "sample_rate": 48000,
            "bit_depth": 24,
            "channels": 2,
            "format": "FLAC",
            "bitrate": 2304,
            "codec": "FLAC",
            "quality": "lossless",
            "peak_level": -3.2,
            "rms_level": -18.5,
            "dynamic_range": 12.8,
            "spectral_features": {
                "centroid": 2500.0,
                "rolloff": 8000.0,
                "zero_crossing_rate": 0.15
            },
            "tempo": 128.5,
            "key": "Am",
            "energy": 7.2
        }
        
        audio_info = AudioInfo(**audio_data)
        
        assert audio_info.codec == "FLAC"
        assert audio_info.quality == "lossless"
        assert audio_info.peak_level == -3.2
        assert audio_info.spectral_features["centroid"] == 2500.0
        assert audio_info.tempo == 128.5
        assert audio_info.key == "Am"
    
    @pytest.mark.unit
    def test_audio_info_validation(self):
        """Test: AudioInfo-Validierung"""
        # Ungültige Duration
        with pytest.raises(ValidationError):
            AudioInfo(
                duration=-10.0,  # Muss >= 0 sein
                sample_rate=44100,
                bit_depth=16,
                channels=2,
                format="WAV"
            )
        
        # Ungültige Sample-Rate
        with pytest.raises(ValidationError):
            AudioInfo(
                duration=180.0,
                sample_rate=100,  # Zu niedrig
                bit_depth=16,
                channels=2,
                format="WAV"
            )
        
        # Ungültige Kanäle
        with pytest.raises(ValidationError):
            AudioInfo(
                duration=180.0,
                sample_rate=44100,
                bit_depth=16,
                channels=0,  # Muss > 0 sein
                format="WAV"
            )


class TestProcessingInfo:
    """Tests für ProcessingInfo-Schema"""
    
    @pytest.mark.unit
    def test_processing_info_basic(self):
        """Test: Grundlegende ProcessingInfo"""
        processing_data = {
            "task_id": "task_123",
            "status": "processing",
            "progress": 65.5,
            "started_at": datetime.now() - timedelta(minutes=5),
            "estimated_completion": datetime.now() + timedelta(minutes=2)
        }
        
        processing_info = ProcessingInfo(**processing_data)
        
        assert processing_info.task_id == "task_123"
        assert processing_info.status == "processing"
        assert processing_info.progress == 65.5
        assert isinstance(processing_info.started_at, datetime)
    
    @pytest.mark.unit
    def test_processing_info_detailed(self):
        """Test: Detaillierte ProcessingInfo"""
        processing_data = {
            "task_id": "task_456",
            "status": "completed",
            "progress": 100.0,
            "started_at": datetime.now() - timedelta(minutes=10),
            "completed_at": datetime.now(),
            "processing_time": 600.5,
            "stage": "finalization",
            "stage_progress": 100.0,
            "throughput": 2.5,
            "resource_usage": {
                "cpu_usage": 75.2,
                "memory_usage": 2048,
                "disk_io": 15.8
            },
            "errors": [],
            "warnings": ["High memory usage detected"],
            "metrics": {
                "items_processed": 100,
                "items_failed": 0,
                "average_processing_time": 6.0
            }
        }
        
        processing_info = ProcessingInfo(**processing_data)
        
        assert processing_info.processing_time == 600.5
        assert processing_info.stage == "finalization"
        assert processing_info.throughput == 2.5
        assert processing_info.resource_usage["cpu_usage"] == 75.2
        assert len(processing_info.errors) == 0
        assert len(processing_info.warnings) == 1
        assert processing_info.metrics["items_processed"] == 100


class TestPerformanceMetrics:
    """Tests für PerformanceMetrics-Schema"""
    
    @pytest.mark.unit
    def test_performance_metrics_basic(self):
        """Test: Grundlegende PerformanceMetrics"""
        metrics_data = {
            "timestamp": datetime.now(),
            "response_time": 0.125,
            "throughput": 150.5,
            "error_rate": 0.02,
            "success_rate": 0.98
        }
        
        metrics = PerformanceMetrics(**metrics_data)
        
        assert metrics.response_time == 0.125
        assert metrics.throughput == 150.5
        assert metrics.error_rate == 0.02
        assert metrics.success_rate == 0.98
    
    @pytest.mark.unit
    def test_performance_metrics_detailed(self):
        """Test: Detaillierte PerformanceMetrics"""
        metrics_data = {
            "timestamp": datetime.now(),
            "response_time": 0.250,
            "response_time_p50": 0.150,
            "response_time_p95": 0.450,
            "response_time_p99": 0.750,
            "throughput": 200.0,
            "requests_per_second": 100.0,
            "error_rate": 0.01,
            "success_rate": 0.99,
            "cache_hit_rate": 0.85,
            "active_connections": 45,
            "queue_depth": 12,
            "resource_utilization": {
                "cpu_usage": 65.5,
                "memory_usage": 78.2,
                "disk_usage": 45.0,
                "network_io": {"in": 1024, "out": 2048}
            }
        }
        
        metrics = PerformanceMetrics(**metrics_data)
        
        assert metrics.response_time_p95 == 0.450
        assert metrics.requests_per_second == 100.0
        assert metrics.cache_hit_rate == 0.85
        assert metrics.active_connections == 45
        assert metrics.resource_utilization["cpu_usage"] == 65.5


class TestTaskInfo:
    """Tests für TaskInfo-Schema"""
    
    @pytest.mark.unit
    def test_task_info_basic(self):
        """Test: Grundlegende TaskInfo"""
        task_data = {
            "task_id": "task_123",
            "task_type": "audio_analysis",
            "status": "running",
            "priority": "normal",
            "created_at": datetime.now() - timedelta(minutes=5),
            "started_at": datetime.now() - timedelta(minutes=3)
        }
        
        task_info = TaskInfo(**task_data)
        
        assert task_info.task_id == "task_123"
        assert task_info.task_type == "audio_analysis"
        assert task_info.status == "running"
        assert task_info.priority == "normal"
    
    @pytest.mark.unit
    def test_task_info_with_dependencies(self):
        """Test: TaskInfo mit Abhängigkeiten"""
        task_data = {
            "task_id": "task_456",
            "task_type": "arrangement_render",
            "status": "pending",
            "priority": "high",
            "dependencies": ["task_123", "task_789"],
            "parameters": {
                "arrangement_id": "arr_abc",
                "output_format": "wav",
                "quality": "high"
            },
            "retry_count": 0,
            "max_retries": 3,
            "timeout": 3600
        }
        
        task_info = TaskInfo(**task_data)
        
        assert "task_123" in task_info.dependencies
        assert task_info.parameters["arrangement_id"] == "arr_abc"
        assert task_info.max_retries == 3
        assert task_info.timeout == 3600


class TestLogEntry:
    """Tests für LogEntry-Schema"""
    
    @pytest.mark.unit
    def test_log_entry_basic(self):
        """Test: Grundlegende LogEntry"""
        log_data = {
            "timestamp": datetime.now(),
            "level": "INFO",
            "logger": "neuromorphe.services.clap",
            "message": "Audio analysis completed successfully",
            "module": "clap_service",
            "function": "analyze_audio"
        }
        
        log_entry = LogEntry(**log_data)
        
        assert log_entry.level == "INFO"
        assert log_entry.logger == "neuromorphe.services.clap"
        assert "successfully" in log_entry.message
        assert log_entry.module == "clap_service"
    
    @pytest.mark.unit
    def test_log_entry_with_context(self):
        """Test: LogEntry mit Kontext"""
        log_data = {
            "timestamp": datetime.now(),
            "level": "ERROR",
            "logger": "neuromorphe.services.render",
            "message": "Render job failed due to insufficient memory",
            "module": "render_service",
            "function": "process_arrangement",
            "line_number": 245,
            "thread_id": "thread_123",
            "process_id": 1234,
            "user_id": "user_456",
            "session_id": "session_789",
            "request_id": "req_abc",
            "context": {
                "job_id": "job_def",
                "arrangement_id": "arr_ghi",
                "memory_usage": 8192,
                "memory_limit": 4096
            },
            "exception": {
                "type": "MemoryError",
                "message": "Out of memory",
                "traceback": "Traceback (most recent call last)..."
            },
            "tags": ["render", "memory", "error"]
        }
        
        log_entry = LogEntry(**log_data)
        
        assert log_entry.level == "ERROR"
        assert log_entry.line_number == 245
        assert log_entry.context["job_id"] == "job_def"
        assert log_entry.exception["type"] == "MemoryError"
        assert "render" in log_entry.tags


class TestFilterCriteria:
    """Tests für FilterCriteria-Schema"""
    
    @pytest.mark.unit
    def test_filter_criteria_basic(self):
        """Test: Grundlegende FilterCriteria"""
        filter_data = {
            "field": "genre",
            "operator": "eq",
            "value": "techno"
        }
        
        filter_criteria = FilterCriteria(**filter_data)
        
        assert filter_criteria.field == "genre"
        assert filter_criteria.operator == "eq"
        assert filter_criteria.value == "techno"
    
    @pytest.mark.unit
    def test_filter_criteria_complex(self):
        """Test: Komplexe FilterCriteria"""
        filter_data = {
            "field": "bpm",
            "operator": "between",
            "value": [120, 140],
            "case_sensitive": False,
            "negate": False
        }
        
        filter_criteria = FilterCriteria(**filter_data)
        
        assert filter_criteria.field == "bpm"
        assert filter_criteria.operator == "between"
        assert filter_criteria.value == [120, 140]
        assert filter_criteria.case_sensitive == False
    
    @pytest.mark.unit
    def test_filter_criteria_validation(self):
        """Test: FilterCriteria-Validierung"""
        # Ungültiger Operator
        with pytest.raises(ValidationError):
            FilterCriteria(
                field="test",
                operator="invalid_operator",
                value="test"
            )
        
        # Leeres Field
        with pytest.raises(ValidationError):
            FilterCriteria(
                field="",
                operator="eq",
                value="test"
            )


class TestExportOptions:
    """Tests für ExportOptions-Schema"""
    
    @pytest.mark.unit
    def test_export_options_basic(self):
        """Test: Grundlegende ExportOptions"""
        export_data = {
            "format": "json",
            "include_metadata": True,
            "compress": False
        }
        
        export_options = ExportOptions(**export_data)
        
        assert export_options.format == "json"
        assert export_options.include_metadata == True
        assert export_options.compress == False
    
    @pytest.mark.unit
    def test_export_options_detailed(self):
        """Test: Detaillierte ExportOptions"""
        export_data = {
            "format": "csv",
            "include_metadata": True,
            "include_embeddings": False,
            "compress": True,
            "compression_format": "gzip",
            "fields": ["id", "name", "genre", "bpm"],
            "filters": {
                "genre": ["techno", "house"],
                "created_after": "2024-01-01"
            },
            "batch_size": 1000,
            "output_path": "/exports/stems_export.csv.gz"
        }
        
        export_options = ExportOptions(**export_data)
        
        assert export_options.format == "csv"
        assert export_options.compression_format == "gzip"
        assert "id" in export_options.fields
        assert export_options.filters["genre"] == ["techno", "house"]
        assert export_options.batch_size == 1000
    
    @pytest.mark.unit
    def test_export_options_validation(self):
        """Test: ExportOptions-Validierung"""
        # Ungültiges Format
        with pytest.raises(ValidationError):
            ExportOptions(
                format="invalid_format",
                include_metadata=True
            )
        
        # Ungültige Batch-Size
        with pytest.raises(ValidationError):
            ExportOptions(
                format="json",
                batch_size=0  # Muss > 0 sein
            )


class TestUserPreferences:
    """Tests für UserPreferences-Schema"""
    
    @pytest.mark.unit
    def test_user_preferences_basic(self):
        """Test: Grundlegende UserPreferences"""
        preferences_data = {
            "user_id": "user_123",
            "theme": "dark",
            "language": "en",
            "timezone": "Europe/Berlin"
        }
        
        preferences = UserPreferences(**preferences_data)
        
        assert preferences.user_id == "user_123"
        assert preferences.theme == "dark"
        assert preferences.language == "en"
        assert preferences.timezone == "Europe/Berlin"
    
    @pytest.mark.unit
    def test_user_preferences_detailed(self):
        """Test: Detaillierte UserPreferences"""
        preferences_data = {
            "user_id": "user_456",
            "theme": "light",
            "language": "de",
            "timezone": "Europe/Berlin",
            "audio_settings": {
                "default_sample_rate": 48000,
                "default_bit_depth": 24,
                "default_format": "FLAC",
                "auto_normalize": True
            },
            "ui_settings": {
                "show_waveforms": True,
                "show_spectrograms": False,
                "auto_save_interval": 300,
                "grid_snap": True
            },
            "notification_settings": {
                "email_notifications": True,
                "push_notifications": False,
                "render_completion": True,
                "system_alerts": True
            },
            "privacy_settings": {
                "analytics_enabled": False,
                "crash_reporting": True,
                "usage_statistics": False
            }
        }
        
        preferences = UserPreferences(**preferences_data)
        
        assert preferences.audio_settings["default_sample_rate"] == 48000
        assert preferences.ui_settings["show_waveforms"] == True
        assert preferences.notification_settings["email_notifications"] == True
        assert preferences.privacy_settings["analytics_enabled"] == False


class TestUtilSchemasIntegration:
    """Integrationstests für Utility-Schemas"""
    
    @pytest.mark.integration
    def test_combined_mixins(self):
        """Test: Kombinierte Mixins"""
        class CompleteModel(BaseModel, UUIDMixin, TimestampMixin, MetadataMixin):
            name: str
            value: int
        
        model = CompleteModel(name="test", value=42)
        
        # UUID-Funktionalität
        assert hasattr(model, 'id')
        assert isinstance(model.id, str)
        
        # Timestamp-Funktionalität
        assert hasattr(model, 'created_at')
        assert hasattr(model, 'updated_at')
        
        # Metadata-Funktionalität
        assert hasattr(model, 'metadata')
        model.set_metadata("test_key", "test_value")
        assert model.get_metadata("test_key") == "test_value"
        
        # Serialisierung
        model_dict = model.dict()
        assert "id" in model_dict
        assert "created_at" in model_dict
        assert "metadata" in model_dict
    
    @pytest.mark.integration
    def test_file_processing_workflow(self):
        """Test: Datei-Verarbeitungs-Workflow"""
        # 1. Datei-Info erstellen
        file_info = FileInfo(
            filename="test_audio.wav",
            file_path="/uploads/test_audio.wav",
            file_size=10485760,
            mime_type="audio/wav",
            checksum="abc123"
        )
        
        # 2. Audio-Info hinzufügen
        audio_info = AudioInfo(
            duration=180.0,
            sample_rate=44100,
            bit_depth=16,
            channels=2,
            format="WAV"
        )
        
        # 3. Processing-Info erstellen
        processing_info = ProcessingInfo(
            task_id="task_123",
            status="processing",
            progress=50.0,
            started_at=datetime.now()
        )
        
        # 4. Task-Info erstellen
        task_info = TaskInfo(
            task_id="task_123",
            task_type="audio_analysis",
            status="running",
            priority="normal",
            parameters={
                "file_path": file_info.file_path,
                "sample_rate": audio_info.sample_rate
            }
        )
        
        # Validierung des Workflows
        assert file_info.filename == "test_audio.wav"
        assert audio_info.duration == 180.0
        assert processing_info.task_id == task_info.task_id
        assert task_info.parameters["file_path"] == file_info.file_path
    
    @pytest.mark.performance
    def test_utility_schemas_performance(self):
        """Test: Performance der Utility-Schemas"""
        import time
        
        # Viele Modelle mit Mixins erstellen
        start_time = time.time()
        
        class TestModel(BaseModel, UUIDMixin, TimestampMixin, MetadataMixin):
            name: str
            value: int
        
        models = []
        for i in range(1000):
            model = TestModel(name=f"test_{i}", value=i)
            model.set_metadata("index", i)
            models.append(model)
        
        creation_time = time.time() - start_time
        
        assert len(models) == 1000
        assert creation_time < 3.0  # Sollte unter 3 Sekunden dauern
        
        # Serialisierung testen
        start_time = time.time()
        
        serialized = [model.dict() for model in models[:100]]
        
        serialization_time = time.time() - start_time
        
        assert len(serialized) == 100
        assert serialization_time < 1.0  # Sollte unter 1 Sekunde dauern