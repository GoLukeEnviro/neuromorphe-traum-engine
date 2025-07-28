"""Tests für Render-Schemas"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock
from enum import Enum

from src.schemas.schemas import (
    RenderOutput, RenderFormat, RenderQuality, RenderStatus,
    RenderRequest, RenderProgress,
    RenderJobBase, RenderJobCreate, RenderJobUpdate, RenderJobResponse,
    RenderPriority, RenderSettings, RenderError,
    RenderMetrics, RenderBatch, RenderTemplate, RenderQueue
)
from src.core.exceptions import ValidationError


class TestRenderStatus:
    """Tests für RenderStatus-Enum"""
    
    @pytest.mark.unit
    def test_render_status_values(self):
        """Test: RenderStatus-Werte"""
        assert RenderStatus.PENDING.value == "pending"
        assert RenderStatus.QUEUED.value == "queued"
        assert RenderStatus.PROCESSING.value == "processing"
        assert RenderStatus.COMPLETED.value == "completed"
        assert RenderStatus.FAILED.value == "failed"
        assert RenderStatus.CANCELLED.value == "cancelled"
    
    @pytest.mark.unit
    def test_render_status_transitions(self):
        """Test: Gültige Status-Übergänge"""
        # Gültige Übergänge
        valid_transitions = {
            RenderStatus.PENDING: [RenderStatus.QUEUED, RenderStatus.CANCELLED],
            RenderStatus.QUEUED: [RenderStatus.PROCESSING, RenderStatus.CANCELLED],
            RenderStatus.PROCESSING: [RenderStatus.COMPLETED, RenderStatus.FAILED, RenderStatus.CANCELLED],
            RenderStatus.COMPLETED: [],  # Terminal state
            RenderStatus.FAILED: [RenderStatus.QUEUED],  # Retry möglich
            RenderStatus.CANCELLED: [RenderStatus.QUEUED]  # Restart möglich
        }
        
        for current_status, allowed_next in valid_transitions.items():
            assert isinstance(current_status, RenderStatus)
            for next_status in allowed_next:
                assert isinstance(next_status, RenderStatus)


class TestRenderPriority:
    """Tests für RenderPriority-Enum"""
    
    @pytest.mark.unit
    def test_render_priority_values(self):
        """Test: RenderPriority-Werte"""
        assert RenderPriority.LOW.value == "low"
        assert RenderPriority.NORMAL.value == "normal"
        assert RenderPriority.HIGH.value == "high"
        assert RenderPriority.URGENT.value == "urgent"
    
    @pytest.mark.unit
    def test_render_priority_ordering(self):
        """Test: Prioritäts-Reihenfolge"""
        priorities = [RenderPriority.LOW, RenderPriority.NORMAL, RenderPriority.HIGH, RenderPriority.URGENT]
        priority_values = [1, 2, 3, 4]  # Angenommene numerische Werte
        
        for i, priority in enumerate(priorities):
            assert isinstance(priority, RenderPriority)


class TestRenderFormat:
    """Tests für RenderFormat-Enum"""
    
    @pytest.mark.unit
    def test_render_format_values(self):
        """Test: RenderFormat-Werte"""
        assert RenderFormat.WAV.value == "wav"
        assert RenderFormat.MP3.value == "mp3"
        assert RenderFormat.FLAC.value == "flac"
        assert RenderFormat.AAC.value == "aac"
        assert RenderFormat.OGG.value == "ogg"
    
    @pytest.mark.unit
    def test_render_format_properties(self):
        """Test: Format-Eigenschaften"""
        # Lossless Formate
        lossless_formats = [RenderFormat.WAV, RenderFormat.FLAC]
        for format in lossless_formats:
            assert isinstance(format, RenderFormat)
        
        # Lossy Formate
        lossy_formats = [RenderFormat.MP3, RenderFormat.AAC, RenderFormat.OGG]
        for format in lossy_formats:
            assert isinstance(format, RenderFormat)


class TestRenderQuality:
    """Tests für RenderQuality-Enum"""
    
    @pytest.mark.unit
    def test_render_quality_values(self):
        """Test: RenderQuality-Werte"""
        assert RenderQuality.DRAFT.value == "draft"
        assert RenderQuality.STANDARD.value == "standard"
        assert RenderQuality.HIGH.value == "high"
        assert RenderQuality.MASTER.value == "master"
    
    @pytest.mark.unit
    def test_render_quality_settings(self):
        """Test: Quality-Settings"""
        quality_settings = {
            RenderQuality.DRAFT: {"sample_rate": 22050, "bit_depth": 16},
            RenderQuality.STANDARD: {"sample_rate": 44100, "bit_depth": 16},
            RenderQuality.HIGH: {"sample_rate": 48000, "bit_depth": 24},
            RenderQuality.MASTER: {"sample_rate": 96000, "bit_depth": 32}
        }
        
        for quality, settings in quality_settings.items():
            assert isinstance(quality, RenderQuality)
            assert settings["sample_rate"] > 0
            assert settings["bit_depth"] in [16, 24, 32]


class TestRenderJobBase:
    """Tests für RenderJobBase-Schema"""
    
    @pytest.mark.unit
    def test_render_job_base_creation(self):
        """Test: RenderJobBase erstellen"""
        job_data = {
            "arrangement_id": "arrangement_123",
            "format": RenderFormat.WAV,
            "quality": RenderQuality.HIGH,
            "priority": RenderPriority.NORMAL,
            "output_filename": "test_render.wav"
        }
        
        job = RenderJobBase(**job_data)
        
        assert job.arrangement_id == "arrangement_123"
        assert job.format == RenderFormat.WAV
        assert job.quality == RenderQuality.HIGH
        assert job.priority == RenderPriority.NORMAL
        assert job.output_filename == "test_render.wav"
    
    @pytest.mark.unit
    def test_render_job_base_defaults(self):
        """Test: RenderJobBase-Standardwerte"""
        minimal_data = {
            "arrangement_id": "arrangement_456"
        }
        
        job = RenderJobBase(**minimal_data)
        
        assert job.arrangement_id == "arrangement_456"
        assert job.format == RenderFormat.WAV  # Standard
        assert job.quality == RenderQuality.STANDARD  # Standard
        assert job.priority == RenderPriority.NORMAL  # Standard
        assert job.output_filename is None  # Optional
    
    @pytest.mark.unit
    def test_render_job_base_validation(self):
        """Test: RenderJobBase-Validierung"""
        # Arrangement-ID ist erforderlich
        with pytest.raises(ValidationError):
            RenderJobBase()
        
        # Leere Arrangement-ID
        with pytest.raises(ValidationError):
            RenderJobBase(arrangement_id="")
        
        # Ungültiges Format
        with pytest.raises(ValidationError):
            RenderJobBase(arrangement_id="test", format="invalid_format")


class TestRenderJobCreate:
    """Tests für RenderJobCreate-Schema"""
    
    @pytest.mark.unit
    def test_render_job_create_basic(self):
        """Test: Grundlegende RenderJobCreate"""
        create_data = {
            "arrangement_id": "arrangement_789",
            "format": RenderFormat.MP3,
            "quality": RenderQuality.HIGH,
            "priority": RenderPriority.HIGH,
            "notify_on_completion": True,
            "notification_email": "user@example.com"
        }
        
        job_create = RenderJobCreate(**create_data)
        
        assert job_create.arrangement_id == "arrangement_789"
        assert job_create.format == RenderFormat.MP3
        assert job_create.notify_on_completion == True
        assert job_create.notification_email == "user@example.com"
    
    @pytest.mark.unit
    def test_render_job_create_with_settings(self):
        """Test: RenderJobCreate mit erweiterten Einstellungen"""
        create_data = {
            "arrangement_id": "arrangement_abc",
            "custom_settings": {
                "sample_rate": 48000,
                "bit_depth": 24,
                "normalize": True,
                "apply_limiter": True,
                "limiter_threshold": -1.0,
                "fade_in": 2.0,
                "fade_out": 4.0
            },
            "export_stems": True,
            "stem_format": RenderFormat.WAV,
            "include_metadata": True,
            "metadata_tags": {
                "artist": "AI Generator",
                "album": "Generated Tracks",
                "genre": "Techno"
            }
        }
        
        job_create = RenderJobCreate(**create_data)
        
        assert job_create.custom_settings["sample_rate"] == 48000
        assert job_create.export_stems == True
        assert job_create.stem_format == RenderFormat.WAV
        assert job_create.metadata_tags["artist"] == "AI Generator"
    
    @pytest.mark.unit
    def test_render_job_create_validation(self):
        """Test: RenderJobCreate-Validierung"""
        # Ungültige E-Mail
        with pytest.raises(ValidationError):
            RenderJobCreate(
                arrangement_id="test",
                notification_email="invalid_email"
            )
        
        # Ungültige Sample-Rate
        with pytest.raises(ValidationError):
            RenderJobCreate(
                arrangement_id="test",
                custom_settings={"sample_rate": 1000}  # Zu niedrig
            )
        
        # Ungültige Bit-Depth
        with pytest.raises(ValidationError):
            RenderJobCreate(
                arrangement_id="test",
                custom_settings={"bit_depth": 12}  # Nicht unterstützt
            )


class TestRenderJobUpdate:
    """Tests für RenderJobUpdate-Schema"""
    
    @pytest.mark.unit
    def test_render_job_update_status(self):
        """Test: Status-Update"""
        update_data = {
            "status": RenderStatus.PROCESSING,
            "progress": 45.5,
            "current_stage": "mixing",
            "estimated_completion": datetime.now() + timedelta(minutes=10)
        }
        
        job_update = RenderJobUpdate(**update_data)
        
        assert job_update.status == RenderStatus.PROCESSING
        assert job_update.progress == 45.5
        assert job_update.current_stage == "mixing"
        assert isinstance(job_update.estimated_completion, datetime)
    
    @pytest.mark.unit
    def test_render_job_update_priority(self):
        """Test: Prioritäts-Update"""
        update_data = {
            "priority": RenderPriority.URGENT,
            "reason": "User requested priority boost"
        }
        
        job_update = RenderJobUpdate(**update_data)
        
        assert job_update.priority == RenderPriority.URGENT
        assert job_update.reason == "User requested priority boost"
    
    @pytest.mark.unit
    def test_render_job_update_error(self):
        """Test: Error-Update"""
        update_data = {
            "status": RenderStatus.FAILED,
            "error_message": "Audio processing failed",
            "error_code": "AUDIO_PROC_001",
            "retry_count": 2,
            "max_retries": 3
        }
        
        job_update = RenderJobUpdate(**update_data)
        
        assert job_update.status == RenderStatus.FAILED
        assert job_update.error_message == "Audio processing failed"
        assert job_update.retry_count == 2
    
    @pytest.mark.unit
    def test_render_job_update_validation(self):
        """Test: RenderJobUpdate-Validierung"""
        # Ungültiger Progress
        with pytest.raises(ValidationError):
            RenderJobUpdate(progress=150.0)  # Max ist 100.0
        
        with pytest.raises(ValidationError):
            RenderJobUpdate(progress=-10.0)  # Min ist 0.0
        
        # Ungültiger Retry-Count
        with pytest.raises(ValidationError):
            RenderJobUpdate(retry_count=-1)  # Muss >= 0 sein


class TestRenderJobResponse:
    """Tests für RenderJobResponse-Schema"""
    
    @pytest.mark.unit
    def test_render_job_response_complete(self):
        """Test: Vollständige RenderJobResponse"""
        response_data = {
            "id": "job_123",
            "arrangement_id": "arrangement_456",
            "status": RenderStatus.COMPLETED,
            "progress": 100.0,
            "format": RenderFormat.WAV,
            "quality": RenderQuality.HIGH,
            "priority": RenderPriority.NORMAL,
            "created_at": datetime.now(),
            "started_at": datetime.now(),
            "completed_at": datetime.now(),
            "duration": 45.2,
            "file_size": 52428800,  # 50 MB
            "output_path": "/renders/job_123.wav",
            "download_url": "https://api.example.com/renders/job_123/download"
        }
        
        job_response = RenderJobResponse(**response_data)
        
        assert job_response.id == "job_123"
        assert job_response.status == RenderStatus.COMPLETED
        assert job_response.progress == 100.0
        assert job_response.file_size == 52428800
        assert job_response.download_url.startswith("https://")
    
    @pytest.mark.unit
    def test_render_job_response_with_metrics(self):
        """Test: RenderJobResponse mit Metriken"""
        response_data = {
            "id": "job_456",
            "arrangement_id": "arrangement_789",
            "status": RenderStatus.COMPLETED,
            "metrics": {
                "processing_time": 42.5,
                "cpu_usage_avg": 75.2,
                "memory_usage_peak": 2048,
                "stems_processed": 8,
                "effects_applied": 15,
                "mix_complexity": 0.7
            },
            "quality_metrics": {
                "peak_level": -0.1,
                "rms_level": -12.5,
                "dynamic_range": 8.2,
                "thd_n": 0.001,
                "frequency_response_score": 0.95
            }
        }
        
        job_response = RenderJobResponse(**response_data)
        
        assert job_response.metrics["processing_time"] == 42.5
        assert job_response.quality_metrics["peak_level"] == -0.1
        assert job_response.metrics["stems_processed"] == 8
    
    @pytest.mark.unit
    def test_render_job_response_failed(self):
        """Test: Fehlgeschlagene RenderJobResponse"""
        response_data = {
            "id": "job_failed",
            "arrangement_id": "arrangement_error",
            "status": RenderStatus.FAILED,
            "progress": 65.0,
            "error_message": "Insufficient memory for processing",
            "error_code": "MEM_001",
            "retry_count": 2,
            "max_retries": 3,
            "can_retry": True,
            "next_retry_at": datetime.now() + timedelta(minutes=5)
        }
        
        job_response = RenderJobResponse(**response_data)
        
        assert job_response.status == RenderStatus.FAILED
        assert job_response.error_message == "Insufficient memory for processing"
        assert job_response.can_retry == True
        assert isinstance(job_response.next_retry_at, datetime)


class TestRenderProgress:
    """Tests für RenderProgress-Schema"""
    
    @pytest.mark.unit
    def test_render_progress_basic(self):
        """Test: Grundlegende RenderProgress"""
        progress_data = {
            "job_id": "job_123",
            "percentage": 75.5,
            "current_stage": "applying_effects",
            "stage_progress": 50.0,
            "estimated_remaining": 120.0,
            "stages_completed": 3,
            "total_stages": 5
        }
        
        progress = RenderProgress(**progress_data)
        
        assert progress.job_id == "job_123"
        assert progress.percentage == 75.5
        assert progress.current_stage == "applying_effects"
        assert progress.estimated_remaining == 120.0
    
    @pytest.mark.unit
    def test_render_progress_with_details(self):
        """Test: RenderProgress mit Details"""
        progress_data = {
            "job_id": "job_456",
            "percentage": 45.0,
            "current_stage": "mixing",
            "stage_details": {
                "current_stem": "bass_001.wav",
                "stems_processed": 3,
                "total_stems": 8,
                "current_effect": "reverb",
                "effects_applied": 5,
                "total_effects": 12
            },
            "performance_metrics": {
                "cpu_usage": 68.5,
                "memory_usage": 1536,
                "processing_speed": 2.3
            }
        }
        
        progress = RenderProgress(**progress_data)
        
        assert progress.stage_details["current_stem"] == "bass_001.wav"
        assert progress.performance_metrics["cpu_usage"] == 68.5
        assert progress.stage_details["stems_processed"] == 3
    
    @pytest.mark.unit
    def test_render_progress_validation(self):
        """Test: RenderProgress-Validierung"""
        # Ungültiger Percentage
        with pytest.raises(ValidationError):
            RenderProgress(job_id="test", percentage=150.0)
        
        # Ungültiger Stage-Progress
        with pytest.raises(ValidationError):
            RenderProgress(job_id="test", percentage=50.0, stage_progress=-10.0)
        
        # Ungültige Stage-Counts
        with pytest.raises(ValidationError):
            RenderProgress(
                job_id="test",
                percentage=50.0,
                stages_completed=5,
                total_stages=3  # Completed > Total
            )


class TestRenderSettings:
    """Tests für RenderSettings-Schema"""
    
    @pytest.mark.unit
    def test_render_settings_audio(self):
        """Test: Audio-Einstellungen"""
        settings_data = {
            "sample_rate": 48000,
            "bit_depth": 24,
            "channels": 2,
            "normalize": True,
            "normalization_target": -1.0,
            "apply_limiter": True,
            "limiter_threshold": -0.5,
            "limiter_release": 50.0
        }
        
        settings = RenderSettings(**settings_data)
        
        assert settings.sample_rate == 48000
        assert settings.bit_depth == 24
        assert settings.normalize == True
        assert settings.limiter_threshold == -0.5
    
    @pytest.mark.unit
    def test_render_settings_processing(self):
        """Test: Processing-Einstellungen"""
        settings_data = {
            "dithering": True,
            "dither_type": "triangular",
            "oversampling": 4,
            "anti_aliasing": True,
            "high_quality_resampling": True,
            "parallel_processing": True,
            "max_threads": 8,
            "buffer_size": 1024
        }
        
        settings = RenderSettings(**settings_data)
        
        assert settings.dithering == True
        assert settings.dither_type == "triangular"
        assert settings.oversampling == 4
        assert settings.max_threads == 8
    
    @pytest.mark.unit
    def test_render_settings_effects(self):
        """Test: Effekt-Einstellungen"""
        settings_data = {
            "master_effects": {
                "eq": {
                    "enabled": True,
                    "low": 0.1,
                    "mid": 0.0,
                    "high": 0.05
                },
                "compressor": {
                    "enabled": True,
                    "ratio": 4.0,
                    "threshold": -12.0,
                    "attack": 10.0,
                    "release": 100.0
                },
                "reverb": {
                    "enabled": False
                }
            },
            "fade_in": 2.0,
            "fade_out": 4.0,
            "crossfade_duration": 1.0
        }
        
        settings = RenderSettings(**settings_data)
        
        assert settings.master_effects["eq"]["enabled"] == True
        assert settings.master_effects["compressor"]["ratio"] == 4.0
        assert settings.fade_in == 2.0
    
    @pytest.mark.unit
    def test_render_settings_validation(self):
        """Test: RenderSettings-Validierung"""
        # Ungültige Sample-Rate
        with pytest.raises(ValidationError):
            RenderSettings(sample_rate=1000)  # Zu niedrig
        
        # Ungültige Bit-Depth
        with pytest.raises(ValidationError):
            RenderSettings(bit_depth=12)  # Nicht unterstützt
        
        # Ungültige Thread-Anzahl
        with pytest.raises(ValidationError):
            RenderSettings(max_threads=0)  # Muss > 0 sein
        
        # Ungültiger Normalization-Target
        with pytest.raises(ValidationError):
            RenderSettings(normalization_target=1.0)  # Muss < 0 sein


class TestRenderOutput:
    """Tests für RenderOutput-Schema"""
    
    @pytest.mark.unit
    def test_render_output_basic(self):
        """Test: Grundlegende RenderOutput"""
        output_data = {
            "job_id": "job_123",
            "main_file": {
                "path": "/renders/job_123_main.wav",
                "size": 52428800,
                "format": "wav",
                "duration": 300.0,
                "checksum": "abc123def456"
            },
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=7)
        }
        
        output = RenderOutput(**output_data)
        
        assert output.job_id == "job_123"
        assert output.main_file["size"] == 52428800
        assert output.main_file["duration"] == 300.0
        assert isinstance(output.created_at, datetime)
    
    @pytest.mark.unit
    def test_render_output_with_stems(self):
        """Test: RenderOutput mit Stems"""
        output_data = {
            "job_id": "job_456",
            "main_file": {
                "path": "/renders/job_456_main.wav",
                "size": 45000000,
                "format": "wav"
            },
            "stem_files": [
                {
                    "stem_id": "kick_001",
                    "path": "/renders/job_456_kick.wav",
                    "size": 8000000,
                    "format": "wav"
                },
                {
                    "stem_id": "bass_001",
                    "path": "/renders/job_456_bass.wav",
                    "size": 12000000,
                    "format": "wav"
                }
            ],
            "metadata_file": {
                "path": "/renders/job_456_metadata.json",
                "size": 2048,
                "format": "json"
            }
        }
        
        output = RenderOutput(**output_data)
        
        assert len(output.stem_files) == 2
        assert output.stem_files[0]["stem_id"] == "kick_001"
        assert output.metadata_file["format"] == "json"
    
    @pytest.mark.unit
    def test_render_output_download_info(self):
        """Test: RenderOutput mit Download-Informationen"""
        output_data = {
            "job_id": "job_789",
            "main_file": {
                "path": "/renders/job_789_main.mp3",
                "size": 8000000,
                "format": "mp3"
            },
            "download_urls": {
                "main": "https://api.example.com/renders/job_789/main",
                "stems": "https://api.example.com/renders/job_789/stems.zip",
                "metadata": "https://api.example.com/renders/job_789/metadata"
            },
            "access_token": "token_abc123",
            "download_count": 0,
            "max_downloads": 10
        }
        
        output = RenderOutput(**output_data)
        
        assert output.download_urls["main"].startswith("https://")
        assert output.access_token == "token_abc123"
        assert output.download_count == 0
        assert output.max_downloads == 10


class TestRenderError:
    """Tests für RenderError-Schema"""
    
    @pytest.mark.unit
    def test_render_error_basic(self):
        """Test: Grundlegende RenderError"""
        error_data = {
            "job_id": "job_error",
            "error_code": "AUDIO_PROC_001",
            "error_message": "Failed to process audio file",
            "error_type": "processing_error",
            "occurred_at": datetime.now(),
            "stage": "audio_processing",
            "recoverable": True
        }
        
        error = RenderError(**error_data)
        
        assert error.job_id == "job_error"
        assert error.error_code == "AUDIO_PROC_001"
        assert error.error_type == "processing_error"
        assert error.recoverable == True
    
    @pytest.mark.unit
    def test_render_error_with_details(self):
        """Test: RenderError mit Details"""
        error_data = {
            "job_id": "job_detailed_error",
            "error_code": "MEM_001",
            "error_message": "Insufficient memory",
            "error_details": {
                "required_memory": "4GB",
                "available_memory": "2GB",
                "current_stem": "lead_synth_001.wav",
                "processing_stage": "reverb_application"
            },
            "stack_trace": "Traceback (most recent call last):\n  File...",
            "system_info": {
                "cpu_usage": 95.2,
                "memory_usage": 98.5,
                "disk_space": 15.2
            },
            "suggested_actions": [
                "Reduce quality settings",
                "Process fewer stems simultaneously",
                "Increase available memory"
            ]
        }
        
        error = RenderError(**error_data)
        
        assert error.error_details["required_memory"] == "4GB"
        assert error.system_info["memory_usage"] == 98.5
        assert len(error.suggested_actions) == 3
        assert "Reduce quality settings" in error.suggested_actions


class TestRenderMetrics:
    """Tests für RenderMetrics-Schema"""
    
    @pytest.mark.unit
    def test_render_metrics_performance(self):
        """Test: Performance-Metriken"""
        metrics_data = {
            "job_id": "job_metrics",
            "processing_time": 125.5,
            "queue_time": 30.2,
            "total_time": 155.7,
            "cpu_usage_avg": 72.3,
            "cpu_usage_peak": 95.1,
            "memory_usage_avg": 2048,
            "memory_usage_peak": 3072,
            "disk_io_read": 1024000000,
            "disk_io_write": 512000000
        }
        
        metrics = RenderMetrics(**metrics_data)
        
        assert metrics.job_id == "job_metrics"
        assert metrics.processing_time == 125.5
        assert metrics.cpu_usage_avg == 72.3
        assert metrics.memory_usage_peak == 3072
    
    @pytest.mark.unit
    def test_render_metrics_audio_quality(self):
        """Test: Audio-Quality-Metriken"""
        metrics_data = {
            "job_id": "job_quality",
            "audio_quality": {
                "peak_level": -0.1,
                "rms_level": -12.5,
                "lufs": -14.2,
                "dynamic_range": 8.5,
                "thd_n": 0.001,
                "snr": 96.3,
                "frequency_response_score": 0.95,
                "stereo_correlation": 0.3
            },
            "processing_quality": {
                "stems_processed": 8,
                "effects_applied": 15,
                "transitions_created": 4,
                "mix_complexity": 0.7,
                "automation_points": 45
            }
        }
        
        metrics = RenderMetrics(**metrics_data)
        
        assert metrics.audio_quality["peak_level"] == -0.1
        assert metrics.audio_quality["lufs"] == -14.2
        assert metrics.processing_quality["stems_processed"] == 8
        assert metrics.processing_quality["mix_complexity"] == 0.7


class TestRenderBatch:
    """Tests für RenderBatch-Schema"""
    
    @pytest.mark.unit
    def test_render_batch_creation(self):
        """Test: RenderBatch erstellen"""
        batch_data = {
            "batch_id": "batch_123",
            "jobs": [
                {
                    "arrangement_id": "arr_1",
                    "format": RenderFormat.WAV,
                    "quality": RenderQuality.HIGH
                },
                {
                    "arrangement_id": "arr_2",
                    "format": RenderFormat.MP3,
                    "quality": RenderQuality.STANDARD
                }
            ],
            "batch_settings": {
                "priority": RenderPriority.HIGH,
                "parallel_jobs": 2,
                "notify_on_completion": True
            }
        }
        
        batch = RenderBatch(**batch_data)
        
        assert batch.batch_id == "batch_123"
        assert len(batch.jobs) == 2
        assert batch.batch_settings["parallel_jobs"] == 2
    
    @pytest.mark.unit
    def test_render_batch_validation(self):
        """Test: RenderBatch-Validierung"""
        # Leere Jobs-Liste
        with pytest.raises(ValidationError):
            RenderBatch(batch_id="test", jobs=[])
        
        # Zu viele Jobs
        large_jobs_list = [{"arrangement_id": f"arr_{i}"} for i in range(101)]
        with pytest.raises(ValidationError):
            RenderBatch(batch_id="test", jobs=large_jobs_list)  # Max 100


class TestRenderTemplate:
    """Tests für RenderTemplate-Schema"""
    
    @pytest.mark.unit
    def test_render_template_creation(self):
        """Test: RenderTemplate erstellen"""
        template_data = {
            "name": "High Quality Techno",
            "description": "Template for high-quality techno renders",
            "format": RenderFormat.WAV,
            "quality": RenderQuality.HIGH,
            "settings": {
                "sample_rate": 48000,
                "bit_depth": 24,
                "normalize": True,
                "apply_limiter": True
            },
            "genre_specific": True,
            "target_genres": ["techno", "tech_house"],
            "created_by": "admin",
            "is_public": True
        }
        
        template = RenderTemplate(**template_data)
        
        assert template.name == "High Quality Techno"
        assert template.format == RenderFormat.WAV
        assert template.settings["sample_rate"] == 48000
        assert template.target_genres == ["techno", "tech_house"]


class TestRenderQueue:
    """Tests für RenderQueue-Schema"""
    
    @pytest.mark.unit
    def test_render_queue_status(self):
        """Test: RenderQueue-Status"""
        queue_data = {
            "total_jobs": 15,
            "pending_jobs": 8,
            "processing_jobs": 3,
            "completed_jobs": 4,
            "failed_jobs": 0,
            "average_processing_time": 45.2,
            "estimated_wait_time": 120.0,
            "queue_health": "good",
            "active_workers": 3,
            "max_workers": 5
        }
        
        queue = RenderQueue(**queue_data)
        
        assert queue.total_jobs == 15
        assert queue.pending_jobs == 8
        assert queue.average_processing_time == 45.2
        assert queue.queue_health == "good"
    
    @pytest.mark.unit
    def test_render_queue_validation(self):
        """Test: RenderQueue-Validierung"""
        # Negative Werte
        with pytest.raises(ValidationError):
            RenderQueue(total_jobs=-1)
        
        # Inkonsistente Job-Counts
        with pytest.raises(ValidationError):
            RenderQueue(
                total_jobs=10,
                pending_jobs=5,
                processing_jobs=3,
                completed_jobs=4,
                failed_jobs=2  # Summe > total_jobs
            )


class TestRenderSchemasIntegration:
    """Integrationstests für Render-Schemas"""
    
    @pytest.mark.integration
    def test_render_job_lifecycle(self):
        """Test: Vollständiger Render-Job-Lifecycle"""
        # 1. Job erstellen
        create_data = {
            "arrangement_id": "arrangement_lifecycle",
            "format": RenderFormat.WAV,
            "quality": RenderQuality.HIGH,
            "priority": RenderPriority.NORMAL,
            "notify_on_completion": True
        }
        
        job_create = RenderJobCreate(**create_data)
        assert job_create.arrangement_id == "arrangement_lifecycle"
        
        # 2. Job-Response (erstellt)
        job_id = "job_lifecycle_123"
        response_data = {
            "id": job_id,
            "arrangement_id": job_create.arrangement_id,
            "status": RenderStatus.PENDING,
            "progress": 0.0,
            "format": job_create.format,
            "quality": job_create.quality,
            "priority": job_create.priority,
            "created_at": datetime.now()
        }
        
        job_response = RenderJobResponse(**response_data)
        assert job_response.status == RenderStatus.PENDING
        
        # 3. Status-Updates
        updates = [
            {"status": RenderStatus.QUEUED, "progress": 0.0},
            {"status": RenderStatus.PROCESSING, "progress": 25.0, "current_stage": "loading_stems"},
            {"status": RenderStatus.PROCESSING, "progress": 50.0, "current_stage": "applying_effects"},
            {"status": RenderStatus.PROCESSING, "progress": 75.0, "current_stage": "mixing"},
            {"status": RenderStatus.PROCESSING, "progress": 90.0, "current_stage": "mastering"},
            {"status": RenderStatus.COMPLETED, "progress": 100.0}
        ]
        
        for update_data in updates:
            job_update = RenderJobUpdate(**update_data)
            assert isinstance(job_update.status, RenderStatus)
        
        # 4. Progress-Updates
        progress_data = {
            "job_id": job_id,
            "percentage": 75.0,
            "current_stage": "mixing",
            "estimated_remaining": 30.0
        }
        
        progress = RenderProgress(**progress_data)
        assert progress.percentage == 75.0
        
        # 5. Final Output
        output_data = {
            "job_id": job_id,
            "main_file": {
                "path": f"/renders/{job_id}_main.wav",
                "size": 50000000,
                "format": "wav",
                "duration": 300.0
            },
            "created_at": datetime.now()
        }
        
        output = RenderOutput(**output_data)
        assert output.main_file["duration"] == 300.0
    
    @pytest.mark.performance
    def test_render_schemas_performance(self):
        """Test: Performance der Render-Schemas"""
        import time
        
        # Viele Render-Jobs erstellen
        start_time = time.time()
        
        jobs = []
        for i in range(100):
            job_data = {
                "arrangement_id": f"arrangement_{i}",
                "format": RenderFormat.WAV if i % 2 == 0 else RenderFormat.MP3,
                "quality": RenderQuality.HIGH if i % 3 == 0 else RenderQuality.STANDARD,
                "priority": RenderPriority.NORMAL
            }
            
            job = RenderJobCreate(**job_data)
            jobs.append(job)
        
        creation_time = time.time() - start_time
        
        assert len(jobs) == 100
        assert creation_time < 1.0  # Sollte unter 1 Sekunde dauern
        
        # Komplexe Render-Responses
        start_time = time.time()
        
        responses = []
        for i in range(50):
            response_data = {
                "id": f"job_{i}",
                "arrangement_id": f"arrangement_{i}",
                "status": RenderStatus.COMPLETED,
                "progress": 100.0,
                "format": RenderFormat.WAV,
                "quality": RenderQuality.HIGH,
                "created_at": datetime.now(),
                "completed_at": datetime.now(),
                "metrics": {
                    "processing_time": 45.0 + i,
                    "cpu_usage_avg": 70.0 + (i % 20),
                    "memory_usage_peak": 2048 + (i * 10)
                }
            }
            
            response = RenderJobResponse(**response_data)
            responses.append(response)
        
        response_creation_time = time.time() - start_time
        
        assert len(responses) == 50
        assert response_creation_time < 1.0  # Sollte unter 1 Sekunde dauern