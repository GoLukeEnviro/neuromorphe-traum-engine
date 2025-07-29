"""Tests für Render-Schemas"""

import pytest
from pydantic import ValidationError
from typing import Dict, Any

from src.schemas.schemas import (
    RenderStatus, RenderPriority, RenderFormat, RenderQuality,
    RenderProgress, RenderSettings, RenderJobCreate, RenderJobUpdate,
    RenderJobBase, RenderJobResponse
)


class TestRenderEnums:
    """Tests für Render-Enums"""

    @pytest.mark.unit
    def test_render_status_values(self):
        """Test: RenderStatus-Werte"""
        assert RenderStatus.PENDING.value == "pending"
        assert RenderStatus.PROCESSING.value == "processing"
        assert RenderStatus.COMPLETED.value == "completed"
        assert RenderStatus.FAILED.value == "failed"

    @pytest.mark.unit
    def test_render_priority_values(self):
        """Test: RenderPriority-Werte"""
        assert RenderPriority.LOW.value == 1
        assert RenderPriority.MEDIUM.value == 5
        assert RenderPriority.HIGH.value == 10

    @pytest.mark.unit
    def test_render_format_values(self):
        """Test: RenderFormat-Werte"""
        assert RenderFormat.WAV.value == "wav"
        assert RenderFormat.MP3.value == "mp3"

    @pytest.mark.unit
    def test_render_quality_values(self):
        """Test: RenderQuality-Werte"""
        assert RenderQuality.LOW.value == "low"
        assert RenderQuality.HIGH.value == "high"


class TestRenderJobBase:
    """Tests für RenderJobBase-Schema"""

    @pytest.mark.unit
    def test_render_job_base_creation(self):
        """Test: RenderJobBase erstellen"""
        job_data = {
            "arrangement_id": "arrangement_123",
            "format": "wav",
            "quality": "high",
        }
        
        job = RenderJobBase(**job_data)
        
        assert job.arrangement_id == "arrangement_123"
        assert job.format == "wav"
        assert job.quality == "high"
        assert job.status == "pending"


class TestRenderJobCreate:
    """Tests für RenderJobCreate-Schema"""

    @pytest.mark.unit
    def test_render_job_create_basic(self):
        """Test: Grundlegende RenderJobCreate"""
        create_data = {
            "arrangement_id": "arrangement_789",
            "format": "mp3",
            "quality": "high",
        }
        
        job_create = RenderJobCreate(**create_data)
        
        assert job_create.arrangement_id == "arrangement_789"
        assert job_create.format == "mp3"


class TestRenderJobUpdate:
    """Tests für RenderJobUpdate-Schema"""

    @pytest.mark.unit
    def test_render_job_update_status(self):
        """Test: Status-Update"""
        update_data = {
            "status": "processing",
            "progress": 45.5,
        }
        
        job_update = RenderJobUpdate(**update_data)
        
        assert job_update.status == "processing"
        assert job_update.progress == 45.5


class TestRenderJobResponse:
    """Tests für RenderJobResponse-Schema"""

    @pytest.mark.unit
    def test_render_job_response_complete(self):
        """Test: Vollständige RenderJobResponse"""
        response_data = {
            "id": 123,
            "arrangement_id": "arrangement_456",
            "status": "completed",
            "progress": 100.0,
            "format": "wav",
            "quality": "high",
        }
        
        job_response = RenderJobResponse(**response_data)
        
        assert job_response.id == 123
        assert job_response.status == "completed"
        assert job_response.progress == 100.0


class TestRenderProgress:
    """Tests für RenderProgress-Schema"""

    @pytest.mark.unit
    def test_render_progress_basic(self):
        """Test: Grundlegende RenderProgress"""
        progress_data = {
            "job_id": "job_123",
            "progress": 75.5,
            "current_step": "applying_effects",
        }
        
        progress = RenderProgress(**progress_data)
        
        assert progress.job_id == "job_123"
        assert progress.progress == 75.5
        assert progress.current_step == "applying_effects"


class TestRenderSettings:
    """Tests für RenderSettings-Schema"""

    @pytest.mark.unit
    def test_render_settings_audio(self):
        """Test: Audio-Einstellungen"""
        settings_data = {
            "sample_rate": 48000,
            "bit_depth": 24,
            "normalize": True,
        }
        
        settings = RenderSettings(**settings_data)
        
        assert settings.sample_rate == 48000
        assert settings.bit_depth == 24
        assert settings.normalize == True

    @pytest.mark.unit
    def test_render_settings_validation(self):
        """Test: RenderSettings-Validierung"""
        with pytest.raises(ValidationError):
            RenderSettings(sample_rate=1000)  # Zu niedrig
