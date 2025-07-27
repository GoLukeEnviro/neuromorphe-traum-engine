"""Tests für Render-Service"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from src.services.render_service import (
    RenderService, RenderEngine, RenderJob,
    RenderQueue, RenderWorker, RenderProgress,
    RenderResult, RenderError, RenderMetrics,
    AudioRenderer, EffectProcessor, MixingEngine
)
from src.core.config import RenderConfig
from src.core.exceptions import (
    RenderingError, AudioProcessingError,
    ValidationError, ConfigurationError
)
from src.schemas.render import (
    RenderJobCreate, RenderJobResponse, RenderStatus,
    RenderPriority, RenderFormat, RenderQuality
)
from src.schemas.arrangement import ArrangementResponse
from src.database.models import RenderJob as RenderJobModel


class TestRenderService:
    """Tests für Render-Service"""
    
    @pytest.fixture
    def render_config(self):
        """Render-Konfiguration für Tests"""
        return RenderConfig(
            max_concurrent_jobs=4,
            default_sample_rate=48000,
            default_bit_depth=24,
            default_format="wav",
            output_directory="./renders",
            temp_directory="./temp",
            max_render_time=3600,
            enable_gpu_acceleration=False,
            quality_presets={
                "low": {"sample_rate": 44100, "bit_depth": 16},
                "medium": {"sample_rate": 48000, "bit_depth": 24},
                "high": {"sample_rate": 96000, "bit_depth": 32}
            }
        )
    
    @pytest.fixture
    def mock_database(self):
        """Mock Datenbank"""
        db = Mock()
        db.create_render_job = AsyncMock()
        db.get_render_job = AsyncMock()
        db.update_render_job = AsyncMock()
        db.list_render_jobs = AsyncMock()
        db.delete_render_job = AsyncMock()
        return db
    
    @pytest.fixture
    def mock_arrangement(self):
        """Mock Arrangement"""
        return ArrangementResponse(
            id="arr_123",
            name="Test Arrangement",
            stems=[
                {
                    "stem_id": "stem_1",
                    "start_time": 0.0,
                    "duration": 30.0,
                    "volume": 0.8,
                    "effects": []
                },
                {
                    "stem_id": "stem_2",
                    "start_time": 15.0,
                    "duration": 45.0,
                    "volume": 0.6,
                    "effects": [{"type": "reverb", "params": {"room_size": 0.5}}]
                }
            ],
            total_duration=60.0,
            bpm=128,
            key="Am"
        )
    
    @pytest.fixture
    def render_service(self, render_config, mock_database):
        """Render-Service für Tests"""
        service = RenderService(render_config, mock_database)
        return service
    
    @pytest.mark.unit
    def test_render_service_initialization(self, render_config, mock_database):
        """Test: Render-Service-Initialisierung"""
        service = RenderService(render_config, mock_database)
        
        assert service.config == render_config
        assert service.database == mock_database
        assert service.max_concurrent_jobs == 4
        assert service.output_directory == Path("./renders")
        assert isinstance(service.render_queue, RenderQueue)
    
    @pytest.mark.unit
    def test_render_service_invalid_config(self, mock_database):
        """Test: Render-Service mit ungültiger Konfiguration"""
        invalid_config = RenderConfig(
            max_concurrent_jobs=0,  # Muss > 0 sein
            output_directory=""
        )
        
        with pytest.raises(ConfigurationError):
            RenderService(invalid_config, mock_database)
    
    @pytest.mark.unit
    async def test_create_render_job(self, render_service, mock_arrangement):
        """Test: Render-Job erstellen"""
        job_data = RenderJobCreate(
            arrangement_id="arr_123",
            format="wav",
            quality="high",
            sample_rate=48000,
            bit_depth=24,
            priority="normal"
        )
        
        # Mock Datenbank-Antwort
        mock_job = RenderJobModel(
            id="job_123",
            arrangement_id="arr_123",
            status=RenderStatus.PENDING,
            format="wav",
            quality="high",
            created_at=datetime.now()
        )
        render_service.database.create_render_job.return_value = mock_job
        
        # Mock Arrangement-Abruf
        with patch.object(render_service, '_get_arrangement') as mock_get:
            mock_get.return_value = mock_arrangement
            
            job = await render_service.create_render_job(job_data)
            
            assert isinstance(job, RenderJobResponse)
            assert job.arrangement_id == "arr_123"
            assert job.status == RenderStatus.PENDING
            assert job.format == "wav"
            
            render_service.database.create_render_job.assert_called_once()
    
    @pytest.mark.unit
    async def test_create_render_job_invalid_arrangement(self, render_service):
        """Test: Render-Job mit ungültigem Arrangement"""
        job_data = RenderJobCreate(
            arrangement_id="invalid_arr",
            format="wav",
            quality="medium"
        )
        
        # Mock Arrangement nicht gefunden
        with patch.object(render_service, '_get_arrangement') as mock_get:
            mock_get.return_value = None
            
            with pytest.raises(ValidationError):
                await render_service.create_render_job(job_data)
    
    @pytest.mark.unit
    async def test_start_render_job(self, render_service, mock_arrangement):
        """Test: Render-Job starten"""
        job_id = "job_123"
        
        # Mock Job-Abruf
        mock_job = RenderJobModel(
            id=job_id,
            arrangement_id="arr_123",
            status=RenderStatus.PENDING,
            format="wav",
            quality="high"
        )
        render_service.database.get_render_job.return_value = mock_job
        
        # Mock Arrangement-Abruf
        with patch.object(render_service, '_get_arrangement') as mock_get:
            mock_get.return_value = mock_arrangement
            
            # Mock Render-Engine
            with patch.object(render_service, 'render_engine') as mock_engine:
                mock_engine.render_arrangement = AsyncMock()
                mock_result = RenderResult(
                    job_id=job_id,
                    output_path="/renders/job_123.wav",
                    duration=60.0,
                    file_size=10485760,
                    format="wav",
                    sample_rate=48000,
                    bit_depth=24
                )
                mock_engine.render_arrangement.return_value = mock_result
                
                result = await render_service.start_render_job(job_id)
                
                assert isinstance(result, RenderResult)
                assert result.job_id == job_id
                assert result.output_path == "/renders/job_123.wav"
                
                # Job-Status sollte aktualisiert werden
                render_service.database.update_render_job.assert_called()
    
    @pytest.mark.unit
    async def test_start_render_job_not_found(self, render_service):
        """Test: Render-Job starten - Job nicht gefunden"""
        job_id = "nonexistent_job"
        
        render_service.database.get_render_job.return_value = None
        
        with pytest.raises(ValidationError):
            await render_service.start_render_job(job_id)
    
    @pytest.mark.unit
    async def test_cancel_render_job(self, render_service):
        """Test: Render-Job abbrechen"""
        job_id = "job_123"
        
        # Mock laufender Job
        mock_job = RenderJobModel(
            id=job_id,
            status=RenderStatus.RENDERING,
            worker_id="worker_1"
        )
        render_service.database.get_render_job.return_value = mock_job
        
        # Mock Worker
        mock_worker = Mock()
        mock_worker.cancel_job = AsyncMock()
        render_service.render_queue.get_worker.return_value = mock_worker
        
        success = await render_service.cancel_render_job(job_id)
        
        assert success == True
        mock_worker.cancel_job.assert_called_once_with(job_id)
        
        # Job-Status sollte auf CANCELLED gesetzt werden
        render_service.database.update_render_job.assert_called()
    
    @pytest.mark.unit
    async def test_get_render_job_status(self, render_service):
        """Test: Render-Job-Status abrufen"""
        job_id = "job_123"
        
        mock_job = RenderJobModel(
            id=job_id,
            status=RenderStatus.RENDERING,
            progress=65.5,
            estimated_completion=datetime.now() + timedelta(minutes=5)
        )
        render_service.database.get_render_job.return_value = mock_job
        
        status = await render_service.get_render_job_status(job_id)
        
        assert isinstance(status, RenderJobResponse)
        assert status.id == job_id
        assert status.status == RenderStatus.RENDERING
        assert status.progress == 65.5
    
    @pytest.mark.unit
    async def test_list_render_jobs(self, render_service):
        """Test: Render-Jobs auflisten"""
        mock_jobs = [
            RenderJobModel(
                id="job_1",
                status=RenderStatus.COMPLETED,
                created_at=datetime.now() - timedelta(hours=1)
            ),
            RenderJobModel(
                id="job_2",
                status=RenderStatus.RENDERING,
                created_at=datetime.now() - timedelta(minutes=30)
            )
        ]
        render_service.database.list_render_jobs.return_value = mock_jobs
        
        jobs = await render_service.list_render_jobs(
            status=None,
            limit=10,
            offset=0
        )
        
        assert len(jobs) == 2
        assert all(isinstance(job, RenderJobResponse) for job in jobs)
        assert jobs[0].id == "job_1"
        assert jobs[1].id == "job_2"
    
    @pytest.mark.unit
    async def test_delete_render_job(self, render_service):
        """Test: Render-Job löschen"""
        job_id = "job_123"
        
        mock_job = RenderJobModel(
            id=job_id,
            status=RenderStatus.COMPLETED,
            output_path="/renders/job_123.wav"
        )
        render_service.database.get_render_job.return_value = mock_job
        
        # Mock Datei-Löschung
        with patch('pathlib.Path.unlink') as mock_unlink:
            success = await render_service.delete_render_job(job_id)
            
            assert success == True
            render_service.database.delete_render_job.assert_called_once_with(job_id)
            mock_unlink.assert_called_once()
    
    @pytest.mark.unit
    async def test_get_render_metrics(self, render_service):
        """Test: Render-Metriken abrufen"""
        # Mock Metriken-Daten
        mock_metrics_data = {
            "total_jobs": 100,
            "completed_jobs": 85,
            "failed_jobs": 10,
            "pending_jobs": 5,
            "average_render_time": 120.5,
            "total_render_time": 10242.5,
            "queue_length": 3,
            "active_workers": 2
        }
        
        with patch.object(render_service, '_calculate_metrics') as mock_calc:
            mock_calc.return_value = mock_metrics_data
            
            metrics = await render_service.get_render_metrics()
            
            assert isinstance(metrics, RenderMetrics)
            assert metrics.total_jobs == 100
            assert metrics.completed_jobs == 85
            assert metrics.success_rate == 0.85
            assert metrics.average_render_time == 120.5


class TestRenderEngine:
    """Tests für Render-Engine"""
    
    @pytest.fixture
    def render_config(self):
        """Render-Konfiguration für Tests"""
        return RenderConfig(
            default_sample_rate=48000,
            default_bit_depth=24,
            enable_gpu_acceleration=False
        )
    
    @pytest.fixture
    def render_engine(self, render_config):
        """Render-Engine für Tests"""
        return RenderEngine(render_config)
    
    @pytest.mark.unit
    def test_render_engine_initialization(self, render_config):
        """Test: Render-Engine-Initialisierung"""
        engine = RenderEngine(render_config)
        
        assert engine.config == render_config
        assert engine.sample_rate == 48000
        assert engine.bit_depth == 24
        assert isinstance(engine.audio_renderer, AudioRenderer)
        assert isinstance(engine.effect_processor, EffectProcessor)
        assert isinstance(engine.mixing_engine, MixingEngine)
    
    @pytest.mark.unit
    async def test_render_arrangement_basic(self, render_engine):
        """Test: Grundlegendes Arrangement-Rendering"""
        arrangement = ArrangementResponse(
            id="arr_123",
            name="Test Arrangement",
            stems=[
                {
                    "stem_id": "stem_1",
                    "start_time": 0.0,
                    "duration": 30.0,
                    "volume": 0.8,
                    "effects": []
                }
            ],
            total_duration=30.0,
            bpm=128
        )
        
        job = RenderJobModel(
            id="job_123",
            format="wav",
            quality="medium",
            sample_rate=48000,
            bit_depth=24
        )
        
        # Mock Audio-Rendering
        with patch.object(render_engine.audio_renderer, 'render_stem') as mock_render:
            mock_render.return_value = np.random.randn(48000 * 30)  # 30 Sekunden Audio
            
            # Mock Mixing
            with patch.object(render_engine.mixing_engine, 'mix_tracks') as mock_mix:
                mock_mix.return_value = np.random.randn(48000 * 30)
                
                # Mock Datei-Export
                with patch.object(render_engine, '_export_audio') as mock_export:
                    mock_export.return_value = "/renders/job_123.wav"
                    
                    result = await render_engine.render_arrangement(arrangement, job)
                    
                    assert isinstance(result, RenderResult)
                    assert result.job_id == "job_123"
                    assert result.output_path == "/renders/job_123.wav"
                    assert result.duration == 30.0
                    assert result.format == "wav"
    
    @pytest.mark.unit
    async def test_render_arrangement_with_effects(self, render_engine):
        """Test: Arrangement-Rendering mit Effekten"""
        arrangement = ArrangementResponse(
            id="arr_123",
            name="Test Arrangement",
            stems=[
                {
                    "stem_id": "stem_1",
                    "start_time": 0.0,
                    "duration": 30.0,
                    "volume": 0.8,
                    "effects": [
                        {"type": "reverb", "params": {"room_size": 0.5}},
                        {"type": "eq", "params": {"low": 1.0, "mid": 1.2, "high": 0.8}}
                    ]
                }
            ],
            total_duration=30.0
        )
        
        job = RenderJobModel(
            id="job_123",
            format="wav",
            quality="high"
        )
        
        # Mock Audio-Rendering
        with patch.object(render_engine.audio_renderer, 'render_stem') as mock_render:
            mock_render.return_value = np.random.randn(48000 * 30)
            
            # Mock Effekt-Verarbeitung
            with patch.object(render_engine.effect_processor, 'apply_effects') as mock_effects:
                mock_effects.return_value = np.random.randn(48000 * 30)
                
                # Mock Mixing und Export
                with patch.object(render_engine.mixing_engine, 'mix_tracks') as mock_mix:
                    mock_mix.return_value = np.random.randn(48000 * 30)
                    
                    with patch.object(render_engine, '_export_audio') as mock_export:
                        mock_export.return_value = "/renders/job_123.wav"
                        
                        result = await render_engine.render_arrangement(arrangement, job)
                        
                        assert isinstance(result, RenderResult)
                        # Effekte sollten angewendet worden sein
                        mock_effects.assert_called_once()
    
    @pytest.mark.unit
    async def test_render_arrangement_progress_tracking(self, render_engine):
        """Test: Fortschrittsverfolgung beim Rendering"""
        arrangement = ArrangementResponse(
            id="arr_123",
            name="Test Arrangement",
            stems=[
                {"stem_id": f"stem_{i}", "start_time": i * 10.0, "duration": 15.0}
                for i in range(5)  # 5 Stems
            ],
            total_duration=60.0
        )
        
        job = RenderJobModel(id="job_123", format="wav")
        
        progress_updates = []
        
        def progress_callback(progress: RenderProgress):
            progress_updates.append(progress)
        
        # Mock Audio-Rendering mit Progress
        with patch.object(render_engine.audio_renderer, 'render_stem') as mock_render:
            mock_render.return_value = np.random.randn(48000 * 15)
            
            with patch.object(render_engine.mixing_engine, 'mix_tracks') as mock_mix:
                mock_mix.return_value = np.random.randn(48000 * 60)
                
                with patch.object(render_engine, '_export_audio') as mock_export:
                    mock_export.return_value = "/renders/job_123.wav"
                    
                    result = await render_engine.render_arrangement(
                        arrangement,
                        job,
                        progress_callback=progress_callback
                    )
                    
                    assert isinstance(result, RenderResult)
                    # Progress-Updates sollten empfangen worden sein
                    assert len(progress_updates) > 0
                    # Letzter Progress sollte 100% sein
                    assert progress_updates[-1].percentage == 100.0
    
    @pytest.mark.unit
    async def test_render_arrangement_error_handling(self, render_engine):
        """Test: Fehlerbehandlung beim Rendering"""
        arrangement = ArrangementResponse(
            id="arr_123",
            name="Test Arrangement",
            stems=[
                {"stem_id": "invalid_stem", "start_time": 0.0, "duration": 30.0}
            ],
            total_duration=30.0
        )
        
        job = RenderJobModel(id="job_123", format="wav")
        
        # Mock Fehler beim Audio-Rendering
        with patch.object(render_engine.audio_renderer, 'render_stem') as mock_render:
            mock_render.side_effect = AudioProcessingError("Stem not found")
            
            with pytest.raises(RenderingError):
                await render_engine.render_arrangement(arrangement, job)


class TestRenderQueue:
    """Tests für Render-Queue"""
    
    @pytest.fixture
    def render_queue(self):
        """Render-Queue für Tests"""
        return RenderQueue(max_workers=2)
    
    @pytest.mark.unit
    def test_render_queue_initialization(self):
        """Test: Render-Queue-Initialisierung"""
        queue = RenderQueue(max_workers=4)
        
        assert queue.max_workers == 4
        assert len(queue.workers) == 0
        assert queue.pending_jobs.empty()
    
    @pytest.mark.unit
    async def test_add_job_to_queue(self, render_queue):
        """Test: Job zur Queue hinzufügen"""
        job = RenderJobModel(
            id="job_123",
            priority=RenderPriority.NORMAL,
            created_at=datetime.now()
        )
        
        await render_queue.add_job(job)
        
        assert not render_queue.pending_jobs.empty()
        assert render_queue.get_queue_length() == 1
    
    @pytest.mark.unit
    async def test_job_priority_ordering(self, render_queue):
        """Test: Job-Prioritäts-Reihenfolge"""
        # Jobs mit verschiedenen Prioritäten hinzufügen
        low_job = RenderJobModel(
            id="job_low",
            priority=RenderPriority.LOW,
            created_at=datetime.now()
        )
        
        high_job = RenderJobModel(
            id="job_high",
            priority=RenderPriority.HIGH,
            created_at=datetime.now()
        )
        
        normal_job = RenderJobModel(
            id="job_normal",
            priority=RenderPriority.NORMAL,
            created_at=datetime.now()
        )
        
        # In umgekehrter Prioritäts-Reihenfolge hinzufügen
        await render_queue.add_job(low_job)
        await render_queue.add_job(normal_job)
        await render_queue.add_job(high_job)
        
        # High-Priority-Job sollte zuerst kommen
        next_job = await render_queue.get_next_job()
        assert next_job.id == "job_high"
        
        # Dann Normal-Priority
        next_job = await render_queue.get_next_job()
        assert next_job.id == "job_normal"
        
        # Zuletzt Low-Priority
        next_job = await render_queue.get_next_job()
        assert next_job.id == "job_low"
    
    @pytest.mark.unit
    async def test_worker_management(self, render_queue):
        """Test: Worker-Management"""
        # Worker starten
        worker_id = await render_queue.start_worker()
        
        assert worker_id is not None
        assert len(render_queue.workers) == 1
        assert worker_id in render_queue.workers
        
        # Worker stoppen
        await render_queue.stop_worker(worker_id)
        
        assert len(render_queue.workers) == 0
        assert worker_id not in render_queue.workers
    
    @pytest.mark.unit
    async def test_max_workers_limit(self, render_queue):
        """Test: Maximale Worker-Anzahl"""
        # Mehr Worker starten als erlaubt
        worker_ids = []
        for i in range(5):  # max_workers = 2
            worker_id = await render_queue.start_worker()
            if worker_id:
                worker_ids.append(worker_id)
        
        # Nur 2 Worker sollten gestartet worden sein
        assert len(worker_ids) == 2
        assert len(render_queue.workers) == 2
    
    @pytest.mark.unit
    async def test_queue_statistics(self, render_queue):
        """Test: Queue-Statistiken"""
        # Jobs hinzufügen
        for i in range(5):
            job = RenderJobModel(
                id=f"job_{i}",
                priority=RenderPriority.NORMAL,
                created_at=datetime.now()
            )
            await render_queue.add_job(job)
        
        stats = render_queue.get_statistics()
        
        assert stats["queue_length"] == 5
        assert stats["active_workers"] == 0
        assert stats["max_workers"] == 2
        assert "average_wait_time" in stats


class TestRenderWorker:
    """Tests für Render-Worker"""
    
    @pytest.fixture
    def mock_render_engine(self):
        """Mock Render-Engine"""
        engine = Mock()
        engine.render_arrangement = AsyncMock()
        return engine
    
    @pytest.fixture
    def render_worker(self, mock_render_engine):
        """Render-Worker für Tests"""
        return RenderWorker(
            worker_id="worker_1",
            render_engine=mock_render_engine
        )
    
    @pytest.mark.unit
    def test_render_worker_initialization(self, mock_render_engine):
        """Test: Render-Worker-Initialisierung"""
        worker = RenderWorker(
            worker_id="worker_test",
            render_engine=mock_render_engine
        )
        
        assert worker.worker_id == "worker_test"
        assert worker.render_engine == mock_render_engine
        assert worker.current_job is None
        assert worker.is_busy == False
    
    @pytest.mark.unit
    async def test_process_job_success(self, render_worker, mock_render_engine):
        """Test: Erfolgreiche Job-Verarbeitung"""
        job = RenderJobModel(
            id="job_123",
            arrangement_id="arr_123",
            format="wav",
            quality="medium"
        )
        
        arrangement = ArrangementResponse(
            id="arr_123",
            name="Test Arrangement",
            stems=[],
            total_duration=30.0
        )
        
        # Mock erfolgreiche Render-Engine
        mock_result = RenderResult(
            job_id="job_123",
            output_path="/renders/job_123.wav",
            duration=30.0,
            file_size=10485760,
            format="wav"
        )
        mock_render_engine.render_arrangement.return_value = mock_result
        
        result = await render_worker.process_job(job, arrangement)
        
        assert isinstance(result, RenderResult)
        assert result.job_id == "job_123"
        assert render_worker.current_job is None
        assert render_worker.is_busy == False
    
    @pytest.mark.unit
    async def test_process_job_failure(self, render_worker, mock_render_engine):
        """Test: Fehlgeschlagene Job-Verarbeitung"""
        job = RenderJobModel(
            id="job_123",
            arrangement_id="arr_123"
        )
        
        arrangement = ArrangementResponse(
            id="arr_123",
            name="Test Arrangement",
            stems=[],
            total_duration=30.0
        )
        
        # Mock Fehler in Render-Engine
        mock_render_engine.render_arrangement.side_effect = RenderingError(
            "Rendering failed"
        )
        
        with pytest.raises(RenderingError):
            await render_worker.process_job(job, arrangement)
        
        assert render_worker.current_job is None
        assert render_worker.is_busy == False
    
    @pytest.mark.unit
    async def test_cancel_job(self, render_worker):
        """Test: Job abbrechen"""
        job = RenderJobModel(id="job_123")
        
        # Job starten (simuliert)
        render_worker.current_job = job
        render_worker.is_busy = True
        
        # Job abbrechen
        success = await render_worker.cancel_job("job_123")
        
        assert success == True
        assert render_worker.current_job is None
        assert render_worker.is_busy == False
    
    @pytest.mark.unit
    async def test_cancel_wrong_job(self, render_worker):
        """Test: Falschen Job abbrechen"""
        job = RenderJobModel(id="job_123")
        render_worker.current_job = job
        
        # Versuche anderen Job abzubrechen
        success = await render_worker.cancel_job("job_456")
        
        assert success == False
        assert render_worker.current_job == job  # Sollte unverändert sein


class TestRenderServiceIntegration:
    """Integrationstests für Render-Service"""
    
    @pytest.mark.integration
    async def test_full_render_workflow(self):
        """Test: Vollständiger Render-Workflow"""
        config = RenderConfig(
            max_concurrent_jobs=2,
            output_directory="./test_renders"
        )
        
        mock_db = Mock()
        mock_db.create_render_job = AsyncMock()
        mock_db.get_render_job = AsyncMock()
        mock_db.update_render_job = AsyncMock()
        
        service = RenderService(config, mock_db)
        
        # 1. Job erstellen
        job_data = RenderJobCreate(
            arrangement_id="arr_123",
            format="wav",
            quality="medium"
        )
        
        mock_job = RenderJobModel(
            id="job_123",
            arrangement_id="arr_123",
            status=RenderStatus.PENDING
        )
        mock_db.create_render_job.return_value = mock_job
        
        arrangement = ArrangementResponse(
            id="arr_123",
            name="Test Arrangement",
            stems=[],
            total_duration=30.0
        )
        
        with patch.object(service, '_get_arrangement') as mock_get:
            mock_get.return_value = arrangement
            
            job = await service.create_render_job(job_data)
            assert job.id == "job_123"
        
        # 2. Job starten
        mock_db.get_render_job.return_value = mock_job
        
        with patch.object(service, '_get_arrangement') as mock_get:
            mock_get.return_value = arrangement
            
            with patch.object(service.render_engine, 'render_arrangement') as mock_render:
                mock_result = RenderResult(
                    job_id="job_123",
                    output_path="/renders/job_123.wav",
                    duration=30.0,
                    file_size=10485760,
                    format="wav"
                )
                mock_render.return_value = mock_result
                
                result = await service.start_render_job("job_123")
                assert result.job_id == "job_123"
        
        # 3. Status abrufen
        mock_job.status = RenderStatus.COMPLETED
        status = await service.get_render_job_status("job_123")
        assert status.status == RenderStatus.COMPLETED
    
    @pytest.mark.performance
    async def test_render_service_performance(self):
        """Test: Render-Service-Performance"""
        import time
        
        config = RenderConfig(
            max_concurrent_jobs=4,
            output_directory="./test_renders"
        )
        
        mock_db = Mock()
        mock_db.create_render_job = AsyncMock()
        mock_db.list_render_jobs = AsyncMock()
        
        service = RenderService(config, mock_db)
        
        # Viele Jobs erstellen
        jobs = []
        for i in range(50):
            job_data = RenderJobCreate(
                arrangement_id=f"arr_{i}",
                format="wav",
                quality="medium"
            )
            
            mock_job = RenderJobModel(
                id=f"job_{i}",
                arrangement_id=f"arr_{i}",
                status=RenderStatus.PENDING
            )
            mock_db.create_render_job.return_value = mock_job
            
            arrangement = ArrangementResponse(
                id=f"arr_{i}",
                name=f"Arrangement {i}",
                stems=[],
                total_duration=30.0
            )
            
            with patch.object(service, '_get_arrangement') as mock_get:
                mock_get.return_value = arrangement
                
                start_time = time.time()
                job = await service.create_render_job(job_data)
                creation_time = time.time() - start_time
                
                jobs.append(job)
                
                # Job-Erstellung sollte schnell sein
                assert creation_time < 0.1
        
        assert len(jobs) == 50
        
        # Jobs auflisten
        mock_db.list_render_jobs.return_value = [
            RenderJobModel(id=f"job_{i}", status=RenderStatus.PENDING)
            for i in range(50)
        ]
        
        start_time = time.time()
        job_list = await service.list_render_jobs(limit=50)
        list_time = time.time() - start_time
        
        assert len(job_list) == 50
        assert list_time < 1.0  # Sollte unter 1 Sekunde dauern