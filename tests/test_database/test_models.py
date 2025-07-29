"""Tests für Datenbank-Modelle"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
import json

from database.database import get_async_db_session

from database.models import (
    Stem, GeneratedTrack, ProcessingJob, 
    RenderStatus, RenderFormat
)


class TestStemModel:
    """Tests für das Stem-Modell"""
    
    @pytest.mark.unit
    def test_stem_creation(self):
        """Test: Stem-Instanz erstellen"""
        stem = Stem(
            id="test_stem_id",
            name="Test Kick",
            file_path="/path/to/kick.wav",
            type="kick",
            genre="techno",
            tempo=128.0,
            key="Am",
            duration=4.0,
            tags=["dark", "heavy"],
            features={
                "spectral_centroid": 1500.0,
                "mfcc": [0.1, 0.2, 0.3]
            },
            embeddings=[0.1] * 512
        )
        
        assert stem.id == "test_stem_id"
        assert stem.name == "Test Kick"
        assert stem.type == StemType.KICK
        assert stem.tempo == 128.0
        assert len(stem.embeddings) == 512
        assert "dark" in stem.tags
        assert stem.features["spectral_centroid"] == 1500.0
    
    @pytest.mark.unit
    def test_stem_validation(self):
        """Test: Stem-Validierung"""
        # Gültiger Stem
        valid_stem = Stem(
            name="Valid Stem",
            file_path="/valid/path.wav",
            type="bass",
            embeddings=[0.0] * 512
        )
        
        assert valid_stem.name == "Valid Stem"
        assert valid_stem.type == StemType.BASS
        
        # Ungültige Embeddings-Länge sollte Fehler verursachen
        with pytest.raises(ValueError):
            Stem(
                name="Invalid Stem",
                file_path="/path.wav",
                type="kick",
                embeddings=[0.0] * 100  # Falsche Länge
            )
    
    @pytest.mark.unit
    def test_stem_serialization(self):
        """Test: Stem-Serialisierung"""
        stem = Stem(
            name="Serialization Test",
            file_path="/path/to/test.wav",
            type="synth",
            genre="house",
            tempo=125.0,
            tags=["melodic", "uplifting"],
            features={"energy": 0.8},
            embeddings=[0.5] * 512
        )
        
        # Zu Dictionary konvertieren
        stem_dict = stem.to_dict()
        
        assert stem_dict["name"] == "Serialization Test"
        assert stem_dict["type"] == "synth"
        assert stem_dict["tempo"] == 125.0
        assert "melodic" in stem_dict["tags"]
        assert len(stem_dict["embeddings"]) == 512
        
        # Von Dictionary erstellen
        new_stem = Stem.from_dict(stem_dict)
        
        assert new_stem.name == stem.name
        assert new_stem.type == stem.type
        assert new_stem.tempo == stem.tempo
        assert new_stem.tags == stem.tags
    
    @pytest.mark.unit
    def test_stem_similarity_calculation(self):
        """Test: Ähnlichkeitsberechnung zwischen Stems"""
        stem1 = Stem(
            name="Stem 1",
            file_path="/path1.wav",
            type="kick",
            embeddings=[1.0] + [0.0] * 511
        )
        
        stem2 = Stem(
            name="Stem 2",
            file_path="/path2.wav",
            type="kick",
            embeddings=[0.9] + [0.1] * 511
        )
        
        stem3 = Stem(
            name="Stem 3",
            file_path="/path3.wav",
            type="kick",
            embeddings=[0.0] * 512
        )
        
        # Ähnlichkeit berechnen
        similarity_1_2 = stem1.calculate_similarity(stem2)
        similarity_1_3 = stem1.calculate_similarity(stem3)
        
        assert similarity_1_2 > similarity_1_3
        assert 0.0 <= similarity_1_2 <= 1.0
        assert 0.0 <= similarity_1_3 <= 1.0
    
    @pytest.mark.unit
    def test_stem_metadata_update(self):
        """Test: Stem-Metadaten aktualisieren"""
        stem = Stem(
            name="Original Name",
            file_path="/path.wav",
            type="fx",
            tags=["original"],
            features={"original": True}
        )
        
        # Metadaten aktualisieren
        update_data = {
            "name": "Updated Name",
            "tags": ["updated", "modified"],
            "features": {"updated": True, "version": 2}
        }
        
        stem.update_metadata(update_data)
        
        assert stem.name == "Updated Name"
        assert "updated" in stem.tags
        assert stem.features["updated"] == True
        assert stem.features["version"] == 2
        assert stem.type == StemType.FX  # Sollte unverändert bleiben


class TestGeneratedTrackOperations:
    """Tests für das Arrangement-Modell"""
    
    @pytest.mark.unit
    def test_generated_track_creation(self):
        """Test: GeneratedTrack-Instanz erstellen"""
        track = GeneratedTrack(
            id="test_track_id",
            original_prompt="Dark techno with heavy bass",
            duration=180,
            target_genre="techno",
            track_structure={
                "sections": [
                    {"name": "intro", "start": 0, "duration": 32},
                    {"name": "main", "start": 32, "duration": 96},
                    {"name": "outro", "start": 128, "duration": 32}
                ],
                "total_duration": 160
            },
            stems=["stem1", "stem2", "stem3"],
            track_metadata={
                "tempo": 128,
                "key": "Am",
                "energy": 0.8
            }
        )
        
        assert track.id == "test_track_id"
        assert track.original_prompt == "Dark techno with heavy bass"
        assert track.duration == 180
        assert track.target_genre == "techno"
        assert len(track.track_structure["sections"]) == 3
        assert len(track.stems) == 3
        assert track.track_metadata["tempo"] == 128
    
    @pytest.mark.unit
    def test_generated_track_validation(self):
        """Test: GeneratedTrack-Validierung"""
        # Gültiger Track
        valid_track = GeneratedTrack(
            original_prompt="Valid prompt",
            duration=120,
            track_structure={"sections": []},
            stems=[]
        )
        
        assert valid_track.original_prompt == "Valid prompt"
        assert valid_track.duration == 120
        
        # Ungültige Dauer sollte Fehler verursachen
        with pytest.raises(ValueError):
            GeneratedTrack(
                original_prompt="Invalid duration",
                duration=-10,  # Negative Dauer
                track_structure={"sections": []},
                stems=[]
            )
        
        # Leerer Prompt sollte Fehler verursachen
        with pytest.raises(ValueError):
            GeneratedTrack(
                original_prompt="",  # Leerer Prompt
                duration=120,
                track_structure={"sections": []},
                stems=[]
            )
    
    @pytest.mark.unit
    def test_generated_track_serialization(self):
        """Test: GeneratedTrack-Serialisierung"""
        track = GeneratedTrack(
            original_prompt="Serialization test",
            duration=240,
            target_genre="house",
            track_structure={
                "sections": [
                    {"name": "build", "start": 0, "duration": 64}
                ]
            },
            stems=["stem1", "stem2"],
            track_metadata={"bpm": 125}
        )
        
        # Zu Dictionary konvertieren
        track_dict = track.to_dict()
        
        assert track_dict["original_prompt"] == "Serialization test"
        assert track_dict["duration"] == 240
        assert track_dict["target_genre"] == "house"
        assert len(track_dict["track_structure"]["sections"]) == 1
        assert len(track_dict["stems"]) == 2
        
        # Von Dictionary erstellen
        new_track = GeneratedTrack.from_dict(track_dict)
        
        assert new_track.original_prompt == track.original_prompt
        assert new_track.duration == track.duration
        assert new_track.target_genre == track.target_genre
        assert new_track.stems == track.stems
    
    
    
    


class TestRenderJobModel:
    """Tests für das RenderJob-Modell"""
    
    @pytest.mark.unit
    def test_render_job_creation(self):
        """Test: RenderJob-Instanz erstellen"""
        job = RenderJob(
            id="test_job_id",
            arrangement_id="arrangement_123",
            format="wav",
            quality="high",
            status="pending",
            progress=0.0,
            options={
                "normalize": True,
                "apply_mastering": True,
                "fade_in": 2.0,
                "fade_out": 4.0
            }
        )
        
        assert job.id == "test_job_id"
        assert job.arrangement_id == "arrangement_123"
        assert job.format == RenderFormat.WAV
        assert job.quality == "high"
        assert job.status == RenderStatus.PENDING
        assert job.progress == 0.0
        assert job.options["normalize"] == True
    
    @pytest.mark.unit
    def test_render_job_status_transitions(self):
        """Test: RenderJob-Status-Übergänge"""
        job = RenderJob(
            arrangement_id="arrangement_123",
            format="mp3",
            status="pending"
        )
        
        # Von PENDING zu PROCESSING
        job.update_status("processing", progress=0.1)
        assert job.status == "processing"
        assert job.progress == 0.1
        assert job.started_at is not None
        
        # Progress aktualisieren
        job.update_status("processing", progress=0.5)
        assert job.progress == 0.5
        
        # Zu COMPLETED
        job.update_status(
            "completed", 
            progress=1.0, 
            output_path="/path/to/output.mp3"
        )
        assert job.status == "completed"
        assert job.progress == 1.0
        assert job.output_path == "/path/to/output.mp3"
        assert job.completed_at is not None
        
        # Rendering-Zeit berechnen
        render_time = job.get_render_time()
        assert render_time is not None
        assert render_time >= 0
    
    @pytest.mark.unit
    def test_render_job_error_handling(self):
        """Test: RenderJob-Fehlerbehandlung"""
        job = RenderJob(
            arrangement_id="arrangement_123",
            format=RenderFormat.WAV,
            status=RenderStatus.PROCESSING,
            progress=0.3
        )
        
        # Fehler setzen
        error_message = "Audio file not found"
        job.update_status(
            RenderStatus.FAILED, 
            error_message=error_message
        )
        
        assert job.status == RenderStatus.FAILED
        assert job.error_message == error_message
        assert job.completed_at is not None
        
        # Retry-Funktionalität
        job.retry()
        assert job.status == RenderStatus.PENDING
        assert job.error_message is None
        assert job.progress == 0.0
        assert job.retry_count == 1
    
    @pytest.mark.unit
    def test_render_job_serialization(self):
        """Test: RenderJob-Serialisierung"""
        job = RenderJob(
            arrangement_id="arrangement_456",
            format="flac",
            quality="lossless",
            status=RenderStatus.COMPLETED,
            progress=1.0,
            output_path="/output/final.flac",
            options={
                "sample_rate": 48000,
                "bit_depth": 24
            }
        )
        
        # Zu Dictionary konvertieren
        job_dict = job.to_dict()
        
        assert job_dict["arrangement_id"] == "arrangement_456"
        assert job_dict["format"] == "flac"
        assert job_dict["quality"] == "lossless"
        assert job_dict["status"] == "completed"
        assert job_dict["progress"] == 1.0
        assert job_dict["output_path"] == "/output/final.flac"
        assert job_dict["options"]["sample_rate"] == 48000
        
        # Von Dictionary erstellen
        new_job = RenderJob.from_dict(job_dict)
        
        assert new_job.arrangement_id == job.arrangement_id
        assert new_job.format == job.format
        assert new_job.quality == job.quality
        assert new_job.status == job.status
        assert new_job.progress == job.progress
        assert new_job.output_path == job.output_path
    
    @pytest.mark.unit
    def test_render_job_validation(self):
        """Test: RenderJob-Validierung"""
        # Gültiger Job
        valid_job = RenderJob(
            arrangement_id="valid_arrangement",
            format=RenderFormat.WAV
        )
        
        assert valid_job.arrangement_id == "valid_arrangement"
        assert valid_job.format == RenderFormat.WAV
        
        # Ungültiger Progress-Wert
        with pytest.raises(ValueError):
            RenderJob(
                arrangement_id="test",
                format="mp3",
                progress=1.5  # > 1.0
            )
        
        # Ungültiger Progress-Wert (negativ)
        with pytest.raises(ValueError):
            RenderJob(
                arrangement_id="test",
                format=RenderFormat.MP3,
                progress=-0.1  # < 0.0
            )
    
    @pytest.mark.unit
    def test_render_job_estimated_time(self):
        """Test: Geschätzte Rendering-Zeit"""
        job = RenderJob(
            arrangement_id="arrangement_789",
            format=RenderFormat.WAV,
            status=RenderStatus.PROCESSING,
            progress=0.4
        )
        
        # Startzeit setzen (vor 10 Sekunden)
        job.started_at = datetime.utcnow() - timedelta(seconds=10)
        
        # Geschätzte verbleibende Zeit berechnen
        estimated_time = job.get_estimated_remaining_time()
        
        assert estimated_time is not None
        assert estimated_time > 0
        
        # Bei 40% Progress und 10 Sekunden vergangen,
        # sollten etwa 15 Sekunden verbleiben (25 Sekunden total - 10 vergangen)
        assert 10 <= estimated_time <= 20


