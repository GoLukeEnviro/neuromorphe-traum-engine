"""Tests für Stem-Schemas"""

import os
import pytest
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import MagicMock

from src.schemas.schemas import (
    StemBase, StemCreate, StemUpdate, StemResponse,
    StemSearch, StemMetadata, StemFeatures,
    StemAnalysis, StemSimilarity, StemBatch
)


class TestStemBase:
    """Tests für StemBase-Schema"""
    
    @pytest.mark.unit
    def test_stem_base_creation(self):
        """Test: StemBase erstellen"""
        stem_data = {
            "filename": "kick_drum.wav",
            "title": "Heavy Kick",
            "genre": "techno",
            "bpm": 128.0,
            "key": "C",
            "duration": 4.0,
            "file_size": 1024000,
            "sample_rate": 44100,
            "bit_depth": 24,
            "channels": 2
        }
        
        stem = StemBase(**stem_data)
        
        assert stem.filename == "kick_drum.wav"
        assert stem.title == "Heavy Kick"
        assert stem.genre == "techno"
        assert stem.bpm == 128.0
        assert stem.key == "C"
        assert stem.duration == 4.0
        assert stem.file_size == 1024000
        assert stem.sample_rate == 44100
        assert stem.bit_depth == 24
        assert stem.channels == 2
    
    @pytest.mark.unit
    def test_stem_base_validation(self):
        """Test: StemBase-Validierung"""
        # Gültige Daten
        valid_data = {
            "filename": "test.wav",
            "duration": 5.0,
            "file_size": 1000,
            "sample_rate": 44100
        }
        
        stem = StemBase(**valid_data)
        assert stem.filename == "test.wav"
        
        # Ungültige BPM
        with pytest.raises(ValidationError):
            StemBase(filename="test.wav", bpm=-10)
        
        # Ungültige Dauer
        with pytest.raises(ValidationError):
            StemBase(filename="test.wav", duration=-1.0)
        
        # Ungültige Sample-Rate
        with pytest.raises(ValidationError):
            StemBase(filename="test.wav", sample_rate=0)
    
    @pytest.mark.unit
    def test_stem_base_optional_fields(self):
        """Test: Optionale Felder in StemBase"""
        # Minimale Daten
        minimal_data = {
            "filename": "minimal.wav"
        }
        
        stem = StemBase(**minimal_data)
        
        assert stem.filename == "minimal.wav"
        assert stem.title is None
        assert stem.genre is None
        assert stem.bpm is None
        assert stem.key is None
        assert stem.tags == []
        assert stem.metadata == {}
    
    @pytest.mark.unit
    def test_stem_base_serialization(self):
        """Test: StemBase-Serialisierung"""
        stem_data = {
            "filename": "test.wav",
            "title": "Test Track",
            "genre": "house",
            "bpm": 120.0,
            "tags": ["deep", "groovy"],
            "metadata": {"producer": "Test Producer"}
        }
        
        stem = StemBase(**stem_data)
        
        # Zu Dictionary
        stem_dict = stem.dict()
        assert stem_dict["filename"] == "test.wav"
        assert stem_dict["tags"] == ["deep", "groovy"]
        assert stem_dict["metadata"]["producer"] == "Test Producer"
        
        # Zu JSON
        stem_json = stem.json()
        assert isinstance(stem_json, str)
        assert "test.wav" in stem_json


class TestStemCreate:
    """Tests für StemCreate-Schema"""
    
    @pytest.mark.unit
    def test_stem_create_basic(self):
        """Test: Grundlegende StemCreate-Erstellung"""
        create_data = {
            "filename": "new_stem.wav",
            "title": "New Stem",
            "genre": "techno",
            "bpm": 130.0
        }
        
        stem_create = StemCreate(**create_data)
        
        assert stem_create.filename == "new_stem.wav"
        assert stem_create.title == "New Stem"
        assert stem_create.genre == "techno"
        assert stem_create.bpm == 130.0
    
    @pytest.mark.unit
    def test_stem_create_with_file_data(self):
        """Test: StemCreate mit Datei-Daten"""
        create_data = {
            "filename": "upload.wav",
            "file_data": b"fake_audio_data",
            "content_type": "audio/wav"
        }
        
        stem_create = StemCreate(**create_data)
        
        assert stem_create.filename == "upload.wav"
        assert stem_create.file_data == b"fake_audio_data"
        assert stem_create.content_type == "audio/wav"
    
    @pytest.mark.unit
    def test_stem_create_validation(self):
        """Test: StemCreate-Validierung"""
        # Filename ist erforderlich
        with pytest.raises(ValidationError):
            StemCreate()
        
        # Gültiger Content-Type
        valid_data = {
            "filename": "test.wav",
            "content_type": "audio/wav"
        }
        
        stem_create = StemCreate(**valid_data)
        assert stem_create.content_type == "audio/wav"
        
        # Ungültiger Content-Type
        with pytest.raises(ValidationError):
            StemCreate(filename="test.txt", content_type="text/plain")
    
    @pytest.mark.unit
    def test_stem_create_auto_analysis(self):
        """Test: Automatische Analyse bei StemCreate"""
        create_data = {
            "filename": "auto_analyze.wav",
            "auto_analyze": True,
            "extract_features": True
        }
        
        stem_create = StemCreate(**create_data)
        
        assert stem_create.auto_analyze == True
        assert stem_create.extract_features == True


class TestStemUpdate:
    """Tests für StemUpdate-Schema"""
    
    @pytest.mark.unit
    def test_stem_update_partial(self):
        """Test: Partielle StemUpdate"""
        update_data = {
            "title": "Updated Title",
            "bpm": 125.0
        }
        
        stem_update = StemUpdate(**update_data)
        
        assert stem_update.title == "Updated Title"
        assert stem_update.bpm == 125.0
        assert stem_update.filename is None  # Nicht gesetzt
        assert stem_update.genre is None  # Nicht gesetzt
    
    @pytest.mark.unit
    def test_stem_update_tags_metadata(self):
        """Test: Tags und Metadata in StemUpdate"""
        update_data = {
            "tags": ["updated", "modified"],
            "metadata": {
                "last_modified": "2024-01-01",
                "editor": "Test User"
            }
        }
        
        stem_update = StemUpdate(**update_data)
        
        assert stem_update.tags == ["updated", "modified"]
        assert stem_update.metadata["last_modified"] == "2024-01-01"
        assert stem_update.metadata["editor"] == "Test User"
    
    @pytest.mark.unit
    def test_stem_update_validation(self):
        """Test: StemUpdate-Validierung"""
        # Alle Felder sind optional
        empty_update = StemUpdate()
        assert empty_update.dict(exclude_unset=True) == {}
        
        # Validierung von gesetzten Feldern
        with pytest.raises(ValidationError):
            StemUpdate(bpm=-10)  # Ungültige BPM
        
        with pytest.raises(ValidationError):
            StemUpdate(duration=-1.0)  # Ungültige Dauer


class TestStemResponse:
    """Tests für StemResponse-Schema"""
    
    @pytest.mark.unit
    def test_stem_response_complete(self):
        """Test: Vollständige StemResponse"""
        response_data = {
            "id": "stem_123",
            "filename": "response_stem.wav",
            "title": "Response Stem",
            "genre": "house",
            "bpm": 124.0,
            "duration": 8.0,
            "file_size": 2048000,
            "sample_rate": 44100,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "file_path": "/path/to/stem.wav",
            "file_url": "https://example.com/stems/stem_123.wav"
        }
        
        stem_response = StemResponse(**response_data)
        
        assert stem_response.id == "stem_123"
        assert stem_response.filename == "response_stem.wav"
        assert stem_response.file_path == "/path/to/stem.wav"
        assert stem_response.file_url == "https://example.com/stems/stem_123.wav"
        assert isinstance(stem_response.created_at, datetime)
        assert isinstance(stem_response.updated_at, datetime)
    
    @pytest.mark.unit
    def test_stem_response_with_features(self):
        """Test: StemResponse mit Features"""
        response_data = {
            "id": "stem_456",
            "filename": "featured_stem.wav",
            "features": {
                "mfcc": [0.1, 0.2, 0.3],
                "spectral_centroid": 1500.0,
                "zero_crossing_rate": 0.05
            },
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        stem_response = StemResponse(**response_data)
        
        assert stem_response.features["mfcc"] == [0.1, 0.2, 0.3]
        assert stem_response.features["spectral_centroid"] == 1500.0
        assert stem_response.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.unit
    def test_stem_response_analysis_status(self):
        """Test: StemResponse mit Analyse-Status"""
        response_data = {
            "id": "stem_789",
            "filename": "analyzed_stem.wav",
            "analysis_status": "completed",
            "analysis_error": None,
            "analysis_progress": 100.0
        }
        
        stem_response = StemResponse(**response_data)
        
        assert stem_response.analysis_status == "completed"
        assert stem_response.analysis_error is None
        assert stem_response.analysis_progress == 100.0


class TestStemSearch:
    """Tests für StemSearch-Schema"""
    
    @pytest.mark.unit
    def test_stem_search_basic(self):
        """Test: Grundlegende StemSearch"""
        search_data = {
            "query": "techno kick",
            "genre": "techno",
            "bpm_min": 120.0,
            "bpm_max": 140.0
        }
        
        stem_search = StemSearch(**search_data)
        
        assert stem_search.query == "techno kick"
        assert stem_search.genre == "techno"
        assert stem_search.bpm_min == 120.0
        assert stem_search.bpm_max == 140.0
    
    @pytest.mark.unit
    def test_stem_search_pagination(self):
        """Test: StemSearch mit Pagination"""
        search_data = {
            "limit": 20,
            "offset": 40,
            "sort_by": "created_at",
            "sort_order": "desc"
        }
        
        stem_search = StemSearch(**search_data)
        
        assert stem_search.limit == 20
        assert stem_search.offset == 40
        assert stem_search.sort_by == "created_at"
        assert stem_search.sort_order == "desc"
    
    @pytest.mark.unit
    def test_stem_search_filters(self):
        """Test: StemSearch mit erweiterten Filtern"""
        search_data = {
            "tags": ["deep", "groovy"],
            "duration_min": 2.0,
            "duration_max": 10.0,
            "key": "Am",
            "has_features": True,
            "created_after": datetime(2024, 1, 1),
            "created_before": datetime(2024, 12, 31)
        }
        
        stem_search = StemSearch(**search_data)
        
        assert stem_search.tags == ["deep", "groovy"]
        assert stem_search.duration_min == 2.0
        assert stem_search.duration_max == 10.0
        assert stem_search.key == "Am"
        assert stem_search.has_features == True
        assert stem_search.created_after == datetime(2024, 1, 1)
        assert stem_search.created_before == datetime(2024, 12, 31)
    
    @pytest.mark.unit
    def test_stem_search_validation(self):
        """Test: StemSearch-Validierung"""
        # Ungültige BPM-Range
        with pytest.raises(ValidationError):
            StemSearch(bpm_min=140.0, bpm_max=120.0)
        
        # Ungültige Duration-Range
        with pytest.raises(ValidationError):
            StemSearch(duration_min=10.0, duration_max=5.0)
        
        # Ungültige Pagination
        with pytest.raises(ValidationError):
            StemSearch(limit=-1)
        
        with pytest.raises(ValidationError):
            StemSearch(offset=-1)
        
        # Ungültige Sort-Order
        with pytest.raises(ValidationError):
            StemSearch(sort_order="invalid")


class TestStemMetadata:
    """Tests für StemMetadata-Schema"""
    
    @pytest.mark.unit
    def test_stem_metadata_creation(self):
        """Test: StemMetadata erstellen"""
        metadata_data = {
            "producer": "Test Producer",
            "label": "Test Label",
            "release_date": "2024-01-01",
            "catalog_number": "TL001",
            "isrc": "US-ABC-12-34567",
            "copyright": "2024 Test Label",
            "description": "A test stem for unit testing",
            "instruments": ["kick", "bass", "synth"],
            "mood": "energetic",
            "energy_level": 8,
            "danceability": 9,
            "custom_fields": {
                "studio": "Test Studio",
                "engineer": "Test Engineer"
            }
        }
        
        metadata = StemMetadata(**metadata_data)
        
        assert metadata.producer == "Test Producer"
        assert metadata.label == "Test Label"
        assert metadata.instruments == ["kick", "bass", "synth"]
        assert metadata.energy_level == 8
        assert metadata.custom_fields["studio"] == "Test Studio"
    
    @pytest.mark.unit
    def test_stem_metadata_validation(self):
        """Test: StemMetadata-Validierung"""
        # Ungültiger Energy-Level
        with pytest.raises(ValidationError):
            StemMetadata(energy_level=11)  # Max ist 10
        
        with pytest.raises(ValidationError):
            StemMetadata(energy_level=0)  # Min ist 1
        
        # Ungültige Danceability
        with pytest.raises(ValidationError):
            StemMetadata(danceability=-1)
        
        # Gültige Werte
        valid_metadata = StemMetadata(
            energy_level=5,
            danceability=7
        )
        
        assert valid_metadata.energy_level == 5
        assert valid_metadata.danceability == 7


class TestStemFeatures:
    """Tests für StemFeatures-Schema"""
    
    @pytest.mark.unit
    def test_stem_features_basic(self):
        """Test: Grundlegende StemFeatures"""
        features_data = {
            "mfcc": [0.1, 0.2, 0.3, 0.4, 0.5],
            "spectral_centroid": 1500.0,
            "spectral_rolloff": 3000.0,
            "zero_crossing_rate": 0.05,
            "tempo": 128.0,
            "onset_strength": 0.8
        }
        
        features = StemFeatures(**features_data)
        
        assert features.mfcc == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert features.spectral_centroid == 1500.0
        assert features.tempo == 128.0
    
    @pytest.mark.unit
    def test_stem_features_advanced(self):
        """Test: Erweiterte StemFeatures"""
        features_data = {
            "chroma": [0.1] * 12,  # 12 Chroma-Features
            "tonnetz": [0.2] * 6,   # 6 Tonnetz-Features
            "spectral_contrast": [0.3] * 7,  # 7 Spectral-Contrast-Features
            "rms_energy": 0.5,
            "spectral_bandwidth": 2000.0,
            "spectral_flatness": 0.1,
            "harmonic_ratio": 0.7,
            "percussive_ratio": 0.3
        }
        
        features = StemFeatures(**features_data)
        
        assert len(features.chroma) == 12
        assert len(features.tonnetz) == 6
        assert len(features.spectral_contrast) == 7
        assert features.harmonic_ratio == 0.7
    
    @pytest.mark.unit
    def test_stem_features_validation(self):
        """Test: StemFeatures-Validierung"""
        # Ungültige Ratio-Werte (müssen zwischen 0 und 1 sein)
        with pytest.raises(ValidationError):
            StemFeatures(harmonic_ratio=1.5)
        
        with pytest.raises(ValidationError):
            StemFeatures(percussive_ratio=-0.1)
        
        # Ungültige Tempo-Werte
        with pytest.raises(ValidationError):
            StemFeatures(tempo=0)
        
        with pytest.raises(ValidationError):
            StemFeatures(tempo=300)  # Zu hoch


class TestStemAnalysis:
    """Tests für StemAnalysis-Schema"""
    
    @pytest.mark.unit
    def test_stem_analysis_complete(self):
        """Test: Vollständige StemAnalysis"""
        analysis_data = {
            "stem_id": "stem_123",
            "status": "completed",
            "progress": 100.0,
            "started_at": datetime.now(),
            "completed_at": datetime.now(),
            "features": {
                "mfcc": [0.1, 0.2, 0.3],
                "tempo": 128.0
            },
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "detected_bpm": 128.5,
            "detected_key": "Am",
            "detected_genre": "techno",
            "confidence_scores": {
                "bpm": 0.95,
                "key": 0.87,
                "genre": 0.92
            }
        }
        
        analysis = StemAnalysis(**analysis_data)
        
        assert analysis.stem_id == "stem_123"
        assert analysis.status == "completed"
        assert analysis.detected_bpm == 128.5
        assert analysis.detected_key == "Am"
        assert analysis.confidence_scores["bpm"] == 0.95
    
    @pytest.mark.unit
    def test_stem_analysis_error(self):
        """Test: StemAnalysis mit Fehler"""
        analysis_data = {
            "stem_id": "stem_456",
            "status": "failed",
            "progress": 50.0,
            "error_message": "Failed to extract features",
            "error_code": "FEATURE_EXTRACTION_ERROR"
        }
        
        analysis = StemAnalysis(**analysis_data)
        
        assert analysis.status == "failed"
        assert analysis.error_message == "Failed to extract features"
        assert analysis.error_code == "FEATURE_EXTRACTION_ERROR"
    
    @pytest.mark.unit
    def test_stem_analysis_validation(self):
        """Test: StemAnalysis-Validierung"""
        # Ungültiger Progress
        with pytest.raises(ValidationError):
            StemAnalysis(stem_id="test", progress=150.0)
        
        with pytest.raises(ValidationError):
            StemAnalysis(stem_id="test", progress=-10.0)
        
        # Ungültiger Status
        with pytest.raises(ValidationError):
            StemAnalysis(stem_id="test", status="invalid_status")


class TestStemSimilarity:
    """Tests für StemSimilarity-Schema"""
    
    @pytest.mark.unit
    def test_stem_similarity_basic(self):
        """Test: Grundlegende StemSimilarity"""
        similarity_data = {
            "stem_id": "stem_123",
            "similar_stem_id": "stem_456",
            "similarity_score": 0.85,
            "similarity_type": "embedding",
            "features_compared": ["mfcc", "spectral_centroid"],
            "metadata": {
                "algorithm": "cosine_similarity",
                "threshold": 0.8
            }
        }
        
        similarity = StemSimilarity(**similarity_data)
        
        assert similarity.stem_id == "stem_123"
        assert similarity.similar_stem_id == "stem_456"
        assert similarity.similarity_score == 0.85
        assert similarity.similarity_type == "embedding"
        assert similarity.features_compared == ["mfcc", "spectral_centroid"]
    
    @pytest.mark.unit
    def test_stem_similarity_validation(self):
        """Test: StemSimilarity-Validierung"""
        # Ungültiger Similarity-Score
        with pytest.raises(ValidationError):
            StemSimilarity(
                stem_id="test1",
                similar_stem_id="test2",
                similarity_score=1.5  # Muss zwischen 0 und 1 sein
            )
        
        with pytest.raises(ValidationError):
            StemSimilarity(
                stem_id="test1",
                similar_stem_id="test2",
                similarity_score=-0.1
            )
        
        # Gleiche Stem-IDs
        with pytest.raises(ValidationError):
            StemSimilarity(
                stem_id="test1",
                similar_stem_id="test1",  # Darf nicht gleich sein
                similarity_score=0.9
            )


class TestStemBatch:
    """Tests für StemBatch-Schema"""
    
    @pytest.mark.unit
    def test_stem_batch_creation(self):
        """Test: StemBatch erstellen"""
        batch_data = {
            "stems": [
                {"filename": "stem1.wav", "title": "Stem 1"},
                {"filename": "stem2.wav", "title": "Stem 2"},
                {"filename": "stem3.wav", "title": "Stem 3"}
            ],
            "batch_id": "batch_123",
            "auto_analyze": True,
            "extract_features": True,
            "metadata": {
                "uploaded_by": "test_user",
                "upload_session": "session_456"
            }
        }
        
        batch = StemBatch(**batch_data)
        
        assert len(batch.stems) == 3
        assert batch.batch_id == "batch_123"
        assert batch.auto_analyze == True
        assert batch.metadata["uploaded_by"] == "test_user"
    
    @pytest.mark.unit
    def test_stem_batch_validation(self):
        """Test: StemBatch-Validierung"""
        # Leere Stems-Liste
        with pytest.raises(ValidationError):
            StemBatch(stems=[])
        
        # Zu viele Stems
        large_stems_list = [{"filename": f"stem_{i}.wav"} for i in range(101)]
        with pytest.raises(ValidationError):
            StemBatch(stems=large_stems_list)  # Max 100
    
    @pytest.mark.unit
    def test_stem_batch_processing_status(self):
        """Test: StemBatch mit Processing-Status"""
        batch_data = {
            "stems": [{"filename": "test.wav"}],
            "processing_status": "processing",
            "processed_count": 5,
            "total_count": 10,
            "failed_count": 1,
            "errors": [
                {
                    "stem_filename": "error_stem.wav",
                    "error_message": "Invalid format",
                    "error_code": "INVALID_FORMAT"
                }
            ]
        }
        
        batch = StemBatch(**batch_data)
        
        assert batch.processing_status == "processing"
        assert batch.processed_count == 5
        assert batch.total_count == 10
        assert batch.failed_count == 1
        assert len(batch.errors) == 1
        assert batch.errors[0]["stem_filename"] == "error_stem.wav"


class TestStemSchemasIntegration:
    """Integrationstests für Stem-Schemas"""
    
    @pytest.mark.integration
    def test_stem_lifecycle_schemas(self):
        """Test: Vollständiger Stem-Lifecycle mit Schemas"""
        # 1. Stem erstellen
        create_data = {
            "filename": "lifecycle_test.wav",
            "title": "Lifecycle Test",
            "genre": "techno",
            "auto_analyze": True
        }
        
        stem_create = StemCreate(**create_data)
        assert stem_create.filename == "lifecycle_test.wav"
        
        # 2. Stem-Response simulieren
        response_data = {
            "id": "stem_lifecycle_123",
            "filename": stem_create.filename,
            "title": stem_create.title,
            "genre": stem_create.genre,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "analysis_status": "pending"
        }
        
        stem_response = StemResponse(**response_data)
        assert stem_response.id == "stem_lifecycle_123"
        
        # 3. Stem aktualisieren
        update_data = {
            "bpm": 128.0,
            "key": "Am",
            "tags": ["dark", "driving"]
        }
        
        stem_update = StemUpdate(**update_data)
        assert stem_update.bpm == 128.0
        
        # 4. Analyse-Ergebnis
        analysis_data = {
            "stem_id": stem_response.id,
            "status": "completed",
            "progress": 100.0,
            "detected_bpm": 127.8,
            "detected_key": "Am",
            "confidence_scores": {"bpm": 0.95, "key": 0.88}
        }
        
        analysis = StemAnalysis(**analysis_data)
        assert analysis.status == "completed"
        
        # 5. Ähnlichkeit finden
        similarity_data = {
            "stem_id": stem_response.id,
            "similar_stem_id": "stem_similar_456",
            "similarity_score": 0.92,
            "similarity_type": "embedding"
        }
        
        similarity = StemSimilarity(**similarity_data)
        assert similarity.similarity_score == 0.92
    
    @pytest.mark.integration
    def test_stem_search_and_filter(self):
        """Test: Stem-Suche und -Filterung"""
        # Komplexe Suche
        search_data = {
            "query": "dark techno kick",
            "genre": "techno",
            "bpm_min": 120.0,
            "bpm_max": 135.0,
            "tags": ["dark", "industrial"],
            "duration_min": 1.0,
            "duration_max": 8.0,
            "has_features": True,
            "limit": 50,
            "sort_by": "similarity_score",
            "sort_order": "desc"
        }
        
        search = StemSearch(**search_data)
        
        # Validierung der Suchparameter
        assert search.query == "dark techno kick"
        assert search.bpm_min < search.bpm_max
        assert search.duration_min < search.duration_max
        assert search.limit > 0
        assert search.sort_order in ["asc", "desc"]
    
    @pytest.mark.performance
    def test_stem_schemas_performance(self):
        """Test: Performance der Stem-Schemas"""
        import time
        
        # Viele Stems erstellen
        start_time = time.time()
        
        stems = []
        for i in range(1000):
            stem_data = {
                "filename": f"performance_test_{i}.wav",
                "title": f"Performance Test {i}",
                "genre": "techno",
                "bpm": 120.0 + (i % 20),
                "tags": [f"tag_{i % 10}"],
                "metadata": {"test_id": i}
            }
            
            stem = StemCreate(**stem_data)
            stems.append(stem)
        
        creation_time = time.time() - start_time
        
        assert len(stems) == 1000
        assert creation_time < 5.0  # Sollte unter 5 Sekunden dauern
        
        # Serialisierung testen
        start_time = time.time()
        
        serialized_stems = [stem.json() for stem in stems[:100]]
        
        serialization_time = time.time() - start_time
        
        assert len(serialized_stems) == 100
        assert serialization_time < 1.0  # Sollte unter 1 Sekunde dauern