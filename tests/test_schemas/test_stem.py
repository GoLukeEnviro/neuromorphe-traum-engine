"""Tests für Stem-Schemas"""

import os
import pytest
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import MagicMock
from pydantic import ValidationError


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
        assert stem.auto_tags == []
        assert stem.manual_tags == []
    
    @pytest.mark.unit
    def test_stem_base_serialization(self):
        """Test: StemBase-Serialisierung"""
        stem_data = {
            "filename": "test.wav",
            "title": "Test Track",
            "genre": "house",
            "bpm": 120.0,
            "manual_tags": ["deep", "groovy"],
            "semantic_analysis": {"producer": "Test Producer"}
        }
        
        stem = StemBase(**stem_data)
        
        # Zu Dictionary
        stem_dict = stem.model_dump()
        assert stem_dict["filename"] == "test.wav"
        assert stem_dict["manual_tags"] == ["deep", "groovy"]
        assert stem_dict["semantic_analysis"]["producer"] == "Test Producer"
        
        # Zu JSON
        stem_json = stem.model_dump_json()
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
    def test_stem_create_validation(self):
        """Test: StemCreate-Validierung"""
        # Filename ist erforderlich
        with pytest.raises(ValidationError):
            StemCreate()
        
        # Gültige Daten
        valid_data = {
            "filename": "test.wav",
        }
        
        stem_create = StemCreate(**valid_data)
        assert stem_create.filename == "test.wav"
        

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
            "manual_tags": ["updated", "modified"],
            "semantic_analysis": {
                "last_modified": "2024-01-01",
                "editor": "Test User"
            }
        }
        
        stem_update = StemUpdate(**update_data)
        
        assert stem_update.manual_tags == ["updated", "modified"]
        assert stem_update.semantic_analysis["last_modified"] == "2024-01-01"
        assert stem_update.semantic_analysis["editor"] == "Test User"
    
    @pytest.mark.unit
    def test_stem_update_validation(self):
        """Test: StemUpdate-Validierung"""
        # Alle Felder sind optional
        empty_update = StemUpdate()
        assert empty_update.model_dump(exclude_unset=True) == {}
        
        # Validierung von gesetzten Feldern
        with pytest.raises(ValidationError):
            StemUpdate(sample_rate=10)


class TestStemResponse:
    """Tests für StemResponse-Schema"""
    
    @pytest.mark.unit
    def test_stem_response_complete(self):
        """Test: Vollständige StemResponse"""
        response_data = {
            "id": 123,
            "filename": "response_stem.wav",
            "title": "Response Stem",
            "genre": "house",
            "bpm": 124.0,
            "duration": 8.0,
            "file_size": 2048000,
            "sample_rate": 44100,
        }
        
        stem_response = StemResponse(**response_data)
        
        assert stem_response.id == 123
        assert stem_response.filename == "response_stem.wav"
        assert isinstance(stem_response.id, int)

    @pytest.mark.unit
    def test_stem_response_with_features(self):
        """Test: StemResponse mit Features"""
        response_data = {
            "id": 456,
            "filename": "featured_stem.wav",
            "neural_features": {
                "mfcc": [0.1, 0.2, 0.3],
                "spectral_centroid": 1500.0,
                "zero_crossing_rate": 0.05
            },
            "audio_embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        stem_response = StemResponse(**response_data)
        
        assert stem_response.neural_features["mfcc"] == [0.1, 0.2, 0.3]
        assert stem_response.neural_features["spectral_centroid"] == 1500.0
        assert stem_response.audio_embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.unit
    def test_stem_response_analysis_status(self):
        """Test: StemResponse mit Analyse-Status"""
        response_data = {
            "id": 789,
            "filename": "analyzed_stem.wav",
            "processing_status": "completed",
            "processing_error": None,
        }
        
        stem_response = StemResponse(**response_data)
        
        assert stem_response.processing_status == "completed"
        assert stem_response.processing_error is None


class TestStemSearch:
    """Tests für StemSearch-Schema"""
    
    @pytest.mark.unit
    def test_stem_search_basic(self):
        """Test: Grundlegende StemSearch"""
        search_data = {
            "query_text": "techno kick",
            "genre": "techno",
            "bpm_min": 120.0,
            "bpm_max": 140.0
        }
        
        stem_search = StemSearch(**search_data)
        
        assert stem_search.query_text == "techno kick"
        assert stem_search.genre == "techno"
        assert stem_search.bpm_min == 120.0
        assert stem_search.bpm_max == 140.0
    
    @pytest.mark.unit
    def test_stem_search_filters(self):
        """Test: StemSearch mit erweiterten Filtern"""
        search_data = {
            "key": "Am",
            "compatible_keys": ["Dm", "G"],
        }
        
        stem_search = StemSearch(**search_data)
        
        assert stem_search.key == "Am"
        assert stem_search.compatible_keys == ["Dm", "G"]

    @pytest.mark.unit
    def test_stem_search_validation(self):
        """Test: StemSearch-Validierung"""
        # Pydantic v2 doesn't validate on assignment by default
        # We need to trigger validation explicitly
        with pytest.raises(ValidationError):
            StemSearch.model_validate({"bpm_min": 140.0, "bpm_max": 120.0})

        with pytest.raises(ValidationError):
            StemSearch.model_validate({"harmonic_complexity_min": 0.8, "harmonic_complexity_max": 0.2})
        

class TestStemMetadata:
    """Tests für StemMetadata-Schema"""
    
    @pytest.mark.unit
    def test_stem_metadata_creation(self):
        """Test: StemMetadata erstellen"""
        metadata_data = {
            "bpm": 120.0,
            "key": "Am",
            "category": "drums",
            "genre": "techno",
            "mood": "dark",
            "energy_level": "high",
            "auto_tags": ["kick", "bass"],
            "harmonic_complexity": 0.8,
            "rhythmic_complexity": 0.9,
            "quality_score": 0.95,
            "complexity_level": "complex"
        }
        
        metadata = StemMetadata(**metadata_data)
        
        assert metadata.bpm == 120.0
        assert metadata.key == "Am"
        assert metadata.energy_level == "high"

    @pytest.mark.unit
    def test_stem_metadata_validation(self):
        """Test: StemMetadata-Validierung"""
        # Gültige Werte
        valid_metadata = StemMetadata(
            bpm=120.0,
            quality_score=0.5
        )
        
        assert valid_metadata.bpm == 120.0
        assert valid_metadata.quality_score == 0.5


class TestStemFeatures:
    """Tests für StemFeatures-Schema"""
    
    @pytest.mark.unit
    def test_stem_features_basic(self):
        """Test: Grundlegende StemFeatures"""
        features_data = {
            "mfcc": [0.1, 0.2, 0.3, 0.4, 0.5],
            "spectral_centroid": 1500.0,
        }
        
        features = StemFeatures(**features_data)
        
        assert features.mfcc == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert features.spectral_centroid == 1500.0
    
    @pytest.mark.unit
    def test_stem_features_validation(self):
        """Test: StemFeatures-Validierung"""
        # Leere Daten sind ok
        features = StemFeatures()
        assert features.mfcc is None


class TestStemAnalysis:
    """Tests für StemAnalysis-Schema"""
    
    @pytest.mark.unit
    def test_stem_analysis_complete(self):
        """Test: Vollständige StemAnalysis"""
        analysis_data = {
            "file_info": {"size": 12345},
            "temporal": {"bpm": 128.0},
            "spectral": {"centroid": 1500.0},
            "rhythmic": {"complexity": 0.8},
            "harmonic": {"key": "Am"},
            "perceptual": {"loudness": -10.5},
            "classification": {"genre": "techno"}
        }
        
        analysis = StemAnalysis(**analysis_data)
        
        assert analysis.file_info["size"] == 12345
        assert analysis.temporal["bpm"] == 128.0
        assert analysis.harmonic["key"] == "Am"

    @pytest.mark.unit
    def test_stem_analysis_validation(self):
        """Test: StemAnalysis-Validierung"""
        # Fehlende Felder
        with pytest.raises(ValidationError):
            StemAnalysis(file_info={}) # other fields missing
        

class TestStemSimilarity:
    """Tests für StemSimilarity-Schema"""
    
    @pytest.mark.unit
    def test_stem_similarity_basic(self):
        """Test: Grundlegende StemSimilarity"""
        similarity_data = {
            "stem_id_1": 123,
            "stem_id_2": 456,
            "similarity_score": 0.85,
        }
        
        similarity = StemSimilarity(**similarity_data)
        
        assert similarity.stem_id_1 == 123
        assert similarity.stem_id_2 == 456
        assert similarity.similarity_score == 0.85
    
    @pytest.mark.unit
    def test_stem_similarity_validation(self):
        """Test: StemSimilarity-Validierung"""
        # Ungültiger Similarity-Score
        with pytest.raises(ValidationError):
            StemSimilarity.model_validate({
                "stem_id_1": 1,
                "stem_id_2": 2,
                "similarity_score": 1.5
            })

        with pytest.raises(ValidationError):
            StemSimilarity.model_validate({
                "stem_id_1": 1,
                "stem_id_2": 2,
                "similarity_score": -0.1
            })


class TestStemBatch:
    """Tests für StemBatch-Schema"""
    
    @pytest.mark.unit
    def test_stem_batch_creation(self):
        """Test: StemBatch erstellen"""
        batch_data = {
            "stem_ids": [1, 2, 3]
        }
        
        batch = StemBatch(**batch_data)
        
        assert len(batch.stem_ids) == 3
        assert batch.stem_ids == [1, 2, 3]
    
    @pytest.mark.unit
    def test_stem_batch_validation(self):
        """Test: StemBatch-Validierung"""
        # Leere Stems-Liste
        with pytest.raises(ValidationError):
            StemBatch.model_validate({"stem_ids": []})
        

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
        }
        
        stem_create = StemCreate(**create_data)
        assert stem_create.filename == "lifecycle_test.wav"
        
        # 2. Stem-Response simulieren
        response_data = {
            "id": 123,
            "filename": stem_create.filename,
            "title": stem_create.title,
            "genre": stem_create.genre,
        }
        
        stem_response = StemResponse(**response_data)
        assert stem_response.id == 123
        
        # 3. Stem aktualisieren
        update_data = {
            "bpm": 128.0,
            "key": "Am",
            "manual_tags": ["dark", "driving"]
        }
        
        stem_update = StemUpdate(**update_data)
        assert stem_update.bpm == 128.0
        
        # 4. Ähnlichkeit finden
        similarity_data = {
            "stem_id_1": stem_response.id,
            "stem_id_2": 456,
            "similarity_score": 0.92,
        }
        
        similarity = StemSimilarity(**similarity_data)
        assert similarity.similarity_score == 0.92
    
    @pytest.mark.integration
    def test_stem_search_and_filter(self):
        """Test: Stem-Suche und -Filterung"""
        # Komplexe Suche
        search_data = {
            "query_text": "dark techno kick",
            "genre": "techno",
            "bpm_min": 120.0,
            "bpm_max": 135.0,
            "key": "Am",
        }
        
        search = StemSearch(**search_data)
        
        # Validierung der Suchparameter
        assert search.query_text == "dark techno kick"
        assert search.bpm_min < search.bpm_max
    
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
                "manual_tags": [f"tag_{i % 10}"],
                "semantic_analysis": {"test_id": i}
            }
            
            stem = StemCreate(**stem_data)
            stems.append(stem)
        
        creation_time = time.time() - start_time
        
        assert len(stems) == 1000
        assert creation_time < 5.0  # Sollte unter 5 Sekunden dauern
        
        # Serialisierung testen
        start_time = time.time()
        
        serialized_stems = [stem.model_dump_json() for stem in stems[:100]]
        
        serialization_time = time.time() - start_time
        
        assert len(serialized_stems) == 100
        assert serialization_time < 1.0  # Sollte unter 1 Sekunde dauern