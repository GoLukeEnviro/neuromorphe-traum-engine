"""Tests für NeuroAnalyzer Service"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

from services.neuro_analyzer import NeuroAnalyzer
from core.config import Settings


class TestNeuroAnalyzer:
    """Test-Suite für NeuroAnalyzer"""
    
    @pytest.fixture
    def analyzer(self) -> NeuroAnalyzer:
        """NeuroAnalyzer-Instanz für Tests"""
        with patch('services.neuro_analyzer.SemanticAnalyzer') as mock_semantic, patch('services.neuro_analyzer.PatternAnalyzer') as mock_pattern:
            analyzer = NeuroAnalyzer()
            analyzer.semantic_analyzer = mock_semantic.return_value
            analyzer.pattern_analyzer = mock_pattern.return_value
            return analyzer
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test: NeuroAnalyzer-Initialisierung"""
        with patch('services.neuro_analyzer.SemanticAnalyzer') as mock_semantic, \
             patch('services.neuro_analyzer.PatternAnalyzer') as mock_pattern:
            
            analyzer = NeuroAnalyzer()
            
            assert analyzer.semantic_analyzer is not None
            assert analyzer.pattern_analyzer is not None
            mock_semantic.assert_called_once()
            mock_pattern.assert_called_once()
    
    @pytest.mark.unit
    async def test_analyze_audio_success(self, analyzer: NeuroAnalyzer, sample_audio_data: bytes):
        """Test: Erfolgreiche Audio-Analyse"""
        # Mock CLAP-Modell-Ausgabe
        mock_embeddings = np.random.rand(512).astype(np.float32)
        analyzer.model.get_audio_embedding_from_data.return_value = mock_embeddings
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (np.random.rand(22050), 22050)
            
            result = await analyzer.analyze_audio(sample_audio_data)
            
            assert "embeddings" in result
            assert "features" in result
            assert len(result["embeddings"]) == 512
            assert "tempo" in result["features"]
            assert "energy" in result["features"]
    
    @pytest.mark.unit
    async def test_analyze_text_prompt(self, analyzer: NeuroAnalyzer):
        """Test: Text-Prompt-Analyse"""
        prompt = "Dark atmospheric techno with heavy bass"
        mock_embeddings = np.random.rand(512).astype(np.float32)
        
        analyzer.processor.return_value = {"input_ids": [[1, 2, 3]]}
        analyzer.model.get_text_embedding.return_value = mock_embeddings
        
        result = await analyzer.analyze_text_prompt(prompt)
        
        assert "embeddings" in result
        assert "prompt" in result
        assert result["prompt"] == prompt
        assert len(result["embeddings"]) == 512
    
    @pytest.mark.unit
    async def test_calculate_similarity(self, analyzer: NeuroAnalyzer):
        """Test: Ähnlichkeitsberechnung"""
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding2 = np.random.rand(512).astype(np.float32)
        
        similarity = await analyzer.calculate_similarity(embedding1, embedding2)
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    @pytest.mark.unit
    async def test_get_similar_stems(self, analyzer: NeuroAnalyzer, test_db_session):
        """Test: Ähnliche Stems finden"""
        query_embedding = np.random.rand(512).astype(np.float32)
        
        with patch.object(analyzer, '_query_similar_stems') as mock_query:
            mock_query.return_value = [
                {"id": 1, "similarity": 0.9, "file_path": "/test/stem1.wav"},
                {"id": 2, "similarity": 0.8, "file_path": "/test/stem2.wav"}
            ]
            
            results = await analyzer.get_similar_stems(
                query_embedding, 
                session=test_db_session,
                limit=5,
                threshold=0.7
            )
            
            assert len(results) == 2
            assert results[0]["similarity"] >= results[1]["similarity"]
            assert all(r["similarity"] >= 0.7 for r in results)
    
    @pytest.mark.unit
    async def test_extract_audio_features(self, analyzer: NeuroAnalyzer):
        """Test: Audio-Feature-Extraktion"""
        # Mock Librosa-Funktionen
        with patch('librosa.beat.tempo') as mock_tempo, \
             patch('librosa.feature.spectral_centroid') as mock_centroid, \
             patch('librosa.feature.mfcc') as mock_mfcc:
            
            mock_tempo.return_value = np.array([128.0])
            mock_centroid.return_value = np.array([[1000.0]])
            mock_mfcc.return_value = np.random.rand(13, 100)
            
            audio_data = np.random.rand(22050)
            sample_rate = 22050
            
            features = await analyzer.extract_audio_features(audio_data, sample_rate)
            
            assert "tempo" in features
            assert "spectral_centroid" in features
            assert "mfcc" in features
            assert "energy" in features
            assert "zero_crossing_rate" in features
    
    @pytest.mark.unit
    async def test_batch_analyze_stems(self, analyzer: NeuroAnalyzer, temp_dir: Path):
        """Test: Batch-Analyse von Stems"""
        # Test-Audio-Dateien erstellen
        audio_files = []
        for i in range(3):
            file_path = temp_dir / f"test_stem_{i}.wav"
            # Einfache Test-Datei erstellen
            file_path.write_bytes(b"fake_audio_data")
            audio_files.append(str(file_path))
        
        with patch.object(analyzer, 'analyze_audio') as mock_analyze:
            mock_analyze.return_value = {
                "embeddings": np.random.rand(512).tolist(),
                "features": {"tempo": 128.0, "energy": 0.8}
            }
            
            results = await analyzer.batch_analyze_stems(audio_files)
            
            assert len(results) == 3
            assert all("embeddings" in r for r in results)
            assert all("features" in r for r in results)
    
    @pytest.mark.unit
    async def test_error_handling_invalid_audio(self, analyzer: NeuroAnalyzer):
        """Test: Fehlerbehandlung bei ungültigen Audio-Daten"""
        invalid_audio = b"not_audio_data"
        
        with pytest.raises(Exception):
            await analyzer.analyze_audio(invalid_audio)
    
    @pytest.mark.unit
    async def test_error_handling_empty_prompt(self, analyzer: NeuroAnalyzer):
        """Test: Fehlerbehandlung bei leerem Prompt"""
        with pytest.raises(ValueError):
            await analyzer.analyze_text_prompt("")
    
    @pytest.mark.clap
    @pytest.mark.slow
    async def test_real_clap_model_integration(self, test_settings: Settings):
        """Test: Integration mit echtem CLAP-Modell (langsam)"""
        # Dieser Test lädt das echte CLAP-Modell
        # Nur ausführen wenn explizit gewünscht
        pytest.skip("Echter CLAP-Test - nur bei Bedarf ausführen")
        
        analyzer = NeuroAnalyzer(test_settings)
        
        # Test mit echtem Audio
        test_prompt = "Electronic music with synthesizers"
        result = await analyzer.analyze_text_prompt(test_prompt)
        
        assert "embeddings" in result
        assert len(result["embeddings"]) > 0
    
    @pytest.mark.performance
    async def test_performance_batch_processing(self, analyzer: NeuroAnalyzer, temp_dir: Path):
        """Test: Performance bei Batch-Verarbeitung"""
        import time
        
        # Viele Test-Dateien erstellen
        audio_files = []
        for i in range(10):
            file_path = temp_dir / f"perf_test_{i}.wav"
            file_path.write_bytes(b"fake_audio_data" * 1000)
            audio_files.append(str(file_path))
        
        with patch.object(analyzer, 'analyze_audio') as mock_analyze:
            mock_analyze.return_value = {
                "embeddings": np.random.rand(512).tolist(),
                "features": {"tempo": 128.0}
            }
            
            start_time = time.time()
            results = await analyzer.batch_analyze_stems(audio_files)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            assert len(results) == 10
            assert processing_time < 5.0  # Sollte unter 5 Sekunden dauern
    
    @pytest.mark.unit
    async def test_embedding_normalization(self, analyzer: NeuroAnalyzer):
        """Test: Embedding-Normalisierung"""
        # Unnormalisierte Embeddings
        raw_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        
        normalized = analyzer._normalize_embedding(raw_embedding)
        
        # Prüfen ob normalisiert (L2-Norm = 1)
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6
    
    @pytest.mark.unit
    async def test_cache_functionality(self, analyzer: NeuroAnalyzer):
        """Test: Caching-Funktionalität"""
        prompt = "Test prompt for caching"
        
        with patch.object(analyzer, '_compute_text_embedding') as mock_compute:
            mock_embedding = np.random.rand(512)
            mock_compute.return_value = mock_embedding
            
            # Erste Anfrage
            result1 = await analyzer.analyze_text_prompt(prompt)
            
            # Zweite Anfrage (sollte gecacht sein)
            result2 = await analyzer.analyze_text_prompt(prompt)
            
            # Compute sollte nur einmal aufgerufen werden
            assert mock_compute.call_count == 1
            assert np.array_equal(result1["embeddings"], result2["embeddings"])