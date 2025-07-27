"""Tests für CLAP-Service"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.services.clap_service import (
    CLAPService, CLAPModelManager, CLAPEmbedding,
    CLAPAnalysisResult, CLAPSimilarityResult,
    CLAPBatchProcessor, CLAPCache
)
from src.core.config import CLAPConfig
from src.core.exceptions import (
    CLAPModelError, AudioProcessingError,
    ValidationError, ConfigurationError
)
from src.schemas.stem import StemCreate, StemResponse
from src.schemas.api import AnalysisRequest, AnalysisResponse


class TestCLAPService:
    """Tests für CLAP-Service"""
    
    @pytest.fixture
    def clap_config(self):
        """CLAP-Konfiguration für Tests"""
        return CLAPConfig(
            model_name="laion/clap-htsat-unfused",
            model_path="./models/clap",
            device="cpu",
            batch_size=4,
            max_audio_length=30.0,
            sample_rate=48000,
            enable_cache=True,
            cache_size=1000,
            embedding_dim=512
        )
    
    @pytest.fixture
    def mock_model(self):
        """Mock CLAP-Modell"""
        model = Mock()
        model.encode_audio.return_value = torch.randn(1, 512)
        model.encode_text.return_value = torch.randn(1, 512)
        model.get_audio_embedding.return_value = torch.randn(1, 512)
        model.get_text_embedding.return_value = torch.randn(1, 512)
        return model
    
    @pytest.fixture
    def clap_service(self, clap_config, mock_model):
        """CLAP-Service für Tests"""
        with patch('src.services.clap_service.CLAPModelManager') as mock_manager:
            mock_manager.return_value.load_model.return_value = mock_model
            mock_manager.return_value.model = mock_model
            service = CLAPService(clap_config)
            return service
    
    @pytest.mark.unit
    def test_clap_service_initialization(self, clap_config):
        """Test: CLAP-Service-Initialisierung"""
        with patch('src.services.clap_service.CLAPModelManager'):
            service = CLAPService(clap_config)
            
            assert service.config == clap_config
            assert service.device == "cpu"
            assert service.batch_size == 4
            assert service.max_audio_length == 30.0
    
    @pytest.mark.unit
    def test_clap_service_invalid_config(self):
        """Test: CLAP-Service mit ungültiger Konfiguration"""
        invalid_config = CLAPConfig(
            model_name="",  # Leerer Modellname
            device="cpu"
        )
        
        with pytest.raises(ConfigurationError):
            CLAPService(invalid_config)
    
    @pytest.mark.unit
    async def test_analyze_audio_file(self, clap_service):
        """Test: Audio-Datei-Analyse"""
        # Mock Audio-Datei
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_path = temp_file.name
        
        try:
            with patch('src.services.clap_service.load_audio') as mock_load:
                mock_load.return_value = (np.random.randn(48000), 48000)
                
                result = await clap_service.analyze_audio_file(audio_path)
                
                assert isinstance(result, CLAPAnalysisResult)
                assert result.file_path == audio_path
                assert result.embedding is not None
                assert len(result.embedding) == 512
                assert result.confidence > 0.0
                assert result.processing_time > 0.0
        finally:
            Path(audio_path).unlink(missing_ok=True)
    
    @pytest.mark.unit
    async def test_analyze_audio_file_invalid_path(self, clap_service):
        """Test: Audio-Datei-Analyse mit ungültigem Pfad"""
        with pytest.raises(AudioProcessingError):
            await clap_service.analyze_audio_file("/nonexistent/file.wav")
    
    @pytest.mark.unit
    async def test_analyze_text_prompt(self, clap_service):
        """Test: Text-Prompt-Analyse"""
        prompt = "energetic techno beat with heavy bass"
        
        result = await clap_service.analyze_text_prompt(prompt)
        
        assert isinstance(result, CLAPAnalysisResult)
        assert result.text_prompt == prompt
        assert result.embedding is not None
        assert len(result.embedding) == 512
        assert result.confidence > 0.0
    
    @pytest.mark.unit
    async def test_analyze_text_prompt_empty(self, clap_service):
        """Test: Text-Prompt-Analyse mit leerem Text"""
        with pytest.raises(ValidationError):
            await clap_service.analyze_text_prompt("")
    
    @pytest.mark.unit
    async def test_find_similar_stems(self, clap_service):
        """Test: Ähnliche Stems finden"""
        query_embedding = np.random.randn(512).tolist()
        
        # Mock Datenbank-Stems
        mock_stems = [
            StemResponse(
                id="stem_1",
                name="Techno Beat 1",
                embedding=np.random.randn(512).tolist(),
                genre="techno",
                bpm=128
            ),
            StemResponse(
                id="stem_2",
                name="House Groove",
                embedding=np.random.randn(512).tolist(),
                genre="house",
                bpm=124
            )
        ]
        
        with patch.object(clap_service, '_get_stems_from_database') as mock_get:
            mock_get.return_value = mock_stems
            
            results = await clap_service.find_similar_stems(
                query_embedding,
                limit=10,
                threshold=0.7
            )
            
            assert isinstance(results, list)
            assert len(results) <= 10
            for result in results:
                assert isinstance(result, CLAPSimilarityResult)
                assert result.similarity >= 0.7
                assert result.stem_id in ["stem_1", "stem_2"]
    
    @pytest.mark.unit
    async def test_find_similar_stems_by_text(self, clap_service):
        """Test: Ähnliche Stems per Text finden"""
        query_text = "dark atmospheric techno"
        
        # Mock Text-Analyse
        with patch.object(clap_service, 'analyze_text_prompt') as mock_analyze:
            mock_analysis = CLAPAnalysisResult(
                text_prompt=query_text,
                embedding=np.random.randn(512).tolist(),
                confidence=0.95,
                processing_time=0.1
            )
            mock_analyze.return_value = mock_analysis
            
            # Mock Ähnlichkeitssuche
            with patch.object(clap_service, 'find_similar_stems') as mock_find:
                mock_results = [
                    CLAPSimilarityResult(
                        stem_id="stem_1",
                        similarity=0.85,
                        embedding=np.random.randn(512).tolist()
                    )
                ]
                mock_find.return_value = mock_results
                
                results = await clap_service.find_similar_stems_by_text(
                    query_text,
                    limit=5
                )
                
                assert len(results) == 1
                assert results[0].similarity == 0.85
                mock_analyze.assert_called_once_with(query_text)
    
    @pytest.mark.unit
    async def test_batch_analyze_audio_files(self, clap_service):
        """Test: Batch-Audio-Analyse"""
        # Mock Audio-Dateien
        audio_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_files.append(temp_file.name)
        
        try:
            with patch('src.services.clap_service.load_audio') as mock_load:
                mock_load.return_value = (np.random.randn(48000), 48000)
                
                results = await clap_service.batch_analyze_audio_files(audio_files)
                
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert isinstance(result, CLAPAnalysisResult)
                    assert result.file_path == audio_files[i]
                    assert result.embedding is not None
        finally:
            for file_path in audio_files:
                Path(file_path).unlink(missing_ok=True)
    
    @pytest.mark.unit
    async def test_batch_analyze_with_errors(self, clap_service):
        """Test: Batch-Analyse mit Fehlern"""
        audio_files = [
            "/valid/file1.wav",
            "/invalid/file.wav",  # Existiert nicht
            "/valid/file2.wav"
        ]
        
        def mock_load_audio(path):
            if "invalid" in path:
                raise FileNotFoundError(f"File not found: {path}")
            return (np.random.randn(48000), 48000)
        
        with patch('src.services.clap_service.load_audio', side_effect=mock_load_audio):
            results = await clap_service.batch_analyze_audio_files(
                audio_files,
                skip_errors=True
            )
            
            # Nur 2 erfolgreiche Ergebnisse
            assert len(results) == 2
            assert all(isinstance(r, CLAPAnalysisResult) for r in results)
    
    @pytest.mark.unit
    def test_calculate_similarity(self, clap_service):
        """Test: Ähnlichkeitsberechnung"""
        embedding1 = np.random.randn(512)
        embedding2 = np.random.randn(512)
        
        # Cosinus-Ähnlichkeit
        similarity = clap_service.calculate_similarity(embedding1, embedding2)
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
        
        # Identische Embeddings
        identical_similarity = clap_service.calculate_similarity(embedding1, embedding1)
        assert abs(identical_similarity - 1.0) < 1e-6
    
    @pytest.mark.unit
    def test_calculate_similarity_different_methods(self, clap_service):
        """Test: Verschiedene Ähnlichkeitsmethoden"""
        embedding1 = np.random.randn(512)
        embedding2 = np.random.randn(512)
        
        # Cosinus-Ähnlichkeit (Standard)
        cosine_sim = clap_service.calculate_similarity(
            embedding1, embedding2, method="cosine"
        )
        
        # Euklidische Distanz
        euclidean_sim = clap_service.calculate_similarity(
            embedding1, embedding2, method="euclidean"
        )
        
        # Dot-Product
        dot_sim = clap_service.calculate_similarity(
            embedding1, embedding2, method="dot"
        )
        
        assert isinstance(cosine_sim, float)
        assert isinstance(euclidean_sim, float)
        assert isinstance(dot_sim, float)
        
        # Verschiedene Methoden sollten verschiedene Ergebnisse liefern
        assert cosine_sim != euclidean_sim or euclidean_sim != dot_sim


class TestCLAPModelManager:
    """Tests für CLAP-Model-Manager"""
    
    @pytest.fixture
    def clap_config(self):
        """CLAP-Konfiguration für Tests"""
        return CLAPConfig(
            model_name="laion/clap-htsat-unfused",
            model_path="./models/clap",
            device="cpu"
        )
    
    @pytest.mark.unit
    def test_model_manager_initialization(self, clap_config):
        """Test: Model-Manager-Initialisierung"""
        manager = CLAPModelManager(clap_config)
        
        assert manager.config == clap_config
        assert manager.device == "cpu"
        assert manager.model is None
    
    @pytest.mark.unit
    def test_load_model_success(self, clap_config):
        """Test: Erfolgreiches Modell-Laden"""
        with patch('transformers.ClapModel.from_pretrained') as mock_from_pretrained:
            with patch('transformers.ClapProcessor.from_pretrained') as mock_processor:
                mock_model = Mock()
                mock_from_pretrained.return_value = mock_model
                mock_processor.return_value = Mock()
                
                manager = CLAPModelManager(clap_config)
                model = manager.load_model()
                
                assert model == mock_model
                assert manager.model == mock_model
                mock_from_pretrained.assert_called_once()
    
    @pytest.mark.unit
    def test_load_model_failure(self, clap_config):
        """Test: Fehlgeschlagenes Modell-Laden"""
        with patch('transformers.ClapModel.from_pretrained') as mock_from_pretrained:
            mock_from_pretrained.side_effect = Exception("Model not found")
            
            manager = CLAPModelManager(clap_config)
            
            with pytest.raises(CLAPModelError):
                manager.load_model()
    
    @pytest.mark.unit
    def test_model_caching(self, clap_config):
        """Test: Modell-Caching"""
        with patch('transformers.ClapModel.from_pretrained') as mock_from_pretrained:
            with patch('transformers.ClapProcessor.from_pretrained'):
                mock_model = Mock()
                mock_from_pretrained.return_value = mock_model
                
                manager = CLAPModelManager(clap_config)
                
                # Erstes Laden
                model1 = manager.load_model()
                
                # Zweites Laden (sollte gecacht sein)
                model2 = manager.load_model()
                
                assert model1 == model2
                # from_pretrained sollte nur einmal aufgerufen werden
                mock_from_pretrained.assert_called_once()
    
    @pytest.mark.unit
    def test_device_handling(self):
        """Test: Device-Handling"""
        # CPU-Konfiguration
        cpu_config = CLAPConfig(device="cpu")
        cpu_manager = CLAPModelManager(cpu_config)
        assert cpu_manager.device == "cpu"
        
        # GPU-Konfiguration (falls verfügbar)
        if torch.cuda.is_available():
            gpu_config = CLAPConfig(device="cuda")
            gpu_manager = CLAPModelManager(gpu_config)
            assert gpu_manager.device == "cuda"
        
        # Auto-Device-Erkennung
        auto_config = CLAPConfig(device="auto")
        auto_manager = CLAPModelManager(auto_config)
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert auto_manager.device == expected_device


class TestCLAPEmbedding:
    """Tests für CLAP-Embedding"""
    
    @pytest.mark.unit
    def test_embedding_creation(self):
        """Test: Embedding-Erstellung"""
        vector = np.random.randn(512).tolist()
        
        embedding = CLAPEmbedding(
            vector=vector,
            dimension=512,
            model_name="laion/clap-htsat-unfused",
            created_at=datetime.now()
        )
        
        assert embedding.vector == vector
        assert embedding.dimension == 512
        assert embedding.model_name == "laion/clap-htsat-unfused"
        assert isinstance(embedding.created_at, datetime)
    
    @pytest.mark.unit
    def test_embedding_validation(self):
        """Test: Embedding-Validierung"""
        # Ungültige Dimension
        with pytest.raises(ValidationError):
            CLAPEmbedding(
                vector=[1.0, 2.0, 3.0],  # 3 Elemente
                dimension=512,  # Aber 512 erwartet
                model_name="test"
            )
        
        # Leerer Vektor
        with pytest.raises(ValidationError):
            CLAPEmbedding(
                vector=[],
                dimension=0,
                model_name="test"
            )
    
    @pytest.mark.unit
    def test_embedding_normalization(self):
        """Test: Embedding-Normalisierung"""
        vector = [3.0, 4.0, 0.0]  # Länge = 5
        
        embedding = CLAPEmbedding(
            vector=vector,
            dimension=3,
            model_name="test"
        )
        
        normalized = embedding.normalize()
        
        # Normalisierter Vektor sollte Länge 1 haben
        norm = np.linalg.norm(normalized.vector)
        assert abs(norm - 1.0) < 1e-6
    
    @pytest.mark.unit
    def test_embedding_similarity(self):
        """Test: Embedding-Ähnlichkeit"""
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [0.0, 1.0, 0.0]
        vector3 = [1.0, 0.0, 0.0]  # Identisch mit vector1
        
        embedding1 = CLAPEmbedding(vector=vector1, dimension=3, model_name="test")
        embedding2 = CLAPEmbedding(vector=vector2, dimension=3, model_name="test")
        embedding3 = CLAPEmbedding(vector=vector3, dimension=3, model_name="test")
        
        # Orthogonale Vektoren
        similarity_12 = embedding1.similarity(embedding2)
        assert abs(similarity_12) < 1e-6
        
        # Identische Vektoren
        similarity_13 = embedding1.similarity(embedding3)
        assert abs(similarity_13 - 1.0) < 1e-6


class TestCLAPBatchProcessor:
    """Tests für CLAP-Batch-Processor"""
    
    @pytest.fixture
    def batch_processor(self):
        """Batch-Processor für Tests"""
        config = CLAPConfig(batch_size=4)
        with patch('src.services.clap_service.CLAPModelManager'):
            return CLAPBatchProcessor(config)
    
    @pytest.mark.unit
    async def test_batch_processing(self, batch_processor):
        """Test: Batch-Verarbeitung"""
        # Mock Eingabedaten
        audio_data = [np.random.randn(48000) for _ in range(10)]
        
        with patch.object(batch_processor, '_process_batch') as mock_process:
            mock_process.return_value = [
                CLAPAnalysisResult(
                    embedding=np.random.randn(512).tolist(),
                    confidence=0.9,
                    processing_time=0.1
                )
            ]
            
            results = await batch_processor.process_audio_batch(audio_data)
            
            assert len(results) == 10
            # Sollte in 3 Batches aufgeteilt werden (4+4+2)
            assert mock_process.call_count == 3
    
    @pytest.mark.unit
    async def test_batch_size_optimization(self, batch_processor):
        """Test: Batch-Size-Optimierung"""
        # Kleine Eingabe (weniger als Batch-Size)
        small_data = [np.random.randn(48000) for _ in range(2)]
        
        with patch.object(batch_processor, '_process_batch') as mock_process:
            mock_process.return_value = [
                CLAPAnalysisResult(
                    embedding=np.random.randn(512).tolist(),
                    confidence=0.9,
                    processing_time=0.1
                )
            ]
            
            results = await batch_processor.process_audio_batch(small_data)
            
            assert len(results) == 2
            # Sollte nur einen Batch geben
            mock_process.assert_called_once()
    
    @pytest.mark.unit
    async def test_parallel_processing(self, batch_processor):
        """Test: Parallele Verarbeitung"""
        audio_data = [np.random.randn(48000) for _ in range(8)]
        
        with patch.object(batch_processor, '_process_batch') as mock_process:
            # Simuliere unterschiedliche Verarbeitungszeiten
            mock_process.side_effect = [
                [CLAPAnalysisResult(
                    embedding=np.random.randn(512).tolist(),
                    confidence=0.9,
                    processing_time=0.1
                )] * 4,
                [CLAPAnalysisResult(
                    embedding=np.random.randn(512).tolist(),
                    confidence=0.9,
                    processing_time=0.2
                )] * 4
            ]
            
            results = await batch_processor.process_audio_batch(
                audio_data,
                parallel=True
            )
            
            assert len(results) == 8
            assert mock_process.call_count == 2


class TestCLAPCache:
    """Tests für CLAP-Cache"""
    
    @pytest.fixture
    def clap_cache(self):
        """CLAP-Cache für Tests"""
        return CLAPCache(max_size=100)
    
    @pytest.mark.unit
    def test_cache_basic_operations(self, clap_cache):
        """Test: Grundlegende Cache-Operationen"""
        key = "test_audio.wav"
        embedding = np.random.randn(512).tolist()
        
        # Cache Miss
        assert clap_cache.get(key) is None
        
        # Cache Set
        clap_cache.set(key, embedding)
        
        # Cache Hit
        cached_embedding = clap_cache.get(key)
        assert cached_embedding == embedding
    
    @pytest.mark.unit
    def test_cache_size_limit(self, clap_cache):
        """Test: Cache-Größenbegrenzung"""
        # Cache mit mehr Einträgen füllen als erlaubt
        for i in range(150):
            key = f"audio_{i}.wav"
            embedding = np.random.randn(512).tolist()
            clap_cache.set(key, embedding)
        
        # Cache sollte auf max_size begrenzt sein
        assert len(clap_cache._cache) <= 100
        
        # Neueste Einträge sollten noch da sein
        assert clap_cache.get("audio_149.wav") is not None
        
        # Älteste Einträge sollten entfernt worden sein
        assert clap_cache.get("audio_0.wav") is None
    
    @pytest.mark.unit
    def test_cache_statistics(self, clap_cache):
        """Test: Cache-Statistiken"""
        key = "test_audio.wav"
        embedding = np.random.randn(512).tolist()
        
        # Initial keine Statistiken
        stats = clap_cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        
        # Cache Miss
        clap_cache.get(key)
        stats = clap_cache.get_stats()
        assert stats["misses"] == 1
        
        # Cache Set und Hit
        clap_cache.set(key, embedding)
        clap_cache.get(key)
        stats = clap_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    @pytest.mark.unit
    def test_cache_clear(self, clap_cache):
        """Test: Cache leeren"""
        # Cache füllen
        for i in range(10):
            key = f"audio_{i}.wav"
            embedding = np.random.randn(512).tolist()
            clap_cache.set(key, embedding)
        
        assert len(clap_cache._cache) == 10
        
        # Cache leeren
        clap_cache.clear()
        
        assert len(clap_cache._cache) == 0
        assert clap_cache.get("audio_0.wav") is None


class TestCLAPServiceIntegration:
    """Integrationstests für CLAP-Service"""
    
    @pytest.mark.integration
    async def test_full_analysis_workflow(self):
        """Test: Vollständiger Analyse-Workflow"""
        config = CLAPConfig(
            model_name="laion/clap-htsat-unfused",
            device="cpu",
            enable_cache=True
        )
        
        with patch('transformers.ClapModel.from_pretrained'):
            with patch('transformers.ClapProcessor.from_pretrained'):
                with patch('src.services.clap_service.load_audio') as mock_load:
                    mock_load.return_value = (np.random.randn(48000), 48000)
                    
                    service = CLAPService(config)
                    
                    # 1. Audio-Analyse
                    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                        audio_result = await service.analyze_audio_file(temp_file.name)
                        assert isinstance(audio_result, CLAPAnalysisResult)
                    
                    # 2. Text-Analyse
                    text_result = await service.analyze_text_prompt("energetic techno")
                    assert isinstance(text_result, CLAPAnalysisResult)
                    
                    # 3. Ähnlichkeitssuche
                    with patch.object(service, '_get_stems_from_database') as mock_get:
                        mock_get.return_value = []
                        
                        similar_stems = await service.find_similar_stems(
                            audio_result.embedding,
                            limit=5
                        )
                        assert isinstance(similar_stems, list)
    
    @pytest.mark.performance
    async def test_clap_service_performance(self):
        """Test: CLAP-Service-Performance"""
        import time
        
        config = CLAPConfig(
            model_name="laion/clap-htsat-unfused",
            device="cpu",
            batch_size=8
        )
        
        with patch('transformers.ClapModel.from_pretrained'):
            with patch('transformers.ClapProcessor.from_pretrained'):
                with patch('src.services.clap_service.load_audio') as mock_load:
                    mock_load.return_value = (np.random.randn(48000), 48000)
                    
                    service = CLAPService(config)
                    
                    # Performance-Test: Batch-Analyse
                    audio_files = [f"/tmp/audio_{i}.wav" for i in range(20)]
                    
                    start_time = time.time()
                    results = await service.batch_analyze_audio_files(audio_files)
                    processing_time = time.time() - start_time
                    
                    assert len(results) == 20
                    assert processing_time < 10.0  # Sollte unter 10 Sekunden dauern
                    
                    # Durchschnittliche Verarbeitungszeit pro Datei
                    avg_time_per_file = processing_time / len(audio_files)
                    assert avg_time_per_file < 0.5  # Unter 0.5 Sekunden pro Datei