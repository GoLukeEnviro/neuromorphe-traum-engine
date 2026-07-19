"""Tests für PreprocessorService"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from pathlib import Path
import tempfile

from services.preprocessor import PreprocessorService
from core.config import Settings
from tests.conftest import create_test_audio_file, assert_audio_file_valid


class TestPreprocessorService:
    """Test-Suite für PreprocessorService"""
    
    @pytest.fixture
    def preprocessor(self, test_settings: Settings, mock_neuro_analyzer) -> PreprocessorService:
        """PreprocessorService-Instanz für Tests"""
        with patch('src.services.preprocessor.essentia'), \
             patch('src.services.preprocessor.es'):
            return PreprocessorService(test_settings, mock_neuro_analyzer)
    
    @pytest.mark.unit
    def test_initialization(self, test_settings: Settings, mock_neuro_analyzer):
        """Test: PreprocessorService-Initialisierung"""
        preprocessor = PreprocessorService(test_settings, mock_neuro_analyzer)
        
        assert preprocessor.settings == test_settings
        assert preprocessor.neuro_analyzer == mock_neuro_analyzer
        assert preprocessor.supported_formats == {".wav", ".mp3", ".flac", ".aiff", ".m4a"}
    
    @pytest.mark.unit
    async def test_process_audio_file_success(self, preprocessor: PreprocessorService, temp_dir: Path, test_db_session):
        """Test: Erfolgreiche Audio-Datei-Verarbeitung"""
        # Test-Audio-Datei erstellen
        audio_file = create_test_audio_file(temp_dir / "test.wav", duration=2.0)
        
        with patch('librosa.load') as mock_load, \
             patch.object(preprocessor, '_save_processed_stem') as mock_save:
            
            # Mock Librosa
            mock_audio_data = np.random.rand(44100 * 2)  # 2 Sekunden
            mock_load.return_value = (mock_audio_data, 44100)
            
            # Mock Stem-Speicherung
            mock_save.return_value = {
                "id": 1,
                "file_path": str(temp_dir / "processed" / "test_processed.wav"),
                "duration": 2.0
            }
            
            result = await preprocessor.process_audio_file(
                str(audio_file), 
                session=test_db_session
            )
            
            assert "stem_id" in result
            assert "duration" in result
            assert "sample_rate" in result
            assert result["duration"] == 2.0
            mock_load.assert_called_once()
            mock_save.assert_called_once()
    
    @pytest.mark.unit
    async def test_process_directory(self, preprocessor: PreprocessorService, temp_dir: Path, test_db_session):
        """Test: Verzeichnis-Verarbeitung"""
        # Mehrere Test-Audio-Dateien erstellen
        audio_files = []
        for i in range(3):
            file_path = create_test_audio_file(temp_dir / f"test_{i}.wav")
            audio_files.append(file_path)
        
        with patch.object(preprocessor, 'process_audio_file') as mock_process:
            mock_process.return_value = {
                "stem_id": 1,
                "duration": 1.0,
                "sample_rate": 44100
            }
            
            results = await preprocessor.process_directory(
                str(temp_dir),
                session=test_db_session
            )
            
            assert len(results) == 3
            assert mock_process.call_count == 3
            assert all("stem_id" in r for r in results)
    
    @pytest.mark.unit
    async def test_extract_features(self, preprocessor: PreprocessorService):
        """Test: Feature-Extraktion"""
        audio_data = np.random.rand(44100)  # 1 Sekunde
        sample_rate = 44100
        
        with patch('librosa.beat.tempo') as mock_tempo, \
             patch('librosa.feature.spectral_centroid') as mock_centroid, \
             patch('librosa.feature.mfcc') as mock_mfcc, \
             patch('librosa.feature.chroma') as mock_chroma:
            
            # Mock Librosa-Funktionen
            mock_tempo.return_value = np.array([128.0])
            mock_centroid.return_value = np.array([[2000.0]])
            mock_mfcc.return_value = np.random.rand(13, 100)
            mock_chroma.return_value = np.random.rand(12, 100)
            
            features = await preprocessor.extract_features(audio_data, sample_rate)
            
            assert "tempo" in features
            assert "spectral_centroid" in features
            assert "mfcc_mean" in features
            assert "chroma_mean" in features
            assert "energy" in features
            assert "zero_crossing_rate" in features
            
            # Werte-Validierung
            assert features["tempo"] == 128.0
            assert features["spectral_centroid"] == 2000.0
            assert len(features["mfcc_mean"]) == 13
            assert len(features["chroma_mean"]) == 12
    
    @pytest.mark.unit
    async def test_normalize_audio(self, preprocessor: PreprocessorService):
        """Test: Audio-Normalisierung"""
        # Audio mit verschiedenen Lautstärken
        quiet_audio = np.random.rand(1000) * 0.1
        loud_audio = np.random.rand(1000) * 2.0
        
        normalized_quiet = await preprocessor.normalize_audio(quiet_audio)
        normalized_loud = await preprocessor.normalize_audio(loud_audio)
        
        # Beide sollten ähnliche Maximalwerte haben
        assert abs(np.max(np.abs(normalized_quiet)) - np.max(np.abs(normalized_loud))) < 0.1
        
        # Sollten im Bereich [-1, 1] liegen
        assert np.max(np.abs(normalized_quiet)) <= 1.0
        assert np.max(np.abs(normalized_loud)) <= 1.0
    
    @pytest.mark.unit
    async def test_detect_silence(self, preprocessor: PreprocessorService):
        """Test: Stille-Erkennung"""
        sample_rate = 44100
        
        # Audio mit Stille am Anfang und Ende
        silence = np.zeros(sample_rate // 2)  # 0.5 Sekunden Stille
        audio_content = np.random.rand(sample_rate) * 0.5  # 1 Sekunde Audio
        audio_with_silence = np.concatenate([silence, audio_content, silence])
        
        trimmed_audio = await preprocessor.detect_silence(audio_with_silence, sample_rate)
        
        # Getrimmt sollte kürzer sein
        assert len(trimmed_audio) < len(audio_with_silence)
        assert len(trimmed_audio) <= len(audio_content) * 1.1  # Etwas Toleranz
    
    @pytest.mark.unit
    async def test_convert_sample_rate(self, preprocessor: PreprocessorService):
        """Test: Sample-Rate-Konvertierung"""
        # Audio mit 22050 Hz
        original_audio = np.random.rand(22050)  # 1 Sekunde bei 22050 Hz
        original_sr = 22050
        target_sr = 44100
        
        with patch('librosa.resample') as mock_resample:
            expected_length = int(len(original_audio) * target_sr / original_sr)
            mock_resample.return_value = np.random.rand(expected_length)
            
            converted_audio = await preprocessor.convert_sample_rate(
                original_audio, original_sr, target_sr
            )
            
            mock_resample.assert_called_once_with(
                original_audio, orig_sr=original_sr, target_sr=target_sr
            )
            assert len(converted_audio) == expected_length
    
    @pytest.mark.unit
    async def test_validate_audio_file(self, preprocessor: PreprocessorService, temp_dir: Path):
        """Test: Audio-Datei-Validierung"""
        # Gültige Audio-Datei
        valid_file = create_test_audio_file(temp_dir / "valid.wav")
        
        # Ungültige Datei
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("This is not audio")
        
        # Nicht existierende Datei
        nonexistent_file = temp_dir / "nonexistent.wav"
        
        assert await preprocessor.validate_audio_file(str(valid_file)) == True
        assert await preprocessor.validate_audio_file(str(invalid_file)) == False
        assert await preprocessor.validate_audio_file(str(nonexistent_file)) == False
    
    @pytest.mark.unit
    async def test_batch_process(self, preprocessor: PreprocessorService, temp_dir: Path, test_db_session):
        """Test: Batch-Verarbeitung"""
        # Mehrere Audio-Dateien erstellen
        audio_files = []
        for i in range(5):
            file_path = create_test_audio_file(temp_dir / f"batch_{i}.wav")
            audio_files.append(str(file_path))
        
        with patch.object(preprocessor, 'process_audio_file') as mock_process:
            mock_process.return_value = {
                "stem_id": 1,
                "duration": 1.0,
                "sample_rate": 44100
            }
            
            results = await preprocessor.batch_process(
                audio_files,
                session=test_db_session,
                max_concurrent=2
            )
            
            assert len(results) == 5
            assert mock_process.call_count == 5
            assert all("stem_id" in r for r in results)
    
    @pytest.mark.unit
    async def test_error_handling_corrupted_file(self, preprocessor: PreprocessorService, temp_dir: Path, test_db_session):
        """Test: Fehlerbehandlung bei korrupter Datei"""
        # Korrupte Audio-Datei simulieren
        corrupted_file = temp_dir / "corrupted.wav"
        corrupted_file.write_bytes(b"corrupted_audio_data")
        
        with patch('librosa.load') as mock_load:
            mock_load.side_effect = Exception("Corrupted audio file")
            
            with pytest.raises(Exception):
                await preprocessor.process_audio_file(
                    str(corrupted_file),
                    session=test_db_session
                )
    
    @pytest.mark.unit
    async def test_unsupported_format(self, preprocessor: PreprocessorService, temp_dir: Path, test_db_session):
        """Test: Nicht unterstütztes Audio-Format"""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_bytes(b"fake_audio")
        
        with pytest.raises(ValueError, match="Unsupported audio format"):
            await preprocessor.process_audio_file(
                str(unsupported_file),
                session=test_db_session
            )
    
    @pytest.mark.audio
    async def test_real_audio_processing(self, preprocessor: PreprocessorService, temp_dir: Path, test_db_session):
        """Test: Echte Audio-Verarbeitung (mit echten Bibliotheken)"""
        # Dieser Test verwendet echte Audio-Bibliotheken
        pytest.skip("Echter Audio-Test - nur bei Bedarf ausführen")
        
        # Echte Audio-Datei erstellen
        audio_file = create_test_audio_file(temp_dir / "real_test.wav", duration=3.0)
        
        result = await preprocessor.process_audio_file(
            str(audio_file),
            session=test_db_session
        )
        
        assert "stem_id" in result
        assert result["duration"] > 0
        assert result["sample_rate"] > 0
    
    @pytest.mark.performance
    async def test_performance_large_file(self, preprocessor: PreprocessorService, temp_dir: Path, test_db_session):
        """Test: Performance bei großen Dateien"""
        import time
        
        # Große Audio-Datei simulieren
        large_audio_file = create_test_audio_file(temp_dir / "large.wav", duration=60.0)
        
        with patch('librosa.load') as mock_load:
            # 60 Sekunden Audio bei 44100 Hz
            mock_load.return_value = (np.random.rand(44100 * 60), 44100)
            
            with patch.object(preprocessor, '_save_processed_stem') as mock_save:
                mock_save.return_value = {"id": 1, "file_path": "/test", "duration": 60.0}
                
                start_time = time.time()
                result = await preprocessor.process_audio_file(
                    str(large_audio_file),
                    session=test_db_session
                )
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                assert "stem_id" in result
                assert processing_time < 10.0  # Sollte unter 10 Sekunden dauern
    
    @pytest.mark.unit
    async def test_metadata_extraction(self, preprocessor: PreprocessorService):
        """Test: Metadaten-Extraktion"""
        audio_data = np.random.rand(44100 * 2)  # 2 Sekunden
        sample_rate = 44100
        file_path = "/test/audio.wav"
        
        metadata = await preprocessor.extract_metadata(audio_data, sample_rate, file_path)
        
        assert "duration" in metadata
        assert "sample_rate" in metadata
        assert "channels" in metadata
        assert "file_size" in metadata
        assert "format" in metadata
        
        assert metadata["duration"] == 2.0
        assert metadata["sample_rate"] == 44100
        assert metadata["format"] == "wav"
    
    @pytest.mark.unit
    async def test_progress_callback(self, preprocessor: PreprocessorService, temp_dir: Path, test_db_session):
        """Test: Progress-Callback-Funktionalität"""
        progress_updates = []
        
        def progress_callback(current: int, total: int, message: str):
            progress_updates.append((current, total, message))
        
        audio_files = []
        for i in range(3):
            file_path = create_test_audio_file(temp_dir / f"progress_{i}.wav")
            audio_files.append(str(file_path))
        
        with patch.object(preprocessor, 'process_audio_file') as mock_process:
            mock_process.return_value = {"stem_id": 1, "duration": 1.0}
            
            await preprocessor.batch_process(
                audio_files,
                session=test_db_session,
                progress_callback=progress_callback
            )
            
            assert len(progress_updates) > 0
            assert any("Processing" in msg for _, _, msg in progress_updates)