"""Tests für Audio-Service"""

import pytest
import numpy as np
import librosa
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import io
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.services.audio_service import (
    AudioService, AudioProcessor, AudioAnalyzer,
    AudioLoader, AudioExporter, AudioValidator,
    AudioMetadata, AudioFeatures, AudioSegment,
    SpectralAnalyzer, TemporalAnalyzer, HarmonicAnalyzer
)
from src.core.config import AudioConfig
from src.core.exceptions import (
    AudioProcessingError, ValidationError,
    ConfigurationError, FileNotFoundError
)
from src.schemas.stem import StemCreate, StemResponse
from src.database.models import Stem as StemModel


class TestAudioService:
    """Tests für Audio-Service"""
    
    @pytest.fixture
    def audio_config(self):
        """Audio-Konfiguration für Tests"""
        return AudioConfig(
            sample_rate=48000,
            bit_depth=24,
            channels=2,
            chunk_size=1024,
            supported_formats=["wav", "mp3", "flac", "aiff"],
            max_file_size=100 * 1024 * 1024,  # 100MB
            enable_normalization=True,
            enable_noise_reduction=False,
            analysis_window_size=2048,
            analysis_hop_length=512
        )
    
    @pytest.fixture
    def mock_database(self):
        """Mock Datenbank"""
        db = Mock()
        db.create_stem = AsyncMock()
        db.get_stem = AsyncMock()
        db.update_stem = AsyncMock()
        db.list_stems = AsyncMock()
        db.delete_stem = AsyncMock()
        return db
    
    @pytest.fixture
    def sample_audio_data(self):
        """Sample Audio-Daten für Tests"""
        # Generiere 2 Sekunden Sinus-Welle bei 440 Hz
        duration = 2.0
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440.0
        audio = np.sin(2 * np.pi * frequency * t)
        return audio, sample_rate
    
    @pytest.fixture
    def audio_service(self, audio_config, mock_database):
        """Audio-Service für Tests"""
        return AudioService(audio_config, mock_database)
    
    @pytest.mark.unit
    def test_audio_service_initialization(self, audio_config, mock_database):
        """Test: Audio-Service-Initialisierung"""
        service = AudioService(audio_config, mock_database)
        
        assert service.config == audio_config
        assert service.database == mock_database
        assert service.sample_rate == 48000
        assert service.bit_depth == 24
        assert isinstance(service.processor, AudioProcessor)
        assert isinstance(service.analyzer, AudioAnalyzer)
        assert isinstance(service.loader, AudioLoader)
        assert isinstance(service.exporter, AudioExporter)
    
    @pytest.mark.unit
    def test_audio_service_invalid_config(self, mock_database):
        """Test: Audio-Service mit ungültiger Konfiguration"""
        invalid_config = AudioConfig(
            sample_rate=0,  # Muss > 0 sein
            supported_formats=[]
        )
        
        with pytest.raises(ConfigurationError):
            AudioService(invalid_config, mock_database)
    
    @pytest.mark.unit
    async def test_load_audio_file(self, audio_service, sample_audio_data):
        """Test: Audio-Datei laden"""
        audio_data, sample_rate = sample_audio_data
        file_path = "/test/audio.wav"
        
        # Mock Audio-Loader
        with patch.object(audio_service.loader, 'load_file') as mock_load:
            mock_load.return_value = (audio_data, sample_rate)
            
            loaded_audio, loaded_sr = await audio_service.load_audio_file(file_path)
            
            assert np.array_equal(loaded_audio, audio_data)
            assert loaded_sr == sample_rate
            mock_load.assert_called_once_with(file_path)
    
    @pytest.mark.unit
    async def test_load_audio_file_not_found(self, audio_service):
        """Test: Audio-Datei laden - Datei nicht gefunden"""
        file_path = "/nonexistent/audio.wav"
        
        with patch.object(audio_service.loader, 'load_file') as mock_load:
            mock_load.side_effect = FileNotFoundError(f"File not found: {file_path}")
            
            with pytest.raises(FileNotFoundError):
                await audio_service.load_audio_file(file_path)
    
    @pytest.mark.unit
    async def test_analyze_audio(self, audio_service, sample_audio_data):
        """Test: Audio-Analyse"""
        audio_data, sample_rate = sample_audio_data
        
        # Mock Audio-Analyzer
        mock_features = AudioFeatures(
            duration=2.0,
            sample_rate=48000,
            channels=1,
            rms_energy=0.5,
            spectral_centroid=2000.0,
            spectral_rolloff=8000.0,
            zero_crossing_rate=0.1,
            mfcc=np.random.randn(13),
            tempo=120.0,
            key="A",
            loudness=-12.0,
            dynamic_range=18.0
        )
        
        with patch.object(audio_service.analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = mock_features
            
            features = await audio_service.analyze_audio(audio_data, sample_rate)
            
            assert isinstance(features, AudioFeatures)
            assert features.duration == 2.0
            assert features.sample_rate == 48000
            assert features.tempo == 120.0
            mock_analyze.assert_called_once_with(audio_data, sample_rate)
    
    @pytest.mark.unit
    async def test_process_audio(self, audio_service, sample_audio_data):
        """Test: Audio-Verarbeitung"""
        audio_data, sample_rate = sample_audio_data
        
        processing_options = {
            "normalize": True,
            "trim_silence": True,
            "apply_fade": True,
            "fade_duration": 0.1
        }
        
        # Mock Audio-Processor
        processed_audio = audio_data * 0.8  # Simuliere Verarbeitung
        
        with patch.object(audio_service.processor, 'process') as mock_process:
            mock_process.return_value = processed_audio
            
            result = await audio_service.process_audio(
                audio_data,
                sample_rate,
                processing_options
            )
            
            assert isinstance(result, np.ndarray)
            assert result.shape == audio_data.shape
            mock_process.assert_called_once_with(
                audio_data,
                sample_rate,
                processing_options
            )
    
    @pytest.mark.unit
    async def test_create_stem_from_file(self, audio_service, sample_audio_data):
        """Test: Stem aus Datei erstellen"""
        audio_data, sample_rate = sample_audio_data
        file_path = "/test/audio.wav"
        
        stem_data = StemCreate(
            name="Test Stem",
            file_path=file_path,
            tags=["test", "audio"],
            metadata={"artist": "Test Artist"}
        )
        
        # Mock Audio-Laden
        with patch.object(audio_service, 'load_audio_file') as mock_load:
            mock_load.return_value = (audio_data, sample_rate)
            
            # Mock Audio-Analyse
            mock_features = AudioFeatures(
                duration=2.0,
                sample_rate=48000,
                channels=1,
                rms_energy=0.5,
                tempo=120.0,
                key="A"
            )
            
            with patch.object(audio_service, 'analyze_audio') as mock_analyze:
                mock_analyze.return_value = mock_features
                
                # Mock Datenbank-Erstellung
                mock_stem = StemModel(
                    id="stem_123",
                    name="Test Stem",
                    file_path=file_path,
                    duration=2.0,
                    sample_rate=48000,
                    features=mock_features.dict()
                )
                audio_service.database.create_stem.return_value = mock_stem
                
                stem = await audio_service.create_stem_from_file(stem_data)
                
                assert isinstance(stem, StemResponse)
                assert stem.name == "Test Stem"
                assert stem.duration == 2.0
                assert stem.sample_rate == 48000
                
                audio_service.database.create_stem.assert_called_once()
    
    @pytest.mark.unit
    async def test_export_audio(self, audio_service, sample_audio_data):
        """Test: Audio exportieren"""
        audio_data, sample_rate = sample_audio_data
        output_path = "/test/output.wav"
        export_format = "wav"
        
        # Mock Audio-Exporter
        with patch.object(audio_service.exporter, 'export') as mock_export:
            mock_export.return_value = output_path
            
            result_path = await audio_service.export_audio(
                audio_data,
                sample_rate,
                output_path,
                export_format
            )
            
            assert result_path == output_path
            mock_export.assert_called_once_with(
                audio_data,
                sample_rate,
                output_path,
                export_format
            )
    
    @pytest.mark.unit
    async def test_validate_audio_file(self, audio_service):
        """Test: Audio-Datei validieren"""
        file_path = "/test/audio.wav"
        
        # Mock Audio-Validator
        validation_result = {
            "is_valid": True,
            "format": "wav",
            "duration": 2.0,
            "sample_rate": 48000,
            "channels": 2,
            "file_size": 384000,
            "errors": []
        }
        
        with patch.object(audio_service, 'validator') as mock_validator:
            mock_validator.validate_file.return_value = validation_result
            
            result = await audio_service.validate_audio_file(file_path)
            
            assert result["is_valid"] == True
            assert result["format"] == "wav"
            assert result["duration"] == 2.0
    
    @pytest.mark.unit
    async def test_get_audio_metadata(self, audio_service):
        """Test: Audio-Metadaten abrufen"""
        file_path = "/test/audio.wav"
        
        mock_metadata = AudioMetadata(
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            genre="Electronic",
            year=2024,
            duration=180.0,
            bitrate=1411,
            sample_rate=44100,
            channels=2,
            format="wav"
        )
        
        with patch.object(audio_service.loader, 'get_metadata') as mock_get_meta:
            mock_get_meta.return_value = mock_metadata
            
            metadata = await audio_service.get_audio_metadata(file_path)
            
            assert isinstance(metadata, AudioMetadata)
            assert metadata.title == "Test Song"
            assert metadata.artist == "Test Artist"
            assert metadata.duration == 180.0
    
    @pytest.mark.unit
    async def test_segment_audio(self, audio_service, sample_audio_data):
        """Test: Audio segmentieren"""
        audio_data, sample_rate = sample_audio_data
        segment_length = 1.0  # 1 Sekunde pro Segment
        
        # Mock Audio-Segmentierung
        mock_segments = [
            AudioSegment(
                start_time=0.0,
                end_time=1.0,
                audio_data=audio_data[:48000],
                features=None
            ),
            AudioSegment(
                start_time=1.0,
                end_time=2.0,
                audio_data=audio_data[48000:],
                features=None
            )
        ]
        
        with patch.object(audio_service.processor, 'segment_audio') as mock_segment:
            mock_segment.return_value = mock_segments
            
            segments = await audio_service.segment_audio(
                audio_data,
                sample_rate,
                segment_length
            )
            
            assert len(segments) == 2
            assert all(isinstance(seg, AudioSegment) for seg in segments)
            assert segments[0].start_time == 0.0
            assert segments[1].start_time == 1.0


class TestAudioProcessor:
    """Tests für Audio-Processor"""
    
    @pytest.fixture
    def audio_config(self):
        """Audio-Konfiguration für Tests"""
        return AudioConfig(
            sample_rate=48000,
            enable_normalization=True,
            enable_noise_reduction=False
        )
    
    @pytest.fixture
    def audio_processor(self, audio_config):
        """Audio-Processor für Tests"""
        return AudioProcessor(audio_config)
    
    @pytest.fixture
    def sample_audio(self):
        """Sample Audio für Tests"""
        duration = 2.0
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Sinus-Welle mit etwas Rauschen
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        return audio, sample_rate
    
    @pytest.mark.unit
    def test_audio_processor_initialization(self, audio_config):
        """Test: Audio-Processor-Initialisierung"""
        processor = AudioProcessor(audio_config)
        
        assert processor.config == audio_config
        assert processor.sample_rate == 48000
        assert processor.enable_normalization == True
    
    @pytest.mark.unit
    def test_normalize_audio(self, audio_processor, sample_audio):
        """Test: Audio-Normalisierung"""
        audio_data, sample_rate = sample_audio
        
        normalized = audio_processor.normalize(audio_data)
        
        # Normalisierte Audio sollte Peak bei 1.0 haben
        assert np.max(np.abs(normalized)) <= 1.0
        assert np.max(np.abs(normalized)) > 0.9  # Sollte nahe 1.0 sein
    
    @pytest.mark.unit
    def test_trim_silence(self, audio_processor):
        """Test: Stille trimmen"""
        # Audio mit Stille am Anfang und Ende
        silence_duration = 0.5  # 0.5 Sekunden Stille
        sample_rate = 48000
        silence_samples = int(silence_duration * sample_rate)
        
        # Signal in der Mitte
        signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))
        
        # Stille hinzufügen
        audio_with_silence = np.concatenate([
            np.zeros(silence_samples),  # Stille am Anfang
            signal,                     # Signal
            np.zeros(silence_samples)   # Stille am Ende
        ])
        
        trimmed = audio_processor.trim_silence(audio_with_silence, threshold=0.01)
        
        # Getrimmte Audio sollte kürzer sein
        assert len(trimmed) < len(audio_with_silence)
        assert len(trimmed) <= len(signal) * 1.1  # Etwas Toleranz
    
    @pytest.mark.unit
    def test_apply_fade(self, audio_processor, sample_audio):
        """Test: Fade-In/Out anwenden"""
        audio_data, sample_rate = sample_audio
        fade_duration = 0.1  # 0.1 Sekunden Fade
        
        faded = audio_processor.apply_fade(
            audio_data,
            sample_rate,
            fade_in=fade_duration,
            fade_out=fade_duration
        )
        
        fade_samples = int(fade_duration * sample_rate)
        
        # Fade-In: Sollte bei 0 starten
        assert np.abs(faded[0]) < 0.01
        
        # Fade-Out: Sollte bei 0 enden
        assert np.abs(faded[-1]) < 0.01
        
        # Mittlerer Teil sollte unverändert sein
        middle_start = fade_samples
        middle_end = len(faded) - fade_samples
        np.testing.assert_array_almost_equal(
            faded[middle_start:middle_end],
            audio_data[middle_start:middle_end],
            decimal=3
        )
    
    @pytest.mark.unit
    def test_resample_audio(self, audio_processor, sample_audio):
        """Test: Audio-Resampling"""
        audio_data, original_sr = sample_audio
        target_sr = 44100
        
        resampled = audio_processor.resample(audio_data, original_sr, target_sr)
        
        # Länge sollte entsprechend angepasst sein
        expected_length = int(len(audio_data) * target_sr / original_sr)
        assert abs(len(resampled) - expected_length) <= 1
    
    @pytest.mark.unit
    def test_apply_eq(self, audio_processor, sample_audio):
        """Test: Equalizer anwenden"""
        audio_data, sample_rate = sample_audio
        
        eq_settings = {
            "low_gain": 1.2,    # +20% bei niedrigen Frequenzen
            "mid_gain": 1.0,    # Keine Änderung bei mittleren Frequenzen
            "high_gain": 0.8    # -20% bei hohen Frequenzen
        }
        
        eq_audio = audio_processor.apply_eq(audio_data, sample_rate, eq_settings)
        
        assert len(eq_audio) == len(audio_data)
        assert not np.array_equal(eq_audio, audio_data)  # Sollte verändert sein
    
    @pytest.mark.unit
    def test_apply_compression(self, audio_processor, sample_audio):
        """Test: Kompression anwenden"""
        audio_data, sample_rate = sample_audio
        
        compression_settings = {
            "threshold": -12.0,  # dB
            "ratio": 4.0,       # 4:1 Kompression
            "attack": 0.003,    # 3ms Attack
            "release": 0.1      # 100ms Release
        }
        
        compressed = audio_processor.apply_compression(
            audio_data,
            sample_rate,
            compression_settings
        )
        
        assert len(compressed) == len(audio_data)
        # Komprimierte Audio sollte geringere Dynamik haben
        original_dynamic_range = np.max(audio_data) - np.min(audio_data)
        compressed_dynamic_range = np.max(compressed) - np.min(compressed)
        assert compressed_dynamic_range <= original_dynamic_range
    
    @pytest.mark.unit
    def test_segment_audio(self, audio_processor, sample_audio):
        """Test: Audio segmentieren"""
        audio_data, sample_rate = sample_audio
        segment_length = 0.5  # 0.5 Sekunden pro Segment
        
        segments = audio_processor.segment_audio(
            audio_data,
            sample_rate,
            segment_length
        )
        
        expected_segments = int(np.ceil(len(audio_data) / (segment_length * sample_rate)))
        assert len(segments) == expected_segments
        
        for i, segment in enumerate(segments):
            assert isinstance(segment, AudioSegment)
            assert segment.start_time == i * segment_length
            assert len(segment.audio_data) <= int(segment_length * sample_rate)


class TestAudioAnalyzer:
    """Tests für Audio-Analyzer"""
    
    @pytest.fixture
    def audio_config(self):
        """Audio-Konfiguration für Tests"""
        return AudioConfig(
            sample_rate=48000,
            analysis_window_size=2048,
            analysis_hop_length=512
        )
    
    @pytest.fixture
    def audio_analyzer(self, audio_config):
        """Audio-Analyzer für Tests"""
        return AudioAnalyzer(audio_config)
    
    @pytest.fixture
    def test_audio(self):
        """Test-Audio für Analyse"""
        duration = 3.0
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Komplexeres Signal: Grundton + Harmonische
        fundamental = 440.0  # A4
        audio = (
            0.5 * np.sin(2 * np.pi * fundamental * t) +
            0.3 * np.sin(2 * np.pi * fundamental * 2 * t) +
            0.2 * np.sin(2 * np.pi * fundamental * 3 * t)
        )
        
        return audio, sample_rate
    
    @pytest.mark.unit
    def test_audio_analyzer_initialization(self, audio_config):
        """Test: Audio-Analyzer-Initialisierung"""
        analyzer = AudioAnalyzer(audio_config)
        
        assert analyzer.config == audio_config
        assert analyzer.window_size == 2048
        assert analyzer.hop_length == 512
        assert isinstance(analyzer.spectral_analyzer, SpectralAnalyzer)
        assert isinstance(analyzer.temporal_analyzer, TemporalAnalyzer)
        assert isinstance(analyzer.harmonic_analyzer, HarmonicAnalyzer)
    
    @pytest.mark.unit
    def test_analyze_basic_features(self, audio_analyzer, test_audio):
        """Test: Grundlegende Audio-Feature-Analyse"""
        audio_data, sample_rate = test_audio
        
        features = audio_analyzer.analyze(audio_data, sample_rate)
        
        assert isinstance(features, AudioFeatures)
        assert features.duration == pytest.approx(3.0, rel=0.1)
        assert features.sample_rate == sample_rate
        assert features.channels == 1
        assert features.rms_energy > 0
        assert features.spectral_centroid > 0
        assert features.zero_crossing_rate >= 0
    
    @pytest.mark.unit
    def test_spectral_analysis(self, audio_analyzer, test_audio):
        """Test: Spektrale Analyse"""
        audio_data, sample_rate = test_audio
        
        spectral_features = audio_analyzer.spectral_analyzer.analyze(
            audio_data,
            sample_rate
        )
        
        assert "spectral_centroid" in spectral_features
        assert "spectral_rolloff" in spectral_features
        assert "spectral_bandwidth" in spectral_features
        assert "mfcc" in spectral_features
        
        # MFCC sollte 13 Koeffizienten haben
        assert len(spectral_features["mfcc"]) == 13
        
        # Spektrale Features sollten positive Werte haben
        assert spectral_features["spectral_centroid"] > 0
        assert spectral_features["spectral_rolloff"] > 0
    
    @pytest.mark.unit
    def test_temporal_analysis(self, audio_analyzer, test_audio):
        """Test: Temporale Analyse"""
        audio_data, sample_rate = test_audio
        
        temporal_features = audio_analyzer.temporal_analyzer.analyze(
            audio_data,
            sample_rate
        )
        
        assert "tempo" in temporal_features
        assert "beat_times" in temporal_features
        assert "onset_times" in temporal_features
        assert "rhythm_pattern" in temporal_features
        
        # Tempo sollte realistisch sein
        assert 60 <= temporal_features["tempo"] <= 200
        
        # Beat-Times sollten aufsteigend sortiert sein
        beat_times = temporal_features["beat_times"]
        assert all(beat_times[i] <= beat_times[i+1] for i in range(len(beat_times)-1))
    
    @pytest.mark.unit
    def test_harmonic_analysis(self, audio_analyzer, test_audio):
        """Test: Harmonische Analyse"""
        audio_data, sample_rate = test_audio
        
        harmonic_features = audio_analyzer.harmonic_analyzer.analyze(
            audio_data,
            sample_rate
        )
        
        assert "fundamental_frequency" in harmonic_features
        assert "harmonics" in harmonic_features
        assert "harmonic_ratio" in harmonic_features
        assert "key" in harmonic_features
        assert "chroma" in harmonic_features
        
        # Grundfrequenz sollte nahe 440 Hz sein
        f0 = harmonic_features["fundamental_frequency"]
        assert 430 <= f0 <= 450
        
        # Chroma sollte 12 Werte haben (für 12 Halbtöne)
        assert len(harmonic_features["chroma"]) == 12
    
    @pytest.mark.unit
    def test_loudness_analysis(self, audio_analyzer, test_audio):
        """Test: Lautstärke-Analyse"""
        audio_data, sample_rate = test_audio
        
        loudness_features = audio_analyzer.analyze_loudness(
            audio_data,
            sample_rate
        )
        
        assert "lufs" in loudness_features
        assert "peak_db" in loudness_features
        assert "rms_db" in loudness_features
        assert "dynamic_range" in loudness_features
        
        # LUFS sollte negativer Wert sein
        assert loudness_features["lufs"] < 0
        
        # Peak sollte <= 0 dB sein
        assert loudness_features["peak_db"] <= 0
        
        # Dynamic Range sollte positiv sein
        assert loudness_features["dynamic_range"] > 0
    
    @pytest.mark.unit
    def test_analyze_stereo_audio(self, audio_analyzer):
        """Test: Stereo-Audio-Analyse"""
        duration = 2.0
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Stereo-Signal: Links 440 Hz, Rechts 880 Hz
        left_channel = 0.5 * np.sin(2 * np.pi * 440 * t)
        right_channel = 0.5 * np.sin(2 * np.pi * 880 * t)
        stereo_audio = np.column_stack([left_channel, right_channel])
        
        features = audio_analyzer.analyze(stereo_audio, sample_rate)
        
        assert features.channels == 2
        assert features.duration == pytest.approx(2.0, rel=0.1)
        
        # Stereo-spezifische Features
        assert hasattr(features, 'stereo_width')
        assert hasattr(features, 'channel_correlation')
    
    @pytest.mark.unit
    def test_analyze_silence(self, audio_analyzer):
        """Test: Stille-Analyse"""
        duration = 1.0
        sample_rate = 48000
        silence = np.zeros(int(sample_rate * duration))
        
        features = audio_analyzer.analyze(silence, sample_rate)
        
        assert features.rms_energy == pytest.approx(0.0, abs=1e-6)
        assert features.zero_crossing_rate == pytest.approx(0.0, abs=1e-6)
        assert features.spectral_centroid == pytest.approx(0.0, abs=100)


class TestAudioServiceIntegration:
    """Integrationstests für Audio-Service"""
    
    @pytest.mark.integration
    async def test_full_audio_processing_workflow(self):
        """Test: Vollständiger Audio-Verarbeitungs-Workflow"""
        config = AudioConfig(
            sample_rate=48000,
            enable_normalization=True,
            supported_formats=["wav", "mp3"]
        )
        
        mock_db = Mock()
        mock_db.create_stem = AsyncMock()
        
        service = AudioService(config, mock_db)
        
        # 1. Audio-Daten generieren
        duration = 3.0
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # 2. Audio verarbeiten
        processing_options = {
            "normalize": True,
            "trim_silence": True,
            "apply_fade": True
        }
        
        with patch.object(service.processor, 'process') as mock_process:
            mock_process.return_value = audio_data * 0.9
            
            processed = await service.process_audio(
                audio_data,
                sample_rate,
                processing_options
            )
            
            assert isinstance(processed, np.ndarray)
        
        # 3. Audio analysieren
        with patch.object(service.analyzer, 'analyze') as mock_analyze:
            mock_features = AudioFeatures(
                duration=3.0,
                sample_rate=48000,
                channels=1,
                rms_energy=0.35,
                tempo=120.0,
                key="A"
            )
            mock_analyze.return_value = mock_features
            
            features = await service.analyze_audio(processed, sample_rate)
            assert features.duration == 3.0
        
        # 4. Audio exportieren
        with patch.object(service.exporter, 'export') as mock_export:
            mock_export.return_value = "/output/processed.wav"
            
            output_path = await service.export_audio(
                processed,
                sample_rate,
                "/output/processed.wav",
                "wav"
            )
            
            assert output_path == "/output/processed.wav"
    
    @pytest.mark.performance
    async def test_audio_service_performance(self):
        """Test: Audio-Service-Performance"""
        import time
        
        config = AudioConfig(sample_rate=48000)
        mock_db = Mock()
        service = AudioService(config, mock_db)
        
        # Große Audio-Datei simulieren (10 Sekunden)
        duration = 10.0
        sample_rate = 48000
        large_audio = np.random.randn(int(sample_rate * duration))
        
        # Performance-Test: Audio-Analyse
        with patch.object(service.analyzer, 'analyze') as mock_analyze:
            mock_features = AudioFeatures(
                duration=duration,
                sample_rate=sample_rate,
                channels=1,
                rms_energy=0.5,
                tempo=120.0
            )
            mock_analyze.return_value = mock_features
            
            start_time = time.time()
            features = await service.analyze_audio(large_audio, sample_rate)
            analysis_time = time.time() - start_time
            
            # Analyse sollte schnell sein (< 1 Sekunde für 10s Audio)
            assert analysis_time < 1.0
        
        # Performance-Test: Audio-Verarbeitung
        with patch.object(service.processor, 'process') as mock_process:
            mock_process.return_value = large_audio * 0.8
            
            start_time = time.time()
            processed = await service.process_audio(
                large_audio,
                sample_rate,
                {"normalize": True}
            )
            processing_time = time.time() - start_time
            
            # Verarbeitung sollte schnell sein
            assert processing_time < 2.0