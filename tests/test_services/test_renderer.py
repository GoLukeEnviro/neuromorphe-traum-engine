"""Tests für RendererService"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
import numpy as np
from pathlib import Path
import tempfile
import os

from services.renderer import RendererService
from core.config import Settings


class TestRendererService:
    """Test-Suite für RendererService"""
    
    @pytest.fixture
    def renderer(self, test_settings: Settings) -> RendererService:
        """RendererService-Instanz für Tests"""
        return RendererService(test_settings)
    
    @pytest.mark.unit
    def test_initialization(self, test_settings: Settings):
        """Test: RendererService-Initialisierung"""
        renderer = RendererService(test_settings)
        
        assert renderer.settings == test_settings
        assert renderer.sample_rate == test_settings.audio.sample_rate
        assert renderer.bit_depth == test_settings.audio.bit_depth
        assert renderer.output_format == test_settings.audio.output_format
    
    @pytest.mark.unit
    async def test_render_arrangement_basic(self, renderer: RendererService, temp_dir: Path):
        """Test: Basis-Arrangement-Rendering"""
        arrangement = {
            "arrangement_id": "test_123",
            "structure": {
                "sections": [
                    {
                        "name": "intro",
                        "start": 0,
                        "duration": 32,
                        "stems": [1, 2]
                    },
                    {
                        "name": "main",
                        "start": 32,
                        "duration": 64,
                        "stems": [1, 2, 3]
                    }
                ],
                "total_duration": 96
            },
            "stems": [1, 2, 3],
            "metadata": {"genre": "techno", "tempo": 128}
        }
        
        stems_data = {
            1: {"file_path": str(temp_dir / "kick.wav"), "type": "kick"},
            2: {"file_path": str(temp_dir / "bass.wav"), "type": "bass"},
            3: {"file_path": str(temp_dir / "synth.wav"), "type": "synth"}
        }
        
        output_path = temp_dir / "output.wav"
        
        with patch.object(renderer, '_load_stem_audio') as mock_load, \
             patch.object(renderer, '_apply_effects') as mock_effects, \
             patch.object(renderer, '_mix_stems') as mock_mix, \
             patch.object(renderer, '_export_audio') as mock_export:
            
            # Mock Audio-Daten (44100 Hz, 2 Sekunden)
            mock_audio = np.random.rand(2, 88200).astype(np.float32)
            mock_load.return_value = mock_audio
            mock_effects.return_value = mock_audio
            mock_mix.return_value = mock_audio
            
            result = await renderer.render_arrangement(
                arrangement=arrangement,
                stems_data=stems_data,
                output_path=str(output_path)
            )
            
            assert "output_path" in result
            assert "duration" in result
            assert "metadata" in result
            assert result["output_path"] == str(output_path)
            
            # Verify method calls
            assert mock_load.call_count >= 3  # Für jeden Stem
            mock_export.assert_called_once()
    
    @pytest.mark.unit
    async def test_load_stem_audio(self, renderer: RendererService, temp_dir: Path, sample_audio_data):
        """Test: Laden von Stem-Audio-Daten"""
        # Test-Audio-Datei erstellen
        audio_file = temp_dir / "test_stem.wav"
        
        with patch('librosa.load') as mock_load:
            # Mock librosa.load
            mock_load.return_value = (sample_audio_data, renderer.sample_rate)
            
            audio_data = await renderer._load_stem_audio(str(audio_file))
            
            assert audio_data is not None
            assert audio_data.shape[0] == 2  # Stereo
            assert audio_data.dtype == np.float32
            
            mock_load.assert_called_once_with(
                str(audio_file),
                sr=renderer.sample_rate,
                mono=False
            )
    
    @pytest.mark.unit
    async def test_load_stem_audio_mono_to_stereo(self, renderer: RendererService, temp_dir: Path):
        """Test: Konvertierung von Mono zu Stereo"""
        audio_file = temp_dir / "mono_stem.wav"
        
        with patch('librosa.load') as mock_load:
            # Mock Mono-Audio
            mono_audio = np.random.rand(44100).astype(np.float32)
            mock_load.return_value = (mono_audio, renderer.sample_rate)
            
            audio_data = await renderer._load_stem_audio(str(audio_file))
            
            assert audio_data.shape[0] == 2  # Sollte zu Stereo konvertiert werden
            assert np.array_equal(audio_data[0], audio_data[1])  # Beide Kanäle identisch
    
    @pytest.mark.unit
    async def test_apply_effects_volume(self, renderer: RendererService, sample_audio_data):
        """Test: Lautstärke-Effekt anwenden"""
        effects = {
            "volume": 0.5,
            "pan": 0.0
        }
        
        processed_audio = await renderer._apply_effects(sample_audio_data, effects)
        
        # Audio sollte leiser sein
        assert np.max(np.abs(processed_audio)) < np.max(np.abs(sample_audio_data))
        assert processed_audio.shape == sample_audio_data.shape
    
    @pytest.mark.unit
    async def test_apply_effects_pan(self, renderer: RendererService, sample_audio_data):
        """Test: Pan-Effekt anwenden"""
        effects = {
            "volume": 1.0,
            "pan": -1.0  # Komplett links
        }
        
        processed_audio = await renderer._apply_effects(sample_audio_data, effects)
        
        # Rechter Kanal sollte stumm sein
        assert np.max(np.abs(processed_audio[1])) < 0.01
        assert processed_audio.shape == sample_audio_data.shape
    
    @pytest.mark.unit
    async def test_apply_effects_fade_in(self, renderer: RendererService, sample_audio_data):
        """Test: Fade-In-Effekt"""
        effects = {
            "fade_in": 1.0  # 1 Sekunde Fade-In
        }
        
        processed_audio = await renderer._apply_effects(sample_audio_data, effects)
        
        # Anfang sollte leise sein
        fade_samples = int(1.0 * renderer.sample_rate)
        assert np.max(np.abs(processed_audio[:, :fade_samples//4])) < 0.1
    
    @pytest.mark.unit
    async def test_apply_effects_fade_out(self, renderer: RendererService, sample_audio_data):
        """Test: Fade-Out-Effekt"""
        effects = {
            "fade_out": 1.0  # 1 Sekunde Fade-Out
        }
        
        processed_audio = await renderer._apply_effects(sample_audio_data, effects)
        
        # Ende sollte leise sein
        fade_samples = int(1.0 * renderer.sample_rate)
        assert np.max(np.abs(processed_audio[:, -fade_samples//4:])) < 0.1
    
    @pytest.mark.unit
    async def test_apply_effects_eq(self, renderer: RendererService, sample_audio_data):
        """Test: EQ-Effekt"""
        effects = {
            "eq": {
                "low": 1.2,    # Bass verstärken
                "mid": 0.8,    # Mitten reduzieren
                "high": 1.1    # Höhen leicht verstärken
            }
        }
        
        with patch('scipy.signal.butter') as mock_butter, \
             patch('scipy.signal.filtfilt') as mock_filtfilt:
            
            mock_butter.return_value = ([1, 0], [1, 0])  # Dummy-Filter
            mock_filtfilt.return_value = sample_audio_data[0]  # Dummy-Output
            
            processed_audio = await renderer._apply_effects(sample_audio_data, effects)
            
            assert processed_audio.shape == sample_audio_data.shape
            # EQ sollte angewendet worden sein
            assert mock_butter.call_count >= 3  # Low, Mid, High
    
    @pytest.mark.unit
    async def test_mix_stems_basic(self, renderer: RendererService, sample_audio_data):
        """Test: Basis-Stem-Mixing"""
        stems_audio = {
            "kick": sample_audio_data * 0.8,
            "bass": sample_audio_data * 0.6,
            "synth": sample_audio_data * 0.4
        }
        
        mixed_audio = await renderer._mix_stems(stems_audio)
        
        assert mixed_audio.shape == sample_audio_data.shape
        # Mixed Audio sollte lauter sein als einzelne Stems
        assert np.max(np.abs(mixed_audio)) > np.max(np.abs(sample_audio_data * 0.8))
    
    @pytest.mark.unit
    async def test_mix_stems_with_compression(self, renderer: RendererService, sample_audio_data):
        """Test: Mixing mit Kompression"""
        stems_audio = {
            "kick": sample_audio_data,
            "bass": sample_audio_data,
            "synth": sample_audio_data
        }
        
        with patch.object(renderer, '_apply_compression') as mock_compression:
            mock_compression.return_value = sample_audio_data * 0.8
            
            mixed_audio = await renderer._mix_stems(
                stems_audio,
                apply_compression=True
            )
            
            mock_compression.assert_called_once()
            assert mixed_audio.shape == sample_audio_data.shape
    
    @pytest.mark.unit
    async def test_apply_compression(self, renderer: RendererService, sample_audio_data):
        """Test: Audio-Kompression"""
        # Sehr lautes Signal erstellen
        loud_audio = sample_audio_data * 2.0
        
        compressed_audio = await renderer._apply_compression(
            loud_audio,
            threshold=-12.0,
            ratio=4.0,
            attack=0.003,
            release=0.1
        )
        
        # Komprimiertes Audio sollte leiser sein
        assert np.max(np.abs(compressed_audio)) < np.max(np.abs(loud_audio))
        assert compressed_audio.shape == loud_audio.shape
    
    @pytest.mark.unit
    async def test_apply_limiter(self, renderer: RendererService, sample_audio_data):
        """Test: Audio-Limiter"""
        # Sehr lautes Signal erstellen
        loud_audio = sample_audio_data * 3.0
        
        limited_audio = await renderer._apply_limiter(
            loud_audio,
            threshold=-1.0
        )
        
        # Limitiertes Audio sollte unter Threshold bleiben
        assert np.max(np.abs(limited_audio)) <= 1.0
        assert limited_audio.shape == loud_audio.shape
    
    @pytest.mark.unit
    async def test_normalize_audio(self, renderer: RendererService, sample_audio_data):
        """Test: Audio-Normalisierung"""
        # Leises Signal erstellen
        quiet_audio = sample_audio_data * 0.1
        
        normalized_audio = await renderer._normalize_audio(
            quiet_audio,
            target_lufs=-14.0
        )
        
        # Normalisiertes Audio sollte lauter sein
        assert np.max(np.abs(normalized_audio)) > np.max(np.abs(quiet_audio))
        assert normalized_audio.shape == quiet_audio.shape
    
    @pytest.mark.unit
    async def test_export_audio_wav(self, renderer: RendererService, sample_audio_data, temp_dir: Path):
        """Test: WAV-Export"""
        output_path = temp_dir / "test_output.wav"
        
        with patch('soundfile.write') as mock_write:
            await renderer._export_audio(
                sample_audio_data,
                str(output_path),
                format="wav"
            )
            
            mock_write.assert_called_once_with(
                str(output_path),
                sample_audio_data.T,  # Transponiert für soundfile
                renderer.sample_rate,
                subtype='PCM_24'
            )
    
    @pytest.mark.unit
    async def test_export_audio_mp3(self, renderer: RendererService, sample_audio_data, temp_dir: Path):
        """Test: MP3-Export"""
        output_path = temp_dir / "test_output.mp3"
        
        with patch('pydub.AudioSegment.from_raw') as mock_from_raw, \
             patch('pydub.AudioSegment.export') as mock_export:
            
            mock_segment = MagicMock()
            mock_from_raw.return_value = mock_segment
            
            await renderer._export_audio(
                sample_audio_data,
                str(output_path),
                format="mp3"
            )
            
            mock_from_raw.assert_called_once()
            mock_segment.export.assert_called_once_with(
                str(output_path),
                format="mp3",
                bitrate="320k"
            )
    
    @pytest.mark.unit
    async def test_calculate_section_timing(self, renderer: RendererService):
        """Test: Berechnung von Sektion-Timing"""
        section = {
            "start": 32,
            "duration": 64
        }
        
        tempo = 128  # BPM
        
        timing = await renderer._calculate_section_timing(section, tempo)
        
        assert "start_samples" in timing
        assert "duration_samples" in timing
        assert "start_beats" in timing
        assert "duration_beats" in timing
        
        # 32 Sekunden bei 44100 Hz
        expected_start_samples = 32 * renderer.sample_rate
        assert timing["start_samples"] == expected_start_samples
    
    @pytest.mark.unit
    async def test_apply_crossfade(self, renderer: RendererService, sample_audio_data):
        """Test: Crossfade zwischen Audio-Segmenten"""
        audio1 = sample_audio_data
        audio2 = sample_audio_data * 0.5
        
        crossfaded = await renderer._apply_crossfade(
            audio1,
            audio2,
            fade_duration=1.0  # 1 Sekunde
        )
        
        assert crossfaded.shape[1] == audio1.shape[1] + audio2.shape[1] - int(1.0 * renderer.sample_rate)
        assert crossfaded.shape[0] == 2  # Stereo
    
    @pytest.mark.unit
    async def test_error_handling_missing_stem(self, renderer: RendererService, temp_dir: Path):
        """Test: Fehlerbehandlung bei fehlendem Stem"""
        non_existent_file = temp_dir / "missing.wav"
        
        with patch('librosa.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(FileNotFoundError):
                await renderer._load_stem_audio(str(non_existent_file))
    
    @pytest.mark.unit
    async def test_error_handling_invalid_audio_format(self, renderer: RendererService, temp_dir: Path):
        """Test: Fehlerbehandlung bei ungültigem Audio-Format"""
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("This is not audio")
        
        with patch('librosa.load') as mock_load:
            mock_load.side_effect = Exception("Invalid audio format")
            
            with pytest.raises(Exception):
                await renderer._load_stem_audio(str(invalid_file))
    
    @pytest.mark.unit
    async def test_memory_efficient_processing(self, renderer: RendererService, sample_audio_data):
        """Test: Speicher-effiziente Verarbeitung großer Audio-Dateien"""
        # Großes Audio-Array simulieren (10 Minuten)
        large_audio = np.random.rand(2, 44100 * 600).astype(np.float32)
        
        with patch.object(renderer, '_process_audio_chunks') as mock_chunks:
            mock_chunks.return_value = large_audio
            
            processed = await renderer._apply_effects(
                large_audio,
                {"volume": 0.8},
                chunk_size=44100  # 1 Sekunde Chunks
            )
            
            mock_chunks.assert_called_once()
            assert processed.shape == large_audio.shape
    
    @pytest.mark.performance
    async def test_rendering_performance(self, renderer: RendererService, temp_dir: Path):
        """Test: Rendering-Performance"""
        import time
        
        # Einfaches Arrangement für Performance-Test
        arrangement = {
            "arrangement_id": "perf_test",
            "structure": {
                "sections": [
                    {"name": "main", "start": 0, "duration": 60, "stems": [1, 2]}
                ],
                "total_duration": 60
            },
            "stems": [1, 2],
            "metadata": {"genre": "techno", "tempo": 128}
        }
        
        stems_data = {
            1: {"file_path": str(temp_dir / "kick.wav"), "type": "kick"},
            2: {"file_path": str(temp_dir / "bass.wav"), "type": "bass"}
        }
        
        output_path = temp_dir / "perf_output.wav"
        
        with patch.object(renderer, '_load_stem_audio') as mock_load, \
             patch.object(renderer, '_export_audio') as mock_export:
            
            # Mock Audio-Daten (1 Minute)
            mock_audio = np.random.rand(2, 44100 * 60).astype(np.float32)
            mock_load.return_value = mock_audio
            
            start_time = time.time()
            
            result = await renderer.render_arrangement(
                arrangement=arrangement,
                stems_data=stems_data,
                output_path=str(output_path)
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Rendering sollte in angemessener Zeit erfolgen
            assert processing_time < 10.0  # Unter 10 Sekunden für 1 Minute Audio
            assert "output_path" in result
    
    @pytest.mark.integration
    async def test_full_rendering_workflow(self, renderer: RendererService, temp_dir: Path, sample_audio_data):
        """Test: Vollständiger Rendering-Workflow"""
        # Komplexes Arrangement
        arrangement = {
            "arrangement_id": "full_test",
            "structure": {
                "sections": [
                    {
                        "name": "intro",
                        "start": 0,
                        "duration": 16,
                        "stems": [1],
                        "effects": {1: {"volume": 0.6, "fade_in": 2.0}}
                    },
                    {
                        "name": "buildup",
                        "start": 16,
                        "duration": 16,
                        "stems": [1, 2],
                        "effects": {
                            1: {"volume": 0.8},
                            2: {"volume": 0.4, "fade_in": 4.0}
                        }
                    },
                    {
                        "name": "drop",
                        "start": 32,
                        "duration": 32,
                        "stems": [1, 2, 3],
                        "effects": {
                            1: {"volume": 1.0},
                            2: {"volume": 0.8},
                            3: {"volume": 0.6, "pan": 0.3}
                        }
                    },
                    {
                        "name": "outro",
                        "start": 64,
                        "duration": 16,
                        "stems": [1],
                        "effects": {1: {"volume": 0.4, "fade_out": 8.0}}
                    }
                ],
                "total_duration": 80
            },
            "stems": [1, 2, 3],
            "metadata": {
                "genre": "techno",
                "tempo": 128,
                "key": "Am",
                "master_effects": {
                    "compression": {"threshold": -12, "ratio": 3.0},
                    "limiter": {"threshold": -1.0},
                    "normalize": {"target_lufs": -14.0}
                }
            }
        }
        
        stems_data = {
            1: {"file_path": str(temp_dir / "kick.wav"), "type": "kick"},
            2: {"file_path": str(temp_dir / "bass.wav"), "type": "bass"},
            3: {"file_path": str(temp_dir / "synth.wav"), "type": "synth"}
        }
        
        output_path = temp_dir / "full_output.wav"
        
        with patch.object(renderer, '_load_stem_audio') as mock_load, \
             patch.object(renderer, '_export_audio') as mock_export:
            
            mock_load.return_value = sample_audio_data
            
            result = await renderer.render_arrangement(
                arrangement=arrangement,
                stems_data=stems_data,
                output_path=str(output_path)
            )
            
            # Vollständiges Ergebnis validieren
            assert "output_path" in result
            assert "duration" in result
            assert "metadata" in result
            assert "sections_rendered" in result
            assert "effects_applied" in result
            
            assert result["output_path"] == str(output_path)
            assert result["duration"] == 80
            assert len(result["sections_rendered"]) == 4
            
            # Alle Stems sollten geladen worden sein
            assert mock_load.call_count >= 3
            
            # Export sollte erfolgt sein
            mock_export.assert_called_once()