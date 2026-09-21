"""Fail-closed-Verhalten des echten Embedding-Pfads.

Vertrag (TE-MODELS): Im realen Modus darf ein Modell-Lade- oder
Inferenzfehler NICHT als Erfolg durchgereicht werden, und ein degeneriertes
(Null-/nicht-endliches) Embedding darf niemals als Embedding gelten.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio.service import AudioProcessingService  # noqa: E402
from core.config import Settings  # noqa: E402
from exceptions import CLAPModelError  # noqa: E402
from schemas import AudioProcessingResponse, ProcessingStatus  # noqa: E402
from services.neuro_analyzer import NeuroAnalyzer  # noqa: E402


def _write_wav(path: Path, seconds: float = 1.0, sr: int = 48000) -> Path:
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 0.05, int(sr * seconds)).astype(np.float32)
    sf.write(str(path), signal, sr)
    return path


@pytest.fixture
def audio_service(tmp_path: Path) -> AudioProcessingService:
    return AudioProcessingService(audio_dir=str(tmp_path / "audio"))


@pytest.mark.unit
class TestAudioServiceFailsClosed:
    """Ein Modellfehler darf nie als COMPLETED ohne Embedding enden."""

    def test_model_load_error_raises_instead_of_silent_success(self, audio_service):
        wav = _write_wav(audio_service.audio_dir / "track.wav")

        def _boom():
            raise RuntimeError("model weights not present")

        audio_service._load_clap_model = _boom  # type: ignore[assignment]

        with pytest.raises(CLAPModelError):
            asyncio.run(audio_service._generate_clap_embedding(wav))

        assert not (audio_service.audio_dir / "track_embedding.npy").exists()
        assert wav.exists()

    def test_model_load_error_does_not_report_completed(self, audio_service):
        _write_wav(audio_service.audio_dir / "track.wav")

        def _boom():
            raise RuntimeError("model weights not present")

        audio_service._load_clap_model = _boom  # type: ignore[assignment]

        response = asyncio.run(audio_service.process_audio_file("track"))

        assert isinstance(response, AudioProcessingResponse)
        assert response.status is not ProcessingStatus.COMPLETED
        assert response.status is ProcessingStatus.FAILED
        assert "CLAP" in response.message
        assert not (audio_service.audio_dir / "track_embedding.npy").exists()

    def test_real_embedding_is_persisted_and_marked(self, audio_service):
        _write_wav(audio_service.audio_dir / "track.wav")
        vector = np.linspace(-0.5, 0.5, 512).astype(np.float32)

        class _FakeCLAP:
            def get_audio_embedding_from_filelist(self, x, use_tensor=False):
                return np.stack([vector] * len(x))

        audio_service._load_clap_model = lambda: _FakeCLAP()  # type: ignore[assignment]

        response = asyncio.run(audio_service.process_audio_file("track"))

        assert response.status is ProcessingStatus.COMPLETED
        stored = np.load(audio_service.audio_dir / "track_embedding.npy")
        assert stored.shape == (512,)
        assert float(np.linalg.norm(stored)) > 0.0

    def test_null_vector_from_model_is_rejected(self, audio_service):
        _write_wav(audio_service.audio_dir / "track.wav")

        class _NullCLAP:
            def get_audio_embedding_from_filelist(self, x, use_tensor=False):
                return np.zeros((len(x), 512), dtype=np.float32)

        audio_service._load_clap_model = lambda: _NullCLAP()  # type: ignore[assignment]

        response = asyncio.run(audio_service.process_audio_file("track"))

        assert response.status is ProcessingStatus.FAILED
        assert "CLAP" in response.message
        assert not (audio_service.audio_dir / "track_embedding.npy").exists()


@pytest.mark.unit
class TestNeuroAnalyzerFailsClosed:
    """Ein Null-/NaN-Embedding ist kein Embedding."""

    def test_zero_embedding_raises(self):
        analyzer = NeuroAnalyzer()
        analyzer.model.get_audio_embedding_from_data.return_value = np.zeros(
            512, dtype=np.float32
        )
        audio = np.random.default_rng(0).normal(0, 0.05, 48000).astype(np.float32)

        with pytest.raises(CLAPModelError):
            asyncio.run(analyzer.analyze_audio(audio, 48000))

    def test_non_finite_embedding_raises(self):
        analyzer = NeuroAnalyzer()
        bad = np.zeros(512, dtype=np.float32)
        bad[3] = np.nan
        analyzer.model.get_audio_embedding_from_data.return_value = bad
        audio = np.random.default_rng(1).normal(0, 0.05, 48000).astype(np.float32)

        with pytest.raises(CLAPModelError):
            asyncio.run(analyzer.analyze_audio(audio, 48000))

    def test_real_embedding_passes_and_is_normalised(self):
        analyzer = NeuroAnalyzer()
        vector = np.random.default_rng(2).normal(0, 1, 512).astype(np.float32)
        analyzer.model.get_audio_embedding_from_data.return_value = vector
        audio = np.random.default_rng(3).normal(0, 0.05, 48000).astype(np.float32)

        with patch.object(analyzer, "extract_audio_features", return_value={}):
            result = asyncio.run(analyzer.analyze_audio(audio, 48000))

        embedding = np.asarray(result["embeddings"], dtype=np.float64)
        assert embedding.shape == (512,)
        assert np.all(np.isfinite(embedding))
        assert float(np.linalg.norm(embedding)) > 0.0
        assert abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-6


@pytest.mark.unit
class TestFailClosedIsConfigurable:
    """Der reale Modus ist Default; ein expliziter lokaler Modus bleibt möglich."""

    def test_default_is_fail_closed(self):
        assert Settings(_env_file=None).EMBEDDING_FAIL_CLOSED is True

    def test_opt_out_keeps_degraded_behaviour_visible(self, audio_service):
        wav = _write_wav(audio_service.audio_dir / "track.wav")

        def _boom():
            raise RuntimeError("model weights not present")

        audio_service._load_clap_model = _boom  # type: ignore[assignment]
        audio_service.fail_closed = False

        response = asyncio.run(audio_service.process_audio_file("track"))

        assert response.status is ProcessingStatus.COMPLETED
        assert response.message.endswith("(CLAP embedding failed)")
        assert not (audio_service.audio_dir / "track_embedding.npy").exists()
        assert wav.exists()
