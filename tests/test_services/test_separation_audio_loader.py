"""Audio-Loader des SeparationService: soundfile statt torchaudio/torchcodec (#8).

Regression: ``torchaudio.load`` delegiert in torchaudio 2.11 auf ``torchcodec``
(kein Projekt-Dependency), wodurch jeder Separationslauf schon beim Einlesen mit
``ImportError: TorchCodec is required`` abbrach. Der Service dekodiert jetzt über
``soundfile``; diese Tests laufen bewusst in einer Umgebung *ohne* torchcodec und
würden mit dem alten Pfad fehlschlagen.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.separation_service import SeparationService  # noqa: E402


def _write_wav(path: Path, seconds: float = 0.5, sr: int = 44100, channels: int = 2) -> Path:
    rng = np.random.default_rng(7)
    data = (0.1 * rng.normal(size=(int(sr * seconds), channels))).astype(np.float32)
    sf.write(str(path), data, sr)
    return path


@pytest.mark.unit
def test_load_waveform_reads_stereo_wav_without_torchcodec(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "stereo.wav")
    waveform, sample_rate = SeparationService()._load_waveform(str(path))

    assert sample_rate == 44100
    assert waveform.shape == (2, int(44100 * 0.5))
    assert waveform.dtype == torch.float32
    assert float(waveform.abs().max()) > 0.0  # nicht still


@pytest.mark.unit
def test_load_waveform_duplicates_mono_to_stereo(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "mono.wav", channels=1)
    waveform, sample_rate = SeparationService()._load_waveform(str(path))

    assert sample_rate == 44100
    assert waveform.shape[0] == 2
    assert torch.equal(waveform[0], waveform[1])


@pytest.mark.unit
def test_load_waveform_truncates_more_than_two_channels(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "quad.wav", channels=4)
    waveform, _ = SeparationService()._load_waveform(str(path))

    assert waveform.shape[0] == 2


@pytest.mark.unit
def test_separation_module_does_not_import_torchcodec_dependent_torchaudio() -> None:
    """Struktureller Wächter: der Service soll torchaudio (und damit den
    torchcodec-Pfad) nicht mehr importieren."""
    import services.separation_service as mod

    assert not hasattr(mod, "torchaudio")
