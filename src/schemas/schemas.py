"""Legacy-Schema-Modul der Neuromorphen Traum-Engine.

Enthält Render- und Stem-Schemas, die von ``tests/test_schemas/``
spezifiziert werden. Die kanonischen Definitionen für Render-Typen liegen in
``schemas/render.py``; dieses Modul hält die von Tests und CLI erwarteten
Namen bereit.

Hinweis: ``RenderStatus`` erlaubt hier zusätzlich ``CANCELLED`` sowie die
numerischen ``RenderPriority``-Werte aus der Test-Spezifikation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RenderStatus(str, Enum):
    """Render-Status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RenderPriority(int, Enum):
    """Render-Priorität (numerisch, höher = wichtiger)."""
    LOW = 1
    MEDIUM = 5
    HIGH = 10


class RenderFormat(str, Enum):
    """Verfügbare Render-Formate."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class RenderQuality(str, Enum):
    """Verfügbare Render-Qualitäten."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"


# ---------------------------------------------------------------------------
# Render-Schemas
# ---------------------------------------------------------------------------


MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000


class RenderSettings(BaseModel):
    """Audio-Einstellungen für einen Render-Vorgang."""
    format: RenderFormat = RenderFormat.WAV
    quality: RenderQuality = RenderQuality.MEDIUM
    priority: RenderPriority = RenderPriority.MEDIUM
    sample_rate: int = Field(default=44100, ge=MIN_SAMPLE_RATE, le=MAX_SAMPLE_RATE)
    bit_depth: int = Field(default=16, ge=8, le=32)
    channels: int = Field(default=2, ge=1, le=8)
    normalize: bool = True

    model_config = ConfigDict(from_attributes=True)


class RenderJobBase(BaseModel):
    """Basis-Schema für Render-Aufträge."""
    arrangement_id: str
    format: RenderFormat = RenderFormat.WAV
    quality: RenderQuality = RenderQuality.HIGH
    status: RenderStatus = RenderStatus.PENDING
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    options: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class RenderJobCreate(RenderJobBase):
    """Schema zum Anlegen eines Render-Auftrags."""
    pass


class RenderJobUpdate(BaseModel):
    """Schema für Status-Updates eines Render-Auftrags."""
    status: Optional[RenderStatus] = None
    progress: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    result_path: Optional[str] = None
    error_message: Optional[str] = None


class RenderJobResponse(RenderJobBase):
    """Schema für die Antwort mit einem Render-Auftrag."""
    id: Optional[int] = None
    result_path: Optional[str] = None
    error_message: Optional[str] = None


class RenderProgress(BaseModel):
    """Fortschritt eines Render-Vorgangs."""
    job_id: Optional[str] = None
    render_id: Optional[str] = None
    status: Optional[RenderStatus] = None
    progress: float = Field(..., ge=0.0, le=100.0)
    current_step: Optional[str] = None
    message: Optional[str] = None
    estimated_time_remaining: Optional[float] = None


class EmptyResponse(BaseModel):
    """Leere Erfolgsantwort."""
    message: str = "Operation successful."


# ---------------------------------------------------------------------------
# Stem-Schemas
# ---------------------------------------------------------------------------


class StemSearch(BaseModel):
    """Suchfilter für Stems."""
    query: Optional[str] = None
    category: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    key: Optional[str] = None
    harmonic_complexity_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    harmonic_complexity_max: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rhythmic_complexity_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rhythmic_complexity_max: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=50, ge=1, le=500)

    @model_validator(mode="after")
    def _check_ranges(self) -> "StemSearch":
        if self.bpm_min is not None and self.bpm_max is not None:
            if self.bpm_min > self.bpm_max:
                raise ValueError("bpm_min must not exceed bpm_max")
        if self.harmonic_complexity_min is not None and self.harmonic_complexity_max is not None:
            if self.harmonic_complexity_min > self.harmonic_complexity_max:
                raise ValueError(
                    "harmonic_complexity_min must not exceed harmonic_complexity_max"
                )
        if self.rhythmic_complexity_min is not None and self.rhythmic_complexity_max is not None:
            if self.rhythmic_complexity_min > self.rhythmic_complexity_max:
                raise ValueError(
                    "rhythmic_complexity_min must not exceed rhythmic_complexity_max"
                )
        return self


class StemFeatures(BaseModel):
    """Audio-Features eines Stems."""
    mfcc: Optional[List[float]] = None
    spectral_centroid: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    rms_energy: Optional[float] = None
    tempo: Optional[float] = None
    chroma: Optional[List[float]] = None

    model_config = ConfigDict(from_attributes=True)


class StemAnalysis(BaseModel):
    """Ergebnis einer Stem-Analyse."""
    stem_id: Optional[int] = None
    features: Optional[StemFeatures] = None
    embeddings: Optional[List[float]] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(from_attributes=True)


class StemBatch(BaseModel):
    """Sammlung von Stem-IDs für Batch-Operationen."""
    stem_ids: List[int] = Field(..., min_length=1)
    total: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)
