"""Render-Schemata für die Neuromorphe Traum-Engine."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from enum import Enum


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


class RenderStatus(str, Enum):
    """Render-Status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RenderOutput(BaseModel):
    """Schema für Render-Output."""
    render_id: str
    arrangement_id: str
    output_path: str
    format: RenderFormat
    quality: RenderQuality
    status: RenderStatus
    duration: Optional[float] = None
    file_size: Optional[int] = None
    sample_rate: Optional[int] = 44100
    bit_depth: Optional[int] = 16
    channels: Optional[int] = 2
    metadata: Optional[Dict[str, Any]] = None
    progress: Optional[float] = 0.0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = datetime.now()
    
    model_config = ConfigDict(from_attributes=True)


class RenderRequest(BaseModel):
    """Schema für Render-Anfragen."""
    arrangement_id: str
    format: RenderFormat = RenderFormat.WAV
    quality: RenderQuality = RenderQuality.HIGH
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2
    normalize: bool = True
    fade_in: Optional[float] = None
    fade_out: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class RenderProgress(BaseModel):
    """Schema für Render-Fortschritt."""
    render_id: str
    status: RenderStatus
    progress: float
    current_step: Optional[str] = None
    estimated_time_remaining: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)