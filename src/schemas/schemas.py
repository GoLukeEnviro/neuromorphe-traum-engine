from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime


class RenderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RenderPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class RenderFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class RenderFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class RenderQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"


class RenderPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class RenderSettings(BaseModel):
    format: RenderFormat = RenderFormat.WAV
    quality: RenderQuality = RenderQuality.MEDIUM
    priority: RenderPriority = RenderPriority.MEDIUM


class RenderJobUpdate(BaseModel):
    status: Optional[RenderStatus] = None
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    result_path: Optional[str] = None
    error_message: Optional[str] = None


class RenderJobBase(BaseModel):
    job_id: str
    stem_id: str
    status: RenderStatus
    progress: float
    settings: RenderSettings
    created_at: datetime
    updated_at: datetime
    result_path: Optional[str] = None
    error_message: Optional[str] = None


class RenderJobResponse(RenderJobBase):
    pass


class RenderJobCreate(BaseModel):
    stem_id: str
    settings: RenderSettings = RenderSettings()


class RenderProgress(BaseModel):
    progress: float = Field(..., ge=0.0, le=1.0)
    message: Optional[str] = None


class EmptyResponse(BaseModel):
    message: str = "Operation successful."
