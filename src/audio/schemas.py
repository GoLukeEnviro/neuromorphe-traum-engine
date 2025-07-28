from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AudioUploadRequest(BaseModel):
    """Schema for audio file upload request"""
    filename: str = Field(..., min_length=1, max_length=255)
    category: Optional[str] = Field(None, max_length=100)
    bpm: Optional[int] = Field(None, ge=60, le=200)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "techno_beat_128.wav",
                "category": "techno",
                "bpm": 128
            }
        }
    )


class AudioProcessingResponse(BaseModel):
    """Schema for audio processing response"""
    id: str
    filename: str
    status: ProcessingStatus
    message: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class EmbeddingResponse(BaseModel):
    """Schema for CLAP embedding response"""
    id: str
    filename: str
    embedding_size: int
    processing_time: float
    created_at: datetime


class AudioFileInfo(BaseModel):
    """Schema for audio file information"""
    id: str
    filename: str
    category: Optional[str]
    bpm: Optional[int]
    duration: Optional[float]
    sample_rate: Optional[int]
    channels: Optional[int]
    file_size: int
    has_embedding: bool
    created_at: datetime
    updated_at: Optional[datetime]