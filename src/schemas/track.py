from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TrackStatus(str, Enum):
    """Status eines generierten Tracks"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GeneratedTrackBase(BaseModel):
    """Basis-Schema für generierte Tracks"""
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    original_prompt: str = Field(..., min_length=1)
    
    # Audio-Eigenschaften
    duration: Optional[float] = Field(None, ge=0)
    sample_rate: int = Field(44100, ge=8000, le=192000)
    channels: Optional[int] = Field(2, ge=1, le=8)
    file_size: Optional[int] = Field(None, ge=0)
    
    # Musik-Parameter
    target_bpm: Optional[float] = Field(None, ge=20, le=300)
    target_key: Optional[str] = Field(None, max_length=10)
    target_genre: Optional[str] = Field(None, max_length=100)
    target_mood: Optional[str] = Field(None, max_length=100)
    target_energy: Optional[str] = Field(None, max_length=50)
    
    # Generierungs-Parameter
    generation_model: Optional[str] = Field(None, max_length=100)
    generation_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Qualität und Bewertung
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    
    # Status und Metadaten
    status: TrackStatus = Field(TrackStatus.PENDING)
    error_message: Optional[str] = Field(None, max_length=1000)
    processing_time: Optional[float] = Field(None, ge=0)
    
    # Tags und Kategorisierung
    tags: List[str] = Field(default_factory=list)
    is_public: bool = Field(False)
    is_featured: bool = Field(False)
    model_config = ConfigDict(use_enum_values=True)


class GeneratedTrackCreate(GeneratedTrackBase):
    """Schema für die Erstellung eines generierten Tracks"""
    pass


class GeneratedTrackUpdate(BaseModel):
    """Schema für die Aktualisierung eines generierten Tracks"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Audio-Eigenschaften
    duration: Optional[float] = Field(None, ge=0)
    file_size: Optional[int] = Field(None, ge=0)
    
    # Musik-Parameter
    target_bpm: Optional[float] = Field(None, ge=20, le=300)
    target_key: Optional[str] = Field(None, max_length=10)
    target_genre: Optional[str] = Field(None, max_length=100)
    target_mood: Optional[str] = Field(None, max_length=100)
    target_energy: Optional[str] = Field(None, max_length=50)
    
    # Qualität und Bewertung
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    
    # Status und Metadaten
    status: Optional[TrackStatus] = None
    error_message: Optional[str] = Field(None, max_length=1000)
    processing_time: Optional[float] = Field(None, ge=0)
    
    # Tags und Kategorisierung
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    is_featured: Optional[bool] = None
    model_config = ConfigDict(use_enum_values=True)


class GeneratedTrackResponse(GeneratedTrackBase):
    """Schema für die Antwort mit Track-Daten"""
    id: int
    output_path: Optional[str] = None
    preview_path: Optional[str] = None
    file_hash: Optional[str] = None
    
    # Zeitstempel
    created_at: datetime
    updated_at: datetime
    
    # Verwendete Stems
    used_stems: List[int] = Field(default_factory=list)
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class GeneratedTrackSearch(BaseModel):
    """Schema für Track-Suche"""
    query: Optional[str] = Field(None, min_length=1, max_length=500)
    genre: Optional[str] = Field(None, max_length=100)
    mood: Optional[str] = Field(None, max_length=100)
    energy: Optional[str] = Field(None, max_length=50)
    key: Optional[str] = Field(None, max_length=10)
    
    # BPM-Bereich
    bpm_min: Optional[float] = Field(None, ge=20, le=300)
    bpm_max: Optional[float] = Field(None, ge=20, le=300)
    
    # Qualitäts-Filter
    quality_min: Optional[float] = Field(None, ge=0, le=1)
    rating_min: Optional[int] = Field(None, ge=1, le=5)
    
    # Status-Filter
    status: Optional[TrackStatus] = None
    is_public: Optional[bool] = None
    is_featured: Optional[bool] = None
    
    # Zeitraum-Filter
    created_after: Optional[datetime] = None