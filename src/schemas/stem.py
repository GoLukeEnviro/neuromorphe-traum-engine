"""Stem-spezifische Schemas für die Neuromorphe Traum-Engine."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


class SearchResult(BaseModel):
    """Ergebnis einer semantischen Suche nach Audio-Stems."""
    
    stem: Any  # Database Stem model
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Ähnlichkeitswert zwischen 0 und 1")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class StemResponse(BaseModel):
    """Response-Schema für einzelne Stems."""
    
    id: int
    filename: str
    original_path: str
    processed_path: Optional[str] = None
    file_hash: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    file_size: int
    bpm: Optional[float] = None
    key: Optional[str] = None
    time_signature: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    category: Optional[str] = None
    energy_level: Optional[str] = None
    source: str
    auto_tags: Optional[List[str]] = None
    manual_tags: Optional[List[str]] = None
    audio_embedding: Optional[List[float]] = None
    semantic_analysis: Optional[Dict[str, Any]] = None
    pattern_analysis: Optional[Dict[str, Any]] = None
    neural_features: Optional[Dict[str, Any]] = None
    perceptual_mapping: Optional[Dict[str, Any]] = None
    harmonic_complexity: Optional[float] = None
    rhythmic_complexity: Optional[float] = None
    quality_score: Optional[float] = None
    complexity_level: Optional[str] = None
    recommended_usage: Optional[List[str]] = None
    processing_status: str
    processing_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class StemSearchRequest(BaseModel):
    """Request-Schema für Stems-Suche."""
    
    query: str = Field(..., min_length=1, description="Textuelle Suchanfrage")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximale Anzahl der Ergebnisse")
    category_filter: Optional[str] = None
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    genre_filter: Optional[str] = None
    mood_filter: Optional[str] = None


class StemUploadRequest(BaseModel):
    """Request-Schema für Stems-Upload."""
    
    filename: str = Field(..., min_length=1)
    category: Optional[str] = None
    tags: List[str] = []
    bpm: Optional[float] = None
    key: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None


class StemBatchResponse(BaseModel):
    """Response-Schema für Stems-Listen."""
    
    stems: List[StemResponse]
    total_count: int
    page: int
    per_page: int
    has_next: bool
    has_previous: bool


class StemAnalysisRequest(BaseModel):
    """Request-Schema für Stems-Analyse."""
    
    stem_id: int
    include_embedding: bool = False
    extract_features: bool = True


class StemAnalysisResponse(BaseModel):
    """Response-Schema für Stems-Analyse-Ergebnisse."""
    
    stem_id: int
    features: Dict[str, Any]
    embedding: Optional[List[float]] = None
    analysis_time: float


class StemCreate(BaseModel):
    """Schema für die Erstellung eines neuen Stems."""
    
    filename: str = Field(..., min_length=1)
    original_path: str = Field(..., min_length=1)
    file_hash: str = Field(..., min_length=32, max_length=64)
    
    # Audio-Eigenschaften
    duration: float = Field(..., gt=0)
    sample_rate: int = Field(..., gt=0)
    channels: int = Field(..., ge=1)
    bit_depth: Optional[int] = None
    file_size: int = Field(..., ge=0)
    
    # Musikalische Eigenschaften
    key: Optional[str] = None
    bpm: Optional[float] = None
    time_signature: Optional[str] = None
    
    # Kategorisierung
    category: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy_level: Optional[str] = None
    source: str = Field(default="original")
    
    # Tags
    auto_tags: Optional[List[str]] = None
    manual_tags: Optional[List[str]] = None
    
    # KI-Features
    audio_embedding: Optional[List[float]] = None
    semantic_analysis: Optional[Dict[str, Any]] = None
    pattern_analysis: Optional[Dict[str, Any]] = None
    neural_features: Optional[Dict[str, Any]] = None
    perceptual_mapping: Optional[Dict[str, Any]] = None
    
    # Qualitäts-Metriken
    harmonic_complexity: Optional[float] = Field(None, ge=0.0, le=1.0)
    rhythmic_complexity: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    complexity_level: Optional[str] = None
    recommended_usage: Optional[List[str]] = None
    
    # Verarbeitungsstatus
    processing_status: str = Field(default="pending")
    processing_error: Optional[str] = None
    processed_path: Optional[str] = None
    processed_at: Optional[datetime] = None


class StemUpdate(BaseModel):
    """Schema für die Aktualisierung eines Stems."""
    
    # Musikalische Eigenschaften
    key: Optional[str] = None
    bpm: Optional[float] = None
    category: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy_level: Optional[str] = None
    
    # Tags
    auto_tags: Optional[List[str]] = None
    manual_tags: Optional[List[str]] = None
    
    # KI-Features
    audio_embedding: Optional[List[float]] = None
    semantic_analysis: Optional[Dict[str, Any]] = None
    pattern_analysis: Optional[Dict[str, Any]] = None
    neural_features: Optional[Dict[str, Any]] = None
    perceptual_mapping: Optional[Dict[str, Any]] = None
    
    # Qualitäts-Metriken
    harmonic_complexity: Optional[float] = Field(None, ge=0.0, le=1.0)
    rhythmic_complexity: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    complexity_level: Optional[str] = None
    recommended_usage: Optional[List[str]] = None
    
    # Verarbeitungsstatus
    processing_status: Optional[str] = None
    processing_error: Optional[str] = None
    processed_path: Optional[str] = None