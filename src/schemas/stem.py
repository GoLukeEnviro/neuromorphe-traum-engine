"""Stem-spezifische Schemas für die Neuromorphe Traum-Engine."""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class StemBase(BaseModel):
    """Basis-Schema für ein Audio-Stem."""
    filename: str
    title: Optional[str] = None
    genre: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    sample_rate: Optional[int] = Field(default=None, gt=0)
    bit_depth: Optional[int] = None
    channels: Optional[int] = None
    auto_tags: List[str] = []
    manual_tags: List[str] = []
    semantic_analysis: Optional[Dict[str, Any]] = None


class StemMetadata(BaseModel):
    """Schema für Metadaten eines Stems."""
    bpm: Optional[float] = None
    key: Optional[str] = None
    time_signature: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    category: Optional[str] = None
    energy_level: Optional[str] = None
    source: Optional[str] = None
    auto_tags: Optional[List[str]] = None
    manual_tags: Optional[List[str]] = None
    harmonic_complexity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rhythmic_complexity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    complexity_level: Optional[str] = None


class StemFeatures(BaseModel):
    """Schema für extrahierte Features eines Stems."""
    mfcc: Optional[List[float]] = None
    spectral_centroid: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    rms_energy: Optional[float] = None
    chroma: Optional[List[float]] = None
    tempo: Optional[float] = None
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


class StemBatch(BaseModel):
    """Sammlung von Stem-IDs für Batch-Operationen."""
    stem_ids: List[int] = Field(..., min_length=1)

    model_config = ConfigDict(from_attributes=True)


class StemAnalysis(BaseModel):
    """Vollständige Analyse eines Stems (mehrdimensionale Auswertung)."""
    file_info: Dict[str, Any] = Field(..., min_length=1)
    temporal: Optional[Dict[str, Any]] = None
    spectral: Optional[Dict[str, Any]] = None
    rhythmic: Optional[Dict[str, Any]] = None
    harmonic: Optional[Dict[str, Any]] = None
    perceptual: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None


class StemSearch(BaseModel):
    """Suchfilter für Stems."""
    query_text: Optional[str] = None
    query: Optional[str] = None
    category: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    key: Optional[str] = None
    compatible_keys: Optional[List[str]] = None
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
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
        if (
            self.harmonic_complexity_min is not None
            and self.harmonic_complexity_max is not None
            and self.harmonic_complexity_min > self.harmonic_complexity_max
        ):
            raise ValueError(
                "harmonic_complexity_min must not exceed harmonic_complexity_max"
            )
        if (
            self.rhythmic_complexity_min is not None
            and self.rhythmic_complexity_max is not None
            and self.rhythmic_complexity_min > self.rhythmic_complexity_max
        ):
            raise ValueError(
                "rhythmic_complexity_min must not exceed rhythmic_complexity_max"
            )
        return self


class SearchResult(BaseModel):
    """Ergebnis einer semantischen Suche nach Audio-Stems."""
    
    stem: Any  # Database Stem model
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Ähnlichkeitswert zwischen 0 und 1")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class StemResponse(BaseModel):
    """Response-Schema für einzelne Stems.

    Nur ``id`` und ``filename`` sind zwingend; die übrigen Felder sind
    optional, damit auch unvollständig verarbeitete Stems ausgegeben werden
    können.
    """
    
    id: int
    filename: str
    original_path: Optional[str] = None
    processed_path: Optional[str] = None
    file_hash: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    file_size: Optional[int] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    musical_key: Optional[str] = None
    time_signature: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    category: Optional[str] = None
    energy_level: Optional[str] = None
    source: Optional[str] = None
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
    processing_status: Optional[str] = None
    processing_error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True, extra="ignore")


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


class StemCreate(StemBase):
    """Schema für die Erstellung eines neuen Stems.

    Erbt alle optionalen Felder von ``StemBase``; nur ``filename`` ist
    zwingend. Die übrigen Audio-Felder werden beim Preprocessing ergänzt.
    """

    original_path: Optional[str] = None
    file_hash: Optional[str] = Field(default=None, min_length=32, max_length=64)

    # Kategorisierung
    time_signature: Optional[str] = None
    source: str = Field(default="original")

    # Verarbeitungsstatus
    processing_status: str = Field(default="pending")
    processing_error: Optional[str] = None
    processed_path: Optional[str] = None
    processed_at: Optional[datetime] = None


class StemSimilarity(BaseModel):
    """Schema für die Ähnlichkeit von Stems."""
    stem_id_1: int
    stem_id_2: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class StemUpdate(BaseModel):
    """Schema für die Aktualisierung eines Stems.
    
    Alle Felder sind optional. Wird ``sample_rate`` gesetzt, muss er
    positiv sein.
    """
    
    # Allgemein
    title: Optional[str] = None
    filename: Optional[str] = None
    
    # Musikalische Eigenschaften
    key: Optional[str] = None
    bpm: Optional[float] = None
    category: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy_level: Optional[str] = None
    
    # Audio-Eigenschaften (Validierung wie in StemBase)
    sample_rate: Optional[int] = Field(default=None, ge=8000, le=192000)
    duration: Optional[float] = Field(default=None, gt=0)
    file_size: Optional[int] = Field(default=None, ge=0)
    bit_depth: Optional[int] = Field(default=None, gt=0)
    channels: Optional[int] = Field(default=None, ge=1)
    
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