"""Pydantic-Schemas für die Neuromorphe Traum-Engine v2.0

Diese Datei definiert alle Pydantic-Modelle für API-Validierung und Serialisierung.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


# Enums für bessere Typisierung
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class GenerationStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    ARRANGING = "arranging"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class EnergyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ComplexityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QualityRating(str, Enum):
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class JobType(str, Enum):
    PREPROCESS = "preprocess"
    GENERATE = "generate"
    ANALYZE = "analyze"
    CLEANUP = "cleanup"


# Base-Schemas
class BaseSchema(BaseModel):
    """Basis-Schema mit gemeinsamen Konfigurationen"""
    
    class Config:
        orm_mode = True
        use_enum_values = True
        validate_assignment = True


# Stem-Schemas
class StemBase(BaseSchema):
    """Basis-Schema für Stems"""
    filename: str = Field(..., min_length=1, max_length=255)
    duration: Optional[float] = Field(None, ge=0)
    sample_rate: Optional[int] = Field(None, ge=8000, le=192000)
    channels: Optional[int] = Field(None, ge=1, le=8)
    bit_depth: Optional[int] = Field(None, ge=8, le=32)
    
    # Musiktheorie
    key: Optional[str] = Field(None, max_length=10)
    bpm: Optional[float] = Field(None, ge=20, le=300)
    time_signature: Optional[str] = Field(None, max_length=10)
    
    # Kategorisierung
    instrument: Optional[str] = Field(None, max_length=100)
    genre: Optional[str] = Field(None, max_length=100)
    mood: Optional[str] = Field(None, max_length=100)
    energy_level: Optional[EnergyLevel] = None
    
    # Tags
    auto_tags: Optional[List[str]] = Field(default_factory=list)
    manual_tags: Optional[List[str]] = Field(default_factory=list)


class StemCreate(StemBase):
    """Schema für Stem-Erstellung"""
    original_path: str = Field(..., min_length=1, max_length=500)
    file_hash: str = Field(..., min_length=32, max_length=64)
    file_size: int = Field(..., ge=0)


class StemUpdate(BaseSchema):
    """Schema für Stem-Updates"""
    filename: Optional[str] = Field(None, min_length=1, max_length=255)
    key: Optional[str] = Field(None, max_length=10)
    bpm: Optional[float] = Field(None, ge=20, le=300)
    time_signature: Optional[str] = Field(None, max_length=10)
    instrument: Optional[str] = Field(None, max_length=100)
    genre: Optional[str] = Field(None, max_length=100)
    mood: Optional[str] = Field(None, max_length=100)
    energy_level: Optional[EnergyLevel] = None
    manual_tags: Optional[List[str]] = Field(default_factory=list)
    processing_status: Optional[ProcessingStatus] = None


class StemResponse(StemBase):
    """Schema für Stem-Antworten"""
    id: int
    original_path: str
    processed_path: Optional[str] = None
    file_hash: str
    file_size: int
    
    # Neuromorphe Analyse
    audio_embedding: Optional[List[float]] = None
    semantic_analysis: Optional[Dict[str, Any]] = None
    pattern_analysis: Optional[Dict[str, Any]] = None
    neural_features: Optional[Dict[str, Any]] = None
    perceptual_mapping: Optional[Dict[str, Any]] = None
    
    # Qualität
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    complexity_level: Optional[ComplexityLevel] = None
    recommended_usage: Optional[List[str]] = Field(default_factory=list)
    
    # Status
    processing_status: ProcessingStatus
    processing_error: Optional[str] = None
    
    # Zeitstempel
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None


class StemSearchRequest(BaseSchema):
    """Schema für Stem-Suche"""
    query: Optional[str] = Field(None, min_length=1, max_length=500)
    instrument: Optional[str] = Field(None, max_length=100)
    genre: Optional[str] = Field(None, max_length=100)
    mood: Optional[str] = Field(None, max_length=100)
    energy_level: Optional[EnergyLevel] = None
    key: Optional[str] = Field(None, max_length=10)
    bpm_min: Optional[float] = Field(None, ge=20, le=300)
    bpm_max: Optional[float] = Field(None, ge=20, le=300)
    quality_min: Optional[float] = Field(None, ge=0, le=1)
    complexity_level: Optional[ComplexityLevel] = None
    processing_status: Optional[ProcessingStatus] = None
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)
    
    @validator('bpm_max')
    def validate_bpm_range(cls, v, values):
        if v is not None and 'bpm_min' in values and values['bpm_min'] is not None:
            if v < values['bpm_min']:
                raise ValueError('bpm_max must be greater than or equal to bpm_min')
        return v


# Track-Schemas
class TrackBase(BaseSchema):
    """Basis-Schema für Tracks"""
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Musik-Parameter
    target_bpm: Optional[float] = Field(None, ge=20, le=300)
    target_key: Optional[str] = Field(None, max_length=10)
    target_genre: Optional[str] = Field(None, max_length=100)
    target_mood: Optional[str] = Field(None, max_length=100)
    target_energy: Optional[EnergyLevel] = None
    
    # Metadaten
    tags: Optional[List[str]] = Field(default_factory=list)


class TrackCreate(TrackBase):
    """Schema für Track-Erstellung"""
    original_prompt: str = Field(..., min_length=1, max_length=2000)
    sample_rate: int = Field(44100, ge=22050, le=192000)
    channels: int = Field(2, ge=1, le=8)


class TrackUpdate(BaseSchema):
    """Schema für Track-Updates"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    target_bpm: Optional[float] = Field(None, ge=20, le=300)
    target_key: Optional[str] = Field(None, max_length=10)
    target_genre: Optional[str] = Field(None, max_length=100)
    target_mood: Optional[str] = Field(None, max_length=100)
    target_energy: Optional[EnergyLevel] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    generation_status: Optional[GenerationStatus] = None


class TrackResponse(TrackBase):
    """Schema für Track-Antworten"""
    id: int
    original_prompt: str
    
    # Datei-Informationen
    output_path: Optional[str] = None
    preview_path: Optional[str] = None
    file_hash: Optional[str] = None
    duration: Optional[float] = Field(None, ge=0)
    sample_rate: int
    channels: int
    file_size: Optional[int] = Field(None, ge=0)
    
    # Arrangement und Rendering
    arrangement_plan: Optional[Dict[str, Any]] = None
    track_structure: Optional[Dict[str, Any]] = None
    rendering_settings: Optional[Dict[str, Any]] = None
    master_effects: Optional[Dict[str, Any]] = None
    
    # Status und Qualität
    generation_status: GenerationStatus
    generation_error: Optional[str] = None
    quality_rating: Optional[QualityRating] = None
    
    # Metadaten
    metadata: Optional[Dict[str, Any]] = None
    
    # Zeitstempel
    created_at: datetime
    updated_at: datetime
    generated_at: Optional[datetime] = None


# Track-Stem-Schemas
class TrackStemBase(BaseSchema):
    """Basis-Schema für Track-Stem-Verknüpfungen"""
    section_name: str = Field(..., min_length=1, max_length=100)
    layer_name: str = Field(..., min_length=1, max_length=100)
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    
    # Audio-Verarbeitung
    volume: float = Field(1.0, ge=0, le=2.0)
    pan: float = Field(0.0, ge=-1.0, le=1.0)
    pitch_shift: float = Field(0.0, ge=-24.0, le=24.0)
    time_stretch: float = Field(1.0, ge=0.25, le=4.0)
    
    # Effekte
    fade_in: float = Field(0.0, ge=0)
    fade_out: float = Field(0.0, ge=0)
    
    # Metadaten
    usage_context: Optional[str] = Field(None, max_length=100)
    importance: float = Field(1.0, ge=0, le=1.0)
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v


class TrackStemCreate(TrackStemBase):
    """Schema für Track-Stem-Erstellung"""
    track_id: int = Field(..., ge=1)
    stem_id: int = Field(..., ge=1)
    effects: Optional[Dict[str, Any]] = None


class TrackStemUpdate(BaseSchema):
    """Schema für Track-Stem-Updates"""
    section_name: Optional[str] = Field(None, min_length=1, max_length=100)
    layer_name: Optional[str] = Field(None, min_length=1, max_length=100)
    start_time: Optional[float] = Field(None, ge=0)
    end_time: Optional[float] = Field(None, ge=0)
    volume: Optional[float] = Field(None, ge=0, le=2.0)
    pan: Optional[float] = Field(None, ge=-1.0, le=1.0)
    pitch_shift: Optional[float] = Field(None, ge=-24.0, le=24.0)
    time_stretch: Optional[float] = Field(None, ge=0.25, le=4.0)
    fade_in: Optional[float] = Field(None, ge=0)
    fade_out: Optional[float] = Field(None, ge=0)
    effects: Optional[Dict[str, Any]] = None
    usage_context: Optional[str] = Field(None, max_length=100)
    importance: Optional[float] = Field(None, ge=0, le=1.0)


class TrackStemResponse(TrackStemBase):
    """Schema für Track-Stem-Antworten"""
    id: int
    track_id: int
    stem_id: int
    effects: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    # Verknüpfte Daten
    stem: Optional[StemResponse] = None


# Processing Job Schemas
class ProcessingJobBase(BaseSchema):
    """Basis-Schema für Verarbeitungsaufträge"""
    job_type: JobType
    priority: int = Field(5, ge=1, le=10)
    input_data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None


class ProcessingJobCreate(ProcessingJobBase):
    """Schema für Job-Erstellung"""
    pass


class ProcessingJobUpdate(BaseSchema):
    """Schema für Job-Updates"""
    job_status: Optional[str] = Field(None, max_length=50)
    progress_percentage: Optional[float] = Field(None, ge=0, le=100)
    current_step: Optional[str] = Field(None, max_length=200)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ProcessingJobResponse(ProcessingJobBase):
    """Schema für Job-Antworten"""
    id: int
    job_status: str
    progress_percentage: float
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    cpu_time: Optional[float] = None
    memory_peak: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Generation Request Schemas
class GenerationRequest(BaseSchema):
    """Schema für Track-Generierungsanfragen"""
    prompt: str = Field(..., min_length=1, max_length=2000)
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    
    # Musik-Parameter (optional, werden aus Prompt extrahiert wenn nicht angegeben)
    target_bpm: Optional[float] = Field(None, ge=20, le=300)
    target_key: Optional[str] = Field(None, max_length=10)
    target_genre: Optional[str] = Field(None, max_length=100)
    target_mood: Optional[str] = Field(None, max_length=100)
    target_energy: Optional[EnergyLevel] = None
    target_duration: Optional[float] = Field(None, ge=10, le=600)  # 10 Sekunden bis 10 Minuten
    
    # Rendering-Optionen
    sample_rate: int = Field(44100, ge=22050, le=192000)
    channels: int = Field(2, ge=1, le=8)
    apply_mastering: bool = Field(True)
    create_preview: bool = Field(True)
    
    # Erweiterte Optionen
    stem_selection_strategy: str = Field("semantic", regex="^(semantic|random|quality)$")
    arrangement_complexity: ComplexityLevel = Field(ComplexityLevel.MEDIUM)
    allow_pitch_shifting: bool = Field(True)
    allow_time_stretching: bool = Field(True)
    
    # Tags
    tags: Optional[List[str]] = Field(default_factory=list)


class PreprocessRequest(BaseSchema):
    """Schema für Preprocessing-Anfragen"""
    file_paths: List[str] = Field(..., min_items=1, max_items=100)
    force_reprocess: bool = Field(False)
    extract_features: bool = Field(True)
    generate_tags: bool = Field(True)
    analyze_quality: bool = Field(True)


# Response Schemas
class HealthResponse(BaseSchema):
    """Schema für Health-Check-Antworten"""
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    timestamp: datetime
    version: str
    uptime_seconds: float
    database_status: str
    clap_model_status: str
    processing_queue_size: int
    system_metrics: Dict[str, Any]


class StatisticsResponse(BaseSchema):
    """Schema für Statistik-Antworten"""
    total_stems: int
    processed_stems: int
    total_tracks: int
    completed_tracks: int
    processing_rate: float
    average_quality: float
    system_performance: Dict[str, Any]
    recent_activity: Dict[str, Any]


class SearchResponse(BaseSchema):
    """Schema für Such-Antworten"""
    results: List[StemResponse]
    total_count: int
    query_time_ms: float
    search_metadata: Dict[str, Any]


class ErrorResponse(BaseSchema):
    """Schema für Fehler-Antworten"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None


class SuccessResponse(BaseSchema):
    """Schema für Erfolgs-Antworten"""
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime


# Arrangement Schemas
class ArrangementSection(BaseSchema):
    """Schema für Arrangement-Sektionen"""
    name: str = Field(..., min_length=1, max_length=100)
    start_time: float = Field(..., ge=0)
    duration: float = Field(..., gt=0)
    energy_level: EnergyLevel
    stem_layers: Dict[str, Dict[str, Any]]  # layer_name -> layer_config
    effects: Optional[Dict[str, Any]] = None
    transitions: Optional[Dict[str, Any]] = None


class ArrangementPlan(BaseSchema):
    """Schema für vollständige Arrangement-Pläne"""
    track_id: Optional[int] = None
    total_duration: float = Field(..., gt=0)
    target_bpm: float = Field(..., ge=20, le=300)
    target_key: str = Field(..., max_length=10)
    genre: str = Field(..., max_length=100)
    mood: str = Field(..., max_length=100)
    energy_progression: List[EnergyLevel]
    sections: List[ArrangementSection]
    global_effects: Optional[Dict[str, Any]] = None
    mastering_settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# Configuration Schemas
class ConfigurationSetting(BaseSchema):
    """Schema für Konfigurationseinstellungen"""
    category: str = Field(..., min_length=1, max_length=100)
    key: str = Field(..., min_length=1, max_length=200)
    value: Any
    data_type: str = Field(..., regex="^(string|integer|float|boolean|json)$")
    description: Optional[str] = Field(None, max_length=1000)
    is_user_configurable: bool = Field(True)
    requires_restart: bool = Field(False)
    validation_rules: Optional[Dict[str, Any]] = None
    default_value: Optional[Any] = None


# Batch Operation Schemas
class BatchProcessRequest(BaseSchema):
    """Schema für Batch-Verarbeitungsanfragen"""
    operation: str = Field(..., regex="^(preprocess|analyze|cleanup|reindex)$")
    filters: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    priority: int = Field(5, ge=1, le=10)


class BatchProcessResponse(BaseSchema):
    """Schema für Batch-Verarbeitungsantworten"""
    job_id: int
    operation: str
    estimated_items: int
    estimated_duration_seconds: Optional[float] = None
    created_at: datetime


# Validation Schemas
class ValidationResult(BaseSchema):
    """Schema für Validierungsergebnisse"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


# Export Schema
class ExportRequest(BaseSchema):
    """Schema für Export-Anfragen"""
    format: str = Field(..., regex="^(json|csv|xml)$")
    include_stems: bool = Field(True)
    include_tracks: bool = Field(True)
    include_metadata: bool = Field(True)
    date_range: Optional[Dict[str, datetime]] = None
    filters: Optional[Dict[str, Any]] = None


# Pagination Schema
class PaginationParams(BaseSchema):
    """Schema für Paginierung"""
    page: int = Field(1, ge=1)
    size: int = Field(50, ge=1, le=200)
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size
    
    @property
    def limit(self) -> int:
        return self.size


class PaginatedResponse(BaseSchema):
    """Schema für paginierte Antworten"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    
    @validator('pages', pre=True, always=True)
    def calculate_pages(cls, v, values):
        if 'total' in values and 'size' in values:
            return max(1, (values['total'] + values['size'] - 1) // values['size'])
        return v