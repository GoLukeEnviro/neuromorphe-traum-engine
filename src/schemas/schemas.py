from pydantic import BaseModel
from pydantic import BaseModel


class EmptyResponse(BaseModel):
    message: str = "Operation successful."

        elif data_type == ConfigDataType.INTEGER and not isinstance(v, int):
            raise ValueError('Value must be an integer')
        elif data_type == ConfigDataType.FLOAT and not isinstance(v, (int, float)):
            raise ValueError('Value must be a number')
        elif data_type == ConfigDataType.BOOLEAN and not isinstance(v, bool):
            raise ValueError('Value must be a boolean')
        elif data_type == ConfigDataType.LIST and not isinstance(v, list):
            raise ValueError('Value must be a list')
        elif data_type == ConfigDataType.JSON and not isinstance(v, (dict, list)):
            raise ValueError('Value must be a JSON object or array')
            
        return v
    model_config = ConfigDict(use_enum_values=True)

class ConfigurationSettingCreate(ConfigurationSettingBase):
    """Schema für die Erstellung einer Konfigurationseinstellung"""
    pass

class ConfigurationSettingUpdate(BaseModel):
    """Schema für die Aktualisierung einer Konfigurationseinstellung"""
    value: Optional[Union[str, int, float, bool, Dict[str, Any], List[Any]]] = None
    description: Optional[str] = None
    is_user_configurable: Optional[bool] = None
    requires_restart: Optional[bool] = None
    validation_rules: Optional[Dict[str, Any]] = None
    default_value: Optional[Union[str, int, float, bool, Dict[str, Any], List[Any]]] = None
    model_config = ConfigDict(use_enum_values=True)

class ConfigurationSettingResponse(ConfigurationSettingBase):
    """Schema für die Antwort mit Konfigurationsdaten"""
    id: int
    
    # Zeitstempel
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

class ConfigurationSearch(BaseModel):
    """Schema für Konfigurationssuche"""
    category: Optional[ConfigCategory] = None
    key_pattern: Optional[str] = Field(None, max_length=200, description="Suchpattern für Schlüssel")
    data_type: Optional[ConfigDataType] = None
    is_user_configurable: Optional[bool] = None
    requires_restart: Optional[bool] = None
    
    # Paginierung
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=200)
    
    # Sortierung
    order_by: str = Field("category", pattern=r"^(category|key|data_type|created_at|updated_at)$")
    order_desc: bool = Field(False)
    model_config = ConfigDict(use_enum_values=True)

class ConfigurationBatch(BaseModel):
    """Schema für Batch-Operationen mit Konfigurationen"""
    settings: List[ConfigurationSettingCreate] = Field(..., min_length=1, max_length=100)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "settings": [
                {
                    "category": "audio",
                    "key": "sample_rate",
                    "value": 44100,
                    "data_type": "integer",
                    "description": "Standard-Samplerate für Audio-Verarbeitung",
                    "validation_rules": {"min": 8000, "max": 192000}
                },
                {
                    "category": "processing",
                    "key": "max_workers",
                    "value": 4,
                    "data_type": "integer",
                    "description": "Maximale Anzahl Worker-Threads",
                    "requires_restart": True
                }
            ]
        }
    })

class ConfigurationExport(BaseModel):
    """Schema für Konfigurationsexport"""
    categories: Optional[List[ConfigCategory]] = Field(None, description="Zu exportierende Kategorien")
    include_system_settings: bool = Field(False, description="System-Einstellungen einschließen")
    format: str = Field("json", pattern=r"^(json|yaml|env)$", description="Export-Format")
    model_config = ConfigDict(use_enum_values=True)

class ConfigurationImport(BaseModel):
    """Schema für Konfigurationsimport"""
    settings: Dict[str, Dict[str, Any]] = Field(..., description="Konfigurationen gruppiert nach Kategorie")
    overwrite_existing: bool = Field(False, description="Bestehende Einstellungen überschreiben")
    validate_only: bool = Field(False, description="Nur validieren, nicht importieren")
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "settings": {
                "audio": {
                    "sample_rate": 44100,
                    "bit_depth": 24,
                    "channels": 2
                },
                "processing": {
                    "max_workers": 4,
                    "chunk_size": 1024
                }
            },
            "overwrite_existing": True,
            "validate_only": False
        }
    })

class ConfigurationStats(BaseModel):
    """Schema für Konfigurationsstatistiken"""
    total_settings: int
    user_configurable: int
    system_settings: int
    requires_restart_count: int
    
    # Verteilung nach Kategorie
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Verteilung nach Datentyp
    data_type_distribution: Dict[str, int] = Field(default_factory=dict)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_settings": 45,
            "user_configurable": 32,
            "system_settings": 13,
            "requires_restart_count": 8,
            "category_distribution": {
                "audio": 12,
                "processing": 10,
                "ui": 8,
                "database": 6
            },
            "data_type_distribution": {
                "string": 18,
                "integer": 12,
                "boolean": 10,
                "float": 5
            }
        }
    })

# From job.py
class JobType(str, Enum):
    """Typ eines Verarbeitungsauftrags"""
    PREPROCESS = "preprocess"
    GENERATE = "generate"
    ANALYZE = "analyze"
    EXPORT = "export"
    CLEANUP = "cleanup"

class JobStatus(str, Enum):
    """Status eines Verarbeitungsauftrags"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ProcessingJobBase(BaseModel):
    """Basis-Schema für Verarbeitungsaufträge"""
    job_type: JobType
    job_status: JobStatus = Field(JobStatus.PENDING)
    priority: int = Field(5, ge=1, le=10, description="Priorität: 1 (hoch) bis 10 (niedrig)")
    
    # Eingabedaten
    input_data: Dict[str, Any] = Field(..., description="Job-spezifische Eingabedaten")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Zusätzliche Parameter")
    
    # Ausgabedaten
    output_data: Optional[Dict[str, Any]] = Field(None, description="Job-Ergebnisse")
    error_message: Optional[str] = Field(None, max_length=1000)
    error_traceback: Optional[str] = Field(None)
    
    # Progress-Tracking
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    current_step: Optional[str] = Field(None, max_length=200)
    total_steps: Optional[int] = Field(None, ge=1)
    
    # Ressourcen-Verbrauch
    cpu_time: Optional[float] = Field(None, ge=0.0, description="CPU-Zeit in Sekunden")
    memory_peak: Optional[int] = Field(None, ge=0, description="Peak-Memory in MB")
    model_config = ConfigDict(use_enum_values=True)

class ProcessingJobCreate(ProcessingJobBase):
    """Schema für die Erstellung eines Verarbeitungsauftrags"""
    pass

class ProcessingJobUpdate(BaseModel):
    """Schema für die Aktualisierung eines Verarbeitungsauftrags"""
    job_status: Optional[JobStatus] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    
    # Ausgabedaten
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = Field(None, max_length=1000)
    error_traceback: Optional[str] = None
    
    # Progress-Tracking
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    current_step: Optional[str] = Field(None, max_length=200)
    total_steps: Optional[int] = Field(None, ge=1)
    
    # Ressourcen-Verbrauch
    cpu_time: Optional[float] = Field(None, ge=0.0)
    memory_peak: Optional[int] = Field(None, ge=0)
    model_config = ConfigDict(use_enum_values=True)

class ProcessingJobResponse(ProcessingJobBase):
    """Schema für die Antwort mit Job-Daten"""
    id: int
    
    # Zeitstempel
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Berechnete Felder
    duration: Optional[float] = Field(None, description="Ausführungszeit in Sekunden")
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

class ProcessingJobSearch(BaseModel):
    """Schema für Job-Suche"""
    job_type: Optional[JobType] = None
    job_status: Optional[JobStatus] = None
    priority_min: Optional[int] = Field(None, ge=1, le=10)
    priority_max: Optional[int] = Field(None, ge=1, le=10)
    
    # Zeitraum-Filter
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    completed_after: Optional[datetime] = None
    completed_before: Optional[datetime] = None
    
    # Progress-Filter
    progress_min: Optional[float] = Field(None, ge=0.0, le=100.0)
    progress_max: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Paginierung
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=100)
    
    # Sortierung
    order_by: str = Field("created_at", pattern=r"^(created_at|updated_at|priority|progress_percentage|job_type|job_status)$")
    order_desc: bool = Field(True)
    model_config = ConfigDict(use_enum_values=True)

class ProcessingJobStats(BaseModel):
    """Schema für Job-Statistiken"""
    total_jobs: int
    pending_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    
    # Durchschnittswerte
    avg_duration: Optional[float] = Field(None, description="Durchschnittliche Ausführungszeit in Sekunden")
    avg_cpu_time: Optional[float] = Field(None, description="Durchschnittliche CPU-Zeit in Sekunden")
    avg_memory_peak: Optional[float] = Field(None, description="Durchschnittlicher Peak-Memory in MB")
    
    # Verteilung nach Typ
    job_type_distribution: Dict[str, int] = Field(default_factory=dict)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_jobs": 150,
            "pending_jobs": 5,
            "running_jobs": 2,
            "completed_jobs": 140,
            "failed_jobs": 3,
            "cancelled_jobs": 0,
            "avg_duration": 45.2,
            "avg_cpu_time": 38.7,
            "avg_memory_peak": 512,
            "job_type_distribution": {
                "preprocess": 80,
                "generate": 50,
                "analyze": 20
            }
        }
    })

# From metric.py
class MetricType(str, Enum):
    """Typ einer System-Metrik"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    PROCESSING = "processing"
    NETWORK = "network"
    DATABASE = "database"
    API = "api"
    AUDIO = "audio"

class SystemMetricBase(BaseModel):
    """Basis-Schema für System-Metriken"""
    metric_type: MetricType
    metric_name: str = Field(..., min_length=1, max_length=100)
    metric_value: float
    metric_unit: Optional[str] = Field(None, max_length=20, description="Einheit wie %, MB, seconds, etc.")
    
    # Kontext und Tags
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Zusätzlicher Kontext")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Tags für Gruppierung")
    model_config = ConfigDict(use_enum_values=True)

class SystemMetricCreate(SystemMetricBase):
    """Schema für die Erstellung einer System-Metrik"""
    timestamp: Optional[datetime] = Field(None, description="Zeitstempel, falls nicht angegeben wird aktuelle Zeit verwendet")

class SystemMetricResponse(SystemMetricBase):
    """Schema für die Antwort mit Metrik-Daten"""
    id: int
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

class SystemMetricSearch(BaseModel):
    """Schema für Metrik-Suche"""
    metric_type: Optional[MetricType] = None
    metric_name: Optional[str] = Field(None, max_length=100)
    
    # Wert-Filter
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    
    # Zeitraum-Filter
    timestamp_after: Optional[datetime] = None
    timestamp_before: Optional[datetime] = None
    
    # Tag-Filter
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    
    # Paginierung
    skip: int = Field(0, ge=0)
    limit: int = Field(100, ge=1, le=1000)
    
    # Sortierung
    order_by: str = Field("timestamp", pattern=r"^(timestamp|metric_type|metric_name|metric_value)$")
    order_desc: bool = Field(True)
    model_config = ConfigDict(use_enum_values=True)

class MetricAggregation(BaseModel):
    """Schema für aggregierte Metriken"""
    metric_type: MetricType
    metric_name: str
    
    # Aggregierte Werte
    count: int
    min_value: float
    max_value: float
    avg_value: float
    sum_value: float
    
    # Zeitraum
    period_start: datetime
    period_end: datetime
    model_config = ConfigDict(use_enum_values=True)

class SystemMetricsBatch(BaseModel):
    """Schema für Batch-Erstellung von Metriken"""
    metrics: List[SystemMetricCreate] = Field(..., min_length=1, max_length=1000)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "metrics": [
                {
                    "metric_type": "cpu",
                    "metric_name": "usage_percent",
                    "metric_value": 75.5,
                    "metric_unit": "%",
                    "context": {"core": 0},
                    "tags": {"host": "server-01"}
                },
                {
                    "metric_type": "memory",
                    "metric_name": "used_mb",
                    "metric_unit": "MB",
                    "metric_value": 2048,
                    "tags": {"host": "server-01"}
                }
            ]
        }
    })

class SystemMetricsStats(BaseModel):
    """Schema für Metrik-Statistiken"""
    total_metrics: int
    unique_types: int
    unique_names: int
    
    # Zeitraum
    oldest_metric: Optional[datetime] = None
    newest_metric: Optional[datetime] = None
    
    # Verteilung nach Typ
    type_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Top Metriken
    top_metrics_by_count: List[Dict[str, Any]] = Field(default_factory=list)
    # TODO[pydantic]: The following keys were removed: `json_encoders`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    }, json_schema_extra={
        "example": {
            "total_metrics": 50000,
            "unique_types": 8,
            "unique_names": 25,
            "oldest_metric": "2024-01-01T00:00:00Z",
            "newest_metric": "2024-07-28T16:00:00Z",
            "type_distribution": {
                "cpu": 15000,
                "memory": 12000,
                "processing": 8000,
                "disk": 5000
            },
            "top_metrics_by_count": [
                {"metric_name": "cpu_usage_percent", "count": 8000},
                {"metric_name": "memory_used_mb", "count": 6000}
            ]
        }
    })

# From render.py
class RenderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RenderPriority(int, Enum):
    LOW = 1
    MEDIUM = 5
    HIGH = 10

class RenderFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"

class RenderQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"

class RenderProgress(BaseModel):
    job_id: str
    progress: float
    current_step: str
    message: Optional[str] = None

class RenderSettings(BaseModel):
    sample_rate: int = Field(44100, ge=8000, le=192000)
    bit_depth: Optional[int] = None
    normalize: Optional[bool] = None
    apply_mastering: Optional[bool] = None
    fade_in: Optional[float] = None
    fade_out: Optional[float] = None

class RenderJobCreate(BaseModel):
    arrangement_id: str
    format: str
    quality: str
    options: Optional[Dict[str, Any]] = None

class RenderJobUpdate(BaseModel):
    arrangement_id: Optional[str] = None
    format: Optional[str] = None
    quality: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[float] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class RenderJobBase(BaseModel):
    arrangement_id: str
    format: str
    quality: str
    status: Optional[str] = "pending"
    progress: Optional[float] = 0.0
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class RenderJobResponse(BaseModel):
    id: int
    arrangement_id: str
    format: str
    quality: str
    status: str
    progress: float
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

# From stem.py
class StemBase(BaseModel):
    filename: str
    title: Optional[str] = None
    original_path: Optional[str] = None
    processed_path: Optional[str] = None
    file_hash: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: int = Field(44100, ge=8000, le=192000)
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    file_size: Optional[int] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    time_signature: Optional[str] = None
    category: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy_level: Optional[str] = None
    source: Optional[str] = None
    auto_tags: List[str] = Field(default_factory=list)
    manual_tags: List[str] = Field(default_factory=list)
    audio_embedding: Optional[List[float]] = None
    semantic_analysis: Dict[str, Any] = Field(default_factory=dict)
    pattern_analysis: Dict[str, Any] = Field(default_factory=dict)
    neural_features: Dict[str, Any] = Field(default_factory=dict)
    perceptual_mapping: Dict[str, Any] = Field(default_factory=dict)
    harmonic_complexity: Optional[float] = None
    rhythmic_complexity: Optional[float] = None
    quality_score: Optional[float] = None
    complexity_level: Optional[str] = None
    recommended_usage: Optional[List[str]] = None
    processing_status: Optional[str] = None
    processing_error: Optional[str] = None

class StemCreate(StemBase):
    pass

class StemUpdate(StemBase):
    filename: Optional[str] = None
    original_path: Optional[str] = None
    file_hash: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: int = Field(44100, ge=8000, le=192000)
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    file_size: Optional[int] = None

class StemResponse(StemBase):
    id: int

    class Config:
        from_attributes = True

class StemList(BaseModel):
    stems: List[StemResponse]
    total_count: int

class SearchResult(BaseModel):
    stem: StemResponse
    similarity_score: float

class StemSearch(BaseModel):
    query_text: Optional[str] = None
    category: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy_level: Optional[str] = None
    key: Optional[str] = None
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    quality_min: Optional[float] = None
    processing_status: Optional[str] = None
    harmonic_complexity_min: Optional[float] = None
    harmonic_complexity_max: Optional[float] = None
    rhythmic_complexity_min: Optional[float] = None
    rhythmic_complexity_max: Optional[float] = None
    compatible_keys: Optional[List[str]] = None
    audio_embedding_is_not_null: Optional[bool] = None
    audio_embedding_is_null: Optional[bool] = None

    @field_validator('bpm_max')
    @classmethod
    def validate_bpm_range(cls, v, info):
        if v is not None and info.data.get('bpm_min') is not None and v < info.data['bpm_min']:
            raise ValueError('bpm_max must be greater than or equal to bpm_min')
        return v

    @field_validator('harmonic_complexity_max')
    @classmethod
    def validate_harmonic_complexity_range(cls, v, info):
        if v is not None and info.data.get('harmonic_complexity_min') is not None and v < info.data['harmonic_complexity_min']:
            raise ValueError('harmonic_complexity_max must be greater than or equal to harmonic_complexity_min')
        return v

    @field_validator('rhythmic_complexity_max')
    @classmethod
    def validate_rhythmic_complexity_range(cls, v, info):
        if v is not None and info.data.get('rhythmic_complexity_min') is not None and v < info.data['rhythmic_complexity_min']:
            raise ValueError('rhythmic_complexity_max must be greater than or equal to rhythmic_complexity_min')
        return v

class StemMetadata(BaseModel):
    bpm: Optional[float] = None
    key: Optional[str] = None
    category: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy_level: Optional[str] = None
    auto_tags: List[str] = Field(default_factory=list)
    harmonic_complexity: Optional[float] = None
    rhythmic_complexity: Optional[float] = None
    quality_score: Optional[float] = None
    complexity_level: Optional[str] = None

class StemFeatures(BaseModel):
    spectral_centroid: Optional[float] = None
    mfcc: Optional[List[float]] = None

class StemAnalysis(BaseModel):
    file_info: Dict[str, Any]
    temporal: Dict[str, Any]
    spectral: Dict[str, Any]
    rhythmic: Dict[str, Any]
    harmonic: Dict[str, Any]
    perceptual: Dict[str, Any]
    classification: Dict[str, Any]

class StemSimilarity(BaseModel):
    stem_id_1: int
    stem_id_2: int
    similarity_score: float = Field(..., ge=0, le=1)

    @field_validator('stem_id_2')
    @classmethod
    def validate_stem_ids_differ(cls, v, info):
        if v == info.data.get('stem_id_1'):
            raise ValueError('stem_id_1 and stem_id_2 must be different')
        return v

class StemBatch(BaseModel):
    stem_ids: List[int] = Field(..., min_length=1)

# From track.py
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
    # TODO[pydantic]: The following keys were removed: `json_encoders`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
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
    # TODO[pydantic]: The following keys were removed: `json_encoders`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
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
    created_before: Optional[datetime] = None
    
    # Paginierung
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=100)
    
    # Sortierung
    order_by: str = Field("created_at", pattern=r"^(created_at|updated_at|title|quality_score|user_rating|duration)$")
    order_desc: bool = Field(True)
    
    @field_validator('bpm_max')
    @classmethod
    def validate_bpm_range(cls, v, info):
        if v is not None and hasattr(info, 'data') and 'bpm_min' in info.data and info.data['bpm_min'] is not None:
            if v < info.data['bpm_min']:
                raise ValueError('bpm_max must be greater than or equal to bpm_min')
        return v
    # TODO[pydantic]: The following keys were removed: `json_encoders`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = ConfigDict(use_enum_values=True)

class GeneratedTrackBatch(BaseModel):
    """Schema für Batch-Operationen mit Tracks"""
    track_ids: List[int] = Field(..., min_length=1, max_length=100)
    operation: str = Field(..., pattern=r"^(delete|export|update_status|add_tags|remove_tags)$")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "track_ids": [1, 2, 3],
            "operation": "add_tags",
            "parameters": {
                "tags": ["experimental", "ai-generated"]
            }
        }
    })

# From websocket.py
class WebSocketMessage(BaseModel):
    event: str
    payload: Any
    timestamp: Optional[str] = None

class WebSocketEvent(BaseModel):
    event_type: str
    data: Any

class WebSocketResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None

class ConnectionMessage(BaseModel):
    client_id: str
    message: str

class DisconnectionMessage(BaseModel):
    client_id: str
    message: str

class ErrorMessage(BaseModel):
    code: int
    message: str

class RenderProgressMessage(BaseModel):
    job_id: str
    progress: float
    current_step: str
    message: Optional[str] = None

class AnalysisProgressMessage(BaseModel):
    job_id: str
    progress: float
    current_step: str
    message: Optional[str] = None

class SystemStatusMessage(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_jobs: int
    queue_size: int

class NotificationMessage(BaseModel):
    type: str
    title: str
    message: str
    timestamp: Optional[str] = None

class BroadcastMessage(BaseModel):
    event: str
    data: Any

class PrivateMessage(BaseModel):
    recipient_id: str
    message: str

class SubscriptionMessage(BaseModel):
    client_id: str
    topic: str

class UnsubscriptionMessage(BaseModel):
    client_id: str
    topic: str

class HeartbeatMessage(BaseModel):
    client_id: str
    timestamp: str
