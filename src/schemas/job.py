from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class JobType(str, Enum):
    """Typ eines Verarbeitung-Jobs"""
    PREPROCESS = "preprocess"
    ANALYZE = "analyze"
    GENERATE = "generate"
    RENDER = "render"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    DELETE = "delete"
    SYSTEM = "system"


class JobStatus(str, Enum):
    """Status eines Verarbeitung-Jobs"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingJobBase(BaseModel):
    """Basis-Schema für einen Verarbeitung-Job"""
    job_type: JobType
    job_status: JobStatus = JobStatus.PENDING
    input_data: Optional[Dict[str, Any]] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = Field(None, max_length=1000)
    error_traceback: Optional[str] = None
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    current_step: Optional[str] = Field(None, max_length=200)
    total_steps: Optional[int] = Field(None, ge=1)
    cpu_time: Optional[float] = Field(None, ge=0.0)
    memory_peak: Optional[int] = Field(None, ge=0)
    model_config = ConfigDict(use_enum_values=True)


class ProcessingJobCreate(ProcessingJobBase):
    """Schema für die Erstellung eines Verarbeitung-Jobs"""
    pass


class ProcessingJobUpdate(ProcessingJobBase):
    """Schema für die Aktualisierung eines Verarbeitung-Jobs"""
    pass


class ProcessingJobResponse(ProcessingJobBase):
    """Schema für die Antwort mit Job-Daten"""
    id: int
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = Field(None, description="Ausführungszeit in Sekunden")
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class ProcessingJobSearch(BaseModel):
    """Schema für Job-Suche"""
    job_type: Optional[JobType] = None
    job_status: Optional[JobStatus] = None
    priority_min: Optional[int] = Field(None, ge=1, le=10)
    priority_max: Optional[int] = Field(None, ge=1, le=10)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    completed_after: Optional[datetime] = None
    completed_before: Optional[datetime] = None
    progress_min: Optional[float] = Field(None, ge=0.0, le=100.0)
    progress_max: Optional[float] = Field(None, ge=0.0, le=100.0)
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=100)
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
    avg_duration: Optional[float] = Field(None, description="Durchschnittliche Ausführungszeit in Sekunden")
    avg_cpu_time: Optional[float] = Field(None, description="Durchschnittliche CPU-Zeit in Sekunden")
    avg_memory_peak: Optional[float] = Field(None, description="Durchschnittlicher Peak-Memory in MB")
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