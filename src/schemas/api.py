"""Shared request and response schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class APIResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class APIError(BaseModel):
    code: str
    message: str
    status_code: int = Field(ge=100, lt=600)
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    documentation_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class APISuccess(BaseModel):
    message: str
    data: Optional[Any] = None
    status_code: int = Field(default=200, ge=200, lt=400)
    location: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class APIPagination(BaseModel):
    page: int = Field(ge=1)
    page_size: int = Field(default=20, gt=0, le=1000)
    total_count: int = Field(default=0, ge=0)
    total_pages: Optional[int] = Field(default=None, ge=0)
    has_next: Optional[bool] = None
    has_previous: Optional[bool] = None
    next_url: Optional[str] = None
    previous_url: Optional[str] = None
    first_url: Optional[str] = None
    last_url: Optional[str] = None

    @model_validator(mode="after")
    def derive_page_information(self):
        if self.total_pages is None:
            self.total_pages = (
                (self.total_count + self.page_size - 1) // self.page_size
                if self.total_count else 0
            )
        if self.has_next is None:
            self.has_next = self.page < self.total_pages
        if self.has_previous is None:
            self.has_previous = self.page > 1
        return self

    @property
    def per_page(self) -> int:
        return self.page_size

    @property
    def total_items(self) -> int:
        return self.total_count


class APIFilter(BaseModel):
    field: str = Field(min_length=1)
    operator: str
    value: Any
    case_sensitive: bool = False

    @model_validator(mode="after")
    def validate_operator(self):
        allowed = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "between"}
        if self.operator not in allowed:
            raise ValueError(f"Unsupported filter operator: {self.operator}")
        return self


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck(BaseModel):
    status: HealthStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    version: Optional[str] = None
    uptime: Optional[float] = None
    services: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    database_status: Optional[str] = None
    message: Optional[str] = None


class SystemInfo(BaseModel):
    hostname: Optional[str] = None
    platform: Optional[str] = None
    python_version: str
    cpu_count: Optional[int] = None
    memory_total: Optional[float] = None
    memory_available: Optional[float] = None
    disk_total: Optional[float] = None
    disk_available: Optional[float] = None
    load_average: List[float] = Field(default_factory=list)
    network_interfaces: Dict[str, Any] = Field(default_factory=dict)
    os_name: Optional[str] = None
    processor_type: Optional[str] = None
    total_memory_gb: Optional[float] = None
    available_memory_gb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None


class ServiceStatus(BaseModel):
    service_name: str
    status: str
    message: Optional[str] = None


class AnalysisRequest(BaseModel):
    type: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self):
        if self.type not in {"text", "audio"}:
            raise ValueError("Analysis type must be 'text' or 'audio'")
        if self.type == "text" and not self.content:
            raise ValueError("Text analysis requires content")
        if self.type == "audio" and not self.file_path:
            raise ValueError("Audio analysis requires file_path")
        return self

    @property
    def analysis_type(self) -> str:
        return self.type


class AnalysisResponse(BaseModel):
    request_id: str
    type: str
    status: str
    results: Dict[str, Any]
    processing_time: Optional[float] = None
    completed_at: Optional[datetime] = None


class SimilarityRequest(BaseModel):
    query_stem_id: str
    limit: int = Field(default=10, gt=0, le=1000)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    include_metadata: bool = False
    filters: Dict[str, Any] = Field(default_factory=dict)
    exclude_ids: List[str] = Field(default_factory=list)
    sort_by: str = "similarity_desc"


class SimilarityResponse(BaseModel):
    query_stem_id: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_found: int = Field(default=0, ge=0)
    processing_time: Optional[float] = None


class UploadRequest(BaseModel):
    filename: str
    content_type: str
    file_size: Optional[int] = Field(default=None, ge=0, le=1024 ** 3)
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_audio_type(self):
        if not self.content_type.startswith("audio/"):
            raise ValueError("Only audio uploads are supported")
        return self

    @property
    def file_name(self) -> str:
        return self.filename


class UploadResponse(BaseModel):
    upload_id: Optional[str] = None
    stem_id: Optional[str] = None
    filename: Optional[str] = None
    file_size: Optional[int] = None
    status: str = "completed"
    upload_url: Optional[str] = None
    processing_status: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.now)
    file_path: Optional[str] = None
    message: Optional[str] = None


class DownloadRequest(BaseModel):
    file_path: str


class DownloadResponse(BaseModel):
    file_path: str
    message: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0)
    category_filter: Optional[str] = None
    bpm_range: Optional[List[float]] = None


class RateLimitInfo(BaseModel):
    limit: int = Field(gt=0)
    remaining: int = Field(ge=0)
    reset_time: datetime
    window_size: int = Field(gt=0)
    retry_after: Optional[int] = Field(default=None, ge=0)
    exceeded: bool = False


class CacheInfo(BaseModel):
    hit: bool
    key: str
    ttl: Optional[int] = Field(default=None, ge=0)
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    reason: Optional[str] = None


class MetricsInfo(BaseModel):
    request_count: int = Field(ge=0)
    response_time_avg: float = Field(ge=0)
    response_time_p95: float = Field(ge=0)
    response_time_p99: float = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    cache_hit_rate: float = Field(ge=0, le=1)
    active_connections: int = Field(ge=0)
    queue_size: int = Field(ge=0)
    memory_usage: float = Field(ge=0)
    cpu_usage: float = Field(ge=0, le=100)
    timestamp: datetime = Field(default_factory=datetime.now)
