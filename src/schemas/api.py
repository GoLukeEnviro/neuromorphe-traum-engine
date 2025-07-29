from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
from datetime import datetime


class APIResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None


class APIError(BaseModel):
    detail: str


class APISuccess(BaseModel):
    message: str


class APIPagination(BaseModel):
    page: int
    per_page: int
    total_items: int
    total_pages: int


class APIFilter(BaseModel):
    field: str
    operator: str
    value: Any


class HealthCheck(BaseModel):
    status: str
    version: str
    database_status: str
    message: Optional[str] = None


class HealthStatus(BaseModel):
    status: str
    message: Optional[str] = None


class SystemInfo(BaseModel):
    python_version: str
    os_name: str
    processor_type: str
    total_memory_gb: float
    available_memory_gb: float
    cpu_usage_percent: float
    disk_usage_percent: float


class ServiceStatus(BaseModel):
    service_name: str
    status: str
    message: Optional[str] = None


class AnalysisRequest(BaseModel):
    file_path: str
    analysis_type: str


class AnalysisResponse(BaseModel):
    file_path: str
    analysis_type: str
    result: Dict[str, Any]


class SimilarityRequest(BaseModel):
    embedding_1: List[float]
    embedding_2: List[float]


class SimilarityResponse(BaseModel):
    similarity_score: float


class UploadRequest(BaseModel):
    file_name: str
    content_type: str
    file_size: Optional[int] = None


class UploadResponse(BaseModel):
    file_path: str
    message: str


class DownloadRequest(BaseModel):
    file_path: str


class DownloadResponse(BaseModel):
    file_path: str
    message: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category_filter: Optional[str] = None
    bpm_range: Optional[List[float]] = None