"""API-Schemata für die Neuromorphe Traum-Engine."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, field_validator
from datetime import datetime


class SearchRequest(BaseModel):
    """Schema für Such-Anfragen."""
    query: str
    limit: Optional[int] = 10
    offset: Optional[int] = 0
    filters: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class APIResponse(BaseModel):
    """Basis-Schema für API-Antworten."""
    success: bool
    message: Optional[str] = None
    timestamp: datetime = datetime.now()
    
    model_config = ConfigDict(from_attributes=True)


class APIError(APIResponse):
    """Schema für API-Fehler."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class APISuccess(APIResponse):
    """Schema für erfolgreiche API-Antworten."""
    success: bool = True
    data: Optional[Any] = None


class APIPagination(BaseModel):
    """Schema für Paginierung."""
    page: int = 1
    per_page: int = 10
    total: int = 0
    pages: int = 0
    
    model_config = ConfigDict(from_attributes=True)


class APIFilter(BaseModel):
    """Schema für API-Filter."""
    field: str
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, in, like
    value: Any
    
    model_config = ConfigDict(from_attributes=True)


class HealthCheck(BaseModel):
    """Schema für Health-Check."""
    status: str = "healthy"
    timestamp: datetime = datetime.now()
    version: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class HealthStatus(BaseModel):
    """Schema für detaillierten Health-Status."""
    overall: str
    database: str
    storage: str
    ai_models: str
    
    model_config = ConfigDict(from_attributes=True)


class SystemInfo(BaseModel):
    """Schema für System-Informationen."""
    version: str
    environment: str
    uptime: float
    memory_usage: float
    cpu_usage: float
    
    model_config = ConfigDict(from_attributes=True)


class ServiceStatus(BaseModel):
    """Schema für Service-Status."""
    name: str
    status: str
    last_check: datetime
    response_time: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)


class AnalysisRequest(BaseModel):
    """Schema für Analyse-Anfragen."""
    audio_path: str
    analysis_type: str = "full"
    parameters: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class AnalysisResponse(BaseModel):
    """Schema für Analyse-Antworten."""
    analysis_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class SimilarityRequest(BaseModel):
    """Schema für Ähnlichkeits-Anfragen."""
    reference_id: str
    threshold: float = 0.7
    limit: int = 10
    
    model_config = ConfigDict(from_attributes=True)


class SimilarityResponse(BaseModel):
    """Schema für Ähnlichkeits-Antworten."""
    reference_id: str
    matches: List[Dict[str, Any]]
    total_matches: int
    
    model_config = ConfigDict(from_attributes=True)


class UploadRequest(BaseModel):
    """Schema für Upload-Anfragen."""
    filename: str
    content_type: str
    size: int
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class UploadResponse(BaseModel):
    """Schema für Upload-Antworten."""
    upload_id: str
    filename: str
    url: str
    status: str = "uploaded"
    
    model_config = ConfigDict(from_attributes=True)


class DownloadRequest(BaseModel):
    """Schema für Download-Anfragen."""
    file_id: str
    format: Optional[str] = None
    quality: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class DownloadResponse(BaseModel):
    """Schema für Download-Antworten."""
    file_id: str
    download_url: str
    expires_at: datetime
    
    model_config = ConfigDict(from_attributes=True)