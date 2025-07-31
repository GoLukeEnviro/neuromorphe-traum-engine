"""Schema-Module für die Neuromorphe Traum-Engine."""

from .stem import *
from .base import TimestampMixin
from .api import *
from .render import RenderOutput
from .websocket import ClientInfo

# Audio schemas
from audio.schemas import (
    AudioUploadRequest,
    AudioProcessingResponse,
    EmbeddingResponse,
    AudioFileInfo,
    ProcessingStatus
)

__all__ = [
    # From base.py
    "TimestampMixin",
    # From api.py
    "SearchRequest",
    # From arrangement.py

    # From render.py
    "RenderOutput",
    # From websocket.py
    "ClientInfo",
    # From schemas.py
    "APIResponse",
    "APIError", 
    "APISuccess",
    "APIPagination",
    "APIFilter",
    "HealthCheck",
    "HealthStatus",
    "SystemInfo",
    "ServiceStatus",
    "AnalysisRequest",
    "AnalysisResponse",
    "SimilarityRequest",
    "SimilarityResponse",
    "UploadRequest",
    "UploadResponse",
    "DownloadRequest",
    "DownloadResponse",
    # From stem.py
    "SearchResult",
    "StemResponse",
    "StemCreate",
    "StemUpdate",
    "StemSearchRequest",
    "StemUploadRequest",
    "StemBatchResponse",
    "StemAnalysisRequest",
    "StemAnalysisResponse",
]