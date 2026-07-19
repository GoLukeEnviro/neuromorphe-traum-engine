from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class SearchRequest(BaseModel):
    """Schema for semantic search request"""
    query: str = Field(..., min_length=1, max_length=500)
    category: Optional[str] = Field(None, max_length=100)
    bpm_min: Optional[int] = Field(None, ge=60, le=200)
    bpm_max: Optional[int] = Field(None, ge=60, le=200)
    limit: int = Field(10, ge=1, le=100)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "energetic techno beat with heavy bass",
                "category": "techno",
                "bpm_min": 120,
                "bpm_max": 140,
                "limit": 10
            }
        }
    )


class SearchResult(BaseModel):
    """Schema for individual search result"""
    id: str
    filename: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    category: Optional[str]
    bpm: Optional[int]
    duration: Optional[float]
    file_path: Optional[str]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "abc123",
                "filename": "techno_beat_128.wav",
                "similarity_score": 0.85,
                "category": "techno",
                "bpm": 128,
                "duration": 180.5,
                "file_path": "/audio_files/abc123.wav"
            }
        }
    )


class SearchResponse(BaseModel):
    """Schema for search response"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    filters_applied: dict
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "energetic techno beat",
                "results": [],
                "total_results": 5,
                "search_time": 0.15,
                "filters_applied": {
                    "category": "techno",
                    "bpm_range": [120, 140]
                }
            }
        }
    )


class SimilarityRequest(BaseModel):
    """Schema for audio-to-audio similarity request"""
    source_file_id: str
    target_file_ids: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=100)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_file_id": "abc123",
                "target_file_ids": ["def456", "ghi789"],
                "limit": 5
            }
        }
    )


class CategoryStats(BaseModel):
    """Schema for category statistics"""
    category: str
    count: int
    avg_bpm: Optional[float]
    avg_duration: Optional[float]


class SearchStats(BaseModel):
    """Schema for search statistics"""
    total_files: int
    total_embeddings: int
    categories: List[CategoryStats]
    bpm_range: Optional[dict]
    last_updated: datetime