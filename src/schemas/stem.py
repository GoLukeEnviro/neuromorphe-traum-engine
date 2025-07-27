from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class StemBase(BaseModel):
    """Base schema for Stem"""
    path: str
    bpm: Optional[float] = None
    key: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None  # JSON string
    features: Optional[str] = None  # JSON string
    quality_ok: bool = True

class StemCreate(StemBase):
    """Schema for creating a new stem"""
    clap_embedding: Optional[bytes] = None

class StemUpdate(BaseModel):
    """Schema for updating a stem"""
    bpm: Optional[float] = None
    key: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    features: Optional[str] = None
    quality_ok: Optional[bool] = None
    clap_embedding: Optional[bytes] = None

class Stem(StemBase):
    """Schema for returning stem data"""
    id: int
    imported_at: datetime
    
    class Config:
        from_attributes = True

class StemList(BaseModel):
    """Schema for paginated stem list"""
    items: List[Stem]
    total: int
    page: int
    size: int
    pages: int

class SearchResult(BaseModel):
    """Schema for search results"""
    stem: Stem
    similarity_score: float