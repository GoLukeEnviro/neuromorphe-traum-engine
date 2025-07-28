import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_async_db_session
from .crud import StemCRUD, GeneratedTrackCRUD, ProcessingJobCRUD, SystemMetricsCRUD, ConfigurationCRUD
from .models import Stem
from ..schemas.schemas import StemCreate
from ..schemas.schemas import GeneratedTrackCreate
from ..schemas.schemas import ProcessingJobCreate
from ..schemas.schemas import SystemMetricCreate
from ..schemas.schemas import ConfigurationSettingCreate

class DatabaseService:
    """Service for database operations using SQLAlchemy"""
    
    def __init__(self):
        # No direct sqlite3 connection here, rely on SQLAlchemy session
        pass
    
    
    
    # New methods for stems table
    async def insert_stem(self, stem_data: StemCreate) -> Optional[Stem]:
        """Insert new stem record using SQLAlchemy"""
        async with get_async_db_session() as session:
            return StemCRUD.create_stem(session, stem_data.dict())
    
    async def get_stems_by_category(self, category: str, source: Optional[str] = None, limit: int = 50) -> List[Stem]:
        """Get stems by category and optionally by source using SQLAlchemy"""
        async with get_async_db_session() as session:
            return StemCRUD.get_stems(session, category=category, source=source, limit=limit)
    
    async def get_stems_by_source(self, source: str, limit: int = 50) -> List[Stem]:
        """Get all stems by source (original, separated, generated) using SQLAlchemy"""
        async with get_async_db_session() as session:
            return StemCRUD.get_stems(session, source=source, limit=limit)
    
    async def get_all_stems(self, 
                           category: Optional[str] = None,
                           source: Optional[str] = None,
                           limit: Optional[int] = None,
                           skip: Optional[int] = None,
                           audio_embedding_is_not_null: Optional[bool] = None,
                           audio_embedding_is_null: Optional[bool] = None) -> List[Stem]:
        """Get all stems with optional filters using SQLAlchemy"""
        async with get_async_db_session() as session:
            return StemCRUD.get_stems(session, category=category, source=source, limit=limit, skip=skip, audio_embedding_is_not_null=audio_embedding_is_not_null, audio_embedding_is_null=audio_embedding_is_null)
    
    async def get_stem_by_id(self, stem_id: int) -> Optional[Stem]:
        """Get single stem by ID using SQLAlchemy"""
        async with get_async_db_session() as session:
            return StemCRUD.get_stem_by_id(session, stem_id)
    
    async def update_stem_processing_status(self, stem_id: int, status: str, error_message: Optional[str] = None) -> Optional[Stem]:
        """Update processing status of a stem using SQLAlchemy"""
        update_data = {"processing_status": status}
        if error_message:
            update_data["error_message"] = error_message
        async with get_async_db_session() as session:
            return StemCRUD.update_stem(session, stem_id, update_data)
    
    async def get_stem_categories(self) -> List[str]:
        """Get list of all stem categories using SQLAlchemy"""
        async with get_async_db_session() as session:
            result = await session.execute(text("SELECT DISTINCT category FROM stems WHERE category IS NOT NULL ORDER BY category"))
            return [row[0] for row in result.fetchall()]
    
    async def get_stem_statistics(self) -> Dict[str, Any]:
        """Get stem statistics by source and category using SQLAlchemy"""
        async with get_async_db_session() as session:
            return StemCRUD.get_stem_statistics(session)

    async def get_stem_count(self, path_pattern: Optional[str] = None) -> int:
        """Holt die Anzahl der Stems, optional gefiltert nach Pfadmuster."""
        async with get_async_db_session() as session:
            return StemCRUD.get_stem_count(session, path_pattern)

    async def search_stems_by_path_pattern(self, path_pattern: str, limit: int = 50) -> List[Stem]:
        """Sucht Stems basierend auf einem Pfadmuster."""
        async with get_async_db_session() as session:
            return StemCRUD.search_stems_by_path_pattern(session, path_pattern, limit)