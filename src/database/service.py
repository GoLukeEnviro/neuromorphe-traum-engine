"""Datenbank-Service für die Neuromorphe Traum-Engine.

Diese Schicht vermittelt zwischen der async FastAPI-Welt und der synchronen
CRUD-Schicht (``database.crud``). Da ``StemCRUD`` & Co. klassisches
``db.query()`` mit einer ``Session`` verwenden, werden die synchronen
Datenbankoperationen über ``run_in_executor`` in einem Worker-Thread
ausgeführt und dürfen deshalb keine ``AsyncSession`` erhalten.
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select

from database.database import get_database_manager
from database.crud import StemCRUD, GeneratedTrackCRUD, ProcessingJobCRUD, SystemMetricsCRUD, ConfigurationCRUD
from database.models import Stem
from schemas.stem import StemCreate
from schemas.track import GeneratedTrackCreate
from schemas.job import ProcessingJobCreate
from schemas.metric import SystemMetricCreate
from schemas.config import ConfigurationSettingBase


class DatabaseService:
    """Service for database operations using SQLAlchemy"""

    def __init__(self):
        # No direct sqlite3 connection here, rely on SQLAlchemy session
        pass

    async def _run_sync(self, func, *args, **kwargs):
        """Führt eine synchrone CRUD-Operation in einem Worker-Thread aus.

        Die CRUD-Schicht arbeitet mit einer synchronen ``Session``. Wir öffnen
        sie im Thread und schließen sie dort wieder, damit keine Session über
        Thread- oder Event-Loop-Grenzen wandert.
        """
        def _work():
            with get_database_manager().get_sync_session() as session:
                return func(session, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(None, _work)

    # New methods for stems table
    async def insert_stem(self, stem_data: StemCreate) -> Optional[Stem]:
        """Insert new stem record using SQLAlchemy"""
        return await self._run_sync(StemCRUD.create_stem, stem_data.model_dump())

    async def get_stem_by_hash(self, file_hash: str) -> Optional[Stem]:
        """Holt einen Stem anhand seines Hashes."""
        return await self._run_sync(StemCRUD.get_stem_by_hash, file_hash)

    async def get_stems_by_category(self, category: str, source: Optional[str] = None, limit: int = 50) -> List[Stem]:
        """Get stems by category and optionally by source using SQLAlchemy"""
        return await self._run_sync(StemCRUD.get_stems, category=category, limit=limit)

    async def get_stems_by_source(self, source: str, limit: int = 50) -> List[Stem]:
        """Get all stems by source (original, separated, generated) using SQLAlchemy"""
        return await self._run_sync(StemCRUD.get_stems, limit=limit)

    async def get_all_stems(self,
                           category: Optional[str] = None,
                           source: Optional[str] = None,
                           limit: Optional[int] = None,
                           skip: Optional[int] = None,
                           audio_embedding_is_not_null: Optional[bool] = None,
                           audio_embedding_is_null: Optional[bool] = None) -> List[Stem]:
        """Get all stems with optional filters using SQLAlchemy"""
        return await self._run_sync(
            StemCRUD.get_stems,
            skip=skip or 0,
            limit=limit or 100,
            category=category,
            audio_embedding_is_not_null=audio_embedding_is_not_null,
            audio_embedding_is_null=audio_embedding_is_null,
        )

    async def get_stem_by_id(self, stem_id: int) -> Optional[Stem]:
        """Get single stem by ID using SQLAlchemy"""
        return await self._run_sync(StemCRUD.get_stem_by_id, stem_id)

    async def update_stem_processing_status(self, stem_id: int, status: str, error_message: Optional[str] = None) -> Optional[Stem]:
        """Update processing status of a stem using SQLAlchemy"""
        update_data = {"processing_status": status}
        if error_message:
            update_data["processing_error"] = error_message
        return await self._run_sync(StemCRUD.update_stem, stem_id, update_data)

    async def get_stem_categories(self) -> List[str]:
        """Get list of all stem categories using SQLAlchemy"""
        def _query(session):
            result = session.execute(
                text("SELECT DISTINCT category FROM stems WHERE category IS NOT NULL ORDER BY category")
            )
            return [row[0] for row in result.fetchall()]

        return await self._run_sync(_query)

    async def get_stem_statistics(self) -> Dict[str, Any]:
        """Get stem statistics by source and category using SQLAlchemy"""
        return await self._run_sync(StemCRUD.get_stem_statistics)

    async def get_stem_count(self, path_pattern: Optional[str] = None) -> int:
        """Holt die Anzahl der Stems, optional gefiltert nach Pfadmuster."""
        return await self._run_sync(StemCRUD.get_stem_count, path_pattern)

    async def search_stems_by_path_pattern(self, path_pattern: str, limit: int = 50) -> List[Stem]:
        """Sucht Stems basierend auf einem Pfadmuster."""
        return await self._run_sync(StemCRUD.search_stems_by_path_pattern, path_pattern, limit)

    # ------------------------------------------------------------------
    # Adapter für die Such-Schicht (src/search/service.py)
    # Die Suche erwartet Stem-Informationen als Dicts inkl. Embedding.
    # ------------------------------------------------------------------

    @staticmethod
    def _stem_to_file_info(stem: Stem) -> Dict[str, Any]:
        """Wandelt ein Stem-Modell in das von der Suche erwartete Dict um."""
        return {
            "id": stem.id,
            "filename": stem.filename,
            "category": stem.category,
            "bpm": stem.bpm,
            "duration": stem.duration,
            "embedding": stem.audio_embedding,
        }

    async def get_audio_files(
        self,
        category: Optional[str] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Liefert Stem-Infos inkl. Embedding als Dicts für die Suche.

        Es werden nur Stems mit vorhandenem Embedding zurückgegeben — ohne
        Vektor ist keine Ähnlichkeitsberechnung möglich.
        """
        stems = await self._run_sync(
            StemCRUD.get_stems,
            skip=0,
            limit=limit,
            category=category,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            audio_embedding_is_not_null=True,
        )
        return [self._stem_to_file_info(s) for s in stems]

    async def get_audio_files_by_ids(self, file_ids: List[Any]) -> List[Dict[str, Any]]:
        """Liefert Stem-Infos für eine explizite Liste von IDs."""
        result: List[Dict[str, Any]] = []
        for stem_id in file_ids:
            stem = await self.get_stem_by_id(stem_id)
            if stem is not None and stem.audio_embedding:
                result.append(self._stem_to_file_info(stem))
        return result

    async def get_search_statistics(self) -> Dict[str, Any]:
        """Statistiken für die Such-API auf Basis der Stem-Tabelle."""
        raw = await self._run_sync(StemCRUD.get_stem_statistics)
        categories = await self._run_sync(StemCRUD.get_stems)

        category_dist = raw.get("category_distribution", {}) or {}
        return {
            "total_files": raw.get("total_stems", 0) or 0,
            "categories": [
                {"category": name, "count": count}
                for name, count in sorted(category_dist.items())
                if name is not None
            ],
            "bpm_range": None,
            "last_updated": datetime.utcnow(),
            "_stem_rows": len(categories),
        }

    def cleanup(self) -> None:
        """Ressourcen freigeben (Symmetrie zu den anderen Services)."""
