#!/usr/bin/env python3
"""Minimales Test-Skript für Stems"""

import asyncio
from database.database import get_database_manager
from database.crud import StemCRUD
from core.logging import get_logger

logger = get_logger(__name__)

async def create_minimal_test_stems():
    """Minimale Test-Stems erstellen"""
    db_manager = get_database_manager()
    
    # Nur die absolut notwendigen Felder
    test_stem = {
        "filename": "test_kick.wav",
        "original_path": "/test/test_kick.wav",
        "file_hash": "test123hash",
        "duration": 2.0,
        "sample_rate": 44100,
        "channels": 2,
        "file_size": 1000000,
        "processing_status": "completed"
    }
    
    try:
        # Sync Session für CRUD-Operationen
        with db_manager.get_sync_session() as session:
            # Prüfen ob bereits vorhanden
            existing = StemCRUD.get_stem_by_hash(session, test_stem["file_hash"])
            if existing:
                logger.info(f"Stem bereits vorhanden: {existing.filename}")
                return
            
            # Neuen Stem erstellen
            new_stem = StemCRUD.create_stem(session, test_stem)
            logger.info(f"Test-Stem erstellt: {new_stem.filename} (ID: {new_stem.id})")
            
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Test-Stems: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(create_minimal_test_stems())
    print("Minimaler Test-Stem erfolgreich erstellt!")