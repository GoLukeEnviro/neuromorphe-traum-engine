#!/usr/bin/env python3
"""Test-Stems für die Neuromorphe Traum-Engine erstellen"""

import asyncio
from pathlib import Path
from database.database import get_database_manager
from database.crud import StemCRUD
from schemas.stem import StemCreate
from core.logging import get_logger

logger = get_logger(__name__)

async def create_test_stems():
    """Test-Stems in der Datenbank erstellen"""
    db_manager = get_database_manager()
    
    # Test-Stems definieren (nur Felder, die im Datenbankmodell existieren)
    test_stems = [
        {
            "filename": "kick_001.wav",
            "original_path": "/test_data/kicks/kick_001.wav",
            "file_hash": "abc123def456",
            "category": "kick",
            "source": "generated",
            "bpm": 128.0,
            "duration": 2.5,
            "file_size": 1024000,
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 1,
            "musical_key": "C"
        },
        {
            "filename": "kick_002.wav",
            "original_path": "/test_data/kicks/kick_002.wav",
            "file_hash": "def456ghi789",
            "category": "kick",
            "source": "generated",
            "bpm": 130.0,
            "duration": 2.2,
            "file_size": 980000,
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 1
        },
        {
            "filename": "snare_001.wav",
            "original_path": "/test_data/snares/snare_001.wav",
            "file_hash": "ghi789jkl012",
            "category": "snare",
            "source": "generated",
            "bpm": 128.0,
            "duration": 1.8,
            "file_size": 720000,
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 1,
            "musical_key": "Am"
        },
        {
            "filename": "hihat_001.wav",
            "original_path": "/test_data/hihats/hihat_001.wav",
            "file_hash": "jkl012mno345",
            "category": "hihat",
            "source": "generated",
            "bpm": 128.0,
            "duration": 0.5,
            "file_size": 200000,
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 1,
            "musical_key": "G"
        },
        {
            "filename": "bass_001.wav",
            "original_path": "/test_data/bass/bass_001.wav",
            "file_hash": "mno345pqr678",
            "category": "bass",
            "source": "generated",
            "bpm": 128.0,
            "duration": 4.0,
            "file_size": 1600000,
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 1,
            "musical_key": "Em"
        }
    ]
    
    try:
        # Sync Session für CRUD-Operationen
        with db_manager.get_sync_session() as session:
            for stem_data in test_stems:
                # Prüfen ob Stem bereits existiert
                existing_stem = StemCRUD.get_stem_by_hash(session, stem_data["file_hash"])
                if existing_stem:
                    logger.info(f"Stem {stem_data['filename']} already exists, skipping")
                    continue
                
                # Stem erstellen
                created_stem = StemCRUD.create_stem(session, stem_data)
                logger.info(f"Created test stem: {created_stem.filename} (ID: {created_stem.id})")
            
            session.commit()
            logger.info(f"Successfully created {len(test_stems)} test stems")
            
    except Exception as e:
        logger.error(f"Error creating test stems: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(create_test_stems())