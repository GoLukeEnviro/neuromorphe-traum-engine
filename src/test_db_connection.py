#!/usr/bin/env python3
"""Test der Datenbankverbindung"""

from database.database import get_database_manager
from database.models import Stem
from core.logging import get_logger

logger = get_logger(__name__)

def test_db_connection():
    """Testet die grundlegende Datenbankverbindung"""
    try:
        db_manager = get_database_manager()
        logger.info("Database Manager erstellt")
        
        # Sync Session testen
        with db_manager.get_sync_session() as session:
            logger.info("Session erstellt")
            
            # Einfache Abfrage
            count = session.query(Stem).count()
            logger.info(f"Anzahl Stems in DB: {count}")
            
            # Neuen Stem direkt erstellen (ohne CRUD)
            new_stem = Stem(
                filename="direct_test.wav",
                original_path="/test/direct_test.wav",
                file_hash="direct123",
                duration=1.0,
                sample_rate=44100,
                channels=1,
                file_size=500000,
                processing_status="completed"
            )
            
            session.add(new_stem)
            session.commit()
            logger.info(f"Stem direkt erstellt: {new_stem.id}")
            
            # Prüfen ob erstellt
            found = session.query(Stem).filter(Stem.file_hash == "direct123").first()
            if found:
                logger.info(f"Stem gefunden: {found.filename}")
            else:
                logger.error("Stem nicht gefunden!")
                
    except Exception as e:
        logger.error(f"Datenbankfehler: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_db_connection()
    print("Datenbanktest abgeschlossen!")