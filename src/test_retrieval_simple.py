#!/usr/bin/env python3
"""Einfacher Test der Retrieval-Funktionalität"""

import asyncio
from core.logging import setup_logging, get_logger
from database.database import get_database_manager
from database.crud import StemCRUD

logger = get_logger(__name__)

def test_simple_retrieval():
    """Einfacher Test der Retrieval-Funktionalität mit synchroner Session"""
    
    try:
        # Datenbank-Manager erstellen (bereits initialisiert)
        db_manager = get_database_manager()
        
        print("=== Einfacher Retrieval-Test ===")
        
        # Synchrone Session verwenden
        with db_manager.get_sync_session() as session:
            # Anzahl der Stems abrufen
            total_stems = StemCRUD.get_stem_count(session)
            print(f"Anzahl der Stems in der Datenbank: {total_stems}")
            
            if total_stems > 0:
                # Erste 10 Stems abrufen
                stems = StemCRUD.get_stems(session, limit=10)
                print(f"\nErste {len(stems)} Stems:")
                for i, stem in enumerate(stems, 1):
                     audio_embedding_status = "✓" if stem.audio_embedding is not None else "✗"
                     print(f"{i:2d}. {stem.filename} (ID: {stem.id})")
                     print(f"    Kategorie: {stem.category}, BPM: {stem.bpm}, Key: {stem.musical_key}")
                     print(f"    Audio Embedding: {audio_embedding_status}")
                     print()
            else:
                print("\nKeine Stems in der Datenbank gefunden.")
        
        print("=== Test erfolgreich abgeschlossen ===")
        
    except Exception as e:
        logger.error(f"Fehler beim Retrieval-Test: {e}")
        print(f"Fehler: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Logging initialisieren
    setup_logging()
    
    # Test ausführen
    success = test_simple_retrieval()
    exit(0 if success else 1)