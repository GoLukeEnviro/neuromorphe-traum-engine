#!/usr/bin/env python3
"""Test des Stem-Retrieval-Systems"""

import asyncio
from core.logging import setup_logging, get_logger
from database.database import get_database_manager
from database.crud import StemCRUD

logger = get_logger(__name__)

def test_retrieval():
    """Testet das Retrieval-System mit synchroner Session"""
    
    try:
        # Datenbank-Manager erstellen (bereits initialisiert)
        db_manager = get_database_manager()
        
        print("\n=== Stem Retrieval Test ===")
        
        # Synchrone Session verwenden für Statistiken
        with db_manager.get_sync_session() as session:
            # Anzahl der Stems abrufen
            total_stems = StemCRUD.get_stem_count(session)
            print(f"Anzahl der Stems in der Datenbank: {total_stems}")
            
            if total_stems == 0:
                print("\nKeine Stems in der Datenbank gefunden.")
                print("Bitte führen Sie zuerst das Preprocessing aus, um Stems zu erstellen.")
                return False
            
            # Stems mit Audio-Embeddings zählen
            stems_with_embeddings = StemCRUD.get_stems(
                session, 
                audio_embedding_is_not_null=True
            )
            print(f"Stems mit Audio-Embeddings: {len(stems_with_embeddings)}")
            
            if len(stems_with_embeddings) == 0:
                print("\nKeine Stems mit Audio-Embeddings gefunden.")
                print("Die semantische Suche benötigt Audio-Embeddings.")
                print("Bitte führen Sie das CLAP-Embedding-Processing aus.")
                
                # Zeige verfügbare Stems
                print("\nVerfügbare Stems (ohne Embeddings):")
                all_stems = StemCRUD.get_stems(session, limit=10)
                for i, stem in enumerate(all_stems, 1):
                    print(f"  {i}. {stem.filename} - {stem.category} - {stem.bpm} BPM")
                
                return False
            
            # Zeige Stems mit Embeddings
            print("\nStems mit Audio-Embeddings:")
            for i, stem in enumerate(stems_with_embeddings, 1):
                print(f"  {i}. {stem.filename} - {stem.category} - {stem.bpm} BPM")
        
        print("\n=== Retrieval-Test abgeschlossen ===")
        return True
        
    except Exception as e:
        logger.error(f"Fehler beim Retrieval-Test: {e}")
        print(f"Fehler: {e}")
        return False

if __name__ == "__main__":
    # Logging initialisieren
    setup_logging()
    
    # Test ausführen
    success = test_retrieval()
    exit(0 if success else 1)