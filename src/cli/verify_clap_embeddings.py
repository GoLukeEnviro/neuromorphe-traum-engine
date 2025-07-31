#!/usr/bin/env python3
"""
Neuromorphe Traum-Engine v2.0 - CLAP-Embeddings Verifikation
Überprüft, ob CLAP-Embeddings erfolgreich in der Datenbank gespeichert wurden.
"""

import numpy as np
import logging
import asyncio

from database.service import DatabaseService
from database.models import Stem

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def verify_clap_embeddings():
    """
    Überprüft die CLAP-Embeddings in der Datenbank.
    """
    db_service = DatabaseService()
    try:
        
        # Gesamtanzahl der Einträge
        total_count = await db_service.get_stem_count()
        logging.info(f"Gesamtanzahl der Stems: {total_count}")
        
        # Anzahl der Einträge mit CLAP-Embeddings
        # Da audio_embedding jetzt ein JSON-Feld ist, prüfen wir auf IS NOT NULL
        stems_with_embeddings = await db_service.get_all_stems(audio_embedding_is_not_null=True) # Annahme: get_all_stems kann nach Feld-Existenz filtern
        embedding_count = len(stems_with_embeddings)
        logging.info(f"Stems mit CLAP-Embeddings: {embedding_count}")
        
        # Prozentsatz
        if total_count > 0:
            percentage = (embedding_count / total_count) * 100
            logging.info(f"Abdeckung: {percentage:.1f}%")
        
        # Beispiel-Embedding anzeigen
        # Wir nehmen das erste Stem mit Embedding, falls vorhanden
        example_stem = next((s for s in stems_with_embeddings if s.audio_embedding), None)
        
        if example_stem:
            stem_id = example_stem.id
            path = example_stem.original_path or example_stem.processed_path
            embedding = np.array(example_stem.audio_embedding)
            logging.info(f"Beispiel-Embedding für {stem_id} ({path}):")
            logging.info(f"  Dimensionen: {embedding.shape}")
            logging.info(f"  Erste 5 Werte: {embedding[:5]}")
            logging.info(f"  Norm: {np.linalg.norm(embedding):.4f}")
        
        # Einträge ohne Embeddings
        stems_without_embeddings = await db_service.get_all_stems(audio_embedding_is_null=True) # Annahme: get_all_stems kann nach Feld-Nullheit filtern
        
        if stems_without_embeddings:
            logging.warning(f"Stems ohne CLAP-Embeddings ({len(stems_without_embeddings)} gezeigt):")
            for stem in stems_without_embeddings[:5]: # Zeige max. 5 Beispiele
                logging.warning(f"  {stem.id}: {stem.original_path or stem.filename}")
        else:
            logging.info("✅ Alle Stems haben CLAP-Embeddings!")
        
        return embedding_count, total_count
        
    except Exception as e:
        logging.error(f"Fehler bei der Verifikation: {e}")
        return 0, 0

if __name__ == "__main__":
    logging.info("Starte CLAP-Embeddings Verifikation...")
    embedding_count, total_count = asyncio.run(verify_clap_embeddings())
    
    if embedding_count == total_count and total_count > 0:
        logging.info("🎉 CLAP-Embeddings erfolgreich für alle Stems berechnet!")
    elif embedding_count > 0:
        logging.info(f"⚠️ CLAP-Embeddings für {embedding_count}/{total_count} Stems berechnet.")
    else:
        logging.error("❌ Keine CLAP-Embeddings gefunden.")