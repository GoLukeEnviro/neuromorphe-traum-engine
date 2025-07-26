#!/usr/bin/env python3
"""
Neuromorphe Traum-Engine v2.0 - CLAP-Embeddings Verifikation
Überprüft, ob CLAP-Embeddings erfolgreich in der Datenbank gespeichert wurden.
"""

import sqlite3
import numpy as np
import logging

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Datenbank-Pfad
DB_PATH = "processed_database/stems.db"

def verify_clap_embeddings():
    """
    Überprüft die CLAP-Embeddings in der Datenbank.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Gesamtanzahl der Einträge
        cursor.execute("SELECT COUNT(*) FROM stems")
        total_count = cursor.fetchone()[0]
        logging.info(f"Gesamtanzahl der Stems: {total_count}")
        
        # Anzahl der Einträge mit CLAP-Embeddings
        cursor.execute("SELECT COUNT(*) FROM stems WHERE clap_embedding IS NOT NULL")
        embedding_count = cursor.fetchone()[0]
        logging.info(f"Stems mit CLAP-Embeddings: {embedding_count}")
        
        # Prozentsatz
        if total_count > 0:
            percentage = (embedding_count / total_count) * 100
            logging.info(f"Abdeckung: {percentage:.1f}%")
        
        # Beispiel-Embedding anzeigen
        cursor.execute("SELECT id, path, clap_embedding FROM stems WHERE clap_embedding IS NOT NULL LIMIT 1")
        result = cursor.fetchone()
        
        if result:
            stem_id, path, embedding_blob = result
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            logging.info(f"Beispiel-Embedding für {stem_id} ({path}):")
            logging.info(f"  Dimensionen: {embedding.shape}")
            logging.info(f"  Erste 5 Werte: {embedding[:5]}")
            logging.info(f"  Norm: {np.linalg.norm(embedding):.4f}")
        
        # Einträge ohne Embeddings
        cursor.execute("SELECT id, path FROM stems WHERE clap_embedding IS NULL LIMIT 5")
        missing = cursor.fetchall()
        
        if missing:
            logging.warning(f"Stems ohne CLAP-Embeddings ({len(missing)} gezeigt):")
            for stem_id, path in missing:
                logging.warning(f"  {stem_id}: {path}")
        else:
            logging.info("✅ Alle Stems haben CLAP-Embeddings!")
        
        conn.close()
        return embedding_count, total_count
        
    except Exception as e:
        logging.error(f"Fehler bei der Verifikation: {e}")
        return 0, 0

if __name__ == "__main__":
    logging.info("Starte CLAP-Embeddings Verifikation...")
    embedding_count, total_count = verify_clap_embeddings()
    
    if embedding_count == total_count and total_count > 0:
        logging.info("🎉 CLAP-Embeddings erfolgreich für alle Stems berechnet!")
    elif embedding_count > 0:
        logging.info(f"⚠️ CLAP-Embeddings für {embedding_count}/{total_count} Stems berechnet.")
    else:
        logging.error("❌ Keine CLAP-Embeddings gefunden.")