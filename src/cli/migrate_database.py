##!/usr/bin/env python3
"""
Neuromorphe Traum-Engine v2.0 - Datenbank-Migration
Fügt die clap_embedding Spalte zur bestehenden stems Tabelle hinzu.
"""

import sqlite3
import os
import logging
from pathlib import Path

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Datenbank-Pfad
DB_PATH = "processed_database/stems.db"

def migrate_database():
    """
    Führt die Datenbank-Migration durch, um die clap_embedding Spalte hinzuzufügen.
    """
    if not os.path.exists(DB_PATH):
        logging.error(f"Datenbank nicht gefunden: {DB_PATH}")
        return False
    
    try:
        # Backup der bestehenden Datenbank erstellen
        backup_path = DB_PATH + ".backup"
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        logging.info(f"Backup erstellt: {backup_path}")
        
        # Verbindung zur Datenbank
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Prüfen, ob die Spalte bereits existiert
        cursor.execute("PRAGMA table_info(stems)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'clap_embedding' in columns:
            logging.info("clap_embedding Spalte existiert bereits.")
            conn.close()
            return True
        
        # Neue Spalte hinzufügen
        logging.info("Füge clap_embedding Spalte hinzu...")
        cursor.execute("ALTER TABLE stems ADD COLUMN clap_embedding BLOB")
        
        # Änderungen speichern
        conn.commit()
        conn.close()
        
        logging.info("Datenbank-Migration erfolgreich abgeschlossen.")
        return True
        
    except Exception as e:
        logging.error(f"Fehler bei der Migration: {e}")
        return False

def verify_migration():
    """
    Überprüft, ob die Migration erfolgreich war.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Tabellen-Schema anzeigen
        cursor.execute("PRAGMA table_info(stems)")
        columns = cursor.fetchall()
        
        logging.info("Aktuelle Tabellen-Struktur:")
        for column in columns:
            logging.info(f"  {column[1]} ({column[2]})")
        
        # Anzahl der Einträge
        cursor.execute("SELECT COUNT(*) FROM stems")
        count = cursor.fetchone()[0]
        logging.info(f"Anzahl der Einträge: {count}")
        
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"Fehler bei der Verifikation: {e}")
        return False

if __name__ == "__main__":
    logging.info("Starte Datenbank-Migration...")
    
    if migrate_database():
        verify_migration()
        logging.info("Migration abgeschlossen. Sie können jetzt prepare_dataset_sql.py ausführen.")
    else:
        logging.error("Migration fehlgeschlagen.")