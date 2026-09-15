#!/usr/bin/env python3
"""Datenbank-Migration: key -> musical_key"""

import sqlite3
from pathlib import Path

def migrate_database():
    """Migriert die Datenbank von key zu musical_key"""
    db_path = Path("E:/VS-code-Projekte-5.2025/neuromorphe-traum-engine/processed_database/stems.db")
    
    if not db_path.exists():
        print("Datenbank existiert nicht!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Prüfen ob key-Spalte existiert
        cursor.execute("PRAGMA table_info(stems);")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'key' in column_names and 'musical_key' not in column_names:
            print("Migriere key -> musical_key...")
            cursor.execute("ALTER TABLE stems RENAME COLUMN key TO musical_key;")
            conn.commit()
            print("Migration erfolgreich!")
        elif 'musical_key' in column_names:
            print("musical_key-Spalte existiert bereits.")
        else:
            print("Weder key noch musical_key gefunden.")
            
    except Exception as e:
        print(f"Fehler bei Migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()