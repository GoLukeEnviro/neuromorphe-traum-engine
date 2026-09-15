#!/usr/bin/env python3
"""Datenbank löschen und neu erstellen"""

import os
import asyncio
from pathlib import Path
from database.database import init_database

def recreate_database():
    """Löscht die alte Datenbank und erstellt eine neue"""
    db_path = Path("E:/VS-code-Projekte-5.2025/neuromorphe-traum-engine/processed_database/stems.db")
    
    # Datenbank löschen falls vorhanden
    if db_path.exists():
        try:
            os.remove(db_path)
            print("Alte Datenbank gelöscht.")
        except Exception as e:
            print(f"Fehler beim Löschen: {e}")
            return False
    
    # Neue Datenbank erstellen
    try:
        asyncio.run(init_database())
        print("Neue Datenbank erstellt.")
        return True
    except Exception as e:
        print(f"Fehler beim Erstellen: {e}")
        return False

if __name__ == "__main__":
    recreate_database()