#!/usr/bin/env python3
"""Erstellt eine frische Datenbank mit neuem Namen"""

import os
import asyncio
from pathlib import Path
from core.config import settings

# Temporär die Datenbank-URL ändern
original_db_url = settings.DATABASE_URL
new_db_path = Path("E:/VS-code-Projekte-5.2025/neuromorphe-traum-engine/processed_database/stems_new.db")
settings.DATABASE_URL = f"sqlite:///{new_db_path}"

from database.database import init_database

async def create_fresh_database():
    """Erstellt eine frische Datenbank"""
    try:
        # Neue Datenbank erstellen
        await init_database()
        print(f"Neue Datenbank erstellt: {new_db_path}")
        
        # Alte Datenbank-Pfade
        old_db_path = Path("E:/VS-code-Projekte-5.2025/neuromorphe-traum-engine/processed_database/stems.db")
        
        # Alte Datenbank löschen falls möglich
        if old_db_path.exists():
            try:
                os.remove(old_db_path)
                print("Alte Datenbank gelöscht.")
                
                # Neue Datenbank umbenennen
                os.rename(new_db_path, old_db_path)
                print("Datenbank umbenannt.")
            except Exception as e:
                print(f"Konnte alte Datenbank nicht ersetzen: {e}")
                print(f"Neue Datenbank verfügbar unter: {new_db_path}")
                # Konfiguration für neue Datenbank beibehalten
                return True
        else:
            # Keine alte Datenbank vorhanden, einfach umbenennen
            os.rename(new_db_path, old_db_path)
            print("Datenbank umbenannt.")
        
        return True
        
    except Exception as e:
        print(f"Fehler beim Erstellen: {e}")
        return False
    finally:
        # Ursprüngliche Konfiguration wiederherstellen
        settings.DATABASE_URL = original_db_url

if __name__ == "__main__":
    asyncio.run(create_fresh_database())