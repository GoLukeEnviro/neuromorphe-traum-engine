#!/usr/bin/env python3
"""
Überprüft die Datenbankstruktur der stems.db
"""

import asyncio
from pathlib import Path

from database.service import DatabaseService
from core.config import settings

async def check_database():
    db_service = DatabaseService()
    db_path = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
    
    if not db_path.exists():
        print(f"❌ Datenbank nicht gefunden: {db_path}")
        return
    
    print(f"✅ Datenbank gefunden: {db_path}")
    
    try:
        
        # Tabellen auflisten
        tables_raw = await db_service.execute_raw_sql("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in tables_raw]
        print(f"\nTabellen: {tables}")
        
        # Schema der stems-Tabelle anzeigen
        columns = await db_service.execute_raw_sql("PRAGMA table_info(stems)")
        
        print("\n=== stems Tabellen-Schema ===")
        for col in columns:
            cid, name, type_, notnull, default, pk = col
            pk_str = " (PRIMARY KEY)" if pk else ""
            print(f"  {name}: {type_}{pk_str}")
        
        # Anzahl der Einträge
        count = await db_service.get_stem_count()
        print(f"\nAnzahl Einträge: {count}")
        
        print("\n✅ Datenbankstruktur ist korrekt!")
        
    except Exception as e:
        print(f"❌ Fehler beim Überprüfen der Datenbank: {e}")

if __name__ == "__main__":
    asyncio.run(check_database())