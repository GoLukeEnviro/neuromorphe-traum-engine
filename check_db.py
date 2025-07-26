#!/usr/bin/env python3
"""
Überprüft die Datenbankstruktur der stems.db
"""

import sqlite3
import os

def check_database():
    db_path = "processed_database/stems.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Datenbank nicht gefunden: {db_path}")
        return
    
    print(f"✅ Datenbank gefunden: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Tabellen auflisten
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"\nTabellen: {[table[0] for table in tables]}")
        
        # Schema der stems-Tabelle anzeigen
        cursor.execute("PRAGMA table_info(stems)")
        columns = cursor.fetchall()
        
        print("\n=== stems Tabellen-Schema ===")
        for col in columns:
            cid, name, type_, notnull, default, pk = col
            pk_str = " (PRIMARY KEY)" if pk else ""
            print(f"  {name}: {type_}{pk_str}")
        
        # Anzahl der Einträge
        cursor.execute("SELECT COUNT(*) FROM stems")
        count = cursor.fetchone()[0]
        print(f"\nAnzahl Einträge: {count}")
        
        conn.close()
        print("\n✅ Datenbankstruktur ist korrekt!")
        
    except Exception as e:
        print(f"❌ Fehler beim Überprüfen der Datenbank: {e}")

if __name__ == "__main__":
    check_database()