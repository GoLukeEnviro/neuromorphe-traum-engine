#!/usr/bin/env python3
"""Datenbankschema überprüfen"""

import sqlite3
from pathlib import Path

def check_schema():
    """Überprüft das Datenbankschema"""
    db_path = Path("E:/VS-code-Projekte-5.2025/neuromorphe-traum-engine/processed_database/stems.db")
    
    if not db_path.exists():
        print("Datenbank existiert nicht!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tabellen auflisten
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tabellen in der Datenbank:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Schema der stems-Tabelle
    if ('stems',) in tables:
        print("\nSchema der stems-Tabelle:")
        cursor.execute("PRAGMA table_info(stems);")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} ({col[2]}) - NOT NULL: {bool(col[3])} - DEFAULT: {col[4]}")
    else:
        print("stems-Tabelle existiert nicht!")
    
    conn.close()

if __name__ == "__main__":
    check_schema()