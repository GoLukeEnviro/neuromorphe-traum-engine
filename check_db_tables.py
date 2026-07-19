#!/usr/bin/env python3
"""
Script to check database tables
"""

import sqlite3
import os

def check_database():
    """Check database structure and content"""
    db_path = "processed_database/stems_new.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Datenbank nicht gefunden: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"📊 Verfügbare Tabellen: {len(tables)}")
        for table in tables:
            table_name = table[0]
            print(f"  • {table_name}")
            
            # Get row count for each table
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"    Anzahl Einträge: {count}")
            
            # Show table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"    Spalten: {[col[1] for col in columns]}")
            print()
        
    except Exception as e:
        print(f"❌ Fehler beim Abfragen der Datenbank: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_database()