#!/usr/bin/env python3
"""
Script to check uploaded audio files in the database
"""

import sys
import os
sys.path.append('src')

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def check_audio_files():
    """Check audio files in database"""
    db_path = "processed_database/stems_new.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Datenbank nicht gefunden: {db_path}")
        return
    
    engine = create_engine(f'sqlite:///{db_path}')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if audio_files table exists
        result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='audio_files'"))
        if not result.fetchone():
            print("❌ Tabelle 'audio_files' existiert nicht")
            return
        
        # Get all audio files
        result = session.execute(text("SELECT COUNT(*) FROM audio_files"))
        count = result.fetchone()[0]
        print(f"📊 Gefundene Audio-Dateien in der Datenbank: {count}")
        
        if count > 0:
            # Get details of first 10 files
            result = session.execute(text("""
                SELECT id, original_filename, processing_status, file_path, created_at 
                FROM audio_files 
                ORDER BY created_at DESC 
                LIMIT 10
            """))
            
            print("\n📋 Letzte 10 hochgeladene Dateien:")
            for row in result:
                print(f"  • ID: {row[0]}, Datei: {row[1]}, Status: {row[2]}")
                print(f"    Pfad: {row[3]}")
                print(f"    Erstellt: {row[4]}")
                print()
        
    except Exception as e:
        print(f"❌ Fehler beim Abfragen der Datenbank: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    check_audio_files()