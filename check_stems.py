#!/usr/bin/env python3
"""
Script to check processed stems
"""

import sqlite3
import os

def check_stems():
    """Check processed stems in database"""
    db_path = "processed_database/stems_new.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Datenbank nicht gefunden: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get stems table structure
        cursor.execute("PRAGMA table_info(stems)")
        columns = cursor.fetchall()
        print("📋 Spalten der stems-Tabelle:")
        for col in columns:
            print(f"  • {col[1]} ({col[2]})")
        print()
        
        # Get stems data (without embeddings)
        cursor.execute("""
            SELECT id, filename, category, bpm, duration, 
                   processing_status, created_at, updated_at
            FROM stems 
            ORDER BY created_at DESC
        """)
        
        rows = cursor.fetchall()
        print(f"📊 Gefundene Stems: {len(rows)}")
        print()
        
        for row in rows:
            print(f"🎵 ID: {row[0]}")
            print(f"   Datei: {row[1]}")
            print(f"   Kategorie: {row[2]}")
            print(f"   BPM: {row[3]}")
            print(f"   Dauer: {row[4]}s")
            print(f"   Status: {row[5]}")
            print(f"   Erstellt: {row[6]}")
            print(f"   Aktualisiert: {row[7]}")
            print()
        
        # Check if actual audio files exist
        print("📁 Überprüfung der Dateien im processed_database/stems Verzeichnis:")
        stems_dir = "processed_database/stems"
        if os.path.exists(stems_dir):
            files = os.listdir(stems_dir)
            wav_files = [f for f in files if f.endswith('.wav')]
            print(f"   Gefundene WAV-Dateien: {len(wav_files)}")
            for f in wav_files[:10]:  # Show first 10
                print(f"   • {f}")
        else:
            print(f"   ❌ Verzeichnis {stems_dir} existiert nicht")
        
    except Exception as e:
        print(f"❌ Fehler beim Abfragen der Datenbank: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_stems()