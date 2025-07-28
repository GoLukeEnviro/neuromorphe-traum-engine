#!/usr/bin/env python3
"""
Überprüft die Pfade in der Datenbank, um zu verstehen, wie die Dateien gespeichert wurden.
"""

import sqlite3
import os
from pathlib import Path

def check_database_paths():
    """
    Überprüft alle Pfade in der Datenbank und sucht nach Techno-Dateien.
    """
    print("=== DATENBANKPFAD-ANALYSE ===")
    print()
    
    # Datenbankverbindung
    db_path = "processed_database/stems.db"
    if not os.path.exists(db_path):
        print("❌ Datenbank nicht gefunden!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Alle Pfade anzeigen
        print("1. ALLE PFADE IN DER DATENBANK:")
        cursor.execute("SELECT path, bpm, category FROM stems ORDER BY imported_at DESC LIMIT 20")
        all_paths = cursor.fetchall()
        
        for i, (path, bpm, category) in enumerate(all_paths, 1):
            filename = Path(path).name
            bpm_str = f"{bpm:.1f} BPM" if bpm > 0 else "No BPM"
            print(f"   {i:2d}. {filename} | {category} | {bpm_str}")
            if i <= 5:  # Zeige vollständige Pfade für die ersten 5
                print(f"       Vollständiger Pfad: {path}")
        print()
        
        # Suche nach verschiedenen Mustern
        print("2. SUCHE NACH TECHNO-DATEIEN:")
        
        patterns = [
            ('%techno%', 'techno'),
            ('%LPE%', 'LPE'),
            ('%drt%', 'drt'),
            ('%130%', '130 (BPM-Hinweis)'),
            ('%LPE126%', 'LPE126')
        ]
        
        for pattern, description in patterns:
            cursor.execute("SELECT COUNT(*) FROM stems WHERE path LIKE ?", (pattern,))
            count = cursor.fetchone()[0]
            print(f"   {description}: {count} Dateien gefunden")
            
            if count > 0:
                cursor.execute("SELECT path FROM stems WHERE path LIKE ? LIMIT 3", (pattern,))
                examples = cursor.fetchall()
                for path, in examples:
                    print(f"      Beispiel: {Path(path).name}")
        print()
        
        # Neueste Einträge
        print("3. NEUESTE EINTRÄGE (letzte 10):")
        cursor.execute("""
            SELECT path, bpm, category, imported_at FROM stems 
            ORDER BY imported_at DESC 
            LIMIT 10
        """)
        
        recent_entries = cursor.fetchall()
        for path, bpm, category, imported_at in recent_entries:
            filename = Path(path).name
            bpm_str = f"{bpm:.1f} BPM" if bpm > 0 else "No BPM"
            print(f"   {filename} | {category} | {bpm_str} | {imported_at}")
        print()
        
        # Statistiken
        print("4. GESAMTSTATISTIKEN:")
        cursor.execute("SELECT COUNT(*) FROM stems")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stems WHERE bpm > 120 AND bpm < 140")
        techno_bpm = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stems WHERE imported_at > datetime('now', '-1 hour')")
        recent = cursor.fetchone()[0]
        
        print(f"   Gesamtanzahl Dateien: {total}")
        print(f"   Dateien mit Techno-BPM (120-140): {techno_bpm}")
        print(f"   In letzter Stunde hinzugefügt: {recent}")
        
    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_database_paths()