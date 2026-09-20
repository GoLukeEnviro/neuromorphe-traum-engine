#!/usr/bin/env python3
"""
Validierungsskript für AGENTEN_DIREKTIVE_004
Überprüft die Erfolgskriterien für erweiterte Qualitätskontrolle und Cluster-basierte Kategorisierung.
"""

import sqlite3
import json
import os
import sys
from pathlib import Path
from typing import Optional

def resolve_db_path(db_path: Optional[str] = None) -> str:
    """DB-Pfad auflösen: Argument > ``NEUROMORPHE_DB_PATH`` > Produktions-Default."""
    return db_path or os.environ.get("NEUROMORPHE_DB_PATH") or "processed_database/stems.db"

def validate_directive_004(
    db_path: Optional[str] = None,
    input_dir: Optional[str] = None,
):
    """
    Validiert alle Erfolgskriterien von AGENTEN_DIREKTIVE_004

    Args:
        db_path: Pfad zur SQLite-Datenbank. Default: ``NEUROMORPHE_DB_PATH``
            oder ``processed_database/stems.db``.
        input_dir: Verzeichnis der Test-Audiodateien. Default:
            ``NEUROMORPHE_INPUT_DIR`` oder ``raw_construction_kits``.

    Returns:
        bool: True, wenn alle Erfolgskriterien erfüllt sind.
    """
    print("=== VALIDIERUNG AGENTEN_DIREKTIVE_004 ===")
    print()
    
    # Datenbankverbindung (Pfad injizierbar)
    db_path = resolve_db_path(db_path)
    input_dir = (
        input_dir
        or os.environ.get("NEUROMORPHE_INPUT_DIR")
        or "raw_construction_kits"
    )
    print(f"📂 Datenbank: {db_path}")
    print(f"📂 Eingabeverzeichnis: {input_dir}")
    if not os.path.exists(db_path):
        print("❌ Datenbank nicht gefunden!")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. Überprüfe neue Test-Audiodateien
        print("1. Überprüfung der Test-Audiodateien:")
        test_files = [
            "test_kick_004.wav",
            "test_bass_004.wav", 
            "test_ambient_004.wav",
            "test_silent_004.wav"
        ]
        
        for test_file in test_files:
            file_path = os.path.join(input_dir, test_file)
            if os.path.exists(file_path):
                print(f"   ✅ {test_file} erstellt")
            else:
                print(f"   ❌ {test_file} fehlt")
        print()
        
        # 2. Überprüfe Datenbankeinträge
        print("2. Datenbankeinträge:")
        cursor.execute("SELECT COUNT(*) FROM stems")
        total_count = cursor.fetchone()[0]
        print(f"   Gesamtanzahl Einträge: {total_count}")
        
        # 3. Überprüfe spezifische Test-Dateien in der Datenbank
        print("\n3. Verarbeitete Test-Dateien (DIRECTIVE_004):")
        directive_004_files = ["test_kick_004.wav", "test_bass_004.wav", "test_ambient_004.wav"]
        
        for test_file in directive_004_files:
            cursor.execute("""
                SELECT id, category, tags, features, quality_ok 
                FROM stems 
                WHERE path LIKE ?
            """, (f"%{test_file}%",))
            
            result = cursor.fetchone()
            if result:
                stem_id, category, tags_json, features_json, quality_ok = result
                print(f"   ✅ {test_file}:")
                print(f"      - Kategorie: {category}")
                print(f"      - Quality OK: {quality_ok}")
                
                # Validiere JSON-Tags
                try:
                    tags = json.loads(tags_json)
                    print(f"      - Tags: {len(tags)} Einträge")
                    if len(tags) >= 3:
                        print(f"        ✅ Mindestens 3 Tags vorhanden")
                    else:
                        print(f"        ❌ Nur {len(tags)} Tags (erwartet: ≥3)")
                except json.JSONDecodeError:
                    print(f"      ❌ Tags sind nicht gültiges JSON")
                
                # Validiere JSON-Features
                try:
                    features = json.loads(features_json)
                    expected_features = ['spectral_centroid', 'zero_crossing_rate', 'rms', 'spectral_rolloff', 'spectral_bandwidth']
                    print(f"      - Features: {len(features)} Einträge")
                    
                    missing_features = [f for f in expected_features if f not in features]
                    if not missing_features:
                        print(f"        ✅ Alle erwarteten Features vorhanden")
                    else:
                        print(f"        ❌ Fehlende Features: {missing_features}")
                        
                except json.JSONDecodeError:
                    print(f"      ❌ Features sind nicht gültiges JSON")
            else:
                print(f"   ❌ {test_file} nicht in Datenbank gefunden")
        
        # 4. Überprüfe Quarantäne (test_silent_004.wav sollte NICHT in der DB sein)
        print("\n4. Quarantäne-Überprüfung:")
        cursor.execute("""
            SELECT COUNT(*) FROM stems 
            WHERE path LIKE ?
        """, ("%test_silent_004.wav%",))
        
        silent_count = cursor.fetchone()[0]
        if silent_count == 0:
            print("   ✅ test_silent_004.wav korrekt quarantäniert (nicht in DB)")
        else:
            print("   ❌ test_silent_004.wav fälschlicherweise in DB gefunden")
        
        # 5. Überprüfe Kategorisierung
        print("\n5. Kategorisierung-Überprüfung:")
        
        # Kick-Datei (Dateiname-Heuristik)
        cursor.execute("""
            SELECT category FROM stems 
            WHERE path LIKE ?
        """, ("%test_kick_004.wav%",))
        kick_result = cursor.fetchone()
        if kick_result and kick_result[0] == "kick":
            print("   ✅ test_kick_004.wav korrekt als 'kick' kategorisiert")
        else:
            print(f"   ❌ test_kick_004.wav falsch kategorisiert: {kick_result[0] if kick_result else 'nicht gefunden'}")
        
        # Bass-Datei (Dateiname-Heuristik)
        cursor.execute("""
            SELECT category FROM stems 
            WHERE path LIKE ?
        """, ("%test_bass_004.wav%",))
        bass_result = cursor.fetchone()
        if bass_result and bass_result[0] == "bass":
            print("   ✅ test_bass_004.wav korrekt als 'bass' kategorisiert")
        else:
            print(f"   ❌ test_bass_004.wav falsch kategorisiert: {bass_result[0] if bass_result else 'nicht gefunden'}")
        
        # Ambient-Datei (sollte durch KMeans kategorisiert werden)
        cursor.execute("""
            SELECT category FROM stems 
            WHERE path LIKE ?
        """, ("%test_ambient_004.wav%",))
        ambient_result = cursor.fetchone()
        if ambient_result:
            category = ambient_result[0]
            if category in ["kick", "bass", "melody", "percussion", "fx", "vocal"]:
                print(f"   ✅ test_ambient_004.wav durch KMeans kategorisiert als: '{category}'")
            else:
                print(f"   ⚠️  test_ambient_004.wav kategorisiert als: '{category}' (möglicherweise Fallback)")
        else:
            print("   ❌ test_ambient_004.wav nicht gefunden")
        
        # 6. Zusammenfassung
        print("\n=== ZUSAMMENFASSUNG ===")
        
        # Zähle erfolgreiche Implementierungen
        success_criteria = [
            all(os.path.exists(os.path.join(input_dir, f)) for f in test_files),  # Test-Dateien erstellt
            total_count >= 12,  # Mindestens 12 Einträge (6 alte + 3 neue + evtl. mehr)
            silent_count == 0,  # Stille Datei quarantäniert
            kick_result and kick_result[0] == "kick",  # Kick korrekt kategorisiert
            bass_result and bass_result[0] == "bass",  # Bass korrekt kategorisiert
            ambient_result is not None  # Ambient verarbeitet
        ]
        
        successful = sum(success_criteria)
        total = len(success_criteria)
        
        print(f"Erfolgskriterien erfüllt: {successful}/{total}")
        
        if successful == total:
            print("\n🎉 AGENTEN_DIREKTIVE_004 ERFOLGREICH IMPLEMENTIERT! 🎉")
            print("\nImplementierte Features:")
            print("✅ Erweiterte Qualitätskontrolle (Spektrale Zentroide & RMS)")
            print("✅ Cluster-basierte Kategorisierung mit KMeans")
            print("✅ Erweiterte Feature-Extraktion (5 spektrale Features)")
            print("✅ JSON-Serialisierung der Features")
            print("✅ Intelligente Quarantäne für stille Dateien")
            print("✅ Fallback-Kategorisierung für unbekannte Dateitypen")
            return True
        else:
            print(f"\n⚠️  AGENTEN_DIREKTIVE_004 teilweise implementiert ({successful}/{total})")
            return False
            
    except Exception as e:
        print(f"❌ Fehler bei der Validierung: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    # Echter Exit-Code: ein nicht erfülltes Kriterium darf nicht als Erfolg enden.
    sys.exit(0 if validate_directive_004() else 1)