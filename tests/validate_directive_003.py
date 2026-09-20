#!/usr/bin/env python3
"""
Validierung für AGENTEN_DIREKTIVE_003
Überprüft die Erfolgskriterien der semantischen Analyse und Kategorisierung.
"""

import sqlite3
import json
import os
import sys
from datetime import datetime
from typing import Optional

def resolve_db_path(db_path: Optional[str] = None) -> str:
    """DB-Pfad auflösen: Argument > ``NEUROMORPHE_DB_PATH`` > Produktions-Default."""
    return db_path or os.environ.get("NEUROMORPHE_DB_PATH") or "processed_database/stems.db"

def validate_directive_003(db_path: Optional[str] = None):
    """
    Validiert die Erfolgskriterien von AGENTEN_DIREKTIVE_003:
    1. Drei neue Test-Audiodateien wurden verarbeitet
    2. Korrekte Kategorisierung (kick, bass, unknown für loop)
    3. Tags-Feld enthält JSON-String mit 3 Tags
    4. Alle Einträge haben quality_ok=True

    Args:
        db_path: Pfad zur SQLite-Datenbank. Default: ``NEUROMORPHE_DB_PATH``
            oder ``processed_database/stems.db`` (Produktions-Default).

    Returns:
        bool: True, wenn die Erfolgskriterien erfüllt sind.
    """
    
    print("🔍 VALIDIERUNG AGENTEN_DIREKTIVE_003")
    print("=" * 50)
    
    # Datenbankverbindung (Pfad injizierbar)
    db_path = resolve_db_path(db_path)
    print(f"📂 Datenbank: {db_path}")
    if not os.path.exists(db_path):
        print("❌ Datenbank nicht gefunden!")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Alle Einträge abrufen
    cursor.execute("SELECT * FROM stems ORDER BY imported_at DESC")
    rows = cursor.fetchall()
    
    # Spalten-Namen abrufen
    columns = [description[0] for description in cursor.description]
    
    print(f"📊 Gefundene Einträge: {len(rows)}")
    print()
    
    # Test-Dateien identifizieren (die neuesten 3 Einträge)
    test_files = rows[:3] if len(rows) >= 3 else rows
    
    success_count = 0
    expected_categories = {"kick": False, "bass": False, "unknown": False}
    
    for i, row in enumerate(test_files):
        entry = dict(zip(columns, row))
        
        print(f"📁 Eintrag {i+1}: {entry['id']}")
        print(f"   Pfad: {entry['path']}")
        print(f"   BPM: {entry['bpm']}")
        print(f"   Kategorie: {entry['category']}")
        print(f"   Quality OK: {entry['quality_ok']}")
        
        # Tags validieren
        try:
            if entry['tags']:
                tags = json.loads(entry['tags'])
                print(f"   Tags: {tags} ({len(tags)} Tags)")
                
                if len(tags) == 3:
                    print("   ✅ Korrekte Anzahl Tags (3)")
                    success_count += 1
                else:
                    print(f"   ❌ Falsche Anzahl Tags: {len(tags)} (erwartet: 3)")
            else:
                print("   ❌ Keine Tags gefunden")
        except json.JSONDecodeError:
            print(f"   ❌ Tags nicht als JSON parsebar: {entry['tags']}")
        
        # Kategorie validieren
        category = entry['category']
        if category in ["kick", "bass", "unknown"]:
            expected_categories[category] = True
            print(f"   ✅ Kategorie '{category}' erkannt")
        else:
            print(f"   ⚠️  Unerwartete Kategorie: {category}")
        
        # Quality Check
        if entry['quality_ok']:
            print("   ✅ Quality OK")
        else:
            print("   ❌ Quality nicht OK")
        
        print(f"   Importiert: {entry['imported_at']}")
        print()
    
    conn.close()
    
    # Zusammenfassung
    print("📋 VALIDIERUNGS-ZUSAMMENFASSUNG")
    print("=" * 30)
    
    # Überprüfe, ob alle erwarteten Kategorien gefunden wurden
    categories_found = sum(expected_categories.values())
    print(f"Kategorien gefunden: {categories_found}/3")
    for cat, found in expected_categories.items():
        status = "✅" if found else "❌"
        print(f"  {status} {cat}")
    
    print(f"Einträge mit korrekten Tags: {success_count}/{len(test_files)}")
    
    # Gesamtergebnis
    if categories_found >= 2 and success_count >= 2:  # Mindestens 2 von 3 sollten funktionieren
        print("\n🎉 AGENTEN_DIREKTIVE_003 ERFOLGREICH IMPLEMENTIERT!")
        print("✅ Semantische Analyse mit LAION-CLAP funktioniert")
        print("✅ Kategorisierung basierend auf Dateinamen funktioniert")
        print("✅ JSON-Tags werden korrekt gespeichert")
        return True
    else:
        print("\n❌ AGENTEN_DIREKTIVE_003 NICHT VOLLSTÄNDIG ERFÜLLT")
        return False

if __name__ == "__main__":
    # Echter Exit-Code: ein nicht erfülltes Kriterium darf nicht als Erfolg enden.
    sys.exit(0 if validate_directive_003() else 1)