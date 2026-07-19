#!/usr/bin/env python3
"""
Überprüft die Pfade in der Datenbank, um zu verstehen, wie die Dateien gespeichert wurden.
"""

import asyncio
from pathlib import Path

from database.service import DatabaseService
from core.config import settings

async def check_database_paths():
    """
    Überprüft alle Pfade in der Datenbank und sucht nach Techno-Dateien.
    """
    print("=== DATENBANKPFAD-ANALYSE ===")
    print()
    
    db_service = DatabaseService()
    db_path = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
    
    if not db_path.exists():
        print("❌ Datenbank nicht gefunden!")
        return
    
    try:
    
    try:
        # Alle Pfade anzeigen
        print("1. ALLE PFADE IN DER DATENBANK:")
        all_stems = await db_service.get_all_stems(limit=20, order_by="created_at", order_desc=True)
        
        for i, stem in enumerate(all_stems, 1):
            filename = Path(stem.original_path).name if stem.original_path else stem.filename
            bpm_str = f"{stem.bpm:.1f} BPM" if stem.bpm is not None and stem.bpm > 0 else "No BPM"
            print(f"   {i:2d}. {filename} | {stem.category} | {bpm_str}")
            if i <= 5:  # Zeige vollständige Pfade für die ersten 5
                print(f"       Vollständiger Pfad: {stem.original_path}")
        print()
        
        # Suche nach verschiedenen Mustern
        print("2. SUCHE NACH TECHNO-DATEIEN:")
        
        patterns = [
            ('techno', 'techno'),
            ('LPE', 'LPE'),
            ('drt', 'drt'),
            ('130', '130 (BPM-Hinweis)'),
            ('LPE126', 'LPE126')
        ]
        
        for pattern, description in patterns:
            count = await db_service.get_stem_count(path_pattern=pattern)
            print(f"   {description}: {count} Dateien gefunden")
            
            if count > 0:
                examples = await db_service.search_stems_by_path_pattern(path_pattern=pattern, limit=3)
                for stem in examples:
                    filename = Path(stem.original_path).name if stem.original_path else stem.filename
                    print(f"      Beispiel: {filename}")
        print()
        
        # Neueste Einträge
        print("3. NEUESTE EINTRÄGE (letzte 10):")
        recent_entries = await db_service.get_all_stems(limit=10, order_by="created_at", order_desc=True)
        
        for stem in recent_entries:
            filename = Path(stem.original_path).name if stem.original_path else stem.filename
            bpm_str = f"{stem.bpm:.1f} BPM" if stem.bpm is not None and stem.bpm > 0 else "No BPM"
            print(f"   {filename} | {stem.category} | {bpm_str} | {stem.created_at}")
        print()
        
        # Statistiken
        print("4. GESAMTSTATISTIKEN:")
        total = await db_service.get_stem_count()
        
        # Annahme: Techno-BPM ist zwischen 120 und 140
        techno_bpm_stems = await db_service.get_all_stems(bpm_min=120, bpm_max=140)
        techno_bpm = len(techno_bpm_stems)
        
        # In letzter Stunde hinzugefügt (requires a method in DatabaseService to filter by time)
        # For now, I'll use a placeholder or skip this if no direct method exists.
        # If needed, a method like get_stems_added_since(datetime_obj) could be added to DatabaseService
        recent = "N/A" # Placeholder for now
        
        print(f"   Gesamtanzahl Dateien: {total}")
        print(f"   Dateien mit Techno-BPM (120-140): {techno_bpm}")
        print(f"   In letzter Stunde hinzugefügt: {recent}")
        
    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")

if __name__ == "__main__":
    asyncio.run(check_database_paths())