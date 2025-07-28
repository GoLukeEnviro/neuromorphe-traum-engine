#!/usr/bin/env python3
"""
Analyse der neuen Techno-Beats aus LPE126.BONUS-techno3
Zeigt detaillierte Informationen über Kategorisierung, BPM, Features und Tags.
"""

import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

from ...database.service import DatabaseService
from ...database.models import Stem

async def analyze_techno_beats():
    """
    Analysiert die verarbeiteten Techno-Beats und zeigt detaillierte Statistiken.
    """
    print("=== ANALYSE DER NEUEN TECHNO-BEATS ===")
    print()
    
    db_service = DatabaseService()
    
    try:
        # 1. Gesamtstatistiken
        print("1. GESAMTSTATISTIKEN:")
        total_count = await db_service.get_stem_count()
        print(f"   Gesamtanzahl Einträge in Datenbank: {total_count}")
        
        # Techno-Beats spezifisch
        techno_count = await db_service.get_stem_count(path_pattern='techno') + \
                       await db_service.get_stem_count(path_pattern='LPE') + \
                       await db_service.get_stem_count(path_pattern='drt')
        print(f"   Techno-Beats gefunden: {techno_count}")
        
        # Note: The original query for 'recent_count' was very specific to a filename pattern.
        # For a general refactoring, I'll assume 'drt130LPE126' is a specific pattern.
        recent_count = await db_service.get_stem_count(path_pattern='drt130LPE126')
        print(f"   Davon in letzter Stunde: {recent_count}")
        print()
        
        # 2. BPM-Analyse der Techno-Beats
        print("2. BPM-ANALYSE DER TECHNO-BEATS:")
        techno_stems_for_bpm = await db_service.search_stems_by_path_pattern(
            path_pattern='techno', limit=1000000)  # Large limit to get all
        techno_stems_for_bpm.extend(await db_service.search_stems_by_path_pattern(
            path_pattern='LPE', limit=1000000))
        techno_stems_for_bpm.extend(await db_service.search_stems_by_path_pattern(
            path_pattern='drt', limit=1000000))

        bpm_values = [stem.bpm for stem in techno_stems_for_bpm if stem.bpm is not None and stem.bpm > 0]
        
        if bpm_values:
            avg_bpm = sum(bpm_values) / len(bpm_values)
            min_bpm = min(bpm_values)
            max_bpm = max(bpm_values)
            print(f"   Durchschnittliche BPM: {avg_bpm:.1f}")
            print(f"   BPM-Bereich: {min_bpm:.1f} - {max_bpm:.1f}")
            print(f"   Anzahl mit erkannter BPM: {len(bpm_values)}/{techno_count}")
        else:
            print("   Keine BPM-Werte erkannt")
        print()
        
        # 3. Kategorisierung
        print("3. KATEGORISIERUNG:")
        techno_stems_all = await db_service.search_stems_by_path_pattern(path_pattern='techno', limit=1000000)
        techno_stems_all.extend(await db_service.search_stems_by_path_pattern(path_pattern='LPE', limit=1000000))
        techno_stems_all.extend(await db_service.search_stems_by_path_pattern(path_pattern='drt', limit=1000000))

        category_counts = Counter(stem.category for stem in techno_stems_all if stem.category)
        
        for category, count in category_counts.most_common():
            percentage = (count / techno_count) * 100
            print(f"   {category}: {count} Dateien ({percentage:.1f}%)")
        print()
        
        # 4. Qualitätskontrolle
        print("4. QUALITÄTSKONTROLLE:")
        quality_counts = Counter(stem.quality_score for stem in techno_stems_all if stem.quality_score is not None)
        
        for quality, count in quality_counts.most_common():
            status = "✅ Qualität OK" if quality else "❌ Quarantäne"
            percentage = (count / techno_count) * 100
            print(f"   {status}: {count} Dateien ({percentage:.1f}%)")
        print()
        
        # 5. Feature-Analyse (Beispiel für erste 5 Dateien)
        print("5. FEATURE-ANALYSE (Beispiele):")
        feature_examples = [stem for stem in techno_stems_all if stem.semantic_analysis is not None and stem.pattern_analysis is not None][:5]
        
        for i, stem in enumerate(feature_examples, 1):
            filename = Path(stem.original_path).name if stem.original_path else stem.filename
            print(f"   Datei {i}: {filename}")
            
            semantic = stem.semantic_analysis
            pattern = stem.pattern_analysis
            
            print(f"      Dominante Stimmung: {semantic.get('semantic_profile', {}).get('dominant_characteristics', {}).get('mood', 'N/A')}")
            print(f"      Rhythmische Komplexität: {pattern.get('rhythmic_patterns', {}).get('rhythm_complexity', 'N/A'):.3f}")
            print(f"      Spektrale Bewegung: {pattern.get('spectral_evolution', {}).get('spectral_movement', 'N/A'):.3f}")
            print()
        
        # 6. Tag-Analyse
        print("6. SEMANTIC TAG-ANALYSE:")
        tag_examples = [stem for stem in techno_stems_all if stem.auto_tags is not None][:3]
        all_tags = []
        
        for i, stem in enumerate(tag_examples, 1):
            try:
                tags = stem.auto_tags
                print(f"   Beispiel {i}: {tags}")
                all_tags.extend(tags)
            except Exception:
                print(f"   Beispiel {i}: Fehler beim Laden der Tags")
        
        if all_tags:
            tag_counter = Counter(all_tags)
            print(f"\n   Häufigste Tags:")
            for tag, count in tag_counter.most_common(10):
                print(f"      '{tag}': {count}x")
        print()
        
        # 7. Detaillierte Dateiliste
        print("7. VERARBEITETE TECHNO-DATEIEN:")
        for stem in techno_stems_all:
            filename = Path(stem.original_path).name if stem.original_path else stem.filename
            bpm_str = f"{stem.bpm:.1f} BPM" if stem.bpm is not None and stem.bpm > 0 else "No BPM"
            quality_str = "✅" if stem.quality_score is not None and stem.quality_score > 0.5 else "❌" # Assuming quality_score > 0.5 is OK
            print(f"   {quality_str} {filename} | {stem.category} | {bpm_str}")
        
        print(f"\n=== ZUSAMMENFASSUNG ===")
        print(f"🎵 {techno_count} Techno-Beats erfolgreich analysiert")
        print(f"🎯 Kategorisierung durch KMeans und Heuristik")
        print(f"🎼 BPM-Erkennung für Tempo-Matching")
        print(f"🔊 Spektrale Features für Klanganalyse")
        print(f"🏷️  Semantische Tags durch LAION-CLAP")
        print(f"✅ Qualitätskontrolle und intelligente Filterung")
        
    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")
    # No finally block needed as db_service handles sessions internally

if __name__ == "__main__":
    analyze_techno_beats()