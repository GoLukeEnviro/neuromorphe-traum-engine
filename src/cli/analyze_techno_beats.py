#!/usr/bin/env python3
"""
Analyse der neuen Techno-Beats aus LPE126.BONUS-techno3
Zeigt detaillierte Informationen über Kategorisierung, BPM, Features und Tags.
"""

import sqlite3
import json
import os
from collections import Counter
from pathlib import Path

def analyze_techno_beats():
    """
    Analysiert die verarbeiteten Techno-Beats und zeigt detaillierte Statistiken.
    """
    print("=== ANALYSE DER NEUEN TECHNO-BEATS ===")
    print()
    
    # Datenbankverbindung
    db_path = "processed_database/stems.db"
    if not os.path.exists(db_path):
        print("❌ Datenbank nicht gefunden!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. Gesamtstatistiken
        print("1. GESAMTSTATISTIKEN:")
        cursor.execute("SELECT COUNT(*) FROM stems")
        total_count = cursor.fetchone()[0]
        print(f"   Gesamtanzahl Einträge in Datenbank: {total_count}")
        
        # Techno-Beats spezifisch
        cursor.execute("""
            SELECT COUNT(*) FROM stems 
            WHERE path LIKE '%techno%' OR path LIKE '%LPE%' OR path LIKE '%drt%'
        """)
        techno_count = cursor.fetchone()[0]
        print(f"   Techno-Beats gefunden: {techno_count}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM stems 
            WHERE path LIKE '%drt130LPE126%'
        """)
        recent_count = cursor.fetchone()[0]
        print(f"   Davon in letzter Stunde: {recent_count}")
        print()
        
        # 2. BPM-Analyse der Techno-Beats
        print("2. BPM-ANALYSE DER TECHNO-BEATS:")
        cursor.execute("""
            SELECT bpm FROM stems 
            WHERE (path LIKE '%techno%' OR path LIKE '%LPE%' OR path LIKE '%drt%') AND bpm > 0
            ORDER BY bpm
        """)
        bpm_values = [row[0] for row in cursor.fetchall()]
        
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
        cursor.execute("""
            SELECT category, COUNT(*) as count FROM stems 
            WHERE path LIKE '%techno%' OR path LIKE '%LPE%' OR path LIKE '%drt%'
            GROUP BY category
            ORDER BY count DESC
        """)
        
        categories = cursor.fetchall()
        for category, count in categories:
            percentage = (count / techno_count) * 100
            print(f"   {category}: {count} Dateien ({percentage:.1f}%)")
        print()
        
        # 4. Qualitätskontrolle
        print("4. QUALITÄTSKONTROLLE:")
        cursor.execute("""
            SELECT quality_ok, COUNT(*) as count FROM stems 
            WHERE path LIKE '%techno%' OR path LIKE '%LPE%' OR path LIKE '%drt%'
            GROUP BY quality_ok
        """)
        
        quality_stats = cursor.fetchall()
        for quality, count in quality_stats:
            status = "✅ Qualität OK" if quality else "❌ Quarantäne"
            percentage = (count / techno_count) * 100
            print(f"   {status}: {count} Dateien ({percentage:.1f}%)")
        print()
        
        # 5. Feature-Analyse (Beispiel für erste 5 Dateien)
        print("5. FEATURE-ANALYSE (Beispiele):")
        cursor.execute("""
            SELECT path, features FROM stems 
            WHERE (path LIKE '%techno%' OR path LIKE '%LPE%' OR path LIKE '%drt%') AND features IS NOT NULL
            LIMIT 5
        """)
        
        feature_examples = cursor.fetchall()
        for i, (path, features_json) in enumerate(feature_examples, 1):
            filename = Path(path).name
            try:
                features = json.loads(features_json)
                print(f"   Datei {i}: {filename}")
                print(f"      Spektrale Zentroide: {features.get('spectral_centroid', 'N/A'):.1f} Hz")
                print(f"      RMS: {features.get('rms', 'N/A'):.3f}")
                print(f"      Zero-Crossing Rate: {features.get('zero_crossing_rate', 'N/A'):.3f}")
                print(f"      Spektraler Rolloff: {features.get('spectral_rolloff', 'N/A'):.1f} Hz")
                print(f"      Spektrale Bandbreite: {features.get('spectral_bandwidth', 'N/A'):.1f} Hz")
            except json.JSONDecodeError:
                print(f"   Datei {i}: {filename} - Fehler beim Laden der Features")
            print()
        
        # 6. Tag-Analyse
        print("6. SEMANTIC TAG-ANALYSE:")
        cursor.execute("""
            SELECT tags FROM stems 
            WHERE (path LIKE '%techno%' OR path LIKE '%LPE%' OR path LIKE '%drt%') AND tags IS NOT NULL
            LIMIT 3
        """)
        
        tag_examples = cursor.fetchall()
        all_tags = []
        
        for i, (tags_json,) in enumerate(tag_examples, 1):
            try:
                tags = json.loads(tags_json)
                print(f"   Beispiel {i}: {tags}")
                all_tags.extend(tags)
            except json.JSONDecodeError:
                print(f"   Beispiel {i}: Fehler beim Laden der Tags")
        
        if all_tags:
            tag_counter = Counter(all_tags)
            print(f"\n   Häufigste Tags:")
            for tag, count in tag_counter.most_common(10):
                print(f"      '{tag}': {count}x")
        print()
        
        # 7. Detaillierte Dateiliste
        print("7. VERARBEITETE TECHNO-DATEIEN:")
        cursor.execute("""
            SELECT path, bpm, category, quality_ok FROM stems 
            WHERE path LIKE '%techno%' OR path LIKE '%LPE%' OR path LIKE '%drt%'
            ORDER BY path
        """)
        
        techno_files = cursor.fetchall()
        for path, bpm, category, quality_ok in techno_files:
            filename = Path(path).name
            bpm_str = f"{bpm:.1f} BPM" if bpm > 0 else "No BPM"
            quality_str = "✅" if quality_ok else "❌"
            print(f"   {quality_str} {filename} | {category} | {bpm_str}")
        
        print(f"\n=== ZUSAMMENFASSUNG ===")
        print(f"🎵 {techno_count} Techno-Beats erfolgreich analysiert")
        print(f"🎯 Kategorisierung durch KMeans und Heuristik")
        print(f"🎼 BPM-Erkennung für Tempo-Matching")
        print(f"🔊 Spektrale Features für Klanganalyse")
        print(f"🏷️  Semantische Tags durch LAION-CLAP")
        print(f"✅ Qualitätskontrolle und intelligente Filterung")
        
    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_techno_beats()