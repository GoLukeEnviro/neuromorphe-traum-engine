#!/usr/bin/env python3
"""Beispiel für die neue musikalische Intelligenz im ArrangerService"""

import asyncio
import sys
import os
from unittest.mock import MagicMock

# Pfad zum Projekt hinzufügen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.arranger import ArrangerService
from src.database.crud import StemCRUD

def create_mock_stem(stem_id, filename, category, key, bpm, harmonic_complexity, rhythmic_complexity):
    """Erstellt einen Mock-Stem für Demonstrationszwecke"""
    mock_stem = MagicMock()
    mock_stem.id = stem_id
    mock_stem.filename = filename
    mock_stem.category = category
    mock_stem.key = key
    mock_stem.bpm = bpm
    mock_stem.harmonic_complexity = harmonic_complexity
    mock_stem.rhythmic_complexity = rhythmic_complexity
    return mock_stem

def mock_database_session():
    """Erstellt eine Mock-Datenbank-Session mit Beispiel-Stems"""
    # Beispiel-Stems in verschiedenen Tonarten
    mock_stems = [
        create_mock_stem(1, "dark_kick_am.wav", "kick", "Am", 128, 0.3, 0.7),
        create_mock_stem(2, "atmospheric_pad_c.wav", "atmo", "C", 128, 0.8, 0.2),
        create_mock_stem(3, "driving_bass_am.wav", "bass", "Am", 128, 0.5, 0.8),
        create_mock_stem(4, "hihat_closed_am.wav", "hihat", "Am", 128, 0.2, 0.9),
        create_mock_stem(5, "lead_synth_c.wav", "lead", "C", 128, 0.9, 0.4),
        create_mock_stem(6, "perc_loop_dm.wav", "perc", "Dm", 128, 0.4, 0.8),
    ]
    
    # Mock der CRUD-Methoden
    def mock_search_harmonically_compatible_stems(db, base_key, category=None, tags=None, limit=50):
        compatible_keys = StemCRUD.get_compatible_keys(base_key)
        filtered_stems = [s for s in mock_stems if s.key in compatible_keys]
        if category:
            filtered_stems = [s for s in filtered_stems if category.lower() in s.category.lower()]
        return filtered_stems[:limit]
    
    def mock_get_stems(db, **kwargs):
        filtered_stems = mock_stems.copy()
        if 'instrument' in kwargs and kwargs['instrument']:
            filtered_stems = [s for s in filtered_stems if kwargs['instrument'].lower() in s.category.lower()]
        if 'bpm_min' in kwargs and kwargs['bpm_min']:
            filtered_stems = [s for s in filtered_stems if s.bpm >= kwargs['bpm_min']]
        if 'bpm_max' in kwargs and kwargs['bmp_max']:
            filtered_stems = [s for s in filtered_stems if s.bpm <= kwargs['bpm_max']]
        return filtered_stems[:kwargs.get('limit', 50)]
    
    # Patche die CRUD-Methoden
    StemCRUD.search_harmonically_compatible_stems = mock_search_harmonically_compatible_stems
    StemCRUD.get_stems = mock_get_stems
    
    return MagicMock()

async def demonstrate_musical_arrangement():
    """Demonstriert die neue musikalische Intelligenz"""
    print("🎵 Neuromorphe Traum-Engine v2.0 - Musikalische Arrangement-Demo 🎵\n")
    
    # ArrangerService initialisieren
    arranger = ArrangerService()
    
    # Mock-Datenbank-Session erstellen
    mock_db = mock_database_session()
    
    # Verschiedene Prompts testen
    test_prompts = [
        "Dark atmospheric techno 128 BPM in Am",
        "Uplifting trance 138 BPM in C major",
        "Industrial harsh mechanical 140 BPM"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"=== Arrangement {i}: '{prompt}' ===")
        
        try:
            # Arrangement erstellen
            arrangement = await arranger.create_arrangement(
                prompt=prompt,
                duration=180,  # 3 Minuten
                db_session=mock_db
            )
            
            print(f"Arrangement ID: {arrangement['arrangement_id']}")
            print(f"Globale Tonart: {arrangement['global_key']}")
            print(f"BPM: {arrangement['bpm']}")
            print(f"Genre: {arrangement['genre']}")
            print(f"Geschätzte Dauer: {arrangement['estimated_duration']:.1f}s")
            print(f"Musikalische Intelligenz: {arrangement['metadata']['created_with_musical_intelligence']}")
            print(f"Harmonische Kohärenz: {arrangement['metadata']['harmonic_coherence']}")
            print(f"Tonart-Kompatibilität verwendet: {arrangement['metadata']['key_compatibility_used']}")
            
            print("\nStruktur:")
            for section in arrangement['structure']:
                print(f"  {section['section']} ({section['bars']} Takte):")
                for stem in section['stems']:
                    print(f"    - {stem['filename']} (Tonart: {stem['key']}, Komplexität: H={stem['harmonic_complexity']:.2f}, R={stem['rhythmic_complexity']:.2f})")
            
            print("\n" + "="*80 + "\n")
            
        except Exception as e:
            print(f"❌ Fehler bei Arrangement-Erstellung: {e}\n")

def demonstrate_harmonic_compatibility():
    """Demonstriert die harmonische Kompatibilität"""
    print("=== Harmonische Kompatibilität ===")
    
    test_keys = ['Am', 'C', 'Dm', 'G', 'F#m']
    
    for key in test_keys:
        compatible = StemCRUD.get_compatible_keys(key)
        print(f"{key}: {', '.join(compatible)}")
    
    print("\n")

def main():
    """Hauptfunktion"""
    demonstrate_harmonic_compatibility()
    
    # Asynchrone Demo ausführen
    asyncio.run(demonstrate_musical_arrangement())
    
    print("🎉 Demo abgeschlossen!")
    print("\nDie Neuromorphe Traum-Engine v2.0 verfügt jetzt über:")
    print("✓ Dynamische Tonart-Erkennung mit Krumhansl-Schmuckler-Algorithmus")
    print("✓ Harmonische und rhythmische Komplexitäts-Analyse")
    print("✓ Harmonisch kohärente Arrangement-Generierung")
    print("✓ Erweiterte Stem-Suche mit musikalischen Filtern")
    print("✓ Intelligente Tonart-Kompatibilität für bessere Arrangements")

if __name__ == "__main__":
    main()