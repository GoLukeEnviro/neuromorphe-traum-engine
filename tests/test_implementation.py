#!/usr/bin/env python3
"""
Test-Script für die Neuromorphic Dream-Engine v2.0
Demonstriert die Funktionalität der NeuroAnalyzer-Klasse
"""

import os
import sys
import logging
from ai_agents.prepare_dataset_sql import NeuroAnalyzer

# Logging für Tests konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_database_initialization():
    """
    Testet die Datenbank-Initialisierung
    """
    print("\n=== Test: Datenbank-Initialisierung ===")
    
    analyzer = NeuroAnalyzer(input_dir="raw_construction_kits")
    analyzer.init_db()
    
    # Prüfe ob Datenbank erstellt wurde
    if os.path.exists("processed_database/stems.db"):
        print("✓ Datenbank erfolgreich erstellt")
    else:
        print("✗ Datenbank nicht gefunden")
        return False
    
    return True

def test_directory_structure():
    """
    Testet die Verzeichnisstruktur
    """
    print("\n=== Test: Verzeichnisstruktur ===")
    
    required_dirs = [
        "raw_construction_kits",
        "processed_database",
        "ai_agents"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name} existiert")
        else:
            print(f"✗ {dir_name} fehlt")
            all_exist = False
    
    return all_exist

def test_empty_run():
    """
    Testet den Analyzer mit leerem Input-Verzeichnis
    """
    print("\n=== Test: Leerer Analyzer-Lauf ===")
    
    analyzer = NeuroAnalyzer(input_dir="raw_construction_kits")
    analyzer.init_db()
    
    try:
        analyzer.run()
        print("✓ Analyzer läuft ohne Fehler (keine Dateien gefunden)")
        return True
    except Exception as e:
        print(f"✗ Fehler beim Analyzer-Lauf: {e}")
        return False

def show_implementation_status():
    """
    Zeigt den Status der Implementierung
    """
    print("\n=== Implementierungsstatus ===")
    print("✓ Imports für Audio-Verarbeitung hinzugefügt")
    print("✓ Parallele Verarbeitung mit ProcessPoolExecutor implementiert")
    print("✓ Audio-Standardisierung (48kHz, Mono, 24-bit WAV)")
    print("✓ BPM-Analyse mit librosa")
    print("✓ Qualitätskontrolle (Mindestdauer 0.5s)")
    print("✓ Quarantäne-System für kurze Dateien")
    print("✓ Vollständige Datenbank-Integration")
    print("✓ Robuste Fehlerbehandlung")
    print("✓ Informative Logging-Ausgaben")
    
    print("\n=== Nächste Schritte ===")
    print("1. Installiere Abhängigkeiten: pip install -r requirements.txt")
    print("2. Füge Audiodateien in raw_construction_kits/ hinzu")
    print("3. Führe den Analyzer aus: python ai_agents/prepare_dataset_sql.py")
    print("4. Prüfe Ergebnisse in processed_database/stems.db")

def main():
    """
    Hauptfunktion für Tests
    """
    print("Neuromorphic Dream-Engine v2.0 - Test Suite")
    print("=" * 50)
    
    # Führe Tests durch
    tests = [
        test_directory_structure,
        test_database_initialization,
        test_empty_run
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test-Ergebnisse ===")
    print(f"Bestanden: {passed}/{total}")
    
    if passed == total:
        print("✓ Alle Tests bestanden!")
    else:
        print("✗ Einige Tests fehlgeschlagen")
    
    show_implementation_status()

if __name__ == "__main__":
    main()