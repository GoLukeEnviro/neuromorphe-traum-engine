#!/usr/bin/env python3
"""
Test-Skript für das aktualisierte prepare_dataset_sql.py
Testet die neuen Funktionen von AGENTEN_DIREKTIVE_006
"""

import sys
import os
import logging
from prepare_dataset_sql import NeuroAnalyzer, ProcessingStatus, ProcessingResult

def test_initialization():
    """Testet die Initialisierung des NeuroAnalyzers"""
    print("=== Test: Initialisierung ===")
    try:
        analyzer = NeuroAnalyzer(
            input_dir="../raw_construction_kits",
            resume_from_checkpoint=False,
            batch_size=4,
            max_retries=2,
            checkpoint_interval=5
        )
        print("✓ NeuroAnalyzer erfolgreich initialisiert")
        print(f"  - Input Directory: {analyzer.input_dir}")
        print(f"  - Batch Size: {analyzer.batch_size}")
        print(f"  - Max Retries: {analyzer.max_retries}")
        print(f"  - Checkpoint Interval: {analyzer.checkpoint_interval}")
        return True
    except Exception as e:
        print(f"✗ Fehler bei Initialisierung: {e}")
        return False

def test_database_schema():
    """Testet das erweiterte Datenbankschema"""
    print("\n=== Test: Datenbankschema ===")
    try:
        analyzer = NeuroAnalyzer(input_dir="../raw_construction_kits")
        analyzer.init_db()
        
        # Prüfe ob processing_status Tabelle existiert
        cursor = analyzer.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='processing_status'")
        result = cursor.fetchone()
        
        if result:
            print("✓ processing_status Tabelle erfolgreich erstellt")
            
            # Prüfe Spalten
            cursor.execute("PRAGMA table_info(processing_status)")
            columns = [row[1] for row in cursor.fetchall()]
            expected_columns = ['file_path', 'file_hash', 'status', 'last_attempt', 'retry_count', 'error_message', 'processing_time']
            
            missing_columns = [col for col in expected_columns if col not in columns]
            if not missing_columns:
                print("✓ Alle erwarteten Spalten vorhanden")
                return True
            else:
                print(f"✗ Fehlende Spalten: {missing_columns}")
                return False
        else:
            print("✗ processing_status Tabelle nicht gefunden")
            return False
            
    except Exception as e:
        print(f"✗ Fehler bei Datenbanktest: {e}")
        return False

def test_processing_result():
    """Testet die ProcessingResult Datenklasse"""
    print("\n=== Test: ProcessingResult ===")
    try:
        result = ProcessingResult(
            file_path="test.wav",
            status=ProcessingStatus.COMPLETED,
            metadata={"test": "data"},
            retry_count=1,
            processing_time=2.5,
            error_message=None
        )
        
        print("✓ ProcessingResult erfolgreich erstellt")
        print(f"  - Status: {result.status}")
        print(f"  - Retry Count: {result.retry_count}")
        print(f"  - Processing Time: {result.processing_time}s")
        return True
        
    except Exception as e:
        print(f"✗ Fehler bei ProcessingResult Test: {e}")
        return False

def main():
    """Hauptfunktion für Tests"""
    print("Neuromorphe Traum-Engine Dataset Processor")
    print("AGENTEN_DIREKTIVE_006 - Funktionstest")
    print("=" * 50)
    
    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tests = [
        test_initialization,
        test_database_schema,
        test_processing_result
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test-Ergebnisse: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("✓ Alle Tests erfolgreich!")
        print("Das System ist bereit für AGENTEN_DIREKTIVE_006")
        return 0
    else:
        print("✗ Einige Tests fehlgeschlagen")
        return 1

if __name__ == "__main__":
    sys.exit(main())