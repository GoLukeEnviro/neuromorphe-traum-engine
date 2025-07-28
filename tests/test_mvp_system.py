#!/usr/bin/env python3
"""
Test-Skript für AGENTEN_DIREKTIVE_007 - Master MVP
Automatisierte Tests für das komplette Text-zu-Stem Retrieval System
"""

import os
import pickle
import sys
from pathlib import Path

# Füge ai_agents zum Python-Pfad hinzu
sys.path.append('ai_agents')

from ai_agents.search_engine_cli import SearchEngine

def test_system_components():
    """
    Testet alle Komponenten des MVP-Systems.
    """
    print("🧪 AGENTEN_DIREKTIVE_007 - Master MVP System Test")
    print("=" * 60)
    
    # Test 1: Verzeichnisstruktur
    print("\n1. Teste Verzeichnisstruktur...")
    required_dirs = ['raw_construction_kits', 'processed_database', 'ai_agents']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✓ {dir_name} existiert")
        else:
            print(f"   ✗ {dir_name} fehlt")
            return False
    
    # Test 2: Audio-Dateien
    print("\n2. Teste Audio-Dateien...")
    audio_files = list(Path('raw_construction_kits').glob('*.wav'))
    print(f"   ✓ {len(audio_files)} Audio-Dateien gefunden")
    if len(audio_files) == 0:
        print("   ⚠️  Keine Audio-Dateien zum Testen")
        return False
    
    # Test 3: Embeddings-Datei
    print("\n3. Teste Embeddings-Datei...")
    embeddings_path = 'processed_database/embeddings.pkl'
    if os.path.exists(embeddings_path):
        print(f"   ✓ {embeddings_path} existiert")
        
        # Lade und prüfe Embeddings
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        print(f"   ✓ {len(embeddings_data)} Embeddings geladen")
        
        # Prüfe Struktur
        if embeddings_data and 'path' in embeddings_data[0] and 'embedding' in embeddings_data[0]:
            print("   ✓ Embedding-Struktur korrekt")
        else:
            print("   ✗ Embedding-Struktur fehlerhaft")
            return False
    else:
        print(f"   ✗ {embeddings_path} fehlt")
        return False
    
    # Test 4: Suchmaschine
    print("\n4. Teste Suchmaschine...")
    try:
        search_engine = SearchEngine(embeddings_path)
        print("   ✓ Suchmaschine erfolgreich initialisiert")
        
        # Test-Suchen
        test_queries = [
            "kick drum",
            "bass line",
            "melody",
            "dark industrial sound",
            "punchy attack"
        ]
        
        print("\n5. Teste Suchanfragen...")
        for i, query in enumerate(test_queries, 1):
            try:
                results = search_engine.search(query, top_k=3)
                print(f"   ✓ Query {i}: '{query}' → {len(results)} Ergebnisse")
                
                # Zeige Top-Ergebnis
                if results:
                    top_result = results[0]
                    file_name = os.path.basename(top_result[0])
                    similarity = top_result[1]
                    print(f"     → Top: {file_name} (Ähnlichkeit: {similarity:.4f})")
                
            except Exception as e:
                print(f"   ✗ Query {i} fehlgeschlagen: {e}")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 ALLE TESTS BESTANDEN!")
        print("Das MVP-System ist vollständig funktionsfähig.")
        print("\nZum Starten der interaktiven Suche:")
        print("python ai_agents/search_engine_cli.py")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"   ✗ Suchmaschine-Initialisierung fehlgeschlagen: {e}")
        return False

def demonstrate_search():
    """
    Demonstriert die Suchfunktionalität mit verschiedenen Queries.
    """
    print("\n🔍 DEMO: Semantische Audio-Suche")
    print("=" * 40)
    
    try:
        search_engine = SearchEngine('processed_database/embeddings.pkl')
        
        demo_queries = [
            "powerful kick drum with punch",
            "melodic bass line",
            "dark atmospheric sound",
            "rhythmic loop"
        ]
        
        for query in demo_queries:
            print(f"\n🎯 Suche: '{query}'")
            print("-" * 30)
            
            results = search_engine.search(query, top_k=3)
            
            for i, (file_path, similarity) in enumerate(results, 1):
                file_name = os.path.basename(file_path)
                print(f"{i}. {file_name}")
                print(f"   Ähnlichkeit: {similarity:.4f}")
        
    except Exception as e:
        print(f"Demo fehlgeschlagen: {e}")

if __name__ == "__main__":
    # Führe System-Tests durch
    success = test_system_components()
    
    if success:
        # Führe Demo durch
        demonstrate_search()
    else:
        print("\n❌ System-Tests fehlgeschlagen!")
        print("Bitte überprüfen Sie die Implementierung.")
        sys.exit(1)