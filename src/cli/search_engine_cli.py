#!/usr/bin/env python3
"""
Neuromorphe Traum-Engine v2.0 - Semantische Audio-Suchmaschine
CLAP-basierte Text-zu-Audio Retrieval CLI

Implementiert gemäß AGENTEN_DIREKTIVE_005:
- Text-zu-Audio Suche mit CLAP-Embeddings
- Kosinus-Ähnlichkeitssuche
- Ranking nach Relevanz
- Interaktive CLI-Schnittstelle
"""

import argparse
import sys
import asyncio
from typing import List, Tuple, Dict
import logging
from pathlib import Path

from ...services.search import SearchService
from ...schemas.stem import SearchResult, Stem

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticAudioSearch:
    def __init__(self):
        """
        Initialisiert die semantische Audio-Suchmaschine.
        """
        logging.info("Initialisiere Semantische Audio-Suchmaschine...")
        self.search_service = SearchService()
        logging.info("Semantische Audio-Suchmaschine bereit.")
    
    
    
    async def search(self, query: str, top_k: int = 10, category_filter: str = None, 
               bpm_range: Tuple[float, float] = None) -> List[SearchResult]:
        """
        Führt eine semantische Suche basierend auf einem Textprompt durch.
        """
        return await self.search_service.search(
            query=query,
            top_k=top_k,
            category_filter=category_filter,
            bpm_range=bpm_range
        )
    
    
    
    def print_results(self, results: List[SearchResult], show_details: bool = False):
        """
        Gibt die Suchergebnisse formatiert aus.
        
        Args:
            results (List[SearchResult]): Die Suchergebnisse
            show_details (bool): Ob detaillierte Informationen angezeigt werden sollen
        """
        if not results:
            print("\n❌ Keine Ergebnisse gefunden.")
            return
        
        print(f"\n🎵 {len(results)} Ergebnisse gefunden:\n")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            similarity_percent = result.similarity_score * 100
            stem = result.stem
            
            print(f"{i:2d}. [{similarity_percent:5.1f}%] {stem.id}")
            print(f"    📁 {Path(stem.original_path).name if stem.original_path else stem.filename}")
            print(f"    🎛️  {stem.category} | 🥁 {stem.bpm:.1f} BPM")
            
            if stem.auto_tags:
                tags_str = ", ".join(stem.auto_tags)
                print(f"    🏷️  {tags_str}")
            
            if show_details:
                print(f"    📂 {stem.original_path}")
            
            print()

async def main():
    """
    Hauptfunktion für die CLI-Schnittstelle.
    """
    parser = argparse.ArgumentParser(
        description="Neuromorphe Traum-Engine v2.0 - Semantische Audio-Suche",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python search_engine_cli.py "dark techno kick"
  python search_engine_cli.py "atmospheric pad" --top-k 5
  python search_engine_cli.py "punchy snare" --category snare --bpm-min 120 --bpm-max 140
  python search_engine_cli.py "hypnotic bass" --details
        """
    )
    
    parser.add_argument("query", help="Suchtext (z.B. 'dark techno kick')")
    parser.add_argument("--top-k", type=int, default=10, help="Anzahl der Ergebnisse (Standard: 10)")
    parser.add_argument("--category", help="Filter nach Kategorie (kick, bass, snare, etc.)")
    parser.add_argument("--bpm-min", type=float, help="Minimaler BPM-Wert")
    parser.add_argument("--bpm-max", type=float, help="Maximaler BPM-Wert")
    parser.add_argument("--details", action="store_true", help="Zeige detaillierte Pfadinformationen")
    parser.add_argument("--interactive", action="store_true", help="Interaktiver Modus")
    
    args = parser.parse_args()
    
    try:
        # Suchmaschine initialisieren
        search_engine = SemanticAudioSearch()
        
        if args.interactive:
            # Interaktiver Modus
            print("\n🎵 Neuromorphe Traum-Engine v2.0 - Interaktive Suche")
            print("Geben Sie 'quit' ein, um zu beenden.\n")
            
            while True:
                try:
                    query = input("🔍 Suche: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not query:
                        continue
                    
                    # BPM-Bereich
                    bpm_range = None
                    if args.bpm_min is not None and args.bpm_max is not None:
                        bpm_range = (args.bpm_min, args.bpm_max)
                    
                    # Suche durchführen
                    results = await search_engine.search(
                        query=query,
                        top_k=args.top_k,
                        category_filter=args.category,
                        bpm_range=bpm_range
                    )
                    
                    # Ergebnisse anzeigen
                    search_engine.print_results(results, show_details=args.details)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Fehler: {e}")
            
            print("\n👋 Auf Wiedersehen!")
        
        else:
            # Einmalige Suche
            bpm_range = None
            if args.bpm_min is not None and args.bpm_max is not None:
                bpm_range = (args.bpm_min, args.bpm_max)
            
            # Suche durchführen
            results = await search_engine.search(
                query=args.query,
                top_k=args.top_k,
                category_filter=args.category,
                bpm_range=bpm_range
            )
            
            # Ergebnisse anzeigen
            search_engine.print_results(results, show_details=args.details)
    
    except Exception as e:
        logging.error(f"Kritischer Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())