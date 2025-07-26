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

import sqlite3
import numpy as np
import argparse
import sys
import os
from typing import List, Tuple, Dict
import logging
from laion_clap import CLAP_Module
import json
from pathlib import Path

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Datenbank-Pfad
DB_PATH = "processed_database/stems.db"

class SemanticAudioSearch:
    def __init__(self):
        """
        Initialisiert die semantische Audio-Suchmaschine.
        """
        logging.info("Initialisiere Semantische Audio-Suchmaschine...")
        
        # LAION-CLAP-Modell laden
        logging.info("Lade LAION-CLAP-Modell...")
        self.clap_model = CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()
        
        # Datenbankverbindung prüfen
        self._check_database()
        
        logging.info("Semantische Audio-Suchmaschine bereit.")
    
    def _check_database(self):
        """
        Prüft, ob die Datenbank existiert und CLAP-Embeddings enthält.
        """
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Datenbank nicht gefunden: {DB_PATH}")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Prüfe, ob clap_embedding Spalte existiert
        cursor.execute("PRAGMA table_info(stems)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'clap_embedding' not in columns:
            conn.close()
            raise ValueError("Datenbank enthält keine CLAP-Embeddings. Bitte führen Sie zuerst prepare_dataset_sql.py aus.")
        
        # Prüfe, ob Embeddings vorhanden sind
        cursor.execute("SELECT COUNT(*) FROM stems WHERE clap_embedding IS NOT NULL")
        embedding_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stems")
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        logging.info(f"Datenbank gefunden: {total_count} Stems, {embedding_count} mit CLAP-Embeddings")
        
        if embedding_count == 0:
            raise ValueError("Keine CLAP-Embeddings in der Datenbank gefunden.")
    
    def search(self, query: str, top_k: int = 10, category_filter: str = None, 
               bpm_range: Tuple[float, float] = None) -> List[Dict]:
        """
        Führt eine semantische Suche basierend auf einem Textprompt durch.
        
        Args:
            query (str): Der Suchtext (z.B. "dark techno kick")
            top_k (int): Anzahl der zurückzugebenden Ergebnisse
            category_filter (str): Optional - Filter nach Kategorie
            bpm_range (Tuple[float, float]): Optional - BPM-Bereich (min, max)
            
        Returns:
            List[Dict]: Liste der gefundenen Stems mit Metadaten und Ähnlichkeitswerten
        """
        logging.info(f"Suche nach: '{query}' (Top {top_k})")
        
        # Text-Embedding für die Suchanfrage berechnen
        query_embedding = self._get_text_embedding(query)
        
        # Alle Stems mit Embeddings aus der Datenbank laden
        stems_data = self._load_stems_with_embeddings(category_filter, bpm_range)
        
        if not stems_data:
            logging.warning("Keine passenden Stems in der Datenbank gefunden.")
            return []
        
        # Ähnlichkeiten berechnen
        similarities = []
        for stem in stems_data:
            similarity = self._calculate_cosine_similarity(query_embedding, stem['embedding'])
            stem['similarity'] = similarity
            similarities.append(stem)
        
        # Nach Ähnlichkeit sortieren (absteigend)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Top-K Ergebnisse zurückgeben
        results = similarities[:top_k]
        
        logging.info(f"Gefunden: {len(results)} relevante Stems")
        return results
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Berechnet CLAP-Embedding für einen Text.
        
        Args:
            text (str): Der Eingabetext
            
        Returns:
            np.ndarray: Das normalisierte Text-Embedding
        """
        try:
            # Text-Embedding berechnen
            text_embed = self.clap_model.get_text_embedding([text])
            
            # Normalisieren
            if len(text_embed.shape) > 1:
                text_embed = text_embed.flatten()
            
            text_embed = text_embed / np.linalg.norm(text_embed)
            
            return text_embed
            
        except Exception as e:
            logging.error(f"Fehler bei Text-Embedding-Berechnung: {e}")
            raise
    
    def _load_stems_with_embeddings(self, category_filter: str = None, 
                                   bpm_range: Tuple[float, float] = None) -> List[Dict]:
        """
        Lädt alle Stems mit ihren CLAP-Embeddings aus der Datenbank.
        
        Args:
            category_filter (str): Optional - Filter nach Kategorie
            bpm_range (Tuple[float, float]): Optional - BPM-Bereich
            
        Returns:
            List[Dict]: Liste der Stems mit Metadaten und Embeddings
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # SQL-Query mit optionalen Filtern erstellen
        query = "SELECT id, path, bpm, category, tags, clap_embedding FROM stems WHERE clap_embedding IS NOT NULL"
        params = []
        
        if category_filter:
            query += " AND category = ?"
            params.append(category_filter)
        
        if bpm_range:
            query += " AND bpm BETWEEN ? AND ?"
            params.extend(bpm_range)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        stems_data = []
        for row in rows:
            stem_id, path, bpm, category, tags, embedding_bytes = row
            
            # Embedding aus Binärdaten rekonstruieren
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Tags parsen
            try:
                tags_list = json.loads(tags) if tags else []
            except:
                tags_list = []
            
            stems_data.append({
                'id': stem_id,
                'path': path,
                'bpm': bpm,
                'category': category,
                'tags': tags_list,
                'embedding': embedding
            })
        
        return stems_data
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Embeddings.
        
        Args:
            embedding1 (np.ndarray): Erstes Embedding
            embedding2 (np.ndarray): Zweites Embedding
            
        Returns:
            float: Kosinus-Ähnlichkeit (0-1)
        """
        try:
            # Kosinus-Ähnlichkeit berechnen
            similarity = np.dot(embedding1, embedding2)
            
            # Sicherstellen, dass der Wert im Bereich [0, 1] liegt
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Fehler bei Ähnlichkeitsberechnung: {e}")
            return 0.0
    
    def print_results(self, results: List[Dict], show_details: bool = False):
        """
        Gibt die Suchergebnisse formatiert aus.
        
        Args:
            results (List[Dict]): Die Suchergebnisse
            show_details (bool): Ob detaillierte Informationen angezeigt werden sollen
        """
        if not results:
            print("\n❌ Keine Ergebnisse gefunden.")
            return
        
        print(f"\n🎵 {len(results)} Ergebnisse gefunden:\n")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            similarity_percent = result['similarity'] * 100
            
            print(f"{i:2d}. [{similarity_percent:5.1f}%] {result['id']}")
            print(f"    📁 {os.path.basename(result['path'])}")
            print(f"    🎛️  {result['category']} | 🥁 {result['bpm']:.1f} BPM")
            
            if result['tags']:
                tags_str = ", ".join(result['tags'])
                print(f"    🏷️  {tags_str}")
            
            if show_details:
                print(f"    📂 {result['path']}")
            
            print()

def main():
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
                    results = search_engine.search(
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
            results = search_engine.search(
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
    main()