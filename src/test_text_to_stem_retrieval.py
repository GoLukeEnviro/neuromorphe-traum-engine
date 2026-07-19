#!/usr/bin/env python3
"""
Test-Skript für das Text-zu-Stem-Retrieval-System
Testet die semantische Suche mit CLAP-Embeddings
"""

import asyncio
import numpy as np
from typing import List, Tuple
from pathlib import Path
import sys

# Projekt-Pfad hinzufügen
sys.path.append(str(Path(__file__).parent))

from core.logging import setup_logging, get_logger
from database.database import get_database_manager
from database.crud import StemCRUD
from services.neuro_analyzer import NeuroAnalyzer
from database.models import Stem

# Logging initialisieren
setup_logging()
logger = get_logger(__name__)

class TextToStemRetrieval:
    """Text-zu-Stem-Retrieval-System mit CLAP-Embeddings"""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.neuro_analyzer = NeuroAnalyzer()
    
    async def initialize(self):
        """Initialisiert das Retrieval-System"""
        logger.info("Initialisiere Text-zu-Stem-Retrieval-System...")
        
        # NeuroAnalyzer ist bereits im Konstruktor initialisiert
        # Lade das CLAP-Modell
        self.neuro_analyzer.semantic_analyzer.embedder.load_model()
        
        logger.info("Text-zu-Stem-Retrieval-System erfolgreich initialisiert")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generiert CLAP-Embedding für Text"""
        return self.neuro_analyzer.semantic_analyzer.embedder.get_text_embedding(text)
    
    def get_stems_with_embeddings(self) -> List[Stem]:
        """Holt alle Stems mit Audio-Embeddings aus der Datenbank"""
        with self.db_manager.get_sync_session() as session:
            stems = StemCRUD.get_stems(
                session,
                audio_embedding_is_not_null=True,
                limit=None
            )
            return stems
    
    def calculate_similarity(self, text_embedding: np.ndarray, audio_embedding: np.ndarray) -> float:
        """Berechnet Cosinus-Ähnlichkeit zwischen Text- und Audio-Embedding"""
        # Normalisierung
        text_norm = text_embedding / np.linalg.norm(text_embedding)
        audio_norm = audio_embedding / np.linalg.norm(audio_embedding)
        
        # Cosinus-Ähnlichkeit
        similarity = np.dot(text_norm, audio_norm)
        return float(similarity)
    
    def search_stems(self, query: str, top_k: int = 5) -> List[Tuple[Stem, float]]:
        """Sucht die ähnlichsten Stems für eine Text-Anfrage"""
        logger.info(f"Suche Stems für Query: '{query}'")
        
        # Text-Embedding generieren
        text_embedding = self.get_text_embedding(query)
        logger.debug(f"Text-Embedding generiert: {text_embedding.shape}")
        
        # Stems mit Embeddings laden
        stems = self.get_stems_with_embeddings()
        logger.info(f"Gefunden: {len(stems)} Stems mit Embeddings")
        
        if not stems:
            logger.warning("Keine Stems mit Embeddings gefunden!")
            return []
        
        # Ähnlichkeiten berechnen
        similarities = []
        for stem in stems:
            audio_embedding = np.array(stem.audio_embedding)
            similarity = self.calculate_similarity(text_embedding, audio_embedding)
            similarities.append((stem, similarity))
        
        # Nach Ähnlichkeit sortieren (absteigend)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Top-K zurückgeben
        return similarities[:top_k]
    
    def print_search_results(self, results: List[Tuple[Stem, float]], query: str):
        """Gibt Suchergebnisse formatiert aus"""
        print(f"\n🔍 Suchergebnisse für: '{query}'")
        print("=" * 60)
        
        if not results:
            print("❌ Keine Ergebnisse gefunden")
            return
        
        for i, (stem, similarity) in enumerate(results, 1):
            print(f"{i}. {stem.filename}")
            print(f"   📁 Kategorie: {stem.category or 'Unbekannt'}")
            print(f"   🎵 Genre: {stem.genre or 'Unbekannt'}")
            print(f"   😊 Stimmung: {stem.mood or 'Unbekannt'}")
            print(f"   ⚡ Energie: {stem.energy_level or 'Unbekannt'}")
            print(f"   🎯 Ähnlichkeit: {similarity:.4f}")
            print(f"   📂 Pfad: {stem.original_path}")
            print()

async def test_retrieval_system():
    """Testet das Text-zu-Stem-Retrieval-System"""
    logger.info("=" * 60)
    logger.info("Neuromorphe Traum-Engine v2.0 - Text-zu-Stem-Retrieval Test")
    logger.info("=" * 60)
    
    try:
        # System initialisieren
        retrieval = TextToStemRetrieval()
        await retrieval.initialize()
        
        # Test-Queries
        test_queries = [
            "powerful kick drum",
            "snappy snare",
            "crisp hi-hat",
            "deep bass",
            "rhythmic percussion",
            "melodic element",
            "aggressive sound",
            "soft ambient texture"
        ]
        
        # Jede Query testen
        for query in test_queries:
            results = retrieval.search_stems(query, top_k=3)
            retrieval.print_search_results(results, query)
            
            # Kurze Pause zwischen Queries
            await asyncio.sleep(0.5)
        
        logger.info("Text-zu-Stem-Retrieval-Test erfolgreich abgeschlossen!")
        
    except KeyboardInterrupt:
        logger.info("Test durch Benutzer unterbrochen")
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_retrieval_system())