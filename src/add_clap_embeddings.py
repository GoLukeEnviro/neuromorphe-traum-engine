#!/usr/bin/env python3
"""
Neuromorphe Traum-Engine v2.0 - CLAP Embedding Processor
Fügt CLAP-Embeddings zu bestehenden Stems in der PostgreSQL-Datenbank hinzu.
"""

import asyncio
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Core imports
from core.logging import setup_logging, get_logger
from database.database import get_database_manager
from database.crud import StemCRUD
from database.models import Stem
from services.neuro_analyzer import NeuroAnalyzer
from core.config import settings

logger = get_logger(__name__)

class CLAPEmbeddingProcessor:
    """Verarbeitet bestehende Stems und fügt CLAP-Embeddings hinzu"""
    
    def __init__(self):
        self.settings = settings
        self.db_manager = get_database_manager()
        self.neuro_analyzer = None
        
    async def initialize(self):
        """Initialisiert den CLAP-Embedding-Processor"""
        logger.info("Initialisiere CLAP-Embedding-Processor...")
        
        # NeuroAnalyzer initialisieren
        self.neuro_analyzer = NeuroAnalyzer()
        
        logger.info("CLAP-Embedding-Processor erfolgreich initialisiert")
    
    def get_stems_without_embeddings(self) -> List[Stem]:
        """Holt alle Stems ohne Audio-Embeddings aus der Datenbank"""
        with self.db_manager.get_sync_session() as session:
            # Stems ohne Audio-Embeddings finden
            stems = StemCRUD.get_stems(
                session,
                audio_embedding_is_null=True,
                limit=None  # Alle Stems ohne Embeddings
            )
            return stems
    
    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """Lädt eine Audio-Datei und konvertiert sie für CLAP"""
        try:
            # Audio laden mit librosa
            audio, sr = librosa.load(file_path, sr=48000, mono=True)
            
            # Validierung
            if len(audio) == 0:
                logger.warning(f"Leere Audio-Datei: {file_path}")
                return None
            
            # Normalisierung
            audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Audio-Datei {file_path}: {e}")
            return None
    
    async def process_stem(self, stem: Stem) -> bool:
        """Verarbeitet einen einzelnen Stem und fügt CLAP-Embedding hinzu"""
        try:
            # Audio-Datei laden
            file_path = stem.processed_path or stem.original_path
            
            # Pfad-Auflösung: Versuche verschiedene Pfad-Varianten
            resolved_path = None
            if file_path:
                # 1. Originaler Pfad
                if Path(file_path).exists():
                    resolved_path = file_path
                # 2. Relativer Pfad vom Projekt-Root
                elif Path(f"..{file_path}").exists():
                    resolved_path = f"..{file_path}"
                # 3. Relativer Pfad vom src-Verzeichnis
                elif Path(f".{file_path}").exists():
                    resolved_path = f".{file_path}"
            
            if not resolved_path:
                logger.warning(f"Audio-Datei nicht gefunden für Stem {stem.id}: {file_path}")
                return False
            
            # Audio laden
            audio = self.load_audio_file(resolved_path)
            if audio is None:
                return False
            
            # CLAP-Embedding generieren über den SemanticAnalyzer
            embedding = self.neuro_analyzer.semantic_analyzer.embedder.get_audio_embedding(audio, sample_rate=48000)
            
            # Embedding als JSON speichern (Liste von Floats)
            embedding_json = embedding.tolist()
            
            # Stem in Datenbank aktualisieren - Stem-Objekt in neuer Session laden
            with self.db_manager.get_sync_session() as session:
                # Stem-Objekt in der aktuellen Session laden
                db_stem = StemCRUD.get_stem_by_id(session, stem.id)
                if db_stem:
                    db_stem.audio_embedding = embedding_json
                    session.commit()
                    logger.debug(f"CLAP-Embedding für Stem {stem.id} erfolgreich hinzugefügt")
                    return True
                else:
                    logger.error(f"Stem {stem.id} nicht in Datenbank gefunden")
                    return False
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von Stem {stem.id}: {e}")
            return False
    
    async def process_all_stems(self):
        """Verarbeitet alle Stems ohne CLAP-Embeddings"""
        logger.info("Starte CLAP-Embedding-Verarbeitung...")
        
        # Stems ohne Embeddings finden
        stems_without_embeddings = self.get_stems_without_embeddings()
        total_stems = len(stems_without_embeddings)
        
        if total_stems == 0:
            logger.info("Alle Stems haben bereits CLAP-Embeddings!")
            return
        
        logger.info(f"Gefunden: {total_stems} Stems ohne CLAP-Embeddings")
        
        # Fortschrittsanzeige
        successful = 0
        failed = 0
        
        with tqdm(total=total_stems, desc="CLAP-Embeddings") as pbar:
            for stem in stems_without_embeddings:
                success = await self.process_stem(stem)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'Erfolgreich': successful,
                    'Fehlgeschlagen': failed
                })
        
        logger.info(f"CLAP-Embedding-Verarbeitung abgeschlossen:")
        logger.info(f"  Erfolgreich: {successful}")
        logger.info(f"  Fehlgeschlagen: {failed}")
        logger.info(f"  Gesamt: {total_stems}")
    
    async def verify_embeddings(self):
        """Überprüft die CLAP-Embeddings in der Datenbank"""
        logger.info("Überprüfe CLAP-Embeddings...")
        
        with self.db_manager.get_sync_session() as session:
            # Gesamtanzahl
            total_count = StemCRUD.get_stem_count(session)
            
            # Mit Embeddings
            stems_with_embeddings = StemCRUD.get_stems(
                session,
                audio_embedding_is_not_null=True
            )
            embedding_count = len(stems_with_embeddings)
            
            # Ohne Embeddings
            stems_without_embeddings = StemCRUD.get_stems(
                session,
                audio_embedding_is_null=True
            )
            missing_count = len(stems_without_embeddings)
        
        logger.info(f"CLAP-Embedding-Status:")
        logger.info(f"  Gesamt Stems: {total_count}")
        logger.info(f"  Mit Embeddings: {embedding_count}")
        logger.info(f"  Ohne Embeddings: {missing_count}")
        
        if total_count > 0:
            percentage = (embedding_count / total_count) * 100
            logger.info(f"  Abdeckung: {percentage:.1f}%")
        
        # Beispiel-Embedding anzeigen
        if stems_with_embeddings:
            example_stem = stems_with_embeddings[0]
            embedding = np.array(example_stem.audio_embedding)
            logger.info(f"Beispiel-Embedding (Stem {example_stem.id}):")
            logger.info(f"  Dimensionen: {embedding.shape}")
            logger.info(f"  Erste 5 Werte: {embedding[:5]}")
            logger.info(f"  Norm: {np.linalg.norm(embedding):.4f}")

async def main():
    """Hauptfunktion"""
    # Logging initialisieren
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("Neuromorphe Traum-Engine v2.0 - CLAP Embedding Processor")
    logger.info("=" * 60)
    
    try:
        # Processor initialisieren
        processor = CLAPEmbeddingProcessor()
        await processor.initialize()
        
        # Status vor der Verarbeitung
        await processor.verify_embeddings()
        
        # CLAP-Embeddings verarbeiten
        await processor.process_all_stems()
        
        # Status nach der Verarbeitung
        await processor.verify_embeddings()
        
        logger.info("CLAP-Embedding-Processing erfolgreich abgeschlossen!")
        
    except KeyboardInterrupt:
        logger.info("Verarbeitung durch Benutzer unterbrochen")
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())