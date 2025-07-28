"""Separation Service für Audio Source Separation mit Demucs

Dieser Service nutzt Demucs von Meta für State-of-the-Art Audio Source Separation.
Er zerlegt Stereo-Tracks in ihre Kern-Stems: drums, bass, vocals, other.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import save_audio

from src.core.config import settings
from src.core.logging import get_logger
logger = get_logger(__name__)


class SeparationService:
    """Service für Audio Source Separation mit Demucs"""
    
    def __init__(self, model_name: str = "htdemucs"):
        """
        Initialisiert den SeparationService
        
        Args:
            model_name: Name des Demucs-Modells (htdemucs, hdemucs, mdx_extra, etc.)
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=1)  # CPU-intensive Aufgaben
        
        # Verzeichnisse
        self.temp_dir = Path(tempfile.gettempdir()) / "neuromorphe_separation"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"SeparationService initialisiert mit Modell: {model_name}, Device: {self.device}")
    
    def _load_model(self) -> None:
        """Lädt das Demucs-Modell"""
        if self.model is None:
            logger.info(f"Lade Demucs-Modell: {self.model_name}")
            try:
                self.model = get_model(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Demucs-Modell erfolgreich geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden des Demucs-Modells: {e}")
                raise
    
    def _separate_audio_sync(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """
        Synchrone Audio-Separation (läuft in ThreadPoolExecutor)
        
        Args:
            audio_path: Pfad zur Audio-Datei
            output_dir: Ausgabe-Verzeichnis
            
        Returns:
            Dictionary mit Stem-Namen als Keys und Dateipfaden als Values
        """
        try:
            # Modell laden falls nötig
            self._load_model()
            
            # Audio laden
            logger.info(f"Lade Audio-Datei: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Auf GPU verschieben falls verfügbar
            waveform = waveform.to(self.device)
            
            # Separation durchführen
            logger.info("Starte Audio-Separation...")
            with torch.no_grad():
                sources = apply_model(self.model, waveform.unsqueeze(0), device=self.device)
            
            # Stems speichern
            output_paths = {}
            stem_names = ["drums", "bass", "other", "vocals"]  # Standard Demucs-Ausgabe
            
            for i, stem_name in enumerate(stem_names):
                if i < sources.shape[1]:  # Prüfen ob Stem existiert
                    stem_audio = sources[0, i].cpu()  # Zurück zur CPU für Speicherung
                    output_path = os.path.join(output_dir, f"{stem_name}.wav")
                    
                    # Audio speichern
                    save_audio(stem_audio, output_path, sample_rate, clip="rescale")
                    output_paths[stem_name] = output_path
                    
                    logger.info(f"Stem gespeichert: {stem_name} -> {output_path}")
            
            logger.info(f"Audio-Separation abgeschlossen. {len(output_paths)} Stems erstellt.")
            return output_paths
            
        except Exception as e:
            logger.error(f"Fehler bei Audio-Separation: {e}")
            raise
    
    async def separate_track(self, file_path: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Zerlegt einen Stereo-Track in seine Kern-Stems
        
        Args:
            file_path: Pfad zur Audio-Datei
            output_dir: Ausgabe-Verzeichnis (optional, verwendet temp_dir falls None)
            
        Returns:
            Dictionary mit Stem-Namen als Keys und Dateipfaden als Values
            
        Raises:
            FileNotFoundError: Wenn die Audio-Datei nicht existiert
            Exception: Bei Fehlern während der Separation
        """
        # Eingabe validieren
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {file_path}")
        
        # Ausgabe-Verzeichnis bestimmen
        if output_dir is None:
            # Temporäres Verzeichnis für diese Separation
            track_name = Path(file_path).stem
            output_dir = self.temp_dir / f"separation_{track_name}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starte Separation für: {file_path}")
        logger.info(f"Ausgabe-Verzeichnis: {output_dir}")
        
        try:
            # Separation in ThreadPoolExecutor ausführen (CPU-intensive)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._separate_audio_sync, 
                file_path, 
                str(output_dir)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Separation fehlgeschlagen für {file_path}: {e}")
            # Aufräumen bei Fehler
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            raise
    
    async def separate_multiple_tracks(self, file_paths: List[str], 
                                     output_base_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """
        Zerlegt mehrere Tracks parallel
        
        Args:
            file_paths: Liste von Audio-Dateipfaden
            output_base_dir: Basis-Ausgabe-Verzeichnis
            
        Returns:
            Dictionary mit Dateipfad als Key und Stem-Dictionary als Value
        """
        if output_base_dir is None:
            output_base_dir = self.temp_dir / "batch_separation"
        
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Sequenzielle Verarbeitung (um GPU-Memory zu schonen)
        for file_path in file_paths:
            try:
                track_name = Path(file_path).stem
                track_output_dir = output_base_dir / track_name
                
                stem_paths = await self.separate_track(file_path, str(track_output_dir))
                results[file_path] = stem_paths
                
            except Exception as e:
                logger.error(f"Fehler bei Separation von {file_path}: {e}")
                results[file_path] = {}
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """
        Gibt unterstützte Audio-Formate zurück
        
        Returns:
            Liste der unterstützten Dateierweiterungen
        """
        return [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
    
    def cleanup_temp_files(self, older_than_hours: int = 24) -> None:
        """
        Räumt temporäre Dateien auf
        
        Args:
            older_than_hours: Dateien älter als diese Stunden werden gelöscht
        """
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)
            
            for item in self.temp_dir.iterdir():
                if item.is_dir():
                    # Prüfe Erstellungszeit des Verzeichnisses
                    if item.stat().st_ctime < cutoff_time:
                        shutil.rmtree(item, ignore_errors=True)
                        logger.info(f"Temporäres Verzeichnis gelöscht: {item}")
                        
        except Exception as e:
            logger.warning(f"Fehler beim Aufräumen temporärer Dateien: {e}")
    
    def __del__(self):
        """Cleanup beim Zerstören der Instanz"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)