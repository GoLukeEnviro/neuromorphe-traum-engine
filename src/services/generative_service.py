"""Generative Service für VAE-basierte Stem-Erzeugung

Dieser Service nutzt trainierte VAE-Modelle zur Generierung neuer Audio-Stems
durch Sampling und Interpolation im latenten Raum.
"""

import os
import json
import pickle
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import istft

from ..core.config import settings
from ..core.logging import get_logger
from ..database.service import DatabaseService
from .training_service import AudioVAE
logger = get_logger(__name__)


class GenerativeService:
    """Service für die Generierung neuer Audio-Stems mit VAE-Modellen"""
    
    def __init__(self):
        self.db_service = DatabaseService()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Verzeichnisse
        self.models_dir = Path("models")
        self.generated_stems_dir = Path("generated_stems")
        self.generated_stems_dir.mkdir(exist_ok=True)
        
        # Cache für geladene Modelle
        self._model_cache = {}
        self._scaler_cache = {}
        
        logger.info(f"GenerativeService initialisiert, Device: {self.device}")
    
    def _load_model_and_scaler(self, category: str) -> Tuple[AudioVAE, StandardScaler, Dict[str, Any]]:
        """
        Lädt VAE-Modell und Scaler für eine Kategorie
        
        Args:
            category: Stem-Kategorie
            
        Returns:
            Tuple aus (model, scaler, metadata)
        """
        # Aus Cache laden falls verfügbar
        if category in self._model_cache:
            return (
                self._model_cache[category],
                self._scaler_cache[category],
                self._model_cache[f"{category}_metadata"]
            )
        
        model_path = self.models_dir / f"{category}_vae.pt"
        scaler_path = self.models_dir / f"{category}_scaler.pkl"
        metadata_path = self.models_dir / f"{category}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"VAE-Modell für Kategorie '{category}' nicht gefunden: {model_path}")
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler für Kategorie '{category}' nicht gefunden: {scaler_path}")
        
        # Modell laden
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = AudioVAE(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Scaler laden
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Metadaten laden
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # In Cache speichern
        self._model_cache[category] = model
        self._scaler_cache[category] = scaler
        self._model_cache[f"{category}_metadata"] = metadata
        
        logger.info(f"VAE-Modell für '{category}' geladen")
        return model, scaler, metadata
    
    def _spectrogram_to_audio(self, spectrogram: np.ndarray, 
                            sample_rate: int = 44100, 
                            n_fft: int = 1024, 
                            hop_length: int = 512) -> np.ndarray:
        """
        Konvertiert Spektrogramm zurück zu Audio-Waveform
        
        Args:
            spectrogram: Log-Spektrogramm
            sample_rate: Sample-Rate
            n_fft: FFT-Größe
            hop_length: Hop-Länge
            
        Returns:
            Audio-Waveform als numpy array
        """
        try:
            # Spektrogramm-Shape wiederherstellen
            # Annahme: Spektrogramm wurde als (freq_bins, time_frames) gespeichert
            freq_bins = (n_fft // 2) + 1
            
            # Spektrogramm reshapen
            if len(spectrogram.shape) == 1:
                # Flattened -> 2D
                time_frames = len(spectrogram) // freq_bins
                spectrogram = spectrogram[:freq_bins * time_frames].reshape(freq_bins, time_frames)
            
            # Aus Log-Spektrogramm zurück zu linearem Spektrogramm
            linear_spectrogram = np.exp(spectrogram) - 1e-8
            
            # Phase-Rekonstruktion mit Griffin-Lim Algorithmus
            # Für einfachheit verwenden wir zufällige Phase
            phase = np.random.uniform(0, 2*np.pi, linear_spectrogram.shape)
            complex_spectrogram = linear_spectrogram * np.exp(1j * phase)
            
            # ISTFT für Audio-Rekonstruktion
            _, audio = istft(
                complex_spectrogram,
                fs=sample_rate,
                window='hann',
                nperseg=n_fft,
                noverlap=n_fft - hop_length
            )
            
            # Normalisierung
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Fehler bei Spektrogramm-zu-Audio Konvertierung: {e}")
            # Fallback: Weißes Rauschen
            duration = 2.0  # 2 Sekunden
            samples = int(sample_rate * duration)
            return np.random.normal(0, 0.1, samples).astype(np.float32)
    
    def _generate_stems_sync(self, category: str, num_variations: int, 
                           generation_mode: str = "random", 
                           interpolation_factor: float = 0.5) -> List[Dict[str, Any]]:
        """
        Synchrone Stem-Generierung (läuft in ThreadPoolExecutor)
        
        Args:
            category: Stem-Kategorie
            num_variations: Anzahl zu generierender Variationen
            generation_mode: "random", "interpolate", oder "hybrid"
            interpolation_factor: Faktor für Interpolation (0.0 - 1.0)
            
        Returns:
            Liste mit generierten Stem-Informationen
        """
        try:
            # Modell und Scaler laden
            model, scaler, metadata = self._load_model_and_scaler(category)
            
            generated_stems = []
            
            with torch.no_grad():
                for i in range(num_variations):
                    # Latenten Vektor generieren
                    if generation_mode == "random":
                        # Zufälliger latenter Vektor
                        z = torch.randn(1, model.latent_dim).to(self.device)
                    
                    elif generation_mode == "interpolate":
                        # Interpolation zwischen zwei zufälligen Punkten
                        z1 = torch.randn(1, model.latent_dim).to(self.device)
                        z2 = torch.randn(1, model.latent_dim).to(self.device)
                        z = interpolation_factor * z1 + (1 - interpolation_factor) * z2
                    
                    elif generation_mode == "hybrid":
                        # Mischung aus strukturiertem und zufälligem Sampling
                        if i % 2 == 0:
                            z = torch.randn(1, model.latent_dim).to(self.device)
                        else:
                            z1 = torch.randn(1, model.latent_dim).to(self.device)
                            z2 = torch.randn(1, model.latent_dim).to(self.device)
                            factor = np.random.uniform(0.2, 0.8)
                            z = factor * z1 + (1 - factor) * z2
                    
                    else:
                        # Fallback: random
                        z = torch.randn(1, model.latent_dim).to(self.device)
                    
                    # Dekodierung
                    generated_spectrogram = model.decode(z)
                    
                    # Zurück zu numpy und denormalisieren
                    spectrogram_np = generated_spectrogram.cpu().numpy().flatten()
                    denormalized_spectrogram = scaler.inverse_transform([spectrogram_np])[0]
                    
                    # Spektrogramm zu Audio konvertieren
                    audio = self._spectrogram_to_audio(denormalized_spectrogram)
                    
                    # Dateiname generieren
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    filename = f"{category}_generated_{timestamp}_{unique_id}_{i+1:03d}.wav"
                    filepath = self.generated_stems_dir / filename
                    
                    # Audio speichern
                    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
                    torchaudio.save(str(filepath), audio_tensor, 44100)
                    
                    # Stem-Informationen sammeln
                    stem_info = {
                        'filename': filename,
                        'filepath': str(filepath),
                        'category': category,
                        'source': 'generated',
                        'generation_mode': generation_mode,
                        'variation_index': i + 1,
                        'latent_vector': z.cpu().numpy().tolist(),
                        'generation_timestamp': datetime.now().isoformat(),
                        'model_metadata': metadata
                    }
                    
                    generated_stems.append(stem_info)
                    
                    logger.debug(f"Generiert: {filename}")
            
            logger.info(f"Erfolgreich {len(generated_stems)} Stems für '{category}' generiert")
            return generated_stems
            
        except Exception as e:
            logger.error(f"Fehler bei Stem-Generierung für '{category}': {e}")
            raise
    
    async def mutate_stems(self, category: str, num_variations: int = 5, 
                          generation_mode: str = "random", 
                          interpolation_factor: float = 0.5) -> Dict[str, Any]:
        """
        Generiert neue Stem-Variationen für eine Kategorie
        
        Args:
            category: Stem-Kategorie (z.B. "kick", "bass", "hihat")
            num_variations: Anzahl zu generierender Variationen
            generation_mode: Generierungsmodus ("random", "interpolate", "hybrid")
            interpolation_factor: Faktor für Interpolation (0.0 - 1.0)
            
        Returns:
            Dictionary mit Generierungs-Ergebnissen
        """
        logger.info(f"Starte Stem-Generierung für '{category}' ({num_variations} Variationen)")
        
        try:
            # Prüfen ob Modell existiert
            model_path = self.models_dir / f"{category}_vae.pt"
            if not model_path.exists():
                return {
                    'success': False,
                    'error': f'Kein trainiertes VAE-Modell für Kategorie "{category}" gefunden. '
                            f'Bitte zuerst Training durchführen.'
                }
            
            # Generierung in ThreadPoolExecutor ausführen
            loop = asyncio.get_event_loop()
            generated_stems = await loop.run_in_executor(
                self.executor,
                self._generate_stems_sync,
                category, num_variations, generation_mode, interpolation_factor
            )
            
            return {
                'success': True,
                'category': category,
                'num_generated': len(generated_stems),
                'generation_mode': generation_mode,
                'generated_stems': generated_stems,
                'output_directory': str(self.generated_stems_dir)
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Stem-Generierung: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_hybrid_stems(self, category1: str, category2: str, 
                                  num_variations: int = 5, 
                                  blend_ratios: List[float] = None) -> Dict[str, Any]:
        """
        Generiert Hybrid-Stems durch Interpolation zwischen zwei Kategorien
        
        Args:
            category1: Erste Stem-Kategorie
            category2: Zweite Stem-Kategorie
            num_variations: Anzahl zu generierender Variationen
            blend_ratios: Liste von Mischungsverhältnissen (0.0 - 1.0)
            
        Returns:
            Dictionary mit Generierungs-Ergebnissen
        """
        if blend_ratios is None:
            blend_ratios = np.linspace(0.1, 0.9, num_variations).tolist()
        
        logger.info(f"Starte Hybrid-Generierung: {category1} + {category2}")
        
        try:
            # Beide Modelle laden
            model1, scaler1, metadata1 = self._load_model_and_scaler(category1)
            model2, scaler2, metadata2 = self._load_model_and_scaler(category2)
            
            # Prüfen ob latente Dimensionen kompatibel sind
            if model1.latent_dim != model2.latent_dim:
                return {
                    'success': False,
                    'error': f'Inkompatible latente Dimensionen: {category1}({model1.latent_dim}) '
                            f'vs {category2}({model2.latent_dim})'
                }
            
            generated_stems = []
            
            with torch.no_grad():
                for i, blend_ratio in enumerate(blend_ratios[:num_variations]):
                    # Latente Vektoren für beide Kategorien sampeln
                    z1 = torch.randn(1, model1.latent_dim).to(self.device)
                    z2 = torch.randn(1, model2.latent_dim).to(self.device)
                    
                    # Interpolation im latenten Raum
                    z_hybrid = blend_ratio * z1 + (1 - blend_ratio) * z2
                    
                    # Mit beiden Modellen dekodieren und mischen
                    output1 = model1.decode(z_hybrid)
                    output2 = model2.decode(z_hybrid)
                    
                    # Spektrogramme mischen
                    hybrid_spectrogram = blend_ratio * output1 + (1 - blend_ratio) * output2
                    
                    # Denormalisierung (verwende Scaler von category1)
                    spectrogram_np = hybrid_spectrogram.cpu().numpy().flatten()
                    denormalized_spectrogram = scaler1.inverse_transform([spectrogram_np])[0]
                    
                    # Zu Audio konvertieren
                    audio = self._spectrogram_to_audio(denormalized_spectrogram)
                    
                    # Dateiname generieren
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    filename = f"hybrid_{category1}_{category2}_{timestamp}_{unique_id}_{i+1:03d}.wav"
                    filepath = self.generated_stems_dir / filename
                    
                    # Audio speichern
                    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
                    torchaudio.save(str(filepath), audio_tensor, 44100)
                    
                    # Stem-Informationen sammeln
                    stem_info = {
                        'filename': filename,
                        'filepath': str(filepath),
                        'category': f"hybrid_{category1}_{category2}",
                        'source': 'generated',
                        'generation_mode': 'hybrid_interpolation',
                        'category1': category1,
                        'category2': category2,
                        'blend_ratio': blend_ratio,
                        'variation_index': i + 1,
                        'generation_timestamp': datetime.now().isoformat()
                    }
                    
                    generated_stems.append(stem_info)
                    
                    logger.debug(f"Hybrid generiert: {filename} (Ratio: {blend_ratio:.2f})")
            
            logger.info(f"Erfolgreich {len(generated_stems)} Hybrid-Stems generiert")
            
            return {
                'success': True,
                'category1': category1,
                'category2': category2,
                'num_generated': len(generated_stems),
                'generated_stems': generated_stems,
                'output_directory': str(self.generated_stems_dir)
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Hybrid-Generierung: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def batch_generate_all_categories(self, num_variations_per_category: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Generiert Stems für alle verfügbaren trainierten Kategorien
        
        Args:
            num_variations_per_category: Anzahl Variationen pro Kategorie
            
        Returns:
            Dictionary mit Ergebnissen pro Kategorie
        """
        # Verfügbare Modelle ermitteln
        available_models = []
        for model_file in self.models_dir.glob("*_vae.pt"):
            category = model_file.stem.replace("_vae", "")
            available_models.append(category)
        
        if not available_models:
            return {
                'error': 'Keine trainierten VAE-Modelle gefunden'
            }
        
        logger.info(f"Starte Batch-Generierung für {len(available_models)} Kategorien")
        
        results = {}
        
        for category in available_models:
            logger.info(f"Generiere Stems für Kategorie: {category}")
            result = await self.mutate_stems(category, num_variations_per_category)
            results[category] = result
        
        return results
    
    def get_generated_stems_info(self) -> List[Dict[str, Any]]:
        """
        Gibt Informationen über alle generierten Stems zurück
        
        Returns:
            Liste mit Stem-Informationen
        """
        stems_info = []
        
        for audio_file in self.generated_stems_dir.glob("*.wav"):
            file_stats = audio_file.stat()
            
            stem_info = {
                'filename': audio_file.name,
                'filepath': str(audio_file),
                'size_bytes': file_stats.st_size,
                'created_timestamp': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'modified_timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            }
            
            # Kategorie aus Dateiname extrahieren
            if "_generated_" in audio_file.name:
                category = audio_file.name.split("_generated_")[0]
                stem_info['category'] = category
                stem_info['source'] = 'generated'
            elif "hybrid_" in audio_file.name:
                stem_info['source'] = 'hybrid'
                # Kategorien aus Hybrid-Namen extrahieren
                parts = audio_file.name.split("_")
                if len(parts) >= 4:
                    stem_info['category1'] = parts[1]
                    stem_info['category2'] = parts[2]
            
            stems_info.append(stem_info)
        
        return sorted(stems_info, key=lambda x: x['created_timestamp'], reverse=True)
    
    def clear_cache(self):
        """Leert den Modell-Cache"""
        self._model_cache.clear()
        self._scaler_cache.clear()
        logger.info("Modell-Cache geleert")
    
    def __del__(self):
        """Cleanup beim Zerstören der Instanz"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)