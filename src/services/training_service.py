"""Training Service für VAE-Modelle

Dieser Service trainiert Variational Autoencoders (VAEs) auf Audio-Stems
für die intelligente Hybridisierung und Generierung neuer Stems.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from core.config import settings
from core.logging import get_logger
from database.service import DatabaseService

logger = get_logger(__name__)


class AudioVAE(nn.Module):
    """Variational Autoencoder für Audio-Spektrogramme"""
    
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dims: List[int] = None):
        super(AudioVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Ausgabe zwischen 0 und 1
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encodiert Input zu latenten Parametern"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization Trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decodiert latente Repräsentation zurück zu Audio"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward Pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class AudioDataset(Dataset):
    """Dataset für Audio-Spektrogramme"""
    
    def __init__(self, audio_paths: List[str], transform=None):
        self.audio_paths = audio_paths
        self.transform = transform
        self.spectrograms = []
        self.scaler = StandardScaler()
        
        # Spektrogramme vorberechnen
        self._precompute_spectrograms()
    
    def _precompute_spectrograms(self):
        """Berechnet Spektrogramme für alle Audio-Dateien vor"""
        logger.info(f"Berechne Spektrogramme für {len(self.audio_paths)} Audio-Dateien...")
        
        all_spectrograms = []
        
        for audio_path in tqdm(self.audio_paths, desc="Spektrogramme berechnen"):
            try:
                # Audio laden
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Auf Mono reduzieren falls Stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Spektrogramm berechnen
                spectrogram_transform = torchaudio.transforms.Spectrogram(
                    n_fft=1024,
                    hop_length=512,
                    power=2.0
                )
                
                spectrogram = spectrogram_transform(waveform)
                
                # Log-Spektrogramm für bessere Repräsentation
                log_spectrogram = torch.log(spectrogram + 1e-8)
                
                # Flatten für VAE
                flattened = log_spectrogram.flatten().numpy()
                all_spectrograms.append(flattened)
                
            except Exception as e:
                logger.warning(f"Fehler beim Verarbeiten von {audio_path}: {e}")
                continue
        
        if all_spectrograms:
            # Normalisierung
            all_spectrograms = np.array(all_spectrograms)
            self.spectrograms = self.scaler.fit_transform(all_spectrograms)
            logger.info(f"Spektrogramme berechnet: {self.spectrograms.shape}")
        else:
            logger.error("Keine gültigen Spektrogramme gefunden")
            self.spectrograms = np.array([])
    
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        spectrogram = torch.FloatTensor(self.spectrograms[idx])
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram


class TrainingService:
    """Service für das Training von VAE-Modellen"""
    
    def __init__(self):
        self.db_service = DatabaseService()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Verzeichnisse
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"TrainingService initialisiert, Device: {self.device}")
    
    def vae_loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """VAE Loss Function (Reconstruction + KL Divergence)"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def _train_vae_sync(self, category: str, audio_paths: List[str], 
                       epochs: int = 100, batch_size: int = 32, 
                       learning_rate: float = 1e-3, latent_dim: int = 128) -> Dict[str, Any]:
        """Synchrones VAE-Training (läuft in ThreadPoolExecutor)"""
        try:
            logger.info(f"Starte VAE-Training für Kategorie: {category}")
            logger.info(f"Anzahl Audio-Dateien: {len(audio_paths)}")
            
            # Dataset erstellen
            dataset = AudioDataset(audio_paths)
            
            if len(dataset) == 0:
                raise ValueError("Keine gültigen Audio-Dateien für Training gefunden")
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Modell erstellen
            input_dim = dataset.spectrograms.shape[1]
            model = AudioVAE(input_dim=input_dim, latent_dim=latent_dim)
            model.to(self.device)
            
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training Loop
            model.train()
            training_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_recon_loss = 0.0
                epoch_kl_loss = 0.0
                
                for batch_idx, data in enumerate(dataloader):
                    data = data.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    recon_batch, mu, logvar = model(data)
                    
                    # Loss berechnen
                    loss, recon_loss, kl_loss = self.vae_loss_function(
                        recon_batch, data, mu, logvar
                    )
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                
                # Durchschnittliche Losses
                avg_loss = epoch_loss / len(dataloader.dataset)
                avg_recon_loss = epoch_recon_loss / len(dataloader.dataset)
                avg_kl_loss = epoch_kl_loss / len(dataloader.dataset)
                
                training_losses.append({
                    'epoch': epoch + 1,
                    'total_loss': avg_loss,
                    'recon_loss': avg_recon_loss,
                    'kl_loss': avg_kl_loss
                })
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, "
                              f"Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")
            
            # Modell speichern
            model_path = self.models_dir / f"{category}_vae.pt"
            scaler_path = self.models_dir / f"{category}_scaler.pkl"
            metadata_path = self.models_dir / f"{category}_metadata.json"
            
            # Modell-State speichern
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'training_losses': training_losses
            }, model_path)
            
            # Scaler speichern
            with open(scaler_path, 'wb') as f:
                pickle.dump(dataset.scaler, f)
            
            # Metadaten speichern
            metadata = {
                'category': category,
                'training_date': datetime.now().isoformat(),
                'num_samples': len(dataset),
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'final_loss': training_losses[-1]['total_loss'],
                'device': self.device
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"VAE-Training abgeschlossen für {category}")
            logger.info(f"Modell gespeichert: {model_path}")
            
            return {
                'success': True,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'metadata_path': str(metadata_path),
                'final_loss': training_losses[-1]['total_loss'],
                'training_losses': training_losses
            }
            
        except Exception as e:
            logger.error(f"Fehler beim VAE-Training für {category}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def train_vae_for_category(self, category: str, 
                                   epochs: int = 100, 
                                   batch_size: int = 32,
                                   learning_rate: float = 1e-3,
                                   latent_dim: int = 128) -> Dict[str, Any]:
        """
        Trainiert ein VAE-Modell für eine bestimmte Kategorie
        
        Args:
            category: Stem-Kategorie (z.B. "kick", "bass", "hihat")
            epochs: Anzahl Trainings-Epochen
            batch_size: Batch-Größe
            learning_rate: Lernrate
            latent_dim: Dimensionalität des latenten Raums
            
        Returns:
            Dictionary mit Trainings-Ergebnissen
        """
        logger.info(f"Sammle Stems für Kategorie: {category}")
        
        # Stems aus Datenbank sammeln
        stems = await self.db_service.get_stems_by_category(category)
        
        if not stems:
            return {
                'success': False,
                'error': f'Keine Stems für Kategorie "{category}" gefunden'
            }
        
        # Pfade extrahieren
        audio_paths = []
        for stem in stems:
            if hasattr(stem, 'processed_path') and stem.processed_path:
                if os.path.exists(stem.processed_path):
                    audio_paths.append(stem.processed_path)
            elif hasattr(stem, 'original_path') and stem.original_path:
                if os.path.exists(stem.original_path):
                    audio_paths.append(stem.original_path)
        
        if not audio_paths:
            return {
                'success': False,
                'error': f'Keine gültigen Audio-Pfade für Kategorie "{category}" gefunden'
            }
        
        logger.info(f"Gefunden: {len(audio_paths)} Audio-Dateien für Training")
        
        # Training in ThreadPoolExecutor ausführen
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._train_vae_sync,
            category, audio_paths, epochs, batch_size, learning_rate, latent_dim
        )
        
        return result
    
    async def train_all_categories(self, min_samples: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Trainiert VAE-Modelle für alle Kategorien mit genügend Samples
        
        Args:
            min_samples: Mindestanzahl Samples pro Kategorie
            
        Returns:
            Dictionary mit Ergebnissen pro Kategorie
        """
        # Verfügbare Kategorien ermitteln
        categories = await self.db_service.get_categories()
        
        results = {}
        
        for category in categories:
            # Anzahl Stems prüfen
            stems = await self.db_service.get_stems_by_category(category)
            
            if len(stems) >= min_samples:
                logger.info(f"Starte Training für Kategorie: {category} ({len(stems)} Samples)")
                result = await self.train_vae_for_category(category)
                results[category] = result
            else:
                logger.info(f"Überspringe Kategorie {category}: nur {len(stems)} Samples (min: {min_samples})")
                results[category] = {
                    'success': False,
                    'error': f'Nicht genügend Samples: {len(stems)} < {min_samples}'
                }
        
        return results
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Gibt Liste aller verfügbaren trainierten Modelle zurück
        
        Returns:
            Liste mit Modell-Informationen
        """
        models = []
        
        for model_file in self.models_dir.glob("*_vae.pt"):
            category = model_file.stem.replace("_vae", "")
            metadata_file = self.models_dir / f"{category}_metadata.json"
            
            model_info = {
                'category': category,
                'model_path': str(model_file),
                'exists': model_file.exists()
            }
            
            # Metadaten laden falls verfügbar
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    model_info.update(metadata)
                except Exception as e:
                    logger.warning(f"Fehler beim Laden der Metadaten für {category}: {e}")
            
            models.append(model_info)
        
        return models
    
    def __del__(self):
        """Cleanup beim Zerstören der Instanz"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)