#!/usr/bin/env python3
"""
Optimized Preprocessor for AGENTEN_DIREKTIVE_008
Generiert CLAP-Embeddings für große Datensätze (500+ Stems)
Mit optimierter Batch-Verarbeitung und Fortschrittsanzeige
"""

import os
import pickle
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import laion_clap
import torch
import gc
from pathlib import Path

class OptimizedPreprocessor:
    def __init__(self, model_name=None, batch_size=8):
        """
        Initialisiert den optimierten Preprocessor
        
        Args:
            model_name: CLAP-Modell Name (None für automatische Auswahl)
            batch_size: Kleinere Batch-Größe für Stabilität
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Verwende Device: {self.device}")
        
    def load_model(self):
        """Lädt das CLAP-Modell"""
        print("🤖 Lade CLAP-Modell...")
        try:
            self.model = laion_clap.CLAP_Module(enable_fusion=False, device=self.device)
            
            # Verwende Standard-Modell wenn keines angegeben
            if self.model_name is None:
                print("📦 Verwende Standard CLAP-Modell")
                # Lade ohne spezifisches Checkpoint (verwendet vortrainiertes Modell)
                pass
            else:
                self.model.load_ckpt(self.model_name)
                
            print("✅ CLAP-Modell erfolgreich geladen.")
        except Exception as e:
            print(f"❌ Fehler beim Laden des Modells: {e}")
            raise
    
    def find_audio_files(self, directory):
        """Findet alle Audio-Dateien rekursiv"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        audio_files = []
        
        print(f"🔍 Suche Audio-Dateien in: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        print(f"📁 Gefunden: {len(audio_files)} Audio-Dateien")
        return audio_files
    
    def load_audio_safe(self, file_path, target_sr=48000, max_duration=30):
        """Lädt Audio-Datei sicher mit Fehlerbehandlung"""
        try:
            # Lade Audio mit librosa
            audio, sr = librosa.load(file_path, sr=target_sr, duration=max_duration)
            
            # Normalisierung
            if len(audio) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            return audio, sr
        except Exception as e:
            print(f"⚠️  Fehler beim Laden von {file_path}: {e}")
            return None, None
    
    def process_batch(self, audio_files_batch):
        """Verarbeitet einen Batch von Audio-Dateien"""
        audio_data = []
        valid_files = []
        
        # Lade Audio-Dateien für den Batch
        for file_path in audio_files_batch:
            audio, sr = self.load_audio_safe(file_path)
            if audio is not None:
                audio_data.append(audio)
                valid_files.append(file_path)
        
        if not audio_data:
            return [], []
        
        try:
            # Generiere Embeddings für den Batch
            with torch.no_grad():
                embeddings = self.model.get_audio_embedding_from_data(
                    x=audio_data, 
                    use_tensor=False
                )
            
            # Garbage Collection
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return embeddings, valid_files
            
        except Exception as e:
            print(f"⚠️  Batch-Verarbeitung fehlgeschlagen: {e}")
            # Fallback: Einzelverarbeitung
            return self.process_individual(valid_files, audio_data)
    
    def process_individual(self, files, audio_data):
        """Fallback: Verarbeitet Dateien einzeln"""
        embeddings = []
        valid_files = []
        
        for file_path, audio in zip(files, audio_data):
            try:
                with torch.no_grad():
                    embedding = self.model.get_audio_embedding_from_data(
                        x=[audio], 
                        use_tensor=False
                    )
                embeddings.extend(embedding)
                valid_files.append(file_path)
                
                # Cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"⚠️  Einzelverarbeitung fehlgeschlagen für {file_path}: {e}")
                continue
        
        return embeddings, valid_files
    
    def process_audio_files(self, audio_files, output_path):
        """Verarbeitet alle Audio-Dateien in Batches"""
        if not self.model:
            self.load_model()
        
        all_embeddings = []
        all_file_paths = []
        
        # Verarbeite in Batches
        print(f"🔄 Verarbeite {len(audio_files)} Dateien in Batches von {self.batch_size}...")
        
        for i in tqdm(range(0, len(audio_files), self.batch_size), desc="Batches"):
            batch = audio_files[i:i + self.batch_size]
            
            embeddings, valid_files = self.process_batch(batch)
            
            if len(embeddings) > 0:
                all_embeddings.extend(embeddings)
                all_file_paths.extend(valid_files)
            
            # Fortschritt anzeigen
            if (i // self.batch_size + 1) % 10 == 0:
                print(f"📊 Verarbeitet: {len(all_embeddings)} von {len(audio_files)} Dateien")
        
        # Speichere Ergebnisse
        self.save_embeddings(all_embeddings, all_file_paths, output_path)
        
        return len(all_embeddings)
    
    def save_embeddings(self, embeddings, file_paths, output_path):
        """Speichert Embeddings und Dateipfade"""
        # Erstelle Output-Verzeichnis
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Konvertiere zu numpy arrays
        embeddings_array = np.array(embeddings)
        
        # Speichere als Pickle
        data = {
            'embeddings': embeddings_array,
            'file_paths': file_paths,
            'model_name': self.model_name,
            'total_files': len(file_paths)
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Embeddings gespeichert: {output_path}")
        print(f"📈 Verarbeitete Dateien: {len(file_paths)}")
        print(f"🔢 Embedding-Dimensionen: {embeddings_array.shape}")

def main():
    """Hauptfunktion für AGENTEN_DIREKTIVE_008"""
    print("🎵 Optimized Preprocessor - AGENTEN_DIREKTIVE_008")
    print("=" * 50)
    
    # Konfiguration
    input_dir = "raw_construction_kits"
    output_path = "processed_database/embeddings.pkl"
    batch_size = 4  # Kleinere Batches für Stabilität
    
    # Initialisiere Preprocessor
    preprocessor = OptimizedPreprocessor(batch_size=batch_size)
    
    # Finde Audio-Dateien
    audio_files = preprocessor.find_audio_files(input_dir)
    
    if not audio_files:
        print("❌ Keine Audio-Dateien gefunden!")
        return
    
    # Verarbeite Dateien
    processed_count = preprocessor.process_audio_files(audio_files, output_path)
    
    print("\n✅ Verarbeitung abgeschlossen!")
    print(f"📊 Erfolgreich verarbeitet: {processed_count}/{len(audio_files)} Dateien")
    print(f"💾 Embeddings gespeichert in: {output_path}")
    print("\n🎉 Embedding-Datenbank für 500+ Stems erstellt!")
    print("Nächster Schritt: Teste die Search Engine mit dem erweiterten Datensatz")

if __name__ == "__main__":
    main()