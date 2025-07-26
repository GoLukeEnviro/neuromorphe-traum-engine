#!/usr/bin/env python3
"""
Minimal Preprocessor für Neuromorphe Traum-Engine v2.0
AGENTEN_DIREKTIVE_007 - Master MVP

Erstellt CLAP-Embeddings für Audio-Dateien und speichert sie in einer kompakten Binärdatei.
"""

import os
import pickle
import librosa
import torch
import numpy as np
from tqdm import tqdm
import laion_clap


class MinimalPreprocessor:
    def __init__(self, input_dir, output_path):
        """
        Initialisiert den MinimalPreprocessor.
        
        Args:
            input_dir (str): Verzeichnis mit den Audio-Dateien
            output_path (str): Pfad für die Ausgabe-Datei (embeddings.pkl)
        """
        self.input_dir = input_dir
        self.output_path = output_path
        
        print("Lade CLAP-Modell...")
        # Lade CLAP-Modell einmalig
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()  # Lädt das vortrainierte Modell
        print("CLAP-Modell erfolgreich geladen.")
    
    def _find_audio_files(self):
        """
        Sucht rekursiv alle .wav-Dateien im input_dir.
        
        Returns:
            list: Liste aller gefundenen .wav-Dateipfade
        """
        audio_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        return audio_files
    
    def _process_batch(self, file_batch):
        """
        Verarbeitet einen Batch von Audio-Dateien.
        
        Args:
            file_batch (list): Liste von Dateipfaden
            
        Returns:
            list: Liste von Dictionaries mit 'path' und 'embedding'
        """
        try:
            # Berechne Audio-Embeddings für den gesamten Batch
            embeddings = self.clap_model.get_audio_embedding_from_filelist(
                x=file_batch, 
                use_tensor=False
            )
            
            # Erstelle Ergebnis-Liste
            results = []
            for i, file_path in enumerate(file_batch):
                results.append({
                    'path': file_path,
                    'embedding': embeddings[i]
                })
            
            return results
            
        except Exception as e:
            print(f"Fehler beim Verarbeiten des Batches: {e}")
            # Fallback: Verarbeite Dateien einzeln
            results = []
            for file_path in file_batch:
                try:
                    embedding = self.clap_model.get_audio_embedding_from_filelist(
                        x=[file_path], 
                        use_tensor=False
                    )[0]
                    results.append({
                        'path': file_path,
                        'embedding': embedding
                    })
                except Exception as single_error:
                    print(f"Fehler bei Datei {file_path}: {single_error}")
                    continue
            
            return results
    
    def run(self):
        """
        Hauptverarbeitungsschleife: Findet Audio-Dateien, verarbeitet sie in Batches
        und speichert die Embeddings.
        """
        print(f"Suche Audio-Dateien in: {self.input_dir}")
        audio_files = self._find_audio_files()
        
        if not audio_files:
            print("Keine .wav-Dateien gefunden!")
            return
        
        print(f"Gefunden: {len(audio_files)} Audio-Dateien")
        
        # Batch-Größe
        batch_size = 32
        all_results = []
        
        # Verarbeite Dateien in Batches
        for i in tqdm(range(0, len(audio_files), batch_size), 
                     desc="Verarbeite Audio-Batches"):
            batch = audio_files[i:i + batch_size]
            batch_results = self._process_batch(batch)
            all_results.extend(batch_results)
        
        # Stelle sicher, dass das Ausgabeverzeichnis existiert
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Speichere alle Ergebnisse als Pickle-Datei
        print(f"Speichere {len(all_results)} Embeddings in: {self.output_path}")
        with open(self.output_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        print("Verarbeitung abgeschlossen!")


if __name__ == "__main__":
    # Konfiguration
    input_directory = "raw_construction_kits"
    output_file = "processed_database/embeddings.pkl"
    
    # Erstelle und führe Preprocessor aus
    preprocessor = MinimalPreprocessor(input_directory, output_file)
    preprocessor.run()