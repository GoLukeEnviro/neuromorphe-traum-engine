#!/usr/bin/env python3
"""
Search Engine CLI für Neuromorphe Traum-Engine v2.0
AGENTEN_DIREKTIVE_007 - Master MVP

Interaktive Kommandozeilen-Suchmaschine für semantische Audio-Suche.
"""

import pickle
import torch
import numpy as np
import laion_clap
import torch.nn.functional as F


class SearchEngine:
    def __init__(self, embeddings_path):
        """
        Initialisiert die Suchmaschine.
        
        Args:
            embeddings_path (str): Pfad zur embeddings.pkl Datei
        """
        print("Initialisiere Suchmaschine...")
        
        # Lade CLAP-Modell
        print("Lade CLAP-Modell...")
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()
        print("CLAP-Modell geladen.")
        
        # Lade Embeddings aus Pickle-Datei
        print(f"Lade Embeddings aus: {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        # Extrahiere Dateipfade und Embeddings
        self.file_paths = []
        embeddings_list = []
        
        for item in embeddings_data:
            self.file_paths.append(item['path'])
            embeddings_list.append(item['embedding'])
        
        # Konvertiere zu PyTorch-Tensor für effiziente Berechnung
        self.embedding_tensors = torch.tensor(np.array(embeddings_list), dtype=torch.float32)
        
        print(f"Suchmaschine bereit mit {len(self.file_paths)} Audio-Dateien.")
    
    def search(self, prompt: str, top_k: int = 5):
        """
        Führt eine semantische Suche durch.
        
        Args:
            prompt (str): Text-Prompt für die Suche
            top_k (int): Anzahl der zurückzugebenden Ergebnisse
            
        Returns:
            list: Liste der Top-K Dateipfade
        """
        # Berechne Text-Embedding für den Prompt
        text_embedding = self.clap_model.get_text_embedding([prompt], use_tensor=True)
        
        # Stelle sicher, dass das Text-Embedding die richtige Form hat
        if text_embedding.dim() > 1:
            text_embedding = text_embedding.squeeze(0)
        
        # Berechne Kosinus-Ähnlichkeit zwischen Text-Embedding und allen Audio-Embeddings
        similarities = F.cosine_similarity(
            text_embedding.unsqueeze(0), 
            self.embedding_tensors, 
            dim=1
        )
        
        # Finde die Top-K ähnlichsten Audio-Dateien
        top_k_values, top_k_indices = torch.topk(similarities, k=min(top_k, len(self.file_paths)))
        
        # Erstelle Ergebnis-Liste mit Dateipfaden und Ähnlichkeitswerten
        results = []
        for i, idx in enumerate(top_k_indices):
            file_path = self.file_paths[idx.item()]
            similarity_score = top_k_values[i].item()
            results.append((file_path, similarity_score))
        
        return results


def main():
    """
    Hauptfunktion für die interaktive CLI-Sitzung.
    """
    embeddings_file = "processed_database/embeddings.pkl"
    
    try:
        # Initialisiere Suchmaschine
        search_engine = SearchEngine(embeddings_file)
        
        print("\n" + "="*60)
        print("Neuromorphe Traum-Engine v2.0 - Semantische Audio-Suche")
        print("AGENTEN_DIREKTIVE_007 - Master MVP")
        print("="*60)
        print("Geben Sie einen Text-Prompt ein, um passende Audio-Dateien zu finden.")
        print("Beispiele: 'dark industrial kick', 'melodic synth pad', 'punchy snare'")
        print("Geben Sie 'exit' ein, um das Programm zu beenden.\n")
        
        # Interaktive Schleife
        while True:
            try:
                # Benutzereingabe
                prompt = input("Prompt eingeben (oder 'exit' zum Beenden): ").strip()
                
                # Beenden bei 'exit'
                if prompt.lower() == 'exit':
                    print("Auf Wiedersehen!")
                    break
                
                # Leere Eingabe überspringen
                if not prompt:
                    continue
                
                # Suche durchführen
                print(f"\nSuche nach: '{prompt}'...")
                results = search_engine.search(prompt, top_k=5)
                
                # Ergebnisse anzeigen
                if results:
                    print(f"\nTop 5 Ergebnisse:")
                    print("-" * 50)
                    for i, (file_path, similarity) in enumerate(results, 1):
                        print(f"{i}. {file_path}")
                        print(f"   Ähnlichkeit: {similarity:.4f}")
                        print()
                else:
                    print("Keine Ergebnisse gefunden.")
                
            except KeyboardInterrupt:
                print("\n\nProgramm durch Benutzer unterbrochen.")
                break
            except Exception as e:
                print(f"Fehler bei der Suche: {e}")
                continue
    
    except FileNotFoundError:
        print(f"Fehler: Embeddings-Datei nicht gefunden: {embeddings_file}")
        print("Bitte führen Sie zuerst 'python ai_agents/minimal_preprocessor.py' aus.")
    except Exception as e:
        print(f"Fehler beim Initialisieren der Suchmaschine: {e}")


if __name__ == "__main__":
    main()