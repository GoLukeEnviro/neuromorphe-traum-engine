#!/usr/bin/env python3
"""
Erstellt einfache Test-Audio-Dateien für die CLAP-Embedding-Verarbeitung.
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path

def create_simple_audio(filename: str, duration: float = 1.0, sample_rate: int = 44100, frequency: float = 440.0):
    """Erstellt eine einfache Sinuswelle als Audio-Datei."""
    # Erstelle Verzeichnis falls es nicht existiert
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Generiere Sinuswelle
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.3  # Reduzierte Amplitude
    
    # Speichere als WAV-Datei
    sf.write(filename, audio, sample_rate)
    print(f"Audio-Datei erstellt: {filename}")

def main():
    """Erstellt alle benötigten Test-Audio-Dateien."""
    base_path = Path("../")
    
    # Audio-Dateien mit verschiedenen Frequenzen erstellen
    audio_files = [
        ("test/test_kick.wav", 60.0),  # Tiefe Frequenz für Kick
        ("test_data/kicks/kick_001.wav", 65.0),
        ("test_data/kicks/kick_002.wav", 70.0),
        ("test_data/snares/snare_001.wav", 200.0),  # Mittlere Frequenz für Snare
        ("test_data/hihats/hihat_001.wav", 8000.0),  # Hohe Frequenz für Hi-Hat
        ("test_data/bass/bass_001.wav", 80.0),  # Tiefe Frequenz für Bass
    ]
    
    for filepath, frequency in audio_files:
        full_path = base_path / filepath
        create_simple_audio(str(full_path), duration=2.0, frequency=frequency)
    
    print("\nAlle Test-Audio-Dateien wurden erfolgreich erstellt!")

if __name__ == "__main__":
    main()