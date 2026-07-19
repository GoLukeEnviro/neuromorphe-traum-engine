#!/usr/bin/env python3
"""
Test Audio Generator für AGENTEN_DIREKTIVE_004
Erstellt verschiedene Arten von Audiodateien für erweiterte Qualitätskontrolle und Cluster-basierte Kategorisierung.
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path

def create_kick_sample(filename: str, duration: float = 2.0, sample_rate: int = 48000):
    """
    Erstellt ein synthetisches Kick-Sample mit charakteristischen Eigenschaften:
    - Hohe RMS-Energie
    - Niedrige spektrale Zentroide (Bass-lastig)
    - Kurze, perkussive Hüllkurve
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Kick-Drum Synthese: Sinus-Sweep + Noise Burst
    # Frequenz-Sweep von 80Hz zu 40Hz
    freq_sweep = 80 * np.exp(-t * 8)  # Exponentieller Abfall
    kick_tone = np.sin(2 * np.pi * freq_sweep * t)
    
    # Envelope: Schneller Attack, exponentieller Decay
    envelope = np.exp(-t * 6)
    
    # Noise Burst für Attack
    noise = np.random.normal(0, 0.3, len(t))
    noise_envelope = np.exp(-t * 20)  # Sehr schneller Decay
    
    # Kombiniere Komponenten
    kick = (kick_tone * envelope + noise * noise_envelope) * 0.8
    
    # Normalisierung
    kick = kick / np.max(np.abs(kick)) * 0.9
    
    sf.write(filename, kick, sample_rate)
    print(f"Kick-Sample erstellt: {filename}")

def create_bass_line(filename: str, duration: float = 4.0, sample_rate: int = 48000):
    """
    Erstellt eine synthetische Bass-Linie mit:
    - Mittlere RMS-Energie
    - Niedrige spektrale Zentroide
    - Kontinuierliche Tonfolge
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Bass-Linie: Grundfrequenzen zwischen 40-120Hz
    bass_notes = [55, 73, 82, 65]  # A1, D2, E2, C2
    note_duration = duration / len(bass_notes)
    
    bass_line = np.zeros_like(t)
    
    for i, freq in enumerate(bass_notes):
        start_idx = int(i * note_duration * sample_rate)
        end_idx = int((i + 1) * note_duration * sample_rate)
        
        if end_idx > len(t):
            end_idx = len(t)
        
        note_t = t[start_idx:end_idx]
        
        # Sawtooth-ähnlicher Bass mit Harmonischen
        fundamental = np.sin(2 * np.pi * freq * note_t)
        harmonic2 = 0.3 * np.sin(2 * np.pi * freq * 2 * note_t)
        harmonic3 = 0.1 * np.sin(2 * np.pi * freq * 3 * note_t)
        
        # Envelope für jeden Ton
        envelope = np.exp(-note_t * 0.5) * 0.6
        
        bass_line[start_idx:end_idx] = (fundamental + harmonic2 + harmonic3) * envelope
    
    # Normalisierung
    bass_line = bass_line / np.max(np.abs(bass_line)) * 0.7
    
    sf.write(filename, bass_line, sample_rate)
    print(f"Bass-Linie erstellt: {filename}")

def create_ambient_pad(filename: str, duration: float = 6.0, sample_rate: int = 48000):
    """
    Erstellt ein Ambient-Pad mit:
    - Niedrige bis mittlere RMS-Energie
    - Höhere spektrale Zentroide (mehr Obertöne)
    - Langsame Entwicklung, atmosphärisch
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mehrere Schichten für Ambient-Textur
    # Grundton: C3 (130.81 Hz)
    fundamental = 130.81
    
    # Schicht 1: Grundton mit langsamer Modulation
    mod_freq = 0.2  # Sehr langsame LFO
    layer1 = np.sin(2 * np.pi * fundamental * t + 0.1 * np.sin(2 * np.pi * mod_freq * t))
    
    # Schicht 2: Quinte (G3 - 196.00 Hz)
    layer2 = 0.6 * np.sin(2 * np.pi * 196.00 * t + 0.05 * np.sin(2 * np.pi * 0.15 * t))
    
    # Schicht 3: Oktave (C4 - 261.63 Hz)
    layer3 = 0.4 * np.sin(2 * np.pi * 261.63 * t + 0.08 * np.sin(2 * np.pi * 0.25 * t))
    
    # Schicht 4: Hohe Harmonische für Glanz
    layer4 = 0.2 * np.sin(2 * np.pi * 523.25 * t + 0.03 * np.sin(2 * np.pi * 0.3 * t))
    
    # Kombiniere alle Schichten
    ambient = layer1 + layer2 + layer3 + layer4
    
    # Langsame Envelope für Fade-in/out
    fade_samples = int(sample_rate * 1.0)  # 1 Sekunde Fade
    envelope = np.ones_like(t)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    ambient = ambient * envelope * 0.4
    
    # Normalisierung
    ambient = ambient / np.max(np.abs(ambient)) * 0.6
    
    sf.write(filename, ambient, sample_rate)
    print(f"Ambient-Pad erstellt: {filename}")

def create_silent_file(filename: str, duration: float = 3.0, sample_rate: int = 48000):
    """
    Erstellt eine "stille" Datei mit:
    - Sehr niedrige RMS-Energie (sollte quarantäniert werden)
    - Nur minimales Rauschen
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Sehr leises Rauschen (unter der Quarantäne-Schwelle)
    silence = np.random.normal(0, 0.001, len(t))  # Sehr niedrige Amplitude
    
    sf.write(filename, silence, sample_rate)
    print(f"Stille Datei erstellt: {filename} (sollte quarantäniert werden)")

def main():
    """
    Erstellt alle Test-Audiodateien für AGENTEN_DIREKTIVE_004
    """
    # Zielverzeichnis
    output_dir = Path("raw_construction_kits")
    output_dir.mkdir(exist_ok=True)
    
    print("Erstelle Test-Audiodateien für AGENTEN_DIREKTIVE_004...")
    
    # Erstelle verschiedene Arten von Audiodateien
    create_kick_sample(output_dir / "test_kick_004.wav")
    create_bass_line(output_dir / "test_bass_004.wav")
    create_ambient_pad(output_dir / "test_ambient_004.wav")
    create_silent_file(output_dir / "test_silent_004.wav")
    
    print("\nAlle Test-Audiodateien für AGENTEN_DIREKTIVE_004 erstellt!")
    print("Erwartete Kategorisierung:")
    print("- test_kick_004.wav: 'kick' (Dateiname-Heuristik)")
    print("- test_bass_004.wav: 'bass' (Dateiname-Heuristik)")
    print("- test_ambient_004.wav: 'unknown' -> KMeans-Kategorisierung")
    print("- test_silent_004.wav: Quarantäne (niedrige RMS)")

if __name__ == "__main__":
    main()