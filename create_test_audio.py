#!/usr/bin/env python3
"""
Test-Audiodateien Generator für AGENTEN_DIREKTIVE_003
Erstellt drei synthetische Audiodateien für die Kategorisierungstests.
"""

import numpy as np
import soundfile as sf
import os

def create_test_audio_files():
    """
    Erstellt drei Test-Audiodateien mit unterschiedlichen Charakteristika:
    - test_kick_sample.wav: Kurzer perkussiver Sound (Kick-Drum)
    - test_bass_line.wav: Tieffrequenter Bass-Sound
    - test_loop_melody.wav: Melodischer Loop
    """
    
    # Sicherstellen, dass das Verzeichnis existiert
    output_dir = "raw_construction_kits"
    os.makedirs(output_dir, exist_ok=True)
    
    # Audio-Parameter
    sample_rate = 44100
    duration = 2.0  # 2 Sekunden
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 1. Kick-Drum Sample: Kurzer perkussiver Sound mit tiefem Thump
    kick_freq = 60  # Grundfrequenz für Kick
    kick_envelope = np.exp(-t * 8)  # Schnell abfallende Hüllkurve
    kick_sound = kick_envelope * np.sin(2 * np.pi * kick_freq * t)
    # Füge Klick-Anteil hinzu
    click = kick_envelope * 0.3 * np.sin(2 * np.pi * 2000 * t)
    kick_final = kick_sound + click
    kick_final = kick_final * 0.7  # Normalisierung
    
    # 2. Bass-Line: Tieffrequenter, sustain Bass
    bass_freq = 80  # Bass-Grundfrequenz
    bass_envelope = np.ones_like(t) * 0.5  # Konstante Lautstärke
    bass_envelope[int(len(t)*0.8):] *= np.linspace(1, 0, len(t) - int(len(t)*0.8))  # Fade-out
    bass_sound = bass_envelope * np.sin(2 * np.pi * bass_freq * t)
    # Füge Obertöne hinzu
    bass_sound += 0.3 * bass_envelope * np.sin(2 * np.pi * bass_freq * 2 * t)
    bass_final = bass_sound * 0.6
    
    # 3. Melodischer Loop: Höhere Frequenzen, melodisch
    melody_freqs = [440, 523, 659, 784]  # A, C, E, G (A-Moll Akkord)
    melody_sound = np.zeros_like(t)
    for i, freq in enumerate(melody_freqs):
        phase_offset = i * np.pi / 4  # Phasenverschiebung für Komplexität
        melody_sound += 0.25 * np.sin(2 * np.pi * freq * t + phase_offset)
    
    # Envelope für melodischen Charakter
    melody_envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t)  # Langsame Modulation
    melody_final = melody_sound * melody_envelope * 0.5
    
    # Dateien speichern
    files_created = []
    
    # Kick-Sample
    kick_path = os.path.join(output_dir, "test_kick_sample.wav")
    sf.write(kick_path, kick_final, sample_rate)
    files_created.append(kick_path)
    print(f"✓ Erstellt: {kick_path}")
    
    # Bass-Line
    bass_path = os.path.join(output_dir, "test_bass_line.wav")
    sf.write(bass_path, bass_final, sample_rate)
    files_created.append(bass_path)
    print(f"✓ Erstellt: {bass_path}")
    
    # Melodischer Loop
    loop_path = os.path.join(output_dir, "test_loop_melody.wav")
    sf.write(loop_path, melody_final, sample_rate)
    files_created.append(loop_path)
    print(f"✓ Erstellt: {loop_path}")
    
    print(f"\n🎵 {len(files_created)} Test-Audiodateien erfolgreich erstellt!")
    print("Bereit für AGENTEN_DIREKTIVE_003 Tests.")
    
    return files_created

if __name__ == "__main__":
    create_test_audio_files()