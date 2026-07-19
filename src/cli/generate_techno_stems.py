#!/usr/bin/env python3
"""
Techno Stem Generator für AGENTEN_DIREKTIVE_008
Generiert 500+ hochwertige Raw-Techno, Hardgroove und verwandte Audio-Stems
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path
import random
from tqdm import tqdm

class TechnoStemGenerator:
    def __init__(self, output_dir="raw_construction_kits", sample_rate=44100):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.bpm_range = [120, 140]  # Typische Techno BPMs
        
        # Erstelle Kategorien-Ordner
        self.categories = {
            'kicks': 'kicks',
            'basslines': 'basslines', 
            'percussion': 'percussion',
            'leads': 'leads',
            'pads': 'pads',
            'fx': 'fx',
            'loops': 'loops',
            'hats': 'hats'
        }
        
        for category in self.categories.values():
            (self.output_dir / category).mkdir(parents=True, exist_ok=True)
    
    def generate_kick_drum(self, duration=0.5, freq_start=60, freq_end=40):
        """Generiert einen kraftvollen Techno-Kick"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Frequenz-Sweep für den Kick
        freq = np.linspace(freq_start, freq_end, len(t))
        
        # Grundwelle mit Harmonischen
        kick = np.sin(2 * np.pi * freq * t)
        kick += 0.3 * np.sin(4 * np.pi * freq * t)  # 2. Harmonische
        kick += 0.1 * np.sin(6 * np.pi * freq * t)  # 3. Harmonische
        
        # Exponentieller Decay
        envelope = np.exp(-t * 8)
        kick *= envelope
        
        # Punch-Envelope für Attack
        punch_env = np.exp(-t * 50)
        kick += 0.2 * np.random.normal(0, 0.1, len(t)) * punch_env
        
        # Normalisierung
        kick = kick / np.max(np.abs(kick)) * 0.9
        
        return kick
    
    def generate_bassline(self, duration=4.0, root_freq=55):
        """Generiert eine hypnotische Techno-Bassline"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Bassline-Pattern (16tel Noten)
        pattern_length = duration / 16
        pattern = []
        
        # Verschiedene Bass-Patterns
        patterns = [
            [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],  # Minimal
            [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0],  # Driving
            [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]   # Syncopated
        ]
        
        chosen_pattern = random.choice(patterns)
        
        bassline = np.zeros(len(t))
        
        for i, note in enumerate(chosen_pattern):
            if note:
                start_idx = int(i * pattern_length * self.sample_rate)
                end_idx = int((i + 0.8) * pattern_length * self.sample_rate)
                
                if end_idx <= len(t):
                    note_t = t[start_idx:end_idx] - t[start_idx]
                    
                    # Frequenz-Variation
                    freq_mult = random.choice([1, 1.25, 1.5, 2])  # Oktaven und Quinten
                    freq = root_freq * freq_mult
                    
                    # Sawtooth-Wave für Bass
                    note_wave = 2 * (note_t * freq - np.floor(note_t * freq + 0.5))
                    
                    # Filter-Envelope
                    filter_env = np.exp(-note_t * 3)
                    note_wave *= filter_env
                    
                    bassline[start_idx:end_idx] += note_wave
        
        # Normalisierung
        bassline = bassline / np.max(np.abs(bassline)) * 0.7
        
        return bassline
    
    def generate_hihat(self, duration=0.1, style='closed'):
        """Generiert Hi-Hats"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Noise-basierte Hi-Hat
        noise = np.random.normal(0, 1, len(t))
        
        if style == 'closed':
            # Geschlossene Hi-Hat - kurz und scharf
            envelope = np.exp(-t * 50)
            # Hochpass-Filter Simulation
            hihat = noise * envelope
            hihat = np.diff(np.concatenate([[0], hihat]))  # Einfacher Hochpass
        else:
            # Offene Hi-Hat - länger und luftiger
            envelope = np.exp(-t * 10)
            hihat = noise * envelope
        
        # Normalisierung
        hihat = hihat / np.max(np.abs(hihat)) * 0.5
        
        return hihat
    
    def generate_lead_synth(self, duration=8.0, root_freq=220):
        """Generiert einen Techno-Lead-Synth"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Melodie-Pattern
        notes = [1, 1.25, 1.5, 1.25, 2, 1.5, 1.25, 1]  # Relative Frequenzen
        note_duration = duration / len(notes)
        
        lead = np.zeros(len(t))
        
        for i, note_mult in enumerate(notes):
            start_idx = int(i * note_duration * self.sample_rate)
            end_idx = int((i + 1) * note_duration * self.sample_rate)
            
            if end_idx <= len(t):
                note_t = t[start_idx:end_idx] - t[start_idx]
                freq = root_freq * note_mult
                
                # Sawtooth mit Filter-Sweep
                note_wave = 2 * (note_t * freq - np.floor(note_t * freq + 0.5))
                
                # LFO für Filter-Modulation
                lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * note_t)
                filter_freq = 1000 + 2000 * lfo
                
                # Einfache Filter-Simulation
                cutoff_env = np.exp(-note_t * 2) * lfo
                note_wave *= cutoff_env
                
                # ADSR-Envelope
                attack = 0.1
                decay = 0.2
                sustain = 0.6
                release = note_duration - attack - decay
                
                envelope = np.ones(len(note_t))
                attack_samples = int(attack * self.sample_rate)
                decay_samples = int(decay * self.sample_rate)
                
                if len(note_t) > attack_samples:
                    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                if len(note_t) > attack_samples + decay_samples:
                    envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain, decay_samples)
                    envelope[attack_samples + decay_samples:] = sustain * np.exp(-np.linspace(0, 5, len(note_t) - attack_samples - decay_samples))
                
                note_wave *= envelope
                lead[start_idx:end_idx] += note_wave
        
        # Normalisierung
        lead = lead / np.max(np.abs(lead)) * 0.6
        
        return lead
    
    def generate_atmospheric_pad(self, duration=16.0):
        """Generiert atmosphärische Pads"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Mehrere Oszillatoren für reicheren Sound
        freqs = [55, 82.5, 110, 165]  # A1, E2, A2, E3
        pad = np.zeros(len(t))
        
        for freq in freqs:
            # Sine Wave mit leichter Verstimmung
            detune = random.uniform(-2, 2)
            osc = np.sin(2 * np.pi * (freq + detune) * t)
            
            # LFO für Tremolo
            lfo_rate = random.uniform(0.1, 0.5)
            lfo = 0.8 + 0.2 * np.sin(2 * np.pi * lfo_rate * t)
            osc *= lfo
            
            pad += osc * 0.25
        
        # Langsame Attack/Release
        envelope = np.ones(len(t))
        fade_samples = int(2 * self.sample_rate)  # 2 Sekunden Fade
        
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        pad *= envelope
        
        # Normalisierung
        pad = pad / np.max(np.abs(pad)) * 0.4
        
        return pad
    
    def generate_fx_sweep(self, duration=2.0):
        """Generiert FX-Sweeps und Risers"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Frequenz-Sweep
        freq_start = random.uniform(100, 500)
        freq_end = random.uniform(1000, 4000)
        freq = np.linspace(freq_start, freq_end, len(t))
        
        # Noise mit Filter-Sweep
        noise = np.random.normal(0, 1, len(t))
        
        # Resonanter Filter-Effekt
        sweep = np.sin(2 * np.pi * freq * t) * 0.3
        sweep += noise * 0.1
        
        # Exponentieller Anstieg
        envelope = (t / duration) ** 2
        sweep *= envelope
        
        # Normalisierung
        sweep = sweep / np.max(np.abs(sweep)) * 0.5
        
        return sweep
    
    def generate_percussion_loop(self, duration=4.0):
        """Generiert Percussion-Loops"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        loop = np.zeros(len(t))
        
        # 16tel-Pattern für verschiedene Percussion-Elemente
        pattern_length = duration / 16
        
        # Clap-Pattern
        clap_pattern = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        
        # Shaker-Pattern
        shaker_pattern = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        
        for i in range(16):
            start_idx = int(i * pattern_length * self.sample_rate)
            
            # Clap
            if clap_pattern[i]:
                clap_duration = 0.1
                clap_samples = int(clap_duration * self.sample_rate)
                clap_t = np.linspace(0, clap_duration, clap_samples)
                
                clap = np.random.normal(0, 1, clap_samples)
                clap_env = np.exp(-clap_t * 20)
                clap *= clap_env * 0.3
                
                end_idx = min(start_idx + clap_samples, len(loop))
                loop[start_idx:end_idx] += clap[:end_idx-start_idx]
            
            # Shaker
            if shaker_pattern[i]:
                shaker_duration = 0.05
                shaker_samples = int(shaker_duration * self.sample_rate)
                shaker_t = np.linspace(0, shaker_duration, shaker_samples)
                
                shaker = np.random.normal(0, 1, shaker_samples)
                shaker_env = np.exp(-shaker_t * 30)
                shaker *= shaker_env * 0.2
                
                end_idx = min(start_idx + shaker_samples, len(loop))
                loop[start_idx:end_idx] += shaker[:end_idx-start_idx]
        
        # Normalisierung
        loop = loop / np.max(np.abs(loop)) * 0.6
        
        return loop
    
    def generate_stems(self, target_count=500):
        """Generiert die gewünschte Anzahl von Stems"""
        current_count = len(list(self.output_dir.rglob('*.wav')))
        needed = target_count - current_count
        
        print(f"Aktuelle Stems: {current_count}")
        print(f"Benötigte Stems: {needed}")
        print(f"Ziel: {target_count} Stems")
        
        if needed <= 0:
            print("Ziel bereits erreicht!")
            return
        
        # Verteilung der Stems auf Kategorien
        distribution = {
            'kicks': int(needed * 0.25),      # 25% Kicks
            'basslines': int(needed * 0.20),  # 20% Basslines
            'hats': int(needed * 0.15),       # 15% Hi-Hats
            'percussion': int(needed * 0.15), # 15% Percussion
            'leads': int(needed * 0.10),      # 10% Leads
            'pads': int(needed * 0.08),       # 8% Pads
            'fx': int(needed * 0.05),         # 5% FX
            'loops': int(needed * 0.02)       # 2% Loops
        }
        
        # Stelle sicher, dass wir die exakte Anzahl erreichen
        total_distributed = sum(distribution.values())
        distribution['kicks'] += needed - total_distributed
        
        print("\nGeneriere Stems...")
        
        # Generiere Kicks
        for i in tqdm(range(distribution['kicks']), desc="Kicks"):
            bpm = random.randint(*self.bpm_range)
            freq_start = random.randint(50, 80)
            freq_end = random.randint(30, 50)
            duration = random.uniform(0.3, 0.8)
            
            kick = self.generate_kick_drum(duration, freq_start, freq_end)
            
            style = random.choice(['punchy', 'deep', 'industrial', 'minimal'])
            filename = f"kick_{style}_{bpm}bpm_{i+1:03d}.wav"
            filepath = self.output_dir / 'kicks' / filename
            
            sf.write(filepath, kick, self.sample_rate)
        
        # Generiere Basslines
        for i in tqdm(range(distribution['basslines']), desc="Basslines"):
            bpm = random.randint(*self.bpm_range)
            root_freq = random.choice([55, 73.4, 82.4, 110])  # A1, D2, E2, A2
            duration = random.choice([4, 8, 16])
            
            bassline = self.generate_bassline(duration, root_freq)
            
            style = random.choice(['hypnotic', 'driving', 'minimal', 'acid'])
            filename = f"bass_{style}_{bpm}bpm_{i+1:03d}.wav"
            filepath = self.output_dir / 'basslines' / filename
            
            sf.write(filepath, bassline, self.sample_rate)
        
        # Generiere Hi-Hats
        for i in tqdm(range(distribution['hats']), desc="Hi-Hats"):
            style = random.choice(['closed', 'open'])
            duration = 0.1 if style == 'closed' else random.uniform(0.2, 0.5)
            
            hihat = self.generate_hihat(duration, style)
            
            filename = f"hihat_{style}_{i+1:03d}.wav"
            filepath = self.output_dir / 'hats' / filename
            
            sf.write(filepath, hihat, self.sample_rate)
        
        # Generiere Percussion
        for i in tqdm(range(distribution['percussion']), desc="Percussion"):
            duration = random.choice([2, 4, 8])
            
            perc = self.generate_percussion_loop(duration)
            
            style = random.choice(['tribal', 'industrial', 'minimal', 'groove'])
            filename = f"perc_{style}_{i+1:03d}.wav"
            filepath = self.output_dir / 'percussion' / filename
            
            sf.write(filepath, perc, self.sample_rate)
        
        # Generiere Leads
        for i in tqdm(range(distribution['leads']), desc="Leads"):
            bpm = random.randint(*self.bpm_range)
            root_freq = random.choice([220, 293.7, 440, 587.3])  # A3, D4, A4, D5
            duration = random.choice([8, 16, 32])
            
            lead = self.generate_lead_synth(duration, root_freq)
            
            style = random.choice(['acid', 'stabbing', 'melodic', 'industrial'])
            filename = f"lead_{style}_{bpm}bpm_{i+1:03d}.wav"
            filepath = self.output_dir / 'leads' / filename
            
            sf.write(filepath, lead, self.sample_rate)
        
        # Generiere Pads
        for i in tqdm(range(distribution['pads']), desc="Pads"):
            duration = random.choice([16, 32, 64])
            
            pad = self.generate_atmospheric_pad(duration)
            
            style = random.choice(['dark', 'atmospheric', 'warm', 'cold'])
            filename = f"pad_{style}_{i+1:03d}.wav"
            filepath = self.output_dir / 'pads' / filename
            
            sf.write(filepath, pad, self.sample_rate)
        
        # Generiere FX
        for i in tqdm(range(distribution['fx']), desc="FX"):
            duration = random.uniform(1, 4)
            
            fx = self.generate_fx_sweep(duration)
            
            style = random.choice(['riser', 'sweep', 'impact', 'reverse'])
            filename = f"fx_{style}_{i+1:03d}.wav"
            filepath = self.output_dir / 'fx' / filename
            
            sf.write(filepath, fx, self.sample_rate)
        
        # Generiere Loops
        for i in tqdm(range(distribution['loops']), desc="Loops"):
            bpm = random.randint(*self.bpm_range)
            duration = random.choice([4, 8, 16])
            
            # Kombiniere verschiedene Elemente für komplexe Loops
            loop = np.zeros(int(duration * self.sample_rate))
            
            # Füge Bass hinzu
            bass = self.generate_bassline(duration, 55)
            loop += bass * 0.6
            
            # Füge Percussion hinzu
            perc = self.generate_percussion_loop(duration)
            loop += perc * 0.4
            
            # Normalisierung
            loop = loop / np.max(np.abs(loop)) * 0.8
            
            style = random.choice(['minimal', 'driving', 'hypnotic', 'industrial'])
            filename = f"loop_{style}_{bpm}bpm_{i+1:03d}.wav"
            filepath = self.output_dir / 'loops' / filename
            
            sf.write(filepath, loop, self.sample_rate)
        
        final_count = len(list(self.output_dir.rglob('*.wav')))
        print(f"\n✅ Generierung abgeschlossen!")
        print(f"Finale Anzahl Stems: {final_count}")
        print(f"Ziel erreicht: {'✅' if final_count >= target_count else '❌'}")

def main():
    print("🎵 Techno Stem Generator - AGENTEN_DIREKTIVE_008")
    print("=" * 50)
    
    generator = TechnoStemGenerator()
    generator.generate_stems(target_count=500)
    
    print("\n🎉 Datensatz-Expansion abgeschlossen!")
    print("Nächster Schritt: Führe 'python ai_agents/minimal_preprocessor.py' aus")

if __name__ == "__main__":
    main()