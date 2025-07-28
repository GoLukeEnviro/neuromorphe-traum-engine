import sounddevice as sd
import numpy as np
import threading
import time
from typing import Dict, Any, List

class LivePlayerService:
    """
    Echtzeit-Loop-Engine zur Interpretation und Wiedergabe von Track-Blaupausen.
    Ersetzt den Offline-Renderer.
    """
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.player_slots: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.stream = None
        self.conductor_thread = None
        self.is_playing = False

    def _audio_callback(self, outdata: np.ndarray, frames: int, time, status):
        """
        Zeitkritische Audio-Callback-Methode.
        Mischt Audio-Daten von aktiven Player-Slots.
        """
        if status:
            print(status) # Optional: Log audio stream warnings/errors

        # Erstelle einen leeren Puffer für die Ausgabe
        outdata.fill(0)

        with self.lock:
            for stem_id, slot_info in self.player_slots.items():
                if slot_info['is_active'] and slot_info['audio_data'] is not None:
                    audio_data = slot_info['audio_data']
                    current_frame = slot_info['current_frame']
                    volume = slot_info.get('volume', 1.0)

                    # Berechne, wie viele Frames vom aktuellen Stem gelesen werden können
                    frames_to_read = min(frames, len(audio_data) - current_frame)

                    # Lese Audio-Daten und füge sie zum Ausgabepuffer hinzu
                    outdata[:frames_to_read] += audio_data[current_frame:current_frame + frames_to_read] * volume

                    # Aktualisiere den Lesezeiger
                    slot_info['current_frame'] += frames_to_read

                    # Loop-Logik: Wenn das Ende erreicht ist, springe zum Anfang zurück
                    if slot_info['current_frame'] >= len(audio_data):
                        slot_info['current_frame'] = 0 # Zurück zum Anfang für Looping

    def _conductor_thread_run(self, arrangement_plan: Dict[str, Any]):
        """
        Musikalische Logik, die in einem separaten Thread läuft.
        Interpretiert die Arrangement-Blaupause und steuert die Wiedergabe.
        """
        print(f"Conductor thread started for arrangement: {arrangement_plan.get('title', 'Untitled')}")
        
        # Annahme: arrangement_plan hat eine 'track_structure' mit 'sections'
        # Jede Sektion hat eine 'duration' in Takten und eine Liste von 'stems' (IDs)
        # Annahme: Tempo ist in BPM in arrangement_plan.get('tempo', 120)

        tempo = arrangement_plan.get('tempo', 120)
        # Berechne die Dauer eines Taktes in Sekunden
        seconds_per_beat = 60 / tempo
        seconds_per_bar = seconds_per_beat * 4 # Annahme: 4/4 Takt

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            channels=2, # Stereo-Ausgabe
            callback=self._audio_callback
        )
        
        try:
            self.stream.start()
            self.is_playing = True
            print("Audio stream started.")

            for section in arrangement_plan.get('track_structure', {}).get('sections', []):
                if not self.is_playing:
                    break # Wiedergabe wurde gestoppt

                section_name = section.get('name', 'Unknown Section')
                section_duration_bars = section.get('duration_bars', 4) # Dauer in Takten
                section_stems = section.get('stems', []) # Stems für diese Sektion

                print(f"Playing section: {section_name} for {section_duration_bars} bars")

                # Stems für die aktuelle Sektion laden und aktivieren
                with self.lock:
                    # Deaktiviere alle aktuell aktiven Stems, die nicht in dieser Sektion sind
                    for stem_id in list(self.player_slots.keys()):
                        if stem_id not in section_stems:
                            self.player_slots[stem_id]['is_active'] = False
                            print(f"Deactivated stem: {stem_id}")
                    
                    # Aktiviere oder lade neue Stems für diese Sektion
                    for stem_id in section_stems:
                        if stem_id not in self.player_slots:
                            # Hier müsste die Logik zum Laden der tatsächlichen Audio-Daten implementiert werden
                            # Für jetzt: Platzhalter-Audio
                            print(f"Loading placeholder audio for stem: {stem_id}")
                            placeholder_audio = np.random.uniform(-0.1, 0.1, size=(self.sample_rate * 10, 2)).astype(np.float32) # 10 Sekunden Rauschen
                            self.player_slots[stem_id] = {
                                'audio_data': placeholder_audio,
                                'current_frame': 0,
                                'is_active': True,
                                'volume': 0.5 # Standard-Lautstärke
                            }
                        else:
                            self.player_slots[stem_id]['is_active'] = True
                        print(f"Activated stem: {stem_id}")

                # Warte für die Dauer der Sektion
                time.sleep(section_duration_bars * seconds_per_bar)

        except Exception as e:
            print(f"Error in conductor thread: {e}")
        finally:
            if self.stream and self.stream.active:
                self.stream.stop()
                self.stream.close()
            self.is_playing = False
            print("Conductor thread finished.")

    def play(self, arrangement_plan: Dict[str, Any]):
        """
        Startet die Live-Wiedergabe basierend auf dem Arrangement-Plan.
        """
        if self.is_playing:
            print("Already playing. Stop current playback first.")
            return

        self.conductor_thread = threading.Thread(
            target=self._conductor_thread_run,
            args=(arrangement_plan,)
        )
        self.conductor_thread.start()
        print("Playback initiated.")

    def stop(self):
        """
        Stoppt den Audio-Stream und beendet den Dirigenten-Thread sauber.
        """
        if self.is_playing:
            self.is_playing = False
            if self.conductor_thread and self.conductor_thread.is_alive():
                self.conductor_thread.join(timeout=2) # Warte auf Thread-Beendigung
                if self.conductor_thread.is_alive():
                    print("Warning: Conductor thread did not terminate gracefully.")
            print("Playback stopped.")
        else:
            print("No playback active.")
