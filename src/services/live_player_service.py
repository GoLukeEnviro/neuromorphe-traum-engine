import sounddevice as sd
import numpy as np
import threading
import time
import librosa
from typing import Dict, Any, List
from ..database.service import DatabaseService

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
        self.db_service = DatabaseService()

    async def _load_stem_audio(self, stem_id: str) -> np.ndarray:
        """
        Lädt die Audio-Daten für einen Stem aus der Datenbank und von der Festplatte.
        """
        try:
            # Stem aus der Datenbank holen
            stem = await self.db_service.get_stem_by_id(int(stem_id))
            if not stem:
                print(f"Stem with ID {stem_id} not found in database")
                return None
            
            # Pfad zur verarbeiteten Audio-Datei
            audio_path = stem.processed_path
            if not audio_path:
                print(f"No processed_path found for stem {stem_id}")
                return None
            
            # Audio-Datei mit librosa laden
            print(f"Loading audio file: {audio_path}")
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
            
            # Sicherstellen, dass wir Stereo-Audio haben
            if audio_data.ndim == 1:
                # Mono zu Stereo konvertieren
                audio_data = np.stack([audio_data, audio_data], axis=0)
            
            # Transponieren für die richtige Form (frames, channels)
            audio_data = audio_data.T.astype(np.float32)
            
            print(f"Successfully loaded audio for stem {stem_id}: {audio_data.shape} samples")
            return audio_data
            
        except Exception as e:
            print(f"Error loading audio for stem {stem_id}: {e}")
            return None

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
                            # Echte Audio-Daten laden
                            audio_data = self._load_stem_audio(stem_id)
                            if audio_data is not None:
                                print(f"Loaded real audio for stem: {stem_id}")
                                self.player_slots[stem_id] = {
                                    'audio_data': audio_data,
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
