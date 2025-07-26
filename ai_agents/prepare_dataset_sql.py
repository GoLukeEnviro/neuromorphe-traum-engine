import os
import sqlite3
import logging
from datetime import datetime
import librosa
import numpy as np
import soundfile as sf
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, List
import hashlib
from pathlib import Path

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Datenbank-Definition
DB_PATH = "processed_database/stems.db"

class NeuroAnalyzer:
    def __init__(self, input_dir: str):
        """
        Initialisiert den Neuro-Analysator.
        
        Args:
            input_dir (str): Das Verzeichnis mit den rohen Audiodaten.
        """
        logging.info("Initialisiere Neuro-Analysator...")
        self.input_dir = input_dir
        # Spätere Initialisierung von CLAP, KMeans etc. kommt hier hin

    def init_db(self):
        """
        Erstellt die SQLite-Datenbank und die 'stems'-Tabelle, falls sie nicht existiert.
        Das Schema muss exakt den Spezifikationen entsprechen.
        """
        logging.info(f"Initialisiere Datenbank unter: {DB_PATH}")
        
        # Stelle sicher, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # Verbindung zur Datenbank herstellen
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Erstelle die stems-Tabelle mit exakt den spezifizierten Spalten
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stems (
                id TEXT PRIMARY KEY,
                path TEXT,
                bpm REAL,
                key TEXT,
                category TEXT,
                tags TEXT,
                features TEXT,
                quality_ok BOOLEAN,
                user_rating INTEGER,
                imported_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logging.info("Datenbank erfolgreich initialisiert.")

    def run(self):
        """
        Die Hauptmethode, die den gesamten Analyseprozess steuert.
        Sucht rekursiv nach Audiodateien und verarbeitet sie parallel.
        """
        logging.info("Neuro-Analyse gestartet...")
        
        # Unterstützte Audioformate
        audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg'}
        
        # Sammle alle Audiodateien
        audio_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if Path(file).suffix.lower() in audio_extensions:
                    audio_files.append(os.path.join(root, file))
        
        logging.info(f"Gefunden: {len(audio_files)} Audiodateien")
        
        if not audio_files:
            logging.warning("Keine Audiodateien gefunden!")
            return
        
        # Parallele Verarbeitung
        processed_count = 0
        quarantined_count = 0
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Starte alle Tasks
            future_to_file = {executor.submit(self._process_file, file_path): file_path 
                            for file_path in audio_files}
            
            # Sammle Ergebnisse
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        if result.get('quarantined', False):
                            quarantined_count += 1
                        else:
                            self._insert_meta(result)
                            processed_count += 1
                except Exception as e:
                    logging.error(f"Fehler bei Verarbeitung von {file_path}: {e}")
        
        logging.info(f"Neuro-Analyse abgeschlossen. Verarbeitet: {processed_count}, Quarantäne: {quarantined_count}")

    def _process_file(self, file_path: str) -> Optional[Dict]:
        """
        Verarbeitet eine einzelne Audiodatei: Lädt, standardisiert, analysiert BPM
        und extrahiert Metadaten.
        
        Args:
            file_path (str): Der Pfad zur zu verarbeitenden Datei.
            
        Returns:
            dict: Ein Dictionary mit den extrahierten Metadaten oder None bei Fehler.
        """
        try:
            # Audio laden
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            # Qualitätskontrolle: Mindestdauer 0.5s
            if duration < 0.5:
                logging.warning(f"Datei zu kurz ({duration:.2f}s): {file_path}")
                # In Quarantäne verschieben
                quarantine_dir = "processed_database/quarantine"
                os.makedirs(quarantine_dir, exist_ok=True)
                quarantine_path = os.path.join(quarantine_dir, os.path.basename(file_path))
                shutil.move(file_path, quarantine_path)
                return {'quarantined': True, 'reason': 'too_short', 'duration': duration}
            
            # Audio standardisieren: 48kHz, Mono, 24-bit WAV
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=48000)
            if len(y_resampled.shape) > 1:
                y_resampled = librosa.to_mono(y_resampled)
            
            # Standardisierte Datei speichern
            stems_dir = "processed_database/stems"
            os.makedirs(stems_dir, exist_ok=True)
            
            # Eindeutige ID generieren
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            standardized_filename = f"{file_hash}_{os.path.splitext(os.path.basename(file_path))[0]}.wav"
            standardized_path = os.path.join(stems_dir, standardized_filename)
            
            # Als 24-bit WAV speichern
            sf.write(standardized_path, y_resampled, 48000, subtype='PCM_24')
            
            # BPM-Analyse
            tempo, beats = librosa.beat.beat_track(y=y_resampled, sr=48000)
            bpm = float(tempo)
            
            # Metadaten zusammenstellen
            metadata = {
                'id': file_hash,
                'path': standardized_path,
                'bpm': bpm,
                'key': None,  # Wird später implementiert
                'category': None,  # Wird später implementiert
                'tags': None,  # Wird später implementiert
                'features': None,  # Wird später implementiert
                'quality_ok': True,
                'user_rating': None,
                'imported_at': datetime.now().isoformat()
            }
            
            logging.info(f"Verarbeitet: {os.path.basename(file_path)} -> BPM: {bpm:.1f}, Dauer: {duration:.2f}s")
            return metadata
            
        except Exception as e:
            logging.error(f"Fehler bei Verarbeitung von {file_path}: {e}")
            return None

    def _insert_meta(self, metadata: Dict):
        """
        Fügt ein Metadaten-Dictionary in die SQLite-Datenbank ein.
        Verwendet Transaktionen für Datenkonsistenz.
        
        Args:
            metadata (dict): Das Dictionary mit den zu speichernden Metadaten.
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Insert mit allen Spalten
            cursor.execute('''
                INSERT OR REPLACE INTO stems 
                (id, path, bpm, key, category, tags, features, quality_ok, user_rating, imported_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata['id'],
                metadata['path'],
                metadata['bpm'],
                metadata['key'],
                metadata['category'],
                metadata['tags'],
                metadata['features'],
                metadata['quality_ok'],
                metadata['user_rating'],
                metadata['imported_at']
            ))
            
            conn.commit()
            conn.close()
            
            logging.debug(f"Metadaten gespeichert für ID: {metadata['id']}")
            
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Metadaten: {e}")
            if 'conn' in locals():
                conn.close()

if __name__ == "__main__":
    analyzer = NeuroAnalyzer(input_dir="raw_construction_kits")
    analyzer.init_db()
    analyzer.run()