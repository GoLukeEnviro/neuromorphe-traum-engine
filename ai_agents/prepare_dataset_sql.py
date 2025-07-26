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
import json
from laion_clap import CLAP_Module
from sklearn.cluster import KMeans

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
        
        # LAION-CLAP-Modell laden
        logging.info("Lade LAION-CLAP-Modell. Dies kann einen Moment dauern...")
        self.clap_model = CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt()  # Lädt Standard-Checkpoint
        
        # Semantische Tag-Kandidaten definieren
        self.tags_candidates = ["dark", "punchy", "hypnotic", "industrial", "gritty", 
                               "atmospheric", "driving", "melodic", "percussive", "minimal"]
        
        # Ziel-Kategorien definieren
        self.categories = ["kick", "bass", "synth", "hihat", "snare", "percussion", "fx", "atmo"]
        
        # KMeans-Modell für Cluster-basierte Kategorisierung
        self.kmeans = KMeans(n_clusters=len(self.categories), random_state=42)

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
        
        # KMeans-Training vor der parallelen Verarbeitung
        logging.info("Trainiere KMeans-Modell für Cluster-basierte Kategorisierung...")
        self._train_kmeans(audio_files)
        
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

    def _train_kmeans(self, audio_files: List[str]):
        """
        Trainiert das KMeans-Modell mit Features aus allen Audiodateien.
        
        Args:
            audio_files (List[str]): Liste aller zu verarbeitenden Audiodateien
        """
        feature_pairs = []
        
        for file_path in audio_files:
            try:
                # Audio laden (nur für Feature-Extraktion)
                y, sr = librosa.load(file_path, sr=48000, duration=10.0)  # Nur erste 10s für Training
                
                # Grundlegende Qualitätsprüfung
                if len(y) / sr < 0.5:
                    continue
                
                # Features extrahieren
                features = self._extract_features(y, sr)
                
                # Nur spectral_centroid und rms für Clustering verwenden
                feature_pairs.append([features['spectral_centroid'], features['rms']])
                
            except Exception as e:
                logging.debug(f"Fehler beim Laden von {file_path} für KMeans-Training: {e}")
                continue
        
        if len(feature_pairs) < len(self.categories):
            logging.warning(f"Zu wenige Features für KMeans-Training: {len(feature_pairs)} < {len(self.categories)}")
            # Fallback: Verwende Dummy-Features
            feature_pairs = [[1000.0 + i*500, 0.1 + i*0.05] for i in range(len(self.categories))]
        
        # KMeans-Modell trainieren
        feature_array = np.array(feature_pairs)
        self.kmeans.fit(feature_array)
        
        logging.info(f"KMeans-Modell trainiert mit {len(feature_pairs)} Feature-Paaren")

    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extrahiert erweiterte Audio-Features für Qualitätskontrolle und Clustering.
        
        Args:
            y (np.ndarray): Audio-Array
            sr (int): Abtastrate
            
        Returns:
            Dict[str, float]: Dictionary mit berechneten Features
        """
        try:
            # Spektraler Schwerpunkt
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            
            # Nulldurchgangsrate
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            
            # RMS-Energie
            rms = float(np.mean(librosa.feature.rms(y=y)))
            
            # Spektraler Rolloff
            spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            
            # Spektrale Bandbreite
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            
            features = {
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zero_crossing_rate,
                'rms': rms,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth
            }
            
            logging.debug(f"Features extrahiert: {features}")
            return features
            
        except Exception as e:
            logging.error(f"Fehler bei Feature-Extraktion: {e}")
            # Fallback mit Standardwerten
            return {
                'spectral_centroid': 1000.0,
                'zero_crossing_rate': 0.1,
                'rms': 0.1,
                'spectral_rolloff': 2000.0,
                'spectral_bandwidth': 1000.0
            }

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
            
            # Erweiterte Feature-Extraktion
            features = self._extract_features(y_resampled, 48000)
            
            # Erweiterte Qualitätskontrolle
            if features['spectral_centroid'] < 100:
                logging.warning(f"Spektraler Schwerpunkt zu niedrig ({features['spectral_centroid']:.2f}): {file_path}")
                quarantine_dir = "processed_database/quarantine"
                os.makedirs(quarantine_dir, exist_ok=True)
                quarantine_path = os.path.join(quarantine_dir, os.path.basename(file_path))
                shutil.move(file_path, quarantine_path)
                return {'quarantined': True, 'reason': 'low_spectral_centroid', 'value': features['spectral_centroid']}
            
            if features['rms'] < 0.001:
                logging.warning(f"RMS zu niedrig ({features['rms']:.6f}): {file_path}")
                quarantine_dir = "processed_database/quarantine"
                os.makedirs(quarantine_dir, exist_ok=True)
                quarantine_path = os.path.join(quarantine_dir, os.path.basename(file_path))
                shutil.move(file_path, quarantine_path)
                return {'quarantined': True, 'reason': 'low_rms', 'value': features['rms']}
            
            # Semantische Analyse mit CLAP
            clap_tags = self._get_clap_tags(y_resampled)
            category = self._get_category(file_path, features)
            
            # Eindeutige ID mit Kategorie generieren
            timestamp = int(datetime.now().timestamp())
            stem_id = f"{category}_{timestamp}_{file_hash[:4]}"
            
            # Metadaten zusammenstellen
            metadata = {
                'id': stem_id,
                'path': standardized_path,
                'bpm': bpm,
                'key': None,  # Wird später implementiert
                'category': category,
                'tags': json.dumps(clap_tags),
                'features': json.dumps(features),
                'quality_ok': True,
                'user_rating': None,
                'imported_at': datetime.now().isoformat()
            }
            
            logging.info(f"Verarbeitet: {os.path.basename(file_path)} -> BPM: {bpm:.1f}, Dauer: {duration:.2f}s")
            return metadata
            
        except Exception as e:
            logging.error(f"Fehler bei Verarbeitung von {file_path}: {e}")
            return None

    def _get_clap_tags(self, audio_array: np.ndarray) -> List[str]:
        """
        Verwendet LAION-CLAP um semantische Tags für ein Audio-Array zu generieren.
        
        Args:
            audio_array (np.ndarray): Das Audio-Array (48kHz, mono)
            
        Returns:
            List[str]: Liste der Top 3 passendsten Tags
        """
        try:
            # Sicherstellen, dass das Audio-Array die richtige Form hat
            if len(audio_array.shape) == 1:
                # Mono-Audio: Füge Batch-Dimension hinzu
                audio_data = audio_array.reshape(1, -1)
            else:
                audio_data = audio_array
            
            # Audio-Embedding berechnen
            audio_embed = self.clap_model.get_audio_embedding_from_data(
                x=audio_data, use_tensor=False
            )
            
            # Text-Embeddings für alle Kandidaten-Tags berechnen
            text_embeds = self.clap_model.get_text_embedding(self.tags_candidates)
            
            # Kosinus-Ähnlichkeit berechnen
            if len(audio_embed.shape) > 1:
                audio_embed = audio_embed.flatten()
            if len(text_embeds.shape) > 2:
                text_embeds = text_embeds.reshape(text_embeds.shape[0], -1)
            
            similarities = np.dot(audio_embed, text_embeds.T).flatten()
            
            # Top 3 Tags auswählen
            top_indices = np.argsort(similarities)[-3:][::-1]
            top_tags = [self.tags_candidates[i] for i in top_indices]
            
            logging.debug(f"CLAP-Tags generiert: {top_tags}")
            return top_tags
            
        except Exception as e:
            logging.error(f"Fehler bei CLAP-Tag-Generierung: {e}")
            # Fallback: Erste drei Tags aus der Liste
            return self.tags_candidates[:3]
    
    def _get_category(self, file_path: str, features: Dict[str, float]) -> str:
        """
        Bestimmt die Kategorie eines Stems basierend auf Dateinamen-Heuristik
        mit Cluster-basierter Fallback-Logik.
        
        Args:
            file_path (str): Der Pfad zur Originaldatei
            features (Dict[str, float]): Extrahierte Audio-Features
            
        Returns:
            str: Die ermittelte Kategorie
        """
        filename_lower = os.path.basename(file_path).lower()
        
        # Durchsuche Dateinamen nach Kategorie-Schlüsselwörtern
        for category in self.categories:
            if category in filename_lower:
                logging.debug(f"Kategorie '{category}' erkannt in: {filename_lower}")
                return category
        
        # Cluster-basierte Fallback-Logik
        try:
            # Verwende spectral_centroid und rms für Clustering
            feature_vector = np.array([[features['spectral_centroid'], features['rms']]])
            
            # Vorhersage mit trainiertem KMeans-Modell
            cluster_prediction = self.kmeans.predict(feature_vector)[0]
            predicted_category = self.categories[cluster_prediction]
            
            logging.debug(f"Cluster-basierte Kategorie '{predicted_category}' für: {filename_lower}")
            return predicted_category
            
        except Exception as e:
            logging.error(f"Fehler bei Cluster-Vorhersage: {e}")
            # Fallback: unknown
            logging.debug(f"Keine Kategorie erkannt in: {filename_lower}, verwende 'unknown'")
            return "unknown"

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