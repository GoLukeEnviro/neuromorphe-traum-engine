import os
import sqlite3
import logging
from datetime import datetime
import librosa
import numpy as np
import soundfile as sf
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import Optional, Dict, List, Tuple
import hashlib
from pathlib import Path
import json
from laion_clap import CLAP_Module
from sklearn.cluster import KMeans
import time
import traceback
import pickle
from dataclasses import dataclass
from enum import Enum
import glob

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Konstanten für Batch-Verarbeitung
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CHECKPOINT_INTERVAL = 50

class ProcessingStatus(Enum):
    """Status-Enum für die Verarbeitung von Dateien"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"

@dataclass
class ProcessingResult:
    """Datenklasse für Verarbeitungsergebnisse"""
    file_path: str
    status: ProcessingStatus
    metadata: Optional[Dict] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time: float = 0.0

# Datenbank-Definition
DB_PATH = "processed_database/stems.db"

class NeuroAnalyzer:
    def __init__(self, input_dir: str, resume_from_checkpoint: bool = True, 
                 batch_size: int = BATCH_SIZE, max_retries: int = MAX_RETRIES, 
                 checkpoint_interval: int = CHECKPOINT_INTERVAL):
        """
        Initialisiert den Neuro-Analysator.
        
        Args:
            input_dir (str): Das Verzeichnis mit den rohen Audiodaten.
            resume_from_checkpoint (bool): Ob von einem Checkpoint fortgesetzt werden soll.
            batch_size (int): Batch-Größe für CLAP-Verarbeitung.
            max_retries (int): Maximale Anzahl von Wiederholungsversuchen.
            checkpoint_interval (int): Intervall für Checkpoint-Speicherung.
        """
        logging.info("Initialisiere Neuro-Analysator...")
        self.input_dir = input_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.checkpoint_interval = checkpoint_interval
        
        # Checkpoint- und Status-Dateien
        self.checkpoint_dir = "processed_database/checkpoints"
        self.progress_file = os.path.join(self.checkpoint_dir, "progress.json")
        self.failed_files_log = os.path.join(self.checkpoint_dir, "failed_files.json")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
        
        # Statistiken
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'quarantined': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }

    def init_db(self):
        """
        Erstellt die SQLite-Datenbank und die 'stems'-Tabelle, falls sie nicht existiert.
        Das Schema muss exakt den Spezifikationen entsprechen.
        """
        logging.info(f"Initialisiere Datenbank unter: {DB_PATH}")
        
        # Stelle sicher, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # Verbindung zur Datenbank herstellen und als Instanzvariable speichern
        self.conn = sqlite3.connect(DB_PATH)
        cursor = self.conn.cursor()
        
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
                imported_at DATETIME,
                clap_embedding BLOB
            )
        ''')
        
        # Erstelle processing_status Tabelle für Resume-Funktionalität
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_status (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT,
                status TEXT,
                last_attempt DATETIME,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                processing_time REAL
            )
        ''')
        
        self.conn.commit()
        
        logging.info("Datenbank erfolgreich initialisiert.")

    def run(self):
        """
        Die Hauptmethode, die den gesamten Analyseprozess steuert.
        Implementiert robuste Batch-Verarbeitung mit Resume-Funktionalität.
        """
        self.stats['start_time'] = time.time()
        logging.info("Neuro-Analyse gestartet...")
        
        try:
            # Sammle alle Audiodateien
            audio_files = self._discover_audio_files()
            self.stats['total_files'] = len(audio_files)
            
            if not audio_files:
                logging.warning("Keine Audiodateien gefunden!")
                return
            
            # Lade vorherigen Fortschritt oder starte neu
            processed_files = self._load_progress() if self.resume_from_checkpoint else set()
            
            # Filtere bereits verarbeitete Dateien
            remaining_files = [f for f in audio_files if f not in processed_files]
            logging.info(f"Gefunden: {len(audio_files)} Audiodateien, {len(remaining_files)} noch zu verarbeiten")
            
            if not remaining_files:
                logging.info("Alle Dateien bereits verarbeitet!")
                return
            
            # KMeans-Training vor der Verarbeitung
            logging.info("Trainiere KMeans-Modell für Cluster-basierte Kategorisierung...")
            self._train_kmeans(remaining_files[:100])  # Verwende nur Subset für Training
            
            # Batch-Verarbeitung mit Resume-Funktionalität
            self._process_files_in_batches(remaining_files)
            
        except Exception as e:
            logging.error(f"Kritischer Fehler in run(): {e}")
            logging.error(traceback.format_exc())
        finally:
            self.stats['end_time'] = time.time()
            self._print_final_statistics()
    
    def _discover_audio_files(self) -> List[str]:
        """
        Entdeckt alle Audiodateien im Input-Verzeichnis.
        
        Returns:
            List[str]: Liste aller gefundenen Audiodateien
        """
        audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg'}
        audio_files = []
        
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if Path(file).suffix.lower() in audio_extensions:
                    audio_files.append(os.path.join(root, file))
        
        return audio_files
    
    def _load_progress(self) -> set:
        """
        Lädt den Fortschritt aus der Datenbank.
        
        Returns:
            set: Set der bereits verarbeiteten Dateipfade
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path FROM processing_status 
                WHERE status IN ('completed', 'quarantined')
            """)
            
            processed_files = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            logging.info(f"Fortschritt geladen: {len(processed_files)} bereits verarbeitete Dateien")
            return processed_files
            
        except Exception as e:
            logging.error(f"Fehler beim Laden des Fortschritts: {e}")
            return set()
    
    def _process_files_in_batches(self, files: List[str]):
        """
        Verarbeitet Dateien in Batches mit robustem Error-Handling.
        
        Args:
            files (List[str]): Liste der zu verarbeitenden Dateien
        """
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(0, len(files), BATCH_SIZE):
            batch_files = files[batch_idx:batch_idx + BATCH_SIZE]
            batch_num = (batch_idx // BATCH_SIZE) + 1
            
            logging.info(f"Verarbeite Batch {batch_num}/{total_batches} ({len(batch_files)} Dateien)")
            
            # Verarbeite Batch mit CLAP-Embeddings
            batch_results = self._process_batch_with_clap(batch_files)
            
            # Speichere Ergebnisse
            self._save_batch_results(batch_results)
            
            # Checkpoint speichern
            if batch_num % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint()
                logging.info(f"Checkpoint gespeichert nach Batch {batch_num}")
    
    def _process_batch_with_clap(self, files: List[str]) -> List[ProcessingResult]:
        """
        Verarbeitet einen Batch von Dateien mit CLAP-Embeddings.
        
        Args:
            files (List[str]): Liste der Dateien im Batch
            
        Returns:
            List[ProcessingResult]: Liste der Verarbeitungsergebnisse
        """
        results = []
        
        # Lade alle Audio-Daten für den Batch
        audio_data_batch = []
        valid_files = []
        
        for file_path in files:
            try:
                result = self._load_and_validate_audio(file_path)
                if result:
                    audio_data_batch.append(result['audio_data'])
                    valid_files.append((file_path, result))
                else:
                    # Datei fehlgeschlagen
                    results.append(ProcessingResult(
                        file_path=file_path,
                        status=ProcessingStatus.FAILED,
                        error_message="Audio-Validierung fehlgeschlagen"
                    ))
            except Exception as e:
                results.append(ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.FAILED,
                    error_message=str(e)
                ))
        
        # Batch-CLAP-Embedding-Berechnung
        if audio_data_batch:
            try:
                clap_embeddings = self._compute_batch_clap_embeddings(audio_data_batch)
                
                # Verarbeite jede Datei mit ihrem CLAP-Embedding
                for i, (file_path, audio_result) in enumerate(valid_files):
                    try:
                        start_time = time.time()
                        
                        # Verwende vorberechnetes CLAP-Embedding
                        clap_embedding = clap_embeddings[i] if i < len(clap_embeddings) else None
                        
                        metadata = self._create_metadata(file_path, audio_result, clap_embedding)
                        
                        processing_time = time.time() - start_time
                        
                        results.append(ProcessingResult(
                            file_path=file_path,
                            status=ProcessingStatus.COMPLETED,
                            metadata=metadata,
                            processing_time=processing_time
                        ))
                        
                    except Exception as e:
                        results.append(ProcessingResult(
                            file_path=file_path,
                            status=ProcessingStatus.FAILED,
                            error_message=str(e)
                        ))
                        
            except Exception as e:
                logging.error(f"Batch-CLAP-Verarbeitung fehlgeschlagen: {e}")
                # Fallback: Einzelverarbeitung
                for file_path, audio_result in valid_files:
                    try:
                        start_time = time.time()
                        metadata = self._create_metadata(file_path, audio_result, None)
                        processing_time = time.time() - start_time
                        
                        results.append(ProcessingResult(
                            file_path=file_path,
                            status=ProcessingStatus.COMPLETED,
                            metadata=metadata,
                            processing_time=processing_time
                        ))
                    except Exception as e:
                        results.append(ProcessingResult(
                            file_path=file_path,
                            status=ProcessingStatus.FAILED,
                            error_message=str(e)
                        ))
        
        return results
    
    def _load_and_validate_audio(self, file_path: str) -> Optional[Dict]:
        """
        Lädt und validiert eine Audiodatei.
        
        Args:
            file_path (str): Pfad zur Audiodatei
            
        Returns:
            Optional[Dict]: Audio-Daten und Metadaten oder None bei Fehler
        """
        try:
            # Audio laden
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            # Qualitätskontrolle: Mindestdauer 0.5s
            if duration < 0.5:
                logging.warning(f"Datei zu kurz ({duration:.2f}s): {file_path}")
                return None
            
            # Audio standardisieren: 48kHz, Mono
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=48000)
            if len(y_resampled.shape) > 1:
                y_resampled = librosa.to_mono(y_resampled)
            
            # Features extrahieren
            features = self._extract_features(y_resampled, 48000)
            
            # Erweiterte Qualitätskontrolle
            if features['spectral_centroid'] < 100 or features['rms'] < 0.001:
                logging.warning(f"Qualitätsprüfung fehlgeschlagen: {file_path}")
                return None
            
            # BPM-Analyse
            tempo, beats = librosa.beat.beat_track(y=y_resampled, sr=48000)
            bpm = float(tempo)
            
            return {
                'audio_data': y_resampled,
                'duration': duration,
                'bpm': bpm,
                'features': features
            }
            
        except Exception as e:
            logging.error(f"Fehler beim Laden von {file_path}: {e}")
            return None
    
    def _compute_batch_clap_embeddings(self, audio_batch: List[np.ndarray]) -> List[bytes]:
        """
        Berechnet CLAP-Embeddings für einen Batch von Audio-Arrays.
        
        Args:
            audio_batch (List[np.ndarray]): Liste von Audio-Arrays
            
        Returns:
            List[bytes]: Liste von CLAP-Embeddings als Binärdaten
        """
        try:
            # Batch-Audio-Daten vorbereiten
            batch_data = np.array(audio_batch)
            
            # Batch-CLAP-Embeddings berechnen
            audio_embeds = self.clap_model.get_audio_embedding_from_data(
                x=batch_data, use_tensor=False
            )
            
            # Embeddings normalisieren und als Binärdaten konvertieren
            embeddings_bytes = []
            for embed in audio_embeds:
                if len(embed.shape) > 1:
                    embed = embed.flatten()
                
                # Normalisierung
                embed = embed / np.linalg.norm(embed)
                
                # Als float32 Binärdaten speichern
                embedding_bytes = embed.astype(np.float32).tobytes()
                embeddings_bytes.append(embedding_bytes)
            
            logging.debug(f"Batch-CLAP-Embeddings berechnet: {len(embeddings_bytes)} Embeddings")
            return embeddings_bytes
            
        except Exception as e:
            logging.error(f"Fehler bei Batch-CLAP-Embedding-Berechnung: {e}")
            # Fallback: Null-Embeddings
            fallback_embeddings = []
            for _ in audio_batch:
                fallback_embedding = np.zeros(512, dtype=np.float32)
                fallback_embeddings.append(fallback_embedding.tobytes())
            return fallback_embeddings
    
    def _create_metadata(self, file_path: str, audio_result: Dict, clap_embedding: Optional[bytes]) -> Dict:
        """
        Erstellt Metadaten-Dictionary für eine Datei.
        
        Args:
            file_path (str): Pfad zur Originaldatei
            audio_result (Dict): Audio-Analyseergebnisse
            clap_embedding (Optional[bytes]): CLAP-Embedding oder None
            
        Returns:
            Dict: Metadaten-Dictionary
        """
        # Standardisierte Datei speichern
        stems_dir = "processed_database/stems"
        os.makedirs(stems_dir, exist_ok=True)
        
        # Eindeutige ID generieren
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        standardized_filename = f"{file_hash}_{os.path.splitext(os.path.basename(file_path))[0]}.wav"
        standardized_path = os.path.join(stems_dir, standardized_filename)
        
        # Als 24-bit WAV speichern
        sf.write(standardized_path, audio_result['audio_data'], 48000, subtype='PCM_24')
        
        # Semantische Analyse
        clap_tags = self._get_clap_tags(audio_result['audio_data'])
        category = self._get_category(file_path, audio_result['features'])
        
        # Fallback für CLAP-Embedding falls nicht vorhanden
        if clap_embedding is None:
            clap_embedding = self._get_clap_embedding(audio_result['audio_data'])
        
        # Eindeutige ID mit Kategorie generieren
        timestamp = int(datetime.now().timestamp())
        stem_id = f"{category}_{timestamp}_{file_hash[:4]}"
        
        return {
            'id': stem_id,
            'path': standardized_path,
            'bpm': audio_result['bpm'],
            'key': None,  # Wird später implementiert
            'category': category,
            'tags': json.dumps(clap_tags),
            'features': json.dumps(audio_result['features']),
            'quality_ok': True,
            'user_rating': None,
            'imported_at': datetime.now().isoformat(),
            'clap_embedding': clap_embedding
        }
    
    def _save_batch_results(self, results: List[ProcessingResult]):
        """
        Speichert die Ergebnisse eines Batches in der Datenbank.
        
        Args:
            results (List[ProcessingResult]): Liste der Verarbeitungsergebnisse
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            for result in results:
                # Update processing_status
                cursor.execute("""
                    INSERT OR REPLACE INTO processing_status 
                    (file_path, file_hash, status, last_attempt, retry_count, error_message, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.file_path,
                    hashlib.md5(result.file_path.encode()).hexdigest(),
                    result.status.value,
                    datetime.now().isoformat(),
                    result.retry_count,
                    result.error_message,
                    result.processing_time
                ))
                
                # Insert metadata if successful
                if result.status == ProcessingStatus.COMPLETED and result.metadata:
                    self._insert_meta_with_cursor(cursor, result.metadata)
                    self.stats['processed'] += 1
                elif result.status == ProcessingStatus.FAILED:
                    self.stats['failed'] += 1
                elif result.status == ProcessingStatus.QUARANTINED:
                    self.stats['quarantined'] += 1
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Batch-Ergebnisse: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _insert_meta_with_cursor(self, cursor, metadata: Dict):
        """
        Fügt Metadaten mit einem bestehenden Cursor ein.
        
        Args:
            cursor: SQLite-Cursor
            metadata (Dict): Metadaten-Dictionary
        """
        cursor.execute('''
            INSERT OR REPLACE INTO stems 
            (id, path, bpm, key, category, tags, features, quality_ok, user_rating, imported_at, clap_embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            metadata['imported_at'],
            metadata['clap_embedding']
        ))
    
    def _save_checkpoint(self):
        """
        Speichert einen Checkpoint des aktuellen Fortschritts.
        """
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy()
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Fehler beim Speichern des Checkpoints: {e}")
    
    def _print_final_statistics(self):
        """
        Druckt finale Statistiken der Verarbeitung.
        """
        if self.stats['start_time'] and self.stats['end_time']:
            total_time = self.stats['end_time'] - self.stats['start_time']
            
            logging.info("=== FINALE STATISTIKEN ===")
            logging.info(f"Gesamtzeit: {total_time:.2f} Sekunden")
            logging.info(f"Dateien gesamt: {self.stats['total_files']}")
            logging.info(f"Erfolgreich verarbeitet: {self.stats['processed']}")
            logging.info(f"Fehlgeschlagen: {self.stats['failed']}")
            logging.info(f"Quarantäne: {self.stats['quarantined']}")
            logging.info(f"Übersprungen: {self.stats['skipped']}")
            
            if self.stats['total_files'] > 0:
                success_rate = (self.stats['processed'] / self.stats['total_files']) * 100
                logging.info(f"Erfolgsrate: {success_rate:.1f}%")
            
            if self.stats['processed'] > 0:
                avg_time = total_time / self.stats['processed']
                logging.info(f"Durchschnittliche Zeit pro Datei: {avg_time:.2f} Sekunden")

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

    def _process_file_with_retry(self, file_path: str, max_retries: int = MAX_RETRIES) -> ProcessingResult:
        """
        Verarbeitet eine einzelne Audiodatei mit Retry-Logik.
        
        Args:
            file_path (str): Der Pfad zur zu verarbeitenden Datei
            max_retries (int): Maximale Anzahl von Wiederholungsversuchen
            
        Returns:
            ProcessingResult: Ergebnis der Verarbeitung
        """
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Audio laden und validieren
                audio_result = self._load_and_validate_audio(file_path)
                if not audio_result:
                    # Datei in Quarantäne verschieben
                    self._quarantine_file(file_path, "validation_failed")
                    return ProcessingResult(
                        file_path=file_path,
                        status=ProcessingStatus.QUARANTINED,
                        error_message="Audio-Validierung fehlgeschlagen",
                        retry_count=attempt
                    )
                
                # Metadaten erstellen
                metadata = self._create_metadata(file_path, audio_result, None)
                
                processing_time = time.time() - start_time
                
                logging.info(f"Verarbeitet: {os.path.basename(file_path)} -> BPM: {audio_result['bpm']:.1f}, Dauer: {audio_result['duration']:.2f}s")
                
                return ProcessingResult(
                    file_path=file_path,
                    status=ProcessingStatus.COMPLETED,
                    metadata=metadata,
                    retry_count=attempt,
                    processing_time=processing_time
                )
                
            except Exception as e:
                error_msg = f"Versuch {attempt + 1}/{max_retries + 1} fehlgeschlagen: {str(e)}"
                logging.warning(f"Fehler bei {file_path}: {error_msg}")
                
                if attempt < max_retries:
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logging.error(f"Alle Versuche fehlgeschlagen für {file_path}: {e}")
                    return ProcessingResult(
                        file_path=file_path,
                        status=ProcessingStatus.FAILED,
                        error_message=str(e),
                        retry_count=attempt
                    )
        
        # Sollte nie erreicht werden
        return ProcessingResult(
            file_path=file_path,
            status=ProcessingStatus.FAILED,
            error_message="Unbekannter Fehler",
            retry_count=max_retries
        )
    
    def _quarantine_file(self, file_path: str, reason: str):
        """
        Verschiebt eine Datei in die Quarantäne.
        
        Args:
            file_path (str): Pfad zur Datei
            reason (str): Grund für die Quarantäne
        """
        try:
            quarantine_dir = "processed_database/quarantine"
            os.makedirs(quarantine_dir, exist_ok=True)
            
            filename = os.path.basename(file_path)
            quarantine_path = os.path.join(quarantine_dir, f"{reason}_{filename}")
            
            # Kopiere statt verschieben, um Originaldaten zu erhalten
            shutil.copy2(file_path, quarantine_path)
            
            logging.info(f"Datei in Quarantäne: {file_path} -> {quarantine_path} (Grund: {reason})")
            
        except Exception as e:
            logging.error(f"Fehler beim Verschieben in Quarantäne: {e}")

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
    
    def _get_clap_embedding(self, audio_array: np.ndarray) -> bytes:
        """
        Berechnet CLAP-Embedding für ein Audio-Array und gibt es als Binärdaten zurück.
        
        Args:
            audio_array (np.ndarray): Das Audio-Array (48kHz, mono)
            
        Returns:
            bytes: Das CLAP-Embedding als Binärdaten für die Datenbank
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
            
            # Embedding normalisieren und als Binärdaten konvertieren
            if len(audio_embed.shape) > 1:
                audio_embed = audio_embed.flatten()
            
            # Normalisierung für bessere Suchperformance
            audio_embed = audio_embed / np.linalg.norm(audio_embed)
            
            # Als float32 Binärdaten speichern
            embedding_bytes = audio_embed.astype(np.float32).tobytes()
            
            logging.debug(f"CLAP-Embedding berechnet: {audio_embed.shape} -> {len(embedding_bytes)} bytes")
            return embedding_bytes
            
        except Exception as e:
            logging.error(f"Fehler bei CLAP-Embedding-Berechnung: {e}")
            # Fallback: Null-Embedding
            fallback_embedding = np.zeros(512, dtype=np.float32)
            return fallback_embedding.tobytes()
    
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
                (id, path, bpm, key, category, tags, features, quality_ok, user_rating, imported_at, clap_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                metadata['imported_at'],
                metadata['clap_embedding']
            ))
            
            conn.commit()
            conn.close()
            
            logging.debug(f"Metadaten gespeichert für ID: {metadata['id']}")
            
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Metadaten: {e}")
            if 'conn' in locals():
                conn.close()

if __name__ == "__main__":
    import argparse
    
    # Argument Parser für Konfiguration
    parser = argparse.ArgumentParser(description='Neuromorphe Traum-Engine Dataset Processor')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size for CLAP processing (default: {BATCH_SIZE})')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES, help=f'Maximum retry attempts (default: {MAX_RETRIES})')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL, help=f'Checkpoint save interval (default: {CHECKPOINT_INTERVAL})')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    parser.add_argument('--input-dir', type=str, default='raw_construction_kits', help='Input directory for audio files')
    
    args = parser.parse_args()
    
    # Logging konfigurieren
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('neuro_analyzer.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 60)
    logging.info("Neuromorphe Traum-Engine Dataset Processor v2.0")
    logging.info("AGENTEN_DIREKTIVE_006: Production-Ready Processing")
    logging.info("=" * 60)
    logging.info(f"Konfiguration:")
    logging.info(f"  - Resume: {args.resume}")
    logging.info(f"  - Batch Size: {args.batch_size}")
    logging.info(f"  - Max Retries: {args.max_retries}")
    logging.info(f"  - Checkpoint Interval: {args.checkpoint_interval}")
    logging.info(f"  - Input Directory: {args.input_dir}")
    logging.info(f"  - Log Level: {args.log_level}")
    logging.info("=" * 60)
    
    try:
        # Neuro-Analyzer initialisieren und ausführen
        analyzer = NeuroAnalyzer(input_dir=args.input_dir, resume_from_checkpoint=args.resume)
        analyzer.init_db()
        analyzer.run()
        
    except KeyboardInterrupt:
        logging.info("\nVerarbeitung durch Benutzer unterbrochen.")
        logging.info("Fortschritt wurde gespeichert. Verwenden Sie --resume zum Fortsetzen.")
    except Exception as e:
        logging.error(f"Kritischer Fehler: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise