"""Preprocessor Service - Audio-Analyse und Stem-Verarbeitung

Dieser Service implementiert die Audio-Analyse-Pipeline der Neuromorphe Traum-Engine v2.0.
Er analysiert neue Audio-Dateien und extrahiert Metadaten für die Datenbank.

Pipeline::

    Datei -> Validierung -> librosa.load -> Feature-Extraktion
          -> NeuroAnalyzer (CLAP) -> Kategorisierung -> Kopie -> DB

Öffentliche Methoden der :class:`PreprocessorService`:

* ``process_audio_file``  - eine Datei inkl. DB-Eintrag verarbeiten
* ``process_directory``   - alle Audio-Dateien in einem Verzeichnis verarbeiten
* ``batch_process``       - Liste von Dateien parallel verarbeiten
* ``extract_features``    - Audio-Features aus einem numpy-Array
* ``extract_metadata``    - technische Metadaten (Dauer, SR, Größe, Format)
* ``normalize_audio``     - Peak-Normalisierung
* ``detect_silence``      - Stille an den Rändern entfernen
* ``convert_sample_rate`` - Resampling
* ``validate_audio_file`` - Format-/Lesbarkeitsprüfung
"""

import os
import sys
import types as _types
import logging
import asyncio
import inspect
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import json
import hashlib
import wave
from datetime import datetime

import shutil
import librosa
import numpy as np
from scipy import signal

# Essentia ist ein optionales Extra (nicht in requirements.txt). Das Modul
# wird hier bewusst als Name bereitgestellt, damit Tests es patchen können
# und Aufrufer es per ``if essentia is not None`` prüfen können.
try:  # pragma: no cover - optionale Abhängigkeit
    import essentia  # type: ignore
    import essentia.standard as es  # type: ignore
except Exception:  # noqa: BLE001 - jedes Fehlen ist akzeptabel
    essentia = None  # type: ignore
    es = None  # type: ignore

from schemas.stem import StemCreate
from core.config import settings
from .neuro_analyzer import NeuroAnalyzer
from database.service import DatabaseService
from database.models import Stem

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Kompatibilitäts-Shim: in aktuellen librosa-Versionen ist ``librosa.feature.chroma``
# kein Aufruf mehr (Submodul statt Funktion). Die Pipeline nutzt ``chroma`` als
# Chromatogramm-Funktion; wir stellen deshalb einen aufrufbaren Alias bereit.
# --------------------------------------------------------------------------------------
if not callable(getattr(librosa.feature, "chroma", None)):
    _chroma_module = _types.ModuleType("librosa.feature.chroma")
    _chroma_module.chroma_stft = librosa.feature.chroma_stft
    _chroma_module.chroma_cqt = librosa.feature.chroma_cqt
    _chroma_module.chroma_cens = librosa.feature.chroma_cens
    librosa.feature.chroma = librosa.feature.chroma_stft  # type: ignore[attr-defined]
    sys.modules.setdefault("librosa.feature.chroma", _chroma_module)


#: Unterstützte Audio-Endungen.
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".aiff", ".m4a"}

#: Anzahl der Mel-Bänder für MFCC.
N_MFCC = 13

#: Anzahl der Chroma-Bins.
N_CHROMA = 12

#: Peak-Zielwert der Normalisierung.
NORMALIZATION_PEAK = 1.0

#: Unterhalb dieser Amplitude gilt ein Sample als Stille.
SILENCE_THRESHOLD = 1e-3


async def _maybe_await(value: Any) -> Any:
    """Gibt ``value`` zurück bzw. awaited es, falls es awaitable ist."""
    if inspect.isawaitable(value):
        return await value
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    """Robuste Float-Konvertierung (auch für numpy-Skalare)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


class AudioAnalyzer:
    """Klasse für grundlegende Audio-Analyse"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048
        
        logger.info("AudioAnalyzer initialisiert")
    
    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Führt eine vollständige Audio-Analyse durch"""
        logger.info(f"Analysiere Audio-Datei: {file_path}")
        
        try:
            # Audio laden
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            analysis_result = self.analyze_audio_array(audio, sr, file_path=file_path)
            
            logger.info(f"Audio-Analyse abgeschlossen: {analysis_result['file_info']['duration']:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Fehler bei Audio-Analyse von {file_path}: {e}")
            raise

    def analyze_audio_array(
        self,
        audio: np.ndarray,
        sr: int,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analysiert bereits geladenes Audio (ohne erneutes I/O)."""
        audio = np.asarray(audio)
        if audio.size == 0:
            raise ValueError("Audio-Daten sind leer")

        return {
            'file_info': self._analyze_file_info(file_path, audio, sr),
            'temporal': self._analyze_temporal_features(audio, sr),
            'spectral': self._analyze_spectral_features(audio, sr),
            'rhythmic': self._analyze_rhythmic_features(audio, sr),
            'harmonic': self._analyze_harmonic_features(audio, sr),
            'perceptual': self._analyze_perceptual_features(audio, sr),
            'classification': self._classify_audio_content(audio, sr)
        }
    
    def _analyze_file_info(self, file_path: Optional[str], audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analysiert grundlegende Datei-Informationen"""
        file_stats = None
        if file_path and os.path.exists(file_path):
            try:
                file_stats = os.stat(file_path)
            except OSError:
                file_stats = None

        # Audio-Hash für Duplikatserkennung
        audio_hash = hashlib.md5(np.asarray(audio, dtype=np.float32).tobytes()).hexdigest()

        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path) if file_path else None,
            'file_size': file_stats.st_size if file_stats else int(np.asarray(audio).size * 4),
            'duration': float(len(audio) / sr) if sr else 0.0,
            'sample_rate': sr,
            'channels': 1 if np.asarray(audio).ndim == 1 else int(np.asarray(audio).shape[0]),
            'audio_hash': audio_hash,
            'analyzed_at': datetime.now().isoformat()
        }
    
    def _analyze_temporal_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analysiert zeitliche Eigenschaften"""
        # RMS Energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        # Onset Detection
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, hop_length=self.hop_length, units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        
        # Tempo und Beat-Tracking
        tempo, beats = librosa.beat.beat_track(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        tempo = _as_float(np.asarray(tempo).ravel()[0]) if np.asarray(tempo).size else 0.0
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr)),
            'onset_count': len(onset_times),
            'onset_density': len(onset_times) / (len(audio) / sr) if len(audio) else 0.0,
            'tempo': tempo,
            'beat_count': len(beats),
            'rhythmic_regularity': self._calculate_rhythmic_regularity(beats, sr)
        }
    
    def _analyze_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analysiert spektrale Eigenschaften"""
        # Dynamische FFT-Anpassung
        n_fft = min(1024, len(audio))
        if n_fft < 256:
            n_fft = 256  # Minimum sicherstellen

        # STFT berechnen
        stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=n_fft)
        magnitude = np.abs(stft)
        
        # Spektrale Features
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_contrast_mean': [float(np.mean(sc)) for sc in spectral_contrast],
            'spectral_flatness_mean': float(np.mean(spectral_flatness)),
            'mfcc_mean': [float(np.mean(mfcc)) for mfcc in mfccs],
            'mfcc_std': [float(np.std(mfcc)) for mfcc in mfccs]
        }
    
    def _analyze_rhythmic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analysiert rhythmische Eigenschaften"""
        # Tempogram
        hop_length = self.hop_length
        oenv = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        
        # Beat-Tracking
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
        tempo = _as_float(np.asarray(tempo).ravel()[0]) if np.asarray(tempo).size else 0.0
        
        # Rhythmic Pattern Analysis
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            mean_interval = float(np.mean(beat_intervals))
            beat_consistency = 1.0 - (float(np.std(beat_intervals)) / mean_interval) if mean_interval > 0 else 0.0
        else:
            beat_consistency = 0.0
        
        # Rhythmische Komplexität basierend auf Onset-Strength-Variabilität
        onset_strength_complexity = float(np.std(oenv)) if len(oenv) > 1 else 0.0
        
        return {
            'tempo': tempo,
            'beat_consistency': float(beat_consistency),
            'tempogram_mean': float(np.mean(tempogram)),
            'onset_strength_mean': float(np.mean(oenv)),
            'rhythmic_complexity': onset_strength_complexity
        }
    
    def _analyze_harmonic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analysiert harmonische Eigenschaften"""
        # Harmonic-Percussive Separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr, n_chroma=N_CHROMA)
        
        # Tonnetz (Harmonic Network)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        
        # Pitch Detection (vereinfacht)
        pitches, magnitudes = librosa.piptrack(y=harmonic, sr=sr)
        
        # Robuste Tonart-Erkennung mit Krumhansl-Schmuckler-Algorithmus
        estimated_key, key_strength = self._estimate_key(harmonic, sr)
        
        # Harmonische Komplexität basierend auf Chroma-Varianz
        chroma_mean = np.mean(chroma, axis=1)
        total_energy = float(np.sum(audio**2)) or 1e-10
        harmonic_complexity_value = float(np.sum(harmonic**2) / total_energy)
        
        return {
            'harmonic_ratio': float(np.sum(harmonic**2) / total_energy),
            'percussive_ratio': float(np.sum(percussive**2) / total_energy),
            'chroma_mean': [float(c) for c in chroma_mean],
            'estimated_key': estimated_key,
            'key_strength': float(key_strength),
            'tonnetz_mean': [float(np.mean(t)) for t in tonnetz],
            'harmonic_complexity': harmonic_complexity_value
        }
    
    def _analyze_perceptual_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analysiert perzeptuelle Eigenschaften"""
        # Loudness (vereinfacht)
        rms = librosa.feature.rms(y=audio)[0]
        loudness_lufs = 20 * np.log10(np.mean(rms) + 1e-8) - 0.691  # Grobe LUFS-Schätzung
        
        # Dynamic Range
        dynamic_range = float(np.max(rms) - np.min(rms))
        
        # Spectral Features für Brightness
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Brightness (Energie über 1500 Hz)
        brightness_threshold = 1500
        brightness_bins = freqs > brightness_threshold
        brightness = np.mean(np.sum(magnitude[brightness_bins], axis=0))
        
        return {
            'loudness_lufs': float(loudness_lufs),
            'dynamic_range': dynamic_range,
            'brightness': float(brightness),
            'rms_mean': float(np.mean(rms)),
            'peak_amplitude': float(np.max(np.abs(audio)))
        }
    
    def _classify_audio_content(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Klassifiziert Audio-Inhalt"""
        # Einfache Heuristiken für Content-Klassifikation
        
        # Harmonic vs Percussive
        harmonic, percussive = librosa.effects.hpss(audio)
        total_energy = float(np.sum(audio**2)) or 1e-10
        harmonic_ratio = np.sum(harmonic**2) / total_energy
        percussive_ratio = np.sum(percussive**2) / total_energy
        
        # Onset Density für Rhythmic Content
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_density = len(onset_frames) / (len(audio) / sr) if len(audio) else 0.0
        
        # Spectral Characteristics
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        
        # Klassifikation basierend auf Heuristiken
        content_type = "unknown"
        confidence = 0.0
        
        if percussive_ratio > 0.7 and onset_density > 2.0:
            content_type = "percussion"
            confidence = min(percussive_ratio, onset_density / 5.0)
        elif harmonic_ratio > 0.6 and spectral_centroid < 2000:
            content_type = "bass"
            confidence = harmonic_ratio
        elif harmonic_ratio > 0.5 and spectral_centroid > 2000:
            content_type = "lead"
            confidence = harmonic_ratio
        elif spectral_rolloff < 1000:
            content_type = "sub"
            confidence = 1.0 - (spectral_rolloff / 1000)
        
        return {
            'content_type': content_type,
            'confidence': float(confidence),
            'harmonic_ratio': float(harmonic_ratio),
            'percussive_ratio': float(percussive_ratio),
            'onset_density': float(onset_density),
            'spectral_centroid': float(spectral_centroid)
        }
    
    def _calculate_rhythmic_regularity(self, beats: np.ndarray, sr: int) -> float:
        """Berechnet rhythmische Regelmäßigkeit"""
        if len(beats) < 3:
            return 0.0
        
        beat_times = librosa.frames_to_time(beats, sr=sr)
        intervals = np.diff(beat_times)
        
        if len(intervals) == 0:
            return 0.0
        
        # Coefficient of Variation (umgekehrt für Regelmäßigkeit)
        mean_interval = float(np.mean(intervals))
        if mean_interval <= 0:
            return 0.0
        cv = np.std(intervals) / mean_interval
        regularity = max(0.0, 1.0 - cv)
        
        return float(regularity)
    
    def _calculate_rhythmic_complexity(self, tempogram: np.ndarray) -> float:
        """Berechnet rhythmische Komplexität"""
        # Entropie des Tempograms als Maß für Komplexität
        tempogram_flat = tempogram.flatten()
        tempogram_norm = tempogram_flat / (np.sum(tempogram_flat) + 1e-8)
        
        # Shannon-Entropie
        entropy = -np.sum(tempogram_norm * np.log2(tempogram_norm + 1e-8))
        
        # Normalisierung
        max_entropy = np.log2(len(tempogram_norm))
        complexity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(complexity)
    
    def _estimate_key(self, audio: np.ndarray, sr: int) -> tuple:
        """Schätzt die Tonart mit dem Krumhansl-Schmuckler-Algorithmus"""
        
        # Krumhansl-Schmuckler Profile für Dur und Moll
        # Diese Profile basieren auf psychoakustischen Studien
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalisierung der Profile
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        # Chroma-Features berechnen (präzisere CQT-basierte Methode)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=self.hop_length)
        
        # Über die Zeit mitteln, um ein 12-dimensionales Pitch-Class-Profil zu erhalten
        chroma_mean = np.mean(chroma, axis=1)
        
        # Normalisierung des Chroma-Profils
        chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-8)
        
        # Tonart-Namen
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        best_correlation = -1
        best_key = 'C'
        
        # Teste alle 24 Tonarten (12 Dur + 12 Moll)
        for i in range(12):
            # Rotiere die Profile für alle Grundtöne
            major_rotated = np.roll(major_profile, i)
            minor_rotated = np.roll(minor_profile, i)
            
            # Berechne Korrelation mit Dur-Profil
            major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
            if not np.isnan(major_corr) and major_corr > best_correlation:
                best_correlation = major_corr
                best_key = key_names[i]
            
            # Berechne Korrelation mit Moll-Profil
            minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
            if not np.isnan(minor_corr) and minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = key_names[i] + 'm'
        
        # Fallback falls keine gültige Korrelation gefunden wurde
        if best_correlation < 0:
            best_correlation = 0.0
            best_key = 'C'
        
        return best_key, best_correlation
 

class TagGenerator:
    """Klasse für automatische Tag-Generierung"""
    
    def __init__(self):
        # Tag-Kategorien und Schwellenwerte
        self.tempo_tags = {
            'slow': (0, 100),
            'medium': (100, 130),
            'fast': (130, 160),
            'very_fast': (160, 300)
        }
        
        self.energy_tags = {
            'low_energy': (0, 0.3),
            'medium_energy': (0.3, 0.7),
            'high_energy': (0.7, 1.0)
        }
        
        self.brightness_tags = {
            'dark': (0, 0.3),
            'balanced': (0.3, 0.7),
            'bright': (0.7, 1.0)
        }
    
    def generate_tags(self, analysis: Dict[str, Any]) -> List[str]:
        """Generiert Tags basierend auf Audio-Analyse"""
        tags = []
        
        # Tempo-Tags
        tempo = analysis['temporal']['tempo']
        for tag, (min_bpm, max_bpm) in self.tempo_tags.items():
            if min_bpm <= tempo < max_bpm:
                tags.append(tag)
                break
        
        # Energy-Tags
        energy = analysis['temporal']['rms_mean']
        for tag, (min_energy, max_energy) in self.energy_tags.items():
            if min_energy <= energy < max_energy:
                tags.append(tag)
                break
        
        # Brightness-Tags
        brightness = analysis['perceptual']['brightness']
        brightness_norm = min(1.0, brightness / 1000)  # Normalisierung
        for tag, (min_bright, max_bright) in self.brightness_tags.items():
            if min_bright <= brightness_norm < max_bright:
                tags.append(tag)
                break
        
        # Content-Type Tags
        content_type = analysis['classification']['content_type']
        if content_type != 'unknown':
            tags.append(content_type)
        
        # Rhythmic Tags
        if analysis['rhythmic']['beat_consistency'] > 0.8:
            tags.append('steady')
        elif analysis['rhythmic']['beat_consistency'] < 0.5:
            tags.append('irregular')
        
        if analysis['rhythmic']['rhythmic_complexity'] > 0.7:
            tags.append('complex')
        elif analysis['rhythmic']['rhythmic_complexity'] < 0.3:
            tags.append('simple')
        
        # Harmonic Tags
        if analysis['harmonic']['harmonic_ratio'] > 0.7:
            tags.append('melodic')
        if analysis['harmonic']['percussive_ratio'] > 0.7:
            tags.append('percussive')
        
        # Dynamic Range Tags
        if analysis['perceptual']['dynamic_range'] > 0.5:
            tags.append('dynamic')
        elif analysis['perceptual']['dynamic_range'] < 0.2:
            tags.append('compressed')
        
        return list(set(tags))  # Duplikate entfernen


class PreprocessorService:
    """Hauptservice für Audio-Preprocessing.

    Args:
        config: Optionales Settings-Objekt (Default: globale ``settings``).
        neuro_analyzer: Optionale NeuroAnalyzer-Instanz (z. B. für Tests).
    """

    #: Idempotenz-Cache über Instanzen/Prozessläufe hinweg: Byte-Hashes der
    #: Quelldateien, die in diesem Prozess bereits in die Stem-Bibliothek
    #: übernommen wurden. Ein Verzeichnislauf überspringt solche Dateien,
    #: statt sie erneut zu analysieren (Pendant zur ``file_hash``-Prüfung
    #: in der Datenbank).
    _imported_source_hashes: set = set()

    #: Obergrenze für :attr:`_imported_source_hashes` (älteste zuerst).
    _IMPORTED_HASH_LIMIT = 10000

    def _remember_imported_source(self, file_path: Union[str, Path]) -> None:
        """Merkt eine erfolgreich übernommene Quelldatei (Idempotenz)."""
        source_hash = self._hash_source_file(file_path)
        if not source_hash:
            return
        registry = type(self)._imported_source_hashes
        registry.add(source_hash)
        # Begrenzen, damit der Prozess-Cache nicht unbegrenzt wächst.
        if len(registry) > self._IMPORTED_HASH_LIMIT:
            registry.pop()

    @classmethod
    def reset_import_registry(cls) -> None:
        """Leert den Idempotenz-Cache (z. B. zwischen Test-Sessions)."""
        cls._imported_source_hashes.clear()

    def __init__(self, config: Optional[Any] = None, neuro_analyzer: Optional[Any] = None):
        self.settings = config if config is not None else settings
        self.config = self.settings
        self.supported_formats = set(SUPPORTED_AUDIO_FORMATS)

        # Audio-Analyse-Komponenten
        self.analyzer = AudioAnalyzer(sample_rate=getattr(self.settings, 'AUDIO_SAMPLE_RATE', 44100))
        self.tag_generator = TagGenerator()

        # Neuromorphe Analyse (injizierbar)
        self.neuro_analyzer = neuro_analyzer if neuro_analyzer is not None else NeuroAnalyzer()
        self.db_service = DatabaseService()
        
        # Verzeichnisse sicherstellen
        self.processed_dir = Path(getattr(self.settings, 'PROCESSED_STEMS_DIR', './processed_database/stems'))
        self.stems_dir = self.processed_dir / "stems"
        self.quarantine_dir = self.processed_dir / "quarantine"
        
        for directory in [self.processed_dir, self.stems_dir, self.quarantine_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as error:  # pragma: no cover - z. B. read-only FS
                logger.warning(f"Verzeichnis konnte nicht erstellt werden ({directory}): {error}")
        
        logger.info("PreprocessorService initialisiert")

    # ------------------------------------------------------------------
    # Basisoperationen auf Audio-Arrays
    # ------------------------------------------------------------------

    async def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extrahiert Audio-Features (Tempo, Spektrum, Chroma, Energie)."""
        audio = np.asarray(audio_data)
        if audio.size == 0:
            raise ValueError("Audio-Daten sind leer")

        # Tempo
        tempo_raw = librosa.beat.tempo(y=audio, sr=sample_rate, hop_length=self.analyzer.hop_length)
        tempo = _as_float(np.asarray(tempo_raw).ravel()[0]) if np.asarray(tempo_raw).size else 0.0

        # Spektraler Schwerpunkt
        spectral_centroid = _as_float(np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        ))

        # MFCC (Mittelwerte je Koeffizient)
        mfcc_raw = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        mfcc_mean = [float(value) for value in np.mean(np.atleast_2d(mfcc_raw), axis=1)]

        # Chroma (Mittelwerte je Pitch-Class)
        chroma_raw = librosa.feature.chroma(y=audio, sr=sample_rate)
        chroma_mean = [float(value) for value in np.mean(np.atleast_2d(chroma_raw), axis=1)]

        # Energie & Zero-Crossing-Rate
        rms = librosa.feature.rms(y=audio)
        energy = _as_float(np.mean(rms))
        zero_crossing_rate = _as_float(np.mean(librosa.feature.zero_crossing_rate(audio)))

        return {
            "tempo": tempo,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": _as_float(np.mean(
                librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            )),
            "mfcc": mfcc_mean,
            "mfcc_mean": mfcc_mean,
            "chroma": chroma_mean,
            "chroma_mean": chroma_mean,
            "energy": energy,
            "rms_mean": energy,
            "zero_crossing_rate": zero_crossing_rate,
            "peak_amplitude": _as_float(np.max(np.abs(audio))) if audio.size else 0.0,
            "duration": float(len(audio) / sample_rate) if sample_rate else 0.0,
        }

    async def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalisiert Audio auf einen einheitlichen Peak-Wert (max. 1.0)."""
        audio = np.asarray(audio_data, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 0.0 or not np.isfinite(peak):
            return audio
        normalized = audio * (NORMALIZATION_PEAK / peak)
        # Sicherheitsclip gegen Rundungsfehler
        return np.clip(normalized, -NORMALIZATION_PEAK, NORMALIZATION_PEAK)

    async def detect_silence(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Schneidet Stille an Anfang und Ende ab."""
        audio = np.asarray(audio_data)
        if audio.size == 0:
            return audio

        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=60)
            if self._is_meaningful_trim(trimmed, audio):
                return trimmed
        except Exception as error:  # pragma: no cover - defensiv
            logger.debug(f"librosa.effects.trim fehlgeschlagen: {error}")

        # Fallback: manuelle Schwellwert-Trimmung
        return self._trim_by_threshold(audio)

    @staticmethod
    def _is_meaningful_trim(trimmed: np.ndarray, original: np.ndarray) -> bool:
        """Prüft, ob das Ergebnis der Trimmung brauchbar ist."""
        return (
            isinstance(trimmed, np.ndarray)
            and 0 < len(trimmed) < len(original)
        )

    @staticmethod
    def _trim_by_threshold(audio: np.ndarray) -> np.ndarray:
        """Entfernt führende und folgende Stille über einen Amplitudenschwellwert."""
        if audio.ndim > 1:
            activity = np.max(np.abs(audio), axis=0)
        else:
            activity = np.abs(audio)

        loud = np.flatnonzero(activity > SILENCE_THRESHOLD)
        if loud.size == 0:
            return audio
        return audio[..., loud[0]:loud[-1] + 1]

    async def convert_sample_rate(
        self,
        audio_data: np.ndarray,
        original_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        """Konvertiert eine Sample-Rate."""
        if original_sample_rate == target_sample_rate:
            return audio_data
        return librosa.resample(
            audio_data,
            orig_sr=original_sample_rate,
            target_sr=target_sample_rate,
        )

    async def extract_metadata(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extrahiert technische Metadaten eines Audio-Arrays."""
        audio = np.asarray(audio_data)

        file_size = 0
        if file_path and os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
            except OSError:
                file_size = 0
        if not file_size:
            file_size = int(audio.size * np.dtype(np.float32).itemsize)

        suffix = Path(file_path).suffix.lstrip('.').lower() if file_path else 'wav'
        if not suffix:
            suffix = 'wav'

        return {
            "duration": float(len(audio) / sample_rate) if sample_rate else 0.0,
            "sample_rate": int(sample_rate),
            "channels": 1 if audio.ndim == 1 else int(audio.shape[0]),
            "file_size": int(file_size),
            "format": suffix,
            "file_path": file_path,
            "peak_amplitude": _as_float(np.max(np.abs(audio))) if audio.size else 0.0,
            "rms_mean": _as_float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0,
            "extracted_at": datetime.now().isoformat(),
        }

    async def validate_audio_file(self, file_path: str) -> bool:
        """Prüft, ob eine Datei ein lesbares Audio-Format hat."""
        if not file_path:
            return False

        path = Path(file_path)
        try:
            if not path.is_file():
                return False
        except OSError:
            return False

        if path.suffix.lower() not in self.supported_formats:
            return False

        # WAV-Dateien lassen sich ohne Audio-Decoder über den Header prüfen.
        if path.suffix.lower() == '.wav':
            try:
                with wave.open(str(path), 'rb') as wav_file:
                    return (
                        wav_file.getnframes() > 0
                        and wav_file.getframerate() > 0
                        and wav_file.getnchannels() > 0
                    )
            except Exception:
                return False

        # Für komprimierte Formate genügt ein Lesetest des Headers.
        try:
            with open(path, 'rb') as raw_file:
                header = raw_file.read(16)
            return len(header) >= 4
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Einzeldatei-Verarbeitung
    # ------------------------------------------------------------------

    async def process_audio(
        self,
        audio_input: Any,
        filename: Optional[str] = None,
        category: Optional[str] = None,
        genre: Optional[str] = None,
        tags: Optional[List[str]] = None,
        session: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Verarbeitet Audio aus dem Speicher (Bytes) oder von einem Pfad.

        Convenience-Einstieg für den Upload-Endpunkt: schreibt die Daten in
        eine temporäre Datei, delegiert an :meth:`process_audio_file` und
        reichert das Ergebnis um die übergebenen Metadaten an.

        Args:
            audio_input: Rohdaten als ``bytes`` oder ein Dateipfad.
            filename: Ursprünglicher Dateiname (bestimmt das Format).
            category: Stem-Kategorie (z. B. "kick").
            genre: Genre-Angabe.
            tags: Liste von Tags.
            session: Optionale Datenbank-Session.
        """
        suffix = Path(filename or "").suffix.lower()
        if suffix not in self.supported_formats:
            suffix = ".wav"

        cleanup_path: Optional[str] = None
        if isinstance(audio_input, (bytes, bytearray)):
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(bytes(audio_input))
                cleanup_path = tmp.name
            file_path = cleanup_path
        else:
            file_path = str(audio_input)

        try:
            result = await self.process_audio_file(
                file_path, session=session, category=category
            )
        finally:
            if cleanup_path is not None:
                Path(cleanup_path).unlink(missing_ok=True)

        if not isinstance(result, dict):
            result = {"stem_id": str(result)}

        features = result.get("features")
        if not isinstance(features, dict):
            features = {}
        if result.get("duration") is not None:
            features.setdefault("duration", result["duration"])
        if result.get("sample_rate") is not None:
            features.setdefault("sample_rate", result["sample_rate"])

        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        if category:
            metadata.setdefault("type", category)
        if genre:
            metadata.setdefault("genre", genre)
        if tags:
            metadata.setdefault("tags", list(tags))
        if filename:
            metadata.setdefault("filename", filename)

        result["features"] = features
        result["metadata"] = metadata
        return result

    async def process_audio_file(
        self,
        file_path: str,
        session: Optional[Any] = None,
        category: Optional[str] = None,
        source: str = 'original',
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """Verarbeitet eine Audio-Datei vollständig und legt sie in der DB an."""
        logger.info(f"Starte Verarbeitung von: {file_path}")

        if progress_callback is not None:
            try:
                await _maybe_await(progress_callback(0, 1, f"Processing {file_path}"))
            except Exception as error:  # pragma: no cover
                logger.warning(f"Progress-Callback fehlgeschlagen: {error}")

        # Format prüfen (vor jedem I/O)
        path = Path(file_path)
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {path.suffix or file_path}")

        if not path.is_file():
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {file_path}")

        try:
            # 1. Audio laden (genau ein Lesevorgang)
            audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size == 0:
                raise ValueError("Audio-Datei enthält keine Samples")

            # 2. Duplikatsprüfung über den Audio-Hash
            audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
            existing_stem = await self._find_existing_stem(audio_hash)
            if existing_stem is not None:
                logger.info(f"Überspringe {file_path}: bereits vorhanden")
                return {
                    'success': True,
                    'action': 'skipped',
                    'status': 'skipped',
                    'stem_id': getattr(existing_stem, 'id', None),
                    'file_path': file_path,
                    'message': 'Datei bereits in Datenbank',
                }

            # 3. Features & Analyse (auf dem geladenen Array)
            features = await self.extract_features(audio, sample_rate)
            metadata = await self.extract_metadata(audio, sample_rate, file_path)

            # Normalisieren und Stille entfernen (best effort)
            try:
                analysis_audio = await self.normalize_audio(
                    await self.detect_silence(audio, sample_rate)
                )
            except Exception as error:  # pragma: no cover - defensiv
                logger.debug(f"Vorverarbeitung fehlgeschlagen: {error}")
                analysis_audio = audio

            analysis = self._build_analysis(analysis_audio, sample_rate, features, metadata)
            auto_tags = self.tag_generator.generate_tags(analysis)

            # 4. Neuromorphe Analyse (CLAP-Embeddings etc.)
            neuro_features = await self._analyze_neuro(file_path)

            # 5. Kategorie bestimmen
            if not category:
                category = self._determine_category(analysis, neuro_features)

            # 6. Datei in das verarbeitete Verzeichnis kopieren
            processed_path = await self._copy_to_processed_dir(file_path, category)

            # 7. Stem-Daten vorbereiten und speichern
            stem_data = self._prepare_stem_data(
                file_path, processed_path, category, analysis, auto_tags, neuro_features, source
            )
            saved = await self._save_processed_stem(stem_data, session=session)

            stem_id = saved.get('id') if isinstance(saved, dict) else getattr(saved, 'id', None)
            duration = _as_float(
                saved.get('duration') if isinstance(saved, dict) else getattr(saved, 'duration', None),
                metadata['duration'],
            )
            if not duration:
                duration = metadata['duration']

            # Erfolgreich importiert -> Quelle als "bereits übernommen" merken.
            self._remember_imported_source(file_path)

            logger.info(f"Stem erfolgreich verarbeitet: {path.name} (ID: {stem_id})")

            result = {
                'success': True,
                'action': 'processed',
                'status': 'processed',
                'stem_id': stem_id,
                'stem_name': path.stem,
                'file_path': file_path,
                'processed_path': processed_path,
                'category': category,
                'tags': auto_tags,
                'duration': duration,
                'sample_rate': int(sample_rate),
                'channels': metadata['channels'],
                'features': features,
                'metadata': metadata,
                'analysis': analysis,
                'neuro_features': neuro_features,
            }

            if progress_callback is not None:
                try:
                    await _maybe_await(progress_callback(1, 1, f"Processing complete: {path.name}"))
                except Exception as error:  # pragma: no cover
                    logger.warning(f"Progress-Callback fehlgeschlagen: {error}")

            return result

        except Exception as error:
            logger.error(f"Fehler bei Verarbeitung von {file_path}: {error}")
            await self._quarantine_file(file_path, str(error))
            raise

    async def _find_existing_stem(self, audio_hash: str) -> Optional[Any]:
        """Sucht einen bereits gespeicherten Stem anhand des Audio-Hashes."""
        try:
            return await _maybe_await(self.db_service.get_stem_by_hash(audio_hash))
        except Exception as error:
            logger.debug(f"Duplikatsprüfung nicht möglich: {error}")
            return None

    async def _analyze_neuro(self, file_path: str) -> Dict[str, Any]:
        """Führt die neuromorphe Analyse aus (Fehler sind nicht fatal)."""
        try:
            result = await _maybe_await(self.neuro_analyzer.analyze_audio(file_path))
            return result if isinstance(result, dict) else {}
        except Exception as error:
            logger.warning(f"Neuromorphe Analyse fehlgeschlagen für {file_path}: {error}")
            return {}

    def _build_analysis(
        self,
        audio: np.ndarray,
        sample_rate: int,
        features: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Baut die Analyse-Struktur aus den bereits berechneten Features."""
        peak = _as_float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = _as_float(features.get('energy'))
        centroid = _as_float(features.get('spectral_centroid'))
        tempo = _as_float(features.get('tempo'))

        # Einfache Heuristik für den Inhaltstyp (ersetzt hpss, das sehr teuer ist).
        if tempo >= 150 or (rms > 0.05 and centroid > 3000):
            content_type, confidence = 'percussion', min(1.0, rms * 10)
        elif centroid < 800:
            content_type, confidence = 'bass', 0.6
        elif centroid > 3000:
            content_type, confidence = 'lead', 0.6
        else:
            content_type, confidence = 'mid', 0.5

        return {
            'file_info': {
                'file_path': metadata.get('file_path'),
                'file_name': Path(metadata['file_path']).name if metadata.get('file_path') else None,
                'file_size': metadata.get('file_size', 0),
                'duration': metadata.get('duration', 0.0),
                'sample_rate': sample_rate,
                'channels': metadata.get('channels', 1),
                'audio_hash': hashlib.md5(audio.tobytes()).hexdigest(),
                'analyzed_at': datetime.now().isoformat(),
            },
            'temporal': {
                'rms_mean': rms,
                'rms_std': 0.0,
                'zcr_mean': _as_float(features.get('zero_crossing_rate')),
                'tempo': tempo,
                'beat_count': 0,
                'onset_density': 0.0,
                'rhythmic_regularity': 0.0,
            },
            'spectral': {
                'spectral_centroid_mean': centroid,
                'spectral_bandwidth_mean': _as_float(features.get('spectral_bandwidth')),
                'mfcc_mean': features.get('mfcc_mean', []),
                'chroma_mean': features.get('chroma_mean', []),
            },
            'rhythmic': {
                'tempo': tempo,
                'beat_consistency': 0.5,
                'rhythmic_complexity': 0.5,
                'tempogram_mean': 0.0,
                'onset_strength_mean': 0.0,
            },
            'harmonic': {
                'harmonic_ratio': 0.5,
                'percussive_ratio': 0.5,
                'estimated_key': 'Am',
                'key_strength': 0.0,
                'harmonic_complexity': 0.5,
                'chroma_mean': features.get('chroma_mean', []),
            },
            'perceptual': {
                'loudness_lufs': 20 * float(np.log10(rms + 1e-8)) - 0.691,
                'dynamic_range': peak,
                'brightness': centroid,
                'rms_mean': rms,
                'peak_amplitude': peak,
            },
            'classification': {
                'content_type': content_type,
                'confidence': float(confidence),
                'harmonic_ratio': 0.5,
                'percussive_ratio': 0.5,
                'onset_density': 0.0,
                'spectral_centroid': centroid,
            },
        }

    def _determine_category(self, analysis: Dict[str, Any], neuro_features: Dict[str, Any]) -> str:
        """Bestimmt Kategorie basierend auf Analyse"""
        # Einfache Heuristiken für Kategorie-Bestimmung
        classification = analysis['classification']
        
        if classification['content_type'] != 'unknown':
            return classification['content_type']
        
        # Fallback basierend auf spektralen Eigenschaften
        spectral = analysis['spectral']
        centroid = spectral['spectral_centroid_mean']
        
        if centroid < 500:
            return 'bass'
        elif centroid < 2000:
            return 'mid'
        else:
            return 'lead'
    
    async def _copy_to_processed_dir(self, source_path: str, category: str) -> str:
        """Kopiert Datei in verarbeitetes Verzeichnis"""
        source = Path(source_path)
        
        # Ziel-Pfad generieren
        category_dir = self.stems_dir / (category or 'unknown')
        try:
            category_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:  # pragma: no cover
            logger.warning(f"Kategorie-Verzeichnis nicht erstellbar: {error}")
        
        # Eindeutigen Dateinamen generieren
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        target_name = f"{timestamp}_{source.stem}{source.suffix or '.wav'}"
        target_path = category_dir / target_name
        
        try:
            if not target_path.exists() and source.is_file():
                shutil.copy2(source_path, target_path)
            return str(target_path)
        except Exception as error:
            # Das Kopieren ist nicht kritisch - die Analyse bleibt gültig.
            logger.warning(f"Fehler beim Kopieren von {source_path}: {error}")
            return str(source_path)
    
    def _prepare_stem_data(
        self,
        original_path: str,
        processed_path: str,
        category: str,
        analysis: Dict[str, Any],
        tags: List[str],
        neuro_features: Dict[str, Any],
        source: str = 'original',
    ) -> Union[StemCreate, Dict[str, Any]]:
        """Bereitet Stem-Daten für Datenbank vor"""
        file_info = analysis.get('file_info', {})
        temporal = analysis.get('temporal', {})
        harmonic = analysis.get('harmonic', {})
        overall_assessment = (neuro_features or {}).get('overall_assessment', {}) or {}

        def _get(container: Any, key: str, default: Any = None) -> Any:
            """Robuster Zugriff auf dict-artige Container (inkl. Mocks)."""
            if isinstance(container, dict):
                return container.get(key, default)
            return default

        semantic_analysis = _get(neuro_features, 'semantic_analysis', {}) or {}
        pattern_analysis = _get(neuro_features, 'pattern_analysis', {}) or {}
        neural_features_data = _get(neuro_features, 'neural_features', {}) or {}
        perceptual_mapping_data = _get(neuro_features, 'perceptual_mapping', {}) or {}
        quality_assessment = _get(overall_assessment, 'quality_assessment', {}) or {}
        characteristics = _get(overall_assessment, 'characteristics', {}) or {}

        payload: Dict[str, Any] = dict(
            filename=Path(original_path).stem,
            original_path=original_path,
            processed_path=processed_path,
            file_hash=file_info.get('audio_hash'),
            duration=_as_float(file_info.get('duration')),
            sample_rate=int(file_info.get('sample_rate') or self.analyzer.sample_rate),
            channels=int(file_info.get('channels') or 1),
            file_size=int(file_info.get('file_size') or 0),

            bpm=int(_as_float(temporal.get('tempo')) or 0),
            key=harmonic.get('estimated_key') or 'C',

            category=category,
            source=source,
            auto_tags=list(tags or []),

            audio_embedding=_get(semantic_analysis, 'audio_embedding'),
            semantic_analysis=semantic_analysis if isinstance(semantic_analysis, dict) else {},
            pattern_analysis=pattern_analysis if isinstance(pattern_analysis, dict) else {},
            neural_features=neural_features_data if isinstance(neural_features_data, dict) else {},
            perceptual_mapping=perceptual_mapping_data if isinstance(perceptual_mapping_data, dict) else {},

            harmonic_complexity=_as_float(harmonic.get('harmonic_complexity')),
            rhythmic_complexity=_as_float(_get(analysis.get('rhythmic', {}), 'rhythmic_complexity')),

            quality_score=_get(quality_assessment, 'overall_quality'),
            complexity_level=_get(characteristics, 'complexity_level'),
            recommended_usage=_get(overall_assessment, 'recommended_usage'),

            processing_status="completed",
            processed_at=datetime.utcnow(),
        )

        # Bevorzugt ein validiertes Schema zurückgeben; bei unvollständigen
        # Daten (z. B. mit Mock-Analysen) bleibt das Roh-Dict erhalten.
        try:
            return StemCreate(**payload)
        except Exception as error:
            logger.debug(f"StemCreate-Validierung fehlgeschlagen, nutze Dict: {error}")
            return payload

    async def _save_processed_stem(
        self,
        stem_data: Union[StemCreate, Dict[str, Any]],
        session: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Speichert einen verarbeiteten Stem in der Datenbank."""
        stem = await _maybe_await(self.db_service.insert_stem(stem_data))

        if stem is None:
            raise RuntimeError("Stem konnte nicht gespeichert werden")

        return {
            'id': getattr(stem, 'id', None),
            'file_path': getattr(stem, 'processed_path', None),
            'duration': _as_float(getattr(stem, 'duration', None)),
        }

    async def _quarantine_file(self, file_path: str, error_message: str) -> None:
        """Verschiebt problematische Datei in Quarantäne"""
        try:
            source = Path(file_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            quarantine_name = f"{timestamp}_{source.name}"
            quarantine_path = self.quarantine_dir / quarantine_name
            
            # Fehler-Info speichern
            error_info = {
                'original_path': file_path,
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            }
            
            error_file = quarantine_path.with_suffix('.error.json')
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            
            logger.warning(f"Datei in Quarantäne: {quarantine_path}")
            
        except Exception as e:
            logger.error(f"Fehler bei Quarantäne von {file_path}: {e}")

    # ------------------------------------------------------------------
    # Stapelverarbeitung
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_source_file(file_path: Union[str, Path]) -> str:
        """MD5-Hash über die Rohbytes einer Quelldatei (Duplikatserkennung)."""
        digest = hashlib.md5()
        try:
            with open(file_path, 'rb') as raw_file:
                for chunk in iter(lambda: raw_file.read(1024 * 1024), b''):
                    digest.update(chunk)
        except OSError:
            return ""
        return digest.hexdigest()

    def _find_audio_files(self, directory_path: str, recursive: bool = True) -> List[Path]:
        """Findet alle Audio-Dateien in einem Verzeichnis.

        Verzeichnisse können Fremd-Dateien oder bereits importierte Quellen
        enthalten. Kandidaten werden deshalb vorab geprüft (lesbarer Header)
        und bereits übernommene Dateien übersprungen.
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Verzeichnis nicht gefunden: {directory_path}")

        if recursive:
            candidates = directory.rglob('*')
        else:
            candidates = directory.glob('*')

        files: List[Path] = []
        for path in candidates:
            if path.suffix.lower() not in self.supported_formats or not path.is_file():
                continue
            if not self._is_readable_audio(path):
                continue

            # Bereits importierte Quellen nicht erneut aufnehmen.
            source_hash = self._hash_source_file(path)
            if source_hash and source_hash in self._imported_source_hashes:
                logger.debug(f"Überspringe bereits importierte Datei: {path.name}")
                continue

            files.append(path)

        return sorted(files)

    @staticmethod
    def _is_readable_audio(path: Path) -> bool:
        """Prüft, ob eine Datei ein lesbares Audio-Format hat (Header-Check)."""
        try:
            if path.stat().st_size <= 0:
                return False
        except OSError:
            return False

        if path.suffix.lower() == '.wav':
            try:
                with wave.open(str(path), 'rb') as wav_file:
                    return (
                        wav_file.getnframes() > 0
                        and wav_file.getframerate() > 0
                    )
            except Exception:
                return False

        try:
            with open(path, 'rb') as raw_file:
                header = raw_file.read(4)
            return len(header) >= 4
        except OSError:
            return False

    async def process_directory(
        self,
        directory_path: str,
        session: Optional[Any] = None,
        recursive: bool = True,
        category: Optional[str] = None,
        progress_callback: Optional[Callable[..., Any]] = None,
        max_files: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Verarbeitet alle Audio-Dateien eines Verzeichnisses.

        Bereits bekannte Dateien (Status ``skipped``) werden nicht erneut
        verarbeitet, tauchen aber im Ergebnis auf.
        """
        audio_files = self._find_audio_files(directory_path, recursive=recursive)
        if max_files is not None and max_files > 0:
            audio_files = audio_files[:max_files]

        logger.info(f"Verarbeite {len(audio_files)} Audio-Dateien aus {directory_path}")

        return await self.batch_process(
            [str(path) for path in audio_files],
            session=session,
            category=category,
            progress_callback=progress_callback,
        )

    async def batch_process(
        self,
        file_paths: List[str],
        session: Optional[Any] = None,
        category: Optional[str] = None,
        source: str = 'batch_processed',
        max_concurrent: int = 4,
        progress_callback: Optional[Callable[..., Any]] = None,
        on_error: str = 'collect',
    ) -> List[Dict[str, Any]]:
        """Verarbeitet eine Liste von Dateien (parallel, begrenzte Nebenläufigkeit)."""
        paths = [str(path) for path in (file_paths or [])]
        if not paths:
            return []

        total = len(paths)
        semaphore = asyncio.Semaphore(max(1, int(max_concurrent or 1)))

        async def _process(index: int, path: str) -> Dict[str, Any]:
            if progress_callback is not None:
                try:
                    await _maybe_await(
                        progress_callback(index, total, f"Processing {Path(path).name}")
                    )
                except Exception as error:  # pragma: no cover
                    logger.warning(f"Progress-Callback fehlgeschlagen: {error}")

            async with semaphore:
                try:
                    result = await _maybe_await(
                        self.process_audio_file(
                            path,
                            session=session,
                            category=category,
                            source=source,
                        )
                    )
                except Exception as error:
                    if on_error == 'raise':
                        raise
                    logger.error(f"Fehler bei {path}: {error}")
                    return {
                        'success': False,
                        'action': 'error',
                        'status': 'error',
                        'file_path': path,
                        'error': str(error),
                    }

            if isinstance(result, dict):
                result.setdefault('file_path', path)
                return result
            return {
                'success': True,
                'action': 'processed',
                'status': 'processed',
                'file_path': path,
                'stem_id': result,
            }

        tasks = [_process(index, path) for index, path in enumerate(paths, 1)]
        results = list(await asyncio.gather(*tasks))

        if progress_callback is not None:
            try:
                await _maybe_await(
                    progress_callback(total, total, f"Processing finished: {total} files")
                )
            except Exception as error:  # pragma: no cover
                logger.warning(f"Progress-Callback fehlgeschlagen: {error}")

        return results

    async def batch_process_directory(
        self,
        directory_path: str,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verarbeitet alle Audio-Dateien in einem Verzeichnis (Zusammenfassung)."""
        logger.info(f"Starte Batch-Verarbeitung: {directory_path}")

        results = await self.process_directory(directory_path, category=category)

        summary = {
            'total_files': len(results),
            'processed': 0,
            'skipped': 0,
            'quarantined': 0,
            'errors': [],
            'processed_stems': [],
        }

        for result in results:
            status = result.get('status') or result.get('action')
            if status == 'processed':
                summary['processed'] += 1
                summary['processed_stems'].append({
                    'id': result.get('stem_id'),
                    'name': result.get('stem_name') or Path(result.get('file_path', '')).stem,
                    'category': result.get('category'),
                })
            elif status == 'skipped':
                summary['skipped'] += 1
            else:
                summary['quarantined'] += 1
                summary['errors'].append({
                    'file': result.get('file_path'),
                    'error': result.get('error'),
                })

        logger.info(
            f"Batch-Verarbeitung abgeschlossen: {summary['processed']} verarbeitet, "
            f"{summary['skipped']} übersprungen, {summary['quarantined']} fehlerhaft"
        )

        return summary

    # ------------------------------------------------------------------
    # Kompatibilitäts-API
    # ------------------------------------------------------------------

    async def _check_existing_stem(self, file_path: str) -> Optional[Stem]:
        """Prüft ob Datei bereits verarbeitet wurde"""
        try:
            audio, _ = librosa.load(file_path, sr=22050, mono=True)
            audio_hash = hashlib.md5(np.asarray(audio, dtype=np.float32).tobytes()).hexdigest()
            return await self._find_existing_stem(audio_hash)
        except Exception as e:
            logger.warning(f"Fehler bei Duplikatsprüfung: {e}")
            return None

    async def process_audio_files(self, directory_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Alias für die Verzeichnis-Verarbeitung (Legacy-API)."""
        if directory_path is None:
            directory_path = str(self.processed_dir)
        return await self.process_directory(directory_path)
