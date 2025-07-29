"""Preprocessor Service - Audio-Analyse und Stem-Verarbeitung

Dieser Service implementiert die Audio-Analyse-Pipeline der Neuromorphe Traum-Engine v2.0.
Er analysiert neue Audio-Dateien und extrahiert Metadaten für die Datenbank.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import hashlib
from datetime import datetime

import librosa
import numpy as np
from scipy import signal
# import essentia
# import essentia.standard as es

from schemas.stem import StemCreate
from core.config import settings
from services.neuro_analyzer import NeuroAnalyzer
from database.service import DatabaseService
from database.models import Stem

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Klasse für grundlegende Audio-Analyse"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048
        
        # Essentia-Algorithmen initialisieren (deaktiviert)
        # self.windowing = es.Windowing(type='hann')
        # self.spectrum = es.Spectrum()
        # self.spectral_peaks = es.SpectralPeaks()
        # self.pitch_detection = es.PitchYinProbabilistic()
        # self.onset_detection = es.OnsetDetection(method='hfc')
        # self.beats_loudness = es.BeatsLoudness()
        
        logger.info("AudioAnalyzer initialisiert")
    
    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Führt eine vollständige Audio-Analyse durch"""
        logger.info(f"Analysiere Audio-Datei: {file_path}")
        
        try:
            # Audio laden
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Grundlegende Eigenschaften
            duration = len(audio) / sr
            
            # Verschiedene Analysen durchführen
            analysis_result = {
                'file_info': self._analyze_file_info(file_path, audio, sr),
                'temporal': self._analyze_temporal_features(audio, sr),
                'spectral': self._analyze_spectral_features(audio, sr),
                'rhythmic': self._analyze_rhythmic_features(audio, sr),
                'harmonic': self._analyze_harmonic_features(audio, sr),
                'perceptual': self._analyze_perceptual_features(audio, sr),
                'classification': self._classify_audio_content(audio, sr)
            }
            
            logger.info(f"Audio-Analyse abgeschlossen: {duration:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Fehler bei Audio-Analyse von {file_path}: {e}")
            raise
    
    def _analyze_file_info(self, file_path: str, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analysiert grundlegende Datei-Informationen"""
        file_stats = os.stat(file_path)
        
        # Audio-Hash für Duplikatserkennung
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
        
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': file_stats.st_size,
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'channels': 1,  # Mono nach librosa.load
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
        
        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr)),
            'onset_count': len(onset_times),
            'onset_density': len(onset_times) / (len(audio) / sr),
            'tempo': float(tempo),
            'beat_count': len(beats),
            'rhythmic_regularity': self._calculate_rhythmic_regularity(beats, sr)
        }
    
    def _analyze_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analysiert spektrale Eigenschaften"""
        # STFT berechnen
        stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.frame_length)
        magnitude = np.abs(stft)
        
        # Spektrale Features
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
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
        
        # Rhythmic Pattern Analysis
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            beat_consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
        else:
            beat_consistency = 0.0
        
        # Rhythmische Komplexität basierend auf Onset-Strength-Variabilität
        onset_strength_complexity = float(np.std(oenv)) if len(oenv) > 1 else 0.0
        
        return {
            'tempo': float(tempo),
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
        chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
        
        # Tonnetz (Harmonic Network)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        
        # Pitch Detection (vereinfacht)
        pitches, magnitudes = librosa.piptrack(y=harmonic, sr=sr)
        
        # Robuste Tonart-Erkennung mit Krumhansl-Schmuckler-Algorithmus
        estimated_key, key_strength = self._estimate_key(harmonic, sr)
        
        # Harmonische Komplexität basierend auf Chroma-Varianz
        chroma_mean = np.mean(chroma, axis=1)
        harmonic_complexity_value = float(np.sum(harmonic**2) / np.sum(audio**2))
        
        return {
            'harmonic_ratio': float(np.sum(harmonic**2) / np.sum(audio**2)),
            'percussive_ratio': float(np.sum(percussive**2) / np.sum(audio**2)),
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
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
        
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
        harmonic_ratio = np.sum(harmonic**2) / np.sum(audio**2)
        percussive_ratio = np.sum(percussive**2) / np.sum(audio**2)
        
        # Onset Density für Rhythmic Content
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_density = len(onset_frames) / (len(audio) / sr)
        
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
        cv = np.std(intervals) / np.mean(intervals)
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
    
    def _estimate_key(self, audio: np.ndarray, sr: int) -> tuple[str, float]:
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
    """Hauptservice für Audio-Preprocessing"""
    
    def __init__(self, config: settings):
        self.config = config
        self.analyzer = AudioAnalyzer(sample_rate=self.config.AUDIO_SAMPLE_RATE)
        self.tag_generator = TagGenerator()
        self.neuro_analyzer = NeuroAnalyzer()
        self.db_service = DatabaseService()
        
        # Verzeichnisse sicherstellen
        self.processed_dir = Path(self.config.PROCESSED_STEMS_DIR)
        self.stems_dir = self.processed_dir / "stems"
        self.quarantine_dir = self.processed_dir / "quarantine"
        
        for directory in [self.processed_dir, self.stems_dir, self.quarantine_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("PreprocessorService initialisiert")
    
    async def process_audio_file(self, file_path: str, category: str = None, source: str = 'original') -> Dict[str, Any]:
        """Verarbeitet eine Audio-Datei vollständig"""
        logger.info(f"Starte Verarbeitung von: {file_path}")
        
        try:
            # Prüfe ob Datei bereits verarbeitet wurde
            existing_stem = await self._check_existing_stem(file_path)
            if existing_stem:
                logger.info(f"Datei bereits verarbeitet: {existing_stem['name']}")
                return {
                    'success': True,
                    'action': 'skipped',
                    'stem_id': existing_stem['id'],
                    'message': 'Datei bereits in Datenbank'
                }
            
            # Audio-Analyse durchführen
            analysis = self.analyzer.analyze_audio_file(file_path)
            
            # Tags generieren
            auto_tags = self.tag_generator.generate_tags(analysis)
            
            # Neuromorphe Analyse
            neuro_features = await self.neuro_analyzer.analyze_audio(file_path)
            
            # Kategorie bestimmen (falls nicht angegeben)
            if not category:
                category = self._determine_category(analysis, neuro_features)
            
            # Datei in verarbeitetes Verzeichnis kopieren
            processed_path = await self._copy_to_processed_dir(file_path, category)
            
            # Stem-Daten für Datenbank vorbereiten
            stem_data = self._prepare_stem_data(
                file_path, processed_path, category, analysis, auto_tags, neuro_features, source
            )
            
            # In Datenbank speichern
            stem = await self.db_service.insert_stem(stem_data)
            
            logger.info(f"Stem erfolgreich verarbeitet: {stem.name} (ID: {stem.id})")
            
            return {
                'success': True,
                'action': 'processed',
                'stem_id': stem.id,
                'stem_name': stem.name,
                'category': category,
                'tags': auto_tags,
                'analysis': analysis,
                'neuro_features': neuro_features
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Verarbeitung von {file_path}: {e}")
            
            # Datei in Quarantäne verschieben
            await self._quarantine_file(file_path, str(e))
            
            return {
                'success': False,
                'action': 'quarantined',
                'error': str(e),
                'file_path': file_path
            }
    
    async def _check_existing_stem(self, file_path: str) -> Optional[Stem]:
        """Prüft ob Datei bereits verarbeitet wurde"""
        try:
            # Lade Audio für Hash-Berechnung
            audio, _ = librosa.load(file_path, sr=22050, mono=True)
            audio_hash = hashlib.md5(audio.tobytes()).hexdigest()
            
            # Prüfe Datenbank
            existing_stem = await self.db_service.get_stem_by_hash(audio_hash)
            
            return existing_stem
            
        except Exception as e:
            logger.warning(f"Fehler bei Duplikatsprüfung: {e}")
            return None
    
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
        category_dir = self.stems_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Eindeutigen Dateinamen generieren
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        target_name = f"{timestamp}_{source.stem}.wav"
        target_path = category_dir / target_name
        
        # Datei kopieren (hier würde man normalerweise shutil.copy2 verwenden)
        # Für Demo: symbolischen Link erstellen
        try:
            if not target_path.exists():
                # In Produktion: shutil.copy2(source_path, target_path)
                target_path.write_text(f"Processed: {source_path}")
            
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Fehler beim Kopieren von {source_path}: {e}")
            raise
    
    def _prepare_stem_data(self, original_path: str, processed_path: str, category: str,
                          analysis: Dict[str, Any], tags: List[str],
                          neuro_features: Dict[str, Any], source: str = 'original') -> StemCreate:
        """Bereitet Stem-Daten für Datenbank vor"""
        file_info = analysis['file_info']
        temporal = analysis['temporal']
        harmonic = analysis['harmonic']
        perceptual = analysis['perceptual']
        
        # Extrahiere Daten aus neuro_features
        semantic_analysis = neuro_features.get('semantic_analysis', {})
        pattern_analysis = neuro_features.get('pattern_analysis', {})
        neural_features_data = neuro_features.get('neural_features', {})
        perceptual_mapping_data = neuro_features.get('perceptual_mapping', {})
        overall_assessment = neuro_features.get('overall_assessment', {})

        return StemCreate(
            filename=Path(original_path).stem,
            original_path=original_path,
            processed_path=processed_path,
            file_hash=file_info['audio_hash'],
            duration=file_info['duration'],
            sample_rate=file_info['sample_rate'],
            channels=file_info['channels'],
            file_size=file_info['file_size'],
            
            bpm=int(temporal['tempo']),
            key=harmonic['estimated_key'],
            
            category=category,
            source=source,
            auto_tags=tags,
            
            audio_embedding=semantic_analysis.get('audio_embedding'),
            semantic_analysis=semantic_analysis,
            pattern_analysis=pattern_analysis,
            neural_features=neural_features_data,
            perceptual_mapping=perceptual_mapping_data,
            
            harmonic_complexity=harmonic['harmonic_complexity'],
            rhythmic_complexity=analysis['rhythmic']['rhythmic_complexity'],
            
            quality_score=overall_assessment.get('quality_assessment', {}).get('overall_quality'),
            complexity_level=overall_assessment.get('characteristics', {}).get('complexity_level'),
            recommended_usage=overall_assessment.get('recommended_usage'),
            
            processing_status="completed",
            processed_at=datetime.utcnow()
        )
    
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
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            
            logger.warning(f"Datei in Quarantäne: {quarantine_path}")
            
        except Exception as e:
            logger.error(f"Fehler bei Quarantäne von {file_path}: {e}")
    
    async def batch_process_directory(self, directory_path: str, 
                                    category: str = None) -> Dict[str, Any]:
        """Verarbeitet alle Audio-Dateien in einem Verzeichnis"""
        logger.info(f"Starte Batch-Verarbeitung: {directory_path}")
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Verzeichnis nicht gefunden: {directory_path}")
        
        # Unterstützte Audio-Formate
        audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.m4a'}
        
        # Alle Audio-Dateien finden
        audio_files = [
            f for f in directory.rglob('*') 
            if f.suffix.lower() in audio_extensions and f.is_file()
        ]
        
        logger.info(f"Gefunden: {len(audio_files)} Audio-Dateien")
        
        results = {
            'total_files': len(audio_files),
            'processed': 0,
            'skipped': 0,
            'quarantined': 0,
            'errors': [],
            'processed_stems': []
        }
        
        # Verarbeite jede Datei
        for i, file_path in enumerate(audio_files, 1):
            logger.info(f"Verarbeite {i}/{len(audio_files)}: {file_path.name}")
            
            try:
                result = await self.process_audio_file(str(file_path), category, source='batch_processed')
                
                if result['success']:
                    if result['action'] == 'processed':
                        results['processed'] += 1
                        results['processed_stems'].append({
                            'id': result['stem_id'],
                            'name': result['stem_name'],
                            'category': result['category']
                        })
                    elif result['action'] == 'skipped':
                        results['skipped'] += 1
                else:
                    results['quarantined'] += 1
                    results['errors'].append({
                        'file': str(file_path),
                        'error': result['error']
                    })
                    
            except Exception as e:
                logger.error(f"Unerwarteter Fehler bei {file_path}: {e}")
                results['quarantined'] += 1
                results['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        logger.info(f"Batch-Verarbeitung abgeschlossen: {results['processed']} verarbeitet, "
                   f"{results['skipped']} übersprungen, {results['quarantined']} in Quarantäne")
        
        return results