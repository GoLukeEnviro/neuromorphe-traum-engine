"""NeuroAnalyzer - CLAP-basierte semantische Audio-Analyse

Diese Klasse implementiert die neuromorphe Analyse-Pipeline der Traum-Engine v2.0.
Sie nutzt CLAP (Contrastive Language-Audio Pre-training) für semantische Audio-Embeddings.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.core.config import settings

logger = logging.getLogger(__name__)


class CLAPEmbedder:
    """CLAP-Modell für Audio-Text-Embeddings"""
    
    def __init__(self, model_name: str = "laion/larger_clap_music_and_speech"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        # Cache-Verzeichnis
        self.cache_dir = Path(settings.CLAP_MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CLAPEmbedder initialisiert (Device: {self.device})")
    
    def load_model(self) -> None:
        """Lädt das CLAP-Modell"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Lade CLAP-Modell: {self.model_name}")
            
            # Modell und Processor laden
            self.processor = ClapProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            self.model = ClapModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            ).to(self.device)
            
            self.model.eval()
            self.is_loaded = True
            
            logger.info("CLAP-Modell erfolgreich geladen")
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des CLAP-Modells: {e}")
            raise
    
    def get_audio_embedding(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Erstellt Audio-Embedding mit CLAP"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Audio für CLAP vorbereiten
            if sample_rate != 48000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=48000)
            
            # Audio-Input verarbeiten
            inputs = self.processor(
                audios=audio,
                sampling_rate=48000,
                return_tensors="pt"
            ).to(self.device)
            
            # Embedding generieren
            with torch.no_grad():
                audio_embeds = self.model.get_audio_features(**inputs)
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
            
            return audio_embeds.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Fehler bei Audio-Embedding: {e}")
            raise
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Erstellt Text-Embedding mit CLAP"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Text-Input verarbeiten
            inputs = self.processor(
                text=[text],
                return_tensors="pt"
            ).to(self.device)
            
            # Embedding generieren
            with torch.no_grad():
                text_embeds = self.model.get_text_features(**inputs)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            return text_embeds.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Fehler bei Text-Embedding: {e}")
            raise
    
    def calculate_similarity(self, audio_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """Berechnet Cosinus-Ähnlichkeit zwischen Audio und Text"""
        similarity = cosine_similarity(
            audio_embedding.reshape(1, -1),
            text_embedding.reshape(1, -1)
        )[0, 0]
        return float(similarity)


class SemanticAnalyzer:
    """Semantische Analyse von Audio-Inhalten"""
    
    def __init__(self):
        self.embedder = CLAPEmbedder()
        
        # Vordefinierte semantische Kategorien
        self.semantic_categories = {
            'energy': [
                'energetic powerful driving force',
                'calm peaceful relaxed gentle',
                'aggressive intense harsh brutal',
                'soft smooth mellow warm'
            ],
            'mood': [
                'dark mysterious ominous sinister',
                'bright happy uplifting positive',
                'melancholic sad emotional deep',
                'euphoric ecstatic joyful celebratory'
            ],
            'texture': [
                'rough distorted gritty industrial',
                'smooth clean polished pristine',
                'organic natural acoustic human',
                'synthetic digital electronic artificial'
            ],
            'movement': [
                'flowing liquid smooth continuous',
                'choppy staccato fragmented broken',
                'pulsing rhythmic steady regular',
                'chaotic random unpredictable wild'
            ],
            'space': [
                'wide spacious reverberant atmospheric',
                'tight close intimate dry',
                'deep cavernous echoing vast',
                'shallow surface immediate present'
            ]
        }
        
        # Cache für Text-Embeddings
        self.text_embeddings_cache = {}
        
        logger.info("SemanticAnalyzer initialisiert")
    
    def analyze_semantic_content(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Führt semantische Analyse des Audio-Inhalts durch"""
        logger.debug("Starte semantische Analyse")
        
        try:
            # Audio-Embedding erstellen
            audio_embedding = self.embedder.get_audio_embedding(audio, sample_rate)
            
            # Semantische Kategorien analysieren
            semantic_scores = {}
            
            for category, descriptions in self.semantic_categories.items():
                category_scores = []
                
                for description in descriptions:
                    # Text-Embedding aus Cache oder neu erstellen
                    if description not in self.text_embeddings_cache:
                        self.text_embeddings_cache[description] = self.embedder.get_text_embedding(description)
                    
                    text_embedding = self.text_embeddings_cache[description]
                    similarity = self.embedder.calculate_similarity(audio_embedding, text_embedding)
                    category_scores.append(similarity)
                
                semantic_scores[category] = {
                    'scores': category_scores,
                    'max_score': max(category_scores),
                    'mean_score': np.mean(category_scores),
                    'dominant_description': descriptions[np.argmax(category_scores)]
                }
            
            # Gesamtbewertung
            overall_semantic_profile = self._create_semantic_profile(semantic_scores)
            
            return {
                'audio_embedding': audio_embedding.tolist(),
                'semantic_scores': semantic_scores,
                'semantic_profile': overall_semantic_profile,
                'embedding_dimension': len(audio_embedding)
            }
            
        except Exception as e:
            logger.error(f"Fehler bei semantischer Analyse: {e}")
            raise
    
    def _create_semantic_profile(self, semantic_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt ein zusammenfassendes semantisches Profil"""
        profile = {
            'dominant_characteristics': {},
            'semantic_vector': [],
            'confidence_scores': {}
        }
        
        # Dominante Eigenschaften pro Kategorie
        for category, scores in semantic_scores.items():
            profile['dominant_characteristics'][category] = scores['dominant_description']
            profile['semantic_vector'].append(scores['max_score'])
            profile['confidence_scores'][category] = scores['max_score']
        
        # Gesamtvertrauen
        profile['overall_confidence'] = np.mean(list(profile['confidence_scores'].values()))
        
        return profile


class PatternAnalyzer:
    """Analyse von Audio-Mustern und -Strukturen"""
    
    def __init__(self):
        self.window_size = 2048
        self.hop_length = 512
        
    def analyze_patterns(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analysiert Muster und Strukturen im Audio"""
        logger.debug("Starte Pattern-Analyse")
        
        try:
            # Spektrogramm berechnen
            stft = librosa.stft(audio, n_fft=self.window_size, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Verschiedene Pattern-Analysen
            patterns = {
                'repetition': self._analyze_repetition_patterns(magnitude),
                'texture': self._analyze_texture_patterns(magnitude),
                'dynamics': self._analyze_dynamic_patterns(audio, sample_rate),
                'spectral_evolution': self._analyze_spectral_evolution(magnitude),
                'rhythmic_patterns': self._analyze_rhythmic_patterns(audio, sample_rate)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Fehler bei Pattern-Analyse: {e}")
            raise
    
    def _analyze_repetition_patterns(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Analysiert Wiederholungsmuster"""
        # Selbstähnlichkeitsmatrix
        similarity_matrix = cosine_similarity(magnitude.T)
        
        # Diagonale Strukturen finden (Wiederholungen)
        diagonal_strength = np.mean([np.mean(np.diag(similarity_matrix, k)) for k in range(1, min(50, similarity_matrix.shape[0]))])
        
        # Periodizität
        autocorr = np.correlate(np.mean(magnitude, axis=0), np.mean(magnitude, axis=0), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Finde Peaks für Periodizität
        peaks = []
        for i in range(1, min(len(autocorr) // 2, 100)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append((i, autocorr[i]))
        
        return {
            'repetition_strength': float(diagonal_strength),
            'periodicity_peaks': peaks[:5],  # Top 5 Peaks
            'overall_periodicity': float(np.std(autocorr[:50])),
            'self_similarity': float(np.mean(similarity_matrix))
        }
    
    def _analyze_texture_patterns(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Analysiert Textur-Eigenschaften"""
        # Spektrale Textur-Features
        spectral_contrast = np.mean(np.std(magnitude, axis=1))
        spectral_smoothness = 1.0 / (1.0 + spectral_contrast)
        
        # Zeitliche Textur
        temporal_contrast = np.mean(np.std(magnitude, axis=0))
        temporal_smoothness = 1.0 / (1.0 + temporal_contrast)
        
        # Granularität
        granularity = np.mean(np.abs(np.diff(magnitude, axis=1)))
        
        return {
            'spectral_contrast': float(spectral_contrast),
            'spectral_smoothness': float(spectral_smoothness),
            'temporal_contrast': float(temporal_contrast),
            'temporal_smoothness': float(temporal_smoothness),
            'granularity': float(granularity)
        }
    
    def _analyze_dynamic_patterns(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analysiert dynamische Muster"""
        # RMS-Energie über Zeit
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Dynamik-Features
        dynamic_range = float(np.max(rms) - np.min(rms))
        dynamic_variance = float(np.var(rms))
        
        # Envelope-Analyse
        envelope = np.abs(audio)
        envelope_smooth = librosa.util.smooth(envelope, length=sample_rate // 10)
        
        # Attack/Decay-Charakteristiken
        diff_envelope = np.diff(envelope_smooth)
        attack_strength = float(np.mean(diff_envelope[diff_envelope > 0]))
        decay_strength = float(np.abs(np.mean(diff_envelope[diff_envelope < 0])))
        
        return {
            'dynamic_range': dynamic_range,
            'dynamic_variance': dynamic_variance,
            'attack_strength': attack_strength,
            'decay_strength': decay_strength,
            'envelope_complexity': float(np.std(diff_envelope))
        }
    
    def _analyze_spectral_evolution(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Analysiert spektrale Entwicklung über Zeit"""
        # Spektraler Schwerpunkt über Zeit
        freqs = np.arange(magnitude.shape[0])
        spectral_centroids = []
        
        for frame in magnitude.T:
            if np.sum(frame) > 0:
                centroid = np.sum(freqs * frame) / np.sum(frame)
                spectral_centroids.append(centroid)
        
        spectral_centroids = np.array(spectral_centroids)
        
        # Spektrale Bewegung
        spectral_movement = float(np.std(spectral_centroids))
        spectral_trend = float(np.polyfit(range(len(spectral_centroids)), spectral_centroids, 1)[0])
        
        return {
            'spectral_movement': spectral_movement,
            'spectral_trend': spectral_trend,
            'spectral_stability': float(1.0 / (1.0 + spectral_movement))
        }
    
    def _analyze_rhythmic_patterns(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analysiert rhythmische Muster"""
        # Onset-Detection
        onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        
        # Rhythmische Regelmäßigkeit
        autocorr_rhythm = np.correlate(onset_envelope, onset_envelope, mode='full')
        autocorr_rhythm = autocorr_rhythm[autocorr_rhythm.size // 2:]
        
        # Rhythmische Komplexität
        rhythm_complexity = float(np.std(onset_envelope))
        
        return {
            'rhythm_complexity': rhythm_complexity,
            'rhythm_regularity': float(np.max(autocorr_rhythm[1:20]) / autocorr_rhythm[0]),
            'onset_density': float(len(librosa.onset.onset_detect(y=audio, sr=sample_rate)) / (len(audio) / sample_rate))
        }


class NeuroAnalyzer:
    """Hauptklasse für neuromorphe Audio-Analyse"""
    
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        
        # Cache für Analysen
        self.analysis_cache = {}
        
        logger.info("NeuroAnalyzer initialisiert")
    
    async def analyze_audio(self, file_path: str) -> Dict[str, Any]:
        """Führt vollständige neuromorphe Analyse durch"""
        logger.info(f"Starte neuromorphe Analyse: {file_path}")
        
        try:
            # Audio laden
            audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
            
            # Verschiedene Analysen durchführen
            analysis_result = {
                'file_path': file_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'semantic_analysis': self.semantic_analyzer.analyze_semantic_content(audio, sample_rate),
                'pattern_analysis': self.pattern_analyzer.analyze_patterns(audio, sample_rate),
                'neural_features': await self._extract_neural_features(audio, sample_rate),
                'perceptual_mapping': self._create_perceptual_mapping(audio, sample_rate)
            }
            
            # Gesamtbewertung erstellen
            analysis_result['overall_assessment'] = self._create_overall_assessment(analysis_result)
            
            logger.info(f"Neuromorphe Analyse abgeschlossen: {file_path}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Fehler bei neuromorpher Analyse von {file_path}: {e}")
            raise
    
    async def _extract_neural_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extrahiert neurale Features"""
        # Simulierte neurale Features (in Produktion würde man echte neurale Netze verwenden)
        
        # Spektrale Features
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # "Neurale" Repräsentation durch PCA
        pca = PCA(n_components=min(50, magnitude.shape[0]))
        neural_representation = pca.fit_transform(magnitude.T)
        
        # Clustering für Struktur-Erkennung
        n_clusters = min(8, neural_representation.shape[0] // 10)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(neural_representation)
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_labels = np.zeros(neural_representation.shape[0])
            cluster_centers = np.array([np.mean(neural_representation, axis=0)])
        
        return {
            'neural_representation_shape': neural_representation.shape,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'n_clusters': n_clusters,
            'cluster_distribution': np.bincount(cluster_labels).tolist(),
            'neural_complexity': float(np.std(neural_representation)),
            'feature_importance': pca.components_[0].tolist()[:20]  # Top 20 Features
        }
    
    def _create_perceptual_mapping(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Erstellt perzeptuelle Zuordnung"""
        # Psychoakustische Features
        
        # Loudness (vereinfacht)
        rms = np.sqrt(np.mean(audio**2))
        loudness_sones = 2**((20 * np.log10(rms + 1e-8) + 40) / 10)  # Grobe Sones-Schätzung
        
        # Sharpness (vereinfacht)
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sample_rate)
        
        # Gewichtung für Sharpness (höhere Frequenzen stärker gewichtet)
        freq_weights = freqs / np.max(freqs)
        sharpness = np.sum(magnitude * freq_weights[:, np.newaxis]) / np.sum(magnitude)
        
        # Roughness (vereinfacht durch Modulationsanalyse)
        envelope = np.abs(audio)
        modulation_spectrum = np.abs(np.fft.fft(envelope[:sample_rate]))  # 1 Sekunde
        roughness_freq_range = (15, 300)  # Hz
        roughness_bins = np.where((np.fft.fftfreq(len(modulation_spectrum), 1/sample_rate) >= roughness_freq_range[0]) & 
                                 (np.fft.fftfreq(len(modulation_spectrum), 1/sample_rate) <= roughness_freq_range[1]))[0]
        roughness = np.sum(modulation_spectrum[roughness_bins])
        
        return {
            'loudness_sones': float(loudness_sones),
            'sharpness': float(sharpness),
            'roughness': float(roughness),
            'perceptual_brightness': float(np.mean(magnitude[len(magnitude)//2:])),
            'perceptual_warmth': float(np.mean(magnitude[:len(magnitude)//4]))
        }
    
    def _create_overall_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Gesamtbewertung der Analyse"""
        semantic = analysis['semantic_analysis']['semantic_profile']
        patterns = analysis['pattern_analysis']
        neural = analysis['neural_features']
        perceptual = analysis['perceptual_mapping']
        
        # Komplexitätsbewertung
        complexity_factors = [
            patterns['texture']['granularity'],
            patterns['repetition']['overall_periodicity'],
            neural['neural_complexity'],
            patterns['rhythmic_patterns']['rhythm_complexity']
        ]
        overall_complexity = np.mean(complexity_factors)
        
        # Charakteristik-Zusammenfassung
        characteristics = {
            'complexity_level': 'high' if overall_complexity > 0.7 else 'medium' if overall_complexity > 0.3 else 'low',
            'dominant_energy': semantic['dominant_characteristics'].get('energy', 'unknown'),
            'dominant_mood': semantic['dominant_characteristics'].get('mood', 'unknown'),
            'texture_type': semantic['dominant_characteristics'].get('texture', 'unknown'),
            'movement_style': semantic['dominant_characteristics'].get('movement', 'unknown')
        }
        
        return {
            'overall_complexity': float(overall_complexity),
            'confidence_score': semantic['overall_confidence'],
            'characteristics': characteristics,
            'recommended_usage': self._recommend_usage(characteristics, patterns),
            'quality_assessment': self._assess_quality(analysis)
        }
    
    def _recommend_usage(self, characteristics: Dict[str, Any], patterns: Dict[str, Any]) -> List[str]:
        """Empfiehlt Verwendungszwecke basierend auf Analyse"""
        recommendations = []
        
        # Basierend auf Komplexität
        if characteristics['complexity_level'] == 'high':
            recommendations.append('lead_element')
            recommendations.append('feature_sound')
        elif characteristics['complexity_level'] == 'low':
            recommendations.append('background_element')
            recommendations.append('foundation_layer')
        
        # Basierend auf Energie
        if 'energetic' in characteristics['dominant_energy']:
            recommendations.append('drop_section')
            recommendations.append('buildup')
        elif 'calm' in characteristics['dominant_energy']:
            recommendations.append('breakdown')
            recommendations.append('intro_outro')
        
        # Basierend auf Rhythmus
        if patterns['rhythmic_patterns']['rhythm_regularity'] > 0.8:
            recommendations.append('rhythmic_foundation')
        
        return list(set(recommendations))
    
    def _assess_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Bewertet Audio-Qualität"""
        neural = analysis['neural_features']
        patterns = analysis['pattern_analysis']
        
        # Qualitätsindikatoren
        quality_score = 0.0
        quality_factors = []
        
        # Spektrale Qualität
        spectral_quality = 1.0 - patterns['texture']['granularity']
        quality_factors.append(spectral_quality)
        
        # Dynamische Qualität
        dynamic_quality = min(1.0, patterns['dynamics']['dynamic_range'])
        quality_factors.append(dynamic_quality)
        
        # Neurale Kohärenz
        neural_quality = 1.0 / (1.0 + neural['neural_complexity'])
        quality_factors.append(neural_quality)
        
        quality_score = np.mean(quality_factors)
        
        return {
            'overall_quality': float(quality_score),
            'quality_factors': {
                'spectral_quality': float(spectral_quality),
                'dynamic_quality': float(dynamic_quality),
                'neural_quality': float(neural_quality)
            },
            'quality_rating': 'excellent' if quality_score > 0.8 else 'good' if quality_score > 0.6 else 'fair' if quality_score > 0.4 else 'poor'
        }
    
    async def compare_audio_similarity(self, file_path1: str, file_path2: str) -> Dict[str, Any]:
        """Vergleicht zwei Audio-Dateien auf Ähnlichkeit"""
        logger.info(f"Vergleiche Audio-Ähnlichkeit: {file_path1} vs {file_path2}")
        
        try:
            # Beide Dateien analysieren
            analysis1 = await self.analyze_audio(file_path1)
            analysis2 = await self.analyze_audio(file_path2)
            
            # Embeddings vergleichen
            embedding1 = np.array(analysis1['semantic_analysis']['audio_embedding'])
            embedding2 = np.array(analysis2['semantic_analysis']['audio_embedding'])
            
            semantic_similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0, 0]
            
            # Pattern-Ähnlichkeit
            pattern_similarity = self._compare_patterns(
                analysis1['pattern_analysis'],
                analysis2['pattern_analysis']
            )
            
            # Gesamtähnlichkeit
            overall_similarity = (semantic_similarity + pattern_similarity) / 2
            
            return {
                'semantic_similarity': float(semantic_similarity),
                'pattern_similarity': float(pattern_similarity),
                'overall_similarity': float(overall_similarity),
                'similarity_rating': 'very_high' if overall_similarity > 0.9 else 'high' if overall_similarity > 0.7 else 'medium' if overall_similarity > 0.5 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Ähnlichkeitsvergleich: {e}")
            raise
    
    def _compare_patterns(self, patterns1: Dict[str, Any], patterns2: Dict[str, Any]) -> float:
        """Vergleicht Pattern-Analysen zweier Audio-Dateien"""
        similarities = []
        
        # Textur-Ähnlichkeit
        texture1 = patterns1['texture']
        texture2 = patterns2['texture']
        
        texture_sim = 1.0 - abs(texture1['granularity'] - texture2['granularity'])
        similarities.append(texture_sim)
        
        # Rhythmus-Ähnlichkeit
        rhythm1 = patterns1['rhythmic_patterns']
        rhythm2 = patterns2['rhythmic_patterns']
        
        rhythm_sim = 1.0 - abs(rhythm1['rhythm_complexity'] - rhythm2['rhythm_complexity'])
        similarities.append(rhythm_sim)
        
        # Dynamik-Ähnlichkeit
        dynamics1 = patterns1['dynamics']
        dynamics2 = patterns2['dynamics']
        
        dynamic_sim = 1.0 - abs(dynamics1['dynamic_range'] - dynamics2['dynamic_range'])
        similarities.append(dynamic_sim)
        
        return np.mean(similarities)