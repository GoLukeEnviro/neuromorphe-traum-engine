"""Renderer Service - Audio-Montage und Track-Generierung

Dieser Service implementiert die Audio-Verarbeitungslogik der Neuromorphe Traum-Engine v2.0.
Er nimmt Arrangement-Pläne entgegen und rendert daraus finale Audio-Tracks.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import uuid

import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from sqlalchemy.orm import Session

from ..database.crud import StemCRUD
from ..db.database import get_db
from ..database.models import GeneratedTrack
from .arranger import ArrangementPlan, ArrangementSection, StemQuery
from ..core.config import settings

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Klasse für grundlegende Audio-Verarbeitung"""
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
        """Lädt eine Audio-Datei und konvertiert sie zur Ziel-Samplerate"""
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=0)  # Mono zu Stereo
            elif audio.ndim == 2 and audio.shape[0] > 2:
                audio = audio[:2]  # Nur erste 2 Kanäle bei Multi-Channel
            return audio, sr
        except Exception as e:
            logger.error(f"Fehler beim Laden von {file_path}: {e}")
            raise
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sr: int = 44100) -> None:
        """Speichert Audio-Daten als Datei"""
        try:
            # Sicherstellen dass Audio im richtigen Format ist
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=0)
            
            # Transponieren für soundfile (channels, samples) -> (samples, channels)
            audio_transposed = audio.T
            
            sf.write(file_path, audio_transposed, sr, format='WAV', subtype='PCM_24')
            logger.info(f"Audio gespeichert: {file_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern von {file_path}: {e}")
            raise
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
        """Normalisiert Audio auf Ziel-LUFS (vereinfacht)"""
        # Vereinfachte Normalisierung - in Produktion würde man pyloudnorm verwenden
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_lufs / 20)
            scaling_factor = target_rms / rms
            return audio * scaling_factor
        return audio
    
    @staticmethod
    def apply_fade(audio: np.ndarray, fade_in_samples: int = 0, fade_out_samples: int = 0) -> np.ndarray:
        """Wendet Fade-In und Fade-Out auf Audio an"""
        result = audio.copy()
        
        if fade_in_samples > 0:
            fade_curve = np.linspace(0, 1, fade_in_samples)
            if audio.ndim == 2:
                fade_curve = fade_curve[np.newaxis, :]
            result[:, :fade_in_samples] *= fade_curve
        
        if fade_out_samples > 0:
            fade_curve = np.linspace(1, 0, fade_out_samples)
            if audio.ndim == 2:
                fade_curve = fade_curve[np.newaxis, :]
            result[:, -fade_out_samples:] *= fade_curve
        
        return result
    
    @staticmethod
    def time_stretch(audio: np.ndarray, sr: int, stretch_factor: float) -> np.ndarray:
        """Zeitdehnung ohne Tonhöhenänderung"""
        try:
            if audio.ndim == 2:
                # Stereo: jeden Kanal einzeln bearbeiten
                stretched_channels = []
                for channel in audio:
                    stretched = librosa.effects.time_stretch(channel, rate=stretch_factor)
                    stretched_channels.append(stretched)
                return np.array(stretched_channels)
            else:
                return librosa.effects.time_stretch(audio, rate=stretch_factor)
        except Exception as e:
            logger.warning(f"Time-Stretch fehlgeschlagen: {e}")
            return audio
    
    @staticmethod
    def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Tonhöhenverschiebung"""
        try:
            if audio.ndim == 2:
                shifted_channels = []
                for channel in audio:
                    shifted = librosa.effects.pitch_shift(channel, sr=sr, n_steps=semitones)
                    shifted_channels.append(shifted)
                return np.array(shifted_channels)
            else:
                return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        except Exception as e:
            logger.warning(f"Pitch-Shift fehlgeschlagen: {e}")
            return audio


class StemMixer:
    """Klasse für das Mischen von Audio-Stems"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.processor = AudioProcessor()
    
    def mix_stems(self, stem_data: List[Dict[str, Any]], section_bars: int, bpm: int) -> np.ndarray:
        """Mischt mehrere Stems zu einem Audio-Signal"""
        # Berechne Ziel-Länge in Samples
        beats_per_bar = 4
        total_beats = section_bars * beats_per_bar
        beat_duration = 60.0 / bpm  # Sekunden pro Beat
        section_duration = total_beats * beat_duration
        target_samples = int(section_duration * self.sample_rate)
        
        # Initialisiere Mix-Buffer (Stereo)
        mix_buffer = np.zeros((2, target_samples))
        
        for stem_info in stem_data:
            try:
                # Lade Stem-Audio
                audio, _ = self.processor.load_audio(stem_info['file_path'], self.sample_rate)
                
                # Anpassungen anwenden
                audio = self._apply_stem_processing(audio, stem_info)
                
                # Auf Ziel-Länge anpassen
                audio = self._fit_to_length(audio, target_samples, bpm)
                
                # Zum Mix hinzufügen
                volume = stem_info.get('volume', 1.0)
                mix_buffer += audio * volume
                
                logger.debug(f"Stem hinzugefügt: {stem_info['name']} (Volume: {volume})")
                
            except Exception as e:
                logger.error(f"Fehler beim Mischen von Stem {stem_info.get('name', 'unknown')}: {e}")
                continue
        
        return mix_buffer
    
    def _apply_stem_processing(self, audio: np.ndarray, stem_info: Dict[str, Any]) -> np.ndarray:
        """Wendet Verarbeitung auf einzelnen Stem an"""
        result = audio.copy()
        
        # Pitch-Shift
        pitch_shift = stem_info.get('pitch_shift', 0)
        if pitch_shift != 0:
            result = self.processor.pitch_shift(result, self.sample_rate, pitch_shift)
        
        # Time-Stretch
        time_stretch = stem_info.get('time_stretch', 1.0)
        if time_stretch != 1.0:
            result = self.processor.time_stretch(result, self.sample_rate, time_stretch)
        
        # Fade-Effekte
        fade_in = stem_info.get('fade_in', 0)
        fade_out = stem_info.get('fade_out', 0)
        if fade_in > 0 or fade_out > 0:
            fade_in_samples = int(fade_in * self.sample_rate)
            fade_out_samples = int(fade_out * self.sample_rate)
            result = self.processor.apply_fade(result, fade_in_samples, fade_out_samples)
        
        return result
    
    def _fit_to_length(self, audio: np.ndarray, target_samples: int, bpm: int) -> np.ndarray:
        """Passt Audio-Länge an Ziel-Länge an"""
        current_samples = audio.shape[1]
        
        if current_samples == target_samples:
            return audio
        elif current_samples < target_samples:
            # Loop das Audio
            return self._loop_audio(audio, target_samples, bpm)
        else:
            # Schneide Audio ab
            return audio[:, :target_samples]
    
    def _loop_audio(self, audio: np.ndarray, target_samples: int, bpm: int) -> np.ndarray:
        """Loopt Audio intelligent basierend auf BPM"""
        current_samples = audio.shape[1]
        
        # Berechne Beat-Länge in Samples
        beat_samples = int((60.0 / bpm) * self.sample_rate)
        bar_samples = beat_samples * 4  # 4/4 Takt
        
        # Finde beste Loop-Länge (Vielfaches von Takten)
        if current_samples >= bar_samples:
            # Schneide auf vollständige Takte
            full_bars = current_samples // bar_samples
            loop_length = full_bars * bar_samples
            audio_trimmed = audio[:, :loop_length]
        else:
            audio_trimmed = audio
            loop_length = current_samples
        
        # Wiederhole bis Ziel-Länge erreicht
        loops_needed = (target_samples + loop_length - 1) // loop_length
        looped_audio = np.tile(audio_trimmed, loops_needed)
        
        # Schneide auf exakte Ziel-Länge
        return looped_audio[:, :target_samples]


class RendererService:
    """Hauptservice für Track-Rendering"""
    
    def __init__(self):
        self.sample_rate = 44100
        self.mixer = StemMixer(self.sample_rate)
        self.processor = AudioProcessor()
        
        # Ausgabe-Verzeichnis sicherstellen
        self.output_dir = Path(settings.GENERATED_TRACKS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("RendererService initialisiert")
    
    async def render_track(self, arrangement_plan: ArrangementPlan, track_name: str = None) -> Dict[str, Any]:
        """Rendert einen kompletten Track aus einem Arrangement-Plan"""
        if not track_name:
            track_name = f"track_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starte Track-Rendering: {track_name}")
        
        try:
            # Sammle alle benötigten Stems
            stem_assignments = await self._collect_stems_for_arrangement(arrangement_plan)
            
            # Rendere jede Sektion
            section_audio_data = []
            for i, section in enumerate(arrangement_plan.structure):
                logger.info(f"Rendere Sektion {i+1}/{len(arrangement_plan.structure)}: {section.section}")
                
                section_stems = stem_assignments.get(section.section, [])
                section_audio = self.mixer.mix_stems(
                    section_stems, 
                    section.bars, 
                    arrangement_plan.bpm
                )
                
                section_audio_data.append(section_audio)
            
            # Kombiniere alle Sektionen
            full_track = np.concatenate(section_audio_data, axis=1)
            
            # Master-Processing
            full_track = self._apply_master_processing(full_track, arrangement_plan)
            
            # Speichere Track
            output_path = self.output_dir / f"{track_name}.wav"
            self.processor.save_audio(full_track, str(output_path), self.sample_rate)
            
            # Erstelle Metadaten
            metadata = self._create_track_metadata(
                arrangement_plan, 
                track_name, 
                str(output_path),
                stem_assignments
            )
            
            # Speichere Metadaten
            metadata_path = self.output_dir / f"{track_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Track erfolgreich gerendert: {output_path}")
            
            return {
                'success': True,
                'track_name': track_name,
                'file_path': str(output_path),
                'metadata_path': str(metadata_path),
                'duration': full_track.shape[1] / self.sample_rate,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Track-Rendering: {e}")
            return {
                'success': False,
                'error': str(e),
                'track_name': track_name
            }
    
    async def _collect_stems_for_arrangement(self, db: Session, plan: ArrangementPlan) -> Dict[str, List[Dict[str, Any]]]:
        """Sammelt alle benötigten Stems für das Arrangement"""
        stem_assignments = {}
        
        for section in plan.structure:
            section_stems = []
            
            for query in section.stem_queries:
                try:
                    # Suche passende Stems
                    stems = await StemCRUD.search_stems_by_tags_and_category(
                        db=db,
                        category=query.category,
                        tags=query.tags,
                        limit=query.count
                    )
                    
                    for stem in stems:
                        stem_info = {
                            'id': stem.id,
                            'name': stem.name,
                            'file_path': stem.file_path,
                            'category': stem.category,
                            'tags': stem.tags,
                            'volume': section.volume,
                            'bpm': stem.bpm,
                            'key': stem.key,
                            # Verarbeitungsparameter
                            'pitch_shift': 0,
                            'time_stretch': 1.0,
                            'fade_in': 0,
                            'fade_out': 0
                        }
                        
                        # BPM-Anpassung berechnen
                        if stem.bpm and stem.bpm != plan.bpm:
                            stem_info['time_stretch'] = plan.bpm / stem.bpm
                        
                        section_stems.append(stem_info)
                        
                except Exception as e:
                    logger.error(f"Fehler beim Sammeln von Stems für {query.category}: {e}")
                    if query.required:
                        raise
            
            stem_assignments[section.section] = section_stems
        
        # db.close() wird hier nicht benötigt, da die Session vom Caller verwaltet wird
        return stem_assignments
    
    def _apply_master_processing(self, audio: np.ndarray, plan: ArrangementPlan) -> np.ndarray:
        """Wendet Master-Processing auf den finalen Track an"""
        result = audio.copy()
        
        # Normalisierung
        result = self.processor.normalize_audio(result, target_lufs=-14.0)
        
        # Soft-Clipping bei Übersteuerung
        max_val = np.max(np.abs(result))
        if max_val > 0.95:
            result = np.tanh(result * 0.95) * 0.95
            logger.info("Soft-Clipping angewendet")
        
        # Fade-In/Out für gesamten Track
        fade_samples = int(0.1 * self.sample_rate)  # 100ms
        result = self.processor.apply_fade(result, fade_samples, fade_samples)
        
        return result
    
    def _create_track_metadata(self, plan: ArrangementPlan, track_name: str, 
                              file_path: str, stem_assignments: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Erstellt Metadaten für den generierten Track"""
        return {
            'track_name': track_name,
            'file_path': file_path,
            'created_at': datetime.now().isoformat(),
            'arrangement_plan': {
                'bpm': plan.bpm,
                'key': plan.key,
                'genre': plan.genre,
                'mood': plan.mood,
                'total_bars': plan.total_bars,
                'estimated_duration': plan.estimated_duration
            },
            'structure': [
                {
                    'section': section.section,
                    'bars': section.bars,
                    'stems_used': len(stem_assignments.get(section.section, []))
                }
                for section in plan.structure
            ],
            'stems_used': {
                section_name: [
                    {
                        'id': stem['id'],
                        'name': stem['name'],
                        'category': stem['category'],
                        'processing': {
                            'volume': stem['volume'],
                            'pitch_shift': stem['pitch_shift'],
                            'time_stretch': stem['time_stretch']
                        }
                    }
                    for stem in stems
                ]
                for section_name, stems in stem_assignments.items()
            },
            'technical_info': {
                'sample_rate': self.sample_rate,
                'bit_depth': 24,
                'channels': 2,
                'format': 'WAV'
            }
        }
    
    async def render_preview(self, arrangement_plan: ArrangementPlan, max_duration: float = 30.0) -> Dict[str, Any]:
        """Rendert eine kurze Vorschau des Tracks"""
        logger.info(f"Erstelle Track-Vorschau (max. {max_duration}s)")
        
        # Wähle repräsentative Sektionen für Vorschau
        preview_sections = self._select_preview_sections(arrangement_plan, max_duration)
        
        # Erstelle temporären Arrangement-Plan für Vorschau
        preview_plan = ArrangementPlan(
            bpm=arrangement_plan.bpm,
            key=arrangement_plan.key,
            genre=arrangement_plan.genre,
            mood=arrangement_plan.mood,
            structure=preview_sections,
            total_bars=sum(s.bars for s in preview_sections),
            estimated_duration=max_duration
        )
        
        # Rendere Vorschau
        track_name = f"preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return await self.render_track(preview_plan, track_name)
    
    def _select_preview_sections(self, plan: ArrangementPlan, max_duration: float) -> List[ArrangementSection]:
        """Wählt Sektionen für eine Vorschau aus"""
        target_bars = int((max_duration * plan.bpm) / (60 * 4))  # Grobe Schätzung
        
        # Priorisiere wichtige Sektionen
        section_priority = {
            'Groove': 3,
            'Drop': 3,
            'Buildup': 2,
            'Breakdown': 2,
            'Intro': 1,
            'Outro': 1
        }
        
        # Sortiere Sektionen nach Priorität
        sorted_sections = sorted(
            plan.structure,
            key=lambda s: section_priority.get(s.section, 0),
            reverse=True
        )
        
        # Wähle Sektionen bis Ziel-Länge erreicht
        selected_sections = []
        current_bars = 0
        
        for section in sorted_sections:
            if current_bars + section.bars <= target_bars:
                selected_sections.append(section)
                current_bars += section.bars
            elif current_bars < target_bars:
                # Verkürze letzte Sektion
                remaining_bars = target_bars - current_bars
                shortened_section = ArrangementSection(
                    section=section.section,
                    bars=remaining_bars,
                    stem_queries=section.stem_queries,
                    volume=section.volume,
                    effects=section.effects
                )
                selected_sections.append(shortened_section)
                break
        
        return selected_sections