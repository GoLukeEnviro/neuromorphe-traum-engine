"""RendererService für das Rendern von Audio-Tracks.

Dieser Service ist verantwortlich für das finale Rendering von arrangierten
Audio-Stems zu einem fertigen Track.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from datetime import datetime

from ..database.service import DatabaseService
from ..core.config import settings

logger = logging.getLogger(__name__)

class RendererService:
    """Service für das Rendern von Audio-Tracks aus arrangierten Stems."""
    
    def __init__(self):
        """Initialisiert den RendererService.
        """
        self.db_service = DatabaseService()
        self.sample_rate = settings.SAMPLE_RATE # Annahme: Sample Rate aus Settings
        self.rendered_dir = Path(settings.RENDERED_DIR)
        self.rendered_dir.mkdir(parents=True, exist_ok=True)
        logger.info("RendererService initialisiert")
    
    async def render_track(
        self,
        arrangement: Dict[str, Any],
        output_format: str = 'wav',
        quality: str = 'high'
    ) -> Dict[str, Any]:
        """Rendert einen Track aus einem Arrangement.
        
        Args:
            arrangement: Das Arrangement-Dictionary mit Stem-Informationen
            output_format: Ausgabeformat (wav, mp3, etc.)
            quality: Qualitätsstufe (low, medium, high)
            
        Returns:
            Dictionary mit Render-Ergebnis
        """
        try:
            logger.info(f"Starte Track-Rendering mit Format: {output_format}, Qualität: {quality}")

            # 1. Gesamtdauer des Tracks berechnen
            bpm = arrangement.get('bpm', 120)
            total_bars = arrangement.get('total_bars', 16) # Annahme: Gesamtanzahl Takte
            
            seconds_per_beat = 60 / bpm
            seconds_per_bar = seconds_per_beat * 4 # Annahme: 4/4 Takt
            total_duration_seconds = total_bars * seconds_per_bar
            
            total_frames = int(total_duration_seconds * self.sample_rate)
            
            # 2. Leeres NumPy-Array für den finalen Stereo-Mixdown
            # Initialisiere mit 2 Kanälen für Stereo
            mixed_audio = np.zeros((total_frames, 2), dtype=np.float32)

            # 3. Iteriere durch den Plan, lade Stems und mische sie
            for section in arrangement.get('track_structure', {}).get('sections', []):
                section_start_bar = section.get('start_bar', 0)
                section_stems = section.get('stems', [])

                for stem_info in section_stems:
                    stem_id = stem_info.get('stem_id')
                    start_offset_bars = stem_info.get('start_offset_bars', 0)
                    duration_bars = stem_info.get('duration_bars', 4) # Standard 4 Takte

                    if stem_id is None:
                        logger.warning(f"Stem ID fehlt in Arrangement-Plan für Sektion {section_start_bar}. Überspringe.")
                        continue

                    stem = await self.db_service.get_stem_by_id(stem_id)
                    if not stem or not stem.processed_path:
                        logger.warning(f"Stem {stem_id} oder processed_path nicht gefunden. Überspringe.")
                        continue

                    # Lade Audio-Daten des Stems
                    stem_audio, sr = librosa.load(stem.processed_path, sr=self.sample_rate, mono=False)
                    
                    # Sicherstellen, dass wir Stereo-Audio haben
                    if stem_audio.ndim == 1:
                        stem_audio = np.stack([stem_audio, stem_audio], axis=0)
                    stem_audio = stem_audio.T # Transponieren für (frames, channels)

                    # Berechne Start- und End-Frame für diesen Stem im Mixdown
                    start_frame_in_mix = int((section_start_bar + start_offset_bars) * seconds_per_bar * self.sample_rate)
                    end_frame_in_mix = int(start_frame_in_mix + (duration_bars * seconds_per_bar * self.sample_rate))

                    # Sicherstellen, dass der Stem nicht über das Ende des Mixdowns hinausgeht
                    if end_frame_in_mix > total_frames:
                        end_frame_in_mix = total_frames
                        stem_audio = stem_audio[:(end_frame_in_mix - start_frame_in_mix)]

                    # Mische den Stem in das Haupt-Audio-Array
                    # Überprüfe die Dimensionen vor dem Mischen
                    if stem_audio.shape[0] > (end_frame_in_mix - start_frame_in_mix):
                        stem_audio = stem_audio[:(end_frame_in_mix - start_frame_in_mix)]
                    
                    if start_frame_in_mix < total_frames and (end_frame_in_mix - start_frame_in_mix) > 0:
                        mixed_audio[start_frame_in_mix:end_frame_in_mix] += stem_audio

            # Normalisiere das Audio, um Clipping zu vermeiden
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio /= max_val

            # 4. Speichere den finalen Mixdown als hochwertige WAV-Datei
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"rendered_track_{timestamp}.{output_format}"
            output_path = self.rendered_dir / output_filename

            sf.write(str(output_path), mixed_audio, self.sample_rate)
            
            logger.info(f"Track erfolgreich gerendert: {output_path}")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'duration': total_duration_seconds,
                'format': output_format,
                'quality': quality
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Track-Rendering: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_supported_formats(self) -> list:
        """Gibt die unterstützten Ausgabeformate zurück.
        
        Returns:
            Liste der unterstützten Formate
        """
        return ['wav', 'mp3', 'flac', 'ogg']
    
    def get_quality_levels(self) -> list:
        """Gibt die verfügbaren Qualitätsstufen zurück.
        
        Returns:
            Liste der Qualitätsstufen
        """
        return ['low', 'medium', 'high', 'lossless']