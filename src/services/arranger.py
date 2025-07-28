"""Arranger Service - Intelligenter Track-Strukturierer

Dieser Service implementiert die kreative Kernlogik der Neuromorphe Traum-Engine v2.0.
Er analysiert Text-Prompts und generiert daraus strukturierte Track-Arrangements.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..database.service import DatabaseService
from ..database.models import Stem

logger = logging.getLogger(__name__)


class TrackSection(Enum):
    """Verfügbare Track-Sektionen"""
    INTRO = "Intro"
    BUILDUP = "Buildup"
    GROOVE = "Groove"
    BREAKDOWN = "Breakdown"
    DROP = "Drop"
    BRIDGE = "Bridge"
    OUTRO = "Outro"


class MusicGenre(Enum):
    """Unterstützte Musik-Genres"""
    TECHNO = "techno"
    HOUSE = "house"
    TRANCE = "trance"
    INDUSTRIAL = "industrial"
    AMBIENT = "ambient"
    DRUM_AND_BASS = "drum_and_bass"


@dataclass
class StemQuery:
    """Definiert eine Suchanfrage für Stems"""
    category: str
    tags: List[str]
    count: int = 1
    required: bool = True


@dataclass
class ArrangementSection:
    """Definiert eine Sektion im Track-Arrangement"""
    section: str
    bars: int
    stem_queries: List[StemQuery]
    volume: float = 1.0
    effects: Optional[List[str]] = None


@dataclass
class ArrangementPlan:
    """Vollständiger Track-Arrangement-Plan"""
    bpm: int
    key: str
    genre: str
    mood: List[str]
    structure: List[ArrangementSection]
    total_bars: int
    estimated_duration: float


class PromptParser:
    """Parser für Text-Prompts zur Extraktion von Musik-Parametern"""
    
    # Regex-Patterns für verschiedene Parameter
    BPM_PATTERN = r'(\d{2,3})\s*bpm'
    GENRE_PATTERNS = {
        MusicGenre.TECHNO: r'\b(techno|tech)\b',
        MusicGenre.HOUSE: r'\b(house|deep house)\b',
        MusicGenre.TRANCE: r'\b(trance|uplifting)\b',
        MusicGenre.INDUSTRIAL: r'\b(industrial|harsh|mechanical)\b',
        MusicGenre.AMBIENT: r'\b(ambient|atmospheric|pad)\b',
        MusicGenre.DRUM_AND_BASS: r'\b(drum.?and.?bass|dnb|jungle)\b'
    }
    
    MOOD_PATTERNS = {
        'dark': r'\b(dark|düster|schwarz|noir)\b',
        'aggressive': r'\b(aggressive|aggressiv|hard|hart)\b',
        'driving': r'\b(driving|treibend|energetic|energisch)\b',
        'melodic': r'\b(melodic|melodisch|harmonic|harmonisch)\b',
        'minimal': r'\b(minimal|minimalistic|reduced)\b',
        'atmospheric': r'\b(atmospheric|atmosphärisch|ambient)\b',
        'punchy': r'\b(punchy|knackig|crisp|sharp)\b',
        'groovy': r'\b(groovy|groove|rhythmic|rhythmisch)\b'
    }
    
    KEY_PATTERNS = {
        'Am': r'\b(a.?minor|am|a.?moll)\b',
        'Dm': r'\b(d.?minor|dm|d.?moll)\b',
        'Em': r'\b(e.?minor|em|e.?moll)\b',
        'Gm': r'\b(g.?minor|gm|g.?moll)\b',
        'C': r'\b(c.?major|c.?dur)\b',
        'F': r'\b(f.?major|f.?dur)\b',
        'G': r'\b(g.?major|g.?dur)\b'
    }
    
    @classmethod
    def parse_prompt(cls, prompt: str) -> Dict[str, Any]:
        """Extrahiert Parameter aus einem Text-Prompt"""
        prompt_lower = prompt.lower()
        
        # BPM extrahieren
        bpm_match = re.search(cls.BPM_PATTERN, prompt_lower)
        bpm = int(bpm_match.group(1)) if bpm_match else 128
        
        # Genre bestimmen
        genre = MusicGenre.TECHNO  # Default
        for genre_enum, pattern in cls.GENRE_PATTERNS.items():
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                genre = genre_enum
                break
        
        # Stimmung/Mood extrahieren
        moods = []
        for mood, pattern in cls.MOOD_PATTERNS.items():
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                moods.append(mood)
        
        # Tonart bestimmen
        key = "Am"  # Default für dunkle Musik
        for key_name, pattern in cls.KEY_PATTERNS.items():
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                key = key_name
                break
        
        return {
            'bpm': bpm,
            'genre': genre,
            'moods': moods if moods else ['driving'],
            'key': key,
            'raw_prompt': prompt
        }


class ArrangerService:
    """Hauptservice für Track-Arrangement-Generierung"""
    
    def __init__(self):
        self.parser = PromptParser()
        self.db_service = DatabaseService()
        logger.info("ArrangerService initialisiert")
    
    def generate_arrangement_plan(self, prompt: str) -> ArrangementPlan:
        """Generiert einen vollständigen Arrangement-Plan aus einem Text-Prompt"""
        logger.info(f"Generiere Arrangement für Prompt: '{prompt}'")
        
        # Prompt analysieren
        params = self.parser.parse_prompt(prompt)
        
        # Track-Struktur basierend auf Genre und BPM generieren
        structure = self._generate_track_structure(
            genre=params['genre'],
            bpm=params['bpm'],
            moods=params['moods']
        )
        
        # Gesamtlänge berechnen
        total_bars = sum(section.bars for section in structure)
        estimated_duration = self._calculate_duration(total_bars, params['bpm'])
        
        plan = ArrangementPlan(
            bpm=params['bpm'],
            key=params['key'],
            genre=params['genre'].value,
            mood=params['moods'],
            structure=structure,
            total_bars=total_bars,
            estimated_duration=estimated_duration
        )
        
        logger.info(f"Arrangement-Plan erstellt: {total_bars} Takte, {estimated_duration:.1f}s")
        return plan
    
    async def create_arrangement(
        self,
        prompt: str,
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Erstellt ein vollständiges Arrangement basierend auf einem Text-Prompt"""
        logger.info(f"Erstelle Arrangement für Prompt: '{prompt}'")
        
        # Arrangement-Plan generieren
        plan = self.generate_arrangement_plan(prompt)
        
        # Stems für jede Sektion auswählen
        selected_stems = []
        global_key = None
        
        for section in plan.structure:
            section_stems = []
            
            for query in section.stem_queries:
                # Erste Stem-Auswahl bestimmt die globale Tonart
                if global_key is None:
                    # Suche nach Stems ohne Tonart-Filter
                    stems = await self._search_stems_for_query(
                        query, None, plan.bpm
                    )
                    if stems:
                        global_key = stems[0].key
                        logger.info(f"Globale Tonart festgelegt: {global_key}")
                        section_stems.extend(stems)
                else:
                    # Suche nach harmonisch kompatiblen Stems
                    stems = await self._search_stems_for_query(
                        query, global_key, plan.bpm
                    )
                    section_stems.extend(stems)
            
            selected_stems.append({
                'section': section.section,
                'bars': section.bars,
                'stems': [{
                    'id': stem.id,
                    'filename': stem.filename,
                    'category': stem.category,
                    'key': stem.key,
                    'bpm': stem.bpm,
                    'harmonic_complexity': stem.harmonic_complexity,
                    'rhythmic_complexity': stem.rhythmic_complexity
                } for stem in section_stems]
            })
        
        # Arrangement-Metadaten zusammenstellen
        arrangement_data = {
            'arrangement_id': f"arr_{hash(prompt)}",
            'prompt': prompt,
            'global_key': global_key or plan.key,
            'bpm': plan.bpm,
            'genre': plan.genre,
            'mood': plan.mood,
            'total_bars': plan.total_bars,
            'estimated_duration': plan.estimated_duration,
            'structure': selected_stems,
            'metadata': {
                'created_with_musical_intelligence': True,
                'harmonic_coherence': True,
                'key_compatibility_used': global_key is not None
            }
        }
        
        logger.info(f"Arrangement erstellt: {len(selected_stems)} Sektionen, Tonart: {global_key}")
        return arrangement_data
    
    async def _search_stems_for_query(
        self,
        query: StemQuery,
        global_key: Optional[str],
        target_bpm: int
    ) -> List[Stem]:
        """Sucht Stems für eine spezifische Query mit musikalischer Intelligenz"""
        try:
            if global_key:
                # Suche nach harmonisch kompatiblen Stems
                stems = await self.db_service.search_harmonically_compatible_stems(
                    base_key=global_key,
                    category=query.category,
                    tags=query.tags,
                    limit=query.count
                )
            else:
                # Erste Suche ohne Tonart-Filter
                stems = await self.db_service.get_stems(
                    category=query.category,
                    bpm_min=target_bpm - 10,
                    bpm_max=target_bpm + 10,
                    processing_status="completed",
                    limit=query.count
                )
            
            if len(stems) < query.count and query.required:
                logger.warning(
                    f"Nur {len(stems)}/{query.count} Stems für {query.category} gefunden"
                )
            
            return stems[:query.count]
            
        except Exception as e:
            logger.error(f"Fehler bei Stem-Suche: {e}")
            return []
    
    def _generate_track_structure(self, genre: MusicGenre, bpm: int, moods: List[str]) -> List[ArrangementSection]:
        """Generiert die Track-Struktur basierend auf Genre und Parametern"""
        
        if genre == MusicGenre.TECHNO:
            return self._generate_techno_structure(bpm, moods)
        elif genre == MusicGenre.HOUSE:
            return self._generate_house_structure(bpm, moods)
        elif genre == MusicGenre.INDUSTRIAL:
            return self._generate_industrial_structure(bpm, moods)
        else:
            # Default: Techno-Struktur
            return self._generate_techno_structure(bpm, moods)
    
    def _generate_techno_structure(self, bpm: int, moods: List[str]) -> List[ArrangementSection]:
        """Generiert eine Techno-Track-Struktur"""
        
        # Basis-Tags aus Moods ableiten
        base_tags = moods.copy()
        
        # BPM-spezifische Anpassungen
        if bpm >= 140:
            base_tags.append('fast')
        elif bpm <= 120:
            base_tags.append('slow')
        
        structure = [
            # Intro: Atmosphäre aufbauen
            ArrangementSection(
                section=TrackSection.INTRO.value,
                bars=16,
                stem_queries=[
                    StemQuery(category="atmo", tags=base_tags[:2], count=1),
                    StemQuery(category="fx", tags=["intro", "sweep"], count=1, required=False)
                ]
            ),
            
            # Buildup: Spannung aufbauen
            ArrangementSection(
                section=TrackSection.BUILDUP.value,
                bars=32,
                stem_queries=[
                    StemQuery(category="kick", tags=base_tags + ["soft"], count=1),
                    StemQuery(category="hihat", tags=["closed"], count=1),
                    StemQuery(category="atmo", tags=base_tags, count=1)
                ]
            ),
            
            # Main Groove: Hauptteil
            ArrangementSection(
                section=TrackSection.GROOVE.value,
                bars=64,
                stem_queries=[
                    StemQuery(category="kick", tags=base_tags + ["punchy"], count=1),
                    StemQuery(category="bass", tags=base_tags + ["driving"], count=1),
                    StemQuery(category="hihat", tags=["open", "closed"], count=2),
                    StemQuery(category="perc", tags=base_tags, count=1, required=False),
                    StemQuery(category="lead", tags=base_tags, count=1, required=False)
                ]
            ),
            
            # Breakdown: Entspannung
            ArrangementSection(
                section=TrackSection.BREAKDOWN.value,
                bars=32,
                stem_queries=[
                    StemQuery(category="bass", tags=base_tags + ["soft"], count=1),
                    StemQuery(category="pad", tags=["atmospheric"], count=1),
                    StemQuery(category="fx", tags=["reverse"], count=1, required=False)
                ]
            ),
            
            # Drop: Höhepunkt
            ArrangementSection(
                section=TrackSection.DROP.value,
                bars=64,
                stem_queries=[
                    StemQuery(category="kick", tags=base_tags + ["heavy"], count=1),
                    StemQuery(category="bass", tags=base_tags + ["massive"], count=1),
                    StemQuery(category="hihat", tags=["aggressive"], count=1),
                    StemQuery(category="lead", tags=base_tags + ["powerful"], count=1),
                    StemQuery(category="perc", tags=base_tags, count=2, required=False)
                ]
            ),
            
            # Outro: Ausklingen
            ArrangementSection(
                section=TrackSection.OUTRO.value,
                bars=16,
                stem_queries=[
                    StemQuery(category="atmo", tags=base_tags + ["fade"], count=1),
                    StemQuery(category="fx", tags=["outro", "reverse"], count=1, required=False)
                ]
            )
        ]
        
        return structure
    
    def _generate_house_structure(self, bpm: int, moods: List[str]) -> List[ArrangementSection]:
        """Generiert eine House-Track-Struktur"""
        # Vereinfachte House-Struktur
        base_tags = moods + ["house", "groovy"]
        
        return [
            ArrangementSection(
                section=TrackSection.INTRO.value,
                bars=16,
                stem_queries=[StemQuery(category="atmo", tags=base_tags, count=1)]
            ),
            ArrangementSection(
                section=TrackSection.GROOVE.value,
                bars=128,
                stem_queries=[
                    StemQuery(category="kick", tags=base_tags + ["four-on-floor"], count=1),
                    StemQuery(category="bass", tags=base_tags, count=1),
                    StemQuery(category="hihat", tags=["shuffle"], count=1),
                    StemQuery(category="vocal", tags=["house"], count=1, required=False)
                ]
            ),
            ArrangementSection(
                section=TrackSection.OUTRO.value,
                bars=16,
                stem_queries=[StemQuery(category="atmo", tags=base_tags, count=1)]
            )
        ]
    
    def _generate_industrial_structure(self, bpm: int, moods: List[str]) -> List[ArrangementSection]:
        """Generiert eine Industrial-Track-Struktur"""
        base_tags = moods + ["industrial", "mechanical", "harsh"]
        
        return [
            ArrangementSection(
                section=TrackSection.INTRO.value,
                bars=8,
                stem_queries=[
                    StemQuery(category="noise", tags=base_tags, count=1),
                    StemQuery(category="fx", tags=["machine", "startup"], count=1)
                ]
            ),
            ArrangementSection(
                section=TrackSection.GROOVE.value,
                bars=96,
                stem_queries=[
                    StemQuery(category="kick", tags=base_tags + ["distorted"], count=1),
                    StemQuery(category="bass", tags=base_tags + ["grinding"], count=1),
                    StemQuery(category="perc", tags=base_tags + ["metal"], count=2),
                    StemQuery(category="noise", tags=base_tags, count=1)
                ]
            ),
            ArrangementSection(
                section=TrackSection.OUTRO.value,
                bars=8,
                stem_queries=[
                    StemQuery(category="fx", tags=["shutdown", "fade"], count=1)
                ]
            )
        ]
    
    def _calculate_duration(self, total_bars: int, bpm: int) -> float:
        """Berechnet die geschätzte Track-Dauer in Sekunden"""
        # 4/4 Takt: 4 Beats pro Takt
        beats_per_bar = 4
        total_beats = total_bars * beats_per_bar
        beats_per_second = bpm / 60
        duration = total_beats / beats_per_second
        return duration
    
    async def validate_arrangement_plan(self, plan: ArrangementPlan) -> Dict[str, Any]:
        """Validiert einen Arrangement-Plan gegen verfügbare Stems"""
        validation_result = {
            'valid': True,
            'warnings': [],
            'missing_stems': [],
            'available_alternatives': {}
        }
        
        for section in plan.structure:
            for query in section.stem_queries:
                # Prüfe ob passende Stems verfügbar sind
                try:
                    stems = await self.db_service.search_stems_by_text(
                        query_text=f"{query.category} {' '.join(query.tags)}",
                        limit=query.count
                    )
                    
                    if len(stems) < query.count and query.required:
                        validation_result['valid'] = False
                        validation_result['missing_stems'].append({
                            'section': section.section,
                            'category': query.category,
                            'tags': query.tags,
                            'needed': query.count,
                            'available': len(stems)
                        })
                    elif len(stems) < query.count:
                        validation_result['warnings'].append(
                            f"Nur {len(stems)}/{query.count} Stems für {query.category} in {section.section}"
                        )
                        
                except Exception as e:
                    logger.error(f"Fehler bei Stem-Validierung: {e}")
                    validation_result['warnings'].append(f"Validierung für {query.category} fehlgeschlagen")
        
        return validation_result
    
    def export_arrangement_plan(self, plan: ArrangementPlan) -> Dict[str, Any]:
        """Exportiert einen Arrangement-Plan als Dictionary"""
        return {
            'bpm': plan.bpm,
            'key': plan.key,
            'genre': plan.genre,
            'mood': plan.mood,
            'total_bars': plan.total_bars,
            'estimated_duration': plan.estimated_duration,
            'structure': [
                {
                    'section': section.section,
                    'bars': section.bars,
                    'volume': section.volume,
                    'effects': section.effects or [],
                    'stem_queries': [
                        {
                            'category': query.category,
                            'tags': query.tags,
                            'count': query.count,
                            'required': query.required
                        }
                        for query in section.stem_queries
                    ]
                }
                for section in plan.structure
            ]
        }