"""Arranger Service - Intelligenter Track-Strukturierer

Dieser Service implementiert die kreative Kernlogik der Neuromorphe Traum-Engine v2.0.
Er analysiert Text-Prompts und generiert daraus strukturierte Track-Arrangements.

Der Service arbeitet zweistufig:

* ``generate_arrangement_plan`` - statischer Plan (Takte, Queries, Dauer)
* ``create_arrangement``        - vollständiges, stem-basiertes Arrangement:
  Prompt-Analyse -> Stem-Auswahl -> Struktur -> Übergänge -> Validierung
"""

import re
import inspect
import logging
import math
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from database.service import DatabaseService
from database.models import Stem
from services.neuro_analyzer import NeuroAnalyzer

logger = logging.getLogger(__name__)

#: Mindestähnlichkeit, ab der ein Stem übernommen wird.
MIN_STEM_SIMILARITY = 0.7

#: Standarddauer, wenn keine Dauer angegeben ist.
DEFAULT_ARRANGEMENT_DURATION = 180.0


async def _maybe_await(value: Any) -> Any:
    """Gibt ``value`` zurück bzw. awaited es, falls es awaitable ist."""
    if inspect.isawaitable(value):
        return await value
    return value


class TrackSection(Enum):
    """Verfügbare Track-Sektionen"""
    INTRO = "Intro"
    BUILDUP = "Buildup"
    VERSE = "Verse"
    GROOVE = "Groove"
    BREAKDOWN = "Breakdown"
    CHORUS = "Chorus"
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


#: Typisches Tempo je Genre (BPM).
GENRE_TEMPO = {
    MusicGenre.TECHNO.value: 128,
    MusicGenre.HOUSE.value: 124,
    MusicGenre.TRANCE.value: 138,
    MusicGenre.INDUSTRIAL.value: 132,
    MusicGenre.AMBIENT.value: 90,
    MusicGenre.DRUM_AND_BASS.value: 174,
}

#: Grundenergie je Genre (0.0 - 1.0).
GENRE_ENERGY = {
    MusicGenre.TECHNO.value: 0.8,
    MusicGenre.HOUSE.value: 0.7,
    MusicGenre.TRANCE.value: 0.8,
    MusicGenre.INDUSTRIAL.value: 0.9,
    MusicGenre.AMBIENT.value: 0.3,
    MusicGenre.DRUM_AND_BASS.value: 0.85,
}

#: Energie-Modifikatoren je Stimmung.
MOOD_ENERGY = {
    'dark': 0.0,
    'aggressive': 0.1,
    'driving': 0.05,
    'energetic': 0.1,
    'uplifting': 0.1,
    'euphoric': 0.1,
    'hypnotic': -0.05,
    'melodic': -0.05,
    'minimal': -0.1,
    'atmospheric': -0.1,
    'punchy': 0.05,
    'groovy': 0.05,
    'calm': -0.2,
}

#: Klang-Elemente, die aus dem Prompt-Text erkannt werden.
ELEMENT_PATTERNS: Dict[str, str] = {
    'kick': r'\b(kick|kickdrum|bassdrum)\b',
    'bass': r'\b(bass|bassline|sub.?bass|bassline)\b',
    'drums': r'\b(drums?|drumkit)\b',
    'hihat': r'\b(hi.?hats?|hats?)\b',
    'percussion': r'\b(perc|percussion|toms?|claps?|shaker)\b',
    'synth': r'\b(synth|synthesizer|synthesizers|synths|arp|arpeggio|arpeggios)\b',
    'piano': r'\b(piano|keys|keyboard)\b',
    'vocal': r'\b(vocals?|voice|choir|chops?)\b',
    'pad': r'\b(pads?|strings?)\b',
    'lead': r'\b(lead|melody|melodies|melodic)\b',
    'fx': r'\b(fx|effects?|sweeps?|risers?|noise)\b',
    'drone': r'\b(drones?|drone)\b',
    'industrial': r'\b(industrial|mechanical|machine|harsh|metal)\b',
    'guitar': r'\b(guitars?|riffs?)\b',
    'bell': r'\b(bells?|chimes?|glockenspiel)\b',
}


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
        'energetic': r'\b(energetic|energisch|powerful|kraftvoll)\b',
        'uplifting': r'\b(uplifting|euphoric|euphorisch|joyful|happy|positiv|positive)\b',
        'hypnotic': r'\b(hypnotic|hypnotisch|trancey|trancey patterns)\b',
        'melodic': r'\b(melodic|melodisch|harmonic|harmonisch|melodies|melody)\b',
        'minimal': r'\b(minimal|minimalistic|reduced)\b',
        'atmospheric': r'\b(atmospheric|atmosphärisch|ambient|spacious)\b',
        'punchy': r'\b(punchy|knackig|crisp|sharp)\b',
        'groovy': r'\b(groovy|groove|rhythmic|rhythmisch)\b',
        'calm': r'\b(calm|ruhig|peaceful|gentle|soft)\b',
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
        prompt_lower = (prompt or "").lower()
        
        # BPM extrahieren
        bpm_match = re.search(cls.BPM_PATTERN, prompt_lower)
        bpm = int(bpm_match.group(1)) if bpm_match else None
        
        # Genre bestimmen - Reihenfolge der Patterns entscheidet bei
        # mehrdeutigen Prompts (z. B. "techno with industrial sounds" -> techno).
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
        self.neuro_analyzer = NeuroAnalyzer()
        logger.info("ArrangerService initialisiert")
    
    # ------------------------------------------------------------------
    # Statischer Plan (Legacy-API)
    # ------------------------------------------------------------------

    def generate_arrangement_plan(self, prompt: str) -> ArrangementPlan:
        """Generiert einen vollständigen Arrangement-Plan aus einem Text-Prompt"""
        logger.info(f"Generiere Arrangement für Prompt: '{prompt}'")
        
        # Prompt analysieren
        params = self.parser.parse_prompt(prompt)
        genre = params['genre']
        bpm = int(params['bpm'] or GENRE_TEMPO.get(genre.value, 128))
        
        # Track-Struktur basierend auf Genre und BPM generieren
        structure = self._generate_track_structure(
            genre=genre,
            bpm=bpm,
            moods=params['moods']
        )
        
        # Gesamtlänge berechnen
        total_bars = sum(section.bars for section in structure)
        estimated_duration = self._calculate_duration(total_bars, bpm)
        
        plan = ArrangementPlan(
            bpm=bpm,
            key=params['key'],
            genre=genre.value,
            mood=params['moods'],
            structure=structure,
            total_bars=total_bars,
            estimated_duration=estimated_duration
        )
        
        logger.info(f"Arrangement-Plan erstellt: {total_bars} Takte, {estimated_duration:.1f}s")
        return plan
    
    # ------------------------------------------------------------------
    # Prompt-Analyse
    # ------------------------------------------------------------------

    def _extract_elements(self, prompt: str) -> List[str]:
        """Extrahiert Klang-Elemente aus einem Prompt (Reihenfolge = Text)."""
        prompt_lower = (prompt or "").lower()
        matches: List[Tuple[int, str]] = []
        for element, pattern in ELEMENT_PATTERNS.items():
            found = re.search(pattern, prompt_lower, re.IGNORECASE)
            if found:
                matches.append((found.start(), element))
        matches.sort(key=lambda item: item[0])
        return [element for _, element in matches]

    @staticmethod
    def _estimate_energy(genre: str, moods: List[str]) -> float:
        """Schätzt die Energie eines Tracks aus Genre und Stimmung."""
        energy = GENRE_ENERGY.get(genre, 0.7)
        for mood in moods or []:
            energy += MOOD_ENERGY.get(mood, 0.0)
        return round(min(1.0, max(0.0, energy)), 3)

    async def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analysiert einen Text-Prompt (Genre, Mood, Tempo, Elemente)."""
        if prompt is None or not str(prompt).strip():
            raise ValueError("Prompt darf nicht leer sein")

        prompt = str(prompt)
        params = self.parser.parse_prompt(prompt)
        genre_enum = params.get('genre')
        genre = genre_enum.value if isinstance(genre_enum, MusicGenre) else str(genre_enum or MusicGenre.TECHNO.value)
        moods = list(params.get('moods') or [])

        explicit_bpm = params.get('bpm')
        tempo = int(explicit_bpm) if explicit_bpm else GENRE_TEMPO.get(genre, 128)

        analysis: Dict[str, Any] = {
            'prompt': prompt,
            'genre': genre,
            'mood': moods,
            'moods': moods,
            'key': params.get('key', 'Am'),
            'tempo': tempo,
            'bpm': tempo,
            'energy': self._estimate_energy(genre, moods),
            'elements': self._extract_elements(prompt),
            'embeddings': None,
        }

        # Text-Embedding über den NeuroAnalyzer (best effort - die Analyse darf
        # auch ohne Modell/Embedding funktionieren).
        try:
            embedding_result = await _maybe_await(
                self.neuro_analyzer.analyze_text_prompt(prompt)
            )
            if isinstance(embedding_result, dict):
                embeddings = embedding_result.get('embeddings')
                if embeddings is not None:
                    analysis['embeddings'] = embeddings
        except Exception as error:
            logger.debug(f"Text-Embedding nicht verfügbar: {error}")

        return analysis

    # ------------------------------------------------------------------
    # Stem-Auswahl
    # ------------------------------------------------------------------

    async def _select_stems(
        self,
        prompt_analysis: Dict[str, Any],
        session: Optional[Any] = None,
        max_stems: int = 8,
        min_similarity: float = MIN_STEM_SIMILARITY,
    ) -> List[Dict[str, Any]]:
        """Wählt Stems anhand ihrer Ähnlichkeit zum Prompt aus.

        Ergebnis ist auf ``max_stems`` begrenzt, nach Ähnlichkeit absteigend
        sortiert und enthält keine Treffer unterhalb von ``min_similarity``.
        """
        if not isinstance(prompt_analysis, dict):
            prompt_analysis = {}

        max_stems = max(1, int(max_stems or 1))
        min_similarity = float(min_similarity)
        embedding = prompt_analysis.get('embeddings')
        elements = [str(element).lower() for element in (prompt_analysis.get('elements') or [])]
        genre = prompt_analysis.get('genre')

        candidates: Any = []
        try:
            candidates = await _maybe_await(
                self.neuro_analyzer.get_similar_stems(
                    embedding,
                    session=session,
                    limit=max(max_stems * 4, 50),
                    threshold=min_similarity,
                )
            )
        except Exception as error:
            logger.error(f"Fehler bei der Stem-Suche: {error}")
            candidates = []

        if not isinstance(candidates, (list, tuple)):
            candidates = []

        scored: List[Tuple[float, float, int, Dict[str, Any]]] = []
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            try:
                similarity = float(candidate.get('similarity', 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(similarity) or similarity < min_similarity:
                continue

            stem_type = str(candidate.get('type') or candidate.get('category') or '').lower()
            boost = 0.0
            if elements and stem_type:
                if any(
                    element in stem_type or stem_type in element
                    for element in elements
                    if element
                ):
                    boost += 0.05
            if genre and str(candidate.get('genre') or '').lower() == str(genre).lower():
                boost += 0.05

            scored.append((similarity + boost, similarity, position, candidate))

        # Beste Treffer zuerst; Gleichstand stabil über die Eingabereihenfolge.
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))

        selected = [dict(item[3]) for item in scored[:max_stems]]
        selected.sort(key=lambda stem: float(stem.get('similarity', 0.0)), reverse=True)
        return selected

    # ------------------------------------------------------------------
    # Struktur
    # ------------------------------------------------------------------

    @staticmethod
    def _structure_template(genre: str) -> List[Tuple[str, float, float, int]]:
        """Sektions-Template je Genre: (Name, Anteil, Ziel-Energie, max. Stems)."""
        templates = {
            MusicGenre.TECHNO.value: [
                ('intro', 0.10, 0.3, 2),
                ('buildup', 0.20, 0.6, 3),
                ('drop', 0.35, 0.9, 4),
                ('breakdown', 0.20, 0.5, 3),
                ('outro', 0.15, 0.3, 2),
            ],
            MusicGenre.HOUSE.value: [
                ('intro', 0.10, 0.3, 2),
                ('verse', 0.25, 0.6, 3),
                ('chorus', 0.30, 0.85, 4),
                ('breakdown', 0.20, 0.5, 3),
                ('outro', 0.15, 0.3, 2),
            ],
            MusicGenre.AMBIENT.value: [
                ('intro', 0.20, 0.2, 2),
                ('breakdown', 0.35, 0.4, 3),
                ('outro', 0.45, 0.2, 2),
            ],
            MusicGenre.TRANCE.value: [
                ('intro', 0.10, 0.4, 2),
                ('buildup', 0.25, 0.7, 3),
                ('drop', 0.30, 0.9, 4),
                ('breakdown', 0.20, 0.5, 3),
                ('outro', 0.15, 0.3, 2),
            ],
            MusicGenre.DRUM_AND_BASS.value: [
                ('intro', 0.10, 0.4, 2),
                ('verse', 0.30, 0.7, 3),
                ('drop', 0.30, 0.95, 4),
                ('breakdown', 0.15, 0.5, 3),
                ('outro', 0.15, 0.3, 2),
            ],
            MusicGenre.INDUSTRIAL.value: [
                ('intro', 0.10, 0.4, 2),
                ('buildup', 0.20, 0.7, 3),
                ('drop', 0.35, 0.95, 4),
                ('breakdown', 0.20, 0.6, 3),
                ('outro', 0.15, 0.35, 2),
            ],
        }
        return templates.get(str(genre), templates[MusicGenre.TECHNO.value])

    @staticmethod
    def _allocate_durations(total: float, weights: List[float]) -> List[int]:
        """Verteilt die Gesamtdauer auf Sektionen (Summe <= total, je >= 1)."""
        total_int = int(max(0, math.floor(total)))
        count = max(1, len(weights))

        durations = [max(1, int(math.floor(total_int * weight))) for weight in weights]

        # Obergrenze einhalten: größte Sektionen zuerst reduzieren.
        while sum(durations) > total_int and total_int >= count:
            index = max(range(len(durations)), key=lambda i: durations[i])
            if durations[index] <= 1:
                break
            durations[index] -= 1

        return durations

    async def _create_structure(
        self,
        prompt_analysis: Dict[str, Any],
        selected_stems: List[Any],
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Erzeugt eine lückenlose Sektionsstruktur für die gewünschte Dauer."""
        if not isinstance(prompt_analysis, dict):
            prompt_analysis = {}

        total_duration = float(duration) if duration else DEFAULT_ARRANGEMENT_DURATION
        if total_duration <= 0:
            raise ValueError("Dauer muss größer als 0 sein")

        genre = str(prompt_analysis.get('genre') or MusicGenre.TECHNO.value)
        template = self._structure_template(genre)
        durations = self._allocate_durations(total_duration, [entry[1] for entry in template])

        sections: List[Dict[str, Any]] = []
        start = 0
        for (name, _weight, target_energy, max_stems), section_duration in zip(template, durations):
            sections.append({
                'name': name,
                'start': int(start),
                'duration': int(section_duration),
                'stems': [],
                'target_energy': float(target_energy),
                'max_stems': int(max_stems),
                'energy': float(target_energy),
            })
            start += section_duration

        # Stems auf die Sektionen verteilen.
        sections = await self._optimize_stem_placement(selected_stems or [], sections)

        structure = {
            'sections': sections,
            'total_duration': int(sum(section['duration'] for section in sections)),
            'genre': genre,
            'energy': prompt_analysis.get('energy', 0.7),
            'tempo': prompt_analysis.get('tempo', GENRE_TEMPO.get(genre, 128)),
        }
        return structure

    async def _optimize_stem_placement(
        self,
        stems: List[Any],
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Verteilt Stems anhand ihrer Energie auf die Sektionen.

        Energetische Stems landen in energiereichen Sektionen; ``max_stems``
        je Sektion wird eingehalten.
        """
        if not isinstance(sections, list):
            sections = []

        optimized: List[Dict[str, Any]] = [
            dict(section) if isinstance(section, dict) else {'name': str(section)}
            for section in sections
        ]
        if not optimized:
            return optimized

        normalized: List[Dict[str, Any]] = []
        for stem in stems or []:
            if isinstance(stem, dict):
                try:
                    energy = float(stem.get('energy', 0.5))
                except (TypeError, ValueError):
                    energy = 0.5
                normalized.append({
                    'id': stem.get('id', stem.get('stem_id')),
                    'energy': min(1.0, max(0.0, energy)),
                })
            else:
                normalized.append({'id': stem, 'energy': 0.5})

        targets = []
        capacities = []
        for section in optimized:
            try:
                targets.append(float(section.get('target_energy', section.get('energy', 0.5))))
            except (TypeError, ValueError):
                targets.append(0.5)
            try:
                capacity = int(section.get('max_stems', len(normalized) or 1))
            except (TypeError, ValueError):
                capacity = len(normalized) or 1
            capacities.append(max(0, min(capacity, len(normalized) or capacity)))

        assigned: List[List[Any]] = [[] for _ in optimized]

        # Energetischste Stems zuerst, dann den energetisch passendsten Slot.
        ordering = sorted(
            range(len(normalized)),
            key=lambda index: (-normalized[index]['energy'], index),
        )
        for index in ordering:
            stem = normalized[index]
            section_order = sorted(
                range(len(optimized)),
                key=lambda i: (abs(targets[i] - stem['energy']), -targets[i], i),
            )
            for section_index in section_order:
                if len(assigned[section_index]) < capacities[section_index]:
                    assigned[section_index].append(stem['id'])
                    break

        for section_index, section in enumerate(optimized):
            section['stems'] = [stem_id for stem_id in assigned[section_index] if stem_id is not None]

        return optimized

    # ------------------------------------------------------------------
    # Übergänge & Validierung
    # ------------------------------------------------------------------

    async def _apply_transitions(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Ergänzt Fade-In/Fade-Out-Übergänge zwischen den Sektionen."""
        if not isinstance(structure, dict):
            structure = {'sections': []}

        sections = structure.get('sections')
        if not isinstance(sections, list):
            sections = []
            structure['sections'] = sections

        for position, section in enumerate(sections):
            if not isinstance(section, dict):
                continue

            try:
                section_duration = float(section.get('duration', 0) or 0)
            except (TypeError, ValueError):
                section_duration = 0.0

            fade = min(4.0, max(0.5, section_duration / 8.0)) if section_duration else 1.0
            transitions: Dict[str, Any] = {
                'fade_in': {
                    'duration': round(fade, 2),
                    'curve': 'exponential' if position == 0 else 'linear',
                },
                'fade_out': {
                    'duration': round(fade, 2),
                    'curve': 'exponential' if position == len(sections) - 1 else 'linear',
                },
            }
            section['transitions'] = transitions

        return structure

    async def _validate_arrangement(self, arrangement: Dict[str, Any]) -> bool:
        """Validiert ein Arrangement (Sektionen lückenlos, positive Dauer)."""
        try:
            if not isinstance(arrangement, dict):
                return False

            structure = arrangement.get('structure')
            if not isinstance(structure, dict):
                return False

            sections = structure.get('sections')
            if not isinstance(sections, list) or not sections:
                return False

            expected_start = 0.0
            for section in sections:
                if not isinstance(section, dict):
                    return False
                if not all(key in section for key in ('name', 'start', 'duration')):
                    return False

                start = float(section['start'])
                section_duration = float(section['duration'])
                if section_duration <= 0 or start < 0:
                    return False
                if abs(start - expected_start) > 1e-6:
                    return False

                stems = section.get('stems', [])
                if stems is not None and not isinstance(stems, (list, tuple)):
                    return False

                expected_start = start + section_duration

            return True

        except (TypeError, ValueError, KeyError):
            return False

    # ------------------------------------------------------------------
    # Haupt-API
    # ------------------------------------------------------------------

    async def create_arrangement(
        self,
        prompt: str,
        duration: Optional[float] = None,
        session: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Erstellt ein vollständiges Arrangement basierend auf einem Text-Prompt.

        Raises:
            ValueError: bei leerem Prompt oder ungültiger Dauer (<= 0).
        """
        # Kompatibilität: manche Aufrufer übergeben die DB-Session positionell.
        if duration is not None and session is None and not isinstance(duration, (int, float, str)):
            session, duration = duration, None

        if prompt is None or not str(prompt).strip():
            raise ValueError("Prompt darf nicht leer sein")

        if duration is None:
            duration = DEFAULT_ARRANGEMENT_DURATION
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            raise ValueError(f"Ungültige Dauer: {duration}")
        if duration <= 0:
            raise ValueError("Dauer muss größer als 0 sein")

        logger.info(f"Erstelle Arrangement für Prompt: '{prompt}' ({duration:.0f}s)")

        # 1. Prompt analysieren (muss genau einmal und positional aufgerufen werden)
        prompt_analysis = await self._analyze_prompt(prompt)

        # 2. Stems auswählen
        selected_stems = await self._select_stems(prompt_analysis, session=session)

        # 3. Struktur erzeugen
        structure = await self._create_structure(
            prompt_analysis,
            selected_stems,
            duration=duration,
        )

        # 4. Übergänge ergänzen
        structure = await _maybe_await(self._apply_transitions(structure))

        stem_ids = [
            stem.get('id') if isinstance(stem, dict) else stem
            for stem in (selected_stems or [])
        ]
        stem_ids = [stem_id for stem_id in stem_ids if stem_id is not None]

        arrangement = {
            'arrangement_id': f"arr_{uuid.uuid4().hex[:12]}",
            'prompt': prompt,
            'structure': structure,
            'stems': stem_ids,
            'metadata': {
                'genre': prompt_analysis.get('genre'),
                'mood': prompt_analysis.get('mood'),
                'tempo': prompt_analysis.get('tempo'),
                'key': prompt_analysis.get('key'),
                'energy': prompt_analysis.get('energy'),
                'elements': prompt_analysis.get('elements'),
                'requested_duration': duration,
                'stem_count': len(stem_ids),
                'created_with_musical_intelligence': True,
            },
        }

        # 5. Validieren (nicht abbrechen - nur protokollieren)
        is_valid = await _maybe_await(self._validate_arrangement(arrangement))
        arrangement['metadata']['valid'] = bool(is_valid)
        if not is_valid:
            logger.warning("Arrangement-Validierung fehlgeschlagen: %s", arrangement['arrangement_id'])

        logger.info(
            "Arrangement erstellt: %s (%d Sektionen, %.0fs)",
            arrangement['arrangement_id'],
            len(structure.get('sections', []) or []),
            structure.get('total_duration', 0),
        )
        return arrangement
    
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
