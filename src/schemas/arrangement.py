from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict


class ArrangementStem(BaseModel):
    stem_id: int
    start_offset_bars: int
    duration_bars: int


class ArrangementSection(BaseModel):
    section: str
    bars: int
    stem_queries: List[Dict[str, Any]]
    volume: float = 1.0
    effects: Optional[List[str]] = None


class ArrangementTransition(BaseModel):
    from_section: str
    to_section: str
    transition_type: str
    duration_bars: int
    effects: Optional[List[str]] = None


class ArrangementStructure(BaseModel):
    sections: List[ArrangementSection]
    transitions: Optional[List[ArrangementTransition]] = None


class ArrangementMetadata(BaseModel):
    created_with_musical_intelligence: bool
    harmonic_coherence: bool
    key_compatibility_used: bool


class ArrangementCreate(BaseModel):
    bpm: int
    total_bars: int
    track_structure: Dict[str, Any]
    stems: List[int]


class ArrangementUpdate(BaseModel):
    bpm: Optional[int] = None
    total_bars: Optional[int] = None
    track_structure: Optional[Dict[str, Any]] = None
    stems: Optional[List[int]] = None


class ArrangementBase(BaseModel):
    bpm: int
    total_bars: int
    track_structure: Dict[str, Any]
    stems: List[int]


class ArrangementResponse(BaseModel):
    arrangement_id: str
    prompt: str
    global_key: str
    bpm: int
    genre: str
    mood: List[str]
    total_bars: int
    estimated_duration: Optional[float] = None
    structure: List[Dict[str, Any]]
    metadata: Dict[str, Any]
