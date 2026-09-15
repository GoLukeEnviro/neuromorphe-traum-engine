from pydantic import BaseModel, Field, model_validator
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
    """Übergang zwischen zwei Sektionen.

    ``type`` ist der kanonische Feldname; ``transition_type`` wird als
    Alias weiter akzeptiert.
    """
    from_section: str
    to_section: str
    type: Optional[str] = None
    transition_type: Optional[str] = None
    duration_bars: int = 0
    effects: Optional[List[str]] = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _sync_type(self) -> "ArrangementTransition":
        if self.type is None and self.transition_type is not None:
            self.type = self.transition_type
        if self.transition_type is None and self.type is not None:
            self.transition_type = self.type
        return self


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
    """Basis-Schema für ein Arrangement."""
    bpm: int = Field(..., gt=0, le=300)
    total_bars: int = Field(..., gt=0)
    track_structure: Dict[str, Any] = Field(..., min_length=1)
    stems: List[int] = Field(..., min_length=1)


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