"""Arrangement-Schemata für die Neuromorphe Traum-Engine."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
from datetime import datetime


class ArrangementAnalysis(BaseModel):
    """Schema für Arrangement-Analyse."""
    arrangement_id: str
    stems: List[Dict[str, Any]]
    structure: Dict[str, Any]
    tempo: float
    key: Optional[str] = None
    time_signature: Optional[str] = "4/4"
    complexity_score: Optional[float] = None
    energy_profile: Optional[List[float]] = None
    harmonic_analysis: Optional[Dict[str, Any]] = None
    rhythmic_patterns: Optional[List[Dict[str, Any]]] = None
    created_at: datetime = datetime.now()
    
    model_config = ConfigDict(from_attributes=True)