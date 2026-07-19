"""Basis-Schemas für die Neuromorphe Traum-Engine."""

from datetime import datetime
from pydantic import BaseModel, Field


class TimestampMixin(BaseModel):
    """Mixin für Zeitstempel-Felder."""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {"extra": "forbid"}