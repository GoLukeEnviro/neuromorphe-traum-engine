"""Datenbankmodelle für die Neuromorphe Traum-Engine.

Definiert die SQLAlchemy-Modelle für die `stems`-Tabelle und andere
Datenbankentitäten.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, LargeBinary
from sqlalchemy.sql import func
from .database import Base
from datetime import datetime
from typing import Optional

class Stem(Base):
    """SQLAlchemy-Modell für die `stems`-Tabelle.

    Speichert Metadaten und extrahierte Features von Audio-Stems.
    """
    __tablename__ = "stems"
    
    id: Column = Column(Integer, primary_key=True, index=True)
    path: Column = Column(String, unique=True, index=True, nullable=False)
    bpm: Column = Column(Float, nullable=True)
    key: Column = Column(String, nullable=True)
    category: Column = Column(String, nullable=True)
    tags: Column = Column(Text, nullable=True)  # JSON string
    features: Column = Column(Text, nullable=True)  # JSON string
    quality_ok: Column = Column(Boolean, default=True)
    clap_embedding: Column = Column(LargeBinary, nullable=True)  # Binary data for embeddings
    imported_at: Column = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        return f"<Stem(id={self.id}, path='{self.path}', category='{self.category}')>"