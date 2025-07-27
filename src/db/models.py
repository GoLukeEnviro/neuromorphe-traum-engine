from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, LargeBinary
from sqlalchemy.sql import func
from .database import Base

class Stem(Base):
    """SQLAlchemy model for stems table"""
    __tablename__ = "stems"
    
    id = Column(Integer, primary_key=True, index=True)
    path = Column(String, unique=True, index=True, nullable=False)
    bpm = Column(Float, nullable=True)
    key = Column(String, nullable=True)
    category = Column(String, nullable=True)
    tags = Column(Text, nullable=True)  # JSON string
    features = Column(Text, nullable=True)  # JSON string
    quality_ok = Column(Boolean, default=True)
    clap_embedding = Column(LargeBinary, nullable=True)  # Binary data for embeddings
    imported_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<Stem(id={self.id}, path='{self.path}', category='{self.category}')>"