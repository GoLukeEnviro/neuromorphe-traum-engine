"""Datenbankkonfiguration und Session-Management für die Neuromorphe Traum-Engine.

Definiert die Datenbank-Engine, Session-Fabrik und Hilfsfunktionen für die
Interaktion mit der SQLite-Datenbank.
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from ..core.config import settings
import os
from typing import Generator

# Ensure database directory exists
db_dir: str = os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", ""))
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# SQLite database URL
DATABASE_URL: str = settings.DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables() -> None:
    """Create all tables"""
    Base.metadata.create_all(bind=engine)