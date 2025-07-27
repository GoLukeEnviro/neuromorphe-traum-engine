from sqlalchemy.orm import Session
from typing import List, Optional
from .models import Stem
from ..schemas.stem import StemCreate

def get_stems(db: Session, skip: int = 0, limit: int = 100) -> List[Stem]:
    """Get all stems with pagination"""
    return db.query(Stem).offset(skip).limit(limit).all()

def get_stem_by_id(db: Session, stem_id: int) -> Optional[Stem]:
    """Get a specific stem by ID"""
    return db.query(Stem).filter(Stem.id == stem_id).first()

def get_stem_by_path(db: Session, path: str) -> Optional[Stem]:
    """Get a stem by its file path"""
    return db.query(Stem).filter(Stem.path == path).first()

def create_stem(db: Session, stem: StemCreate) -> Stem:
    """Create a new stem entry"""
    db_stem = Stem(**stem.dict())
    db.add(db_stem)
    db.commit()
    db.refresh(db_stem)
    return db_stem

def delete_stem(db: Session, stem_id: int) -> bool:
    """Delete a stem by ID"""
    stem = db.query(Stem).filter(Stem.id == stem_id).first()
    if stem:
        db.delete(stem)
        db.commit()
        return True
    return False

def get_stems_by_category(db: Session, category: str) -> List[Stem]:
    """Get all stems of a specific category"""
    return db.query(Stem).filter(Stem.category == category).all()

def search_stems_by_text(db: Session, search_text: str) -> List[Stem]:
    """Search stems by tags or category (simple text search)"""
    return db.query(Stem).filter(
        (Stem.tags.contains(search_text)) | 
        (Stem.category.contains(search_text))
    ).all()