from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from ...db.database import get_db
from ...db import crud
from ...schemas.stem import Stem, StemList, SearchResult
from ...services.search import search_service

router = APIRouter()

@router.get("/", response_model=List[Stem])
def get_stems(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    db: Session = Depends(get_db)
):
    """Get all stems with pagination"""
    stems = crud.get_stems(db, skip=skip, limit=limit)
    return stems

@router.get("/{stem_id}", response_model=Stem)
def get_stem(
    stem_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific stem by ID"""
    stem = crud.get_stem_by_id(db, stem_id=stem_id)
    if stem is None:
        raise HTTPException(status_code=404, detail="Stem not found")
    return stem

@router.get("/category/{category}", response_model=List[Stem])
def get_stems_by_category(
    category: str,
    db: Session = Depends(get_db)
):
    """Get all stems of a specific category"""
    stems = crud.get_stems_by_category(db, category=category)
    return stems

@router.get("/search/", response_model=List[SearchResult])
def search_stems(
    prompt: str = Query(..., description="Search query text"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db)
):
    """Search stems using semantic similarity"""
    try:
        results = search_service.search(
            db=db,
            query=prompt,
            top_k=top_k,
            category_filter=category
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")