from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from .service import SearchService
from .dependencies import get_search_service
from schemas import (
    SearchRequest,
    SearchResponse,
    SimilarityRequest,
    SearchStats
)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/text", response_model=SearchResponse)
async def search_by_text(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service)
):
    """Perform semantic search using text query"""
    try:
        response = await service.search_by_text(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/text", response_model=SearchResponse)
async def search_by_text_get(
    query: str = Query(..., min_length=1, max_length=500),
    category: str = Query(None),
    bpm_min: int = Query(None, ge=60, le=200),
    bpm_max: int = Query(None, ge=60, le=200),
    limit: int = Query(10, ge=1, le=100),
    service: SearchService = Depends(get_search_service)
):
    """Perform semantic search using text query (GET method)"""
    try:
        request = SearchRequest(
            query=query,
            category=category,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            limit=limit
        )
        response = await service.search_by_text(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/similar", response_model=SearchResponse)
async def find_similar_audio(
    request: SimilarityRequest,
    service: SearchService = Depends(get_search_service)
):
    """Find audio files similar to a given audio file"""
    try:
        response = await service.find_similar_audio(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {str(e)}"
        )


@router.get("/similar/{file_id}", response_model=SearchResponse)
async def find_similar_audio_get(
    file_id: str,
    limit: int = Query(10, ge=1, le=100),
    service: SearchService = Depends(get_search_service)
):
    """Find audio files similar to a given audio file (GET method)"""
    try:
        request = SimilarityRequest(
            source_file_id=file_id,
            limit=limit
        )
        response = await service.find_similar_audio(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {str(e)}"
        )


@router.get("/stats", response_model=SearchStats)
async def get_search_statistics(
    service: SearchService = Depends(get_search_service)
):
    """Get search and database statistics"""
    try:
        stats = await service.get_search_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search statistics: {str(e)}"
        )


@router.get("/categories", response_model=List[str])
async def get_categories(
    service: SearchService = Depends(get_search_service)
):
    """Get list of all available categories"""
    try:
        categories = await service.db_service.get_categories()
        return categories
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get categories: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "search"}