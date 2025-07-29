"""API-Endpunkte für die Verwaltung und Suche von Audio-Stems.

Definiert Routen für das Abrufen, Suchen und Generieren von Tracks
basierend auf Audio-Stems.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from ...database.database import get_database_manager
from ...database.service import DatabaseService
from ...database.models import Stem
from ...schemas.schemas import StemList, SearchResult, StemResponse
from ...services.search import SearchService
from ...services.arranger import ArrangerService
from ...services.renderer import RendererService
from ...core.logging import get_logger
from ...core.config import settings

logger = get_logger(__name__)

# Dependency-Funktionen für Services
def get_search_service() -> SearchService:
    return SearchService()

def get_arranger_service() -> ArrangerService:
    return ArrangerService(settings=settings) # Annahme: ArrangerService benötigt Settings

def get_renderer_service() -> RendererService:
    return RendererService(config=settings)

def get_database_service() -> DatabaseService:
    return DatabaseService()


class TrackGenerationRequest(BaseModel):
    """Request für Track-Generierung"""
    prompt: str = Field(..., description="Beschreibung des gewünschten Tracks")
    bpm: Optional[int] = Field(None, ge=60, le=200, description="Beats per Minute")
    duration: Optional[float] = Field(None, ge=10.0, le=300.0, description="Track-Dauer in Sekunden")
    include_generated: bool = Field(True, description="Generierte Stems einbeziehen")
    include_separated: bool = Field(True, description="Separierte Stems einbeziehen")
    include_original: bool = Field(True, description="Originale Stems einbeziehen")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt": "aggressive industrial techno 140 bpm",
            "bpm": 140,
            "duration": 120.0,
            "include_generated": True,
            "include_separated": True,
            "include_original": True
        }
    })

router = APIRouter()

@router.get("/", response_model=List[StemResponse])
async def get_stems(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get all stems with pagination"""
    stems = await db_service.get_all_stems(skip=skip, limit=limit)
    return stems

@router.get("/{stem_id}", response_model=StemResponse)
async def get_stem(
    stem_id: int,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get a specific stem by ID"""
    stem = await db_service.get_stem_by_id(stem_id=stem_id)
    if stem is None:
        raise HTTPException(status_code=404, detail="Stem not found")
    return stem

@router.get("/category/{category}", response_model=List[StemResponse])
async def get_stems_by_category(
    category: str,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get all stems of a specific category"""
    stems = await db_service.get_stems_by_category(category=category)
    return stems

@router.get("/search/", response_model=List[SearchResult])
async def search_stems(
    prompt: str = Query(..., description="Search query text"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    search_service: SearchService = Depends(get_search_service)
):
    """Search stems using semantic similarity"""
    try:
        results = await search_service.search(
            query=prompt,
            top_k=top_k,
            category_filter=category
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/generate-track")
async def generate_track(
    request: TrackGenerationRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
    arranger_service: ArrangerService = Depends(get_arranger_service),
    renderer_service: RendererService = Depends(get_renderer_service)
):
    """
    Generiert einen vollständigen Track basierend auf einem Prompt
    
    Dieser Endpunkt nutzt den ArrangerService und RendererService um einen Track
    aus verfügbaren Stems zu komponieren. Er kann auf originale, separierte und
    generierte Stems zugreifen, je nach den Request-Parametern.
    """
    try:
        logger.info(f"Starte Track-Generierung mit Prompt: {request.prompt}")
        
        # Verfügbare Stem-Quellen basierend auf Request-Parametern bestimmen
        allowed_sources = []
        if request.include_original:
            allowed_sources.append('original')
        if request.include_separated:
            allowed_sources.append('separated')
        if request.include_generated:
            allowed_sources.append('generated')
        
        if not allowed_sources:
            raise HTTPException(status_code=400, detail="Mindestens eine Stem-Quelle muss aktiviert sein")
        
        # Stems basierend auf Prompt und erlaubten Quellen suchen
        search_results = await search_service.search(
            query=request.prompt,
            top_k=50,  # Mehr Stems für bessere Auswahl
            category_filter=None
        )
        
        # Stems nach erlaubten Quellen filtern
        filtered_stems = []
        for result in search_results:
            stem = result.stem # result.stem is already the Stem object
            if stem and hasattr(stem, 'source') and stem.source in allowed_sources:
                filtered_stems.append(stem)
        
        if not filtered_stems:
            raise HTTPException(
                status_code=404, 
                detail=f"Keine passenden Stems für Prompt '{request.prompt}' mit den gewählten Quellen gefunden"
            )
        
        logger.info(f"Gefunden: {len(filtered_stems)} passende Stems für Track-Generierung")
        
        # Track-Arrangement erstellen
        arrangement_result = await arranger_service.create_arrangement(
            stems=filtered_stems,
            prompt=request.prompt,
            target_bpm=request.bpm,
            target_duration=request.duration
        )
        
        if not arrangement_result['success']:
            raise HTTPException(
                status_code=500, 
                detail=f"Arrangement-Erstellung fehlgeschlagen: {arrangement_result.get('error', 'Unbekannter Fehler')}"
            )
        
        # Track rendern
        render_result = await renderer_service.render_track(
            arrangement=arrangement_result['arrangement'],
            output_format='wav',
            quality='high'
        )
        
        if not render_result['success']:
            raise HTTPException(
                status_code=500, 
                detail=f"Track-Rendering fehlgeschlagen: {render_result.get('error', 'Unbekannter Fehler')}"
            )
        
        # Erfolgreiche Antwort
        return {
            'success': True,
            'message': f'Track erfolgreich generiert: {render_result["output_path"]}',
            'prompt': request.prompt,
            'track_info': {
                'output_path': render_result['output_path'],
                'duration': render_result.get('duration'),
                'bpm': arrangement_result.get('detected_bpm') or request.bpm,
                'stems_used': len(arrangement_result['arrangement']['stems']),
                'stem_sources': {
                    'original': sum(1 for s in filtered_stems if getattr(s, 'source', '') == 'original'),
                    'separated': sum(1 for s in filtered_stems if getattr(s, 'source', '') == 'separated'),
                    'generated': sum(1 for s in filtered_stems if getattr(s, 'source', '') == 'generated')
                }
            },
            'arrangement_details': arrangement_result.get('arrangement_details'),
            'render_details': render_result.get('render_details')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler bei Track-Generierung: {e}")
        raise HTTPException(status_code=500, detail=f"Track-Generierung fehlgeschlagen: {str(e)}")