"""API-Endpunkte für die Neuromorphe Traum-Engine v2.0

Diese Datei definiert alle FastAPI-Endpunkte für das Backend.
"""

import os
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database.database import get_db
from ..database import crud
from ..database.models import Stem, GeneratedTrack, ProcessingJob
from ..schemas.schemas import (
    # Stem schemas
    StemResponse, StemCreate, StemUpdate, StemSearchRequest, SearchResponse,
    # Track schemas
    TrackResponse, TrackCreate, TrackUpdate, GenerationRequest,
    # Job schemas
    ProcessingJobResponse, ProcessingJobCreate, BatchProcessRequest, BatchProcessResponse,
    # Utility schemas
    HealthResponse, StatisticsResponse, SuccessResponse, ErrorResponse,
    PreprocessRequest, ValidationResult, PaginationParams, PaginatedResponse
)
from ..services.preprocessor import PreprocessorService
from ..services.arranger import ArrangerService
from ..services.renderer import RendererService
from ..services.neuro_analyzer import NeuroAnalyzer
from ..core.config import settings
from ..core.logging import get_logger

# Router-Instanzen
api_router = APIRouter()
stem_router = APIRouter(prefix="/stems", tags=["stems"])
track_router = APIRouter(prefix="/tracks", tags=["tracks"])
job_router = APIRouter(prefix="/jobs", tags=["jobs"])
system_router = APIRouter(prefix="/system", tags=["system"])

# Services und Konfiguration
logger = get_logger(__name__)

# Service-Instanzen (werden bei Bedarf initialisiert)
_preprocessor_service = None
_arranger_service = None
_renderer_service = None
_neuro_analyzer = None


def get_preprocessor_service() -> PreprocessorService:
    """Dependency für PreprocessorService"""
    global _preprocessor_service
    if _preprocessor_service is None:
        _preprocessor_service = PreprocessorService()
    return _preprocessor_service


def get_arranger_service() -> ArrangerService:
    """Dependency für ArrangerService"""
    global _arranger_service
    if _arranger_service is None:
        _arranger_service = ArrangerService()
    return _arranger_service


def get_renderer_service() -> RendererService:
    """Dependency für RendererService"""
    global _renderer_service
    if _renderer_service is None:
        _renderer_service = RendererService()
    return _renderer_service


def get_neuro_analyzer() -> NeuroAnalyzer:
    """Dependency für NeuroAnalyzer"""
    global _neuro_analyzer
    if _neuro_analyzer is None:
        _neuro_analyzer = NeuroAnalyzer()
    return _neuro_analyzer


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@system_router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """System-Health-Check"""
    try:
        # Datenbankverbindung testen
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    # CLAP-Modell-Status prüfen
    try:
        analyzer = get_neuro_analyzer()
        clap_status = "healthy" if analyzer.clap_embedder else "unhealthy"
    except Exception as e:
        logger.error(f"CLAP model health check failed: {e}")
        clap_status = "unhealthy"
    
    # Verarbeitungsqueue-Größe
    try:
        queue_size = crud.get_active_jobs_count(db)
    except Exception:
        queue_size = -1
    
    # Gesamtstatus bestimmen
    if db_status == "healthy" and clap_status == "healthy":
        overall_status = "healthy"
    elif db_status == "unhealthy" or clap_status == "unhealthy":
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.VERSION,
        uptime_seconds=0.0,  # TODO: Implementieren
        database_status=db_status,
        clap_model_status=clap_status,
        processing_queue_size=queue_size,
        system_metrics={
            "memory_usage": "unknown",
            "cpu_usage": "unknown",
            "disk_usage": "unknown"
        }
    )


@system_router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """System-Statistiken abrufen"""
    try:
        stats = crud.get_system_statistics(db)
        return StatisticsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@system_router.post("/cleanup", response_model=SuccessResponse)
async def cleanup_system(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """System-Cleanup durchführen"""
    try:
        # Cleanup-Job erstellen
        job_data = ProcessingJobCreate(
            job_type="cleanup",
            priority=3,
            input_data={"cleanup_type": "full"},
            parameters={"remove_orphaned": True, "compress_old": True}
        )
        
        job = crud.create_processing_job(db, job_data)
        
        # Background-Task für Cleanup
        background_tasks.add_task(run_cleanup_job, job.id)
        
        return SuccessResponse(
            message="Cleanup job started",
            data={"job_id": job.id},
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to start cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to start cleanup")


# ============================================================================
# STEM ENDPOINTS
# ============================================================================

@stem_router.get("/", response_model=PaginatedResponse)
async def list_stems(
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db)
):
    """Alle Stems auflisten"""
    try:
        stems, total = crud.get_stems_paginated(db, pagination.offset, pagination.limit)
        
        return PaginatedResponse(
            items=[StemResponse.from_orm(stem) for stem in stems],
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=max(1, (total + pagination.size - 1) // pagination.size)
        )
    except Exception as e:
        logger.error(f"Failed to list stems: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stems")


@stem_router.get("/{stem_id}", response_model=StemResponse)
async def get_stem(stem_id: int, db: Session = Depends(get_db)):
    """Einzelnen Stem abrufen"""
    stem = crud.get_stem(db, stem_id)
    if not stem:
        raise HTTPException(status_code=404, detail="Stem not found")
    return StemResponse.from_orm(stem)


@stem_router.put("/{stem_id}", response_model=StemResponse)
async def update_stem(
    stem_id: int,
    stem_update: StemUpdate,
    db: Session = Depends(get_db)
):
    """Stem aktualisieren"""
    stem = crud.get_stem(db, stem_id)
    if not stem:
        raise HTTPException(status_code=404, detail="Stem not found")
    
    try:
        updated_stem = crud.update_stem(db, stem_id, stem_update)
        return StemResponse.from_orm(updated_stem)
    except Exception as e:
        logger.error(f"Failed to update stem {stem_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update stem")


@stem_router.delete("/{stem_id}", response_model=SuccessResponse)
async def delete_stem(stem_id: int, db: Session = Depends(get_db)):
    """Stem löschen"""
    stem = crud.get_stem(db, stem_id)
    if not stem:
        raise HTTPException(status_code=404, detail="Stem not found")
    
    try:
        crud.delete_stem(db, stem_id)
        
        # Dateien löschen
        if stem.processed_path and os.path.exists(stem.processed_path):
            os.remove(stem.processed_path)
        
        return SuccessResponse(
            message="Stem deleted successfully",
            data={"stem_id": stem_id},
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to delete stem {stem_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete stem")


@stem_router.post("/search", response_model=SearchResponse)
async def search_stems(
    search_request: StemSearchRequest,
    db: Session = Depends(get_db),
    analyzer: NeuroAnalyzer = Depends(get_neuro_analyzer)
):
    """Stems durchsuchen"""
    try:
        start_time = datetime.utcnow()
        
        # Semantische Suche wenn Query vorhanden
        if search_request.query:
            results = crud.semantic_search_stems(
                db, 
                search_request.query, 
                analyzer,
                limit=search_request.limit,
                offset=search_request.offset,
                filters=search_request.dict(exclude={'query', 'limit', 'offset'})
            )
        else:
            # Normale Filtersuche
            results = crud.filter_stems(
                db,
                filters=search_request.dict(exclude={'limit', 'offset'}),
                limit=search_request.limit,
                offset=search_request.offset
            )
        
        query_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=[StemResponse.from_orm(stem) for stem in results],
            total_count=len(results),
            query_time_ms=query_time,
            search_metadata={
                "query": search_request.query,
                "filters_applied": len([v for v in search_request.dict().values() if v is not None])
            }
        )
    except Exception as e:
        logger.error(f"Failed to search stems: {e}")
        raise HTTPException(status_code=500, detail="Failed to search stems")


@stem_router.get("/{stem_id}/download")
async def download_stem(stem_id: int, db: Session = Depends(get_db)):
    """Stem-Datei herunterladen"""
    stem = crud.get_stem(db, stem_id)
    if not stem:
        raise HTTPException(status_code=404, detail="Stem not found")
    
    file_path = stem.processed_path or stem.original_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stem file not found")
    
    return FileResponse(
        path=file_path,
        filename=stem.filename,
        media_type="audio/wav"
    )


# ============================================================================
# TRACK ENDPOINTS
# ============================================================================

@track_router.get("/", response_model=PaginatedResponse)
async def list_tracks(
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db)
):
    """Alle Tracks auflisten"""
    try:
        tracks, total = crud.get_tracks_paginated(db, pagination.offset, pagination.limit)
        
        return PaginatedResponse(
            items=[TrackResponse.from_orm(track) for track in tracks],
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=max(1, (total + pagination.size - 1) // pagination.size)
        )
    except Exception as e:
        logger.error(f"Failed to list tracks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tracks")


@track_router.get("/{track_id}", response_model=TrackResponse)
async def get_track(track_id: int, db: Session = Depends(get_db)):
    """Einzelnen Track abrufen"""
    track = crud.get_track(db, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return TrackResponse.from_orm(track)


@track_router.put("/{track_id}", response_model=TrackResponse)
async def update_track(
    track_id: int,
    track_update: TrackUpdate,
    db: Session = Depends(get_db)
):
    """Track aktualisieren"""
    track = crud.get_track(db, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    try:
        updated_track = crud.update_track(db, track_id, track_update)
        return TrackResponse.from_orm(updated_track)
    except Exception as e:
        logger.error(f"Failed to update track {track_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update track")


@track_router.delete("/{track_id}", response_model=SuccessResponse)
async def delete_track(track_id: int, db: Session = Depends(get_db)):
    """Track löschen"""
    track = crud.get_track(db, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    try:
        crud.delete_track(db, track_id)
        
        # Dateien löschen
        if track.output_path and os.path.exists(track.output_path):
            os.remove(track.output_path)
        if track.preview_path and os.path.exists(track.preview_path):
            os.remove(track.preview_path)
        
        return SuccessResponse(
            message="Track deleted successfully",
            data={"track_id": track_id},
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to delete track {track_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete track")


@track_router.get("/{track_id}/download")
async def download_track(track_id: int, db: Session = Depends(get_db)):
    """Track-Datei herunterladen"""
    track = crud.get_track(db, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    if not track.output_path or not os.path.exists(track.output_path):
        raise HTTPException(status_code=404, detail="Track file not found")
    
    return FileResponse(
        path=track.output_path,
        filename=f"{track.title}.wav",
        media_type="audio/wav"
    )


@track_router.get("/{track_id}/preview")
async def get_track_preview(track_id: int, db: Session = Depends(get_db)):
    """Track-Vorschau herunterladen"""
    track = crud.get_track(db, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    if not track.preview_path or not os.path.exists(track.preview_path):
        raise HTTPException(status_code=404, detail="Track preview not found")
    
    return FileResponse(
        path=track.preview_path,
        filename=f"{track.title}_preview.wav",
        media_type="audio/wav"
    )


# ============================================================================
# GENERATION ENDPOINTS
# ============================================================================

@api_router.post("/preprocess", response_model=ProcessingJobResponse)
async def preprocess_audio(
    request: PreprocessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Audio-Dateien vorverarbeiten"""
    try:
        # Dateien validieren
        valid_files = []
        for file_path in request.file_paths:
            if os.path.exists(file_path) and file_path.lower().endswith(('.wav', '.mp3', '.flac', '.aiff')):
                valid_files.append(file_path)
            else:
                logger.warning(f"Invalid or missing file: {file_path}")
        
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid audio files found")
        
        # Processing-Job erstellen
        job_data = ProcessingJobCreate(
            job_type="preprocess",
            priority=5,
            input_data={
                "file_paths": valid_files,
                "force_reprocess": request.force_reprocess,
                "extract_features": request.extract_features,
                "generate_tags": request.generate_tags,
                "analyze_quality": request.analyze_quality
            }
        )
        
        job = crud.create_processing_job(db, job_data)
        
        # Background-Task starten
        background_tasks.add_task(run_preprocessing_job, job.id)
        
        return ProcessingJobResponse.from_orm(job)
        
    except Exception as e:
        logger.error(f"Failed to start preprocessing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start preprocessing")


@api_router.post("/generate-track", response_model=ProcessingJobResponse)
async def generate_track(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Neuen Track generieren"""
    try:
        # Track-Eintrag erstellen
        track_data = TrackCreate(
            title=request.title or f"Generated Track {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            original_prompt=request.prompt,
            target_bpm=request.target_bpm,
            target_key=request.target_key,
            target_genre=request.target_genre,
            target_mood=request.target_mood,
            target_energy=request.target_energy,
            sample_rate=request.sample_rate,
            channels=request.channels,
            tags=request.tags
        )
        
        track = crud.create_track(db, track_data)
        
        # Generation-Job erstellen
        job_data = ProcessingJobCreate(
            job_type="generate",
            priority=7,
            input_data={
                "track_id": track.id,
                "generation_request": request.dict()
            }
        )
        
        job = crud.create_processing_job(db, job_data)
        
        # Background-Task starten
        background_tasks.add_task(run_generation_job, job.id)
        
        return ProcessingJobResponse.from_orm(job)
        
    except Exception as e:
        logger.error(f"Failed to start track generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to start track generation")


# ============================================================================
# JOB ENDPOINTS
# ============================================================================

@job_router.get("/", response_model=List[ProcessingJobResponse])
async def list_jobs(
    status: Optional[str] = Query(None),
    job_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Verarbeitungsaufträge auflisten"""
    try:
        jobs = crud.get_processing_jobs(
            db, 
            status=status, 
            job_type=job_type, 
            limit=limit
        )
        return [ProcessingJobResponse.from_orm(job) for job in jobs]
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")


@job_router.get("/{job_id}", response_model=ProcessingJobResponse)
async def get_job(job_id: int, db: Session = Depends(get_db)):
    """Einzelnen Job abrufen"""
    job = crud.get_processing_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return ProcessingJobResponse.from_orm(job)


@job_router.delete("/{job_id}", response_model=SuccessResponse)
async def cancel_job(job_id: int, db: Session = Depends(get_db)):
    """Job abbrechen"""
    job = crud.get_processing_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.job_status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    try:
        crud.update_processing_job(db, job_id, {"job_status": "cancelled"})
        return SuccessResponse(
            message="Job cancelled successfully",
            data={"job_id": job_id},
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def run_preprocessing_job(job_id: int):
    """Preprocessing-Job im Hintergrund ausführen"""
    db = next(get_db())
    preprocessor = get_preprocessor_service()
    
    try:
        job = crud.get_processing_job(db, job_id)
        if not job:
            return
        
        # Job als "processing" markieren
        crud.update_processing_job(db, job_id, {
            "job_status": "processing",
            "started_at": datetime.utcnow(),
            "current_step": "Initializing preprocessing"
        })
        
        file_paths = job.input_data["file_paths"]
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            # Progress aktualisieren
            progress = (i / total_files) * 100
            crud.update_processing_job(db, job_id, {
                "progress_percentage": progress,
                "current_step": f"Processing file {i+1}/{total_files}: {os.path.basename(file_path)}"
            })
            
            # Datei verarbeiten
            try:
                await preprocessor.process_file(file_path, db)
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue
        
        # Job als abgeschlossen markieren
        crud.update_processing_job(db, job_id, {
            "job_status": "completed",
            "progress_percentage": 100.0,
            "current_step": "Preprocessing completed",
            "completed_at": datetime.utcnow(),
            "output_data": {"processed_files": total_files}
        })
        
    except Exception as e:
        logger.error(f"Preprocessing job {job_id} failed: {e}")
        crud.update_processing_job(db, job_id, {
            "job_status": "failed",
            "error_message": str(e),
            "completed_at": datetime.utcnow()
        })
    finally:
        db.close()


async def run_generation_job(job_id: int):
    """Generation-Job im Hintergrund ausführen"""
    db = next(get_db())
    arranger = get_arranger_service()
    renderer = get_renderer_service()
    
    try:
        job = crud.get_processing_job(db, job_id)
        if not job:
            return
        
        track_id = job.input_data["track_id"]
        generation_request = job.input_data["generation_request"]
        
        # Job als "processing" markieren
        crud.update_processing_job(db, job_id, {
            "job_status": "processing",
            "started_at": datetime.utcnow(),
            "current_step": "Analyzing prompt"
        })
        
        # Track-Status aktualisieren
        crud.update_track(db, track_id, {"generation_status": "analyzing"})
        
        # Schritt 1: Arrangement erstellen
        crud.update_processing_job(db, job_id, {
            "progress_percentage": 20.0,
            "current_step": "Creating arrangement plan"
        })
        crud.update_track(db, track_id, {"generation_status": "arranging"})
        
        arrangement_plan = await arranger.create_arrangement(
            generation_request["prompt"],
            db,
            **{k: v for k, v in generation_request.items() if k != "prompt"}
        )
        
        # Arrangement-Plan speichern
        crud.update_track(db, track_id, {
            "arrangement_plan": arrangement_plan.dict(),
            "track_structure": arrangement_plan.dict()["sections"]
        })
        
        # Schritt 2: Track rendern
        crud.update_processing_job(db, job_id, {
            "progress_percentage": 60.0,
            "current_step": "Rendering audio track"
        })
        crud.update_track(db, track_id, {"generation_status": "rendering"})
        
        output_path, preview_path, metadata = await renderer.render_track(
            arrangement_plan,
            track_id,
            db,
            progress_callback=lambda p: crud.update_processing_job(db, job_id, {
                "progress_percentage": 60.0 + (p * 0.4)
            })
        )
        
        # Track finalisieren
        crud.update_track(db, track_id, {
            "output_path": output_path,
            "preview_path": preview_path,
            "metadata": metadata,
            "generation_status": "completed",
            "generated_at": datetime.utcnow()
        })
        
        # Job als abgeschlossen markieren
        crud.update_processing_job(db, job_id, {
            "job_status": "completed",
            "progress_percentage": 100.0,
            "current_step": "Track generation completed",
            "completed_at": datetime.utcnow(),
            "output_data": {
                "track_id": track_id,
                "output_path": output_path,
                "preview_path": preview_path
            }
        })
        
    except Exception as e:
        logger.error(f"Generation job {job_id} failed: {e}")
        
        # Track-Status auf failed setzen
        if "track_id" in job.input_data:
            crud.update_track(db, job.input_data["track_id"], {
                "generation_status": "failed",
                "generation_error": str(e)
            })
        
        crud.update_processing_job(db, job_id, {
            "job_status": "failed",
            "error_message": str(e),
            "completed_at": datetime.utcnow()
        })
    finally:
        db.close()


async def run_cleanup_job(job_id: int):
    """Cleanup-Job im Hintergrund ausführen"""
    db = next(get_db())
    
    try:
        job = crud.get_processing_job(db, job_id)
        if not job:
            return
        
        crud.update_processing_job(db, job_id, {
            "job_status": "processing",
            "started_at": datetime.utcnow(),
            "current_step": "Starting system cleanup"
        })
        
        # Cleanup-Operationen durchführen
        # TODO: Implementieren
        
        crud.update_processing_job(db, job_id, {
            "job_status": "completed",
            "progress_percentage": 100.0,
            "current_step": "Cleanup completed",
            "completed_at": datetime.utcnow()
        })
        
    except Exception as e:
        logger.error(f"Cleanup job {job_id} failed: {e}")
        crud.update_processing_job(db, job_id, {
            "job_status": "failed",
            "error_message": str(e),
            "completed_at": datetime.utcnow()
        })
    finally:
        db.close()


# Router registrieren
api_router.include_router(stem_router)
api_router.include_router(track_router)
api_router.include_router(job_router)
api_router.include_router(system_router)