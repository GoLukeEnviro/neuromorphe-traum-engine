"""API-Endpunkte für die Neuromorphic Dream Engine

Diese Endpunkte implementieren den vollständigen lernenden Kreislauf:
- Preprocessing: Stem-Separation und Analyse
- Training: VAE-Modell Training
- Generation: Neue Stem-Erzeugung
- Track Generation: Vollständige Track-Komposition
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, UploadFile, File, Depends
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
import os
import tempfile
from pathlib import Path
import logging

from ...services.separation_service import SeparationService
from ...services.training_service import TrainingService
from ...services.generative_service import GenerativeService
from ...services.preprocessor import PreprocessorService
from ...services.renderer import RendererService
from ...core.logging import get_logger
from ...core.config import settings
from ...database.crud import ProcessingJobCRUD
from ...database.database import get_db
from sqlalchemy.orm import Session

logger = get_logger(__name__)
router = APIRouter()

# Dependency-Funktionen für Services
def get_separation_service() -> SeparationService:
    return SeparationService()

def get_training_service() -> TrainingService:
    return TrainingService()

def get_generative_service() -> GenerativeService:
    return GenerativeService()

def get_preprocessor_service() -> PreprocessorService:
    return PreprocessorService(config=settings)

def get_renderer_service() -> RendererService:
    return RendererService(config=settings)


# Pydantic Models für Request/Response
class PreprocessRequest(BaseModel):
    """Request für Preprocessing"""
    stereo_track_path: Optional[str] = Field(None, description="Pfad zu einem Stereo-Track für Separation")
    stem_paths: Optional[List[str]] = Field(None, description="Liste von Stem-Pfaden für direktes Processing")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "stereo_track_path": "/path/to/track.wav",
            "stem_paths": ["/path/to/stem1.wav", "/path/to/stem2.wav"]
        }
    })


class RenderRequest(BaseModel):
    """Request für Track-Rendering"""
    arrangement_plan: Dict[str, Any] = Field(..., description="Der Arrangement-Plan des Tracks")
    output_format: str = Field('wav', description="Ausgabeformat (z.B. wav, mp3)")
    quality: str = Field('high', description="Qualitätsstufe (low, medium, high)")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "arrangement_plan": {
                "bpm": 120,
                "total_bars": 32,
                "track_structure": {
                    "sections": [
                        {
                            "name": "intro",
                            "start_bar": 0,
                            "stems": [
                                {"stem_id": 1, "start_offset_bars": 0, "duration_bars": 8},
                                {"stem_id": 2, "start_offset_bars": 0, "duration_bars": 8}
                            ]
                        }
                    ]
                }
            },
            "output_format": "wav",
            "quality": "high"
        }
    })


class TrainingRequest(BaseModel):
    """Request für VAE-Training"""
    category: str = Field(..., description="Stem-Kategorie für Training")
    epochs: int = Field(100, ge=10, le=1000, description="Anzahl Trainings-Epochen")
    batch_size: int = Field(32, ge=1, le=128, description="Batch-Größe")
    learning_rate: float = Field(1e-3, gt=0, lt=1, description="Lernrate")
    latent_dim: int = Field(128, ge=32, le=512, description="Latente Dimensionalität")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "category": "kick",
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "latent_dim": 128
        }
    })


class GenerationRequest(BaseModel):
    """Request für Stem-Generierung"""
    category: str = Field(..., description="Stem-Kategorie für Generierung")
    num_variations: int = Field(5, ge=1, le=50, description="Anzahl zu generierender Variationen")
    generation_mode: str = Field("random", description="Generierungsmodus: random, interpolate, hybrid")
    interpolation_factor: float = Field(0.5, ge=0.0, le=1.0, description="Interpolationsfaktor")
    auto_process: bool = Field(True, description="Automatisches Processing der generierten Stems")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "category": "kick",
            "num_variations": 10,
            "generation_mode": "random",
            "interpolation_factor": 0.5,
            "auto_process": True
        }
    })


class HybridGenerationRequest(BaseModel):
    """Request für Hybrid-Stem-Generierung"""
    category1: str = Field(..., description="Erste Stem-Kategorie")
    category2: str = Field(..., description="Zweite Stem-Kategorie")
    num_variations: int = Field(5, ge=1, le=50, description="Anzahl zu generierender Variationen")
    blend_ratios: Optional[List[float]] = Field(None, description="Liste von Mischungsverhältnissen")
    auto_process: bool = Field(True, description="Automatisches Processing der generierten Stems")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "category1": "kick",
            "category2": "bass",
            "num_variations": 5,
            "blend_ratios": [0.2, 0.4, 0.6, 0.8],
            "auto_process": True
        }
    })


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


# Background Task Functions
async def background_training_task(job_id: int, category: str, epochs: int, batch_size: int, 
                                 learning_rate: float, latent_dim: int,
                                 training_service: TrainingService = Depends(get_training_service),
                                 db: Session = Depends(get_db)):
    """Background Task für VAE-Training"""
    try:
        ProcessingJobCRUD.update_job_status(db, job_id, "running", current_step="Starte Training")
        logger.info(f"Starte Background-Training für Kategorie: {category} (Job ID: {job_id})")
        result = await training_service.train_vae_for_category(
            category=category,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            latent_dim=latent_dim
        )
        if result['success']:
            ProcessingJobCRUD.update_job_status(db, job_id, "completed", output_data=result)
            logger.info(f"Background-Training abgeschlossen für {category}: {result['success']} (Job ID: {job_id})")
        else:
            ProcessingJobCRUD.update_job_status(db, job_id, "failed", error_message=result.get('error', 'Unbekannter Fehler'))
            logger.error(f"Background-Training fehlgeschlagen für {category}: {result.get('error', 'Unbekannter Fehler')} (Job ID: {job_id})")
    except Exception as e:
        ProcessingJobCRUD.update_job_status(db, job_id, "failed", error_message=str(e))
        logger.error(f"Fehler im Background-Training für {category}: {e} (Job ID: {job_id})")


async def background_generation_and_processing_task(job_id: int, category: str, num_variations: int, 
                                                  generation_mode: str, interpolation_factor: float,
                                                  generative_service: GenerativeService = Depends(get_generative_service),
                                                  preprocessor_service: PreprocessorService = Depends(get_preprocessor_service),
                                                  db: Session = Depends(get_db)):
    """Background Task für Generierung und automatisches Processing"""
    try:
        ProcessingJobCRUD.update_job_status(db, job_id, "running", current_step="Starte Generierung")
        logger.info(f"Starte Background-Generierung für Kategorie: {category} (Job ID: {job_id})")
        
        # Stems generieren
        generation_result = await generative_service.mutate_stems(
            category=category,
            num_variations=num_variations,
            generation_mode=generation_mode,
            interpolation_factor=interpolation_factor
        )
        
        if generation_result['success']:
            ProcessingJobCRUD.update_job_status(db, job_id, "running", current_step="Verarbeite generierte Stems")
            # Generierte Stems automatisch verarbeiten
            generated_stems = generation_result['generated_stems']
            for stem_info in generated_stems:
                try:
                    await preprocessor_service.process_audio_file(
                        file_path=stem_info['filepath'],
                        source='generated'
                    )
                    logger.info(f"Generierter Stem verarbeitet: {stem_info['filename']}")
                except Exception as e:
                    logger.warning(f"Fehler beim Verarbeiten von {stem_info['filename']}: {e}")
            ProcessingJobCRUD.update_job_status(db, job_id, "completed", output_data=generation_result)
            logger.info(f"Background-Generierung abgeschlossen für {category} (Job ID: {job_id})")
        else:
            ProcessingJobCRUD.update_job_status(db, job_id, "failed", error_message=generation_result.get('error', 'Unbekannter Fehler'))
            logger.error(f"Background-Generierung fehlgeschlagen für {category}: {generation_result.get('error', 'Unbekannter Fehler')} (Job ID: {job_id})")
    except Exception as e:
        ProcessingJobCRUD.update_job_status(db, job_id, "failed", error_message=str(e))
        logger.error(f"Fehler in Background-Generierung für {category}: {e} (Job ID: {job_id})")


# API Endpoints
@router.post("/preprocess")
async def preprocess_audio(request: PreprocessRequest, background_tasks: BackgroundTasks,
                           separation_service: SeparationService = Depends(get_separation_service),
                           preprocessor_service: PreprocessorService = Depends(get_preprocessor_service),
                           db: Session = Depends(get_db)):
    """
    Preprocessing-Endpunkt für Stem-Separation und -Analyse
    
    Dieser Endpunkt kann:
    1. Einen Stereo-Track in Stems zerlegen und diese analysieren
    2. Direkt eine Liste von Stem-Pfaden analysieren
    """
    try:
        # Job in Datenbank erstellen
        job_data = {
            "job_type": "preprocessing",
            "job_status": "pending",
            "input_data": request.dict()
        }
        job = ProcessingJobCRUD.create_job(db, job_data)

        background_tasks.add_task(
            background_preprocessing_task,
            job.id,
            request.stereo_track_path,
            request.stem_paths,
            separation_service,
            preprocessor_service,
            db
        )
        
        return {
            'success': True,
            'message': f'Preprocessing-Job gestartet (Job ID: {job.id})',
            'job_id': job.id
        }
        
    except Exception as e:
        logger.error(f"Fehler beim Starten des Preprocessing-Jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing-Job Start fehlgeschlagen: {str(e)}")


async def background_preprocessing_task(job_id: int, stereo_track_path: Optional[str], stem_paths: Optional[List[str]],
                                        separation_service: SeparationService,
                                        preprocessor_service: PreprocessorService,
                                        db: Session):
    """Background Task für Preprocessing"""
    try:
        ProcessingJobCRUD.update_job_status(db, job_id, "running", current_step="Starte Preprocessing")
        logger.info(f"Starte Background-Preprocessing (Job ID: {job_id})")
        
        results = []
        
        # Fall 1: Stereo-Track Separation
        if stereo_track_path:
            if not os.path.exists(stereo_track_path):
                raise HTTPException(status_code=404, detail=f"Stereo-Track nicht gefunden: {stereo_track_path}")
            
            ProcessingJobCRUD.update_job_status(db, job_id, "running", current_step="Separiere Track")
            logger.info(f"Starte Separation für: {stereo_track_path}")
            
            separation_result = await separation_service.separate_track(stereo_track_path)
            
            if not separation_result['success']:
                raise HTTPException(status_code=500, detail=f"Separation fehlgeschlagen: {separation_result['error']}")
            
            ProcessingJobCRUD.update_job_status(db, job_id, "running", current_step="Verarbeite separierte Stems")
            for stem_path in separation_result['stem_paths']:
                try:
                    process_result = await preprocessor_service.process_audio_file(
                        file_path=stem_path,
                        source='separated'
                    )
                    results.append({
                        'stem_path': stem_path,
                        'processing_result': process_result
                    })
                except Exception as e:
                    logger.warning(f"Fehler beim Verarbeiten von {stem_path}: {e}")
                    results.append({
                        'stem_path': stem_path,
                        'processing_result': {'success': False, 'error': str(e)}
                    })
        
        # Fall 2: Direkte Stem-Verarbeitung
        if stem_paths:
            ProcessingJobCRUD.update_job_status(db, job_id, "running", current_step="Verarbeite direkte Stems")
            for stem_path in stem_paths:
                if not os.path.exists(stem_path):
                    logger.warning(f"Stem-Pfad nicht gefunden: {stem_path}")
                    continue
                
                try:
                    process_result = await preprocessor_service.process_audio_file(
                        file_path=stem_path,
                        source='original'
                    )
                    results.append({
                        'stem_path': stem_path,
                        'processing_result': process_result
                    })
                except Exception as e:
                    logger.warning(f"Fehler beim Verarbeiten von {stem_path}: {e}")
                    results.append({
                        'stem_path': stem_path,
                        'processing_result': {'success': False, 'error': str(e)}
                    })
        
        if not results:
            raise HTTPException(status_code=400, detail="Keine gültigen Audio-Pfade zum Verarbeiten gefunden")
        
        ProcessingJobCRUD.update_job_status(db, job_id, "completed", output_data={'results': results})
        logger.info(f"Background-Preprocessing abgeschlossen (Job ID: {job_id})")
        
    except Exception as e:
        ProcessingJobCRUD.update_job_status(db, job_id, "failed", error_message=str(e))
        logger.error(f"Fehler im Background-Preprocessing (Job ID: {job_id}): {e}")


@router.post("/preprocess/upload")
async def preprocess_uploaded_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                                    preprocessor_service: PreprocessorService = Depends(get_preprocessor_service)):
    """
    Preprocessing für hochgeladene Audio-Dateien
    """
    try:
        # Temporäre Datei erstellen
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Verarbeitung im Hintergrund starten
        background_tasks.add_task(
            preprocessor_service.process_audio_file,
            file_path=temp_path,
            source='uploaded'
        )
        
        return {
            'success': True,
            'message': f'Datei {file.filename} wird im Hintergrund verarbeitet',
            'temp_path': temp_path
        }
        
    except Exception as e:
        logger.error(f"Fehler beim Upload-Preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Upload-Preprocessing fehlgeschlagen: {str(e)}")


@router.post("/train")
async def train_vae_model(request: TrainingRequest, background_tasks: BackgroundTasks,
                          training_service: TrainingService = Depends(get_training_service),
                          db: Session = Depends(get_db)):
    """
    Training-Endpunkt für VAE-Modelle
    
    Startet asynchrones Training eines VAE-Modells für eine bestimmte Kategorie.
    """
    try:
        # Job in Datenbank erstellen
        job_data = {
            "job_type": "training",
            "job_status": "pending",
            "input_data": request.dict(),
            "parameters": {
                "category": request.category,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "latent_dim": request.latent_dim
            }
        }
        job = ProcessingJobCRUD.create_job(db, job_data)
        
        # Training im Hintergrund starten
        background_tasks.add_task(
            background_training_task,
            job.id, # Job ID übergeben
            request.category,
            request.epochs,
            request.batch_size,
            request.learning_rate,
            request.latent_dim,
            training_service, # Übergabe des Service an die Background-Task
            db # Übergabe der DB-Session an die Background-Task
        )
        
        return {
            'success': True,
            'message': f'VAE-Training für Kategorie "{request.category}" gestartet (Job ID: {job.id})',
            'job_id': job.id,
            'category': request.category,
            'training_parameters': {
                'epochs': request.epochs,
                'batch_size': request.batch_size,
                'learning_rate': request.learning_rate,
                'latent_dim': request.latent_dim
            }
        }
        
    except Exception as e:
        logger.error(f"Fehler beim Starten des Trainings: {e}")
        raise HTTPException(status_code=500, detail=f"Training-Start fehlgeschlagen: {str(e)}")


@router.post("/train/batch")
async def train_all_categories(background_tasks: BackgroundTasks, min_samples: int = Query(10, ge=5, le=100),
                               training_service: TrainingService = Depends(get_training_service)):
    """
    Batch-Training für alle verfügbaren Kategorien
    """
    try:
        # Training im Hintergrund starten
        background_tasks.add_task(
            training_service.train_all_categories,
            min_samples
        )
        
        return {
            'success': True,
            'message': f'Batch-Training für alle Kategorien gestartet (min. {min_samples} Samples)',
            'min_samples': min_samples
        }
        
    except Exception as e:
        logger.error(f"Fehler beim Batch-Training: {e}")
        raise HTTPException(status_code=500, detail=f"Batch-Training fehlgeschlagen: {str(e)}")


@router.post("/generate")
async def generate_stems(request: GenerationRequest, background_tasks: BackgroundTasks,
                         generative_service: GenerativeService = Depends(get_generative_service),
                         preprocessor_service: PreprocessorService = Depends(get_preprocessor_service),
                         db: Session = Depends(get_db)):
    """
    Generierungs-Endpunkt für neue Stems
    
    Generiert neue Stems mit einem trainierten VAE-Modell und verarbeitet sie optional automatisch.
    """
    try:
        if request.auto_process:
            # Job in Datenbank erstellen
            job_data = {
                "job_type": "generation",
                "job_status": "pending",
                "input_data": request.dict()
            }
            job = ProcessingJobCRUD.create_job(db, job_data)

            # Generierung und Processing im Hintergrund
            background_tasks.add_task(
                background_generation_and_processing_task,
                job.id,
                request.category,
                request.num_variations,
                request.generation_mode,
                request.interpolation_factor,
                generative_service, # Übergabe des Service an die Background-Task
                preprocessor_service, # Übergabe des Service an die Background-Task
                db
            )
            
            return {
                'success': True,
                'message': f'{request.num_variations} Stems für "{request.category}" werden generiert und verarbeitet (Job ID: {job.id})',
                'job_id': job.id,
                'category': request.category,
                'num_variations': request.num_variations,
                'generation_mode': request.generation_mode,
                'auto_processing': True
            }
        else:
            # Nur Generierung, synchron
            result = await generative_service.mutate_stems(
                category=request.category,
                num_variations=request.num_variations,
                generation_mode=request.generation_mode,
                interpolation_factor=request.interpolation_factor
            )
            
            return result
        
    except Exception as e:
        logger.error(f"Fehler bei Stem-Generierung: {e}")
        raise HTTPException(status_code=500, detail=f"Stem-Generierung fehlgeschlagen: {str(e)}")


@router.post("/generate/hybrid")
async def generate_hybrid_stems(request: HybridGenerationRequest, background_tasks: BackgroundTasks,
                                generative_service: GenerativeService = Depends(get_generative_service),
                                preprocessor_service: PreprocessorService = Depends(get_preprocessor_service)):
    """
    Generierung von Hybrid-Stems durch Interpolation zwischen zwei Kategorien
    """
    try:
        result = await generative_service.generate_hybrid_stems(
            category1=request.category1,
            category2=request.category2,
            num_variations=request.num_variations,
            blend_ratios=request.blend_ratios
        )
        
        if result['success'] and request.auto_process:
            # Generierte Hybrid-Stems automatisch verarbeiten
            generated_stems = result['generated_stems']
            for stem_info in generated_stems:
                background_tasks.add_task(
                    preprocessor_service.process_audio_file,
                    stem_info['filepath'],
                    'generated'
                )
        
        return result
        
    except Exception as e:
        logger.error(f"Fehler bei Hybrid-Generierung: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid-Generierung fehlgeschlagen: {str(e)}")


@router.post("/generate/batch")
async def batch_generate_all_categories(background_tasks: BackgroundTasks, 
                                      num_variations: int = Query(5, ge=1, le=20),
                                      generative_service: GenerativeService = Depends(get_generative_service)):
    """
    Batch-Generierung für alle verfügbaren trainierten Kategorien
    """
    try:
        result = await generative_service.batch_generate_all_categories(num_variations)
        
        return {
            'success': True,
            'message': f'Batch-Generierung abgeschlossen ({num_variations} Variationen pro Kategorie)',
            'results': result
        }
        
    except Exception as e:
        logger.error(f"Fehler bei Batch-Generierung: {e}")
        raise HTTPException(status_code=500, detail=f"Batch-Generierung fehlgeschlagen: {str(e)}")


@router.post("/render")
async def render_track(request: RenderRequest, background_tasks: BackgroundTasks,
                       renderer_service: RendererService = Depends(get_renderer_service),
                       db: Session = Depends(get_db)):
    """
    Rendering-Endpunkt für Tracks
    
    Startet asynchrones Rendering eines Tracks basierend auf einem Arrangement-Plan.
    """
    try:
        # Job in Datenbank erstellen
        job_data = {
            "job_type": "rendering",
            "job_status": "pending",
            "input_data": request.dict()
        }
        job = ProcessingJobCRUD.create_job(db, job_data)

        background_tasks.add_task(
            background_rendering_task,
            job.id,
            request.arrangement_plan,
            request.output_format,
            request.quality,
            renderer_service,
            db
        )
        
        return {
            'success': True,
            'message': f'Track-Rendering-Job gestartet (Job ID: {job.id})',
            'job_id': job.id
        }
        
    except Exception as e:
        logger.error(f"Fehler beim Starten des Rendering-Jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Rendering-Job Start fehlgeschlagen: {str(e)}")


async def background_rendering_task(job_id: int, arrangement_plan: Dict[str, Any],
                                    output_format: str, quality: str,
                                    renderer_service: RendererService,
                                    db: Session):
    """Background Task für Track-Rendering"""
    try:
        ProcessingJobCRUD.update_job_status(db, job_id, "running", current_step="Starte Rendering")
        logger.info(f"Starte Background-Rendering (Job ID: {job_id})")
        
        render_result = await renderer_service.render_track(
            arrangement=arrangement_plan,
            output_format=output_format,
            quality=quality
        )
        
        if render_result['success']:
            ProcessingJobCRUD.update_job_status(db, job_id, "completed", output_data=render_result)
            logger.info(f"Background-Rendering abgeschlossen (Job ID: {job_id}): {render_result['output_path']}")
        else:
            ProcessingJobCRUD.update_job_status(db, job_id, "failed", error_message=render_result.get('error', 'Unbekannter Fehler'))
            logger.error(f"Background-Rendering fehlgeschlagen (Job ID: {job_id}): {render_result.get('error', 'Unbekannter Fehler')}")
            
    except Exception as e:
        ProcessingJobCRUD.update_job_status(db, job_id, "failed", error_message=str(e))
        logger.error(f"Fehler im Background-Rendering (Job ID: {job_id}): {e}")


# Status und Info Endpoints
@router.get("/models/available")
async def get_available_models(training_service: TrainingService = Depends(get_training_service)):
    """
    Gibt Liste aller verfügbaren trainierten VAE-Modelle zurück
    """
    try:
        models = training_service.get_available_models()
        return {
            'success': True,
            'models': models,
            'count': len(models)
        }
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Modelle: {e}")
        raise HTTPException(status_code=500, detail=f"Modell-Abfrage fehlgeschlagen: {str(e)}")


@router.get("/generated/info")
async def get_generated_stems_info(generative_service: GenerativeService = Depends(get_generative_service)):
    """
    Gibt Informationen über alle generierten Stems zurück
    """
    try:
        stems_info = generative_service.get_generated_stems_info()
        return {
            'success': True,
            'generated_stems': stems_info,
            'count': len(stems_info)
        }
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der generierten Stems: {e}")
        raise HTTPException(status_code=500, detail=f"Generierte Stems Abfrage fehlgeschlagen: {str(e)}")


@router.post("/cache/clear")
async def clear_model_cache(generative_service: GenerativeService = Depends(get_generative_service)):
    """
    Leert den Modell-Cache
    """
    try:
        generative_service.clear_cache()
        return {
            'success': True,
            'message': 'Modell-Cache geleert'
        }
    except Exception as e:
        logger.error(f"Fehler beim Leeren des Caches: {e}")
        raise HTTPException(status_code=500, detail=f"Cache-Leerung fehlgeschlagen: {str(e)}")