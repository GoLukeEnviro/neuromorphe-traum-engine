# FastAPI Backend - Technische Spezifikation
## Neuromorphe Traum-Engine v2.0

### Architektur-Übersicht

Das Backend folgt einer modularen Service-Architektur mit klarer Trennung von Verantwortlichkeiten:

```
backend/
├── main.py                     # FastAPI App & Router Setup
├── core/
│   ├── __init__.py
│   ├── config.py               # Konfiguration & Environment
│   ├── database.py             # SQLite/PostgreSQL Setup
│   ├── dependencies.py         # Dependency Injection
│   └── security.py             # Authentication & Authorization
├── services/
│   ├── __init__.py
│   ├── audio_processor.py      # Audio-Verarbeitung
│   ├── embedding_service.py    # CLAP Embeddings
│   ├── search_engine.py        # Semantische Suche
│   ├── file_manager.py         # Datei-Management
│   └── background_tasks.py     # Asynchrone Tasks
├── models/
│   ├── __init__.py
│   ├── database_models.py      # SQLAlchemy ORM Models
│   ├── pydantic_models.py      # Request/Response Models
│   └── enums.py                # Enumerations
├── api/
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── audio.py            # Audio-Endpunkte
│   │   ├── search.py           # Such-Endpunkte
│   │   ├── analytics.py        # Analytics-Endpunkte
│   │   └── health.py           # Health-Check Endpunkte
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py          # Audio-Hilfsfunktionen
│   ├── vector_utils.py         # Vektor-Operationen
│   ├── file_utils.py           # Datei-Operationen
│   └── logging_utils.py        # Logging-Konfiguration
└── tests/
    ├── __init__.py
    ├── test_audio.py
    ├── test_search.py
    └── test_integration.py
```

## Core Module

### config.py
```python
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # App Settings
    app_name: str = "Neuromorphe Traum-Engine API"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./data/traum_engine.db"
    
    # File Storage
    upload_dir: str = "./uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = [".wav", ".mp3", ".flac", ".m4a"]
    
    # CLAP Model
    clap_model_name: str = "laion/larger_clap_music_and_speech"
    clap_cache_dir: str = "./models"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour
    
    # Processing
    max_concurrent_jobs: int = 3
    chunk_size: int = 1024
    
    # Security
    secret_key: str = "your-secret-key-here"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### database.py
```python
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Services Module

### audio_processor.py
```python
import asyncio
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import UploadFile
from ..core.config import settings
from ..models.pydantic_models import AudioMetadata, ProcessingStatus

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 22050
        self.max_duration = 600  # 10 minutes
    
    async def validate_audio_file(self, file: UploadFile) -> bool:
        """Validiert Audio-Datei Format und Größe"""
        # Dateiendung prüfen
        if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
            return False
        
        # Dateigröße prüfen
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        if len(content) > settings.max_file_size:
            return False
        
        return True
    
    async def extract_metadata(self, file_path: Path) -> AudioMetadata:
        """Extrahiert Metadaten aus Audio-Datei"""
        try:
            # Audio laden
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Grundlegende Metadaten
            duration = len(y) / sr
            
            # Erweiterte Audio-Features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # RMS Energy
            rms = np.mean(librosa.feature.rms(y=y))
            
            return AudioMetadata(
                filename=file_path.name,
                duration=duration,
                sample_rate=sr,
                channels=1,  # Mono nach librosa.load
                tempo=float(tempo),
                spectral_centroid=float(spectral_centroid),
                zero_crossing_rate=float(zero_crossing_rate),
                rms_energy=float(rms),
                file_size=file_path.stat().st_size
            )
        except Exception as e:
            raise ValueError(f"Fehler beim Extrahieren der Metadaten: {str(e)}")
    
    async def preprocess_audio(self, file_path: Path) -> np.ndarray:
        """Bereitet Audio für CLAP-Verarbeitung vor"""
        try:
            # Audio laden und normalisieren
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Auf maximale Dauer begrenzen
            max_samples = self.max_duration * sr
            if len(y) > max_samples:
                y = y[:max_samples]
            
            # Normalisierung
            y = librosa.util.normalize(y)
            
            # Silence Trimming
            y, _ = librosa.effects.trim(y, top_db=20)
            
            return y
        except Exception as e:
            raise ValueError(f"Fehler bei der Audio-Vorverarbeitung: {str(e)}")
    
    async def save_uploaded_file(self, file: UploadFile, filename: str) -> Path:
        """Speichert hochgeladene Datei"""
        upload_path = Path(settings.upload_dir)
        upload_path.mkdir(exist_ok=True)
        
        file_path = upload_path / filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        return file_path
```

### embedding_service.py
```python
import torch
import numpy as np
from typing import List, Optional
from laion_clap import CLAP_Module
from ..core.config import settings
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model: Optional[CLAP_Module] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"EmbeddingService initialisiert mit Device: {self.device}")
    
    async def initialize_model(self):
        """Initialisiert CLAP-Modell"""
        try:
            logger.info("Lade CLAP-Modell...")
            self.model = CLAP_Module(
                enable_fusion=False,
                device=self.device,
                amodel='HTSAT-base',
                tmodel='roberta'
            )
            self.model.load_ckpt()
            logger.info("CLAP-Modell erfolgreich geladen")
        except Exception as e:
            logger.error(f"Fehler beim Laden des CLAP-Modells: {str(e)}")
            raise
    
    async def generate_audio_embedding(self, audio_data: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """Generiert Embedding für Audio-Daten"""
        if self.model is None:
            await self.initialize_model()
        
        try:
            # Audio-Embedding generieren
            audio_embed = self.model.get_audio_embedding_from_data(
                x=audio_data,
                use_tensor=False
            )
            
            return audio_embed[0]  # Erstes (und einziges) Embedding
        except Exception as e:
            logger.error(f"Fehler bei der Audio-Embedding-Generierung: {str(e)}")
            raise
    
    async def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generiert Embedding für Text"""
        if self.model is None:
            await self.initialize_model()
        
        try:
            # Text-Embedding generieren
            text_embed = self.model.get_text_embedding([text], use_tensor=False)
            
            return text_embed[0]  # Erstes (und einziges) Embedding
        except Exception as e:
            logger.error(f"Fehler bei der Text-Embedding-Generierung: {str(e)}")
            raise
    
    async def batch_generate_embeddings(self, audio_data_list: List[np.ndarray]) -> List[np.ndarray]:
        """Generiert Embeddings für mehrere Audio-Dateien"""
        if self.model is None:
            await self.initialize_model()
        
        embeddings = []
        for audio_data in audio_data_list:
            embedding = await self.generate_audio_embedding(audio_data)
            embeddings.append(embedding)
        
        return embeddings
```

### search_engine.py
```python
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from ..models.database_models import AudioFile
from ..models.pydantic_models import SearchResult, SearchFilters
from ..utils.vector_utils import normalize_vector
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class SearchEngine:
    def __init__(self):
        self.similarity_threshold = 0.1
    
    async def semantic_search(
        self,
        query_embedding: np.ndarray,
        db: Session,
        filters: Optional[SearchFilters] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """Führt semantische Suche durch"""
        try:
            # Basis-Query
            query = db.query(AudioFile)
            
            # Filter anwenden
            if filters:
                if filters.min_duration:
                    query = query.filter(AudioFile.duration >= filters.min_duration)
                if filters.max_duration:
                    query = query.filter(AudioFile.duration <= filters.max_duration)
                if filters.min_tempo:
                    query = query.filter(AudioFile.tempo >= filters.min_tempo)
                if filters.max_tempo:
                    query = query.filter(AudioFile.tempo <= filters.max_tempo)
            
            # Alle relevanten Audio-Dateien laden
            audio_files = query.all()
            
            if not audio_files:
                return []
            
            # Embeddings sammeln
            embeddings = []
            file_data = []
            
            for audio_file in audio_files:
                if audio_file.embedding:
                    # Embedding aus Datenbank laden (als numpy array)
                    embedding = np.frombuffer(audio_file.embedding, dtype=np.float32)
                    embeddings.append(embedding)
                    file_data.append(audio_file)
            
            if not embeddings:
                return []
            
            # Similarity-Berechnung
            embeddings_matrix = np.vstack(embeddings)
            query_embedding = normalize_vector(query_embedding.reshape(1, -1))
            embeddings_matrix = normalize_vector(embeddings_matrix)
            
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
            
            # Ergebnisse sortieren und filtern
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.similarity_threshold:
                    audio_file = file_data[i]
                    results.append(SearchResult(
                        id=audio_file.id,
                        filename=audio_file.filename,
                        similarity_score=float(similarity),
                        duration=audio_file.duration,
                        tempo=audio_file.tempo,
                        spectral_centroid=audio_file.spectral_centroid,
                        file_path=audio_file.file_path,
                        created_at=audio_file.created_at
                    ))
            
            # Nach Similarity sortieren
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return results[:limit]
        
        except Exception as e:
            logger.error(f"Fehler bei der semantischen Suche: {str(e)}")
            raise
    
    async def text_search(
        self,
        query: str,
        embedding_service,
        db: Session,
        filters: Optional[SearchFilters] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """Führt textbasierte semantische Suche durch"""
        try:
            # Text zu Embedding konvertieren
            query_embedding = await embedding_service.generate_text_embedding(query)
            
            # Semantische Suche durchführen
            return await self.semantic_search(query_embedding, db, filters, limit)
        
        except Exception as e:
            logger.error(f"Fehler bei der Text-Suche: {str(e)}")
            raise
    
    async def audio_search(
        self,
        audio_data: np.ndarray,
        embedding_service,
        db: Session,
        filters: Optional[SearchFilters] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """Führt audiobasierte semantische Suche durch"""
        try:
            # Audio zu Embedding konvertieren
            query_embedding = await embedding_service.generate_audio_embedding(audio_data)
            
            # Semantische Suche durchführen
            return await self.semantic_search(query_embedding, db, filters, limit)
        
        except Exception as e:
            logger.error(f"Fehler bei der Audio-Suche: {str(e)}")
            raise
```

## API Endpunkte

### api/v1/audio.py
```python
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import uuid4
import asyncio

from ...core.database import get_db
from ...core.dependencies import get_audio_processor, get_embedding_service
from ...models.pydantic_models import (
    AudioUploadResponse,
    ProcessingStatus,
    AudioFileResponse,
    AudioMetadataResponse
)
from ...services.background_tasks import process_audio_file
from ...models.database_models import AudioFile, ProcessingJob

router = APIRouter(prefix="/audio", tags=["audio"])

# In-Memory Job-Tracking (in Production: Redis)
processing_jobs = {}

@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    audio_processor = Depends(get_audio_processor),
    embedding_service = Depends(get_embedding_service)
):
    """Audio-Datei hochladen und Verarbeitung starten"""
    
    # Datei validieren
    if not await audio_processor.validate_audio_file(file):
        raise HTTPException(status_code=400, detail="Ungültige Audio-Datei")
    
    # Eindeutige Job-ID generieren
    job_id = str(uuid4())
    
    # Job-Status initialisieren
    processing_jobs[job_id] = {
        "status": "uploading",
        "progress": 0,
        "message": "Datei wird hochgeladen..."
    }
    
    # Background-Task für Verarbeitung starten
    background_tasks.add_task(
        process_audio_file,
        file,
        job_id,
        db,
        audio_processor,
        embedding_service,
        processing_jobs
    )
    
    return AudioUploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="uploading",
        message="Upload gestartet"
    )

@router.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Status der Audio-Verarbeitung abrufen"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    
    job_data = processing_jobs[job_id]
    
    return ProcessingStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        message=job_data["message"],
        audio_file_id=job_data.get("audio_file_id")
    )

@router.get("/files", response_model=List[AudioFileResponse])
async def list_audio_files(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Liste aller Audio-Dateien abrufen"""
    
    audio_files = db.query(AudioFile).offset(skip).limit(limit).all()
    
    return [
        AudioFileResponse(
            id=af.id,
            filename=af.filename,
            duration=af.duration,
            tempo=af.tempo,
            file_size=af.file_size,
            created_at=af.created_at,
            processed=af.embedding is not None
        )
        for af in audio_files
    ]

@router.get("/files/{file_id}/metadata", response_model=AudioMetadataResponse)
async def get_audio_metadata(
    file_id: int,
    db: Session = Depends(get_db)
):
    """Detaillierte Metadaten einer Audio-Datei abrufen"""
    
    audio_file = db.query(AudioFile).filter(AudioFile.id == file_id).first()
    
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio-Datei nicht gefunden")
    
    return AudioMetadataResponse(
        id=audio_file.id,
        filename=audio_file.filename,
        duration=audio_file.duration,
        sample_rate=audio_file.sample_rate,
        channels=audio_file.channels,
        tempo=audio_file.tempo,
        spectral_centroid=audio_file.spectral_centroid,
        zero_crossing_rate=audio_file.zero_crossing_rate,
        rms_energy=audio_file.rms_energy,
        file_size=audio_file.file_size,
        file_path=audio_file.file_path,
        created_at=audio_file.created_at,
        processed=audio_file.embedding is not None
    )

@router.delete("/files/{file_id}")
async def delete_audio_file(
    file_id: int,
    db: Session = Depends(get_db)
):
    """Audio-Datei löschen"""
    
    audio_file = db.query(AudioFile).filter(AudioFile.id == file_id).first()
    
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio-Datei nicht gefunden")
    
    # Datei vom Dateisystem löschen
    try:
        Path(audio_file.file_path).unlink(missing_ok=True)
    except Exception:
        pass  # Datei bereits gelöscht oder nicht vorhanden
    
    # Aus Datenbank löschen
    db.delete(audio_file)
    db.commit()
    
    return {"message": "Audio-Datei erfolgreich gelöscht"}
```

Diese Spezifikation bietet eine solide Grundlage für die FastAPI-Backend-Implementierung mit klarer Modularität, asynchroner Verarbeitung und umfassender API-Dokumentation.