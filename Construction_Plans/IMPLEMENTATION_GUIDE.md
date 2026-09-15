# Implementierungsanleitung - Neuromorphe Traum-Engine v2.0
## Schritt-für-Schritt Transformation zur Service-Architektur

### Übersicht

Diese Anleitung führt Sie durch die vollständige Transformation der bestehenden Neuromorphen Traum-Engine in eine moderne, serviceorientierte Architektur mit FastAPI-Backend und Streamlit-Frontend.

## Sprint-Planung

### Sprint 1: Backend-Grundlagen (Woche 1-2)
**Ziel:** Funktionsfähiges FastAPI-Backend mit Core-Services

#### Tag 1-2: Projektstruktur & Konfiguration
```bash
# 1. Neue Projektstruktur erstellen
mkdir -p backend/{core,services,api,models,utils}
mkdir -p frontend/{pages,components,utils,config}
mkdir -p data/{uploads,processed,embeddings}
mkdir -p models/clap
mkdir -p logs

# 2. Backend-Abhängigkeiten definieren
```

**backend/requirements.txt erstellen:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
sqlalchemy==2.0.23
aiosqlite==0.19.0
librosa==0.10.1
soundfile==0.12.1
numpy==1.24.4
laion-clap==1.1.4
torch==2.1.1
torchaudio==2.1.1
scikit-learn==1.3.2
redis==5.0.1
pydantic==2.5.0
pydantic-settings==2.0.3
aiofiles==23.2.1
requests==2.31.0
python-dotenv==1.0.0
```

#### Tag 3-4: Core-Module implementieren

**1. Konfiguration (backend/core/config.py):**
```python
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # App
    app_name: str = "Neuromorphe Traum-Engine"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./data/traum_engine.db"
    
    # File Storage
    upload_dir: str = "./data/uploads"
    processed_dir: str = "./data/processed"
    embeddings_dir: str = "./data/embeddings"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    # CLAP Model
    model_cache_dir: str = "./models"
    clap_model_name: str = "laion/larger_clap_music_and_speech"
    
    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:8501"]
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**2. Datenbank-Setup (backend/core/database.py):**
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

**3. Datenmodelle (backend/models/database_models.py):**
```python
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.sql import func
from core.database import Base

class AudioFile(Base):
    __tablename__ = "audio_files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    original_filename = Column(String)
    file_path = Column(String)
    file_size = Column(Integer)
    duration = Column(Float)
    sample_rate = Column(Integer)
    channels = Column(Integer)
    format = Column(String)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String, default="pending")
    error_message = Column(Text, nullable=True)
    
    # Embeddings
    embedding_path = Column(String, nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
class SearchQuery(Base):
    __tablename__ = "search_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)
    query_type = Column(String)  # "text" or "audio"
    results_count = Column(Integer)
    execution_time = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

#### Tag 5-7: Audio-Processing Service

**backend/services/audio_processor.py:**
```python
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
import aiofiles
from core.config import settings
from models.database_models import AudioFile
from sqlalchemy.orm import Session

class AudioProcessor:
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.processed_dir = Path(settings.processed_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to disk"""
        file_path = self.upload_dir / filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        return str(file_path)
    
    async def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio file and extract metadata"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            
            # Extract basic information
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get file info
            info = sf.info(file_path)
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": info.channels,
                "format": info.format,
                "file_size": Path(file_path).stat().st_size,
                "frames": len(y)
            }
        except Exception as e:
            raise Exception(f"Error analyzing audio file: {str(e)}")
    
    async def preprocess_audio(self, file_path: str, target_sr: int = 48000) -> str:
        """Preprocess audio file for embedding generation"""
        try:
            # Load and resample
            y, sr = librosa.load(file_path, sr=target_sr)
            
            # Normalize
            y = librosa.util.normalize(y)
            
            # Save processed file
            processed_filename = f"processed_{Path(file_path).stem}.wav"
            processed_path = self.processed_dir / processed_filename
            
            sf.write(processed_path, y, target_sr)
            
            return str(processed_path)
        except Exception as e:
            raise Exception(f"Error preprocessing audio: {str(e)}")
    
    async def process_audio_file(
        self, 
        file_path: str, 
        db: Session, 
        audio_record: AudioFile
    ) -> Dict[str, Any]:
        """Complete audio processing pipeline"""
        try:
            # Update status
            audio_record.processing_status = "analyzing"
            db.commit()
            
            # Analyze file
            analysis = await self.analyze_audio_file(file_path)
            
            # Update database with analysis results
            audio_record.duration = analysis["duration"]
            audio_record.sample_rate = analysis["sample_rate"]
            audio_record.channels = analysis["channels"]
            audio_record.format = analysis["format"]
            audio_record.file_size = analysis["file_size"]
            
            # Preprocess for embedding
            audio_record.processing_status = "preprocessing"
            db.commit()
            
            processed_path = await self.preprocess_audio(file_path)
            
            # Mark as processed
            audio_record.is_processed = True
            audio_record.processing_status = "completed"
            db.commit()
            
            return {
                "status": "success",
                "processed_path": processed_path,
                "analysis": analysis
            }
            
        except Exception as e:
            audio_record.processing_status = "failed"
            audio_record.error_message = str(e)
            db.commit()
            raise e
```

#### Tag 8-10: Embedding Service

**backend/services/embedding_service.py:**
```python
import laion_clap
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import pickle
from core.config import settings
from models.database_models import AudioFile
from sqlalchemy.orm import Session

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings_dir = Path(settings.embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir = Path(settings.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize_model(self):
        """Initialize CLAP model"""
        if self.model is None:
            print("Loading CLAP model...")
            
            # Set cache directory
            import os
            os.environ['TORCH_HOME'] = str(self.model_cache_dir)
            os.environ['HF_HOME'] = str(self.model_cache_dir)
            
            # Load model
            self.model = laion_clap.CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()  # Load default checkpoint
            
            print(f"CLAP model loaded on {self.device}")
    
    async def generate_audio_embedding(self, audio_path: str) -> np.ndarray:
        """Generate embedding for audio file"""
        if self.model is None:
            await self.initialize_model()
        
        try:
            # Generate embedding
            audio_embed = self.model.get_audio_embedding_from_filelist(
                x=[audio_path], 
                use_tensor=False
            )
            
            return audio_embed[0]  # Return first (and only) embedding
            
        except Exception as e:
            raise Exception(f"Error generating audio embedding: {str(e)}")
    
    async def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text query"""
        if self.model is None:
            await self.initialize_model()
        
        try:
            # Generate embedding
            text_embed = self.model.get_text_embedding([text], use_tensor=False)
            
            return text_embed[0]  # Return first (and only) embedding
            
        except Exception as e:
            raise Exception(f"Error generating text embedding: {str(e)}")
    
    async def save_embedding(self, embedding: np.ndarray, filename: str) -> str:
        """Save embedding to disk"""
        embedding_path = self.embeddings_dir / f"{filename}.pkl"
        
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding, f)
        
        return str(embedding_path)
    
    async def load_embedding(self, embedding_path: str) -> np.ndarray:
        """Load embedding from disk"""
        with open(embedding_path, 'rb') as f:
            return pickle.load(f)
    
    async def process_audio_embedding(
        self, 
        audio_path: str, 
        db: Session, 
        audio_record: AudioFile
    ) -> str:
        """Complete embedding generation pipeline for audio"""
        try:
            # Generate embedding
            embedding = await self.generate_audio_embedding(audio_path)
            
            # Save embedding
            embedding_filename = f"audio_{audio_record.id}"
            embedding_path = await self.save_embedding(embedding, embedding_filename)
            
            # Update database
            audio_record.embedding_path = embedding_path
            audio_record.embedding_dimension = len(embedding)
            db.commit()
            
            return embedding_path
            
        except Exception as e:
            raise Exception(f"Error processing audio embedding: {str(e)}")
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
```

#### Tag 11-14: Search Engine Service

**backend/services/search_engine.py:**
```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
from models.database_models import AudioFile, SearchQuery
from .embedding_service import EmbeddingService
import time

class SearchEngine:
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    async def search_by_text(
        self, 
        query: str, 
        db: Session, 
        limit: int = 10,
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Search audio files by text query"""
        start_time = time.time()
        
        try:
            # Generate text embedding
            query_embedding = await self.embedding_service.generate_text_embedding(query)
            
            # Get all processed audio files
            audio_files = db.query(AudioFile).filter(
                AudioFile.is_processed == True,
                AudioFile.embedding_path.isnot(None)
            ).all()
            
            # Calculate similarities
            similarities = []
            for audio_file in audio_files:
                try:
                    # Load audio embedding
                    audio_embedding = await self.embedding_service.load_embedding(
                        audio_file.embedding_path
                    )
                    
                    # Calculate similarity
                    similarity = self.embedding_service.calculate_similarity(
                        query_embedding, audio_embedding
                    )
                    
                    if similarity >= threshold:
                        similarities.append({
                            "audio_file": audio_file,
                            "similarity": similarity
                        })
                        
                except Exception as e:
                    print(f"Error processing file {audio_file.filename}: {e}")
                    continue
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limit results
            similarities = similarities[:limit]
            
            # Format results
            results = []
            for item in similarities:
                audio_file = item["audio_file"]
                results.append({
                    "id": audio_file.id,
                    "filename": audio_file.original_filename,
                    "similarity": item["similarity"],
                    "duration": audio_file.duration,
                    "file_size": audio_file.file_size,
                    "created_at": audio_file.created_at.isoformat(),
                    "file_path": audio_file.file_path
                })
            
            # Log search query
            execution_time = time.time() - start_time
            search_query = SearchQuery(
                query_text=query,
                query_type="text",
                results_count=len(results),
                execution_time=execution_time
            )
            db.add(search_query)
            db.commit()
            
            return results
            
        except Exception as e:
            raise Exception(f"Error in text search: {str(e)}")
    
    async def search_by_audio(
        self, 
        audio_path: str, 
        db: Session, 
        limit: int = 10,
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Search audio files by audio query"""
        start_time = time.time()
        
        try:
            # Generate audio embedding for query
            query_embedding = await self.embedding_service.generate_audio_embedding(audio_path)
            
            # Get all processed audio files
            audio_files = db.query(AudioFile).filter(
                AudioFile.is_processed == True,
                AudioFile.embedding_path.isnot(None)
            ).all()
            
            # Calculate similarities
            similarities = []
            for audio_file in audio_files:
                try:
                    # Skip if it's the same file
                    if audio_file.file_path == audio_path:
                        continue
                    
                    # Load audio embedding
                    audio_embedding = await self.embedding_service.load_embedding(
                        audio_file.embedding_path
                    )
                    
                    # Calculate similarity
                    similarity = self.embedding_service.calculate_similarity(
                        query_embedding, audio_embedding
                    )
                    
                    if similarity >= threshold:
                        similarities.append({
                            "audio_file": audio_file,
                            "similarity": similarity
                        })
                        
                except Exception as e:
                    print(f"Error processing file {audio_file.filename}: {e}")
                    continue
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limit results
            similarities = similarities[:limit]
            
            # Format results
            results = []
            for item in similarities:
                audio_file = item["audio_file"]
                results.append({
                    "id": audio_file.id,
                    "filename": audio_file.original_filename,
                    "similarity": item["similarity"],
                    "duration": audio_file.duration,
                    "file_size": audio_file.file_size,
                    "created_at": audio_file.created_at.isoformat(),
                    "file_path": audio_file.file_path
                })
            
            # Log search query
            execution_time = time.time() - start_time
            search_query = SearchQuery(
                query_text=f"Audio query: {audio_path}",
                query_type="audio",
                results_count=len(results),
                execution_time=execution_time
            )
            db.add(search_query)
            db.commit()
            
            return results
            
        except Exception as e:
            raise Exception(f"Error in audio search: {str(e)}")
    
    async def get_similar_files(
        self, 
        file_id: int, 
        db: Session, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get files similar to a specific file"""
        # Get the target file
        target_file = db.query(AudioFile).filter(AudioFile.id == file_id).first()
        if not target_file or not target_file.embedding_path:
            return []
        
        # Use the file's path for audio search
        return await self.search_by_audio(target_file.file_path, db, limit)
    
    def get_search_statistics(self, db: Session) -> Dict[str, Any]:
        """Get search statistics"""
        total_queries = db.query(SearchQuery).count()
        text_queries = db.query(SearchQuery).filter(SearchQuery.query_type == "text").count()
        audio_queries = db.query(SearchQuery).filter(SearchQuery.query_type == "audio").count()
        
        # Average execution time
        avg_execution_time = db.query(SearchQuery).with_entities(
            SearchQuery.execution_time
        ).all()
        
        if avg_execution_time:
            avg_time = sum(t[0] for t in avg_execution_time) / len(avg_execution_time)
        else:
            avg_time = 0
        
        return {
            "total_queries": total_queries,
            "text_queries": text_queries,
            "audio_queries": audio_queries,
            "average_execution_time": avg_time
        }
```

### Sprint 2: API-Endpunkte (Woche 3)
**Ziel:** Vollständige REST API mit allen Endpunkten

#### Tag 15-17: FastAPI Hauptanwendung

**backend/main.py:**
```python
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
import os
from pathlib import Path

# Core imports
from core.config import settings
from core.database import engine, get_db, Base
from models.database_models import AudioFile, SearchQuery

# Services
from services.audio_processor import AudioProcessor
from services.embedding_service import EmbeddingService
from services.search_engine import SearchEngine

# API routes
from api.v1 import audio, search, files

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Neuromorphe Traum-Engine - Audio Similarity Search API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(audio.router, prefix=settings.api_v1_prefix)
app.include_router(search.router, prefix=settings.api_v1_prefix)
app.include_router(files.router, prefix=settings.api_v1_prefix)

# Initialize services
audio_processor = AudioProcessor()
embedding_service = EmbeddingService()
search_engine = SearchEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting Neuromorphe Traum-Engine...")
    
    # Create directories
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)
    os.makedirs(settings.embeddings_dir, exist_ok=True)
    os.makedirs(settings.model_cache_dir, exist_ok=True)
    
    # Initialize CLAP model
    print("🧠 Initializing CLAP model...")
    await embedding_service.initialize_model()
    print("✅ CLAP model ready!")
    
    print("🎉 Neuromorphe Traum-Engine started successfully!")

@app.get("/")
async def root():
    return {
        "message": "Neuromorphe Traum-Engine API",
        "version": settings.app_version,
        "docs": "/docs"
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.app_version,
        "model_loaded": embedding_service.model is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
```

#### Tag 18-21: API-Router implementieren

**backend/api/v1/audio.py:**
```python
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import uuid
from pathlib import Path

from core.database import get_db
from core.config import settings
from models.database_models import AudioFile
from services.audio_processor import AudioProcessor
from services.embedding_service import EmbeddingService

router = APIRouter(prefix="/audio", tags=["audio"])
audio_processor = AudioProcessor()
embedding_service = EmbeddingService()

async def process_audio_background(
    file_path: str, 
    audio_id: int,
    db: Session
):
    """Background task for audio processing"""
    try:
        # Get audio record
        audio_record = db.query(AudioFile).filter(AudioFile.id == audio_id).first()
        if not audio_record:
            return
        
        # Process audio file
        result = await audio_processor.process_audio_file(file_path, db, audio_record)
        
        # Generate embedding
        if result["status"] == "success":
            await embedding_service.process_audio_embedding(
                result["processed_path"], db, audio_record
            )
            
    except Exception as e:
        print(f"Background processing error: {e}")
        audio_record = db.query(AudioFile).filter(AudioFile.id == audio_id).first()
        if audio_record:
            audio_record.processing_status = "failed"
            audio_record.error_message = str(e)
            db.commit()

@router.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process audio file"""
    
    # Validate file type
    allowed_types = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed: {allowed_types}"
        )
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
        )
    
    try:
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Save file
        file_path = await audio_processor.save_uploaded_file(file_content, unique_filename)
        
        # Create database record
        audio_record = AudioFile(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_path,
            processing_status="pending"
        )
        
        db.add(audio_record)
        db.commit()
        db.refresh(audio_record)
        
        # Start background processing
        background_tasks.add_task(
            process_audio_background,
            file_path,
            audio_record.id,
            db
        )
        
        return {
            "id": audio_record.id,
            "filename": audio_record.original_filename,
            "status": "uploaded",
            "message": "File uploaded successfully. Processing started."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/status/{audio_id}")
async def get_audio_status(
    audio_id: int,
    db: Session = Depends(get_db)
):
    """Get processing status of audio file"""
    
    audio_file = db.query(AudioFile).filter(AudioFile.id == audio_id).first()
    
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return {
        "id": audio_file.id,
        "filename": audio_file.original_filename,
        "status": audio_file.processing_status,
        "is_processed": audio_file.is_processed,
        "error_message": audio_file.error_message,
        "duration": audio_file.duration,
        "created_at": audio_file.created_at
    }

@router.get("/list")
async def list_audio_files(
    skip: int = 0,
    limit: int = 100,
    status: str = None,
    db: Session = Depends(get_db)
):
    """List audio files with optional filtering"""
    
    query = db.query(AudioFile)
    
    if status:
        query = query.filter(AudioFile.processing_status == status)
    
    audio_files = query.offset(skip).limit(limit).all()
    
    return {
        "files": [
            {
                "id": af.id,
                "filename": af.original_filename,
                "status": af.processing_status,
                "is_processed": af.is_processed,
                "duration": af.duration,
                "file_size": af.file_size,
                "created_at": af.created_at
            }
            for af in audio_files
        ],
        "total": query.count()
    }

@router.delete("/delete/{audio_id}")
async def delete_audio_file(
    audio_id: int,
    db: Session = Depends(get_db)
):
    """Delete audio file and associated data"""
    
    audio_file = db.query(AudioFile).filter(AudioFile.id == audio_id).first()
    
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    try:
        # Delete files from disk
        if Path(audio_file.file_path).exists():
            Path(audio_file.file_path).unlink()
        
        if audio_file.embedding_path and Path(audio_file.embedding_path).exists():
            Path(audio_file.embedding_path).unlink()
        
        # Delete from database
        db.delete(audio_file)
        db.commit()
        
        return {"message": "Audio file deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
```

**backend/api/v1/search.py:**
```python
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from core.database import get_db
from services.search_engine import SearchEngine
from services.audio_processor import AudioProcessor

router = APIRouter(prefix="/search", tags=["search"])
search_engine = SearchEngine()
audio_processor = AudioProcessor()

class TextSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    threshold: Optional[float] = 0.1

@router.post("/text")
async def search_by_text(
    request: TextSearchRequest,
    db: Session = Depends(get_db)
):
    """Search audio files by text description"""
    
    try:
        results = await search_engine.search_by_text(
            query=request.query,
            db=db,
            limit=request.limit,
            threshold=request.threshold
        )
        
        return {
            "query": request.query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/audio")
async def search_by_audio(
    file: UploadFile = File(...),
    limit: int = 10,
    threshold: float = 0.1,
    db: Session = Depends(get_db)
):
    """Search audio files by audio similarity"""
    
    # Validate file type
    allowed_types = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported"
        )
    
    try:
        # Save temporary file
        file_content = await file.read()
        temp_filename = f"temp_query_{uuid.uuid4()}{file_extension}"
        temp_path = await audio_processor.save_uploaded_file(file_content, temp_filename)
        
        # Perform search
        results = await search_engine.search_by_audio(
            audio_path=temp_path,
            db=db,
            limit=limit,
            threshold=threshold
        )
        
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)
        
        return {
            "query_filename": file.filename,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Audio search failed: {str(e)}")

@router.get("/similar/{file_id}")
async def get_similar_files(
    file_id: int,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get files similar to a specific file"""
    
    try:
        results = await search_engine.get_similar_files(
            file_id=file_id,
            db=db,
            limit=limit
        )
        
        return {
            "file_id": file_id,
            "similar_files": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")

@router.get("/statistics")
async def get_search_statistics(
    db: Session = Depends(get_db)
):
    """Get search statistics"""
    
    try:
        stats = search_engine.get_search_statistics(db)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")
```

### Sprint 3: Frontend-Entwicklung (Woche 4-5)
**Ziel:** Vollständiges Streamlit-Frontend

#### Tag 22-28: Streamlit-Anwendung

**frontend/requirements.txt:**
```txt
streamlit==1.28.1
streamlit-option-menu==0.3.6
streamlit-aggrid==0.3.4.post3
requests==2.31.0
pandas==2.1.3
plotly==5.17.0
numpy==1.24.4
pillow==10.1.0
python-dotenv==1.0.0
```

**frontend/streamlit_app.py:**
```python
import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from config.settings import load_settings
from pages import upload, search, results, analytics, settings_page
from utils.api_client import APIClient
from utils.theme import apply_custom_theme

# Page configuration
st.set_page_config(
    page_title="Neuromorphe Traum-Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load settings
settings = load_settings()

# Apply custom theme
apply_custom_theme()

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient(settings.backend_url)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Upload"

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Main navigation
with st.sidebar:
    st.title("🧠 Neuromorphe Traum-Engine")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Upload", "Search", "Results", "Analytics", "Settings"],
        icons=["cloud-upload", "search", "list-ul", "bar-chart", "gear"],
        menu_icon="brain",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0e1117"},
            "icon": {"color": "#ff6b6b", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#262730",
            },
            "nav-link-selected": {"background-color": "#ff6b6b"},
        },
    )
    
    st.session_state.current_page = selected
    
    # Backend status
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    try:
        health = st.session_state.api_client.get_health()
        if health.get("status") == "healthy":
            st.success("✅ Backend Online")
            if health.get("model_loaded"):
                st.success("🧠 CLAP Model Ready")
            else:
                st.warning("⏳ Model Loading...")
        else:
            st.error("❌ Backend Offline")
    except:
        st.error("❌ Backend Offline")

# Main content area
if st.session_state.current_page == "Upload":
    upload.show_upload_page()
elif st.session_state.current_page == "Search":
    search.show_search_page()
elif st.session_state.current_page == "Results":
    results.show_results_page()
elif st.session_state.current_page == "Analytics":
    analytics.show_analytics_page()
elif st.session_state.current_page == "Settings":
    settings_page.show_settings_page()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Neuromorphe Traum-Engine v2.0 | Powered by CLAP & FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True
)
```

### Sprint 4: Docker & Deployment (Woche 6)
**Ziel:** Vollständige Containerisierung und Deployment

#### Tag 29-35: Docker-Setup

**Dockerfile.backend:**
```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# Create directories
RUN mkdir -p data uploads models logs

# Download CLAP model
RUN python -c "import laion_clap; model = laion_clap.CLAP_Module(); model.load_ckpt()"

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
    environment:
      - DATABASE_URL=sqlite:///data/traum_engine.db
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  data:
  uploads:
```

## Erfolgskriterien

### Sprint 1 Abnahme:
- [ ] FastAPI-Backend läuft auf Port 8000
- [ ] CLAP-Model wird erfolgreich geladen
- [ ] Audio-Upload und -Verarbeitung funktioniert
- [ ] Embedding-Generierung funktioniert
- [ ] Datenbank-Integration funktioniert

### Sprint 2 Abnahme:
- [ ] Alle API-Endpunkte sind implementiert
- [ ] Text-Suche funktioniert
- [ ] Audio-Suche funktioniert
- [ ] API-Dokumentation ist verfügbar (/docs)
- [ ] Error-Handling ist implementiert

### Sprint 3 Abnahme:
- [ ] Streamlit-Frontend läuft auf Port 8501
- [ ] Upload-Interface funktioniert
- [ ] Such-Interface funktioniert
- [ ] Ergebnisse werden korrekt angezeigt
- [ ] Navigation zwischen Seiten funktioniert

### Sprint 4 Abnahme:
- [ ] Docker-Container bauen erfolgreich
- [ ] docker-compose startet alle Services
- [ ] Frontend und Backend kommunizieren
- [ ] Persistente Datenspeicherung funktioniert
- [ ] Deployment-Dokumentation ist vollständig

## Nächste Schritte

1. **Sofort beginnen:** Sprint 1, Tag 1-2 (Projektstruktur)
2. **Woche 1:** Backend-Grundlagen implementieren
3. **Woche 2:** Services vervollständigen
4. **Woche 3:** API-Endpunkte entwickeln
5. **Woche 4-5:** Frontend-Entwicklung
6. **Woche 6:** Docker & Deployment

## Hilfreiche Befehle

```bash
# Backend starten (Development)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend starten (Development)
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py

# Docker Deployment
docker-compose up --build

# Tests ausführen
pytest backend/tests/

# Code-Qualität prüfen
black backend/
isort backend/
flake8 backend/
```

Diese Implementierungsanleitung führt Sie systematisch durch die Transformation der Neuromorphen Traum-Engine in eine moderne, serviceorientierte Architektur.