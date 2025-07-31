import os
import uuid
import time
import asyncio
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# CLAP import is optional for MVP
try:
    from laion_clap import CLAP_Module
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    print("Warning: CLAP module not available, embeddings will be skipped")

from schemas import (
    AudioUploadRequest, 
    AudioProcessingResponse, 
    EmbeddingResponse, 
    AudioFileInfo,
    ProcessingStatus
)


class AudioProcessingService:
    """Service for audio processing and CLAP embedding generation"""
    
    def __init__(self, 
                 audio_dir: str = "audio_files"):
        self.audio_dir = Path(audio_dir)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._clap_model = None
        
        # Create directory if it doesn't exist
        self.audio_dir.mkdir(exist_ok=True)
    
    def _load_clap_model(self):
        """Load CLAP model for embedding generation"""
        if not CLAP_AVAILABLE:
            raise RuntimeError("CLAP module not available")
        
        if self._clap_model is None:
            self._clap_model = CLAP_Module(enable_fusion=False)
            self._clap_model.load_ckpt()
        return self._clap_model
    
    async def _generate_clap_embedding(self, audio_path: Path) -> Optional[np.ndarray]:
        """Generate CLAP embedding for audio file"""
        try:
            if not CLAP_AVAILABLE:
                return None
            
            # Load CLAP model if not already loaded
            if self._clap_model is None:
                self._clap_model = self._load_clap_model()
            
            # Generate embedding directly from file (more reliable)
            audio_embed = self._clap_model.get_audio_embedding_from_filelist(
                x=[str(audio_path)], use_tensor=False
            )
            
            return audio_embed[0] if len(audio_embed) > 0 else None
            
        except Exception as e:
            return None
    
    async def save_uploaded_file(self, 
                                file_content: bytes, 
                                request: AudioUploadRequest) -> str:
        """Save uploaded audio file and return file ID"""
        file_id = str(uuid.uuid4())
        file_extension = Path(request.filename).suffix
        file_path = self.audio_dir / f"{file_id}{file_extension}"
        
        # Save file asynchronously
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: file_path.write_bytes(file_content)
        )
        
        return file_id
    
    async def get_audio_info(self, file_id: str) -> Optional[AudioFileInfo]:
        """Get audio file information"""
        audio_files = list(self.audio_dir.glob(f"{file_id}.*"))
        if not audio_files:
            return None
        
        file_path = audio_files[0]
        loop = asyncio.get_event_loop()
        
        try:
            # Load audio info in thread pool
            audio_data, sample_rate = await loop.run_in_executor(
                self._executor,
                lambda: librosa.load(str(file_path), sr=None)
            )
            
            duration = len(audio_data) / sample_rate
            channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]
            
            return AudioFileInfo(
                id=file_id,
                filename=file_path.name,
                category=None,  # Will be set from database
                bpm=None,       # Will be set from database
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                file_size=file_path.stat().st_size,
                has_embedding=False,  # Always false in MVP
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                updated_at=None
            )
        except Exception as e:
            print(f"Error getting audio info for {file_id}: {e}")
            return None
    
    async def process_audio_file(self, file_id: str) -> AudioProcessingResponse:
        """Process audio file and generate CLAP embedding"""
        start_time = time.time()
        
        try:
            # Find audio file
            audio_files = list(self.audio_dir.glob(f"{file_id}.*"))
            if not audio_files:
                return AudioProcessingResponse(
                    id=file_id,
                    filename="unknown",
                    status=ProcessingStatus.FAILED,
                    message="Audio file not found",
                    created_at=datetime.now()
                )
            
            file_path = audio_files[0]
            
            # Generate CLAP embedding
            embedding = await self._generate_clap_embedding(file_path)
            
            processing_time = time.time() - start_time
            
            if embedding is not None:
                # Save embedding to file for later retrieval
                embedding_path = self.audio_dir / f"{file_id}_embedding.npy"
                np.save(embedding_path, embedding)
                
                message = f"Audio processed successfully in {processing_time:.2f}s with CLAP embedding ({embedding.shape[0]} dimensions)"
            else:
                message = f"Audio uploaded in {processing_time:.2f}s (CLAP embedding failed)"
            
            return AudioProcessingResponse(
                id=file_id,
                filename=file_path.name,
                status=ProcessingStatus.COMPLETED,
                message=message,
                created_at=datetime.now()
            )
            
        except Exception as e:
            return AudioProcessingResponse(
                id=file_id,
                filename="unknown",
                status=ProcessingStatus.FAILED,
                message=f"Processing failed: {str(e)}",
                created_at=datetime.now()
            )
    
    async def get_embedding(self, file_id: str) -> Optional[np.ndarray]:
        """Load CLAP embedding for file ID"""
        try:
            embedding_path = self.audio_dir / f"{file_id}_embedding.npy"
            if embedding_path.exists():
                return np.load(embedding_path)
            return None
        except Exception as e:
            print(f"Error loading embedding for {file_id}: {e}")
            return None
    
    async def list_audio_files(self) -> List[str]:
        """List all audio file IDs"""
        audio_files = []
        for file_path in self.audio_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.wav', '.mp3', '.flac', '.ogg']:
                file_id = file_path.stem
                audio_files.append(file_id)
        return audio_files
    
    def cleanup(self):
        """Cleanup resources"""
        if self._executor:
            self._executor.shutdown(wait=True)