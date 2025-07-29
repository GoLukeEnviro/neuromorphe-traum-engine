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

from .schemas import (
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
    
    # CLAP functionality removed for MVP - will be added later
    
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
        """Process audio file (simplified for MVP - no CLAP embedding)"""
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
            processing_time = time.time() - start_time
            
            return AudioProcessingResponse(
                id=file_id,
                filename=file_path.name,
                status=ProcessingStatus.COMPLETED,
                message=f"Audio uploaded successfully in {processing_time:.2f}s (CLAP processing skipped in MVP)",
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
    
    # CLAP embedding generation removed for MVP
    
    async def get_embedding(self, file_id: str) -> Optional[np.ndarray]:
        """Load embedding for file ID - not available in MVP"""
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