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
from laion_clap import CLAP_Module
from concurrent.futures import ThreadPoolExecutor

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
                 audio_dir: str = "audio_files",
                 embedding_dir: str = "embeddings",
                 model_version: str = "630k-audioset-best"):
        self.audio_dir = Path(audio_dir)
        self.embedding_dir = Path(embedding_dir)
        self.model_version = model_version
        self._clap_model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Create directories if they don't exist
        self.audio_dir.mkdir(exist_ok=True)
        self.embedding_dir.mkdir(exist_ok=True)
    
    async def get_clap_model(self) -> CLAP_Module:
        """Lazy loading of CLAP model"""
        if self._clap_model is None:
            loop = asyncio.get_event_loop()
            self._clap_model = await loop.run_in_executor(
                self._executor, 
                self._load_clap_model
            )
        return self._clap_model
    
    def _load_clap_model(self) -> CLAP_Module:
        """Load CLAP model in thread pool"""
        model = CLAP_Module(enable_fusion=False)
        model.load_ckpt(self.model_version)
        return model
    
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
            
            # Check if embedding exists
            embedding_path = self.embedding_dir / f"{file_id}.npy"
            has_embedding = embedding_path.exists()
            
            return AudioFileInfo(
                id=file_id,
                filename=file_path.name,
                category=None,  # Will be set from database
                bpm=None,       # Will be set from database
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                file_size=file_path.stat().st_size,
                has_embedding=has_embedding,
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
            
            # Load CLAP model
            model = await self.get_clap_model()
            
            # Process audio in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                self._generate_embedding,
                model,
                str(file_path)
            )
            
            # Save embedding
            embedding_path = self.embedding_dir / f"{file_id}.npy"
            await loop.run_in_executor(
                self._executor,
                lambda: np.save(str(embedding_path), embedding)
            )
            
            processing_time = time.time() - start_time
            
            return AudioProcessingResponse(
                id=file_id,
                filename=file_path.name,
                status=ProcessingStatus.COMPLETED,
                message=f"Processing completed in {processing_time:.2f}s",
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
    
    def _generate_embedding(self, model: CLAP_Module, file_path: str) -> np.ndarray:
        """Generate CLAP embedding for audio file"""
        # Load and preprocess audio
        audio_data, sample_rate = librosa.load(file_path, sr=48000)
        
        # Ensure audio is the right length (CLAP expects specific duration)
        target_length = 48000 * 10  # 10 seconds at 48kHz
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        
        # Generate embedding
        audio_embed = model.get_audio_embedding_from_data(
            x=audio_data, 
            use_tensor=False
        )
        
        return audio_embed
    
    async def get_embedding(self, file_id: str) -> Optional[np.ndarray]:
        """Load embedding for file ID"""
        embedding_path = self.embedding_dir / f"{file_id}.npy"
        if not embedding_path.exists():
            return None
        
        loop = asyncio.get_event_loop()
        try:
            embedding = await loop.run_in_executor(
                self._executor,
                lambda: np.load(str(embedding_path))
            )
            return embedding
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