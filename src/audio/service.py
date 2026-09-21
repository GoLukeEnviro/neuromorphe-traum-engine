import os
import logging
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

from core.config import settings
from exceptions import CLAPModelError

from schemas import (
    AudioUploadRequest, 
    AudioProcessingResponse, 
    EmbeddingResponse, 
    AudioFileInfo,
    ProcessingStatus
)

logger = logging.getLogger(__name__)


def _is_usable_embedding(embedding: Optional[np.ndarray]) -> bool:
    """Ein Embedding zaehlt nur, wenn es endlich und nicht degeneriert ist.

    Null-Vektor, NaN/Inf oder ein leeres Array sind keine Embeddings - sie
    sehen fuer den Aufrufer sonst wie ein gueltiges Ergebnis aus.
    """
    if embedding is None:
        return False
    array = np.asarray(embedding, dtype=np.float64).ravel()
    if array.size == 0 or not np.all(np.isfinite(array)):
        return False
    return float(np.linalg.norm(array)) > 0.0


class AudioProcessingService:
    """Service for audio processing and CLAP embedding generation"""
    
    def __init__(self, 
                 audio_dir: str = "audio_files"):
        self.audio_dir = Path(audio_dir)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._clap_model = None
        # Realer Modus: Modellfehler und degenerierte Embeddings fail-closed.
        self.fail_closed = bool(settings.EMBEDDING_FAIL_CLOSED)
        
        # Create directory if it doesn't exist
        self.audio_dir.mkdir(exist_ok=True)
    
    def _load_clap_model(self):
        """Load CLAP model for embedding generation"""
        try:
            from laion_clap import CLAP_Module
        except ImportError as exc:
            raise CLAPModelError(
                "CLAP-Modul nicht verfuegbar (laion_clap fehlt)",
                operation="import",
            ) from exc

        if self._clap_model is None:
            try:
                self._clap_model = CLAP_Module(enable_fusion=False)
                self._clap_model.load_ckpt()
            except Exception as exc:
                raise CLAPModelError(
                    f"CLAP-Modell konnte nicht geladen werden: {exc}",
                    operation="load",
                ) from exc
        return self._clap_model
    
    async def _generate_clap_embedding(self, audio_path: Path) -> Optional[np.ndarray]:
        """Generate CLAP embedding for audio file.

        Raises:
            CLAPModelError: im realen Modus, wenn das Modell nicht laedt, die
                Inferenz fehlschlaegt oder das Ergebnis kein brauchbares
                Embedding ist (Null-Vektor, NaN/Inf, leer).
        """
        model = self._clap_model
        if model is None:
            try:
                model = self._load_clap_model()
            except Exception as exc:
                if self.fail_closed:
                    logger.error("CLAP-Modell nicht verfuegbar fuer %s: %s", audio_path, exc)
                    raise CLAPModelError(
                        f"CLAP-Modell nicht verfuegbar: {exc}",
                        operation="load",
                    ) from exc
                logger.warning("CLAP-Modell nicht verfuegbar fuer %s: %s", audio_path, exc)
                return None

        try:
            audio_embed = model.get_audio_embedding_from_filelist(
                x=[str(audio_path)], use_tensor=False
            )
        except CLAPModelError:
            raise
        except Exception as exc:
            if self.fail_closed:
                logger.error("CLAP-Embedding fehlgeschlagen fuer %s: %s", audio_path, exc)
                raise CLAPModelError(
                    f"CLAP-Embedding fehlgeschlagen: {exc}",
                    operation="inference",
                ) from exc
            logger.warning("CLAP-Embedding fehlgeschlagen fuer %s: %s", audio_path, exc)
            return None

        embedding = audio_embed[0] if len(audio_embed) > 0 else None

        if not _is_usable_embedding(embedding):
            if self.fail_closed:
                logger.error("CLAP lieferte kein brauchbares Embedding fuer %s", audio_path)
                raise CLAPModelError(
                    "CLAP lieferte ein degeneriertes Embedding (Null-Vektor oder NaN)",
                    operation="inference",
                    details={"audio_path": str(audio_path)},
                )
            logger.warning("CLAP lieferte kein brauchbares Embedding fuer %s", audio_path)
            return None

        return embedding
    
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
            elif self.fail_closed:
                # Fail-closed: kein Ergebnis ist ein Fehler, kein Erfolg.
                return AudioProcessingResponse(
                    id=file_id,
                    filename=file_path.name,
                    status=ProcessingStatus.FAILED,
                    message="CLAP embedding failed (fail-closed mode)",
                    created_at=datetime.now()
                )
            else:
                message = f"Audio uploaded in {processing_time:.2f}s (CLAP embedding failed)"
            
            return AudioProcessingResponse(
                id=file_id,
                filename=file_path.name,
                status=ProcessingStatus.COMPLETED,
                message=message,
                created_at=datetime.now()
            )
            
        except CLAPModelError as e:
            logger.error(f"CLAP-Embedding nicht moeglich fuer {file_id}: {e}")
            return AudioProcessingResponse(
                id=file_id,
                filename=file_id,
                status=ProcessingStatus.FAILED,
                message=f"CLAP embedding failed: {e}",
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
