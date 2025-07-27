from functools import lru_cache
from .service import AudioProcessingService


@lru_cache()
def get_audio_service() -> AudioProcessingService:
    """Dependency injection for AudioProcessingService"""
    return AudioProcessingService(
        audio_dir="audio_files",
        embedding_dir="embeddings",
        model_version="630k-audioset-best"
    )