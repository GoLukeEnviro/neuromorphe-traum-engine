"""
Konfigurationseinstellungen für die Neuromorphe Traum-Engine.

Definiert verschiedene Einstellungen für die Anwendung, Datenbank, Dateispeicherung,
CLAP-Modell und API-Zugriff.
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator, ConfigDict
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Anwendungseinstellungen, geladen aus Umgebungsvariablen oder .env-Datei."""
    # App
    PROJECT_NAME: str = "Neuromorphe Traum-Engine v2.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    DEVELOPMENT_MODE: bool = False
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE_MB: int = 5
    LOG_FILE_BACKUP_COUNT: int = 3
    LOG_LEVEL: str = "INFO"
    PERFORMANCE_TRACKING: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite:///processed_database/stems.db"
    DATABASE_ECHO: bool = False
    ENABLE_DATABASE_MONITORING: bool = False
    SLOW_QUERY_THRESHOLD: float = 0.5
    
    # File Storage
    UPLOAD_DIR: str = "./raw_construction_kits"
    PROCESSED_DIR: str = "./processed_database/stems"
    EMBEDDINGS_DIR: str = "./dataembeddings"
    GENERATED_TRACKS_DIR: str = "./generated_tracks"
    STEREO_TRACKS_DIR: str = "./stereo_tracks_for_analysis"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # CLAP Model
    MODEL_CACHE_DIR: str = "./models"
    CLAP_MODEL_NAME: str = "laion/larger_clap_music_and_speech"
    
    # API
    CORS_ORIGINS: List[str] = ["http://localhost:8501"]

    @field_validator("CORS_ORIGINS", mode="before")
    def split_cors_origins(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v

    def get_logs_path(self) -> Path:
        """Gibt den Pfad zum Log-Verzeichnis zurück."""
        return Path("./logs")
    
    model_config = ConfigDict(env_file=".env", extra="allow")

settings = Settings()