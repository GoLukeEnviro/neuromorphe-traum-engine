# src/core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "Neuromorphe Traum-Engine v2.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite:///processed_database/stems.db"
    
    # File Storage
    UPLOAD_DIR: str = "./raw_construction_kits"
    PROCESSED_DIR: str = "./processed_database/stems"
    EMBEDDINGS_DIR: str = "./dataembeddings"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # CLAP Model
    MODEL_CACHE_DIR: str = "./models"
    CLAP_MODEL_NAME: str = "laion/larger_clap_music_and_speech"
    
    # API
    CORS_ORIGINS: List[str] = ["http://localhost:8501", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()