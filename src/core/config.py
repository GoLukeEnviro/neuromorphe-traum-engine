"""
Konfigurationseinstellungen für die Neuromorphe Traum-Engine.

Definiert verschiedene Einstellungen für die Anwendung, Datenbank, Dateispeicherung,
CLAP-Modell und API-Zugriff.
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator, ConfigDict
from typing import Dict, List
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
    LOGS_DIR: str = "./logs"
    PERFORMANCE_TRACKING: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite:///processed_database/stems.db"
    DATABASE_ECHO: bool = False
    ENABLE_DATABASE_MONITORING: bool = False
    SLOW_QUERY_THRESHOLD: float = 0.5
    
    # File Storage
    UPLOAD_DIR: str = "./raw_construction_kits"
    PROCESSED_STEMS_DIR: str = "./processed_database/stems"
    EMBEDDINGS_DIR: str = "./dataembeddings"
    RENDERED_TRACKS_DIR: str = "./rendered_tracks"
    GENERATED_STEMS_DIR: str = "./generated_stems"
    STEREO_TRACKS_DIR: str = "./stereo_tracks_for_analysis"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

    # Audio Settings
    AUDIO_SAMPLE_RATE: int = 44100
    
    # CLAP Model
    MODEL_CACHE_DIR: str = "./models"
    CLAP_MODEL_NAME: str = "laion/larger_clap_music_and_speech"
    #: Realer Modus ist der Default: ein Modell-Lade-/Inferenzfehler oder ein
    #: degeneriertes (Null-/nicht-endliches) Embedding darf NIE als Erfolg
    #: durchgereicht werden. ``False`` erlaubt nur den degradierten Altpfad
    #: (Audio bleibt nutzbar, aber ohne Embedding) - bewusst gesetzt, nie
    #: stillschweigend.
    EMBEDDING_FAIL_CLOSED: bool = True
    
    # API
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://localhost:3000",
    ]
    #: Betriebsmodus der API. ``local`` (Default) ist der bisherige, lokale
    #: Betrieb ohne Token. ``shared`` verlangt für jede sensible Route einen
    #: konfigurierten Client-Token (Header ``X-API-Token``).
    OPERATION_MODE: str = "local"
    #: Client-Tokens als ``token:owner``-Paare, kommagetrennt. Leer = keine
    #: Clients konfiguriert; im Modus ``shared`` antwortet der Dienst dann
    #: fail-closed (503) statt still offen weiterzulaufen.
    #: Bewusst ein String: pydantic-settings würde ein Dict-Feld als JSON
    #: parsen und an einer einfachen ``token:owner``-Liste scheitern.
    API_CLIENT_TOKENS: str = ""

    @field_validator("CORS_ORIGINS")
    @classmethod
    def reject_wildcard_origin(cls, value: List[str]) -> List[str]:
        """Ein Wildcard-Origin ist keine Freigabe, sondern ein Loch."""
        if any(origin.strip() == "*" for origin in value):
            raise ValueError("CORS_ORIGINS must not contain '*'")
        return value

    @field_validator("OPERATION_MODE")
    @classmethod
    def validate_operation_mode(cls, value: str) -> str:
        """Nur die beiden bekannten Betriebsmodi sind zulässig."""
        normalized = value.strip().lower()
        if normalized not in {"local", "shared"}:
            raise ValueError(f"Unsupported operation mode: {value}")
        return normalized

    @property
    def client_tokens(self) -> Dict[str, str]:
        """Client-Tokens als ``{token: owner}``-Abbildung."""
        tokens: Dict[str, str] = {}
        for entry in self.API_CLIENT_TOKENS.split(","):
            entry = entry.strip()
            if not entry:
                continue
            token, _, owner = entry.partition(":")
            token = token.strip()
            owner = owner.strip()
            if token and owner:
                tokens[token] = owner
        return tokens

    @property
    def cors_origins(self) -> List[str]:
        """Freigegebene Origins (ohne Leereinträge, ohne Wildcard)."""
        return [origin.strip() for origin in self.CORS_ORIGINS if origin.strip()]

    @field_validator("CORS_ORIGINS", mode="before")
    def split_cors_origins(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Accept only standard Python logging levels."""
        normalized = value.upper()
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed_levels:
            raise ValueError(f"Unsupported log level: {value}")
        return normalized

    @field_validator("MAX_FILE_SIZE")
    @classmethod
    def validate_max_file_size(cls, value: int) -> int:
        """File-size limits must be positive."""
        if value <= 0:
            raise ValueError("MAX_FILE_SIZE must be greater than zero")
        return value

    def get_logs_path(self) -> Path:
        """Gibt den Pfad zum Log-Verzeichnis zurück (injizierbar via LOGS_DIR)."""
        return Path(self.LOGS_DIR)
    
    model_config = ConfigDict(env_file=".env", extra="allow")

settings = Settings()
