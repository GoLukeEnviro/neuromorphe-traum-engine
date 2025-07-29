import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.config import Settings, settings as default_settings
from exceptions import ConfigurationError


class TestSettings:
    """Tests für Haupt-Settings-Klasse"""

    @pytest.fixture(autouse=True)
    def reset_settings(self):
        """Setzt die Settings nach jedem Test zurück, um Isolation zu gewährleisten."""
        original_env = os.environ.copy()
        yield
        os.environ.clear()
        os.environ.update(original_env)
        # Optional: Reset default_settings if it's a global singleton
        # For Pydantic BaseSettings, re-instantiating is usually enough.

    @pytest.mark.unit
    def test_settings_creation(self):
        """Test: Settings-Instanz mit spezifischen Werten erstellen"""
        settings = Settings(
            PROJECT_NAME="Test Engine",
            DEBUG=True,
            DATABASE_URL="sqlite:///test.db",
            AUDIO_SAMPLE_RATE=48000,
            CLAP_MODEL_NAME="test-model",
            MAX_CONCURRENT_JOBS=4,
            CORS_ORIGINS=["http://localhost:3000"],
            LOG_LEVEL="DEBUG"
        )

        assert settings.PROJECT_NAME == "Test Engine"
        assert settings.DEBUG is True
        assert settings.DATABASE_URL == "sqlite:///test.db"
        assert settings.AUDIO_SAMPLE_RATE == 48000
        assert settings.CLAP_MODEL_NAME == "test-model"
        assert settings.MAX_CONCURRENT_JOBS == 4
        assert "http://localhost:3000" in settings.CORS_ORIGINS
        assert settings.LOG_LEVEL == "DEBUG"

    @pytest.mark.unit
    def test_settings_defaults(self):
        """Test: Settings-Standardwerte"""
        settings = Settings() # Load defaults

        assert settings.PROJECT_NAME == "Neuromorphe Traum-Engine v2.0"
        assert settings.DEBUG is False
        assert settings.DATABASE_URL == "sqlite:///processed_database/stems.db"
        assert settings.UPLOAD_DIR == "./raw_construction_kits"
        assert settings.MODEL_CACHE_DIR == "./models"
        assert settings.CLAP_MODEL_NAME == "laion/larger_clap_music_and_speech"
        assert settings.LOG_LEVEL == "INFO"
        assert "http://localhost:8501" in settings.CORS_ORIGINS
        assert "http://localhost:3000" in settings.CORS_ORIGINS

    @pytest.mark.unit
    def test_settings_from_env(self):
        """Test: Settings aus Umgebungsvariablen laden"""
        env_vars = {
            "PROJECT_NAME": "Env Test",
            "DEBUG": "True",
            "DATABASE_URL": "postgresql://localhost/env_db",
            "AUDIO_SAMPLE_RATE": "96000",
            "CLAP_MODEL_NAME": "env-model",
            "LOG_LEVEL": "WARNING"
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings() # Load from env

            assert settings.PROJECT_NAME == "Env Test"
            assert settings.DEBUG is True
            assert settings.DATABASE_URL == "postgresql://localhost/env_db"
            assert settings.AUDIO_SAMPLE_RATE == 96000
            assert settings.CLAP_MODEL_NAME == "env-model"
            assert settings.LOG_LEVEL == "WARNING"

    @pytest.mark.unit
    def test_settings_from_env_file(self, temp_dir: Path):
        """Test: Settings aus .env-Datei laden"""
        env_file_path = temp_dir / ".env"
        env_content = """
PROJECT_NAME="File Test"
DEBUG=False
DATABASE_URL="sqlite:///file_db.db"
AUDIO_SAMPLE_RATE=44100
CLAP_MODEL_NAME="file-model"
LOG_LEVEL="ERROR"
"""
        with open(env_file_path, "w") as f:
            f.write(env_content)

        # Temporarily change the working directory to where the .env file is
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            settings = Settings() # Load from .env in current dir
            assert settings.PROJECT_NAME == "File Test"
            assert settings.DEBUG is False
            assert settings.DATABASE_URL == "sqlite:///file_db.db"
            assert settings.AUDIO_SAMPLE_RATE == 44100
            assert settings.CLAP_MODEL_NAME == "file-model"
            assert settings.LOG_LEVEL == "ERROR"
        finally:
            os.chdir(original_cwd) # Change back

    @pytest.mark.unit
    def test_settings_validation(self):
        """Test: Settings-Validierung"""
        # Invalid LOG_LEVEL
        with pytest.raises(ValueError): # Pydantic raises ValueError for invalid enums/types
            Settings(LOG_LEVEL="INVALID_LEVEL")

        # Invalid MAX_FILE_SIZE (e.g., negative)
        with pytest.raises(ValueError):
            Settings(MAX_FILE_SIZE=-100)

    @pytest.mark.unit
    def test_get_logs_path(self):
        """Test: get_logs_path Methode"""
        settings = Settings()
        logs_path = settings.get_logs_path()
        assert logs_path == Path("./logs")
        assert isinstance(logs_path, Path)

    @pytest.mark.unit
    def test_settings_priority(self, temp_dir: Path):
        """Test: Priorität der Settings (Umgebung > .env > Standard)"""
        # Create a .env file
        env_file_path = temp_dir / ".env"
        with open(env_file_path, "w") as f:
            f.write("PROJECT_NAME=EnvFileProject\nDEBUG=True")

        # Set an environment variable
        os.environ["PROJECT_NAME"] = "EnvVarProject"
        os.environ["LOG_LEVEL"] = "CRITICAL"

        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            settings = Settings() # Load from env, then .env, then defaults
            assert settings.PROJECT_NAME == "EnvVarProject" # Env var has highest priority
            assert settings.DEBUG is True # .env has higher priority than default
            assert settings.LOG_LEVEL == "CRITICAL" # Env var has highest priority
        finally:
            os.chdir(original_cwd)
            del os.environ["PROJECT_NAME"]
            del os.environ["LOG_LEVEL"]


class TestConfigurationErrors:
    """Tests für Konfigurationsfehler"""

    @pytest.mark.unit
    def test_invalid_env_var_types(self):
        """Test: Ungültige Typen in Umgebungsvariablen"""
        env_vars = {
            "DEBUG": "not_a_boolean",
            "MAX_FILE_SIZE": "not_a_number",
        }

        with patch.dict(os.environ, env_vars):
            with pytest.raises(ValueError): # Pydantic raises ValueError for type coercion failures
                Settings()
