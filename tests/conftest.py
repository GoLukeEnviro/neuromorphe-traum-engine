"""Pytest-Konfiguration und gemeinsame Fixtures für Tests"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Test-spezifische Imports
from src.core.config import Settings
from src.database.database import DatabaseManager
from src.database.database import get_async_db_session
from src.database.models import Base
from src.main import app
from src.services.arranger import ArrangerService
from src.services.neuro_analyzer import NeuroAnalyzer
from src.services.preprocessor import PreprocessorService
from src.services.renderer import RendererService


@pytest.fixture(scope="session")
def event_loop():
    """Event Loop für async Tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test-Konfiguration"""
    return Settings(
        # Test-Datenbank
        database_url="sqlite:///./test_database.db",
        async_database_url="sqlite+aiosqlite:///./test_database.db",
        
        # Test-Verzeichnisse
        audio_input_dir="./test_data/audio_input",
        audio_output_dir="./test_data/audio_output",
        processed_database_dir="./test_data/processed_database",
        generated_tracks_dir="./test_data/generated_tracks",
        
        # Test-spezifische Einstellungen
        log_level="DEBUG",
        enable_performance_monitoring=False,
        max_concurrent_jobs=1,
        
        # CLAP-Modell (Mock für Tests)
        clap_model_name="test_model",
        clap_cache_dir="./test_cache",
        
        # Audio-Einstellungen
        audio_sample_rate=22050,  # Niedrigere Sample-Rate für Tests
        audio_bit_depth=16,
        max_audio_duration=30,  # Kurze Clips für Tests
    )


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Temporäres Verzeichnis für Tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
async def test_db_manager(test_settings: Settings, temp_dir: Path) -> AsyncGenerator[DatabaseManager, None]:
    """Test-Datenbankmanager"""
    # Test-Datenbank in temporärem Verzeichnis
    test_db_path = temp_dir / "test.db"
    test_settings.database_url = f"sqlite:///{test_db_path}"
    test_settings.async_database_url = f"sqlite+aiosqlite:///{test_db_path}"
    
    db_manager = DatabaseManager(test_settings)
    await db_manager.initialize()
    
    # Tabellen erstellen
    await db_manager.create_tables()
    
    yield db_manager
    
    # Cleanup
    await db_manager.close()
    if test_db_path.exists():
        test_db_path.unlink()


@pytest.fixture(scope="function")
async def test_db_session(test_db_manager: DatabaseManager) -> AsyncGenerator[AsyncSession, None]:
    """Test-Datenbank-Session"""
    async with test_db_manager.get_async_session() as session:
        yield session


@pytest.fixture(scope="function")
def test_client(test_settings: Settings, test_db_manager: DatabaseManager) -> TestClient:
    """FastAPI Test-Client"""
    app = create_app(test_settings)
    
    # Dependency Override für Datenbank
    async def override_get_db():
        async with test_db_manager.get_async_session() as session:
            yield session
    
    app.dependency_overrides[get_database_session] = override_get_db
    
    return TestClient(app)


@pytest.fixture(scope="function")
def mock_neuro_analyzer() -> MagicMock:
    """Mock für NeuroAnalyzer"""
    mock = MagicMock(spec=NeuroAnalyzer)
    
    # Mock-Methoden
    mock.analyze_audio.return_value = {
        "embeddings": [0.1, 0.2, 0.3],
        "features": {
            "tempo": 128.0,
            "key": "C",
            "energy": 0.8,
            "valence": 0.6
        }
    }
    
    mock.get_similar_stems.return_value = [
        {"id": 1, "similarity": 0.9},
        {"id": 2, "similarity": 0.8}
    ]
    
    return mock


@pytest.fixture(scope="function")
def mock_preprocessor_service() -> MagicMock:
    """Mock für PreprocessorService"""
    mock = MagicMock(spec=PreprocessorService)
    
    mock.process_audio_file.return_value = {
        "stem_id": 1,
        "duration": 30.0,
        "sample_rate": 44100,
        "channels": 2
    }
    
    mock.extract_features.return_value = {
        "tempo": 128.0,
        "key": "C",
        "spectral_features": [0.1, 0.2, 0.3]
    }
    
    return mock


@pytest.fixture(scope="function")
def mock_arranger_service() -> MagicMock:
    """Mock für ArrangerService"""
    mock = MagicMock(spec=ArrangerService)
    
    mock.create_arrangement.return_value = {
        "arrangement_id": "test_arrangement_123",
        "structure": [
            {"section": "intro", "start": 0, "duration": 16},
            {"section": "verse", "start": 16, "duration": 32},
            {"section": "chorus", "start": 48, "duration": 32}
        ],
        "stems": [1, 2, 3],
        "total_duration": 80
    }
    
    return mock


@pytest.fixture(scope="function")
def mock_renderer_service() -> MagicMock:
    """Mock für RendererService"""
    mock = MagicMock(spec=RendererService)
    
    mock.render_track.return_value = {
        "track_id": "test_track_123",
        "file_path": "/test/output/track.wav",
        "duration": 180.0,
        "file_size": 1024000
    }
    
    return mock


@pytest.fixture(scope="function")
def sample_audio_data() -> bytes:
    """Beispiel-Audio-Daten für Tests"""
    # Einfache Sinus-Welle als WAV-Daten
    import wave
    import numpy as np
    from io import BytesIO
    
    sample_rate = 22050
    duration = 1.0  # 1 Sekunde
    frequency = 440  # A4
    
    # Sinus-Welle generieren
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Als 16-bit WAV konvertieren
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # WAV-Datei in BytesIO schreiben
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return buffer.getvalue()


@pytest.fixture(scope="function")
def sample_text_prompts() -> list[str]:
    """Beispiel-Text-Prompts für Tests"""
    return [
        "Dark atmospheric techno with heavy bass",
        "Uplifting house music with piano melodies",
        "Minimal techno with hypnotic arpeggios",
        "Aggressive industrial track with distorted synths"
    ]


@pytest.fixture(autouse=True)
def setup_test_environment(test_settings: Settings, temp_dir: Path):
    """Test-Umgebung einrichten"""
    # Test-Verzeichnisse erstellen
    for dir_path in [
        test_settings.audio_input_dir,
        test_settings.audio_output_dir,
        test_settings.processed_database_dir,
        test_settings.generated_tracks_dir,
        test_settings.clap_cache_dir
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Umgebungsvariablen für Tests setzen
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


# Pytest-Marker für verschiedene Test-Kategorien
pytestmark = [
    pytest.mark.asyncio,
]


# Hilfsfunktionen für Tests
def create_test_audio_file(file_path: Path, duration: float = 1.0) -> Path:
    """Test-Audio-Datei erstellen"""
    import wave
    import numpy as np
    
    sample_rate = 22050
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with wave.open(str(file_path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return file_path


def assert_audio_file_valid(file_path: Path) -> bool:
    """Audio-Datei validieren"""
    import wave
    
    try:
        with wave.open(str(file_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            
            return frames > 0 and sample_rate > 0 and channels > 0
    except Exception:
        return False