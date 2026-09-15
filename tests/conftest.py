"""Pytest-Konfiguration und gemeinsame Fixtures für Tests"""

import sys
from pathlib import Path
import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator, List
from unittest.mock import MagicMock, patch

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add project root to Python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Add src directory to Python path
sys.path.insert(0, str(root_dir / "src"))

# src.X und X auf dieselben Modulobjekte abbilden, damit patch()-Ziele in der
# Test-Suite (z. B. src.services.renderer.RendererService) auch für den
# Produktivcode (services.renderer) wirksam sind.
import _src_alias  # noqa: E402,F401

# Test-spezifische Imports
from core.config import Settings
from database.database import get_async_db_session
from database.database import create_tables
from database.models import Base
from main import app
from services.arranger import ArrangerService
from services.neuro_analyzer import NeuroAnalyzer
from services.preprocessor import PreprocessorService
from services.renderer import RendererService


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
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


@pytest.fixture(scope="session")
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_tables(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    async with db_engine.begin() as connection:
        async with AsyncSession(bind=connection) as session:
            yield session


@pytest.fixture
async def test_db_session(db_session):
    """Alias für Tests, die den historischen Fixture-Namen nutzen"""
    yield db_session


@pytest.fixture(scope="function")
def test_client(test_settings: Settings, db_session: AsyncSession, monkeypatch) -> Generator[TestClient, None, None]:
    """FastAPI Test-Client.

    Wichtig: Es wird nicht nur ``get_async_db_session`` überschrieben, sondern
    auch ``get_database_manager``. Die API-Router nutzen den Manager direkt
    (``Depends(get_db_manager)``); ohne diesen zweiten Override würden Tests
    auf die ECHTE Produktionsdatenbank unter ``processed_database/stems.db``
    schreiben. Der Manager wird deshalb auf eine temporäre SQLite-Datei
    umgebogen.
    """
    import tempfile
    from database.database import DatabaseManager, get_database_manager

    tmpdir = tempfile.mkdtemp(prefix="nt-test-db-")
    test_db_path = Path(tmpdir) / "test_api.db"

    # Eigenen Manager für die Testdauer erzeugen und global einsetzen.
    isolated_settings = Settings(
        DATABASE_URL=f"sqlite:///{test_db_path}", LOG_LEVEL="DEBUG"
    )
    isolated_manager = DatabaseManager(isolated_settings)

    monkeypatch.setattr(
        "database.database._database_manager", isolated_manager, raising=False
    )
    monkeypatch.setattr(
        "database.database.get_database_manager", lambda: isolated_manager, raising=False
    )

    # Test-DB initialisieren (Tabellen anlegen) und Manager global ersetzen.
    import asyncio as _asyncio

    _asyncio.run(isolated_manager.create_tables())

    # Dependency Override für Datenbank-Session
    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_async_db_session] = override_get_db_session

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.pop(get_async_db_session, None)


@pytest.fixture(scope="function")
def db_manager(test_settings: Settings):
    """DatabaseManager für Tests, mit einer temporären SQLite-Datei."""
    import tempfile
    from database.database import DatabaseManager

    tmpdir = tempfile.mkdtemp(prefix="nt-test-db-")
    db_path = Path(tmpdir) / "test.db"
    settings = Settings(
        DATABASE_URL=f"sqlite:///{db_path}",
        LOG_LEVEL="DEBUG",
    )
    manager = DatabaseManager(settings)
    yield manager


@pytest.fixture(scope="function")
def sample_audio_file(tmp_path: Path) -> Path:
    """Pfad zu einer gültigen Test-WAV-Datei."""
    import wave
    import numpy as np

    sample_rate = 22050
    duration = 1.0
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration))
    mono = np.sin(2 * np.pi * frequency * t)
    audio_int16 = (mono * 32767).astype(np.int16)

    file_path = tmp_path / "sample_audio.wav"
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return file_path


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