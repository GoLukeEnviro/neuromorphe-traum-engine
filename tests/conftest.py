"""Pytest-Konfiguration und gemeinsame Fixtures für Tests.

Test-Isolation
--------------
Die Suite darf ausschließlich in temporäre Verzeichnisse schreiben. Beim Import
dieses Moduls passiert deshalb dreierlei:

1. Eine Sandbox unter ``$TMPDIR`` wird angelegt.
2. Die komplette Pfad-Konfiguration der Anwendung wird per Umgebungsvariable
   auf diese Sandbox umgebogen (Datenbank, Input, Output, Modelle, Cache,
   Logs). Das wirkt auch auf das globale ``settings``-Singleton und damit auf
   alle Services, die ohne explizite Konfiguration konstruiert werden.
3. Ein :class:`tests._isolation_guard.ProductionWriteGuard` wird installiert,
   der jeden Schreibzugriff auf einen Produktionspfad sofort scheitern lässt.

Der Import von ``src``-Modulen erfolgt bewusst *nach* dem Setzen der
Umgebungsvariablen — sonst wäre das Settings-Singleton bereits auf die
Produktionspfade eingefroren.
"""

import asyncio
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator, List
from unittest.mock import MagicMock

import pytest

# ----------------------------------------------------------------------
# 1. Sandbox anlegen — VOR dem Import der Anwendungsmodule
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_ROOT = Path(tempfile.mkdtemp(prefix="nt-test-sandbox-")).resolve()

if REPO_ROOT == SANDBOX_ROOT or REPO_ROOT in SANDBOX_ROOT.parents:  # pragma: no cover
    raise RuntimeError(
        f"Test-Sandbox darf nicht im Repository liegen: {SANDBOX_ROOT}"
    )

SANDBOX_DIRS = {
    "input": SANDBOX_ROOT / "raw_construction_kits",
    "processed_database": SANDBOX_ROOT / "processed_database",
    "stems": SANDBOX_ROOT / "processed_database" / "stems",
    "checkpoints": SANDBOX_ROOT / "processed_database" / "checkpoints",
    "quarantine": SANDBOX_ROOT / "processed_database" / "quarantine",
    "models": SANDBOX_ROOT / "models",
    "logs": SANDBOX_ROOT / "logs",
    "dataembeddings": SANDBOX_ROOT / "dataembeddings",
    "rendered_tracks": SANDBOX_ROOT / "rendered_tracks",
    "generated_stems": SANDBOX_ROOT / "generated_stems",
    "stereo_tracks": SANDBOX_ROOT / "stereo_tracks_for_analysis",
    "audio_input": SANDBOX_ROOT / "test_data" / "audio_input",
    "audio_output": SANDBOX_ROOT / "test_data" / "audio_output",
    "test_cache": SANDBOX_ROOT / "test_cache",
}
for _directory in SANDBOX_DIRS.values():
    _directory.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. Pfade injizieren — wirkt auf das Settings-Singleton und alle Services
# ----------------------------------------------------------------------
SANDBOX_DB_PATH = SANDBOX_DIRS["processed_database"] / "stems.db"

SANDBOX_ENV = {
    # Anwendungs-Settings (src/core/config.py)
    "DATABASE_URL": f"sqlite:///{SANDBOX_DB_PATH}",
    "UPLOAD_DIR": str(SANDBOX_DIRS["input"]),
    "PROCESSED_STEMS_DIR": str(SANDBOX_DIRS["stems"]),
    "EMBEDDINGS_DIR": str(SANDBOX_DIRS["dataembeddings"]),
    "RENDERED_TRACKS_DIR": str(SANDBOX_DIRS["rendered_tracks"]),
    "GENERATED_STEMS_DIR": str(SANDBOX_DIRS["generated_stems"]),
    "STEREO_TRACKS_DIR": str(SANDBOX_DIRS["stereo_tracks"]),
    "MODEL_CACHE_DIR": str(SANDBOX_DIRS["models"]),
    "LOGS_DIR": str(SANDBOX_DIRS["logs"]),
    # Skripte unter ai_agents/ (ai_agents/prepare_dataset_sql.py).
    # Eigene Datei: das Legacy-Schema (mit ``path``-Spalte) und das ORM-Schema
    # belegen produktiv denselben Dateinamen — in der Sandbox bekommen beide
    # getrennte Dateien, damit beide Schreibpfade prüfbar bleiben.
    "NEUROMORPHE_DB_PATH": str(SANDBOX_DIRS["processed_database"] / "legacy_stems.db"),
    "NEUROMORPHE_CHECKPOINT_DIR": str(SANDBOX_DIRS["checkpoints"]),
    "NEUROMORPHE_STEMS_DIR": str(SANDBOX_DIRS["stems"]),
    "NEUROMORPHE_QUARANTINE_DIR": str(SANDBOX_DIRS["quarantine"]),
    # Kein echtes CLAP-Modell im Testlauf: deterministisch, keine Downloads.
    "USE_REAL_CLAP": "0",
}
os.environ.update(SANDBOX_ENV)
os.environ["TESTING"] = "true"

# ----------------------------------------------------------------------
# 3. Produktionspfade hart sperren (gilt auch für Imports/Kollektierung)
# ----------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests._isolation_guard import (  # noqa: E402
    ProductionWriteGuard,
    ProductionWriteViolation,  # noqa: F401  (Re-Export für Tests)
    match_production_path,  # noqa: F401  (Re-Export für Tests)
    snapshot_production,  # noqa: F401  (Re-Export für Tests)
)

PRODUCTION_WRITE_GUARD = ProductionWriteGuard(root=REPO_ROOT)
PRODUCTION_WRITE_GUARD.install()


def _cleanup_sandbox() -> None:
    """Sandbox am Prozessende entfernen (ohne Sandbox-Guard)."""
    PRODUCTION_WRITE_GUARD.uninstall()
    shutil.rmtree(SANDBOX_ROOT, ignore_errors=True)


atexit.register(_cleanup_sandbox)

# src.X und X auf dieselben Modulobjekte abbilden, damit patch()-Ziele in der
# Test-Suite (z. B. src.services.renderer.RendererService) auch für den
# Produktivcode (services.renderer) wirksam sind.
import _src_alias  # noqa: E402,F401

# Test-spezifische Imports (erst jetzt — nach der Pfad-Injektion)
from core.config import Settings  # noqa: E402
from database.database import get_async_db_session  # noqa: E402
from database.database import create_tables  # noqa: E402
from database.models import Base  # noqa: E402
from main import app  # noqa: E402
from services.arranger import ArrangerService  # noqa: E402
from services.neuro_analyzer import NeuroAnalyzer  # noqa: E402
from services.preprocessor import PreprocessorService  # noqa: E402
from services.renderer import RendererService  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine  # noqa: E402


# ----------------------------------------------------------------------
# Sandbox-Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def initialize_sandbox_database() -> Generator[None, None, None]:
    """Legt das Schema der Sandbox-Datenbank an (globaler Manager).

    Ohne das würde jeder direkte Zugriff über ``get_database_manager()`` an
    fehlenden Tabellen scheitern — die Suite würde dann faktisch auf die
    Produktionsdatenbank ausweichen.
    """
    from database.database import create_tables, get_database_manager

    manager = get_database_manager()
    asyncio.run(create_tables(manager.async_engine))
    yield


@pytest.fixture(scope="session")
def sandbox_root() -> Path:
    """Wurzel der Test-Sandbox (temporär, außerhalb des Repos)."""
    return SANDBOX_ROOT


@pytest.fixture(scope="session")
def production_write_guard() -> Generator[ProductionWriteGuard, None, None]:
    """Zugriff auf den installierten Sandbox-Guard + Sammelprüfung am Ende."""
    yield PRODUCTION_WRITE_GUARD
    PRODUCTION_WRITE_GUARD.assert_clean()


def sandbox_settings(**overrides) -> Settings:
    """Settings-Instanz, die vollständig in die Sandbox zeigt."""
    values = {
        "DATABASE_URL": f"sqlite:///{SANDBOX_DB_PATH}",
        "UPLOAD_DIR": str(SANDBOX_DIRS["input"]),
        "PROCESSED_STEMS_DIR": str(SANDBOX_DIRS["stems"]),
        "EMBEDDINGS_DIR": str(SANDBOX_DIRS["dataembeddings"]),
        "RENDERED_TRACKS_DIR": str(SANDBOX_DIRS["rendered_tracks"]),
        "GENERATED_STEMS_DIR": str(SANDBOX_DIRS["generated_stems"]),
        "STEREO_TRACKS_DIR": str(SANDBOX_DIRS["stereo_tracks"]),
        "MODEL_CACHE_DIR": str(SANDBOX_DIRS["models"]),
        "LOGS_DIR": str(SANDBOX_DIRS["logs"]),
        "LOG_LEVEL": "DEBUG",
        # Test-spezifische Einstellungen
        "PERFORMANCE_TRACKING": False,
        "ENABLE_DATABASE_MONITORING": False,
        "AUDIO_SAMPLE_RATE": 22050,  # niedrigere Sample-Rate für Tests
        # Historische Feldnamen (Services lesen sie als Extra-Felder)
        "async_database_url": f"sqlite+aiosqlite:///{SANDBOX_DB_PATH}",
        "audio_input_dir": str(SANDBOX_DIRS["audio_input"]),
        "audio_output_dir": str(SANDBOX_DIRS["audio_output"]),
        "processed_database_dir": str(SANDBOX_DIRS["processed_database"]),
        "generated_tracks_dir": str(SANDBOX_DIRS["generated_stems"]),
        "audio_sample_rate": 22050,
        "audio_bit_depth": 16,
        "audio_output_format": "wav",
        "max_audio_duration": 30,
        "max_concurrent_jobs": 1,
        "enable_performance_monitoring": False,
        "clap_model_name": "test_model",
        "clap_cache_dir": str(SANDBOX_DIRS["models"]),
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test-Konfiguration — alle Pfade zeigen in die Sandbox."""
    return sandbox_settings()


@pytest.fixture(autouse=True)
def restore_shared_test_settings(test_settings: Settings) -> Generator[None, None, None]:
    """Stellt die geteilte Session-Konfiguration nach jedem Test wieder her.

    Einzelne Tests (z. B. ``test_health_check_failure``) mutieren
    ``test_settings.database.url``. Ohne Wiederherstellung leckt diese
    Mutation in alle nachfolgenden Tests.
    """
    original_url = test_settings.DATABASE_URL
    yield
    test_settings.DATABASE_URL = original_url


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Temporäres Verzeichnis für Tests (liegt in der Sandbox)."""
    path = SANDBOX_ROOT / "scratch"
    path.mkdir(parents=True, exist_ok=True)
    yield path


@pytest.fixture(scope="session")
async def db_engine():
    engine_file = SANDBOX_DIRS["processed_database"] / "engine.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{engine_file}")
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


@pytest.fixture
def db_manager(test_settings: Settings) -> Generator[object, None, None]:
    """DatabaseManager für Tests, mit einer temporären SQLite-Datei."""
    from database.database import DatabaseManager

    tmpdir = tempfile.mkdtemp(prefix="nt-test-db-", dir=str(SANDBOX_ROOT))
    db_path = Path(tmpdir) / "test.db"
    settings = sandbox_settings(DATABASE_URL=f"sqlite:///{db_path}")
    manager = DatabaseManager(settings)
    yield manager


@pytest.fixture(scope="function")
def test_client(test_settings: Settings, db_session: AsyncSession, monkeypatch) -> Generator[TestClient, None, None]:
    """FastAPI Test-Client.

    Es wird nicht nur ``get_async_db_session`` überschrieben, sondern auch
    ``get_database_manager``. Die API-Router nutzen den Manager direkt
    (``Depends(get_db_manager)``); der Manager wird auf eine temporäre
    SQLite-Datei in der Sandbox umgebogen.
    """
    from database.database import DatabaseManager, get_database_manager

    tmpdir = tempfile.mkdtemp(prefix="nt-test-db-", dir=str(SANDBOX_ROOT))
    test_db_path = Path(tmpdir) / "test_api.db"

    isolated_settings = sandbox_settings(DATABASE_URL=f"sqlite:///{test_db_path}")
    isolated_manager = DatabaseManager(isolated_settings)

    monkeypatch.setattr(
        "database.database._database_manager", isolated_manager, raising=False
    )
    monkeypatch.setattr(
        "database.database.get_database_manager", lambda: isolated_manager, raising=False
    )

    import asyncio as _asyncio

    _asyncio.run(isolated_manager.create_tables())

    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_async_db_session] = override_get_db_session

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.pop(get_async_db_session, None)


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

    mock.analyze_audio.return_value = {
        "embeddings": [0.1, 0.2, 0.3],
        "features": {
            "tempo": 128.0,
            "key": "C",
            "energy": 0.8,
            "valence": 0.6,
        },
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
        "channels": 2,
    }

    mock.extract_features.return_value = {
        "tempo": 128.0,
        "key": "C",
        "spectral_features": [0.1, 0.2, 0.3],
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
            {"section": "chorus", "start": 48, "duration": 32},
        ],
        "stems": [1, 2, 3],
        "total_duration": 80,
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
        "file_size": 1024000,
    }

    return mock


@pytest.fixture(scope="function")
def sample_audio_data() -> bytes:
    """Beispiel-Audio-Daten für Tests"""
    import wave
    import numpy as np
    from io import BytesIO

    sample_rate = 22050
    duration = 1.0
    frequency = 440

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return buffer.getvalue()


@pytest.fixture(scope="function")
def sample_text_prompts() -> List[str]:
    """Beispiel-Text-Prompts für Tests"""
    return [
        "Dark atmospheric techno with heavy bass",
        "Uplifting house music with piano melodies",
        "Minimal techno with hypnotic arpeggios",
        "Aggressive industrial track with distorted synths",
    ]


@pytest.fixture(autouse=True)
def setup_test_environment(test_settings: Settings) -> Generator[None, None, None]:
    """Test-Umgebung einrichten — ausschließlich innerhalb der Sandbox."""
    for dir_path in [
        test_settings.UPLOAD_DIR,
        test_settings.PROCESSED_STEMS_DIR,
        test_settings.EMBEDDINGS_DIR,
        test_settings.RENDERED_TRACKS_DIR,
        test_settings.GENERATED_STEMS_DIR,
        test_settings.MODEL_CACHE_DIR,
        test_settings.LOGS_DIR,
        test_settings.audio_input_dir,
        test_settings.audio_output_dir,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"

    yield

    os.environ.pop("TESTING", None)


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

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return file_path


def assert_audio_file_valid(file_path: Path) -> None:
    """Audio-Datei validieren — echte Assertion statt bool-Rückgabe."""
    import wave

    with wave.open(str(file_path), "rb") as wav_file:
        assert wav_file.getnframes() > 0, f"Keine Frames in {file_path}"
        assert wav_file.getframerate() > 0, f"Ungültige Sample-Rate in {file_path}"
        assert wav_file.getnchannels() > 0, f"Ungültige Kanalzahl in {file_path}"
