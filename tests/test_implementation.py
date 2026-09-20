"""Tests für den Dataset-Processor (``ai_agents.prepare_dataset_sql.NeuroAnalyzer``).

Ausgangslage (Defekt): Die drei ursprünglichen „Tests“ gaben ausschließlich
``True``/``False`` zurück und druckten ihren Status. pytest verwirft
Rückgabewerte (es gab nur eine ``PytestReturnNotNoneWarning``) — der Lauf war
damit immer grün, unabhängig vom Ergebnis („stilles Grün“). Gleichzeitig
schrieben die Tests über feste relative Pfade in die Produktionsdaten
(``processed_database/stems.db``, ``processed_database/checkpoints``,
``processed_database/stems``, ``processed_database/quarantine``) und riefen
``init_db()`` doppelt auf, wobei eine SQLite-Verbindung leakte.

Jetzt: echte Assertions, alle Pfade über Konstruktor-Parameter in ``tmp_path``
injiziert — inklusive Nachweis, dass der Konstruktor keine Seiteneffekte hat.
"""

import sqlite3
import wave
from pathlib import Path

import numpy as np
import pytest

from ai_agents.prepare_dataset_sql import (
    DB_PATH as ANALYZER_DEFAULT_DB,
    NeuroAnalyzer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_test_wav(
    path: Path,
    seconds: float = 1.0,
    sample_rate: int = 22050,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> Path:
    """Erzeugt eine gültige Mono-WAV-Datei (Sinus) für die Verarbeitung."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return path


@pytest.fixture
def injected_paths(tmp_path: Path) -> dict:
    """Alle Pfade, die der Analyzer beschreiben darf — ausschließlich tmp_path."""
    return {
        "input_dir": tmp_path / "input",
        "db_path": tmp_path / "database" / "stems.db",
        "checkpoint_dir": tmp_path / "checkpoints",
        "stems_dir": tmp_path / "stems",
        "quarantine_dir": tmp_path / "quarantine",
    }


@pytest.fixture
def analyzer(injected_paths: dict) -> NeuroAnalyzer:
    """Analyzer mit vollständig injizierten Pfaden."""
    injected_paths["input_dir"].mkdir(parents=True, exist_ok=True)
    instance = NeuroAnalyzer(
        input_dir=str(injected_paths["input_dir"]),
        resume_from_checkpoint=False,
        checkpoint_interval=1,
        db_path=str(injected_paths["db_path"]),
        checkpoint_dir=str(injected_paths["checkpoint_dir"]),
        stems_dir=str(injected_paths["stems_dir"]),
        quarantine_dir=str(injected_paths["quarantine_dir"]),
    )
    yield instance
    if getattr(instance, "conn", None) is not None:
        instance.conn.close()
        instance.conn = None


# ----------------------------------------------------------------------
# Pfad-Injektion
# ----------------------------------------------------------------------
def test_constructor_has_no_filesystem_side_effect(injected_paths: dict):
    """Der Konstruktor darf keine Verzeichnisse anlegen (früher: checkpoints/)."""
    instance = NeuroAnalyzer(
        input_dir=str(injected_paths["input_dir"]),
        db_path=str(injected_paths["db_path"]),
        checkpoint_dir=str(injected_paths["checkpoint_dir"]),
        stems_dir=str(injected_paths["stems_dir"]),
        quarantine_dir=str(injected_paths["quarantine_dir"]),
    )

    assert not injected_paths["checkpoint_dir"].exists()
    assert not injected_paths["db_path"].exists()
    assert not injected_paths["stems_dir"].exists()
    assert not injected_paths["quarantine_dir"].exists()

    assert instance.db_path == str(injected_paths["db_path"])
    assert instance.checkpoint_dir == str(injected_paths["checkpoint_dir"])
    assert instance.stems_dir == str(injected_paths["stems_dir"])
    assert instance.quarantine_dir == str(injected_paths["quarantine_dir"])


def test_injected_paths_win_over_production_defaults(injected_paths: dict):
    """Die Defaults sind überschreibbar — Beweis gegen einen Rückfall auf Produktion."""
    instance = NeuroAnalyzer(
        input_dir=str(injected_paths["input_dir"]),
        db_path=str(injected_paths["db_path"]),
        checkpoint_dir=str(injected_paths["checkpoint_dir"]),
        stems_dir=str(injected_paths["stems_dir"]),
        quarantine_dir=str(injected_paths["quarantine_dir"]),
    )

    for attribute in ("db_path", "checkpoint_dir", "stems_dir", "quarantine_dir"):
        resolved = Path(getattr(instance, attribute)).resolve()
        assert injected_paths["db_path"].parent.parent in resolved.parents, (
            f"{attribute} zeigt nicht in das injizierte Testverzeichnis: {resolved}"
        )
        assert REPO_ROOT not in resolved.parents


def test_module_default_db_path_is_env_injected(sandbox_root: Path):
    """Der Modul-Default folgt ``NEUROMORPHE_DB_PATH`` (Env-Injektion)."""
    default = Path(ANALYZER_DEFAULT_DB).resolve()

    assert str(sandbox_root) in str(default), f"Default-DB nicht in der Sandbox: {default}"
    assert REPO_ROOT not in default.parents

    # Auch ohne Konstruktor-Parameter landet ein Analyzer in der Sandbox.
    instance = NeuroAnalyzer(input_dir=str(sandbox_root / "raw_construction_kits"))
    try:
        assert Path(instance.db_path).resolve() == default
        assert str(sandbox_root) in str(Path(instance.checkpoint_dir).resolve())
    finally:
        if getattr(instance, "conn", None) is not None:
            instance.conn.close()
            instance.conn = None


# ----------------------------------------------------------------------
# Datenbank-Initialisierung
# ----------------------------------------------------------------------
def test_init_db_creates_schema_at_injected_path(analyzer: NeuroAnalyzer, injected_paths: dict):
    """``init_db()`` legt Schema und Datei am injizierten Pfad an."""
    assert not injected_paths["db_path"].exists()

    analyzer.init_db()

    assert injected_paths["db_path"].is_file()

    connection = sqlite3.connect(str(injected_paths["db_path"]))
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"stems", "processing_status"} <= tables

        stem_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(stems)")
        }
        assert {
            "id",
            "path",
            "bpm",
            "key",
            "category",
            "tags",
            "features",
            "quality_ok",
            "user_rating",
            "imported_at",
            "clap_embedding",
        } <= stem_columns

        status_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(processing_status)")
        }
        assert {"file_path", "file_hash", "status", "last_attempt"} <= status_columns
    finally:
        connection.close()


def test_init_db_twice_is_idempotent_and_closes_previous_connection(
    analyzer: NeuroAnalyzer, injected_paths: dict
):
    """Zweiter ``init_db()``-Aufruf ist erlaubt und leakt keine Verbindung.

    Vorher wurde ``init_db()`` in der Suite doppelt aufgerufen, wobei die erste
    Verbindung offen blieb.
    """
    analyzer.init_db()
    first_connection = analyzer.conn
    assert first_connection is not None

    analyzer.init_db()  # darf nicht fehlschlagen

    assert analyzer.conn is not first_connection

    # Die alte Verbindung ist geschlossen — Nutzung muss fehlschlagen.
    with pytest.raises(sqlite3.ProgrammingError):
        first_connection.execute("SELECT 1 FROM stems")

    connection = sqlite3.connect(str(injected_paths["db_path"]))
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        connection.close()
    assert {"stems", "processing_status"} <= tables


# ----------------------------------------------------------------------
# Verzeichnisstruktur / Dateierkennung
# ----------------------------------------------------------------------
def test_discover_audio_files_only_returns_audio(analyzer: NeuroAnalyzer, injected_paths: dict):
    """Nur Audiodateien gelten als Kandidaten — Nicht-Audio wird ignoriert."""
    input_dir = injected_paths["input_dir"]
    (input_dir / "nested").mkdir(parents=True, exist_ok=True)

    expected = {
        str(write_test_wav(input_dir / "test_kick_sample.wav")),
        str(write_test_wav(input_dir / "nested" / "test_bass_line.flac")),
    }
    (input_dir / "README.txt").write_text("kein Audio")
    (input_dir / "cover.png").write_bytes(b"\x89PNG")

    discovered = analyzer._discover_audio_files()

    assert set(discovered) == expected
    assert all(Path(path).is_file() for path in discovered)


# ----------------------------------------------------------------------
# Läufe
# ----------------------------------------------------------------------
def test_run_on_empty_input_dir_writes_nothing(analyzer: NeuroAnalyzer, injected_paths: dict):
    """Ohne Audiodateien bleibt die Datenbank leer (statt stillem Erfolg)."""
    analyzer.init_db()
    analyzer.run()

    assert analyzer.stats["total_files"] == 0
    assert analyzer.stats["processed"] == 0

    connection = sqlite3.connect(str(injected_paths["db_path"]))
    try:
        assert connection.execute("SELECT COUNT(*) FROM stems").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM processing_status").fetchone()[0] == 0
    finally:
        connection.close()


def test_run_processes_audio_into_injected_paths(analyzer: NeuroAnalyzer, injected_paths: dict):
    """Ein echter Lauf schreibt Stem und Status ausschließlich in tmp_path."""
    source = write_test_wav(injected_paths["input_dir"] / "test_kick_sample.wav")

    analyzer.init_db()
    analyzer.run()

    assert analyzer.stats["total_files"] == 1
    assert analyzer.stats["processed"] == 1
    assert analyzer.stats["failed"] == 0

    connection = sqlite3.connect(str(injected_paths["db_path"]))
    try:
        rows = connection.execute(
            "SELECT id, path, category, tags, features, clap_embedding FROM stems"
        ).fetchall()
        status_rows = connection.execute(
            "SELECT file_path, status FROM processing_status"
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 1, f"Erwartet 1 Stem, gefunden: {rows}"
    stem_id, standardized_path, category, tags, features, embedding = rows[0]

    # Dateiname-Heuristik muss greifen (früher nie geprüft).
    assert category == "kick"
    assert stem_id.startswith("kick_")

    # Der standardisierte Stem liegt im injizierten Verzeichnis und existiert.
    standardized = Path(standardized_path).resolve()
    assert injected_paths["stems_dir"].resolve() in standardized.parents
    assert standardized.is_file()
    assert standardized.stat().st_size > 0

    # Tags/Features sind JSON und das Embedding hat die CLAP-Dimension.
    import json

    assert len(json.loads(tags)) == 3
    assert {"spectral_centroid", "rms", "zero_crossing_rate"} <= set(json.loads(features))
    assert len(embedding) == 512 * 4

    assert status_rows == [(str(source), "completed")]

    # Die Originaldatei bleibt unverändert liegen.
    assert source.is_file()


def test_run_is_resumable_per_processing_status(analyzer: NeuroAnalyzer, injected_paths: dict):
    """Bereits verarbeitete Dateien werden beim zweiten Lauf übersprungen."""
    write_test_wav(injected_paths["input_dir"] / "test_kick_sample.wav")

    analyzer.init_db()
    analyzer.run()
    processed_after_first_run = analyzer.stats["processed"]

    analyzer.stats["processed"] = 0
    analyzer.resume_from_checkpoint = True
    analyzer.run()

    assert processed_after_first_run == 1
    assert analyzer.stats["processed"] == 0, "Datei wurde erneut verarbeitet"

    connection = sqlite3.connect(str(injected_paths["db_path"]))
    try:
        assert connection.execute("SELECT COUNT(*) FROM stems").fetchone()[0] == 1
    finally:
        connection.close()


# ----------------------------------------------------------------------
# Checkpoints, Metadaten, Quarantäne
# ----------------------------------------------------------------------
def test_checkpoint_is_written_to_injected_dir(analyzer: NeuroAnalyzer, injected_paths: dict):
    """Checkpoints landen im injizierten Verzeichnis (früher: Produktion)."""
    output = write_test_wav(injected_paths["input_dir"] / "test_kick_sample.wav")
    assert output.is_file()

    analyzer.init_db()
    analyzer.run()

    progress_file = injected_paths["checkpoint_dir"] / "progress.json"
    assert progress_file.is_file(), "Checkpoint wurde nicht geschrieben"

    import json

    payload = json.loads(progress_file.read_text())
    assert payload["stats"]["processed"] == 1
    assert payload["stats"]["total_files"] == 1


def test_quarantine_writes_into_injected_dir(analyzer: NeuroAnalyzer, injected_paths: dict):
    """Quarantäne-Kopien landen im injizierten Verzeichnis."""
    source = write_test_wav(injected_paths["input_dir"] / "broken.wav")

    analyzer._quarantine_file(str(source), "validation_failed")

    quarantined = sorted(injected_paths["quarantine_dir"].glob("*.wav"))
    assert len(quarantined) == 1
    assert quarantined[0].name.endswith("broken.wav")
    assert quarantined[0].read_bytes() == source.read_bytes()
    assert source.is_file(), "Originaldatei darf nicht verschwinden"


def test_create_metadata_never_escapes_injected_dirs(analyzer: NeuroAnalyzer, injected_paths: dict):
    """``_create_metadata`` schreibt ausschließlich in ``stems_dir``."""
    source = write_test_wav(injected_paths["input_dir"] / "test_kick_sample.wav")
    audio = np.zeros(22050, dtype=np.float32)
    audio_result = {
        "audio_data": audio,
        "duration": 1.0,
        "bpm": 128.0,
        "features": {
            "spectral_centroid": 1000.0,
            "zero_crossing_rate": 0.1,
            "rms": 0.2,
            "spectral_rolloff": 2000.0,
            "spectral_bandwidth": 1000.0,
        },
    }

    metadata = analyzer._create_metadata(str(source), audio_result, embedding := b"\x00" * 8)

    written = Path(metadata["path"]).resolve()
    assert injected_paths["stems_dir"].resolve() in written.parents
    assert written.is_file()
    assert metadata["category"] == "kick"
    assert metadata["clap_embedding"] == embedding
