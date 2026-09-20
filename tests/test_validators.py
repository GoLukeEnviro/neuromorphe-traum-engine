"""Tests für die Validierungsskripte ``validate_directive_003`` / ``004``.

Ausgangslage (Defekt): Beide Skripte lasen über einen fest verdrahteten
relativen Pfad direkt aus der Produktionsdatenbank (``sqlite3.connect``, ohne
Fixture), gaben ausschließlich ``True``/``False`` zurück und beendeten sich im
Fehlerfall trotzdem mit Exit-Code 0 — ein „stilles Grün“. Sie wurden außerdem
von pytest nie eingesammelt (Funktionsnamen ohne ``test_``-Präfix).

Jetzt: echte Assertions gegen eine temporär erzeugte Datenbank; die Skripte
akzeptieren DB- und Eingabepfad als Parameter bzw. über Umgebungsvariablen und
liefern einen echten Exit-Code.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from ai_agents.prepare_dataset_sql import NeuroAnalyzer
from tests._isolation_guard import snapshot_production, describe_snapshot_diff
from tests.validate_directive_003 import validate_directive_003
from tests.validate_directive_004 import validate_directive_004

REPO_ROOT = Path(__file__).resolve().parent.parent
DIRECTIVE_004_FILES = [
    "test_kick_004.wav",
    "test_bass_004.wav",
    "test_ambient_004.wav",
    "test_silent_004.wav",
]


def create_schema(db_path: Path) -> None:
    """Legt das echte ``stems``-Schema an (DDL aus dem Produktivcode)."""
    analyzer = NeuroAnalyzer(
        input_dir=str(db_path.parent),
        db_path=str(db_path),
        checkpoint_dir=str(db_path.parent / "checkpoints"),
        stems_dir=str(db_path.parent / "stems"),
        quarantine_dir=str(db_path.parent / "quarantine"),
    )
    analyzer.init_db()
    analyzer.conn.close()
    analyzer.conn = None


def insert_stem(
    db_path: Path,
    stem_id: str,
    category: str,
    tags: str = '["dark", "punchy", "driving"]',
    quality_ok: int = 1,
    imported_at: str = "2026-01-01T00:00:00",
    path: str = "raw_construction_kits/sample.wav",
) -> None:
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO stems
            (id, path, bpm, key, category, tags, features, quality_ok,
             user_rating, imported_at, clap_embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stem_id,
                path,
                128.0,
                None,
                category,
                tags,
                '{"spectral_centroid": 1000.0, "rms": 0.2}',
                quality_ok,
                None,
                imported_at,
                b"\x00" * 8,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def write_wav(path: Path) -> Path:
    """Kleine, aber gültige WAV-Datei (Inhalt ist für die Validierung egal)."""
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(struct.pack("<h", 0) * 2205)
    return path


# ----------------------------------------------------------------------
# validate_directive_003
# ----------------------------------------------------------------------
def test_directive_003_passes_on_valid_data(tmp_path: Path):
    """Drei passende Einträge erfüllen die Kriterien → True."""
    db_path = tmp_path / "stems.db"
    create_schema(db_path)
    insert_stem(db_path, "kick_1", "kick", imported_at="2026-01-03T00:00:00")
    insert_stem(db_path, "bass_1", "bass", imported_at="2026-01-02T00:00:00")
    insert_stem(db_path, "unknown_1", "unknown", imported_at="2026-01-01T00:00:00")

    assert validate_directive_003(db_path=str(db_path)) is True


def test_directive_003_fails_on_empty_database(tmp_path: Path):
    """Ohne Einträge ist die Validierung nicht erfüllt → False statt still grün."""
    db_path = tmp_path / "empty.db"
    create_schema(db_path)

    assert validate_directive_003(db_path=str(db_path)) is False


def test_directive_003_fails_on_incomplete_tags(tmp_path: Path):
    """Zu wenige Tags dürfen nicht als Erfolg durchgehen."""
    db_path = tmp_path / "bad_tags.db"
    create_schema(db_path)
    insert_stem(db_path, "kick_1", "kick", tags='["dark"]', imported_at="2026-01-03T00:00:00")
    insert_stem(db_path, "bass_1", "bass", tags='["dark"]', imported_at="2026-01-02T00:00:00")
    insert_stem(db_path, "unknown_1", "unknown", tags='["dark"]', imported_at="2026-01-01T00:00:00")

    assert validate_directive_003(db_path=str(db_path)) is False


def test_directive_003_fails_on_missing_database(tmp_path: Path):
    """Fehlende Datenbank ist ein Fehlschlag, kein stiller Erfolg."""
    assert validate_directive_003(db_path=str(tmp_path / "gibt-es-nicht.db")) is False


def test_directive_003_exit_code_is_nonzero_on_failure(tmp_path: Path):
    """Als Skript beendet sich die Validierung im Fehlerfall mit Exit-Code ≠ 0."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tests" / "validate_directive_003.py")],
        cwd=str(tmp_path),
        env={**os.environ, "NEUROMORPHE_DB_PATH": str(tmp_path / "fehlt.db")},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "nicht gefunden" in result.stdout


# ----------------------------------------------------------------------
# validate_directive_004
# ----------------------------------------------------------------------
def test_directive_004_passes_on_complete_data(tmp_path: Path):
    """Vollständige Testdaten erfüllen alle Kriterien → True."""
    db_path = tmp_path / "stems.db"
    input_dir = tmp_path / "input"
    create_schema(db_path)

    for name in DIRECTIVE_004_FILES:
        write_wav(input_dir / name)

    for index in range(9):
        insert_stem(
            db_path,
            f"filler_{index}",
            "percussion",
            imported_at=f"2025-12-0{index + 1}T00:00:00",
        )
    insert_stem(
        db_path,
        "kick_004",
        "kick",
        imported_at="2026-01-03T00:00:00",
        path="raw_construction_kits/test_kick_004.wav",
    )
    insert_stem(
        db_path,
        "bass_004",
        "bass",
        imported_at="2026-01-02T00:00:00",
        path="raw_construction_kits/test_bass_004.wav",
    )
    insert_stem(
        db_path,
        "ambient_004",
        "percussion",
        imported_at="2026-01-01T00:00:00",
        path="raw_construction_kits/test_ambient_004.wav",
    )

    assert validate_directive_004(db_path=str(db_path), input_dir=str(input_dir)) is True


def test_directive_004_fails_when_input_files_are_missing(tmp_path: Path):
    """Fehlende Test-Audiodateien → False (kein stilles Grün)."""
    db_path = tmp_path / "stems.db"
    create_schema(db_path)

    for index in range(12):
        insert_stem(db_path, f"filler_{index}", "percussion")

    assert validate_directive_004(db_path=str(db_path), input_dir=str(tmp_path / "leer")) is False


def test_directive_004_fails_on_missing_database(tmp_path: Path):
    assert validate_directive_004(db_path=str(tmp_path / "fehlt.db")) is False


def test_directive_004_exit_code_is_nonzero_on_failure(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tests" / "validate_directive_004.py")],
        cwd=str(tmp_path),
        env={**os.environ, "NEUROMORPHE_DB_PATH": str(tmp_path / "fehlt.db")},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0, result.stdout + result.stderr


# ----------------------------------------------------------------------
# Isolation
# ----------------------------------------------------------------------
def test_validators_never_touch_production_paths(tmp_path: Path):
    """Beide Validierungen laufen vollständig gegen temporäre Pfade.

    Vorher griffen sie direkt (ohne Fixture) auf
    ``processed_database/stems.db`` und ``raw_construction_kits/`` zu.
    """
    db_path = tmp_path / "stems.db"
    input_dir = tmp_path / "input"
    create_schema(db_path)
    for name in DIRECTIVE_004_FILES:
        write_wav(input_dir / name)
    for index in range(12):
        insert_stem(db_path, f"filler_{index}", "percussion")

    before = snapshot_production()
    validate_directive_003(db_path=str(db_path))
    validate_directive_004(db_path=str(db_path), input_dir=str(input_dir))
    after = snapshot_production()

    differences = describe_snapshot_diff(before, after)
    assert not differences, "Produktionsdaten verändert:\n" + "\n".join(differences)


def test_validators_honour_env_injection(tmp_path: Path):
    """``NEUROMORPHE_DB_PATH`` wird ohne Argument respektiert."""
    db_path = tmp_path / "stems.db"
    create_schema(db_path)
    insert_stem(db_path, "kick_1", "kick", imported_at="2026-01-03T00:00:00")
    insert_stem(db_path, "bass_1", "bass", imported_at="2026-01-02T00:00:00")
    insert_stem(db_path, "unknown_1", "unknown", imported_at="2026-01-01T00:00:00")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("NEUROMORPHE_DB_PATH", str(db_path))
        assert validate_directive_003() is True
    finally:
        monkeypatch.undo()
