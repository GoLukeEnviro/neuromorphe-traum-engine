"""Sentinel-/Sandbox-Beweis für die Test-Isolation.

Dieser Test beweist in einem Durchlauf, dass die kritischen Schreibpfade der
Anwendung während der Tests **ausschließlich** in temporäre Verzeichnisse
schreiben und die Produktionsdaten unberührt bleiben:

1. Snapshot aller Produktionsbäume (Größe + mtime + SHA-256 je Datei).
2. Ausführen der Risiko-Pfade der Suite — inklusive der zuvor fehlerhaften
   Zugriffe: ``NeuroAnalyzer`` (Legacy-``sqlite3``-Zugriff, doppeltes
   ``init_db()``, Checkpoints, Stems, Quarantäne), ``PreprocessorService``
   (Kategorie-Kopie, Quarantäne) und ein schreibender API-Aufruf.
3. Nachweis **positiver** Evidenz: die Schreibvorgänge haben tatsächlich
   stattgefunden — nur eben innerhalb der Sandbox. Ohne diese Prüfung wäre der
   Sentinel-Test vakuum-grün.
4. Vergleich der Snapshots sowie explizite Prüfung auf WAL-/SHM-/Journal- und
   Checkpoint-Dateien in den Produktionspfaden.

Zusätzlich: der Sandbox-Guard wird negativ getestet (er muss direkte
Produktionszugriffe tatsächlich abweisen), damit der Beweis nicht auf einem
wirkungslosen Guard beruht.
"""

import asyncio
import os
import sqlite3
import wave
from pathlib import Path

import numpy as np
import pytest

from core.config import settings
from tests._isolation_guard import (
    ProductionWriteViolation,
    describe_snapshot_diff,
    match_production_path,
    snapshot_production,
    sqlite_sidecar_paths,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_DB = REPO_ROOT / "processed_database" / "stems.db"
PRODUCTION_DIRS = [
    "processed_database",
    "raw_construction_kits",
    "models",
    "logs",
    "dataembeddings",
    "rendered_tracks",
    "generated_stems",
    "stereo_tracks_for_analysis",
]


def write_wav(path: Path, seconds: float = 1.0, sample_rate: int = 22050) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return path


def listing(root: Path) -> set:
    return {str(p.relative_to(root)) for p in root.rglob("*")} if root.is_dir() else set()


# ----------------------------------------------------------------------
# Grundlagen
# ----------------------------------------------------------------------
def test_sandbox_root_is_outside_the_repository(sandbox_root: Path):
    """Die Sandbox darf nicht im Repository liegen — sonst wäre der Beweis wertlos."""
    resolved = sandbox_root.resolve()
    assert resolved != REPO_ROOT
    assert REPO_ROOT not in resolved.parents
    assert resolved.is_dir()


def test_all_settings_paths_point_into_the_sandbox(sandbox_root: Path):
    """Alle Pfad-Einstellungen der Anwendung sind in die Sandbox injiziert."""
    sandbox = sandbox_root.resolve()
    values = {
        "DATABASE_URL": Path(settings.DATABASE_URL.replace("sqlite:///", "")),
        "UPLOAD_DIR": Path(settings.UPLOAD_DIR),
        "PROCESSED_STEMS_DIR": Path(settings.PROCESSED_STEMS_DIR),
        "EMBEDDINGS_DIR": Path(settings.EMBEDDINGS_DIR),
        "RENDERED_TRACKS_DIR": Path(settings.RENDERED_TRACKS_DIR),
        "GENERATED_STEMS_DIR": Path(settings.GENERATED_STEMS_DIR),
        "STEREO_TRACKS_DIR": Path(settings.STEREO_TRACKS_DIR),
        "MODEL_CACHE_DIR": Path(settings.MODEL_CACHE_DIR),
        "LOGS_DIR": Path(settings.LOGS_DIR),
    }

    for name, value in values.items():
        resolved = value.resolve()
        assert sandbox in resolved.parents or resolved == sandbox, (
            f"{name} zeigt nicht in die Sandbox: {resolved}"
        )
        assert REPO_ROOT not in resolved.parents, f"{name} zeigt in die Produktion: {resolved}"

    # Vorab-Defaults der Anwendung dürfen nicht mehr aktiv sein.
    configured_db = Path(settings.DATABASE_URL.replace("sqlite:///", "")).resolve()
    assert configured_db != PRODUCTION_DB.resolve()


# ----------------------------------------------------------------------
# Sentinel-Beweis
# ----------------------------------------------------------------------
def test_risky_write_paths_touch_only_sandbox(sandbox_root: Path, tmp_path: Path, test_client):
    """Die kritischen Schreibpfade der Suite lassen Produktionsdaten unverändert."""
    sandbox = sandbox_root.resolve()
    legacy_input = sandbox / "raw_construction_kits"
    legacy_input.mkdir(parents=True, exist_ok=True)
    wav_path = write_wav(legacy_input / "test_kick_sample.wav")

    production_before = snapshot_production()
    sidecars = sqlite_sidecar_paths(PRODUCTION_DB)
    sidecars_before = {str(path): path.exists() for path in sidecars}
    checkpoints_dir = REPO_ROOT / "processed_database" / "checkpoints"
    checkpoints_before = listing(checkpoints_dir)

    # --- Risiko-Pfade ---------------------------------------------------
    from ai_agents.prepare_dataset_sql import NeuroAnalyzer
    from services.preprocessor import PreprocessorService

    # 1) Legacy-Analyzer ohne Pfadargumente: muss über die Umgebung in der
    #    Sandbox landen (vorher: fester relativer Produktionspfad).
    legacy_analyzer = NeuroAnalyzer(input_dir=str(legacy_input))
    assert sandbox in Path(legacy_analyzer.db_path).resolve().parents
    assert sandbox in Path(legacy_analyzer.checkpoint_dir).resolve().parents
    legacy_analyzer.init_db()
    legacy_analyzer.init_db()  # doppelter Aufruf (früherer Defekt)
    legacy_analyzer.run()
    legacy_analyzer._save_checkpoint()
    legacy_analyzer._quarantine_file(str(wav_path), "isolation_probe")
    if getattr(legacy_analyzer, "conn", None) is not None:
        legacy_analyzer.conn.close()
        legacy_analyzer.conn = None

    # 2) Analyzer mit explizit injizierten tmp_path-Pfaden.
    injected_db = tmp_path / "injected" / "stems.db"
    injected_analyzer = NeuroAnalyzer(
        input_dir=str(tmp_path / "injected_input"),
        db_path=str(injected_db),
        checkpoint_dir=str(tmp_path / "injected" / "checkpoints"),
        stems_dir=str(tmp_path / "injected" / "stems"),
        quarantine_dir=str(tmp_path / "injected" / "quarantine"),
        checkpoint_interval=1,
    )
    (tmp_path / "injected_input").mkdir(parents=True, exist_ok=True)
    write_wav(tmp_path / "injected_input" / "test_bass_line.wav")
    injected_analyzer.init_db()
    injected_analyzer.run()
    if getattr(injected_analyzer, "conn", None) is not None:
        injected_analyzer.conn.close()
        injected_analyzer.conn = None

    # 3) PreprocessorService nutzt das globale Settings-Singleton.
    preprocessor = PreprocessorService()
    copied = asyncio.run(preprocessor._copy_to_processed_dir(str(wav_path), "kick"))
    asyncio.run(preprocessor._quarantine_file(str(wav_path), "isolation_probe"))

    # 4) Schreibender API-Aufruf über den TestClient (isolierter Manager).
    response = test_client.post(
        "/api/v1/arrangements",
        json={"prompt": "driving industrial techno 138 bpm", "duration": 180, "genre": "techno"},
    )
    api_status = response.status_code

    # --- Positive Evidenz: es wurde wirklich geschrieben, aber in der Sandbox ---
    assert sandbox in Path(copied).resolve().parents, f"Kopie nicht in der Sandbox: {copied}"
    assert Path(copied).is_file()
    assert (sandbox / "processed_database" / "stems.db").is_file(), "Sandbox-DB fehlt"
    assert (sandbox / "processed_database" / "checkpoints" / "progress.json").is_file()
    assert list((sandbox / "processed_database" / "quarantine").glob("*")), "Quarantäne leer"
    stem_copies = list(Path(settings.PROCESSED_STEMS_DIR).rglob(Path(copied).name))
    assert stem_copies, "Stem-Kopie fehlt im Sandbox-Verzeichnis"
    assert injected_db.is_file(), "Injizierte DB wurde nicht angelegt"
    assert (tmp_path / "injected" / "checkpoints" / "progress.json").is_file()
    assert api_status in (200, 201, 202), f"API-Schreibzugriff fehlgeschlagen: {api_status}"

    sandbox_connection = sqlite3.connect(str(Path(legacy_analyzer.db_path)))
    try:
        stem_rows = sandbox_connection.execute("SELECT COUNT(*) FROM stems").fetchone()[0]
    finally:
        sandbox_connection.close()
    assert stem_rows >= 1, "Legacy-Analyzer hat nichts in die Sandbox-DB geschrieben"
    assert legacy_analyzer.stats["processed"] >= 1

    # --- Sentinel: Produktionsdaten unverändert -------------------------
    production_after = snapshot_production()
    differences = describe_snapshot_diff(production_before, production_after)
    assert not differences, "Produktionsdaten wurden verändert:\n" + "\n".join(differences)

    # WAL-/SHM-/Journal-Dateien dürfen in Produktionspfaden weder entstehen noch
    # verschwinden.
    for path in sidecars:
        assert path.exists() == sidecars_before[str(path)], f"Sidecar-Datei verändert: {path}"

    # Checkpoint-Verzeichnis der Produktion bleibt unverändert.
    assert listing(checkpoints_dir) == checkpoints_before


def test_production_database_content_is_unchanged_by_the_suite():
    """Belegt mitten in der Suite, dass die Produktionsdaten unangetastet sind.

    Existiert (noch) keine Produktionsdatenbank — etwa in einem frischen
    Checkout —, wird geprüft, dass auch keine entsteht. Der Test bleibt damit in
    beiden Umgebungen scharf statt still zu überspringen.
    """
    before = snapshot_production()

    # Zugriff, der früher auf die Produktionsdatenbank zielte.
    from services.preprocessor import PreprocessorService

    service = PreprocessorService()
    assert service.processed_dir == Path(settings.PROCESSED_STEMS_DIR)

    after = snapshot_production()
    differences = describe_snapshot_diff(before, after)
    assert not differences, "Produktionsdaten verändert:\n" + "\n".join(differences)

    if str(PRODUCTION_DB) in before:
        db_size, db_mtime, db_hash = before[str(PRODUCTION_DB)]
        assert len(db_hash) == 64
        assert after[str(PRODUCTION_DB)] == (db_size, db_mtime, db_hash)
    else:
        assert str(PRODUCTION_DB) not in after, (
            "Der Testlauf hat eine Produktionsdatenbank angelegt"
        )


# ----------------------------------------------------------------------
# Negativ-Test des Guards: die Sperre muss wirklich greifen
# ----------------------------------------------------------------------
def test_guard_blocks_direct_production_writes(production_write_guard, tmp_path: Path):
    """Direkte Produktionszugriffe werden hart abgewiesen (kein wirkungsloser Guard)."""
    guard = production_write_guard

    # Klassifikation
    assert match_production_path(PRODUCTION_DB) is not None
    assert match_production_path("processed_database/stems.db") is not None
    assert match_production_path(REPO_ROOT / "logs" / "x.log") is not None
    assert match_production_path(REPO_ROOT / "models" / "vae.pt") is not None
    assert match_production_path(f"{PRODUCTION_DB}-wal") is not None
    assert match_production_path(tmp_path / "frei.db") is None
    assert match_production_path("/tmp/anything/x.wav") is None

    with guard.expected_violation():
        with pytest.raises(ProductionWriteViolation):
            open(PRODUCTION_DB, "a").close()

        with pytest.raises(ProductionWriteViolation):
            sqlite3.connect(str(PRODUCTION_DB))

        with pytest.raises(ProductionWriteViolation):
            (REPO_ROOT / "logs").mkdir(exist_ok=True)

        with pytest.raises(ProductionWriteViolation):
            (REPO_ROOT / "models" / "schaden.pt").write_text("boom")

        with pytest.raises(ProductionWriteViolation):
            os.makedirs(REPO_ROOT / "generated_stems" / "neu", exist_ok=True)

    # Gegenprobe: Schreiben in temporäre Verzeichnisse bleibt erlaubt.
    allowed = tmp_path / "erlaubt.txt"
    allowed.write_text("ok")
    assert allowed.read_text() == "ok"

    assert guard.violations == [], "Guard hat unerwartete Verstöße protokolliert"
