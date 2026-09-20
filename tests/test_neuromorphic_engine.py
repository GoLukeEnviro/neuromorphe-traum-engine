"""End-to-End-Tests der Neuromorphen Traum-Engine (Service- und API-Ebene).

Ausgangslage (Defekt): ``test_neuromorphic_engine`` und ``test_api_endpoints``
waren reine Statusdrucker. Jeder Fehler endete in ``print(...)`` plus
``return``; der Test war damit immer grün („stilles Grün“). Im Lauf vor dieser
Änderung kehrte ``test_neuromorphic_engine`` bereits beim ersten
``return`` zurück, ohne eine einzige Zusicherung zu prüfen.
``test_api_endpoints`` verlangte zusätzlich einen laufenden Server auf Port
8000 und schluckte alle Ausnahmen.

Jetzt: echte Assertions gegen die Sandbox (Datenbank, Modelle, Ausgabepfade);
der API-Teil läuft gegen den ``TestClient`` — kein Live-Server, keine
Netzabhängigkeit.
"""

import asyncio
from pathlib import Path

import pytest

from core.config import settings
from database.database import get_database_manager
from database.service import DatabaseService
from schemas.stem import StemCreate
from services.generative_service import GenerativeService
from services.separation_service import SeparationService
from services.training_service import TrainingService

REPO_ROOT = Path(__file__).resolve().parent.parent


def _assert_inside_sandbox(path: Path, sandbox_root: Path, label: str) -> None:
    resolved = Path(path).resolve()
    assert sandbox_root.resolve() in resolved.parents or resolved == sandbox_root.resolve(), (
        f"{label} zeigt nicht in die Sandbox: {resolved}"
    )
    assert REPO_ROOT not in resolved.parents, f"{label} zeigt in die Produktion: {resolved}"


# ----------------------------------------------------------------------
# Datenbank
# ----------------------------------------------------------------------
def test_database_manager_is_bound_to_sandbox(sandbox_root: Path):
    """Der globale Datenbank-Manager zeigt auf die Sandbox, nicht auf Produktion."""
    manager = get_database_manager()

    _assert_inside_sandbox(
        Path(manager.database_url.replace("sqlite:///", "")), sandbox_root, "DATABASE_URL"
    )
    assert Path(manager.database_url.replace("sqlite:///", "")).resolve() != (
        REPO_ROOT / "processed_database" / "stems.db"
    ).resolve()


def test_database_service_reads_are_bound_to_sandbox(sandbox_root: Path):
    """Lesezugriffe des DatabaseService laufen gegen die Sandbox-Datenbank."""
    service = DatabaseService()

    async def scenario():
        stats = await service.get_stem_statistics()
        assert isinstance(stats, dict)
        assert {"total_stems", "processed_stems", "category_distribution"} <= set(stats)
        assert isinstance(stats["total_stems"], int)
        assert stats["total_stems"] >= 0

        categories = await service.get_stem_categories()
        assert isinstance(categories, list)

        count = await service.get_stem_count()
        assert isinstance(count, int)
        assert count >= 0

        service.cleanup()

    asyncio.run(scenario())


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason=(
        "Bekannter Schema-Drift: StemCreate enthält 'title'/'key', das ORM-Modell "
        "Stem nicht (dort 'musical_key'). StemCRUD.create_stem reicht die Felder "
        "ungefiltert an Stem(**...) weiter — insert_stem ist dadurch funktionsunfähig. "
        "Nicht Teil dieses Auftrags (Test-Isolation), aber hier sichtbar statt still grün."
    ),
)
def test_database_service_insert_stem_is_blocked_by_schema_drift():
    """Dokumentiert den Defekt: ``insert_stem`` scheitert an unbekannten Feldern."""
    service = DatabaseService()

    async def scenario():
        created = await service.insert_stem(
            StemCreate(filename="test_kick_engine.wav", category="kick")
        )
        assert created is not None

    try:
        asyncio.run(scenario())
    finally:
        service.cleanup()


# ----------------------------------------------------------------------
# Services
# ----------------------------------------------------------------------
def test_separation_service_reports_formats_and_rejects_missing_file(tmp_path: Path):
    """Fehlende Eingaben scheitern sofort — und laden kein Demucs-Modell."""
    service = SeparationService()

    formats = service.get_supported_formats()
    assert formats, "Es werden keine unterstützten Formate gemeldet"
    assert all(isinstance(item, str) for item in formats)

    with pytest.raises(FileNotFoundError):
        asyncio.run(service.separate_track(str(tmp_path / "gibt-es-nicht.wav")))

    assert service.model is None, (
        "Das Demucs-Modell darf vor der Existenzprüfung nicht geladen werden"
    )
    service.cleanup()


def test_training_service_uses_sandbox_model_dir(sandbox_root: Path):
    """Modelle landen im injizierten Modellverzeichnis, nicht in ``./models``."""
    service = TrainingService()

    _assert_inside_sandbox(service.models_dir, sandbox_root, "models_dir")
    _assert_inside_sandbox(Path(settings.MODEL_CACHE_DIR), sandbox_root, "MODEL_CACHE_DIR")
    assert service.models_dir.is_dir()
    assert service.get_available_models() == []


def test_generative_service_uses_sandbox_output_dirs(sandbox_root: Path):
    """Generierte Stems landen im injizierten Ausgabeverzeichnis."""
    service = GenerativeService()

    _assert_inside_sandbox(service.models_dir, sandbox_root, "models_dir")
    _assert_inside_sandbox(service.generated_stems_dir, sandbox_root, "generated_stems_dir")
    assert service.generated_stems_dir.is_dir()
    assert service.get_generated_stems_info() == []


# ----------------------------------------------------------------------
# API (TestClient statt Live-Server)
# ----------------------------------------------------------------------
def test_api_health_endpoint(test_client):
    """Der Health-Endpunkt antwortet mit 200 und einem Statusfeld."""
    response = test_client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload, "Health-Antwort ist leer"


def test_api_stems_endpoint_lists_stems(test_client):
    """``/api/v1/stems/`` liefert eine Liste — gegen die Test-Datenbank."""
    response = test_client.get("/api/v1/stems/")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, (dict, list))


def test_api_documented_post_endpoints_exist(test_client):
    """Die dokumentierten POST-Endpunkte existieren (422/400 statt 404)."""
    endpoints = [
        "/api/v1/neuromorphic/preprocess",
        "/api/v1/neuromorphic/train",
        "/api/v1/neuromorphic/generate",
    ]

    for endpoint in endpoints:
        response = test_client.post(endpoint)
        assert response.status_code != 404, f"{endpoint} existiert nicht"
        assert response.status_code in (400, 422, 405), (
            f"{endpoint} antwortet unerwartet: {response.status_code}"
        )


def test_api_arrangements_endpoint_works(test_client):
    """Arrangements lassen sich auflisten (leere Liste ist gültig)."""
    response = test_client.get("/api/v1/arrangements?page=1&per_page=10")

    assert response.status_code == 200
    assert isinstance(response.json(), (dict, list))
