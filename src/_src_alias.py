"""Alias-Registrierung: ``src.X`` und ``X`` als dasselbe Modul.

Das Projekt importiert intern aus ``src/`` (also ``services.renderer``,
``database.database``), die Test-Suite patcht dagegen die Pfade
``src.services.renderer``, ``src.database.manager`` usw. Python behandelt
``services.renderer`` und ``src.services.renderer`` als zwei verschiedene
Module, obwohl beide auf dieselbe Datei zeigen — ein ``patch()`` auf den
einen Pfad bliebe für den anderen unsichtbar.

Dieses Modul registriert beide Schreibweisen in ``sys.modules`` unter
demselben Modulobjekt. Es wird importiert, sobald die Test-Suite lädt
(über ``tests/conftest.py``), und ist idempotent.
"""

import importlib
import sys

# (Kurzname, Modulpfad relativ zu src/)
_MODULES = (
    "services.arranger",
    "services.generative_service",
    "services.live_player_service",
    "services.neuro_analyzer",
    "services.preprocessor",
    "services.renderer",
    "services.search",
    "services.separation_service",
    "services.training_service",
    "database.database",
    "database.manager",
    "database.models",
    "database.crud",
    "database.service",
    "core.config",
    "core.logging",
    "core.utils",
    "schemas.stem",
    "schemas.render",
    "schemas.arrangement",
    "schemas.websocket",
    "audio.service",
    "audio.router",
    "main",
)


def register_src_aliases() -> None:
    """Registriert ``src.<modul>`` als Alias für ``<modul>``."""
    for name in _MODULES:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:  # noqa: BLE001 - fehlende Optional-Module sind ok
                continue
        # Unter dem src-Pfad denselben Modulnamen auflösen
        sys.modules.setdefault(f"src.{name}", module)
        # Zwischenpakete (src.services, src.database, ...) ebenfalls verknüpfen
        parts = name.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            pkg = sys.modules.get(prefix)
            if pkg is not None:
                sys.modules.setdefault(f"src.{prefix}", pkg)


register_src_aliases()
