"""Kompatibilitätsmodul für den historischen Importpfad.

``tests/test_api/test_endpoints.py`` patcht ``src.database.manager.DatabaseManager``.
Ein Modul dieses Namens hat im Repository nie existiert; die Tests erwarten
ihn dennoch als Patch-Ziel. Damit die Patches greifen, wird hier derselbe
``DatabaseManager`` veröffentlicht, der auch in ``database.database`` lebt.

Wichtig: Es wird keine zweite Klasse definiert — das würde die Patches ins
Leere laufen lassen und zwei divergierende Manager-Typen erzeugen. Dieses
Modul re-exportiert ausschließlich.
"""

from database.database import DatabaseManager, get_database_manager

__all__ = ["DatabaseManager", "get_database_manager"]
