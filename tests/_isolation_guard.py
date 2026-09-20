"""Sandbox-Guard für die Testsuite.

Zweck: Tests dürfen ausschließlich in temporäre Verzeichnisse schreiben.
Produktionsdaten (Datenbank, Checkpoints, Input/Output, Modelle, Cache) sind
für Testprozesse nicht schreibbar.

Das Modul stellt zwei voneinander unabhängige Schutzschichten bereit:

1. :class:`ProductionWriteGuard` — patcht die Schreib-Schnittstellen des
   Interpreters (``io.open``, ``os.*``, ``shutil.*``, ``sqlite3.connect``) und
   lässt jeden Schreibzugriff auf einen Produktionspfad sofort mit
   :class:`ProductionWriteViolation` fehlschlagen.
2. :func:`snapshot_paths` — Größen-/mtime-/Hash-Snapshot der Produktionsbäume.
   Damit kann ein Test beweisen, dass ein Lauf die Produktionsdaten nicht
   berührt hat (Sentinel-Beweis, unabhängig von Schicht 1).
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# Verzeichnisse mit Produktionsdaten (relativ zum Repo-Root).
PRODUCTION_TREES: Tuple[str, ...] = (
    "processed_database",
    "raw_construction_kits",
    "dataembeddings",
    "dataaudio",
    "rendered_tracks",
    "stereo_tracks_for_analysis",
    "generated_stems",
    "audio_files",
    "models",
    "logs",
    "test_data",
    "test_cache",
    "processing_logs",
)

# Einzelne Produktionsdateien (relativ zum Repo-Root).
PRODUCTION_FILES: Tuple[str, ...] = (
    "test_database.db",
    "neuro_analyzer.log",
    "embeddings.pkl",
)

# SQLite legt diese Begleitdateien neben der Datenbank an.
SQLITE_SIDECARS: Tuple[str, ...] = ("-wal", "-shm", "-journal")


class ProductionWriteViolation(AssertionError):
    """Ein Test wollte in einen Produktionspfad schreiben."""


def repo_root() -> Path:
    """Repo-Root (zwei Ebenen über diesem Modul)."""
    return Path(__file__).resolve().parent.parent


def _dir_fd_base(dir_fd: Any) -> Optional[Path]:
    """Basisverzeichnis eines ``dir_fd``-Arguments auflösen (Linux: /proc)."""
    if dir_fd is None:
        return None
    try:
        return Path(os.readlink(f"/proc/self/fd/{int(dir_fd)}"))
    except (OSError, ValueError):  # pragma: no cover - exotische Umgebungen
        return None


def _candidates(target: Any, root: Path, dir_fd: Any = None) -> List[Path]:
    """Alle plausiblen Auflösungen eines Ziels.

    Relative Pfade werden sowohl gegen das aktuelle Arbeitsverzeichnis als
    auch gegen das Repo-Root aufgelöst — je nachdem, wo der Testprozess
    gestartet wurde, kann beides die Produktionsdatei treffen. Bei relativem
    Pfad *und* gesetztem ``dir_fd`` ist allein der ``dir_fd`` maßgeblich.
    """
    if target is None or isinstance(target, int):
        return []
    try:
        raw = Path(os.fspath(target))
    except TypeError:
        return []

    fd_base = _dir_fd_base(dir_fd)
    if fd_base is not None and not raw.is_absolute():
        try:
            return [(fd_base / raw).resolve()]
        except OSError:  # pragma: no cover
            return [(fd_base / raw).absolute()]

    candidates: List[Path] = []
    for base in (None, Path.cwd(), root):
        candidate = raw if (base is None or raw.is_absolute()) else base / raw
        try:
            candidates.append(candidate.resolve())
        except OSError:  # pragma: no cover - z. B. defekter Symlink
            candidates.append(candidate.absolute())

    # Duplikate entfernen, Reihenfolge behalten
    unique: List[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def match_production_path(
    target: Any, root: Optional[Path] = None, dir_fd: Any = None
) -> Optional[Path]:
    """Liefert den getroffenen Produktionspfad oder ``None``.

    Geprüft werden nur Pfade innerhalb des Repo-Roots. Alles außerhalb
    (``/tmp``, pytest-``tmp_path``, Sandbox-Wurzel) gilt als temporär.
    """
    root = root or repo_root()
    trees = [(root / name).resolve() for name in PRODUCTION_TREES]
    files = [(root / name).resolve() for name in PRODUCTION_FILES]
    sidecar_names = {f"{f.name}{suffix}" for f in files for suffix in SQLITE_SIDECARS}

    for candidate in _candidates(target, root, dir_fd):
        for tree in trees:
            if candidate == tree or tree in candidate.parents:
                return candidate
        for file_path in files:
            if candidate == file_path:
                return candidate
        # Noch nicht existierende Begleitdateien (-wal/-shm/-journal) im Root
        if candidate.parent == root and candidate.name in sidecar_names:
            return candidate
    return None


class ProductionWriteGuard:
    """Harte Sperre gegen Schreibzugriffe auf Produktionspfade."""

    #: APIs, die als schreibend gelten und einzeln geprüft werden
    WRITE_APIS: Tuple[str, ...] = (
        "io.open",
        "builtins.open",
        "os.open",
        "os.mkdir",
        "os.makedirs",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.rename",
        "os.replace",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.move",
        "shutil.rmtree",
        "sqlite3.connect",
    )

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or repo_root()).resolve()
        self.violations: List[str] = []
        self._installed: List[Tuple[str, Any]] = []

    # ------------------------------------------------------------------
    # Prüfung
    # ------------------------------------------------------------------
    def check(self, api: str, *targets: Any, dir_fd: Any = None) -> None:
        """Schlägt hart fehl, wenn eines der Ziele im Produktionsbereich liegt."""
        for target in targets:
            hit = match_production_path(target, self.root, dir_fd)
            if hit is None:
                continue
            message = (
                f"Test-Isolation verletzt: {api}() wollte den Produktionspfad "
                f"'{hit}' schreiben. Tests müssen in temporäre Verzeichnisse "
                f"schreiben (tmp_path/Sandbox)."
            )
            self.violations.append(message)
            raise ProductionWriteViolation(message)

    def assert_clean(self) -> None:
        """Sammelprüfung am Sitzungsende."""
        assert not self.violations, "Produktionsschreibzugriffe:\n" + "\n".join(self.violations)

    @contextmanager
    def expected_violation(self) -> Iterator[None]:
        """Negative Tests: erwartete Verstöße werden nicht als Sitzungsfehler gewertet."""
        recorded = len(self.violations)
        try:
            yield
        finally:
            del self.violations[recorded:]

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------
    def install(self) -> None:
        """Alle Schreib-APIs durch geprüfte Varianten ersetzen (idempotent)."""
        if self._installed:
            return

        def replace(api: str, wrapper: Any) -> None:
            module_name, _, attribute = api.rpartition(".")
            module = {
                "io": io,
                "os": os,
                "shutil": shutil,
                "sqlite3": sqlite3,
                "builtins": __import__("builtins"),
            }[module_name]
            self._installed.append((api, getattr(module, attribute)))
            setattr(module, attribute, wrapper)

        original_open = io.open

        def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            effective_mode = kwargs.get("mode", mode)
            if isinstance(effective_mode, str) and any(flag in effective_mode for flag in "wax+"):
                self.check("io.open", file)
            return original_open(file, mode, *args, **kwargs)

        replace("io.open", guarded_open)
        # ``builtins.open`` verweist auf dasselbe Funktionsobjekt, muss aber
        # separat ersetzt werden — sonst greift der Guard bei ``open(...)`` nicht.
        replace("builtins.open", guarded_open)

        original_os_open = os.open

        def guarded_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> Any:
            write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
            if flags & write_flags:
                self.check("os.open", path, dir_fd=kwargs.get("dir_fd"))
            return original_os_open(path, flags, *args, **kwargs)

        replace("os.open", guarded_os_open)

        def single_path(name: str, original: Any) -> Any:
            def guarded(path: Any, *args: Any, **kwargs: Any) -> Any:
                self.check(name, path, dir_fd=kwargs.get("dir_fd"))
                return original(path, *args, **kwargs)

            return guarded

        replace("os.mkdir", single_path("os.mkdir", os.mkdir))
        replace("os.makedirs", single_path("os.makedirs", os.makedirs))
        replace("os.remove", single_path("os.remove", os.remove))
        replace("os.unlink", single_path("os.unlink", os.unlink))
        replace("os.rmdir", single_path("os.rmdir", os.rmdir))
        replace("shutil.rmtree", single_path("shutil.rmtree", shutil.rmtree))

        for name, original in (
            ("os.rename", os.rename),
            ("os.replace", os.replace),
            ("shutil.move", shutil.move),
            ("shutil.copy", shutil.copy),
            ("shutil.copy2", shutil.copy2),
            ("shutil.copyfile", shutil.copyfile),
        ):
            def two_paths(name: str = name, original: Any = original) -> Any:
                def guarded(source: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
                    self.check(name, source, destination)
                    return original(source, destination, *args, **kwargs)

                return guarded

            replace(name, two_paths())

        original_connect = sqlite3.connect

        def guarded_connect(database: Any, *args: Any, **kwargs: Any) -> Any:
            # Auch ein reiner Lesezugriff kann schreiben (CREATE TABLE, Journal).
            self.check("sqlite3.connect", database)
            return original_connect(database, *args, **kwargs)

        replace("sqlite3.connect", guarded_connect)

    def uninstall(self) -> None:
        """Ursprüngliche APIs wiederherstellen."""
        modules = {
            "io": io,
            "os": os,
            "shutil": shutil,
            "sqlite3": sqlite3,
            "builtins": __import__("builtins"),
        }
        while self._installed:
            api, original = self._installed.pop()
            module_name, _, attribute = api.rpartition(".")
            setattr(modules[module_name], attribute, original)


# ----------------------------------------------------------------------
# Sentinel-/Snapshot-Helfer
# ----------------------------------------------------------------------
def production_paths(root: Optional[Path] = None) -> List[Path]:
    """Alle Produktionsbäume und -dateien als absolute Pfade."""
    root = root or repo_root()
    paths = [root / name for name in PRODUCTION_TREES]
    paths += [root / name for name in PRODUCTION_FILES]
    return paths


def iter_production_files(root: Optional[Path] = None) -> List[Path]:
    """Alle vorhandenen Dateien in den Produktionsbäumen (rekursiv)."""
    files: List[Path] = []
    for path in production_paths(root):
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
    return files


def snapshot_paths(paths: Iterable[Path]) -> Dict[str, Tuple[int, int, str]]:
    """Snapshot aus (Größe, mtime_ns, sha256) je Datei."""
    snapshot: Dict[str, Tuple[int, int, str]] = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot[str(path)] = (stat.st_size, stat.st_mtime_ns, digest)
    return snapshot


def snapshot_production(root: Optional[Path] = None) -> Dict[str, Tuple[int, int, str]]:
    """Snapshot aller vorhandenen Produktionsdateien."""
    return snapshot_paths(iter_production_files(root))


def describe_snapshot_diff(
    before: Dict[str, Tuple[int, int, str]],
    after: Dict[str, Tuple[int, int, str]],
) -> List[str]:
    """Menschlesbare Unterschiede zwischen zwei Snapshots."""
    differences: List[str] = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            differences.append(f"NEU:      {path}")
        elif path not in after:
            differences.append(f"GELÖSCHT: {path}")
        elif before[path] != after[path]:
            differences.append(
                f"GEÄNDERT: {path} (vorher {before[path]}, nachher {after[path]})"
            )
    return differences


def sqlite_sidecar_paths(db_path: Path) -> List[Path]:
    """Mögliche WAL-/SHM-/Journal-Begleitdateien einer SQLite-Datenbank."""
    return [Path(f"{db_path}{suffix}") for suffix in SQLITE_SIDECARS]
