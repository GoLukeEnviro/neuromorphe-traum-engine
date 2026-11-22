"""Indiziert eine Sample-Bibliothek: CLAP-Embedding berechnen und in die DB schreiben.

Aufruf:
    PYTHONPATH=src .venv/bin/python scripts/import_library.py [--dir PFAD] [--limit N]

Liest optional ``manifest.csv`` im Zielordner, um Kategorie, Quelle und Lizenz
zu übernehmen. Ohne Manifest wird die Kategorie aus dem Elternordner abgeleitet.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _load_manifest(folder: Path) -> dict:
    manifest = folder / "manifest.csv"
    if not manifest.exists():
        return {}
    out = {}
    with manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[str(Path(row["path"]).resolve())] = row
    return out


async def run(folder: Path, limit: int | None) -> None:
    from core.logging import setup_logging  # noqa: F401
    from database.database import get_database_manager, create_tables
    from search.service import SearchService

    await create_tables()
    manager = get_database_manager()
    service = SearchService()

    files = sorted(folder.rglob("*.wav"))
    if limit:
        files = files[:limit]

    manifest = _load_manifest(folder)
    print(f"Dateien: {len(files)}")
    print(f"Manifest-Einträge: {len(manifest)}")

    # Service EINMAL instanziieren — sonst lädt jede Datei das CLAP-Modell neu.
    from audio.service import AudioProcessingService

    processor = AudioProcessingService(audio_dir=str(folder))
    await processor._generate_clap_embedding(files[0])  # Modell vorwärmen
    print("CLAP-Modell geladen.")

    ok = failed = skipped = 0
    started = time.time()

    for index, path in enumerate(files, 1):
        key = str(path.resolve())
        meta = manifest.get(key, {})
        category = meta.get("category") or path.parent.name
        bpm_raw = meta.get("bpm") or ""
        try:
            bpm = float(bpm_raw) if bpm_raw else None
        except ValueError:
            bpm = None

        # Bereits indiziert? Direkt in der DB prüfen — search_stems liefert
        # nur IDs, keine Datensätze.
        async with manager.get_async_session() as session:
            from sqlalchemy import text as _text

            hit = (
                await session.execute(
                    _text("SELECT 1 FROM stems WHERE filename = :fn LIMIT 1"),
                    {"fn": path.name},
                )
            ).first()
        if hit:
            skipped += 1
            print(f"[{index}/{len(files)}] übersprungen (vorhanden): {path.name}")
            continue

        try:
            info = sf.info(str(path))
            embedding = await service.generate_text_embedding(path.name) if hasattr(
                service, "generate_text_embedding"
            ) else None
        except Exception:
            info = None
            embedding = None

        # Audio-Embedding über denselben Pfad, den auch der Upload nutzt.
        try:
            vector = await processor._generate_clap_embedding(path)
            arr = np.asarray(vector, dtype=np.float32).ravel() if vector is not None else None
        except Exception as exc:
            print(f"[{index}/{len(files)}] FEHLER {path.name}: {exc}")
            failed += 1
            continue

        if arr is None or arr.size == 0 or not np.all(np.isfinite(arr)):
            print(f"[{index}/{len(files)}] kein brauchbares Embedding: {path.name}")
            failed += 1
            continue

        payload = {
            "filename": path.name,
            "original_path": str(path),
            "processed_path": str(path),
            "category": category,
            "bpm": bpm,
            "duration": float(info.duration) if info else 0.0,
            "sample_rate": int(info.samplerate) if info else 44100,
            "channels": int(info.channels) if info else 2,
            "file_size": path.stat().st_size,
            "audio_embedding": arr.tolist(),
            "processing_status": "completed",
            "source": meta.get("source", "library"),
            "manual_tags": [category] if category else [],
        }
        try:
            stem_id = await manager.create_stem(payload)
            ok += 1
            elapsed = time.time() - started
            rate = elapsed / max(1, ok)
            remaining = rate * (len(files) - index)
            print(
                f"[{index}/{len(files)}] {path.name} -> {stem_id} "
                f"({arr.size}d, {elapsed:.0f}s, ~{remaining:.0f}s übrig)"
            )
        except Exception as exc:
            print(f"[{index}/{len(files)}] DB-FEHLER {path.name}: {exc}")
            failed += 1

    print("\n=== ZUSAMMENFASSUNG ===")
    print(f"  importiert:  {ok}")
    print(f"  übersprungen:{skipped}")
    print(f"  fehlerhaft:  {failed}")
    print(f"  Dauer:       {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="raw_construction_kits/library")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(Path(args.dir), args.limit))


if __name__ == "__main__":
    main()
