"""Lädt CC0-Samples von Freesound und legt sie in der Bibliothek ab.

Voraussetzung: ein Freesound-API-Token (kostenlos).
  https://freesound.org/apiv2/apply/

Aufruf:
    export FREESOUND_API_KEY=...
    PYTHONPATH=src .venv/bin/python scripts/fetch_freesound.py --limit 300

Sucht gezielt nach Techno-Elementen, filtert hart auf CC0 und lädt nur
Originaldateien im WAV-Format. Jede Datei landet unter
``raw_construction_kits/library/<kategorie>/`` und bekommt einen
Manifest-Eintrag mit Quelle und Lizenz — damit später nachvollziehbar ist,
woher jeder Stem stammt.

Nur CC0. Keine "royalty-free"-Grauzonen: Sample-Packs, die kommerzielle
Nutzung erlauben, aber Weiterverbreitung verbieten, sind für eine
Datenbank, die die Dateien selbst ausliefert, nicht geeignet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://freesound.org/apiv2"

# Suchbegriffe -> Zielkategorie. Bewusst breit: Freesound-Tags sind
# inkonsistent, mehrere Queries pro Kategorie fangen mehr ab.
QUERIES: dict[str, list[str]] = {
    "kick": [
        "techno kick", "kick drum techno", "dark techno kick", "rumble kick",
        "industrial kick", "hard techno kick",
    ],
    "bass": [
        "techno bass", "acid bassline", "rolling bass loop", "sub bass loop",
        "303 acid",
    ],
    "hihat": [
        "techno hihat", "closed hihat", "open hihat loop", "hat loop techno",
    ],
    "snare": [
        "techno snare", "techno clap", "industrial clap", "rimshot",
    ],
    "percussion": [
        "techno percussion", "metal percussion", "industrial percussion loop",
        "tom loop techno",
    ],
    "atmosphere": [
        "dark drone", "industrial atmosphere", "techno pad", "eerie texture",
    ],
    "synth": [
        "techno stab", "chord stab", "acid synth loop", "rave stab",
    ],
    "loop": [
        "techno loop 130 bpm", "techno loop 138 bpm", "warehouse loop",
        "hypnotic techno loop",
    ],
}

MIN_DURATION = 0.15   # kürzer ist kein brauchbares Sample
MAX_DURATION = 32.0   # längere Dateien sind Mixe, keine Samples
MAX_BYTES = 25_000_000


def api_get(path: str, token: str, **params) -> dict:
    params["token"] = token
    url = f"{API}{path}/?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search(query: str, token: str, page_size: int = 150) -> list[dict]:
    """Sucht Sounds und filtert auf CC0 + WAV + brauchbare Länge."""
    fields = (
        "id,name,previews,license,duration,filesize,type,"
        "samplerate,channels,download,tags"
    )
    data = api_get(
        "search",
        token,
        query=query,
        filter='license:"Creative Commons 0"',
        fields=fields,
        page_size=page_size,
    )
    out = []
    for snd in data.get("results", []):
        if snd.get("type") != "wav":
            continue
        dur = float(snd.get("duration") or 0)
        if not (MIN_DURATION <= dur <= MAX_DURATION):
            continue
        if int(snd.get("filesize") or 0) > MAX_BYTES:
            continue
        out.append(snd)
    return out


def download(sound: dict, target: Path, token: str) -> bool:
    """Lädt die Originaldatei (OAuth-frei über die download-URL)."""
    url = sound.get("download")
    if not url:
        return False
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}token={token}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"    Fehler: {type(exc).__name__}: {exc}")
        return False
    if len(payload) < 1000:
        return False
    target.write_bytes(payload)
    return True


def safe_name(name: str) -> str:
    keep = "-_() "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return (cleaned or "sound")[:80]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="max. Dateien gesamt")
    parser.add_argument("--per-query", type=int, default=40, help="max. je Suchbegriff")
    parser.add_argument("--target", default="raw_construction_kits/library")
    args = parser.parse_args()

    token = os.environ.get("FREESOUND_API_KEY", "").strip()
    if not token:
        print("FEHLER: FREESOUND_API_KEY nicht gesetzt.")
        print("Token anfordern: https://freesound.org/apiv2/apply/")
        sys.exit(2)

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.csv"

    existing = set()
    rows: list[dict] = []
    if manifest_path.exists():
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
        existing = {r["path"] for r in rows}

    seen_ids: set[int] = set()
    downloaded = 0
    started = time.time()

    for category, queries in QUERIES.items():
        if downloaded >= args.limit:
            break
        folder = target / category
        folder.mkdir(parents=True, exist_ok=True)

        for query in queries:
            if downloaded >= args.limit:
                break
            print(f"\n[{category}] Suche: {query}")
            try:
                results = search(query, token)
            except Exception as exc:  # noqa: BLE001
                print(f"  Suchfehler: {type(exc).__name__}: {exc}")
                continue

            taken = 0
            for snd in results:
                if downloaded >= args.limit or taken >= args.per_query:
                    break
                sid = snd.get("id")
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)

                filename = f"fs_{sid}_{safe_name(snd.get('name', ''))}.wav"
                path = folder / filename
                if str(path) in existing:
                    continue

                if download(snd, path, token):
                    rows.append({
                        "path": str(path),
                        "category": category,
                        "bpm": "",
                        "source": "cc0",
                        "license": "CC0 1.0 (Freesound)",
                    })
                    existing.add(str(path))
                    downloaded += 1
                    taken += 1
                    print(f"    + {filename[:56]} ({downloaded}/{args.limit})")
                time.sleep(0.25)  # Rücksicht auf das Rate-Limit

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["path", "category", "bpm", "source", "license"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== FERTIG ===")
    print(f"  geladen: {downloaded}")
    print(f"  Manifest: {len(rows)} Einträge")
    print(f"  Dauer: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
