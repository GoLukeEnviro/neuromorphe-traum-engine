"""Baut eine Test-Bibliothek für die Neuromorphe Traum-Engine auf.

Zwei Quellen, beide lizenzsauber:

1. ``synthetic`` — hier synthetisierte Techno-Elemente (Kick, HiHat, Clap,
   Bass, Percussion, Atmo, Stab). Eigene Werke, keine fremden Rechte.
   Das ist KEIN Ersatz für eine echte Sample-Bibliothek: Es prüft die
   Pipeline, nicht den Musikgeschmack.
2. ``cc0`` — Samples aus der Producer-Space-CC0-Library auf archive.org
   (CC0 1.0, keine Attribution nötig).

Jede Datei bekommt einen Eintrag in ``manifest.csv`` mit Quelle und Lizenz,
damit später nachvollziehbar ist, woher ein Stem stammt.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
RNG = np.random.default_rng(20260915)


# --------------------------------------------------------------------------
# Synthese-Bausteine
# --------------------------------------------------------------------------
def _env(n: int, attack: float, decay: float, curve: float = 2.0) -> np.ndarray:
    """Attack-Decay-Hüllkurve."""
    a = max(1, int(attack * SR))
    d = max(1, n - a)
    env = np.concatenate([
        np.linspace(0.0, 1.0, a, endpoint=False),
        np.exp(-np.linspace(0, curve, d)),
    ])
    return env[:n]


def _sine_sweep(n: int, f_start: float, f_end: float) -> np.ndarray:
    """Exponentieller Frequenz-Sweep."""
    t = np.arange(n) / SR
    k = math.log(f_end / f_start) / max(1e-9, n / SR)
    phase = 2 * np.pi * f_start * (np.exp(k * t) - 1) / k
    return np.sin(phase)


def make_kick(pitch: float = 48.0, decay: float = 0.45, click: float = 0.35) -> np.ndarray:
    n = int(SR * decay)
    body = _sine_sweep(n, 160.0, pitch) * _env(n, 0.002, decay, 6.0)
    transient = RNG.normal(0, 1, int(SR * 0.006)) * np.linspace(1, 0, int(SR * 0.006))
    out = body.copy()
    m = min(len(transient), n)
    out[:m] += transient[:m] * click
    out = np.tanh(out * 1.6)
    return out / (np.abs(out).max() + 1e-9) * 0.9


def make_hat(decay: float = 0.06, open_hat: bool = False) -> np.ndarray:
    d = 0.28 if open_hat else decay
    n = int(SR * d)
    noise = RNG.normal(0, 1, n)
    # Einfacher Hochpass über Differenzbildung
    hp = np.diff(noise, prepend=0.0)
    out = hp * _env(n, 0.0005, d, 8.0)
    return out / (np.abs(out).max() + 1e-9) * 0.55


def make_clap() -> np.ndarray:
    n = int(SR * 0.25)
    out = np.zeros(n)
    for offset, gain in ((0.000, 1.0), (0.011, 0.7), (0.022, 0.5), (0.034, 0.35)):
        i = int(offset * SR)
        seg = RNG.normal(0, 1, int(SR * 0.02)) * np.exp(-np.linspace(0, 5, int(SR * 0.02)))
        m = min(len(seg), n - i)
        out[i:i + m] += seg[:m] * gain
    body = out * np.exp(-np.linspace(0, 3, n))
    return body / (np.abs(body).max() + 1e-9) * 0.7


def make_snare() -> np.ndarray:
    n = int(SR * 0.3)
    tone = _sine_sweep(n, 220.0, 150.0) * _env(n, 0.001, 0.3, 7.0) * 0.5
    noise = RNG.normal(0, 1, n) * _env(n, 0.001, 0.3, 6.0)
    out = tone + noise * 0.7
    return out / (np.abs(out).max() + 1e-9) * 0.75


def make_bass(freq: float = 55.0, length: float = 0.5, cutoff: float = 900.0) -> np.ndarray:
    n = int(SR * length)
    t = np.arange(n) / SR
    saw = 2 * ((t * freq) % 1.0) - 1.0
    square = np.sign(np.sin(2 * np.pi * freq * t)) * 0.4
    raw = saw * 0.7 + square
    # Einfacher Tiefpass (Ein-Pol)
    alpha = 1 - math.exp(-2 * math.pi * cutoff / SR)
    out = np.zeros(n)
    acc = 0.0
    for i, v in enumerate(raw):
        acc += alpha * (v - acc)
        out[i] = acc
    out *= _env(n, 0.005, length, 3.0)
    return out / (np.abs(out).max() + 1e-9) * 0.8


def make_perc(freq: float = 320.0, length: float = 0.18) -> np.ndarray:
    n = int(SR * length)
    t = np.arange(n) / SR
    mod = np.sin(2 * np.pi * freq * 1.41 * t) * 4.0
    out = np.sin(2 * np.pi * freq * t + mod) * _env(n, 0.001, length, 9.0)
    return out / (np.abs(out).max() + 1e-9) * 0.6


def make_atmo(length: float = 4.0) -> np.ndarray:
    n = int(SR * length)
    noise = RNG.normal(0, 1, n)
    # Langsamer Filter-Sweep über gleitenden Mittelwert
    out = np.zeros(n)
    acc = 0.0
    for i in range(n):
        cutoff = 300 + 1800 * (i / n)
        alpha = 1 - math.exp(-2 * math.pi * cutoff / SR)
        acc += alpha * (noise[i] - acc)
        out[i] = acc
    env = np.sin(np.linspace(0, np.pi, n)) ** 1.5
    out *= env
    return out / (np.abs(out).max() + 1e-9) * 0.45


def make_stab(freq: float = 220.0) -> np.ndarray:
    n = int(SR * 0.45)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for mult, gain in ((1.0, 1.0), (1.5, 0.5), (2.0, 0.35)):
        out += np.sin(2 * np.pi * freq * mult * t) * gain
    out *= _env(n, 0.004, 0.45, 5.0)
    return out / (np.abs(out).max() + 1e-9) * 0.6


def make_loop_4bar(kind: str, bpm: float = 134.0) -> np.ndarray:
    """Vier Takte 4/4, 16tel-Raster."""
    beat = 60.0 / bpm
    bar = beat * 4
    total = int(SR * bar * 4)
    out = np.zeros(total)

    def place(sample: np.ndarray, position_sec: float, gain: float = 1.0) -> None:
        i = int(position_sec * SR)
        m = min(len(sample), total - i)
        if m > 0:
            out[i:i + m] += sample[:m] * gain

    if kind == "kick_loop":
        kick = make_kick(46, 0.4)
        for b in range(16):
            place(kick, b * beat, 0.9)
    elif kind == "hat_loop":
        hat = make_hat(0.05)
        ohat = make_hat(open_hat=True)
        for b in range(16 * 4):
            pos = b * beat / 4
            place(ohat if b % 4 == 2 else hat, pos, 0.35 if b % 2 else 0.22)
    elif kind == "bass_loop":
        notes = [55.0, 55.0, 61.74, 49.0]
        for b in range(16):
            place(make_bass(notes[b % 4], beat * 0.9, 700), b * beat, 0.8)
    elif kind == "perc_loop":
        for b in range(16 * 4):
            if b % 4 in (1, 3):
                place(make_perc(300 + (b % 5) * 60), b * beat / 4, 0.3)
    else:
        raise ValueError(kind)

    peak = np.abs(out).max()
    if peak > 0:
        out = out / peak * 0.85
    return out


# --------------------------------------------------------------------------
# Aufbau
# --------------------------------------------------------------------------
SYNTH_ONE_SHOTS = {
    "kick": [
        ("kick_deep_46hz", lambda: make_kick(46, 0.50)),
        ("kick_punch_52hz", lambda: make_kick(52, 0.38, 0.55)),
        ("kick_short_58hz", lambda: make_kick(58, 0.26, 0.30)),
        ("kick_long_42hz", lambda: make_kick(42, 0.70)),
        ("kick_clicky_60hz", lambda: make_kick(60, 0.30, 0.80)),
        ("kick_sub_38hz", lambda: make_kick(38, 0.85, 0.15)),
    ],
    "hihat": [
        ("hat_closed_tight", lambda: make_hat(0.04)),
        ("hat_closed_soft", lambda: make_hat(0.09)),
        ("hat_open_long", lambda: make_hat(open_hat=True)),
        ("hat_tick", lambda: make_hat(0.02)),
    ],
    "snare": [
        ("snare_techno", make_snare),
        ("clap_dry", make_clap),
    ],
    "bass": [
        ("bass_a1_55hz", lambda: make_bass(55.0, 0.45, 800)),
        ("bass_b1_61hz", lambda: make_bass(61.74, 0.45, 900)),
        ("bass_g1_49hz", lambda: make_bass(49.0, 0.30, 1100)),
        ("bass_acid_long", lambda: make_bass(55.0, 0.90, 1400)),
    ],
    "percussion": [
        ("perc_wood_320hz", lambda: make_perc(320)),
        ("perc_metal_480hz", lambda: make_perc(480, 0.12)),
        ("perc_tom_180hz", lambda: make_perc(180, 0.25)),
    ],
    "atmosphere": [
        ("atmo_rise_4s", lambda: make_atmo(4.0)),
        ("atmo_dark_8s", lambda: make_atmo(8.0)),
    ],
    "synth": [
        ("stab_am_220hz", lambda: make_stab(220)),
        ("stab_dm_293hz", lambda: make_stab(293.66)),
    ],
}

SYNTH_LOOPS = [
    ("loop_kick_4bar_134", lambda: make_loop_4bar("kick_loop", 134)),
    ("loop_hat_4bar_134", lambda: make_loop_4bar("hat_loop", 134)),
    ("loop_bass_4bar_134", lambda: make_loop_4bar("bass_loop", 134)),
    ("loop_perc_4bar_134", lambda: make_loop_4bar("perc_loop", 134)),
    ("loop_kick_4bar_138", lambda: make_loop_4bar("kick_loop", 138)),
    ("loop_bass_4bar_138", lambda: make_loop_4bar("bass_loop", 138)),
]


def build_synthetic(target: Path, manifest: list) -> int:
    count = 0
    for category, items in SYNTH_ONE_SHOTS.items():
        folder = target / category
        folder.mkdir(parents=True, exist_ok=True)
        for name, fn in items:
            audio = fn()
            audio = audio / (np.abs(audio).max() + 1e-9) * 0.85
            path = folder / f"{name}.wav"
            sf.write(path, audio.astype(np.float32), SR, subtype="PCM_16")
            manifest.append({
                "path": str(path), "category": category, "bpm": "",
                "source": "synthetic", "license": "eigenes Werk",
            })
            count += 1

    loop_dir = target / "loops"
    loop_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in SYNTH_LOOPS:
        audio = fn()
        audio = audio / (np.abs(audio).max() + 1e-9) * 0.85
        path = loop_dir / f"{name}.wav"
        sf.write(path, audio.astype(np.float32), SR, subtype="PCM_16")
        bpm = 138 if "138" in name else 134
        manifest.append({
            "path": str(path), "category": "loop", "bpm": bpm,
            "source": "synthetic", "license": "eigenes Werk",
        })
        count += 1

    return count


CC0_BASE = "https://archive.org/download/producer-space-cc0-sample-library/"
CC0_FILES = [
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Drums%201.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Drums%202.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Drums%203.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Drums%204.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Bass%201.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Bass%202.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Bass%203.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Bass%204.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Bass%205.wav",
    "Producer%20Space/Packs/Dance%20Vocal%20Shouts/DVS%20-%20Bass%206.wav",
]


def build_cc0(target: Path, manifest: list) -> int:
    import urllib.request

    folder = target / "cc0"
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for rel in CC0_FILES:
        name = rel.split("/")[-1].replace("%20", " ")
        path = folder / name
        if path.exists():
            count += 1
            continue
        try:
            with urllib.request.urlopen(CC0_BASE + rel, timeout=60) as resp:
                path.write_bytes(resp.read())
            category = "percussion" if "Drum" in name else "bass"
            manifest.append({
                "path": str(path), "category": category, "bpm": "",
                "source": "cc0",
                "license": "CC0 1.0 (Producer Space, archive.org)",
            })
            print(f"  geladen: {name}")
            count += 1
        except Exception as exc:
            print(f"  FEHLER bei {name}: {exc}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="raw_construction_kits/library")
    parser.add_argument("--no-cc0", action="store_true")
    args = parser.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    manifest: list = []

    synth = build_synthetic(target, manifest)
    print(f"Synthetisch erzeugt: {synth} Dateien")

    cc0 = 0
    if not args.no_cc0:
        cc0 = build_cc0(target, manifest)
        print(f"CC0 geladen: {cc0} Dateien")

    manifest_path = target / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "category", "bpm", "source", "license"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nGesamt: {synth + cc0} Dateien")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
