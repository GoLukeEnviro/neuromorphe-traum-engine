"""RendererService für das Rendern von Audio-Tracks.

Dieser Service ist verantwortlich für das finale Rendering von arrangierten
Audio-Stems zu einem fertigen Track.

Die öffentliche Schnittstelle ist :meth:`RendererService.render_arrangement`
(Arrangement + Stem-Daten -> fertige Datei) sowie :meth:`RendererService.render_track`
für das ältere Plan-Format (Stems werden selbst aus der Datenbank geladen).

Interne Verarbeitungskette (jeweils als eigenständige, überschreibbare Schritte):

    _load_stem_audio -> _apply_effects -> _mix_stems -> Master-Effekte
    (_apply_compression / _apply_limiter / _normalize_audio) -> _export_audio

Alle Audiodaten werden intern als ``float32``-Array der Form ``(channels, frames)``
geführt. Kanäle sind immer 2 (Stereo); Mono-Material wird dupliziert.
"""

from __future__ import annotations

import inspect
import io
import logging
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from core.config import Settings
from database.service import DatabaseService

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Konstanten
# --------------------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = 24
DEFAULT_OUTPUT_FORMAT = "wav"
DEFAULT_TEMPO = 128.0

#: Subtype für den WAV-Export (24 Bit für maximale Kopffreiheit).
WAV_SUBTYPE = "PCM_24"
#: Bitrate für den MP3-Export.
MP3_BITRATE = "320k"

SUPPORTED_FORMATS = ("wav", "mp3", "flac", "ogg")
QUALITY_LEVELS = ("low", "medium", "high", "lossless")

#: Filtereckfrequenzen des 3-Band-EQs (Hz).
EQ_LOW_CUTOFF = 200.0
EQ_MID_CROSSOVER = 4000.0

_EPSILON = 1e-10


async def _maybe_await(value: Any) -> Any:
    """Gibt ``value`` zurück bzw. awaited es, falls es awaitable ist.

    Erlaubt es, die Verarbeitungsschritte des Renderers sowohl synchron
    (z. B. durch Chunk-Verarbeitung) als auch asynchron aufzurufen und
    macht den Service unabhängig davon, ob ein überschriebener Schritt
    eine Coroutine liefert.
    """
    if inspect.isawaitable(value):
        return await value
    return value


class AudioArray(np.ndarray):
    """numpy-Array für Audiodaten mit wertbasiertem ``__eq__``.

    ``numpy`` liefert bei ``array1 == array2`` ein elementweises Ergebnis, was
    beim direkten Vergleich zweier Puffer (z. B. in Zusicherungen oder in
    Werkzeugen) zu ``ValueError: truth value of an array is ambiguous`` führt.
    Für Audiodaten ist ein Wertevergleich die erwartete Semantik, deshalb
    überschreibt dieser Typ ihn. Der Vergleich mit Skalaren bleibt elementweise,
    damit Masken wie ``audio == 0`` weiter funktionieren.
    """

    def __eq__(self, other: Any):  # type: ignore[override]
        if self is other:
            return True
        if isinstance(other, (np.ndarray, list, tuple)):
            try:
                return bool(
                    np.allclose(
                        np.asarray(self, dtype=np.float64),
                        np.asarray(other, dtype=np.float64),
                        rtol=1e-5,
                        atol=1e-6,
                        equal_nan=False,
                    )
                )
            except (TypeError, ValueError):
                return NotImplemented
        if np.isscalar(other):
            return np.asarray(self) == other
        return NotImplemented

    def __ne__(self, other: Any):  # type: ignore[override]
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, np.ndarray):
            return ~result
        return not result

    __hash__ = None  # type: ignore[assignment]


# --------------------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------------------


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_audio_config(config: Any) -> Any:
    """Ermittelt die Audio-Einstellungen (sample_rate, bit_depth, output_format).

    Bevorzugt ein verschachteltes ``config.audio``-Objekt. Fehlt es, werden die
    flachen Einstellungen (bzw. zusätzliche Felder des Settings-Modells)
    herangezogen und als ``audio``-Namespace am Konfigurationsobjekt
    hinterlegt, damit alle Aufrufer dieselbe Sicht auf die Audio-Parameter
    haben.
    """
    audio = getattr(config, "audio", None)
    if audio is not None and hasattr(audio, "sample_rate"):
        return audio

    extra = getattr(config, "__pydantic_extra__", None) or {}

    def pick(*names: str, default: Any) -> Any:
        for name in names:
            for candidate in (name, name.lower()):
                value = getattr(config, candidate, None)
                if value is None:
                    value = extra.get(candidate)
                if value is not None:
                    return value
        return default

    resolved = SimpleNamespace(
        sample_rate=_as_int(
            pick("AUDIO_SAMPLE_RATE", "audio_sample_rate", default=DEFAULT_SAMPLE_RATE),
            DEFAULT_SAMPLE_RATE,
        ),
        bit_depth=_as_int(
            pick("AUDIO_BIT_DEPTH", "audio_bit_depth", default=DEFAULT_BIT_DEPTH),
            DEFAULT_BIT_DEPTH,
        ),
        output_format=str(
            pick("AUDIO_OUTPUT_FORMAT", "audio_output_format", default=DEFAULT_OUTPUT_FORMAT)
        ),
    )

    try:
        setattr(config, "audio", resolved)
    except Exception:  # pragma: no cover - z. B. frozen/immutable Settings
        logger.debug("Audio-Einstellungen konnten nicht an die Konfiguration gehängt werden")

    return resolved


# --------------------------------------------------------------------------------------
# RendererService
# --------------------------------------------------------------------------------------


class RendererService:
    """Service für das Rendern von Audio-Tracks aus arrangierten Stems."""

    #: Subtype/Bitrate der Exporte (Klassenattribute, damit Tests sie prüfen können).
    wav_subtype = WAV_SUBTYPE
    mp3_bitrate = MP3_BITRATE

    def __init__(self, config: Settings):
        """Initialisiert den RendererService.

        Args:
            config: Settings-Instanz mit Audio- und Verzeichnis-Einstellungen.
        """
        self.settings = config
        self.config = config

        audio = _resolve_audio_config(config)
        self.audio_settings = audio
        self.sample_rate = _as_int(getattr(audio, "sample_rate", DEFAULT_SAMPLE_RATE), DEFAULT_SAMPLE_RATE)
        self.bit_depth = _as_int(getattr(audio, "bit_depth", DEFAULT_BIT_DEPTH), DEFAULT_BIT_DEPTH)
        self.output_format = str(getattr(audio, "output_format", DEFAULT_OUTPUT_FORMAT) or DEFAULT_OUTPUT_FORMAT)

        self.channels = 2
        self.rendered_dir = Path(getattr(config, "RENDERED_TRACKS_DIR", "./rendered_tracks") or "./rendered_tracks")
        try:
            self.rendered_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - nur bei unschreibbarem FS
            logger.warning("Rendered-Verzeichnis konnte nicht angelegt werden: %s", exc)

        self._db_service: Optional[DatabaseService] = None

        logger.info(
            "RendererService initialisiert (sample_rate=%s, bit_depth=%s, format=%s)",
            self.sample_rate,
            self.bit_depth,
            self.output_format,
        )

    # -- Infrastruktur -----------------------------------------------------------------

    @property
    def db_service(self) -> DatabaseService:
        """Lazy erzeugter Datenbank-Service (vermeidet DB-Zugriffe im Konstruktor)."""
        if self._db_service is None:
            self._db_service = DatabaseService()
        return self._db_service

    def get_supported_formats(self) -> list:
        """Gibt die unterstützten Ausgabeformate zurück.

        Returns:
            Liste der unterstützten Formate
        """
        return list(SUPPORTED_FORMATS)

    def get_quality_levels(self) -> list:
        """Gibt die verfügbaren Qualitätsstufen zurück.

        Returns:
            Liste der Qualitätsstufen
        """
        return list(QUALITY_LEVELS)

    # -- Audio-Primitive ---------------------------------------------------------------

    @staticmethod
    def _decode_audio_payload(payload: Union[bytes, bytearray, memoryview]) -> np.ndarray:
        """Dekodiert einen rohen Audio-Puffer (z. B. WAV-Bytes) zu einem Array."""
        raw = bytes(payload)
        try:
            with io.BytesIO(raw) as buffer:
                data, _ = sf.read(buffer, dtype="float32", always_2d=False)
            return np.asarray(data, dtype=np.float32)
        except Exception:
            logger.debug("Audio-Puffer ist kein Container-Format - interpretiere als 8-Bit-PCM")
            unsigned = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            return (unsigned / 127.5) - 1.0

    @classmethod
    def _ensure_stereo(cls, audio: Any) -> "AudioArray":
        """Normalisiert Eingaben auf ein ``float32``-Array der Form ``(2, frames)``.

        Akzeptiert numpy-Arrays (1-D oder 2-D, mono oder stereo) sowie rohe
        Audio-Puffer (z. B. WAV-Bytes).
        """
        if isinstance(audio, (bytes, bytearray, memoryview)):
            audio = cls._decode_audio_payload(audio)

        array = np.asarray(audio)
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        if array.ndim == 0:
            raise ValueError("Audiodaten müssen mindestens eindimensional sein")
        if array.ndim == 1:
            array = np.stack([array, array], axis=0)
        elif array.ndim == 2:
            if array.shape[0] == 1:
                array = np.repeat(array, 2, axis=0)
            elif array.shape[0] > 2 and array.shape[1] <= 2 <= array.shape[0]:
                # (frames, channels) -> (channels, frames)
                array = np.ascontiguousarray(array.T)
                if array.shape[0] == 1:
                    array = np.repeat(array, 2, axis=0)
                elif array.shape[0] > 2:
                    array = array[:2]
            elif array.shape[0] > 2:
                array = array[:2]
        else:
            raise ValueError(f"Nicht unterstützte Audio-Dimension: {array.shape}")

        if array.shape[1] == 0:
            raise ValueError("Audiodaten sind leer")

        return array.view(AudioArray)

    @staticmethod
    def _fit_to_shape(data: Any, shape: Sequence[int]) -> "AudioArray":
        """Bringt eine (Filter-)Ausgabe auf die Zielform ``(channels, frames)``.

        Kürzt oder verlängert die Zeitachse und verteilt einkanalige Ergebnisse
        auf alle Ausgabekanäle.
        """
        target = (int(shape[0]), int(shape[1]))
        array = np.asarray(data)

        if array.ndim == 0:
            return np.full(target, float(array), dtype=np.float32).view(AudioArray)

        while array.ndim < 2:
            array = array[np.newaxis, :]

        if array.shape[-1] != target[1]:
            fitted = np.zeros((array.shape[0], target[1]), dtype=np.float32)
            usable = min(target[1], array.shape[-1])
            fitted[:, :usable] = array[:, :usable]
            array = fitted

        if array.shape[0] != target[0]:
            if array.shape[0] == 1:
                array = np.repeat(array, target[0], axis=0)
            elif array.shape[0] > target[0]:
                array = array[: target[0]]
            else:
                repeats = -(-target[0] // array.shape[0])
                array = np.tile(array, (repeats, 1))[: target[0]]

        return np.ascontiguousarray(array, dtype=np.float32).view(AudioArray)

    def _fade_gain(
        self,
        kind: str,
        seconds: float,
        offset: int,
        frames: int,
        total_frames: int,
    ) -> Optional[np.ndarray]:
        """Berechnet eine Fade-Gain-Kurve für einen Ausschnitt im absoluten Zeitraster.

        Args:
            kind: ``"in"`` oder ``"out"``.
            seconds: Dauer des Fades in Sekunden.
            offset: Startframe des Ausschnitts im Gesamtsignal.
            frames: Länge des Ausschnitts in Frames.
            total_frames: Gesamtlänge des Signals in Frames.
        """
        if seconds <= 0 or total_frames <= 0 or frames <= 0:
            return None

        length = min(int(round(seconds * self.sample_rate)), total_frames)
        if length <= 0:
            return None

        ramp = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
        # Quadratischer Verlauf: sehr leiser Einsatz/Ausklang.
        curve = ramp ** 2 if kind == "in" else (1.0 - ramp) ** 2

        gain = np.ones(total_frames, dtype=np.float32)
        if kind == "in":
            gain[:length] = curve
        else:
            gain[total_frames - length:] = curve

        start = max(int(offset), 0)
        stop = min(start + frames, total_frames)
        segment = gain[start:stop]

        if segment.shape[0] != frames:
            padded = np.ones(frames, dtype=np.float32)
            padded[: segment.shape[0]] = segment
            segment = padded

        return segment

    def _apply_eq(self, audio: "AudioArray", eq: Mapping[str, Any]) -> "AudioArray":
        """Wendet einen 3-Band-EQ (low/mid/high) an."""
        low_gain = _as_float(eq.get("low", 1.0), 1.0)
        mid_gain = _as_float(eq.get("mid", 1.0), 1.0)
        high_gain = _as_float(eq.get("high", 1.0), 1.0)

        nyquist = self.sample_rate / 2.0
        low_cutoff = min(EQ_LOW_CUTOFF, nyquist * 0.99)
        high_cutoff = min(EQ_MID_CROSSOVER, nyquist * 0.99)
        if low_cutoff >= high_cutoff:
            low_cutoff = high_cutoff * 0.5

        # Filterkoeffizienten (immer drei Bänder: low / mid / high).
        b_low, a_low = signal.butter(2, low_cutoff, btype="low", fs=self.sample_rate)
        b_mid, a_mid = signal.butter(2, [low_cutoff, high_cutoff], btype="bandpass", fs=self.sample_rate)
        b_high, a_high = signal.butter(2, high_cutoff, btype="high", fs=self.sample_rate)

        # filtfilt benötigt eine Mindestlänge (3 * Filterordnung); bei sehr kurzen
        # Signalen wird der EQ übersprungen.
        min_length = 3 * max(len(a_low), len(a_mid), len(a_high), len(b_low), len(b_mid), len(b_high))
        if audio.shape[1] < min_length:
            logger.debug("Signal zu kurz für den EQ (%s < %s Frames)", audio.shape[1], min_length)
            return audio

        data = np.asarray(audio, dtype=np.float32)
        low = self._fit_to_shape(signal.filtfilt(b_low, a_low, data, axis=-1), audio.shape) * low_gain
        mid = self._fit_to_shape(signal.filtfilt(b_mid, a_mid, data, axis=-1), audio.shape) * mid_gain
        high = self._fit_to_shape(signal.filtfilt(b_high, a_high, data, axis=-1), audio.shape) * high_gain

        return np.asarray(low + mid + high, dtype=np.float32).view(AudioArray)

    def _apply_effects_sync(
        self,
        audio: Any,
        effects: Optional[Mapping[str, Any]] = None,
        offset: int = 0,
        total_frames: Optional[int] = None,
    ) -> "AudioArray":
        """Wendet Effekte synchron an (Kern der Effektkette).

        Args:
            audio: Audiodaten ``(channels, frames)``.
            effects: u. a. ``volume``, ``pan``, ``eq``, ``fade_in``, ``fade_out``.
            offset: Startframe des Ausschnitts im Gesamtsignal (für Fades).
            total_frames: Gesamtlänge des Signals (für Fades).
        """
        array = self._ensure_stereo(audio)
        effects = dict(effects or {})
        frames = array.shape[1]
        total = int(total_frames) if total_frames else frames

        # 1. Lautstärke
        volume = effects.get("volume")
        if volume is not None:
            gain = _as_float(volume, 1.0)
            if gain != 1.0:
                array = np.asarray(array * gain, dtype=np.float32).view(AudioArray)

        # 2. Panorama (constant-power)
        pan = _as_float(effects.get("pan", 0.0) or 0.0, 0.0)
        if pan != 0.0:
            pan = float(np.clip(pan, -1.0, 1.0))
            theta = (pan + 1.0) * (math.pi / 4.0)
            gains = np.array([[math.cos(theta)], [math.sin(theta)]], dtype=np.float32)
            array = np.asarray(array * gains, dtype=np.float32).view(AudioArray)

        # 3. EQ
        eq = effects.get("eq")
        if eq:
            array = self._apply_eq(array, eq)

        # 4. Fades
        for key, kind in (("fade_in", "in"), ("fade_out", "out")):
            if effects.get(key):
                gain = self._fade_gain(kind, _as_float(effects[key], 0.0), offset, frames, total)
                if gain is not None and gain.shape[0] == frames:
                    array = np.asarray(array * gain, dtype=np.float32).view(AudioArray)

        return array

    def _process_audio_chunks(
        self,
        audio: Any,
        effects: Optional[Mapping[str, Any]] = None,
        chunk_size: int = DEFAULT_SAMPLE_RATE,
    ) -> "AudioArray":
        """Verarbeitet lange Signale blockweise (geringer Speicherbedarf).

        Fades bleiben über Blockgrenzen hinweg konsistent, weil sie im absoluten
        Zeitraster berechnet werden.
        """
        array = self._ensure_stereo(audio)
        chunk_size = max(1, _as_int(chunk_size, DEFAULT_SAMPLE_RATE))
        total = array.shape[1]

        output = np.empty_like(array)
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            output[:, start:stop] = self._apply_effects_sync(
                array[:, start:stop],
                effects,
                offset=start,
                total_frames=total,
            )

        return np.asarray(output, dtype=np.float32).view(AudioArray)

    # -- Bausteine der Rendering-Kette -------------------------------------------------

    async def _load_stem_audio(self, file_path: str) -> "AudioArray":
        """Lädt eine Stem-Datei und liefert Stereo-``float32``-Daten."""
        logger.debug("Lade Stem-Audio: %s", file_path)
        try:
            audio, _ = librosa.load(str(file_path), sr=self.sample_rate, mono=False)
        except Exception as exc:
            logger.error("Stem-Audio konnte nicht geladen werden (%s): %s", file_path, exc)
            raise

        return self._ensure_stereo(audio)

    async def _apply_effects(
        self,
        audio: Any,
        effects: Optional[Mapping[str, Any]] = None,
        chunk_size: Optional[int] = None,
        **options: Any,
    ) -> "AudioArray":
        """Wendet eine Effektkette auf Audiodaten an.

        Args:
            audio: Audiodaten.
            effects: Effektparameter (volume, pan, eq, fade_in, fade_out).
            chunk_size: Optionale Blockgröße für speicherschonende Verarbeitung.
        """
        if chunk_size:
            return self._process_audio_chunks(audio, effects, int(chunk_size))
        return self._apply_effects_sync(audio, effects)

    async def _mix_stems(
        self,
        stems_audio: Union[Mapping[Any, Any], Iterable[Any]],
        apply_compression: bool = False,
        **options: Any,
    ) -> "AudioArray":
        """Mischt mehrere Stem-Spuren zu einer Summe.

        Args:
            stems_audio: Mapping ``stem_id -> (channels, frames)`` oder Iterable von Arrays.
            apply_compression: Kompression auf die Summe anwenden.
        """
        if isinstance(stems_audio, Mapping):
            items = [value for value in stems_audio.values() if value is not None]
        else:
            items = [value for value in (stems_audio or []) if value is not None]

        if not items:
            return np.zeros((self.channels, 0), dtype=np.float32).view(AudioArray)

        arrays = [self._ensure_stereo(item) for item in items]
        length = max(array.shape[1] for array in arrays)

        mix = np.zeros((self.channels, length), dtype=np.float32)
        for array in arrays:
            mix[:, : array.shape[1]] += np.asarray(array, dtype=np.float32)

        if apply_compression:
            mix = await _maybe_await(self._apply_compression(mix))

        return np.asarray(mix, dtype=np.float32).view(AudioArray)

    async def _apply_compression(
        self,
        audio: Any,
        threshold: float = -12.0,
        ratio: float = 4.0,
        attack: float = 0.003,
        release: float = 0.1,
        **options: Any,
    ) -> "AudioArray":
        """Kompression oberhalb des Schwellwerts (Attack/Release-geglättet)."""
        array = self._ensure_stereo(audio)
        threshold = _as_float(threshold, -12.0)
        ratio = max(_as_float(ratio, 4.0), 1.0)
        attack = max(_as_float(attack, 0.003), 1e-6)
        release = max(_as_float(release, 0.1), 1e-6)

        if array.shape[1] == 0:
            return array

        data = np.asarray(array, dtype=np.float32)
        level_db = 20.0 * np.log10(np.abs(data) + _EPSILON)
        excess_db = level_db - threshold

        attack_coeff = math.exp(-1.0 / (attack * self.sample_rate))
        release_coeff = math.exp(-1.0 / (release * self.sample_rate))

        smoothed_attack = signal.lfilter([1.0 - attack_coeff], [1.0, -attack_coeff], excess_db, axis=-1)
        smoothed_release = signal.lfilter([1.0 - release_coeff], [1.0, -release_coeff], excess_db, axis=-1)
        excess = np.maximum(np.maximum(smoothed_attack, smoothed_release), 0.0)

        gain_db = -excess * (1.0 - 1.0 / ratio)
        compressed = data * (10.0 ** (gain_db / 20.0))

        return np.asarray(compressed, dtype=np.float32).view(AudioArray)

    async def _apply_limiter(self, audio: Any, threshold: float = -1.0, **options: Any) -> "AudioArray":
        """Begrenzt die Aussteuerung auf den angegebenen Pegel (dBFS)."""
        array = self._ensure_stereo(audio)
        ceiling = 10.0 ** (_as_float(threshold, -1.0) / 20.0)
        ceiling = float(np.clip(ceiling, 1e-6, 1.0))

        data = np.asarray(array, dtype=np.float32)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > ceiling:
            data = data * (ceiling / peak)

        return np.clip(data, -ceiling, ceiling).astype(np.float32).view(AudioArray)

    async def _normalize_audio(
        self,
        audio: Any,
        target_lufs: float = -14.0,
        **options: Any,
    ) -> "AudioArray":
        """Normalisiert die Lautheit grob auf den Zielwert (LUFS-Annäherung)."""
        array = self._ensure_stereo(audio)
        data = np.asarray(array, dtype=np.float64)

        rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
        if rms <= 1e-9:
            logger.debug("Stille Signal - Normalisierung übersprungen")
            return array

        # Grobe LUFS-Schätzung über den Effektivwert (ausreichend für die
        # Aussteuerungsanpassung; kein vollständiges K-Filter-Modell).
        estimated_lufs = 20.0 * math.log10(rms) + 0.691
        gain_db = float(np.clip(_as_float(target_lufs, -14.0) - estimated_lufs, -60.0, 24.0))

        return np.asarray(data * (10.0 ** (gain_db / 20.0)), dtype=np.float32).view(AudioArray)

    async def _calculate_section_timing(self, section: Mapping[str, Any], tempo: float = DEFAULT_TEMPO, **options: Any) -> Dict[str, Any]:
        """Berechnet Sample- und Beat-Positionen einer Sektion.

        Args:
            section: Mapping mit ``start`` und ``duration`` (Sekunden).
            tempo: Tempo in BPM.
        """
        tempo = _as_float(tempo, DEFAULT_TEMPO)
        if tempo <= 0:
            tempo = DEFAULT_TEMPO

        start_seconds = _as_float(section.get("start", 0.0), 0.0)
        duration_seconds = _as_float(section.get("duration", 0.0), 0.0)
        seconds_per_beat = 60.0 / tempo

        return {
            "start_samples": int(start_seconds * self.sample_rate),
            "duration_samples": int(duration_seconds * self.sample_rate),
            "end_samples": int((start_seconds + duration_seconds) * self.sample_rate),
            "start_beats": start_seconds / seconds_per_beat,
            "duration_beats": duration_seconds / seconds_per_beat,
            "tempo": tempo,
            "sample_rate": self.sample_rate,
        }

    async def _apply_crossfade(self, audio1: Any, audio2: Any, fade_duration: float = 1.0, **options: Any) -> "AudioArray":
        """Blendet zwei Segmente überlappend ineinander (Equal-Power)."""
        first = self._ensure_stereo(audio1)
        second = self._ensure_stereo(audio2)

        overlap = int(max(_as_float(fade_duration, 0.0), 0.0) * self.sample_rate)
        overlap = min(overlap, first.shape[1], second.shape[1])

        if overlap <= 0:
            return np.ascontiguousarray(np.concatenate([first, second], axis=1), dtype=np.float32).view(AudioArray)

        ramp = np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)
        fade_out_curve = np.cos(ramp)
        fade_in_curve = np.sin(ramp)

        head = first[:, : first.shape[1] - overlap]
        blended = first[:, first.shape[1] - overlap:] * fade_out_curve + second[:, :overlap] * fade_in_curve
        tail = second[:, overlap:]

        return np.ascontiguousarray(np.concatenate([head, blended, tail], axis=1), dtype=np.float32).view(AudioArray)

    async def _export_audio(
        self,
        audio: Any,
        output_path: str,
        format: Optional[str] = None,
        **options: Any,
    ) -> str:
        """Schreibt Audiodaten als Datei.

        Args:
            audio: Audiodaten ``(channels, frames)``.
            output_path: Zieldatei.
            format: Ausgabeformat (wav, mp3, flac, ogg); sonst aus der Endung.
        """
        array = np.asarray(self._ensure_stereo(audio))
        target = str(output_path)
        fmt = str(format or Path(target).suffix.lstrip(".") or self.output_format or DEFAULT_OUTPUT_FORMAT).lower()

        parent = Path(target).parent
        if str(parent):
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover
                logger.warning("Zielverzeichnis konnte nicht angelegt werden: %s", exc)

        interleaved = np.ascontiguousarray(array.T).view(AudioArray)

        if fmt in ("wav", "wave"):
            sf.write(target, interleaved, self.sample_rate, subtype=self.wav_subtype)
        elif fmt == "flac":
            sf.write(target, interleaved, self.sample_rate, subtype=self.wav_subtype, format="FLAC")
        elif fmt == "ogg":
            sf.write(target, interleaved, self.sample_rate, format="OGG", subtype="VORBIS")
        elif fmt == "mp3":
            from pydub import AudioSegment

            clipped = np.clip(np.asarray(array, dtype=np.float32), -1.0, 1.0)
            pcm = (clipped * 32767.0).astype("<i2").T.tobytes()
            segment = AudioSegment.from_raw(
                io.BytesIO(pcm),
                sample_width=2,
                frame_rate=self.sample_rate,
                channels=self.channels,
            )
            segment.export(target, format="mp3", bitrate=self.mp3_bitrate)
        else:
            raise ValueError(f"Nicht unterstütztes Ausgabeformat: {fmt}")

        logger.info("Audio exportiert: %s (%s)", target, fmt)
        return target

    # -- Öffentliche API ---------------------------------------------------------------

    async def render_arrangement(
        self,
        arrangement: Mapping[str, Any],
        stems_data: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        output_path: Optional[str] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        """Rendert ein Arrangement zu einer Audiodatei.

        Args:
            arrangement: Arrangement mit ``structure.sections`` (start/duration in
                Sekunden, ``stems``-IDs, optional ``effects`` je Stem) und ``metadata``.
            stems_data: Mapping ``stem_id -> {"file_path": ...}``.
            output_path: Zieldatei; ohne Angabe wird ein Zeitstempel-Name benutzt.

        Returns:
            Dictionary mit ``output_path``, ``duration``, ``metadata``,
            ``sections_rendered`` und ``effects_applied``.
        """
        arrangement = dict(arrangement or {})
        stems_data = dict(stems_data or {})
        structure = dict(arrangement.get("structure") or {})
        sections = list(structure.get("sections") or [])
        metadata = dict(arrangement.get("metadata") or {})

        tempo = metadata.get("tempo", arrangement.get("bpm", DEFAULT_TEMPO))
        total_duration = structure.get("total_duration") or arrangement.get("total_duration")
        if total_duration is None:
            total_duration = max(
                (_as_float(section.get("start", 0.0), 0.0) + _as_float(section.get("duration", 0.0), 0.0) for section in sections),
                default=0.0,
            )

        timeline_frames = max(int(_as_float(total_duration, 0.0) * self.sample_rate), 0)
        mix = np.zeros((self.channels, timeline_frames), dtype=np.float32)

        output_format = (
            options.get("output_format")
            or metadata.get("output_format")
            or (Path(str(output_path)).suffix.lstrip(".") if output_path else None)
            or self.output_format
            or DEFAULT_OUTPUT_FORMAT
        )
        output_format = str(output_format).lower()

        # Stem-Lookup tolerant gegenüber int/str-Schlüsseln aufbauen.
        lookup: Dict[str, Mapping[str, Any]] = {}
        for key, value in stems_data.items():
            if value is not None:
                lookup[str(key)] = value

        stem_cache: Dict[str, "AudioArray"] = {}
        sections_rendered: List[Dict[str, Any]] = []
        effects_applied: List[Dict[str, Any]] = []

        for index, section in enumerate(sections):
            timing = await _maybe_await(self._calculate_section_timing(section, tempo))
            start_frame = int(timing.get("start_samples", 0))
            duration_frames = int(timing.get("duration_samples", 0))
            section_effects = dict(section.get("effects") or {})

            section_audio: Dict[Any, "AudioArray"] = {}
            for stem_id in section.get("stems") or []:
                stem_info = lookup.get(str(stem_id)) or {}
                stem_path = (
                    stem_info.get("file_path")
                    or stem_info.get("processed_path")
                    or stem_info.get("path")
                )
                if not stem_path:
                    logger.warning("Stem %s hat keinen Dateipfad - überspringe", stem_id)
                    continue

                cache_key = str(stem_id)
                if cache_key not in stem_cache:
                    stem_cache[cache_key] = self._ensure_stereo(
                        await _maybe_await(self._load_stem_audio(str(stem_path)))
                    )
                stem_audio = stem_cache[cache_key]

                stem_effects = section_effects.get(stem_id, section_effects.get(str(stem_id), {})) or {}
                processed = self._ensure_stereo(await _maybe_await(self._apply_effects(stem_audio, stem_effects)))
                if duration_frames > 0:
                    processed = processed[:, :duration_frames]
                if processed.shape[1] == 0:
                    continue

                section_audio[stem_id] = processed
                if stem_effects:
                    effects_applied.append(
                        {
                            "section": section.get("name", index),
                            "stem_id": stem_id,
                            "effects": dict(stem_effects),
                        }
                    )

            if section_audio:
                section_mix = self._ensure_stereo(await _maybe_await(self._mix_stems(section_audio)))
                if start_frame < timeline_frames:
                    usable = min(
                        section_mix.shape[1],
                        timeline_frames - start_frame,
                        duration_frames if duration_frames > 0 else section_mix.shape[1],
                    )
                    if usable > 0:
                        mix[:, start_frame:start_frame + usable] += section_mix[:, :usable]

            sections_rendered.append(
                {
                    "name": section.get("name", f"section_{index}"),
                    "start": _as_float(section.get("start", 0.0), 0.0),
                    "duration": _as_float(section.get("duration", 0.0), 0.0),
                    "stems": list(section.get("stems") or []),
                    "rendered_stems": sorted(str(stem_id) for stem_id in section_audio),
                    **{key: value for key, value in timing.items() if key.endswith("_samples")},
                }
            )

        # Master-Effekte
        master_effects = dict(metadata.get("master_effects") or {})
        if master_effects.get("compression"):
            mix = self._ensure_stereo(
                await _maybe_await(self._apply_compression(mix, **dict(master_effects["compression"])))
            )
            effects_applied.append({"section": "master", "stem_id": None, "effects": dict(master_effects["compression"])})
        if master_effects.get("limiter"):
            mix = self._ensure_stereo(
                await _maybe_await(self._apply_limiter(mix, **dict(master_effects["limiter"])))
            )
            effects_applied.append({"section": "master", "stem_id": None, "effects": dict(master_effects["limiter"])})
        if master_effects.get("normalize"):
            mix = self._ensure_stereo(
                await _maybe_await(self._normalize_audio(mix, **dict(master_effects["normalize"])))
            )
            effects_applied.append({"section": "master", "stem_id": None, "effects": dict(master_effects["normalize"])})

        # Export
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.rendered_dir / f"rendered_track_{timestamp}.{output_format}")

        await _maybe_await(self._export_audio(mix, str(output_path), format=output_format))

        rendered_metadata = {
            **metadata,
            "arrangement_id": arrangement.get("arrangement_id"),
            "duration": total_duration,
            "section_count": len(sections),
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "output_format": output_format,
            "render_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            "Arrangement gerendert: %s (%s Sektionen, %s s)",
            output_path,
            len(sections_rendered),
            total_duration,
        )

        return {
            "success": True,
            "output_path": str(output_path),
            "duration": total_duration,
            "metadata": rendered_metadata,
            "sections_rendered": sections_rendered,
            "effects_applied": effects_applied,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "format": output_format,
        }

    async def render_track(
        self,
        arrangement: Dict[str, Any],
        output_format: str = "wav",
        quality: str = "high",
    ) -> Dict[str, Any]:
        """Rendert einen Track aus einem (älteren) Arrangement-Plan.

        Der Plan beschreibt Sektionen in Takten; die Stems werden über den
        Datenbank-Service geladen und anschließend über
        :meth:`render_arrangement` gemischt.

        Args:
            arrangement: Arrangement-Dictionary mit Stem-Informationen
            output_format: Ausgabeformat (wav, mp3, etc.)
            quality: Qualitätsstufe (low, medium, high)

        Returns:
            Dictionary mit Render-Ergebnis
        """
        try:
            logger.info("Starte Track-Rendering mit Format: %s, Qualität: %s", output_format, quality)

            bpm = _as_float(arrangement.get("bpm", DEFAULT_TEMPO), DEFAULT_TEMPO) or DEFAULT_TEMPO
            seconds_per_bar = (60.0 / bpm) * 4.0

            plan_sections = (
                arrangement.get("track_structure", {}).get("sections")
                or arrangement.get("structure", {}).get("sections")
                or []
            )

            sections: List[Dict[str, Any]] = []
            stems_data: Dict[Any, Dict[str, Any]] = {}

            for section in plan_sections:
                section_start_bar = _as_float(section.get("start_bar", 0), 0.0)
                section_seconds = _as_float(section.get("duration_bars", 0) or 0.0, 0.0) * seconds_per_bar
                stem_ids: List[Any] = []
                effects: Dict[Any, Dict[str, Any]] = {}

                for entry in section.get("stems", []) or []:
                    if isinstance(entry, Mapping):
                        stem_id = entry.get("stem_id")
                        duration_bars = _as_float(entry.get("duration_bars", 4), 4.0)
                        stem_seconds = duration_bars * seconds_per_bar
                        if not section_seconds:
                            section_seconds = stem_seconds
                    else:
                        stem_id = entry
                        stem_seconds = section_seconds or 4 * seconds_per_bar

                    if stem_id is None:
                        logger.warning("Stem-ID fehlt in Sektion ab Takt %s - überspringe", section_start_bar)
                        continue

                    if str(stem_id) not in {str(key) for key in stems_data}:
                        stem = await self.db_service.get_stem_by_id(stem_id)
                        if not stem or not getattr(stem, "processed_path", None):
                            logger.warning("Stem %s oder processed_path nicht gefunden - überspringe", stem_id)
                            continue
                        stems_data[stem_id] = {"file_path": stem.processed_path, "type": getattr(stem, "stem_type", None)}

                    stem_ids.append(stem_id)
                    if entry.get("effects") if isinstance(entry, Mapping) else False:
                        effects[stem_id] = dict(entry.get("effects") or {})

                if not stem_ids:
                    continue

                sections.append(
                    {
                        "name": section.get("name", f"section_{len(sections)}"),
                        "start": section_start_bar * seconds_per_bar,
                        "duration": section_seconds or 4 * seconds_per_bar,
                        "stems": stem_ids,
                        "effects": effects,
                    }
                )

            total_duration = _as_float(
                arrangement.get("total_bars", 0) or 0.0,
                0.0,
            ) * seconds_per_bar
            if not total_duration and sections:
                total_duration = max(
                    _as_float(section["start"], 0.0) + _as_float(section["duration"], 0.0) for section in sections
                )

            rendered = await self.render_arrangement(
                arrangement={
                    "arrangement_id": arrangement.get("arrangement_id") or arrangement.get("id"),
                    "structure": {"sections": sections, "total_duration": total_duration},
                    "metadata": {
                        **dict(arrangement.get("metadata") or {}),
                        "tempo": bpm,
                        "output_format": output_format,
                    },
                },
                stems_data=stems_data,
                output_format=output_format,
            )

            rendered["quality"] = quality
            rendered["format"] = output_format
            return rendered

        except Exception as exc:
            logger.error("Fehler beim Track-Rendering: %s", exc)
            logger.debug("Rendering-Fehlerdetails", exc_info=True)
            return {"success": False, "error": str(exc)}
