"""Hilfsfunktionen für die Neuromorphe Traum-Engine.

Sammlung von Werkzeugen für Datei-, Audio-, Daten-, String-, Zeit-,
Async-, Validierungs-, Performance- und Fehlerbehandlung. Die Semantik
orientiert sich an der Test-Suite (``tests/test_core/test_utils.py``).
"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import re
import time
import traceback
import unicodedata
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


# ---------------------------------------------------------------------------
# Datei-Hilfsfunktionen
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


def ensure_directory(path: str) -> None:
    """Erstellt ein Verzeichnis samt Elternordnern (idempotent)."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_file_path(path: str) -> str:
    """Bereinigt einen relativen Pfad für sichere Dateisystem-Nutzung.

    Behält Verzeichnistrenner bei, entfernt aber Traversal-Sequenzen und
    macht den Pfad relativ.
    """
    # Windows-Trenner vereinheitlichen
    path = str(path).replace("\\", "/")
    # Absolute Pfade relativ machen
    path = path.lstrip("/")
    # Traversal entfernen
    parts = [p for p in path.split("/") if p not in ("", ".", "..")]
    cleaned = "/".join(parts)
    # Gefährliche Zeichen je Segment entfernen (Trenner bleiben erhalten)
    cleaned = re.sub(r'[<>:"|?*]+', "_", cleaned)
    return cleaned.strip("._")[:200]


def get_file_size(path: str) -> int:
    """Gibt die Dateigröße in Bytes zurück."""
    return os.path.getsize(path)


def get_file_extension(path: str) -> str:
    """Gibt die Dateiendung inkl. Punkt zurück.

    ``.hidden`` (reine Hidden-Datei) hat keine Endung und liefert "".
    """
    name = os.path.basename(str(path))
    if name.startswith(".") and name.count(".") == 1:
        return ""
    return os.path.splitext(name)[1]


def copy_file(src: str, dst: str) -> None:
    """Kopiert eine Datei (inkl. Zielverzeichnis)."""
    import shutil

    ensure_directory(str(Path(dst).parent))
    shutil.copy2(src, dst)


def move_file(src: str, dst: str) -> None:
    """Verschiebt eine Datei (inkl. Zielverzeichnis)."""
    import shutil

    ensure_directory(str(Path(dst).parent))
    shutil.move(src, dst)


def delete_file(path: str) -> None:
    """Löscht eine Datei, falls vorhanden."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def list_files(
    directory: str,
    pattern: str = "*",
    extensions: Optional[List[str]] = None,
) -> List[str]:
    """Listet Dateien in einem Verzeichnis (nicht rekursiv).

    Args:
        directory: Zu durchsuchendes Verzeichnis.
        pattern: Glob-Muster.
        extensions: Optionale Liste erlaubter Endungen (z. B. [".wav"]).
    """
    files = [str(p) for p in Path(directory).glob(pattern) if p.is_file()]
    if extensions:
        normalized = {e.lower() for e in extensions}
        files = [f for f in files if Path(f).suffix.lower() in normalized]
    return files


def find_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = True,
) -> List[str]:
    """Findet Dateien nach Muster, optional rekursiv."""
    glob_pattern = f"**/{pattern}" if recursive else pattern
    return [str(p) for p in Path(directory).glob(glob_pattern) if p.is_file()]


# ---------------------------------------------------------------------------
# Audio-Hilfsfunktionen
# ---------------------------------------------------------------------------


def validate_audio_format(path: str) -> bool:
    """Prüft, ob die Dateiendung ein unterstütztes Audioformat ist."""
    return get_file_extension(path).lower() in AUDIO_EXTENSIONS


def get_audio_duration(path: str) -> float:
    """Ermittelt die Dauer einer Audiodatei in Sekunden."""
    import librosa

    return float(librosa.get_duration(path=path))


def normalize_audio_path(path: str) -> str:
    """Normalisiert einen Audio-Pfad: keine Leerzeichen, keine Umlaute.

    Relative Pfade bleiben relativ; Verzeichnisstruktur bleibt erhalten.
    """
    result = str(path).replace(" ", "_")
    # Umlaute und Akzente transliterieren (ä -> a, ö -> o, ü -> u, ß -> ss)
    result = (
        result.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("Ä", "A")
        .replace("Ö", "O")
        .replace("Ü", "U")
        .replace("ß", "ss")
    )
    result = unicodedata.normalize("NFKD", result)
    result = "".join(c for c in result if not unicodedata.combining(c))
    return result


def convert_sample_rate(input_path: str, output_path: str, target_rate: int) -> None:
    """Konvertiert die Samplerate einer Audiodatei."""
    import librosa
    import soundfile as sf

    audio, _ = librosa.load(input_path, sr=target_rate, mono=False)
    sf.write(output_path, audio.T if audio.ndim > 1 else audio, target_rate)


def get_audio_info(path: str) -> Dict[str, Any]:
    """Liest Audio-Metadaten (Samplerate, Dauer, Sample-Anzahl)."""
    import librosa

    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    samples = len(audio)
    return {
        "sample_rate": sample_rate,
        "duration": samples / sample_rate if sample_rate else 0.0,
        "samples": samples,
        "path": str(path),
    }


def create_silence(duration: float, sample_rate: int = 44100) -> List[float]:
    """Erzeugt Stille als Sample-Liste.

    Länge entspricht der Sample-Anzahl (Monokanal, float-Samples).
    """
    return [0.0] * int(duration * sample_rate)


# ---------------------------------------------------------------------------
# Daten-Hilfsfunktionen
# ---------------------------------------------------------------------------


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Führt zwei Dictionaries tief zusammen.

    Verschachtelte Dicts werden rekursiv gemergt, Listen konkateniert,
    andere Werte aus ``dict2`` überschreiben.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Macht ein verschachteltes Dictionary flach."""
    items: List[Tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Stellt ein flaches Dictionary wieder hierarchisch her."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def sanitize_dict(
    d: Dict[str, Any],
    sensitive_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Ersetzt sensible Werte durch ``***``.

    Args:
        d: Zu bereinigendes Dictionary.
        sensitive_keys: Schlüsselnamen, deren Werte maskiert werden.
            Standard: password, api_key, token, secret, key.
    """
    keys = sensitive_keys if sensitive_keys is not None else [
        "password", "api_key", "token", "secret", "key",
    ]
    lowered = {k.lower() for k in keys}
    result: Dict[str, Any] = {}
    for key, value in d.items():
        if key.lower() in lowered:
            result[key] = "***"
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value, sensitive_keys)
        else:
            result[key] = value
    return result


def serialize_data(data: Any) -> str:
    """Serialisiert Daten zu einem JSON-String."""
    return json.dumps(data, default=str)


def deserialize_data(data: str) -> Any:
    """Deserialisiert einen JSON-String."""
    return json.loads(data)


def hash_data(data: Any) -> str:
    """Erzeugt einen SHA-256-Hash über die Daten.

    Die Reihenfolge der Dict-Schlüssel ist irrelevant (sortierte Ausgabe).
    """
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compare_with_tolerance(a: Any, b: Any, tolerance: Optional[float]) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_compare_with_tolerance(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_compare_with_tolerance(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if tolerance is None:
            return a == b
        try:
            return abs(a - b) <= tolerance
        except TypeError:
            return False
    return a == b


def compare_data(
    data1: Any,
    data2: Any,
    tolerance: Optional[float] = None,
) -> bool:
    """Vergleicht zwei Datenstrukturen, optional mit Float-Toleranz."""
    return _compare_with_tolerance(data1, data2, tolerance)


# ---------------------------------------------------------------------------
# String-Hilfsfunktionen
# ---------------------------------------------------------------------------


def sanitize_filename(filename: str) -> str:
    """Bereinigt einen Dateinamen für sichere Nutzung."""
    name = os.path.basename(str(filename).replace("\\", "/"))
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._") or "unnamed"


def generate_id(prefix: str = "", length: int = 16) -> str:
    """Erzeugt eine eindeutige ID, optional mit Präfix."""
    import uuid

    token = uuid.uuid4().hex[: max(length, 8)]
    return f"{prefix}_{token}" if prefix else token


def slugify(text: str) -> str:
    """Erzeugt einen URL-freundlichen Slug."""
    value = (
        text.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("Ä", "A")
        .replace("Ö", "O")
        .replace("Ü", "U")
        .replace("ß", "ss")
    )
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def truncate_string(text: str, max_length: int, ellipsis: str = "...", suffix: Optional[str] = None) -> str:
    """Kürzt einen String auf ``max_length`` inkl. Ellipsis."""
    marker = suffix if suffix is not None else ellipsis
    if len(text) <= max_length:
        return text
    if not marker:
        return text[:max_length]
    return text[: max_length - len(marker)] + marker


def format_duration(seconds: float) -> str:
    """Formatiert Sekunden als ``M:SS`` bzw. ``H:MM:SS``."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_file_size(size_bytes: int) -> str:
    """Formatiert Bytes menschenlesbar."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def validate_email(email: str) -> bool:
    """Validiert eine E-Mail-Adresse."""
    if not email or not isinstance(email, str):
        return False
    try:
        from email_validator import validate_email as _validate, EmailNotValidError

        _validate(email, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False
    except Exception:
        pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        return bool(re.match(pattern, email))


# ---------------------------------------------------------------------------
# Zeit-Hilfsfunktionen
# ---------------------------------------------------------------------------


def get_timestamp() -> float:
    """Aktueller Unix-Zeitstempel als Float."""
    return time.time()


def _to_datetime(value: Union[int, float, str, datetime]) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.strptime(str(value), "%Y-%m-%d")


def format_timestamp(
    timestamp: Union[int, float, str, datetime],
    format_str: Optional[str] = None,
    format: Optional[str] = None,
) -> str:
    """Formatiert einen Zeitstempel.

    ``format="iso"`` liefert ISO-8601, ein strftime-Muster wird direkt
    verwendet.
    """
    dt = _to_datetime(timestamp)
    spec = format if format is not None else format_str
    if spec is None or spec == "iso":
        return dt.isoformat()
    return dt.strftime(spec)


def parse_timestamp(timestamp_str: str, format: Optional[str] = None) -> float:
    """Parst einen Zeitstempel-String zu einem Unix-Zeitstempel (Float)."""
    if format:
        dt = datetime.strptime(timestamp_str, format)
    else:
        dt = _to_datetime(timestamp_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def time_ago(timestamp: Union[datetime, int, float, str]) -> str:
    """Beschreibt, wie lange ein Zeitpunkt zurückliegt (englisch)."""
    dt = _to_datetime(timestamp)
    now = datetime.now(tz=dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    seconds = int(delta.total_seconds())

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = seconds // 86400
    return f"{days} day{'s' if days != 1 else ''} ago"


def duration_to_seconds(
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
) -> int:
    """Rechnet Stunden/Minuten/Sekunden in Sekunden um."""
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_duration(seconds: int, format: Optional[str] = None) -> Union[Dict[str, int], str]:
    """Rechnet Sekunden in Stunden/Minuten/Sekunden um.

    ``format="string"`` liefert z. B. ``"1h 1m 1s"``.
    """
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if format == "string":
        return f"{hours}h {minutes}m {secs}s"
    return {"hours": hours, "minutes": minutes, "seconds": secs}


# ---------------------------------------------------------------------------
# Async-Hilfsfunktionen
# ---------------------------------------------------------------------------


async def run_async(func: Callable, *args, **kwargs) -> Any:
    """Führt eine (async) Funktion aus und gibt das Ergebnis zurück."""
    result = func(*args, **kwargs)
    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
        return await result
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)


async def gather_with_concurrency(tasks: List[Any], max_concurrency: int = 10) -> List[Any]:
    """Führt Awaitables mit begrenzter Parallelität aus (Reihenfolge bleibt)."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run(item, index):
        async with semaphore:
            if callable(item):
                item = item()
            if asyncio.iscoroutine(item) or asyncio.isfuture(item):
                return index, await item
            return index, item

    results = await asyncio.gather(*(_run(t, i) for i, t in enumerate(tasks)))
    return [value for _, value in sorted(results, key=lambda pair: pair[0])]


async def retry_async(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    *args,
    **kwargs,
) -> Any:
    """Wiederholt eine (async) Funktion bei Fehlern bis ``max_retries``."""
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                return await result
            return result
        except Exception as exc:  # noqa: BLE001 - Retry soll alles fangen
            last_error = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


async def timeout_async(coro: Any, timeout: float) -> Any:
    """Wartet auf ein Awaitable mit Zeitlimit."""
    return await asyncio.wait_for(coro, timeout=timeout)


def debounce(delay: float = 0.3):
    """Decorator: führt nur den letzten Aufruf nach ``delay`` aus.

    Nutzt einen Hintergrund-Thread, damit auch synchrone Aufrufer
    (ohne Event-Loop) funktionieren.
    """
    import threading

    def decorator(func: Callable) -> Callable:
        timer_holder: Dict[str, Any] = {"timer": None}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if timer_holder["timer"] is not None:
                timer_holder["timer"].cancel()
            timer = threading.Timer(delay, lambda: func(*args, **kwargs))
            timer.daemon = True
            timer_holder["timer"] = timer
            timer.start()
            return None

        return wrapper

    return decorator


def throttle(interval: float = 1.0):
    """Decorator: erlaubt einen Aufruf pro ``interval`` Sekunden.

    Gedrosselte Aufrufe geben ``None`` zurück.
    """
    def decorator(func: Callable) -> Callable:
        state: Dict[str, float] = {"last": 0.0}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if now - state["last"] >= interval:
                state["last"] = now
                return func(*args, **kwargs)
            return None

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Validierungs-Hilfsfunktionen
# ---------------------------------------------------------------------------


class ValidationResult:
    """Ergebnis einer Validierung."""

    def __init__(
        self,
        is_valid: bool = True,
        errors: Optional[List[str]] = None,
        missing_fields: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors if errors is not None else []
        self.missing_fields = missing_fields if missing_fields is not None else []

    def summary(self) -> str:
        if self.is_valid:
            return "Validation successful"
        return f"{len(self.errors)} validation errors"

    def __bool__(self) -> bool:
        return bool(self.is_valid)

    def __repr__(self) -> str:
        return f"<ValidationResult valid={self.is_valid} errors={len(self.errors)}>"


def validate_type(value: Any, expected_type: Union[Type, Tuple[Type, ...]]) -> bool:
    """Prüft den Typ eines Werts."""
    return isinstance(value, expected_type)


def validate_range(value: Any, min_val: Any = None, max_val: Any = None) -> bool:
    """Prüft, ob ein Wert innerhalb der Grenzen liegt (inklusiv)."""
    try:
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    except TypeError:
        return False


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> ValidationResult:
    """Prüft, ob alle erforderlichen Felder vorhanden und nicht leer sind."""
    missing = [
        field for field in required_fields
        if field not in data or data[field] is None
    ]
    if missing:
        errors = [f"Field '{name}' is required" for name in missing]
        return ValidationResult(is_valid=False, errors=errors, missing_fields=missing)
    return ValidationResult(is_valid=True)


def validate_schema(data: Any, schema: Dict[str, Any]) -> ValidationResult:
    """Validiert Daten gegen ein (vereinfachtes) JSON-Schema."""
    errors: List[str] = []
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    if schema.get("type") == "object":
        if not isinstance(data, dict):
            return ValidationResult(is_valid=False, errors=["Value is not an object"])
        for field in required:
            if field not in data:
                errors.append(f"Field '{field}' is required")
        for field, rules in properties.items():
            if field not in data:
                continue
            value = data[field]
            expected = rules.get("type")
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            if expected in type_map and not isinstance(value, type_map[expected]):
                errors.append(f"Field '{field}' must be of type {expected}")
                continue
            if "minimum" in rules and isinstance(value, (int, float)):
                if value < rules["minimum"]:
                    errors.append(f"Field '{field}' must be >= {rules['minimum']}")
            if "maximum" in rules and isinstance(value, (int, float)):
                if value > rules["maximum"]:
                    errors.append(f"Field '{field}' must be <= {rules['maximum']}")

    return ValidationResult(is_valid=not errors, errors=errors)


# ---------------------------------------------------------------------------
# Performance-Hilfsfunktionen
# ---------------------------------------------------------------------------


class Timer:
    """Einfacher Zeitmesser, auch als Context-Manager nutzbar."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.elapsed = self.end_time - self.start_time
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class MemoryProfiler:
    """Misst den Speicherverbrauch eines Codeblocks."""

    def __init__(self):
        self._process = None
        self.start_memory: int = 0
        self.peak_memory: int = 0
        self.memory_delta: int = 0
        self._start_tracemalloc = False

    def _current_memory(self) -> int:
        try:
            import psutil

            if self._process is None:
                self._process = psutil.Process(os.getpid())
            return self._process.memory_info().rss
        except Exception:
            import tracemalloc

            if tracemalloc.is_tracing():
                current, _ = tracemalloc.get_traced_memory()
                return current
            return 0

    def start(self) -> "MemoryProfiler":
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._start_tracemalloc = True
        self.start_memory = self._current_memory()
        self.peak_memory = self.start_memory
        return self

    def stop(self) -> Dict[str, int]:
        import tracemalloc

        current = self._current_memory()
        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            self.peak_memory = max(self.peak_memory, current, peak)
        else:
            self.peak_memory = max(self.peak_memory, current)
        self.memory_delta = max(0, current - self.start_memory)
        if self.peak_memory == 0:
            # Fallback, falls keine Messung möglich war
            self.peak_memory = max(1, current)
        if self._start_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
        return {
            "start_memory": self.start_memory,
            "peak_memory": self.peak_memory,
            "memory_delta": self.memory_delta,
        }

    def __enter__(self) -> "MemoryProfiler":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def performance_monitor(operation: Optional[str] = None):
    """Decorator: misst Laufzeit und loggt sie."""
    def decorator(func: Callable) -> Callable:
        name = operation or func.__name__
        logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - started
                logger.debug(f"{name} took {duration:.4f}s")

        return wrapper

    return decorator


def cache_result(max_size: int = 128, ttl: Optional[float] = None):
    """Decorator: cached Ergebnisse (LRU, optional mit TTL)."""
    def decorator(func: Callable) -> Callable:
        cache: "OrderedDict[Any, Tuple[float, Any]]" = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                stored_at, value = cache[key]
                if ttl is None or (now - stored_at) < ttl:
                    cache.move_to_end(key)
                    return value
                del cache[key]
            result = func(*args, **kwargs)
            cache[key] = (now, result)
            cache.move_to_end(key)
            while len(cache) > max_size:
                cache.popitem(last=False)
            return result

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def memoize(func: Callable) -> Callable:
    """Decorator: unbegrenztes Caching der Ergebnisse."""
    cache: Dict[Any, Any] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
    return wrapper


def rate_limit(max_calls: int = 10, time_window: float = 1.0):
    """Decorator: begrenzt Aufrufe pro Zeitfenster.

    Überschreitungen lösen eine ``RuntimeError`` aus.
    """
    def decorator(func: Callable) -> Callable:
        calls: List[float] = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [t for t in calls if now - t < time_window]
            if len(calls) >= max_calls:
                raise RuntimeError(
                    f"Rate limit exceeded: max {max_calls} calls per {time_window}s"
                )
            calls.append(now)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Fehlerbehandlungs-Hilfsfunktionen
# ---------------------------------------------------------------------------


class ExecutionResult:
    """Ergebnis einer abgesicherten Ausführung."""

    def __init__(self, success: bool, result: Any = None, error: Optional[Exception] = None):
        self.success = success
        self.result = result
        self.error = error

    def __bool__(self) -> bool:
        return bool(self.success)

    def __repr__(self) -> str:
        return f"<ExecutionResult success={self.success} error={self.error!r}>"


def safe_execute(func: Callable, default: Any = None, *args, **kwargs) -> ExecutionResult:
    """Führt eine Funktion aus und fängt Exceptions ab."""
    try:
        return ExecutionResult(success=True, result=func(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - bewusst alles fangen
        return ExecutionResult(success=False, result=default, error=exc)


def error_handler(
    default_return: Any = None,
    handler: Optional[Callable[[Exception], Any]] = None,
):
    """Decorator: fängt Exceptions ab und liefert einen Ersatzwert."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if handler is not None:
                    return handler(exc)
                return default_return

        return wrapper

    return decorator


def exception_to_dict(exc: Exception) -> Dict[str, Any]:
    """Wandelt eine Exception in ein Dictionary um."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def log_exception(
    exc: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Loggt eine Exception samt Kontext."""
    log = logger or logging.getLogger(__name__)
    details = f" | context={context}" if context else ""
    log.error(f"{type(exc).__name__}: {exc}{details}", exc_info=True)


def create_error_response(
    message: str,
    error_code: Optional[str] = None,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Erzeugt eine standardisierte Fehlerantwort."""
    return {
        "error": {
            "message": message,
            "code": error_code,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "status_code": status_code,
    }
