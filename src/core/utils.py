from pathlib import Path
import re
import os
import shutil
import json
import hashlib
import time
import asyncio
import functools
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from dataclasses import dataclass
import uuid
import email_validator

T = TypeVar('T')

# File utilities
def ensure_directory(path: str) -> None:
    """Ensures that a directory exists; creates it if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_file_path(path: str) -> str:
    """Sanitizes a string to be safe for use as a file path component."""
    s = re.sub(r'[<>:"/\\|?*\s]+', '_', path)
    s = s.strip('._')
    return s[:200]

def get_file_size(path: str) -> int:
    """Returns file size in bytes."""
    return os.path.getsize(path)

def get_file_extension(path: str) -> str:
    """Returns file extension."""
    return Path(path).suffix.lower()

def copy_file(src: str, dst: str) -> None:
    """Copies a file."""
    shutil.copy2(src, dst)

def move_file(src: str, dst: str) -> None:
    """Moves a file."""
    shutil.move(src, dst)

def delete_file(path: str) -> None:
    """Deletes a file."""
    os.remove(path)

def list_files(directory: str, pattern: str = "*") -> List[str]:
    """Lists files in directory."""
    return [str(p) for p in Path(directory).glob(pattern) if p.is_file()]

def find_files(directory: str, pattern: str, recursive: bool = True) -> List[str]:
    """Finds files matching pattern."""
    glob_pattern = f"**/{pattern}" if recursive else pattern
    return [str(p) for p in Path(directory).glob(glob_pattern) if p.is_file()]

# Audio utilities
def validate_audio_format(path: str) -> bool:
    """Validates if file is a supported audio format."""
    valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    return get_file_extension(path) in valid_extensions

def get_audio_duration(path: str) -> float:
    """Returns audio duration in seconds."""
    # Placeholder - would use librosa in real implementation
    return 0.0

def normalize_audio_path(path: str) -> str:
    """Normalizes audio file path."""
    return str(Path(path).resolve())

def convert_sample_rate(input_path: str, output_path: str, target_rate: int) -> None:
    """Converts audio sample rate."""
    # Placeholder implementation
    pass

def get_audio_info(path: str) -> Dict[str, Any]:
    """Returns audio file information."""
    return {
        'duration': 0.0,
        'sample_rate': 44100,
        'channels': 2,
        'format': get_file_extension(path)
    }

def create_silence(duration: float, sample_rate: int = 44100) -> bytes:
    """Creates silence audio data."""
    return b'\x00' * int(duration * sample_rate * 2 * 2)  # 16-bit stereo

# Data utilities
def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merges two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflattens a dictionary."""
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    return result

def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitizes dictionary by removing None values."""
    return {k: v for k, v in d.items() if v is not None}

def serialize_data(data: Any) -> str:
    """Serializes data to JSON string."""
    return json.dumps(data, default=str)

def deserialize_data(data: str) -> Any:
    """Deserializes JSON string to data."""
    return json.loads(data)

def hash_data(data: Any) -> str:
    """Creates hash of data."""
    return hashlib.sha256(serialize_data(data).encode()).hexdigest()

def compare_data(data1: Any, data2: Any) -> bool:
    """Compares two data structures."""
    return hash_data(data1) == hash_data(data2)

# String utilities
def sanitize_filename(filename: str) -> str:
    """Sanitizes filename."""
    return safe_file_path(filename)

def generate_id(prefix: str = '') -> str:
    """Generates unique ID."""
    return f"{prefix}{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:8]

def slugify(text: str) -> str:
    """Creates URL-friendly slug."""
    return re.sub(r'[^a-zA-Z0-9-_]', '-', text.lower()).strip('-')

def truncate_string(text: str, max_length: int, suffix: str = '...') -> str:
    """Truncates string to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_duration(seconds: float) -> str:
    """Formats duration in human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def format_file_size(size_bytes: int) -> str:
    """Formats file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def validate_email(email: str) -> bool:
    """Validates email address."""
    try:
        email_validator.validate_email(email)
        return True
    except:
        return False

# Time utilities
def get_timestamp() -> str:
    """Returns current timestamp."""
    return datetime.utcnow().isoformat()

def format_timestamp(timestamp: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Formats timestamp."""
    return timestamp.strftime(format_str)

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parses timestamp string."""
    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

def time_ago(timestamp: datetime) -> str:
    """Returns human readable time ago."""
    now = datetime.utcnow()
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600} hours ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60} minutes ago"
    else:
        return "just now"

def duration_to_seconds(duration_str: str) -> float:
    """Converts duration string to seconds."""
    parts = duration_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float(duration_str)

def seconds_to_duration(seconds: float) -> str:
    """Converts seconds to duration string."""
    return format_duration(seconds)

# Async utilities
async def run_async(func: Callable, *args, **kwargs) -> Any:
    """Runs function asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)

async def gather_with_concurrency(tasks: List[Callable], max_concurrency: int = 10) -> List[Any]:
    """Runs tasks with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def run_with_semaphore(task):
        async with semaphore:
            return await task()
    
    return await asyncio.gather(*[run_with_semaphore(task) for task in tasks])

async def retry_async(func: Callable, max_retries: int = 3, delay: float = 1.0) -> Any:
    """Retries async function on failure."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(delay * (2 ** attempt))

async def timeout_async(func: Callable, timeout_seconds: float) -> Any:
    """Runs function with timeout."""
    return await asyncio.wait_for(func(), timeout=timeout_seconds)

def debounce(wait_time: float):
    """Debounce decorator."""
    def decorator(func):
        last_called = [0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_called[0] >= wait_time:
                last_called[0] = now
                return func(*args, **kwargs)
        return wrapper
    return decorator

def throttle(rate_limit: float):
    """Throttle decorator."""
    def decorator(func):
        last_called = [0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_called[0] >= 1.0 / rate_limit:
                last_called[0] = now
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Validation utilities
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None

def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """Validates data against schema."""
    errors = []
    for key, expected_type in schema.items():
        if key not in data:
            errors.append(f"Missing required field: {key}")
        elif not isinstance(data[key], expected_type):
            errors.append(f"Invalid type for {key}: expected {expected_type.__name__}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)

def validate_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> bool:
    """Validates if value is in range."""
    return min_val <= value <= max_val

def validate_type(value: Any, expected_type: type) -> bool:
    """Validates value type."""
    return isinstance(value, expected_type)

def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> ValidationResult:
    """Validates required fields."""
    errors = []
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)

# Performance utilities
class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

class MemoryProfiler:
    def __init__(self):
        self.initial_memory = 0
        self.peak_memory = 0
    
    def start(self):
        # Placeholder - would use psutil in real implementation
        self.initial_memory = 0
    
    def stop(self):
        # Placeholder - would use psutil in real implementation
        self.peak_memory = 0
    
    def get_usage(self) -> Dict[str, float]:
        return {'initial': self.initial_memory, 'peak': self.peak_memory}

def performance_monitor(func):
    """Performance monitoring decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer()
        timer.start()
        result = func(*args, **kwargs)
        timer.stop()
        print(f"{func.__name__} took {timer.elapsed():.4f} seconds")
        return result
    return wrapper

def cache_result(ttl: int = 300):
    """Cache result decorator."""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = hash_data((args, kwargs))
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator

def memoize(func):
    """Memoization decorator."""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = hash_data((args, kwargs))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

def rate_limit(calls_per_second: float):
    """Rate limiting decorator."""
    def decorator(func):
        last_called = [0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            time_since_last = now - last_called[0]
            min_interval = 1.0 / calls_per_second
            
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Error utilities
def safe_execute(func: Callable, default_value: Any = None, *args, **kwargs) -> Any:
    """Safely executes function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return default_value

def error_handler(default_value: Any = None):
    """Error handler decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return default_value
        return wrapper
    return decorator

def exception_to_dict(exception: Exception) -> Dict[str, Any]:
    """Converts exception to dictionary."""
    return {
        'type': type(exception).__name__,
        'message': str(exception),
        'args': exception.args
    }

def log_exception(exception: Exception, context: str = '') -> None:
    """Logs exception with context."""
    print(f"Exception in {context}: {exception}")

def create_error_response(message: str, code: int = 500) -> Dict[str, Any]:
    """Creates standardized error response."""
    return {
        'error': True,
        'message': message,
        'code': code,
        'timestamp': get_timestamp()
    }
