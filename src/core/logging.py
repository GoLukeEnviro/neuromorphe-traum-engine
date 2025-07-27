"""Logging-Konfiguration für die Neuromorphe Traum-Engine v2.0

Diese Datei definiert das Logging-System für die Anwendung.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from functools import lru_cache

from .config import get_settings


class ColoredFormatter(logging.Formatter):
    """Farbiger Formatter für Console-Output"""
    
    # ANSI-Farbcodes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Grün
        'WARNING': '\033[33m',    # Gelb
        'ERROR': '\033[31m',      # Rot
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Farbe basierend auf Log-Level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Original-Formatter anwenden
        formatted = super().format(record)
        
        # Farbe nur für Terminal-Output
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            return f"{color}{formatted}{reset}"
        return formatted


class StructuredFormatter(logging.Formatter):
    """Strukturierter Formatter für JSON-ähnliche Logs"""
    
    def format(self, record):
        # Basis-Informationen
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Exception-Informationen hinzufügen
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Extra-Felder hinzufügen
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        # Als JSON-String formatieren
        import json
        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))


class PerformanceFilter(logging.Filter):
    """Filter für Performance-Logging"""
    
    def filter(self, record):
        # Performance-Metriken hinzufügen
        if hasattr(record, 'duration'):
            if record.duration > 1.0:  # Langsame Operationen markieren
                record.performance_warning = True
        
        return True


class SensitiveDataFilter(logging.Filter):
    """Filter zum Entfernen sensibler Daten aus Logs"""
    
    SENSITIVE_PATTERNS = [
        'password', 'token', 'key', 'secret', 'auth',
        'credential', 'api_key', 'access_token'
    ]
    
    def filter(self, record):
        # Nachricht auf sensible Daten prüfen
        message = record.getMessage().lower()
        
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                # Sensible Daten durch Platzhalter ersetzen
                record.msg = record.msg.replace(
                    record.args[0] if record.args else '',
                    '[REDACTED]'
                )
                break
        
        return True


class LoggerManager:
    """Manager für Logger-Konfiguration"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers_configured = False
        
    def setup_logging(self):
        """Logging-System konfigurieren"""
        if self._handlers_configured:
            return
        
        # Root-Logger konfigurieren
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.settings.LOG_LEVEL))
        
        # Bestehende Handler entfernen
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console-Handler
        console_handler = self._create_console_handler()
        root_logger.addHandler(console_handler)
        
        # File-Handler
        file_handler = self._create_file_handler()
        if file_handler:
            root_logger.addHandler(file_handler)
        
        # Error-File-Handler
        error_handler = self._create_error_handler()
        if error_handler:
            root_logger.addHandler(error_handler)
        
        # Performance-Handler (wenn aktiviert)
        if self.settings.PERFORMANCE_TRACKING:
            perf_handler = self._create_performance_handler()
            if perf_handler:
                root_logger.addHandler(perf_handler)
        
        # Externe Logger konfigurieren
        self._configure_external_loggers()
        
        self._handlers_configured = True
        
        # Startup-Nachricht
        logger = self.get_logger('neuromorphe_traum_engine')
        logger.info(f"Logging system initialized - Level: {self.settings.LOG_LEVEL}")
    
    def _create_console_handler(self) -> logging.Handler:
        """Console-Handler erstellen"""
        handler = logging.StreamHandler(sys.stdout)
        
        if self.settings.DEBUG or self.settings.DEVELOPMENT_MODE:
            # Farbiger Formatter für Entwicklung
            formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            # Einfacher Formatter für Produktion
            formatter = logging.Formatter(
                fmt=self.settings.LOG_FORMAT,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        handler.setFormatter(formatter)
        handler.setLevel(getattr(logging, self.settings.LOG_LEVEL))
        
        # Filter hinzufügen
        handler.addFilter(SensitiveDataFilter())
        
        return handler
    
    def _create_file_handler(self) -> Optional[logging.Handler]:
        """File-Handler erstellen"""
        try:
            log_file = self.settings.get_logs_path() / 'neuromorphe_traum_engine.log'
            
            # Rotating File Handler
            handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self.settings.LOG_FILE_MAX_SIZE_MB * 1024 * 1024,
                backupCount=self.settings.LOG_FILE_BACKUP_COUNT,
                encoding='utf-8'
            )
            
            # Strukturierter Formatter für Datei-Logs
            if self.settings.DEBUG:
                formatter = logging.Formatter(
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            else:
                formatter = StructuredFormatter()
            
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)  # Alle Logs in Datei
            
            # Filter hinzufügen
            handler.addFilter(SensitiveDataFilter())
            
            return handler
            
        except Exception as e:
            print(f"Failed to create file handler: {e}")
            return None
    
    def _create_error_handler(self) -> Optional[logging.Handler]:
        """Error-Handler für separate Error-Logs erstellen"""
        try:
            error_file = self.settings.get_logs_path() / 'errors.log'
            
            handler = logging.handlers.RotatingFileHandler(
                filename=error_file,
                maxBytes=self.settings.LOG_FILE_MAX_SIZE_MB * 1024 * 1024,
                backupCount=self.settings.LOG_FILE_BACKUP_COUNT,
                encoding='utf-8'
            )
            
            # Nur Errors und Critical
            handler.setLevel(logging.ERROR)
            
            # Detaillierter Formatter für Errors
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d\n'
                    'Function: %(funcName)s\n'
                    'Message: %(message)s\n'
                    '%(exc_text)s\n' + '-' * 80,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            handler.setFormatter(formatter)
            
            return handler
            
        except Exception as e:
            print(f"Failed to create error handler: {e}")
            return None
    
    def _create_performance_handler(self) -> Optional[logging.Handler]:
        """Performance-Handler erstellen"""
        try:
            perf_file = self.settings.get_logs_path() / 'performance.log'
            
            handler = logging.handlers.RotatingFileHandler(
                filename=perf_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=3,
                encoding='utf-8'
            )
            
            # Performance-spezifischer Formatter
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            
            # Performance-Filter
            handler.addFilter(PerformanceFilter())
            
            return handler
            
        except Exception as e:
            print(f"Failed to create performance handler: {e}")
            return None
    
    def _configure_external_loggers(self):
        """Externe Logger konfigurieren"""
        # SQLAlchemy Logger
        sqlalchemy_logger = logging.getLogger('sqlalchemy')
        if self.settings.DATABASE_ECHO:
            sqlalchemy_logger.setLevel(logging.INFO)
        else:
            sqlalchemy_logger.setLevel(logging.WARNING)
        
        # FastAPI Logger
        fastapi_logger = logging.getLogger('fastapi')
        fastapi_logger.setLevel(logging.INFO)
        
        # Uvicorn Logger
        uvicorn_logger = logging.getLogger('uvicorn')
        uvicorn_logger.setLevel(logging.INFO)
        
        # Transformers Logger (für CLAP)
        transformers_logger = logging.getLogger('transformers')
        transformers_logger.setLevel(logging.WARNING)
        
        # Librosa Logger
        librosa_logger = logging.getLogger('librosa')
        librosa_logger.setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Logger für spezifischen Namen abrufen"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Performance-Metrik loggen"""
        if not self.settings.PERFORMANCE_TRACKING:
            return
        
        perf_logger = self.get_logger('performance')
        
        # Performance-Daten strukturieren
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        # Als JSON loggen
        import json
        perf_logger.info(json.dumps(perf_data, ensure_ascii=False))
    
    def log_system_metrics(self, metrics: Dict[str, Any]):
        """System-Metriken loggen"""
        if not self.settings.ENABLE_METRICS:
            return
        
        metrics_logger = self.get_logger('system_metrics')
        
        # Metriken mit Timestamp
        metrics_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        }
        
        import json
        metrics_logger.info(json.dumps(metrics_data, ensure_ascii=False))


# Globaler Logger-Manager
_logger_manager: Optional[LoggerManager] = None


def get_logger_manager() -> LoggerManager:
    """Logger-Manager-Instanz abrufen"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
        _logger_manager.setup_logging()
    return _logger_manager


@lru_cache(maxsize=128)
def get_logger(name: str) -> logging.Logger:
    """Logger für spezifischen Namen abrufen (gecacht)"""
    manager = get_logger_manager()
    return manager.get_logger(name)


def setup_logging(settings=None):
    """Logging-System initialisieren"""
    global _logger_manager
    _logger_manager = LoggerManager(settings)
    _logger_manager.setup_logging()


def log_performance(operation: str, duration: float, **kwargs):
    """Performance-Metrik loggen"""
    manager = get_logger_manager()
    manager.log_performance(operation, duration, **kwargs)


def log_system_metrics(metrics: Dict[str, Any]):
    """System-Metriken loggen"""
    manager = get_logger_manager()
    manager.log_system_metrics(metrics)


# Context Manager für Performance-Logging
class PerformanceLogger:
    """Context Manager für automatisches Performance-Logging"""
    
    def __init__(self, operation: str, logger_name: str = None, **kwargs):
        self.operation = operation
        self.logger = get_logger(logger_name or 'performance')
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            
            if exc_type is None:
                # Erfolgreiche Operation
                self.logger.info(
                    f"Operation completed: {self.operation}",
                    extra={'duration': duration, **self.kwargs}
                )
                log_performance(self.operation, duration, **self.kwargs)
            else:
                # Fehlerhafte Operation
                self.logger.error(
                    f"Operation failed: {self.operation} - {exc_val}",
                    extra={'duration': duration, 'error': str(exc_val), **self.kwargs}
                )


# Decorator für automatisches Performance-Logging
def log_performance_decorator(operation: str = None, logger_name: str = None):
    """Decorator für automatisches Performance-Logging"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            with PerformanceLogger(op_name, logger_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Hilfsfunktionen für strukturiertes Logging
def log_audio_processing(logger: logging.Logger, file_path: str, operation: str, 
                        duration: float = None, **kwargs):
    """Audio-Verarbeitung loggen"""
    extra_data = {
        'file_path': file_path,
        'operation': operation,
        'category': 'audio_processing',
        **kwargs
    }
    
    if duration is not None:
        extra_data['duration'] = duration
    
    logger.info(f"Audio processing: {operation} - {Path(file_path).name}", extra=extra_data)


def log_api_request(logger: logging.Logger, method: str, endpoint: str, 
                   status_code: int, duration: float, **kwargs):
    """API-Request loggen"""
    extra_data = {
        'method': method,
        'endpoint': endpoint,
        'status_code': status_code,
        'duration': duration,
        'category': 'api_request',
        **kwargs
    }
    
    logger.info(f"{method} {endpoint} - {status_code}", extra=extra_data)


def log_database_operation(logger: logging.Logger, operation: str, table: str, 
                          duration: float = None, **kwargs):
    """Datenbank-Operation loggen"""
    extra_data = {
        'operation': operation,
        'table': table,
        'category': 'database',
        **kwargs
    }
    
    if duration is not None:
        extra_data['duration'] = duration
    
    logger.info(f"Database {operation}: {table}", extra=extra_data)


# Logging für Entwicklung vs. Produktion
def configure_for_environment(environment: str = "development"):
    """Logging für spezifische Umgebung konfigurieren"""
    if environment == "development":
        # Mehr Details für Entwicklung
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Alle externen Logger auf INFO
        for logger_name in ['sqlalchemy', 'fastapi', 'uvicorn']:
            logging.getLogger(logger_name).setLevel(logging.INFO)
    
    elif environment == "production":
        # Weniger Details für Produktion
        logging.getLogger().setLevel(logging.INFO)
        
        # Externe Logger auf WARNING
        for logger_name in ['sqlalchemy', 'transformers', 'librosa']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    elif environment == "testing":
        # Minimales Logging für Tests
        logging.getLogger().setLevel(logging.WARNING)
        
        # Alle externen Logger stumm
        for logger_name in ['sqlalchemy', 'fastapi', 'uvicorn', 'transformers', 'librosa']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)