"""Tests für Logging-Service"""

import pytest
import logging
import tempfile
import shutil
import json
import time
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from io import StringIO

from src.services.logging_service import (
    LoggingService, LogManager, LogFormatter,
    LogHandler, LogFilter, LogRotator,
    LogAggregator, LogAnalyzer, LogExporter,
    LogLevel, LogFormat, LogDestination,
    LogEntry, LogMetrics, LogAlert,
    LogConfig, LogRule, LogPattern,
    StructuredLogger, AsyncLogHandler, LogBuffer,
    LogCompressor, LogArchiver, LogCleaner,
    LogMonitor, LogDashboard, LogSearch
)
from src.core.config import LoggingConfig
from src.core.exceptions import (
    LoggingError, LogConfigurationError, LogHandlerError,
    LogFormatterError, LogRotationError, LogExportError
)
from src.schemas.logging import (
    LogData, LogMetricsData, LogAlertData,
    LogConfigData, LogRuleData, LogPatternData,
    LogSearchData, LogStatsData, LogExportData
)


class TestLoggingService:
    """Tests für Logging-Service"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Temporäres Log-Verzeichnis für Tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def logging_config(self, temp_log_dir):
        """Logging-Konfiguration für Tests"""
        return LoggingConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file_enabled=True,
            file_path=str(Path(temp_log_dir) / "neuromorphe.log"),
            file_max_size="10MB",
            file_backup_count=5,
            console_enabled=True,
            console_format="%(levelname)s: %(message)s",
            structured_logging=True,
            json_format=True,
            async_logging=True,
            buffer_size=1000,
            flush_interval=5,
            compression_enabled=True,
            rotation_enabled=True,
            rotation_when="midnight",
            rotation_interval=1,
            archival_enabled=True,
            archival_days=30,
            cleanup_enabled=True,
            cleanup_days=90,
            monitoring_enabled=True,
            metrics_enabled=True,
            alert_enabled=True,
            export_enabled=True,
            export_format="json",
            search_enabled=True,
            dashboard_enabled=True
        )
    
    @pytest.fixture
    def logging_service(self, logging_config):
        """Logging-Service für Tests"""
        return LoggingService(logging_config)
    
    @pytest.mark.unit
    def test_logging_service_initialization(self, logging_config):
        """Test: Logging-Service-Initialisierung"""
        service = LoggingService(logging_config)
        
        assert service.config == logging_config
        assert service.level == logging.INFO
        assert service.file_enabled == True
        assert service.console_enabled == True
        assert isinstance(service.log_manager, LogManager)
        assert isinstance(service.log_formatter, LogFormatter)
        assert isinstance(service.log_aggregator, LogAggregator)
    
    @pytest.mark.unit
    def test_logging_service_invalid_config(self):
        """Test: Logging-Service mit ungültiger Konfiguration"""
        invalid_config = LoggingConfig(
            level="INVALID_LEVEL",  # Ungültiger Log-Level
            file_max_size="invalid_size",  # Ungültige Größe
            file_backup_count=-1,  # Negative Anzahl
            buffer_size=0  # Null-Puffergröße
        )
        
        with pytest.raises(LogConfigurationError):
            LoggingService(invalid_config)
    
    @pytest.mark.unit
    def test_setup_logging(self, logging_service, temp_log_dir):
        """Test: Logging einrichten"""
        # Logging einrichten
        logging_service.setup_logging()
        
        # Logger sollte konfiguriert sein
        logger = logging.getLogger("neuromorphe")
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        
        # File-Handler sollte existieren
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        
        # Console-Handler sollte existieren
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) > 0
    
    @pytest.mark.unit
    def test_log_messages(self, logging_service, temp_log_dir):
        """Test: Log-Nachrichten"""
        logging_service.setup_logging()
        logger = logging_service.get_logger("test_logger")
        
        # Verschiedene Log-Level testen
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Log-Datei sollte erstellt worden sein
        log_file = Path(temp_log_dir) / "neuromorphe.log"
        assert log_file.exists()
        
        # Log-Inhalt prüfen
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Debug sollte nicht geloggt werden (Level ist INFO)
        assert "Debug message" not in log_content
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content
        assert "Critical message" in log_content
    
    @pytest.mark.unit
    def test_structured_logging(self, logging_service):
        """Test: Strukturiertes Logging"""
        logging_service.setup_logging()
        logger = logging_service.get_structured_logger("structured_test")
        
        # Strukturierte Log-Nachricht
        logger.info(
            "User action performed",
            extra={
                "user_id": "12345",
                "action": "stem_upload",
                "file_size": 1024000,
                "duration": 2.5,
                "success": True
            }
        )
        
        # Log-Entry sollte strukturiert sein
        log_entries = logging_service.get_recent_logs(limit=1)
        assert len(log_entries) > 0
        
        entry = log_entries[0]
        assert entry.message == "User action performed"
        assert entry.extra["user_id"] == "12345"
        assert entry.extra["action"] == "stem_upload"
        assert entry.extra["success"] == True
    
    @pytest.mark.unit
    async def test_async_logging(self, logging_service):
        """Test: Asynchrones Logging"""
        logging_service.setup_logging()
        async_logger = logging_service.get_async_logger("async_test")
        
        # Viele Log-Nachrichten asynchron senden
        tasks = []
        for i in range(100):
            task = async_logger.info(f"Async message {i}")
            tasks.append(task)
        
        # Alle Tasks abwarten
        await asyncio.gather(*tasks)
        
        # Buffer sollte geleert werden
        await logging_service.flush_async_logs()
        
        # Alle Nachrichten sollten geloggt worden sein
        log_entries = logging_service.get_recent_logs(limit=100)
        assert len(log_entries) >= 100
        
        # Nachrichten sollten in der richtigen Reihenfolge sein
        messages = [entry.message for entry in log_entries if "Async message" in entry.message]
        assert len(messages) == 100
    
    @pytest.mark.unit
    def test_log_filtering(self, logging_service):
        """Test: Log-Filterung"""
        logging_service.setup_logging()
        
        # Filter hinzufügen
        sensitive_filter = LogFilter(
            name="sensitive_filter",
            pattern=r"password|secret|token",
            action="mask",
            replacement="***MASKED***"
        )
        
        logging_service.add_filter(sensitive_filter)
        
        logger = logging_service.get_logger("filter_test")
        
        # Nachrichten mit sensiblen Daten
        logger.info("User login with password: secret123")
        logger.info("API token: abc123token")
        logger.info("Normal message without sensitive data")
        
        # Log-Einträge abrufen
        log_entries = logging_service.get_recent_logs(limit=3)
        
        # Sensitive Daten sollten maskiert sein
        password_entry = next(e for e in log_entries if "password" in e.message)
        assert "***MASKED***" in password_entry.message
        assert "secret123" not in password_entry.message
        
        token_entry = next(e for e in log_entries if "token" in e.message)
        assert "***MASKED***" in token_entry.message
        assert "abc123token" not in token_entry.message
        
        # Normale Nachricht sollte unverändert sein
        normal_entry = next(e for e in log_entries if "Normal message" in e.message)
        assert "Normal message without sensitive data" in normal_entry.message
    
    @pytest.mark.unit
    def test_log_rotation(self, logging_service, temp_log_dir):
        """Test: Log-Rotation"""
        logging_service.setup_logging()
        logger = logging_service.get_logger("rotation_test")
        
        # Viele Log-Nachrichten schreiben (um Rotation auszulösen)
        large_message = "X" * 1000  # 1KB Nachricht
        
        for i in range(15000):  # 15MB an Daten
            logger.info(f"Large message {i}: {large_message}")
        
        # Log-Rotation sollte stattgefunden haben
        log_files = list(Path(temp_log_dir).glob("neuromorphe.log*"))
        
        # Sollte mehrere Log-Dateien geben (Original + rotierte)
        assert len(log_files) > 1
        
        # Backup-Dateien sollten existieren
        backup_files = [f for f in log_files if f.name != "neuromorphe.log"]
        assert len(backup_files) > 0
    
    @pytest.mark.unit
    def test_log_search(self, logging_service):
        """Test: Log-Suche"""
        logging_service.setup_logging()
        logger = logging_service.get_logger("search_test")
        
        # Test-Nachrichten loggen
        logger.info("User 12345 uploaded stem file.wav")
        logger.warning("High CPU usage detected: 85%")
        logger.error("Database connection failed")
        logger.info("User 67890 downloaded stem file.mp3")
        logger.critical("System out of memory")
        
        # Nach Benutzer-Aktionen suchen
        user_logs = logging_service.search_logs(
            query="User",
            level="INFO",
            start_time=datetime.now() - timedelta(minutes=1)
        )
        
        assert len(user_logs) == 2
        assert all("User" in entry.message for entry in user_logs)
        assert all(entry.level == "INFO" for entry in user_logs)
        
        # Nach Fehlern suchen
        error_logs = logging_service.search_logs(
            query="failed|memory",
            level=["ERROR", "CRITICAL"]
        )
        
        assert len(error_logs) == 2
        assert any("Database connection failed" in entry.message for entry in error_logs)
        assert any("System out of memory" in entry.message for entry in error_logs)
        
        # Nach spezifischer Benutzer-ID suchen
        user_12345_logs = logging_service.search_logs(query="12345")
        assert len(user_12345_logs) == 1
        assert "12345" in user_12345_logs[0].message
    
    @pytest.mark.unit
    def test_log_metrics(self, logging_service):
        """Test: Log-Metriken"""
        logging_service.setup_logging()
        logger = logging_service.get_logger("metrics_test")
        
        # Verschiedene Log-Level verwenden
        for _ in range(10):
            logger.info("Info message")
        
        for _ in range(5):
            logger.warning("Warning message")
        
        for _ in range(3):
            logger.error("Error message")
        
        for _ in range(1):
            logger.critical("Critical message")
        
        # Metriken abrufen
        metrics = logging_service.get_metrics()
        
        assert metrics.total_logs >= 19
        assert metrics.info_count >= 10
        assert metrics.warning_count >= 5
        assert metrics.error_count >= 3
        assert metrics.critical_count >= 1
        
        # Log-Rate berechnen
        assert metrics.logs_per_second > 0
        
        # Top-Logger
        assert "metrics_test" in [logger_stat.name for logger_stat in metrics.top_loggers]
    
    @pytest.mark.unit
    def test_log_alerts(self, logging_service):
        """Test: Log-Alerts"""
        logging_service.setup_logging()
        
        # Alert-Regel hinzufügen
        error_alert = LogAlert(
            name="high_error_rate",
            condition="error_count > 5 in 1 minute",
            level="WARNING",
            notification_channels=["email", "webhook"]
        )
        
        logging_service.add_alert_rule(error_alert)
        
        logger = logging_service.get_logger("alert_test")
        
        # Viele Fehler loggen (um Alert auszulösen)
        for i in range(10):
            logger.error(f"Test error {i}")
        
        # Alerts prüfen
        active_alerts = logging_service.get_active_alerts()
        
        # High error rate Alert sollte ausgelöst worden sein
        assert len(active_alerts) > 0
        assert any(alert.name == "high_error_rate" for alert in active_alerts)
    
    @pytest.mark.unit
    def test_log_export(self, logging_service, temp_log_dir):
        """Test: Log-Export"""
        logging_service.setup_logging()
        logger = logging_service.get_logger("export_test")
        
        # Test-Daten loggen
        logger.info("Export test message 1")
        logger.warning("Export test message 2")
        logger.error("Export test message 3")
        
        # Logs als JSON exportieren
        export_file = Path(temp_log_dir) / "exported_logs.json"
        
        logging_service.export_logs(
            output_file=str(export_file),
            format="json",
            start_time=datetime.now() - timedelta(minutes=1),
            end_time=datetime.now()
        )
        
        assert export_file.exists()
        
        # Export-Inhalt prüfen
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert isinstance(exported_data, list)
        assert len(exported_data) >= 3
        
        # Alle Test-Nachrichten sollten enthalten sein
        messages = [entry["message"] for entry in exported_data]
        assert any("Export test message 1" in msg for msg in messages)
        assert any("Export test message 2" in msg for msg in messages)
        assert any("Export test message 3" in msg for msg in messages)


class TestLogManager:
    """Tests für Log-Manager"""
    
    @pytest.fixture
    def log_manager(self, temp_log_dir):
        """Log-Manager für Tests"""
        config = LoggingConfig(
            file_path=str(Path(temp_log_dir) / "test.log"),
            file_max_size="1MB",
            file_backup_count=3,
            rotation_enabled=True
        )
        return LogManager(config)
    
    @pytest.mark.unit
    def test_create_file_handler(self, log_manager):
        """Test: File-Handler erstellen"""
        handler = log_manager.create_file_handler()
        
        assert isinstance(handler, logging.FileHandler)
        assert handler.level == logging.INFO
        
        # Formatter sollte gesetzt sein
        assert handler.formatter is not None
    
    @pytest.mark.unit
    def test_create_console_handler(self, log_manager):
        """Test: Console-Handler erstellen"""
        handler = log_manager.create_console_handler()
        
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.INFO
        
        # Formatter sollte gesetzt sein
        assert handler.formatter is not None
    
    @pytest.mark.unit
    def test_create_rotating_handler(self, log_manager):
        """Test: Rotating-Handler erstellen"""
        handler = log_manager.create_rotating_handler()
        
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 1024 * 1024  # 1MB
        assert handler.backupCount == 3
    
    @pytest.mark.unit
    def test_setup_logger(self, log_manager):
        """Test: Logger einrichten"""
        logger_name = "test_logger"
        logger = log_manager.setup_logger(logger_name)
        
        assert logger.name == logger_name
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        
        # Handler sollten konfiguriert sein
        for handler in logger.handlers:
            assert handler.formatter is not None
    
    @pytest.mark.unit
    def test_add_custom_handler(self, log_manager):
        """Test: Benutzerdefinierten Handler hinzufügen"""
        # Custom Handler erstellen
        custom_handler = logging.StreamHandler(StringIO())
        custom_handler.setLevel(logging.DEBUG)
        
        logger = log_manager.setup_logger("custom_test")
        log_manager.add_handler(logger, custom_handler)
        
        # Custom Handler sollte hinzugefügt worden sein
        assert custom_handler in logger.handlers
    
    @pytest.mark.unit
    def test_remove_handler(self, log_manager):
        """Test: Handler entfernen"""
        logger = log_manager.setup_logger("remove_test")
        initial_handler_count = len(logger.handlers)
        
        # Ersten Handler entfernen
        if logger.handlers:
            handler_to_remove = logger.handlers[0]
            log_manager.remove_handler(logger, handler_to_remove)
            
            assert len(logger.handlers) == initial_handler_count - 1
            assert handler_to_remove not in logger.handlers


class TestLogFormatter:
    """Tests für Log-Formatter"""
    
    @pytest.fixture
    def log_formatter(self):
        """Log-Formatter für Tests"""
        return LogFormatter(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            date_format="%Y-%m-%d %H:%M:%S",
            json_format=True,
            structured=True
        )
    
    @pytest.mark.unit
    def test_standard_formatting(self, log_formatter):
        """Test: Standard-Formatierung"""
        # Log-Record erstellen
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Standard-Formatierung
        formatted = log_formatter.format_standard(record)
        
        assert "test_logger" in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted
        assert "-" in formatted  # Trennzeichen
    
    @pytest.mark.unit
    def test_json_formatting(self, log_formatter):
        """Test: JSON-Formatierung"""
        # Log-Record mit Extra-Daten erstellen
        record = logging.LogRecord(
            name="json_logger",
            level=logging.WARNING,
            pathname="/test/path.py",
            lineno=42,
            msg="JSON test message",
            args=(),
            exc_info=None
        )
        
        # Extra-Daten hinzufügen
        record.user_id = "12345"
        record.action = "test_action"
        record.duration = 1.5
        
        # JSON-Formatierung
        formatted = log_formatter.format_json(record)
        
        # JSON parsen
        log_data = json.loads(formatted)
        
        assert log_data["logger"] == "json_logger"
        assert log_data["level"] == "WARNING"
        assert log_data["message"] == "JSON test message"
        assert log_data["user_id"] == "12345"
        assert log_data["action"] == "test_action"
        assert log_data["duration"] == 1.5
        assert "timestamp" in log_data
    
    @pytest.mark.unit
    def test_structured_formatting(self, log_formatter):
        """Test: Strukturierte Formatierung"""
        # Strukturierte Log-Daten
        structured_data = {
            "event": "user_login",
            "user_id": "67890",
            "ip_address": "192.168.1.100",
            "success": True,
            "login_time": "2024-01-15T10:30:00Z"
        }
        
        # Log-Record erstellen
        record = logging.LogRecord(
            name="auth_logger",
            level=logging.INFO,
            pathname="/auth/login.py",
            lineno=25,
            msg="User login attempt",
            args=(),
            exc_info=None
        )
        
        # Strukturierte Daten hinzufügen
        for key, value in structured_data.items():
            setattr(record, key, value)
        
        # Strukturierte Formatierung
        formatted = log_formatter.format_structured(record)
        
        # Alle strukturierten Daten sollten enthalten sein
        for key, value in structured_data.items():
            assert f"{key}={value}" in formatted or f'"{key}": "{value}"' in formatted
    
    @pytest.mark.unit
    def test_exception_formatting(self, log_formatter):
        """Test: Exception-Formatierung"""
        # Exception erstellen
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = True
        
        # Log-Record mit Exception erstellen
        record = logging.LogRecord(
            name="error_logger",
            level=logging.ERROR,
            pathname="/test/error.py",
            lineno=10,
            msg="An error occurred",
            args=(),
            exc_info=exc_info
        )
        
        # Exception-Formatierung
        formatted = log_formatter.format_exception(record)
        
        assert "An error occurred" in formatted
        assert "ValueError" in formatted
        assert "Test exception" in formatted
        assert "Traceback" in formatted
    
    @pytest.mark.unit
    def test_sensitive_data_masking(self, log_formatter):
        """Test: Maskierung sensibler Daten"""
        # Sensitive Patterns hinzufügen
        log_formatter.add_sensitive_pattern(r"password=\w+", "password=***")
        log_formatter.add_sensitive_pattern(r"token=\w+", "token=***")
        
        # Log-Record mit sensiblen Daten
        record = logging.LogRecord(
            name="security_logger",
            level=logging.INFO,
            pathname="/auth/login.py",
            lineno=15,
            msg="Login attempt with password=secret123 and token=abc456",
            args=(),
            exc_info=None
        )
        
        # Formatierung mit Maskierung
        formatted = log_formatter.format_with_masking(record)
        
        assert "password=***" in formatted
        assert "token=***" in formatted
        assert "secret123" not in formatted
        assert "abc456" not in formatted


class TestLoggingServiceIntegration:
    """Integrationstests für Logging-Service"""
    
    @pytest.mark.integration
    def test_full_logging_workflow(self, temp_log_dir):
        """Test: Vollständiger Logging-Workflow"""
        config = LoggingConfig(
            level="DEBUG",
            file_enabled=True,
            file_path=str(Path(temp_log_dir) / "integration.log"),
            console_enabled=True,
            structured_logging=True,
            json_format=True,
            rotation_enabled=True,
            file_max_size="1MB",
            metrics_enabled=True,
            search_enabled=True
        )
        
        service = LoggingService(config)
        service.setup_logging()
        
        # 1. Verschiedene Logger erstellen
        auth_logger = service.get_logger("auth")
        api_logger = service.get_logger("api")
        db_logger = service.get_logger("database")
        
        # 2. Strukturierte Logs
        structured_logger = service.get_structured_logger("structured")
        
        # 3. Verschiedene Log-Nachrichten
        auth_logger.info("User login successful", extra={"user_id": "12345"})
        api_logger.warning("Rate limit approaching", extra={"endpoint": "/api/stems"})
        db_logger.error("Connection timeout", extra={"host": "localhost", "port": 5432})
        
        structured_logger.info(
            "Stem processing completed",
            extra={
                "stem_id": "stem_67890",
                "duration": 2.5,
                "file_size": 1024000,
                "success": True
            }
        )
        
        # 4. Log-Suche
        user_logs = service.search_logs(query="user_id")
        assert len(user_logs) > 0
        
        api_logs = service.search_logs(query="Rate limit")
        assert len(api_logs) > 0
        
        error_logs = service.search_logs(level="ERROR")
        assert len(error_logs) > 0
        
        # 5. Metriken abrufen
        metrics = service.get_metrics()
        assert metrics.total_logs >= 4
        assert metrics.info_count >= 2
        assert metrics.warning_count >= 1
        assert metrics.error_count >= 1
        
        # 6. Log-Export
        export_file = Path(temp_log_dir) / "exported.json"
        service.export_logs(
            output_file=str(export_file),
            format="json"
        )
        
        assert export_file.exists()
        
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) >= 4
    
    @pytest.mark.performance
    async def test_logging_service_performance(self, temp_log_dir):
        """Test: Logging-Service-Performance"""
        config = LoggingConfig(
            level="INFO",
            file_enabled=True,
            file_path=str(Path(temp_log_dir) / "performance.log"),
            async_logging=True,
            buffer_size=10000,
            flush_interval=1
        )
        
        service = LoggingService(config)
        service.setup_logging()
        
        # Performance-Test: Synchrones Logging
        sync_logger = service.get_logger("sync_perf")
        
        start_time = time.time()
        
        for i in range(1000):
            sync_logger.info(f"Sync log message {i}")
        
        sync_time = time.time() - start_time
        
        # Performance-Test: Asynchrones Logging
        async_logger = service.get_async_logger("async_perf")
        
        start_time = time.time()
        
        tasks = []
        for i in range(1000):
            task = async_logger.info(f"Async log message {i}")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        await service.flush_async_logs()
        
        async_time = time.time() - start_time
        
        # Asynchrones Logging sollte schneller sein
        assert async_time < sync_time
        
        # Performance-Test: Strukturiertes Logging
        structured_logger = service.get_structured_logger("struct_perf")
        
        start_time = time.time()
        
        for i in range(500):
            structured_logger.info(
                f"Structured message {i}",
                extra={
                    "iteration": i,
                    "timestamp": time.time(),
                    "data": {"key": f"value_{i}"},
                    "success": i % 2 == 0
                }
            )
        
        structured_time = time.time() - start_time
        
        # Sollte unter 2 Sekunden für 500 strukturierte Logs dauern
        assert structured_time < 2.0
        
        # Performance-Test: Log-Suche
        start_time = time.time()
        
        for _ in range(100):
            results = service.search_logs(query="message", limit=10)
            assert len(results) <= 10
        
        search_time = time.time() - start_time
        
        # Sollte unter 1 Sekunde für 100 Suchen dauern
        assert search_time < 1.0