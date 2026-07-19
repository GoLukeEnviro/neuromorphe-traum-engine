import pytest
import logging
import tempfile
import os
import json
from unittest.mock import patch
from pathlib import Path

from core.logging import setup_logging, get_logger, LoggerManager, ColoredFormatter, StructuredFormatter
from core.config import Settings


class TestLoggerSetup:
    """Tests für das grundlegende Logger-Setup und get_logger."""

    @pytest.fixture(autouse=True)
    def reset_logging_config(self):
        """Setzt die Logging-Konfiguration nach jedem Test zurück."""
        # Speichert den ursprünglichen Zustand der Logger
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        # Stellt sicher, dass der LoggerManager Singleton zurückgesetzt wird
        # Dies ist ein Workaround, da LoggerManager kein explizites Reset hat
        if hasattr(logging, '_logger_manager'):
            del logging._logger_manager

        yield

        # Stellt den ursprünglichen Zustand wieder her
        root_logger.handlers = []
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)

    @pytest.mark.unit
    def test_setup_logging_basic(self):
        """Testet das grundlegende Logging-Setup mit Standardeinstellungen."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                LOG_LEVEL="INFO",
                LOG_FILE_MAX_SIZE_MB=1,
                LOG_FILE_BACKUP_COUNT=1,
                PROCESSED_DIR=temp_dir # Um get_logs_path zu beeinflussen
            )
            # Überschreibe get_logs_path, um in temp_dir zu schreiben
            with patch.object(settings, 'get_logs_path', return_value=Path(temp_dir)):
                setup_logging(settings)

                logger = get_logger("test_logger")
                logger.info("Test message")

                log_file = Path(temp_dir) / 'neuromorphe_traum_engine.log'
                assert log_file.exists()

                with open(log_file, 'r') as f:
                    content = f.read()
                    assert "Test message" in content
                    assert "INFO" in content

    @pytest.mark.unit
    def test_get_logger_singleton(self):
        """Testet, ob get_logger Singleton-Instanzen zurückgibt."""
        settings = Settings(LOG_LEVEL="DEBUG")
        setup_logging(settings)

        logger1 = get_logger("my_module")
        logger2 = get_logger("my_module")
        logger3 = get_logger("another_module")

        assert logger1 is logger2
        assert logger1 is not logger3
        assert logger1.name == "my_module"
        assert logger3.name == "another_module"


class TestLoggerManager:
    """Tests für die LoggerManager-Klasse."""

    @pytest.mark.unit
    def test_logger_manager_initialization(self):
        """Testet die Initialisierung des LoggerManagers."""
        settings = Settings(LOG_LEVEL="DEBUG")
        manager = LoggerManager(settings)
        assert manager.settings == settings
        assert not manager._handlers_configured

    @pytest.mark.unit
    def test_logger_manager_setup_logging(self):
        """Testet die setup_logging-Methode des LoggerManagers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                LOG_LEVEL="DEBUG",
                LOG_FILE_MAX_SIZE_MB=1,
                LOG_FILE_BACKUP_COUNT=1,
                PROCESSED_DIR=temp_dir # Um get_logs_path zu beeinflussen
            )
            with patch.object(settings, 'get_logs_path', return_value=Path(temp_dir)):
                manager = LoggerManager(settings)
                manager.setup_logging()

                assert manager._handlers_configured
                root_logger = logging.getLogger()
                assert len(root_logger.handlers) >= 1 # Console handler

                log_file = Path(temp_dir) / 'neuromorphe_traum_engine.log'
                assert log_file.exists()

    @pytest.mark.unit
    def test_logger_manager_get_logger(self):
        """Testet das Abrufen von Loggern über den Manager."""
        settings = Settings(LOG_LEVEL="INFO")
        manager = LoggerManager(settings)
        manager.setup_logging()

        logger = manager.get_logger("test_manager_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_manager_logger"


class TestFormatters:
    """Tests für die benutzerdefinierten Formatter."""

    @pytest.mark.unit
    def test_colored_formatter(self):
        """Testet den ColoredFormatter."""
        formatter = ColoredFormatter(
            fmt="%(levelname)s - %(message)s",
            datefmt='%H:%M:%S'
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Info message", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        assert "\033[32m" in formatted # Grüner Farbcode für INFO
        assert "Info message" in formatted

    @pytest.mark.unit
    def test_structured_formatter(self):
        """Testet den StructuredFormatter."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_structured", level=logging.INFO, pathname="/app/main.py", lineno=10,
            msg="Structured log message", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_structured"
        assert log_data["message"] == "Structured log message"
        assert log_data["module"] == "main"
        assert log_data["line"] == 10
        assert "timestamp" in log_data
