"""Tests für Core-Logging"""

import pytest
import logging
import tempfile
import os
import json
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
from pathlib import Path

from src.core.logging import (
    setup_logging, get_logger, LoggerManager, StructuredLogger,
    PerformanceLogger, ErrorLogger, AuditLogger, SecurityLogger,
    LogFormatter, JSONFormatter, ColoredFormatter,
    LogFilter, SensitiveDataFilter, RateLimitFilter,
    LogRotationHandler, DatabaseLogHandler, WebSocketLogHandler
)
from src.core.config import LoggingConfig


class TestLoggerSetup:
    """Tests für Logger-Setup"""
    
    @pytest.mark.unit
    def test_setup_logging_basic(self):
        """Test: Basis-Logging-Setup"""
        config = LoggingConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file_path="test.log"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            config.file_path = log_file
            
            setup_logging(config)
            
            # Logger testen
            logger = logging.getLogger("test_logger")
            logger.info("Test message")
            
            # Log-Datei sollte existieren
            assert os.path.exists(log_file)
            
            # Log-Inhalt prüfen
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
                assert "INFO" in content
    
    @pytest.mark.unit
    def test_setup_logging_with_rotation(self):
        """Test: Logging-Setup mit Rotation"""
        config = LoggingConfig(
            level="DEBUG",
            file_path="rotating.log",
            max_file_size="1MB",
            backup_count=3,
            rotation_enabled=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "rotating.log")
            config.file_path = log_file
            
            setup_logging(config)
            
            logger = logging.getLogger("rotation_test")
            
            # Viele Nachrichten schreiben
            for i in range(1000):
                logger.info(f"Test message {i} with some additional content to increase file size")
            
            # Log-Datei sollte existieren
            assert os.path.exists(log_file)
    
    @pytest.mark.unit
    def test_get_logger(self):
        """Test: Logger abrufen"""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        logger3 = get_logger("other_module")
        
        # Gleicher Name sollte gleichen Logger zurückgeben
        assert logger1 is logger2
        
        # Verschiedene Namen sollten verschiedene Logger zurückgeben
        assert logger1 is not logger3
        
        # Logger sollten korrekte Namen haben
        assert logger1.name == "neuromorphe.test_module"
        assert logger3.name == "neuromorphe.other_module"


class TestLoggerManager:
    """Tests für LoggerManager"""
    
    @pytest.mark.unit
    def test_logger_manager_creation(self):
        """Test: LoggerManager erstellen"""
        config = LoggingConfig(level="INFO")
        manager = LoggerManager(config)
        
        assert manager.config == config
        assert manager.loggers == {}
        assert manager.handlers == []
    
    @pytest.mark.unit
    def test_logger_manager_get_logger(self):
        """Test: Logger über Manager abrufen"""
        config = LoggingConfig(level="DEBUG")
        manager = LoggerManager(config)
        
        logger1 = manager.get_logger("test1")
        logger2 = manager.get_logger("test1")
        logger3 = manager.get_logger("test2")
        
        assert logger1 is logger2
        assert logger1 is not logger3
        assert len(manager.loggers) == 2
    
    @pytest.mark.unit
    def test_logger_manager_configure_handlers(self):
        """Test: Handler-Konfiguration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "manager_test.log")
            
            config = LoggingConfig(
                level="INFO",
                file_path=log_file,
                console_enabled=True,
                json_format=True
            )
            
            manager = LoggerManager(config)
            manager.configure_handlers()
            
            assert len(manager.handlers) >= 1  # Mindestens File-Handler
            
            # Logger testen
            logger = manager.get_logger("handler_test")
            logger.info("Handler test message")
            
            assert os.path.exists(log_file)
    
    @pytest.mark.unit
    def test_logger_manager_update_level(self):
        """Test: Log-Level aktualisieren"""
        config = LoggingConfig(level="INFO")
        manager = LoggerManager(config)
        
        logger = manager.get_logger("level_test")
        
        # Initial INFO-Level
        assert logger.level == logging.INFO
        
        # Level auf DEBUG ändern
        manager.update_level("DEBUG")
        assert logger.level == logging.DEBUG
        
        # Level auf ERROR ändern
        manager.update_level("ERROR")
        assert logger.level == logging.ERROR


class TestStructuredLogger:
    """Tests für StructuredLogger"""
    
    @pytest.mark.unit
    def test_structured_logger_creation(self):
        """Test: StructuredLogger erstellen"""
        logger = StructuredLogger("test_structured")
        
        assert logger.name == "neuromorphe.test_structured"
        assert hasattr(logger, 'log_event')
        assert hasattr(logger, 'log_metric')
        assert hasattr(logger, 'log_error')
    
    @pytest.mark.unit
    def test_structured_logger_log_event(self):
        """Test: Event-Logging"""
        with patch('logging.Logger.info') as mock_info:
            logger = StructuredLogger("event_test")
            
            logger.log_event(
                event_type="user_action",
                action="create_arrangement",
                user_id="user_123",
                metadata={"prompt": "test prompt", "duration": 120}
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            
            # JSON-Format prüfen
            log_data = json.loads(call_args)
            assert log_data["event_type"] == "user_action"
            assert log_data["action"] == "create_arrangement"
            assert log_data["user_id"] == "user_123"
            assert log_data["metadata"]["prompt"] == "test prompt"
    
    @pytest.mark.unit
    def test_structured_logger_log_metric(self):
        """Test: Metric-Logging"""
        with patch('logging.Logger.info') as mock_info:
            logger = StructuredLogger("metric_test")
            
            logger.log_metric(
                metric_name="render_time",
                value=45.2,
                unit="seconds",
                tags={"format": "wav", "quality": "high"}
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["metric_name"] == "render_time"
            assert log_data["value"] == 45.2
            assert log_data["unit"] == "seconds"
            assert log_data["tags"]["format"] == "wav"
    
    @pytest.mark.unit
    def test_structured_logger_log_error(self):
        """Test: Error-Logging"""
        with patch('logging.Logger.error') as mock_error:
            logger = StructuredLogger("error_test")
            
            try:
                raise ValueError("Test error")
            except Exception as e:
                logger.log_error(
                    error=e,
                    context={"operation": "test_operation", "file": "test.wav"},
                    user_id="user_123"
                )
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["error_type"] == "ValueError"
            assert log_data["error_message"] == "Test error"
            assert log_data["context"]["operation"] == "test_operation"
            assert log_data["user_id"] == "user_123"


class TestPerformanceLogger:
    """Tests für PerformanceLogger"""
    
    @pytest.mark.unit
    def test_performance_logger_timing(self):
        """Test: Performance-Timing"""
        with patch('logging.Logger.info') as mock_info:
            logger = PerformanceLogger("perf_test")
            
            with logger.time_operation("test_operation", metadata={"size": "large"}):
                import time
                time.sleep(0.1)  # 100ms warten
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["operation"] == "test_operation"
            assert log_data["duration_ms"] >= 100
            assert log_data["metadata"]["size"] == "large"
    
    @pytest.mark.unit
    def test_performance_logger_memory_usage(self):
        """Test: Memory-Usage-Logging"""
        with patch('logging.Logger.info') as mock_info:
            logger = PerformanceLogger("memory_test")
            
            logger.log_memory_usage(
                operation="model_loading",
                memory_before=1024,
                memory_after=2048,
                peak_memory=2200
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["operation"] == "model_loading"
            assert log_data["memory_before_mb"] == 1024
            assert log_data["memory_after_mb"] == 2048
            assert log_data["memory_delta_mb"] == 1024
            assert log_data["peak_memory_mb"] == 2200
    
    @pytest.mark.unit
    def test_performance_logger_throughput(self):
        """Test: Throughput-Logging"""
        with patch('logging.Logger.info') as mock_info:
            logger = PerformanceLogger("throughput_test")
            
            logger.log_throughput(
                operation="audio_processing",
                items_processed=1000,
                duration_seconds=10.5,
                unit="samples"
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["operation"] == "audio_processing"
            assert log_data["items_processed"] == 1000
            assert log_data["duration_seconds"] == 10.5
            assert abs(log_data["throughput"] - 95.24) < 0.01  # 1000/10.5
            assert log_data["unit"] == "samples"


class TestErrorLogger:
    """Tests für ErrorLogger"""
    
    @pytest.mark.unit
    def test_error_logger_exception_logging(self):
        """Test: Exception-Logging"""
        with patch('logging.Logger.error') as mock_error:
            logger = ErrorLogger("error_test")
            
            try:
                1 / 0
            except Exception as e:
                logger.log_exception(
                    exception=e,
                    context={"operation": "division", "values": [1, 0]},
                    severity="high",
                    user_id="user_123"
                )
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["exception_type"] == "ZeroDivisionError"
            assert "division by zero" in log_data["exception_message"]
            assert log_data["severity"] == "high"
            assert log_data["context"]["operation"] == "division"
            assert "traceback" in log_data
    
    @pytest.mark.unit
    def test_error_logger_validation_error(self):
        """Test: Validation-Error-Logging"""
        with patch('logging.Logger.warning') as mock_warning:
            logger = ErrorLogger("validation_test")
            
            validation_errors = [
                {"field": "duration", "message": "must be positive"},
                {"field": "prompt", "message": "cannot be empty"}
            ]
            
            logger.log_validation_error(
                errors=validation_errors,
                input_data={"duration": -10, "prompt": ""},
                endpoint="/api/arrangements"
            )
            
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["error_type"] == "validation_error"
            assert len(log_data["validation_errors"]) == 2
            assert log_data["endpoint"] == "/api/arrangements"
    
    @pytest.mark.unit
    def test_error_logger_system_error(self):
        """Test: System-Error-Logging"""
        with patch('logging.Logger.critical') as mock_critical:
            logger = ErrorLogger("system_test")
            
            logger.log_system_error(
                component="database",
                error_message="Connection pool exhausted",
                system_state={"active_connections": 100, "max_connections": 100},
                impact="service_unavailable"
            )
            
            mock_critical.assert_called_once()
            call_args = mock_critical.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["error_type"] == "system_error"
            assert log_data["component"] == "database"
            assert log_data["impact"] == "service_unavailable"
            assert log_data["system_state"]["active_connections"] == 100


class TestAuditLogger:
    """Tests für AuditLogger"""
    
    @pytest.mark.unit
    def test_audit_logger_user_action(self):
        """Test: User-Action-Logging"""
        with patch('logging.Logger.info') as mock_info:
            logger = AuditLogger("audit_test")
            
            logger.log_user_action(
                user_id="user_123",
                action="create_arrangement",
                resource="arrangement_456",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0...",
                metadata={"prompt": "techno beat", "duration": 180}
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["audit_type"] == "user_action"
            assert log_data["user_id"] == "user_123"
            assert log_data["action"] == "create_arrangement"
            assert log_data["resource"] == "arrangement_456"
            assert log_data["ip_address"] == "192.168.1.100"
    
    @pytest.mark.unit
    def test_audit_logger_data_access(self):
        """Test: Data-Access-Logging"""
        with patch('logging.Logger.info') as mock_info:
            logger = AuditLogger("data_access_test")
            
            logger.log_data_access(
                user_id="user_123",
                resource_type="stem",
                resource_id="stem_789",
                access_type="read",
                query_params={"format": "wav", "quality": "high"}
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["audit_type"] == "data_access"
            assert log_data["resource_type"] == "stem"
            assert log_data["access_type"] == "read"
            assert log_data["query_params"]["format"] == "wav"
    
    @pytest.mark.unit
    def test_audit_logger_system_change(self):
        """Test: System-Change-Logging"""
        with patch('logging.Logger.warning') as mock_warning:
            logger = AuditLogger("system_change_test")
            
            logger.log_system_change(
                admin_user="admin_123",
                change_type="configuration",
                component="database",
                old_value={"max_connections": 50},
                new_value={"max_connections": 100},
                reason="Increased load"
            )
            
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["audit_type"] == "system_change"
            assert log_data["admin_user"] == "admin_123"
            assert log_data["change_type"] == "configuration"
            assert log_data["old_value"]["max_connections"] == 50
            assert log_data["new_value"]["max_connections"] == 100


class TestSecurityLogger:
    """Tests für SecurityLogger"""
    
    @pytest.mark.unit
    def test_security_logger_authentication_attempt(self):
        """Test: Authentication-Attempt-Logging"""
        with patch('logging.Logger.warning') as mock_warning:
            logger = SecurityLogger("security_test")
            
            logger.log_authentication_attempt(
                user_id="user_123",
                success=False,
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0...",
                failure_reason="invalid_password",
                attempt_count=3
            )
            
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["security_event"] == "authentication_attempt"
            assert log_data["success"] == False
            assert log_data["failure_reason"] == "invalid_password"
            assert log_data["attempt_count"] == 3
    
    @pytest.mark.unit
    def test_security_logger_suspicious_activity(self):
        """Test: Suspicious-Activity-Logging"""
        with patch('logging.Logger.error') as mock_error:
            logger = SecurityLogger("suspicious_test")
            
            logger.log_suspicious_activity(
                activity_type="rate_limit_exceeded",
                ip_address="192.168.1.100",
                user_id="user_123",
                details={
                    "requests_per_minute": 1000,
                    "limit": 100,
                    "endpoint": "/api/analyze"
                },
                severity="high"
            )
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["security_event"] == "suspicious_activity"
            assert log_data["activity_type"] == "rate_limit_exceeded"
            assert log_data["severity"] == "high"
            assert log_data["details"]["requests_per_minute"] == 1000
    
    @pytest.mark.unit
    def test_security_logger_access_violation(self):
        """Test: Access-Violation-Logging"""
        with patch('logging.Logger.error') as mock_error:
            logger = SecurityLogger("access_test")
            
            logger.log_access_violation(
                user_id="user_123",
                resource="admin_panel",
                required_permission="admin",
                user_permissions=["user", "read"],
                ip_address="192.168.1.100"
            )
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            
            log_data = json.loads(call_args)
            assert log_data["security_event"] == "access_violation"
            assert log_data["resource"] == "admin_panel"
            assert log_data["required_permission"] == "admin"
            assert "user" in log_data["user_permissions"]


class TestLogFormatters:
    """Tests für Log-Formatter"""
    
    @pytest.mark.unit
    def test_json_formatter(self):
        """Test: JSON-Formatter"""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["pathname"] == "/test/path.py"
        assert log_data["lineno"] == 42
        assert "timestamp" in log_data
    
    @pytest.mark.unit
    def test_colored_formatter(self):
        """Test: Colored-Formatter"""
        formatter = ColoredFormatter(
            fmt="%(levelname)s - %(message)s",
            use_colors=True
        )
        
        # INFO-Record
        info_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Info message", args=(), exc_info=None
        )
        
        # ERROR-Record
        error_record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="Error message", args=(), exc_info=None
        )
        
        info_formatted = formatter.format(info_record)
        error_formatted = formatter.format(error_record)
        
        # Farb-Codes sollten enthalten sein (ANSI-Escape-Sequenzen)
        assert "\033[" in info_formatted or "INFO" in info_formatted
        assert "\033[" in error_formatted or "ERROR" in error_formatted
    
    @pytest.mark.unit
    def test_log_formatter_with_extra_fields(self):
        """Test: Log-Formatter mit Extra-Feldern"""
        formatter = LogFormatter(
            include_extra=True,
            extra_fields=["user_id", "request_id"]
        )
        
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test with extra", args=(), exc_info=None
        )
        
        # Extra-Felder hinzufügen
        record.user_id = "user_123"
        record.request_id = "req_456"
        record.other_field = "should_be_ignored"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["user_id"] == "user_123"
        assert log_data["request_id"] == "req_456"
        assert "other_field" not in log_data


class TestLogFilters:
    """Tests für Log-Filter"""
    
    @pytest.mark.unit
    def test_sensitive_data_filter(self):
        """Test: Sensitive-Data-Filter"""
        filter_obj = SensitiveDataFilter(
            sensitive_patterns=[
                r'password["\']?\s*[:=]\s*["\']?([^"\',\s]+)',
                r'token["\']?\s*[:=]\s*["\']?([^"\',\s]+)',
                r'api_key["\']?\s*[:=]\s*["\']?([^"\',\s]+)'
            ]
        )
        
        # Record mit sensiblen Daten
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg='User login: {"username": "john", "password": "secret123", "token": "abc123xyz"}',
            args=(), exc_info=None
        )
        
        # Filter anwenden
        filtered = filter_obj.filter(record)
        
        assert filtered == True  # Record sollte durchgelassen werden
        assert "secret123" not in record.msg  # Passwort sollte maskiert sein
        assert "abc123xyz" not in record.msg  # Token sollte maskiert sein
        assert "john" in record.msg  # Username sollte erhalten bleiben
        assert "***" in record.msg  # Maskierung sollte vorhanden sein
    
    @pytest.mark.unit
    def test_rate_limit_filter(self):
        """Test: Rate-Limit-Filter"""
        filter_obj = RateLimitFilter(
            max_messages=3,
            time_window=1.0  # 1 Sekunde
        )
        
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Repeated message", args=(), exc_info=None
        )
        
        # Erste 3 Nachrichten sollten durchgelassen werden
        assert filter_obj.filter(record) == True
        assert filter_obj.filter(record) == True
        assert filter_obj.filter(record) == True
        
        # 4. Nachricht sollte blockiert werden
        assert filter_obj.filter(record) == False
        
        # Nach Zeitfenster sollten wieder Nachrichten durchgelassen werden
        import time
        time.sleep(1.1)
        assert filter_obj.filter(record) == True
    
    @pytest.mark.unit
    def test_log_filter_by_level(self):
        """Test: Filter nach Log-Level"""
        filter_obj = LogFilter(
            min_level=logging.WARNING,
            max_level=logging.ERROR
        )
        
        debug_record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="", lineno=0,
            msg="Debug message", args=(), exc_info=None
        )
        
        warning_record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="Warning message", args=(), exc_info=None
        )
        
        critical_record = logging.LogRecord(
            name="test", level=logging.CRITICAL, pathname="", lineno=0,
            msg="Critical message", args=(), exc_info=None
        )
        
        assert filter_obj.filter(debug_record) == False  # Zu niedrig
        assert filter_obj.filter(warning_record) == True  # Im Bereich
        assert filter_obj.filter(critical_record) == False  # Zu hoch


class TestLogHandlers:
    """Tests für spezielle Log-Handler"""
    
    @pytest.mark.unit
    def test_log_rotation_handler(self):
        """Test: Log-Rotation-Handler"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "rotation_test.log")
            
            handler = LogRotationHandler(
                filename=log_file,
                max_bytes=1024,  # 1KB
                backup_count=3
            )
            
            logger = logging.getLogger("rotation_test")
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            # Viele Nachrichten schreiben um Rotation auszulösen
            for i in range(100):
                logger.info(f"Test message {i} with additional content to reach size limit")
            
            # Haupt-Log-Datei sollte existieren
            assert os.path.exists(log_file)
            
            # Backup-Dateien könnten existieren
            backup_files = [f for f in os.listdir(temp_dir) if f.startswith("rotation_test.log.")]
            # Je nach Nachrichtengröße könnten Backups erstellt worden sein
    
    @pytest.mark.unit
    def test_database_log_handler(self):
        """Test: Database-Log-Handler"""
        # Mock-Database-Session
        mock_session = MagicMock()
        
        handler = DatabaseLogHandler(
            session=mock_session,
            table_name="logs"
        )
        
        record = logging.LogRecord(
            name="db_test", level=logging.ERROR, pathname="/test.py", lineno=42,
            msg="Database test message", args=(), exc_info=None
        )
        
        handler.emit(record)
        
        # Session sollte aufgerufen worden sein
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.unit
    def test_websocket_log_handler(self):
        """Test: WebSocket-Log-Handler"""
        # Mock-WebSocket-Manager
        mock_ws_manager = MagicMock()
        
        handler = WebSocketLogHandler(
            websocket_manager=mock_ws_manager,
            channel="logs"
        )
        
        record = logging.LogRecord(
            name="ws_test", level=logging.INFO, pathname="", lineno=0,
            msg="WebSocket test message", args=(), exc_info=None
        )
        
        handler.emit(record)
        
        # WebSocket-Manager sollte aufgerufen worden sein
        mock_ws_manager.broadcast.assert_called_once()
        call_args = mock_ws_manager.broadcast.call_args
        assert call_args[1]["channel"] == "logs"


class TestLoggingIntegration:
    """Integrationstests für Logging"""
    
    @pytest.mark.integration
    def test_complete_logging_setup(self):
        """Test: Vollständiges Logging-Setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "integration_test.log")
            
            config = LoggingConfig(
                level="DEBUG",
                file_path=log_file,
                console_enabled=True,
                json_format=True,
                rotation_enabled=True,
                max_file_size="1MB",
                backup_count=3
            )
            
            setup_logging(config)
            
            # Verschiedene Logger testen
            structured_logger = StructuredLogger("integration_structured")
            performance_logger = PerformanceLogger("integration_performance")
            error_logger = ErrorLogger("integration_error")
            audit_logger = AuditLogger("integration_audit")
            security_logger = SecurityLogger("integration_security")
            
            # Verschiedene Log-Nachrichten
            structured_logger.log_event("test_event", action="integration_test")
            
            with performance_logger.time_operation("test_operation"):
                import time
                time.sleep(0.01)
            
            try:
                raise ValueError("Integration test error")
            except Exception as e:
                error_logger.log_exception(e, context={"test": "integration"})
            
            audit_logger.log_user_action(
                user_id="test_user",
                action="integration_test",
                resource="test_resource"
            )
            
            security_logger.log_authentication_attempt(
                user_id="test_user",
                success=True,
                ip_address="127.0.0.1"
            )
            
            # Log-Datei sollte alle Nachrichten enthalten
            assert os.path.exists(log_file)
            
            with open(log_file, 'r') as f:
                content = f.read()
                assert "test_event" in content
                assert "test_operation" in content
                assert "Integration test error" in content
                assert "integration_test" in content
                assert "authentication_attempt" in content
    
    @pytest.mark.performance
    def test_logging_performance(self):
        """Test: Logging-Performance"""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "performance_test.log")
            
            config = LoggingConfig(
                level="INFO",
                file_path=log_file,
                console_enabled=False  # Nur File-Logging für Performance-Test
            )
            
            setup_logging(config)
            logger = get_logger("performance_test")
            
            start_time = time.time()
            
            # Viele Log-Nachrichten
            for i in range(10000):
                logger.info(f"Performance test message {i}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 10.000 Nachrichten sollten in unter 5 Sekunden geloggt werden
            assert duration < 5.0
            
            # Log-Datei sollte alle Nachrichten enthalten
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 10000