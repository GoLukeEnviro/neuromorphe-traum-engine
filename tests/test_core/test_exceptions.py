"""Tests für Core-Exceptions"""

import pytest
from unittest.mock import MagicMock
import traceback
from datetime import datetime

from src.core.exceptions import (
    NeuromorpheError, ConfigurationError, AudioProcessingError,
    CLAPModelError, DatabaseError, RenderingError, APIError,
    ValidationError, FileNotFoundError, InsufficientResourcesError,
    TimeoutError, AuthenticationError, AuthorizationError,
    RateLimitError, ExternalServiceError
)


class TestBaseException:
    """Tests für die Basis-Exception-Klasse"""
    
    @pytest.mark.unit
    def test_neuromorphe_error_creation(self):
        """Test: NeuromorpheError-Instanz erstellen"""
        error = NeuromorpheError(
            message="Test error message",
            error_code="TEST_001",
            details={"key": "value"},
            original_exception=ValueError("Original error")
        )
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code == "TEST_001"
        assert error.details == {"key": "value"}
        assert isinstance(error.original_exception, ValueError)
        assert error.timestamp is not None
    
    @pytest.mark.unit
    def test_neuromorphe_error_defaults(self):
        """Test: NeuromorpheError-Standardwerte"""
        error = NeuromorpheError("Simple error")
        
        assert error.message == "Simple error"
        assert error.error_code is None
        assert error.details == {}
        assert error.original_exception is None
        assert isinstance(error.timestamp, datetime)
    
    @pytest.mark.unit
    def test_neuromorphe_error_serialization(self):
        """Test: NeuromorpheError-Serialisierung"""
        original_error = ValueError("Original error")
        error = NeuromorpheError(
            message="Serialization test",
            error_code="SER_001",
            details={"component": "test", "value": 42},
            original_exception=original_error
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["message"] == "Serialization test"
        assert error_dict["error_code"] == "SER_001"
        assert error_dict["details"]["component"] == "test"
        assert error_dict["details"]["value"] == 42
        assert "timestamp" in error_dict
        assert "traceback" in error_dict
        assert "original_exception" in error_dict
    
    @pytest.mark.unit
    def test_neuromorphe_error_chaining(self):
        """Test: Exception-Chaining"""
        original_error = ValueError("Original error")
        
        try:
            raise original_error
        except ValueError as e:
            neuromorphe_error = NeuromorpheError(
                "Wrapped error",
                original_exception=e
            )
        
        assert neuromorphe_error.original_exception is original_error
        assert "ValueError: Original error" in str(neuromorphe_error.original_exception)


class TestConfigurationError:
    """Tests für ConfigurationError"""
    
    @pytest.mark.unit
    def test_configuration_error_creation(self):
        """Test: ConfigurationError erstellen"""
        error = ConfigurationError(
            "Invalid configuration",
            config_key="database.url",
            config_value="invalid_url"
        )
        
        assert error.message == "Invalid configuration"
        assert error.config_key == "database.url"
        assert error.config_value == "invalid_url"
        assert "CONF_" in error.error_code
    
    @pytest.mark.unit
    def test_configuration_error_validation(self):
        """Test: ConfigurationError bei Validierungsfehlern"""
        error = ConfigurationError.validation_error(
            key="api.port",
            value=70000,
            reason="Port must be between 1 and 65535"
        )
        
        assert "validation" in error.message.lower()
        assert error.config_key == "api.port"
        assert error.config_value == 70000
        assert "Port must be between 1 and 65535" in error.message
    
    @pytest.mark.unit
    def test_configuration_error_missing_key(self):
        """Test: ConfigurationError bei fehlendem Schlüssel"""
        error = ConfigurationError.missing_key("required.setting")
        
        assert "missing" in error.message.lower()
        assert error.config_key == "required.setting"
        assert error.config_value is None


class TestAudioProcessingError:
    """Tests für AudioProcessingError"""
    
    @pytest.mark.unit
    def test_audio_processing_error_creation(self):
        """Test: AudioProcessingError erstellen"""
        error = AudioProcessingError(
            "Failed to process audio",
            file_path="/path/to/audio.wav",
            operation="normalize",
            audio_info={
                "sample_rate": 44100,
                "channels": 2,
                "duration": 120.5
            }
        )
        
        assert error.message == "Failed to process audio"
        assert error.file_path == "/path/to/audio.wav"
        assert error.operation == "normalize"
        assert error.audio_info["sample_rate"] == 44100
        assert "AUDIO_" in error.error_code
    
    @pytest.mark.unit
    def test_audio_processing_error_invalid_format(self):
        """Test: AudioProcessingError bei ungültigem Format"""
        error = AudioProcessingError.invalid_format(
            file_path="/path/to/file.xyz",
            supported_formats=["wav", "mp3", "flac"]
        )
        
        assert "format" in error.message.lower()
        assert error.file_path == "/path/to/file.xyz"
        assert "wav" in str(error.details)
    
    @pytest.mark.unit
    def test_audio_processing_error_corrupted_file(self):
        """Test: AudioProcessingError bei korrupter Datei"""
        error = AudioProcessingError.corrupted_file(
            file_path="/path/to/corrupted.wav",
            reason="Invalid header"
        )
        
        assert "corrupted" in error.message.lower()
        assert error.file_path == "/path/to/corrupted.wav"
        assert "Invalid header" in error.message


class TestCLAPModelError:
    """Tests für CLAPModelError"""
    
    @pytest.mark.unit
    def test_clap_model_error_creation(self):
        """Test: CLAPModelError erstellen"""
        error = CLAPModelError(
            "Model loading failed",
            model_name="laion/clap-htsat-unfused",
            operation="load",
            model_info={
                "device": "cuda",
                "memory_required": "2GB"
            }
        )
        
        assert error.message == "Model loading failed"
        assert error.model_name == "laion/clap-htsat-unfused"
        assert error.operation == "load"
        assert error.model_info["device"] == "cuda"
        assert "CLAP_" in error.error_code
    
    @pytest.mark.unit
    def test_clap_model_error_model_not_found(self):
        """Test: CLAPModelError bei nicht gefundenem Modell"""
        error = CLAPModelError.model_not_found(
            model_name="nonexistent/model",
            available_models=["model1", "model2"]
        )
        
        assert "not found" in error.message.lower()
        assert error.model_name == "nonexistent/model"
        assert "model1" in str(error.details)
    
    @pytest.mark.unit
    def test_clap_model_error_insufficient_memory(self):
        """Test: CLAPModelError bei unzureichendem Speicher"""
        error = CLAPModelError.insufficient_memory(
            model_name="large/model",
            required_memory="8GB",
            available_memory="4GB"
        )
        
        assert "memory" in error.message.lower()
        assert error.model_name == "large/model"
        assert "8GB" in error.message
        assert "4GB" in error.message


class TestDatabaseError:
    """Tests für DatabaseError"""
    
    @pytest.mark.unit
    def test_database_error_creation(self):
        """Test: DatabaseError erstellen"""
        error = DatabaseError(
            "Connection failed",
            operation="connect",
            table="stems",
            query="SELECT * FROM stems",
            connection_info={
                "url": "postgresql://localhost/test",
                "pool_size": 5
            }
        )
        
        assert error.message == "Connection failed"
        assert error.operation == "connect"
        assert error.table == "stems"
        assert error.query == "SELECT * FROM stems"
        assert error.connection_info["url"] == "postgresql://localhost/test"
        assert "DB_" in error.error_code
    
    @pytest.mark.unit
    def test_database_error_connection_failed(self):
        """Test: DatabaseError bei Verbindungsfehler"""
        error = DatabaseError.connection_failed(
            database_url="postgresql://localhost/nonexistent",
            reason="Database does not exist"
        )
        
        assert "connection" in error.message.lower()
        assert "postgresql://localhost/nonexistent" in error.message
        assert "Database does not exist" in error.message
    
    @pytest.mark.unit
    def test_database_error_query_failed(self):
        """Test: DatabaseError bei Query-Fehler"""
        error = DatabaseError.query_failed(
            query="SELECT * FROM nonexistent_table",
            table="nonexistent_table",
            reason="Table does not exist"
        )
        
        assert "query" in error.message.lower()
        assert error.query == "SELECT * FROM nonexistent_table"
        assert error.table == "nonexistent_table"
        assert "Table does not exist" in error.message


class TestRenderingError:
    """Tests für RenderingError"""
    
    @pytest.mark.unit
    def test_rendering_error_creation(self):
        """Test: RenderingError erstellen"""
        error = RenderingError(
            "Rendering failed",
            job_id="job_123",
            arrangement_id="arr_456",
            stage="mixing",
            progress=0.75,
            render_info={
                "format": "wav",
                "quality": "high",
                "duration": 180
            }
        )
        
        assert error.message == "Rendering failed"
        assert error.job_id == "job_123"
        assert error.arrangement_id == "arr_456"
        assert error.stage == "mixing"
        assert error.progress == 0.75
        assert error.render_info["format"] == "wav"
        assert "RENDER_" in error.error_code
    
    @pytest.mark.unit
    def test_rendering_error_stem_not_found(self):
        """Test: RenderingError bei nicht gefundenem Stem"""
        error = RenderingError.stem_not_found(
            stem_id="stem_789",
            job_id="job_123"
        )
        
        assert "stem" in error.message.lower()
        assert "not found" in error.message.lower()
        assert error.job_id == "job_123"
        assert "stem_789" in error.message
    
    @pytest.mark.unit
    def test_rendering_error_insufficient_disk_space(self):
        """Test: RenderingError bei unzureichendem Speicherplatz"""
        error = RenderingError.insufficient_disk_space(
            required_space="500MB",
            available_space="100MB",
            job_id="job_123"
        )
        
        assert "disk space" in error.message.lower()
        assert error.job_id == "job_123"
        assert "500MB" in error.message
        assert "100MB" in error.message


class TestAPIError:
    """Tests für APIError"""
    
    @pytest.mark.unit
    def test_api_error_creation(self):
        """Test: APIError erstellen"""
        error = APIError(
            "Bad request",
            status_code=400,
            endpoint="/api/arrangements",
            method="POST",
            request_data={"prompt": "test"},
            headers={"Content-Type": "application/json"}
        )
        
        assert error.message == "Bad request"
        assert error.status_code == 400
        assert error.endpoint == "/api/arrangements"
        assert error.method == "POST"
        assert error.request_data["prompt"] == "test"
        assert "API_" in error.error_code
    
    @pytest.mark.unit
    def test_api_error_bad_request(self):
        """Test: APIError für Bad Request"""
        error = APIError.bad_request(
            message="Invalid prompt",
            endpoint="/api/analyze",
            details={"field": "prompt", "issue": "too_short"}
        )
        
        assert error.status_code == 400
        assert error.endpoint == "/api/analyze"
        assert "Invalid prompt" in error.message
        assert error.details["field"] == "prompt"
    
    @pytest.mark.unit
    def test_api_error_not_found(self):
        """Test: APIError für Not Found"""
        error = APIError.not_found(
            resource="arrangement",
            resource_id="arr_123",
            endpoint="/api/arrangements/arr_123"
        )
        
        assert error.status_code == 404
        assert "arrangement" in error.message.lower()
        assert "arr_123" in error.message
        assert error.endpoint == "/api/arrangements/arr_123"
    
    @pytest.mark.unit
    def test_api_error_internal_server_error(self):
        """Test: APIError für Internal Server Error"""
        original_error = ValueError("Database connection failed")
        
        error = APIError.internal_server_error(
            endpoint="/api/stems",
            original_exception=original_error
        )
        
        assert error.status_code == 500
        assert "internal server error" in error.message.lower()
        assert error.endpoint == "/api/stems"
        assert error.original_exception is original_error


class TestValidationError:
    """Tests für ValidationError"""
    
    @pytest.mark.unit
    def test_validation_error_creation(self):
        """Test: ValidationError erstellen"""
        error = ValidationError(
            "Validation failed",
            field="duration",
            value=-10,
            constraint="must be positive",
            validation_errors=[
                {"field": "duration", "message": "must be positive"}
            ]
        )
        
        assert error.message == "Validation failed"
        assert error.field == "duration"
        assert error.value == -10
        assert error.constraint == "must be positive"
        assert len(error.validation_errors) == 1
        assert "VALID_" in error.error_code
    
    @pytest.mark.unit
    def test_validation_error_required_field(self):
        """Test: ValidationError für erforderliches Feld"""
        error = ValidationError.required_field(
            field="prompt",
            context="arrangement creation"
        )
        
        assert "required" in error.message.lower()
        assert error.field == "prompt"
        assert "arrangement creation" in error.message
    
    @pytest.mark.unit
    def test_validation_error_invalid_type(self):
        """Test: ValidationError für ungültigen Typ"""
        error = ValidationError.invalid_type(
            field="duration",
            expected_type="number",
            actual_type="string",
            value="not_a_number"
        )
        
        assert "type" in error.message.lower()
        assert error.field == "duration"
        assert "number" in error.message
        assert "string" in error.message
        assert error.value == "not_a_number"


class TestSpecializedErrors:
    """Tests für spezialisierte Error-Klassen"""
    
    @pytest.mark.unit
    def test_file_not_found_error(self):
        """Test: FileNotFoundError"""
        error = FileNotFoundError(
            file_path="/path/to/missing.wav",
            operation="load_audio"
        )
        
        assert "not found" in error.message.lower()
        assert error.file_path == "/path/to/missing.wav"
        assert error.operation == "load_audio"
    
    @pytest.mark.unit
    def test_insufficient_resources_error(self):
        """Test: InsufficientResourcesError"""
        error = InsufficientResourcesError(
            resource_type="memory",
            required="8GB",
            available="4GB",
            operation="model_loading"
        )
        
        assert "insufficient" in error.message.lower()
        assert error.resource_type == "memory"
        assert error.required == "8GB"
        assert error.available == "4GB"
        assert error.operation == "model_loading"
    
    @pytest.mark.unit
    def test_timeout_error(self):
        """Test: TimeoutError"""
        error = TimeoutError(
            operation="render_job",
            timeout_seconds=300,
            elapsed_seconds=450
        )
        
        assert "timeout" in error.message.lower()
        assert error.operation == "render_job"
        assert error.timeout_seconds == 300
        assert error.elapsed_seconds == 450
    
    @pytest.mark.unit
    def test_authentication_error(self):
        """Test: AuthenticationError"""
        error = AuthenticationError(
            reason="invalid_token",
            user_id="user_123",
            endpoint="/api/protected"
        )
        
        assert "authentication" in error.message.lower()
        assert error.reason == "invalid_token"
        assert error.user_id == "user_123"
        assert error.endpoint == "/api/protected"
    
    @pytest.mark.unit
    def test_authorization_error(self):
        """Test: AuthorizationError"""
        error = AuthorizationError(
            required_permission="admin",
            user_permissions=["user", "read"],
            resource="system_settings",
            user_id="user_123"
        )
        
        assert "authorization" in error.message.lower()
        assert error.required_permission == "admin"
        assert "user" in error.user_permissions
        assert error.resource == "system_settings"
        assert error.user_id == "user_123"
    
    @pytest.mark.unit
    def test_rate_limit_error(self):
        """Test: RateLimitError"""
        error = RateLimitError(
            limit="100/minute",
            current_usage=150,
            reset_time=datetime.now(),
            client_id="client_123"
        )
        
        assert "rate limit" in error.message.lower()
        assert error.limit == "100/minute"
        assert error.current_usage == 150
        assert error.client_id == "client_123"
    
    @pytest.mark.unit
    def test_external_service_error(self):
        """Test: ExternalServiceError"""
        error = ExternalServiceError(
            service_name="huggingface",
            operation="model_download",
            status_code=503,
            response_data={"error": "Service unavailable"}
        )
        
        assert "external service" in error.message.lower()
        assert error.service_name == "huggingface"
        assert error.operation == "model_download"
        assert error.status_code == 503
        assert error.response_data["error"] == "Service unavailable"


class TestErrorHandling:
    """Tests für Error-Handling-Funktionalität"""
    
    @pytest.mark.unit
    def test_error_context_manager(self):
        """Test: Error-Context-Manager"""
        from src.core.exceptions import error_context
        
        with pytest.raises(AudioProcessingError) as exc_info:
            with error_context("audio_processing", file_path="/test.wav"):
                raise ValueError("Original error")
        
        error = exc_info.value
        assert isinstance(error, AudioProcessingError)
        assert error.file_path == "/test.wav"
        assert isinstance(error.original_exception, ValueError)
    
    @pytest.mark.unit
    def test_error_logging(self):
        """Test: Error-Logging"""
        import logging
        from unittest.mock import patch
        
        with patch('logging.getLogger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            error = NeuromorpheError(
                "Test error for logging",
                error_code="LOG_001"
            )
            
            error.log_error()
            
            # Logger sollte aufgerufen worden sein
            mock_log.error.assert_called_once()
            call_args = mock_log.error.call_args[0]
            assert "Test error for logging" in call_args[0]
    
    @pytest.mark.unit
    def test_error_recovery_suggestions(self):
        """Test: Error-Recovery-Vorschläge"""
        error = AudioProcessingError.invalid_format(
            file_path="/test.xyz",
            supported_formats=["wav", "mp3"]
        )
        
        suggestions = error.get_recovery_suggestions()
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("convert" in suggestion.lower() for suggestion in suggestions)
    
    @pytest.mark.unit
    def test_error_aggregation(self):
        """Test: Error-Aggregation"""
        from src.core.exceptions import ErrorAggregator
        
        aggregator = ErrorAggregator()
        
        # Mehrere Errors hinzufügen
        aggregator.add_error(ValidationError("Error 1", field="field1"))
        aggregator.add_error(ValidationError("Error 2", field="field2"))
        aggregator.add_error(AudioProcessingError("Error 3", file_path="/test.wav"))
        
        assert aggregator.has_errors() == True
        assert aggregator.error_count() == 3
        
        # Errors nach Typ gruppieren
        grouped_errors = aggregator.group_by_type()
        assert ValidationError in grouped_errors
        assert AudioProcessingError in grouped_errors
        assert len(grouped_errors[ValidationError]) == 2
        assert len(grouped_errors[AudioProcessingError]) == 1
        
        # Alle Errors als eine Exception werfen
        with pytest.raises(NeuromorpheError) as exc_info:
            aggregator.raise_if_errors()
        
        combined_error = exc_info.value
        assert "multiple errors" in combined_error.message.lower()
        assert len(combined_error.details["errors"]) == 3


class TestErrorSerialization:
    """Tests für Error-Serialisierung"""
    
    @pytest.mark.unit
    def test_error_json_serialization(self):
        """Test: Error zu JSON serialisieren"""
        import json
        
        error = AudioProcessingError(
            "JSON serialization test",
            file_path="/test.wav",
            operation="normalize",
            audio_info={"sample_rate": 44100}
        )
        
        error_dict = error.to_dict()
        json_str = json.dumps(error_dict, default=str)
        
        # JSON sollte parsbar sein
        parsed_data = json.loads(json_str)
        
        assert parsed_data["message"] == "JSON serialization test"
        assert parsed_data["file_path"] == "/test.wav"
        assert parsed_data["operation"] == "normalize"
        assert parsed_data["audio_info"]["sample_rate"] == 44100
    
    @pytest.mark.unit
    def test_error_from_dict(self):
        """Test: Error aus Dictionary erstellen"""
        error_dict = {
            "message": "Deserialization test",
            "error_code": "DESER_001",
            "details": {"key": "value"},
            "timestamp": "2024-01-01T12:00:00"
        }
        
        error = NeuromorpheError.from_dict(error_dict)
        
        assert error.message == "Deserialization test"
        assert error.error_code == "DESER_001"
        assert error.details["key"] == "value"
    
    @pytest.mark.unit
    def test_error_http_response(self):
        """Test: Error zu HTTP-Response konvertieren"""
        error = APIError(
            "Test API error",
            status_code=400,
            endpoint="/api/test"
        )
        
        response_data = error.to_http_response()
        
        assert response_data["status_code"] == 400
        assert response_data["error"]["message"] == "Test API error"
        assert response_data["error"]["endpoint"] == "/api/test"
        assert "timestamp" in response_data["error"]
        assert "error_code" in response_data["error"]


class TestErrorPerformance:
    """Tests für Error-Performance"""
    
    @pytest.mark.performance
    def test_error_creation_performance(self):
        """Test: Performance der Error-Erstellung"""
        import time
        
        start_time = time.time()
        
        # Viele Errors erstellen
        errors = []
        for i in range(1000):
            error = NeuromorpheError(
                f"Error {i}",
                error_code=f"ERR_{i:03d}",
                details={"index": i, "data": f"value_{i}"}
            )
            errors.append(error)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        assert len(errors) == 1000
        assert creation_time < 1.0  # Sollte unter 1 Sekunde dauern
    
    @pytest.mark.performance
    def test_error_serialization_performance(self):
        """Test: Performance der Error-Serialisierung"""
        import time
        
        error = AudioProcessingError(
            "Performance test error",
            file_path="/test/performance.wav",
            operation="complex_processing",
            audio_info={
                "sample_rate": 44100,
                "channels": 2,
                "duration": 300.0,
                "features": [float(i) for i in range(100)]
            }
        )
        
        start_time = time.time()
        
        # Mehrfache Serialisierung
        for _ in range(1000):
            error_dict = error.to_dict()
        
        end_time = time.time()
        serialization_time = end_time - start_time
        
        assert "message" in error_dict
        assert serialization_time < 1.0  # Sollte unter 1 Sekunde dauern