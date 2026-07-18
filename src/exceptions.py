"""Application exceptions and their serialization helpers."""

from __future__ import annotations

import logging
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Type


class NeuromorpheError(Exception):
    """Base class for errors that can safely cross application boundaries."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[BaseException] = None,
        code: Optional[Any] = None,
        **attributes: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code if error_code is not None else code
        self.details = dict(details or {})
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        for name, value in attributes.items():
            setattr(self, name, value)

    @property
    def code(self) -> Optional[Any]:
        """Backward-compatible alias for the former numeric error code."""
        return self.error_code

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "original_exception": (
                str(self.original_exception) if self.original_exception else None
            ),
            "traceback": self._format_traceback(),
        }
        excluded = {
            "message", "error_code", "details", "original_exception", "timestamp"
        }
        data.update(
            (name, value)
            for name, value in self.__dict__.items()
            if name not in excluded and not name.startswith("_")
        )
        return data

    def _format_traceback(self) -> Optional[str]:
        if self.original_exception is None:
            return None
        return "".join(
            traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__,
            )
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuromorpheError":
        error = cls(
            data["message"],
            error_code=data.get("error_code"),
            details=data.get("details"),
        )
        if data.get("timestamp"):
            error.timestamp = datetime.fromisoformat(data["timestamp"])
        return error

    def log_error(self) -> None:
        logging.getLogger(__name__).error(
            f"{self.message} (error_code={self.error_code})"
        )

    def get_recovery_suggestions(self) -> List[str]:
        return []


class ConfigurationError(NeuromorpheError):
    def __init__(self, message: str = "Configuration error", config_key=None,
                 config_value=None, **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "CONF_ERROR"),
                         **kwargs)
        self.config_key = config_key
        self.config_value = config_value

    @classmethod
    def validation_error(cls, key, value, reason):
        return cls(f"Configuration validation failed for '{key}': {reason}", key, value)

    @classmethod
    def missing_key(cls, key):
        return cls(f"Missing required configuration key: {key}", key, None)


class AudioProcessingError(NeuromorpheError):
    def __init__(self, message: str = "Audio processing failed", file_path=None,
                 operation=None, audio_info=None, **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "AUDIO_ERROR"),
                         **kwargs)
        self.file_path = file_path
        self.operation = operation
        self.audio_info = dict(audio_info or {})

    @classmethod
    def invalid_format(cls, file_path, supported_formats):
        return cls(
            f"Unsupported audio format for {file_path}",
            file_path=file_path,
            operation="validate_format",
            details={"supported_formats": list(supported_formats)},
            error_code="AUDIO_INVALID_FORMAT",
        )

    @classmethod
    def corrupted_file(cls, file_path, reason):
        return cls(
            f"Corrupted audio file {file_path}: {reason}",
            file_path=file_path,
            operation="read",
            error_code="AUDIO_CORRUPTED_FILE",
        )

    def get_recovery_suggestions(self) -> List[str]:
        if self.error_code == "AUDIO_INVALID_FORMAT":
            return ["Convert the file to one of the supported audio formats."]
        return ["Verify the audio file and retry the operation."]


class CLAPModelError(NeuromorpheError):
    def __init__(self, message: str = "CLAP model error", model_name=None,
                 operation=None, model_info=None, **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "CLAP_ERROR"),
                         **kwargs)
        self.model_name = model_name
        self.operation = operation
        self.model_info = dict(model_info or {})

    @classmethod
    def model_not_found(cls, model_name, available_models):
        return cls(f"CLAP model not found: {model_name}", model_name=model_name,
                   operation="load", details={"available_models": list(available_models)})

    @classmethod
    def insufficient_memory(cls, model_name, required_memory, available_memory):
        return cls(
            f"Insufficient memory for {model_name}: requires {required_memory}, "
            f"only {available_memory} available",
            model_name=model_name,
            operation="load",
        )


class DatabaseError(NeuromorpheError):
    def __init__(self, message: str = "Database operation failed", operation=None,
                 table=None, query=None, connection_info=None, **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "DB_ERROR"),
                         **kwargs)
        self.operation = operation
        self.table = table
        self.query = query
        self.connection_info = dict(connection_info or {})

    @classmethod
    def connection_failed(cls, database_url, reason):
        return cls(f"Database connection failed for {database_url}: {reason}",
                   operation="connect", connection_info={"url": database_url})

    @classmethod
    def query_failed(cls, query, table, reason):
        return cls(f"Database query failed: {reason}", operation="query",
                   query=query, table=table)


class RenderingError(NeuromorpheError):
    def __init__(self, message: str = "Audio rendering failed", job_id=None,
                 arrangement_id=None, stage=None, progress=None, render_info=None,
                 **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "RENDER_ERROR"),
                         **kwargs)
        self.job_id = job_id
        self.arrangement_id = arrangement_id
        self.stage = stage
        self.progress = progress
        self.render_info = dict(render_info or {})

    @classmethod
    def stem_not_found(cls, stem_id, job_id=None):
        return cls(f"Stem not found: {stem_id}", job_id=job_id, stage="load_stems")

    @classmethod
    def insufficient_disk_space(cls, required_space, available_space, job_id=None):
        return cls(
            f"Insufficient disk space: requires {required_space}, "
            f"only {available_space} available",
            job_id=job_id,
            stage="write_output",
        )


class APIError(NeuromorpheError):
    def __init__(self, message: str = "API error", status_code: int = 500,
                 endpoint=None, method=None, request_data=None, headers=None,
                 **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "API_ERROR"),
                         **kwargs)
        self.status_code = status_code
        self.endpoint = endpoint
        self.method = method
        self.request_data = dict(request_data or {})
        self.headers = dict(headers or {})

    @classmethod
    def bad_request(cls, message, endpoint=None, details=None):
        return cls(message, status_code=400, endpoint=endpoint, details=details,
                   error_code="API_BAD_REQUEST")

    @classmethod
    def not_found(cls, resource, resource_id, endpoint=None):
        return cls(f"{resource} not found: {resource_id}", status_code=404,
                   endpoint=endpoint, error_code="API_NOT_FOUND")

    @classmethod
    def internal_server_error(cls, endpoint=None, original_exception=None):
        return cls("Internal server error", status_code=500, endpoint=endpoint,
                   original_exception=original_exception,
                   error_code="API_INTERNAL_ERROR")

    def to_http_response(self) -> Dict[str, Any]:
        return {"status_code": self.status_code, "error": self.to_dict()}


class ValidationError(NeuromorpheError):
    def __init__(self, message: str = "Validation failed", field=None, value=None,
                 constraint=None, validation_errors=None, **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "VALID_ERROR"),
                         **kwargs)
        self.field = field
        self.value = value
        self.constraint = constraint
        self.validation_errors = list(validation_errors or [])

    @classmethod
    def required_field(cls, field, context=None):
        suffix = f" for {context}" if context else ""
        return cls(f"Required field '{field}' is missing{suffix}", field=field,
                   constraint="required")

    @classmethod
    def invalid_type(cls, field, expected_type, actual_type, value=None):
        return cls(
            f"Invalid type for '{field}': expected {expected_type}, got {actual_type}",
            field=field,
            value=value,
            constraint=f"type:{expected_type}",
        )


class FileNotFoundError(NeuromorpheError):
    def __init__(self, message: Optional[str] = None, file_path=None, operation=None,
                 **kwargs: Any) -> None:
        message = message or f"File not found: {file_path}"
        super().__init__(message, error_code=kwargs.pop("error_code", "FILE_NOT_FOUND"),
                         **kwargs)
        self.file_path = file_path
        self.operation = operation


class InsufficientResourcesError(NeuromorpheError):
    def __init__(self, message: Optional[str] = None, resource_type=None, required=None,
                 available=None, operation=None, **kwargs: Any) -> None:
        message = message or f"Insufficient {resource_type} resources"
        super().__init__(message, error_code=kwargs.pop("error_code", "RESOURCE_INSUFFICIENT"),
                         **kwargs)
        self.resource_type = resource_type
        self.required = required
        self.available = available
        self.operation = operation


class TimeoutError(NeuromorpheError):
    def __init__(self, message: Optional[str] = None, operation=None,
                 timeout_seconds=None, elapsed_seconds=None, **kwargs: Any) -> None:
        message = message or f"Operation timeout: {operation}"
        super().__init__(message, error_code=kwargs.pop("error_code", "TIMEOUT"), **kwargs)
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class AuthenticationError(NeuromorpheError):
    def __init__(self, message: Optional[str] = None, reason=None, user_id=None,
                 endpoint=None, **kwargs: Any) -> None:
        super().__init__(message or "Authentication failed",
                         error_code=kwargs.pop("error_code", "AUTHENTICATION"), **kwargs)
        self.reason = reason
        self.user_id = user_id
        self.endpoint = endpoint


class AuthorizationError(NeuromorpheError):
    def __init__(self, message: Optional[str] = None, required_permission=None,
                 user_permissions=None, resource=None, user_id=None, **kwargs: Any) -> None:
        super().__init__(message or "Authorization failed",
                         error_code=kwargs.pop("error_code", "AUTHORIZATION"), **kwargs)
        self.required_permission = required_permission
        self.user_permissions = list(user_permissions or [])
        self.resource = resource
        self.user_id = user_id


class RateLimitError(NeuromorpheError):
    def __init__(self, message: Optional[str] = None, limit=None, current_usage=None,
                 reset_time=None, client_id=None, **kwargs: Any) -> None:
        super().__init__(message or "Rate limit exceeded",
                         error_code=kwargs.pop("error_code", "RATE_LIMIT"), **kwargs)
        self.limit = limit
        self.current_usage = current_usage
        self.reset_time = reset_time
        self.client_id = client_id


class ExternalServiceError(NeuromorpheError):
    def __init__(self, message: Optional[str] = None, service_name=None, operation=None,
                 status_code=None, response_data=None, **kwargs: Any) -> None:
        super().__init__(message or "External service error",
                         error_code=kwargs.pop("error_code", "EXTERNAL_SERVICE"), **kwargs)
        self.service_name = service_name
        self.operation = operation
        self.status_code = status_code
        self.response_data = dict(response_data or {})


class NotFoundError(NeuromorpheError):
    def __init__(self, message: str = "Resource not found", **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "NOT_FOUND"), **kwargs)


class ServiceInitializationError(NeuromorpheError):
    def __init__(self, message: str = "Service initialization failed", **kwargs: Any) -> None:
        super().__init__(message, error_code=kwargs.pop("error_code", "SERVICE_INIT"), **kwargs)


class ExternalAPIError(ExternalServiceError):
    pass


_CONTEXT_ERRORS: Dict[str, Type[NeuromorpheError]] = {
    "audio_processing": AudioProcessingError,
    "database": DatabaseError,
    "rendering": RenderingError,
    "clap": CLAPModelError,
    "api": APIError,
}


@contextmanager
def error_context(context: str, **attributes: Any):
    """Translate an unexpected exception into the context-specific error type."""
    try:
        yield
    except NeuromorpheError:
        raise
    except Exception as exc:
        error_type = _CONTEXT_ERRORS.get(context, NeuromorpheError)
        raise error_type(
            f"{context.replace('_', ' ').title()} failed: {exc}",
            original_exception=exc,
            **attributes,
        ) from exc


class ErrorAggregator:
    def __init__(self) -> None:
        self.errors: List[NeuromorpheError] = []

    def add_error(self, error: NeuromorpheError) -> None:
        self.errors.append(error)

    def has_errors(self) -> bool:
        return bool(self.errors)

    def error_count(self) -> int:
        return len(self.errors)

    def group_by_type(self) -> Dict[Type[NeuromorpheError], List[NeuromorpheError]]:
        grouped: Dict[Type[NeuromorpheError], List[NeuromorpheError]] = {}
        for error in self.errors:
            grouped.setdefault(type(error), []).append(error)
        return grouped

    def raise_if_errors(self) -> None:
        if self.errors:
            raise NeuromorpheError(
                "Multiple errors occurred",
                error_code="MULTIPLE_ERRORS",
                details={"errors": [error.to_dict() for error in self.errors]},
            )
