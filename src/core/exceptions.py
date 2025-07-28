class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class NeuromorpheError(Exception):
    """Base exception for Neuromorphe Dream Engine errors."""
    pass

class AudioProcessingError(NeuromorpheError):
    """Exception raised for errors during audio processing."""
    pass

class CLAPModelError(NeuromorpheError):
    """Exception raised for errors related to CLAP model operations."""
    pass

class DatabaseError(NeuromorpheError):
    """Exception raised for database-related errors."""
    pass

class RenderingError(NeuromorpheError):
    """Exception raised for errors during audio rendering."""
    pass

class APIError(NeuromorpheError):
    """Exception raised for API-related errors."""
    pass

class ValidationError(NeuromorpheError):
    """Exception raised for data validation errors."""
    pass

class FileNotFoundError(NeuromorpheError):
    """Exception raised when a required file is not found."""
    pass

class InsufficientResourcesError(NeuromorpheError):
    """Exception raised when there are insufficient system resources."""
    pass

class TimeoutError(NeuromorpheError):
    """Exception raised when an operation times out."""
    pass

class AuthenticationError(NeuromorpheError):
    """Exception raised for authentication failures."""
    pass

class AuthorizationError(NeuromorpheError):
    """Exception raised for authorization failures."""
    pass

class RateLimitError(NeuromorpheError):
    """Exception raised when a rate limit is exceeded."""
    pass

class ExternalServiceError(NeuromorpheError):
    """Exception raised for errors from external services."""
    pass
