"""
Definiert benutzerdefinierte Ausnahmen für die Neuromorphe Traum-Engine.
"""

from typing import Optional, Dict, Any


class NeuromorpheError(Exception):
    """Basis-Ausnahme für alle anwendungsspezifischen Fehler.

    Attribute:
        message (str): Eine beschreibende Fehlermeldung.
        code (Optional[int]): Ein optionaler Fehlercode.
        details (Optional[Dict[str, Any]]): Zusätzliche Fehlerdetails.
    """

    def __init__(
        self, message: str, code: Optional[int] = None, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details

    def __str__(self):
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class DatabaseError(NeuromorpheError):
    """Ausnahme für Datenbankoperationen.
    """

    def __init__(self, message: str = "Database operation failed", **kwargs):
        super().__init__(message, code=5001, **kwargs)


class NotFoundError(NeuromorpheError):
    """Ausnahme, wenn eine Ressource nicht gefunden wurde.
    """

    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, code=4004, **kwargs)


class ValidationError(NeuromorpheError):
    """Ausnahme für Validierungsfehler (z.B. Pydantic).
    """

    def __init__(self, message: str = "Validation failed", **kwargs):
        super().__init__(message, code=4000, **kwargs)


class ServiceInitializationError(NeuromorpheError):
    """Ausnahme für Fehler bei der Initialisierung von Diensten.
    """

    def __init__(self, message: str = "Service initialization failed", **kwargs):
        super().__init__(message, code=5002, **kwargs)


class AudioProcessingError(NeuromorpheError):
    """Ausnahme für Fehler während der Audioverarbeitung.
    """

    def __init__(self, message: str = "Audio processing failed", **kwargs):
        super().__init__(message, code=5003, **kwargs)


class RenderingError(NeuromorpheError):
    """Ausnahme für Fehler während des Renderings.
    """

    def __init__(self, message: str = "Audio rendering failed", **kwargs):
        super().__init__(message, code=5004, **kwargs)


class ExternalAPIError(NeuromorpheError):
    """Ausnahme für Fehler bei der Kommunikation mit externen APIs.
    """

    def __init__(self, message: str = "External API call failed", **kwargs):
        super().__init__(message, code=5005, **kwargs)


class AuthenticationError(NeuromorpheError):
    """Ausnahme für Authentifizierungsfehler.
    """

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, code=4001, **kwargs)


class AuthorizationError(NeuromorpheError):
    """Ausnahme für Autorisierungsfehler.
    """

    def __init__(self, message: str = "Authorization failed", **kwargs):
        super().__init__(message, code=4003, **kwargs)


class ConfigurationError(NeuromorpheError):
    """Ausnahme für Konfigurationsfehler.
    """

    def __init__(self, message: str = "Configuration error", **kwargs):
        super().__init__(message, code=5006, **kwargs)