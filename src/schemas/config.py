from pydantic import Field, ConfigDict, field_validator
from typing import Any, Optional, List, Dict, Union
from enum import Enum


class ConfigDataType(str, Enum):
    """Datentyp einer Konfigurationseinstellung"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    LIST = "list"


class ConfigCategory(str, Enum):
    """Kategorie einer Konfigurationseinstellung"""
    AUDIO = "audio"
    PROCESSING = "processing"
    UI = "ui"
    DATABASE = "database"
    API = "api"
    SECURITY = "security"
    LOGGING = "logging"
    PERFORMANCE = "performance"
    GENERATION = "generation"
    ANALYSIS = "analysis"


class ConfigurationSettingBase(BaseModel):
    """Basis-Schema für Konfigurationseinstellungen"""
    category: ConfigCategory
    key: str = Field(..., min_length=1, max_length=200)
    value: Union[str, int, float, bool, Dict[str, Any], List[Any]] = Field(..., description="Konfigurationswert")

    # Metadaten
    description: Optional[str] = Field(None, description="Beschreibung der Einstellung")
    data_type: ConfigDataType
    is_user_configurable: bool = Field(True, description="Kann vom Benutzer geändert werden")
    requires_restart: bool = Field(False, description="Erfordert Neustart nach Änderung")

    # Validierung
    validation_rules: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validierungsregeln")
    default_value: Optional[Union[str, int, float, bool, Dict[str, Any], List[Any]]] = Field(None, description="Standardwert")

    @field_validator('value')
    @classmethod
    def validate_value_type(cls, v, info):
        """Validiert, dass der Wert dem angegebenen Datentyp entspricht"""
        if not hasattr(info, 'data') or 'data_type' not in info.data:
            return v

        data_type = info.data['data_type']

        if data_type == ConfigDataType.STRING and not isinstance(v, str):
            raise ValueError('Value must be a string')