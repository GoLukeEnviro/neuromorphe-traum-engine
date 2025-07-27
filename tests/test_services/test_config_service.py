"""Tests für Configuration-Service"""

import pytest
import os
import json
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from src.services.config_service import (
    ConfigService, ConfigManager, ConfigValidator,
    ConfigLoader, ConfigWatcher, ConfigCache,
    EnvironmentManager, SecretManager, ConfigMerger,
    ConfigFormat, ConfigSource, ValidationRule,
    ConfigSchema, ConfigValue, ConfigSection,
    ConfigTemplate, ConfigProfile, ConfigHistory,
    ConfigBackup, ConfigSync, ConfigEncryption
)
from src.core.config import (
    Config, DatabaseConfig, CacheConfig, APIConfig,
    MonitoringConfig, RenderConfig, CLAPConfig,
    WebSocketConfig, LoggingConfig, SecurityConfig
)
from src.core.exceptions import (
    ConfigurationError, ConfigValidationError,
    ConfigLoadError, ConfigSaveError, ConfigNotFoundError,
    ConfigFormatError, ConfigPermissionError, ConfigEncryptionError
)
from src.schemas.config import (
    ConfigData, ConfigUpdateData, ConfigValidationData,
    ConfigHistoryData, ConfigBackupData, ConfigProfileData,
    ConfigTemplateData, ConfigSyncData, ConfigStatsData
)


class TestConfigService:
    """Tests für Configuration-Service"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Temporäres Konfigurationsverzeichnis für Tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config_data(self):
        """Beispiel-Konfigurationsdaten"""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "neuromorphe_db",
                "user": "admin",
                "password": "secret123",
                "pool_size": 10,
                "ssl_mode": "require"
            },
            "cache": {
                "backend": "redis",
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None,
                "ttl": 3600,
                "max_connections": 20
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "cors_origins": ["*"],
                "rate_limit": {
                    "requests_per_minute": 100,
                    "burst_size": 20
                },
                "auth": {
                    "jwt_secret": "super_secret_key",
                    "jwt_expiry": 3600,
                    "api_keys": ["key1", "key2"]
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics_collection_interval": 10,
                "health_check_interval": 30,
                "alert_check_interval": 60,
                "dashboard_enabled": True,
                "dashboard_port": 3001
            },
            "render": {
                "max_concurrent_jobs": 4,
                "default_quality": "high",
                "default_format": "wav",
                "output_directory": "/tmp/renders",
                "temp_directory": "/tmp/render_temp",
                "cleanup_after_days": 7
            },
            "clap": {
                "model_name": "laion/clap-htsat-unfused",
                "device": "auto",
                "batch_size": 32,
                "cache_embeddings": True,
                "embedding_dim": 512
            },
            "websocket": {
                "host": "0.0.0.0",
                "port": 8001,
                "max_connections": 100,
                "heartbeat_interval": 30,
                "message_queue_size": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_enabled": True,
                "file_path": "/var/log/neuromorphe.log",
                "file_max_size": "10MB",
                "file_backup_count": 5,
                "console_enabled": True
            },
            "security": {
                "encryption_key": "encryption_key_here",
                "hash_algorithm": "sha256",
                "password_min_length": 8,
                "session_timeout": 1800,
                "max_login_attempts": 5
            }
        }
    
    @pytest.fixture
    def config_service(self, temp_config_dir, sample_config_data):
        """Configuration-Service für Tests"""
        # Konfigurationsdatei erstellen
        config_file = Path(temp_config_dir) / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_data, f)
        
        return ConfigService(
            config_dir=temp_config_dir,
            config_file="config.yaml",
            environment="test",
            auto_reload=False,
            validation_enabled=True,
            encryption_enabled=False
        )
    
    @pytest.mark.unit
    def test_config_service_initialization(self, temp_config_dir):
        """Test: Configuration-Service-Initialisierung"""
        service = ConfigService(
            config_dir=temp_config_dir,
            config_file="config.yaml",
            environment="test"
        )
        
        assert service.config_dir == temp_config_dir
        assert service.config_file == "config.yaml"
        assert service.environment == "test"
        assert isinstance(service.config_manager, ConfigManager)
        assert isinstance(service.config_validator, ConfigValidator)
        assert isinstance(service.config_loader, ConfigLoader)
    
    @pytest.mark.unit
    def test_config_service_invalid_directory(self):
        """Test: Configuration-Service mit ungültigem Verzeichnis"""
        with pytest.raises(ConfigurationError):
            ConfigService(
                config_dir="/nonexistent/directory",
                config_file="config.yaml"
            )
    
    @pytest.mark.unit
    def test_load_config(self, config_service):
        """Test: Konfiguration laden"""
        # Konfiguration laden
        config = config_service.load_config()
        
        assert isinstance(config, Config)
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.api.port == 8000
        assert config.monitoring.enabled == True
    
    @pytest.mark.unit
    def test_load_config_file_not_found(self, temp_config_dir):
        """Test: Konfiguration laden - Datei nicht gefunden"""
        service = ConfigService(
            config_dir=temp_config_dir,
            config_file="nonexistent.yaml"
        )
        
        with pytest.raises(ConfigNotFoundError):
            service.load_config()
    
    @pytest.mark.unit
    def test_save_config(self, config_service, temp_config_dir):
        """Test: Konfiguration speichern"""
        # Konfiguration laden und modifizieren
        config = config_service.load_config()
        config.api.port = 9000
        config.database.pool_size = 20
        
        # Konfiguration speichern
        config_service.save_config(config)
        
        # Konfiguration erneut laden und prüfen
        reloaded_config = config_service.load_config()
        assert reloaded_config.api.port == 9000
        assert reloaded_config.database.pool_size == 20
    
    @pytest.mark.unit
    def test_validate_config(self, config_service):
        """Test: Konfiguration validieren"""
        config = config_service.load_config()
        
        # Gültige Konfiguration validieren
        is_valid, errors = config_service.validate_config(config)
        assert is_valid == True
        assert len(errors) == 0
        
        # Ungültige Konfiguration erstellen
        config.database.port = -1  # Ungültiger Port
        config.api.host = ""       # Leerer Host
        
        is_valid, errors = config_service.validate_config(config)
        assert is_valid == False
        assert len(errors) > 0
        assert any("port" in error.lower() for error in errors)
        assert any("host" in error.lower() for error in errors)
    
    @pytest.mark.unit
    def test_get_config_value(self, config_service):
        """Test: Konfigurationswert abrufen"""
        # Einfacher Wert
        db_host = config_service.get_config_value("database.host")
        assert db_host == "localhost"
        
        # Verschachtelter Wert
        rate_limit = config_service.get_config_value("api.rate_limit.requests_per_minute")
        assert rate_limit == 100
        
        # Nicht existierender Wert
        nonexistent = config_service.get_config_value("nonexistent.key", default="default_value")
        assert nonexistent == "default_value"
        
        # Nicht existierender Wert ohne Default
        with pytest.raises(ConfigNotFoundError):
            config_service.get_config_value("nonexistent.key")
    
    @pytest.mark.unit
    def test_set_config_value(self, config_service):
        """Test: Konfigurationswert setzen"""
        # Einfacher Wert setzen
        config_service.set_config_value("database.host", "new_host")
        assert config_service.get_config_value("database.host") == "new_host"
        
        # Verschachtelter Wert setzen
        config_service.set_config_value("api.rate_limit.requests_per_minute", 200)
        assert config_service.get_config_value("api.rate_limit.requests_per_minute") == 200
        
        # Neuen Wert erstellen
        config_service.set_config_value("new_section.new_key", "new_value")
        assert config_service.get_config_value("new_section.new_key") == "new_value"
    
    @pytest.mark.unit
    def test_environment_variable_override(self, config_service):
        """Test: Umgebungsvariablen-Override"""
        with patch.dict(os.environ, {
            "NEUROMORPHE_DATABASE_HOST": "env_host",
            "NEUROMORPHE_API_PORT": "9000",
            "NEUROMORPHE_MONITORING_ENABLED": "false"
        }):
            # Konfiguration mit Umgebungsvariablen laden
            config = config_service.load_config_with_env_override()
            
            assert config.database.host == "env_host"
            assert config.api.port == 9000
            assert config.monitoring.enabled == False
    
    @pytest.mark.unit
    def test_config_profiles(self, config_service, temp_config_dir):
        """Test: Konfigurationsprofile"""
        # Development-Profil erstellen
        dev_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"debug": True, "port": 8000}
        }
        
        dev_file = Path(temp_config_dir) / "config.dev.yaml"
        with open(dev_file, 'w') as f:
            yaml.dump(dev_config, f)
        
        # Production-Profil erstellen
        prod_config = {
            "database": {"host": "prod-db.example.com", "port": 5432},
            "api": {"debug": False, "port": 80}
        }
        
        prod_file = Path(temp_config_dir) / "config.prod.yaml"
        with open(prod_file, 'w') as f:
            yaml.dump(prod_config, f)
        
        # Development-Profil laden
        dev_config_obj = config_service.load_profile("dev")
        assert dev_config_obj.database.host == "localhost"
        assert dev_config_obj.api.debug == True
        
        # Production-Profil laden
        prod_config_obj = config_service.load_profile("prod")
        assert prod_config_obj.database.host == "prod-db.example.com"
        assert prod_config_obj.api.debug == False
    
    @pytest.mark.unit
    def test_config_templates(self, config_service, temp_config_dir):
        """Test: Konfigurationsvorlagen"""
        # Template erstellen
        template_data = {
            "database": {
                "host": "{{DB_HOST}}",
                "port": "{{DB_PORT}}",
                "name": "{{DB_NAME}}",
                "user": "{{DB_USER}}",
                "password": "{{DB_PASSWORD}}"
            },
            "api": {
                "host": "{{API_HOST}}",
                "port": "{{API_PORT}}",
                "debug": "{{DEBUG}}"
            }
        }
        
        template_file = Path(temp_config_dir) / "config.template.yaml"
        with open(template_file, 'w') as f:
            yaml.dump(template_data, f)
        
        # Template-Variablen
        variables = {
            "DB_HOST": "template-db.example.com",
            "DB_PORT": 5432,
            "DB_NAME": "template_db",
            "DB_USER": "template_user",
            "DB_PASSWORD": "template_pass",
            "API_HOST": "0.0.0.0",
            "API_PORT": 8080,
            "DEBUG": False
        }
        
        # Konfiguration aus Template generieren
        config = config_service.generate_from_template("config.template.yaml", variables)
        
        assert config.database.host == "template-db.example.com"
        assert config.database.port == 5432
        assert config.api.port == 8080
        assert config.api.debug == False


class TestConfigManager:
    """Tests für Config-Manager"""
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Config-Manager für Tests"""
        return ConfigManager(
            config_dir=temp_config_dir,
            backup_enabled=True,
            history_enabled=True,
            max_backups=5,
            max_history_entries=10
        )
    
    @pytest.mark.unit
    def test_config_backup(self, config_manager, temp_config_dir):
        """Test: Konfiguration-Backup"""
        # Original-Konfiguration erstellen
        config_file = Path(temp_config_dir) / "config.yaml"
        original_config = {"test": "value", "number": 42}
        
        with open(config_file, 'w') as f:
            yaml.dump(original_config, f)
        
        # Backup erstellen
        backup_path = config_manager.create_backup("config.yaml")
        
        assert backup_path.exists()
        assert "backup" in backup_path.name
        
        # Backup-Inhalt prüfen
        with open(backup_path, 'r') as f:
            backup_data = yaml.safe_load(f)
        
        assert backup_data == original_config
    
    @pytest.mark.unit
    def test_config_history(self, config_manager, temp_config_dir):
        """Test: Konfiguration-Historie"""
        config_file = Path(temp_config_dir) / "config.yaml"
        
        # Mehrere Konfigurationsversionen erstellen
        configs = [
            {"version": 1, "value": "first"},
            {"version": 2, "value": "second"},
            {"version": 3, "value": "third"}
        ]
        
        for config in configs:
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            # Änderung in Historie speichern
            config_manager.add_to_history("config.yaml", config, "Test change")
        
        # Historie abrufen
        history = config_manager.get_history("config.yaml")
        
        assert len(history) == 3
        assert history[0]["config"]["version"] == 1
        assert history[1]["config"]["version"] == 2
        assert history[2]["config"]["version"] == 3
    
    @pytest.mark.unit
    def test_config_rollback(self, config_manager, temp_config_dir):
        """Test: Konfiguration-Rollback"""
        config_file = Path(temp_config_dir) / "config.yaml"
        
        # Original-Konfiguration
        original_config = {"version": 1, "important": "data"}
        with open(config_file, 'w') as f:
            yaml.dump(original_config, f)
        config_manager.add_to_history("config.yaml", original_config, "Initial")
        
        # Geänderte Konfiguration
        changed_config = {"version": 2, "important": "changed_data"}
        with open(config_file, 'w') as f:
            yaml.dump(changed_config, f)
        config_manager.add_to_history("config.yaml", changed_config, "Changed")
        
        # Rollback zur ersten Version
        config_manager.rollback("config.yaml", version=0)
        
        # Konfiguration prüfen
        with open(config_file, 'r') as f:
            current_config = yaml.safe_load(f)
        
        assert current_config == original_config
        assert current_config["version"] == 1
        assert current_config["important"] == "data"
    
    @pytest.mark.unit
    def test_config_merge(self, config_manager):
        """Test: Konfiguration-Merge"""
        base_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "base_db"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
        
        override_config = {
            "database": {
                "host": "override_host",
                "user": "override_user"  # Neuer Schlüssel
            },
            "cache": {  # Neue Sektion
                "backend": "redis",
                "host": "localhost"
            }
        }
        
        # Konfigurationen mergen
        merged_config = config_manager.merge_configs(base_config, override_config)
        
        # Database-Sektion sollte gemergt sein
        assert merged_config["database"]["host"] == "override_host"  # Überschrieben
        assert merged_config["database"]["port"] == 5432            # Beibehalten
        assert merged_config["database"]["name"] == "base_db"       # Beibehalten
        assert merged_config["database"]["user"] == "override_user" # Hinzugefügt
        
        # API-Sektion sollte unverändert sein
        assert merged_config["api"] == base_config["api"]
        
        # Cache-Sektion sollte hinzugefügt sein
        assert merged_config["cache"] == override_config["cache"]


class TestConfigValidator:
    """Tests für Config-Validator"""
    
    @pytest.fixture
    def config_validator(self):
        """Config-Validator für Tests"""
        return ConfigValidator(
            schema_file="config_schema.json",
            strict_mode=True,
            custom_validators={
                "port": lambda x: 1 <= x <= 65535,
                "host": lambda x: len(x) > 0 and "." in x or x == "localhost"
            }
        )
    
    @pytest.mark.unit
    def test_validate_database_config(self, config_validator):
        """Test: Datenbank-Konfiguration validieren"""
        # Gültige Konfiguration
        valid_db_config = {
            "host": "localhost",
            "port": 5432,
            "name": "test_db",
            "user": "admin",
            "password": "secret",
            "pool_size": 10
        }
        
        is_valid, errors = config_validator.validate_section("database", valid_db_config)
        assert is_valid == True
        assert len(errors) == 0
        
        # Ungültige Konfiguration
        invalid_db_config = {
            "host": "",           # Leerer Host
            "port": 70000,       # Port außerhalb des gültigen Bereichs
            "name": "test_db",
            "user": "",          # Leerer User
            "pool_size": -1      # Negative Pool-Größe
        }
        
        is_valid, errors = config_validator.validate_section("database", invalid_db_config)
        assert is_valid == False
        assert len(errors) > 0
        assert any("host" in error.lower() for error in errors)
        assert any("port" in error.lower() for error in errors)
        assert any("user" in error.lower() for error in errors)
        assert any("pool_size" in error.lower() for error in errors)
    
    @pytest.mark.unit
    def test_validate_api_config(self, config_validator):
        """Test: API-Konfiguration validieren"""
        # Gültige Konfiguration
        valid_api_config = {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
            "cors_origins": ["*"],
            "rate_limit": {
                "requests_per_minute": 100,
                "burst_size": 20
            }
        }
        
        is_valid, errors = config_validator.validate_section("api", valid_api_config)
        assert is_valid == True
        assert len(errors) == 0
        
        # Ungültige Konfiguration
        invalid_api_config = {
            "host": "",                    # Leerer Host
            "port": 0,                     # Ungültiger Port
            "debug": "not_boolean",        # Falscher Typ
            "cors_origins": "not_list",    # Falscher Typ
            "rate_limit": {
                "requests_per_minute": -1,  # Negative Anzahl
                "burst_size": 0            # Null-Wert
            }
        }
        
        is_valid, errors = config_validator.validate_section("api", invalid_api_config)
        assert is_valid == False
        assert len(errors) > 0
    
    @pytest.mark.unit
    def test_custom_validation_rules(self, config_validator):
        """Test: Benutzerdefinierte Validierungsregeln"""
        # Port-Validator testen
        assert config_validator.custom_validators["port"](8000) == True
        assert config_validator.custom_validators["port"](0) == False
        assert config_validator.custom_validators["port"](70000) == False
        
        # Host-Validator testen
        assert config_validator.custom_validators["host"]("localhost") == True
        assert config_validator.custom_validators["host"]("example.com") == True
        assert config_validator.custom_validators["host"]("") == False
        assert config_validator.custom_validators["host"]("invalid") == False
    
    @pytest.mark.unit
    def test_validate_required_fields(self, config_validator):
        """Test: Pflichtfelder validieren"""
        # Konfiguration mit fehlenden Pflichtfeldern
        incomplete_config = {
            "database": {
                "host": "localhost"
                # port, name, user fehlen
            },
            "api": {
                "host": "0.0.0.0"
                # port fehlt
            }
        }
        
        is_valid, errors = config_validator.validate_config(incomplete_config)
        assert is_valid == False
        assert len(errors) > 0
        assert any("required" in error.lower() for error in errors)
    
    @pytest.mark.unit
    def test_validate_data_types(self, config_validator):
        """Test: Datentypen validieren"""
        # Konfiguration mit falschen Datentypen
        wrong_types_config = {
            "database": {
                "host": "localhost",
                "port": "not_a_number",    # Sollte int sein
                "name": "test_db",
                "user": "admin",
                "pool_size": "not_a_number" # Sollte int sein
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": "not_boolean",     # Sollte bool sein
                "cors_origins": "not_list"  # Sollte list sein
            }
        }
        
        is_valid, errors = config_validator.validate_config(wrong_types_config)
        assert is_valid == False
        assert len(errors) > 0
        assert any("type" in error.lower() for error in errors)


class TestConfigServiceIntegration:
    """Integrationstests für Configuration-Service"""
    
    @pytest.mark.integration
    def test_full_config_workflow(self, temp_config_dir):
        """Test: Vollständiger Konfiguration-Workflow"""
        # 1. Service initialisieren
        service = ConfigService(
            config_dir=temp_config_dir,
            config_file="config.yaml",
            environment="test",
            validation_enabled=True,
            backup_enabled=True
        )
        
        # 2. Initial-Konfiguration erstellen
        initial_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"host": "0.0.0.0", "port": 8000}
        }
        
        config_file = Path(temp_config_dir) / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(initial_config, f)
        
        # 3. Konfiguration laden
        config = service.load_config()
        assert config.database.host == "localhost"
        assert config.api.port == 8000
        
        # 4. Konfiguration validieren
        is_valid, errors = service.validate_config(config)
        assert is_valid == True
        
        # 5. Konfiguration ändern
        config.database.host = "new_host"
        config.api.port = 9000
        
        # 6. Geänderte Konfiguration speichern
        service.save_config(config)
        
        # 7. Konfiguration erneut laden und prüfen
        reloaded_config = service.load_config()
        assert reloaded_config.database.host == "new_host"
        assert reloaded_config.api.port == 9000
        
        # 8. Backup sollte erstellt worden sein
        backup_files = list(Path(temp_config_dir).glob("*backup*"))
        assert len(backup_files) > 0
    
    @pytest.mark.integration
    def test_config_with_environment_override(self, temp_config_dir):
        """Test: Konfiguration mit Umgebungsvariablen-Override"""
        # Basis-Konfiguration erstellen
        base_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"host": "0.0.0.0", "port": 8000, "debug": False}
        }
        
        config_file = Path(temp_config_dir) / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(base_config, f)
        
        service = ConfigService(
            config_dir=temp_config_dir,
            config_file="config.yaml"
        )
        
        # Umgebungsvariablen setzen
        with patch.dict(os.environ, {
            "NEUROMORPHE_DATABASE_HOST": "env_db_host",
            "NEUROMORPHE_API_PORT": "9000",
            "NEUROMORPHE_API_DEBUG": "true"
        }):
            # Konfiguration mit Umgebungsvariablen laden
            config = service.load_config_with_env_override()
            
            # Umgebungsvariablen sollten Basis-Konfiguration überschreiben
            assert config.database.host == "env_db_host"
            assert config.database.port == 5432  # Nicht überschrieben
            assert config.api.port == 9000
            assert config.api.debug == True
    
    @pytest.mark.performance
    def test_config_service_performance(self, temp_config_dir):
        """Test: Configuration-Service-Performance"""
        import time
        
        # Große Konfiguration erstellen
        large_config = {}
        for i in range(100):
            large_config[f"section_{i}"] = {
                f"key_{j}": f"value_{j}" for j in range(50)
            }
        
        config_file = Path(temp_config_dir) / "large_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(large_config, f)
        
        service = ConfigService(
            config_dir=temp_config_dir,
            config_file="large_config.yaml"
        )
        
        # Performance-Test: Laden
        start_time = time.time()
        
        for _ in range(10):
            config = service.load_config()
            assert len(config.__dict__) > 0
        
        load_time = time.time() - start_time
        
        # Sollte unter 1 Sekunde für 10 Ladevorgänge dauern
        assert load_time < 1.0
        
        # Performance-Test: Speichern
        start_time = time.time()
        
        for _ in range(10):
            service.save_config(config)
        
        save_time = time.time() - start_time
        
        # Sollte unter 1 Sekunde für 10 Speichervorgänge dauern
        assert save_time < 1.0
        
        # Performance-Test: Werte abrufen
        start_time = time.time()
        
        for i in range(100):
            for j in range(10):  # Nur 10 von 50 Keys pro Sektion
                value = service.get_config_value(f"section_{i}.key_{j}")
                assert value == f"value_{j}"
        
        get_time = time.time() - start_time
        
        # Sollte unter 0.5 Sekunden für 1000 Abrufe dauern
        assert get_time < 0.5