"""Tests für Core-Konfiguration"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.config import (
    Settings, DatabaseConfig, AudioConfig, CLAPConfig,
    RenderConfig, APIConfig, LoggingConfig
)
from src.core.exceptions import ConfigurationError


class TestDatabaseConfig:
    """Tests für Datenbank-Konfiguration"""
    
    @pytest.mark.unit
    def test_database_config_creation(self):
        """Test: DatabaseConfig-Instanz erstellen"""
        config = DatabaseConfig(
            url="sqlite:///test.db",
            echo=True,
            pool_size=10,
            max_overflow=20
        )
        
        assert config.url == "sqlite:///test.db"
        assert config.echo == True
        assert config.pool_size == 10
        assert config.max_overflow == 20
    
    @pytest.mark.unit
    def test_database_config_defaults(self):
        """Test: DatabaseConfig-Standardwerte"""
        config = DatabaseConfig()
        
        assert config.url == "sqlite:///neuromorphe_traum.db"
        assert config.echo == False
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.pool_timeout == 30
        assert config.pool_recycle == 3600
    
    @pytest.mark.unit
    def test_database_config_validation(self):
        """Test: DatabaseConfig-Validierung"""
        # Gültige Konfiguration
        valid_config = DatabaseConfig(url="postgresql://user:pass@localhost/db")
        assert valid_config.url.startswith("postgresql://")
        
        # Ungültige URL sollte Fehler verursachen
        with pytest.raises(ValueError):
            DatabaseConfig(url="invalid_url")
        
        # Negative Pool-Größe sollte Fehler verursachen
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=-1)


class TestAudioConfig:
    """Tests für Audio-Konfiguration"""
    
    @pytest.mark.unit
    def test_audio_config_creation(self):
        """Test: AudioConfig-Instanz erstellen"""
        config = AudioConfig(
            sample_rate=48000,
            channels=2,
            bit_depth=24,
            buffer_size=1024,
            supported_formats=["wav", "mp3", "flac"]
        )
        
        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.bit_depth == 24
        assert config.buffer_size == 1024
        assert "wav" in config.supported_formats
    
    @pytest.mark.unit
    def test_audio_config_defaults(self):
        """Test: AudioConfig-Standardwerte"""
        config = AudioConfig()
        
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.bit_depth == 16
        assert config.buffer_size == 512
        assert config.max_file_size == 100 * 1024 * 1024  # 100MB
        assert "wav" in config.supported_formats
        assert "mp3" in config.supported_formats
    
    @pytest.mark.unit
    def test_audio_config_validation(self):
        """Test: AudioConfig-Validierung"""
        # Ungültige Sample-Rate
        with pytest.raises(ValueError):
            AudioConfig(sample_rate=0)
        
        # Ungültige Kanal-Anzahl
        with pytest.raises(ValueError):
            AudioConfig(channels=0)
        
        # Ungültige Bit-Tiefe
        with pytest.raises(ValueError):
            AudioConfig(bit_depth=7)  # Nicht unterstützt
        
        # Leere Format-Liste
        with pytest.raises(ValueError):
            AudioConfig(supported_formats=[])


class TestCLAPConfig:
    """Tests für CLAP-Konfiguration"""
    
    @pytest.mark.unit
    def test_clap_config_creation(self):
        """Test: CLAPConfig-Instanz erstellen"""
        config = CLAPConfig(
            model_name="custom-clap-model",
            model_path="/path/to/model",
            device="cuda",
            batch_size=16,
            embedding_dim=1024
        )
        
        assert config.model_name == "custom-clap-model"
        assert config.model_path == "/path/to/model"
        assert config.device == "cuda"
        assert config.batch_size == 16
        assert config.embedding_dim == 1024
    
    @pytest.mark.unit
    def test_clap_config_defaults(self):
        """Test: CLAPConfig-Standardwerte"""
        config = CLAPConfig()
        
        assert config.model_name == "laion/clap-htsat-unfused"
        assert config.device == "auto"
        assert config.batch_size == 8
        assert config.embedding_dim == 512
        assert config.max_audio_length == 30.0
        assert config.cache_embeddings == True
    
    @pytest.mark.unit
    def test_clap_config_device_auto_detection(self):
        """Test: Automatische Device-Erkennung"""
        config = CLAPConfig(device="auto")
        
        # Device sollte automatisch erkannt werden
        detected_device = config.get_device()
        assert detected_device in ["cpu", "cuda", "mps"]
    
    @pytest.mark.unit
    def test_clap_config_validation(self):
        """Test: CLAPConfig-Validierung"""
        # Ungültige Batch-Größe
        with pytest.raises(ValueError):
            CLAPConfig(batch_size=0)
        
        # Ungültige Embedding-Dimension
        with pytest.raises(ValueError):
            CLAPConfig(embedding_dim=-1)
        
        # Ungültige Audio-Länge
        with pytest.raises(ValueError):
            CLAPConfig(max_audio_length=0)


class TestRenderConfig:
    """Tests für Render-Konfiguration"""
    
    @pytest.mark.unit
    def test_render_config_creation(self):
        """Test: RenderConfig-Instanz erstellen"""
        config = RenderConfig(
            output_dir="/custom/output",
            temp_dir="/custom/temp",
            max_concurrent_jobs=4,
            default_format="flac",
            quality_presets={
                "ultra": {"sample_rate": 96000, "bit_depth": 32}
            }
        )
        
        assert config.output_dir == "/custom/output"
        assert config.temp_dir == "/custom/temp"
        assert config.max_concurrent_jobs == 4
        assert config.default_format == "flac"
        assert "ultra" in config.quality_presets
    
    @pytest.mark.unit
    def test_render_config_defaults(self):
        """Test: RenderConfig-Standardwerte"""
        config = RenderConfig()
        
        assert config.output_dir == "./output"
        assert config.temp_dir == "./temp"
        assert config.max_concurrent_jobs == 2
        assert config.default_format == "wav"
        assert config.cleanup_temp_files == True
        assert "high" in config.quality_presets
        assert "medium" in config.quality_presets
    
    @pytest.mark.unit
    def test_render_config_directory_creation(self, temp_dir: Path):
        """Test: Automatische Verzeichnis-Erstellung"""
        output_dir = temp_dir / "test_output"
        temp_render_dir = temp_dir / "test_temp"
        
        config = RenderConfig(
            output_dir=str(output_dir),
            temp_dir=str(temp_render_dir)
        )
        
        # Verzeichnisse sollten erstellt werden
        config.ensure_directories()
        
        assert output_dir.exists()
        assert temp_render_dir.exists()
    
    @pytest.mark.unit
    def test_render_config_validation(self):
        """Test: RenderConfig-Validierung"""
        # Ungültige Job-Anzahl
        with pytest.raises(ValueError):
            RenderConfig(max_concurrent_jobs=0)
        
        # Ungültiges Format
        with pytest.raises(ValueError):
            RenderConfig(default_format="invalid_format")


class TestAPIConfig:
    """Tests für API-Konfiguration"""
    
    @pytest.mark.unit
    def test_api_config_creation(self):
        """Test: APIConfig-Instanz erstellen"""
        config = APIConfig(
            host="0.0.0.0",
            port=8080,
            debug=True,
            cors_origins=["http://localhost:3000"],
            rate_limit="100/minute"
        )
        
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.debug == True
        assert "http://localhost:3000" in config.cors_origins
        assert config.rate_limit == "100/minute"
    
    @pytest.mark.unit
    def test_api_config_defaults(self):
        """Test: APIConfig-Standardwerte"""
        config = APIConfig()
        
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.debug == False
        assert config.workers == 1
        assert config.rate_limit == "60/minute"
        assert config.max_upload_size == 50 * 1024 * 1024  # 50MB
    
    @pytest.mark.unit
    def test_api_config_validation(self):
        """Test: APIConfig-Validierung"""
        # Ungültiger Port
        with pytest.raises(ValueError):
            APIConfig(port=0)
        
        with pytest.raises(ValueError):
            APIConfig(port=70000)  # Zu hoch
        
        # Ungültige Worker-Anzahl
        with pytest.raises(ValueError):
            APIConfig(workers=0)


class TestLoggingConfig:
    """Tests für Logging-Konfiguration"""
    
    @pytest.mark.unit
    def test_logging_config_creation(self):
        """Test: LoggingConfig-Instanz erstellen"""
        config = LoggingConfig(
            level="DEBUG",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file_path="/var/log/neuromorphe.log",
            max_file_size=10 * 1024 * 1024,
            backup_count=5
        )
        
        assert config.level == "DEBUG"
        assert "%(asctime)s" in config.format
        assert config.file_path == "/var/log/neuromorphe.log"
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.backup_count == 5
    
    @pytest.mark.unit
    def test_logging_config_defaults(self):
        """Test: LoggingConfig-Standardwerte"""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.console_output == True
        assert config.file_output == True
        assert config.json_format == False
        assert config.max_file_size == 5 * 1024 * 1024  # 5MB
        assert config.backup_count == 3
    
    @pytest.mark.unit
    def test_logging_config_validation(self):
        """Test: LoggingConfig-Validierung"""
        # Ungültiges Log-Level
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")
        
        # Ungültige Datei-Größe
        with pytest.raises(ValueError):
            LoggingConfig(max_file_size=0)
        
        # Ungültige Backup-Anzahl
        with pytest.raises(ValueError):
            LoggingConfig(backup_count=-1)


class TestSettings:
    """Tests für Haupt-Settings-Klasse"""
    
    @pytest.mark.unit
    def test_settings_creation(self):
        """Test: Settings-Instanz erstellen"""
        settings = Settings(
            environment="development",
            debug=True,
            secret_key="test-secret-key",
            database=DatabaseConfig(url="sqlite:///test.db"),
            audio=AudioConfig(sample_rate=48000),
            clap=CLAPConfig(device="cpu"),
            render=RenderConfig(max_concurrent_jobs=4),
            api=APIConfig(port=8080),
            logging=LoggingConfig(level="DEBUG")
        )
        
        assert settings.environment == "development"
        assert settings.debug == True
        assert settings.secret_key == "test-secret-key"
        assert settings.database.url == "sqlite:///test.db"
        assert settings.audio.sample_rate == 48000
        assert settings.clap.device == "cpu"
        assert settings.render.max_concurrent_jobs == 4
        assert settings.api.port == 8080
        assert settings.logging.level == "DEBUG"
    
    @pytest.mark.unit
    def test_settings_defaults(self):
        """Test: Settings-Standardwerte"""
        settings = Settings()
        
        assert settings.environment == "production"
        assert settings.debug == False
        assert settings.secret_key is not None
        assert len(settings.secret_key) >= 32
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.audio, AudioConfig)
        assert isinstance(settings.clap, CLAPConfig)
        assert isinstance(settings.render, RenderConfig)
        assert isinstance(settings.api, APIConfig)
        assert isinstance(settings.logging, LoggingConfig)
    
    @pytest.mark.unit
    def test_settings_from_env(self):
        """Test: Settings aus Umgebungsvariablen laden"""
        env_vars = {
            "NEUROMORPHE_ENVIRONMENT": "development",
            "NEUROMORPHE_DEBUG": "true",
            "NEUROMORPHE_SECRET_KEY": "env-secret-key",
            "NEUROMORPHE_DATABASE_URL": "postgresql://localhost/test",
            "NEUROMORPHE_AUDIO_SAMPLE_RATE": "48000",
            "NEUROMORPHE_CLAP_DEVICE": "cuda",
            "NEUROMORPHE_API_PORT": "9000",
            "NEUROMORPHE_LOGGING_LEVEL": "WARNING"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings.from_env()
            
            assert settings.environment == "development"
            assert settings.debug == True
            assert settings.secret_key == "env-secret-key"
            assert settings.database.url == "postgresql://localhost/test"
            assert settings.audio.sample_rate == 48000
            assert settings.clap.device == "cuda"
            assert settings.api.port == 9000
            assert settings.logging.level == "WARNING"
    
    @pytest.mark.unit
    def test_settings_from_file(self, temp_dir: Path):
        """Test: Settings aus Datei laden"""
        config_file = temp_dir / "config.json"
        
        config_data = {
            "environment": "testing",
            "debug": True,
            "secret_key": "file-secret-key",
            "database": {
                "url": "sqlite:///file_test.db",
                "echo": True
            },
            "audio": {
                "sample_rate": 96000,
                "channels": 2
            },
            "clap": {
                "model_name": "custom-model",
                "device": "cpu"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 7000
            },
            "logging": {
                "level": "ERROR",
                "console_output": False
            }
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        settings = Settings.from_file(str(config_file))
        
        assert settings.environment == "testing"
        assert settings.debug == True
        assert settings.secret_key == "file-secret-key"
        assert settings.database.url == "sqlite:///file_test.db"
        assert settings.database.echo == True
        assert settings.audio.sample_rate == 96000
        assert settings.clap.model_name == "custom-model"
        assert settings.api.host == "0.0.0.0"
        assert settings.api.port == 7000
        assert settings.logging.level == "ERROR"
        assert settings.logging.console_output == False
    
    @pytest.mark.unit
    def test_settings_validation(self):
        """Test: Settings-Validierung"""
        # Ungültige Umgebung
        with pytest.raises(ValueError):
            Settings(environment="invalid_env")
        
        # Zu kurzer Secret-Key
        with pytest.raises(ValueError):
            Settings(secret_key="short")
    
    @pytest.mark.unit
    def test_settings_to_dict(self):
        """Test: Settings zu Dictionary konvertieren"""
        settings = Settings(
            environment="development",
            debug=True
        )
        
        settings_dict = settings.to_dict()
        
        assert settings_dict["environment"] == "development"
        assert settings_dict["debug"] == True
        assert "database" in settings_dict
        assert "audio" in settings_dict
        assert "clap" in settings_dict
        assert "render" in settings_dict
        assert "api" in settings_dict
        assert "logging" in settings_dict
    
    @pytest.mark.unit
    def test_settings_merge(self):
        """Test: Settings zusammenführen"""
        base_settings = Settings(
            environment="production",
            debug=False,
            database=DatabaseConfig(url="sqlite:///base.db")
        )
        
        override_settings = Settings(
            environment="development",
            debug=True,
            database=DatabaseConfig(url="sqlite:///override.db", echo=True)
        )
        
        merged_settings = base_settings.merge(override_settings)
        
        assert merged_settings.environment == "development"
        assert merged_settings.debug == True
        assert merged_settings.database.url == "sqlite:///override.db"
        assert merged_settings.database.echo == True
    
    @pytest.mark.unit
    def test_settings_copy(self):
        """Test: Settings kopieren"""
        original_settings = Settings(
            environment="development",
            debug=True,
            secret_key="original-key"
        )
        
        copied_settings = original_settings.copy()
        
        # Kopie sollte identisch sein
        assert copied_settings.environment == original_settings.environment
        assert copied_settings.debug == original_settings.debug
        assert copied_settings.secret_key == original_settings.secret_key
        
        # Aber separate Instanz
        assert copied_settings is not original_settings
        
        # Änderungen an Kopie sollten Original nicht beeinflussen
        copied_settings.debug = False
        assert original_settings.debug == True


class TestConfigurationErrors:
    """Tests für Konfigurationsfehler"""
    
    @pytest.mark.unit
    def test_invalid_config_file(self, temp_dir: Path):
        """Test: Ungültige Konfigurationsdatei"""
        # Nicht existierende Datei
        with pytest.raises(ConfigurationError):
            Settings.from_file("/nonexistent/config.json")
        
        # Ungültiges JSON
        invalid_json_file = temp_dir / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(ConfigurationError):
            Settings.from_file(str(invalid_json_file))
    
    @pytest.mark.unit
    def test_missing_required_env_vars(self):
        """Test: Fehlende erforderliche Umgebungsvariablen"""
        # Alle Umgebungsvariablen löschen
        with patch.dict(os.environ, {}, clear=True):
            # Sollte trotzdem funktionieren mit Standardwerten
            settings = Settings.from_env()
            assert settings.environment == "production"
    
    @pytest.mark.unit
    def test_invalid_env_var_types(self):
        """Test: Ungültige Typen in Umgebungsvariablen"""
        env_vars = {
            "NEUROMORPHE_DEBUG": "not_a_boolean",
            "NEUROMORPHE_API_PORT": "not_a_number",
            "NEUROMORPHE_AUDIO_SAMPLE_RATE": "invalid_number"
        }
        
        with patch.dict(os.environ, env_vars):
            with pytest.raises(ConfigurationError):
                Settings.from_env()


class TestConfigurationPerformance:
    """Tests für Konfiguration-Performance"""
    
    @pytest.mark.performance
    def test_settings_loading_performance(self, temp_dir: Path):
        """Test: Performance beim Laden der Settings"""
        import time
        
        # Große Konfigurationsdatei erstellen
        config_file = temp_dir / "large_config.json"
        
        large_config = {
            "environment": "testing",
            "database": {"url": "sqlite:///test.db"},
            "audio": {"sample_rate": 44100},
            "clap": {"model_name": "test-model"},
            "render": {"max_concurrent_jobs": 2},
            "api": {"port": 8000},
            "logging": {"level": "INFO"}
        }
        
        # Viele zusätzliche Konfigurationsoptionen hinzufügen
        for i in range(1000):
            large_config[f"custom_option_{i}"] = f"value_{i}"
        
        import json
        with open(config_file, 'w') as f:
            json.dump(large_config, f)
        
        start_time = time.time()
        
        settings = Settings.from_file(str(config_file))
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        assert settings.environment == "testing"
        assert loading_time < 1.0  # Sollte unter 1 Sekunde dauern
    
    @pytest.mark.performance
    def test_settings_serialization_performance(self):
        """Test: Performance der Settings-Serialisierung"""
        import time
        
        settings = Settings(
            environment="development",
            debug=True
        )
        
        start_time = time.time()
        
        # Mehrfache Serialisierung
        for _ in range(1000):
            settings_dict = settings.to_dict()
        
        end_time = time.time()
        serialization_time = end_time - start_time
        
        assert "environment" in settings_dict
        assert serialization_time < 1.0  # Sollte unter 1 Sekunde dauern