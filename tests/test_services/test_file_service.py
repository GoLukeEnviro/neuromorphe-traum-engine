"""Tests für File-Service"""

import pytest
import os
import tempfile
import shutil
import hashlib
import mimetypes
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, List, Any, Optional, BinaryIO
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO, StringIO

from src.services.file_service import (
    FileService, FileManager, FileStorage, FileValidator,
    FileProcessor, FileMetadata, FileUploader, FileDownloader,
    FileCompressor, FileEncryptor, FileThumbnail, FileCache,
    FileWatcher, FileIndexer, FileSearcher, FileSynchronizer,
    StorageBackend, LocalStorage, CloudStorage, S3Storage,
    FilePermissions, FileAccess, FileQuota, FileVersioning,
    FileBackup, FileReplication, FileIntegrity, FileAudit,
    FileStats, FileAnalyzer, FileConverter, FileOptimizer,
    UploadSession, DownloadSession, TransferProgress,
    FileOperation, FileEvent, FileNotification, FileAlert,
    FileConfig, StorageConfig, UploadConfig, ProcessingConfig,
    FileInfo, FileContent, FileChunk, FileStream,
    FileFilter, FileSorter, FileGrouper, FileArchiver
)
from src.core.config import FileConfig as CoreFileConfig
from src.core.exceptions import (
    FileError, FileNotFoundError, FilePermissionError,
    FileValidationError, FileStorageError, FileProcessingError,
    FileUploadError, FileDownloadError, FileCompressionError,
    FileEncryptionError, FileThumbnailError, FileCacheError,
    FileWatcherError, FileIndexError, FileSearchError,
    FileSyncError, FileQuotaError, FileVersionError,
    FileBackupError, FileIntegrityError, FileAuditError
)
from src.schemas.file import (
    FileData, FileMetadataData, UploadData, DownloadData,
    FileStatsData, FileConfigData, StorageConfigData,
    FileOperationData, FileEventData, FileNotificationData
)


@dataclass
class TestFile:
    """Test-Datei"""
    name: str
    content: bytes
    mime_type: str
    size: int
    checksum: str
    created_at: datetime = None
    modified_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.modified_at is None:
            self.modified_at = self.created_at


class TestFileService:
    """Tests für File-Service"""
    
    @pytest.fixture
    def file_config(self):
        """File-Konfiguration für Tests"""
        return CoreFileConfig(
            storage_backend="local",
            storage_path="/tmp/neuromorphe_test_storage",
            max_file_size=100 * 1024 * 1024,  # 100 MB
            allowed_extensions=[".wav", ".mp3", ".flac", ".aiff", ".ogg"],
            allowed_mime_types=[
                "audio/wav", "audio/mpeg", "audio/flac",
                "audio/aiff", "audio/ogg"
            ],
            upload_chunk_size=1024 * 1024,  # 1 MB
            download_chunk_size=1024 * 1024,  # 1 MB
            compression_enabled=True,
            compression_level=6,
            encryption_enabled=True,
            encryption_algorithm="AES-256-GCM",
            thumbnail_enabled=True,
            thumbnail_sizes=[(128, 128), (256, 256), (512, 512)],
            cache_enabled=True,
            cache_size=1024 * 1024 * 1024,  # 1 GB
            cache_ttl=3600,  # 1 Stunde
            versioning_enabled=True,
            max_versions=10,
            backup_enabled=True,
            backup_interval=86400,  # 24 Stunden
            integrity_check_enabled=True,
            integrity_check_interval=3600,  # 1 Stunde
            audit_enabled=True,
            quota_enabled=True,
            default_quota=10 * 1024 * 1024 * 1024,  # 10 GB
            watch_enabled=True,
            index_enabled=True,
            search_enabled=True,
            sync_enabled=False,  # Für Tests deaktiviert
            replication_enabled=False  # Für Tests deaktiviert
        )
    
    @pytest.fixture
    def temp_dir(self):
        """Temporäres Verzeichnis für Tests"""
        temp_dir = tempfile.mkdtemp(prefix="neuromorphe_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def file_service(self, file_config, temp_dir):
        """File-Service für Tests"""
        # Storage-Pfad auf temporäres Verzeichnis setzen
        file_config.storage_path = temp_dir
        return FileService(file_config)
    
    @pytest.fixture
    def test_audio_file(self):
        """Test-Audio-Datei"""
        # Simulierte WAV-Datei (vereinfacht)
        content = b"RIFF" + b"\x00" * 36 + b"WAVE" + b"\x00" * 1000
        checksum = hashlib.sha256(content).hexdigest()
        
        return TestFile(
            name="test_audio.wav",
            content=content,
            mime_type="audio/wav",
            size=len(content),
            checksum=checksum
        )
    
    @pytest.fixture
    def test_large_file(self):
        """Test-Große-Datei"""
        # 5 MB Test-Datei
        content = b"A" * (5 * 1024 * 1024)
        checksum = hashlib.sha256(content).hexdigest()
        
        return TestFile(
            name="large_test.wav",
            content=content,
            mime_type="audio/wav",
            size=len(content),
            checksum=checksum
        )
    
    @pytest.mark.unit
    def test_file_service_initialization(self, file_config, temp_dir):
        """Test: File-Service-Initialisierung"""
        file_config.storage_path = temp_dir
        service = FileService(file_config)
        
        assert service.config == file_config
        assert isinstance(service.file_manager, FileManager)
        assert isinstance(service.storage, FileStorage)
        assert isinstance(service.validator, FileValidator)
        assert isinstance(service.processor, FileProcessor)
        assert isinstance(service.uploader, FileUploader)
        assert isinstance(service.downloader, FileDownloader)
        
        # Storage-Verzeichnis sollte erstellt worden sein
        assert os.path.exists(temp_dir)
    
    @pytest.mark.unit
    def test_file_service_invalid_config(self):
        """Test: File-Service mit ungültiger Konfiguration"""
        invalid_config = CoreFileConfig(
            storage_path="",  # Leerer Storage-Pfad
            max_file_size=0,  # Keine maximale Dateigröße
            upload_chunk_size=0,  # Keine Chunk-Größe
            allowed_extensions=[],  # Keine erlaubten Erweiterungen
            cache_size=-1  # Negative Cache-Größe
        )
        
        with pytest.raises(FileError):
            FileService(invalid_config)
    
    @pytest.mark.unit
    def test_file_upload(self, file_service, test_audio_file):
        """Test: Datei-Upload"""
        # Datei hochladen
        file_stream = BytesIO(test_audio_file.content)
        
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="user_12345"
        )
        
        assert upload_result.success == True
        assert upload_result.file_id is not None
        assert upload_result.filename == test_audio_file.name
        assert upload_result.size == test_audio_file.size
        assert upload_result.checksum == test_audio_file.checksum
        
        # Datei sollte im Storage existieren
        file_exists = file_service.file_exists(upload_result.file_id)
        assert file_exists == True
        
        # Datei-Metadaten abrufen
        metadata = file_service.get_file_metadata(upload_result.file_id)
        assert metadata.filename == test_audio_file.name
        assert metadata.mime_type == test_audio_file.mime_type
        assert metadata.size == test_audio_file.size
    
    @pytest.mark.unit
    def test_file_upload_validation(self, file_service):
        """Test: Datei-Upload-Validierung"""
        # Ungültige Dateierweiterung
        invalid_file = BytesIO(b"invalid content")
        
        with pytest.raises(FileValidationError):
            file_service.upload_file(
                file_stream=invalid_file,
                filename="invalid.txt",
                mime_type="text/plain",
                user_id="user_12345"
            )
        
        # Zu große Datei
        large_content = b"A" * (200 * 1024 * 1024)  # 200 MB
        large_file = BytesIO(large_content)
        
        with pytest.raises(FileValidationError):
            file_service.upload_file(
                file_stream=large_file,
                filename="too_large.wav",
                mime_type="audio/wav",
                user_id="user_12345"
            )
        
        # Ungültiger MIME-Type
        audio_file = BytesIO(b"RIFF" + b"\x00" * 1000)
        
        with pytest.raises(FileValidationError):
            file_service.upload_file(
                file_stream=audio_file,
                filename="test.wav",
                mime_type="video/mp4",  # Falscher MIME-Type
                user_id="user_12345"
            )
    
    @pytest.mark.unit
    def test_file_download(self, file_service, test_audio_file):
        """Test: Datei-Download"""
        # Datei hochladen
        file_stream = BytesIO(test_audio_file.content)
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="user_12345"
        )
        
        # Datei herunterladen
        download_stream = file_service.download_file(upload_result.file_id)
        
        assert download_stream is not None
        
        # Inhalt vergleichen
        downloaded_content = download_stream.read()
        assert downloaded_content == test_audio_file.content
        
        # Download-Statistiken
        download_stats = file_service.get_download_stats(upload_result.file_id)
        assert download_stats.download_count >= 1
        assert download_stats.last_download is not None
    
    @pytest.mark.unit
    def test_file_download_nonexistent(self, file_service):
        """Test: Download nicht existierender Datei"""
        with pytest.raises(FileNotFoundError):
            file_service.download_file("nonexistent_file_id")
    
    @pytest.mark.unit
    def test_chunked_upload(self, file_service, test_large_file):
        """Test: Chunked-Upload"""
        # Upload-Session starten
        upload_session = file_service.start_upload_session(
            filename=test_large_file.name,
            file_size=test_large_file.size,
            mime_type=test_large_file.mime_type,
            user_id="user_12345"
        )
        
        assert upload_session.session_id is not None
        assert upload_session.chunk_size > 0
        assert upload_session.total_chunks > 1
        
        # Datei in Chunks aufteilen und hochladen
        chunk_size = upload_session.chunk_size
        content = test_large_file.content
        
        for chunk_index in range(upload_session.total_chunks):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, len(content))
            chunk_data = content[start:end]
            
            chunk_result = file_service.upload_chunk(
                session_id=upload_session.session_id,
                chunk_index=chunk_index,
                chunk_data=chunk_data
            )
            
            assert chunk_result.success == True
            assert chunk_result.chunk_index == chunk_index
        
        # Upload abschließen
        final_result = file_service.complete_upload_session(upload_session.session_id)
        
        assert final_result.success == True
        assert final_result.file_id is not None
        assert final_result.size == test_large_file.size
        
        # Datei sollte vollständig und korrekt sein
        downloaded_stream = file_service.download_file(final_result.file_id)
        downloaded_content = downloaded_stream.read()
        assert downloaded_content == test_large_file.content
    
    @pytest.mark.unit
    def test_file_compression(self, file_service, test_large_file):
        """Test: Datei-Kompression"""
        # Datei mit Kompression hochladen
        file_stream = BytesIO(test_large_file.content)
        
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_large_file.name,
            mime_type=test_large_file.mime_type,
            user_id="user_12345",
            compress=True
        )
        
        # Datei-Metadaten prüfen
        metadata = file_service.get_file_metadata(upload_result.file_id)
        
        assert metadata.is_compressed == True
        assert metadata.compressed_size < metadata.original_size
        assert metadata.compression_ratio > 0
        
        # Datei herunterladen (sollte automatisch dekomprimiert werden)
        download_stream = file_service.download_file(upload_result.file_id)
        downloaded_content = download_stream.read()
        
        assert downloaded_content == test_large_file.content
    
    @pytest.mark.unit
    def test_file_encryption(self, file_service, test_audio_file):
        """Test: Datei-Verschlüsselung"""
        # Datei mit Verschlüsselung hochladen
        file_stream = BytesIO(test_audio_file.content)
        
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="user_12345",
            encrypt=True
        )
        
        # Datei-Metadaten prüfen
        metadata = file_service.get_file_metadata(upload_result.file_id)
        
        assert metadata.is_encrypted == True
        assert metadata.encryption_algorithm is not None
        
        # Datei herunterladen (sollte automatisch entschlüsselt werden)
        download_stream = file_service.download_file(upload_result.file_id)
        downloaded_content = download_stream.read()
        
        assert downloaded_content == test_audio_file.content
        
        # Rohe verschlüsselte Datei sollte anders sein
        raw_encrypted_content = file_service.get_raw_file_content(upload_result.file_id)
        assert raw_encrypted_content != test_audio_file.content
    
    @pytest.mark.unit
    def test_file_versioning(self, file_service, test_audio_file):
        """Test: Datei-Versionierung"""
        # Erste Version hochladen
        file_stream = BytesIO(test_audio_file.content)
        
        upload_result_v1 = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="user_12345"
        )
        
        # Zweite Version hochladen (gleicher Dateiname)
        modified_content = test_audio_file.content + b"MODIFIED"
        file_stream_v2 = BytesIO(modified_content)
        
        upload_result_v2 = file_service.upload_file(
            file_stream=file_stream_v2,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="user_12345",
            replace_existing=True
        )
        
        # Versionen abrufen
        versions = file_service.get_file_versions(upload_result_v2.file_id)
        
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2
        assert versions[1].is_current == True
        
        # Aktuelle Version herunterladen
        current_stream = file_service.download_file(upload_result_v2.file_id)
        current_content = current_stream.read()
        assert current_content == modified_content
        
        # Vorherige Version herunterladen
        previous_stream = file_service.download_file_version(
            upload_result_v2.file_id, version=1
        )
        previous_content = previous_stream.read()
        assert previous_content == test_audio_file.content
    
    @pytest.mark.unit
    def test_file_thumbnail_generation(self, file_service):
        """Test: Datei-Thumbnail-Generierung"""
        # Simulierte Bild-Datei für Thumbnail
        image_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000  # Fake PNG
        file_stream = BytesIO(image_content)
        
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename="test_image.png",
            mime_type="image/png",
            user_id="user_12345",
            generate_thumbnails=True
        )
        
        # Thumbnails abrufen
        thumbnails = file_service.get_file_thumbnails(upload_result.file_id)
        
        assert len(thumbnails) > 0
        
        for thumbnail in thumbnails:
            assert thumbnail.width > 0
            assert thumbnail.height > 0
            assert thumbnail.size > 0
            assert thumbnail.format in ["PNG", "JPEG"]
            
            # Thumbnail herunterladen
            thumbnail_stream = file_service.download_thumbnail(
                upload_result.file_id, thumbnail.size_key
            )
            assert thumbnail_stream is not None
    
    @pytest.mark.unit
    def test_file_search(self, file_service, test_audio_file):
        """Test: Datei-Suche"""
        # Mehrere Dateien hochladen
        files_to_upload = [
            ("beat_track_1.wav", b"RIFF" + b"\x00" * 1000),
            ("melody_sample.wav", b"RIFF" + b"\x00" * 1500),
            ("drum_loop.wav", b"RIFF" + b"\x00" * 2000),
            ("bass_line.wav", b"RIFF" + b"\x00" * 1200)
        ]
        
        uploaded_files = []
        for filename, content in files_to_upload:
            file_stream = BytesIO(content)
            result = file_service.upload_file(
                file_stream=file_stream,
                filename=filename,
                mime_type="audio/wav",
                user_id="user_12345"
            )
            uploaded_files.append(result)
        
        # Nach Dateinamen suchen
        search_results = file_service.search_files(
            query="beat",
            user_id="user_12345"
        )
        
        assert len(search_results) >= 1
        assert any("beat" in result.filename.lower() for result in search_results)
        
        # Nach MIME-Type filtern
        audio_files = file_service.search_files(
            mime_type="audio/wav",
            user_id="user_12345"
        )
        
        assert len(audio_files) >= 4
        
        # Nach Dateigröße filtern
        large_files = file_service.search_files(
            min_size=1500,
            user_id="user_12345"
        )
        
        assert len(large_files) >= 2
        
        # Nach Datum filtern
        recent_files = file_service.search_files(
            created_after=datetime.now() - timedelta(minutes=1),
            user_id="user_12345"
        )
        
        assert len(recent_files) >= 4
    
    @pytest.mark.unit
    def test_file_permissions(self, file_service, test_audio_file):
        """Test: Datei-Berechtigungen"""
        # Datei hochladen
        file_stream = BytesIO(test_audio_file.content)
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="owner_12345"
        )
        
        # Berechtigungen setzen
        file_service.set_file_permissions(
            file_id=upload_result.file_id,
            owner_id="owner_12345",
            permissions={
                "read": ["user_67890", "user_11111"],
                "write": ["user_67890"],
                "delete": ["owner_12345"]
            }
        )
        
        # Berechtigungen prüfen
        can_read_owner = file_service.check_file_permission(
            file_id=upload_result.file_id,
            user_id="owner_12345",
            permission="read"
        )
        assert can_read_owner == True
        
        can_read_user = file_service.check_file_permission(
            file_id=upload_result.file_id,
            user_id="user_67890",
            permission="read"
        )
        assert can_read_user == True
        
        can_write_user = file_service.check_file_permission(
            file_id=upload_result.file_id,
            user_id="user_67890",
            permission="write"
        )
        assert can_write_user == True
        
        can_delete_user = file_service.check_file_permission(
            file_id=upload_result.file_id,
            user_id="user_67890",
            permission="delete"
        )
        assert can_delete_user == False
        
        can_read_unauthorized = file_service.check_file_permission(
            file_id=upload_result.file_id,
            user_id="unauthorized_user",
            permission="read"
        )
        assert can_read_unauthorized == False
    
    @pytest.mark.unit
    def test_file_quota_management(self, file_service, test_audio_file):
        """Test: Datei-Quota-Management"""
        user_id = "quota_user_12345"
        
        # Benutzer-Quota setzen (5 MB)
        file_service.set_user_quota(user_id, 5 * 1024 * 1024)
        
        # Quota-Info abrufen
        quota_info = file_service.get_user_quota(user_id)
        assert quota_info.total_quota == 5 * 1024 * 1024
        assert quota_info.used_quota == 0
        assert quota_info.available_quota == 5 * 1024 * 1024
        
        # Datei hochladen
        file_stream = BytesIO(test_audio_file.content)
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id=user_id
        )
        
        # Quota sollte aktualisiert sein
        updated_quota = file_service.get_user_quota(user_id)
        assert updated_quota.used_quota == test_audio_file.size
        assert updated_quota.available_quota == (5 * 1024 * 1024) - test_audio_file.size
        
        # Große Datei hochladen (sollte Quota überschreiten)
        large_content = b"A" * (6 * 1024 * 1024)  # 6 MB
        large_file_stream = BytesIO(large_content)
        
        with pytest.raises(FileQuotaError):
            file_service.upload_file(
                file_stream=large_file_stream,
                filename="large_file.wav",
                mime_type="audio/wav",
                user_id=user_id
            )
    
    @pytest.mark.unit
    def test_file_integrity_check(self, file_service, test_audio_file):
        """Test: Datei-Integritätsprüfung"""
        # Datei hochladen
        file_stream = BytesIO(test_audio_file.content)
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="user_12345"
        )
        
        # Integritätsprüfung durchführen
        integrity_result = file_service.check_file_integrity(upload_result.file_id)
        
        assert integrity_result.is_valid == True
        assert integrity_result.checksum_match == True
        assert integrity_result.size_match == True
        assert integrity_result.corruption_detected == False
        
        # Datei-Metadaten mit korrekter Prüfsumme
        metadata = file_service.get_file_metadata(upload_result.file_id)
        assert metadata.checksum == test_audio_file.checksum
    
    @pytest.mark.unit
    def test_file_backup_and_restore(self, file_service, test_audio_file):
        """Test: Datei-Backup und -Wiederherstellung"""
        # Datei hochladen
        file_stream = BytesIO(test_audio_file.content)
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="user_12345"
        )
        
        # Backup erstellen
        backup_result = file_service.create_file_backup(upload_result.file_id)
        
        assert backup_result.success == True
        assert backup_result.backup_id is not None
        assert backup_result.backup_location is not None
        
        # Backup-Liste abrufen
        backups = file_service.list_file_backups(upload_result.file_id)
        assert len(backups) >= 1
        assert backups[0].backup_id == backup_result.backup_id
        
        # Datei löschen
        file_service.delete_file(upload_result.file_id, user_id="user_12345")
        
        # Datei sollte nicht mehr existieren
        assert file_service.file_exists(upload_result.file_id) == False
        
        # Datei aus Backup wiederherstellen
        restore_result = file_service.restore_file_from_backup(
            backup_id=backup_result.backup_id,
            restore_location=upload_result.file_id
        )
        
        assert restore_result.success == True
        assert restore_result.file_id == upload_result.file_id
        
        # Datei sollte wieder existieren
        assert file_service.file_exists(upload_result.file_id) == True
        
        # Inhalt sollte identisch sein
        restored_stream = file_service.download_file(upload_result.file_id)
        restored_content = restored_stream.read()
        assert restored_content == test_audio_file.content
    
    @pytest.mark.unit
    def test_file_deletion(self, file_service, test_audio_file):
        """Test: Datei-Löschung"""
        # Datei hochladen
        file_stream = BytesIO(test_audio_file.content)
        upload_result = file_service.upload_file(
            file_stream=file_stream,
            filename=test_audio_file.name,
            mime_type=test_audio_file.mime_type,
            user_id="user_12345"
        )
        
        # Datei sollte existieren
        assert file_service.file_exists(upload_result.file_id) == True
        
        # Datei löschen
        delete_result = file_service.delete_file(
            file_id=upload_result.file_id,
            user_id="user_12345"
        )
        
        assert delete_result.success == True
        assert delete_result.deleted_at is not None
        
        # Datei sollte nicht mehr existieren
        assert file_service.file_exists(upload_result.file_id) == False
        
        # Download sollte fehlschlagen
        with pytest.raises(FileNotFoundError):
            file_service.download_file(upload_result.file_id)
    
    @pytest.mark.unit
    def test_file_statistics(self, file_service, test_audio_file):
        """Test: Datei-Statistiken"""
        user_id = "stats_user_12345"
        
        # Mehrere Dateien hochladen
        for i in range(3):
            content = test_audio_file.content + bytes(f"_{i}", 'utf-8')
            file_stream = BytesIO(content)
            
            file_service.upload_file(
                file_stream=file_stream,
                filename=f"test_file_{i}.wav",
                mime_type="audio/wav",
                user_id=user_id
            )
        
        # Benutzer-Statistiken abrufen
        user_stats = file_service.get_user_file_stats(user_id)
        
        assert user_stats.total_files >= 3
        assert user_stats.total_size > 0
        assert user_stats.average_file_size > 0
        assert "audio/wav" in user_stats.mime_type_distribution
        
        # System-Statistiken abrufen
        system_stats = file_service.get_system_file_stats()
        
        assert system_stats.total_files >= 3
        assert system_stats.total_size > 0
        assert system_stats.total_users >= 1
        assert system_stats.storage_usage > 0


class TestFileManager:
    """Tests für File-Manager"""
    
    @pytest.fixture
    def file_manager(self, file_config, temp_dir):
        """File-Manager für Tests"""
        file_config.storage_path = temp_dir
        return FileManager(file_config)
    
    @pytest.mark.unit
    def test_file_path_generation(self, file_manager):
        """Test: Datei-Pfad-Generierung"""
        file_id = "test_file_12345"
        user_id = "user_67890"
        
        # Pfad generieren
        file_path = file_manager.generate_file_path(file_id, user_id)
        
        assert file_path is not None
        assert isinstance(file_path, Path)
        assert file_id in str(file_path)
        
        # Pfad sollte im Storage-Verzeichnis sein
        assert str(file_path).startswith(file_manager.config.storage_path)
        
        # Verzeichnisstruktur sollte erstellt werden
        file_manager.ensure_directory_exists(file_path.parent)
        assert file_path.parent.exists()
    
    @pytest.mark.unit
    def test_file_metadata_management(self, file_manager):
        """Test: Datei-Metadaten-Management"""
        file_id = "metadata_test_12345"
        
        # Metadaten erstellen
        metadata = FileMetadata(
            file_id=file_id,
            filename="test.wav",
            mime_type="audio/wav",
            size=1024,
            checksum="abc123",
            user_id="user_12345",
            created_at=datetime.now(),
            modified_at=datetime.now()
        )
        
        # Metadaten speichern
        file_manager.save_metadata(metadata)
        
        # Metadaten abrufen
        retrieved_metadata = file_manager.get_metadata(file_id)
        
        assert retrieved_metadata is not None
        assert retrieved_metadata.file_id == file_id
        assert retrieved_metadata.filename == "test.wav"
        assert retrieved_metadata.size == 1024
        
        # Metadaten aktualisieren
        metadata.size = 2048
        metadata.modified_at = datetime.now()
        
        file_manager.update_metadata(metadata)
        
        updated_metadata = file_manager.get_metadata(file_id)
        assert updated_metadata.size == 2048
        
        # Metadaten löschen
        file_manager.delete_metadata(file_id)
        
        deleted_metadata = file_manager.get_metadata(file_id)
        assert deleted_metadata is None


class TestFileServiceIntegration:
    """Integrationstests für File-Service"""
    
    @pytest.mark.integration
    def test_full_file_lifecycle(self, temp_dir):
        """Test: Vollständiger Datei-Lebenszyklus"""
        config = CoreFileConfig(
            storage_path=temp_dir,
            max_file_size=10 * 1024 * 1024,
            compression_enabled=True,
            encryption_enabled=True,
            versioning_enabled=True,
            backup_enabled=True
        )
        
        service = FileService(config)
        
        # Test-Datei erstellen
        test_content = b"RIFF" + b"\x00" * 5000 + b"WAVE" + b"\x00" * 5000
        test_filename = "integration_test.wav"
        user_id = "integration_user_12345"
        
        # 1. Datei hochladen
        file_stream = BytesIO(test_content)
        upload_result = service.upload_file(
            file_stream=file_stream,
            filename=test_filename,
            mime_type="audio/wav",
            user_id=user_id,
            compress=True,
            encrypt=True
        )
        
        assert upload_result.success == True
        file_id = upload_result.file_id
        
        # 2. Datei-Metadaten prüfen
        metadata = service.get_file_metadata(file_id)
        assert metadata.filename == test_filename
        assert metadata.is_compressed == True
        assert metadata.is_encrypted == True
        
        # 3. Datei herunterladen
        download_stream = service.download_file(file_id)
        downloaded_content = download_stream.read()
        assert downloaded_content == test_content
        
        # 4. Neue Version hochladen
        modified_content = test_content + b"MODIFIED"
        modified_stream = BytesIO(modified_content)
        
        upload_v2_result = service.upload_file(
            file_stream=modified_stream,
            filename=test_filename,
            mime_type="audio/wav",
            user_id=user_id,
            replace_existing=True
        )
        
        # 5. Versionen prüfen
        versions = service.get_file_versions(file_id)
        assert len(versions) == 2
        
        # 6. Backup erstellen
        backup_result = service.create_file_backup(file_id)
        assert backup_result.success == True
        
        # 7. Integritätsprüfung
        integrity_result = service.check_file_integrity(file_id)
        assert integrity_result.is_valid == True
        
        # 8. Datei suchen
        search_results = service.search_files(
            query="integration",
            user_id=user_id
        )
        assert len(search_results) >= 1
        
        # 9. Statistiken abrufen
        user_stats = service.get_user_file_stats(user_id)
        assert user_stats.total_files >= 1
        
        # 10. Datei löschen
        delete_result = service.delete_file(file_id, user_id)
        assert delete_result.success == True
        assert service.file_exists(file_id) == False
    
    @pytest.mark.performance
    def test_file_service_performance(self, temp_dir):
        """Test: File-Service-Performance"""
        config = CoreFileConfig(
            storage_path=temp_dir,
            compression_enabled=False,  # Für Performance-Test deaktiviert
            encryption_enabled=False,   # Für Performance-Test deaktiviert
            versioning_enabled=False,   # Für Performance-Test deaktiviert
            backup_enabled=False        # Für Performance-Test deaktiviert
        )
        
        service = FileService(config)
        
        # Performance-Test: Viele kleine Dateien hochladen
        start_time = time.time()
        
        uploaded_files = []
        for i in range(100):
            content = f"Test file content {i}".encode('utf-8') * 100  # ~2KB pro Datei
            file_stream = BytesIO(content)
            
            result = service.upload_file(
                file_stream=file_stream,
                filename=f"perf_test_{i}.txt",
                mime_type="text/plain",
                user_id="perf_user"
            )
            uploaded_files.append(result.file_id)
        
        upload_time = time.time() - start_time
        
        # Sollte unter 10 Sekunden für 100 Dateien dauern
        assert upload_time < 10.0
        
        # Performance-Test: Viele Dateien herunterladen
        start_time = time.time()
        
        for file_id in uploaded_files:
            download_stream = service.download_file(file_id)
            content = download_stream.read()
            assert len(content) > 0
        
        download_time = time.time() - start_time
        
        # Sollte unter 5 Sekunden für 100 Downloads dauern
        assert download_time < 5.0
        
        # Performance-Test: Datei-Suche
        start_time = time.time()
        
        for i in range(50):
            search_results = service.search_files(
                query=f"perf_test_{i}",
                user_id="perf_user"
            )
            assert len(search_results) >= 1
        
        search_time = time.time() - start_time
        
        # Sollte unter 2 Sekunden für 50 Suchen dauern
        assert search_time < 2.0
        
        # Performance-Test: Metadaten-Abruf
        start_time = time.time()
        
        for file_id in uploaded_files:
            metadata = service.get_file_metadata(file_id)
            assert metadata is not None
        
        metadata_time = time.time() - start_time
        
        # Sollte unter 1 Sekunde für 100 Metadaten-Abrufe dauern
        assert metadata_time < 1.0