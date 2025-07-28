"""Tests für DatabaseManager"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import sqlite3

from src.database.database import DatabaseManager
from src.database.models import Stem, GeneratedTrack, ProcessingJob
from src.core.config import Settings


class TestDatabaseManager:
    """Test-Suite für DatabaseManager"""
    
    @pytest.fixture
    async def db_manager(self, test_settings: Settings) -> DatabaseManager:
        """DatabaseManager-Instanz für Tests"""
        manager = DatabaseManager(test_settings)
        await manager.initialize()
        yield manager
        await manager.close()
    
    @pytest.mark.unit
    async def test_initialization(self, test_settings: Settings):
        """Test: DatabaseManager-Initialisierung"""
        manager = DatabaseManager(test_settings)
        
        assert manager.settings == test_settings
        assert manager.database_url == test_settings.database.url
        
        await manager.initialize()
        assert manager.engine is not None
        assert manager.session_factory is not None
        
        await manager.close()
    
    @pytest.mark.unit
    async def test_create_tables(self, db_manager: DatabaseManager):
        """Test: Tabellen-Erstellung"""
        # Tabellen sollten bereits durch initialize() erstellt worden sein
        async with db_manager.get_session() as session:
            # Prüfen ob Tabellen existieren
            result = await session.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in result.fetchall()]
            
            assert "stems" in tables
            assert "arrangements" in tables
            assert "render_jobs" in tables
    
    @pytest.mark.unit
    async def test_session_management(self, db_manager: DatabaseManager):
        """Test: Session-Management"""
        # Session erstellen
        async with db_manager.get_session() as session:
            assert session is not None
            
            # Einfache Abfrage
            result = await session.execute("SELECT 1")
            assert result.scalar() == 1
        
        # Session sollte automatisch geschlossen werden
    
    @pytest.mark.unit
    async def test_health_check(self, db_manager: DatabaseManager):
        """Test: Datenbank-Health-Check"""
        is_healthy = await db_manager.health_check()
        assert is_healthy == True
    
    @pytest.mark.unit
    async def test_health_check_failure(self, test_settings: Settings):
        """Test: Health-Check bei Datenbankfehler"""
        # Ungültige Datenbank-URL
        test_settings.database.url = "sqlite:///nonexistent/path/test.db"
        manager = DatabaseManager(test_settings)
        
        with pytest.raises(Exception):
            await manager.initialize()


class TestStemOperations:
    """Tests für Stem-Operationen"""
    
    @pytest.mark.unit
    async def test_create_stem(self, db_manager: DatabaseManager):
        """Test: Stem erstellen"""
        stem_data = {
            "name": "Test Kick",
            "file_path": "/path/to/kick.wav",
            "type": "kick",
            "genre": "techno",
            "tempo": 128.0,
            "key": "Am",
            "duration": 4.0,
            "tags": ["dark", "heavy"],
            "features": {
                "spectral_centroid": 1500.0,
                "mfcc": [0.1, 0.2, 0.3],
                "chroma": [0.5, 0.6, 0.7]
            },
            "embeddings": [0.1] * 512
        }
        
        stem_id = await db_manager.create_stem(stem_data)
        
        assert stem_id is not None
        assert isinstance(stem_id, str)
        
        # Stem abrufen und validieren
        stem = await db_manager.get_stem(stem_id)
        assert stem is not None
        assert stem["name"] == "Test Kick"
        assert stem["type"] == "kick"
        assert stem["tempo"] == 128.0
    
    @pytest.mark.unit
    async def test_get_stem(self, db_manager: DatabaseManager):
        """Test: Stem abrufen"""
        # Erst einen Stem erstellen
        stem_data = {
            "name": "Test Bass",
            "file_path": "/path/to/bass.wav",
            "type": "bass",
            "genre": "house",
            "tempo": 125.0,
            "duration": 8.0,
            "embeddings": [0.2] * 512
        }
        
        stem_id = await db_manager.create_stem(stem_data)
        
        # Stem abrufen
        stem = await db_manager.get_stem(stem_id)
        
        assert stem is not None
        assert stem["id"] == stem_id
        assert stem["name"] == "Test Bass"
        assert stem["type"] == "bass"
    
    @pytest.mark.unit
    async def test_get_stem_not_found(self, db_manager: DatabaseManager):
        """Test: Nicht existierenden Stem abrufen"""
        stem = await db_manager.get_stem("nonexistent_id")
        assert stem is None
    
    @pytest.mark.unit
    async def test_update_stem(self, db_manager: DatabaseManager):
        """Test: Stem aktualisieren"""
        # Stem erstellen
        stem_data = {
            "name": "Original Name",
            "file_path": "/path/to/original.wav",
            "type": "synth",
            "genre": "techno",
            "embeddings": [0.3] * 512
        }
        
        stem_id = await db_manager.create_stem(stem_data)
        
        # Stem aktualisieren
        update_data = {
            "name": "Updated Name",
            "tags": ["updated", "modified"]
        }
        
        updated_stem = await db_manager.update_stem(stem_id, update_data)
        
        assert updated_stem is not None
        assert updated_stem["name"] == "Updated Name"
        assert "updated" in updated_stem["tags"]
        assert updated_stem["type"] == "synth"  # Unverändert
    
    @pytest.mark.unit
    async def test_delete_stem(self, db_manager: DatabaseManager):
        """Test: Stem löschen"""
        # Stem erstellen
        stem_data = {
            "name": "To Delete",
            "file_path": "/path/to/delete.wav",
            "type": "fx",
            "embeddings": [0.4] * 512
        }
        
        stem_id = await db_manager.create_stem(stem_data)
        
        # Stem löschen
        deleted = await db_manager.delete_stem(stem_id)
        assert deleted == True
        
        # Stem sollte nicht mehr existieren
        stem = await db_manager.get_stem(stem_id)
        assert stem is None
    
    @pytest.mark.unit
    async def test_search_stems(self, db_manager: DatabaseManager):
        """Test: Stems suchen"""
        # Mehrere Stems erstellen
        stems_data = [
            {
                "name": "Techno Kick 1",
                "file_path": "/path/to/kick1.wav",
                "type": "kick",
                "genre": "techno",
                "tempo": 128.0,
                "tags": ["dark", "heavy"],
                "embeddings": [0.1] * 512
            },
            {
                "name": "House Kick 1",
                "file_path": "/path/to/kick2.wav",
                "type": "kick",
                "genre": "house",
                "tempo": 125.0,
                "tags": ["punchy", "clean"],
                "embeddings": [0.2] * 512
            },
            {
                "name": "Techno Bass 1",
                "file_path": "/path/to/bass1.wav",
                "type": "bass",
                "genre": "techno",
                "tempo": 128.0,
                "tags": ["dark", "distorted"],
                "embeddings": [0.3] * 512
            }
        ]
        
        for stem_data in stems_data:
            await db_manager.create_stem(stem_data)
        
        # Nach Genre suchen
        techno_stems = await db_manager.search_stems(
            filters={"genre": "techno"},
            limit=10
        )
        
        assert len(techno_stems["stems"]) == 2
        assert all(stem["genre"] == "techno" for stem in techno_stems["stems"])
        
        # Nach Typ suchen
        kick_stems = await db_manager.search_stems(
            filters={"type": "kick"},
            limit=10
        )
        
        assert len(kick_stems["stems"]) == 2
        assert all(stem["type"] == "kick" for stem in kick_stems["stems"])
        
        # Nach Text suchen
        house_stems = await db_manager.search_stems(
            query="house",
            limit=10
        )
        
        assert len(house_stems["stems"]) >= 1
        assert any("house" in stem["name"].lower() or stem["genre"] == "house" 
                  for stem in house_stems["stems"])
    
    @pytest.mark.unit
    async def test_get_similar_stems(self, db_manager: DatabaseManager):
        """Test: Ähnliche Stems finden"""
        # Stems mit verschiedenen Embeddings erstellen
        stems_data = [
            {
                "name": "Similar 1",
                "file_path": "/path/to/similar1.wav",
                "type": "kick",
                "embeddings": [1.0] + [0.0] * 511  # Sehr ähnlich zu Ziel
            },
            {
                "name": "Similar 2",
                "file_path": "/path/to/similar2.wav",
                "type": "kick",
                "embeddings": [0.9] + [0.1] * 511  # Ähnlich zu Ziel
            },
            {
                "name": "Different",
                "file_path": "/path/to/different.wav",
                "type": "kick",
                "embeddings": [0.0] * 512  # Sehr unterschiedlich
            }
        ]
        
        for stem_data in stems_data:
            await db_manager.create_stem(stem_data)
        
        # Ähnliche Stems finden
        target_embeddings = [1.0] + [0.0] * 511
        
        similar_stems = await db_manager.get_similar_stems(
            embeddings=target_embeddings,
            limit=2,
            threshold=0.5
        )
        
        assert len(similar_stems) <= 2
        assert all(stem["similarity"] >= 0.5 for stem in similar_stems)
        
        # Sollte nach Ähnlichkeit sortiert sein
        similarities = [stem["similarity"] for stem in similar_stems]
        assert similarities == sorted(similarities, reverse=True)


class TestArrangementOperations:
    """Tests für Arrangement-Operationen"""
    
    @pytest.mark.unit
    async def test_create_arrangement(self, db_manager: DatabaseManager):
        """Test: Arrangement erstellen"""
        track_data = {
            "prompt": "Dark techno with heavy bass",
            "duration": 180,
            "genre": "techno",
            "track_structure": {
                "sections": [
                    {"name": "intro", "start": 0, "duration": 32, "stems": [1, 2]},
                    {"name": "main", "start": 32, "duration": 96, "stems": [1, 2, 3]},
                    {"name": "outro", "start": 128, "duration": 32, "stems": [1]}
                ],
                "total_duration": 160
            },
            "stems": [1, 2, 3],
            "metadata": {
                "tempo": 128,
                "key": "Am",
                "energy": 0.8
            }
        }
        
        track_id = await db_manager.create_generated_track(track_data)
        
        assert track_id is not None
        assert isinstance(track_id, str)
        
        # GeneratedTrack abrufen und validieren
        track = await db_manager.get_generated_track(track_id)
        assert track is not None
        assert track["prompt"] == "Dark techno with heavy bass"
        assert track["duration"] == 180
        assert len(track["track_structure"]["sections"]) == 3
    
    @pytest.mark.unit
    async def test_list_generated_tracks(self, db_manager: DatabaseManager):
        """Test: GeneratedTracks auflisten"""
        # Mehrere GeneratedTracks erstellen
        tracks_data = [
            {
                "original_prompt": "Track 1",
                "duration": 120,
                "track_structure": {"sections": []},
                "stems": []
            },
            {
                "original_prompt": "Track 2",
                "duration": 180,
                "track_structure": {"sections": []},
                "stems": []
            },
            {
                "original_prompt": "Track 3",
                "duration": 240,
                "track_structure": {"sections": []},
                "stems": []
            }
        ]
        
        for track_data in tracks_data:
            await db_manager.create_generated_track(track_data)
        
        # GeneratedTracks auflisten
        result = await db_manager.list_generated_tracks(
            page=1,
            per_page=2
        )
        
        assert "tracks" in result
        assert "total" in result
        assert "page" in result
        assert "per_page" in result
        
        assert len(result["tracks"]) <= 2
        assert result["total"] >= 3
        assert result["page"] == 1
        assert result["per_page"] == 2
    
    @pytest.mark.unit
    async def test_update_generated_track(self, db_manager: DatabaseManager):
        """Test: GeneratedTrack aktualisieren"""
        # GeneratedTrack erstellen
        track_data = {
            "original_prompt": "Original prompt",
            "duration": 120,
            "track_structure": {"sections": []},
            "stems": []
        }
        
        track_id = await db_manager.create_generated_track(track_data)
        
        # GeneratedTrack aktualisieren
        update_data = {
            "original_prompt": "Updated prompt",
            "track_metadata": {"updated": True}
        }
        
        updated_track = await db_manager.update_generated_track(
            track_id, 
            update_data
        )
        
        assert updated_track is not None
        assert updated_track["original_prompt"] == "Updated prompt"
        assert updated_track["track_metadata"]["updated"] == True
        assert updated_track["duration"] == 120  # Unverändert


class TestRenderJobOperations:
    """Tests für RenderJob-Operationen"""
    
    @pytest.mark.unit
    async def test_create_render_job(self, db_manager: DatabaseManager):
        """Test: Render-Job erstellen"""
        # Erst ein GeneratedTrack erstellen
        track_data = {
            "original_prompt": "Test generated track",
            "duration": 180,
            "track_structure": {"sections": []},
            "stems": []
        }
        
        track_id = await db_manager.create_generated_track(track_data)
        
        # Render-Job erstellen
        job_data = {
            "arrangement_id": track_id,
            "format": "wav",
            "quality": "high",
            "options": {
                "normalize": True,
                "apply_mastering": True
            }
        }
        
        job_id = await db_manager.create_render_job(job_data)
        
        assert job_id is not None
        assert isinstance(job_id, str)
        
        # Job abrufen und validieren
        job = await db_manager.get_render_job(job_id)
        assert job is not None
        assert job["arrangement_id"] == track_id
        assert job["format"] == "wav"
        assert job["status"] == "pending"
    
    @pytest.mark.unit
    async def test_update_render_job_status(self, db_manager: DatabaseManager):
        """Test: Render-Job-Status aktualisieren"""
        # GeneratedTrack und Job erstellen
        track_data = {
            "original_prompt": "Test",
            "duration": 60,
            "track_structure": {"sections": []},
            "stems": []
        }
        
        track_id = await db_manager.create_generated_track(track_data)
        
        job_data = {
            "arrangement_id": track_id,
            "format": "mp3"
        }
        
        job_id = await db_manager.create_render_job(job_data)
        
        # Status auf "processing" setzen
        await db_manager.update_render_job_status(
            job_id, 
            "processing", 
            progress=0.5
        )
        
        job = await db_manager.get_render_job(job_id)
        assert job["status"] == "processing"
        assert job["progress"] == 0.5
        
        # Status auf "completed" setzen
        await db_manager.update_render_job_status(
            job_id, 
            "completed", 
            progress=1.0,
            output_path="/path/to/output.mp3"
        )
        
        job = await db_manager.get_render_job(job_id)
        assert job["status"] == "completed"
        assert job["progress"] == 1.0
        assert job["output_path"] == "/path/to/output.mp3"
    
    @pytest.mark.unit
    async def test_list_render_jobs(self, db_manager: DatabaseManager):
        """Test: Render-Jobs auflisten"""
        # GeneratedTrack erstellen
        track_data = {
            "original_prompt": "Test",
            "duration": 60,
            "track_structure": {"sections": []},
            "stems": []
        }
        
        track_id = await db_manager.create_generated_track(track_data)
        
        # Mehrere Jobs erstellen
        for i in range(3):
            job_data = {
                "arrangement_id": track_id,
                "format": "wav" if i % 2 == 0 else "mp3"
            }
            await db_manager.create_render_job(job_data)
        
        # Jobs auflisten
        result = await db_manager.list_render_jobs(
            filters={"format": "wav"},
            limit=10
        )
        
        assert "jobs" in result
        assert "total" in result
        
        # Sollte nur WAV-Jobs enthalten
        wav_jobs = [job for job in result["jobs"] if job["format"] == "wav"]
        assert len(wav_jobs) >= 1


class TestDatabasePerformance:
    """Tests für Datenbank-Performance"""
    
    @pytest.mark.performance
    async def test_bulk_stem_insertion(self, db_manager: DatabaseManager):
        """Test: Bulk-Insertion von Stems"""
        import time
        
        # 100 Stems erstellen
        stems_data = []
        for i in range(100):
            stems_data.append({
                "name": f"Bulk Stem {i}",
                "file_path": f"/path/to/stem_{i}.wav",
                "type": "kick" if i % 3 == 0 else "bass" if i % 3 == 1 else "synth",
                "genre": "techno" if i % 2 == 0 else "house",
                "tempo": 120.0 + (i % 20),
                "embeddings": [float(i % 100) / 100.0] * 512
            })
        
        start_time = time.time()
        
        # Stems einzeln einfügen
        stem_ids = []
        for stem_data in stems_data:
            stem_id = await db_manager.create_stem(stem_data)
            stem_ids.append(stem_id)
        
        end_time = time.time()
        insertion_time = end_time - start_time
        
        assert len(stem_ids) == 100
        assert insertion_time < 10.0  # Sollte unter 10 Sekunden dauern
        
        # Alle Stems sollten abrufbar sein
        for stem_id in stem_ids[:5]:  # Nur erste 5 testen
            stem = await db_manager.get_stem(stem_id)
            assert stem is not None
    
    @pytest.mark.performance
    async def test_similarity_search_performance(self, db_manager: DatabaseManager):
        """Test: Performance der Ähnlichkeitssuche"""
        import time
        
        # 50 Stems mit verschiedenen Embeddings erstellen
        for i in range(50):
            stem_data = {
                "name": f"Search Test {i}",
                "file_path": f"/path/to/search_{i}.wav",
                "type": "kick",
                "embeddings": [float(j % 100) / 100.0 for j in range(i, i + 512)]
            }
            await db_manager.create_stem(stem_data)
        
        # Ähnlichkeitssuche durchführen
        target_embeddings = [0.5] * 512
        
        start_time = time.time()
        
        similar_stems = await db_manager.get_similar_stems(
            embeddings=target_embeddings,
            limit=10,
            threshold=0.0
        )
        
        end_time = time.time()
        search_time = end_time - start_time
        
        assert len(similar_stems) <= 10
        assert search_time < 2.0  # Sollte unter 2 Sekunden dauern


class TestDatabaseMigrations:
    """Tests für Datenbank-Migrationen"""
    
    @pytest.mark.unit
    async def test_schema_version(self, db_manager: DatabaseManager):
        """Test: Schema-Version prüfen"""
        version = await db_manager.get_schema_version()
        assert version is not None
        assert isinstance(version, str)
    
    @pytest.mark.unit
    async def test_migration_status(self, db_manager: DatabaseManager):
        """Test: Migrations-Status"""
        status = await db_manager.get_migration_status()
        assert "current_version" in status
        assert "pending_migrations" in status
        assert isinstance(status["pending_migrations"], list)


class TestDatabaseBackup:
    """Tests für Datenbank-Backup"""
    
    @pytest.mark.unit
    async def test_create_backup(self, db_manager: DatabaseManager, temp_dir: Path):
        """Test: Datenbank-Backup erstellen"""
        # Einige Test-Daten erstellen
        stem_data = {
            "name": "Backup Test",
            "file_path": "/path/to/backup.wav",
            "type": "kick",
            "embeddings": [0.1] * 512
        }
        
        await db_manager.create_stem(stem_data)
        
        # Backup erstellen
        backup_path = temp_dir / "backup.db"
        
        success = await db_manager.create_backup(str(backup_path))
        assert success == True
        assert backup_path.exists()
        
        # Backup-Datei sollte gültige SQLite-Datenbank sein
        conn = sqlite3.connect(str(backup_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert "stems" in tables
    
    @pytest.mark.unit
    async def test_restore_backup(self, db_manager: DatabaseManager, temp_dir: Path):
        """Test: Datenbank-Backup wiederherstellen"""
        # Backup-Datei erstellen (vereinfacht)
        backup_path = temp_dir / "restore_test.db"
        
        # Einfache SQLite-Datenbank erstellen
        conn = sqlite3.connect(str(backup_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE stems (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO stems (id, name, type) 
            VALUES ('test_id', 'Restored Stem', 'kick')
        """)
        conn.commit()
        conn.close()
        
        # Backup wiederherstellen
        success = await db_manager.restore_backup(str(backup_path))
        assert success == True
        
        # Wiederhergestellte Daten prüfen
        stem = await db_manager.get_stem("test_id")
        assert stem is not None
        assert stem["name"] == "Restored Stem"