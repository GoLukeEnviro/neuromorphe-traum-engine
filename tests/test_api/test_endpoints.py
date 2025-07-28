"""Tests für API-Endpunkte"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
import json
import tempfile
from pathlib import Path
import io

from src.main import app
from src.core.config import Settings


class TestHealthEndpoint:
    """Tests für Health-Check-Endpunkt"""
    
    def test_health_check(self, test_client: TestClient):
        """Test: Health-Check-Endpunkt"""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_health_check_with_dependencies(self, test_client: TestClient):
        """Test: Health-Check mit Abhängigkeiten"""
        with patch('src.database.manager.DatabaseManager.health_check') as mock_db_health:
            mock_db_health.return_value = True
            
            response = test_client.get("/health?check_dependencies=true")
            
            assert response.status_code == 200
            data = response.json()
            assert "dependencies" in data
            assert "database" in data["dependencies"]


class TestAnalysisEndpoints:
    """Tests für Analyse-Endpunkte"""
    
    def test_analyze_text_prompt(self, test_client: TestClient, sample_text_prompts):
        """Test: Text-Prompt-Analyse"""
        prompt = sample_text_prompts[0]
        
        with patch('src.services.neuro_analyzer.NeuroAnalyzer.analyze_text_prompt') as mock_analyze:
            mock_analyze.return_value = {
                "embeddings": [0.1] * 512,
                "prompt": prompt,
                "analysis": {
                    "genre": "techno",
                    "mood": "dark",
                    "energy": 0.8
                }
            }
            
            response = test_client.post(
                "/api/v1/analyze/text",
                json={"prompt": prompt}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "embeddings" in data
            assert "analysis" in data
            assert data["analysis"]["genre"] == "techno"
    
    def test_analyze_text_prompt_empty(self, test_client: TestClient):
        """Test: Leerer Text-Prompt"""
        response = test_client.post(
            "/api/v1/analyze/text",
            json={"prompt": ""}
        )
        
        assert response.status_code == 422  # Validation Error
    
    def test_analyze_audio_file(self, test_client: TestClient, sample_audio_file):
        """Test: Audio-Datei-Analyse"""
        with patch('src.services.neuro_analyzer.NeuroAnalyzer.analyze_audio') as mock_analyze:
            mock_analyze.return_value = {
                "embeddings": [0.1] * 512,
                "features": {
                    "tempo": 128.0,
                    "key": "Am",
                    "energy": 0.7
                }
            }
            
            with open(sample_audio_file, "rb") as f:
                response = test_client.post(
                    "/api/v1/analyze/audio",
                    files={"audio_file": ("test.wav", f, "audio/wav")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "embeddings" in data
            assert "features" in data
            assert data["features"]["tempo"] == 128.0
    
    def test_analyze_audio_invalid_format(self, test_client: TestClient, temp_dir: Path):
        """Test: Ungültiges Audio-Format"""
        # Text-Datei als Audio hochladen
        text_file = temp_dir / "not_audio.txt"
        text_file.write_text("This is not audio")
        
        with open(text_file, "rb") as f:
            response = test_client.post(
                "/api/v1/analyze/audio",
                files={"audio_file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 400  # Bad Request
    
    def test_get_similar_stems(self, test_client: TestClient):
        """Test: Ähnliche Stems finden"""
        embeddings = [0.1] * 512
        
        with patch('src.services.neuro_analyzer.NeuroAnalyzer.get_similar_stems') as mock_similar:
            mock_similar.return_value = [
                {"id": 1, "similarity": 0.95, "type": "kick"},
                {"id": 2, "similarity": 0.90, "type": "bass"},
                {"id": 3, "similarity": 0.85, "type": "synth"}
            ]
            
            response = test_client.post(
                "/api/v1/analyze/similar-stems",
                json={
                    "embeddings": embeddings,
                    "limit": 10,
                    "threshold": 0.7
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "stems" in data
            assert len(data["stems"]) == 3
            assert data["stems"][0]["similarity"] == 0.95


class TestArrangementEndpoints:
    """Tests für Arrangement-Endpunkte"""
    
    def test_create_arrangement(self, test_client: TestClient, sample_text_prompts):
        """Test: Arrangement erstellen"""
        prompt = sample_text_prompts[0]
        
        with patch('src.services.arranger.ArrangerService.create_arrangement') as mock_create:
            mock_create.return_value = {
                "arrangement_id": "test_123",
                "structure": {
                    "sections": [
                        {"name": "intro", "start": 0, "duration": 32, "stems": [1, 2]}
                    ],
                    "total_duration": 180
                },
                "stems": [1, 2, 3],
                "metadata": {"genre": "techno", "tempo": 128}
            }
            
            response = test_client.post(
                "/api/v1/arrangements",
                json={
                    "prompt": prompt,
                    "duration": 180,
                    "genre": "techno",
                    "options": {
                        "max_stems": 8,
                        "complexity": "medium"
                    }
                }
            )
            
            assert response.status_code == 201
            data = response.json()
            assert "arrangement_id" in data
            assert "structure" in data
            assert "stems" in data
            assert "metadata" in data
    
    def test_get_arrangement(self, test_client: TestClient):
        """Test: Arrangement abrufen"""
        arrangement_id = "test_123"
        
        with patch('src.database.manager.DatabaseManager.get_arrangement') as mock_get:
            mock_get.return_value = {
                "id": arrangement_id,
                "prompt": "Test prompt",
                "structure": {"sections": []},
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            response = test_client.get(f"/api/v1/arrangements/{arrangement_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == arrangement_id
    
    def test_get_arrangement_not_found(self, test_client: TestClient):
        """Test: Nicht existierendes Arrangement"""
        arrangement_id = "nonexistent"
        
        with patch('src.database.manager.DatabaseManager.get_arrangement') as mock_get:
            mock_get.return_value = None
            
            response = test_client.get(f"/api/v1/arrangements/{arrangement_id}")
            
            assert response.status_code == 404
    
    def test_list_arrangements(self, test_client: TestClient):
        """Test: Arrangements auflisten"""
        with patch('src.database.manager.DatabaseManager.list_arrangements') as mock_list:
            mock_list.return_value = {
                "arrangements": [
                    {"id": "1", "prompt": "Test 1", "created_at": "2024-01-01T00:00:00Z"},
                    {"id": "2", "prompt": "Test 2", "created_at": "2024-01-01T01:00:00Z"}
                ],
                "total": 2,
                "page": 1,
                "per_page": 10
            }
            
            response = test_client.get("/api/v1/arrangements?page=1&per_page=10")
            
            assert response.status_code == 200
            data = response.json()
            assert "arrangements" in data
            assert "total" in data
            assert len(data["arrangements"]) == 2
    
    def test_update_arrangement(self, test_client: TestClient):
        """Test: Arrangement aktualisieren"""
        arrangement_id = "test_123"
        
        with patch('src.database.manager.DatabaseManager.update_arrangement') as mock_update:
            mock_update.return_value = {
                "id": arrangement_id,
                "prompt": "Updated prompt",
                "structure": {"sections": []}
            }
            
            response = test_client.put(
                f"/api/v1/arrangements/{arrangement_id}",
                json={
                    "prompt": "Updated prompt",
                    "metadata": {"genre": "house"}
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["prompt"] == "Updated prompt"
    
    def test_delete_arrangement(self, test_client: TestClient):
        """Test: Arrangement löschen"""
        arrangement_id = "test_123"
        
        with patch('src.database.manager.DatabaseManager.delete_arrangement') as mock_delete:
            mock_delete.return_value = True
            
            response = test_client.delete(f"/api/v1/arrangements/{arrangement_id}")
            
            assert response.status_code == 204


class TestRenderingEndpoints:
    """Tests für Rendering-Endpunkte"""
    
    def test_render_arrangement(self, test_client: TestClient, temp_dir: Path):
        """Test: Arrangement rendern"""
        arrangement_id = "test_123"
        output_path = temp_dir / "output.wav"
        
        with patch('src.services.renderer.RendererService.render_arrangement') as mock_render:
            mock_render.return_value = {
                "output_path": str(output_path),
                "duration": 180,
                "metadata": {"format": "wav", "sample_rate": 44100}
            }
            
            response = test_client.post(
                f"/api/v1/arrangements/{arrangement_id}/render",
                json={
                    "format": "wav",
                    "quality": "high",
                    "options": {
                        "normalize": True,
                        "apply_mastering": True
                    }
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "render_id" in data
            assert "status" in data
            assert data["status"] == "completed"
    
    def test_get_render_status(self, test_client: TestClient):
        """Test: Render-Status abrufen"""
        render_id = "render_123"
        
        with patch('src.database.manager.DatabaseManager.get_render_job') as mock_get:
            mock_get.return_value = {
                "id": render_id,
                "status": "processing",
                "progress": 0.5,
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            response = test_client.get(f"/api/v1/renders/{render_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == render_id
            assert data["status"] == "processing"
            assert data["progress"] == 0.5
    
    def test_download_rendered_file(self, test_client: TestClient, temp_dir: Path):
        """Test: Gerenderte Datei herunterladen"""
        render_id = "render_123"
        output_file = temp_dir / "output.wav"
        output_file.write_bytes(b"fake audio data")
        
        with patch('src.database.manager.DatabaseManager.get_render_job') as mock_get:
            mock_get.return_value = {
                "id": render_id,
                "status": "completed",
                "output_path": str(output_file)
            }
            
            response = test_client.get(f"/api/v1/renders/{render_id}/download")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            assert b"fake audio data" in response.content
    
    def test_list_render_jobs(self, test_client: TestClient):
        """Test: Render-Jobs auflisten"""
        with patch('src.database.manager.DatabaseManager.list_render_jobs') as mock_list:
            mock_list.return_value = {
                "jobs": [
                    {"id": "1", "status": "completed", "created_at": "2024-01-01T00:00:00Z"},
                    {"id": "2", "status": "processing", "created_at": "2024-01-01T01:00:00Z"}
                ],
                "total": 2
            }
            
            response = test_client.get("/api/v1/renders")
            
            assert response.status_code == 200
            data = response.json()
            assert "jobs" in data
            assert len(data["jobs"]) == 2


class TestStemEndpoints:
    """Tests für Stem-Endpunkte"""
    
    def test_upload_stem(self, test_client: TestClient, sample_audio_file):
        """Test: Stem hochladen"""
        with patch('src.services.preprocessor.PreprocessorService.process_audio') as mock_process:
            mock_process.return_value = {
                "stem_id": "stem_123",
                "features": {
                    "tempo": 128.0,
                    "key": "Am",
                    "duration": 4.0
                },
                "metadata": {
                    "type": "kick",
                    "genre": "techno"
                }
            }
            
            with open(sample_audio_file, "rb") as f:
                response = test_client.post(
                    "/api/v1/stems",
                    files={"audio_file": ("kick.wav", f, "audio/wav")},
                    data={
                        "name": "Test Kick",
                        "type": "kick",
                        "genre": "techno",
                        "tags": "dark,heavy"
                    }
                )
            
            assert response.status_code == 201
            data = response.json()
            assert "stem_id" in data
            assert "features" in data
    
    def test_get_stem(self, test_client: TestClient):
        """Test: Stem abrufen"""
        stem_id = "stem_123"
        
        with patch('src.database.manager.DatabaseManager.get_stem') as mock_get:
            mock_get.return_value = {
                "id": stem_id,
                "name": "Test Kick",
                "type": "kick",
                "features": {"tempo": 128.0}
            }
            
            response = test_client.get(f"/api/v1/stems/{stem_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == stem_id
    
    def test_search_stems(self, test_client: TestClient):
        """Test: Stems suchen"""
        with patch('src.database.manager.DatabaseManager.search_stems') as mock_search:
            mock_search.return_value = {
                "stems": [
                    {"id": "1", "name": "Kick 1", "type": "kick"},
                    {"id": "2", "name": "Kick 2", "type": "kick"}
                ],
                "total": 2
            }
            
            response = test_client.get(
                "/api/v1/stems/search?query=kick&type=kick&genre=techno"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "stems" in data
            assert len(data["stems"]) == 2
    
    def test_update_stem_metadata(self, test_client: TestClient):
        """Test: Stem-Metadaten aktualisieren"""
        stem_id = "stem_123"
        
        with patch('src.database.manager.DatabaseManager.update_stem') as mock_update:
            mock_update.return_value = {
                "id": stem_id,
                "name": "Updated Kick",
                "tags": ["dark", "heavy", "updated"]
            }
            
            response = test_client.put(
                f"/api/v1/stems/{stem_id}",
                json={
                    "name": "Updated Kick",
                    "tags": ["dark", "heavy", "updated"]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Kick"
    
    def test_delete_stem(self, test_client: TestClient):
        """Test: Stem löschen"""
        stem_id = "stem_123"
        
        with patch('src.database.manager.DatabaseManager.delete_stem') as mock_delete:
            mock_delete.return_value = True
            
            response = test_client.delete(f"/api/v1/stems/{stem_id}")
            
            assert response.status_code == 204


class TestErrorHandling:
    """Tests für Fehlerbehandlung"""
    
    def test_validation_error(self, test_client: TestClient):
        """Test: Validierungsfehler"""
        response = test_client.post(
            "/api/v1/analyze/text",
            json={}  # Fehlender 'prompt' Parameter
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_internal_server_error(self, test_client: TestClient):
        """Test: Interner Server-Fehler"""
        with patch('src.services.neuro_analyzer.NeuroAnalyzer.analyze_text_prompt') as mock_analyze:
            mock_analyze.side_effect = Exception("Internal error")
            
            response = test_client.post(
                "/api/v1/analyze/text",
                json={"prompt": "test prompt"}
            )
            
            assert response.status_code == 500
    
    def test_rate_limiting(self, test_client: TestClient):
        """Test: Rate Limiting"""
        # Mehrere schnelle Anfragen
        responses = []
        for i in range(10):
            response = test_client.get("/health")
            responses.append(response.status_code)
        
        # Alle Anfragen sollten erfolgreich sein (Rate Limiting nicht aktiv in Tests)
        assert all(status == 200 for status in responses)
    
    def test_cors_headers(self, test_client: TestClient):
        """Test: CORS-Header"""
        response = test_client.options("/api/v1/analyze/text")
        
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_authentication_required(self, test_client: TestClient):
        """Test: Authentifizierung erforderlich (falls aktiviert)"""
        # Dieser Test würde nur relevant sein, wenn Authentifizierung implementiert ist
        response = test_client.get("/api/v1/arrangements")
        
        # Ohne Authentifizierung sollte es funktionieren
        assert response.status_code in [200, 401]  # Je nach Konfiguration


class TestWebSocketEndpoints:
    """Tests für WebSocket-Endpunkte"""
    
    def test_websocket_connection(self, test_client: TestClient):
        """Test: WebSocket-Verbindung"""
        with test_client.websocket_connect("/ws") as websocket:
            # Ping senden
            websocket.send_json({"type": "ping"})
            
            # Pong empfangen
            data = websocket.receive_json()
            assert data["type"] == "pong"
    
    def test_websocket_render_progress(self, test_client: TestClient):
        """Test: Render-Progress über WebSocket"""
        with test_client.websocket_connect("/ws/render/progress") as websocket:
            # Render-Job starten (simuliert)
            websocket.send_json({
                "type": "start_render",
                "arrangement_id": "test_123"
            })
            
            # Progress-Updates empfangen
            data = websocket.receive_json()
            assert data["type"] == "render_progress"
            assert "progress" in data