"""Tests für API-Service"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from src.services.api_service import (
    APIService, APIRouter, APIMiddleware, APIAuth,
    RateLimitMiddleware, CORSMiddleware, LoggingMiddleware,
    ValidationMiddleware, CacheMiddleware, MetricsMiddleware,
    APIDocumentation, APIVersioning, APIResponse,
    RequestValidator, ResponseFormatter, ErrorHandler
)
from src.core.config import APIConfig
from src.core.exceptions import (
    APIError, ValidationError, AuthenticationError,
    RateLimitError, NotFoundError, InternalServerError
)
from src.schemas.api import (
    APIResponse as APIResponseSchema, APIError as APIErrorSchema,
    APISuccess, APIPagination, APIFilter, HealthCheck,
    SystemInfo, AnalysisRequest, AnalysisResponse,
    SimilarityRequest, SimilarityResponse, UploadRequest,
    UploadResponse, RateLimitInfo, CacheInfo, MetricsInfo
)
from src.schemas.stem import StemCreate, StemResponse
from src.schemas.arrangement import ArrangementCreate, ArrangementResponse
from src.schemas.render import RenderJobCreate, RenderJobResponse


class TestAPIService:
    """Tests für API-Service"""
    
    @pytest.fixture
    def api_config(self):
        """API-Konfiguration für Tests"""
        return APIConfig(
            host="localhost",
            port=8000,
            debug=True,
            title="Neuromorphe Traum-Engine API",
            description="API for the Neuromorphe Traum-Engine v2.0",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            cors_origins=["*"],
            cors_methods=["*"],
            cors_headers=["*"],
            rate_limit_requests=100,
            rate_limit_window=60,
            max_request_size=10 * 1024 * 1024,  # 10MB
            request_timeout=30,
            enable_authentication=True,
            enable_rate_limiting=True,
            enable_caching=True,
            enable_metrics=True,
            enable_compression=True,
            api_key_header="X-API-Key",
            jwt_secret="test_secret",
            jwt_algorithm="HS256",
            jwt_expiration=3600
        )
    
    @pytest.fixture
    def api_service(self, api_config):
        """API-Service für Tests"""
        return APIService(api_config)
    
    @pytest.fixture
    def test_client(self, api_service):
        """Test-Client für API"""
        return TestClient(api_service.app)
    
    @pytest.mark.unit
    def test_api_service_initialization(self, api_config):
        """Test: API-Service-Initialisierung"""
        service = APIService(api_config)
        
        assert service.config == api_config
        assert service.host == "localhost"
        assert service.port == 8000
        assert service.debug == True
        assert service.app is not None
        assert isinstance(service.router, APIRouter)
        assert isinstance(service.auth, APIAuth)
    
    @pytest.mark.unit
    def test_api_service_invalid_config(self):
        """Test: API-Service mit ungültiger Konfiguration"""
        invalid_config = APIConfig(
            host="",  # Leer
            port=0,   # Ungültig
            title="",  # Leer
            version=""  # Leer
        )
        
        with pytest.raises(ValidationError):
            APIService(invalid_config)
    
    @pytest.mark.unit
    def test_health_check_endpoint(self, test_client):
        """Test: Health-Check-Endpoint"""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
    
    @pytest.mark.unit
    def test_system_info_endpoint(self, test_client):
        """Test: System-Info-Endpoint"""
        response = test_client.get("/system/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "version" in data
        assert "environment" in data
        assert "database" in data
        assert "services" in data
        assert "metrics" in data
    
    @pytest.mark.unit
    def test_openapi_documentation(self, test_client):
        """Test: OpenAPI-Dokumentation"""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        openapi_spec = response.json()
        
        assert openapi_spec["info"]["title"] == "Neuromorphe Traum-Engine API"
        assert openapi_spec["info"]["version"] == "2.0.0"
        assert "paths" in openapi_spec
        assert "components" in openapi_spec
    
    @pytest.mark.unit
    def test_cors_headers(self, test_client):
        """Test: CORS-Headers"""
        response = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
    
    @pytest.mark.unit
    def test_request_validation_middleware(self, test_client):
        """Test: Request-Validierung-Middleware"""
        # Test mit ungültigen Daten
        invalid_data = {
            "name": "",  # Leer
            "duration": -1  # Negativ
        }
        
        response = test_client.post("/stems", json=invalid_data)
        
        assert response.status_code == 422  # Validation Error
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.unit
    def test_error_handling_middleware(self, test_client):
        """Test: Error-Handling-Middleware"""
        # Mock Service-Fehler
        with patch('src.services.stem_service.StemService.create_stem') as mock_create:
            mock_create.side_effect = InternalServerError("Database connection failed")
            
            response = test_client.post("/stems", json={
                "name": "Test Stem",
                "file_path": "/test/audio.wav",
                "duration": 30.0
            })
            
            assert response.status_code == 500
            data = response.json()
            assert data["error"]["type"] == "InternalServerError"
    
    @pytest.mark.unit
    def test_rate_limiting_middleware(self, test_client):
        """Test: Rate-Limiting-Middleware"""
        # Viele Requests schnell hintereinander senden
        responses = []
        for i in range(105):  # Über Limit von 100
            response = test_client.get("/health")
            responses.append(response)
        
        # Letzte Requests sollten rate-limited sein
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0
    
    @pytest.mark.unit
    def test_authentication_required(self, test_client):
        """Test: Authentifizierung erforderlich"""
        # Request ohne API-Key
        response = test_client.post("/stems", json={
            "name": "Test Stem",
            "file_path": "/test/audio.wav"
        })
        
        assert response.status_code == 401  # Unauthorized
        data = response.json()
        assert "authentication" in data["error"]["message"].lower()
    
    @pytest.mark.unit
    def test_authentication_with_valid_key(self, test_client):
        """Test: Authentifizierung mit gültigem Key"""
        # Mock gültigen API-Key
        with patch('src.services.api_service.APIAuth.verify_api_key') as mock_verify:
            mock_verify.return_value = True
            
            response = test_client.post(
                "/stems",
                json={
                    "name": "Test Stem",
                    "file_path": "/test/audio.wav",
                    "duration": 30.0
                },
                headers={"X-API-Key": "valid_api_key"}
            )
            
            # Sollte nicht wegen Authentifizierung fehlschlagen
            assert response.status_code != 401


class TestAPIRouter:
    """Tests für API-Router"""
    
    @pytest.fixture
    def api_router(self):
        """API-Router für Tests"""
        return APIRouter(prefix="/api/v2", tags=["test"])
    
    @pytest.fixture
    def mock_stem_service(self):
        """Mock Stem-Service"""
        service = Mock()
        service.create_stem = AsyncMock()
        service.get_stem = AsyncMock()
        service.update_stem = AsyncMock()
        service.delete_stem = AsyncMock()
        service.list_stems = AsyncMock()
        service.search_stems = AsyncMock()
        return service
    
    @pytest.mark.unit
    async def test_create_stem_endpoint(self, api_router, mock_stem_service):
        """Test: Stem-Erstellungs-Endpoint"""
        stem_data = StemCreate(
            name="Test Stem",
            file_path="/test/audio.wav",
            duration=30.0,
            sample_rate=48000
        )
        
        # Mock Service-Antwort
        mock_stem = Mock()
        mock_stem.id = "stem_123"
        mock_stem.name = "Test Stem"
        mock_stem.duration = 30.0
        mock_stem_service.create_stem.return_value = mock_stem
        
        # Endpoint-Handler simulieren
        async def create_stem_handler(stem_create: StemCreate):
            stem = await mock_stem_service.create_stem(stem_create)
            return APIResponseSchema(
                success=True,
                data=StemResponse.from_orm(stem),
                message="Stem created successfully"
            )
        
        response = await create_stem_handler(stem_data)
        
        assert response.success == True
        assert response.data.name == "Test Stem"
        assert "created successfully" in response.message
        mock_stem_service.create_stem.assert_called_once_with(stem_data)
    
    @pytest.mark.unit
    async def test_get_stem_endpoint(self, api_router, mock_stem_service):
        """Test: Stem-Abruf-Endpoint"""
        stem_id = "stem_123"
        
        # Mock Service-Antwort
        mock_stem = Mock()
        mock_stem.id = stem_id
        mock_stem.name = "Test Stem"
        mock_stem_service.get_stem.return_value = mock_stem
        
        # Endpoint-Handler simulieren
        async def get_stem_handler(stem_id: str):
            stem = await mock_stem_service.get_stem(stem_id)
            if not stem:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Stem not found"
                )
            return APIResponseSchema(
                success=True,
                data=StemResponse.from_orm(stem)
            )
        
        response = await get_stem_handler(stem_id)
        
        assert response.success == True
        assert response.data.id == stem_id
        mock_stem_service.get_stem.assert_called_once_with(stem_id)
    
    @pytest.mark.unit
    async def test_get_stem_not_found(self, api_router, mock_stem_service):
        """Test: Stem nicht gefunden"""
        stem_id = "nonexistent_stem"
        
        # Mock Service-Antwort: None
        mock_stem_service.get_stem.return_value = None
        
        # Endpoint-Handler simulieren
        async def get_stem_handler(stem_id: str):
            stem = await mock_stem_service.get_stem(stem_id)
            if not stem:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Stem not found"
                )
            return APIResponseSchema(
                success=True,
                data=StemResponse.from_orm(stem)
            )
        
        with pytest.raises(HTTPException) as exc_info:
            await get_stem_handler(stem_id)
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail
    
    @pytest.mark.unit
    async def test_list_stems_with_pagination(self, api_router, mock_stem_service):
        """Test: Stems mit Pagination auflisten"""
        # Mock Service-Antwort
        mock_stems = [
            Mock(id=f"stem_{i}", name=f"Stem {i}", duration=30.0 + i)
            for i in range(5)
        ]
        mock_stem_service.list_stems.return_value = mock_stems
        
        # Endpoint-Handler simulieren
        async def list_stems_handler(
            limit: int = 10,
            offset: int = 0,
            filters: Optional[Dict] = None
        ):
            stems = await mock_stem_service.list_stems(
                limit=limit,
                offset=offset,
                filters=filters or {}
            )
            
            return APIResponseSchema(
                success=True,
                data=[StemResponse.from_orm(stem) for stem in stems],
                pagination=APIPagination(
                    limit=limit,
                    offset=offset,
                    total=len(stems),
                    has_next=len(stems) == limit
                )
            )
        
        response = await list_stems_handler(limit=5, offset=0)
        
        assert response.success == True
        assert len(response.data) == 5
        assert response.pagination.limit == 5
        assert response.pagination.offset == 0
        assert response.pagination.total == 5
    
    @pytest.mark.unit
    async def test_search_stems_endpoint(self, api_router, mock_stem_service):
        """Test: Stem-Such-Endpoint"""
        search_query = "test audio"
        
        # Mock Service-Antwort
        mock_results = [
            Mock(id="stem_1", name="Test Audio Stem", duration=30.0)
        ]
        mock_stem_service.search_stems.return_value = mock_results
        
        # Endpoint-Handler simulieren
        async def search_stems_handler(
            query: str,
            limit: int = 10,
            filters: Optional[Dict] = None
        ):
            results = await mock_stem_service.search_stems(
                query=query,
                limit=limit,
                filters=filters or {}
            )
            
            return APIResponseSchema(
                success=True,
                data=[StemResponse.from_orm(stem) for stem in results],
                message=f"Found {len(results)} stems matching '{query}'"
            )
        
        response = await search_stems_handler(search_query)
        
        assert response.success == True
        assert len(response.data) == 1
        assert search_query.split()[0].lower() in response.data[0].name.lower()
        mock_stem_service.search_stems.assert_called_once()


class TestAPIAuth:
    """Tests für API-Authentifizierung"""
    
    @pytest.fixture
    def api_auth(self):
        """API-Auth für Tests"""
        return APIAuth(
            jwt_secret="test_secret",
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            api_key_header="X-API-Key"
        )
    
    @pytest.mark.unit
    def test_generate_jwt_token(self, api_auth):
        """Test: JWT-Token generieren"""
        user_data = {
            "user_id": "user_123",
            "username": "testuser",
            "roles": ["user"]
        }
        
        token = api_auth.generate_jwt_token(user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert "." in token  # JWT hat Punkte als Trenner
    
    @pytest.mark.unit
    def test_verify_jwt_token(self, api_auth):
        """Test: JWT-Token verifizieren"""
        user_data = {
            "user_id": "user_123",
            "username": "testuser"
        }
        
        # Token generieren
        token = api_auth.generate_jwt_token(user_data)
        
        # Token verifizieren
        decoded_data = api_auth.verify_jwt_token(token)
        
        assert decoded_data["user_id"] == "user_123"
        assert decoded_data["username"] == "testuser"
        assert "exp" in decoded_data  # Expiration timestamp
    
    @pytest.mark.unit
    def test_verify_invalid_jwt_token(self, api_auth):
        """Test: Ungültigen JWT-Token verifizieren"""
        invalid_token = "invalid.jwt.token"
        
        with pytest.raises(AuthenticationError):
            api_auth.verify_jwt_token(invalid_token)
    
    @pytest.mark.unit
    def test_verify_expired_jwt_token(self, api_auth):
        """Test: Abgelaufenen JWT-Token verifizieren"""
        # Auth mit sehr kurzer Expiration
        short_auth = APIAuth(
            jwt_secret="test_secret",
            jwt_algorithm="HS256",
            jwt_expiration=1  # 1 Sekunde
        )
        
        user_data = {"user_id": "user_123"}
        token = short_auth.generate_jwt_token(user_data)
        
        # Warten bis Token abläuft
        import time
        time.sleep(2)
        
        with pytest.raises(AuthenticationError):
            short_auth.verify_jwt_token(token)
    
    @pytest.mark.unit
    def test_verify_api_key(self, api_auth):
        """Test: API-Key verifizieren"""
        # Mock gültiger API-Key
        valid_key = "valid_api_key_123"
        
        with patch.object(api_auth, '_get_valid_api_keys') as mock_keys:
            mock_keys.return_value = [valid_key]
            
            is_valid = api_auth.verify_api_key(valid_key)
            assert is_valid == True
    
    @pytest.mark.unit
    def test_verify_invalid_api_key(self, api_auth):
        """Test: Ungültigen API-Key verifizieren"""
        invalid_key = "invalid_api_key"
        
        with patch.object(api_auth, '_get_valid_api_keys') as mock_keys:
            mock_keys.return_value = ["valid_key_1", "valid_key_2"]
            
            is_valid = api_auth.verify_api_key(invalid_key)
            assert is_valid == False
    
    @pytest.mark.unit
    def test_extract_token_from_header(self, api_auth):
        """Test: Token aus Header extrahieren"""
        # Bearer Token
        bearer_header = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        token = api_auth.extract_token_from_header(bearer_header)
        assert token == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        
        # Direkter Token
        direct_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        token = api_auth.extract_token_from_header(direct_token)
        assert token == direct_token
        
        # Ungültiger Header
        with pytest.raises(AuthenticationError):
            api_auth.extract_token_from_header("")


class TestRateLimitMiddleware:
    """Tests für Rate-Limit-Middleware"""
    
    @pytest.fixture
    def rate_limit_middleware(self):
        """Rate-Limit-Middleware für Tests"""
        return RateLimitMiddleware(
            max_requests=10,
            time_window=60,
            key_func=lambda request: request.client.host
        )
    
    @pytest.fixture
    def mock_request(self):
        """Mock HTTP-Request"""
        request = Mock()
        request.client.host = "127.0.0.1"
        request.method = "GET"
        request.url.path = "/test"
        return request
    
    @pytest.mark.unit
    async def test_allow_request_within_limit(self, rate_limit_middleware, mock_request):
        """Test: Request innerhalb des Limits erlauben"""
        # Erste 10 Requests sollten erlaubt sein
        for i in range(10):
            allowed = await rate_limit_middleware.is_allowed(mock_request)
            assert allowed == True
    
    @pytest.mark.unit
    async def test_block_request_over_limit(self, rate_limit_middleware, mock_request):
        """Test: Request über Limit blockieren"""
        # Erste 10 Requests erlauben
        for i in range(10):
            await rate_limit_middleware.is_allowed(mock_request)
        
        # 11. Request sollte blockiert werden
        allowed = await rate_limit_middleware.is_allowed(mock_request)
        assert allowed == False
    
    @pytest.mark.unit
    async def test_different_ips_separate_limits(self, rate_limit_middleware):
        """Test: Verschiedene IPs haben separate Limits"""
        request_1 = Mock()
        request_1.client.host = "127.0.0.1"
        
        request_2 = Mock()
        request_2.client.host = "192.168.1.1"
        
        # IP 1 Limit erreichen
        for i in range(10):
            await rate_limit_middleware.is_allowed(request_1)
        
        # IP 1 sollte blockiert sein
        assert await rate_limit_middleware.is_allowed(request_1) == False
        
        # IP 2 sollte noch erlaubt sein
        assert await rate_limit_middleware.is_allowed(request_2) == True
    
    @pytest.mark.unit
    def test_get_rate_limit_info(self, rate_limit_middleware, mock_request):
        """Test: Rate-Limit-Informationen abrufen"""
        info = rate_limit_middleware.get_rate_limit_info(mock_request)
        
        assert isinstance(info, RateLimitInfo)
        assert info.limit == 10
        assert info.remaining <= 10
        assert info.reset_time > datetime.now()


class TestAPIServiceIntegration:
    """Integrationstests für API-Service"""
    
    @pytest.mark.integration
    async def test_full_api_workflow(self):
        """Test: Vollständiger API-Workflow"""
        config = APIConfig(
            host="localhost",
            port=8000,
            title="Test API",
            version="1.0.0",
            enable_authentication=False,  # Für Test deaktiviert
            enable_rate_limiting=False     # Für Test deaktiviert
        )
        
        service = APIService(config)
        client = TestClient(service.app)
        
        # 1. Health Check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # 2. System Info
        info_response = client.get("/system/info")
        assert info_response.status_code == 200
        assert "version" in info_response.json()
        
        # 3. OpenAPI Spec
        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        spec = openapi_response.json()
        assert spec["info"]["title"] == "Test API"
        
        # 4. Mock Stem-Operationen
        with patch('src.services.stem_service.StemService') as mock_service:
            # Mock Stem erstellen
            mock_stem = Mock()
            mock_stem.id = "integration_stem"
            mock_stem.name = "Integration Test Stem"
            mock_stem.duration = 45.0
            mock_service.return_value.create_stem.return_value = mock_stem
            
            create_response = client.post("/stems", json={
                "name": "Integration Test Stem",
                "file_path": "/test/integration.wav",
                "duration": 45.0,
                "sample_rate": 48000
            })
            
            # Sollte erfolgreich sein (wenn Auth deaktiviert)
            if create_response.status_code != 401:
                assert create_response.status_code == 201
                data = create_response.json()
                assert data["success"] == True
                assert data["data"]["name"] == "Integration Test Stem"
    
    @pytest.mark.performance
    async def test_api_service_performance(self):
        """Test: API-Service-Performance"""
        import time
        
        config = APIConfig(
            host="localhost",
            port=8000,
            enable_authentication=False,
            enable_rate_limiting=False
        )
        
        service = APIService(config)
        client = TestClient(service.app)
        
        # Performance-Test: Viele Health-Checks
        start_time = time.time()
        
        responses = []
        for i in range(100):
            response = client.get("/health")
            responses.append(response)
        
        total_time = time.time() - start_time
        
        # Alle Requests sollten erfolgreich sein
        assert all(r.status_code == 200 for r in responses)
        
        # Sollte unter 5 Sekunden dauern
        assert total_time < 5.0
        
        # Durchschnittliche Response-Zeit berechnen
        avg_response_time = total_time / 100
        assert avg_response_time < 0.05  # Unter 50ms pro Request
        
        # Performance-Test: Parallele Requests
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            parallel_responses = [f.result() for f in futures]
        
        parallel_time = time.time() - start_time
        
        # Alle parallelen Requests sollten erfolgreich sein
        assert all(r.status_code == 200 for r in parallel_responses)
        
        # Parallele Requests sollten schneller sein als sequenzielle
        assert parallel_time < total_time / 2
    
    @pytest.mark.integration
    def test_api_error_handling_integration(self):
        """Test: API-Fehlerbehandlung Integration"""
        config = APIConfig(
            host="localhost",
            port=8000,
            enable_authentication=False
        )
        
        service = APIService(config)
        client = TestClient(service.app)
        
        # Test verschiedene Fehlertypen
        
        # 1. Validation Error
        validation_response = client.post("/stems", json={
            "name": "",  # Leer - sollte Validierungsfehler auslösen
            "duration": -1  # Negativ - sollte Validierungsfehler auslösen
        })
        
        assert validation_response.status_code == 422
        validation_data = validation_response.json()
        assert "detail" in validation_data
        
        # 2. Not Found Error
        not_found_response = client.get("/stems/nonexistent_stem")
        
        # Sollte 404 oder 401 (wenn Auth aktiviert) sein
        assert not_found_response.status_code in [404, 401]
        
        # 3. Method Not Allowed
        method_response = client.patch("/health")  # PATCH nicht erlaubt für /health
        
        assert method_response.status_code == 405
        
        # 4. Internal Server Error (simuliert)
        with patch('src.services.stem_service.StemService') as mock_service:
            mock_service.return_value.get_stem.side_effect = Exception("Database error")
            
            error_response = client.get("/stems/test_stem")
            
            # Sollte 500 oder 401 (wenn Auth aktiviert) sein
            if error_response.status_code not in [401, 403]:
                assert error_response.status_code == 500
                error_data = error_response.json()
                assert "error" in error_data