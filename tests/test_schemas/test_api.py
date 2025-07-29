"""Tests für API-Schemas"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock
from enum import Enum

from schemas.api import (
    APIResponse, APIError, APISuccess, APIPagination, APIFilter,
    HealthCheck, HealthStatus, SystemInfo, ServiceStatus,
    AnalysisRequest, AnalysisResponse, SimilarityRequest, SimilarityResponse,
    UploadRequest, UploadResponse, DownloadRequest, DownloadResponse,
    SearchRequest
)
from schemas.websocket import (
    WebSocketMessage, NotificationMessage
)
from exceptions import ValidationError


class TestAPIResponse:
    """Tests für APIResponse-Schema"""
    
    @pytest.mark.unit
    def test_api_response_success(self):
        """Test: Erfolgreiche APIResponse"""
        response_data = {
            "success": True,
            "message": "Operation completed successfully",
            "data": {"id": "123", "name": "Test Item"},
            "timestamp": datetime.now(),
            "request_id": "req_abc123"
        }
        
        response = APIResponse(**response_data)
        
        assert response.success == True
        assert response.message == "Operation completed successfully"
        assert response.data["id"] == "123"
        assert isinstance(response.timestamp, datetime)
        assert response.request_id == "req_abc123"
    
    @pytest.mark.unit
    def test_api_response_error(self):
        """Test: Fehler-APIResponse"""
        response_data = {
            "success": False,
            "message": "Validation failed",
            "error": {
                "code": "VALIDATION_ERROR",
                "details": "Invalid input parameters",
                "field": "email"
            },
            "timestamp": datetime.now()
        }
        
        response = APIResponse(**response_data)
        
        assert response.success == False
        assert response.error["code"] == "VALIDATION_ERROR"
        assert response.error["field"] == "email"
    
    @pytest.mark.unit
    def test_api_response_with_metadata(self):
        """Test: APIResponse mit Metadaten"""
        response_data = {
            "success": True,
            "data": ["item1", "item2", "item3"],
            "metadata": {
                "total_count": 150,
                "page": 1,
                "page_size": 20,
                "has_next": True,
                "processing_time": 0.045
            }
        }
        
        response = APIResponse(**response_data)
        
        assert response.metadata["total_count"] == 150
        assert response.metadata["has_next"] == True
        assert response.metadata["processing_time"] == 0.045


class TestAPIError:
    """Tests für APIError-Schema"""
    
    @pytest.mark.unit
    def test_api_error_basic(self):
        """Test: Grundlegende APIError"""
        error_data = {
            "code": "NOT_FOUND",
            "message": "Resource not found",
            "status_code": 404,
            "timestamp": datetime.now()
        }
        
        error = APIError(**error_data)
        
        assert error.code == "NOT_FOUND"
        assert error.message == "Resource not found"
        assert error.status_code == 404
        assert isinstance(error.timestamp, datetime)
    
    @pytest.mark.unit
    def test_api_error_with_details(self):
        """Test: APIError mit Details"""
        error_data = {
            "code": "VALIDATION_ERROR",
            "message": "Input validation failed",
            "status_code": 400,
            "details": {
                "field_errors": {
                    "email": "Invalid email format",
                    "age": "Must be between 18 and 120"
                },
                "validation_rules": ["email_format", "age_range"]
            },
            "suggestion": "Please check the input format and try again",
            "documentation_url": "https://docs.example.com/validation"
        }
        
        error = APIError(**error_data)
        
        assert error.details["field_errors"]["email"] == "Invalid email format"
        assert "email_format" in error.details["validation_rules"]
        assert error.suggestion == "Please check the input format and try again"
    
    @pytest.mark.unit
    def test_api_error_validation(self):
        """Test: APIError-Validierung"""
        # Ungültiger Status-Code
        with pytest.raises(ValidationError):
            APIError(
                code="TEST",
                message="Test",
                status_code=99  # Muss >= 100 sein
            )
        
        with pytest.raises(ValidationError):
            APIError(
                code="TEST",
                message="Test",
                status_code=600  # Muss < 600 sein
            )


class TestAPISuccess:
    """Tests für APISuccess-Schema"""
    
    @pytest.mark.unit
    def test_api_success_basic(self):
        """Test: Grundlegende APISuccess"""
        success_data = {
            "message": "Operation completed",
            "data": {"result": "success"},
            "status_code": 200
        }
        
        success = APISuccess(**success_data)
        
        assert success.message == "Operation completed"
        assert success.data["result"] == "success"
        assert success.status_code == 200
    
    @pytest.mark.unit
    def test_api_success_with_location(self):
        """Test: APISuccess mit Location (für Created-Responses)"""
        success_data = {
            "message": "Resource created",
            "data": {"id": "new_123"},
            "status_code": 201,
            "location": "/api/v1/resources/new_123"
        }
        
        success = APISuccess(**success_data)
        
        assert success.status_code == 201
        assert success.location == "/api/v1/resources/new_123"


class TestAPIPagination:
    """Tests für APIPagination-Schema"""
    
    @pytest.mark.unit
    def test_api_pagination_basic(self):
        """Test: Grundlegende APIPagination"""
        pagination_data = {
            "page": 2,
            "page_size": 20,
            "total_count": 150,
            "total_pages": 8,
            "has_next": True,
            "has_previous": True
        }
        
        pagination = APIPagination(**pagination_data)
        
        assert pagination.page == 2
        assert pagination.page_size == 20
        assert pagination.total_count == 150
        assert pagination.total_pages == 8
        assert pagination.has_next == True
        assert pagination.has_previous == True
    
    @pytest.mark.unit
    def test_api_pagination_with_urls(self):
        """Test: APIPagination mit URLs"""
        pagination_data = {
            "page": 3,
            "page_size": 10,
            "total_count": 100,
            "total_pages": 10,
            "has_next": True,
            "has_previous": True,
            "next_url": "/api/v1/items?page=4&page_size=10",
            "previous_url": "/api/v1/items?page=2&page_size=10",
            "first_url": "/api/v1/items?page=1&page_size=10",
            "last_url": "/api/v1/items?page=10&page_size=10"
        }
        
        pagination = APIPagination(**pagination_data)
        
        assert pagination.next_url == "/api/v1/items?page=4&page_size=10"
        assert pagination.previous_url == "/api/v1/items?page=2&page_size=10"
    
    @pytest.mark.unit
    def test_api_pagination_validation(self):
        """Test: APIPagination-Validierung"""
        # Ungültige Page
        with pytest.raises(ValidationError):
            APIPagination(page=0, page_size=10, total_count=100)  # Page muss >= 1
        
        # Ungültige Page-Size
        with pytest.raises(ValidationError):
            APIPagination(page=1, page_size=0, total_count=100)  # Page-Size muss > 0
        
        # Page-Size zu groß
        with pytest.raises(ValidationError):
            APIPagination(page=1, page_size=1001, total_count=100)  # Max 1000


class TestAPIFilter:
    """Tests für APIFilter-Schema"""
    
    @pytest.mark.unit
    def test_api_filter_basic(self):
        """Test: Grundlegende APIFilter"""
        filter_data = {
            "field": "genre",
            "operator": "eq",
            "value": "techno"
        }
        
        filter_obj = APIFilter(**filter_data)
        
        assert filter_obj.field == "genre"
        assert filter_obj.operator == "eq"
        assert filter_obj.value == "techno"
    
    @pytest.mark.unit
    def test_api_filter_complex(self):
        """Test: Komplexe APIFilter"""
        filter_data = {
            "field": "bpm",
            "operator": "between",
            "value": [120, 140],
            "case_sensitive": False
        }
        
        filter_obj = APIFilter(**filter_data)
        
        assert filter_obj.field == "bpm"
        assert filter_obj.operator == "between"
        assert filter_obj.value == [120, 140]
        assert filter_obj.case_sensitive == False
    
    @pytest.mark.unit
    def test_api_filter_validation(self):
        """Test: APIFilter-Validierung"""
        # Ungültiger Operator
        with pytest.raises(ValidationError):
            APIFilter(
                field="test",
                operator="invalid_op",
                value="test"
            )
        
        # Leeres Field
        with pytest.raises(ValidationError):
            APIFilter(
                field="",
                operator="eq",
                value="test"
            )


class TestHealthCheck:
    """Tests für HealthCheck-Schema"""
    
    @pytest.mark.unit
    def test_health_check_healthy(self):
        """Test: Gesunder HealthCheck"""
        health_data = {
            "status": HealthStatus.HEALTHY,
            "timestamp": datetime.now(),
            "version": "2.0.0",
            "uptime": 3600.5,
            "services": {
                "database": {"status": "healthy", "response_time": 0.05},
                "clap_model": {"status": "healthy", "response_time": 0.12},
                "audio_processor": {"status": "healthy", "response_time": 0.08}
            }
        }
        
        health = HealthCheck(**health_data)
        
        assert health.status == HealthStatus.HEALTHY
        assert health.version == "2.0.0"
        assert health.uptime == 3600.5
        assert health.services["database"]["status"] == "healthy"
    
    @pytest.mark.unit
    def test_health_check_degraded(self):
        """Test: Degradierter HealthCheck"""
        health_data = {
            "status": HealthStatus.DEGRADED,
            "timestamp": datetime.now(),
            "services": {
                "database": {"status": "healthy", "response_time": 0.05},
                "clap_model": {"status": "degraded", "response_time": 2.5, "error": "High latency"},
                "audio_processor": {"status": "healthy", "response_time": 0.08}
            },
            "warnings": ["CLAP model experiencing high latency"],
            "performance_metrics": {
                "cpu_usage": 85.2,
                "memory_usage": 78.5,
                "disk_usage": 65.0
            }
        }
        
        health = HealthCheck(**health_data)
        
        assert health.status == HealthStatus.DEGRADED
        assert "High latency" in health.services["clap_model"]["error"]
        assert len(health.warnings) == 1
        assert health.performance_metrics["cpu_usage"] == 85.2
    
    @pytest.mark.unit
    def test_health_check_unhealthy(self):
        """Test: Ungesunder HealthCheck"""
        health_data = {
            "status": HealthStatus.UNHEALTHY,
            "timestamp": datetime.now(),
            "services": {
                "database": {"status": "unhealthy", "error": "Connection timeout"},
                "clap_model": {"status": "healthy", "response_time": 0.12},
                "audio_processor": {"status": "unhealthy", "error": "Service unavailable"}
            },
            "errors": [
                "Database connection failed",
                "Audio processor service down"
            ]
        }
        
        health = HealthCheck(**health_data)
        
        assert health.status == HealthStatus.UNHEALTHY
        assert len(health.errors) == 2
        assert "Database connection failed" in health.errors


class TestSystemInfo:
    """Tests für SystemInfo-Schema"""
    
    @pytest.mark.unit
    def test_system_info_complete(self):
        """Test: Vollständige SystemInfo"""
        system_data = {
            "hostname": "neuromorphe-server-01",
            "platform": "linux",
            "python_version": "3.11.5",
            "cpu_count": 8,
            "memory_total": 16384,  # MB
            "memory_available": 8192,
            "disk_total": 1024000,  # MB
            "disk_available": 512000,
            "load_average": [1.2, 1.5, 1.8],
            "network_interfaces": {
                "eth0": {"ip": "192.168.1.100", "status": "up"},
                "lo": {"ip": "127.0.0.1", "status": "up"}
            }
        }
        
        system_info = SystemInfo(**system_data)
        
        assert system_info.hostname == "neuromorphe-server-01"
        assert system_info.cpu_count == 8
        assert system_info.memory_total == 16384
        assert len(system_info.load_average) == 3
        assert system_info.network_interfaces["eth0"]["ip"] == "192.168.1.100"


class TestAnalysisRequest:
    """Tests für AnalysisRequest-Schema"""
    
    @pytest.mark.unit
    def test_analysis_request_text(self):
        """Test: Text-Analyse-Request"""
        request_data = {
            "type": "text",
            "content": "Create a dark techno track with heavy kicks and atmospheric pads",
            "options": {
                "extract_genre": True,
                "extract_mood": True,
                "extract_instruments": True,
                "confidence_threshold": 0.7
            }
        }
        
        analysis_request = AnalysisRequest(**request_data)
        
        assert analysis_request.type == "text"
        assert "dark techno" in analysis_request.content
        assert analysis_request.options["extract_genre"] == True
        assert analysis_request.options["confidence_threshold"] == 0.7
    
    @pytest.mark.unit
    def test_analysis_request_audio(self):
        """Test: Audio-Analyse-Request"""
        request_data = {
            "type": "audio",
            "file_path": "/uploads/audio_sample.wav",
            "options": {
                "analyze_tempo": True,
                "analyze_key": True,
                "analyze_energy": True,
                "segment_duration": 30.0
            }
        }
        
        analysis_request = AnalysisRequest(**request_data)
        
        assert analysis_request.type == "audio"
        assert analysis_request.file_path == "/uploads/audio_sample.wav"
        assert analysis_request.options["segment_duration"] == 30.0
    
    @pytest.mark.unit
    def test_analysis_request_validation(self):
        """Test: AnalysisRequest-Validierung"""
        # Ungültiger Type
        with pytest.raises(ValidationError):
            AnalysisRequest(
                type="invalid_type",
                content="test"
            )
        
        # Fehlender Content für Text-Analyse
        with pytest.raises(ValidationError):
            AnalysisRequest(type="text")
        
        # Fehlender File-Path für Audio-Analyse
        with pytest.raises(ValidationError):
            AnalysisRequest(type="audio")


class TestAnalysisResponse:
    """Tests für AnalysisResponse-Schema"""
    
    @pytest.mark.unit
    def test_analysis_response_text(self):
        """Test: Text-Analyse-Response"""
        response_data = {
            "request_id": "analysis_123",
            "type": "text",
            "status": "completed",
            "results": {
                "genre": {"value": "techno", "confidence": 0.92},
                "subgenre": {"value": "dark_techno", "confidence": 0.85},
                "mood": {"value": "dark", "confidence": 0.88},
                "energy": {"value": "high", "confidence": 0.79},
                "instruments": [
                    {"name": "kick", "confidence": 0.95},
                    {"name": "bass", "confidence": 0.87},
                    {"name": "pad", "confidence": 0.82}
                ]
            },
            "processing_time": 1.25,
            "completed_at": datetime.now()
        }
        
        analysis_response = AnalysisResponse(**response_data)
        
        assert analysis_response.request_id == "analysis_123"
        assert analysis_response.results["genre"]["value"] == "techno"
        assert analysis_response.results["genre"]["confidence"] == 0.92
        assert len(analysis_response.results["instruments"]) == 3
        assert analysis_response.processing_time == 1.25
    
    @pytest.mark.unit
    def test_analysis_response_audio(self):
        """Test: Audio-Analyse-Response"""
        response_data = {
            "request_id": "analysis_456",
            "type": "audio",
            "status": "completed",
            "results": {
                "tempo": {"value": 128.5, "confidence": 0.96},
                "key": {"value": "Am", "confidence": 0.84},
                "energy_curve": [0.2, 0.4, 0.7, 0.9, 0.8, 0.6, 0.3],
                "spectral_features": {
                    "centroid": 2500.0,
                    "rolloff": 8000.0,
                    "zero_crossing_rate": 0.15
                },
                "segments": [
                    {"start": 0.0, "end": 30.0, "energy": 0.6, "tempo": 128.0},
                    {"start": 30.0, "end": 60.0, "energy": 0.8, "tempo": 128.5}
                ]
            }
        }
        
        analysis_response = AnalysisResponse(**response_data)
        
        assert analysis_response.results["tempo"]["value"] == 128.5
        assert len(analysis_response.results["energy_curve"]) == 7
        assert len(analysis_response.results["segments"]) == 2
        assert analysis_response.results["spectral_features"]["centroid"] == 2500.0


class TestSimilarityRequest:
    """Tests für SimilarityRequest-Schema"""
    
    @pytest.mark.unit
    def test_similarity_request_basic(self):
        """Test: Grundlegende SimilarityRequest"""
        request_data = {
            "query_stem_id": "stem_123",
            "limit": 10,
            "threshold": 0.7,
            "include_metadata": True
        }
        
        similarity_request = SimilarityRequest(**request_data)
        
        assert similarity_request.query_stem_id == "stem_123"
        assert similarity_request.limit == 10
        assert similarity_request.threshold == 0.7
        assert similarity_request.include_metadata == True
    
    @pytest.mark.unit
    def test_similarity_request_with_filters(self):
        """Test: SimilarityRequest mit Filtern"""
        request_data = {
            "query_stem_id": "stem_456",
            "filters": {
                "genre": ["techno", "house"],
                "bpm_range": [120, 140],
                "energy_level": {"min": 6, "max": 10}
            },
            "exclude_ids": ["stem_789", "stem_abc"],
            "sort_by": "similarity_desc"
        }
        
        similarity_request = SimilarityRequest(**request_data)
        
        assert similarity_request.filters["genre"] == ["techno", "house"]
        assert similarity_request.exclude_ids == ["stem_789", "stem_abc"]
        assert similarity_request.sort_by == "similarity_desc"
    
    @pytest.mark.unit
    def test_similarity_request_validation(self):
        """Test: SimilarityRequest-Validierung"""
        # Ungültiges Limit
        with pytest.raises(ValidationError):
            SimilarityRequest(
                query_stem_id="test",
                limit=0  # Muss > 0 sein
            )
        
        # Ungültiger Threshold
        with pytest.raises(ValidationError):
            SimilarityRequest(
                query_stem_id="test",
                threshold=1.5  # Muss <= 1.0 sein
            )


class TestSimilarityResponse:
    """Tests für SimilarityResponse-Schema"""
    
    @pytest.mark.unit
    def test_similarity_response_basic(self):
        """Test: Grundlegende SimilarityResponse"""
        response_data = {
            "query_stem_id": "stem_123",
            "results": [
                {
                    "stem_id": "stem_456",
                    "similarity_score": 0.92,
                    "distance": 0.08,
                    "metadata": {"genre": "techno", "bpm": 128}
                },
                {
                    "stem_id": "stem_789",
                    "similarity_score": 0.87,
                    "distance": 0.13,
                    "metadata": {"genre": "techno", "bpm": 130}
                }
            ],
            "total_found": 25,
            "processing_time": 0.15
        }
        
        similarity_response = SimilarityResponse(**response_data)
        
        assert similarity_response.query_stem_id == "stem_123"
        assert len(similarity_response.results) == 2
        assert similarity_response.results[0]["similarity_score"] == 0.92
        assert similarity_response.total_found == 25
        assert similarity_response.processing_time == 0.15


class TestUploadRequest:
    """Tests für UploadRequest-Schema"""
    
    @pytest.mark.unit
    def test_upload_request_basic(self):
        """Test: Grundlegende UploadRequest"""
        request_data = {
            "filename": "test_audio.wav",
            "content_type": "audio/wav",
            "file_size": 10485760,  # 10 MB
            "checksum": "abc123def456",
            "metadata": {
                "title": "Test Audio",
                "genre": "techno",
                "bpm": 128
            }
        }
        
        upload_request = UploadRequest(**request_data)
        
        assert upload_request.filename == "test_audio.wav"
        assert upload_request.content_type == "audio/wav"
        assert upload_request.file_size == 10485760
        assert upload_request.metadata["title"] == "Test Audio"
    
    @pytest.mark.unit
    def test_upload_request_validation(self):
        """Test: UploadRequest-Validierung"""
        # Ungültiger Content-Type
        with pytest.raises(ValidationError):
            UploadRequest(
                filename="test.txt",
                content_type="text/plain",  # Nur Audio erlaubt
                file_size=1000
            )
        
        # Datei zu groß
        with pytest.raises(ValidationError):
            UploadRequest(
                filename="huge.wav",
                content_type="audio/wav",
                file_size=1073741825  # > 1GB
            )


class TestUploadResponse:
    """Tests für UploadResponse-Schema"""
    
    @pytest.mark.unit
    def test_upload_response_success(self):
        """Test: Erfolgreiche UploadResponse"""
        response_data = {
            "upload_id": "upload_123",
            "stem_id": "stem_456",
            "filename": "uploaded_audio.wav",
            "file_size": 10485760,
            "status": "completed",
            "upload_url": "https://storage.example.com/stems/stem_456.wav",
            "processing_status": "analyzing",
            "uploaded_at": datetime.now()
        }
        
        upload_response = UploadResponse(**response_data)
        
        assert upload_response.upload_id == "upload_123"
        assert upload_response.stem_id == "stem_456"
        assert upload_response.status == "completed"
        assert upload_response.processing_status == "analyzing"


class TestWebSocketMessage:
    """Tests für WebSocketMessage-Schema"""
    
    @pytest.mark.unit
    def test_websocket_message_basic(self):
        """Test: Grundlegende WebSocketMessage"""
        message_data = {
            "type": "render_progress",
            "data": {
                "job_id": "job_123",
                "progress": 75.5,
                "stage": "mixing"
            },
            "timestamp": datetime.now(),
            "client_id": "client_abc"
        }
        
        ws_message = WebSocketMessage(**message_data)
        
        assert ws_message.type == "render_progress"
        assert ws_message.data["job_id"] == "job_123"
        assert ws_message.data["progress"] == 75.5
        assert ws_message.client_id == "client_abc"
    
    @pytest.mark.unit
    def test_websocket_message_event(self):
        """Test: WebSocketMessage für Events"""
        message_data = {
            "type": "system_event",
            "event": "service_status_changed",
            "data": {
                "service": "clap_model",
                "old_status": "healthy",
                "new_status": "degraded",
                "reason": "High latency detected"
            }
        }
        
        ws_message = WebSocketMessage(**message_data)
        
        assert ws_message.event == "service_status_changed"
        assert ws_message.data["service"] == "clap_model"
        assert ws_message.data["new_status"] == "degraded"


class TestRateLimitInfo:
    """Tests für RateLimitInfo-Schema"""
    
    @pytest.mark.unit
    def test_rate_limit_info_basic(self):
        """Test: Grundlegende RateLimitInfo"""
        rate_limit_data = {
            "limit": 100,
            "remaining": 75,
            "reset_time": datetime.now() + timedelta(hours=1),
            "window_size": 3600,  # 1 Stunde
            "retry_after": None
        }
        
        rate_limit = RateLimitInfo(**rate_limit_data)
        
        assert rate_limit.limit == 100
        assert rate_limit.remaining == 75
        assert rate_limit.window_size == 3600
        assert rate_limit.retry_after is None
    
    @pytest.mark.unit
    def test_rate_limit_info_exceeded(self):
        """Test: RateLimitInfo bei Überschreitung"""
        rate_limit_data = {
            "limit": 100,
            "remaining": 0,
            "reset_time": datetime.now() + timedelta(minutes=30),
            "window_size": 3600,
            "retry_after": 1800,  # 30 Minuten
            "exceeded": True
        }
        
        rate_limit = RateLimitInfo(**rate_limit_data)
        
        assert rate_limit.remaining == 0
        assert rate_limit.retry_after == 1800
        assert rate_limit.exceeded == True


class TestCacheInfo:
    """Tests für CacheInfo-Schema"""
    
    @pytest.mark.unit
    def test_cache_info_basic(self):
        """Test: Grundlegende CacheInfo"""
        cache_data = {
            "hit": True,
            "key": "stems:search:techno:128bpm",
            "ttl": 3600,
            "created_at": datetime.now() - timedelta(minutes=10),
            "expires_at": datetime.now() + timedelta(minutes=50)
        }
        
        cache_info = CacheInfo(**cache_data)
        
        assert cache_info.hit == True
        assert cache_info.key == "stems:search:techno:128bpm"
        assert cache_info.ttl == 3600
        assert isinstance(cache_info.created_at, datetime)
    
    @pytest.mark.unit
    def test_cache_info_miss(self):
        """Test: CacheInfo bei Cache-Miss"""
        cache_data = {
            "hit": False,
            "key": "arrangements:new_query",
            "reason": "key_not_found"
        }
        
        cache_info = CacheInfo(**cache_data)
        
        assert cache_info.hit == False
        assert cache_info.reason == "key_not_found"


class TestMetricsInfo:
    """Tests für MetricsInfo-Schema"""
    
    @pytest.mark.unit
    def test_metrics_info_complete(self):
        """Test: Vollständige MetricsInfo"""
        metrics_data = {
            "request_count": 1250,
            "response_time_avg": 0.125,
            "response_time_p95": 0.350,
            "response_time_p99": 0.750,
            "error_rate": 0.02,
            "cache_hit_rate": 0.85,
            "active_connections": 45,
            "queue_size": 12,
            "memory_usage": 2048,
            "cpu_usage": 65.5,
            "timestamp": datetime.now()
        }
        
        metrics = MetricsInfo(**metrics_data)
        
        assert metrics.request_count == 1250
        assert metrics.response_time_avg == 0.125
        assert metrics.error_rate == 0.02
        assert metrics.cache_hit_rate == 0.85
        assert metrics.cpu_usage == 65.5


class TestAPISchemasIntegration:
    """Integrationstests für API-Schemas"""
    
    @pytest.mark.integration
    def test_api_request_response_cycle(self):
        """Test: Vollständiger API-Request-Response-Zyklus"""
        # 1. Analysis Request
        analysis_request = AnalysisRequest(
            type="text",
            content="Create a dark techno track with heavy bass",
            options={"extract_genre": True, "extract_mood": True}
        )
        
        # 2. Analysis Response
        analysis_response = AnalysisResponse(
            request_id="analysis_123",
            type="text",
            status="completed",
            results={
                "genre": {"value": "techno", "confidence": 0.92},
                "mood": {"value": "dark", "confidence": 0.88}
            },
            processing_time=1.25
        )
        
        # 3. API Success Response
        api_response = APISuccess(
            message="Analysis completed successfully",
            data=analysis_response.dict(),
            status_code=200
        )
        
        assert api_response.status_code == 200
        assert api_response.data["results"]["genre"]["value"] == "techno"
    
    @pytest.mark.integration
    def test_error_handling_flow(self):
        """Test: Fehlerbehandlungs-Flow"""
        # 1. Ungültige Request
        try:
            invalid_request = AnalysisRequest(
                type="invalid_type",
                content="test"
            )
        except ValidationError as e:
            # 2. API Error Response
            api_error = APIError(
                code="VALIDATION_ERROR",
                message="Invalid analysis type",
                status_code=400,
                details={"field": "type", "allowed_values": ["text", "audio"]}
            )
            
            assert api_error.status_code == 400
            assert api_error.code == "VALIDATION_ERROR"
    
    @pytest.mark.performance
    def test_api_schemas_performance(self):
        """Test: Performance der API-Schemas"""
        import time
        
        # Viele API-Responses erstellen
        start_time = time.time()
        
        responses = []
        for i in range(1000):
            response_data = {
                "success": True,
                "message": f"Operation {i} completed",
                "data": {"id": f"item_{i}", "value": i * 10},
                "timestamp": datetime.now(),
                "request_id": f"req_{i}"
            }
            
            response = APIResponse(**response_data)
            responses.append(response)
        
        creation_time = time.time() - start_time
        
        assert len(responses) == 1000
        assert creation_time < 2.0  # Sollte unter 2 Sekunden dauern
        
        # Komplexe Health-Checks
        start_time = time.time()
        
        health_checks = []
        for i in range(100):
            health_data = {
                "status": HealthStatus.HEALTHY if i % 10 != 0 else HealthStatus.DEGRADED,
                "timestamp": datetime.now(),
                "version": "2.0.0",
                "uptime": 3600.0 + i,
                "services": {
                    "database": {"status": "healthy", "response_time": 0.05 + (i * 0.001)},
                    "clap_model": {"status": "healthy", "response_time": 0.12 + (i * 0.002)}
                },
                "performance_metrics": {
                    "cpu_usage": 50.0 + (i % 30),
                    "memory_usage": 60.0 + (i % 20)
                }
            }
            
            health_check = HealthCheck(**health_data)
            health_checks.append(health_check)
        
        health_creation_time = time.time() - start_time
        
        assert len(health_checks) == 100
        assert health_creation_time < 1.0  # Sollte unter 1 Sekunde dauern