"""Tests für WebSocket-Schemas"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock
from enum import Enum

from src.schemas.websocket import (
    WebSocketMessage, WebSocketEvent, WebSocketResponse,
    ConnectionMessage, DisconnectionMessage, ErrorMessage,
    RenderProgressMessage, AnalysisProgressMessage, SystemStatusMessage,
    NotificationMessage, BroadcastMessage, PrivateMessage,
    SubscriptionMessage, UnsubscriptionMessage, HeartbeatMessage,
    ClientInfo, ConnectionStatus, MessageType, EventType,
    ProgressUpdate, StatusUpdate, SystemMetrics
)
from src.core.exceptions import ValidationError


class TestWebSocketMessage:
    """Tests für WebSocketMessage-Schema"""
    
    @pytest.mark.unit
    def test_websocket_message_basic(self):
        """Test: Grundlegende WebSocketMessage"""
        message_data = {
            "id": "msg_123",
            "type": MessageType.RENDER_PROGRESS,
            "timestamp": datetime.now(),
            "data": {
                "job_id": "job_456",
                "progress": 75.5,
                "stage": "mixing"
            }
        }
        
        message = WebSocketMessage(**message_data)
        
        assert message.id == "msg_123"
        assert message.type == MessageType.RENDER_PROGRESS
        assert message.data["job_id"] == "job_456"
        assert message.data["progress"] == 75.5
        assert isinstance(message.timestamp, datetime)
    
    @pytest.mark.unit
    def test_websocket_message_with_client_info(self):
        """Test: WebSocketMessage mit Client-Info"""
        message_data = {
            "id": "msg_789",
            "type": MessageType.NOTIFICATION,
            "data": {"message": "Analysis completed"},
            "client_id": "client_abc",
            "session_id": "session_def",
            "user_id": "user_123"
        }
        
        message = WebSocketMessage(**message_data)
        
        assert message.client_id == "client_abc"
        assert message.session_id == "session_def"
        assert message.user_id == "user_123"
    
    @pytest.mark.unit
    def test_websocket_message_serialization(self):
        """Test: WebSocketMessage-Serialisierung"""
        message = WebSocketMessage(
            id="msg_serialize",
            type=MessageType.SYSTEM_STATUS,
            data={"status": "healthy", "uptime": 3600}
        )
        
        # Zu Dict
        message_dict = message.dict()
        assert message_dict["id"] == "msg_serialize"
        assert message_dict["type"] == MessageType.SYSTEM_STATUS
        
        # JSON-Serialisierung
        message_json = message.json()
        assert "msg_serialize" in message_json
        assert "SYSTEM_STATUS" in message_json
    
    @pytest.mark.unit
    def test_websocket_message_validation(self):
        """Test: WebSocketMessage-Validierung"""
        # Leere ID
        with pytest.raises(ValidationError):
            WebSocketMessage(
                id="",
                type=MessageType.NOTIFICATION,
                data={}
            )
        
        # Ungültiger Message-Type
        with pytest.raises(ValidationError):
            WebSocketMessage(
                id="test",
                type="invalid_type",
                data={}
            )


class TestWebSocketEvent:
    """Tests für WebSocketEvent-Schema"""
    
    @pytest.mark.unit
    def test_websocket_event_basic(self):
        """Test: Grundlegendes WebSocketEvent"""
        event_data = {
            "event_id": "event_123",
            "event_type": EventType.RENDER_STARTED,
            "source": "render_service",
            "timestamp": datetime.now(),
            "data": {
                "job_id": "job_456",
                "arrangement_id": "arr_789",
                "estimated_duration": 120.5
            }
        }
        
        event = WebSocketEvent(**event_data)
        
        assert event.event_id == "event_123"
        assert event.event_type == EventType.RENDER_STARTED
        assert event.source == "render_service"
        assert event.data["job_id"] == "job_456"
    
    @pytest.mark.unit
    def test_websocket_event_with_metadata(self):
        """Test: WebSocketEvent mit Metadaten"""
        event_data = {
            "event_id": "event_456",
            "event_type": EventType.ANALYSIS_COMPLETED,
            "source": "clap_service",
            "data": {"stem_id": "stem_123", "results": {"genre": "techno"}},
            "metadata": {
                "processing_time": 2.5,
                "model_version": "1.2.3",
                "confidence_score": 0.92
            },
            "tags": ["analysis", "clap", "completed"]
        }
        
        event = WebSocketEvent(**event_data)
        
        assert event.metadata["processing_time"] == 2.5
        assert event.metadata["model_version"] == "1.2.3"
        assert "analysis" in event.tags
        assert len(event.tags) == 3


class TestWebSocketResponse:
    """Tests für WebSocketResponse-Schema"""
    
    @pytest.mark.unit
    def test_websocket_response_success(self):
        """Test: Erfolgreiche WebSocketResponse"""
        response_data = {
            "request_id": "req_123",
            "success": True,
            "message": "Subscription successful",
            "data": {
                "subscription_id": "sub_456",
                "topics": ["render_progress", "system_status"]
            }
        }
        
        response = WebSocketResponse(**response_data)
        
        assert response.request_id == "req_123"
        assert response.success == True
        assert response.message == "Subscription successful"
        assert response.data["subscription_id"] == "sub_456"
    
    @pytest.mark.unit
    def test_websocket_response_error(self):
        """Test: Fehler-WebSocketResponse"""
        response_data = {
            "request_id": "req_789",
            "success": False,
            "message": "Authentication failed",
            "error": {
                "code": "AUTH_FAILED",
                "details": "Invalid token provided",
                "retry_after": 60
            }
        }
        
        response = WebSocketResponse(**response_data)
        
        assert response.success == False
        assert response.error["code"] == "AUTH_FAILED"
        assert response.error["retry_after"] == 60


class TestConnectionMessage:
    """Tests für ConnectionMessage-Schema"""
    
    @pytest.mark.unit
    def test_connection_message_basic(self):
        """Test: Grundlegende ConnectionMessage"""
        connection_data = {
            "client_id": "client_123",
            "session_id": "session_456",
            "user_id": "user_789",
            "connected_at": datetime.now(),
            "client_info": {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "ip_address": "192.168.1.100",
                "platform": "web"
            }
        }
        
        connection = ConnectionMessage(**connection_data)
        
        assert connection.client_id == "client_123"
        assert connection.session_id == "session_456"
        assert connection.user_id == "user_789"
        assert connection.client_info["platform"] == "web"
    
    @pytest.mark.unit
    def test_connection_message_with_capabilities(self):
        """Test: ConnectionMessage mit Capabilities"""
        connection_data = {
            "client_id": "client_456",
            "session_id": "session_789",
            "connected_at": datetime.now(),
            "capabilities": {
                "supports_binary": True,
                "max_message_size": 1048576,
                "compression": ["gzip", "deflate"]
            },
            "subscriptions": ["render_progress", "system_alerts"]
        }
        
        connection = ConnectionMessage(**connection_data)
        
        assert connection.capabilities["supports_binary"] == True
        assert connection.capabilities["max_message_size"] == 1048576
        assert "render_progress" in connection.subscriptions


class TestDisconnectionMessage:
    """Tests für DisconnectionMessage-Schema"""
    
    @pytest.mark.unit
    def test_disconnection_message_basic(self):
        """Test: Grundlegende DisconnectionMessage"""
        disconnection_data = {
            "client_id": "client_123",
            "session_id": "session_456",
            "disconnected_at": datetime.now(),
            "reason": "client_disconnect",
            "code": 1000
        }
        
        disconnection = DisconnectionMessage(**disconnection_data)
        
        assert disconnection.client_id == "client_123"
        assert disconnection.reason == "client_disconnect"
        assert disconnection.code == 1000
    
    @pytest.mark.unit
    def test_disconnection_message_with_stats(self):
        """Test: DisconnectionMessage mit Statistiken"""
        disconnection_data = {
            "client_id": "client_789",
            "session_id": "session_abc",
            "disconnected_at": datetime.now(),
            "reason": "timeout",
            "code": 1006,
            "session_duration": 3600.5,
            "messages_sent": 150,
            "messages_received": 75,
            "bytes_transferred": 2048576
        }
        
        disconnection = DisconnectionMessage(**disconnection_data)
        
        assert disconnection.session_duration == 3600.5
        assert disconnection.messages_sent == 150
        assert disconnection.bytes_transferred == 2048576


class TestRenderProgressMessage:
    """Tests für RenderProgressMessage-Schema"""
    
    @pytest.mark.unit
    def test_render_progress_message_basic(self):
        """Test: Grundlegende RenderProgressMessage"""
        progress_data = {
            "job_id": "job_123",
            "arrangement_id": "arr_456",
            "progress": 65.5,
            "stage": "mixing",
            "estimated_remaining": 45.2,
            "current_stem": "bass_001.wav"
        }
        
        progress = RenderProgressMessage(**progress_data)
        
        assert progress.job_id == "job_123"
        assert progress.arrangement_id == "arr_456"
        assert progress.progress == 65.5
        assert progress.stage == "mixing"
        assert progress.estimated_remaining == 45.2
    
    @pytest.mark.unit
    def test_render_progress_message_detailed(self):
        """Test: Detaillierte RenderProgressMessage"""
        progress_data = {
            "job_id": "job_789",
            "arrangement_id": "arr_abc",
            "progress": 85.0,
            "stage": "effects_processing",
            "stage_progress": 75.0,
            "total_stages": 5,
            "current_stage": 4,
            "processed_stems": 8,
            "total_stems": 12,
            "processing_stats": {
                "cpu_usage": 78.5,
                "memory_usage": 2048,
                "disk_io": 15.2
            },
            "quality_metrics": {
                "peak_level": -3.2,
                "rms_level": -18.5,
                "dynamic_range": 12.8
            }
        }
        
        progress = RenderProgressMessage(**progress_data)
        
        assert progress.stage_progress == 75.0
        assert progress.current_stage == 4
        assert progress.total_stages == 5
        assert progress.processed_stems == 8
        assert progress.processing_stats["cpu_usage"] == 78.5
        assert progress.quality_metrics["peak_level"] == -3.2
    
    @pytest.mark.unit
    def test_render_progress_validation(self):
        """Test: RenderProgressMessage-Validierung"""
        # Ungültiger Progress-Wert
        with pytest.raises(ValidationError):
            RenderProgressMessage(
                job_id="test",
                arrangement_id="test",
                progress=150.0,  # Muss <= 100 sein
                stage="test"
            )
        
        # Negativer Progress
        with pytest.raises(ValidationError):
            RenderProgressMessage(
                job_id="test",
                arrangement_id="test",
                progress=-10.0,  # Muss >= 0 sein
                stage="test"
            )


class TestAnalysisProgressMessage:
    """Tests für AnalysisProgressMessage-Schema"""
    
    @pytest.mark.unit
    def test_analysis_progress_message_basic(self):
        """Test: Grundlegende AnalysisProgressMessage"""
        progress_data = {
            "analysis_id": "analysis_123",
            "stem_id": "stem_456",
            "progress": 45.0,
            "stage": "feature_extraction",
            "model_name": "clap_v1.2.3"
        }
        
        progress = AnalysisProgressMessage(**progress_data)
        
        assert progress.analysis_id == "analysis_123"
        assert progress.stem_id == "stem_456"
        assert progress.progress == 45.0
        assert progress.stage == "feature_extraction"
        assert progress.model_name == "clap_v1.2.3"
    
    @pytest.mark.unit
    def test_analysis_progress_message_detailed(self):
        """Test: Detaillierte AnalysisProgressMessage"""
        progress_data = {
            "analysis_id": "analysis_789",
            "stem_id": "stem_abc",
            "progress": 75.5,
            "stage": "embedding_generation",
            "model_name": "clap_v1.2.3",
            "batch_progress": {
                "current_batch": 3,
                "total_batches": 4,
                "batch_size": 32
            },
            "intermediate_results": {
                "tempo_detected": 128.5,
                "key_detected": "Am",
                "energy_level": 7.2
            },
            "performance_metrics": {
                "gpu_utilization": 85.2,
                "memory_usage": 4096,
                "processing_speed": 2.5
            }
        }
        
        progress = AnalysisProgressMessage(**progress_data)
        
        assert progress.batch_progress["current_batch"] == 3
        assert progress.intermediate_results["tempo_detected"] == 128.5
        assert progress.performance_metrics["gpu_utilization"] == 85.2


class TestSystemStatusMessage:
    """Tests für SystemStatusMessage-Schema"""
    
    @pytest.mark.unit
    def test_system_status_message_basic(self):
        """Test: Grundlegende SystemStatusMessage"""
        status_data = {
            "status": "healthy",
            "timestamp": datetime.now(),
            "uptime": 86400.5,
            "version": "2.0.0",
            "services": {
                "database": {"status": "healthy", "response_time": 0.05},
                "clap_model": {"status": "healthy", "response_time": 0.12},
                "render_engine": {"status": "healthy", "response_time": 0.08}
            }
        }
        
        status = SystemStatusMessage(**status_data)
        
        assert status.status == "healthy"
        assert status.uptime == 86400.5
        assert status.version == "2.0.0"
        assert status.services["database"]["status"] == "healthy"
    
    @pytest.mark.unit
    def test_system_status_message_detailed(self):
        """Test: Detaillierte SystemStatusMessage"""
        status_data = {
            "status": "degraded",
            "timestamp": datetime.now(),
            "uptime": 3600.0,
            "version": "2.0.0",
            "services": {
                "database": {"status": "healthy", "response_time": 0.05},
                "clap_model": {"status": "degraded", "response_time": 2.5, "error": "High latency"}
            },
            "system_metrics": {
                "cpu_usage": 85.2,
                "memory_usage": 78.5,
                "disk_usage": 65.0,
                "network_io": {"in": 1024, "out": 2048},
                "active_connections": 45
            },
            "alerts": [
                {"level": "warning", "message": "High CPU usage detected", "timestamp": datetime.now()},
                {"level": "info", "message": "CLAP model latency increased", "timestamp": datetime.now()}
            ],
            "performance_trends": {
                "response_time_trend": "increasing",
                "error_rate_trend": "stable",
                "throughput_trend": "decreasing"
            }
        }
        
        status = SystemStatusMessage(**status_data)
        
        assert status.status == "degraded"
        assert status.system_metrics["cpu_usage"] == 85.2
        assert len(status.alerts) == 2
        assert status.alerts[0]["level"] == "warning"
        assert status.performance_trends["response_time_trend"] == "increasing"


class TestNotificationMessage:
    """Tests für NotificationMessage-Schema"""
    
    @pytest.mark.unit
    def test_notification_message_basic(self):
        """Test: Grundlegende NotificationMessage"""
        notification_data = {
            "notification_id": "notif_123",
            "type": "info",
            "title": "Analysis Complete",
            "message": "Your audio analysis has been completed successfully",
            "timestamp": datetime.now()
        }
        
        notification = NotificationMessage(**notification_data)
        
        assert notification.notification_id == "notif_123"
        assert notification.type == "info"
        assert notification.title == "Analysis Complete"
        assert "successfully" in notification.message
    
    @pytest.mark.unit
    def test_notification_message_with_actions(self):
        """Test: NotificationMessage mit Aktionen"""
        notification_data = {
            "notification_id": "notif_456",
            "type": "warning",
            "title": "Render Queue Full",
            "message": "The render queue is at capacity. Consider upgrading your plan.",
            "priority": "high",
            "category": "system",
            "actions": [
                {"id": "upgrade", "label": "Upgrade Plan", "url": "/upgrade"},
                {"id": "dismiss", "label": "Dismiss", "action": "dismiss"}
            ],
            "metadata": {
                "queue_size": 50,
                "max_queue_size": 50,
                "estimated_wait_time": 300
            },
            "expires_at": datetime.now() + timedelta(hours=24)
        }
        
        notification = NotificationMessage(**notification_data)
        
        assert notification.priority == "high"
        assert notification.category == "system"
        assert len(notification.actions) == 2
        assert notification.actions[0]["label"] == "Upgrade Plan"
        assert notification.metadata["queue_size"] == 50


class TestBroadcastMessage:
    """Tests für BroadcastMessage-Schema"""
    
    @pytest.mark.unit
    def test_broadcast_message_basic(self):
        """Test: Grundlegende BroadcastMessage"""
        broadcast_data = {
            "broadcast_id": "broadcast_123",
            "channel": "system_announcements",
            "message": "Scheduled maintenance will begin in 30 minutes",
            "timestamp": datetime.now(),
            "sender": "system"
        }
        
        broadcast = BroadcastMessage(**broadcast_data)
        
        assert broadcast.broadcast_id == "broadcast_123"
        assert broadcast.channel == "system_announcements"
        assert "maintenance" in broadcast.message
        assert broadcast.sender == "system"
    
    @pytest.mark.unit
    def test_broadcast_message_targeted(self):
        """Test: Gezielte BroadcastMessage"""
        broadcast_data = {
            "broadcast_id": "broadcast_456",
            "channel": "premium_users",
            "message": "New premium features are now available!",
            "target_criteria": {
                "user_tier": "premium",
                "active_since": "2024-01-01"
            },
            "priority": "normal",
            "delivery_options": {
                "immediate": True,
                "persistent": True,
                "max_retries": 3
            }
        }
        
        broadcast = BroadcastMessage(**broadcast_data)
        
        assert broadcast.target_criteria["user_tier"] == "premium"
        assert broadcast.delivery_options["immediate"] == True
        assert broadcast.delivery_options["max_retries"] == 3


class TestSubscriptionMessage:
    """Tests für SubscriptionMessage-Schema"""
    
    @pytest.mark.unit
    def test_subscription_message_basic(self):
        """Test: Grundlegende SubscriptionMessage"""
        subscription_data = {
            "subscription_id": "sub_123",
            "client_id": "client_456",
            "topics": ["render_progress", "system_status"],
            "created_at": datetime.now()
        }
        
        subscription = SubscriptionMessage(**subscription_data)
        
        assert subscription.subscription_id == "sub_123"
        assert subscription.client_id == "client_456"
        assert "render_progress" in subscription.topics
        assert len(subscription.topics) == 2
    
    @pytest.mark.unit
    def test_subscription_message_with_filters(self):
        """Test: SubscriptionMessage mit Filtern"""
        subscription_data = {
            "subscription_id": "sub_789",
            "client_id": "client_abc",
            "topics": ["render_progress"],
            "filters": {
                "user_id": "user_123",
                "job_types": ["arrangement", "stem_analysis"]
            },
            "options": {
                "buffer_size": 100,
                "delivery_guarantee": "at_least_once",
                "compression": True
            }
        }
        
        subscription = SubscriptionMessage(**subscription_data)
        
        assert subscription.filters["user_id"] == "user_123"
        assert "arrangement" in subscription.filters["job_types"]
        assert subscription.options["buffer_size"] == 100
        assert subscription.options["compression"] == True


class TestHeartbeatMessage:
    """Tests für HeartbeatMessage-Schema"""
    
    @pytest.mark.unit
    def test_heartbeat_message_basic(self):
        """Test: Grundlegende HeartbeatMessage"""
        heartbeat_data = {
            "client_id": "client_123",
            "timestamp": datetime.now(),
            "sequence": 42
        }
        
        heartbeat = HeartbeatMessage(**heartbeat_data)
        
        assert heartbeat.client_id == "client_123"
        assert heartbeat.sequence == 42
        assert isinstance(heartbeat.timestamp, datetime)
    
    @pytest.mark.unit
    def test_heartbeat_message_with_stats(self):
        """Test: HeartbeatMessage mit Statistiken"""
        heartbeat_data = {
            "client_id": "client_456",
            "timestamp": datetime.now(),
            "sequence": 100,
            "client_stats": {
                "messages_sent": 250,
                "messages_received": 180,
                "connection_quality": "good",
                "latency": 45.2
            },
            "server_stats": {
                "active_connections": 150,
                "server_load": 65.5,
                "queue_depth": 25
            }
        }
        
        heartbeat = HeartbeatMessage(**heartbeat_data)
        
        assert heartbeat.client_stats["messages_sent"] == 250
        assert heartbeat.client_stats["connection_quality"] == "good"
        assert heartbeat.server_stats["active_connections"] == 150
        assert heartbeat.server_stats["server_load"] == 65.5


class TestClientInfo:
    """Tests für ClientInfo-Schema"""
    
    @pytest.mark.unit
    def test_client_info_basic(self):
        """Test: Grundlegende ClientInfo"""
        client_data = {
            "client_id": "client_123",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "ip_address": "192.168.1.100",
            "platform": "web",
            "connected_at": datetime.now()
        }
        
        client_info = ClientInfo(**client_data)
        
        assert client_info.client_id == "client_123"
        assert "Windows" in client_info.user_agent
        assert client_info.ip_address == "192.168.1.100"
        assert client_info.platform == "web"
    
    @pytest.mark.unit
    def test_client_info_detailed(self):
        """Test: Detaillierte ClientInfo"""
        client_data = {
            "client_id": "client_456",
            "user_agent": "NeuromorpheApp/2.0.0 (iOS 17.0)",
            "ip_address": "10.0.0.50",
            "platform": "mobile",
            "connected_at": datetime.now(),
            "device_info": {
                "device_type": "smartphone",
                "os_version": "iOS 17.0",
                "app_version": "2.0.0",
                "screen_resolution": "1170x2532"
            },
            "capabilities": {
                "websocket_version": "13",
                "compression_support": ["gzip", "deflate"],
                "max_message_size": 1048576,
                "binary_support": True
            },
            "location": {
                "country": "DE",
                "region": "Bavaria",
                "timezone": "Europe/Berlin"
            }
        }
        
        client_info = ClientInfo(**client_data)
        
        assert client_info.device_info["device_type"] == "smartphone"
        assert "gzip" in client_info.capabilities["compression_support"]
        assert client_info.location["country"] == "DE"
        assert client_info.capabilities["binary_support"] == True


class TestProgressUpdate:
    """Tests für ProgressUpdate-Schema"""
    
    @pytest.mark.unit
    def test_progress_update_basic(self):
        """Test: Grundlegendes ProgressUpdate"""
        progress_data = {
            "task_id": "task_123",
            "progress": 65.5,
            "stage": "processing",
            "timestamp": datetime.now()
        }
        
        progress = ProgressUpdate(**progress_data)
        
        assert progress.task_id == "task_123"
        assert progress.progress == 65.5
        assert progress.stage == "processing"
        assert isinstance(progress.timestamp, datetime)
    
    @pytest.mark.unit
    def test_progress_update_detailed(self):
        """Test: Detailliertes ProgressUpdate"""
        progress_data = {
            "task_id": "task_456",
            "progress": 85.0,
            "stage": "finalizing",
            "stage_progress": 70.0,
            "estimated_remaining": 30.5,
            "throughput": 2.5,
            "details": {
                "current_item": "stem_789.wav",
                "items_processed": 17,
                "total_items": 20,
                "errors_encountered": 0
            }
        }
        
        progress = ProgressUpdate(**progress_data)
        
        assert progress.stage_progress == 70.0
        assert progress.estimated_remaining == 30.5
        assert progress.throughput == 2.5
        assert progress.details["items_processed"] == 17
        assert progress.details["errors_encountered"] == 0


class TestWebSocketSchemasIntegration:
    """Integrationstests für WebSocket-Schemas"""
    
    @pytest.mark.integration
    def test_websocket_message_flow(self):
        """Test: Vollständiger WebSocket-Message-Flow"""
        # 1. Client verbindet sich
        connection = ConnectionMessage(
            client_id="client_123",
            session_id="session_456",
            connected_at=datetime.now(),
            client_info={
                "user_agent": "TestClient/1.0",
                "ip_address": "127.0.0.1",
                "platform": "test"
            }
        )
        
        # 2. Client abonniert Topics
        subscription = SubscriptionMessage(
            subscription_id="sub_789",
            client_id="client_123",
            topics=["render_progress", "system_status"]
        )
        
        # 3. Render-Progress wird gesendet
        progress = RenderProgressMessage(
            job_id="job_abc",
            arrangement_id="arr_def",
            progress=50.0,
            stage="mixing"
        )
        
        # 4. WebSocket-Message wird erstellt
        ws_message = WebSocketMessage(
            id="msg_progress",
            type=MessageType.RENDER_PROGRESS,
            data=progress.dict(),
            client_id="client_123"
        )
        
        # 5. Client trennt Verbindung
        disconnection = DisconnectionMessage(
            client_id="client_123",
            session_id="session_456",
            disconnected_at=datetime.now(),
            reason="client_disconnect",
            code=1000
        )
        
        # Validierung des Flows
        assert connection.client_id == subscription.client_id
        assert subscription.client_id == ws_message.client_id
        assert ws_message.client_id == disconnection.client_id
        assert ws_message.data["job_id"] == "job_abc"
    
    @pytest.mark.integration
    def test_notification_system_flow(self):
        """Test: Notification-System-Flow"""
        # 1. System-Event tritt auf
        system_event = WebSocketEvent(
            event_id="event_123",
            event_type=EventType.SYSTEM_ALERT,
            source="monitoring_service",
            data={
                "alert_type": "high_cpu_usage",
                "cpu_usage": 95.2,
                "threshold": 90.0
            }
        )
        
        # 2. Notification wird erstellt
        notification = NotificationMessage(
            notification_id="notif_456",
            type="warning",
            title="High CPU Usage Alert",
            message="CPU usage has exceeded 90% threshold",
            priority="high",
            category="system"
        )
        
        # 3. Broadcast wird gesendet
        broadcast = BroadcastMessage(
            broadcast_id="broadcast_789",
            channel="system_alerts",
            message="System performance alert: High CPU usage detected",
            sender="monitoring_service"
        )
        
        # Validierung
        assert system_event.data["alert_type"] == "high_cpu_usage"
        assert notification.priority == "high"
        assert "High CPU" in broadcast.message
    
    @pytest.mark.performance
    def test_websocket_schemas_performance(self):
        """Test: Performance der WebSocket-Schemas"""
        import time
        
        # Viele WebSocket-Messages erstellen
        start_time = time.time()
        
        messages = []
        for i in range(1000):
            message_data = {
                "id": f"msg_{i}",
                "type": MessageType.RENDER_PROGRESS,
                "data": {
                    "job_id": f"job_{i}",
                    "progress": (i % 100),
                    "stage": "processing"
                },
                "client_id": f"client_{i % 10}"
            }
            
            message = WebSocketMessage(**message_data)
            messages.append(message)
        
        creation_time = time.time() - start_time
        
        assert len(messages) == 1000
        assert creation_time < 2.0  # Sollte unter 2 Sekunden dauern
        
        # Serialisierung testen
        start_time = time.time()
        
        serialized = [msg.json() for msg in messages[:100]]
        
        serialization_time = time.time() - start_time
        
        assert len(serialized) == 100
        assert serialization_time < 1.0  # Sollte unter 1 Sekunde dauern