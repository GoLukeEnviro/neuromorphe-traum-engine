"""Tests für WebSocket-Schemas"""

import pytest
from pydantic import ValidationError
from typing import Any

from src.schemas.schemas import (
    WebSocketMessage, WebSocketResponse, ConnectionMessage, DisconnectionMessage,
    ErrorMessage, RenderProgressMessage, AnalysisProgressMessage, SystemStatusMessage,
    NotificationMessage, BroadcastMessage, PrivateMessage, SubscriptionMessage,
    UnsubscriptionMessage, HeartbeatMessage
)


class TestWebSocketMessage:
    """Tests für WebSocketMessage-Schema"""

    @pytest.mark.unit
    def test_websocket_message_basic(self):
        """Test: Grundlegende WebSocketMessage"""
        message_data = {
            "event": "render_progress",
            "payload": {
                "job_id": "job_456",
                "progress": 75.5,
                "current_step": "mixing"
            }
        }
        
        message = WebSocketMessage(**message_data)
        
        assert message.event == "render_progress"
        assert message.payload["job_id"] == "job_456"
        assert message.payload["progress"] == 75.5


class TestWebSocketResponse:
    """Tests für WebSocketResponse-Schema"""

    @pytest.mark.unit
    def test_websocket_response_success(self):
        """Test: Erfolgreiche WebSocketResponse"""
        response_data = {
            "status": "success",
            "message": "Subscription successful",
            "data": {
                "topic": "render_progress"
            }
        }
        
        response = WebSocketResponse(**response_data)
        
        assert response.status == "success"
        assert response.message == "Subscription successful"
        assert response.data["topic"] == "render_progress"

    @pytest.mark.unit
    def test_websocket_response_error(self):
        """Test: Fehler-WebSocketResponse"""
        response_data = {
            "status": "error",
            "message": "Authentication failed",
        }
        
        response = WebSocketResponse(**response_data)
        
        assert response.status == "error"
        assert response.message == "Authentication failed"


class TestConnectionMessage:
    """Tests für ConnectionMessage-Schema"""

    @pytest.mark.unit
    def test_connection_message_basic(self):
        """Test: Grundlegende ConnectionMessage"""
        connection_data = {
            "client_id": "client_123",
            "message": "Client connected"
        }
        
        connection = ConnectionMessage(**connection_data)
        
        assert connection.client_id == "client_123"
        assert connection.message == "Client connected"


class TestRenderProgressMessage:
    """Tests für RenderProgressMessage-Schema"""

    @pytest.mark.unit
    def test_render_progress_message_basic(self):
        """Test: Grundlegende RenderProgressMessage"""
        progress_data = {
            "job_id": "job_123",
            "progress": 65.5,
            "current_step": "mixing",
        }
        
        progress = RenderProgressMessage(**progress_data)
        
        assert progress.job_id == "job_123"
        assert progress.progress == 65.5
        assert progress.current_step == "mixing"


class TestSubscriptionMessage:
    """Tests für SubscriptionMessage-Schema"""

    @pytest.mark.unit
    def test_subscription_message_basic(self):
        """Test: Grundlegende SubscriptionMessage"""
        subscription_data = {
            "client_id": "client_456",
            "topic": "render_progress"
        }
        
        subscription = SubscriptionMessage(**subscription_data)
        
        assert subscription.client_id == "client_456"
        assert subscription.topic == "render_progress"
