"""Tests für WebSocket-Service"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

from src.services.websocket_service import (
    WebSocketService, WebSocketConnection, WebSocketManager,
    ConnectionHandler, MessageHandler, BroadcastManager,
    SubscriptionManager, HeartbeatManager, RateLimiter,
    WebSocketMetrics, ConnectionPool
)
from src.core.config import WebSocketConfig
from src.core.exceptions import (
    WebSocketError, ConnectionError, ValidationError,
    RateLimitError, AuthenticationError
)
from src.schemas.websocket import (
    WebSocketMessage, WebSocketEvent, WebSocketResponse,
    ConnectionMessage, DisconnectionMessage, RenderProgressMessage,
    AnalysisProgressMessage, SystemStatusMessage, NotificationMessage,
    BroadcastMessage, SubscriptionMessage, HeartbeatMessage
)


class TestWebSocketService:
    """Tests für WebSocket-Service"""
    
    @pytest.fixture
    def websocket_config(self):
        """WebSocket-Konfiguration für Tests"""
        return WebSocketConfig(
            host="localhost",
            port=8765,
            max_connections=1000,
            heartbeat_interval=30,
            connection_timeout=60,
            message_timeout=10,
            max_message_size=1024 * 1024,  # 1MB
            rate_limit_requests=100,
            rate_limit_window=60,
            enable_compression=True,
            enable_authentication=True,
            cors_origins=["*"],
            ssl_enabled=False,
            ssl_cert_path=None,
            ssl_key_path=None
        )
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket-Verbindung"""
        websocket = Mock()
        websocket.send = AsyncMock()
        websocket.recv = AsyncMock()
        websocket.close = AsyncMock()
        websocket.closed = False
        websocket.remote_address = ("127.0.0.1", 12345)
        return websocket
    
    @pytest.fixture
    def websocket_service(self, websocket_config):
        """WebSocket-Service für Tests"""
        return WebSocketService(websocket_config)
    
    @pytest.mark.unit
    def test_websocket_service_initialization(self, websocket_config):
        """Test: WebSocket-Service-Initialisierung"""
        service = WebSocketService(websocket_config)
        
        assert service.config == websocket_config
        assert service.host == "localhost"
        assert service.port == 8765
        assert service.max_connections == 1000
        assert isinstance(service.connection_manager, WebSocketManager)
        assert isinstance(service.message_handler, MessageHandler)
        assert isinstance(service.broadcast_manager, BroadcastManager)
    
    @pytest.mark.unit
    def test_websocket_service_invalid_config(self):
        """Test: WebSocket-Service mit ungültiger Konfiguration"""
        invalid_config = WebSocketConfig(
            host="",  # Leer
            port=0,   # Ungültig
            max_connections=-1  # Negativ
        )
        
        with pytest.raises(ValidationError):
            WebSocketService(invalid_config)
    
    @pytest.mark.unit
    async def test_start_server(self, websocket_service):
        """Test: WebSocket-Server starten"""
        with patch('websockets.serve') as mock_serve:
            mock_server = Mock()
            mock_serve.return_value = mock_server
            
            await websocket_service.start_server()
            
            assert websocket_service.server == mock_server
            assert websocket_service.is_running == True
            mock_serve.assert_called_once()
    
    @pytest.mark.unit
    async def test_stop_server(self, websocket_service):
        """Test: WebSocket-Server stoppen"""
        # Mock laufender Server
        mock_server = Mock()
        mock_server.close = AsyncMock()
        mock_server.wait_closed = AsyncMock()
        websocket_service.server = mock_server
        websocket_service.is_running = True
        
        await websocket_service.stop_server()
        
        assert websocket_service.is_running == False
        mock_server.close.assert_called_once()
        mock_server.wait_closed.assert_called_once()
    
    @pytest.mark.unit
    async def test_handle_connection(self, websocket_service, mock_websocket):
        """Test: WebSocket-Verbindung behandeln"""
        path = "/ws"
        
        # Mock Connection Handler
        with patch.object(websocket_service.connection_manager, 'add_connection') as mock_add:
            with patch.object(websocket_service.connection_manager, 'remove_connection') as mock_remove:
                with patch.object(websocket_service, '_handle_messages') as mock_handle:
                    mock_handle.side_effect = ConnectionError("Connection closed")
                    
                    await websocket_service.handle_connection(mock_websocket, path)
                    
                    mock_add.assert_called_once()
                    mock_remove.assert_called_once()
    
    @pytest.mark.unit
    async def test_send_message(self, websocket_service, mock_websocket):
        """Test: Nachricht senden"""
        message = WebSocketMessage(
            type="notification",
            data={"title": "Test", "message": "Hello World"},
            timestamp=datetime.now()
        )
        
        await websocket_service.send_message(mock_websocket, message)
        
        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        parsed_data = json.loads(sent_data)
        
        assert parsed_data["type"] == "notification"
        assert parsed_data["data"]["title"] == "Test"
    
    @pytest.mark.unit
    async def test_send_message_to_closed_connection(self, websocket_service, mock_websocket):
        """Test: Nachricht an geschlossene Verbindung senden"""
        mock_websocket.closed = True
        
        message = WebSocketMessage(
            type="test",
            data={"test": True}
        )
        
        # Sollte keine Exception werfen
        await websocket_service.send_message(mock_websocket, message)
        
        # Send sollte nicht aufgerufen werden
        mock_websocket.send.assert_not_called()
    
    @pytest.mark.unit
    async def test_broadcast_message(self, websocket_service):
        """Test: Nachricht an alle Verbindungen senden"""
        # Mock Verbindungen
        mock_connections = [
            Mock(closed=False),
            Mock(closed=False),
            Mock(closed=True)  # Geschlossene Verbindung
        ]
        
        for conn in mock_connections:
            conn.send = AsyncMock()
        
        websocket_service.connection_manager.connections = mock_connections
        
        message = BroadcastMessage(
            type="system_status",
            data={"status": "healthy"},
            target="all"
        )
        
        await websocket_service.broadcast_message(message)
        
        # Nur offene Verbindungen sollten Nachricht erhalten
        mock_connections[0].send.assert_called_once()
        mock_connections[1].send.assert_called_once()
        mock_connections[2].send.assert_not_called()
    
    @pytest.mark.unit
    async def test_subscribe_to_events(self, websocket_service, mock_websocket):
        """Test: Events abonnieren"""
        subscription = SubscriptionMessage(
            type="subscription",
            action="subscribe",
            events=["render_progress", "analysis_complete"],
            filters={"user_id": "user_123"}
        )
        
        await websocket_service.subscribe_to_events(mock_websocket, subscription)
        
        # Verbindung sollte zu Subscription Manager hinzugefügt werden
        assert mock_websocket in websocket_service.subscription_manager.subscriptions
    
    @pytest.mark.unit
    async def test_unsubscribe_from_events(self, websocket_service, mock_websocket):
        """Test: Events abbestellen"""
        # Erst abonnieren
        subscription = SubscriptionMessage(
            type="subscription",
            action="subscribe",
            events=["render_progress"]
        )
        await websocket_service.subscribe_to_events(mock_websocket, subscription)
        
        # Dann abbestellen
        unsubscription = SubscriptionMessage(
            type="subscription",
            action="unsubscribe",
            events=["render_progress"]
        )
        
        await websocket_service.unsubscribe_from_events(mock_websocket, unsubscription)
        
        # Verbindung sollte nicht mehr abonniert sein
        assert mock_websocket not in websocket_service.subscription_manager.subscriptions


class TestWebSocketConnection:
    """Tests für WebSocket-Connection"""
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket"""
        websocket = Mock()
        websocket.remote_address = ("127.0.0.1", 12345)
        websocket.closed = False
        websocket.send = AsyncMock()
        websocket.close = AsyncMock()
        return websocket
    
    @pytest.fixture
    def websocket_connection(self, mock_websocket):
        """WebSocket-Connection für Tests"""
        return WebSocketConnection(
            websocket=mock_websocket,
            connection_id="conn_123",
            user_id="user_456",
            path="/ws"
        )
    
    @pytest.mark.unit
    def test_connection_initialization(self, websocket_connection, mock_websocket):
        """Test: WebSocket-Connection-Initialisierung"""
        assert websocket_connection.websocket == mock_websocket
        assert websocket_connection.connection_id == "conn_123"
        assert websocket_connection.user_id == "user_456"
        assert websocket_connection.path == "/ws"
        assert websocket_connection.is_alive == True
        assert isinstance(websocket_connection.connected_at, datetime)
    
    @pytest.mark.unit
    async def test_send_message(self, websocket_connection, mock_websocket):
        """Test: Nachricht über Connection senden"""
        message = {"type": "test", "data": "hello"}
        
        await websocket_connection.send(message)
        
        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        assert json.loads(sent_data) == message
    
    @pytest.mark.unit
    async def test_close_connection(self, websocket_connection, mock_websocket):
        """Test: Connection schließen"""
        await websocket_connection.close()
        
        assert websocket_connection.is_alive == False
        mock_websocket.close.assert_called_once()
    
    @pytest.mark.unit
    def test_connection_info(self, websocket_connection):
        """Test: Connection-Informationen abrufen"""
        info = websocket_connection.get_info()
        
        assert info["connection_id"] == "conn_123"
        assert info["user_id"] == "user_456"
        assert info["path"] == "/ws"
        assert info["remote_address"] == "127.0.0.1:12345"
        assert "connected_at" in info
        assert "is_alive" in info
    
    @pytest.mark.unit
    def test_connection_equality(self, mock_websocket):
        """Test: Connection-Gleichheit"""
        conn1 = WebSocketConnection(
            websocket=mock_websocket,
            connection_id="conn_123",
            user_id="user_456"
        )
        
        conn2 = WebSocketConnection(
            websocket=mock_websocket,
            connection_id="conn_123",
            user_id="user_456"
        )
        
        conn3 = WebSocketConnection(
            websocket=mock_websocket,
            connection_id="conn_789",
            user_id="user_456"
        )
        
        assert conn1 == conn2
        assert conn1 != conn3


class TestWebSocketManager:
    """Tests für WebSocket-Manager"""
    
    @pytest.fixture
    def websocket_manager(self):
        """WebSocket-Manager für Tests"""
        return WebSocketManager(max_connections=100)
    
    @pytest.fixture
    def mock_connection(self):
        """Mock WebSocket-Connection"""
        connection = Mock()
        connection.connection_id = "conn_123"
        connection.user_id = "user_456"
        connection.is_alive = True
        connection.close = AsyncMock()
        return connection
    
    @pytest.mark.unit
    async def test_add_connection(self, websocket_manager, mock_connection):
        """Test: Connection hinzufügen"""
        await websocket_manager.add_connection(mock_connection)
        
        assert mock_connection in websocket_manager.connections
        assert websocket_manager.get_connection_count() == 1
    
    @pytest.mark.unit
    async def test_remove_connection(self, websocket_manager, mock_connection):
        """Test: Connection entfernen"""
        # Erst hinzufügen
        await websocket_manager.add_connection(mock_connection)
        assert websocket_manager.get_connection_count() == 1
        
        # Dann entfernen
        await websocket_manager.remove_connection(mock_connection)
        
        assert mock_connection not in websocket_manager.connections
        assert websocket_manager.get_connection_count() == 0
    
    @pytest.mark.unit
    async def test_get_connection_by_id(self, websocket_manager, mock_connection):
        """Test: Connection per ID abrufen"""
        await websocket_manager.add_connection(mock_connection)
        
        found_connection = websocket_manager.get_connection_by_id("conn_123")
        
        assert found_connection == mock_connection
    
    @pytest.mark.unit
    async def test_get_connections_by_user(self, websocket_manager):
        """Test: Connections per User-ID abrufen"""
        # Mehrere Connections für denselben User
        connections = [
            Mock(connection_id=f"conn_{i}", user_id="user_456", is_alive=True)
            for i in range(3)
        ]
        
        for conn in connections:
            await websocket_manager.add_connection(conn)
        
        user_connections = websocket_manager.get_connections_by_user("user_456")
        
        assert len(user_connections) == 3
        assert all(conn.user_id == "user_456" for conn in user_connections)
    
    @pytest.mark.unit
    async def test_max_connections_limit(self, websocket_manager):
        """Test: Maximale Anzahl Connections"""
        # Manager auf 2 Connections begrenzen
        websocket_manager.max_connections = 2
        
        # Erste zwei Connections hinzufügen
        conn1 = Mock(connection_id="conn_1", is_alive=True)
        conn2 = Mock(connection_id="conn_2", is_alive=True)
        
        await websocket_manager.add_connection(conn1)
        await websocket_manager.add_connection(conn2)
        
        # Dritte Connection sollte fehlschlagen
        conn3 = Mock(connection_id="conn_3", is_alive=True)
        
        with pytest.raises(ConnectionError):
            await websocket_manager.add_connection(conn3)
    
    @pytest.mark.unit
    async def test_cleanup_dead_connections(self, websocket_manager):
        """Test: Tote Connections aufräumen"""
        # Lebende und tote Connections hinzufügen
        alive_conn = Mock(connection_id="alive", is_alive=True)
        dead_conn = Mock(connection_id="dead", is_alive=False)
        
        await websocket_manager.add_connection(alive_conn)
        await websocket_manager.add_connection(dead_conn)
        
        assert websocket_manager.get_connection_count() == 2
        
        # Cleanup ausführen
        cleaned_count = await websocket_manager.cleanup_dead_connections()
        
        assert cleaned_count == 1
        assert websocket_manager.get_connection_count() == 1
        assert alive_conn in websocket_manager.connections
        assert dead_conn not in websocket_manager.connections
    
    @pytest.mark.unit
    def test_get_statistics(self, websocket_manager):
        """Test: Manager-Statistiken abrufen"""
        stats = websocket_manager.get_statistics()
        
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "max_connections" in stats
        assert "connections_by_user" in stats
        
        assert stats["max_connections"] == 100


class TestMessageHandler:
    """Tests für Message-Handler"""
    
    @pytest.fixture
    def message_handler(self):
        """Message-Handler für Tests"""
        return MessageHandler()
    
    @pytest.fixture
    def mock_connection(self):
        """Mock WebSocket-Connection"""
        connection = Mock()
        connection.connection_id = "conn_123"
        connection.user_id = "user_456"
        connection.send = AsyncMock()
        return connection
    
    @pytest.mark.unit
    async def test_handle_connection_message(self, message_handler, mock_connection):
        """Test: Connection-Nachricht behandeln"""
        message_data = {
            "type": "connection",
            "data": {
                "user_id": "user_456",
                "client_info": {
                    "browser": "Chrome",
                    "version": "91.0"
                }
            }
        }
        
        response = await message_handler.handle_message(mock_connection, message_data)
        
        assert response["type"] == "connection_ack"
        assert response["data"]["connection_id"] == "conn_123"
    
    @pytest.mark.unit
    async def test_handle_subscription_message(self, message_handler, mock_connection):
        """Test: Subscription-Nachricht behandeln"""
        message_data = {
            "type": "subscription",
            "data": {
                "action": "subscribe",
                "events": ["render_progress", "analysis_complete"]
            }
        }
        
        response = await message_handler.handle_message(mock_connection, message_data)
        
        assert response["type"] == "subscription_ack"
        assert "subscribed_events" in response["data"]
    
    @pytest.mark.unit
    async def test_handle_heartbeat_message(self, message_handler, mock_connection):
        """Test: Heartbeat-Nachricht behandeln"""
        message_data = {
            "type": "heartbeat",
            "data": {
                "timestamp": datetime.now().isoformat()
            }
        }
        
        response = await message_handler.handle_message(mock_connection, message_data)
        
        assert response["type"] == "heartbeat_ack"
        assert "server_timestamp" in response["data"]
    
    @pytest.mark.unit
    async def test_handle_invalid_message(self, message_handler, mock_connection):
        """Test: Ungültige Nachricht behandeln"""
        invalid_message = {
            "type": "unknown_type",
            "data": {}
        }
        
        response = await message_handler.handle_message(mock_connection, invalid_message)
        
        assert response["type"] == "error"
        assert "Unknown message type" in response["data"]["message"]
    
    @pytest.mark.unit
    async def test_handle_malformed_message(self, message_handler, mock_connection):
        """Test: Fehlerhaft formatierte Nachricht behandeln"""
        malformed_message = {
            # Fehlt 'type' Feld
            "data": {"test": True}
        }
        
        response = await message_handler.handle_message(mock_connection, malformed_message)
        
        assert response["type"] == "error"
        assert "validation" in response["data"]["message"].lower()


class TestBroadcastManager:
    """Tests für Broadcast-Manager"""
    
    @pytest.fixture
    def broadcast_manager(self):
        """Broadcast-Manager für Tests"""
        return BroadcastManager()
    
    @pytest.fixture
    def mock_connections(self):
        """Mock WebSocket-Connections"""
        connections = []
        for i in range(5):
            conn = Mock()
            conn.connection_id = f"conn_{i}"
            conn.user_id = f"user_{i % 3}"  # 3 verschiedene User
            conn.is_alive = True
            conn.send = AsyncMock()
            connections.append(conn)
        return connections
    
    @pytest.mark.unit
    async def test_broadcast_to_all(self, broadcast_manager, mock_connections):
        """Test: Nachricht an alle senden"""
        message = {
            "type": "system_announcement",
            "data": {"message": "System maintenance in 10 minutes"}
        }
        
        await broadcast_manager.broadcast_to_all(mock_connections, message)
        
        # Alle Connections sollten Nachricht erhalten haben
        for conn in mock_connections:
            conn.send.assert_called_once()
    
    @pytest.mark.unit
    async def test_broadcast_to_user(self, broadcast_manager, mock_connections):
        """Test: Nachricht an spezifischen User senden"""
        target_user = "user_1"
        message = {
            "type": "user_notification",
            "data": {"message": "You have a new message"}
        }
        
        await broadcast_manager.broadcast_to_user(
            mock_connections, target_user, message
        )
        
        # Nur Connections von user_1 sollten Nachricht erhalten haben
        user_1_connections = [
            conn for conn in mock_connections 
            if conn.user_id == target_user
        ]
        
        for conn in user_1_connections:
            conn.send.assert_called_once()
        
        # Andere User sollten keine Nachricht erhalten haben
        other_connections = [
            conn for conn in mock_connections 
            if conn.user_id != target_user
        ]
        
        for conn in other_connections:
            conn.send.assert_not_called()
    
    @pytest.mark.unit
    async def test_broadcast_to_subscribers(self, broadcast_manager, mock_connections):
        """Test: Nachricht an Event-Abonnenten senden"""
        # Mock Subscription Manager
        subscription_manager = Mock()
        subscription_manager.get_subscribers.return_value = mock_connections[:3]
        
        event_type = "render_progress"
        message = {
            "type": event_type,
            "data": {"progress": 50, "job_id": "job_123"}
        }
        
        await broadcast_manager.broadcast_to_subscribers(
            subscription_manager, event_type, message
        )
        
        # Nur abonnierte Connections sollten Nachricht erhalten haben
        for i in range(3):
            mock_connections[i].send.assert_called_once()
        
        # Nicht abonnierte Connections sollten keine Nachricht erhalten haben
        for i in range(3, 5):
            mock_connections[i].send.assert_not_called()
    
    @pytest.mark.unit
    async def test_broadcast_with_filter(self, broadcast_manager, mock_connections):
        """Test: Nachricht mit Filter senden"""
        message = {
            "type": "filtered_message",
            "data": {"content": "Only for specific users"}
        }
        
        # Filter: Nur Connections mit gerader ID
        def connection_filter(conn):
            conn_num = int(conn.connection_id.split('_')[1])
            return conn_num % 2 == 0
        
        await broadcast_manager.broadcast_with_filter(
            mock_connections, message, connection_filter
        )
        
        # Nur Connections mit gerader ID sollten Nachricht erhalten haben
        for conn in mock_connections:
            conn_num = int(conn.connection_id.split('_')[1])
            if conn_num % 2 == 0:
                conn.send.assert_called_once()
            else:
                conn.send.assert_not_called()


class TestRateLimiter:
    """Tests für Rate-Limiter"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Rate-Limiter für Tests"""
        return RateLimiter(
            max_requests=10,
            time_window=60  # 10 Requests pro Minute
        )
    
    @pytest.mark.unit
    async def test_allow_request_within_limit(self, rate_limiter):
        """Test: Request innerhalb des Limits erlauben"""
        client_id = "client_123"
        
        # Erste 10 Requests sollten erlaubt sein
        for i in range(10):
            allowed = await rate_limiter.is_allowed(client_id)
            assert allowed == True
    
    @pytest.mark.unit
    async def test_block_request_over_limit(self, rate_limiter):
        """Test: Request über Limit blockieren"""
        client_id = "client_123"
        
        # Erste 10 Requests erlauben
        for i in range(10):
            await rate_limiter.is_allowed(client_id)
        
        # 11. Request sollte blockiert werden
        allowed = await rate_limiter.is_allowed(client_id)
        assert allowed == False
    
    @pytest.mark.unit
    async def test_reset_after_time_window(self, rate_limiter):
        """Test: Reset nach Zeitfenster"""
        client_id = "client_123"
        
        # Limit erreichen
        for i in range(10):
            await rate_limiter.is_allowed(client_id)
        
        # Sollte blockiert sein
        assert await rate_limiter.is_allowed(client_id) == False
        
        # Zeit vorspulen (Mock)
        with patch('time.time') as mock_time:
            mock_time.return_value = mock_time.return_value + 61  # 61 Sekunden später
            
            # Sollte wieder erlaubt sein
            allowed = await rate_limiter.is_allowed(client_id)
            assert allowed == True
    
    @pytest.mark.unit
    async def test_different_clients_separate_limits(self, rate_limiter):
        """Test: Verschiedene Clients haben separate Limits"""
        client_1 = "client_123"
        client_2 = "client_456"
        
        # Client 1 Limit erreichen
        for i in range(10):
            await rate_limiter.is_allowed(client_1)
        
        # Client 1 sollte blockiert sein
        assert await rate_limiter.is_allowed(client_1) == False
        
        # Client 2 sollte noch erlaubt sein
        assert await rate_limiter.is_allowed(client_2) == True
    
    @pytest.mark.unit
    def test_get_remaining_requests(self, rate_limiter):
        """Test: Verbleibende Requests abrufen"""
        client_id = "client_123"
        
        # Initial sollten 10 Requests verfügbar sein
        remaining = rate_limiter.get_remaining_requests(client_id)
        assert remaining == 10
        
        # Nach 3 Requests sollten 7 übrig sein
        for i in range(3):
            rate_limiter.is_allowed(client_id)
        
        remaining = rate_limiter.get_remaining_requests(client_id)
        assert remaining == 7


class TestWebSocketServiceIntegration:
    """Integrationstests für WebSocket-Service"""
    
    @pytest.mark.integration
    async def test_full_websocket_workflow(self):
        """Test: Vollständiger WebSocket-Workflow"""
        config = WebSocketConfig(
            host="localhost",
            port=8765,
            max_connections=100
        )
        
        service = WebSocketService(config)
        
        # Mock WebSocket
        mock_websocket = Mock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        mock_websocket.closed = False
        mock_websocket.send = AsyncMock()
        mock_websocket.recv = AsyncMock()
        
        # 1. Connection herstellen
        connection = WebSocketConnection(
            websocket=mock_websocket,
            connection_id="integration_test",
            user_id="test_user",
            path="/ws"
        )
        
        await service.connection_manager.add_connection(connection)
        assert service.connection_manager.get_connection_count() == 1
        
        # 2. Events abonnieren
        subscription = SubscriptionMessage(
            type="subscription",
            action="subscribe",
            events=["render_progress", "analysis_complete"]
        )
        
        await service.subscribe_to_events(mock_websocket, subscription)
        
        # 3. Nachricht senden
        message = WebSocketMessage(
            type="test_message",
            data={"content": "Integration test message"}
        )
        
        await service.send_message(mock_websocket, message)
        mock_websocket.send.assert_called()
        
        # 4. Broadcast senden
        broadcast = BroadcastMessage(
            type="system_announcement",
            data={"message": "System update complete"},
            target="all"
        )
        
        await service.broadcast_message(broadcast)
        
        # 5. Connection schließen
        await service.connection_manager.remove_connection(connection)
        assert service.connection_manager.get_connection_count() == 0
    
    @pytest.mark.performance
    async def test_websocket_service_performance(self):
        """Test: WebSocket-Service-Performance"""
        import time
        
        config = WebSocketConfig(
            host="localhost",
            port=8765,
            max_connections=1000
        )
        
        service = WebSocketService(config)
        
        # Performance-Test: Viele Connections hinzufügen
        connections = []
        for i in range(100):
            mock_websocket = Mock()
            mock_websocket.remote_address = ("127.0.0.1", 12345 + i)
            mock_websocket.closed = False
            mock_websocket.send = AsyncMock()
            
            connection = WebSocketConnection(
                websocket=mock_websocket,
                connection_id=f"perf_test_{i}",
                user_id=f"user_{i % 10}",  # 10 verschiedene User
                path="/ws"
            )
            connections.append(connection)
        
        start_time = time.time()
        
        # Connections parallel hinzufügen
        tasks = [
            service.connection_manager.add_connection(conn)
            for conn in connections
        ]
        await asyncio.gather(*tasks)
        
        add_time = time.time() - start_time
        
        assert service.connection_manager.get_connection_count() == 100
        # Sollte unter 1 Sekunde dauern
        assert add_time < 1.0
        
        # Performance-Test: Broadcast an alle Connections
        broadcast_message = BroadcastMessage(
            type="performance_test",
            data={"message": "Performance test broadcast"},
            target="all"
        )
        
        start_time = time.time()
        await service.broadcast_message(broadcast_message)
        broadcast_time = time.time() - start_time
        
        # Sollte unter 0.5 Sekunden dauern
        assert broadcast_time < 0.5
        
        # Alle Connections sollten Nachricht erhalten haben
        for conn in connections:
            conn.websocket.send.assert_called()