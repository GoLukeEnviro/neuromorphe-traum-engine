"""WebSocket-Schemata für die Neuromorphe Traum-Engine."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """WebSocket-Nachrichtentypen."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    PROGRESS_UPDATE = "progress_update"
    ERROR = "error"
    NOTIFICATION = "notification"
    COMMAND = "command"
    RESPONSE = "response"


class ClientInfo(BaseModel):
    """Schema für Client-Informationen."""
    client_id: str
    session_id: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    connected_at: datetime = datetime.now()
    last_activity: datetime = datetime.now()
    subscriptions: List[str] = []
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class WebSocketMessage(BaseModel):
    """Schema für WebSocket-Nachrichten."""
    message_id: str
    type: MessageType
    payload: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()
    client_id: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class WebSocketResponse(BaseModel):
    """Schema für WebSocket-Antworten."""
    message_id: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = datetime.now()
    
    model_config = ConfigDict(from_attributes=True)


class SubscriptionRequest(BaseModel):
    """Schema für Abonnement-Anfragen."""
    topics: List[str]
    filters: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class NotificationMessage(BaseModel):
    """Schema für Benachrichtigungen."""
    topic: str
    title: str
    message: str
    level: str = "info"  # info, warning, error, success
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()
    
    model_config = ConfigDict(from_attributes=True)


class ConnectionMessage(BaseModel):
    """Schema für Verbindungsnachrichten."""
    client_id: str
    message: str

class DisconnectionMessage(BaseModel):
    """Schema für Trennungsnachrichten."""
    client_id: str
    message: str

class ErrorMessage(BaseModel):
    """Schema für Fehlernachrichten."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class RenderProgressMessage(BaseModel):
    """Schema für Render-Fortschrittsnachrichten."""
    job_id: str
    progress: float
    current_step: str

class AnalysisProgressMessage(BaseModel):
    """Schema für Analyse-Fortschrittsnachrichten."""
    analysis_id: str
    progress: float
    current_step: str

class SystemStatusMessage(BaseModel):
    """Schema für Systemstatus-Nachrichten."""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class BroadcastMessage(BaseModel):
    """Schema für Broadcast-Nachrichten."""
    topic: str
    message: str
    data: Optional[Dict[str, Any]] = None

class PrivateMessage(BaseModel):
    """Schema für private Nachrichten."""
    recipient_client_id: str
    message: str
    data: Optional[Dict[str, Any]] = None

class SubscriptionMessage(BaseModel):
    """Schema für Abonnement-Nachrichten."""
    client_id: str
    topic: str

class UnsubscriptionMessage(BaseModel):
    """Schema für Abbestellungs-Nachrichten."""
    client_id: str
    topic: str

class HeartbeatMessage(BaseModel):
    """Schema für Heartbeat-Nachrichten."""
    client_id: str
    timestamp: datetime

class ProgressUpdate(BaseModel):
    """Schema für Fortschritts-Updates."""
    task_id: str
    task_type: str
    progress: float  # 0.0 - 1.0
    status: str
    message: Optional[str] = None
    estimated_time_remaining: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True)