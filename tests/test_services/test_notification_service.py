"""Tests für Notification-Service"""

import pytest
import json
import time
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.services.notification_service import (
    NotificationService, NotificationManager, NotificationSender,
    NotificationQueue, NotificationProcessor, NotificationTemplate,
    NotificationChannel, NotificationPreferences, NotificationHistory,
    NotificationScheduler, NotificationBatch, NotificationFilter,
    NotificationAnalytics, NotificationMetrics, NotificationAudit,
    EmailNotification, SMSNotification, PushNotification,
    WebSocketNotification, SlackNotification, DiscordNotification,
    NotificationProvider, EmailProvider, SMSProvider, PushProvider,
    NotificationDelivery, DeliveryStatus, DeliveryAttempt,
    NotificationEvent, NotificationTrigger, NotificationRule,
    NotificationSubscription, NotificationGroup, NotificationUser,
    NotificationConfig, ChannelConfig, ProviderConfig,
    NotificationType, NotificationPriority, NotificationStatus,
    Notification, NotificationData, NotificationRecipient,
    NotificationContent, NotificationAttachment, NotificationAction,
    NotificationContext, NotificationVariable, NotificationPersonalization
)
from src.core.config import NotificationConfig as CoreNotificationConfig
from src.core.exceptions import (
    NotificationError, NotificationSendError, NotificationTemplateError,
    NotificationChannelError, NotificationProviderError,
    NotificationScheduleError, NotificationDeliveryError,
    NotificationValidationError, NotificationConfigError,
    NotificationQuotaError, NotificationRateLimitError
)
from src.schemas.notification import (
    NotificationData as NotificationSchemaData,
    NotificationConfigData, NotificationStatsData,
    NotificationEventData, NotificationDeliveryData,
    NotificationPreferencesData, NotificationTemplateData
)


class TestNotificationType(Enum):
    """Test-Notification-Typen"""
    WELCOME = "welcome"
    STEM_UPLOADED = "stem_uploaded"
    ARRANGEMENT_CREATED = "arrangement_created"
    COLLABORATION_INVITE = "collaboration_invite"
    SYSTEM_ALERT = "system_alert"
    SECURITY_WARNING = "security_warning"
    MAINTENANCE_NOTICE = "maintenance_notice"
    FEATURE_ANNOUNCEMENT = "feature_announcement"


@dataclass
class TestNotification:
    """Test-Notification"""
    id: str
    type: TestNotificationType
    recipient_id: str
    title: str
    message: str
    data: Dict[str, Any]
    channels: List[str]
    priority: str = "normal"
    scheduled_at: datetime = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at


class TestNotificationService:
    """Tests für Notification-Service"""
    
    @pytest.fixture
    def notification_config(self):
        """Notification-Konfiguration für Tests"""
        return CoreNotificationConfig(
            enabled=True,
            default_channels=["email", "push"],
            rate_limit_enabled=True,
            rate_limit_per_minute=60,
            rate_limit_per_hour=1000,
            batch_enabled=True,
            batch_size=100,
            batch_timeout=30,
            retry_enabled=True,
            max_retry_attempts=3,
            retry_delay=60,
            queue_enabled=True,
            queue_max_size=10000,
            priority_queue_enabled=True,
            scheduling_enabled=True,
            template_enabled=True,
            personalization_enabled=True,
            analytics_enabled=True,
            audit_enabled=True,
            # Email-Konfiguration
            email_enabled=True,
            email_provider="smtp",
            email_smtp_host="smtp.example.com",
            email_smtp_port=587,
            email_smtp_username="test@example.com",
            email_smtp_password="test_password",
            email_from_address="noreply@neuromorphe.com",
            email_from_name="Neuromorphe Traum Engine",
            # SMS-Konfiguration
            sms_enabled=True,
            sms_provider="twilio",
            sms_api_key="test_sms_api_key",
            sms_api_secret="test_sms_api_secret",
            sms_from_number="+1234567890",
            # Push-Konfiguration
            push_enabled=True,
            push_provider="firebase",
            push_api_key="test_push_api_key",
            push_project_id="test_project",
            # WebSocket-Konfiguration
            websocket_enabled=True,
            websocket_endpoint="ws://localhost:8000/ws",
            # Slack-Konfiguration
            slack_enabled=False,  # Für Tests deaktiviert
            slack_webhook_url="",
            # Discord-Konfiguration
            discord_enabled=False,  # Für Tests deaktiviert
            discord_webhook_url=""
        )
    
    @pytest.fixture
    def notification_service(self, notification_config):
        """Notification-Service für Tests"""
        return NotificationService(notification_config)
    
    @pytest.fixture
    def test_notification(self):
        """Test-Notification"""
        return TestNotification(
            id="notif_12345",
            type=TestNotificationType.STEM_UPLOADED,
            recipient_id="user_67890",
            title="New Stem Uploaded",
            message="A new stem 'Epic Beat' has been uploaded to your library.",
            data={
                "stem_id": "stem_abc123",
                "stem_name": "Epic Beat",
                "uploader_id": "user_11111",
                "uploader_name": "Producer123",
                "genre": "Electronic",
                "bpm": 128
            },
            channels=["email", "push"],
            priority="normal"
        )
    
    @pytest.fixture
    def test_user_preferences(self):
        """Test-Benutzer-Präferenzen"""
        return {
            "user_67890": {
                "email_enabled": True,
                "push_enabled": True,
                "sms_enabled": False,
                "websocket_enabled": True,
                "notification_types": {
                    "stem_uploaded": True,
                    "arrangement_created": True,
                    "collaboration_invite": True,
                    "system_alert": True,
                    "security_warning": True,
                    "maintenance_notice": False,
                    "feature_announcement": False
                },
                "quiet_hours": {
                    "enabled": True,
                    "start_time": "22:00",
                    "end_time": "08:00",
                    "timezone": "UTC"
                },
                "frequency_limits": {
                    "max_per_hour": 10,
                    "max_per_day": 50
                }
            }
        }
    
    @pytest.mark.unit
    def test_notification_service_initialization(self, notification_config):
        """Test: Notification-Service-Initialisierung"""
        service = NotificationService(notification_config)
        
        assert service.config == notification_config
        assert isinstance(service.notification_manager, NotificationManager)
        assert isinstance(service.notification_sender, NotificationSender)
        assert isinstance(service.notification_queue, NotificationQueue)
        assert isinstance(service.notification_processor, NotificationProcessor)
        assert isinstance(service.notification_scheduler, NotificationScheduler)
        assert isinstance(service.template_manager, NotificationTemplate)
        assert isinstance(service.analytics, NotificationAnalytics)
    
    @pytest.mark.unit
    def test_notification_service_invalid_config(self):
        """Test: Notification-Service mit ungültiger Konfiguration"""
        invalid_config = CoreNotificationConfig(
            enabled=True,
            rate_limit_per_minute=0,  # Ungültiges Rate-Limit
            batch_size=0,  # Ungültige Batch-Größe
            max_retry_attempts=-1,  # Negative Retry-Versuche
            queue_max_size=0,  # Ungültige Queue-Größe
            email_smtp_port=-1,  # Ungültiger Port
            email_from_address="invalid_email"  # Ungültige E-Mail
        )
        
        with pytest.raises(NotificationConfigError):
            NotificationService(invalid_config)
    
    @pytest.mark.unit
    def test_send_notification(self, notification_service, test_notification):
        """Test: Notification senden"""
        # Mock: Provider für verschiedene Kanäle
        with patch.object(notification_service.notification_sender, 'send_email') as mock_email, \
             patch.object(notification_service.notification_sender, 'send_push') as mock_push:
            
            mock_email.return_value = True
            mock_push.return_value = True
            
            # Notification senden
            result = notification_service.send_notification(
                notification_type=test_notification.type.value,
                recipient_id=test_notification.recipient_id,
                title=test_notification.title,
                message=test_notification.message,
                data=test_notification.data,
                channels=test_notification.channels
            )
            
            assert result.success == True
            assert result.notification_id is not None
            assert result.channels_sent == ["email", "push"]
            assert result.channels_failed == []
            
            # Provider sollten aufgerufen worden sein
            mock_email.assert_called_once()
            mock_push.assert_called_once()
    
    @pytest.mark.unit
    def test_send_notification_with_template(self, notification_service, test_notification):
        """Test: Notification mit Template senden"""
        # Template registrieren
        template_data = {
            "subject": "New Stem: {{stem_name}}",
            "body": "Hello {{recipient_name}}, a new stem '{{stem_name}}' by {{uploader_name}} has been uploaded.",
            "html_body": "<h1>New Stem: {{stem_name}}</h1><p>Hello {{recipient_name}}, a new stem <strong>{{stem_name}}</strong> by {{uploader_name}} has been uploaded.</p>"
        }
        
        notification_service.template_manager.register_template(
            template_id="stem_uploaded",
            template_data=template_data
        )
        
        # Template-Variablen
        template_variables = {
            "recipient_name": "TestUser",
            "stem_name": test_notification.data["stem_name"],
            "uploader_name": test_notification.data["uploader_name"]
        }
        
        with patch.object(notification_service.notification_sender, 'send_email') as mock_email:
            mock_email.return_value = True
            
            result = notification_service.send_notification(
                notification_type=test_notification.type.value,
                recipient_id=test_notification.recipient_id,
                template_id="stem_uploaded",
                template_variables=template_variables,
                channels=["email"]
            )
            
            assert result.success == True
            
            # E-Mail sollte mit gerenderten Template-Daten gesendet worden sein
            call_args = mock_email.call_args[1]
            assert "Epic Beat" in call_args["subject"]
            assert "TestUser" in call_args["body"]
            assert "Producer123" in call_args["body"]
    
    @pytest.mark.unit
    def test_send_notification_with_preferences(self, notification_service, test_notification, test_user_preferences):
        """Test: Notification mit Benutzer-Präferenzen senden"""
        # Mock: Benutzer-Präferenzen laden
        with patch.object(notification_service.notification_manager, 'get_user_preferences') as mock_prefs:
            mock_prefs.return_value = test_user_preferences[test_notification.recipient_id]
            
            with patch.object(notification_service.notification_sender, 'send_email') as mock_email, \
                 patch.object(notification_service.notification_sender, 'send_push') as mock_push, \
                 patch.object(notification_service.notification_sender, 'send_sms') as mock_sms:
                
                mock_email.return_value = True
                mock_push.return_value = True
                mock_sms.return_value = True
                
                result = notification_service.send_notification(
                    notification_type=test_notification.type.value,
                    recipient_id=test_notification.recipient_id,
                    title=test_notification.title,
                    message=test_notification.message,
                    channels=["email", "push", "sms"]  # SMS sollte durch Präferenzen gefiltert werden
                )
                
                assert result.success == True
                assert "email" in result.channels_sent
                assert "push" in result.channels_sent
                assert "sms" not in result.channels_sent  # SMS deaktiviert in Präferenzen
                
                # SMS sollte nicht gesendet worden sein
                mock_sms.assert_not_called()
    
    @pytest.mark.unit
    def test_send_notification_with_scheduling(self, notification_service, test_notification):
        """Test: Geplante Notification senden"""
        # Notification für die Zukunft planen
        scheduled_time = datetime.now() + timedelta(hours=1)
        
        result = notification_service.schedule_notification(
            notification_type=test_notification.type.value,
            recipient_id=test_notification.recipient_id,
            title=test_notification.title,
            message=test_notification.message,
            channels=test_notification.channels,
            scheduled_at=scheduled_time
        )
        
        assert result.success == True
        assert result.notification_id is not None
        assert result.scheduled_at == scheduled_time
        assert result.status == "scheduled"
        
        # Geplante Notifications abrufen
        scheduled_notifications = notification_service.get_scheduled_notifications(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2)
        )
        
        assert len(scheduled_notifications) >= 1
        assert any(n.notification_id == result.notification_id for n in scheduled_notifications)
    
    @pytest.mark.unit
    def test_send_batch_notifications(self, notification_service):
        """Test: Batch-Notifications senden"""
        # Mehrere Notifications erstellen
        notifications = []
        for i in range(5):
            notifications.append({
                "notification_type": "system_alert",
                "recipient_id": f"user_{i}",
                "title": f"System Alert {i}",
                "message": f"This is system alert number {i}",
                "channels": ["email"]
            })
        
        with patch.object(notification_service.notification_sender, 'send_email') as mock_email:
            mock_email.return_value = True
            
            result = notification_service.send_batch_notifications(notifications)
            
            assert result.success == True
            assert result.total_notifications == 5
            assert result.successful_notifications == 5
            assert result.failed_notifications == 0
            
            # E-Mail sollte 5 Mal gesendet worden sein
            assert mock_email.call_count == 5
    
    @pytest.mark.unit
    def test_notification_rate_limiting(self, notification_service, test_notification):
        """Test: Notification-Rate-Limiting"""
        # Rate-Limiter konfigurieren (5 Notifications pro Minute)
        notification_service.rate_limiter.set_limit(
            user_id=test_notification.recipient_id,
            limit=5,
            window=60
        )
        
        with patch.object(notification_service.notification_sender, 'send_email') as mock_email:
            mock_email.return_value = True
            
            # 5 Notifications senden (sollten alle durchgehen)
            for i in range(5):
                result = notification_service.send_notification(
                    notification_type=test_notification.type.value,
                    recipient_id=test_notification.recipient_id,
                    title=f"{test_notification.title} {i}",
                    message=test_notification.message,
                    channels=["email"]
                )
                assert result.success == True
            
            # 6. Notification sollte durch Rate-Limiting blockiert werden
            with pytest.raises(NotificationRateLimitError):
                notification_service.send_notification(
                    notification_type=test_notification.type.value,
                    recipient_id=test_notification.recipient_id,
                    title="Rate Limited Notification",
                    message=test_notification.message,
                    channels=["email"]
                )
    
    @pytest.mark.unit
    def test_notification_retry_mechanism(self, notification_service, test_notification):
        """Test: Notification-Retry-Mechanismus"""
        # Mock: E-Mail-Versand schlägt zunächst fehl, dann erfolgreich
        with patch.object(notification_service.notification_sender, 'send_email') as mock_email:
            mock_email.side_effect = [False, False, True]  # 2 Fehlschläge, dann Erfolg
            
            result = notification_service.send_notification(
                notification_type=test_notification.type.value,
                recipient_id=test_notification.recipient_id,
                title=test_notification.title,
                message=test_notification.message,
                channels=["email"],
                retry_on_failure=True
            )
            
            assert result.success == True
            assert result.retry_attempts == 2
            
            # E-Mail sollte 3 Mal versucht worden sein
            assert mock_email.call_count == 3
    
    @pytest.mark.unit
    def test_notification_delivery_tracking(self, notification_service, test_notification):
        """Test: Notification-Delivery-Tracking"""
        with patch.object(notification_service.notification_sender, 'send_email') as mock_email:
            mock_email.return_value = True
            
            result = notification_service.send_notification(
                notification_type=test_notification.type.value,
                recipient_id=test_notification.recipient_id,
                title=test_notification.title,
                message=test_notification.message,
                channels=["email"]
            )
            
            notification_id = result.notification_id
            
            # Delivery-Status abrufen
            delivery_status = notification_service.get_delivery_status(notification_id)
            
            assert delivery_status is not None
            assert delivery_status.notification_id == notification_id
            assert delivery_status.status == "delivered"
            assert delivery_status.delivered_at is not None
            
            # Delivery-Historie abrufen
            delivery_history = notification_service.get_delivery_history(notification_id)
            
            assert len(delivery_history) >= 1
            assert delivery_history[0].channel == "email"
            assert delivery_history[0].status == "delivered"
    
    @pytest.mark.unit
    def test_notification_analytics(self, notification_service, test_notification):
        """Test: Notification-Analytics"""
        # Mehrere Notifications senden
        notification_ids = []
        
        with patch.object(notification_service.notification_sender, 'send_email') as mock_email, \
             patch.object(notification_service.notification_sender, 'send_push') as mock_push:
            
            mock_email.return_value = True
            mock_push.return_value = True
            
            for i in range(10):
                result = notification_service.send_notification(
                    notification_type=test_notification.type.value,
                    recipient_id=f"user_{i}",
                    title=f"{test_notification.title} {i}",
                    message=test_notification.message,
                    channels=["email", "push"]
                )
                notification_ids.append(result.notification_id)
        
        # Analytics abrufen
        analytics = notification_service.get_notification_analytics(
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now() + timedelta(hours=1)
        )
        
        assert analytics.total_notifications >= 10
        assert analytics.successful_notifications >= 10
        assert analytics.failed_notifications == 0
        assert "email" in analytics.channel_stats
        assert "push" in analytics.channel_stats
        assert analytics.channel_stats["email"]["sent"] >= 10
        assert analytics.channel_stats["push"]["sent"] >= 10
        
        # Notification-Typen-Statistiken
        assert test_notification.type.value in analytics.type_stats
        assert analytics.type_stats[test_notification.type.value]["sent"] >= 10
    
    @pytest.mark.unit
    def test_notification_preferences_management(self, notification_service):
        """Test: Notification-Präferenzen-Management"""
        user_id = "pref_user_12345"
        
        # Präferenzen setzen
        preferences = {
            "email_enabled": True,
            "push_enabled": False,
            "sms_enabled": True,
            "notification_types": {
                "stem_uploaded": True,
                "arrangement_created": False,
                "system_alert": True
            },
            "quiet_hours": {
                "enabled": True,
                "start_time": "23:00",
                "end_time": "07:00"
            }
        }
        
        notification_service.set_user_preferences(user_id, preferences)
        
        # Präferenzen abrufen
        retrieved_prefs = notification_service.get_user_preferences(user_id)
        
        assert retrieved_prefs["email_enabled"] == True
        assert retrieved_prefs["push_enabled"] == False
        assert retrieved_prefs["sms_enabled"] == True
        assert retrieved_prefs["notification_types"]["stem_uploaded"] == True
        assert retrieved_prefs["notification_types"]["arrangement_created"] == False
        assert retrieved_prefs["quiet_hours"]["enabled"] == True
        
        # Einzelne Präferenz aktualisieren
        notification_service.update_user_preference(
            user_id, "push_enabled", True
        )
        
        updated_prefs = notification_service.get_user_preferences(user_id)
        assert updated_prefs["push_enabled"] == True
    
    @pytest.mark.unit
    def test_notification_subscription_management(self, notification_service):
        """Test: Notification-Subscription-Management"""
        user_id = "sub_user_12345"
        
        # Subscription erstellen
        subscription = notification_service.create_subscription(
            user_id=user_id,
            notification_type="stem_uploaded",
            channels=["email", "push"],
            filters={
                "genre": ["Electronic", "Hip-Hop"],
                "bpm_min": 120,
                "bpm_max": 140
            }
        )
        
        assert subscription.subscription_id is not None
        assert subscription.user_id == user_id
        assert subscription.notification_type == "stem_uploaded"
        assert subscription.is_active == True
        
        # Subscriptions abrufen
        subscriptions = notification_service.get_user_subscriptions(user_id)
        
        assert len(subscriptions) >= 1
        assert any(s.subscription_id == subscription.subscription_id for s in subscriptions)
        
        # Subscription aktualisieren
        notification_service.update_subscription(
            subscription.subscription_id,
            channels=["email"],  # Push entfernen
            filters={
                "genre": ["Electronic"],  # Hip-Hop entfernen
                "bpm_min": 125,
                "bpm_max": 135
            }
        )
        
        updated_subscription = notification_service.get_subscription(subscription.subscription_id)
        assert updated_subscription.channels == ["email"]
        assert updated_subscription.filters["genre"] == ["Electronic"]
        
        # Subscription deaktivieren
        notification_service.deactivate_subscription(subscription.subscription_id)
        
        deactivated_subscription = notification_service.get_subscription(subscription.subscription_id)
        assert deactivated_subscription.is_active == False
    
    @pytest.mark.unit
    def test_notification_templates(self, notification_service):
        """Test: Notification-Templates"""
        template_id = "custom_template"
        
        # Template erstellen
        template_data = {
            "subject": "{{event_type}}: {{title}}",
            "body": "Hello {{user_name}}, {{message}}\n\nBest regards,\nThe Team",
            "html_body": "<h2>{{event_type}}: {{title}}</h2><p>Hello {{user_name}},</p><p>{{message}}</p><p>Best regards,<br>The Team</p>",
            "variables": ["event_type", "title", "user_name", "message"]
        }
        
        notification_service.template_manager.create_template(
            template_id=template_id,
            template_data=template_data
        )
        
        # Template abrufen
        template = notification_service.template_manager.get_template(template_id)
        
        assert template.template_id == template_id
        assert template.subject == "{{event_type}}: {{title}}"
        assert "{{user_name}}" in template.body
        
        # Template rendern
        variables = {
            "event_type": "New Feature",
            "title": "Advanced Audio Processing",
            "user_name": "John Doe",
            "message": "We've added new audio processing capabilities to enhance your music production experience."
        }
        
        rendered = notification_service.template_manager.render_template(
            template_id, variables
        )
        
        assert "New Feature: Advanced Audio Processing" in rendered.subject
        assert "Hello John Doe" in rendered.body
        assert "Advanced Audio Processing" in rendered.html_body
        
        # Template aktualisieren
        updated_data = {
            "subject": "[{{event_type}}] {{title}}",
            "body": template_data["body"],
            "html_body": template_data["html_body"]
        }
        
        notification_service.template_manager.update_template(
            template_id, updated_data
        )
        
        updated_template = notification_service.template_manager.get_template(template_id)
        assert updated_template.subject == "[{{event_type}}] {{title}}"
        
        # Template löschen
        notification_service.template_manager.delete_template(template_id)
        
        with pytest.raises(NotificationTemplateError):
            notification_service.template_manager.get_template(template_id)
    
    @pytest.mark.unit
    def test_notification_channels(self, notification_service):
        """Test: Notification-Channels"""
        # Verfügbare Channels abrufen
        channels = notification_service.get_available_channels()
        
        assert "email" in channels
        assert "push" in channels
        assert "sms" in channels
        assert "websocket" in channels
        
        # Channel-Status prüfen
        email_status = notification_service.get_channel_status("email")
        assert email_status.channel == "email"
        assert email_status.is_enabled == True
        assert email_status.is_healthy == True
        
        # Channel-Konfiguration abrufen
        email_config = notification_service.get_channel_config("email")
        assert email_config.provider == "smtp"
        assert email_config.from_address == "noreply@neuromorphe.com"
        
        # Channel temporär deaktivieren
        notification_service.disable_channel("sms")
        
        sms_status = notification_service.get_channel_status("sms")
        assert sms_status.is_enabled == False
        
        # Channel wieder aktivieren
        notification_service.enable_channel("sms")
        
        sms_status = notification_service.get_channel_status("sms")
        assert sms_status.is_enabled == True


class TestNotificationSender:
    """Tests für Notification-Sender"""
    
    @pytest.fixture
    def notification_sender(self, notification_config):
        """Notification-Sender für Tests"""
        return NotificationSender(notification_config)
    
    @pytest.mark.unit
    def test_email_sending(self, notification_sender):
        """Test: E-Mail-Versand"""
        # Mock: SMTP-Client
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            
            result = notification_sender.send_email(
                to_address="test@example.com",
                subject="Test Email",
                body="This is a test email.",
                html_body="<p>This is a test email.</p>"
            )
            
            assert result == True
            
            # SMTP-Server sollte konfiguriert worden sein
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.send_message.assert_called_once()
            mock_server.quit.assert_called_once()
    
    @pytest.mark.unit
    def test_sms_sending(self, notification_sender):
        """Test: SMS-Versand"""
        # Mock: Twilio-Client
        with patch('twilio.rest.Client') as mock_twilio:
            mock_client = MagicMock()
            mock_twilio.return_value = mock_client
            
            mock_message = MagicMock()
            mock_message.sid = "SMS123456"
            mock_client.messages.create.return_value = mock_message
            
            result = notification_sender.send_sms(
                to_number="+1234567890",
                message="This is a test SMS."
            )
            
            assert result == True
            
            # SMS sollte gesendet worden sein
            mock_client.messages.create.assert_called_once_with(
                body="This is a test SMS.",
                from_="+1234567890",
                to="+1234567890"
            )
    
    @pytest.mark.unit
    def test_push_notification_sending(self, notification_sender):
        """Test: Push-Notification-Versand"""
        # Mock: Firebase-Client
        with patch('firebase_admin.messaging.send') as mock_firebase:
            mock_firebase.return_value = "projects/test_project/messages/msg_123"
            
            result = notification_sender.send_push(
                device_token="device_token_123",
                title="Test Push",
                body="This is a test push notification.",
                data={"key": "value"}
            )
            
            assert result == True
            
            # Push-Notification sollte gesendet worden sein
            mock_firebase.assert_called_once()
    
    @pytest.mark.unit
    def test_websocket_notification_sending(self, notification_sender):
        """Test: WebSocket-Notification-Versand"""
        # Mock: WebSocket-Client
        with patch('websockets.connect') as mock_websocket:
            mock_connection = AsyncMock()
            mock_websocket.return_value.__aenter__.return_value = mock_connection
            
            result = notification_sender.send_websocket(
                user_id="user_12345",
                message={
                    "type": "notification",
                    "title": "Test WebSocket",
                    "body": "This is a test WebSocket notification."
                }
            )
            
            assert result == True


class TestNotificationServiceIntegration:
    """Integrationstests für Notification-Service"""
    
    @pytest.mark.integration
    def test_full_notification_workflow(self):
        """Test: Vollständiger Notification-Workflow"""
        config = CoreNotificationConfig(
            enabled=True,
            email_enabled=True,
            push_enabled=True,
            template_enabled=True,
            analytics_enabled=True,
            retry_enabled=True,
            max_retry_attempts=2
        )
        
        service = NotificationService(config)
        
        # 1. Template erstellen
        template_data = {
            "subject": "Welcome to {{app_name}}!",
            "body": "Hello {{user_name}}, welcome to {{app_name}}! We're excited to have you on board.",
            "html_body": "<h1>Welcome to {{app_name}}!</h1><p>Hello {{user_name}}, welcome to {{app_name}}! We're excited to have you on board.</p>"
        }
        
        service.template_manager.create_template(
            template_id="welcome_template",
            template_data=template_data
        )
        
        # 2. Benutzer-Präferenzen setzen
        user_id = "integration_user_12345"
        preferences = {
            "email_enabled": True,
            "push_enabled": True,
            "notification_types": {
                "welcome": True
            }
        }
        
        service.set_user_preferences(user_id, preferences)
        
        # 3. Notification mit Template senden
        with patch.object(service.notification_sender, 'send_email') as mock_email, \
             patch.object(service.notification_sender, 'send_push') as mock_push:
            
            mock_email.return_value = True
            mock_push.return_value = True
            
            result = service.send_notification(
                notification_type="welcome",
                recipient_id=user_id,
                template_id="welcome_template",
                template_variables={
                    "app_name": "Neuromorphe Traum Engine",
                    "user_name": "John Doe"
                },
                channels=["email", "push"]
            )
            
            assert result.success == True
            notification_id = result.notification_id
            
            # 4. Delivery-Status prüfen
            delivery_status = service.get_delivery_status(notification_id)
            assert delivery_status.status == "delivered"
            
            # 5. Analytics prüfen
            analytics = service.get_notification_analytics(
                start_date=datetime.now() - timedelta(minutes=1),
                end_date=datetime.now() + timedelta(minutes=1)
            )
            
            assert analytics.total_notifications >= 1
            assert analytics.successful_notifications >= 1
            
            # 6. Notification-Historie prüfen
            history = service.get_user_notification_history(
                user_id=user_id,
                limit=10
            )
            
            assert len(history) >= 1
            assert any(n.notification_id == notification_id for n in history)
    
    @pytest.mark.performance
    def test_notification_service_performance(self):
        """Test: Notification-Service-Performance"""
        config = CoreNotificationConfig(
            enabled=True,
            batch_enabled=True,
            batch_size=50,
            queue_enabled=True,
            rate_limit_enabled=False  # Für Performance-Test deaktiviert
        )
        
        service = NotificationService(config)
        
        # Performance-Test: Viele Notifications senden
        start_time = time.time()
        
        notifications = []
        for i in range(1000):
            notifications.append({
                "notification_type": "performance_test",
                "recipient_id": f"user_{i}",
                "title": f"Performance Test {i}",
                "message": f"This is performance test notification {i}",
                "channels": ["email"]
            })
        
        with patch.object(service.notification_sender, 'send_email') as mock_email:
            mock_email.return_value = True
            
            # Batch-Versand
            result = service.send_batch_notifications(notifications)
            
            assert result.success == True
            assert result.total_notifications == 1000
            assert result.successful_notifications == 1000
        
        batch_time = time.time() - start_time
        
        # Sollte unter 5 Sekunden für 1000 Notifications dauern
        assert batch_time < 5.0
        
        # Performance-Test: Analytics-Abfrage
        start_time = time.time()
        
        for i in range(100):
            analytics = service.get_notification_analytics(
                start_date=datetime.now() - timedelta(hours=1),
                end_date=datetime.now()
            )
            assert analytics.total_notifications >= 1000
        
        analytics_time = time.time() - start_time
        
        # Sollte unter 2 Sekunden für 100 Analytics-Abfragen dauern
        assert analytics_time < 2.0
        
        # Performance-Test: Template-Rendering
        template_data = {
            "subject": "{{title}} - {{timestamp}}",
            "body": "Hello {{user_name}}, this is message {{message_id}} sent at {{timestamp}}."
        }
        
        service.template_manager.create_template(
            template_id="perf_template",
            template_data=template_data
        )
        
        start_time = time.time()
        
        for i in range(1000):
            variables = {
                "title": f"Performance Test {i}",
                "user_name": f"User {i}",
                "message_id": str(i),
                "timestamp": datetime.now().isoformat()
            }
            
            rendered = service.template_manager.render_template(
                "perf_template", variables
            )
            
            assert f"Performance Test {i}" in rendered.subject
            assert f"User {i}" in rendered.body
        
        template_time = time.time() - start_time
        
        # Sollte unter 1 Sekunde für 1000 Template-Renderings dauern
        assert template_time < 1.0