"""Tests für Event-Service"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.services.event_service import (
    EventService, EventBus, EventDispatcher,
    EventHandler, EventListener, EventSubscriber,
    EventPublisher, EventQueue, EventProcessor,
    EventFilter, EventTransformer, EventValidator,
    EventStore, EventReplay, EventSnapshot,
    EventMetrics, EventMonitor, EventLogger,
    Event, EventType, EventPriority, EventStatus,
    EventData, EventMetadata, EventContext,
    EventSubscription, EventRule, EventPattern,
    AsyncEventHandler, EventMiddleware, EventRouter,
    EventAggregator, EventProjector, EventSourcing
)
from src.core.config import EventConfig
from src.core.exceptions import (
    EventError, EventHandlerError, EventValidationError,
    EventDispatchError, EventSubscriptionError, EventStoreError
)
from src.schemas.events import (
    EventData as EventDataSchema, EventMetricsData,
    EventSubscriptionData, EventRuleData, EventPatternData,
    EventStoreData, EventReplayData, EventSnapshotData
)


class TestEventTypes(Enum):
    """Test-Event-Typen"""
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    STEM_UPLOADED = "stem.uploaded"
    STEM_PROCESSED = "stem.processed"
    STEM_DELETED = "stem.deleted"
    RENDER_STARTED = "render.started"
    RENDER_COMPLETED = "render.completed"
    RENDER_FAILED = "render.failed"
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    ERROR_OCCURRED = "error.occurred"


@dataclass
class TestEventData:
    """Test-Event-Daten"""
    user_id: Optional[str] = None
    stem_id: Optional[str] = None
    render_id: Optional[str] = None
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TestEventService:
    """Tests für Event-Service"""
    
    @pytest.fixture
    def event_config(self):
        """Event-Konfiguration für Tests"""
        return EventConfig(
            enabled=True,
            async_processing=True,
            max_queue_size=10000,
            batch_size=100,
            batch_timeout=1.0,
            retry_attempts=3,
            retry_delay=0.1,
            dead_letter_queue=True,
            event_store_enabled=True,
            event_store_type="memory",
            event_replay_enabled=True,
            event_sourcing_enabled=True,
            metrics_enabled=True,
            monitoring_enabled=True,
            logging_enabled=True,
            middleware_enabled=True,
            validation_enabled=True,
            filtering_enabled=True,
            transformation_enabled=True,
            routing_enabled=True,
            aggregation_enabled=True,
            projection_enabled=True
        )
    
    @pytest.fixture
    def event_service(self, event_config):
        """Event-Service für Tests"""
        return EventService(event_config)
    
    @pytest.mark.unit
    def test_event_service_initialization(self, event_config):
        """Test: Event-Service-Initialisierung"""
        service = EventService(event_config)
        
        assert service.config == event_config
        assert service.enabled == True
        assert service.async_processing == True
        assert isinstance(service.event_bus, EventBus)
        assert isinstance(service.event_dispatcher, EventDispatcher)
        assert isinstance(service.event_store, EventStore)
        assert isinstance(service.event_queue, EventQueue)
    
    @pytest.mark.unit
    def test_event_service_invalid_config(self):
        """Test: Event-Service mit ungültiger Konfiguration"""
        invalid_config = EventConfig(
            max_queue_size=0,  # Ungültige Queue-Größe
            batch_size=-1,  # Negative Batch-Größe
            retry_attempts=-1,  # Negative Retry-Versuche
            retry_delay=-0.1  # Negative Retry-Verzögerung
        )
        
        with pytest.raises(EventError):
            EventService(invalid_config)
    
    @pytest.mark.unit
    async def test_start_stop_service(self, event_service):
        """Test: Service starten und stoppen"""
        # Service starten
        await event_service.start()
        assert event_service.is_running == True
        
        # Service stoppen
        await event_service.stop()
        assert event_service.is_running == False
    
    @pytest.mark.unit
    async def test_publish_event(self, event_service):
        """Test: Event veröffentlichen"""
        await event_service.start()
        
        # Event erstellen
        event_data = TestEventData(
            user_id="12345",
            metadata={"action": "login", "ip": "192.168.1.100"}
        )
        
        event = Event(
            type=TestEventTypes.USER_CREATED.value,
            data=event_data,
            source="auth_service",
            timestamp=datetime.now()
        )
        
        # Event veröffentlichen
        event_id = await event_service.publish(event)
        
        assert event_id is not None
        assert isinstance(event_id, str)
        
        # Event sollte im Store gespeichert sein
        stored_event = await event_service.get_event(event_id)
        assert stored_event is not None
        assert stored_event.type == TestEventTypes.USER_CREATED.value
        assert stored_event.data.user_id == "12345"
        
        await event_service.stop()
    
    @pytest.mark.unit
    async def test_subscribe_to_events(self, event_service):
        """Test: Events abonnieren"""
        await event_service.start()
        
        # Event-Handler erstellen
        received_events = []
        
        async def user_event_handler(event: Event):
            received_events.append(event)
        
        # Events abonnieren
        subscription_id = await event_service.subscribe(
            event_type="user.*",
            handler=user_event_handler
        )
        
        assert subscription_id is not None
        
        # Events veröffentlichen
        user_created_event = Event(
            type=TestEventTypes.USER_CREATED.value,
            data=TestEventData(user_id="12345"),
            source="auth_service"
        )
        
        user_updated_event = Event(
            type=TestEventTypes.USER_UPDATED.value,
            data=TestEventData(user_id="12345"),
            source="user_service"
        )
        
        stem_uploaded_event = Event(
            type=TestEventTypes.STEM_UPLOADED.value,
            data=TestEventData(stem_id="stem_67890"),
            source="upload_service"
        )
        
        await event_service.publish(user_created_event)
        await event_service.publish(user_updated_event)
        await event_service.publish(stem_uploaded_event)
        
        # Kurz warten für Event-Verarbeitung
        await asyncio.sleep(0.1)
        
        # Nur User-Events sollten empfangen worden sein
        assert len(received_events) == 2
        assert all(event.type.startswith("user.") for event in received_events)
        
        await event_service.stop()
    
    @pytest.mark.unit
    async def test_unsubscribe_from_events(self, event_service):
        """Test: Event-Abonnement kündigen"""
        await event_service.start()
        
        received_events = []
        
        async def event_handler(event: Event):
            received_events.append(event)
        
        # Abonnieren
        subscription_id = await event_service.subscribe(
            event_type="stem.*",
            handler=event_handler
        )
        
        # Event veröffentlichen (sollte empfangen werden)
        event1 = Event(
            type=TestEventTypes.STEM_UPLOADED.value,
            data=TestEventData(stem_id="stem_1"),
            source="upload_service"
        )
        
        await event_service.publish(event1)
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        
        # Abonnement kündigen
        await event_service.unsubscribe(subscription_id)
        
        # Weiteres Event veröffentlichen (sollte nicht empfangen werden)
        event2 = Event(
            type=TestEventTypes.STEM_PROCESSED.value,
            data=TestEventData(stem_id="stem_2"),
            source="processing_service"
        )
        
        await event_service.publish(event2)
        await asyncio.sleep(0.1)
        
        # Sollte immer noch nur 1 Event empfangen haben
        assert len(received_events) == 1
        
        await event_service.stop()
    
    @pytest.mark.unit
    async def test_event_filtering(self, event_service):
        """Test: Event-Filterung"""
        await event_service.start()
        
        received_events = []
        
        async def filtered_handler(event: Event):
            received_events.append(event)
        
        # Filter-Funktion: Nur Events von bestimmtem User
        def user_filter(event: Event) -> bool:
            return hasattr(event.data, 'user_id') and event.data.user_id == "12345"
        
        # Mit Filter abonnieren
        await event_service.subscribe(
            event_type="user.*",
            handler=filtered_handler,
            filter_func=user_filter
        )
        
        # Events von verschiedenen Usern veröffentlichen
        event1 = Event(
            type=TestEventTypes.USER_CREATED.value,
            data=TestEventData(user_id="12345"),  # Sollte durchkommen
            source="auth_service"
        )
        
        event2 = Event(
            type=TestEventTypes.USER_CREATED.value,
            data=TestEventData(user_id="67890"),  # Sollte gefiltert werden
            source="auth_service"
        )
        
        event3 = Event(
            type=TestEventTypes.USER_UPDATED.value,
            data=TestEventData(user_id="12345"),  # Sollte durchkommen
            source="user_service"
        )
        
        await event_service.publish(event1)
        await event_service.publish(event2)
        await event_service.publish(event3)
        
        await asyncio.sleep(0.1)
        
        # Nur Events von User 12345 sollten empfangen worden sein
        assert len(received_events) == 2
        assert all(event.data.user_id == "12345" for event in received_events)
        
        await event_service.stop()
    
    @pytest.mark.unit
    async def test_event_transformation(self, event_service):
        """Test: Event-Transformation"""
        await event_service.start()
        
        received_events = []
        
        async def transformed_handler(event: Event):
            received_events.append(event)
        
        # Transformer-Funktion: User-ID zu Großbuchstaben
        def user_id_transformer(event: Event) -> Event:
            if hasattr(event.data, 'user_id') and event.data.user_id:
                event.data.user_id = event.data.user_id.upper()
            return event
        
        # Mit Transformer abonnieren
        await event_service.subscribe(
            event_type="user.*",
            handler=transformed_handler,
            transformer_func=user_id_transformer
        )
        
        # Event mit Kleinbuchstaben-User-ID veröffentlichen
        event = Event(
            type=TestEventTypes.USER_CREATED.value,
            data=TestEventData(user_id="abc123"),
            source="auth_service"
        )
        
        await event_service.publish(event)
        await asyncio.sleep(0.1)
        
        # User-ID sollte transformiert worden sein
        assert len(received_events) == 1
        assert received_events[0].data.user_id == "ABC123"
        
        await event_service.stop()
    
    @pytest.mark.unit
    async def test_event_priority_handling(self, event_service):
        """Test: Event-Prioritäts-Behandlung"""
        await event_service.start()
        
        processed_order = []
        
        async def priority_handler(event: Event):
            processed_order.append(event.priority)
            # Kleine Verzögerung für realistische Verarbeitung
            await asyncio.sleep(0.01)
        
        # Handler für alle Events registrieren
        await event_service.subscribe(
            event_type="*",
            handler=priority_handler
        )
        
        # Events mit verschiedenen Prioritäten veröffentlichen
        low_priority_event = Event(
            type=TestEventTypes.USER_UPDATED.value,
            data=TestEventData(user_id="12345"),
            priority=EventPriority.LOW,
            source="user_service"
        )
        
        high_priority_event = Event(
            type=TestEventTypes.ERROR_OCCURRED.value,
            data=TestEventData(error_message="Critical error"),
            priority=EventPriority.HIGH,
            source="system"
        )
        
        normal_priority_event = Event(
            type=TestEventTypes.STEM_UPLOADED.value,
            data=TestEventData(stem_id="stem_123"),
            priority=EventPriority.NORMAL,
            source="upload_service"
        )
        
        # Events in umgekehrter Prioritäts-Reihenfolge veröffentlichen
        await event_service.publish(low_priority_event)
        await event_service.publish(high_priority_event)
        await event_service.publish(normal_priority_event)
        
        # Warten bis alle Events verarbeitet sind
        await asyncio.sleep(0.2)
        
        # Events sollten nach Priorität verarbeitet worden sein
        assert len(processed_order) == 3
        assert processed_order[0] == EventPriority.HIGH
        assert processed_order[1] == EventPriority.NORMAL
        assert processed_order[2] == EventPriority.LOW
        
        await event_service.stop()
    
    @pytest.mark.unit
    async def test_event_retry_mechanism(self, event_service):
        """Test: Event-Retry-Mechanismus"""
        await event_service.start()
        
        attempt_count = 0
        
        async def failing_handler(event: Event):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:  # Erste 2 Versuche fehlschlagen
                raise EventHandlerError("Temporary failure")
            
            # Dritter Versuch erfolgreich
            return "success"
        
        # Handler mit Retry-Konfiguration registrieren
        await event_service.subscribe(
            event_type="test.retry",
            handler=failing_handler,
            retry_attempts=3,
            retry_delay=0.05
        )
        
        # Event veröffentlichen
        event = Event(
            type="test.retry",
            data=TestEventData(),
            source="test"
        )
        
        await event_service.publish(event)
        
        # Warten bis alle Retry-Versuche abgeschlossen sind
        await asyncio.sleep(0.3)
        
        # Handler sollte 3 Mal aufgerufen worden sein
        assert attempt_count == 3
        
        await event_service.stop()
    
    @pytest.mark.unit
    async def test_dead_letter_queue(self, event_service):
        """Test: Dead Letter Queue"""
        await event_service.start()
        
        async def always_failing_handler(event: Event):
            raise EventHandlerError("Always fails")
        
        # Handler registrieren, der immer fehlschlägt
        await event_service.subscribe(
            event_type="test.dlq",
            handler=always_failing_handler,
            retry_attempts=2
        )
        
        # Event veröffentlichen
        event = Event(
            type="test.dlq",
            data=TestEventData(error_message="Test DLQ"),
            source="test"
        )
        
        event_id = await event_service.publish(event)
        
        # Warten bis alle Retry-Versuche fehlgeschlagen sind
        await asyncio.sleep(0.3)
        
        # Event sollte in Dead Letter Queue sein
        dlq_events = await event_service.get_dead_letter_events()
        assert len(dlq_events) > 0
        
        # Event sollte das veröffentlichte Event sein
        dlq_event = next((e for e in dlq_events if e.id == event_id), None)
        assert dlq_event is not None
        assert dlq_event.status == EventStatus.FAILED
        
        await event_service.stop()
    
    @pytest.mark.unit
    async def test_event_metrics(self, event_service):
        """Test: Event-Metriken"""
        await event_service.start()
        
        processed_events = []
        
        async def metrics_handler(event: Event):
            processed_events.append(event)
        
        # Handler registrieren
        await event_service.subscribe(
            event_type="*",
            handler=metrics_handler
        )
        
        # Verschiedene Events veröffentlichen
        events = [
            Event(type=TestEventTypes.USER_CREATED.value, data=TestEventData(), source="auth"),
            Event(type=TestEventTypes.USER_UPDATED.value, data=TestEventData(), source="user"),
            Event(type=TestEventTypes.STEM_UPLOADED.value, data=TestEventData(), source="upload"),
            Event(type=TestEventTypes.RENDER_STARTED.value, data=TestEventData(), source="render"),
            Event(type=TestEventTypes.RENDER_COMPLETED.value, data=TestEventData(), source="render")
        ]
        
        for event in events:
            await event_service.publish(event)
        
        await asyncio.sleep(0.2)
        
        # Metriken abrufen
        metrics = await event_service.get_metrics()
        
        assert metrics.total_events_published >= 5
        assert metrics.total_events_processed >= 5
        assert metrics.events_per_second > 0
        
        # Event-Typ-Statistiken
        assert "user.created" in metrics.events_by_type
        assert "stem.uploaded" in metrics.events_by_type
        assert "render.started" in metrics.events_by_type
        
        # Handler-Statistiken
        assert len(metrics.handler_stats) > 0
        
        await event_service.stop()


class TestEventBus:
    """Tests für Event-Bus"""
    
    @pytest.fixture
    def event_bus(self):
        """Event-Bus für Tests"""
        return EventBus(
            max_subscribers=1000,
            async_processing=True,
            metrics_enabled=True
        )
    
    @pytest.mark.unit
    async def test_subscribe_and_publish(self, event_bus):
        """Test: Abonnieren und Veröffentlichen"""
        received_events = []
        
        async def test_handler(event: Event):
            received_events.append(event)
        
        # Event-Typ abonnieren
        subscription_id = await event_bus.subscribe("test.event", test_handler)
        assert subscription_id is not None
        
        # Event veröffentlichen
        event = Event(
            type="test.event",
            data=TestEventData(),
            source="test"
        )
        
        await event_bus.publish(event)
        await asyncio.sleep(0.1)
        
        # Event sollte empfangen worden sein
        assert len(received_events) == 1
        assert received_events[0].type == "test.event"
    
    @pytest.mark.unit
    async def test_wildcard_subscriptions(self, event_bus):
        """Test: Wildcard-Abonnements"""
        received_events = []
        
        async def wildcard_handler(event: Event):
            received_events.append(event)
        
        # Wildcard-Pattern abonnieren
        await event_bus.subscribe("user.*", wildcard_handler)
        
        # Verschiedene Events veröffentlichen
        events = [
            Event(type="user.created", data=TestEventData(), source="auth"),
            Event(type="user.updated", data=TestEventData(), source="user"),
            Event(type="user.deleted", data=TestEventData(), source="user"),
            Event(type="stem.uploaded", data=TestEventData(), source="upload")  # Sollte nicht matchen
        ]
        
        for event in events:
            await event_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        # Nur User-Events sollten empfangen worden sein
        assert len(received_events) == 3
        assert all(event.type.startswith("user.") for event in received_events)
    
    @pytest.mark.unit
    async def test_multiple_subscribers(self, event_bus):
        """Test: Mehrere Abonnenten"""
        handler1_events = []
        handler2_events = []
        handler3_events = []
        
        async def handler1(event: Event):
            handler1_events.append(event)
        
        async def handler2(event: Event):
            handler2_events.append(event)
        
        async def handler3(event: Event):
            handler3_events.append(event)
        
        # Mehrere Handler für denselben Event-Typ registrieren
        await event_bus.subscribe("multi.test", handler1)
        await event_bus.subscribe("multi.test", handler2)
        await event_bus.subscribe("multi.test", handler3)
        
        # Event veröffentlichen
        event = Event(
            type="multi.test",
            data=TestEventData(),
            source="test"
        )
        
        await event_bus.publish(event)
        await asyncio.sleep(0.1)
        
        # Alle Handler sollten das Event empfangen haben
        assert len(handler1_events) == 1
        assert len(handler2_events) == 1
        assert len(handler3_events) == 1
    
    @pytest.mark.unit
    async def test_unsubscribe(self, event_bus):
        """Test: Abonnement kündigen"""
        received_events = []
        
        async def test_handler(event: Event):
            received_events.append(event)
        
        # Abonnieren
        subscription_id = await event_bus.subscribe("unsub.test", test_handler)
        
        # Event veröffentlichen (sollte empfangen werden)
        event1 = Event(type="unsub.test", data=TestEventData(), source="test")
        await event_bus.publish(event1)
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        
        # Abonnement kündigen
        await event_bus.unsubscribe(subscription_id)
        
        # Weiteres Event veröffentlichen (sollte nicht empfangen werden)
        event2 = Event(type="unsub.test", data=TestEventData(), source="test")
        await event_bus.publish(event2)
        await asyncio.sleep(0.1)
        
        # Sollte immer noch nur 1 Event empfangen haben
        assert len(received_events) == 1


class TestEventStore:
    """Tests für Event-Store"""
    
    @pytest.fixture
    def event_store(self):
        """Event-Store für Tests"""
        return EventStore(
            store_type="memory",
            max_events=10000,
            compression_enabled=False,
            encryption_enabled=False
        )
    
    @pytest.mark.unit
    async def test_store_and_retrieve_event(self, event_store):
        """Test: Event speichern und abrufen"""
        # Event erstellen
        event = Event(
            type=TestEventTypes.USER_CREATED.value,
            data=TestEventData(user_id="12345"),
            source="auth_service",
            timestamp=datetime.now()
        )
        
        # Event speichern
        event_id = await event_store.store(event)
        assert event_id is not None
        
        # Event abrufen
        retrieved_event = await event_store.get(event_id)
        assert retrieved_event is not None
        assert retrieved_event.type == TestEventTypes.USER_CREATED.value
        assert retrieved_event.data.user_id == "12345"
        assert retrieved_event.source == "auth_service"
    
    @pytest.mark.unit
    async def test_query_events_by_type(self, event_store):
        """Test: Events nach Typ abfragen"""
        # Verschiedene Events speichern
        events = [
            Event(type="user.created", data=TestEventData(user_id="1"), source="auth"),
            Event(type="user.updated", data=TestEventData(user_id="1"), source="user"),
            Event(type="stem.uploaded", data=TestEventData(stem_id="s1"), source="upload"),
            Event(type="user.created", data=TestEventData(user_id="2"), source="auth"),
            Event(type="render.started", data=TestEventData(render_id="r1"), source="render")
        ]
        
        for event in events:
            await event_store.store(event)
        
        # User-Events abfragen
        user_events = await event_store.query_by_type("user.*")
        assert len(user_events) == 3
        assert all(event.type.startswith("user.") for event in user_events)
        
        # Spezifische Events abfragen
        created_events = await event_store.query_by_type("user.created")
        assert len(created_events) == 2
        assert all(event.type == "user.created" for event in created_events)
    
    @pytest.mark.unit
    async def test_query_events_by_timerange(self, event_store):
        """Test: Events nach Zeitraum abfragen"""
        now = datetime.now()
        
        # Events mit verschiedenen Zeitstempeln speichern
        events = [
            Event(type="test.old", data=TestEventData(), source="test", timestamp=now - timedelta(hours=2)),
            Event(type="test.recent", data=TestEventData(), source="test", timestamp=now - timedelta(minutes=30)),
            Event(type="test.new", data=TestEventData(), source="test", timestamp=now)
        ]
        
        for event in events:
            await event_store.store(event)
        
        # Events der letzten Stunde abfragen
        recent_events = await event_store.query_by_timerange(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=1)
        )
        
        assert len(recent_events) == 2
        assert all(event.type in ["test.recent", "test.new"] for event in recent_events)
    
    @pytest.mark.unit
    async def test_event_stream(self, event_store):
        """Test: Event-Stream"""
        # Events speichern
        events = []
        for i in range(10):
            event = Event(
                type=f"stream.test.{i}",
                data=TestEventData(user_id=str(i)),
                source="stream_test"
            )
            event_id = await event_store.store(event)
            events.append((event_id, event))
        
        # Event-Stream abrufen
        stream_events = []
        async for event in event_store.stream():
            stream_events.append(event)
            if len(stream_events) >= 10:
                break
        
        assert len(stream_events) == 10
        
        # Events sollten in der richtigen Reihenfolge sein
        for i, event in enumerate(stream_events):
            assert event.type == f"stream.test.{i}"


class TestEventServiceIntegration:
    """Integrationstests für Event-Service"""
    
    @pytest.mark.integration
    async def test_full_event_workflow(self):
        """Test: Vollständiger Event-Workflow"""
        config = EventConfig(
            enabled=True,
            async_processing=True,
            event_store_enabled=True,
            metrics_enabled=True,
            retry_attempts=2
        )
        
        service = EventService(config)
        await service.start()
        
        # 1. Event-Handler registrieren
        user_events = []
        stem_events = []
        all_events = []
        
        async def user_handler(event: Event):
            user_events.append(event)
        
        async def stem_handler(event: Event):
            stem_events.append(event)
        
        async def all_handler(event: Event):
            all_events.append(event)
        
        await service.subscribe("user.*", user_handler)
        await service.subscribe("stem.*", stem_handler)
        await service.subscribe("*", all_handler)
        
        # 2. Events veröffentlichen
        events_to_publish = [
            Event(type="user.created", data=TestEventData(user_id="u1"), source="auth"),
            Event(type="user.updated", data=TestEventData(user_id="u1"), source="user"),
            Event(type="stem.uploaded", data=TestEventData(stem_id="s1"), source="upload"),
            Event(type="stem.processed", data=TestEventData(stem_id="s1"), source="processing"),
            Event(type="render.started", data=TestEventData(render_id="r1"), source="render")
        ]
        
        published_ids = []
        for event in events_to_publish:
            event_id = await service.publish(event)
            published_ids.append(event_id)
        
        # Warten auf Event-Verarbeitung
        await asyncio.sleep(0.3)
        
        # 3. Handler sollten entsprechende Events empfangen haben
        assert len(user_events) == 2  # user.created, user.updated
        assert len(stem_events) == 2  # stem.uploaded, stem.processed
        assert len(all_events) == 5   # alle Events
        
        # 4. Events sollten im Store gespeichert sein
        for event_id in published_ids:
            stored_event = await service.get_event(event_id)
            assert stored_event is not None
        
        # 5. Event-Suche
        user_stored_events = await service.query_events(event_type="user.*")
        assert len(user_stored_events) >= 2
        
        # 6. Metriken prüfen
        metrics = await service.get_metrics()
        assert metrics.total_events_published >= 5
        assert metrics.total_events_processed >= 15  # 5 Events * 3 Handler
        
        await service.stop()
    
    @pytest.mark.performance
    async def test_event_service_performance(self):
        """Test: Event-Service-Performance"""
        config = EventConfig(
            enabled=True,
            async_processing=True,
            max_queue_size=50000,
            batch_size=1000,
            event_store_enabled=True
        )
        
        service = EventService(config)
        await service.start()
        
        processed_count = 0
        
        async def performance_handler(event: Event):
            nonlocal processed_count
            processed_count += 1
        
        await service.subscribe("perf.*", performance_handler)
        
        # Performance-Test: Viele Events schnell veröffentlichen
        start_time = time.time()
        
        tasks = []
        for i in range(10000):
            event = Event(
                type=f"perf.test.{i % 100}",  # 100 verschiedene Event-Typen
                data=TestEventData(user_id=str(i)),
                source="performance_test"
            )
            task = service.publish(event)
            tasks.append(task)
        
        # Alle Events veröffentlichen
        await asyncio.gather(*tasks)
        
        publish_time = time.time() - start_time
        
        # Warten bis alle Events verarbeitet sind
        await asyncio.sleep(2.0)
        
        total_time = time.time() - start_time
        
        # Performance-Assertions
        assert publish_time < 5.0  # Veröffentlichung sollte unter 5 Sekunden dauern
        assert total_time < 10.0   # Gesamtverarbeitung sollte unter 10 Sekunden dauern
        assert processed_count == 10000  # Alle Events sollten verarbeitet worden sein
        
        # Metriken prüfen
        metrics = await service.get_metrics()
        assert metrics.total_events_published >= 10000
        assert metrics.events_per_second > 1000  # Mindestens 1000 Events/Sekunde
        
        await service.stop()