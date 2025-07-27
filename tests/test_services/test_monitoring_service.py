"""Tests für Monitoring-Service"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
from dataclasses import dataclass

from src.services.monitoring_service import (
    MonitoringService, MetricsCollector, HealthChecker,
    AlertManager, LogAggregator, PerformanceMonitor,
    SystemMonitor, ServiceMonitor, DatabaseMonitor,
    CacheMonitor, APIMonitor, ResourceMonitor,
    MetricType, AlertLevel, HealthStatus, MonitoringConfig,
    Metric, Alert, HealthCheck, LogEntry, PerformanceMetric,
    SystemMetric, ServiceMetric, Threshold, Dashboard,
    NotificationChannel, MetricsExporter, MonitoringRule
)
from src.core.config import MonitoringConfig as CoreMonitoringConfig
from src.core.exceptions import (
    MonitoringError, MetricsCollectionError, AlertingError,
    HealthCheckError, MonitoringConfigurationError
)
from src.schemas.monitoring import (
    MetricData, AlertData, HealthCheckData, LogData,
    PerformanceData, SystemData, ServiceData, ThresholdData,
    DashboardData, NotificationData, MonitoringStats
)


class TestMonitoringService:
    """Tests für Monitoring-Service"""
    
    @pytest.fixture
    def monitoring_config(self):
        """Monitoring-Konfiguration für Tests"""
        return CoreMonitoringConfig(
            enabled=True,
            metrics_collection_interval=10,
            health_check_interval=30,
            alert_check_interval=60,
            log_aggregation_interval=5,
            performance_monitoring=True,
            system_monitoring=True,
            service_monitoring=True,
            database_monitoring=True,
            cache_monitoring=True,
            api_monitoring=True,
            resource_monitoring=True,
            metrics_retention_days=30,
            log_retention_days=7,
            alert_retention_days=90,
            dashboard_enabled=True,
            dashboard_port=3001,
            notification_channels=[
                {
                    "type": "email",
                    "config": {
                        "smtp_host": "localhost",
                        "smtp_port": 587,
                        "recipients": ["admin@example.com"]
                    }
                },
                {
                    "type": "webhook",
                    "config": {
                        "url": "https://hooks.slack.com/test",
                        "method": "POST"
                    }
                }
            ],
            thresholds={
                "cpu_usage": {"warning": 70, "critical": 90},
                "memory_usage": {"warning": 80, "critical": 95},
                "disk_usage": {"warning": 85, "critical": 95},
                "response_time": {"warning": 1000, "critical": 5000},
                "error_rate": {"warning": 5, "critical": 10}
            },
            export_enabled=True,
            export_format="prometheus",
            export_endpoint="/metrics"
        )
    
    @pytest.fixture
    def monitoring_service(self, monitoring_config):
        """Monitoring-Service für Tests"""
        return MonitoringService(monitoring_config)
    
    @pytest.mark.unit
    def test_monitoring_service_initialization(self, monitoring_config):
        """Test: Monitoring-Service-Initialisierung"""
        service = MonitoringService(monitoring_config)
        
        assert service.config == monitoring_config
        assert service.enabled == True
        assert service.metrics_interval == 10
        assert service.health_check_interval == 30
        assert isinstance(service.metrics_collector, MetricsCollector)
        assert isinstance(service.health_checker, HealthChecker)
        assert isinstance(service.alert_manager, AlertManager)
        assert isinstance(service.log_aggregator, LogAggregator)
        assert isinstance(service.performance_monitor, PerformanceMonitor)
    
    @pytest.mark.unit
    def test_monitoring_service_invalid_config(self):
        """Test: Monitoring-Service mit ungültiger Konfiguration"""
        invalid_config = CoreMonitoringConfig(
            enabled=True,
            metrics_collection_interval=-1,  # Ungültig
            health_check_interval=0,         # Ungültig
            metrics_retention_days=-5        # Ungültig
        )
        
        with pytest.raises(MonitoringConfigurationError):
            MonitoringService(invalid_config)
    
    @pytest.mark.unit
    async def test_start_monitoring(self, monitoring_service):
        """Test: Monitoring starten"""
        # Mock Komponenten
        monitoring_service.metrics_collector.start = AsyncMock()
        monitoring_service.health_checker.start = AsyncMock()
        monitoring_service.alert_manager.start = AsyncMock()
        monitoring_service.log_aggregator.start = AsyncMock()
        
        # Monitoring starten
        await monitoring_service.start()
        
        assert monitoring_service.is_running == True
        
        # Alle Komponenten sollten gestartet worden sein
        monitoring_service.metrics_collector.start.assert_called_once()
        monitoring_service.health_checker.start.assert_called_once()
        monitoring_service.alert_manager.start.assert_called_once()
        monitoring_service.log_aggregator.start.assert_called_once()
    
    @pytest.mark.unit
    async def test_stop_monitoring(self, monitoring_service):
        """Test: Monitoring stoppen"""
        # Mock Komponenten
        monitoring_service.metrics_collector.stop = AsyncMock()
        monitoring_service.health_checker.stop = AsyncMock()
        monitoring_service.alert_manager.stop = AsyncMock()
        monitoring_service.log_aggregator.stop = AsyncMock()
        
        # Monitoring als laufend markieren
        monitoring_service.is_running = True
        
        # Monitoring stoppen
        await monitoring_service.stop()
        
        assert monitoring_service.is_running == False
        
        # Alle Komponenten sollten gestoppt worden sein
        monitoring_service.metrics_collector.stop.assert_called_once()
        monitoring_service.health_checker.stop.assert_called_once()
        monitoring_service.alert_manager.stop.assert_called_once()
        monitoring_service.log_aggregator.stop.assert_called_once()
    
    @pytest.mark.unit
    async def test_collect_metrics(self, monitoring_service):
        """Test: Metriken sammeln"""
        # Mock Metrics Collector
        mock_metrics = [
            Metric(
                name="cpu_usage",
                value=45.5,
                type=MetricType.GAUGE,
                timestamp=datetime.now(),
                labels={"host": "localhost"}
            ),
            Metric(
                name="memory_usage",
                value=67.2,
                type=MetricType.GAUGE,
                timestamp=datetime.now(),
                labels={"host": "localhost"}
            )
        ]
        
        monitoring_service.metrics_collector.collect_all = AsyncMock(return_value=mock_metrics)
        
        # Metriken sammeln
        metrics = await monitoring_service.collect_metrics()
        
        assert len(metrics) == 2
        assert metrics[0].name == "cpu_usage"
        assert metrics[0].value == 45.5
        assert metrics[1].name == "memory_usage"
        assert metrics[1].value == 67.2
    
    @pytest.mark.unit
    async def test_health_check(self, monitoring_service):
        """Test: Health-Check"""
        # Mock Health Checker
        mock_health_checks = [
            HealthCheck(
                service="database",
                status=HealthStatus.HEALTHY,
                response_time=50,
                timestamp=datetime.now(),
                details={"connections": 5}
            ),
            HealthCheck(
                service="cache",
                status=HealthStatus.DEGRADED,
                response_time=200,
                timestamp=datetime.now(),
                details={"hit_rate": 0.85}
            )
        ]
        
        monitoring_service.health_checker.check_all = AsyncMock(return_value=mock_health_checks)
        
        # Health-Check durchführen
        health_checks = await monitoring_service.check_health()
        
        assert len(health_checks) == 2
        assert health_checks[0].service == "database"
        assert health_checks[0].status == HealthStatus.HEALTHY
        assert health_checks[1].service == "cache"
        assert health_checks[1].status == HealthStatus.DEGRADED
    
    @pytest.mark.unit
    async def test_alert_processing(self, monitoring_service):
        """Test: Alert-Verarbeitung"""
        # Mock Alert Manager
        mock_alerts = [
            Alert(
                id="alert_1",
                name="High CPU Usage",
                level=AlertLevel.WARNING,
                message="CPU usage is above 70%",
                timestamp=datetime.now(),
                source="system_monitor",
                metric_name="cpu_usage",
                metric_value=75.0,
                threshold=70.0
            )
        ]
        
        monitoring_service.alert_manager.process_alerts = AsyncMock(return_value=mock_alerts)
        
        # Alerts verarbeiten
        alerts = await monitoring_service.process_alerts()
        
        assert len(alerts) == 1
        assert alerts[0].name == "High CPU Usage"
        assert alerts[0].level == AlertLevel.WARNING
        assert alerts[0].metric_value == 75.0
    
    @pytest.mark.unit
    def test_get_monitoring_stats(self, monitoring_service):
        """Test: Monitoring-Statistiken abrufen"""
        # Mock Statistiken
        monitoring_service.metrics_collector.get_stats = Mock(return_value={
            "total_metrics": 1000,
            "metrics_per_second": 10.5
        })
        
        monitoring_service.health_checker.get_stats = Mock(return_value={
            "total_checks": 500,
            "healthy_services": 8,
            "degraded_services": 1,
            "unhealthy_services": 0
        })
        
        monitoring_service.alert_manager.get_stats = Mock(return_value={
            "total_alerts": 25,
            "active_alerts": 3,
            "resolved_alerts": 22
        })
        
        # Statistiken abrufen
        stats = monitoring_service.get_stats()
        
        assert stats["metrics"]["total_metrics"] == 1000
        assert stats["health"]["healthy_services"] == 8
        assert stats["alerts"]["active_alerts"] == 3


class TestMetricsCollector:
    """Tests für Metrics-Collector"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Metrics-Collector für Tests"""
        return MetricsCollector(
            collection_interval=10,
            retention_days=30,
            enabled_monitors=[
                "system", "service", "database", "cache", "api"
            ]
        )
    
    @pytest.mark.unit
    async def test_collect_system_metrics(self, metrics_collector):
        """Test: System-Metriken sammeln"""
        # Mock System Monitor
        mock_system_monitor = Mock()
        mock_system_monitor.collect_metrics = AsyncMock(return_value=[
            Metric(
                name="cpu_usage",
                value=45.5,
                type=MetricType.GAUGE,
                timestamp=datetime.now()
            ),
            Metric(
                name="memory_usage",
                value=67.2,
                type=MetricType.GAUGE,
                timestamp=datetime.now()
            )
        ])
        
        metrics_collector.system_monitor = mock_system_monitor
        
        # System-Metriken sammeln
        metrics = await metrics_collector.collect_system_metrics()
        
        assert len(metrics) == 2
        assert metrics[0].name == "cpu_usage"
        assert metrics[1].name == "memory_usage"
    
    @pytest.mark.unit
    async def test_collect_service_metrics(self, metrics_collector):
        """Test: Service-Metriken sammeln"""
        # Mock Service Monitor
        mock_service_monitor = Mock()
        mock_service_monitor.collect_metrics = AsyncMock(return_value=[
            Metric(
                name="request_count",
                value=1500,
                type=MetricType.COUNTER,
                timestamp=datetime.now(),
                labels={"service": "api", "endpoint": "/stems"}
            ),
            Metric(
                name="response_time",
                value=250.5,
                type=MetricType.HISTOGRAM,
                timestamp=datetime.now(),
                labels={"service": "api", "endpoint": "/stems"}
            )
        ])
        
        metrics_collector.service_monitor = mock_service_monitor
        
        # Service-Metriken sammeln
        metrics = await metrics_collector.collect_service_metrics()
        
        assert len(metrics) == 2
        assert metrics[0].name == "request_count"
        assert metrics[0].type == MetricType.COUNTER
        assert metrics[1].name == "response_time"
        assert metrics[1].type == MetricType.HISTOGRAM
    
    @pytest.mark.unit
    async def test_store_metrics(self, metrics_collector):
        """Test: Metriken speichern"""
        metrics = [
            Metric(
                name="test_metric",
                value=42.0,
                type=MetricType.GAUGE,
                timestamp=datetime.now()
            )
        ]
        
        # Mock Storage
        metrics_collector.storage = AsyncMock()
        
        # Metriken speichern
        await metrics_collector.store_metrics(metrics)
        
        # Storage sollte aufgerufen worden sein
        metrics_collector.storage.store_batch.assert_called_once_with(metrics)
    
    @pytest.mark.unit
    async def test_get_metrics_by_name(self, metrics_collector):
        """Test: Metriken nach Name abrufen"""
        metric_name = "cpu_usage"
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        # Mock Storage
        mock_metrics = [
            Metric(
                name=metric_name,
                value=45.5,
                type=MetricType.GAUGE,
                timestamp=datetime.now() - timedelta(minutes=30)
            ),
            Metric(
                name=metric_name,
                value=52.1,
                type=MetricType.GAUGE,
                timestamp=datetime.now() - timedelta(minutes=15)
            )
        ]
        
        metrics_collector.storage = AsyncMock()
        metrics_collector.storage.get_metrics.return_value = mock_metrics
        
        # Metriken abrufen
        metrics = await metrics_collector.get_metrics(
            name=metric_name,
            start_time=start_time,
            end_time=end_time
        )
        
        assert len(metrics) == 2
        assert all(m.name == metric_name for m in metrics)
    
    @pytest.mark.unit
    def test_calculate_metric_statistics(self, metrics_collector):
        """Test: Metrik-Statistiken berechnen"""
        metrics = [
            Metric(name="test", value=10.0, type=MetricType.GAUGE, timestamp=datetime.now()),
            Metric(name="test", value=20.0, type=MetricType.GAUGE, timestamp=datetime.now()),
            Metric(name="test", value=30.0, type=MetricType.GAUGE, timestamp=datetime.now()),
            Metric(name="test", value=40.0, type=MetricType.GAUGE, timestamp=datetime.now()),
            Metric(name="test", value=50.0, type=MetricType.GAUGE, timestamp=datetime.now())
        ]
        
        # Statistiken berechnen
        stats = metrics_collector.calculate_statistics(metrics)
        
        assert stats["count"] == 5
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["std_dev"] > 0


class TestHealthChecker:
    """Tests für Health-Checker"""
    
    @pytest.fixture
    def health_checker(self):
        """Health-Checker für Tests"""
        return HealthChecker(
            check_interval=30,
            timeout=5,
            enabled_services=[
                "database", "cache", "api", "render_service", "clap_service"
            ]
        )
    
    @pytest.mark.unit
    async def test_check_database_health(self, health_checker):
        """Test: Datenbank-Health-Check"""
        # Mock Database Service
        mock_db_service = AsyncMock()
        mock_db_service.ping.return_value = True
        mock_db_service.get_connection_count.return_value = 5
        
        health_checker.database_service = mock_db_service
        
        # Health-Check durchführen
        health_check = await health_checker.check_database_health()
        
        assert health_check.service == "database"
        assert health_check.status == HealthStatus.HEALTHY
        assert health_check.response_time > 0
        assert health_check.details["connections"] == 5
    
    @pytest.mark.unit
    async def test_check_database_health_failure(self, health_checker):
        """Test: Datenbank-Health-Check-Fehler"""
        # Mock Database Service mit Fehler
        mock_db_service = AsyncMock()
        mock_db_service.ping.side_effect = Exception("Connection failed")
        
        health_checker.database_service = mock_db_service
        
        # Health-Check durchführen
        health_check = await health_checker.check_database_health()
        
        assert health_check.service == "database"
        assert health_check.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in health_check.details["error"]
    
    @pytest.mark.unit
    async def test_check_cache_health(self, health_checker):
        """Test: Cache-Health-Check"""
        # Mock Cache Service
        mock_cache_service = AsyncMock()
        mock_cache_service.ping.return_value = True
        mock_cache_service.get_stats.return_value = {
            "hit_rate": 0.85,
            "memory_usage": 67.5
        }
        
        health_checker.cache_service = mock_cache_service
        
        # Health-Check durchführen
        health_check = await health_checker.check_cache_health()
        
        assert health_check.service == "cache"
        assert health_check.status == HealthStatus.HEALTHY
        assert health_check.details["hit_rate"] == 0.85
    
    @pytest.mark.unit
    async def test_check_api_health(self, health_checker):
        """Test: API-Health-Check"""
        # Mock API Service
        mock_api_service = AsyncMock()
        mock_api_service.is_running.return_value = True
        mock_api_service.get_stats.return_value = {
            "active_connections": 25,
            "avg_response_time": 150.5
        }
        
        health_checker.api_service = mock_api_service
        
        # Health-Check durchführen
        health_check = await health_checker.check_api_health()
        
        assert health_check.service == "api"
        assert health_check.status == HealthStatus.HEALTHY
        assert health_check.details["active_connections"] == 25
    
    @pytest.mark.unit
    async def test_check_all_services(self, health_checker):
        """Test: Alle Services Health-Check"""
        # Mock alle Service-Checker
        health_checker.check_database_health = AsyncMock(return_value=HealthCheck(
            service="database",
            status=HealthStatus.HEALTHY,
            response_time=50,
            timestamp=datetime.now()
        ))
        
        health_checker.check_cache_health = AsyncMock(return_value=HealthCheck(
            service="cache",
            status=HealthStatus.DEGRADED,
            response_time=200,
            timestamp=datetime.now()
        ))
        
        health_checker.check_api_health = AsyncMock(return_value=HealthCheck(
            service="api",
            status=HealthStatus.HEALTHY,
            response_time=75,
            timestamp=datetime.now()
        ))
        
        # Alle Health-Checks durchführen
        health_checks = await health_checker.check_all()
        
        assert len(health_checks) == 3
        
        # Services nach Status gruppieren
        healthy = [hc for hc in health_checks if hc.status == HealthStatus.HEALTHY]
        degraded = [hc for hc in health_checks if hc.status == HealthStatus.DEGRADED]
        
        assert len(healthy) == 2
        assert len(degraded) == 1
    
    @pytest.mark.unit
    def test_determine_overall_health(self, health_checker):
        """Test: Gesamte System-Health bestimmen"""
        # Alle Services gesund
        healthy_checks = [
            HealthCheck(service="db", status=HealthStatus.HEALTHY, response_time=50, timestamp=datetime.now()),
            HealthCheck(service="cache", status=HealthStatus.HEALTHY, response_time=30, timestamp=datetime.now()),
            HealthCheck(service="api", status=HealthStatus.HEALTHY, response_time=75, timestamp=datetime.now())
        ]
        
        overall_status = health_checker.determine_overall_health(healthy_checks)
        assert overall_status == HealthStatus.HEALTHY
        
        # Ein Service degradiert
        mixed_checks = [
            HealthCheck(service="db", status=HealthStatus.HEALTHY, response_time=50, timestamp=datetime.now()),
            HealthCheck(service="cache", status=HealthStatus.DEGRADED, response_time=200, timestamp=datetime.now()),
            HealthCheck(service="api", status=HealthStatus.HEALTHY, response_time=75, timestamp=datetime.now())
        ]
        
        overall_status = health_checker.determine_overall_health(mixed_checks)
        assert overall_status == HealthStatus.DEGRADED
        
        # Ein Service ungesund
        unhealthy_checks = [
            HealthCheck(service="db", status=HealthStatus.HEALTHY, response_time=50, timestamp=datetime.now()),
            HealthCheck(service="cache", status=HealthStatus.UNHEALTHY, response_time=0, timestamp=datetime.now()),
            HealthCheck(service="api", status=HealthStatus.HEALTHY, response_time=75, timestamp=datetime.now())
        ]
        
        overall_status = health_checker.determine_overall_health(unhealthy_checks)
        assert overall_status == HealthStatus.UNHEALTHY


class TestAlertManager:
    """Tests für Alert-Manager"""
    
    @pytest.fixture
    def alert_manager(self):
        """Alert-Manager für Tests"""
        thresholds = {
            "cpu_usage": {"warning": 70, "critical": 90},
            "memory_usage": {"warning": 80, "critical": 95},
            "response_time": {"warning": 1000, "critical": 5000}
        }
        
        return AlertManager(
            thresholds=thresholds,
            check_interval=60,
            notification_channels=[
                {"type": "email", "config": {"recipients": ["admin@test.com"]}},
                {"type": "webhook", "config": {"url": "https://hooks.test.com"}}
            ]
        )
    
    @pytest.mark.unit
    def test_evaluate_metric_thresholds(self, alert_manager):
        """Test: Metrik-Schwellenwerte evaluieren"""
        # CPU-Metrik über Warning-Schwelle
        cpu_metric = Metric(
            name="cpu_usage",
            value=75.0,
            type=MetricType.GAUGE,
            timestamp=datetime.now()
        )
        
        alert = alert_manager.evaluate_metric(cpu_metric)
        
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert alert.metric_name == "cpu_usage"
        assert alert.metric_value == 75.0
        assert alert.threshold == 70.0
        
        # Memory-Metrik über Critical-Schwelle
        memory_metric = Metric(
            name="memory_usage",
            value=97.0,
            type=MetricType.GAUGE,
            timestamp=datetime.now()
        )
        
        alert = alert_manager.evaluate_metric(memory_metric)
        
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL
        assert alert.metric_name == "memory_usage"
        assert alert.metric_value == 97.0
        assert alert.threshold == 95.0
        
        # Metrik unter Schwelle
        normal_metric = Metric(
            name="cpu_usage",
            value=45.0,
            type=MetricType.GAUGE,
            timestamp=datetime.now()
        )
        
        alert = alert_manager.evaluate_metric(normal_metric)
        assert alert is None
    
    @pytest.mark.unit
    async def test_send_alert_notifications(self, alert_manager):
        """Test: Alert-Benachrichtigungen senden"""
        alert = Alert(
            id="test_alert",
            name="High CPU Usage",
            level=AlertLevel.CRITICAL,
            message="CPU usage is critically high at 95%",
            timestamp=datetime.now(),
            source="system_monitor",
            metric_name="cpu_usage",
            metric_value=95.0,
            threshold=90.0
        )
        
        # Mock Notification Channels
        alert_manager.email_notifier = AsyncMock()
        alert_manager.webhook_notifier = AsyncMock()
        
        # Benachrichtigungen senden
        await alert_manager.send_notifications(alert)
        
        # Beide Channels sollten aufgerufen worden sein
        alert_manager.email_notifier.send.assert_called_once_with(alert)
        alert_manager.webhook_notifier.send.assert_called_once_with(alert)
    
    @pytest.mark.unit
    def test_alert_deduplication(self, alert_manager):
        """Test: Alert-Deduplizierung"""
        # Erste Alert
        alert_1 = Alert(
            id="alert_1",
            name="High CPU Usage",
            level=AlertLevel.WARNING,
            message="CPU usage is high",
            timestamp=datetime.now(),
            source="system_monitor",
            metric_name="cpu_usage",
            metric_value=75.0,
            threshold=70.0
        )
        
        # Identische Alert (sollte dedupliziert werden)
        alert_2 = Alert(
            id="alert_2",
            name="High CPU Usage",
            level=AlertLevel.WARNING,
            message="CPU usage is high",
            timestamp=datetime.now(),
            source="system_monitor",
            metric_name="cpu_usage",
            metric_value=76.0,  # Leicht unterschiedlicher Wert
            threshold=70.0
        )
        
        # Erste Alert hinzufügen
        is_duplicate_1 = alert_manager.is_duplicate_alert(alert_1)
        assert is_duplicate_1 == False
        alert_manager.add_active_alert(alert_1)
        
        # Zweite Alert prüfen (sollte als Duplikat erkannt werden)
        is_duplicate_2 = alert_manager.is_duplicate_alert(alert_2)
        assert is_duplicate_2 == True
    
    @pytest.mark.unit
    def test_alert_resolution(self, alert_manager):
        """Test: Alert-Auflösung"""
        # Alert erstellen und als aktiv markieren
        alert = Alert(
            id="resolvable_alert",
            name="High Memory Usage",
            level=AlertLevel.WARNING,
            message="Memory usage is high",
            timestamp=datetime.now(),
            source="system_monitor",
            metric_name="memory_usage",
            metric_value=85.0,
            threshold=80.0
        )
        
        alert_manager.add_active_alert(alert)
        assert len(alert_manager.active_alerts) == 1
        
        # Metrik unter Schwelle (sollte Alert auflösen)
        resolved_metric = Metric(
            name="memory_usage",
            value=75.0,
            type=MetricType.GAUGE,
            timestamp=datetime.now()
        )
        
        resolved_alerts = alert_manager.check_alert_resolution([resolved_metric])
        
        assert len(resolved_alerts) == 1
        assert resolved_alerts[0].id == "resolvable_alert"
        assert len(alert_manager.active_alerts) == 0


class TestMonitoringServiceIntegration:
    """Integrationstests für Monitoring-Service"""
    
    @pytest.mark.integration
    async def test_full_monitoring_workflow(self):
        """Test: Vollständiger Monitoring-Workflow"""
        config = CoreMonitoringConfig(
            enabled=True,
            metrics_collection_interval=1,  # Kurz für Test
            health_check_interval=2,
            alert_check_interval=1,
            performance_monitoring=True,
            system_monitoring=True,
            thresholds={
                "cpu_usage": {"warning": 70, "critical": 90},
                "memory_usage": {"warning": 80, "critical": 95}
            }
        )
        
        service = MonitoringService(config)
        
        # Mock externe Services
        service.metrics_collector.system_monitor = Mock()
        service.metrics_collector.system_monitor.collect_metrics = AsyncMock(return_value=[
            Metric(name="cpu_usage", value=75.0, type=MetricType.GAUGE, timestamp=datetime.now()),
            Metric(name="memory_usage", value=85.0, type=MetricType.GAUGE, timestamp=datetime.now())
        ])
        
        service.health_checker.database_service = AsyncMock()
        service.health_checker.database_service.ping.return_value = True
        
        # 1. Monitoring starten
        await service.start()
        assert service.is_running == True
        
        # 2. Metriken sammeln
        metrics = await service.collect_metrics()
        assert len(metrics) == 2
        assert any(m.name == "cpu_usage" and m.value == 75.0 for m in metrics)
        
        # 3. Health-Checks durchführen
        health_checks = await service.check_health()
        assert len(health_checks) >= 1
        
        # 4. Alerts verarbeiten
        alerts = await service.process_alerts()
        # CPU und Memory über Schwelle -> sollte Alerts geben
        assert len(alerts) >= 1
        
        # 5. Statistiken abrufen
        stats = service.get_stats()
        assert "metrics" in stats
        assert "health" in stats
        assert "alerts" in stats
        
        # 6. Monitoring stoppen
        await service.stop()
        assert service.is_running == False
    
    @pytest.mark.performance
    async def test_monitoring_service_performance(self):
        """Test: Monitoring-Service-Performance"""
        import time
        
        config = CoreMonitoringConfig(
            enabled=True,
            metrics_collection_interval=0.1,  # Sehr kurz für Performance-Test
            performance_monitoring=True
        )
        
        service = MonitoringService(config)
        
        # Mock schnelle Metrik-Sammlung
        service.metrics_collector.collect_all = AsyncMock(return_value=[
            Metric(name=f"metric_{i}", value=float(i), type=MetricType.GAUGE, timestamp=datetime.now())
            for i in range(100)  # 100 Metriken
        ])
        
        # Performance-Test: Metrik-Sammlung
        start_time = time.time()
        
        for _ in range(10):
            metrics = await service.collect_metrics()
            assert len(metrics) == 100
        
        collection_time = time.time() - start_time
        
        # Sollte unter 1 Sekunde für 10 Sammlungen mit je 100 Metriken dauern
        assert collection_time < 1.0
        
        # Durchschnittliche Zeit pro Sammlung
        avg_collection_time = collection_time / 10
        assert avg_collection_time < 0.1  # Unter 100ms pro Sammlung
        
        # Performance-Test: Alert-Verarbeitung
        service.alert_manager.evaluate_metric = Mock(return_value=None)  # Keine Alerts
        
        start_time = time.time()
        
        for _ in range(100):
            await service.process_alerts()
        
        alert_processing_time = time.time() - start_time
        
        # Sollte unter 0.5 Sekunden für 100 Alert-Verarbeitungen dauern
        assert alert_processing_time < 0.5
        
        # Performance-Test: Health-Checks
        service.health_checker.check_all = AsyncMock(return_value=[
            HealthCheck(service="test", status=HealthStatus.HEALTHY, response_time=10, timestamp=datetime.now())
        ])
        
        start_time = time.time()
        
        for _ in range(50):
            health_checks = await service.check_health()
            assert len(health_checks) == 1
        
        health_check_time = time.time() - start_time
        
        # Sollte unter 0.5 Sekunden für 50 Health-Checks dauern
        assert health_check_time < 0.5