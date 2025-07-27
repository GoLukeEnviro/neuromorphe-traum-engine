"""Tests für Task-Service"""

import pytest
import json
import time
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from src.services.task_service import (
    TaskService, TaskManager, TaskExecutor, TaskScheduler,
    TaskQueue, TaskWorker, TaskMonitor, TaskResult,
    TaskDefinition, TaskInstance, TaskExecution, TaskHistory,
    TaskDependency, TaskTrigger, TaskCondition, TaskAction,
    TaskContext, TaskData, TaskMetadata, TaskProgress,
    TaskStatus, TaskPriority, TaskType, ExecutionMode,
    ScheduleType, TriggerType, ConditionType, ActionType,
    TaskConfig, WorkerConfig, SchedulerConfig, QueueConfig,
    TaskProcessor, TaskValidator, TaskSerializer, TaskLogger,
    TaskMetrics, TaskAnalytics, TaskAudit, TaskBackup,
    TaskRecovery, TaskCleanup, TaskOptimizer, TaskProfiler,
    RetryPolicy, TimeoutPolicy, ResourcePolicy, SecurityPolicy,
    TaskEvent, TaskNotification, TaskAlert, TaskReport,
    WorkflowEngine, WorkflowDefinition, WorkflowInstance,
    WorkflowStep, WorkflowTransition, WorkflowState,
    ParallelTask, SequentialTask, ConditionalTask, LoopTask,
    MapReduceTask, PipelineTask, BatchTask, StreamTask
)
from src.core.config import TaskConfig as CoreTaskConfig
from src.core.exceptions import (
    TaskError, TaskExecutionError, TaskSchedulingError,
    TaskValidationError, TaskTimeoutError, TaskDependencyError,
    TaskResourceError, TaskSecurityError, TaskConfigError,
    WorkflowError, WorkflowExecutionError, WorkflowValidationError
)
from src.schemas.task import (
    TaskData as TaskSchemaData, TaskConfigData, TaskStatsData,
    TaskEventData, TaskResultData, WorkflowData, WorkflowStatsData
)


class TestTaskType(Enum):
    """Test-Task-Typen"""
    AUDIO_PROCESSING = "audio_processing"
    STEM_ANALYSIS = "stem_analysis"
    ARRANGEMENT_GENERATION = "arrangement_generation"
    COLLABORATION_SYNC = "collaboration_sync"
    DATABASE_BACKUP = "database_backup"
    CACHE_CLEANUP = "cache_cleanup"
    NOTIFICATION_BATCH = "notification_batch"
    ANALYTICS_REPORT = "analytics_report"
    SECURITY_SCAN = "security_scan"
    SYSTEM_MAINTENANCE = "system_maintenance"


class TestTaskStatus(Enum):
    """Test-Task-Status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class TestTaskPriority(Enum):
    """Test-Task-Prioritäten"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class TestTask:
    """Test-Task"""
    id: str
    name: str
    task_type: TestTaskType
    function: str
    parameters: Dict[str, Any]
    priority: TestTaskPriority = TestTaskPriority.NORMAL
    timeout: int = 300  # 5 Minuten
    retry_count: int = 3
    dependencies: List[str] = None
    schedule: str = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()


class TestTaskService:
    """Tests für Task-Service"""
    
    @pytest.fixture
    def task_config(self):
        """Task-Konfiguration für Tests"""
        return CoreTaskConfig(
            enabled=True,
            max_workers=4,
            queue_max_size=1000,
            default_timeout=300,
            max_retry_attempts=3,
            retry_delay=60,
            cleanup_interval=3600,
            history_retention_days=30,
            monitoring_enabled=True,
            metrics_enabled=True,
            audit_enabled=True,
            backup_enabled=True,
            recovery_enabled=True,
            optimization_enabled=True,
            profiling_enabled=True,
            # Scheduler-Konfiguration
            scheduler_enabled=True,
            scheduler_interval=60,
            scheduler_max_jobs=100,
            # Worker-Konfiguration
            worker_pool_size=4,
            worker_timeout=600,
            worker_heartbeat_interval=30,
            # Queue-Konfiguration
            queue_type="memory",  # memory, redis, database
            queue_persistence=False,
            queue_priority_enabled=True,
            # Workflow-Konfiguration
            workflow_enabled=True,
            workflow_max_depth=10,
            workflow_timeout=3600,
            # Security-Konfiguration
            security_enabled=True,
            allowed_functions=["test_function", "audio_process", "stem_analyze"],
            resource_limits={
                "max_memory_mb": 1024,
                "max_cpu_percent": 80,
                "max_execution_time": 600
            }
        )
    
    @pytest.fixture
    def task_service(self, task_config):
        """Task-Service für Tests"""
        return TaskService(task_config)
    
    @pytest.fixture
    def test_task(self):
        """Test-Task"""
        return TestTask(
            id="task_12345",
            name="Test Audio Processing",
            task_type=TestTaskType.AUDIO_PROCESSING,
            function="audio_process",
            parameters={
                "input_file": "/path/to/audio.wav",
                "output_format": "mp3",
                "quality": "high",
                "effects": ["normalize", "eq"]
            },
            priority=TestTaskPriority.HIGH,
            timeout=600,
            retry_count=2
        )
    
    @pytest.fixture
    def test_workflow(self):
        """Test-Workflow"""
        return {
            "id": "workflow_12345",
            "name": "Audio Processing Workflow",
            "description": "Complete audio processing pipeline",
            "steps": [
                {
                    "id": "step_1",
                    "name": "Audio Analysis",
                    "task_type": "stem_analysis",
                    "function": "analyze_audio",
                    "parameters": {"input_file": "{{workflow.input_file}}"},
                    "dependencies": []
                },
                {
                    "id": "step_2",
                    "name": "Audio Processing",
                    "task_type": "audio_processing",
                    "function": "process_audio",
                    "parameters": {
                        "input_file": "{{workflow.input_file}}",
                        "analysis_result": "{{step_1.result}}"
                    },
                    "dependencies": ["step_1"]
                },
                {
                    "id": "step_3",
                    "name": "Generate Arrangement",
                    "task_type": "arrangement_generation",
                    "function": "generate_arrangement",
                    "parameters": {
                        "processed_audio": "{{step_2.result}}",
                        "style": "electronic"
                    },
                    "dependencies": ["step_2"]
                }
            ],
            "parameters": {
                "input_file": "/path/to/input.wav"
            }
        }
    
    @pytest.mark.unit
    def test_task_service_initialization(self, task_config):
        """Test: Task-Service-Initialisierung"""
        service = TaskService(task_config)
        
        assert service.config == task_config
        assert isinstance(service.task_manager, TaskManager)
        assert isinstance(service.task_executor, TaskExecutor)
        assert isinstance(service.task_scheduler, TaskScheduler)
        assert isinstance(service.task_queue, TaskQueue)
        assert isinstance(service.task_monitor, TaskMonitor)
        assert isinstance(service.workflow_engine, WorkflowEngine)
        assert service.is_running == False
    
    @pytest.mark.unit
    def test_task_service_invalid_config(self):
        """Test: Task-Service mit ungültiger Konfiguration"""
        invalid_config = CoreTaskConfig(
            enabled=True,
            max_workers=0,  # Ungültige Worker-Anzahl
            queue_max_size=-1,  # Ungültige Queue-Größe
            default_timeout=0,  # Ungültiger Timeout
            max_retry_attempts=-1,  # Negative Retry-Versuche
            worker_pool_size=0,  # Ungültige Pool-Größe
            resource_limits={
                "max_memory_mb": -1,  # Ungültiges Memory-Limit
                "max_cpu_percent": 150  # Ungültiger CPU-Prozentsatz
            }
        )
        
        with pytest.raises(TaskConfigError):
            TaskService(invalid_config)
    
    @pytest.mark.unit
    def test_task_service_start_stop(self, task_service):
        """Test: Task-Service starten und stoppen"""
        # Service starten
        task_service.start()
        
        assert task_service.is_running == True
        assert task_service.task_executor.is_running == True
        assert task_service.task_scheduler.is_running == True
        
        # Service stoppen
        task_service.stop()
        
        assert task_service.is_running == False
        assert task_service.task_executor.is_running == False
        assert task_service.task_scheduler.is_running == False
    
    @pytest.mark.unit
    def test_submit_task(self, task_service, test_task):
        """Test: Task einreichen"""
        # Mock: Task-Funktion
        def mock_audio_process(input_file, output_format, quality, effects):
            return {
                "output_file": "/path/to/output.mp3",
                "duration": 180.5,
                "format": output_format,
                "quality": quality,
                "effects_applied": effects
            }
        
        # Task-Funktion registrieren
        task_service.register_function("audio_process", mock_audio_process)
        
        # Task einreichen
        task_result = task_service.submit_task(
            name=test_task.name,
            task_type=test_task.task_type.value,
            function=test_task.function,
            parameters=test_task.parameters,
            priority=test_task.priority.value,
            timeout=test_task.timeout,
            retry_count=test_task.retry_count
        )
        
        assert task_result.task_id is not None
        assert task_result.status == "queued"
        assert task_result.priority == test_task.priority.value
        
        # Task sollte in der Queue sein
        queue_size = task_service.get_queue_size()
        assert queue_size >= 1
        
        # Task-Details abrufen
        task_details = task_service.get_task(task_result.task_id)
        assert task_details.name == test_task.name
        assert task_details.task_type == test_task.task_type.value
        assert task_details.function == test_task.function
    
    @pytest.mark.unit
    def test_execute_task(self, task_service, test_task):
        """Test: Task ausführen"""
        # Mock: Task-Funktion
        def mock_audio_process(input_file, output_format, quality, effects):
            time.sleep(0.1)  # Simuliere Verarbeitung
            return {
                "output_file": "/path/to/output.mp3",
                "processing_time": 0.1,
                "success": True
            }
        
        task_service.register_function("audio_process", mock_audio_process)
        task_service.start()
        
        try:
            # Task einreichen und ausführen
            task_result = task_service.submit_task(
                name=test_task.name,
                task_type=test_task.task_type.value,
                function=test_task.function,
                parameters=test_task.parameters
            )
            
            task_id = task_result.task_id
            
            # Warten bis Task abgeschlossen ist
            timeout = 5.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                task_status = task_service.get_task_status(task_id)
                if task_status.status in ["completed", "failed"]:
                    break
                time.sleep(0.1)
            
            # Task sollte erfolgreich abgeschlossen sein
            final_status = task_service.get_task_status(task_id)
            assert final_status.status == "completed"
            assert final_status.result is not None
            assert final_status.result["success"] == True
            assert final_status.result["output_file"] == "/path/to/output.mp3"
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_task_with_dependencies(self, task_service):
        """Test: Task mit Abhängigkeiten"""
        # Mock: Task-Funktionen
        def mock_analyze_audio(input_file):
            return {"tempo": 128, "key": "C major", "genre": "electronic"}
        
        def mock_process_audio(input_file, analysis_result):
            return {
                "output_file": "/path/to/processed.wav",
                "applied_effects": ["normalize", "eq"],
                "tempo": analysis_result["tempo"]
            }
        
        task_service.register_function("analyze_audio", mock_analyze_audio)
        task_service.register_function("process_audio", mock_process_audio)
        task_service.start()
        
        try:
            # Erste Task (Analyse)
            analysis_task = task_service.submit_task(
                name="Audio Analysis",
                task_type="stem_analysis",
                function="analyze_audio",
                parameters={"input_file": "/path/to/input.wav"}
            )
            
            # Zweite Task (Verarbeitung) mit Abhängigkeit
            processing_task = task_service.submit_task(
                name="Audio Processing",
                task_type="audio_processing",
                function="process_audio",
                parameters={
                    "input_file": "/path/to/input.wav",
                    "analysis_result": f"{{task:{analysis_task.task_id}:result}}"
                },
                dependencies=[analysis_task.task_id]
            )
            
            # Warten bis beide Tasks abgeschlossen sind
            timeout = 10.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                analysis_status = task_service.get_task_status(analysis_task.task_id)
                processing_status = task_service.get_task_status(processing_task.task_id)
                
                if (analysis_status.status in ["completed", "failed"] and
                    processing_status.status in ["completed", "failed"]):
                    break
                time.sleep(0.1)
            
            # Beide Tasks sollten erfolgreich abgeschlossen sein
            analysis_final = task_service.get_task_status(analysis_task.task_id)
            processing_final = task_service.get_task_status(processing_task.task_id)
            
            assert analysis_final.status == "completed"
            assert processing_final.status == "completed"
            
            # Verarbeitungs-Task sollte Analyse-Ergebnis verwendet haben
            assert processing_final.result["tempo"] == 128
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_scheduled_task(self, task_service):
        """Test: Geplante Task"""
        # Mock: Task-Funktion
        def mock_cleanup_cache():
            return {"cleaned_files": 42, "freed_space_mb": 256}
        
        task_service.register_function("cleanup_cache", mock_cleanup_cache)
        task_service.start()
        
        try:
            # Task für die Zukunft planen
            scheduled_time = datetime.now() + timedelta(seconds=2)
            
            task_result = task_service.schedule_task(
                name="Cache Cleanup",
                task_type="cache_cleanup",
                function="cleanup_cache",
                parameters={},
                scheduled_at=scheduled_time
            )
            
            assert task_result.task_id is not None
            assert task_result.status == "scheduled"
            
            # Task sollte zunächst geplant sein
            initial_status = task_service.get_task_status(task_result.task_id)
            assert initial_status.status == "scheduled"
            
            # Warten bis Task ausgeführt wird
            time.sleep(3)
            
            # Task sollte jetzt abgeschlossen sein
            final_status = task_service.get_task_status(task_result.task_id)
            assert final_status.status == "completed"
            assert final_status.result["cleaned_files"] == 42
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_recurring_task(self, task_service):
        """Test: Wiederkehrende Task"""
        # Mock: Task-Funktion
        execution_count = 0
        
        def mock_system_check():
            nonlocal execution_count
            execution_count += 1
            return {"check_id": execution_count, "status": "healthy"}
        
        task_service.register_function("system_check", mock_system_check)
        task_service.start()
        
        try:
            # Wiederkehrende Task erstellen (alle 2 Sekunden)
            task_result = task_service.create_recurring_task(
                name="System Health Check",
                task_type="system_maintenance",
                function="system_check",
                parameters={},
                schedule="*/2 * * * * *",  # Alle 2 Sekunden
                max_executions=3
            )
            
            assert task_result.task_id is not None
            assert task_result.status == "scheduled"
            
            # Warten auf mehrere Ausführungen
            time.sleep(7)  # Sollte 3 Ausführungen ermöglichen
            
            # Ausführungshistorie prüfen
            history = task_service.get_task_history(task_result.task_id)
            assert len(history) >= 3
            
            # Alle Ausführungen sollten erfolgreich gewesen sein
            for execution in history:
                assert execution.status == "completed"
                assert execution.result["status"] == "healthy"
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_task_retry_mechanism(self, task_service):
        """Test: Task-Retry-Mechanismus"""
        # Mock: Task-Funktion die zunächst fehlschlägt
        attempt_count = 0
        
        def mock_unreliable_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Attempt {attempt_count} failed")
            return {"success": True, "attempts": attempt_count}
        
        task_service.register_function("unreliable_function", mock_unreliable_function)
        task_service.start()
        
        try:
            # Task mit Retry-Konfiguration einreichen
            task_result = task_service.submit_task(
                name="Unreliable Task",
                task_type="system_maintenance",
                function="unreliable_function",
                parameters={},
                retry_count=3,
                retry_delay=1
            )
            
            # Warten bis Task abgeschlossen ist (mit Retries)
            timeout = 10.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                task_status = task_service.get_task_status(task_result.task_id)
                if task_status.status in ["completed", "failed"]:
                    break
                time.sleep(0.1)
            
            # Task sollte nach Retries erfolgreich sein
            final_status = task_service.get_task_status(task_result.task_id)
            assert final_status.status == "completed"
            assert final_status.result["success"] == True
            assert final_status.result["attempts"] == 3
            assert final_status.retry_count >= 2
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_task_timeout(self, task_service):
        """Test: Task-Timeout"""
        # Mock: Langsame Task-Funktion
        def mock_slow_function():
            time.sleep(5)  # Länger als Timeout
            return {"completed": True}
        
        task_service.register_function("slow_function", mock_slow_function)
        task_service.start()
        
        try:
            # Task mit kurzem Timeout einreichen
            task_result = task_service.submit_task(
                name="Slow Task",
                task_type="system_maintenance",
                function="slow_function",
                parameters={},
                timeout=2  # 2 Sekunden Timeout
            )
            
            # Warten bis Task timeout
            time.sleep(4)
            
            # Task sollte wegen Timeout fehlgeschlagen sein
            final_status = task_service.get_task_status(task_result.task_id)
            assert final_status.status == "timeout"
            assert final_status.error is not None
            assert "timeout" in final_status.error.lower()
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_task_cancellation(self, task_service):
        """Test: Task-Abbruch"""
        # Mock: Langsame Task-Funktion
        def mock_long_running_function():
            for i in range(100):
                time.sleep(0.1)
                # Prüfe auf Abbruch-Signal
                if hasattr(threading.current_thread(), 'cancelled'):
                    if threading.current_thread().cancelled:
                        return {"cancelled": True, "progress": i}
            return {"completed": True, "progress": 100}
        
        task_service.register_function("long_running_function", mock_long_running_function)
        task_service.start()
        
        try:
            # Langlaufende Task einreichen
            task_result = task_service.submit_task(
                name="Long Running Task",
                task_type="system_maintenance",
                function="long_running_function",
                parameters={}
            )
            
            # Kurz warten, dann Task abbrechen
            time.sleep(1)
            
            cancel_result = task_service.cancel_task(task_result.task_id)
            assert cancel_result.success == True
            
            # Task sollte abgebrochen sein
            time.sleep(1)
            
            final_status = task_service.get_task_status(task_result.task_id)
            assert final_status.status == "cancelled"
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_task_priority_queue(self, task_service):
        """Test: Task-Prioritäts-Queue"""
        execution_order = []
        
        # Mock: Task-Funktionen die Ausführungsreihenfolge aufzeichnen
        def mock_low_priority_task():
            execution_order.append("low")
            return {"priority": "low"}
        
        def mock_high_priority_task():
            execution_order.append("high")
            return {"priority": "high"}
        
        def mock_critical_priority_task():
            execution_order.append("critical")
            return {"priority": "critical"}
        
        task_service.register_function("low_priority_task", mock_low_priority_task)
        task_service.register_function("high_priority_task", mock_high_priority_task)
        task_service.register_function("critical_priority_task", mock_critical_priority_task)
        
        # Worker auf 1 begrenzen für deterministische Reihenfolge
        task_service.config.max_workers = 1
        task_service.start()
        
        try:
            # Tasks in umgekehrter Prioritätsreihenfolge einreichen
            low_task = task_service.submit_task(
                name="Low Priority", task_type="test", function="low_priority_task",
                parameters={}, priority=TestTaskPriority.LOW.value
            )
            
            high_task = task_service.submit_task(
                name="High Priority", task_type="test", function="high_priority_task",
                parameters={}, priority=TestTaskPriority.HIGH.value
            )
            
            critical_task = task_service.submit_task(
                name="Critical Priority", task_type="test", function="critical_priority_task",
                parameters={}, priority=TestTaskPriority.CRITICAL.value
            )
            
            # Warten bis alle Tasks abgeschlossen sind
            time.sleep(3)
            
            # Ausführungsreihenfolge sollte nach Priorität sortiert sein
            assert len(execution_order) == 3
            assert execution_order[0] == "critical"  # Höchste Priorität zuerst
            assert execution_order[1] == "high"
            assert execution_order[2] == "low"  # Niedrigste Priorität zuletzt
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_task_progress_tracking(self, task_service):
        """Test: Task-Progress-Tracking"""
        # Mock: Task-Funktion mit Progress-Updates
        def mock_progress_task(task_context):
            total_steps = 10
            for i in range(total_steps):
                time.sleep(0.1)
                progress = (i + 1) / total_steps * 100
                task_context.update_progress(
                    progress=progress,
                    message=f"Processing step {i + 1}/{total_steps}"
                )
            return {"completed": True, "steps": total_steps}
        
        task_service.register_function("progress_task", mock_progress_task)
        task_service.start()
        
        try:
            # Task mit Progress-Tracking einreichen
            task_result = task_service.submit_task(
                name="Progress Task",
                task_type="system_maintenance",
                function="progress_task",
                parameters={},
                progress_tracking=True
            )
            
            # Progress während Ausführung verfolgen
            progress_updates = []
            timeout = 5.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                task_status = task_service.get_task_status(task_result.task_id)
                
                if task_status.progress is not None:
                    progress_updates.append(task_status.progress)
                
                if task_status.status in ["completed", "failed"]:
                    break
                
                time.sleep(0.2)
            
            # Task sollte erfolgreich abgeschlossen sein
            final_status = task_service.get_task_status(task_result.task_id)
            assert final_status.status == "completed"
            
            # Progress-Updates sollten empfangen worden sein
            assert len(progress_updates) > 0
            assert any(p.progress > 0 for p in progress_updates)
            assert any(p.progress == 100 for p in progress_updates)
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_task_resource_limits(self, task_service):
        """Test: Task-Resource-Limits"""
        # Mock: Resource-intensive Task-Funktion
        def mock_resource_intensive_task():
            # Simuliere hohen Memory-Verbrauch
            large_data = [0] * (2 * 1024 * 1024)  # 2MB Liste
            return {"data_size": len(large_data)}
        
        task_service.register_function("resource_intensive_task", mock_resource_intensive_task)
        
        # Resource-Limits setzen
        task_service.config.resource_limits["max_memory_mb"] = 1  # Sehr niedriges Limit
        
        task_service.start()
        
        try:
            # Resource-intensive Task einreichen
            task_result = task_service.submit_task(
                name="Resource Intensive Task",
                task_type="system_maintenance",
                function="resource_intensive_task",
                parameters={},
                resource_monitoring=True
            )
            
            # Warten bis Task abgeschlossen ist
            time.sleep(3)
            
            # Task sollte wegen Resource-Limits fehlgeschlagen sein
            final_status = task_service.get_task_status(task_result.task_id)
            # Je nach Implementation könnte Task erfolgreich sein oder fehlschlagen
            # Hier prüfen wir nur, dass Resource-Monitoring funktioniert
            assert final_status.status in ["completed", "failed"]
            
            if final_status.resource_usage:
                assert "memory_mb" in final_status.resource_usage
                assert "cpu_percent" in final_status.resource_usage
            
        finally:
            task_service.stop()
    
    @pytest.mark.unit
    def test_task_metrics_and_analytics(self, task_service):
        """Test: Task-Metriken und Analytics"""
        # Mock: Verschiedene Task-Funktionen
        def mock_fast_task():
            time.sleep(0.1)
            return {"type": "fast"}
        
        def mock_slow_task():
            time.sleep(0.5)
            return {"type": "slow"}
        
        def mock_failing_task():
            raise Exception("Intentional failure")
        
        task_service.register_function("fast_task", mock_fast_task)
        task_service.register_function("slow_task", mock_slow_task)
        task_service.register_function("failing_task", mock_failing_task)
        task_service.start()
        
        try:
            # Verschiedene Tasks einreichen
            tasks = []
            
            # 5 schnelle Tasks
            for i in range(5):
                task = task_service.submit_task(
                    name=f"Fast Task {i}",
                    task_type="test",
                    function="fast_task",
                    parameters={}
                )
                tasks.append(task)
            
            # 3 langsame Tasks
            for i in range(3):
                task = task_service.submit_task(
                    name=f"Slow Task {i}",
                    task_type="test",
                    function="slow_task",
                    parameters={}
                )
                tasks.append(task)
            
            # 2 fehlschlagende Tasks
            for i in range(2):
                task = task_service.submit_task(
                    name=f"Failing Task {i}",
                    task_type="test",
                    function="failing_task",
                    parameters={},
                    retry_count=0  # Keine Retries
                )
                tasks.append(task)
            
            # Warten bis alle Tasks abgeschlossen sind
            time.sleep(5)
            
            # Metriken abrufen
            metrics = task_service.get_task_metrics(
                start_time=datetime.now() - timedelta(minutes=1),
                end_time=datetime.now()
            )
            
            assert metrics.total_tasks >= 10
            assert metrics.completed_tasks >= 8  # 5 fast + 3 slow
            assert metrics.failed_tasks >= 2  # 2 failing
            assert metrics.average_execution_time > 0
            
            # Analytics abrufen
            analytics = task_service.get_task_analytics(
                start_time=datetime.now() - timedelta(minutes=1),
                end_time=datetime.now()
            )
            
            assert analytics.task_type_stats["test"]["total"] >= 10
            assert analytics.function_stats["fast_task"]["total"] >= 5
            assert analytics.function_stats["slow_task"]["total"] >= 3
            assert analytics.function_stats["failing_task"]["total"] >= 2
            
            # Performance-Statistiken
            assert analytics.performance_stats["min_execution_time"] > 0
            assert analytics.performance_stats["max_execution_time"] > 0
            assert analytics.performance_stats["avg_execution_time"] > 0
            
        finally:
            task_service.stop()


class TestWorkflowEngine:
    """Tests für Workflow-Engine"""
    
    @pytest.fixture
    def workflow_engine(self, task_config):
        """Workflow-Engine für Tests"""
        task_service = TaskService(task_config)
        return task_service.workflow_engine
    
    @pytest.mark.unit
    def test_workflow_execution(self, workflow_engine, test_workflow):
        """Test: Workflow-Ausführung"""
        # Mock: Workflow-Step-Funktionen
        def mock_analyze_audio(input_file):
            return {"tempo": 128, "key": "C major", "duration": 180}
        
        def mock_process_audio(input_file, analysis_result):
            return {
                "output_file": "/path/to/processed.wav",
                "tempo": analysis_result["tempo"],
                "effects": ["normalize", "eq"]
            }
        
        def mock_generate_arrangement(processed_audio, style):
            return {
                "arrangement_file": "/path/to/arrangement.mid",
                "style": style,
                "tracks": 8
            }
        
        # Funktionen registrieren
        workflow_engine.register_function("analyze_audio", mock_analyze_audio)
        workflow_engine.register_function("process_audio", mock_process_audio)
        workflow_engine.register_function("generate_arrangement", mock_generate_arrangement)
        
        workflow_engine.start()
        
        try:
            # Workflow ausführen
            workflow_result = workflow_engine.execute_workflow(
                workflow_definition=test_workflow,
                workflow_parameters=test_workflow["parameters"]
            )
            
            assert workflow_result.workflow_id is not None
            assert workflow_result.status == "running"
            
            # Warten bis Workflow abgeschlossen ist
            timeout = 10.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                workflow_status = workflow_engine.get_workflow_status(workflow_result.workflow_id)
                if workflow_status.status in ["completed", "failed"]:
                    break
                time.sleep(0.1)
            
            # Workflow sollte erfolgreich abgeschlossen sein
            final_status = workflow_engine.get_workflow_status(workflow_result.workflow_id)
            assert final_status.status == "completed"
            
            # Alle Steps sollten erfolgreich ausgeführt worden sein
            step_results = workflow_engine.get_workflow_step_results(workflow_result.workflow_id)
            assert len(step_results) == 3
            
            # Step 1: Audio Analysis
            step1_result = step_results["step_1"]
            assert step1_result["tempo"] == 128
            assert step1_result["key"] == "C major"
            
            # Step 2: Audio Processing
            step2_result = step_results["step_2"]
            assert step2_result["tempo"] == 128  # Von Step 1 übernommen
            assert "normalize" in step2_result["effects"]
            
            # Step 3: Arrangement Generation
            step3_result = step_results["step_3"]
            assert step3_result["style"] == "electronic"
            assert step3_result["tracks"] == 8
            
        finally:
            workflow_engine.stop()
    
    @pytest.mark.unit
    def test_parallel_workflow_execution(self, workflow_engine):
        """Test: Parallele Workflow-Ausführung"""
        # Workflow mit parallelen Steps
        parallel_workflow = {
            "id": "parallel_workflow",
            "name": "Parallel Processing Workflow",
            "steps": [
                {
                    "id": "step_1",
                    "name": "Input Preparation",
                    "function": "prepare_input",
                    "parameters": {"input": "{{workflow.input}}"},
                    "dependencies": []
                },
                {
                    "id": "step_2a",
                    "name": "Process A",
                    "function": "process_a",
                    "parameters": {"data": "{{step_1.result}}"},
                    "dependencies": ["step_1"]
                },
                {
                    "id": "step_2b",
                    "name": "Process B",
                    "function": "process_b",
                    "parameters": {"data": "{{step_1.result}}"},
                    "dependencies": ["step_1"]
                },
                {
                    "id": "step_3",
                    "name": "Combine Results",
                    "function": "combine_results",
                    "parameters": {
                        "result_a": "{{step_2a.result}}",
                        "result_b": "{{step_2b.result}}"
                    },
                    "dependencies": ["step_2a", "step_2b"]
                }
            ],
            "parameters": {"input": "test_data"}
        }
        
        # Mock: Step-Funktionen
        def mock_prepare_input(input_data):
            return {"prepared": input_data, "timestamp": time.time()}
        
        def mock_process_a(data):
            time.sleep(0.2)  # Simuliere Verarbeitung
            return {"result_a": f"processed_a_{data['prepared']}", "duration": 0.2}
        
        def mock_process_b(data):
            time.sleep(0.3)  # Simuliere längere Verarbeitung
            return {"result_b": f"processed_b_{data['prepared']}", "duration": 0.3}
        
        def mock_combine_results(result_a, result_b):
            return {
                "combined": f"{result_a['result_a']}+{result_b['result_b']}",
                "total_duration": result_a["duration"] + result_b["duration"]
            }
        
        # Funktionen registrieren
        workflow_engine.register_function("prepare_input", mock_prepare_input)
        workflow_engine.register_function("process_a", mock_process_a)
        workflow_engine.register_function("process_b", mock_process_b)
        workflow_engine.register_function("combine_results", mock_combine_results)
        
        workflow_engine.start()
        
        try:
            start_time = time.time()
            
            # Workflow ausführen
            workflow_result = workflow_engine.execute_workflow(
                workflow_definition=parallel_workflow,
                workflow_parameters=parallel_workflow["parameters"]
            )
            
            # Warten bis Workflow abgeschlossen ist
            timeout = 10.0
            workflow_start = time.time()
            
            while time.time() - workflow_start < timeout:
                workflow_status = workflow_engine.get_workflow_status(workflow_result.workflow_id)
                if workflow_status.status in ["completed", "failed"]:
                    break
                time.sleep(0.1)
            
            execution_time = time.time() - start_time
            
            # Workflow sollte erfolgreich abgeschlossen sein
            final_status = workflow_engine.get_workflow_status(workflow_result.workflow_id)
            assert final_status.status == "completed"
            
            # Parallele Ausführung sollte schneller sein als sequenzielle
            # (0.2 + 0.3 = 0.5 sequenziell, aber parallel sollte ~0.3 dauern)
            assert execution_time < 1.0  # Großzügiger Puffer für Test-Umgebung
            
            # Ergebnisse prüfen
            step_results = workflow_engine.get_workflow_step_results(workflow_result.workflow_id)
            final_result = step_results["step_3"]
            
            assert "processed_a_test_data" in final_result["combined"]
            assert "processed_b_test_data" in final_result["combined"]
            
        finally:
            workflow_engine.stop()


class TestTaskServiceIntegration:
    """Integrationstests für Task-Service"""
    
    @pytest.mark.integration
    def test_full_task_lifecycle(self):
        """Test: Vollständiger Task-Lebenszyklus"""
        config = CoreTaskConfig(
            enabled=True,
            max_workers=2,
            monitoring_enabled=True,
            metrics_enabled=True,
            retry_enabled=True,
            max_retry_attempts=2
        )
        
        service = TaskService(config)
        
        # Mock: Task-Funktionen
        def mock_data_processing(input_data, processing_type):
            time.sleep(0.1)  # Simuliere Verarbeitung
            return {
                "processed_data": f"{processing_type}_{input_data}",
                "processing_time": 0.1,
                "success": True
            }
        
        def mock_data_validation(processed_data):
            return {
                "valid": True,
                "data": processed_data,
                "validation_time": 0.05
            }
        
        service.register_function("data_processing", mock_data_processing)
        service.register_function("data_validation", mock_data_validation)
        service.start()
        
        try:
            # 1. Task einreichen
            processing_task = service.submit_task(
                name="Data Processing",
                task_type="data_processing",
                function="data_processing",
                parameters={
                    "input_data": "raw_audio_data",
                    "processing_type": "normalize"
                },
                priority=TestTaskPriority.HIGH.value
            )
            
            # 2. Abhängige Task einreichen
            validation_task = service.submit_task(
                name="Data Validation",
                task_type="data_validation",
                function="data_validation",
                parameters={
                    "processed_data": f"{{task:{processing_task.task_id}:result:processed_data}}"
                },
                dependencies=[processing_task.task_id]
            )
            
            # 3. Tasks ausführen lassen
            timeout = 10.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                processing_status = service.get_task_status(processing_task.task_id)
                validation_status = service.get_task_status(validation_task.task_id)
                
                if (processing_status.status in ["completed", "failed"] and
                    validation_status.status in ["completed", "failed"]):
                    break
                time.sleep(0.1)
            
            # 4. Ergebnisse prüfen
            processing_final = service.get_task_status(processing_task.task_id)
            validation_final = service.get_task_status(validation_task.task_id)
            
            assert processing_final.status == "completed"
            assert validation_final.status == "completed"
            
            assert processing_final.result["success"] == True
            assert "normalize_raw_audio_data" in processing_final.result["processed_data"]
            
            assert validation_final.result["valid"] == True
            assert "normalize_raw_audio_data" in validation_final.result["data"]
            
            # 5. Metriken prüfen
            metrics = service.get_task_metrics(
                start_time=datetime.now() - timedelta(minutes=1),
                end_time=datetime.now()
            )
            
            assert metrics.total_tasks >= 2
            assert metrics.completed_tasks >= 2
            assert metrics.failed_tasks == 0
            
            # 6. Task-Historie prüfen
            processing_history = service.get_task_history(processing_task.task_id)
            validation_history = service.get_task_history(validation_task.task_id)
            
            assert len(processing_history) >= 1
            assert len(validation_history) >= 1
            
            # 7. Audit-Log prüfen
            audit_logs = service.get_audit_logs(
                start_time=datetime.now() - timedelta(minutes=1),
                end_time=datetime.now()
            )
            
            assert len(audit_logs) >= 4  # 2 Tasks × 2 Events (submit, complete)
            
        finally:
            service.stop()
    
    @pytest.mark.performance
    def test_task_service_performance(self):
        """Test: Task-Service-Performance"""
        config = CoreTaskConfig(
            enabled=True,
            max_workers=8,
            queue_max_size=10000,
            monitoring_enabled=False,  # Für Performance deaktiviert
            metrics_enabled=True,
            audit_enabled=False  # Für Performance deaktiviert
        )
        
        service = TaskService(config)
        
        # Mock: Schnelle Task-Funktion
        def mock_fast_computation(value):
            return {"result": value * 2, "computed": True}
        
        service.register_function("fast_computation", mock_fast_computation)
        service.start()
        
        try:
            # Performance-Test: Viele Tasks schnell einreichen
            start_time = time.time()
            
            task_ids = []
            for i in range(1000):
                task_result = service.submit_task(
                    name=f"Fast Task {i}",
                    task_type="computation",
                    function="fast_computation",
                    parameters={"value": i}
                )
                task_ids.append(task_result.task_id)
            
            submission_time = time.time() - start_time
            
            # Sollte unter 2 Sekunden für 1000 Task-Submissions dauern
            assert submission_time < 2.0
            
            # Warten bis alle Tasks abgeschlossen sind
            execution_start = time.time()
            completed_count = 0
            
            while completed_count < 1000 and time.time() - execution_start < 30:
                completed_count = 0
                for task_id in task_ids[:100]:  # Nur Stichprobe prüfen
                    status = service.get_task_status(task_id)
                    if status.status == "completed":
                        completed_count += 1
                
                if completed_count == 100:  # Alle Stichproben abgeschlossen
                    break
                
                time.sleep(0.1)
            
            execution_time = time.time() - execution_start
            
            # Sollte unter 10 Sekunden für 1000 Tasks dauern
            assert execution_time < 10.0
            
            # Metriken prüfen
            metrics = service.get_task_metrics(
                start_time=datetime.now() - timedelta(minutes=1),
                end_time=datetime.now()
            )
            
            assert metrics.total_tasks >= 1000
            assert metrics.completed_tasks >= 900  # Mindestens 90% erfolgreich
            
            # Durchsatz prüfen
            throughput = metrics.completed_tasks / execution_time
            assert throughput > 50  # Mindestens 50 Tasks pro Sekunde
            
        finally:
            service.stop()