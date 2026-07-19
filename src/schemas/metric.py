from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MetricType(str, Enum):
    """Typ einer System-Metrik"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    PROCESSING = "processing"
    NETWORK = "network"
    DATABASE = "database"
    API = "api"
    AUDIO = "audio"


class SystemMetricBase(BaseModel):
    """Basis-Schema für System-Metriken"""
    metric_type: MetricType
    metric_name: str = Field(..., min_length=1, max_length=100)
    metric_value: float
    metric_unit: Optional[str] = Field(None, max_length=20, description="Einheit wie %, MB, seconds, etc.")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Zusätzlicher Kontext")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Tags für Gruppierung")
    model_config = ConfigDict(use_enum_values=True)


class SystemMetricCreate(SystemMetricBase):
    """Schema für die Erstellung einer System-Metrik"""
    timestamp: Optional[datetime] = Field(None, description="Zeitstempel, falls nicht angegeben wird aktuelle Zeit verwendet")


class SystemMetricResponse(SystemMetricBase):
    """Schema für die Antwort mit Metrik-Daten"""
    id: int
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class SystemMetricSearch(BaseModel):
    """Schema für Metrik-Suche"""
    metric_type: Optional[MetricType] = None
    metric_name: Optional[str] = Field(None, max_length=100)
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    timestamp_after: Optional[datetime] = None
    timestamp_before: Optional[datetime] = None
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    skip: int = Field(0, ge=0)
    limit: int = Field(100, ge=1, le=1000)
    order_by: str = Field("timestamp", pattern=r"^(timestamp|metric_type|metric_name|metric_value)$")
    order_desc: bool = Field(True)
    model_config = ConfigDict(use_enum_values=True)


class MetricAggregation(BaseModel):
    """Schema für aggregierte Metriken"""
    metric_type: MetricType
    metric_name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    sum_value: float
    period_start: datetime
    period_end: datetime
    model_config = ConfigDict(use_enum_values=True)


class SystemMetricsBatch(BaseModel):
    """Schema für Batch-Erstellung von Metriken"""
    metrics: List[SystemMetricCreate] = Field(..., min_length=1, max_length=1000)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "metrics": [
                {
                    "metric_type": "cpu",
                    "metric_name": "usage_percent",
                    "metric_value": 75.5,
                    "metric_unit": "%",
                    "context": {"core": 0},
                    "tags": {"host": "server-01"}
                },
                {
                    "metric_type": "memory",
                    "metric_name": "used_mb",
                    "metric_unit": "MB",
                    "metric_value": 2048,
                    "tags": {"host": "server-01"}
                }
            ]
        }
    })


class SystemMetricsStats(BaseModel):
    """Schema für Metrik-Statistiken"""
    total_metrics: int
    unique_types: int
    unique_names: int
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    type_distribution: Dict[str, int] = Field(default_factory=dict)
    top_metrics_by_count: List[Dict[str, Any]] = Field(default_factory=list)
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_metrics": 50000,
            "unique_types": 8,
            "unique_names": 25,
            "period_start": "2024-01-01T00:00:00Z",
            "period_end": "2024-07-28T16:00:00Z",
            "type_distribution": {
                "cpu": 15000,
                "memory": 12000,
                "processing": 8000,
                "disk": 5000
            },
            "top_metrics_by_count": [
                {"metric_name": "cpu_usage_percent", "count": 8000},
                {"metric_name": "memory_used_mb", "count": 6000}
            ]
        }
    })