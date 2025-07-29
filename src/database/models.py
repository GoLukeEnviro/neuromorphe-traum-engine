"""Datenbankmodelle für die Neuromorphe Traum-Engine v2.0

Diese Datei definiert alle SQLAlchemy-Modelle für die Datenbank.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from enum import Enum

class Base(DeclarativeBase):
    pass

class RenderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RenderFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class Stem(Base):
    """Stem-Modell für Audio-Dateien"""
    __tablename__ = "stems"
    
    # Primärschlüssel
    id = Column(Integer, primary_key=True, index=True)
    
    # Datei-Informationen
    filename = Column(String(255), nullable=False, index=True)
    original_path = Column(String(500), nullable=False)
    processed_path = Column(String(500), nullable=True)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    
    # Audio-Metadaten
    duration = Column(Float, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    channels = Column(Integer, nullable=False)
    bit_depth = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=False)
    
    # Musiktheorie-Tags
    musical_key = Column(String(10), nullable=True, index=True)
    bpm = Column(Float, nullable=True, index=True)
    time_signature = Column(String(10), nullable=True)
    
    # Kategorisierung
    category = Column(String(100), nullable=True, index=True)
    genre = Column(String(100), nullable=True, index=True)
    mood = Column(String(100), nullable=True, index=True)
    energy_level = Column(String(50), nullable=True, index=True)
    source = Column(String(20), nullable=False, default="original", index=True)  # original, separated, generated
    
    # Generierte Tags
    auto_tags = Column(SQLiteJSON, nullable=True)  # Liste von automatisch generierten Tags
    manual_tags = Column(SQLiteJSON, nullable=True)  # Liste von manuell hinzugefügten Tags
    
    # Neuromorphe Analyse
    audio_embedding = Column(SQLiteJSON, nullable=True)  # CLAP-Embedding als JSON
    semantic_analysis = Column(SQLiteJSON, nullable=True)  # Semantische Analyse-Ergebnisse
    pattern_analysis = Column(SQLiteJSON, nullable=True)  # Pattern-Analyse-Ergebnisse
    neural_features = Column(SQLiteJSON, nullable=True)  # Neurale Features
    perceptual_mapping = Column(SQLiteJSON, nullable=True)  # Perzeptuelle Zuordnung
    
    # Musikalische Analyse
    harmonic_complexity = Column(Float, nullable=True, index=True)  # Harmonische Komplexität (0.0 - 1.0)
    rhythmic_complexity = Column(Float, nullable=True, index=True)  # Rhythmische Komplexität (0.0 - 1.0)
    
    # Qualitätsbewertung
    quality_score = Column(Float, nullable=True, index=True)
    complexity_level = Column(String(20), nullable=True, index=True)  # low, medium, high
    recommended_usage = Column(SQLiteJSON, nullable=True)  # Liste von Verwendungsempfehlungen
    
    # Verarbeitungsstatus
    processing_status = Column(String(50), nullable=False, default="pending", index=True)
    processing_error = Column(Text, nullable=True)
    
    # Zeitstempel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Beziehungen
    track_stems = relationship("TrackStem", back_populates="stem")
    
    # Indizes für bessere Performance
    __table_args__ = (
        Index('idx_stem_search', 'category', 'genre', 'mood', 'energy_level'),
        Index('idx_stem_music', 'musical_key', 'bpm', 'time_signature'),
        Index('idx_stem_quality', 'quality_score', 'complexity_level'),
        Index('idx_stem_processing', 'processing_status', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Stem(id={self.id}, filename='{self.filename}', category='{self.category}')>"


class GeneratedTrack(Base):
    """Modell für generierte Tracks"""
    __tablename__ = "generated_tracks"
    
    # Primärschlüssel
    id = Column(Integer, primary_key=True, index=True)
    
    # Track-Informationen
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    original_prompt = Column(Text, nullable=False)
    
    # Datei-Informationen
    output_path = Column(String(500), nullable=True)
    preview_path = Column(String(500), nullable=True)
    file_hash = Column(String(64), nullable=True, unique=True)
    
    # Audio-Eigenschaften
    duration = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=False, default=44100)
    channels = Column(Integer, nullable=False, default=2)
    file_size = Column(Integer, nullable=True)
    
    # Musik-Parameter
    target_bpm = Column(Float, nullable=True)
    target_key = Column(String(10), nullable=True)
    target_genre = Column(String(100), nullable=True)
    target_mood = Column(String(100), nullable=True)
    target_energy = Column(String(50), nullable=True)
    
    # Arrangement-Plan
    arrangement_plan = Column(SQLiteJSON, nullable=True)  # Vollständiger Arrangement-Plan
    track_structure = Column(SQLiteJSON, nullable=True)  # Track-Struktur (Intro, Verse, etc.)
    
    # Rendering-Parameter
    rendering_settings = Column(SQLiteJSON, nullable=True)  # Rendering-Einstellungen
    master_effects = Column(SQLiteJSON, nullable=True)  # Master-Effekte
    
    # Status und Qualität
    generation_status = Column(String(50), nullable=False, default="pending", index=True)
    generation_error = Column(Text, nullable=True)
    quality_rating = Column(String(20), nullable=True)  # excellent, good, fair, poor
    
    # Metadaten
    track_metadata = Column(SQLiteJSON, nullable=True)  # Zusätzliche Metadaten
    tags = Column(SQLiteJSON, nullable=True)  # Track-Tags
    
    # Zeitstempel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    generated_at = Column(DateTime, nullable=True)
    
    # Beziehungen
    track_stems = relationship("TrackStem", back_populates="track")
    
    # Indizes
    __table_args__ = (
        Index('idx_track_search', 'target_genre', 'target_mood', 'target_energy'),
        Index('idx_track_music', 'target_key', 'target_bpm'),
        Index('idx_track_status', 'generation_status', 'created_at'),
        Index('idx_track_quality', 'quality_rating', 'generated_at'),
    )
    
    def __repr__(self):
        return f"<GeneratedTrack(id={self.id}, title='{self.title}', status='{self.generation_status}')>"


class TrackStem(Base):
    """Verknüpfungstabelle zwischen Tracks und Stems"""
    __tablename__ = "track_stems"
    
    # Primärschlüssel
    id = Column(Integer, primary_key=True, index=True)
    
    # Fremdschlüssel
    track_id = Column(Integer, ForeignKey("generated_tracks.id"), nullable=False, index=True)
    stem_id = Column(Integer, ForeignKey("stems.id"), nullable=False, index=True)
    
    # Position im Track
    section_name = Column(String(100), nullable=False)  # intro, verse, chorus, etc.
    layer_name = Column(String(100), nullable=False)  # kick, bass, lead, etc.
    start_time = Column(Float, nullable=False)  # Start-Zeit in Sekunden
    end_time = Column(Float, nullable=False)  # End-Zeit in Sekunden
    
    # Audio-Verarbeitung
    volume = Column(Float, nullable=False, default=1.0)  # Lautstärke (0.0 - 2.0)
    pan = Column(Float, nullable=False, default=0.0)  # Panorama (-1.0 bis 1.0)
    pitch_shift = Column(Float, nullable=False, default=0.0)  # Tonhöhenverschiebung in Halbtönen
    time_stretch = Column(Float, nullable=False, default=1.0)  # Zeitdehnung (0.5 - 2.0)
    
    # Effekte
    effects = Column(SQLiteJSON, nullable=True)  # Effekt-Parameter
    fade_in = Column(Float, nullable=False, default=0.0)  # Fade-In-Dauer in Sekunden
    fade_out = Column(Float, nullable=False, default=0.0)  # Fade-Out-Dauer in Sekunden
    
    # Metadaten
    usage_context = Column(String(100), nullable=True)  # Verwendungskontext
    importance = Column(Float, nullable=False, default=1.0)  # Wichtigkeit (0.0 - 1.0)
    
    # Zeitstempel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Beziehungen
    track = relationship("GeneratedTrack", back_populates="track_stems")
    stem = relationship("Stem", back_populates="track_stems")
    
    # Indizes
    __table_args__ = (
        Index('idx_track_stem_position', 'track_id', 'section_name', 'start_time'),
        Index('idx_track_stem_layer', 'track_id', 'layer_name'),
        Index('idx_stem_usage', 'stem_id', 'usage_context'),
    )
    
    def __repr__(self):
        return f"<TrackStem(track_id={self.track_id}, stem_id={self.stem_id}, section='{self.section_name}')>"


class ProcessingJob(Base):
    """Modell für Verarbeitungsaufträge"""
    __tablename__ = "processing_jobs"
    
    # Primärschlüssel
    id = Column(Integer, primary_key=True, index=True)
    
    # Job-Informationen
    job_type = Column(String(50), nullable=False, index=True)  # preprocess, generate, analyze
    job_status = Column(String(50), nullable=False, default="pending", index=True)
    priority = Column(Integer, nullable=False, default=5)  # 1 (hoch) bis 10 (niedrig)
    
    # Eingabedaten
    input_data = Column(SQLiteJSON, nullable=False)  # Job-spezifische Eingabedaten
    parameters = Column(SQLiteJSON, nullable=True)  # Zusätzliche Parameter
    
    # Ausgabedaten
    output_data = Column(SQLiteJSON, nullable=True)  # Job-Ergebnisse
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    
    # Progress-Tracking
    progress_percentage = Column(Float, nullable=False, default=0.0)
    current_step = Column(String(200), nullable=True)
    total_steps = Column(Integer, nullable=True)
    
    # Ressourcen-Verbrauch
    cpu_time = Column(Float, nullable=True)  # CPU-Zeit in Sekunden
    memory_peak = Column(Integer, nullable=True)  # Peak-Memory in MB
    
    # Zeitstempel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Indizes
    __table_args__ = (
        Index('idx_job_queue', 'job_status', 'priority', 'created_at'),
        Index('idx_job_type', 'job_type', 'job_status'),
        Index('idx_job_timing', 'created_at', 'completed_at'),
    )
    
    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, type='{self.job_type}', status='{self.job_status}')>"


class SystemMetrics(Base):
    """Modell für System-Metriken und Performance-Daten"""
    __tablename__ = "system_metrics"
    
    # Primärschlüssel
    id = Column(Integer, primary_key=True, index=True)
    
    # Metrik-Informationen
    metric_type = Column(String(50), nullable=False, index=True)  # cpu, memory, disk, processing
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # %, MB, seconds, etc.
    
    # Kontext
    context = Column(SQLiteJSON, nullable=True)  # Zusätzlicher Kontext
    tags = Column(SQLiteJSON, nullable=True)  # Tags für Gruppierung
    
    # Zeitstempel
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Indizes
    __table_args__ = (
        Index('idx_metrics_type_time', 'metric_type', 'timestamp'),
        Index('idx_metrics_name_time', 'metric_name', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(type='{self.metric_type}', name='{self.metric_name}', value={self.metric_value})>"


class UserSession(Base):
    """Modell für Benutzer-Sessions (für zukünftige Erweiterungen)"""
    __tablename__ = "user_sessions"
    
    # Primärschlüssel
    id = Column(Integer, primary_key=True, index=True)
    
    # Session-Informationen
    session_id = Column(String(64), nullable=False, unique=True, index=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6-kompatibel
    
    # Session-Daten
    session_data = Column(SQLiteJSON, nullable=True)  # Session-spezifische Daten
    preferences = Column(SQLiteJSON, nullable=True)  # Benutzer-Präferenzen
    
    # Aktivität
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Zeitstempel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    # Indizes
    __table_args__ = (
        Index('idx_session_active', 'is_active', 'last_activity'),
        Index('idx_session_expiry', 'expires_at'),
    )
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, session_id='{self.session_id[:8]}...', active={self.is_active})>"


class ConfigurationSetting(Base):
    """Modell für Konfigurationseinstellungen"""
    __tablename__ = "configuration_settings"
    
    # Primärschlüssel
    id = Column(Integer, primary_key=True, index=True)
    
    # Einstellungs-Informationen
    category = Column(String(100), nullable=False, index=True)  # audio, processing, ui, etc.
    key = Column(String(200), nullable=False, index=True)
    value = Column(SQLiteJSON, nullable=False)  # Flexibler Wert-Typ
    
    # Metadaten
    description = Column(Text, nullable=True)
    data_type = Column(String(50), nullable=False)  # string, integer, float, boolean, json
    is_user_configurable = Column(Boolean, nullable=False, default=True)
    requires_restart = Column(Boolean, nullable=False, default=False)
    
    # Validierung
    validation_rules = Column(SQLiteJSON, nullable=True)  # Validierungsregeln
    default_value = Column(SQLiteJSON, nullable=True)  # Standardwert
    
    # Zeitstempel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Eindeutigkeit
    __table_args__ = (
        Index('idx_config_category_key', 'category', 'key', unique=True),
        Index('idx_config_user', 'is_user_configurable'),
    )
    
    def __repr__(self):
        return f"<ConfigurationSetting(category='{self.category}', key='{self.key}')>"


# Hilfsfunktionen für Datenbankoperationen
def create_all_tables(engine):
    """Erstellt alle Tabellen in der Datenbank"""
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine):
    """Löscht alle Tabellen aus der Datenbank"""
    Base.metadata.drop_all(bind=engine)