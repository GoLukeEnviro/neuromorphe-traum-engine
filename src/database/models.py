"""Datenbankmodelle für die Neuromorphe Traum-Engine v2.0

Diese Datei definiert alle SQLAlchemy-Modelle für die Datenbank.
"""

import math
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, relationship, synonym, validates
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.ext.hybrid import hybrid_property
from enum import Enum

class Base(DeclarativeBase):
    pass

#: Erwartete Dimension der CLAP-/Audio-Embeddings (siehe ``Stem.embeddings``).
EMBEDDING_DIM = 512

#: Schema-Version, die von ``DatabaseManager.get_schema_version`` gemeldet wird.
SCHEMA_VERSION = "2.0.0"

class RenderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RenderFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"

class StemType(str, Enum):
    """Kategorien eines Stems.

    Entspricht den Werten der Spalte ``stems.category``. Die Aufzählung wird
    nicht erzwungen — ``Stem.type`` bleibt ein normaler String —, damit die
    bestehende Datenbank und die API unverändert weiterlaufen.
    """

    KICK = "kick"
    BASS = "bass"
    HIHAT = "hihat"
    SNARE = "snare"
    PERCUSSION = "percussion"
    SYNTH = "synth"
    FX = "fx"
    ATMO = "atmo"
    LOOP = "loop"
    VOCAL = "vocal"
    UNKNOWN = "unknown"
    OTHER = "other"


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
    
    # ------------------------------------------------------------------
    # Kompatibilitäts-Aliase (rein additiv)
    #
    # Diese Synonyme und Hybrid-Properties mappen die historischen, in Tests
    # und Services verwendeten Feldnamen auf die realen Spalten des
    # ausgelieferten Schemas. Es werden KEINE Spalten umbenannt, entfernt
    # oder hinzugefügt — die API und die bestehende ``processed_database``
    # bleiben unverändert nutzbar.
    #   name       -> filename
    #   file_path  -> original_path
    #   type       -> category
    #   tempo      -> bpm
    #   key        -> musical_key
    #   tags       -> manual_tags
    #   features   -> neural_features
    #   embeddings -> audio_embedding (mit Längen-Validierung)
    # ------------------------------------------------------------------
    name = synonym("filename")
    file_path = synonym("original_path")
    type = synonym("category")
    tempo = synonym("bpm")
    key = synonym("musical_key")
    tags = synonym("manual_tags")
    features = synonym("neural_features")
    
    @hybrid_property
    def embeddings(self) -> Optional[List[float]]:
        """Historischer Name des Audio-/CLAP-Embeddings."""
        return self.audio_embedding
    
    @embeddings.setter
    def embeddings(self, value: Optional[List[float]]) -> None:
        if value is not None and len(value) != EMBEDDING_DIM:
            raise ValueError(
                f"embeddings must contain exactly {EMBEDDING_DIM} values, "
                f"got {len(value)}"
            )
        self.audio_embedding = value
    
    # Indizes für bessere Performance
    __table_args__ = (
        Index('idx_stem_search', 'category', 'genre', 'mood', 'energy_level'),
        Index('idx_stem_music', 'musical_key', 'bpm', 'time_signature'),
        Index('idx_stem_quality', 'quality_score', 'complexity_level'),
        Index('idx_stem_processing', 'processing_status', 'created_at'),
    )
    
    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Stem als Dictionary serialisieren.

        Enthält sowohl die realen Spaltennamen als auch die historischen
        Alias-Namen (name/type/tempo/...), damit beide Konsumenten-Gruppen
        weiterarbeiten.
        """
        return {
            "id": self.id,
            "filename": self.filename,
            "name": self.filename,
            "original_path": self.original_path,
            "file_path": self.original_path,
            "processed_path": self.processed_path,
            "file_hash": self.file_hash,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "file_size": self.file_size,
            "musical_key": self.musical_key,
            "key": self.musical_key,
            "bpm": self.bpm,
            "tempo": self.bpm,
            "time_signature": self.time_signature,
            "category": self.category,
            "type": self.category,
            "genre": self.genre,
            "mood": self.mood,
            "energy_level": self.energy_level,
            "source": self.source,
            "auto_tags": self.auto_tags,
            "manual_tags": self.manual_tags,
            "tags": self.manual_tags,
            "audio_embedding": self.audio_embedding,
            "embeddings": self.audio_embedding,
            "semantic_analysis": self.semantic_analysis,
            "pattern_analysis": self.pattern_analysis,
            "neural_features": self.neural_features,
            "features": self.neural_features,
            "perceptual_mapping": self.perceptual_mapping,
            "harmonic_complexity": self.harmonic_complexity,
            "rhythmic_complexity": self.rhythmic_complexity,
            "quality_score": self.quality_score,
            "complexity_level": self.complexity_level,
            "recommended_usage": self.recommended_usage,
            "processing_status": self.processing_status,
            "processing_error": self.processing_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "processed_at": self.processed_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stem":
        """Stem aus einem Dictionary erstellen (akzeptiert Alias-Namen)."""
        aliases = {
            "name": "filename",
            "file_path": "original_path",
            "type": "category",
            "tempo": "bpm",
            "key": "musical_key",
            "tags": "manual_tags",
            "features": "neural_features",
            "embeddings": "audio_embedding",
        }
        skip = {"id", "created_at", "updated_at", "processed_at"}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in skip:
                continue
            mapped = aliases.get(key, key)
            if mapped in cls.__table__.columns:
                kwargs[mapped] = value
        return cls(**kwargs)
    
    def update_metadata(self, update_data: Dict[str, Any]) -> None:
        """Metadaten aktualisieren (akzeptiert Alias-Namen)."""
        aliases = {
            "name": "filename",
            "file_path": "original_path",
            "type": "category",
            "tempo": "bpm",
            "key": "musical_key",
            "tags": "manual_tags",
            "features": "neural_features",
            "embeddings": "audio_embedding",
        }
        for key, value in update_data.items():
            mapped = aliases.get(key, key)
            if hasattr(type(self), mapped):
                setattr(self, mapped, value)
        self.updated_at = datetime.utcnow()
    
    def calculate_similarity(self, other: "Stem") -> float:
        """Kosinus-Ähnlichkeit zwischen den Embeddings zweier Stems (0.0-1.0).

        Ein fehlendes oder nulllanges Embedding ergibt 0.0.
        """
        a = self.audio_embedding
        b = getattr(other, "audio_embedding", None)
        if not a or not b:
            return 0.0
        
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += float(x) * float(y)
            norm_a += float(x) * float(x)
            norm_b += float(y) * float(y)
        
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        
        similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
        # Numerische Rundungsfehler abfangen
        return max(0.0, min(1.0, similarity))
    
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
    
    # ------------------------------------------------------------------
    # Konstruktor/Validierung
    #
    # ``stems`` ist bewusst KEINE Spalte: die Track<->Stem-Verknüpfung liegt
    # im echten Schema in ``track_stems``. Der Konstruktor akzeptiert die
    # historische Liste trotzdem, damit bestehende Aufrufer weiterlaufen.
    # ------------------------------------------------------------------
    def __init__(self, **kwargs: Any) -> None:
        stems = kwargs.pop("stems", None)
        super().__init__(**kwargs)
        self.stems = list(stems) if stems else []
    
    @property
    def stems(self) -> List[Any]:
        """Historische Liste der Stem-IDs/Objekte (nicht persistiert)."""
        return self.__dict__.setdefault("_stems", [])
    
    @stems.setter
    def stems(self, value: Any) -> None:
        self.__dict__["_stems"] = list(value) if value else []
    
    @validates("duration")
    def _validate_duration(self, key: str, value: Any) -> Any:
        if value is not None and value < 0:
            raise ValueError("duration must not be negative")
        return value
    
    @validates("original_prompt")
    def _validate_original_prompt(self, key: str, value: Any) -> Any:
        if value is None or not str(value).strip():
            raise ValueError("original_prompt must not be empty")
        return value
    
    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Track als Dictionary serialisieren (inkl. historischer Alias-Namen)."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "original_prompt": self.original_prompt,
            "prompt": self.original_prompt,
            "output_path": self.output_path,
            "preview_path": self.preview_path,
            "file_hash": self.file_hash,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "file_size": self.file_size,
            "target_bpm": self.target_bpm,
            "target_key": self.target_key,
            "target_genre": self.target_genre,
            "genre": self.target_genre,
            "target_mood": self.target_mood,
            "target_energy": self.target_energy,
            "arrangement_plan": self.arrangement_plan,
            "track_structure": self.track_structure,
            "rendering_settings": self.rendering_settings,
            "master_effects": self.master_effects,
            "generation_status": self.generation_status,
            "status": self.generation_status,
            "generation_error": self.generation_error,
            "quality_rating": self.quality_rating,
            "track_metadata": self.track_metadata,
            "metadata": self.track_metadata,
            "tags": self.tags,
            "stems": list(self.stems),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "generated_at": self.generated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedTrack":
        """Track aus einem Dictionary erstellen (akzeptiert Alias-Namen)."""
        aliases = {
            "prompt": "original_prompt",
            "genre": "target_genre",
            "metadata": "track_metadata",
            "status": "generation_status",
        }
        skip = {"id", "created_at", "updated_at", "stems"}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in skip:
                continue
            mapped = aliases.get(key, key)
            if mapped in cls.__table__.columns:
                kwargs[mapped] = value
        if "stems" in data:
            kwargs["stems"] = data["stems"]
        return cls(**kwargs)
    
    def __repr__(self):
        return f"<GeneratedTrack(id={self.id}, title='{self.title}', status='{self.generation_status}')>"


class Arrangement(Base):
    """Modell für Arrangements (Prompt + Struktur eines generierten Tracks).

    Eigene Tabelle ``arrangements``. Die Spalten sind kompatibel zum
    historischen ``Arrangement``-Modell (prompt/duration/genre/…), damit
    bestehende Aufrufer unverändert weiterarbeiten.
    """
    __tablename__ = "arrangements"
    
    # Primärschlüssel
    id = Column(String(36), primary_key=True, index=True)
    
    # Inhalt
    prompt = Column(Text, nullable=False)
    duration = Column(Float, nullable=True)
    genre = Column(String(100), nullable=True, index=True)
    track_structure = Column(SQLiteJSON, nullable=True)
    stems = Column(SQLiteJSON, nullable=True)
    arrangement_metadata = Column(SQLiteJSON, nullable=True)
    
    # Status
    status = Column(String(50), nullable=False, default="pending", index=True)
    error_message = Column(Text, nullable=True)
    
    # Zeitstempel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __init__(self, **kwargs: Any) -> None:
        # Historischer Name ``metadata`` kollidiert mit der Deklarativ-API,
        # daher auf ``arrangement_metadata`` mappen.
        if "metadata" in kwargs:
            kwargs.setdefault("arrangement_metadata", kwargs.pop("metadata"))
        if "id" not in kwargs or kwargs.get("id") is None:
            kwargs["id"] = str(uuid4())
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Arrangement als Dictionary serialisieren."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "duration": self.duration,
            "genre": self.genre,
            "track_structure": self.track_structure,
            "stems": self.stems,
            "metadata": self.arrangement_metadata,
            "arrangement_metadata": self.arrangement_metadata,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Arrangement":
        """Arrangement aus einem Dictionary erstellen."""
        aliases = {"metadata": "arrangement_metadata"}
        skip = {"created_at", "updated_at"}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in skip:
                continue
            mapped = aliases.get(key, key)
            if mapped in cls.__table__.columns:
                kwargs[mapped] = value
        return cls(**kwargs)
    
    def __repr__(self):
        return f"<Arrangement(id={self.id}, genre='{self.genre}', status='{self.status}')>"


class RenderJob(Base):
    """Modell für Render-Jobs.

    Eigene Tabelle ``render_jobs``. Die Spalte ``arrangement_id`` referenziert
    logisch eine ``Arrangement.id``; es wird bewusst KEIN harter Fremdschlüssel
    gesetzt, damit auch die historischen ``generated_tracks``-IDs (Integer)
    unverändert übergeben werden können.
    """
    __tablename__ = "render_jobs"
    
    # Primärschlüssel
    id = Column(String(36), primary_key=True, index=True)
    
    # Zuordnung
    arrangement_id = Column(String(36), nullable=True, index=True)
    
    # Render-Parameter
    format = Column(String(20), nullable=False, default="wav")
    quality = Column(String(20), nullable=False, default="high")
    options = Column(SQLiteJSON, nullable=True)
    
    # Status
    status = Column(String(50), nullable=False, default=RenderStatus.PENDING.value, index=True)
    progress = Column(Float, nullable=False, default=0.0)
    error_message = Column(Text, nullable=True)
    output_path = Column(String(500), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    
    # Zeitstempel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    def __init__(self, **kwargs: Any) -> None:
        if "id" not in kwargs or kwargs.get("id") is None:
            kwargs["id"] = str(uuid4())
        # Enum-Werte auf reine Strings normalisieren
        for key in ("format", "status"):
            if key in kwargs and isinstance(kwargs[key], Enum):
                kwargs[key] = kwargs[key].value
        super().__init__(**kwargs)
    
    @validates("progress")
    def _validate_progress(self, key: str, value: Any) -> Any:
        if value is None:
            return 0.0
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("progress must be between 0.0 and 1.0")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """RenderJob als Dictionary serialisieren."""
        return {
            "id": self.id,
            "arrangement_id": self.arrangement_id,
            "format": self.format,
            "quality": self.quality,
            "options": self.options,
            "status": self.status,
            "progress": self.progress,
            "error_message": self.error_message,
            "output_path": self.output_path,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderJob":
        """RenderJob aus einem Dictionary erstellen."""
        skip = {"created_at", "updated_at", "started_at", "completed_at"}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in skip or key not in cls.__table__.columns:
                continue
            kwargs[key] = value
        return cls(**kwargs)
    
    def update_status(
        self,
        status: Any,
        progress: Optional[float] = None,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Status (und optional Progress/Ausgabe/Fehler) aktualisieren."""
        if isinstance(status, Enum):
            status = status.value
        self.status = status
        
        if progress is not None:
            self.progress = progress
        
        if output_path is not None:
            self.output_path = output_path
        
        if error_message is not None:
            self.error_message = error_message
        
        if status == RenderStatus.PROCESSING.value and self.started_at is None:
            self.started_at = datetime.utcnow()
        
        if status in (RenderStatus.COMPLETED.value, RenderStatus.FAILED.value):
            self.completed_at = datetime.utcnow()
        
        self.updated_at = datetime.utcnow()
    
    def retry(self) -> None:
        """Job für einen erneuten Versuch zurücksetzen."""
        self.status = RenderStatus.PENDING.value
        self.progress = 0.0
        self.error_message = None
        self.output_path = None
        self.started_at = None
        self.completed_at = None
        self.retry_count = (self.retry_count or 0) + 1
        self.updated_at = datetime.utcnow()
    
    def get_render_time(self) -> Optional[float]:
        """Renderdauer in Sekunden (None, solange nicht gestartet/beendet)."""
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()
    
    def get_estimated_remaining_time(self) -> Optional[float]:
        """Geschätzte verbleibende Renderzeit in Sekunden."""
        if self.started_at is None or not self.progress or self.progress <= 0.0:
            return None
        
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        if elapsed <= 0.0:
            return None
        
        total_estimate = elapsed / self.progress
        return max(0.0, total_estimate - elapsed)
    
    def __repr__(self):
        return f"<RenderJob(id={self.id}, format='{self.format}', status='{self.status}')>"


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