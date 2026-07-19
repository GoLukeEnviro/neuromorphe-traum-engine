"""CRUD-Operationen für die Neuromorphe Traum-Engine v2.0

Diese Datei enthält alle Create, Read, Update, Delete-Operationen für die Datenbankmodelle.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
import numpy as np
from scipy.spatial.distance import cosine
from sqlalchemy.exc import IntegrityError

from .models import (
    Stem, GeneratedTrack, TrackStem, ProcessingJob,
    SystemMetrics, UserSession, ConfigurationSetting
)

logger = logging.getLogger(__name__)


class StemCRUD:
    """CRUD-Operationen für Stems"""
    
    @staticmethod
    def create_stem(db: Session, stem_data: Dict[str, Any]) -> Stem:
        """Erstellt einen neuen Stem"""
        try:
            stem = Stem(**stem_data)
            db.add(stem)
            db.commit()
            db.refresh(stem)
            logger.info(f"Stem erstellt: {stem.filename} (ID: {stem.id})")
            return stem
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Fehler beim Erstellen des Stems: {e}")
            raise
    
    @staticmethod
    def get_stem_by_id(db: Session, stem_id: int) -> Optional[Stem]:
        """Holt einen Stem anhand der ID"""
        return db.query(Stem).filter(Stem.id == stem_id).first()
    
    @staticmethod
    def get_stem_by_hash(db: Session, file_hash: str) -> Optional[Stem]:
        """Holt einen Stem anhand des Datei-Hashes"""
        return db.query(Stem).filter(Stem.file_hash == file_hash).first()
    
    @staticmethod
    def get_stem_by_filename(db: Session, filename: str) -> Optional[Stem]:
        """Holt einen Stem anhand des Dateinamens"""
        return db.query(Stem).filter(Stem.filename == filename).first()
    
    @staticmethod
    def get_stems(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        category: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        energy_level: Optional[str] = None,
        key: Optional[str] = None,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
        quality_min: Optional[float] = None,
        processing_status: Optional[str] = None,
        harmonic_complexity_min: Optional[float] = None,
        harmonic_complexity_max: Optional[float] = None,
        rhythmic_complexity_min: Optional[float] = None,
        rhythmic_complexity_max: Optional[float] = None,
        compatible_keys: Optional[List[str]] = None,
        audio_embedding_is_not_null: Optional[bool] = None,
        audio_embedding_is_null: Optional[bool] = None,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[Stem]:
        """Holt Stems mit Filteroptionen"""
        query = db.query(Stem)
        
        # Filter anwenden
        if category:
            query = query.filter(Stem.category.ilike(f"%{category}%"))
        if genre:
            query = query.filter(Stem.genre.ilike(f"%{genre}%"))
        if mood:
            query = query.filter(Stem.mood.ilike(f"%{mood}%"))
        if energy_level:
            query = query.filter(Stem.energy_level == energy_level)
        if key:
            query = query.filter(Stem.key == key)
        if compatible_keys:
            query = query.filter(Stem.key.in_(compatible_keys))
        if bpm_min is not None:
            query = query.filter(Stem.bpm >= bpm_min)
        if bpm_max is not None:
            query = query.filter(Stem.bpm <= bpm_max)
        if quality_min is not None:
            query = query.filter(Stem.quality_score >= quality_min)
        if processing_status:
            query = query.filter(Stem.processing_status == processing_status)
        if harmonic_complexity_min is not None:
            query = query.filter(Stem.harmonic_complexity >= harmonic_complexity_min)
        if harmonic_complexity_max is not None:
            query = query.filter(Stem.harmonic_complexity <= harmonic_complexity_max)
        if rhythmic_complexity_min is not None:
            query = query.filter(Stem.rhythmic_complexity >= rhythmic_complexity_min)
        if rhythmic_complexity_max is not None:
            query = query.filter(Stem.rhythmic_complexity <= rhythmic_complexity_max)
        if audio_embedding_is_not_null is True:
            query = query.filter(Stem.audio_embedding.isnot(None))
        if audio_embedding_is_null is True:
            query = query.filter(Stem.audio_embedding.is_(None))
        
        # Sortierung
        if hasattr(Stem, order_by):
            order_column = getattr(Stem, order_by)
            if order_desc:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(asc(order_column))
        
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def _calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Berechnet die Kosinus-Ähnlichkeit zwischen zwei Vektoren."""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0  # Vermeidet Division durch Null
        return dot_product / (norm_vec1 * norm_vec2)

    @staticmethod
    def search_stems_semantic(
        db: Session,
        query_embedding: List[float],
        limit: int = 50,
        similarity_threshold: float = 0.5
    ) -> List[Stem]:
        """Semantische Suche in Stems basierend auf CLAP-Embeddings."""
        logger.info(f"Starte semantische Suche mit Limit {limit} und Threshold {similarity_threshold}")

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Alle Stems mit Embeddings abrufen
        stems_with_embeddings = db.query(Stem).filter(
            Stem.processing_status == "completed",
            Stem.audio_embedding.isnot(None)
        ).all()

        results = []
        for stem in stems_with_embeddings:
            try:
                # Konvertiere JSON-Embedding zu NumPy-Array
                stem_embedding = np.array(stem.audio_embedding, dtype=np.float32)
                
                # Kosinus-Ähnlichkeit berechnen
                similarity = StemCRUD._calculate_cosine_similarity(query_vec, stem_embedding)
                
                if similarity >= similarity_threshold:
                    results.append((similarity, stem))
            except Exception as e:
                logger.warning(f"Fehler beim Verarbeiten des Embeddings für Stem ID {stem.id}: {e}")
                continue

        # Sortieren nach Ähnlichkeit (absteigend)
        results.sort(key=lambda x: x[0], reverse=True)

        # Nur die Stem-Objekte zurückgeben
        return [stem for similarity, stem in results[:limit]]

    @staticmethod
    async def search_stems_by_tags_and_category(
        db: Session,
        category: str,
        tags: List[str],
        limit: int = 1
    ) -> List[Stem]:
        """
        Sucht Stems, die zu einer Kategorie und einer Liste von Tags passen.
        Die Suche ist so konzipiert, dass sie die besten Treffer findet, auch wenn nicht alle Tags passen.
        """
        logger.debug(f"Suche Stems: Kategorie='{category}', Tags={tags}, Limit={limit}")
        
        # Basis-Query: Filter nach Kategorie und Status
        query = db.query(Stem).filter(
            Stem.category.ilike(f"%{category}%"),
            Stem.processing_status == "completed"
        )

        if not tags:
            # Wenn keine Tags gegeben sind, zufällige Stems der Kategorie zurückgeben
            return query.order_by(func.random()).limit(limit).all()

        # Erstelle eine gewichtete Suche basierend auf Tag-Übereinstimmung.
        # Jeder passende Tag erhöht die Relevanz.
        # HINWEIS: Dies ist eine vereinfachte Implementierung für SQLite.
        # In PostgreSQL oder mit FTS5 könnte dies effizienter gestaltet werden.
        
        conditions = []
        for tag in tags:
            conditions.append(Stem.tags.ilike(f'%"{tag}"%'))
        
        # Filtere nach Stems, die mindestens einen der Tags enthalten
        query = query.filter(or_(*conditions))
        return query.order_by(func.random()).limit(limit).all()
    
    @staticmethod
    def update_stem(db: Session, stem_id: int, update_data: Dict[str, Any]) -> Optional[Stem]:
        """Aktualisiert einen Stem"""
        stem = db.query(Stem).filter(Stem.id == stem_id).first()
        if not stem:
            return None
        
        for key, value in update_data.items():
            if hasattr(stem, key):
                setattr(stem, key, value)
        
        stem.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(stem)
        
        logger.info(f"Stem aktualisiert: {stem.filename} (ID: {stem.id})")
        return stem
    
    @staticmethod
    def get_stem_by_path(db: Session, file_path: str) -> Optional[Stem]:
        """Holt einen Stem anhand des Dateipfads"""
        return db.query(Stem).filter(
            or_(
                Stem.original_path == file_path,
                Stem.processed_path == file_path
            )
        ).first()
    
    @staticmethod
    def get_stems_by_category(db: Session, category: str, limit: int = 50) -> List[Stem]:
        """Holt Stems nach Kategorie"""
        return db.query(Stem).filter(
            Stem.category.ilike(f"%{category}%"),
            Stem.processing_status == "completed"
        ).order_by(func.random()).limit(limit).all()
    
    
    
    @staticmethod
    def get_compatible_keys(base_key: str) -> List[str]:
        """Gibt harmonisch kompatible Tonarten zurück"""
        # Definiere harmonische Kompatibilität basierend auf Quintenzirkel
        compatibility_map = {
            'C': ['C', 'Am', 'F', 'G', 'Dm', 'Em'],
            'Am': ['Am', 'C', 'F', 'G', 'Dm', 'Em'],
            'G': ['G', 'Em', 'C', 'D', 'Am', 'Bm'],
            'Em': ['Em', 'G', 'C', 'D', 'Am', 'Bm'],
            'D': ['D', 'Bm', 'G', 'A', 'Em', 'F#m'],
            'Bm': ['Bm', 'D', 'G', 'A', 'Em', 'F#m'],
            'A': ['A', 'F#m', 'D', 'E', 'Bm', 'C#m'],
            'F#m': ['F#m', 'A', 'D', 'E', 'Bm', 'C#m'],
            'E': ['E', 'C#m', 'A', 'B', 'F#m', 'G#m'],
            'C#m': ['C#m', 'E', 'A', 'B', 'F#m', 'G#m'],
            'B': ['B', 'G#m', 'E', 'F#', 'C#m', 'D#m'],
            'G#m': ['G#m', 'B', 'E', 'F#', 'C#m', 'D#m'],
            'F#': ['F#', 'D#m', 'B', 'C#', 'G#m', 'A#m'],
            'D#m': ['D#m', 'F#', 'B', 'C#', 'G#m', 'A#m'],
            'F': ['F', 'Dm', 'Bb', 'C', 'Gm', 'Am'],
            'Dm': ['Dm', 'F', 'Bb', 'C', 'Gm', 'Am'],
            'Bb': ['Bb', 'Gm', 'F', 'Eb', 'Dm', 'Cm'],
            'Gm': ['Gm', 'Bb', 'F', 'Eb', 'Dm', 'Cm'],
            'Eb': ['Eb', 'Cm', 'Bb', 'Ab', 'Gm', 'Fm'],
            'Cm': ['Cm', 'Eb', 'Bb', 'Ab', 'Gm', 'Fm'],
            'Ab': ['Ab', 'Fm', 'Eb', 'Db', 'Cm', 'Bbm'],
            'Fm': ['Fm', 'Ab', 'Eb', 'Db', 'Cm', 'Bbm'],
            'Db': ['Db', 'Bbm', 'Ab', 'Gb', 'Fm', 'Ebm'],
            'Bbm': ['Bbm', 'Db', 'Ab', 'Gb', 'Fm', 'Ebm'],
            'Gb': ['Gb', 'Ebm', 'Db', 'Cb', 'Bbm', 'Abm'],
            'Ebm': ['Ebm', 'Gb', 'Db', 'Cb', 'Bbm', 'Abm']
        }
        
        return compatibility_map.get(base_key, [base_key])
    
    @staticmethod
    def search_harmonically_compatible_stems(
        db: Session,
        base_key: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Stem]:
        """Sucht harmonisch kompatible Stems basierend auf der Basis-Tonart"""
        compatible_keys = StemCRUD.get_compatible_keys(base_key)
        
        query = db.query(Stem).filter(
            Stem.processing_status == "completed",
            Stem.key.in_(compatible_keys)
        )
        
        if category:
            query = query.filter(Stem.category.ilike(f"%{category}%"))
        
        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append(Stem.tags.ilike(f'%"{tag}"%'))
            if tag_conditions:
                query = query.filter(or_(*tag_conditions))
        
        return query.order_by(desc(Stem.quality_score)).limit(limit).all()
    
    @staticmethod
    def delete_stem(db: Session, stem_id: int) -> bool:
        """Löscht einen Stem"""
        stem = db.query(Stem).filter(Stem.id == stem_id).first()
        if not stem:
            return False
        
        # Prüfen ob Stem in Tracks verwendet wird
        track_usage = db.query(TrackStem).filter(TrackStem.stem_id == stem_id).first()
        if track_usage:
            logger.warning(f"Stem {stem_id} wird in Tracks verwendet und kann nicht gelöscht werden")
            return False
        
        db.delete(stem)
        db.commit()
        
        logger.info(f"Stem gelöscht: {stem.filename} (ID: {stem.id})")
        return True
    
    @staticmethod
    def get_stem_statistics(db: Session) -> Dict[str, Any]:
        """Holt Statistiken über Stems"""
        total_stems = db.query(func.count(Stem.id)).scalar()
        processed_stems = db.query(func.count(Stem.id)).filter(
            Stem.processing_status == "completed"
        ).scalar()
        
        # Durchschnittliche Qualität
        avg_quality = db.query(func.avg(Stem.quality_score)).filter(
            Stem.quality_score.isnot(None)
        ).scalar()
        
        # Verteilung nach Kategorien
        category_dist = db.query(
            Stem.category,
            func.count(Stem.id).label('count')
        ).filter(
            Stem.category.isnot(None)
        ).group_by(Stem.category).all()
        
        # Verteilung nach Genres
        genre_dist = db.query(
            Stem.genre,
            func.count(Stem.id).label('count')
        ).filter(
            Stem.genre.isnot(None)
        ).group_by(Stem.genre).all()
        
        return {
            'total_stems': total_stems,
            'processed_stems': processed_stems,
            'processing_rate': processed_stems / total_stems if total_stems > 0 else 0,
            'average_quality': float(avg_quality) if avg_quality else 0,
            'category_distribution': {item.category: item.count for item in category_dist},
            'genre_distribution': {item.genre: item.count for item in genre_dist}
        }

    @staticmethod
    def get_stem_count(db: Session, path_pattern: Optional[str] = None) -> int:
        """Holt die Anzahl der Stems, optional gefiltert nach Pfadmuster."""
        query = db.query(func.count(Stem.id))
        if path_pattern:
            query = query.filter(or_(
                Stem.original_path.ilike(f"%{path_pattern}%"),
                Stem.processed_path.ilike(f"%{path_pattern}%")
            ))
        return query.scalar()

    @staticmethod
    def search_stems_by_path_pattern(db: Session, path_pattern: str, limit: int = 50) -> List[Stem]:
        """Sucht Stems basierend auf einem Pfadmuster."""
        query = db.query(Stem).filter(or_(
            Stem.original_path.ilike(f"%{path_pattern}%"),
            Stem.processed_path.ilike(f"%{path_pattern}%")
        ))
        return query.limit(limit).all()


class GeneratedTrackCRUD:
    """CRUD-Operationen für generierte Tracks"""
    
    @staticmethod
    def create_track(db: Session, track_data: Dict[str, Any]) -> GeneratedTrack:
        """Erstellt einen neuen Track"""
        try:
            track = GeneratedTrack(**track_data)
            db.add(track)
            db.commit()
            db.refresh(track)
            logger.info(f"Track erstellt: {track.title} (ID: {track.id})")
            return track
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Fehler beim Erstellen des Tracks: {e}")
            raise
    
    @staticmethod
    def get_track_by_id(db: Session, track_id: int) -> Optional[GeneratedTrack]:
        """Holt einen Track anhand der ID"""
        return db.query(GeneratedTrack).filter(GeneratedTrack.id == track_id).first()
    
    @staticmethod
    def get_tracks(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        energy: Optional[str] = None,
        status: Optional[str] = None,
        quality_rating: Optional[str] = None,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[GeneratedTrack]:
        """Holt Tracks mit Filteroptionen"""
        query = db.query(GeneratedTrack)
        
        # Filter anwenden
        if genre:
            query = query.filter(GeneratedTrack.target_genre.ilike(f"%{genre}%"))
        if mood:
            query = query.filter(GeneratedTrack.target_mood.ilike(f"%{mood}%"))
        if energy:
            query = query.filter(GeneratedTrack.target_energy == energy)
        if status:
            query = query.filter(GeneratedTrack.generation_status == status)
        if quality_rating:
            query = query.filter(GeneratedTrack.quality_rating == quality_rating)
        
        # Sortierung
        if hasattr(GeneratedTrack, order_by):
            order_column = getattr(GeneratedTrack, order_by)
            if order_desc:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(asc(order_column))
        
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def update_track(db: Session, track_id: int, update_data: Dict[str, Any]) -> Optional[GeneratedTrack]:
        """Aktualisiert einen Track"""
        track = db.query(GeneratedTrack).filter(GeneratedTrack.id == track_id).first()
        if not track:
            return None
        
        for key, value in update_data.items():
            if hasattr(track, key):
                setattr(track, key, value)
        
        track.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(track)
        
        logger.info(f"Track aktualisiert: {track.title} (ID: {track.id})")
        return track
    
    @staticmethod
    def delete_track(db: Session, track_id: int) -> bool:
        """Löscht einen Track"""
        track = db.query(GeneratedTrack).filter(GeneratedTrack.id == track_id).first()
        if not track:
            return False
        
        # Zuerst TrackStem-Verknüpfungen löschen
        db.query(TrackStem).filter(TrackStem.track_id == track_id).delete()
        
        db.delete(track)
        db.commit()
        
        logger.info(f"Track gelöscht: {track.title} (ID: {track.id})")
        return True


class TrackStemCRUD:
    """CRUD-Operationen für Track-Stem-Verknüpfungen"""
    
    @staticmethod
    def create_track_stem(db: Session, track_stem_data: Dict[str, Any]) -> TrackStem:
        """Erstellt eine neue Track-Stem-Verknüpfung"""
        try:
            track_stem = TrackStem(**track_stem_data)
            db.add(track_stem)
            db.commit()
            db.refresh(track_stem)
            logger.debug(f"TrackStem erstellt: Track {track_stem.track_id}, Stem {track_stem.stem_id}")
            return track_stem
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Fehler beim Erstellen der TrackStem-Verknüpfung: {e}")
            raise
    
    @staticmethod
    def get_track_stems(db: Session, track_id: int) -> List[TrackStem]:
        """Holt alle Stems eines Tracks"""
        return db.query(TrackStem).filter(
            TrackStem.track_id == track_id
        ).order_by(TrackStem.start_time).all()
    
    @staticmethod
    def get_stem_usage(db: Session, stem_id: int) -> List[TrackStem]:
        """Holt alle Tracks, die einen bestimmten Stem verwenden"""
        return db.query(TrackStem).filter(TrackStem.stem_id == stem_id).all()
    
    @staticmethod
    def update_track_stem(db: Session, track_stem_id: int, update_data: Dict[str, Any]) -> Optional[TrackStem]:
        """Aktualisiert eine Track-Stem-Verknüpfung"""
        track_stem = db.query(TrackStem).filter(TrackStem.id == track_stem_id).first()
        if not track_stem:
            return None
        
        for key, value in update_data.items():
            if hasattr(track_stem, key):
                setattr(track_stem, key, value)
        
        db.commit()
        db.refresh(track_stem)
        return track_stem
    
    @staticmethod
    def delete_track_stem(db: Session, track_stem_id: int) -> bool:
        """Löscht eine Track-Stem-Verknüpfung"""
        track_stem = db.query(TrackStem).filter(TrackStem.id == track_stem_id).first()
        if not track_stem:
            return False
        
        db.delete(track_stem)
        db.commit()
        return True


class ProcessingJobCRUD:
    """CRUD-Operationen für Verarbeitungsaufträge"""
    
    @staticmethod
    def create_job(db: Session, job_data: Dict[str, Any]) -> ProcessingJob:
        """Erstellt einen neuen Verarbeitungsauftrag"""
        try:
            job = ProcessingJob(**job_data)
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(f"Verarbeitungsauftrag erstellt: {job.job_type} (ID: {job.id})")
            return job
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Fehler beim Erstellen des Verarbeitungsauftrags: {e}")
            raise
    
    @staticmethod
    def get_job_by_id(db: Session, job_id: int) -> Optional[ProcessingJob]:
        """Holt einen Verarbeitungsauftrag anhand der ID"""
        return db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    
    @staticmethod
    def get_pending_jobs(db: Session, job_type: Optional[str] = None, limit: int = 10) -> List[ProcessingJob]:
        """Holt ausstehende Verarbeitungsaufträge"""
        query = db.query(ProcessingJob).filter(
            ProcessingJob.job_status == "pending"
        )
        
        if job_type:
            query = query.filter(ProcessingJob.job_type == job_type)
        
        return query.order_by(
            asc(ProcessingJob.priority),
            asc(ProcessingJob.created_at)
        ).limit(limit).all()
    
    @staticmethod
    def update_job_status(
        db: Session,
        job_id: int,
        status: str,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[ProcessingJob]:
        """Aktualisiert den Status eines Verarbeitungsauftrags"""
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            return None
        
        job.job_status = status
        
        if progress is not None:
            job.progress_percentage = progress
        if current_step is not None:
            job.current_step = current_step
        if output_data is not None:
            job.output_data = output_data
        if error_message is not None:
            job.error_message = error_message
        
        # Zeitstempel setzen
        if status == "running" and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in ["completed", "failed"]:
            job.completed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(job)
        return job
    
    @staticmethod
    def get_job_statistics(db: Session, hours: int = 24) -> Dict[str, Any]:
        """Holt Statistiken über Verarbeitungsaufträge"""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Gesamtstatistiken
        total_jobs = db.query(func.count(ProcessingJob.id)).filter(
            ProcessingJob.created_at >= since
        ).scalar()
        
        completed_jobs = db.query(func.count(ProcessingJob.id)).filter(
            and_(
                ProcessingJob.created_at >= since,
                ProcessingJob.job_status == "completed"
            )
        ).scalar()
        
        failed_jobs = db.query(func.count(ProcessingJob.id)).filter(
            and_(
                ProcessingJob.created_at >= since,
                ProcessingJob.job_status == "failed"
            )
        ).scalar()
        
        # Durchschnittliche Verarbeitungszeit
        avg_processing_time = db.query(
            func.avg(
                func.julianday(ProcessingJob.completed_at) - 
                func.julianday(ProcessingJob.started_at)
            ) * 24 * 3600  # Konvertierung zu Sekunden
        ).filter(
            and_(
                ProcessingJob.created_at >= since,
                ProcessingJob.job_status == "completed",
                ProcessingJob.started_at.isnot(None),
                ProcessingJob.completed_at.isnot(None)
            )
        ).scalar()
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'success_rate': completed_jobs / total_jobs if total_jobs > 0 else 0,
            'average_processing_time_seconds': float(avg_processing_time) if avg_processing_time else 0
        }


class SystemMetricsCRUD:
    """CRUD-Operationen für System-Metriken"""
    
    @staticmethod
    def record_metric(
        db: Session,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> SystemMetrics:
        """Zeichnet eine System-Metrik auf"""
        metric = SystemMetrics(
            metric_type=metric_type,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            context=context,
            tags=tags
        )
        
        db.add(metric)
        db.commit()
        db.refresh(metric)
        return metric
    
    @staticmethod
    def get_metrics(
        db: Session,
        metric_type: Optional[str] = None,
        metric_name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[SystemMetrics]:
        """Holt System-Metriken"""
        query = db.query(SystemMetrics)
        
        if metric_type:
            query = query.filter(SystemMetrics.metric_type == metric_type)
        if metric_name:
            query = query.filter(SystemMetrics.metric_name == metric_name)
        if since:
            query = query.filter(SystemMetrics.timestamp >= since)
        
        return query.order_by(desc(SystemMetrics.timestamp)).limit(limit).all()
    
    @staticmethod
    def cleanup_old_metrics(db: Session, days: int = 30) -> int:
        """Löscht alte Metriken"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        deleted_count = db.query(SystemMetrics).filter(
            SystemMetrics.timestamp < cutoff_date
        ).delete()
        
        db.commit()
        logger.info(f"Alte Metriken gelöscht: {deleted_count} Einträge")
        return deleted_count


class ConfigurationCRUD:
    """CRUD-Operationen für Konfigurationseinstellungen"""
    
    @staticmethod
    def get_setting(
        db: Session,
        category: str,
        key: str
    ) -> Optional[ConfigurationSetting]:
        """Holt eine Konfigurationseinstellung"""
        return db.query(ConfigurationSetting).filter(
            and_(
                ConfigurationSetting.category == category,
                ConfigurationSetting.key == key
            )
        ).first()
    
    @staticmethod
    def set_setting(
        db: Session,
        category: str,
        key: str,
        value: Any,
        data_type: str = "string",
        description: Optional[str] = None
    ) -> ConfigurationSetting:
        """Setzt eine Konfigurationseinstellung"""
        setting = ConfigurationCRUD.get_setting(db, category, key)
        
        if setting:
            setting.value = value
            setting.data_type = data_type
            if description:
                setting.description = description
            setting.updated_at = datetime.utcnow()
        else:
            setting = ConfigurationSetting(
                category=category,
                key=key,
                value=value,
                data_type=data_type,
                description=description
            )
            db.add(setting)
        
        db.commit()
        db.refresh(setting)
        return setting
    
    @staticmethod
    def get_settings_by_category(db: Session, category: str) -> List[ConfigurationSetting]:
        """Holt alle Einstellungen einer Kategorie"""
        return db.query(ConfigurationSetting).filter(
            ConfigurationSetting.category == category
        ).order_by(ConfigurationSetting.key).all()


# Convenience functions for direct import
def get_stem_by_id(db: Session, stem_id: int) -> Optional[Stem]:
    """Holt einen Stem anhand der ID"""
    return StemCRUD.get_stem_by_id(db, stem_id)


def get_stem_by_path(db: Session, file_path: str) -> Optional[Stem]:
    """Holt einen Stem anhand des Dateipfads"""
    return StemCRUD.get_stem_by_path(db, file_path)


def get_stems_by_category(db: Session, category: str, limit: int = 50) -> List[Stem]:
    """Holt Stems nach Kategorie"""
    return StemCRUD.get_stems_by_category(db, category, limit)





def create_stem(db: Session, stem_data: Dict[str, Any]) -> Stem:
    """Erstellt einen neuen Stem"""
    return StemCRUD.create_stem(db, stem_data)


def get_processing_job(db: Session, job_id: int) -> Optional[ProcessingJob]:
    """Holt einen Verarbeitungsauftrag anhand der ID"""
    return ProcessingJobCRUD.get_job_by_id(db, job_id)


def update_processing_job(db: Session, job_id: int, update_data: Dict[str, Any]) -> Optional[ProcessingJob]:
    """Aktualisiert einen Verarbeitungsauftrag"""
    return ProcessingJobCRUD.update_job_status(db, job_id, **update_data)


def create_processing_job(db: Session, job_data: Dict[str, Any]) -> ProcessingJob:
    """Erstellt einen neuen Verarbeitungsauftrag"""
    return ProcessingJobCRUD.create_job(db, job_data)