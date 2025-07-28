import sqlite3
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class DatabaseService:
    """Service for SQLite database operations"""
    
    def __init__(self, db_path: str = "stems.db"):
        self.db_path = Path(db_path)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Legacy audio_files table for backward compatibility
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    category TEXT,
                    bpm INTEGER,
                    duration REAL,
                    sample_rate INTEGER,
                    channels INTEGER,
                    file_size INTEGER,
                    has_embedding BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # New stems table with source column
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stems (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'original',
                    category TEXT,
                    subcategory TEXT,
                    bpm REAL,
                    key_signature TEXT,
                    time_signature TEXT,
                    duration REAL,
                    sample_rate INTEGER,
                    channels INTEGER,
                    bit_depth INTEGER,
                    file_size INTEGER,
                    file_format TEXT,
                    rms_energy REAL,
                    peak_amplitude REAL,
                    zero_crossing_rate REAL,
                    spectral_centroid REAL,
                    spectral_rolloff REAL,
                    mfcc_features TEXT,
                    chroma_features TEXT,
                    tempo_confidence REAL,
                    onset_strength REAL,
                    harmonic_ratio REAL,
                    percussive_ratio REAL,
                    genre_primary TEXT,
                    genre_secondary TEXT,
                    mood_tags TEXT,
                    instrument_tags TEXT,
                    style_tags TEXT,
                    generated_tags TEXT,
                    spike_patterns TEXT,
                    membrane_potential REAL,
                    synaptic_weights TEXT,
                    neural_complexity REAL,
                    adaptation_rate REAL,
                    resonance_frequency REAL,
                    phase_coherence REAL,
                    temporal_dynamics TEXT,
                    frequency_selectivity REAL,
                    plasticity_index REAL,
                    audio_quality_score REAL,
                    noise_level REAL,
                    dynamic_range REAL,
                    frequency_balance REAL,
                    stereo_width REAL,
                    clipping_detected BOOLEAN DEFAULT FALSE,
                    artifacts_detected BOOLEAN DEFAULT FALSE,
                    mastering_quality TEXT,
                    loudness_lufs REAL,
                    true_peak_dbfs REAL,
                    processing_status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    processing_time REAL,
                    last_processed TIMESTAMP,
                    embedding_vector TEXT,
                    embedding_model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON audio_files(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bpm ON audio_files(bpm)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_has_embedding ON audio_files(has_embedding)")
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stems_category ON stems(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stems_source ON stems(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stems_bpm ON stems(bpm)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stems_processing_status ON stems(processing_status)")
            
            conn.commit()
    
    async def insert_audio_file(self, 
                               file_id: str,
                               filename: str,
                               category: Optional[str] = None,
                               bpm: Optional[int] = None,
                               duration: Optional[float] = None,
                               sample_rate: Optional[int] = None,
                               channels: Optional[int] = None,
                               file_size: Optional[int] = None) -> bool:
        """Insert new audio file record"""
        loop = asyncio.get_event_loop()
        
        def _insert():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO audio_files 
                    (id, filename, category, bpm, duration, sample_rate, channels, file_size, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_id, filename, category, bpm, duration, 
                    sample_rate, channels, file_size, datetime.now()
                ))
                conn.commit()
                return True
        
        try:
            return await loop.run_in_executor(self._executor, _insert)
        except Exception as e:
            print(f"Error inserting audio file {file_id}: {e}")
            return False
    
    async def update_embedding_status(self, file_id: str, has_embedding: bool) -> bool:
        """Update embedding status for audio file"""
        loop = asyncio.get_event_loop()
        
        def _update():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE audio_files 
                    SET has_embedding = ?, updated_at = ?
                    WHERE id = ?
                """, (has_embedding, datetime.now(), file_id))
                conn.commit()
                return conn.rowcount > 0
        
        try:
            return await loop.run_in_executor(self._executor, _update)
        except Exception as e:
            print(f"Error updating embedding status for {file_id}: {e}")
            return False
    
    async def get_audio_files(self,
                             category: Optional[str] = None,
                             bpm_min: Optional[int] = None,
                             bpm_max: Optional[int] = None,
                             has_embedding: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get audio files with optional filters"""
        loop = asyncio.get_event_loop()
        
        def _get_files():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM audio_files WHERE 1=1"
                params = []
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                if bpm_min is not None:
                    query += " AND (bpm IS NULL OR bpm >= ?)"
                    params.append(bpm_min)
                
                if bpm_max is not None:
                    query += " AND (bpm IS NULL OR bpm <= ?)"
                    params.append(bpm_max)
                
                if has_embedding is not None:
                    query += " AND has_embedding = ?"
                    params.append(has_embedding)
                
                query += " ORDER BY created_at DESC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
        
        try:
            return await loop.run_in_executor(self._executor, _get_files)
        except Exception as e:
            print(f"Error getting audio files: {e}")
            return []
    
    async def get_audio_files_by_ids(self, file_ids: List[str]) -> List[Dict[str, Any]]:
        """Get audio files by list of IDs"""
        if not file_ids:
            return []
        
        loop = asyncio.get_event_loop()
        
        def _get_files():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                placeholders = ','.join('?' * len(file_ids))
                query = f"SELECT * FROM audio_files WHERE id IN ({placeholders})"
                
                cursor = conn.execute(query, file_ids)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
        
        try:
            return await loop.run_in_executor(self._executor, _get_files)
        except Exception as e:
            print(f"Error getting audio files by IDs: {e}")
            return []
    
    async def get_audio_file_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get single audio file by ID"""
        loop = asyncio.get_event_loop()
        
        def _get_file():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute(
                    "SELECT * FROM audio_files WHERE id = ?", 
                    (file_id,)
                )
                row = cursor.fetchone()
                
                return dict(row) if row else None
        
        try:
            return await loop.run_in_executor(self._executor, _get_file)
        except Exception as e:
            print(f"Error getting audio file {file_id}: {e}")
            return None
    
    async def delete_audio_file(self, file_id: str) -> bool:
        """Delete audio file record"""
        loop = asyncio.get_event_loop()
        
        def _delete():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM audio_files WHERE id = ?", (file_id,))
                conn.commit()
                return conn.rowcount > 0
        
        try:
            return await loop.run_in_executor(self._executor, _delete)
        except Exception as e:
            print(f"Error deleting audio file {file_id}: {e}")
            return False
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get search and database statistics"""
        loop = asyncio.get_event_loop()
        
        def _get_stats():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Total files
                cursor = conn.execute("SELECT COUNT(*) as total FROM audio_files")
                total_files = cursor.fetchone()['total']
                
                # Category statistics
                cursor = conn.execute("""
                    SELECT 
                        category,
                        COUNT(*) as count,
                        AVG(bpm) as avg_bpm,
                        AVG(duration) as avg_duration
                    FROM audio_files 
                    WHERE category IS NOT NULL
                    GROUP BY category
                    ORDER BY count DESC
                """)
                categories = [dict(row) for row in cursor.fetchall()]
                
                # BPM range
                cursor = conn.execute("""
                    SELECT MIN(bpm) as min_bpm, MAX(bpm) as max_bpm 
                    FROM audio_files 
                    WHERE bpm IS NOT NULL
                """)
                bpm_data = cursor.fetchone()
                bpm_range = {
                    'min': bpm_data['min_bpm'],
                    'max': bpm_data['max_bpm']
                } if bpm_data['min_bpm'] is not None else None
                
                return {
                    'total_files': total_files,
                    'categories': categories,
                    'bpm_range': bpm_range,
                    'last_updated': datetime.now()
                }
        
        try:
            return await loop.run_in_executor(self._executor, _get_stats)
        except Exception as e:
            print(f"Error getting search statistics: {e}")
            return {
                'total_files': 0,
                'categories': [],
                'bpm_range': None,
                'last_updated': datetime.now()
            }
    
    async def get_categories(self) -> List[str]:
        """Get list of all categories"""
        loop = asyncio.get_event_loop()
        
        def _get_categories():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT category 
                    FROM audio_files 
                    WHERE category IS NOT NULL 
                    ORDER BY category
                """)
                return [row[0] for row in cursor.fetchall()]
        
        try:
            return await loop.run_in_executor(self._executor, _get_categories)
        except Exception as e:
            print(f"Error getting categories: {e}")
            return []
    
    # New methods for stems table
    async def insert_stem(self, stem_data: Dict[str, Any]) -> bool:
        """Insert new stem record"""
        loop = asyncio.get_event_loop()
        
        def _insert():
            with sqlite3.connect(self.db_path) as conn:
                # Build dynamic insert query based on provided data
                columns = list(stem_data.keys())
                placeholders = ','.join('?' * len(columns))
                column_names = ','.join(columns)
                
                query = f"INSERT OR REPLACE INTO stems ({column_names}) VALUES ({placeholders})"
                values = list(stem_data.values())
                
                conn.execute(query, values)
                conn.commit()
                return True
        
        try:
            return await loop.run_in_executor(self._executor, _insert)
        except Exception as e:
            print(f"Error inserting stem: {e}")
            return False
    
    async def get_stems_by_category(self, category: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get stems by category and optionally by source"""
        loop = asyncio.get_event_loop()
        
        def _get_stems():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM stems WHERE category = ?"
                params = [category]
                
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                query += " ORDER BY created_at DESC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
        
        try:
            return await loop.run_in_executor(self._executor, _get_stems)
        except Exception as e:
            print(f"Error getting stems by category {category}: {e}")
            return []
    
    async def get_stems_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all stems by source (original, separated, generated)"""
        loop = asyncio.get_event_loop()
        
        def _get_stems():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute(
                    "SELECT * FROM stems WHERE source = ? ORDER BY created_at DESC",
                    (source,)
                )
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
        
        try:
            return await loop.run_in_executor(self._executor, _get_stems)
        except Exception as e:
            print(f"Error getting stems by source {source}: {e}")
            return []
    
    async def get_all_stems(self, 
                           category: Optional[str] = None,
                           source: Optional[str] = None,
                           limit: Optional[int] = None,
                           offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all stems with optional filters"""
        loop = asyncio.get_event_loop()
        
        def _get_stems():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM stems WHERE 1=1"
                params = []
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                query += " ORDER BY created_at DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                    
                    if offset:
                        query += " OFFSET ?"
                        params.append(offset)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
        
        try:
            return await loop.run_in_executor(self._executor, _get_stems)
        except Exception as e:
            print(f"Error getting stems: {e}")
            return []
    
    async def get_stem_by_id(self, stem_id: str) -> Optional[Dict[str, Any]]:
        """Get single stem by ID"""
        loop = asyncio.get_event_loop()
        
        def _get_stem():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute(
                    "SELECT * FROM stems WHERE id = ?",
                    (stem_id,)
                )
                row = cursor.fetchone()
                
                return dict(row) if row else None
        
        try:
            return await loop.run_in_executor(self._executor, _get_stem)
        except Exception as e:
            print(f"Error getting stem {stem_id}: {e}")
            return None
    
    async def update_stem_processing_status(self, stem_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """Update processing status of a stem"""
        loop = asyncio.get_event_loop()
        
        def _update():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE stems 
                    SET processing_status = ?, error_message = ?, updated_at = ?
                    WHERE id = ?
                """, (status, error_message, datetime.now(), stem_id))
                conn.commit()
                return conn.rowcount > 0
        
        try:
            return await loop.run_in_executor(self._executor, _update)
        except Exception as e:
            print(f"Error updating stem processing status for {stem_id}: {e}")
            return False
    
    async def get_stem_categories(self) -> List[str]:
        """Get list of all stem categories"""
        loop = asyncio.get_event_loop()
        
        def _get_categories():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT category 
                    FROM stems 
                    WHERE category IS NOT NULL 
                    ORDER BY category
                """)
                return [row[0] for row in cursor.fetchall()]
        
        try:
            return await loop.run_in_executor(self._executor, _get_categories)
        except Exception as e:
            print(f"Error getting stem categories: {e}")
            return []
    
    async def get_stem_statistics(self) -> Dict[str, Any]:
        """Get stem statistics by source and category"""
        loop = asyncio.get_event_loop()
        
        def _get_stats():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Total stems by source
                cursor = conn.execute("""
                    SELECT source, COUNT(*) as count
                    FROM stems
                    GROUP BY source
                    ORDER BY count DESC
                """)
                source_stats = [dict(row) for row in cursor.fetchall()]
                
                # Category statistics
                cursor = conn.execute("""
                    SELECT 
                        category,
                        source,
                        COUNT(*) as count,
                        AVG(bpm) as avg_bpm,
                        AVG(duration) as avg_duration
                    FROM stems 
                    WHERE category IS NOT NULL
                    GROUP BY category, source
                    ORDER BY category, count DESC
                """)
                category_stats = [dict(row) for row in cursor.fetchall()]
                
                # Total count
                cursor = conn.execute("SELECT COUNT(*) as total FROM stems")
                total_stems = cursor.fetchone()['total']
                
                return {
                    'total_stems': total_stems,
                    'by_source': source_stats,
                    'by_category': category_stats,
                    'last_updated': datetime.now()
                }
        
        try:
            return await loop.run_in_executor(self._executor, _get_stats)
        except Exception as e:
            print(f"Error getting stem statistics: {e}")
            return {
                'total_stems': 0,
                'by_source': [],
                'by_category': [],
                'last_updated': datetime.now()
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if self._executor:
            self._executor.shutdown(wait=True)