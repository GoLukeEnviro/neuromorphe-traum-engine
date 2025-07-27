import time
import asyncio
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
from laion_clap import CLAP_Module
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

from .schemas import (
    SearchRequest,
    SearchResult,
    SearchResponse,
    SimilarityRequest,
    CategoryStats,
    SearchStats
)
from ..database.service import DatabaseService


class SearchService:
    """Service for semantic audio search using CLAP embeddings"""
    
    def __init__(self, 
                 embedding_dir: str = "embeddings",
                 model_version: str = "630k-audioset-best"):
        self.embedding_dir = Path(embedding_dir)
        self.model_version = model_version
        self._clap_model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.db_service = DatabaseService()
    
    async def get_clap_model(self) -> CLAP_Module:
        """Lazy loading of CLAP model"""
        if self._clap_model is None:
            loop = asyncio.get_event_loop()
            self._clap_model = await loop.run_in_executor(
                self._executor, 
                self._load_clap_model
            )
        return self._clap_model
    
    def _load_clap_model(self) -> CLAP_Module:
        """Load CLAP model in thread pool"""
        model = CLAP_Module(enable_fusion=False)
        model.load_ckpt(self.model_version)
        return model
    
    async def search_by_text(self, request: SearchRequest) -> SearchResponse:
        """Perform semantic search using text query"""
        start_time = time.time()
        
        try:
            # Generate text embedding
            model = await self.get_clap_model()
            loop = asyncio.get_event_loop()
            
            text_embedding = await loop.run_in_executor(
                self._executor,
                self._generate_text_embedding,
                model,
                request.query
            )
            
            # Get audio files from database with filters
            audio_files = await self.db_service.get_audio_files(
                category=request.category,
                bpm_min=request.bpm_min,
                bpm_max=request.bpm_max
            )
            
            # Calculate similarities
            similarities = await self._calculate_similarities(
                text_embedding, 
                audio_files,
                request.limit
            )
            
            # Create search results
            results = []
            for file_info, similarity in similarities:
                result = SearchResult(
                    id=file_info['id'],
                    filename=file_info['filename'],
                    similarity_score=float(similarity),
                    category=file_info.get('category'),
                    bpm=file_info.get('bpm'),
                    duration=file_info.get('duration'),
                    file_path=f"/audio_files/{file_info['id']}.{file_info['filename'].split('.')[-1]}"
                )
                results.append(result)
            
            search_time = time.time() - start_time
            
            # Build filters applied
            filters_applied = {}
            if request.category:
                filters_applied['category'] = request.category
            if request.bpm_min or request.bpm_max:
                filters_applied['bpm_range'] = [request.bpm_min, request.bpm_max]
            
            return SearchResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                filters_applied=filters_applied
            )
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    def _generate_text_embedding(self, model: CLAP_Module, text: str) -> np.ndarray:
        """Generate CLAP embedding for text query"""
        text_embed = model.get_text_embedding([text], use_tensor=False)
        return text_embed[0]  # Return first (and only) embedding
    
    async def _calculate_similarities(self, 
                                    query_embedding: np.ndarray, 
                                    audio_files: List[dict],
                                    limit: int) -> List[Tuple[dict, float]]:
        """Calculate cosine similarities between query and audio embeddings"""
        similarities = []
        
        for file_info in audio_files:
            # Load audio embedding
            embedding_path = self.embedding_dir / f"{file_info['id']}.npy"
            if not embedding_path.exists():
                continue
            
            try:
                loop = asyncio.get_event_loop()
                audio_embedding = await loop.run_in_executor(
                    self._executor,
                    lambda: np.load(str(embedding_path))
                )
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    audio_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append((file_info, similarity))
                
            except Exception as e:
                print(f"Error loading embedding for {file_info['id']}: {e}")
                continue
        
        # Sort by similarity (descending) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    async def find_similar_audio(self, request: SimilarityRequest) -> SearchResponse:
        """Find audio files similar to a given audio file"""
        start_time = time.time()
        
        try:
            # Load source embedding
            source_embedding_path = self.embedding_dir / f"{request.source_file_id}.npy"
            if not source_embedding_path.exists():
                raise Exception(f"Source embedding not found for {request.source_file_id}")
            
            loop = asyncio.get_event_loop()
            source_embedding = await loop.run_in_executor(
                self._executor,
                lambda: np.load(str(source_embedding_path))
            )
            
            # Get target files
            if request.target_file_ids:
                audio_files = await self.db_service.get_audio_files_by_ids(
                    request.target_file_ids
                )
            else:
                audio_files = await self.db_service.get_audio_files()
                # Remove source file from results
                audio_files = [f for f in audio_files if f['id'] != request.source_file_id]
            
            # Calculate similarities
            similarities = await self._calculate_similarities(
                source_embedding,
                audio_files,
                request.limit
            )
            
            # Create search results
            results = []
            for file_info, similarity in similarities:
                result = SearchResult(
                    id=file_info['id'],
                    filename=file_info['filename'],
                    similarity_score=float(similarity),
                    category=file_info.get('category'),
                    bpm=file_info.get('bpm'),
                    duration=file_info.get('duration'),
                    file_path=f"/audio_files/{file_info['id']}.{file_info['filename'].split('.')[-1]}"
                )
                results.append(result)
            
            search_time = time.time() - start_time
            
            return SearchResponse(
                query=f"Similar to {request.source_file_id}",
                results=results,
                total_results=len(results),
                search_time=search_time,
                filters_applied={"similarity_search": True}
            )
            
        except Exception as e:
            raise Exception(f"Similarity search failed: {str(e)}")
    
    async def get_search_stats(self) -> SearchStats:
        """Get search statistics"""
        try:
            stats = await self.db_service.get_search_statistics()
            
            # Count embeddings
            embedding_count = len(list(self.embedding_dir.glob("*.npy")))
            
            # Build category stats
            category_stats = []
            for cat_data in stats.get('categories', []):
                category_stats.append(CategoryStats(
                    category=cat_data['category'],
                    count=cat_data['count'],
                    avg_bpm=cat_data.get('avg_bpm'),
                    avg_duration=cat_data.get('avg_duration')
                ))
            
            return SearchStats(
                total_files=stats.get('total_files', 0),
                total_embeddings=embedding_count,
                categories=category_stats,
                bpm_range=stats.get('bpm_range'),
                last_updated=stats.get('last_updated')
            )
            
        except Exception as e:
            raise Exception(f"Failed to get search stats: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self._executor:
            self._executor.shutdown(wait=True)
        if self.db_service:
            self.db_service.cleanup()