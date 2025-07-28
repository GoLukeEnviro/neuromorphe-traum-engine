import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from laion_clap import CLAP_Module
import json

from ..database.service import DatabaseService
from ..schemas.stem import SearchResult
from ..database.models import Stem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchService:
    """Service for semantic audio search using CLAP embeddings"""
    
    def __init__(self):
        """Initialize the search service with CLAP model"""
        logger.info("Initializing SearchService...")
        self.clap_model = None
        self.db_service = DatabaseService()
        self._load_clap_model()
        logger.info("SearchService ready.")
    
    def _load_clap_model(self):
        """Load the LAION-CLAP model"""
        try:
            logger.info("Loading LAION-CLAP model...")
            self.clap_model = CLAP_Module(enable_fusion=False)
            self.clap_model.load_ckpt()
            logger.info("CLAP model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CLAP model: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        category_filter: Optional[str] = None,
        bpm_range: Optional[Tuple[float, float]] = None
    ) -> List[SearchResult]:
        """Perform semantic search based on text query"""
        logger.info(f"Searching for: '{query}' (Top {top_k})")
        
        if not self.clap_model:
            raise RuntimeError("CLAP model not loaded")
        
        # Get text embedding for the query
        query_embedding = self._get_text_embedding(query)
        
        # Load stems with embeddings from database
        stems_data = await self._load_stems_with_embeddings(
            category_filter, bpm_range
        )
        
        if not stems_data:
            logger.warning("No matching stems found in database.")
            return []
        
        # Calculate similarities
        similarities = []
        for stem_data in stems_data:
            similarity = self._calculate_cosine_similarity(
                query_embedding, np.array(stem_data['embedding'])
            )
            similarities.append({
                'stem': stem_data['stem'],
                'similarity_score': float(similarity)
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top-k results
        results = similarities[:top_k]
        
        logger.info(f"Found: {len(results)} relevant stems")
        return [SearchResult(**result) for result in results]
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Calculate CLAP embedding for text"""
        try:
            # Calculate text embedding
            text_embed = self.clap_model.get_text_embedding([text])
            
            # Normalize
            if len(text_embed.shape) > 1:
                text_embed = text_embed.flatten()
            
            text_embed = text_embed / np.linalg.norm(text_embed)
            
            return text_embed
            
        except Exception as e:
            logger.error(f"Error calculating text embedding: {e}")
            raise
    
    async def _load_stems_with_embeddings(
        self, 
        category_filter: Optional[str] = None,
        bpm_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict]:
        """Load all stems with their CLAP embeddings from database"""
        # Get stems from database
        stems = await self.db_service.get_all_stems(category=category_filter)  # Use db_service
        
        stems_data = []
        for stem in stems:
            # Skip stems without embeddings
            if not stem.audio_embedding:
                continue
            
            # Apply BPM filter
            if bpm_range and stem.bpm:
                if not (bpm_range[0] <= stem.bpm <= bpm_range[1]):
                    continue
            
            stems_data.append({
                'stem': stem,
                'embedding': stem.audio_embedding
            })
        
        return stems_data
    
    def _calculate_cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

