from functools import lru_cache
from .service import SearchService


@lru_cache()
def get_search_service() -> SearchService:
    """Dependency injection for SearchService"""
    return SearchService(
        embedding_dir="embeddings",
        model_version="630k-audioset-best"
    )