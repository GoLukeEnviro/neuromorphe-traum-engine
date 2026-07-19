from fastapi import APIRouter
from audio.router import router as audio_router
from search.router import router as search_router

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include domain routers
api_router.include_router(audio_router)
api_router.include_router(search_router)


@api_router.get("/health")
async def health_check():
    """Main health check endpoint"""
    return {
        "status": "healthy",
        "service": "neuromorphe-traum-engine",
        "version": "2.0",
        "components": {
            "audio": "healthy",
            "search": "healthy",
            "database": "healthy"
        }
    }


@api_router.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Neuromorphe Traum-Engine v2.0 API",
        "description": "Semantic audio search engine using CLAP embeddings",
        "docs": "/docs",
        "health": "/api/v1/health"
    }