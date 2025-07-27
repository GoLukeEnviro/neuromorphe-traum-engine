from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.router import api_router
from src.audio.dependencies import get_audio_service
from src.search.dependencies import get_search_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("Starting Neuromorphe Traum-Engine v2.0...")
    
    # Initialize services (lazy loading will happen on first use)
    audio_service = get_audio_service()
    search_service = get_search_service()
    
    yield
    
    # Shutdown
    print("Shutting down Neuromorphe Traum-Engine v2.0...")
    audio_service.cleanup()
    search_service.cleanup()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Neuromorphe Traum-Engine v2.0",
    description="Semantic audio search engine using CLAP embeddings with domain-driven architecture",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Neuromorphe Traum-Engine v2.0",
        "description": "Semantic audio search engine using CLAP embeddings",
        "architecture": "Domain-driven design with FastAPI",
        "docs": "/docs",
        "api": "/api/v1",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )