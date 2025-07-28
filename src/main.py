# src/main.py
from fastapi import FastAPI
from src.core.config import settings
from src.api.endpoints.health import router as health_router
from src.api.endpoints.stems import router as stems_router
from src.api.endpoints.neuromorphic import router as neuromorphic_router
from src.db.database import create_tables

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="Neuromorphic Dream Engine - AI-powered music creation and stem processing"
)

# Create database tables on startup
@app.on_event("startup")
def startup_event():
    create_tables()

# Include routers
app.include_router(health_router, prefix="/system", tags=["system"])
app.include_router(stems_router, prefix="/api/v1/stems", tags=["stems"])
app.include_router(neuromorphic_router, prefix="/api/v1/neuromorphic", tags=["neuromorphic"])

@app.get("/")
def read_root():
    return {"message": f"Willkommen bei der {settings.PROJECT_NAME}"}