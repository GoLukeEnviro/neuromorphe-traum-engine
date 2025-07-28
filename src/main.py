"""
Hauptdatei für die FastAPI-Anwendung der Neuromorphen Traum-Engine.

Definiert die FastAPI-App-Instanz, bindet Router ein und konfiguriert
Startup-Events wie die Datenbankinitialisierung.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import Dict, Any
from src.core.config import settings
from src.api.endpoints.health import router as health_router
from src.api.endpoints.stems import router as stems_router
from src.api.endpoints.neuromorphic import router as neuromorphic_router
from src.database.database import create_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan-Context-Manager für Startup- und Shutdown-Events."""
    # Startup
    await create_tables()
    yield
    # Shutdown (falls benötigt)
    pass

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="Neuromorphic Dream Engine - AI-powered music creation and stem processing",
    lifespan=lifespan
)

# Include routers
app.include_router(health_router, prefix="/system", tags=["system"])
app.include_router(stems_router, prefix="/api/v1/stems", tags=["stems"])
app.include_router(neuromorphic_router, prefix="/api/v1/neuromorphic", tags=["neuromorphic"])

@app.get("/")
def read_root() -> Dict[str, str]:
    """Gibt eine Willkommensnachricht für den Root-Endpunkt zurück."""
    return {"message": f"Willkommen bei der {settings.PROJECT_NAME}"}