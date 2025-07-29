"""
Hauptdatei für die FastAPI-Anwendung der Neuromorphen Traum-Engine.

Definiert die FastAPI-App-Instanz, bindet Router ein und konfiguriert
Startup-Events wie die Datenbankinitialisierung.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import Dict, Any
from core.config import settings
from database.database import create_tables

# Importiere Router nach der Datenbankinitialisierung
from audio.router import router as audio_router
from api.endpoints.health import router as health_router
from api.endpoints.stems import router as stems_router
from api.endpoints.neuromorphic import router as neuromorphic_router

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

# Include routers - Audio zuerst registrieren
app.include_router(audio_router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(health_router, prefix="/system", tags=["system"])
app.include_router(stems_router, prefix="/api/v1/stems", tags=["stems"])
app.include_router(neuromorphic_router, prefix="/api/v1/neuromorphic", tags=["neuromorphic"])

@app.get("/")
def read_root() -> Dict[str, str]:
    """Gibt eine Willkommensnachricht für den Root-Endpunkt zurück."""
    return {"message": f"Willkommen bei der {settings.PROJECT_NAME}"}

@app.get("/debug/routes")
async def debug_routes():
    """Zeige alle registrierten Routen für Debugging"""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods) if hasattr(route, 'methods') else []
        })
    return {"routes": routes}