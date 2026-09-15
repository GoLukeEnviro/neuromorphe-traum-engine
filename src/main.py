"""
Hauptdatei für die FastAPI-Anwendung der Neuromorphen Traum-Engine.

Definiert die FastAPI-App-Instanz, bindet Router ein und konfiguriert
Startup-Events wie die Datenbankinitialisierung.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Dict, Any
from core.config import settings
from database.database import create_tables

# Importiere Router nach der Datenbankinitialisierung
from audio.router import router as audio_router
from api.endpoints.health import router as health_router
from api.endpoints.stems import router as stems_router
from api.endpoints.neuromorphic import router as neuromorphic_router
from search.router import router as search_router
from api.endpoints.system_extras import router as system_extras_router

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Erlaube alle Ursprünge für Entwicklung
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explizite OPTIONS-Routen: die CORSMiddleware beantwortet Preflight nur,
# wenn der Request als Preflight erkannt wird (Origin + Access-Control-
# Request-Method). Ein nackter OPTIONS-Aufruf läuft sonst in einen 405.
@app.options("/{full_path:path}", include_in_schema=False)
async def preflight_handler(full_path: str):
    """Beantwortet CORS-Preflight-Anfragen für beliebige Pfade."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
        },
    )

# Include routers - Audio zuerst registrieren
app.include_router(audio_router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(health_router, prefix="/system", tags=["system"])
app.include_router(stems_router, prefix="/api/v1/stems", tags=["stems"])
app.include_router(neuromorphic_router, prefix="/api/v1/neuromorphic", tags=["neuromorphic"])
app.include_router(search_router, prefix="/api/v1", tags=["search"])
app.include_router(system_extras_router, tags=["system-extras"])

@app.get("/")
def read_root() -> Dict[str, str]:
    """Gibt eine Willkommensnachricht für den Root-Endpunkt zurück."""
    return {"message": f"Willkommen bei der {settings.PROJECT_NAME}"}

@app.get("/debug/routes")
async def debug_routes():
    """Zeige alle registrierten Routen für Debugging."""
    routes = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue  # Sub-Router-Container ohne eigenen Pfad überspringen
        routes.append({
            "path": path,
            "name": getattr(route, "name", None),
            "methods": sorted(route.methods) if hasattr(route, "methods") else []
        })
    return {"routes": routes}