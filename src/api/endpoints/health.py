"""API-Endpunkte für Health-Checks und Systemstatus.

Definiert Routen zur Überprüfung der Verfügbarkeit und des Zustands der API.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    message: str

@router.get("/health", response_model=HealthResponse, status_code=200)
def check_health():
    """
    Überprüft den Status des API-servers.
    """
    return {"status": "ok", "message": "API is running"}