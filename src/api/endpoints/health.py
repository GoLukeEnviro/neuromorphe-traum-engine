# src/api/endpoints/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health", status_code=200)
def check_health():
    """
    Überprüft den Status des API-Servers.
    """
    return {"status": "ok", "message": "API is running"}