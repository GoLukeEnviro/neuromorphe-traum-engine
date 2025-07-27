# src/main.py
from fastapi import FastAPI
from .core.config import settings
from .api.endpoints.health import router as health_router
from .api.endpoints.stems import router as stems_router
from .db.database import create_tables

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0"
)

# Create database tables on startup
@app.on_event("startup")
def startup_event():
    create_tables()

# Include routers
app.include_router(health_router, prefix="/system", tags=["system"])
app.include_router(stems_router, prefix="/api/v1/stems", tags=["stems"])

@app.get("/")
def read_root():
    return {"message": f"Willkommen bei der {settings.PROJECT_NAME}"}