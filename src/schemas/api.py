from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class StemResponse(BaseModel):
    id: str
    name: str

class ArrangementResponse(BaseModel):
    id: str
    prompt: str

class RenderJobResponse(BaseModel):
    id: str
    status: str
