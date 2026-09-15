from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
from ai_agents.search_engine_cli import SearchEngine

APP_DIR = Path(__file__).resolve().parent
RAW_DIR = APP_DIR.parent / "raw_construction_kits"
EMBEDDINGS_PATH = APP_DIR.parent / "processed_database" / "embeddings.pkl"

app = FastAPI(title="Neuromorphe Traum-Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = None


@app.on_event("startup")
def startup_event():
    global search_engine
    if EMBEDDINGS_PATH.exists():
        search_engine = SearchEngine(str(EMBEDDINGS_PATH))


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
def upload_audio(file: UploadFile = File(...)):
    RAW_DIR.mkdir(exist_ok=True)
    dest = RAW_DIR / file.filename
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}


@app.post("/search")
def search(prompt: str):
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not ready")
    results = search_engine.search(prompt, top_k=5)
    return [{"path": p, "score": s} for p, s in results]
