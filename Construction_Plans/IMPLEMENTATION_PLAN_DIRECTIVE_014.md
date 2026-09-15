# 🎵 IMPLEMENTIERUNGSPLAN: AGENTEN-DIREKTIVE 014

**Neuromorphe Traum-Engine v2.0 - Generatives Musik-Backend**

## 📋 ÜBERSICHT

Dieser Plan implementiert ein vollständiges generatives Musik-System basierend auf AGENTEN-DIREKTIVE 014. Das System kann aus Text-Prompts komplette Musik-Tracks generieren.

## 🏗️ SYSTEM-ARCHITEKTUR

### Kernkomponenten
1. **Preprocessor**: Analysiert Audio-Stems und erstellt CLAP-Embeddings
2. **Arranger**: Generiert Track-Strukturpläne aus Text-Prompts
3. **Renderer**: Montiert Stems zu finalen Audio-Tracks
4. **API**: REST-Endpunkte für Preprocessing und Track-Generierung

### Datenfluss
```
Text-Prompt → Arranger → Track-Plan → Renderer → Audio-Track
     ↑                                    ↓
  API-Input                        Generated WAV
```

## 📁 VERZEICHNISSTRUKTUR

```
neuromorphe-traum-engine/
├── raw_construction_kits/          # Input: Audio-Stems
├── processed_database/             # Verarbeitete Daten
│   ├── stems/                      # Stem-Metadaten
│   └── quarantine/                 # Fehlerhafte Dateien
├── generated_tracks/               # Output: Generierte Tracks
├── src/                           # Hauptquellcode
│   ├── __init__.py
│   ├── core/                      # Kernkonfiguration
│   │   ├── __init__.py
│   │   └── config.py
│   ├── db/                        # Datenbanklogik
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── models.py
│   │   └── crud.py
│   ├── services/                  # Business Logic
│   │   ├── __init__.py
│   │   ├── preprocessor.py        # NeuroAnalyzer
│   │   ├── arranger.py           # Track-Strukturierung
│   │   └── renderer.py           # Audio-Montage
│   ├── schemas/                   # Pydantic-Schemas
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── api/                       # REST-API
│   │   ├── __init__.py
│   │   └── endpoints.py
│   └── main.py                    # FastAPI-App
└── requirements.txt               # Dependencies
```

## 🔧 PHASE 1: PROJEKT-SETUP

### 1.1 Verzeichnisstruktur erstellen
- [x] Bestehende Struktur analysieren
- [ ] Fehlende Verzeichnisse erstellen
- [ ] Neue Struktur gemäß Direktive anpassen

### 1.2 Dependencies aktualisieren
```txt
fastapi
uvicorn[standard]
sqlalchemy
pydantic
librosa
soundfile
numpy
laion-clap-pytorch
torch
scikit-learn
tqdm
pydantic-settings
```

## 🗄️ PHASE 2: DATEN-INFRASTRUKTUR

### 2.1 Datenbankmodelle (src/db/models.py)
```python
class Stem(Base):
    id: int (Primary Key)
    path: str (Dateipfad)
    filename: str
    category: str (kick, bass, hihat, etc.)
    tags: str (JSON-Array)
    bpm: float
    key: str
    duration: float
    embedding: bytes (CLAP-Embedding)
    created_at: datetime
    updated_at: datetime
```

### 2.2 CRUD-Operationen (src/db/crud.py)
- `create_stem()`
- `get_stem_by_id()`
- `search_stems_by_tags_and_category()` ⭐ **NEU für Arranger**
- `get_stems_by_category()`
- `update_stem()`
- `delete_stem()`

### 2.3 Pydantic-Schemas (src/schemas/schemas.py)
```python
class StemBase(BaseModel):
    path: str
    category: str
    tags: List[str]
    bpm: Optional[float]
    
class ArrangementPlan(BaseModel):  # ⭐ NEU
    bpm: int
    key: str
    structure: List[ArrangementSection]
    
class TrackGenerationRequest(BaseModel):  # ⭐ NEU
    prompt: str
    
class TrackGenerationResponse(BaseModel):  # ⭐ NEU
    track_path: str
    arrangement_plan: ArrangementPlan
```

## 🧠 PHASE 3: KREATIVE KERNLOGIK

### 3.1 Preprocessor Service (src/services/preprocessor.py)
**Basiert auf NeuroAnalyzer aus Direktive 006**

```python
class PreprocessorService:
    def __init__(self):
        # CLAP-Modell laden
        # Datenbank-Verbindung
        
    async def process_audio_files(self):
        # Batch-Verarbeitung mit Resume-Mechanismus
        # CLAP-Embeddings generieren
        # Metadaten extrahieren
        # In Datenbank speichern
```

### 3.2 Arranger Service (src/services/arranger.py) ⭐ **NEU**
**Intelligenter Track-Strukturierer**

```python
class ArrangerService:
    def __init__(self):
        # Datenbank-Zugriff über CRUD
        
    def generate_arrangement_plan(self, prompt: str) -> dict:
        # 1. Prompt-Parsing (BPM, Genre, Stimmung)
        # 2. Track-Struktur generieren
        # 3. Stem-Queries definieren
        
        return {
            "bpm": 135,
            "key": "Am",
            "structure": [
                {
                    "section": "Intro",
                    "bars": 16,
                    "stem_queries": [
                        {"category": "atmo", "tags": ["dark"], "count": 1}
                    ]
                },
                {
                    "section": "Groove",
                    "bars": 64,
                    "stem_queries": [
                        {"category": "kick", "tags": ["aggressive"], "count": 1},
                        {"category": "bass", "tags": ["dark"], "count": 1}
                    ]
                }
            ]
        }
```

### 3.3 Renderer Service (src/services/renderer.py) ⭐ **NEU**
**Audio-Montage-Engine**

```python
class RendererService:
    def __init__(self):
        # Audio-Processing-Tools
        
    async def render_track(self, arrangement_plan: dict) -> str:
        # 1. Stems aus DB laden basierend auf Queries
        # 2. Audio-Timeline berechnen
        # 3. Stems zeitlich arrangieren
        # 4. Audio-Mixdown erstellen
        # 5. WAV-Datei speichern
        
        return "generated_tracks/track_20250126_150000.wav"
```

## 🌐 PHASE 4: API-INTEGRATION

### 4.1 API-Endpunkte (src/api/endpoints.py)

```python
@router.post("/preprocess")
async def preprocess_audio():
    # Startet Preprocessing asynchron
    
@router.post("/generate-track")  # ⭐ HAUPTFUNKTION
async def generate_track(request: TrackGenerationRequest):
    # 1. Arranger: Plan erstellen
    # 2. Renderer: Track generieren
    # 3. Pfad zurückgeben
```

### 4.2 FastAPI-App (src/main.py)
```python
app = FastAPI(title="Neuromorphe Traum-Engine v2.0")
app.include_router(router, prefix="/api/v1")
```

## 🎯 ERFOLGSKRITERIEN

### Technische Tests
- [ ] System startet fehlerfrei: `uvicorn src.main:app --reload`
- [ ] POST `/preprocess` verarbeitet Audio-Stems
- [ ] Datenbank wird korrekt gefüllt

### Funktionale Tests
- [ ] POST `/generate-track` mit Prompt: `{"prompt": "driving industrial techno 138 bpm"}`
- [ ] System generiert Track-Plan
- [ ] System findet passende Stems
- [ ] System erstellt WAV-Datei in `generated_tracks/`
- [ ] Generierte Datei ist abspielbar

### Qualitätskriterien
- [ ] Track entspricht musikalisch dem Prompt
- [ ] Audio-Qualität ist hochwertig
- [ ] Timing und BPM sind korrekt
- [ ] Stems sind harmonisch arrangiert

## 🔄 IMPLEMENTIERUNGSREIHENFOLGE

1. **Setup & Struktur** (30 min)
   - Verzeichnisse erstellen
   - Dependencies aktualisieren

2. **Datenbank-Layer** (60 min)
   - Models erweitern
   - CRUD-Funktionen implementieren
   - Schemas definieren

3. **Preprocessor** (45 min)
   - Bestehenden Code refaktorisieren
   - In Service-Struktur integrieren

4. **Arranger Service** (90 min)
   - Prompt-Parser implementieren
   - Track-Struktur-Generator
   - Regel-basierte Logik

5. **Renderer Service** (120 min)
   - Stem-Suche implementieren
   - Audio-Timeline-Berechnung
   - Audio-Montage-Engine

6. **API-Integration** (45 min)
   - Endpunkte implementieren
   - Error-Handling
   - Response-Schemas

7. **Testing & Validation** (60 min)
   - Funktionale Tests
   - Audio-Qualität prüfen
   - Performance-Optimierung

**Gesamtzeit: ~7.5 Stunden**

## 🚨 KRITISCHE PUNKTE

1. **Audio-Synchronisation**: Stems müssen zeitlich exakt aligniert werden
2. **BPM-Matching**: Tempo-Anpassung zwischen verschiedenen Stems
3. **Memory-Management**: Große Audio-Dateien effizient verarbeiten
4. **Error-Handling**: Robuste Fehlerbehandlung bei Audio-Processing
5. **Performance**: Rendering sollte < 30 Sekunden dauern

## 📊 METRIKEN

- **Preprocessing-Zeit**: < 10s pro Stem
- **Arrangement-Zeit**: < 2s pro Prompt
- **Rendering-Zeit**: < 30s pro Track
- **Audio-Qualität**: 44.1kHz, 16-bit WAV
- **Track-Länge**: 5-8 Minuten

---

**Status**: 🔄 Bereit zur Implementierung
**Priorität**: 🔥 Hoch
**Komplexität**: ⭐⭐⭐⭐⭐ (5/5)