# 🎵 Neuromorphe Traum-Engine v2.0

Eine semantische Audio-Suchmaschine basierend auf CLAP-Embeddings mit FastAPI-Backend und Streamlit-Frontend.

## 🚀 Features

- **Semantische Audio-Suche**: Finden Sie Audio-Stems durch natürliche Sprachbeschreibungen
- **CLAP-Embeddings**: Nutzt das LAION CLAP-Modell für hochpräzise Audio-zu-Text-Retrieval
- **FastAPI Backend**: Moderne, asynchrone REST-API mit automatischer Dokumentation
- **Streamlit Frontend**: Intuitive Web-Oberfläche für Suche und Verwaltung
- **Docker-Ready**: Vollständig containerisiert für einfaches Deployment
- **SQLite Database**: Effiziente Speicherung von Metadaten und Embeddings

## 🏗️ Architektur

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │    SQLite       │
│   Frontend      │◄──►│    Backend      │◄──►│   Database      │
│   (Port 8501)   │    │   (Port 8000)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Installation

### Voraussetzungen

- Python 3.12+
- Docker & Docker Compose (für Container-Deployment)
- Git

### Lokale Entwicklung

1. **Repository klonen**
```bash
git clone <repository-url>
cd neuromorphe-traum-engine
```

2. **Abhängigkeiten installieren**
```bash
pip install -r requirements.txt
```

3. **Umgebungsvariablen konfigurieren**
```bash
cp .env.example .env
# Bearbeiten Sie .env nach Bedarf
```

4. **Backend starten**
```bash
python -m uvicorn src.main:app --reload
```

5. **Frontend starten** (neues Terminal)
```bash
python -m streamlit run frontend/app.py --server.port 8501
```

### Docker Deployment

1. **Mit Docker Compose starten**
```bash
docker-compose up --build
```

2. **Services einzeln bauen**
```bash
# Backend
docker build -f Dockerfile.backend -t neuromorphe-backend .

# Frontend
docker build -f Dockerfile.frontend -t neuromorphe-frontend .
```

## 🔧 Konfiguration

### Umgebungsvariablen

| Variable | Beschreibung | Standard |
|----------|--------------|----------|
| `PROJECT_NAME` | Name der Anwendung | "Neuromorphe Traum-Engine v2.0" |
| `DATABASE_URL` | SQLite Datenbankpfad | "sqlite:///processed_database/stems.db" |
| `UPLOAD_DIR` | Upload-Verzeichnis | "./raw_construction_kits" |
| `MODEL_CACHE_DIR` | CLAP-Modell Cache | "./models" |
| `API_BASE_URL` | Backend-URL für Frontend | "http://localhost:8000" |

## 📚 API Dokumentation

### Endpunkte

- **GET /** - Willkommensnachricht
- **GET /system/health** - Health Check
- **GET /api/v1/stems/** - Alle Stems abrufen (mit Paginierung)
- **GET /api/v1/stems/{id}** - Einzelnen Stem abrufen
- **GET /api/v1/stems/search/** - Semantische Suche
- **GET /api/v1/stems/category/{category}** - Stems nach Kategorie

### Automatische Dokumentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔍 Verwendung

### Semantische Suche

```python
import requests

# Suche nach "energetic drum loop"
response = requests.get(
    "http://localhost:8000/api/v1/stems/search/",
    params={"prompt": "energetic drum loop", "top_k": 5}
)

results = response.json()
for result in results:
    print(f"Datei: {result['path']}")
    print(f"Ähnlichkeit: {result['similarity']:.3f}")
```

### Frontend-Nutzung

1. Öffnen Sie http://localhost:8501
2. Wählen Sie "🔍 Suche" in der Sidebar
3. Geben Sie eine Beschreibung ein (z.B. "melodic bass line")
4. Durchsuchen Sie die Ergebnisse

## 🗂️ Projektstruktur

```
neuromorphe-traum-engine/
├── src/                     # Backend-Quellcode
│   ├── api/                 # API-Endpunkte
│   ├── core/                # Kernkonfiguration
│   ├── db/                  # Datenbankmodelle und CRUD
│   ├── schemas/             # Pydantic-Schemas
│   ├── services/            # Business Logic
│   └── main.py              # FastAPI-App
├── frontend/                # Streamlit-Frontend
│   └── app.py               # Hauptanwendung
├── processed_database/      # SQLite-Datenbank
├── raw_construction_kits/   # Audio-Dateien
├── models/                  # CLAP-Modell Cache
├── Dockerfile.backend       # Backend-Container
├── Dockerfile.frontend      # Frontend-Container
├── docker-compose.yml       # Orchestrierung
└── requirements.txt         # Python-Abhängigkeiten
```

## 🧪 Tests

```bash
# API-Tests
curl http://localhost:8000/system/health

# Suche testen
curl "http://localhost:8000/api/v1/stems/search/?prompt=drum%20loop&top_k=3"
```

## 🔧 Entwicklung

### Backend erweitern

1. Neue Endpunkte in `src/api/endpoints/` hinzufügen
2. Schemas in `src/schemas/` definieren
3. Business Logic in `src/services/` implementieren

### Frontend anpassen

1. Neue Seiten in `frontend/app.py` hinzufügen
2. API-Aufrufe erweitern
3. UI-Komponenten anpassen

## 📊 Performance

- **CLAP-Modell**: ~1.5GB Download beim ersten Start
- **Suchzeit**: ~100-500ms pro Anfrage
- **Speicher**: ~2-4GB RAM für CLAP-Modell

## 🐛 Troubleshooting

### Häufige Probleme

1. **"ModuleNotFoundError"**
   ```bash
   pip install -r requirements.txt
   ```

2. **"Backend nicht erreichbar"**
   - Prüfen Sie, ob FastAPI läuft: http://localhost:8000/system/health
   - Firewall-Einstellungen überprüfen

3. **"CLAP-Modell lädt nicht"**
   - Internetverbindung prüfen
   - Ausreichend Speicherplatz sicherstellen

4. **Docker-Probleme**
   ```bash
   docker-compose down
   docker-compose up --build
   ```

## 🤝 Beitragen

1. Fork des Repositories
2. Feature-Branch erstellen
3. Änderungen committen
4. Pull Request erstellen

## 📄 Lizenz

MIT License - siehe LICENSE-Datei für Details.

## 🙏 Danksagungen

- [LAION](https://laion.ai/) für das CLAP-Modell
- [FastAPI](https://fastapi.tiangolo.com/) für das Backend-Framework
- [Streamlit](https://streamlit.io/) für das Frontend-Framework

---

**Neuromorphe Traum-Engine v2.0** - Semantische Audio-Suche neu definiert 🎵