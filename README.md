# 🎵 Neuromorphe Traum-Engine v2.0

> *Ein KI-gestützter kreativer Partner für die Generierung von neuartigem Raw-Techno*
>
> *"Wo das Kollektive Unbewusste auf intelligente Dirigenten trifft"*

Die Neuromorphe Traum-Engine ist keine gewöhnliche Audio-Suchmaschine - sie ist ein semantisches KI-System, das die Grenzen zwischen menschlicher Kreativität und maschineller Intuition verwischt. Basierend auf CLAP-Embeddings und modernster Audio-Technologie schafft sie einen luziden Traumzustand für Musikproduzenten.

## 🌌 Vision & Philosophie

### Das Kollektive Unbewusste
Unsere Engine greift auf ein semantisches Gedächtnis zu, das durch millionenfache Audio-Text-Korrelationen trainiert wurde. Jede Suche wird zu einer Reise durch das digitale Kollektivbewusstsein der elektronischen Musik.

### Intelligenter Dirigent
Der KI-Dirigent interpretiert nicht nur Begriffe, sondern versteht emotionale Nuancen, rhythmische Intentionen und klangliche Atmosphären. Er transformiert "energetic drum loop" in eine multidimensionale Suchanfrage, die über reine Metadaten hinausgeht.

### Luzides Träumen
Produzenten können in Echtzeit durch ihre Audio-Bibliotheken navigieren, als wären sie in einem luziden Traum - jede Suchanfrage wird zu einer explorativen Reise durch unentdeckte Klanglandschaften.

## 🚀 Core Features

### 🔍 Semantische Audio-Suche
- **Natürlichsprachige Queries**: "dunkler, hypnotischer Techno-Kick mit industrieller Textur"
- **Multimodale Suche**: Kombination aus Text, Audio-Beispielen und Stimmungsanalyse
- **Stem-Mutation**: Generative Variationen gefundener Stems in Echtzeit

### 🧠 KI-Engine
- **CLAP-Embeddings**: State-of-the-art Audio-Text-Understanding via LAION CLAP
- **Adaptive Retrieval**: Lernende Suchalgorithmen basierend auf Nutzer-Feedback
- **Generatives Paradox**: KI-generierte Stem-Vorschläge, die die ursprüngliche Suche erweitern

### 🎛️ Produktions-Workflow
- **Live-Session Integration**: Direkte Einbindung in DAW-Workflows via API
- **Stem-Kategorisierung**: Intelligente Klassifizierung (Kick, Bass, Hat, Percussion, FX)
- **Qualitäts-Scoring**: Automatische Bewertung von technischer und kreativer Qualität

### 🏗️ Architektur (Zwei-Framework-Prinzip)

```
┌─────────────────────────────────────────────────────────────┐
│                  INTELLIGENTER DIRIGENT                      │
│                    (Live-Framework)                        │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Frontend    │    FastAPI Backend    │  Echtzeit- │
│  (Port 8501)          │    (Port 8000)        │  Audio     │
│                      │                       │  Engine    │
├─────────────────────────────────────────────────────────────┤
│                   PRODUKTIONS-FABRIK                        │
│                  (Trainings-Framework)                      │
├─────────────────────────────────────────────────────────────┤
│  CLAP Model Training  │  Embeddings Pipeline  │  Database  │
│  Stem Preprocessing   │  Semantic Indexing    │  SQLite    │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 MVP-Fokus

Unsere aktuelle Mission ist die Validierung der kreativen Kernhypothese: **"Kann eine KI wirklich verstehen, was ein Produzent meint, wenn er 'einen treibenden Bass mit dunkler Atmosphäre' sucht?"**

### Erfolgsmetriken
- **Relevanz-Score**: >85% zufriedenstellende Ergebnisse bei semantischen Suchanfragen
- **Discovery-Rate**: >40% der Ergebnisse sind neue, unbekannte Stems
- **Workflow-Effizienz**: Reduktion der Suchzeit um 70% gegenüber traditionellen Methoden

## 🛠️ Technischer Stack

### Backend Core
- **FastAPI**: Moderne, asynchrone REST-API mit OpenAPI-Dokumentation
- **SQLAlchemy**: ORM für SQLite mit migrationsfähiger Architektur
- **Pydantic**: Type-safe Datenvalidierung und -serialisierung

### KI/ML Pipeline
- **LAION CLAP**: Contrastive Language-Audio Pre-training
- **Librosa**: Audio-Analyse und -verarbeitung
- **Demucs**: Source Separation für Stem-Extraktion
- **PyTorch**: Deep Learning Framework für Modelle

### Frontend & UX
- **Streamlit**: Rapid Prototyping für kreative Interfaces
- **WebSocket**: Echtzeit-Kommunikation für Live-Sessions
- **Responsive Design**: Mobile-first Ansatz für Studio- und Live-Einsatz

### Deployment & DevOps
- **Docker**: Containerisierung für beide Frameworks
- **Docker Compose**: Orchestrierung von Multi-Service-Setup
- **GitHub Actions**: CI/CD Pipeline mit automatisierten Tests

## 📦 Quick Start

### ⚡ 30-Sekunden Setup (Docker)
```bash
git clone https://github.com/your-org/neuromorphe-traum-engine.git
cd neuromorphe-traum-engine
docker-compose up --build
```

### 🔧 Lokale Entwicklung

#### Voraussetzungen
- Python 3.12+
- 8GB+ RAM (für CLAP-Modell)
- 10GB+ Speicherplatz

#### Schritt-für-Schritt
```bash
# 1. Repository klonen
git clone <repository-url>
cd neuromorphe-traum-engine

# 2. Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Umgebungsvariablen konfigurieren
cp .env.example .env
# .env anpassen: DATABASE_URL, UPLOAD_DIR, etc.

# 5. Datenbank initialisieren
python -m src.cli.migrate_database

# 6. Backend starten
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 7. Frontend starten (neues Terminal)
python -m streamlit run frontend/app.py --server.port 8501
```

### 🌐 Zugang nach Setup
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API-Dokumentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/system/health

## 🎯 Verwendung

### 🔍 Semantische Suche - Die Magie entdecken

#### Beispiel-Suchanfragen
```bash
# Basis-Suche
curl "http://localhost:8000/api/v1/stems/search/?prompt=dark%20techno%20kick"

# Komplexe emotionale Suche
curl "http://localhost:8000/api/v1/stems/search/?prompt=melancholic%20bass%20with%20industrial%20texture"

# Tempo-basierte Suche
curl "http://localhost:8000/api/v1/stems/search/?prompt=fast%20percussion%20loop%20130bpm"

# Top-K Ergebnisse
curl "http://localhost:8000/api/v1/stems/search/?prompt=atmospheric%20pad&top_k=10"
```

#### Python Integration
```python
import requests

# Semantische Suche durchführen
response = requests.get(
    "http://localhost:8000/api/v1/stems/search/",
    params={
        "prompt": "driving techno bass with analog warmth",
        "category": "bass",
        "top_k": 5,
        "min_similarity": 0.7
    }
)

results = response.json()
for stem in results:
    print(f"🎵 {stem['filename']}")
    print(f"   Kategorie: {stem['category']}")
    print(f"   Ähnlichkeit: {stem['similarity']:.3f}")
    print(f"   Pfad: {stem['path']}")
```

### 🎛️ Frontend - Der Luzide Navigator

#### Interface-Komponenten
1. **Search Portal**: Natürlichsprachige Eingabe mit Live-Vorschlägen
2. **Stem Explorer**: Visuelle Navigation durch Klang-Universen
3. **Session Recorder**: Live-Aufnahme von Discovery-Sessions
4. **AI Assistant**: Kontextuelle Empfehlungen basierend auf Suchverhalten

#### Power-User Features
- **Batch Processing**: Mehrere Stems gleichzeitig analysieren
- **Custom Embeddings**: Eigene Modelle trainieren für spezifische Genres
- **Export Pipeline**: Direkte Integration mit DAWs via MIDI-Steuerung

## 🏗️ Projektstruktur - Das neuronale Netzwerk

```
neuromorphe-traum-engine/
├── 🧠 src/                          # Backend - Das neuronale Zentrum
│   ├── api/                         # REST-API Endpunkte
│   ├── core/                        # Kernkonfiguration & Logging
│   ├── database/                    # SQLAlchemy Models & CRUD
│   ├── schemas/                     # Pydantic Type-Definitionen
│   ├── services/                    # Business Logic & KI-Services
│   ├── audio/                       # Audio-Verarbeitung Pipeline
│   └── search/                      # Semantische Such-Engine
├── 🎛️ frontend/                     # Streamlit Interface
│   ├── pages/                       # Multi-Page App Struktur
│   ├── components/                  # Wiederverwendbare UI-Komponenten
│   └── utils/                       # Frontend-Hilfsfunktionen
├── 🗄️ processed_database/           # SQLite + Embeddings
├── 📁 raw_construction_kits/        # Roh-Audio-Dateien
├── 🎯 models/                       # CLAP & Custom Modelle
├── 🐳 docker/                       # Container-Konfiguration
├── 📊 tests/                        # Test-Suite (pytest)
└── 📝 docs/                         # Technische Dokumentation
```

## 🔧 Konfiguration & Anpassung

### Umgebungsvariablen (Environment Variables)

| Variable | Beschreibung | Standardwert | Entwicklung |
|----------|--------------|--------------|-------------|
| `PROJECT_NAME` | Anwendungsname | "Neuromorphe Traum-Engine v2.0" | Anpassen |
| `DATABASE_URL` | SQLite Pfad | "sqlite:///processed_database/stems.db" | Anpassen |
| `UPLOAD_DIR` | Audio-Uploads | "./raw_construction_kits" | Anpassen |
| `MODEL_CACHE_DIR` | CLAP Cache | "./models" | Anpassen |
| `API_BASE_URL` | Backend URL | "http://localhost:8000" | Anpassen |
| `LOG_LEVEL` | Logging Level | "INFO" | DEBUG für Dev |
| `MAX_FILE_SIZE` | Max Upload Size | "100MB" | Erhöhen |

### Advanced Configuration

#### Custom CLAP Model Training
```python
# src/services/training_service.py
class CustomCLAPTrainer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.model_config = {
            'audio_encoder': 'HTSAT-base',
            'text_encoder': 'RoBERTa-base',
            'embed_dim': 512,
            'temperature': 0.07
        }
```

#### Stem-Kategorisierung erweitern
```python
# src/core/config.py
STEM_CATEGORIES = {
    'kick': ['kick', 'bassdrum', 'bd'],
    'snare': ['snare', 'clap', 'rimshot'],
    'hihat': ['hihat', 'hat', 'openhat', 'closedhat'],
    'bass': ['bass', 'sub', 'lowend'],
    'percussion': ['perc', 'shaker', 'conga', 'tom'],
    'fx': ['sweep', 'riser', 'impact', 'texture'],
    'synth': ['lead', 'pad', 'stab', 'chord'],
    'vocal': ['vox', 'voice', 'spoken', 'chant']
}
```

## 🧪 Testing & Qualitätssicherung

### Test-Suite ausführen
```bash
# Alle Tests
pytest tests/ -v

# Nur API-Tests
pytest tests/test_api/ -v

# Mit Coverage
pytest tests/ --cov=src --cov-report=html

# Performance-Tests
pytest tests/test_performance.py -v
```

### Manuelle Tests
```bash
# Health Check
curl http://localhost:8000/system/health

# API-Dokumentation
curl http://localhost:8000/docs

# Suche testen
curl "http://localhost:8000/api/v1/stems/search/?prompt=test"
```

### Test-Daten erstellen
```bash
# Beispiel-Stems generieren
python -m src.cli.create_test_audio

# Test-Datenbank befüllen
python -m src.cli.migrate_database --test-data
```

## 📊 Performance & Skalierung

### System-Anforderungen
| Komponente | Minimum | Empfohlen | Cloud |
|------------|---------|-----------|-------|
| **CPU** | 4 Cores | 8 Cores | 16 Cores |
| **RAM** | 8GB | 16GB | 32GB |
| **Storage** | 50GB | 100GB SSD | 500GB SSD |
| **Network** | 10 Mbps | 100 Mbps | 1 Gbps |

### Performance-Metriken
- **CLAP-Modell Initialisierung**: 30-60 Sekunden
- **Embedding-Berechnung**: 1-2 Sekunden pro Audio-Datei
- **Such-Latenz**: 100-500ms (abhängig von Datenbank-Größe)
- **Memory-Footprint**: 2-4GB für CLAP-Modell
- **Datenbank-Größe**: ~1MB pro 1000 Stems

### Skalierungs-Strategien
1. **Phase 1**: SQLite → PostgreSQL Migration
2. **Phase 2**: Single Instance → Microservices
3. **Phase 3**: CPU → GPU-Acceleration
4. **Phase 4**: On-Premise → Cloud-Native

## 🐛 Troubleshooting & Support

### Häufige Probleme

#### 1. CLAP-Modell lädt nicht
```bash
# Cache löschen
rm -rf models/CLAP/
# Neu herunterladen beim nächsten Start
```

#### 2. Datenbank-Lock Probleme
```bash
# SQLite Lock beheben
sqlite3 processed_database/stems.db "PRAGMA journal_mode=WAL;"
```

#### 3. Audio-Verarbeitung hängt
```bash
# Memory-Limit erhöhen
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 4. Docker Container starten nicht
```bash
# Logs prüfen
docker-compose logs -f backend
docker-compose logs -f frontend

# Ports freigeben
sudo lsof -i :8000
sudo lsof -i :8501
```

### Debug-Modus aktivieren
```bash
# Backend mit Debug-Logging
LOG_LEVEL=DEBUG python -m uvicorn src.main:app --reload

# Frontend mit Debug-Modus
STREAMLIT_DEBUG=true python -m streamlit run frontend/app.py
```

### Support-Kanäle
- **GitHub Issues**: Bug-Reports und Feature-Requests
- **Discord**: Community-Support und Diskussionen
- **Wiki**: Erweiterte Dokumentation und Tutorials

## 🚀 Roadmap & Future Vision

### Phase 1: MVP ✅
- [x] Semantische Audio-Suche
- [x] CLAP-Integration
- [x] Basic Streamlit Frontend
- [x] Docker-Containerisierung

### Phase 2: Enhanced Intelligence 🚧
- [ ] Real-time Stem-Mutation
- [ ] Adaptive Learning from User Feedback
- [ ] Advanced Audio Analysis (BPM, Key, Mood)
- [ ] Multi-language Support

### Phase 3: Creative Ecosystem 🎯
- [ ] DAW Plugin Integration (VST/AU)
- [ ] Collaborative Sessions
- [ ] Cloud-based Model Training
- [ ] Mobile Companion App

### Phase 4: Neural Synthesis 🚀
- [ ] Generative Stem Creation
- [ ] Style Transfer between Tracks
- [ ] AI-Powered Arrangement Suggestions
- [ ] Real-time Collaboration

## 🤝 Contributing & Community

### Beitragen
1. **Fork** das Repository
2. **Feature Branch** erstellen: `feature/semantic-melody-search`
3. **Commit** mit Conventional Commits: `feat: add melody search capability`
4. **Push** zum Feature Branch
5. **Pull Request** erstellen

### Code-Standards
- **PEP 8** für Python
- **Type Hints** für alle Funktionen
- **Docstrings** für öffentliche APIs
- **Tests** für neue Features
- **Performance** über Optimierung

### Community
- **Discord**: [Join our Server](https://discord.gg/neuromorphic)
- **Twitter**: [@NeuromorphicAI](https://twitter.com/neuromorphicai)
- **YouTube**: Tutorials und Live-Coding
- **Blog**: Deep-Dive Artikel zur KI-Musikproduktion

## 📄 Lizenz & Attribution

### Lizenz
```
MIT License - Siehe LICENSE Datei
```

### Drittanbieter
- **CLAP Model**: LAION-AI (MIT License)
- **Demucs**: Facebook Research (MIT License)
- **Librosa**: librosa.org (ISC License)
- **Streamlit**: Streamlit Inc (Apache 2.0)

### Zitierung
```bibtex
@software{neuromorphic_dream_engine,
  title={Neuromorphic Dream Engine: Semantic Audio Search for Electronic Music},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/neuromorphe-traum-engine}
}
```

---

<div align="center">

**🎵 *"Where neural networks dream in sound"* 🎵**

*Built with ❤️ for the electronic music community*

[Getting Started](#-quick-start) • [Documentation](docs/) • [API Reference](http://localhost:8000/docs) • [Report Bug](../../issues)

</div>