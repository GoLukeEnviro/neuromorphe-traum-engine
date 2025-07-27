# 🧠 Neuromorphe Traum-Engine v2.0

*Semantic Audio Search with CLAP Embeddings*

## 📋 Übersicht

Die Neuromorphe Traum-Engine v2.0 ist eine moderne, semantische Audio-Suchmaschine, die CLAP (Contrastive Language-Audio Pre-training) Embeddings verwendet, um natürlichsprachliche Suchanfragen in Audio-Inhalten zu ermöglichen. Das System kombiniert eine FastAPI-Backend-Architektur mit einem benutzerfreundlichen Streamlit-Frontend.

## ✨ Features

### 🎵 Audio-Verarbeitung
- **Multi-Format-Support**: WAV, MP3, FLAC, OGG, M4A
- **Automatische Metadaten-Extraktion**: BPM, Dauer, Sampling-Rate
- **CLAP-Embedding-Generierung**: Semantische Audio-Repräsentationen
- **Batch-Verarbeitung**: Effiziente Verarbeitung mehrerer Dateien

### 🔍 Semantische Suche
- **Text-zu-Audio-Suche**: Natürlichsprachliche Suchanfragen
- **Ähnlichkeitssuche**: Finde ähnliche Audio-Dateien
- **Erweiterte Filter**: Kategorie, BPM-Bereich, Ähnlichkeitsschwelle
- **Fuzzy-Suche**: Tolerante Textsuche

### 📊 Analytics & Statistiken
- **Suchstatistiken**: Anzahl Dateien, Kategorien, Embeddings
- **Performance-Metriken**: Suchzeiten, Cache-Effizienz
- **Kategorie-Analyse**: Verteilung und Trends

### 🎨 Benutzeroberfläche
- **Moderne Streamlit-UI**: Responsive und intuitiv
- **Real-time Updates**: Live-Status und Fortschrittsanzeigen
- **Export-Funktionen**: CSV, JSON, Dateilisten
- **Konfigurierbare Einstellungen**: Backend-Verbindung, Suchparameter

## 🏗️ Architektur

### Backend (FastAPI)
```
src/
├── api/                 # API Router und Endpunkte
├── audio/              # Audio-Verarbeitung und -Management
├── search/             # Semantische Suchlogik
├── database/           # SQLite-Datenbankoperationen
└── main.py            # FastAPI-Anwendung
```

### Frontend (Streamlit)
```
pages/
├── audio_upload.py     # Audio-Upload und -Verarbeitung
├── search.py          # Suchinterface
├── results.py         # Ergebnisanzeige
└── settings.py        # Anwendungseinstellungen
```

## 🚀 Installation

### Voraussetzungen
- Python 3.8+
- Git
- Mindestens 4GB RAM (für CLAP-Modell)

### Setup

1. **Repository klonen**
```bash
git clone <repository-url>
cd neuromorphe-traum-engine
```

2. **Virtual Environment erstellen**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Dependencies installieren**
```bash
pip install -r requirements.txt
```

4. **Verzeichnisstruktur erstellen**
```bash
mkdir -p data/audio data/embeddings data/database
```

## 🎯 Verwendung

### Backend starten
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend starten
```bash
streamlit run streamlit_app.py
```

### API-Dokumentation
Nach dem Start des Backends ist die interaktive API-Dokumentation verfügbar unter:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API-Endpunkte

### Audio-Management
- `POST /api/audio/upload` - Audio-Datei hochladen
- `GET /api/audio/files` - Alle Audio-Dateien auflisten
- `GET /api/audio/files/{file_id}` - Datei-Details abrufen
- `POST /api/audio/files/{file_id}/process` - Embedding generieren
- `GET /api/audio/files/{file_id}/embedding` - Embedding abrufen

### Semantische Suche
- `POST /api/search/text` - Text-zu-Audio-Suche
- `POST /api/search/similar` - Ähnlichkeitssuche
- `GET /api/search/stats` - Suchstatistiken
- `GET /api/search/categories` - Verfügbare Kategorien

### System
- `GET /api/health` - System-Gesundheitsprüfung
- `GET /api/audio/health` - Audio-Service-Status
- `GET /api/search/health` - Search-Service-Status

## 🔧 Konfiguration

### Umgebungsvariablen
```bash
# Backend-Konfiguration
BACKEND_HOST=localhost
BACKEND_PORT=8000

# Verzeichnisse
AUDIO_DIR=./data/audio
EMBEDDINGS_DIR=./data/embeddings
DATABASE_PATH=./data/database/audio_files.db

# CLAP-Modell
CLAP_MODEL_VERSION=630k-audioset-best

# Upload-Limits
MAX_FILE_SIZE=100MB
ALLOWED_EXTENSIONS=wav,mp3,flac,ogg,m4a
```

### Streamlit-Konfiguration
```toml
# .streamlit/config.toml
[server]
port = 8501
maxUploadSize = 100

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🧪 Testing

### Unit Tests ausführen
```bash
pytest tests/ -v
```

### API Tests
```bash
pytest tests/test_api.py -v
```

### Integration Tests
```bash
pytest tests/test_integration.py -v
```

## 📊 Performance

### Benchmarks
- **Audio-Upload**: ~2-5s pro Datei (abhängig von Größe)
- **Embedding-Generierung**: ~10-30s pro Datei (GPU empfohlen)
- **Suchzeit**: <1s für 1000+ Dateien
- **Speicherverbrauch**: ~2-4GB (CLAP-Modell geladen)

### Optimierungen
- **GPU-Beschleunigung**: CUDA-Support für CLAP-Modell
- **Caching**: LRU-Cache für häufige Suchanfragen
- **Batch-Processing**: Parallele Embedding-Generierung
- **Database-Indizierung**: Optimierte SQLite-Abfragen

## 🛠️ Entwicklung

### Code-Struktur
- **Domain-Driven Design**: Modulare Architektur
- **Dependency Injection**: Lose gekoppelte Services
- **Async/Await**: Asynchrone Verarbeitung
- **Type Hints**: Vollständige Typisierung
- **Pydantic**: Datenvalidierung und -serialisierung

### Beitragen
1. Fork des Repositories
2. Feature-Branch erstellen (`git checkout -b feature/amazing-feature`)
3. Änderungen committen (`git commit -m 'Add amazing feature'`)
4. Branch pushen (`git push origin feature/amazing-feature`)
5. Pull Request erstellen

## Projektstruktur

```
Neuromorphe Traum-Engine v2.0/
├── Construction_Plans/          # Projektdirektiven und Architektur
│   ├── AGENTEN_DIREKTIVE_001.md
│   ├── AGENTEN_DIREKTIVE_002.md
│   ├── AGENTEN_DIREKTIVE_003.md
│   ├── AGENTEN_DIREKTIVE_004.md
│   ├── AGENTEN_DIREKTIVE_005_(MASTER_MVP_auf_bestehender_Architektur).md
│   ├── AGENTEN_DIREKTIVE_006.md
│   ├── AGENTEN_DIREKTIVE_007_(MASTER_MVP).md
│   └── AGENTEN_DIREKTIVE_008
├── ai_agents/                   # Hauptkomponenten des Systems
│   ├── minimal_preprocessor.py  # Einfacher CLAP-Embedding Preprocessor
│   ├── optimized_preprocessor.py
│   ├── prepare_dataset_sql.py   # Erweiterte SQL-basierte Verarbeitung
│   ├── search_engine_cli.py     # Kommandozeilen-Suchinterface
│   └── test_prepare_dataset.py
├── raw_construction_kits/       # Input-Verzeichnis für Audio-Dateien
├── processed_database/          # Output-Verzeichnis für verarbeitete Daten
├── requirements.txt             # Python-Abhängigkeiten
└── [verschiedene Test- und Validierungsskripte]
```

## Kernkomponenten

### 1. Minimal Preprocessor (`ai_agents/minimal_preprocessor.py`)

**Zweck**: Erstellt CLAP-Embeddings für Audio-Dateien und speichert sie in einer kompakten Binärdatei.

**Funktionalität**:
- Lädt das LAION-CLAP-Modell
- Durchsucht rekursiv alle .wav-Dateien im Input-Verzeichnis
- Verarbeitet Audio-Dateien in Batches (Standardgröße: 32)
- Speichert Embeddings als Pickle-Datei (`processed_database/embeddings.pkl`)

**Verwendung**:
```bash
python ai_agents/minimal_preprocessor.py
```

### 2. Search Engine CLI (`ai_agents/search_engine_cli.py`)

**Zweck**: Interaktive Kommandozeilen-Suchmaschine für semantische Audio-Suche.

**Funktionalität**:
- Lädt vorverarbeitete CLAP-Embeddings
- Berechnet Text-Embeddings für Benutzer-Prompts
- Findet die ähnlichsten Audio-Dateien mittels Kosinus-Ähnlichkeit
- Gibt Top-5 Ergebnisse mit Ähnlichkeitswerten aus

**Verwendung**:
```bash
python ai_agents/search_engine_cli.py
```

**Beispiel-Prompts**:
- "dark industrial kick with heavy bass"
- "melodic synth pad"
- "punchy snare with reverb"

### 3. SQL-basierter Preprocessor (`ai_agents/prepare_dataset_sql.py`)

**Zweck**: Erweiterte Verarbeitung mit SQLite-Datenbank und zusätzlichen Metadaten.

**Funktionalität**:
- Erstellt SQLite-Datenbank (`processed_database/stems.db`)
- Extrahiert umfangreiche Audio-Metadaten (BPM, Tonart, etc.)
- Speichert CLAP-Embeddings in der Datenbank
- Implementiert robuste Batch-Verarbeitung mit Resume-Funktionalität
- Kategorisiert Audio-Dateien automatisch

## Installation und Setup

### Voraussetzungen
- Python 3.8+
- CUDA-kompatible GPU (empfohlen für bessere Performance)

### Installation

1. **Repository klonen/herunterladen**

2. **Abhängigkeiten installieren**:
```bash
pip install -r requirements.txt
```

3. **Verzeichnisstruktur erstellen**:
```bash
mkdir raw_construction_kits
mkdir processed_database
```

## Verwendung

### Schnellstart (Minimal MVP)

1. **Audio-Dateien hinzufügen**:
   - Legen Sie .wav-Dateien in den Ordner `raw_construction_kits/`

2. **Embeddings erstellen**:
```bash
python ai_agents/minimal_preprocessor.py
```

3. **Suche starten**:
```bash
python ai_agents/search_engine_cli.py
```

### Erweiterte Verwendung (SQL-basiert)

1. **Audio-Dateien verarbeiten**:
```bash
python ai_agents/prepare_dataset_sql.py
```

2. **Erweiterte Suche** (falls implementiert):
```bash
python search_engine_cli.py
```

## Abhängigkeiten

### Kern-Bibliotheken
- **librosa** (≥0.10.0): Audio-Analyse und -Verarbeitung
- **laion-clap**: CLAP-Modell für Audio-Text-Embeddings
- **torch** (≥1.9.0): PyTorch für Deep Learning
- **numpy** (≥1.21.0): Numerische Berechnungen
- **soundfile** (≥0.12.0): Audio-Dateien lesen/schreiben
- **scikit-learn** (≥1.0.0): Machine Learning Algorithmen
- **tqdm**: Fortschrittsanzeige

### Standard-Python-Bibliotheken
- sqlite3, os, logging, datetime, shutil, concurrent.futures, typing, hashlib, pathlib, json

## Architektur

### CLAP-Embeddings
Das System basiert auf LAION-CLAP (Contrastive Language-Audio Pre-training), einem multimodalen Modell, das:
- Audio-Signale in hochdimensionale Vektorräume einbettet
- Text-Beschreibungen in denselben Vektorraum projiziert
- Semantische Ähnlichkeit durch Kosinus-Ähnlichkeit berechnet

### Verarbeitungspipeline
1. **Audio-Eingabe**: .wav-Dateien im `raw_construction_kits/` Verzeichnis
2. **Preprocessing**: Extraktion von CLAP-Embeddings
3. **Speicherung**: Pickle-Datei oder SQLite-Datenbank
4. **Suche**: Text-zu-Audio-Matching via Embedding-Ähnlichkeit

## Konfiguration

### Batch-Verarbeitung
- **Batch-Größe**: 32 (anpassbar in `minimal_preprocessor.py`)
- **Retry-Mechanismus**: Automatische Wiederholung bei Fehlern
- **Checkpoint-System**: Resume-Funktionalität für große Datensätze

### Performance-Optimierung
- **GPU-Beschleunigung**: Automatische CUDA-Nutzung wenn verfügbar
- **Parallele Verarbeitung**: Multi-Threading für I/O-Operationen
- **Speicher-Effizienz**: Batch-weise Verarbeitung großer Datensätze

## Entwicklung und Testing

### Test-Skripte
- `test_implementation.py`: Allgemeine Implementierungstests
- `test_mvp_system.py`: MVP-System-Tests
- `validate_directive_003.py`: Validierung spezifischer Direktiven
- `validate_directive_004.py`: Weitere Direktiven-Validierung

### Debugging-Tools
- `check_db.py`: Datenbank-Integritätsprüfung
- `check_paths.py`: Pfad-Validierung
- `verify_clap_embeddings.py`: CLAP-Embedding-Verifikation

## Fehlerbehebung

### Häufige Probleme

1. **CLAP-Modell lädt nicht**:
   - Überprüfen Sie die Internetverbindung (Modell wird beim ersten Start heruntergeladen)
   - Stellen Sie sicher, dass genügend Speicherplatz verfügbar ist

2. **Keine Audio-Dateien gefunden**:
   - Überprüfen Sie, dass .wav-Dateien im `raw_construction_kits/` Verzeichnis liegen
   - Stellen Sie sicher, dass die Dateien gültige Audio-Formate sind

3. **Speicher-Probleme**:
   - Reduzieren Sie die Batch-Größe in der Konfiguration
   - Verwenden Sie die SQL-basierte Verarbeitung für große Datensätze

4. **Performance-Probleme**:
   - Stellen Sie sicher, dass CUDA verfügbar ist für GPU-Beschleunigung
   - Überprüfen Sie die verfügbaren Systemressourcen

### Logs und Debugging
- Logs werden in der Konsole ausgegeben
- Detaillierte Fehlerinformationen in `processed_database/checkpoints/failed_files.json`
- Progress-Tracking in `processed_database/checkpoints/progress.json`

## Roadmap und Erweiterungen

### Geplante Features
- Web-basierte Benutzeroberfläche
- Erweiterte Audio-Metadaten-Extraktion
- Batch-Export-Funktionalität
- Plugin-System für DAWs
- Cloud-basierte Verarbeitung

### Experimentelle Features
- Automatische Tag-Generierung
- Cluster-basierte Kategorisierung
- Qualitätsbewertung von Audio-Dateien
- Benutzer-Rating-System

## Lizenz und Beiträge

Dieses Projekt ist Teil der Neuromorphe Traum-Engine Initiative. Beiträge sind willkommen - bitte folgen Sie den Coding-Standards und erstellen Sie Tests für neue Features.

## Support

Für technische Fragen und Support:
1. Überprüfen Sie die Dokumentation und häufigen Probleme
2. Schauen Sie in die Construction_Plans für detaillierte Architektur-Informationen
3. Erstellen Sie ein Issue mit detaillierter Problembeschreibung

---

*Neuromorphe Traum-Engine v2.0 - Semantische Audio-Suche der nächsten Generation*