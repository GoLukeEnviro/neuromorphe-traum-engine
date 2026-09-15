# Projektarchitektur: Neuromorphe Traum-Engine v2.0

## Überblick

Die Neuromorphe Traum-Engine v2.0 ist ein modulares System für semantische Audio-Suche und generative Musikproduktion. Es basiert auf modernen Deep Learning-Technologien und folgt einem "Zwei-Framework-Prinzip":

- **Produktions-Fabrik (Backend)**: Kümmert sich um rechenintensive Aufgaben wie Audio-Verarbeitung, KI-Embeddings, Datenbankverwaltung und API-Endpunkte.
- **Intelligenter Dirigent (Frontend)**: Bietet eine intuitive Benutzeroberfläche für die Interaktion mit dem System, einschließlich Suche, Upload und Track-Generierung.

## Systemarchitektur (Mermaid Diagramm)

```mermaid
graph TD
    subgraph Frontend (Intelligenter Dirigent)
        F[Streamlit App] -->|HTTP/API Calls| B
        F -->|User Interaction| U[User]
    end

    subgraph Backend (Produktions-Fabrik)
        B[FastAPI Backend]
        B -->|SQL Queries| D(SQLite Database stems.db)
        B -->|File Operations| RA[raw_construction_kits/]
        B -->|File Operations| PD[processed_database/stems/]
        B -->|File Operations| GT[generated_tracks/]
        B -->|Uses| CLAP[CLAP AI Model]
        B -->|Uses| VAE[VAE Models]
        B -->|Uses| DEMUCS[Demucs Separation Model]
    end

    subgraph Data Storage
        D -->|Stores Metadata & Embeddings| B
        RA -->|Input Audio| B
        PD -->|Processed Stems| B
        GT -->|Generated Tracks| B
    end

    subgraph Core Services
        B --> PS(PreprocessorService)
        B --> SS(SearchService)
        B --> TS(TrainingService)
        B --> GS(GenerativeService)
        B --> AS(ArrangerService)
        B --> RS(RendererService)
        B --> LS(LivePlayerService)
        B --> NA(NeuroAnalyzer)
        B --> SepS(SeparationService)
    end

    U --> F
    PS --> D
    PS --> PD
    SS --> D
    TS --> D
    TS --> VAE
    GS --> VAE
    GS --> PD
    AS --> SS
    AS --> D
    RS --> GT
    SepS --> RA
    SepS --> PD
    NA --> D
    NA --> CLAP
    CLAP --> PS
    CLAP --> SS
```

## Hauptkomponenten und deren Interaktionen

### 1. Frontend (Streamlit App)
- **`frontend/app.py`**: Haupt-Einstiegspunkt der Streamlit-Anwendung. Verwaltet die Navigation und den globalen Zustand.
- **`frontend/main_app.py`**: Eine weitere Hauptanwendung, die die Seiten initialisiert und den Backend-Status prüft.
- **`frontend/pages/`**: Enthält verschiedene Seiten der Anwendung wie `search.py` (semantische Suche), `upload.py` (Audio-Upload), `results.py` (Suchergebnisse) und `settings.py` (Einstellungen).
- **`frontend/utils/api.py`**: Stellt Funktionen für die Kommunikation mit dem FastAPI-Backend bereit.

### 2. Backend (FastAPI)
- **`src/main.py`**: Der zentrale Einstiegspunkt für die FastAPI-Anwendung.
- **`src/api/endpoints/`**: Definiert die REST-API-Endpunkte.
    - **`neuromorphic.py`**: Endpunkte für Preprocessing, VAE-Training, Stem-Generierung und Track-Rendering.
    - **`stems.py`**: Endpunkte für die Verwaltung und semantische Suche von Audio-Stems.
    - **`health.py`**: Health-Check-Endpunkt.
- **`src/services/`**: Enthält die Business-Logik und Kernfunktionalitäten.
    - **`preprocessor.py`**: Analysiert Audio-Dateien, extrahiert Metadaten und CLAP-Embeddings.
    - **`search.py`**: Führt semantische Suchen basierend auf Text-Prompts und CLAP-Embeddings durch.
    - **`arranger.py`**: Generiert musikalische Arrangements aus Text-Prompts und wählt passende Stems aus.
    - **`renderer.py`**: Rendert den finalen Track basierend auf dem Arrangement-Plan.
    - **`separation_service.py`**: Nutzt Demucs zur Trennung von Stereo-Tracks in einzelne Stems.
    - **`training_service.py`**: Verantwortlich für das Training der VAE-Modelle.
    - **`generative_service.py`**: Erzeugt neue Stems mittels VAEs.
    - **`neuro_analyzer.py`**: Führt erweiterte neuromorphe Audio-Analysen durch.
    - **`live_player_service.py`**: (Zukünftiges Feature) Für Echtzeit-Wiedergabe.

### 3. Datenbank (SQLite mit SQLAlchemy)
- **`src/database/models.py`**: Definiert die SQLAlchemy-Modelle für `Stem`, `GeneratedTrack`, `TrackStem`, `ProcessingJob`, `SystemMetrics` und `ConfigurationSetting`.
- **`src/database/crud.py`**: Implementiert CRUD-Operationen für alle Datenbankmodelle.
- **`src/database/database.py`**: Verwaltet die Datenbankverbindungen und -sessions (asynchron und synchron).

### 4. KI-Module und Datenflüsse
- **CLAP Model**: Wird von `PreprocessorService` und `SearchService` für Audio- und Text-Embeddings verwendet.
- **VAE Models**: Werden von `TrainingService` trainiert und von `GenerativeService` für die Stem-Generierung genutzt.
- **Demucs**: Wird von `SeparationService` für die Audio-Trennung eingesetzt.
- **Datenflüsse**:
    - Roh-Audio-Dateien (`raw_construction_kits/`) werden vom `PreprocessorService` verarbeitet.
    - Verarbeitete Stems und Metadaten werden in `processed_database/stems/` und `stems.db` gespeichert.
    - Generierte Tracks werden in `generated_tracks/` abgelegt.

### 5. Konfiguration und Build-Prozess
- **`src/core/config.py`**: Enthält globale Anwendungseinstellungen, Datenbank-URLs, Dateipfade und API-Konfigurationen.
- **`pyproject.toml` & `requirements.txt`**: Definieren die Projektmetadaten und Python-Abhängigkeiten.
- **`Makefile`**: Bietet vereinfachte Befehle für Installation, Entwicklung, Produktion, Tests und Docker-Operationen.
- **`Dockerfile.backend` & `Dockerfile.frontend`**: Dockerfiles für die Containerisierung des Backend- und Frontend-Services.
- **`docker-compose.yml`**: Orchestriert die Docker-Container für Backend und Frontend.

### 6. Teststrategie
- **`tests/test_neuromorphic_engine.py`**: Umfassender End-to-End-Test des gesamten Workflows (Separation -> Training -> Generation -> Track Creation).
- **`tests/test_api/test_endpoints.py`**: Testet die Funktionalität der FastAPI-Endpunkte.
- Weitere Testdateien in `tests/test_core/`, `tests/test_database/`, `tests/test_schemas/`, `tests/test_services/` für Unit-Tests der einzelnen Komponenten.

## Zusammenfassung des Projektverständnisses

Die Neuromorphe Traum-Engine v2.0 ist ein komplexes, aber gut strukturiertes System, das auf einer Microservice-Architektur basiert. Das Backend (FastAPI) und Frontend (Streamlit) kommunizieren über REST-APIs. Die Kernfunktionalität liegt in der KI-gestützten Audio-Analyse, semantischen Suche, generativen Stem-Erstellung und intelligenten Track-Komposition. Die Datenbank (SQLite) speichert Metadaten und Embeddings, während Dateisysteme für rohe und verarbeitete Audio-Dateien verwendet werden. Der Build-Prozess ist durch Docker und Makefile-Befehle gut definiert.

## Klärungsfragen an den Benutzer

1. Gibt es spezifische Bereiche des Projekts, die ich tiefergehend analysieren soll, oder ist dieser allgemeine Überblick ausreichend?
2. Gibt es bestimmte Anwendungsfälle oder Features, die für Sie von höchster Priorität sind und die ich bei der weiteren Planung berücksichtigen sollte?
3. Sind Sie mit dem vorgeschlagenen Plan zur weiteren Vorgehensweise (Implementierung in 'code' Modus) einverstanden?