# Neuromorphe Traum-Engine v2.0

## Überblick

Die Neuromorphe Traum-Engine v2.0 ist ein fortschrittliches System für semantische Audio-Suche, das auf CLAP (Contrastive Language-Audio Pre-training) Embeddings basiert. Das System ermöglicht es Benutzern, Audio-Dateien durch natürlichsprachliche Beschreibungen zu finden und zu kategorisieren.

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