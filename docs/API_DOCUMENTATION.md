# API Dokumentation - Neuromorphe Traum-Engine v2.0

## Überblick

Diese Dokumentation beschreibt die APIs und Schnittstellen der Neuromorphe Traum-Engine v2.0 Komponenten.

## MinimalPreprocessor API

### Klasse: `MinimalPreprocessor`

**Datei**: `ai_agents/minimal_preprocessor.py`

#### Konstruktor

```python
MinimalPreprocessor(input_dir: str, output_path: str)
```

**Parameter**:
- `input_dir` (str): Verzeichnis mit den Audio-Dateien
- `output_path` (str): Pfad für die Ausgabe-Datei (embeddings.pkl)

**Beschreibung**: Initialisiert den MinimalPreprocessor und lädt das CLAP-Modell.

#### Methoden

##### `run()`

```python
def run(self) -> None
```

**Beschreibung**: Hauptverarbeitungsschleife, die Audio-Dateien findet, in Batches verarbeitet und Embeddings speichert.

**Workflow**:
1. Sucht rekursiv alle .wav-Dateien im input_dir
2. Verarbeitet Dateien in Batches (Standardgröße: 32)
3. Erstellt CLAP-Embeddings für jeden Batch
4. Speichert Ergebnisse als Pickle-Datei

**Ausgabe**: Erstellt `embeddings.pkl` mit folgender Struktur:
```python
[
    {
        'path': '/pfad/zur/datei.wav',
        'embedding': numpy.ndarray  # CLAP-Embedding-Vektor
    },
    ...
]
```

##### `_find_audio_files()`

```python
def _find_audio_files(self) -> List[str]
```

**Rückgabe**: Liste aller gefundenen .wav-Dateipfade

##### `_process_batch(file_batch: List[str])`

```python
def _process_batch(self, file_batch: List[str]) -> List[Dict]
```

**Parameter**:
- `file_batch`: Liste von Dateipfaden für Batch-Verarbeitung

**Rückgabe**: Liste von Dictionaries mit 'path' und 'embedding'

## SearchEngine API

### Klasse: `SearchEngine`

**Datei**: `ai_agents/search_engine_cli.py`

#### Konstruktor

```python
SearchEngine(embeddings_path: str)
```

**Parameter**:
- `embeddings_path` (str): Pfad zur embeddings.pkl Datei

**Beschreibung**: Initialisiert die Suchmaschine, lädt CLAP-Modell und Embeddings.

#### Methoden

##### `search(prompt: str, top_k: int = 5)`

```python
def search(self, prompt: str, top_k: int = 5) -> List[Tuple[str, float]]
```

**Parameter**:
- `prompt` (str): Text-Prompt für die Suche
- `top_k` (int): Anzahl der zurückzugebenden Ergebnisse (Standard: 5)

**Rückgabe**: Liste von Tupeln (Dateipfad, Ähnlichkeitswert)

**Beispiel**:
```python
search_engine = SearchEngine("processed_database/embeddings.pkl")
results = search_engine.search("dark industrial kick", top_k=5)
# Rückgabe: [('/path/to/kick1.wav', 0.8234), ('/path/to/kick2.wav', 0.7891), ...]
```

**Algorithmus**:
1. Berechnet Text-Embedding für den Prompt
2. Berechnet Kosinus-Ähnlichkeit zu allen Audio-Embeddings
3. Findet Top-K ähnlichste Dateien mit `torch.topk`
4. Gibt sortierte Ergebnisse zurück

## NeuroAnalyzer API (SQL-basiert)

### Klasse: `NeuroAnalyzer`

**Datei**: `ai_agents/prepare_dataset_sql.py`

#### Konstruktor

```python
NeuroAnalyzer(
    input_dir: str,
    resume_from_checkpoint: bool = True,
    batch_size: int = 10,
    max_retries: int = 3,
    checkpoint_interval: int = 50
)
```

**Parameter**:
- `input_dir` (str): Verzeichnis mit den rohen Audiodaten
- `resume_from_checkpoint` (bool): Ob von einem Checkpoint fortgesetzt werden soll
- `batch_size` (int): Batch-Größe für CLAP-Verarbeitung
- `max_retries` (int): Maximale Anzahl von Wiederholungsversuchen
- `checkpoint_interval` (int): Intervall für Checkpoint-Speicherung

#### Methoden

##### `init_db()`

```python
def init_db(self) -> None
```

**Beschreibung**: Erstellt SQLite-Datenbank und Tabellen.

**Datenbank-Schema**:
```sql
CREATE TABLE stems (
    id TEXT PRIMARY KEY,
    path TEXT,
    bpm REAL,
    key TEXT,
    category TEXT,
    tags TEXT,
    features TEXT,
    quality_ok BOOLEAN,
    user_rating INTEGER,
    imported_at DATETIME,
    clap_embedding BLOB
);

CREATE TABLE processing_status (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT,
    status TEXT,
    last_attempt DATETIME,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    processing_time REAL
);
```

##### `run()`

```python
def run(self) -> None
```

**Beschreibung**: Hauptverarbeitungsschleife mit robuster Batch-Verarbeitung und Resume-Funktionalität.

**Features**:
- Automatische Wiederaufnahme nach Unterbrechungen
- Fehlerbehandlung und Retry-Mechanismus
- Progress-Tracking und Statistiken
- KMeans-basierte Kategorisierung

## Datenstrukturen

### Embedding-Datenformat (Pickle)

```python
# embeddings.pkl Struktur
List[Dict[str, Any]] = [
    {
        'path': str,           # Absoluter Pfad zur Audio-Datei
        'embedding': np.ndarray # CLAP-Embedding (512-dimensional)
    },
    ...
]
```

### SQL-Datenbank-Schema

#### Tabelle: `stems`

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| id | TEXT | Eindeutige ID (Hash der Datei) |
| path | TEXT | Pfad zur Audio-Datei |
| bpm | REAL | Beats per Minute |
| key | TEXT | Musikalische Tonart |
| category | TEXT | Automatisch erkannte Kategorie |
| tags | TEXT | JSON-Array mit semantischen Tags |
| features | TEXT | JSON mit Audio-Features |
| quality_ok | BOOLEAN | Qualitätsbewertung |
| user_rating | INTEGER | Benutzer-Rating (1-5) |
| imported_at | DATETIME | Import-Zeitstempel |
| clap_embedding | BLOB | Serialisiertes CLAP-Embedding |

#### Tabelle: `processing_status`

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| file_path | TEXT | Pfad zur Datei (Primary Key) |
| file_hash | TEXT | SHA-256 Hash der Datei |
| status | TEXT | Verarbeitungsstatus (Enum) |
| last_attempt | DATETIME | Letzter Verarbeitungsversuch |
| retry_count | INTEGER | Anzahl Wiederholungsversuche |
| error_message | TEXT | Fehlermeldung bei Problemen |
| processing_time | REAL | Verarbeitungszeit in Sekunden |

## Konfigurationskonstanten

### Batch-Verarbeitung

```python
BATCH_SIZE = 10              # Standard-Batch-Größe für SQL-Verarbeitung
MAX_RETRIES = 3              # Maximale Wiederholungsversuche
RETRY_DELAY = 2.0            # Verzögerung zwischen Versuchen (Sekunden)
CHECKPOINT_INTERVAL = 50     # Checkpoint-Speicherung alle N Dateien
```

### Dateipfade

```python
DB_PATH = "processed_database/stems.db"                    # SQLite-Datenbank
EMBEDDINGS_PATH = "processed_database/embeddings.pkl"      # Pickle-Embeddings
CHECKPOINT_DIR = "processed_database/checkpoints"          # Checkpoint-Verzeichnis
INPUT_DIR = "raw_construction_kits"                        # Audio-Input-Verzeichnis
```

## Status-Enums

### ProcessingStatus

```python
class ProcessingStatus(Enum):
    PENDING = "pending"           # Wartet auf Verarbeitung
    PROCESSING = "processing"     # Wird gerade verarbeitet
    COMPLETED = "completed"       # Erfolgreich abgeschlossen
    FAILED = "failed"             # Verarbeitung fehlgeschlagen
    QUARANTINED = "quarantined"   # Datei in Quarantäne
```

## Fehlerbehandlung

### Exception-Typen

- **FileNotFoundError**: Embeddings-Datei oder Audio-Dateien nicht gefunden
- **RuntimeError**: CLAP-Modell-Ladefehler
- **ValueError**: Ungültige Parameter oder Datenformate
- **sqlite3.Error**: Datenbankfehler
- **torch.cuda.OutOfMemoryError**: GPU-Speicher-Probleme

### Retry-Mechanismus

```python
# Automatische Wiederholung bei temporären Fehlern
for attempt in range(MAX_RETRIES):
    try:
        # Verarbeitungslogik
        break
    except TemporaryError as e:
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
            continue
        else:
            # Endgültiger Fehler
            raise
```

## Performance-Optimierungen

### GPU-Beschleunigung

```python
# Automatische CUDA-Erkennung
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Speicher-Management

```python
# Batch-weise Verarbeitung für große Datensätze
for batch in batches:
    embeddings = model.process_batch(batch)
    # Sofortige Speicherung und Freigabe
    save_embeddings(embeddings)
    del embeddings
    torch.cuda.empty_cache()  # GPU-Speicher freigeben
```

### Parallele Verarbeitung

```python
# ThreadPoolExecutor für I/O-Operationen
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_file, file) for file in files]
    results = [future.result() for future in as_completed(futures)]
```

## Beispiel-Workflows

### Einfacher Workflow (Minimal MVP)

```python
# 1. Preprocessing
preprocessor = MinimalPreprocessor(
    input_dir="raw_construction_kits",
    output_path="processed_database/embeddings.pkl"
)
preprocessor.run()

# 2. Suche
search_engine = SearchEngine("processed_database/embeddings.pkl")
results = search_engine.search("dark techno kick", top_k=5)

for i, (path, similarity) in enumerate(results, 1):
    print(f"{i}. {path} (Ähnlichkeit: {similarity:.4f})")
```

### Erweiteter Workflow (SQL-basiert)

```python
# 1. Erweiterte Verarbeitung
analyzer = NeuroAnalyzer(
    input_dir="raw_construction_kits",
    batch_size=16,
    resume_from_checkpoint=True
)
analyzer.init_db()
analyzer.run()

# 2. Datenbankabfrage
import sqlite3
conn = sqlite3.connect("processed_database/stems.db")
cursor = conn.cursor()

# Finde alle Kick-Drums mit hoher Qualität
cursor.execute("""
    SELECT path, category, tags, quality_ok 
    FROM stems 
    WHERE category = 'kick' AND quality_ok = 1
""")
results = cursor.fetchall()
```

## Integration und Erweiterung

### Custom Search Engine

```python
class CustomSearchEngine(SearchEngine):
    def __init__(self, embeddings_path, custom_weights=None):
        super().__init__(embeddings_path)
        self.custom_weights = custom_weights or {}
    
    def weighted_search(self, prompt, category_filter=None, top_k=5):
        # Implementierung einer gewichteten Suche
        # mit Kategorie-Filtern
        pass
```

### Plugin-Interface

```python
class AudioProcessor:
    """Basis-Interface für Audio-Verarbeitungs-Plugins"""
    
    def process(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Verarbeitet Audio-Daten und gibt Metadaten zurück"""
        raise NotImplementedError
    
    def get_features(self) -> List[str]:
        """Gibt Liste der extrahierten Features zurück"""
        raise NotImplementedError
```

---

*Diese API-Dokumentation wird kontinuierlich aktualisiert. Für die neueste Version siehe das Git-Repository.*