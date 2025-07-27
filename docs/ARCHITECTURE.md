# Technische Architektur - Neuromorphe Traum-Engine v2.0

## Überblick

Die Neuromorphe Traum-Engine v2.0 ist ein modulares System für semantische Audio-Suche, das auf modernen Deep Learning-Technologien basiert. Die Architektur folgt dem Prinzip der Trennung von Preprocessing und Retrieval, um Skalierbarkeit und Performance zu optimieren.

## Systemarchitektur

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neuromorphe Traum-Engine v2.0                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │ Processing Layer│    │  Output Layer   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Audio Files   │───▶│ • CLAP Model    │───▶│ • Search Results│
│ • Text Queries  │    │ • Embeddings    │    │ • Similarity    │
│ • Metadata      │    │ • Indexing      │    │ • Rankings      │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Storage Layer                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Raw Audio Files │ Embedding Cache │      Metadata Database      │
│ (.wav, .flac)   │ (.pkl, .npy)    │     (SQLite, JSON)         │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Kernkomponenten

### 1. CLAP-Embedding-Engine

**Technologie**: LAION-CLAP (Contrastive Language-Audio Pre-training)

**Architektur**:
```
Audio Input (Waveform)
        ↓
┌─────────────────┐
│  Audio Encoder  │ ← ResNet-based CNN
│  (ResNet-38)    │
└─────────────────┘
        ↓
┌─────────────────┐
│ Audio Embedding │ ← 512-dimensional vector
│   (512-dim)     │
└─────────────────┘

Text Input (Natural Language)
        ↓
┌─────────────────┐
│  Text Encoder   │ ← Transformer-based
│  (RoBERTa)      │
└─────────────────┘
        ↓
┌─────────────────┐
│ Text Embedding  │ ← 512-dimensional vector
│   (512-dim)     │
└─────────────────┘

┌─────────────────┐
│ Shared Latent   │ ← Cosine similarity space
│     Space       │
└─────────────────┘
```

**Eigenschaften**:
- **Dimensionalität**: 512-dimensional embeddings
- **Normalisierung**: L2-normalisiert für Kosinus-Ähnlichkeit
- **Multimodalität**: Gemeinsamer Vektorraum für Audio und Text
- **Pre-training**: Auf großen Audio-Text-Paaren trainiert

### 2. Preprocessing-Pipeline

#### Minimal Preprocessor

```python
class MinimalPreprocessor:
    """
    Einfache, effiziente Preprocessing-Pipeline
    Optimiert für schnelle Prototyping und kleine bis mittlere Datensätze
    """
    
    def __init__(self, input_dir, output_path):
        self.clap_model = CLAP_Module(enable_fusion=False)
        self.batch_size = 32
    
    def run(self):
        # 1. Audio-Dateien entdecken
        # 2. Batch-weise CLAP-Embedding-Extraktion
        # 3. Pickle-Serialisierung
        pass
```

**Datenfluss**:
```
Audio Files (.wav)
        ↓
┌─────────────────┐
│ File Discovery  │ ← Recursive directory scan
└─────────────────┘
        ↓
┌─────────────────┐
│ Batch Formation │ ← Group files (batch_size=32)
└─────────────────┘
        ↓
┌─────────────────┐
│ CLAP Processing │ ← get_audio_embedding_from_filelist()
└─────────────────┘
        ↓
┌─────────────────┐
│ Serialization   │ ← pickle.dump(embeddings.pkl)
└─────────────────┘
```

#### SQL-basierter Preprocessor

```python
class NeuroAnalyzer:
    """
    Erweiterte Preprocessing-Pipeline mit robuster Fehlerbehandlung
    Optimiert für große Datensätze und Produktionsumgebungen
    """
    
    def __init__(self, input_dir, **kwargs):
        self.clap_model = CLAP_Module(enable_fusion=False)
        self.kmeans = KMeans(n_clusters=8)  # Kategorisierung
        self.checkpoint_system = CheckpointManager()
    
    def run(self):
        # 1. Resume-Funktionalität
        # 2. Robuste Batch-Verarbeitung
        # 3. Erweiterte Metadaten-Extraktion
        # 4. SQLite-Persistierung
        pass
```

**Erweiterte Features**:
- **Resume-Funktionalität**: Checkpoint-basierte Wiederaufnahme
- **Fehlerbehandlung**: Retry-Mechanismus mit exponential backoff
- **Metadaten-Extraktion**: BPM, Tonart, Audio-Features
- **Kategorisierung**: KMeans-basierte automatische Klassifikation
- **Progress-Tracking**: Detaillierte Fortschritts- und Fehler-Logs

### 3. Search Engine

```python
class SearchEngine:
    """
    Hochperformante semantische Suchmaschine
    Optimiert für Echtzeit-Abfragen
    """
    
    def __init__(self, embeddings_path):
        self.clap_model = CLAP_Module(enable_fusion=False)
        self.embedding_tensors = self._load_embeddings()
        self.file_paths = self._load_file_paths()
    
    def search(self, prompt, top_k=5):
        # 1. Text-Embedding-Generierung
        # 2. Kosinus-Ähnlichkeits-Berechnung
        # 3. Top-K-Retrieval
        # 4. Ergebnis-Ranking
        pass
```

**Algorithmus**:
```
Text Query
    ↓
┌─────────────────┐
│ Text Embedding  │ ← CLAP text encoder
│   Generation    │
└─────────────────┘
    ↓
┌─────────────────┐
│ Similarity      │ ← cosine_similarity(text_emb, audio_embs)
│  Computation    │
└─────────────────┘
    ↓
┌─────────────────┐
│ Top-K Selection │ ← torch.topk(similarities, k)
└─────────────────┘
    ↓
┌─────────────────┐
│ Result Ranking  │ ← Sort by similarity score
└─────────────────┘
```

## Datenstrukturen

### Embedding-Format (Pickle)

```python
# embeddings.pkl
List[Dict[str, Any]] = [
    {
        'path': str,           # Absoluter Dateipfad
        'embedding': np.ndarray # Shape: (512,), dtype: float32
    },
    ...
]
```

### SQL-Schema (Erweitert)

```sql
-- Haupttabelle für Audio-Metadaten
CREATE TABLE stems (
    id TEXT PRIMARY KEY,              -- SHA-256 Hash
    path TEXT NOT NULL,               -- Dateipfad
    filename TEXT,                    -- Dateiname
    file_size INTEGER,                -- Dateigröße in Bytes
    duration REAL,                    -- Länge in Sekunden
    sample_rate INTEGER,              -- Sample Rate
    channels INTEGER,                 -- Anzahl Kanäle
    bit_depth INTEGER,                -- Bit-Tiefe
    
    -- Audio-Analyse
    bpm REAL,                         -- Beats per Minute
    key TEXT,                         -- Musikalische Tonart
    tempo_confidence REAL,            -- BPM-Erkennungsgenauigkeit
    
    -- Kategorisierung
    category TEXT,                    -- Automatisch erkannte Kategorie
    category_confidence REAL,         -- Kategorisierungs-Konfidenz
    tags TEXT,                        -- JSON-Array semantischer Tags
    
    -- Audio-Features (JSON)
    spectral_features TEXT,           -- Spektrale Eigenschaften
    temporal_features TEXT,           -- Zeitliche Eigenschaften
    harmonic_features TEXT,           -- Harmonische Eigenschaften
    
    -- Qualitätsbewertung
    quality_ok BOOLEAN,               -- Automatische Qualitätsprüfung
    quality_score REAL,               -- Qualitätswert (0-1)
    noise_level REAL,                 -- Rauschpegel
    dynamic_range REAL,               -- Dynamikumfang
    
    -- Benutzer-Metadaten
    user_rating INTEGER,              -- Benutzer-Rating (1-5)
    user_tags TEXT,                   -- Benutzer-definierte Tags
    usage_count INTEGER DEFAULT 0,   -- Verwendungshäufigkeit
    
    -- System-Metadaten
    imported_at DATETIME,             -- Import-Zeitstempel
    last_accessed DATETIME,           -- Letzter Zugriff
    processing_version TEXT,          -- Verarbeitungsversion
    
    -- Embeddings
    clap_embedding BLOB NOT NULL      -- Serialisiertes CLAP-Embedding
);

-- Verarbeitungsstatus für Resume-Funktionalität
CREATE TABLE processing_status (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT,                   -- SHA-256 für Änderungserkennung
    status TEXT,                      -- ProcessingStatus enum
    last_attempt DATETIME,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    processing_time REAL,
    
    -- Checkpoint-Daten
    checkpoint_data TEXT              -- JSON mit Zwischenergebnissen
);

-- Indizes für Performance
CREATE INDEX idx_stems_category ON stems(category);
CREATE INDEX idx_stems_bpm ON stems(bpm);
CREATE INDEX idx_stems_key ON stems(key);
CREATE INDEX idx_stems_quality ON stems(quality_ok, quality_score);
CREATE INDEX idx_stems_imported ON stems(imported_at);
CREATE INDEX idx_processing_status ON processing_status(status, last_attempt);
```

## Performance-Optimierungen

### GPU-Beschleunigung

```python
class GPUOptimizedProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clap_model = CLAP_Module(enable_fusion=False)
        self.clap_model.to(self.device)
        
        # Mixed Precision für bessere Performance
        self.scaler = torch.cuda.amp.GradScaler()
    
    def process_batch(self, audio_batch):
        with torch.cuda.amp.autocast():
            embeddings = self.clap_model.get_audio_embedding_from_data(
                x=audio_batch.to(self.device)
            )
        return embeddings.cpu().numpy()
```

### Speicher-Management

```python
class MemoryEfficientProcessor:
    def __init__(self, max_memory_gb=8):
        self.max_memory = max_memory_gb * 1024**3  # Bytes
        self.current_memory = 0
    
    def adaptive_batch_size(self, file_sizes):
        """Dynamische Batch-Größen-Anpassung basierend auf verfügbarem Speicher"""
        estimated_memory_per_file = np.mean(file_sizes) * 10  # Heuristik
        max_batch_size = int(self.max_memory / estimated_memory_per_file)
        return min(max_batch_size, 64)  # Maximum 64
    
    def memory_cleanup(self):
        """Aggressive Speicherbereinigung"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Parallele Verarbeitung

```python
class ParallelProcessor:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or min(8, os.cpu_count())
    
    def parallel_file_loading(self, file_paths):
        """Paralleles Laden von Audio-Dateien"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(librosa.load, path) for path in file_paths]
            results = [future.result() for future in as_completed(futures)]
        return results
    
    def parallel_feature_extraction(self, audio_data_list):
        """Parallele Extraktion von Audio-Features"""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.extract_features, audio) 
                      for audio in audio_data_list]
            results = [future.result() for future in as_completed(futures)]
        return results
```

## Skalierbarkeits-Strategien

### Horizontale Skalierung

```python
class DistributedProcessor:
    """Verteilte Verarbeitung für sehr große Datensätze"""
    
    def __init__(self, worker_nodes):
        self.worker_nodes = worker_nodes
        self.task_queue = Queue()
    
    def distribute_workload(self, file_list, chunk_size=1000):
        """Verteilt Dateien auf Worker-Nodes"""
        chunks = [file_list[i:i+chunk_size] 
                 for i in range(0, len(file_list), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            worker_id = i % len(self.worker_nodes)
            self.assign_task(worker_id, chunk)
    
    def collect_results(self):
        """Sammelt Ergebnisse von allen Worker-Nodes"""
        all_embeddings = []
        for worker in self.worker_nodes:
            embeddings = worker.get_results()
            all_embeddings.extend(embeddings)
        return all_embeddings
```

### Caching-Strategien

```python
class EmbeddingCache:
    """Intelligentes Caching für Embeddings"""
    
    def __init__(self, cache_dir="cache", max_size_gb=10):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size_gb * 1024**3
        self.cache_index = self._load_cache_index()
    
    def get_embedding(self, file_path, file_hash):
        """Lädt Embedding aus Cache oder berechnet neu"""
        cache_key = f"{file_hash}.npy"
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            return np.load(cache_path)
        else:
            embedding = self._compute_embedding(file_path)
            self._store_in_cache(cache_key, embedding)
            return embedding
    
    def _cleanup_cache(self):
        """LRU-basierte Cache-Bereinigung"""
        cache_files = list(self.cache_dir.glob("*.npy"))
        cache_files.sort(key=lambda x: x.stat().st_atime)  # Nach Zugriff sortieren
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        while total_size > self.max_size and cache_files:
            oldest_file = cache_files.pop(0)
            total_size -= oldest_file.stat().st_size
            oldest_file.unlink()
```

## Monitoring und Debugging

### Logging-System

```python
import logging
from datetime import datetime

class NeuroLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('neuromorphe_engine')
        self.logger.setLevel(log_level)
        
        # File Handler
        file_handler = logging.FileHandler(
            f'logs/neuromorphe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_processing_stats(self, stats):
        self.logger.info(f"Processing Statistics: {stats}")
    
    def log_error_with_context(self, error, context):
        self.logger.error(f"Error: {error}, Context: {context}")
```

### Performance-Metriken

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'processing_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'batch_sizes': [],
            'error_rates': []
        }
    
    @contextmanager
    def measure_processing_time(self):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        yield
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        self.metrics['processing_times'].append(end_time - start_time)
        self.metrics['memory_usage'].append(end_memory - start_memory)
    
    def get_performance_report(self):
        return {
            'avg_processing_time': np.mean(self.metrics['processing_times']),
            'max_memory_usage': max(self.metrics['memory_usage']),
            'total_files_processed': len(self.metrics['processing_times']),
            'error_rate': np.mean(self.metrics['error_rates'])
        }
```

## Erweiterbarkeit

### Plugin-System

```python
class PluginInterface:
    """Basis-Interface für Plugins"""
    
    def process_audio(self, audio_data, sample_rate):
        """Verarbeitet Audio-Daten"""
        raise NotImplementedError
    
    def extract_features(self, audio_data, sample_rate):
        """Extrahiert Features aus Audio-Daten"""
        raise NotImplementedError
    
    def get_metadata(self):
        """Gibt Plugin-Metadaten zurück"""
        raise NotImplementedError

class PluginManager:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, name, plugin_class):
        """Registriert ein neues Plugin"""
        self.plugins[name] = plugin_class()
    
    def apply_plugins(self, audio_data, sample_rate):
        """Wendet alle registrierten Plugins an"""
        results = {}
        for name, plugin in self.plugins.items():
            try:
                results[name] = plugin.process_audio(audio_data, sample_rate)
            except Exception as e:
                logging.error(f"Plugin {name} failed: {e}")
        return results
```

### Custom Embedding Models

```python
class EmbeddingModelInterface:
    """Interface für alternative Embedding-Modelle"""
    
    def load_model(self):
        raise NotImplementedError
    
    def get_audio_embedding(self, audio_data):
        raise NotImplementedError
    
    def get_text_embedding(self, text):
        raise NotImplementedError
    
    def get_embedding_dimension(self):
        raise NotImplementedError

class CustomCLAPModel(EmbeddingModelInterface):
    """Beispiel für ein angepasstes CLAP-Modell"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        if self.model_path:
            # Lade custom model
            self.model = torch.load(self.model_path)
        else:
            # Lade Standard-CLAP
            self.model = CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()
    
    def get_audio_embedding(self, audio_data):
        return self.model.get_audio_embedding_from_data(audio_data)
    
    def get_text_embedding(self, text):
        return self.model.get_text_embedding([text])
    
    def get_embedding_dimension(self):
        return 512  # CLAP embedding dimension
```

## Deployment-Strategien

### Docker-Container

```dockerfile
# Dockerfile für Production Deployment
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Application code
COPY . .

# Create necessary directories
RUN mkdir -p raw_construction_kits processed_database logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Default command
CMD ["python3", "ai_agents/search_engine_cli.py"]
```

### Kubernetes-Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuromorphe-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuromorphe-engine
  template:
    metadata:
      labels:
        app: neuromorphe-engine
    spec:
      containers:
      - name: neuromorphe-engine
        image: neuromorphe-engine:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: audio-storage
          mountPath: /app/raw_construction_kits
        - name: processed-storage
          mountPath: /app/processed_database
      volumes:
      - name: audio-storage
        persistentVolumeClaim:
          claimName: audio-pvc
      - name: processed-storage
        persistentVolumeClaim:
          claimName: processed-pvc
```

---

*Diese Architektur-Dokumentation beschreibt die technischen Grundlagen der Neuromorphe Traum-Engine v2.0. Für Implementierungsdetails siehe die entsprechenden Quellcode-Dateien.*