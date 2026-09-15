# 🧠 Neuromorphic Dream Engine

## Übersicht

Die **Neuromorphic Dream Engine** ist ein vollständig autonomes, lernendes Ökosystem für die kreative Audio-Produktion. Das System kombiniert modernste KI-Technologien für Audio Source Separation, Variational Autoencoders (VAEs) für generative Stem-Mutation und neuromorphe Analyse für intelligente Track-Komposition.

## 🎯 Kernfunktionen

### 1. **Audio Source Separation** (Demucs)
- Zerlegt Stereo-Tracks in ihre Komponenten (drums, bass, vocals, other)
- State-of-the-Art Qualität durch Meta's Demucs-Modell
- Automatische Kategorisierung und Datenbankintegration

### 2. **Generative Stem-Mutation** (VAE)
- Trainiert VAE-Modelle auf Stem-Kategorien
- Erzeugt neue, hybride Stems durch latente Raum-Interpolation
- Intelligente Variation statt chaotische Zufälligkeit

### 3. **Neuromorphe Analyse**
- Tiefe Audio-Analyse mit neuromorphen Metriken
- Qualitätsbewertung und Charakterisierung
- CLAP-Embeddings für semantische Suche

### 4. **Intelligente Track-Komposition**
- Zugriff auf originale, separierte UND generierte Stems
- Prompt-basierte Track-Generierung
- Automatisches Arrangement und Rendering

## 🏗️ Systemarchitektur

```
neuromorphe-traum-engine/
├── stereo_tracks_for_analysis/    # Input: Ganze Tracks für Separation
├── models/                        # Trainierte VAE-Modelle
├── generated_stems/               # Output: Neue generierte Stems
├── generated_tracks/              # Output: Finale Tracks
├── processed_database/stems/      # Verarbeitete Stems
└── src/
    ├── services/
    │   ├── separation_service.py   # 🎵 Audio Source Separation
    │   ├── training_service.py     # 🧠 VAE Training
    │   ├── generative_service.py   # ✨ Stem Generation
    │   ├── preprocessor.py         # 📊 Audio Analysis
    │   ├── arranger.py            # 🎼 Track Arrangement
    │   └── renderer.py            # 🎚️ Audio Rendering
    ├── api/endpoints/
    │   ├── neuromorphic.py        # 🧠 Neuromorphic API
    │   └── stems.py               # 🎵 Stems & Track API
    └── database/
        ├── models.py              # 📊 Datenbank-Modelle
        └── service.py             # 🗄️ Datenbank-Service
```

## 🚀 Installation & Setup

### 1. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 2. Verzeichnisse erstellen
Die notwendigen Verzeichnisse werden automatisch erstellt:
- `stereo_tracks_for_analysis/`
- `models/`
- `generated_stems/`
- `generated_tracks/` (falls nicht vorhanden)

### 3. Datenbank initialisieren
Die SQLite-Datenbank wird beim ersten Start automatisch erstellt.

## 🎮 Verwendung

### Server starten
```bash
cd src
python main.py
```

Der Server läuft auf `http://localhost:8000`

### API-Dokumentation
Interaktive API-Docs: `http://localhost:8000/docs`

## 🔄 Workflow: Der lernende Kreislauf

### Phase 1: Daten-Akquise
```bash
# Stereo-Track in stereo_tracks_for_analysis/ platzieren
# Dann via API verarbeiten:
POST /api/v1/neuromorphic/preprocess
{
    "track_path": "path/to/track.wav",
    "separate_first": true
}
```

**Ergebnis:** Track wird in Stems zerlegt, analysiert und mit `source: "separated"` in die Datenbank eingefügt.

### Phase 2: Lernen
```bash
POST /api/v1/neuromorphic/train
{
    "category": "kick"
}
```

**Ergebnis:** VAE-Modell wird auf alle Kick-Drums trainiert und als `models/kick_vae.pt` gespeichert.

### Phase 3: Kreation
```bash
POST /api/v1/neuromorphic/generate
{
    "category": "kick",
    "num_variations": 10,
    "mode": "random"
}
```

**Ergebnis:** 10 neue Kick-Stems werden generiert, analysiert und mit `source: "generated"` gespeichert.

### Phase 4: Track-Komposition
```bash
POST /api/v1/stems/generate-track
{
    "prompt": "aggressive industrial techno 140 bpm",
    "include_generated": true,
    "include_separated": true,
    "include_original": true
}
```

**Ergebnis:** Kompletter Track aus allen verfügbaren Stem-Quellen komponiert.

## 🧪 System testen

```bash
python test_neuromorphic_engine.py
```

Dieser Test prüft:
- ✅ Datenbank-Initialisierung
- ✅ Audio-Separation (Demucs)
- ✅ VAE-Training
- ✅ Generative Stem-Erstellung
- ✅ API-Endpoints (optional)

## 📊 API-Endpoints

### Neuromorphic Engine
- `POST /api/v1/neuromorphic/preprocess` - Audio-Verarbeitung & Separation
- `POST /api/v1/neuromorphic/train` - VAE-Training
- `POST /api/v1/neuromorphic/generate` - Stem-Generierung

### Stems & Tracks
- `GET /api/v1/stems/` - Alle Stems abrufen
- `GET /api/v1/stems/{id}` - Einzelnen Stem abrufen
- `GET /api/v1/stems/category/{category}` - Stems nach Kategorie
- `POST /api/v1/stems/search/` - Semantische Suche
- `POST /api/v1/stems/generate-track` - Track-Generierung

### System
- `GET /health` - System-Status

## 🎛️ Konfiguration

### Generative Modi
- **`random`**: Zufällige Latent-Vektoren
- **`interpolated`**: Interpolation zwischen existierenden Stems
- **`hybrid`**: Mischung verschiedener Kategorien

### VAE-Parameter
- **Latent Dimension**: 128 (konfigurierbar)
- **Training Epochs**: 100 (konfigurierbar)
- **Batch Size**: 32 (konfigurierbar)

### Demucs-Modelle
- **Standard**: `htdemucs` (ausgewogen)
- **Qualität**: `htdemucs_ft` (höchste Qualität)
- **Geschwindigkeit**: `htdemucs_6s` (schneller)

## 🔧 Erweiterte Funktionen

### Batch-Verarbeitung
```python
# Mehrere Tracks gleichzeitig verarbeiten
separation_service = SeparationService()
results = await separation_service.separate_multiple_tracks_async(track_paths)
```

### Hybrid-Generierung
```python
# Stems zwischen Kategorien interpolieren
generative_service = GenerativeService()
hybrid_stems = await generative_service.generate_hybrid_stems_async(
    category_a="kick",
    category_b="snare", 
    interpolation_factor=0.7
)
```

### Qualitäts-Filtering
```python
# Nur hochwertige Stems für Training verwenden
db_service = DatabaseService()
high_quality_stems = await db_service.get_all_stems(
    category="kick",
    # Zusätzliche Filter über SQL möglich
)
```

## 📈 Monitoring & Statistiken

```bash
# Stem-Statistiken abrufen
GET /api/v1/stems/statistics

# Antwort:
{
    "total_stems": 1250,
    "by_source": [
        {"source": "original", "count": 500},
        {"source": "separated", "count": 400},
        {"source": "generated", "count": 350}
    ],
    "by_category": [...]
}
```

## 🚨 Troubleshooting

### Häufige Probleme

1. **Demucs-Installation**
   ```bash
   pip install demucs==4.0.1
   ```

2. **CUDA-Support** (optional, für GPU-Beschleunigung)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Speicherplatz**
   - Demucs-Modelle: ~2GB
   - VAE-Modelle: ~50MB pro Kategorie
   - Generierte Stems: Variable Größe

4. **Performance-Optimierung**
   - GPU verwenden für Training/Generierung
   - Batch-Größe anpassen
   - Temporäre Dateien regelmäßig löschen

### Logs & Debugging
```python
# Detaillierte Logs aktivieren
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Erfolgskriterien

✅ **End-to-End Workflow 1**: Stereo-Track → Separation → Datenbank  
✅ **End-to-End Workflow 2**: Kategorie → VAE-Training → Modell  
✅ **End-to-End Workflow 3**: Kategorie → Generierung → Neue Stems  
✅ **Finaler Test**: Prompt → Track mit allen Stem-Quellen  

## 🔮 Zukunft & Erweiterungen

- **Multi-Modal VAEs**: Text-zu-Audio Generierung
- **Style Transfer**: Stil-Übertragung zwischen Stems
- **Real-time Processing**: Live-Performance Integration
- **Cloud Deployment**: Skalierbare Infrastruktur
- **Advanced Neuromorphics**: Spiking Neural Networks

---

**Die Neuromorphic Dream Engine ist ein geschlossener, lernender Kreislauf für kreative Audio-Produktion. Jeder generierte Stem erweitert das kreative Potenzial des Systems exponentiell.**

🧠 **Dream. Learn. Create. Repeat.**