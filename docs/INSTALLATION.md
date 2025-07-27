# Installationsanleitung - Neuromorphe Traum-Engine v2.0

## Systemanforderungen

### Mindestanforderungen
- **Betriebssystem**: Windows 10/11, macOS 10.15+, oder Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 oder höher
- **RAM**: Mindestens 8 GB (16 GB empfohlen)
- **Speicherplatz**: 10 GB freier Speicherplatz
- **Internetverbindung**: Für das Herunterladen der CLAP-Modelle

### Empfohlene Konfiguration
- **GPU**: NVIDIA GPU mit CUDA-Unterstützung (für bessere Performance)
- **RAM**: 16 GB oder mehr
- **CPU**: Multi-Core-Prozessor (Intel i5/AMD Ryzen 5 oder besser)
- **SSD**: Für schnellere I/O-Operationen

## Schritt 1: Python-Installation

### Windows

1. **Python herunterladen**:
   - Besuchen Sie [python.org](https://www.python.org/downloads/)
   - Laden Sie Python 3.8+ herunter
   - **Wichtig**: Aktivieren Sie "Add Python to PATH" während der Installation

2. **Installation überprüfen**:
```cmd
python --version
pip --version
```

### macOS

1. **Mit Homebrew** (empfohlen):
```bash
# Homebrew installieren (falls nicht vorhanden)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python installieren
brew install python@3.9
```

2. **Direkt von python.org**:
   - Laden Sie den macOS-Installer herunter und führen Sie ihn aus

### Linux (Ubuntu/Debian)

```bash
# System aktualisieren
sudo apt update
sudo apt upgrade

# Python und pip installieren
sudo apt install python3 python3-pip python3-venv

# Entwicklungstools installieren
sudo apt install build-essential python3-dev
```

## Schritt 2: Projekt-Setup

### Repository herunterladen

**Option A: Git Clone** (falls verfügbar):
```bash
git clone [repository-url]
cd "Neuromorphe Traum-Engine v2.0"
```

**Option B: Manueller Download**:
1. Laden Sie das Projekt-Archiv herunter
2. Extrahieren Sie es in ein Verzeichnis Ihrer Wahl
3. Navigieren Sie zum Projektverzeichnis

### Virtuelle Umgebung erstellen (empfohlen)

```bash
# Virtuelle Umgebung erstellen
python -m venv neuromorphe_env

# Aktivieren
# Windows:
neuromorphe_env\Scripts\activate

# macOS/Linux:
source neuromorphe_env/bin/activate
```

## Schritt 3: Abhängigkeiten installieren

### Basis-Installation

```bash
# Abhängigkeiten aus requirements.txt installieren
pip install -r requirements.txt
```

### Manuelle Installation (falls requirements.txt fehlt)

```bash
# Kern-Bibliotheken
pip install librosa>=0.10.0
pip install numpy>=1.21.0
pip install soundfile>=0.12.0
pip install laion-clap
pip install torch>=1.9.0
pip install scikit-learn>=1.0.0
pip install tqdm
```

### GPU-Unterstützung (optional, aber empfohlen)

**NVIDIA GPU mit CUDA**:
```bash
# PyTorch mit CUDA-Unterstützung
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Überprüfung der GPU-Unterstützung**:
```python
import torch
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
print(f"CUDA-Geräte: {torch.cuda.device_count()}")
```

## Schritt 4: Verzeichnisstruktur erstellen

```bash
# Erforderliche Verzeichnisse erstellen
mkdir raw_construction_kits
mkdir processed_database
mkdir processed_database/checkpoints
```

**Vollständige Verzeichnisstruktur**:
```
Neuromorphe Traum-Engine v2.0/
├── raw_construction_kits/      # Ihre Audio-Dateien hier hinzufügen
├── processed_database/         # Verarbeitete Daten
│   ├── checkpoints/           # Automatisch erstellt
│   ├── embeddings.pkl         # Wird generiert
│   └── stems.db              # Wird generiert (SQL-Version)
├── ai_agents/                 # Hauptkomponenten
├── Construction_Plans/        # Projektdokumentation
└── requirements.txt
```

## Schritt 5: Installation testen

### Test 1: Python-Importe

```python
# test_imports.py
try:
    import librosa
    import torch
    import numpy as np
    import laion_clap
    import soundfile as sf
    import sklearn
    print("✅ Alle Abhängigkeiten erfolgreich importiert!")
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
```

### Test 2: CLAP-Modell laden

```python
# test_clap.py
import laion_clap

try:
    print("Lade CLAP-Modell...")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    print("✅ CLAP-Modell erfolgreich geladen!")
except Exception as e:
    print(f"❌ CLAP-Fehler: {e}")
```

### Test 3: Audio-Verarbeitung

```python
# test_audio.py
import librosa
import numpy as np

# Test-Audio erstellen
test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))  # 1s, 440Hz

try:
    # Audio-Features extrahieren
    mfccs = librosa.feature.mfcc(y=test_audio, sr=22050)
    print(f"✅ Audio-Verarbeitung funktioniert! MFCC-Shape: {mfccs.shape}")
except Exception as e:
    print(f"❌ Audio-Fehler: {e}")
```

## Schritt 6: Erste Verwendung

### Test-Audio-Dateien hinzufügen

1. **Eigene Audio-Dateien**:
   - Kopieren Sie .wav-Dateien in `raw_construction_kits/`
   - Unterstützte Formate: .wav (empfohlen), .mp3, .flac

2. **Test-Audio erstellen** (falls keine Dateien vorhanden):
```python
# create_test_audio.py
import numpy as np
import soundfile as sf
import os

# Verzeichnis erstellen
os.makedirs("raw_construction_kits", exist_ok=True)

# Test-Sounds generieren
sr = 44100
duration = 2.0
t = np.linspace(0, duration, int(sr * duration))

# Kick-Drum-ähnlicher Sound
kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 10)
sf.write("raw_construction_kits/test_kick.wav", kick, sr)

# Hi-Hat-ähnlicher Sound
hihat = np.random.normal(0, 0.1, len(t)) * np.exp(-t * 20)
sf.write("raw_construction_kits/test_hihat.wav", hihat, sr)

print("✅ Test-Audio-Dateien erstellt!")
```

### Ersten Preprocessing-Lauf starten

```bash
# Minimal Preprocessor ausführen
python ai_agents/minimal_preprocessor.py
```

**Erwartete Ausgabe**:
```
Lade CLAP-Modell...
CLAP-Modell erfolgreich geladen.
Suche Audio-Dateien in: raw_construction_kits
Gefunden: 2 Audio-Dateien
Verarbeite Audio-Batches: 100%|████████| 1/1 [00:05<00:00,  5.23s/it]
Speichere 2 Embeddings in: processed_database/embeddings.pkl
Verarbeitung abgeschlossen!
```

### Erste Suche durchführen

```bash
# Search Engine starten
python ai_agents/search_engine_cli.py
```

**Interaktive Sitzung**:
```
Neuromorphe Traum-Engine v2.0 - Semantische Audio-Suche
============================================================
Prompt eingeben (oder 'exit' zum Beenden): kick drum

Suche nach: 'kick drum'...

Top 5 Ergebnisse:
--------------------------------------------------
1. raw_construction_kits/test_kick.wav
   Ähnlichkeit: 0.8234

2. raw_construction_kits/test_hihat.wav
   Ähnlichkeit: 0.3421
```

## Fehlerbehebung

### Häufige Installationsprobleme

#### Problem: "ModuleNotFoundError: No module named 'laion_clap'"

**Lösung**:
```bash
# Spezifische CLAP-Installation
pip install git+https://github.com/LAION-AI/CLAP.git

# Oder alternative Installation
pip install laion-clap-pytorch
```

#### Problem: "CUDA out of memory"

**Lösung**:
1. **Batch-Größe reduzieren**:
   ```python
   # In minimal_preprocessor.py, Zeile ~89
   batch_size = 8  # Statt 32
   ```

2. **CPU-Modus erzwingen**:
   ```python
   # Vor CLAP-Modell-Initialisierung
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = ''
   ```

#### Problem: "Permission denied" (Linux/macOS)

**Lösung**:
```bash
# Berechtigungen setzen
chmod +x ai_agents/*.py

# Oder mit sudo ausführen (nicht empfohlen)
sudo python ai_agents/minimal_preprocessor.py
```

#### Problem: "Audio-Dateien nicht gefunden"

**Lösung**:
1. **Pfade überprüfen**:
   ```python
   import os
   print(os.path.abspath("raw_construction_kits"))
   print(os.listdir("raw_construction_kits"))
   ```

2. **Unterstützte Formate**:
   - Nur .wav-Dateien werden standardmäßig unterstützt
   - Andere Formate zu .wav konvertieren

### Performance-Optimierung

#### GPU-Speicher optimieren

```python
# In den Preprocessing-Skripten hinzufügen
import torch

# Nach jeder Batch-Verarbeitung
torch.cuda.empty_cache()

# Speicher-effiziente Einstellungen
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

#### Batch-Größen anpassen

| GPU-Speicher | Empfohlene Batch-Größe |
|--------------|------------------------|
| 4 GB         | 8-16                   |
| 8 GB         | 16-32                  |
| 12 GB+       | 32-64                  |

## Erweiterte Installation

### Entwicklungsumgebung

```bash
# Zusätzliche Entwicklungstools
pip install jupyter notebook ipython
pip install matplotlib seaborn  # Für Visualisierungen
pip install pytest              # Für Tests
pip install black flake8        # Code-Formatierung
```

### Docker-Installation (optional)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "ai_agents/search_engine_cli.py"]
```

```bash
# Docker-Container erstellen und ausführen
docker build -t neuromorphe-engine .
docker run -it -v $(pwd)/raw_construction_kits:/app/raw_construction_kits neuromorphe-engine
```

### Cloud-Installation (Google Colab)

```python
# In Google Colab Notebook
!pip install librosa laion-clap torch soundfile scikit-learn tqdm

# Projekt-Dateien hochladen
from google.colab import files
uploaded = files.upload()

# Audio-Dateien verarbeiten
!python ai_agents/minimal_preprocessor.py
```

## Nächste Schritte

Nach erfolgreicher Installation:

1. **Lesen Sie die [README.md](README.md)** für eine Übersicht
2. **Konsultieren Sie die [API_DOCUMENTATION.md](API_DOCUMENTATION.md)** für Details
3. **Experimentieren Sie** mit verschiedenen Audio-Dateien und Suchbegriffen
4. **Erweitern Sie das System** nach Ihren Bedürfnissen

## Support

Bei Installationsproblemen:

1. **Überprüfen Sie die Systemanforderungen**
2. **Führen Sie die Testsuite aus**
3. **Konsultieren Sie die Fehlerbehebung**
4. **Erstellen Sie ein Issue** mit detaillierter Problembeschreibung

---

*Neuromorphe Traum-Engine v2.0 - Erfolgreiche Installation ist der erste Schritt zur semantischen Audio-Revolution!*