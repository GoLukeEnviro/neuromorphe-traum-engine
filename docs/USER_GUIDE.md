# Benutzeranleitung - Neuromorphe Traum-Engine v2.0

## Inhaltsverzeichnis

1. [Erste Schritte](#erste-schritte)
2. [Audio-Dateien vorbereiten](#audio-dateien-vorbereiten)
3. [Preprocessing durchführen](#preprocessing-durchführen)
4. [Semantische Suche verwenden](#semantische-suche-verwenden)
5. [Erweiterte Features](#erweiterte-features)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Tipps und Tricks](#tipps-und-tricks)

## Erste Schritte

### Was ist die Neuromorphe Traum-Engine?

Die Neuromorphe Traum-Engine v2.0 ist ein KI-gestütztes System zur semantischen Audio-Suche. Sie ermöglicht es Ihnen:

- **Audio-Dateien durch Textbeschreibungen zu finden**
- **Ähnliche Sounds basierend auf semantischen Eigenschaften zu entdecken**
- **Große Audio-Bibliotheken effizient zu durchsuchen**
- **Kreative Workflows zu beschleunigen**

### Grundlegendes Konzept

1. **Preprocessing**: Audio-Dateien werden analysiert und in mathematische Vektoren (Embeddings) umgewandelt
2. **Indexierung**: Diese Vektoren werden in einer durchsuchbaren Datenbank gespeichert
3. **Suche**: Textbeschreibungen werden in denselben Vektorraum projiziert und mit Audio-Vektoren verglichen
4. **Ergebnisse**: Die ähnlichsten Audio-Dateien werden zurückgegeben

## Audio-Dateien vorbereiten

### Unterstützte Formate

**Primär unterstützt**:
- **.wav** (empfohlen) - Unkomprimiert, beste Qualität
- **.flac** - Verlustfreie Kompression

**Sekundär unterstützt** (je nach librosa-Installation):
- **.mp3** - Komprimiert, kann Qualitätsverluste haben
- **.aiff** - Apple-Format
- **.ogg** - Open-Source-Format

### Audio-Qualität optimieren

**Empfohlene Spezifikationen**:
- **Sample Rate**: 44.1 kHz oder 48 kHz
- **Bit Depth**: 16-bit oder 24-bit
- **Länge**: 1-30 Sekunden (optimal für Stems und Loops)
- **Lautstärke**: Normalisiert, aber nicht überkomprimiert

### Dateiorganisation

```
raw_construction_kits/
├── kicks/
│   ├── dark_industrial_kick_01.wav
│   ├── punchy_techno_kick_02.wav
│   └── deep_house_kick_03.wav
├── bass/
│   ├── rolling_bass_line_01.wav
│   ├── sub_bass_drone_02.wav
│   └── acid_bass_sequence_03.wav
├── synths/
│   ├── atmospheric_pad_01.wav
│   ├── lead_synth_melody_02.wav
│   └── ambient_texture_03.wav
└── percussion/
    ├── tribal_percussion_loop.wav
    ├── minimal_hihat_pattern.wav
    └── industrial_clap_sequence.wav
```

**Naming-Konventionen** (empfohlen):
- Verwenden Sie beschreibende Namen
- Inkludieren Sie Genre, Instrument und Charakteristika
- Vermeiden Sie Sonderzeichen und Umlaute in Dateinamen
- Nutzen Sie Unterstriche statt Leerzeichen

## Preprocessing durchführen

### Methode 1: Minimal Preprocessor (Empfohlen für Einsteiger)

**Schritt 1: Audio-Dateien hinzufügen**
```bash
# Kopieren Sie Ihre .wav-Dateien in das Input-Verzeichnis
cp /pfad/zu/ihren/sounds/*.wav raw_construction_kits/
```

**Schritt 2: Preprocessing starten**
```bash
python ai_agents/minimal_preprocessor.py
```

**Erwartete Ausgabe**:
```
Lade CLAP-Modell...
CLAP-Modell erfolgreich geladen.
Suche Audio-Dateien in: raw_construction_kits
Gefunden: 150 Audio-Dateien
Verarbeite Audio-Batches: 100%|████████| 5/5 [02:30<00:00, 30.2s/it]
Speichere 150 Embeddings in: processed_database/embeddings.pkl
Verarbeitung abgeschlossen!
```

**Verarbeitungszeit** (Richtwerte):
- 100 Dateien: ~2-5 Minuten (mit GPU)
- 500 Dateien: ~10-15 Minuten (mit GPU)
- 1000+ Dateien: ~30+ Minuten (mit GPU)

### Methode 2: SQL-basierter Preprocessor (Für große Bibliotheken)

**Vorteile**:
- Resume-Funktionalität bei Unterbrechungen
- Erweiterte Metadaten-Extraktion
- Robuste Fehlerbehandlung
- Fortschritts-Tracking

**Verwendung**:
```bash
python ai_agents/prepare_dataset_sql.py
```

**Erweiterte Konfiguration**:
```python
# Anpassung in prepare_dataset_sql.py
analyzer = NeuroAnalyzer(
    input_dir="raw_construction_kits",
    batch_size=16,          # Reduzieren bei Speicherproblemen
    max_retries=3,          # Wiederholungsversuche bei Fehlern
    checkpoint_interval=25  # Checkpoint alle 25 Dateien
)
```

### Preprocessing-Monitoring

**Progress-Tracking**:
```bash
# Fortschritt überwachen (SQL-Version)
tail -f processed_database/checkpoints/progress.json
```

**Fehler-Logs überprüfen**:
```bash
# Fehlgeschlagene Dateien anzeigen
cat processed_database/checkpoints/failed_files.json
```

## Semantische Suche verwenden

### Grundlegende Suche

**Suchmaschine starten**:
```bash
python ai_agents/search_engine_cli.py
```

**Interaktive Sitzung**:
```
Neuromorphe Traum-Engine v2.0 - Semantische Audio-Suche
============================================================
Prompt eingeben (oder 'exit' zum Beenden): 
```

### Effektive Suchbegriffe

#### Instrumenten-basierte Suche
```
# Spezifische Instrumente
kick drum
bass guitar
synth pad
hihat
snare
clap

# Kombinationen
kick and bass
synth lead melody
percussion loop
```

#### Genre-basierte Suche
```
# Genres
techno kick
house bass
ambient pad
industrial percussion
acid synth
dubstep wobble

# Sub-Genres
minimal techno
deep house
dark ambient
breakbeat
```

#### Charakteristika-basierte Suche
```
# Klangcharakter
dark and gritty
bright and punchy
warm and analog
cold and digital
organic texture

# Dynamik
loud and aggressive
soft and subtle
driving rhythm
hypnotic loop
```

#### Emotionale Beschreibungen
```
# Stimmungen
melancholic melody
energetic beat
atmospheric soundscape
tense and dramatic
uplifting harmony

# Energie-Level
high energy
low energy
building tension
relaxing ambient
```

### Erweiterte Suchtechniken

#### Kombinierte Suchbegriffe
```
# Mehrere Eigenschaften
dark industrial kick with reverb
melodic techno synth in minor key
organic percussion with tribal feel

# Negative Beschreibungen
kick without reverb
synth not too bright
bass without distortion
```

#### Kontext-spezifische Suche
```
# Verwendungszweck
intro kick drum
breakdown synth
fill percussion
transition effect

# Arrangement-Position
low-end bass
mid-range synth
high-frequency hihat
```

### Suchergebnisse interpretieren

**Beispiel-Ausgabe**:
```
Suche nach: 'dark industrial kick'...

Top 5 Ergebnisse:
--------------------------------------------------
1. raw_construction_kits/kicks/industrial_kick_heavy.wav
   Ähnlichkeit: 0.8934

2. raw_construction_kits/kicks/dark_techno_kick_02.wav
   Ähnlichkeit: 0.8721

3. raw_construction_kits/percussion/metal_hit_processed.wav
   Ähnlichkeit: 0.7456

4. raw_construction_kits/kicks/distorted_kick_analog.wav
   Ähnlichkeit: 0.7234

5. raw_construction_kits/bass/sub_kick_hybrid.wav
   Ähnlichkeit: 0.6891
```

**Ähnlichkeitswerte verstehen**:
- **0.9-1.0**: Sehr hohe Übereinstimmung
- **0.8-0.9**: Hohe Übereinstimmung
- **0.7-0.8**: Gute Übereinstimmung
- **0.6-0.7**: Moderate Übereinstimmung
- **<0.6**: Geringe Übereinstimmung

## Erweiterte Features

### Batch-Suche (Programmierung erforderlich)

```python
# batch_search.py
from ai_agents.search_engine_cli import SearchEngine

search_engine = SearchEngine("processed_database/embeddings.pkl")

# Mehrere Suchbegriffe
queries = [
    "dark kick",
    "melodic synth",
    "tribal percussion",
    "ambient pad"
]

for query in queries:
    results = search_engine.search(query, top_k=3)
    print(f"\nSuche: {query}")
    for i, (path, similarity) in enumerate(results, 1):
        print(f"{i}. {path} ({similarity:.3f})")
```

### Ähnlichkeits-Matrix erstellen

```python
# similarity_matrix.py
import numpy as np
import torch.nn.functional as F
from ai_agents.search_engine_cli import SearchEngine

search_engine = SearchEngine("processed_database/embeddings.pkl")

# Berechne Ähnlichkeit zwischen allen Audio-Dateien
similarity_matrix = F.cosine_similarity(
    search_engine.embedding_tensors.unsqueeze(1),
    search_engine.embedding_tensors.unsqueeze(0),
    dim=2
)

# Finde ähnlichste Dateien für jede Datei
for i, file_path in enumerate(search_engine.file_paths[:5]):  # Erste 5 Dateien
    similarities = similarity_matrix[i]
    top_indices = similarities.topk(6)[1][1:]  # Top 5 (ohne sich selbst)
    
    print(f"\nÄhnlich zu {file_path}:")
    for j, idx in enumerate(top_indices, 1):
        similar_file = search_engine.file_paths[idx]
        similarity = similarities[idx].item()
        print(f"{j}. {similar_file} ({similarity:.3f})")
```

### Export-Funktionen

```python
# export_results.py
import json
import csv
from ai_agents.search_engine_cli import SearchEngine

search_engine = SearchEngine("processed_database/embeddings.pkl")

def export_search_results(query, filename, format='json'):
    results = search_engine.search(query, top_k=10)
    
    if format == 'json':
        data = {
            'query': query,
            'results': [{'path': path, 'similarity': float(sim)} for path, sim in results]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format == 'csv':
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query', 'File Path', 'Similarity'])
            for path, similarity in results:
                writer.writerow([query, path, similarity])

# Verwendung
export_search_results("dark techno kick", "search_results.json")
export_search_results("melodic synth", "synth_results.csv", format='csv')
```

## Best Practices

### Audio-Bibliothek organisieren

1. **Konsistente Benennung**:
   ```
   # Gut
   genre_instrument_characteristic_number.wav
   techno_kick_dark_01.wav
   house_bass_rolling_02.wav
   
   # Schlecht
   kick1.wav
   MyAwesomeSound.wav
   untitled_audio_file.wav
   ```

2. **Kategorische Struktur**:
   ```
   raw_construction_kits/
   ├── drums/
   │   ├── kicks/
   │   ├── snares/
   │   └── percussion/
   ├── bass/
   ├── synths/
   │   ├── leads/
   │   ├── pads/
   │   └── arps/
   └── fx/
   ```

3. **Metadaten in Dateinamen**:
   ```
   # Inkludieren Sie wichtige Informationen
   bpm120_key_Am_dark_techno_kick.wav
   bpm128_key_Gm_rolling_house_bass.wav
   ```

### Suchstrategien optimieren

1. **Spezifisch beginnen, dann erweitern**:
   ```
   # Start: "kick"
   # Verfeinern: "techno kick"
   # Spezifizieren: "dark techno kick with reverb"
   ```

2. **Verschiedene Beschreibungsebenen nutzen**:
   ```
   # Technisch: "808 kick drum"
   # Emotional: "aggressive kick"
   # Kontextuell: "breakdown kick"
   ```

3. **Synonyme und Variationen testen**:
   ```
   # Für Bass:
   "bass", "bassline", "low-end", "sub bass"
   
   # Für Atmosphäre:
   "ambient", "atmospheric", "textural", "soundscape"
   ```

### Performance optimieren

1. **Batch-Größen anpassen**:
   ```python
   # Für wenig GPU-Speicher
   batch_size = 8
   
   # Für viel GPU-Speicher
   batch_size = 64
   ```

2. **Regelmäßige Datenbank-Wartung**:
   ```bash
   # Alte Embeddings löschen bei Änderungen
   rm processed_database/embeddings.pkl
   
   # Neu verarbeiten
   python ai_agents/minimal_preprocessor.py
   ```

3. **Speicher-Management**:
   ```python
   # In eigenen Skripten
   import torch
   torch.cuda.empty_cache()  # GPU-Speicher freigeben
   ```

## Troubleshooting

### Häufige Probleme und Lösungen

#### Problem: Schlechte Suchergebnisse

**Mögliche Ursachen**:
- Unpassende Suchbegriffe
- Zu wenig Audio-Material
- Schlechte Audio-Qualität

**Lösungen**:
1. **Suchbegriffe variieren**:
   ```
   # Statt: "beat"
   # Versuchen: "drum pattern", "rhythm", "percussion loop"
   ```

2. **Audio-Bibliothek erweitern**:
   - Mindestens 50-100 Dateien für gute Ergebnisse
   - Diverse Genres und Stile inkludieren

3. **Audio-Qualität prüfen**:
   ```python
   # Audio-Datei analysieren
   import librosa
   y, sr = librosa.load('problematic_file.wav')
   print(f"Sample Rate: {sr}, Länge: {len(y)/sr:.2f}s")
   print(f"RMS: {librosa.feature.rms(y=y).mean():.4f}")
   ```

#### Problem: Langsame Verarbeitung

**Lösungen**:
1. **GPU-Nutzung überprüfen**:
   ```python
   import torch
   print(f"CUDA verfügbar: {torch.cuda.is_available()}")
   print(f"Aktuelle GPU: {torch.cuda.get_device_name()}")
   ```

2. **Batch-Größe reduzieren**:
   ```python
   # In minimal_preprocessor.py
   batch_size = 16  # Statt 32
   ```

3. **Parallele Verarbeitung nutzen**:
   ```python
   # Für CPU-intensive Aufgaben
   from concurrent.futures import ProcessPoolExecutor
   ```

#### Problem: Speicher-Probleme

**Lösungen**:
1. **Speicher-Monitoring**:
   ```python
   import psutil
   print(f"RAM-Nutzung: {psutil.virtual_memory().percent}%")
   ```

2. **Batch-Verarbeitung optimieren**:
   ```python
   # Kleinere Batches
   batch_size = 8
   
   # Speicher nach jedem Batch freigeben
   del embeddings
   torch.cuda.empty_cache()
   ```

## Tipps und Tricks

### Kreative Suchstrategien

1. **Emotionale Beschreibungen**:
   ```
   "melancholic piano"
   "aggressive distortion"
   "dreamy atmosphere"
   "nostalgic melody"
   ```

2. **Produktions-Kontext**:
   ```
   "intro element"
   "breakdown sound"
   "transition effect"
   "outro fade"
   ```

3. **Referenz-basierte Suche**:
   ```
   "like a 909 kick"
   "moog-style bass"
   "vintage analog"
   "modern digital"
   ```

### Workflow-Integration

1. **DAW-Integration** (manuell):
   ```bash
   # Suchergebnisse in DAW-Ordner kopieren
   cp "gefundene_datei.wav" "/pfad/zu/daw/project/samples/"
   ```

2. **Playlist-Erstellung**:
   ```python
   # playlist_creator.py
   def create_playlist(search_terms, output_dir):
       for term in search_terms:
           results = search_engine.search(term, top_k=5)
           for i, (path, _) in enumerate(results):
               new_name = f"{term}_{i+1}.wav"
               shutil.copy(path, os.path.join(output_dir, new_name))
   ```

3. **Batch-Export**:
   ```python
   # Alle Ergebnisse einer Suche exportieren
   def export_search_batch(query, output_dir, max_results=10):
       results = search_engine.search(query, top_k=max_results)
       os.makedirs(output_dir, exist_ok=True)
       
       for i, (path, similarity) in enumerate(results):
           filename = f"{query}_{i+1}_{similarity:.3f}.wav"
           shutil.copy(path, os.path.join(output_dir, filename))
   ```

### Erweiterte Anwendungen

1. **Sample-Pack-Analyse**:
   ```python
   # Analysiere Ähnlichkeiten in einem Sample-Pack
   def analyze_sample_pack(pack_directory):
       pack_files = [f for f in search_engine.file_paths 
                    if pack_directory in f]
       
       for file in pack_files:
           # Finde ähnliche Dateien außerhalb des Packs
           results = search_engine.search_by_file(file, top_k=5)
           print(f"Ähnlich zu {file}:")
           for path, sim in results:
               if pack_directory not in path:
                   print(f"  {path} ({sim:.3f})")
   ```

2. **Genre-Klassifikation**:
   ```python
   # Automatische Genre-Erkennung basierend auf Ähnlichkeit
   genre_examples = {
       'techno': ['techno_kick.wav', 'techno_bass.wav'],
       'house': ['house_kick.wav', 'house_bass.wav'],
       'ambient': ['ambient_pad.wav', 'ambient_texture.wav']
   }
   
   def classify_genre(audio_file):
       best_genre = None
       best_score = 0
       
       for genre, examples in genre_examples.items():
           scores = []
           for example in examples:
               similarity = calculate_similarity(audio_file, example)
               scores.append(similarity)
           
           avg_score = sum(scores) / len(scores)
           if avg_score > best_score:
               best_score = avg_score
               best_genre = genre
       
       return best_genre, best_score
   ```

---

*Diese Benutzeranleitung wird kontinuierlich erweitert. Experimentieren Sie mit verschiedenen Suchstrategien und entdecken Sie die Möglichkeiten der semantischen Audio-Suche!*