# Detaillierte Analyse der KI-Module in der Neuromorphen Traum-Engine v2.0

Die Neuromorphe Traum-Engine v2.0 integriert mehrere fortschrittliche KI-Module, um ihre Kernfunktionen der semantischen Audio-Suche, generativen Stem-Erstellung und Audio-Trennung zu ermöglichen. Die Hauptmodule sind CLAP, Variational Autoencoders (VAEs) und Demucs.

## 1. CLAP (Contrastive Language-Audio Pre-training)

**Rolle im System**: CLAP ist das Herzstück der semantischen Suche und Audio-Analyse. Es ermöglicht dem System, die Bedeutung von Audio-Dateien und Textbeschreibungen in einem gemeinsamen latenten Raum zu verstehen.

**Implementierung (siehe `src/services/neuro_analyzer.py` und `src/services/search.py`)**:

- **`CLAPEmbedder` Klasse**:
    - Lädt das vortrainierte CLAP-Modell (`laion/larger_clap_music_and_speech`) von Hugging Face Transformers.
    - Nutzt `ClapProcessor` und `ClapModel` für die Verarbeitung.
    - Unterstützt sowohl Audio- als auch Text-Embedding-Generierung.
    - Audio-Input wird auf 48 kHz resampelt und dann vom `audio_encoder` des CLAP-Modells verarbeitet.
    - Text-Input wird vom `text_encoder` verarbeitet.
    - Beide Encoder erzeugen 512-dimensionale Embeddings, die L2-normalisiert werden, um die Kosinus-Ähnlichkeit zu ermöglichen.
    - Verwendet CUDA (falls verfügbar) für GPU-Beschleunigung.
- **`SemanticAnalyzer` Klasse**:
    - Nutzt den `CLAPEmbedder`, um semantische Analysen von Audio-Inhalten durchzuführen.
    - Vergleicht Audio-Embeddings mit vordefinierten Text-Embeddings für Kategorien wie 'energy', 'mood', 'texture', 'movement' und 'space'.
    - Erstellt ein zusammenfassendes semantisches Profil für jeden Stem.
- **`SearchService` Klasse**:
    - Nutzt den `CLAPEmbedder`, um Text-Suchanfragen in Embeddings umzuwandeln.
    - Führt eine Kosinus-Ähnlichkeitssuche zwischen dem Text-Embedding der Anfrage und den Audio-Embeddings der Stems in der Datenbank durch.
    - Gibt die Top-K-Ergebnisse basierend auf der Ähnlichkeit zurück.

**Bedeutung**: CLAP ist entscheidend für die Fähigkeit des Systems, natürliche Sprachabfragen zu verstehen und relevante Audio-Inhalte zu finden, was über einfache Stichwortsuche hinausgeht.

## 2. VAE (Variational Autoencoder)

**Rolle im System**: VAEs sind für die generative Stem-Mutation und Hybridisierung verantwortlich. Sie lernen die Verteilung von Audio-Features (Spektrogramme) einer bestimmten Kategorie und können daraus neue, ähnliche Stems erzeugen.

**Implementierung (siehe `src/services/training_service.py` und `src/services/generative_service.py`)**:

- **`AudioVAE` Klasse (in `src/services/training_service.py`)**:
    - Eine PyTorch `nn.Module`-Implementierung eines VAE.
    - **Encoder**: Mehrere lineare Schichten mit ReLU-Aktivierung und Dropout, die den Input (flattened Log-Spektrogramm) in die Parameter des latenten Raums (Mean `mu` und Log-Varianz `logvar`) abbilden.
    - **Reparameterization Trick**: Ermöglicht das Sampling aus der latenten Verteilung während des Trainings, um den Gradientenfluss zu gewährleisten.
    - **Decoder**: Mehrere lineare Schichten mit ReLU und einer finalen Sigmoid-Aktivierung, die einen latenten Vektor zurück in ein Spektrogramm dekodieren.
    - **Loss Function**: Kombiniert den Rekonstruktions-Loss (MSE zwischen Original- und rekonstruiertem Spektrogramm) und den KL-Divergenz-Loss (misst den Unterschied zwischen der latenten Verteilung und einer Standardnormalverteilung).
- **`AudioDataset` Klasse (in `src/services/training_service.py`)**:
    - Lädt Audio-Dateien, konvertiert sie in Mono und berechnet Log-Spektrogramme.
    - Flattened die Spektrogramme und normalisiert sie mit `StandardScaler` für das Training.
- **`TrainingService` Klasse**:
    - Verwaltet den Trainingsprozess für VAE-Modelle.
    - Ruft Stems einer bestimmten Kategorie aus der Datenbank ab.
    - Führt das Training asynchron in einem `ThreadPoolExecutor` aus, um die Haupt-Event-Loop nicht zu blockieren.
    - Speichert die trainierten Modelle (`.pt`), Scaler (`.pkl`) und Metadaten (`.json`) im `models/` Verzeichnis.
- **`GenerativeService` Klasse**:
    - Lädt trainierte VAE-Modelle und Scaler aus dem Cache oder von der Festplatte.
    - **`_spectrogram_to_audio`**: Konvertiert generierte Spektrogramme zurück in Audio-Waveforms (verwendet `istft` und eine zufällige Phase für die Rekonstruktion).
    - **`_generate_stems_sync`**: Führt die eigentliche Stem-Generierung durch:
        - **Random Mode**: Sampelt zufällige Vektoren aus dem latenten Raum.
        - **Interpolate Mode**: Interpoliert zwischen zwei zufälligen latenten Vektoren.
        - **Hybrid Mode**: Eine Mischung aus strukturiertem und zufälligem Sampling.
    - Speichert die generierten Stems im `generated_stems/` Verzeichnis.
    - Bietet Funktionen für die Hybrid-Generierung zwischen zwei Kategorien und die Batch-Generierung für alle Kategorien.

**Bedeutung**: VAEs ermöglichen die Erzeugung neuer, einzigartiger Audio-Stems, die die gelernten musikalischen Eigenschaften der Trainingsdaten widerspiegeln, was die kreativen Möglichkeiten des Systems erweitert.

## 3. Demucs

**Rolle im System**: Demucs ist ein leistungsstarkes Modell für die Audio Source Separation. Es zerlegt einen vollständigen Stereo-Track in seine einzelnen Komponenten (z.B. Drums, Bass, Vocals, Other), was für die weitere Analyse und Komposition unerlässlich ist.

**Implementierung (siehe `src/services/separation_service.py`)**:

- **`SeparationService` Klasse**:
    - Nutzt das `demucs` Python-Paket.
    - Lädt das `htdemucs` Modell (oder andere konfigurierbare Modelle) für die Trennung.
    - Die `separate_track` Methode nimmt einen Stereo-Track-Pfad entgegen.
    - Trennt den Track in separate Stem-Dateien (z.B. `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`).
    - Speichert die separierten Stems in einem temporären Verzeichnis oder einem konfigurierten Output-Pfad.
    - Die separierten Stems werden dann an den `PreprocessorService` weitergeleitet, um analysiert und in die Datenbank aufgenommen zu werden.
    - Unterstützt GPU-Beschleunigung, falls PyTorch mit CUDA verfügbar ist.

**Bedeutung**: Demucs ist der erste Schritt im Workflow, um komplexe musikalische Arrangements in ihre Grundbausteine zu zerlegen, die dann einzeln analysiert, gesucht und neu kombiniert werden können.

## Zusammenfassung der KI-Integration

Die drei KI-Module arbeiten synergetisch zusammen:
1. **Demucs** zerlegt rohe Audio-Tracks in ihre Einzelteile (Stems).
2. Der **PreprocessorService** nutzt **CLAP** und den **NeuroAnalyzer**, um diese Stems zu analysieren, semantische Embeddings zu extrahieren und Metadaten zu generieren, die in der Datenbank gespeichert werden.
3. Der **TrainingService** trainiert **VAEs** auf Sammlungen dieser analysierten Stems.
4. Der **GenerativeService** verwendet die trainierten **VAEs**, um neue, einzigartige Stems zu erzeugen.
5. Der **SearchService** nutzt **CLAP**-Embeddings, um semantische Suchen durchzuführen und relevante Stems (original, separiert oder generiert) basierend auf Text-Prompts zu finden.

Diese modulare Architektur ermöglicht eine flexible und leistungsstarke Verarbeitung und Generierung von Audio-Inhalten, die das "Kollektive Unbewusste" der elektronischen Musik erschließt.