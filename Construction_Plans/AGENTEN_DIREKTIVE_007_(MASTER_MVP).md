HYPER-PROMPT v3.0 (MASTER-PROMPT für MVP)
AGENTEN-DIREKTIVE: 007 (MASTER-MVP)
PROJEKT: Neuromorphe Traum-Engine v2.0 (Pragmatischer MVP)
MISSION: Erschaffe ein voll funktionsfähiges, kommandozeilenbasiertes "Text-zu-Stem"-Retrieval-System von Grund auf, um die Kernhypothese der semantischen Audio-Suche schnellstmöglich zu validieren.
BEGINN DER DIREKTIVE
Du bist ein autonomer Full-Stack AI-Engineer. Deine Mission ist die Implementierung eines vollständigen MVP-Systems in Python. Das System soll es einem Nutzer ermöglichen, über die Kommandozeile eine textuelle Beschreibung eines Sounds einzugeben und als Ergebnis die fünf passendsten Audio-Dateien aus einer vorverarbeiteten Bibliothek zu erhalten.
Das Projekt besteht aus ZWEI KERN-KOMPONENTEN, die du von Grund auf neu erstellen wirst:
Ein Pre-Processing-Skript (minimal_preprocessor.py): Analysiert eine Audio-Bibliothek und erstellt eine kompakte Datenbank mit CLAP-Embeddings.
Ein Kommandozeilen-Such-Tool (search_engine_cli.py): Nutzt die erstellte Embedding-Datenbank, um auf Text-Prompts zu reagieren.
PHASE 1: PROJEKT-SETUP & UMFELD
Verzeichnisstruktur erstellen:
Stelle sicher, dass die folgende Ordnerstruktur im Hauptverzeichnis des Projekts existiert. Erstelle sie, falls sie fehlt:
Generated code
/
|-- raw_construction_kits/      # Input für rohe Audiodaten (bleibt vorerst leer)
|-- processed_database/         # Output für die Embedding-Datenbank
|-- ai_agents/                  # Dein Arbeitsverzeichnis
|   |-- minimal_preprocessor.py
|   `-- search_engine_cli.py
`-- requirements.txt            # Datei für Abhängigkeiten
Use code with caution.
Abhängigkeiten definieren (requirements.txt):
Erstelle eine requirements.txt-Datei mit exakt folgendem Inhalt, um die Reproduzierbarkeit zu gewährleisten:
Generated code
librosa
soundfile
laion-clap-pytorch
torch
numpy
tqdm
Use code with caution.
Ein Agent muss diese Abhängigkeiten vor der Ausführung der Skripte installieren (z.B. mittels pip install -r requirements.txt).
PHASE 2: IMPLEMENTIERUNG DES PRE-PROCESSORS (minimal_preprocessor.py)
ZIEL: Erstelle ein effizientes Skript, das einen Ordner mit Audiodateien durchsucht, für jede Datei ein CLAP-Embedding berechnet und die Ergebnisse in einer einzigen, kompakten Binärdatei speichert.
Implementierungsdetails:
Importe: os, pickle, librosa, torch, numpy, tqdm und laion_clap.
Klasse MinimalPreprocessor:
__init__(self, input_dir, output_path): Initialisiert die Pfade und lädt das CLAP-Modell (enable_fusion=False) einmalig.
run(self):
Sucht rekursiv alle .wav-Dateien im input_dir.
Verwendet tqdm für eine visuelle Fortschrittsanzeige in der Konsole.
Implementiere eine Batch-Verarbeitung, um die Effizienz zu maximieren. Verarbeite die Audiodateien in Batches der Größe 32.
Für jeden Batch: Lade die Audio-Dateien, berechne die Audio-Embeddings mit self.clap_model.get_audio_embedding_from_filelist(...).
Sammle alle Ergebnisse (Dateipfade und die zugehörigen Embedding-Vektoren als NumPy-Arrays) in einer einzigen Liste von Dictionaries. Jedes Dictionary soll die Form {'path': 'dateipfad', 'embedding': numpy_array} haben.
Speichere die finale Liste mit allen verarbeiteten Dateien als Binärdatei unter Verwendung des pickle-Moduls in processed_database/embeddings.pkl.
Haupt-Ausführungsblock (if __name__ == "__main__":):
Erstellt eine Instanz von MinimalPreprocessor (input: raw_construction_kits, output: processed_database/embeddings.pkl) und ruft die run-Methode auf.
PHASE 3: IMPLEMENTIERUNG DER SUCHMASCHINE (search_engine_cli.py)
ZIEL: Erstelle ein interaktives Kommandozeilen-Tool, das einen Text-Prompt entgegennimmt, die 5 besten Audio-Matches aus der Embedding-Datenbank findet und deren Pfade ausgibt.
Implementierungsdetails:
Importe: pickle, torch, numpy und laion_clap.
Klasse SearchEngine:
__init__(self, embeddings_path):
Lädt das CLAP-Modell einmalig.
Lädt die embeddings.pkl-Datei.
Extrahiert alle Dateipfade in eine Liste (self.file_paths) und alle Embeddings in einen einzigen, großen PyTorch-Tensor (self.embedding_tensors) für maximale Recheneffizienz. Stelle sicher, dass die Indizes übereinstimmen.
search(self, prompt: str, top_k: int = 5):
Nimmt einen Text-Prompt entgegen.
Berechnet das Text-Embedding für den Prompt mit self.clap_model.get_text_embedding(...).
Berechnet die Kosinus-Ähnlichkeit zwischen dem Text-Embedding und allen Audio-Embeddings in self.embedding_tensors (nutze torch.nn.functional.cosine_similarity).
Verwendet torch.topk, um die Indizes der top_k Audio-Dateien mit der höchsten Ähnlichkeit effizient zu finden.
Gibt eine Liste der Dateipfade dieser Top-K-Matches zurück.
Haupt-Ausführungsblock (if __name__ == "__main__":):
Erstellt eine Instanz von SearchEngine und übergibt den Pfad zur embeddings.pkl-Datei.
Startet eine while True-Schleife für eine interaktive Nutzersitzung (input("Prompt eingeben (oder 'exit' zum Beenden): ")).
Bei der Eingabe exit wird die Schleife beendet.
Für jede andere Eingabe wird die search-Methode aufgerufen.
Die Ergebnisse (die Top 5 Dateipfade) werden sauber nummeriert und formatiert auf der Konsole ausgegeben.
ERFOLGSKRITERIEN DES GESAMTEN PROJEKTS:
Ein Nutzer kann eine kleine Sammlung (20-100) von Test-WAV-Dateien in den raw_construction_kits-Ordner legen.
Ein Nutzer führt python ai_agents/minimal_preprocessor.py aus. Das Skript verarbeitet alle Dateien in Batches und erstellt eine processed_database/embeddings.pkl-Datei.
Ein Nutzer startet python ai_agents/search_engine_cli.py. Das Programm lädt die Embeddings aus der .pkl-Datei und wartet auf eine Nutzereingabe.
Wenn der Nutzer einen relevanten Prompt wie "dark industrial kick with a punchy attack" eingibt, gibt das Programm eine nummerierte Liste mit den 5 Dateipfaden aus der raw_construction_kits-Bibliothek aus, die am besten zu dieser Beschreibung passen.
Das gesamte System ist von Anfang bis Ende funktionsfähig und läuft stabil und fehlerfrei.
ENDE DER DIREKTIVE