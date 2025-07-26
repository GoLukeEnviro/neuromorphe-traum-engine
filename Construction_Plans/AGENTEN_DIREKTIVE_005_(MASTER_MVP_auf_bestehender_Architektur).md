HYPER-PROMPT v2.1 (MASTER-PROMPT)
AGENTEN-DIREKTIVE: 005 (MASTER-MVP auf bestehender Architektur)
PROJEKT: Neuromorphe Traum-Engine v2.0 (MVP)
MISSION: Erweitere das bestehende "Neuro-Analyse"-System um semantische Vektoren und implementiere ein Kommandozeilen-Tool für Text-basiertes Audio-Retrieval, um die Kern-Hypothese zu validieren.
BEGINN DER DIREKTIVE
Du bist ein autonomer Full-Stack AI-Engineer. Deine Mission ist die Implementierung eines voll funktionsfähigen "Text-zu-Stem"-Retrieval-MVP, indem du die existierende Code-Basis modifizierst und erweiterst.
Das Projekt besteht aus ZWEI TEILAUFGABEN, die du umsetzen wirst:
MODIFIKATION des Pre-Processing-Skripts (prepare_dataset_sql.py): Anreicherung der bestehenden Datenbank um Vektor-Embeddings.
NEUERSTELLUNG eines Kommandozeilen-Such-Tools (search_engine_cli.py): Implementierung der eigentlichen Suchfunktion.
PHASE 1: MODIFIKATION DES NEURO-ANALYSATORS (prepare_dataset_sql.py)
ZIEL: Erweitere das Skript, sodass es für jede Audiodatei ein CLAP-Embedding berechnet und dieses direkt in der SQLite-Datenbank speichert.
Implementierungsdetails:
Abhängigkeiten sicherstellen: Vergewissere dich, dass die requirements.txt die Einträge laion-clap-pytorch, torch und numpy enthält.
Klasse NeuroAnalyzer modifizieren:
Importe: Füge die notwendigen Importe hinzu: from laion_clap import CLAP_Module.
__init__-Methode: Lade das CLAP-Modell beim Start des Analysators:
Generated python
# Innerhalb von __init__
logging.info("Lade LAION-CLAP-Modell...")
self.clap_model = CLAP_Module(enable_fusion=False)
self.clap_model.load_ckpt()
Use code with caution.
Python
init_db-Methode: Passe das CREATE TABLE-Statement an. Füge eine neue Spalte hinzu:
Generated sql
-- SQL-Statement anpassen
...
imported_at DATETIME,
clap_embedding BLOB
...
Use code with caution.
SQL
_process_file-Methode:
Nach dem Laden der Audiodatei (y, sr = librosa.load(...)), berechne das Audio-Embedding. Nutze self.clap_model.get_audio_embedding_from_data(x=torch.from_numpy(y).unsqueeze(0)).
Konvertiere den Embedding-Tensor in ein serialisierbares Format, z.B. einen NumPy-Array (embedding.cpu().numpy()).
Erweitere das zurückgegebene Metadaten-Dictionary um den Schlüssel clap_embedding, der den NumPy-Array enthält.
_insert_meta-Methode:
Passe das INSERT-Statement an, um die neue clap_embedding-Spalte zu berücksichtigen.
SQLite's BLOB-Typ kann direkt mit dem NumPy-Array (.tobytes()) oder einem serialisierten Objekt (z.B. via pickle) befüllt werden. Stelle sicher, dass die Daten korrekt gespeichert werden.
PHASE 2: NEUERSTELLUNG DER SUCHMASCHINE (search_engine_cli.py)
ZIEL: Erstelle ein eigenständiges Kommandozeilen-Tool, das die in Phase 1 erstellte Datenbank nutzt, um auf Text-Prompts die ähnlichsten Sounds zu finden.
Implementierungsdetails:
Datei erstellen: ai_agents/search_engine_cli.py.
Importe: sqlite3, torch, numpy, laion_clap.
Klasse SearchEngine:
__init__(self, db_path):
Lade das CLAP-Modell, genau wie im Pre-Processor.
Stelle eine Verbindung zur SQLite-Datenbank her (db_path).
Lade alle Stems aus der Datenbank in den Arbeitsspeicher. Lese die id, den path und das clap_embedding für jeden Eintrag.
Deserialisiere die clap_embedding BLOBs zurück in PyTorch-Tensoren und speichere sie zusammen mit den Pfaden in zwei Listen (z.B. self.embedding_tensors und self.file_paths), die denselben Index haben.
search(self, prompt: str, top_k: int = 5):
Nimm einen Text-Prompt entgegen.
Berechne das Text-Embedding für den Prompt mit self.clap_model.get_text_embedding(...).
Berechne die Kosinus-Ähnlichkeit zwischen dem Text-Embedding und allen Audio-Embeddings in self.embedding_tensors.
Finde die Indizes der top_k Audio-Embeddings mit der höchsten Ähnlichkeit.
Nutze diese Indizes, um die entsprechenden Dateipfade aus self.file_paths zu holen.
Gib eine Liste dieser Dateipfade zurück.
Haupt-Ausführungsblock (if __name__ == "__main__":):
Definiere den Pfad zur Datenbank: DB_PATH = "processed_database/stems.db".
Erstelle eine Instanz von SearchEngine.
Starte eine Endlosschleife für die interaktive Eingabe (input("Prompt eingeben...")).
Rufe die search-Methode auf und gib die Ergebnisse formatiert auf der Konsole aus.
ERFOLGSKRITERIEN DES GESAMTEN PROJEKTS:
Ein Nutzer legt Test-Audiodateien in raw_construction_kits/.
Ein Nutzer führt python ai_agents/prepare_dataset_sql.py aus. Das Skript verarbeitet die Dateien und füllt die SQLite-Datenbank, inklusive der clap_embedding-Spalte.
Ein Nutzer kann python ai_agents/search_engine_cli.py starten. Das Programm lädt die Daten aus der DB.
Wenn der Nutzer einen Prompt wie "dark industrial kick with heavy bass" eingibt, gibt das Programm eine nummerierte Liste der 5 Dateipfade aus, deren Klänge am besten zur Beschreibung passen.
Das System ist robust, die beiden Skripte arbeiten fehlerfrei zusammen.
ENDE DER DIREKTIVE