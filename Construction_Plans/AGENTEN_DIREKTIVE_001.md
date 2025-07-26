HYPER-PROMPT v1.0
AGENTEN-DIREKTIVE: 001
PROJEKT: Neuromorphe Traum-Engine v2.0
MEILENSTEIN: 1b - Implementierung des Grundgerüsts des Neuro-Analysators
ZIEL: Erschaffe das programmatische Fundament für unsere Daten-Pipeline.
BEGINN DER DIREKTIVE
Du bist ein autonomer Python-DevOps-Engineer. Deine Aufgabe ist es, die grundlegende Infrastruktur für den "Neuro-Analysator" zu erschaffen. Das Endergebnis dieser Aufgabe ist ein Python-Skript, das eine leere, aber korrekt strukturierte SQLite-Datenbank initialisiert und die zentrale NeuroAnalyzer-Klasse mit allen erforderlichen Methoden als Stubs (leere Methoden) definiert.
ANFORDERUNGEN IM DETAIL:
1. Verzeichnisstruktur überprüfen und erstellen:
Stelle sicher, dass die folgende Ordnerstruktur im Hauptverzeichnis des Projekts existiert. Erstelle sie, falls sie fehlt:
Generated code
/
|-- raw_construction_kits/      # Input für rohe Audiodaten (bleibt leer)
|-- processed_database/
|   |-- stems/                  # Output für verarbeitete WAV-Dateien
|   |-- quarantine/             # Für fehlerhafte Dateien
`-- ai_agents/                  # Dein Arbeitsverzeichnis
    `-- prepare_dataset_sql.py  # Das Skript, das du erstellst
Use code with caution.
2. Python-Skript erstellen: ai_agents/prepare_dataset_sql.py
Erstelle eine neue Python-Datei mit exakt diesem Namen und folgendem Inhalt:
2.1. Importe: Binde alle notwendigen Bibliotheken für die zukünftige Funktionalität ein.
Generated python
import os
import sqlite3
import logging
from datetime import datetime
Use code with caution.
Python
2.2. Logging-Konfiguration: Richte ein einfaches Logging ein, um den Fortschritt und Fehler zu protokollieren.
Generated python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
Use code with caution.
Python
2.3. Datenbank-Definition: Definiere den Pfad zur Datenbank und das exakte SQL-Schema.
Generated python
DB_PATH = "processed_database/stems.db"
Use code with caution.
Python
2.4. Klasse NeuroAnalyzer: Erstelle die Hauptklasse. Sie dient als Kapselung für die gesamte Logik.
Generated python
class NeuroAnalyzer:
    def __init__(self, input_dir: str):
        """
        Initialisiert den Neuro-Analysator.
        
        Args:
            input_dir (str): Das Verzeichnis mit den rohen Audiodaten.
        """
        logging.info("Initialisiere Neuro-Analysator...")
        self.input_dir = input_dir
        # Spätere Initialisierung von CLAP, KMeans etc. kommt hier hin

    def init_db(self):
        """
        Erstellt die SQLite-Datenbank und die 'stems'-Tabelle, falls sie nicht existiert.
        Das Schema muss exakt den Spezifikationen entsprechen.
        """
        logging.info(f"Initialisiere Datenbank unter: {DB_PATH}")
        # Dein Code zum Erstellen der DB und Tabelle hier

    def run(self):
        """
        Die Hauptmethode, die den gesamten Analyseprozess steuert.
        Momentan nur ein Platzhalter.
        """
        logging.info("Neuro-Analyse gestartet...")
        # Implementierung des Loops über Dateien kommt hier hin
        logging.info("Neuro-Analyse abgeschlossen.")

    def _process_file(self, file_path: str) -> dict:
        """
        Verarbeitet eine einzelne Audiodatei. Diese Methode wird später
        die gesamte Analyse-Logik enthalten. Momentan nur ein Stub.
        
        Args:
            file_path (str): Der Pfad zur zu verarbeitenden Datei.
            
        Returns:
            dict: Ein Dictionary mit den extrahierten Metadaten oder None bei Fehler.
        """
        # Platzhalter für zukünftige Implementierung
        pass

    def _insert_meta(self, metadata: dict):
        """
        Fügt ein Metadaten-Dictionary in die SQLite-Datenbank ein.
        Momentan nur ein Stub.
        
        Args:
            metadata (dict): Das Dictionary mit den zu speichernden Metadaten.
        """
        # Platzhalter für zukünftige Implementierung
        pass
Use code with caution.
Python
3. Implementierung der Methode init_db:
Implementiere die Methode init_db so, dass sie eine Verbindung zur stems.db herstellt und eine Tabelle namens stems mit exakt den folgenden Spalten und Datentypen erstellt:
Spaltenname	Datentyp	Beschreibung
id	TEXT	Primary Key, eine einzigartige ID für jeden Stem (z.B. "kick_20250726103000")
path	TEXT	Absoluter Pfad zur verarbeiteten WAV-Datei im stems-Ordner
bpm	REAL	Beats Per Minute als Fließkommazahl
key	TEXT	Geschätzte Tonart (z.B. "C#m", "atonal")
category	TEXT	Hauptkategorie (z.B. "kick", "bass", "synth")
tags	TEXT	JSON-kodiertes Array von CLAP-Tags (z.B. ["dark", "industrial"])
features	TEXT	JSON-kodiertes Objekt mit erweiterten Audio-Features (z.B. {"spectral_centroid": 1234.5, "rms": 0.2})
quality_ok	BOOLEAN	True, wenn die Qualitätskontrolle bestanden wurde, sonst False
user_rating	INTEGER	Platzhalter für späteres Reinforcement Learning (RLHF), Standardwert NULL
imported_at	DATETIME	Zeitstempel der Verarbeitung im ISO 8601-Format
4. Haupt-Ausführungsblock (if __name__ == "__main__":)
Füge am Ende des Skripts einen Standard-Ausführungsblock hinzu. Dieser Block soll eine Instanz des NeuroAnalyzer erstellen und die init_db-Methode aufrufen, um die Datenbank zu erstellen. Anschließend soll run aufgerufen werden.
Generated python
if __name__ == "__main__":
    analyzer = NeuroAnalyzer(input_dir="raw_construction_kits")
    analyzer.init_db()
    analyzer.run()
Use code with caution.
Python
ERFOLGSKRITERIEN:
Das Skript prepare_dataset_sql.py existiert im korrekten Verzeichnis.
Wenn das Skript ausgeführt wird (python ai_agents/prepare_dataset_sql.py), läuft es ohne Fehler durch.
Nach der Ausführung existiert die Datei processed_database/stems.db.
Wenn die stems.db-Datei mit einem SQLite-Browser geöffnet wird, enthält sie eine leere Tabelle namens stems mit exakt den 10 oben definierten Spalten.
Es werden informative Log-Nachrichten auf der Konsole ausgegeben, die den Start des Analysators und die Initialisierung der Datenbank bestätigen.
ENDE DER DIREKTIVE