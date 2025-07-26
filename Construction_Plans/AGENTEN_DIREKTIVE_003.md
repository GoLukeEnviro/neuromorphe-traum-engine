HYPER-PROMPT v1.2
AGENTEN-DIREKTIVE: 003
PROJEKT: Neuromorphe Traum-Engine v2.0
MEILENSTEIN: 1c - Semantische Analyse & Intelligente Kategorisierung
ZIEL: Integriere das LAION-CLAP-Modell, um jedem Stem automatisch aussagekräftige, genre-spezifische Tags zuzuordnen. Implementiere eine erste Version der intelligenten Kategorisierung.
BEGINN DER DIREKTIVE
Du bist ein autonomer Python-DevOps-Engineer. Deine Aufgabe ist es, das bestehende Skript ai_agents/prepare_dataset_sql.py um semantische Analysefähigkeiten zu erweitern.
ANFORDERUNGEN IM DETAIL:
1. Skript prepare_dataset_sql.py modifizieren:
1.1. Zusätzliche Importe: Füge die laion_clap Bibliothek hinzu und erweitere die Typ-Hinweise.
Generated python
from laion_clap import CLAP_Module
# ... andere Importe ...
from typing import List, Dict, Optional
Use code with caution.
Python
Stelle sicher, dass die Bibliothek installiert ist: pip install laion-clap-pytorch.
1.2. NeuroAnalyzer-Klasse erweitern:
Modifiziere die __init__-Methode:
Lade das CLAP-Modell. Dies ist ein rechenintensiver Schritt und sollte nur einmal bei der Initialisierung der Klasse geschehen. Nutze enable_fusion=False für eine breitere Kompatibilität.
Definiere eine Liste von Kandidaten-Tags, die für unser Genre relevant sind. Dies ist unsere "semantische Palette".
Definiere eine Liste der Ziel-Kategorien.
Generated python
class NeuroAnalyzer:
    def __init__(self, input_dir: str):
        # ... bestehender Code ...
        logging.info("Lade LAION-CLAP-Modell. Dies kann einen Moment dauern...")
        self.clap_model = CLAP_Module(enable_fusion=False)
        self.clap_model.load_ckpt() # Lädt Standard-Checkpoint
        
        self.tags_candidates = ["dark", "punchy", "hypnotic", "industrial", "gritty", "atmospheric", "driving", "melodic", "percussive", "minimal"]
        self.categories = ["kick", "bass", "synth", "hihat", "snare", "percussion", "fx", "atmo"]
Use code with caution.
Python
1.3. Neue Methode _get_clap_tags() implementieren:
Erstelle eine neue private Methode, die ein Audio-Array entgegennimmt und eine Liste der passendsten Tags zurückgibt.
Die Methode soll die Audio- und Text-Embeddings mithilfe des geladenen clap_model berechnen.
Berechne die Kosinus-Ähnlichkeit zwischen dem Audio-Embedding und allen Text-Embeddings der Kandidaten-Tags.
Wähle die Top 3 Tags mit der höchsten Ähnlichkeit aus.
Gib diese drei Tags als Liste von Strings zurück.
1.4. Neue Methode _get_category() implementieren:
Erstelle eine neue private Methode zur Kategorisierung. Dies ist eine erste, heuristische Implementierung.
Dateinamen-Heuristik: Durchsuche den (kleingeschriebenen) Dateinamen der Originaldatei nach Schlüsselwörtern aus der self.categories-Liste. Wenn ein passendes Schlüsselwort gefunden wird (z.B. "kick" im Dateinamen), gib diese Kategorie zurück.
Fallback: Wenn kein Schlüsselwort gefunden wird, gib vorerst die Kategorie "unknown" zurück.
1.5. _process_file()-Methode anpassen:
Integriere die neuen Methoden in den Hauptverarbeitungsschritt:
Rufe nach der Qualitätskontrolle die neue Methode _get_clap_tags() mit dem geladenen Audio-Array auf, um die semantischen Tags zu erhalten.
Rufe die neue Methode _get_category() auf, um die Kategorie des Stems zu bestimmen.
Aktualisiere das Metadaten-Dictionary:
Fülle den tags-Schlüssel mit dem JSON-serialisierten String der zurückgegebenen Tag-Liste.
Fülle den category-Schlüssel mit der zurückgegebenen Kategorie.
Passe den stem_id-Generator an, sodass er die ermittelte Kategorie verwendet (z.B. {category}_{timestamp}).
ERFOLGSKRITERIEN:
Lege drei neue Test-Audiodateien in raw_construction_kits/ ab, deren Dateinamen jeweils die Wörter kick, bass und loop enthalten.
Wenn das modifizierte Skript ausgeführt wird, läuft es ohne Fehler durch. Das erstmalige Laden des CLAP-Modells kann einige Zeit dauern und signifikanten Speicher verbrauchen.
Die SQLite-Datenbank enthält nach der Ausführung drei neue Einträge.
Überprüfe die drei neuen Einträge:
Der category-Wert für die ersten beiden Stems muss korrekt kick und bass sein. Der dritte Stem muss die Kategorie unknown haben.
Das tags-Feld darf nicht NULL sein. Es muss einen JSON-String enthalten, der eine Liste von drei Wörtern aus der tags_candidates-Liste darstellt (z.B. ["percussive", "punchy", "dark"]).
Die Konsolenausgabe zeigt eine Log-Nachricht, die das Laden des CLAP-Modells bestätigt.
ENDE DER DIREKTIVE