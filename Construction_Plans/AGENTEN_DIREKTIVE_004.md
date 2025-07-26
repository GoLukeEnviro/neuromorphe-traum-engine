HYPER-PROMPT v1.3
AGENTEN-DIREKTIVE: 004
PROJEKT: Neuromorphe Traum-Engine v2.0
MEILENSTEIN: 1d - Erweiterte Qualitätskontrolle & Cluster-basierte Kategorisierung
ZIEL: Finalisiere den Neuro-Analysator durch Implementierung von erweiterten Audio-Features, einer robusten Qualitätskontrolle und einer intelligenten, auf maschinellem Lernen basierenden Fallback-Kategorisierung.
BEGINN DER DIREKTIVE
Du bist ein autonomer Python-DevOps-Engineer. Deine Aufgabe ist es, ai_agents/prepare_dataset_sql.py zu einem produktionsreifen Analyse-Werkzeug auszubauen.
ANFORDERUNGEN IM DETAIL:
1. Skript prepare_dataset_sql.py modifizieren:
1.1. Zusätzliche Importe: Füge die KMeans-Klasse von Scikit-learn hinzu.
Generated python
from sklearn.cluster import KMeans
# ... andere Importe ...
Use code with caution.
Python
Stelle sicher, dass die Bibliothek installiert ist: pip install scikit-learn.
1.2. NeuroAnalyzer-Klasse erweitern (__init__-Methode):
Initialisiere das KMeans-Modell. Die Anzahl der Cluster (n_clusters) soll der Anzahl der definierten Kategorien entsprechen.
Generated python
class NeuroAnalyzer:
    def __init__(self, input_dir: str):
        # ... bestehender Code ...
        self.kmeans = KMeans(n_clusters=len(self.categories), random_state=42)
Use code with caution.
Python
1.3. Erweiterung der Feature-Extraktion:
Erstelle eine neue private Methode _extract_features(y, sr).
Diese Methode soll ein Audio-Array y und die Abtastrate sr als Input erhalten.
Sie soll ein Dictionary mit den folgenden erweiterten Features berechnen und zurückgeben:
spectral_centroid: Der Mittelwert des spektralen Schwerpunkts.
zero_crossing_rate: Die mittlere Nulldurchgangsrate.
rms: Der Mittelwert der RMS-Energie.
spectral_rolloff: Der Mittelwert des spektralen Rolloffs.
spectral_bandwidth: Der Mittelwert der spektralen Bandbreite.
Alle Werte sollen als float konvertiert werden, um die JSON-Serialisierung zu gewährleisten.
1.4. _process_file()-Methode anpassen:
Rufe die neue Methode _extract_features() auf, um das Dictionary mit den erweiterten Features zu erhalten.
Erweiterte Qualitätskontrolle: Erweitere den bestehenden Qualitäts-Check um eine Prüfung dieser neuen Features. Verschiebe eine Datei in die quarantine, wenn eine der folgenden Bedingungen zutrifft:
spectral_centroid < 100 (Indikator für stark dumpfes Rauschen oder Stille).
rms < 0.001 (Indikator für Fast-Stille).
Aktualisiere das Metadaten-Dictionary, um das gesamte Feature-Dictionary als JSON-serialisierten String im features-Feld zu speichern.
1.5. _get_category()-Methode modifizieren (Cluster-basierte Fallback-Logik):
Passe die bestehende Methode an. Die Logik soll nun wie folgt ablaufen:
Versuche zuerst die Kategorisierung über die Dateinamen-Heuristik (wie bisher).
Wenn die Heuristik fehlschlägt (und "unknown" zurückgeben würde):
Nutze die zuvor berechneten spectral_centroid und rms aus dem Feature-Dictionary.
Forme diese beiden Werte in ein 2D-NumPy-Array [[spectral_centroid, rms]].
Nutze das trainierte self.kmeans-Modell, um eine Cluster-ID für diesen Feature-Vektor vorherzusagen (kmeans.predict(...)).
Gib den Kategorienamen zurück, der dem Index der vorhergesagten Cluster-ID entspricht (z.B. self.categories[prediction]).
1.6. run()-Methode anpassen (KMeans-Training):
Passe die Haupt-run-Methode an.
Vor dem Start der parallelen Verarbeitung musst du das KMeans-Modell trainieren.
Durchlaufe vorab einmal alle Audiodateien und extrahiere nur die für das Clustering relevanten Features (spectral_centroid und rms).
Sammle diese Feature-Paare in einer Liste.
Trainiere das self.kmeans-Modell mit dieser Liste von Feature-Paaren (kmeans.fit(...)).
Fahre erst danach mit der parallelen Verarbeitung der Dateien fort, damit die _get_category-Methode auf ein trainiertes Modell zugreifen kann.
ERFOLGSKRITERIEN:
Lege vier neue Test-Audiodateien in raw_construction_kits/ ab:
test_KICK_01.wav
sample_BASS_heavy.wav
ambient_texture.wav (ein Dateiname ohne Kategorie-Schlüsselwort)
Eine fast stille Audiodatei (z.B. eine exportierte leere Spur).
Wenn das modifizierte Skript ausgeführt wird, läuft es ohne Fehler durch. Der initiale Durchlauf für das KMeans-Training kann die Startzeit leicht verlängern.
Die stille Audiodatei wird korrekt in den quarantine-Ordner verschoben.
Die SQLite-Datenbank enthält drei neue Einträge.
Überprüfe die drei neuen Einträge:
Die category-Werte für kick und bass wurden durch die Dateinamen-Heuristik korrekt erkannt.
Der category-Wert für ambient_texture.wav ist nicht "unknown", sondern eine der definierten Kategorien, die durch das KMeans-Clustering zugewiesen wurde.
Das features-Feld ist für alle drei Einträge gefüllt und enthält einen JSON-String mit den fünf berechneten erweiterten Features.
ENDE DER DIREKTIVE