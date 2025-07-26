HYPER-PROMPT v1.1
AGENTEN-DIREKTIVE: 002
PROJEKT: Neuromorphe Traum-Engine v2.0
MEILENSTEIN: 1b - Implementierung der Kernlogik des Neuro-Analysators
ZIEL: Erweitere die NeuroAnalyzer-Klasse um die Fähigkeit, Audiodateien zu finden, zu laden, zu standardisieren und grundlegende Metadaten (BPM, Dauer, Pfad) zu extrahieren und in die Datenbank zu schreiben.
BEGINN DER DIREKTIVE
Du bist ein autonomer Python-DevOps-Engineer. Deine Aufgabe ist es, die bestehende Datei ai_agents/prepare_dataset_sql.py zu modifizieren und die Kernlogik zu implementieren.
ANFORDERUNGEN IM DETAIL:
1. Skript prepare_dataset_sql.py modifizieren:
1.1. Zusätzliche Importe: Füge die folgenden Importe am Anfang der Datei hinzu:
Generated python
import librosa
import numpy as np
import soundfile as sf
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
Use code with caution.
Python
Falls librosa oder andere Bibliotheken nicht installiert sind, installiere sie (pip install librosa soundfile).
1.2. run()-Methode implementieren:
Implementiere die run-Methode. Ihre Aufgabe ist es:
Das input_dir ("raw_construction_kits/") rekursiv nach allen Audio-Dateien (Endungen: .wav, .mp3, .aif, .flac) zu durchsuchen.
Eine Liste aller gefundenen Dateipfade zu erstellen.
Eine parallele Verarbeitung dieser Liste mittels concurrent.futures.ProcessPoolExecutor zu starten. Jeder gefundene Dateipfad soll an die _process_file-Methode übergeben werden.
Die Ergebnisse (die Metadaten-Dictionaries) aus der parallelen Verarbeitung zu sammeln.
Für jedes erfolgreiche Ergebnis die _insert_meta-Methode aufzurufen, um die Daten in die Datenbank zu schreiben.
Nach Abschluss eine zusammenfassende Log-Nachricht auszugeben, wie viele Dateien erfolgreich verarbeitet wurden.
1.3. _process_file()-Methode implementieren:
Implementiere die _process_file-Methode. Ihre Aufgabe für eine einzelne Datei ist:
Laden & Standardisieren: Die Audiodatei mit librosa.load(file_path, sr=48000, mono=True) laden. Dadurch wird die Datei direkt auf 48kHz Abtastrate resampelt und in ein Monosignal umgewandelt.
Dauer & Qualitäts-Check (einfach): Die Dauer des Audiosignals in Sekunden ermitteln. Wenn die Dauer unter 0.5 Sekunden liegt:
Die Datei in den quarantine-Ordner verschieben (mittels shutil.move).
None zurückgeben, um die Verarbeitung dieser Datei abzubrechen.
BPM-Analyse: Das Tempo (BPM) des Audiosignals mit librosa.beat.tempo schätzen.
Speichern des standardisierten Audios:
Einen einzigartigen stem_id erstellen. Ein gutes Format ist "{category}_{timestamp}_{index}", aber für jetzt reicht "{filename_without_ext}_{timestamp}".
Den neuen Pfad für die Ausgabedatei im stems-Ordner erstellen.
Das standardisierte Audio-Array (y) mit soundfile.write als 24-bit WAV-Datei an diesem neuen Pfad speichern.
Metadaten-Dictionary erstellen: Ein Dictionary mit den bisher extrahierten Daten erstellen, das exakt dem Datenbankschema entspricht. Platzhalter für noch nicht implementierte Felder (wie key, tags, features) sollen None sein.
Das ausgefüllte Metadaten-Dictionary zurückgeben.
Umschließe die gesamte Methode mit einem try...except-Block. Bei jeglichem Fehler soll eine Fehlermeldung geloggt und None zurückgegeben werden.
1.4. _insert_meta()-Methode implementieren:
Implementiere die _insert_meta-Methode. Ihre Aufgabe ist es:
Eine Verbindung zur SQLite-Datenbank herzustellen.
Ein INSERT-Statement auszuführen, das die Werte aus dem übergebenen Metadaten-Dictionary in die stems-Tabelle einfügt.
Die Transaktion mit commit() abzuschließen und die Verbindung zu schließen.
Eine Log-Nachricht auszugeben, die bestätigt, dass ein Stem hinzugefügt wurde.
ERFOLGSKRITERIEN:
Lege zwei Test-Audiodateien in den Ordner raw_construction_kits/: eine längere als 0.5s, eine kürzere.
Wenn das modifizierte Skript ausgeführt wird, läuft es ohne Fehler durch und nutzt mehrere CPU-Kerne.
Die kürzere Testdatei wird in den quarantine-Ordner verschoben.
Die längere Testdatei wird verarbeitet und eine standardisierte 48kHz/24-bit Mono-WAV-Datei wird im stems-Ordner gespeichert.
Die SQLite-Datenbank stems.db enthält nach der Ausführung genau einen neuen Eintrag in der stems-Tabelle.
Dieser Eintrag enthält die korrekten Werte für id, path, bpm und imported_at. Die anderen Felder sind mit NULL gefüllt.
Die Konsolenausgabe zeigt informative Log-Nachrichten über den gesamten Prozess.
ENDE DER DIREKTIVE