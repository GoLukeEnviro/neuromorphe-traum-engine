HYPER-PROMPT v2.2
AGENTEN-DIREKTIVE: 006
PROJEKT: Neuromorphe Traum-Engine v2.0
MEILENSTEIN: 1e - Stabilisierung & Vervollständigung der Daten-Pipeline
ZIEL: Modifiziere das prepare_dataset_sql.py-Skript, um eine 100%ige Verarbeitungsrate zu gewährleisten. Implementiere robustes Error-Handling, einen "Resume"-Mechanismus und eine Batch-Verarbeitung für CLAP-Embeddings, um Effizienz und Stabilität zu maximieren.
BEGINN DER DIREKTIVE
Du bist ein autonomer Python-DevOps-Engineer. Deine Aufgabe ist es, die Daten-Pipeline (prepare_dataset_sql.py) zu einem produktionsreifen, ausfallsicheren System zu machen.
ANFORDERUNGEN IM DETAIL:
1. Skript prepare_dataset_sql.py modifizieren:
1.1. Robuste Fehlerbehandlung in _process_file:
Umschließe den gesamten Inhalt der _process_file-Methode mit einem try...except Exception as e-Block.
Im Fehlerfall:
Logge eine detaillierte Fehlermeldung, die den Dateipfad und die spezifische Exception enthält.
Verschiebe die fehlerhafte Datei in den quarantine-Ordner mit einem aussagekräftigen Präfix (z.B. ERROR_...).
Die Methode soll None zurückgeben, um sicherzustellen, dass fehlerhafte Einträge niemals in die Datenbank gelangen.
1.2. Implementierung eines "Resume"-Mechanismus:
Verhindere, dass bei jedem Neustart alle Dateien erneut verarbeitet werden.
In der run-Methode: Bevor die Dateiliste zur Verarbeitung an den ProcessPoolExecutor übergeben wird:
Frage die SQLite-Datenbank ab und hole eine Liste aller original_file-Pfade, die bereits erfolgreich verarbeitet wurden (SELECT original_file FROM stems).
Filtere die Liste der zu verarbeitenden Dateien, sodass nur noch die Dateien übrig bleiben, die nicht bereits in der Datenbank vorhanden sind.
Gib eine Log-Nachricht aus, wie viele Dateien bereits verarbeitet waren und wie viele neue Dateien verarbeitet werden.
1.3. Effiziente Batch-Verarbeitung für CLAP-Embeddings:
Die Inferenz für jede Datei einzeln ist ineffizient. Wir stellen auf eine Batch-Verarbeitung um.
Struktur anpassen: Die _process_file-Methode soll das CLAP-Embedding nicht mehr selbst berechnen. Sie soll nur noch das geladene und standardisierte Audio-Array (y) zurückgeben, zusammen mit den restlichen Metadaten.
Neue Logik in der run-Methode:
Nachdem die run-Methode die Ergebnisse der _process_file-Aufrufe (Metadaten-Dictionaries und Audio-Arrays) gesammelt hat:
Erstelle eine "Batch-Liste" aller Audio-Arrays.
Übergebe diese komplette Liste in einem einzigen Aufruf an self.clap_model.get_audio_embedding_from_data(...), um alle Embeddings auf einmal zu berechnen.
Füge die zurückgegebenen Embedding-Vektoren den entsprechenden Metadaten-Dictionaries hinzu.
Führe erst danach die _insert_meta-Operation für alle vervollständigten Metadaten durch.
ERFOLGSKRITERIEN:
Das modifizierte Skript wird auf einen Ordner mit 74 Testdateien angewendet, von denen einige absichtlich korrupt sind (z.B. eine leere Datei oder eine Textdatei mit .wav-Endung).
Das Skript läuft ohne Absturz durch. Die korrupten Dateien werden in den quarantine-Ordner verschoben, und es werden entsprechende Fehler im Log protokolliert.
Die Datenbank-Abdeckung erreicht 100% der gültigen Dateien. Alle 70+ nicht-korrupten Dateien sind in der Datenbank eingetragen.
Wenn das Skript ein zweites Mal auf denselben Ordner ausgeführt wird, erkennt der "Resume"-Mechanismus, dass keine neuen Dateien zu verarbeiten sind, und beendet den Vorgang schnell mit einer entsprechenden Log-Nachricht.
Die Verarbeitungsgeschwindigkeit (insbesondere die CLAP-Embedding-Berechnung) ist durch die Batch-Verarbeitung spürbar höher als zuvor.
ENDE DER DIREKTIVE