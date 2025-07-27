HYPER-PROMPT v3.2
AGENTEN-DIREKTIVE: 009
PROJEKT: Neuromorphe Traum-Engine v2.0
MEILENSTEIN: Service-Orientierte Architektur & Web-UI
ZIEL: Transformiere die bestehenden Python-Skripte in eine modulare, Docker-fähige Service-Architektur mit einem einfachen Web-Frontend für den Datei-Upload und die Prozess-Steuerung.
BEGINN DER DIREKTIVE
Du bist ein autonomer Full-Stack AI-Engineer. Deine Mission ist es, die bestehende MVP-Implementierung in eine robuste, auf einem VPS deploybare Service-Architektur umzuwandeln.
PHASE 1: REFAKTORISIERUNG ZU SERVICES (BACKEND)
Web-Framework einführen: Wir nutzen FastAPI für seine hohe Performance und einfache Bedienung. Füge fastapi und uvicorn zur requirements.txt hinzu.
Neues Skript main.py erstellen: Dieses Skript wird der zentrale Einstiegspunkt für unser Backend.
Refaktorisierung des Pre-Processors:
Die Logik aus minimal_preprocessor.py wird in main.py integriert.
Erstelle einen API-Endpunkt POST /process: Dieser Endpunkt startet asynchron den Pre-Processing-Lauf für alle neuen Dateien im raw_construction_kits-Ordner. Er soll sofort eine Bestätigung zurückgeben (z.B. {"message": "Processing started"}) und den Prozess im Hintergrund ausführen.
Refaktorisierung der Suche:
Die Logik aus search_engine_cli.py wird ebenfalls in main.py integriert.
Die SearchEngine-Klasse wird beim Start des Servers einmalig initialisiert und lädt die Embeddings in den Speicher.
Erstelle einen API-Endpunkt GET /search: Dieser Endpunkt nimmt einen Query-Parameter prompt entgegen (z.B. /search?prompt=dark kick). Er führt die Suche durch und gibt die Top-5-Ergebnisse als JSON zurück.
PHASE 2: ERSTELLUNG DER WEB-OBERFLÄCHE (FRONTEND)
Technologie-Wahl: Wir verwenden Streamlit für eine extrem schnelle und einfache Erstellung der UI, da es sich perfekt in die Python-Welt integriert. Füge streamlit zur requirements.txt hinzu.
Neues Skript app.py erstellen: Dies wird die Web-App.
UI-Komponenten implementieren:
Titel: "Neuromorphe Traum-Engine v2.0 - Upload & Search"
Datei-Upload: Implementiere eine Upload-Funktion (st.file_uploader), die es dem Nutzer erlaubt, eine oder mehrere .wav-Dateien von seinem Rechner auszuwählen. Hochgeladene Dateien sollen im raw_construction_kits-Ordner auf dem Server gespeichert werden.
Prozess-Start: Ein Button "Verarbeitung starten" (st.button), der bei Klick eine Anfrage an den POST /process-Endpunkt unseres FastAPI-Backends sendet.
Such-Interface: Ein Text-Eingabefeld (st.text_input) für den Such-Prompt und ein "Suchen"-Button. Bei Klick wird der GET /search-Endpunkt aufgerufen, und die Ergebnisse werden übersichtlich auf der Seite angezeigt.
PHASE 3: DOCKER-KAPSELUNG
Dockerfile erstellen: Erstelle ein Dockerfile im Hauptverzeichnis des Projekts.
Inhalt des Dockerfiles:
Basis-Image: python:3.9-slim
Kopiere requirements.txt und installiere die Abhängigkeiten.
Kopiere den gesamten restlichen Projekt-Code in das Image.
Lade das CLAP-Modell während des Image-Baus herunter und cache es, um Startzeiten zu beschleunigen.
Definiere CMD, um sowohl den uvicorn-Server für das Backend als auch die streamlit-App für das Frontend zu starten (dies kann über ein Start-Skript start.sh realisiert werden).
ERFOLGSKRITERIEN:
Ein Nutzer kann mit docker build -t traum-engine . ein funktionierendes Docker-Image des gesamten Projekts erstellen.
Ein Nutzer kann mit docker run -p 8501:8501 -p 8000:8000 -v $(pwd)/raw_construction_kits:/app/raw_construction_kits traum-engine den Container starten.
Der Nutzer kann im Browser auf http://<VPS_IP>:8501 die Streamlit-Web-Oberfläche aufrufen.
Der Nutzer kann über die UI eine WAV-Datei hochladen. Die Datei erscheint im raw_construction_kits-Ordner auf dem Server.
Ein Klick auf "Verarbeitung starten" stößt den Pre-Processing-Lauf im Backend an, was in den Server-Logs sichtbar ist.
Nach Abschluss der Verarbeitung kann der Nutzer über das Suchfeld in der UI einen Prompt eingeben, und die gefundenen Ergebnisse werden auf der Webseite angezeigt.
Das System ist modular, robust und bereit für das Deployment auf deinem VPS.
ENDE DER DIREKTIVE