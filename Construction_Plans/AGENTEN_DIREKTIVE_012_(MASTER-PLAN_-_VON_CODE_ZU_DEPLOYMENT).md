HYPER-PROMPT v5.0 (MASTER-PROMPT – KOMPLETT-IMPLEMENTIERUNG)
AGENTEN-DIREKTIVE: 012 (MASTER-PLAN - VON CODE ZU DEPLOYMENT)
PROJEKT: Neuromorphe Traum-Engine v2.0
REFERENZ-DOKUMENTE: BACKEND_ARCHITECTURE_SPEC.md, FRONTEND_ARCHITECTURE_SPEC.md, DOCKER_DEPLOYMENT_SPEC.md
MISSION: Transformiere die bestehende Code-Basis in eine voll funktionsfähige, Docker-gekapselte, Multi-Container-Anwendung mit Backend-API und Web-Frontend, bereit für das Deployment auf einem VPS.
BEGINN DER DIREKTIVE
Du bist ein autonomer Full-Stack AI- & DevOps-Engineer. Deine Mission ist die vollständige Implementierung und Containerisierung der "Neuromorphen Traum-Engine". Du wirst die bestehende Code-Basis erweitern, ein interaktives Frontend erstellen und alles in eine produktionsreife Docker-Umgebung verpacken.
PHASE 1: FINALISIERUNG DES BACKEND-SERVICE (FastAPI)
1.1. Datenbank- & CRUD-Integration (basierend auf Direktive 011):
Struktur: Erstelle die Verzeichnisse src/db und src/schemas und die darin enthaltenen Dateien (database.py, models.py, crud.py, stem.py) exakt wie in Direktive 011 beschrieben.
Logik: Implementiere die SQLAlchemy-Modelle, Pydantic-Schemas und die get_stems / get_stem_by_id CRUD-Funktionen.
1.2. API-Endpunkte für Stems (src/api/endpoints/stems.py):
Erstelle den Router, der die Endpunkte GET / (Liste aller Stems) und GET /{stem_id} bereitstellt.
Implementiere die get_db Dependency für saubere Datenbank-Sessions.
Binde den neuen Router in src/main.py unter dem Präfix /api/v1/stems ein.
1.3. Semantische Suche-API:
Struktur: Erstelle eine neue Datei src/services/search.py.
Logik search.py: Refaktorisiere die Logik aus dem alten search_engine_cli.py in eine SearchService-Klasse. Lade das CLAP-Modell und die Embeddings aus der DB beim Start des Servers.
API-Endpunkt: Erstelle einen neuen Endpunkt GET /api/v1/search/ im stems.py-Router.
Dieser Endpunkt nimmt einen Query-Parameter prompt: str entgegen.
Er nutzt den SearchService, um die Top-5-Ergebnisse zu finden.
Er gibt die Ergebnisse als Liste von schemas.stem.Stem zurück.
PHASE 2: IMPLEMENTIERUNG DES FRONTEND-SERVICE (Streamlit)
2.1. Frontend-Struktur:
Erstelle eine neue Haupt-Applikationsdatei app.py im Projekt-Root-Verzeichnis.
Erstelle die Verzeichnisstruktur pages/ für die einzelnen Unterseiten.
2.2. Haupt-App app.py:
Logik: Dies ist der Einstiegspunkt für die Streamlit-App. Konfiguriere hier den Seitentitel, das Layout und die grundlegende Navigation zwischen den Seiten, die in pages/ liegen.
2.3. Upload-Seite (pages/1_Upload_Stems.py):
UI: Implementiere eine Seite mit einem st.file_uploader, der mehrere .wav-Dateien akzeptiert.
Logik: Wenn Dateien hochgeladen werden, sende sie via requests.post an einen (noch zu erstellenden) POST /api/v1/stems/upload-Endpunkt im Backend. Zeige eine Erfolgsmeldung an.
2.4. Such-Seite (pages/2_Search_Stems.py):
UI: Implementiere eine Seite mit einem st.text_input für den Prompt und einem "Suchen"-Button.
Logik: Bei Klick auf "Suchen" sende eine Anfrage an den GET /api/v1/search/?prompt=...-Endpunkt des Backends.
Ergebnisanzeige: Stelle die zurückgegebenen Stem-Informationen (z.B. Pfad, BPM, Kategorie) übersichtlich dar. Füge für jeden Treffer einen st.audio-Player hinzu, damit der Nutzer die Datei direkt im Browser anhören kann.
PHASE 3: DOCKER-CONTAINERISIERUNG
3.1. Dockerfile für das Backend:
Datei: backend.dockerfile
Basis: python:3.9-slim
Logik: Installiert requirements.txt, kopiert das src/-Verzeichnis und startet den uvicorn-Server auf Port 8000.
3.2. Dockerfile für das Frontend:
Datei: frontend.dockerfile
Basis: python:3.9-slim
Logik: Installiert requirements.txt (oder eine separate frontend-requirements.txt), kopiert app.py und das pages/-Verzeichnis und startet die Streamlit-App auf Port 8501.
3.3. docker-compose.yml:
Datei: docker-compose.yml im Projekt-Root.
Services: Definiere zwei Services: backend und frontend.
backend: Baut aus backend.dockerfile. Mappt Port 8000. Bindet die Volumes processed_database und raw_construction_kits ein, damit die Daten persistent bleiben.
frontend: Baut aus frontend.dockerfile. Mappt Port 8501. Hängt vom backend-Service ab (depends_on).
Netzwerk: Definiere ein gemeinsames Netzwerk, damit die Services miteinander kommunizieren können (z.B. kann das Frontend das Backend unter http://backend:8000 erreichen).
PHASE 4: FERTIGSTELLUNG & DOKUMENTATION
4.1. README.md aktualisieren:
Erstelle eine Sektion "Quickstart with Docker".
Füge den Befehl docker-compose up --build hinzu.
Erkläre kurz, wie man auf das Frontend (http://localhost:8501) zugreift und die API-Docs (http://localhost:8000/docs).
4.2. .env.example:
Erstelle eine .env.example-Datei, um zukünftige Konfigurationsvariablen (z.B. Datenbank-URL, Passwörter) zu dokumentieren.
ERFOLGSKRITERIEN DES GESAMTEN PROJEKTS:
Der Befehl docker-compose up --build im Projekt-Root startet erfolgreich zwei Container (backend und frontend) ohne Fehler.
Das Frontend ist unter http://localhost:8501 erreichbar.
Die Backend-API-Dokumentation ist unter http://localhost:8000/docs erreichbar.
Ein Nutzer kann über das Web-Frontend eine WAV-Datei hochladen. Diese Datei landet im raw_construction_kits-Ordner, und ein entsprechender API-Aufruf wird in den Backend-Logs sichtbar.
Ein Nutzer kann im Web-Frontend einen Text-Prompt eingeben. Die Suchanfrage wird an das Backend gesendet, und die Ergebnisse werden inklusive Audio-Playern im Frontend angezeigt.
Die Daten (stems.db und die Audio-Dateien) bleiben auch nach einem Neustart der Container erhalten (persistente Volumes).
Das gesamte System ist gekapselt, reproduzierbar und bereit für das Deployment auf einem VPS, auf dem Docker und Docker Compose installiert sind.
ENDE DER DIREKTIVE