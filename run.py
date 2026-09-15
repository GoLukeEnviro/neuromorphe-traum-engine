#!/usr/bin/env python3
"""
Entry Point für die Neuromorphe Traum-Engine v2.0

Diese Datei dient als zentraler Startpunkt für die Anwendung,
unabhängig davon, ob sie lokal oder in Docker läuft.
"""

import uvicorn
import argparse
import os
import sys
from pathlib import Path

# Füge das src-Verzeichnis zum Python-Path hinzu
# Dies ist notwendig für Docker und lokale Ausführung
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Hauptfunktion zum Starten der Anwendung."""
    parser = argparse.ArgumentParser(description="Neuromorphe Traum-Engine v2.0")
    parser.add_argument(
        "--mode", 
        choices=["dev", "prod"], 
        default="dev",
        help="Startmodus (dev für Entwicklung, prod für Produktion)"
    )
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Host-Adresse für den Server"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port für den Server"
    )
    parser.add_argument(
        "--log-level", 
        choices=["debug", "info", "warning", "error"], 
        default="info",
        help="Log-Level für die Anwendung"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Auto-reload bei Code-Änderungen (nur dev-mode)"
    )

    args = parser.parse_args()

    # Konfiguriere Umgebungsvariablen
    os.environ.setdefault("ENVIRONMENT", args.mode)
    os.environ.setdefault("LOG_LEVEL", args.log_level.upper())

    # Bestimme die Anwendung zu starten
    app_module = "main:app"

    # Konfiguriere uvicorn basierend auf dem Modus
    uvicorn_config = {
        "app": app_module,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": True,
    }

    if args.mode == "dev":
        uvicorn_config.update({
            "reload": args.reload or True,
            "reload_dirs": [str(src_path)],
        })
    else:
        uvicorn_config.update({
            "reload": False,
            "workers": 1,  # Single worker für unsere Anwendung
        })

    print(f"🎵 Starte Neuromorphe Traum-Engine v2.0 im {args.mode.upper()}-Modus")
    print(f"   Host: {args.host}:{args.port}")
    print(f"   Log-Level: {args.log_level}")
    print(f"   Dokumentation: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n🛑 Server wurde durch Benutzer gestoppt")
    except Exception as e:
        print(f"💥 Fehler beim Starten des Servers: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()