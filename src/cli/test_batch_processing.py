import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import asyncio
import logging
import os
from pathlib import Path

from .create_test_audio import create_test_audio_files
from src.services.preprocessor import PreprocessorService
from src.core.config import settings

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_batch_processing_test():
    """
    Führt einen Test der Batch-Verarbeitung von Audio-Dateien durch.
    """
    logger.info("Starte Test der Batch-Verarbeitung...")

    # 1. Test-Audiodateien generieren
    logger.info("Generiere Test-Audiodateien...")
    test_audio_dir = "raw_construction_kits"
    # Sicherstellen, dass das Verzeichnis existiert und leer ist
    if Path(test_audio_dir).exists():
        for f in Path(test_audio_dir).iterdir():
            if f.is_file():
                os.remove(f)
    Path(test_audio_dir).mkdir(exist_ok=True)
    
    created_files = create_test_audio_files()
    if not created_files:
        logger.error("Keine Test-Audiodateien erstellt. Breche Test ab.")
        return

    # 2. PreprocessorService initialisieren
    logger.info("Initialisiere PreprocessorService...")
    preprocessor = PreprocessorService(config=settings)

    # 3. Batch-Verarbeitung starten
    logger.info(f"Starte Batch-Verarbeitung für Verzeichnis: {test_audio_dir}")
    results = await preprocessor.batch_process_directory(directory_path=test_audio_dir)

    # 4. Ergebnisse protokollieren
    logger.info("\n--- Batch-Verarbeitung Ergebnisse ---")
    logger.info(f"Gesamtzahl der Dateien: {results['total_files']}")
    logger.info(f"Erfolgreich verarbeitet: {results['processed']}")
    logger.info(f"Übersprungen (bereits vorhanden): {results['skipped']}")
    logger.info(f"In Quarantäne (Fehler): {results['quarantined']}")

    if results['errors']:
        logger.warning("Fehlerdetails:")
        for error in results['errors']:
            logger.warning(f"  Datei: {error['file']}, Fehler: {error['error']}")
    
    if results['processed_stems']:
        logger.info("Verarbeitete Stems:")
        for stem in results['processed_stems']:
            logger.info(f"  ID: {stem['id']}, Name: {stem['name']}, Kategorie: {stem['category']}")

    logger.info("Test der Batch-Verarbeitung abgeschlossen.")

if __name__ == "__main__":
    # Führe den Async-Test aus
    asyncio.run(run_batch_processing_test())