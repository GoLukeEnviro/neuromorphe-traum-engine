#!/usr/bin/env python3
"""Startup-Script für die Neuromorphe Traum-Engine v2.0

Dieses Script startet die Anwendung mit verschiedenen Modi:
- Development Server
- Production Server
- Database Setup
- System Tests
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional

# Projekt-Root zum Python-Path hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import settings
from core.logging import setup_logging, get_logger
from database.database import get_database_manager, init_database
from services.preprocessor import PreprocessorService
from services.neuro_analyzer import NeuroAnalyzer


def setup_argument_parser() -> argparse.ArgumentParser:
    """Command-line Argument Parser einrichten"""
    parser = argparse.ArgumentParser(
        description="Neuromorphe Traum-Engine v2.0 - Generatives Musiksystem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python run.py --mode dev                    # Development Server starten
  python run.py --mode prod --port 8080       # Production Server auf Port 8080
  python run.py --mode setup                  # Datenbank einrichten
  python run.py --mode test                   # System-Tests ausführen
  python run.py --mode preprocess --input ./audio  # Audio-Preprocessing
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["dev", "prod", "setup", "test", "preprocess", "health"],
        default="dev",
        help="Ausführungsmodus (default: dev)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server Host (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server Port (default: 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Anzahl Worker-Prozesse für Production (default: 1)"
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        help="Input-Verzeichnis für Preprocessing"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output-Verzeichnis für generierte Tracks"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log-Level (default: INFO)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Pfad zu alternativer Konfigurationsdatei"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Erzwinge Ausführung (z.B. Datenbank-Reset)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose Output"
    )
    
    return parser


async def setup_database(force: bool = False):
    """Datenbank einrichten"""
    logger = get_logger("setup")
    
    try:
        logger.info("Setting up database...")
        
        db_manager = get_database_manager()
        
        # Verbindungstest
        if not await db_manager.test_connection():
            logger.error("Database connection failed")
            return False
        
        # Tabellen erstellen (falls nicht vorhanden)
        if force:
            logger.warning("Force mode: Dropping existing tables")
            await db_manager.drop_tables()
        
        await db_manager.create_tables()
        
        # Basis-Konfiguration einfügen
        from database.crud import ConfigurationCRUD
from schemas.schemas import ConfigurationSettingCreate
        
        async with db_manager.get_async_session() as session:
            
            # Standard-Konfigurationen
            default_configs = [
                {"key": "system.version", "value": "2.0.0", "description": "System Version"},
                {"key": "audio.sample_rate", "value": "44100", "description": "Standard Sample Rate"},
                {"key": "audio.bit_depth", "value": "16", "description": "Standard Bit Depth"},
                {"key": "generation.default_duration", "value": "180", "description": "Standard Track-Länge in Sekunden"},
                {"key": "clap.model_name", "value": "laion/larger_clap_music", "description": "CLAP Model Name"}
            ]
            
            for config_data in default_configs:
                # Prüfen ob bereits vorhanden
                existing = await ConfigurationCRUD.get_setting(session, "system", config_data["key"])
                if not existing:
                    await ConfigurationCRUD.set_setting(session, category="system", key=config_data["key"], value=config_data["value"], data_type="string", description=config_data["description"])
                    logger.info(f"Created config: {config_data['key']}")
        
        logger.info("Database setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}", exc_info=True)
        return False


async def run_system_tests():
    """System-Tests ausführen"""
    logger = get_logger("test")
    
    try:
        logger.info("Running system tests...")
        
        # 1. Datenbank-Test
        logger.info("Testing database connection...")
        db_manager = get_database_manager()
        if not await db_manager.test_connection():
            logger.error("Database test failed")
            return False
        
        # 2. Verzeichnis-Test
        logger.info("Testing directory structure...")
        
        required_dirs = [
            settings.UPLOAD_DIR,
            settings.PROCESSED_DIR,
            settings.GENERATED_TRACKS_DIR,
            settings.EMBEDDINGS_DIR,
            settings.get_logs_path()
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.warning(f"Creating missing directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # 3. Service-Tests
        logger.info("Testing services...")
        
        # NeuroAnalyzer Test
        try:
            neuro_analyzer = NeuroAnalyzer()
            logger.info("NeuroAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"NeuroAnalyzer test failed: {e}")
            return False
        
        # PreprocessorService Test
        try:
            preprocessor = PreprocessorService()
            logger.info("PreprocessorService initialized successfully")
        except Exception as e:
            logger.error(f"PreprocessorService test failed: {e}")
            return False
        
        # 4. Health Check
        health_status = await db_manager.health_check()
        if health_status['status'] not in ['healthy', 'degraded']:
            logger.error(f"System health check failed: {health_status}")
            return False
        
        logger.info("All system tests passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"System tests failed: {e}", exc_info=True)
        return False


async def run_preprocessing(input_dir: Path, force: bool = False):
    """Audio-Preprocessing ausführen"""
    logger = get_logger("preprocess")
    
    try:
        if not input_dir or not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return False
        
        logger.info(f"Starting preprocessing for: {input_dir}")
        
        db_manager = get_database_manager()
        # PreprocessorService instanziiert seinen eigenen DatabaseService
        preprocessor = PreprocessorService()
            
        # Audio-Dateien finden
        audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.m4a'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(input_dir.glob(f"**/*{ext}"))
            audio_files.extend(input_dir.glob(f"**/*{ext.upper()}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in: {input_dir}")
            return True
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Preprocessing durchführen
        success_count = 0
        error_count = 0
        
        for audio_file in audio_files:
            try:
                logger.info(f"Processing: {audio_file.name}")
                result = await preprocessor.process_audio_file(str(audio_file))
                
                if result['success']:
                    success_count += 1
                    logger.info(f"Successfully processed: {audio_file.name}")
                else:
                    error_count += 1
                    logger.error(f"Failed to process: {audio_file.name} - {result.get('error')}")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {audio_file.name}: {e}", exc_info=True)
        
        logger.info(f"Preprocessing completed: {success_count} success, {error_count} errors")
        return error_count == 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return False


async def show_health_status():
    """System-Health-Status anzeigen"""
    logger = get_logger("health")
    
    try:
        logger.info("Checking system health...")
        
        # Datenbank-Health
        db_manager = get_database_manager()
        db_health = await db_manager.health_check()
        
        print("\n=== SYSTEM HEALTH STATUS ===")
        print(f"Database Status: {db_health['status']}")
        print(f"Connection Test: {'✓' if db_health['connection_test'] else '✗'}")
        print(f"Response Time: {db_health.get('response_time_ms', 'N/A'):.2f}ms")
        
        if db_health.get('pool_status'):
            pool = db_health['pool_status']
            print(f"Pool Size: {pool.get('size', 'N/A')}")
            print(f"Checked Out: {pool.get('checked_out', 'N/A')}")
        
        # Datenbank-Info
        db_info = await db_manager.get_database_info()
        if 'database_size_mb' in db_info:
            print(f"Database Size: {db_info['database_size_mb']:.2f} MB")
        
        # Tabellen-Größen
        table_sizes = await db_manager.get_table_sizes()
        if table_sizes:
            print("\nTable Sizes:")
            for table, size in table_sizes.items():
                print(f"  {table}: {size} rows")
        
        # Verzeichnis-Status
        print("\nDirectory Status:")
        dirs_to_check = {
            "Upload Dir": settings.UPLOAD_DIR,
            "Processed Dir": settings.PROCESSED_DIR,
            "Embeddings Dir": settings.EMBEDDINGS_DIR,
            "Generated Tracks Dir": settings.GENERATED_TRACKS_DIR,
            "Logs Dir": settings.get_logs_path()
        }
        
        for name, path in dirs_to_check.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {name}: {status} {path}")
        
        print("\n" + "="*30)
        
        return db_health['status'] in ['healthy', 'degraded']
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return False


def run_development_server(host: str, port: int, log_level: str):
    """Development Server starten"""
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level.lower(),
        access_log=True
    )


def run_production_server(host: str, port: int, workers: int, log_level: str):
    """Production Server starten"""
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level.lower(),
        access_log=True,
        loop="uvloop" if sys.platform != "win32" else "asyncio"
    )


async def main():
    """Hauptfunktion"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Logging einrichten
    setup_logging(level=args.log_level, verbose=args.verbose)
    logger = get_logger("main")
    
    logger.info(f"Starting Neuromorphe Traum-Engine v2.0 in {args.mode} mode")
    
    try:
        if args.mode == "setup":
            success = await setup_database(force=args.force)
            sys.exit(0 if success else 1)
            
        elif args.mode == "test":
            success = await run_system_tests()
            sys.exit(0 if success else 1)
            
        elif args.mode == "preprocess":
            if not args.input:
                logger.error("--input directory required for preprocessing mode")
                sys.exit(1)
            success = await run_preprocessing(args.input, force=args.force)
            sys.exit(0 if success else 1)
            
        elif args.mode == "health":
            success = await show_health_status()
            sys.exit(0 if success else 1)
            
        elif args.mode == "dev":
            logger.info(f"Starting development server on {args.host}:{args.port}")
            run_development_server(args.host, args.port, args.log_level)
            
        elif args.mode == "prod":
            logger.info(f"Starting production server on {args.host}:{args.port} with {args.workers} workers")
            run_production_server(args.host, args.port, args.workers, args.log_level)
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Windows-spezifische Asyncio-Policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())