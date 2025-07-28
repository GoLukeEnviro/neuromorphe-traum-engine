#!/usr/bin/env python3
"""
Startup script for the Neuromorphic Dream Engine
Handles initialization, dependency checks, and server startup
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
import importlib.util

from ..core.config import settings


def check_dependencies():
    """
    Check if all required dependencies are installed
    """
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('demucs', 'Demucs'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('librosa', 'Librosa'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('soundfile', 'SoundFile'),
        ('transformers', 'Transformers'),
        ('laion_clap', 'LAION-CLAP')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True


def create_directories():
    """
    Create necessary directories if they don't exist
    """
    print("\n📁 Creating directories...")
    
    directories = [
        settings.UPLOAD_DIR,
        settings.PROCESSED_DIR,
        settings.EMBEDDINGS_DIR,
        settings.GENERATED_TRACKS_DIR,
        settings.get_logs_path()
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        else:
            print(f"📁 Exists: {directory}")


async def initialize_database():
    """
    Initialize the database and check connectivity
    """
    print("\n🗄️  Initializing database...")
    
    try:
        from database.service import DatabaseService
        
        db_service = DatabaseService()
        
        # Test database connectivity
        stats = await db_service.get_stem_statistics()
        print(f"✅ Database initialized - {stats['total_stems']} stems found")
        
        # Show statistics if stems exist
        if stats['total_stems'] > 0:
            print("📊 Current stem distribution:")
            for source_stat in stats['by_source']:
                print(f"   - {source_stat['source']}: {source_stat['count']} stems")
        
        db_service.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def check_gpu_availability():
    """
    Check if GPU is available for acceleration
    """
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} ({gpu_count} device(s))")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU available - using CPU (slower training/generation)")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False


def check_audio_files():
    """
    Check for available audio files for testing
    """
    print("\n🎵 Checking for audio files...")
    
    stereo_dir = Path(settings.STEREO_TRACKS_DIR)
    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.aiff"]
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(stereo_dir.glob(ext))
    
    if audio_files:
        print(f"✅ Found {len(audio_files)} audio files for separation:")
        for file in audio_files[:5]:  # Show first 5
            print(f"   - {file.name}")
        if len(audio_files) > 5:
            print(f"   ... and {len(audio_files) - 5} more")
    else:
        print("⚠️  No audio files found in stereo_tracks_for_analysis/")
        print("   Add .wav, .mp3, .flac, or .aiff files to test separation")
    
    return len(audio_files) > 0


def start_server():
    """
    Start the FastAPI server
    """
    print("\n🚀 Starting Neuromorphic Dream Engine server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\n🧠 Neuromorphic endpoints:")
    print("   POST /api/v1/neuromorphic/preprocess - Audio separation & analysis")
    print("   POST /api/v1/neuromorphic/train - VAE training")
    print("   POST /api/v1/neuromorphic/generate - Stem generation")
    print("\n🎵 Stem endpoints:")
    print("   GET  /api/v1/stems/ - List all stems")
    print("   POST /api/v1/stems/search/ - Semantic search")
    print("   POST /api/v1/stems/generate-track - Generate complete track")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Start uvicorn server from the main directory
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.main:app", 
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")


async def main():
    """
    Main startup sequence
    """
    print("🧠 Neuromorphic Dream Engine - Startup")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Startup aborted - missing dependencies")
        return
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Initialize database
    if not await initialize_database():
        print("\n❌ Startup aborted - database initialization failed")
        return
    
    # Step 4: Check GPU (optional)
    gpu_available = check_gpu_availability()
    
    # Step 5: Check audio files (optional)
    audio_files_available = check_audio_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Startup Summary:")
    print(f"   ✅ Dependencies: OK")
    print(f"   ✅ Directories: OK")
    print(f"   ✅ Database: OK")
    print(f"   {'✅' if gpu_available else '⚠️ '} GPU: {'Available' if gpu_available else 'CPU only'}")
    print(f"   {'✅' if audio_files_available else '⚠️ '} Audio files: {'Available' if audio_files_available else 'None found'}")
    
    if not audio_files_available:
        print("\n💡 Tip: Add audio files to stereo_tracks_for_analysis/ to test separation")
    
    print("\n🚀 System ready! Starting server...")
    
    # Step 6: Start server
    start_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()