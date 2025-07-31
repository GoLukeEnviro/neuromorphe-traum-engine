#!/usr/bin/env python3
"""
Systemtest für die Neuromorphe Traum-Engine.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_database():
    """Teste Datenbankverbindung."""
    try:
        from database.database import DatabaseManager
        from core.config import settings
        
        print(f"✓ Database URL: {settings.DATABASE_URL}")
        dm = DatabaseManager()
        print("✓ Database Manager initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_fastapi():
    """Teste FastAPI-Anwendung."""
    try:
        from main import app
        print("✓ FastAPI app loaded successfully")
        
        print("\nAvailable routes:")
        for route in app.routes:
            methods = getattr(route, 'methods', {'N/A'})
            print(f"  {methods} {route.path}")
        return True
    except Exception as e:
        print(f"✗ FastAPI test failed: {e}")
        return False

def test_services():
    """Teste Service-Layer."""
    try:
        from services.preprocessor import PreprocessorService
        print("✓ PreprocessorService imported")
        
        from services.arranger import ArrangerService
        print("✓ ArrangerService imported")
        
        from services.renderer import RendererService
        print("✓ RendererService imported")
        
        return True
    except Exception as e:
        print(f"✗ Services test failed: {e}")
        return False

def test_schemas():
    """Teste Schema-Definitionen."""
    try:
        from schemas.stem import StemCreate, StemResponse
        print("✓ Stem schemas imported")
        
        from schemas.api import SearchRequest, APIResponse
        print("✓ API schemas imported")
        
        return True
    except Exception as e:
        print(f"✗ Schemas test failed: {e}")
        return False

def main():
    """Hauptfunktion für Systemtest."""
    print("🧠 Neuromorphe Traum-Engine v2.0 - Systemtest\n")
    
    tests = [
        ("Database", test_database),
        ("FastAPI", test_fastapi),
        ("Services", test_services),
        ("Schemas", test_schemas),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- Testing {name} ---")
        result = test_func()
        results.append((name, result))
    
    print("\n" + "="*50)
    print("SYSTEMTEST ERGEBNISSE:")
    print("="*50)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:15} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nGesamt: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("\n🎉 Alle Systemtests erfolgreich!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} Tests fehlgeschlagen")
        return 1

if __name__ == '__main__':
    sys.exit(main())