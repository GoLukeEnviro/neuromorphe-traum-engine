#!/usr/bin/env python3
"""
Skript zur Korrektur aller relativen Imports im src-Verzeichnis.
"""

import os
import re
from pathlib import Path

def fix_relative_imports(file_path: Path):
    """Korrigiert relative Imports in einer Python-Datei."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Patterns für relative Imports
        patterns = [
            # from ..module import something -> from module import something
            (r'from \.\.(\w+)', r'from \1'),
            # from ...module import something -> from module import something  
            (r'from \.\.\.([\w\.]+)', r'from \1'),
            # from ....module import something -> from module import something
            (r'from \.\.\.\.(\w+)', r'from \1'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Spezielle Korrekturen
        content = content.replace('from .database', 'from database')
        content = content.replace('from .core', 'from core')
        content = content.replace('from .schemas', 'from schemas')
        content = content.replace('from .services', 'from services')
        content = content.replace('from .api', 'from api')
        content = content.replace('from .search', 'from search')
        content = content.replace('from .audio', 'from audio')
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Korrigiert: {file_path}")
            return True
        else:
            print(f"- Keine Änderungen: {file_path}")
            return False
            
    except Exception as e:
        print(f"✗ Fehler bei {file_path}: {e}")
        return False

def main():
    """Hauptfunktion."""
    src_dir = Path('src')
    
    if not src_dir.exists():
        print("Fehler: src-Verzeichnis nicht gefunden!")
        return
    
    python_files = list(src_dir.rglob('*.py'))
    print(f"Gefunden: {len(python_files)} Python-Dateien")
    
    fixed_count = 0
    for file_path in python_files:
        if fix_relative_imports(file_path):
            fixed_count += 1
    
    print(f"\nAbgeschlossen: {fixed_count} Dateien korrigiert")

if __name__ == '__main__':
    main()