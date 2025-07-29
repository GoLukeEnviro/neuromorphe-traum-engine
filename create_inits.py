import os
from pathlib import Path

project_root = Path(__file__).parent

for dirpath, dirnames, filenames in os.walk(project_root):
    # Ignoriere __pycache__ Verzeichnisse
    if '__pycache__' in dirpath:
        continue
    
    # Erstelle __init__.py, wenn nicht vorhanden
    init_file = Path(dirpath) / '__init__.py'
    if not init_file.exists():
        print(f"Creating {init_file}")
        init_file.touch()

print("Finished creating __init__.py files.")