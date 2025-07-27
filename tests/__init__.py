"""Test-Suite für die Neuromorphe Traum-Engine v2.0

Diese Test-Suite umfasst:
- Unit Tests für alle Services
- Integration Tests für API-Endpunkte
- Performance Tests
- Audio-Verarbeitungstests
"""

import sys
from pathlib import Path

# Projekt-Root zum Python-Path hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))