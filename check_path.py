import sys
import os

print("Current working directory:", os.getcwd())
print("PYTHONPATH environment variable:", os.environ.get('PYTHONPATH'))
print("sys.path before import:")
for p in sys.path:
    print(f"  {p}")

try:
    import src.schemas.websocket
    print("Successfully imported src.schemas.websocket")
except ImportError as e:
    print(f"Failed to import src.schemas.websocket: {e}")

print("sys.path after import attempt:")
for p in sys.path:
    print(f"  {p}")