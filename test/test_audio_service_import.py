"""Import-safety regression tests for optional audio dependencies."""

from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module_name", ["audio.service", "services.search", "search.service"])
def test_importing_service_module_does_not_import_clap(module_name):
    """Importing service modules must not initialize optional ML packages."""
    probe = f"""
import builtins
import importlib
import sys

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "laion_clap" or name.startswith("laion_clap."):
        raise AssertionError("laion_clap imported during module import")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.path.insert(0, "src")
importlib.import_module({module_name!r})
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
