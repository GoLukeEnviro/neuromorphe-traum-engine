"""Regression tests for production CLAP defaults in legacy agent entry points."""

from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module_name", "model_attribute"),
    [
        ("ai_agents.minimal_preprocessor", "laion_clap"),
        ("ai_agents.search_engine_cli", "laion_clap"),
        ("ai_agents.prepare_dataset_sql", "_RealCLAP_Module"),
    ],
)
def test_real_clap_is_attempted_by_default(module_name, model_attribute):
    """An available real CLAP implementation must win without an opt-in flag."""
    probe = f"""
import importlib
import os
import sys
import types

os.environ.pop("USE_REAL_CLAP", None)

class FakeCLAPModule:
    pass

fake_package = types.ModuleType("laion_clap")
fake_package.CLAP_Module = FakeCLAPModule
sys.modules["laion_clap"] = fake_package

module = importlib.import_module({module_name!r})
selected = getattr(module, {model_attribute!r})
expected = fake_package if {model_attribute!r} == "laion_clap" else FakeCLAPModule
raise SystemExit(0 if selected is expected else 1)
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
