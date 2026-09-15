"""Regression tests for the package-initializer maintenance script."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "create_inits.py"


@pytest.mark.parametrize(
    "excluded_directory",
    [Path(".git/objects/aa"), Path("venv/Lib/site-packages"), Path("src/node_modules/pkg")],
)
def test_script_skips_non_project_directories(tmp_path, excluded_directory):
    """The script must never turn metadata or dependency trees into packages."""
    project_root = tmp_path / "project"
    script_copy = project_root / "create_inits.py"
    source_directory = project_root / "src"
    excluded_path = project_root / excluded_directory

    source_directory.mkdir(parents=True)
    excluded_path.mkdir(parents=True)
    shutil.copy2(SCRIPT_PATH, script_copy)

    subprocess.run(
        [sys.executable, str(script_copy)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (source_directory / "__init__.py").exists()
    assert not list(excluded_path.rglob("__init__.py"))
