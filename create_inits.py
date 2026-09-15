import os
from pathlib import Path


PACKAGE_ROOT_NAMES = ("ai_agents", "backend", "frontend", "src", "test", "tests")
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "env",
    "node_modules",
    "venv",
}


def create_init_files(project_root: Path) -> list[Path]:
    """Create package markers only inside known Python source trees."""
    created_files: list[Path] = []

    for package_root_name in PACKAGE_ROOT_NAMES:
        package_root = project_root / package_root_name
        if not package_root.is_dir():
            continue

        for dirpath, dirnames, _filenames in os.walk(package_root):
            dirnames[:] = [
                name for name in dirnames if name not in EXCLUDED_DIRECTORY_NAMES
            ]
            init_file = Path(dirpath) / "__init__.py"
            if not init_file.exists():
                print(f"Creating {init_file}")
                init_file.touch()
                created_files.append(init_file)

    return created_files


if __name__ == "__main__":
    create_init_files(Path(__file__).resolve().parent)
    print("Finished creating __init__.py files.")
