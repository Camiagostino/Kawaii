from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


def find_project_root(start_path: Path | None = None, markers: Iterable[str] | None = None) -> Path:
    """Find the project root by walking upward until expected marker files are present."""

    search_markers = tuple(markers or ("README.md", "config", "scripts", "src"))
    current = (start_path or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in search_markers):
            return candidate

    raise RuntimeError(f"Could not find the project root from {current}")


def get_project_root() -> Path:
    return find_project_root()


def resolve_project_path(relative_path: str | Path, root: Path | None = None) -> Path:
    base = root or get_project_root()
    return base / Path(relative_path)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config at {path} must contain a mapping at the top level")
    return loaded


def load_project_config(config_name: str, root: Path | None = None) -> dict[str, Any]:
    project_root = root or get_project_root()
    return read_yaml_config(project_root / "config" / config_name)
