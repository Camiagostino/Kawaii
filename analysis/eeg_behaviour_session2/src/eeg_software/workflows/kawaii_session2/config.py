from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from eeg_software.project import load_project_config, read_yaml_config
from eeg_software.project import get_project_root

CONFIG_NAME = "session2_eeg.yaml"


def load_session2_eeg_config(root: Path | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the Session 2 EEG scaffold configuration.

    Parameters
    ----------
    root:
        Optional project root used when loading the default config name.
    config_path:
        Optional explicit YAML path. Relative paths are resolved from the
        project root so entry scripts behave consistently across working dirs.
    """

    project_root = root or get_project_root()
    if config_path is None:
        return load_project_config(CONFIG_NAME, root=project_root)

    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    return read_yaml_config(path)


def dump_resolved_config(config: Mapping[str, Any]) -> str:
    """Return a stable YAML representation for scaffold entry-script output."""

    return yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)


def participant_defaults(config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return included/excluded participant defaults from the EEG config."""

    participants = config.get("participants", {})
    if not isinstance(participants, Mapping):
        return {"included": [], "excluded": []}
    defaults = participants.get("defaults", {})
    if not isinstance(defaults, Mapping):
        return {"included": [], "excluded": []}
    included = defaults.get("included", []) or []
    excluded = defaults.get("excluded", []) or []
    return {"included": list(included), "excluded": list(excluded)}
