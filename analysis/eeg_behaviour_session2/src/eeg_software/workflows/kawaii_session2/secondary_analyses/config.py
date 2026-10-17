"""Configuration and provenance helpers for Kawaii Session 2 secondary analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import hashlib

import yaml


SECONDARY_CONFIG_NAME = "kawaii_session2_secondary_analyses.yaml"
SECONDARY_SCHEMA_VERSION = "kawaii_session2_secondary_analyses_v1"
APPROVED_FIGURE_STYLE_PATH = "config/kawaii_session2_figure_style.yaml"


def load_secondary_analysis_config(
    root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    project_root = Path(root or ".").resolve()
    path = Path(config_path or f"config/{SECONDARY_CONFIG_NAME}")
    if not path.is_absolute():
        path = project_root / path
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Secondary-analysis config must be a mapping: {path}")
    if value.get("schema_version") != SECONDARY_SCHEMA_VERSION:
        raise ValueError(
            f"Secondary-analysis config schema_version must be {SECONDARY_SCHEMA_VERSION!r}; "
            f"got {value.get('schema_version')!r}."
        )
    _validate_secondary_config(value)
    return dict(value)


def _validate_secondary_config(config: Mapping[str, Any]) -> None:
    participants = [str(v) for v in config.get("participants", []) or []]
    if not participants or len(set(participants)) != len(participants):
        raise ValueError("Secondary-analysis participants must be a non-empty unique list.")

    cells = list(config.get("recognition_cells", []) or [])
    if len(cells) != 4:
        raise ValueError("Secondary analyses must define exactly four recognition cells.")
    seen = set()
    for cell in cells:
        key = (str(cell.get("roi", "")), str(cell.get("time_window", "")))
        if not all(key) or key in seen:
            raise ValueError(f"Invalid/duplicate recognition cell: {key!r}")
        seen.add(key)

    figure_style_path = str(config.get("figure_style_path", "")).strip()
    if figure_style_path != APPROVED_FIGURE_STYLE_PATH:
        raise ValueError(
            f"Secondary analyses must reuse {APPROVED_FIGURE_STYLE_PATH}; "
            "do not duplicate or replace the approved palette/layout authority."
        )
    expected_style_sha = str(config.get("figure_style_sha256", "")).strip()
    if len(expected_style_sha) != 64:
        raise ValueError("Secondary-analysis config must pin figure_style_sha256.")

    family = dict((config.get("families", {}) or {}).get("rating_memory_association", {}) or {})
    if int(family.get("family_size", -1)) != 4:
        raise ValueError("rating_memory_association must define exactly four tests.")
    if str(family.get("correction", "")) != "holm":
        raise ValueError("rating_memory_association must use Holm correction across its four tests.")
    comparisons = list(family.get("comparisons", []) or [])
    if family.get("implementation_status") == "IMPLEMENTED":
        if len(comparisons) != 4:
            raise ValueError("Implemented rating_memory_association must define four comparisons.")
        required = {"name", "label", "rating", "rating_column", "condition_a", "condition_b"}
        for comparison in comparisons:
            missing = sorted(required - set(comparison))
            if missing:
                raise ValueError(f"Rating-memory comparison missing fields {missing}: {comparison!r}")
        observed = {
            (
                str(c["rating"]),
                str(c["rating_column"]),
                str(c["condition_a"]),
                str(c["condition_b"]),
            )
            for c in comparisons
        }
        expected = {
            ("cuteness", "kawaii_rating", "hit", "miss"),
            ("cuteness", "kawaii_rating", "false_alarm", "correct_rejection"),
            ("arousal", "arousal_rating", "hit", "miss"),
            ("arousal", "arousal_rating", "false_alarm", "correct_rejection"),
        }
        if observed != expected:
            raise ValueError(
                "rating_memory_association must contain exactly the four frozen cuteness/arousal × recognition-outcome contrasts."
            )

    arousal = dict(config.get("existing_arousal_coefficient", {}) or {})
    if arousal.get("implementation_status") == "IMPLEMENTED":
        if bool(arousal.get("refit_allowed", True)):
            raise ValueError("existing_arousal_coefficient must keep refit_allowed=false.")
        if str(arousal.get("coefficient", "")) != "arousal_centered":
            raise ValueError("existing_arousal_coefficient must extract arousal_centered.")
        if str(arousal.get("model_id", "")) != "cuteness_primary_continuous_500_800_centroparietal":
            raise ValueError("existing_arousal_coefficient must reference the accepted continuous cuteness model.")
        inputs = dict(config.get("inputs", {}) or {})
        for key in (
            "cuteness_mixed_model_coefficients",
            "cuteness_mixed_model_diagnostics",
            "cuteness_mixed_model_specification",
        ):
            if not str(inputs.get(key, "")).strip():
                raise ValueError(f"Implemented arousal coefficient extraction requires inputs.{key}.")


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_figure_style(root: Path, config: Mapping[str, Any]) -> Path:
    style_path = resolve_path(root, str(config["figure_style_path"]))
    observed = sha256_file(style_path)
    expected = str(config["figure_style_sha256"])
    if observed != expected:
        raise ValueError(
            f"Approved Kawaii figure-style identity mismatch: expected {expected}, observed {observed}."
        )
    return style_path
