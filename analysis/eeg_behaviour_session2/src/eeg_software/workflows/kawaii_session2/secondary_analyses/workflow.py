"""Reproducible workflow for Kawaii Session 2 secondary analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from ..config import load_session2_eeg_config
from .arousal_erp import extract_existing_arousal_erp, plot_existing_arousal_erp_coefficient
from .config import (
    SECONDARY_CONFIG_NAME,
    load_secondary_analysis_config,
    resolve_path,
    sha256_file,
    verify_figure_style,
)
from .recognition_behavior import (
    compute_signal_detection_metrics,
    dprime_inference,
    load_source_verified_behavioural_trials,
    plot_recognition_rates,
    signal_detection_group_summary,
)
from .rating_memory import compute_rating_memory_family, plot_rating_memory_forest
from .recognition_erp import (
    compute_erp_family,
    load_recognition_measurements,
    plot_erp_family_forest,
    validate_secondary_against_gate5,
)


STAGE_BEHAVIOUR = "recognition-behavior"
STAGE_OUTCOME_ERP = "recognition-outcome-erp"
STAGE_TRUE_FALSE_ERP = "true-vs-false-erp"
STAGE_RATING_MEMORY = "rating-memory"
STAGE_AROUSAL_ERP = "arousal-erp"
SUPPORTED_STAGES = (
    STAGE_BEHAVIOUR,
    STAGE_OUTCOME_ERP,
    STAGE_TRUE_FALSE_ERP,
    STAGE_RATING_MEMORY,
    STAGE_AROUSAL_ERP,
)


@dataclass(frozen=True)
class SecondaryAnalysisRunResult:
    output_files: dict[str, Path]
    manifest_path: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _write_json(value: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def run_secondary_analyses(
    *,
    root: str | Path,
    core_config_path: str | Path = "config/session2_eeg.yaml",
    secondary_config_path: str | Path = f"config/{SECONDARY_CONFIG_NAME}",
    stages: Sequence[str] = SUPPORTED_STAGES,
) -> SecondaryAnalysisRunResult:
    """Run tracked secondary analyses from validated downstream inputs only."""

    project_root = Path(root).resolve()
    core_config = load_session2_eeg_config(project_root, core_config_path)
    secondary_config = load_secondary_analysis_config(project_root, secondary_config_path)
    stage_set = {str(v).lower() for v in stages}
    unknown = sorted(stage_set - set(SUPPORTED_STAGES))
    if unknown:
        raise ValueError(f"Unsupported secondary-analysis stage(s): {unknown}")

    style_path = verify_figure_style(project_root, secondary_config)
    output_root = resolve_path(project_root, secondary_config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    input_paths: list[Path] = [
        resolve_path(project_root, core_config_path),
        resolve_path(project_root, secondary_config_path),
        style_path,
    ]

    trials: pd.DataFrame | None = None
    source_validation: pd.DataFrame | None = None
    if stage_set & {STAGE_BEHAVIOUR, STAGE_RATING_MEMORY}:
        trials, source_validation, behaviour_inputs = load_source_verified_behavioural_trials(
            project_root, core_config, secondary_config
        )
        input_paths.extend(behaviour_inputs)

    if STAGE_BEHAVIOUR in stage_set:
        metrics = compute_signal_detection_metrics(trials)
        summary = signal_detection_group_summary(metrics)
        inference = dprime_inference(metrics, core_config)
        outputs["recognition_behavior_source_validation"] = _write_csv(
            source_validation,
            output_root / "recognition_behavior" / "source_validation.csv",
        )
        outputs["recognition_behavior_participants"] = _write_csv(
            metrics,
            output_root / "recognition_behavior" / "participant_metrics.csv",
        )
        outputs["recognition_behavior_group_summary"] = _write_csv(
            summary,
            output_root / "recognition_behavior" / "group_summary.csv",
        )
        outputs["recognition_behavior_dprime"] = _write_json(
            inference,
            output_root / "recognition_behavior" / "dprime_inference.json",
        )
        outputs["recognition_behavior_figure"] = plot_recognition_rates(
            metrics,
            output_root / "figures" / "behavioural_recognition_rates.png",
            root=project_root,
        )


    if STAGE_AROUSAL_ERP in stage_set:
        table, arousal_inputs = extract_existing_arousal_erp(project_root, secondary_config)
        input_paths.extend(arousal_inputs)
        outputs["arousal_erp_coefficient"] = _write_csv(
            table, output_root / "arousal_erp" / "existing_arousal_coefficient.csv"
        )
        outputs["arousal_erp_figure"] = plot_existing_arousal_erp_coefficient(
            table, output_root / "figures" / "arousal_erp_coefficient.png", root=project_root
        )

    if STAGE_RATING_MEMORY in stage_set:
        family = secondary_config["families"]["rating_memory_association"]
        participant_means, table = compute_rating_memory_family(
            trials,
            family,
            core_config,
        )
        outputs["rating_memory_source_validation"] = _write_csv(
            source_validation,
            output_root / "rating_memory" / "source_validation.csv",
        )
        outputs["rating_memory_participant_means"] = _write_csv(
            participant_means,
            output_root / "rating_memory" / "participant_condition_means.csv",
        )
        outputs["rating_memory_statistics"] = _write_csv(
            table,
            output_root / "rating_memory" / "rating_memory_associations.csv",
        )
        outputs["rating_memory_figure"] = plot_rating_memory_forest(
            table,
            output_root / "figures" / "rating_memory_associations_forest.png",
            root=project_root,
        )

    measurements: pd.DataFrame | None = None
    if stage_set & {STAGE_OUTCOME_ERP, STAGE_TRUE_FALSE_ERP}:
        measurements = load_recognition_measurements(project_root, secondary_config)
        input_paths.append(
            resolve_path(project_root, secondary_config["inputs"]["recognition_measurements"])
        )

    if STAGE_OUTCOME_ERP in stage_set:
        family = secondary_config["families"]["recognition_secondary_memory_outcome"]
        table = compute_erp_family(
            measurements,
            family,
            core_config,
            secondary_config["recognition_cells"],
        )
        validation = validate_secondary_against_gate5(project_root, secondary_config, table)
        input_paths.append(
            resolve_path(project_root, secondary_config["inputs"]["recognition_statistics"])
        )
        outputs["recognition_outcome_erp_statistics"] = _write_csv(
            table,
            output_root / "recognition_erp" / "recognition_outcome_contrasts.csv",
        )
        outputs["recognition_outcome_erp_gate5_validation"] = _write_csv(
            validation,
            output_root / "recognition_erp" / "gate5_reproduction_validation.csv",
        )
        outputs["recognition_outcome_erp_figure"] = plot_erp_family_forest(
            table,
            output_root / "figures" / "recognition_outcome_erp_forest.png",
            root=project_root,
            title="Secondary recognition ERP contrasts",
        )

    if STAGE_TRUE_FALSE_ERP in stage_set:
        family = secondary_config["families"]["recognition_true_vs_false_exploratory"]
        table = compute_erp_family(
            measurements,
            family,
            core_config,
            secondary_config["recognition_cells"],
        )
        outputs["true_vs_false_erp_statistics"] = _write_csv(
            table,
            output_root / "recognition_erp" / "hit_vs_false_alarm.csv",
        )
        outputs["true_vs_false_erp_figure"] = plot_erp_family_forest(
            table,
            output_root / "figures" / "true_vs_false_recognition_erp_forest.png",
            root=project_root,
            title="True vs false recognition ERP",
        )

    unique_inputs: list[Path] = []
    seen = set()
    for path in input_paths:
        resolved = Path(path).resolve()
        if resolved.exists() and resolved not in seen:
            unique_inputs.append(resolved)
            seen.add(resolved)

    manifest_path = output_root / "manifest.json"
    manifest = {
        "schema_version": "kawaii_session2_secondary_analyses_manifest_v1",
        "analysis_scope": "SECONDARY_AND_EXPLORATORY",
        "primary_analysis_modified": False,
        "stages": sorted(stage_set),
        "inputs": [
            {
                "path": str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in unique_inputs
        ],
        "outputs": {
            label: {
                "path": str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(outputs.items())
        },
        "figure_style": {
            "path": secondary_config["figure_style_path"],
            "sha256": sha256_file(style_path),
        },
    }
    _write_json(manifest, manifest_path)
    return SecondaryAnalysisRunResult(output_files=outputs, manifest_path=manifest_path)
