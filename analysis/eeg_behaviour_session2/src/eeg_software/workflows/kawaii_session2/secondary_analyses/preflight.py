"""Read-only checks for the inputs selected by a Kawaii secondary-analysis run.

This preflight is intentionally not a substitute for the validated publication
lineage or the checks performed by the actual scientific workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..config import load_session2_eeg_config
from .config import (
    SECONDARY_CONFIG_NAME,
    load_secondary_analysis_config,
    resolve_path,
    verify_figure_style,
)
from .workflow import (
    STAGE_AROUSAL_ERP,
    STAGE_BEHAVIOUR,
    STAGE_OUTCOME_ERP,
    STAGE_RATING_MEMORY,
    STAGE_TRUE_FALSE_ERP,
    SUPPORTED_STAGES,
)


def check_secondary_inputs(
    *,
    root: str | Path,
    core_config_path: str | Path = "config/session2_eeg.yaml",
    secondary_config_path: str | Path = f"config/{SECONDARY_CONFIG_NAME}",
    stages: Sequence[str] = SUPPORTED_STAGES,
) -> dict[str, Any]:
    """Inspect and source-validate the selected inputs, creating no files."""

    project_root = Path(root).resolve()
    stage_set = {str(v).lower() for v in stages}
    unknown = sorted(stage_set - set(SUPPORTED_STAGES))
    if unknown or not stage_set:
        raise ValueError(f"Unknown or empty secondary-analysis stage selection: {unknown!r}.")

    core_config = load_session2_eeg_config(project_root, core_config_path)
    config = load_secondary_analysis_config(project_root, secondary_config_path)
    file_paths: set[Path] = {
        resolve_path(project_root, core_config_path).resolve(),
        resolve_path(project_root, secondary_config_path).resolve(),
        verify_figure_style(project_root, config).resolve(),
    }
    warnings: list[str] = []

    if stage_set & {STAGE_BEHAVIOUR, STAGE_RATING_MEMORY}:
        # This checks original source-log hashes, cohort counts and eligibility,
        # independently of rejected EEG epochs; it does not write results.
        from .recognition_behavior import load_source_verified_behavioural_trials

        _, _, inputs = load_source_verified_behavioural_trials(project_root, core_config, config)
        file_paths.update(Path(path).resolve() for path in inputs)

    if stage_set & {STAGE_OUTCOME_ERP, STAGE_TRUE_FALSE_ERP}:
        from .recognition_erp import load_recognition_measurements

        load_recognition_measurements(project_root, config)
        file_paths.add(resolve_path(project_root, config["inputs"]["recognition_measurements"]).resolve())
        if STAGE_OUTCOME_ERP in stage_set:
            # Required for the existing gate-5 numerical-reproduction check.
            file_paths.add(resolve_path(project_root, config["inputs"]["recognition_statistics"]).resolve())

    if STAGE_AROUSAL_ERP in stage_set:
        from .arousal_erp import extract_existing_arousal_erp

        _, inputs = extract_existing_arousal_erp(project_root, config)
        file_paths.update(Path(path).resolve() for path in inputs)

    for path in sorted(file_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Selected analysis input is missing: {path}")

    # The tracked config refers to a historical analysis/group path. Warn
    # rather than silently remap to another generation of EEG results.
    if stage_set & {STAGE_AROUSAL_ERP, STAGE_OUTCOME_ERP, STAGE_TRUE_FALSE_ERP}:
        selection = config.get("inputs", {})
        selected = [
            str(selection[key]).replace("\\", "/")
            for key in (
                "recognition_measurements",
                "recognition_statistics",
                "cuteness_mixed_model_coefficients",
            )
            if key in selection
        ]
        if any("/analysis/group/" in f"/{value}" for value in selected):
            warnings.append(
                "Configuration refers to workspace/.../analysis/group/. "
                "Confirm these files belong to the intended EEG publication "
                "before running or reporting results. No input was remapped."
            )

    return {
        "stages": sorted(stage_set),
        "verified_file_count": len(file_paths),
        "warnings": warnings,
    }
