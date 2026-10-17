"""Extract the existing arousal coefficient from the accepted Kawaii ERP mixed model.

This module never fits or refits a model. It reads the accepted continuous
cuteness-model outputs and exposes the already-estimated arousal adjustment
coefficient as a secondary ERP result together with its original diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..figure_style import load_figure_style, style_axes
from .config import resolve_path


def _single_row(table: pd.DataFrame, mask: pd.Series, *, label: str) -> pd.Series:
    selected = table.loc[mask].copy()
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one {label} row, found {len(selected)}.")
    return selected.iloc[0]


def extract_existing_arousal_erp(
    root: str | Path,
    secondary_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[Path]]:
    """Extract the existing arousal coefficient and accepted model diagnostics."""

    project_root = Path(root).resolve()
    spec = dict(secondary_config.get("existing_arousal_coefficient", {}) or {})
    if bool(spec.get("refit_allowed", True)):
        raise ValueError("Existing arousal ERP extraction must not allow model refitting.")

    model_id = str(spec.get("model_id", "")).strip()
    term = str(spec.get("coefficient", "")).strip()
    if not model_id or not term:
        raise ValueError("existing_arousal_coefficient must define model_id and coefficient.")

    inputs = dict(secondary_config.get("inputs", {}) or {})
    required_input_keys = [
        "cuteness_mixed_model_coefficients",
        "cuteness_mixed_model_diagnostics",
        "cuteness_mixed_model_specification",
    ]
    paths: dict[str, Path] = {}
    for key in required_input_keys:
        value = str(inputs.get(key, "")).strip()
        if not value:
            raise ValueError(f"Secondary-analysis config is missing required input path: {key}")
        path = resolve_path(project_root, value)
        if not path.exists():
            raise FileNotFoundError(f"Required accepted mixed-model artifact does not exist: {path}")
        paths[key] = path

    coefficients = pd.read_csv(paths["cuteness_mixed_model_coefficients"])
    diagnostics = pd.read_csv(paths["cuteness_mixed_model_diagnostics"])
    specification = pd.read_csv(paths["cuteness_mixed_model_specification"])

    coef = _single_row(
        coefficients,
        coefficients["model_id"].astype(str).eq(model_id)
        & coefficients["term"].astype(str).eq(term),
        label="configured arousal coefficient",
    )
    diag = _single_row(
        diagnostics,
        diagnostics["model_id"].astype(str).eq(model_id),
        label="configured model diagnostic",
    )
    model_spec = _single_row(
        specification,
        specification["model_id"].astype(str).eq(model_id),
        label="configured model specification",
    )

    if not bool(diag["arousal_included"]):
        raise ValueError("Accepted model diagnostics state that arousal was not included.")
    if not bool(model_spec["arousal_included"]):
        raise ValueError("Accepted model specification states that arousal was not included.")
    fixed_effects = {part.strip() for part in str(model_spec["fixed_effects"]).split(";") if part.strip()}
    if term not in fixed_effects:
        raise ValueError(f"Configured coefficient {term!r} is absent from accepted fixed effects.")
    if not bool(diag["converged"]):
        raise ValueError("Accepted full mixed model did not converge; extraction is not accepted.")

    for field in (
        "estimate",
        "standard_error",
        "confidence_interval_low",
        "confidence_interval_high",
        "test_statistic",
        "p_value",
    ):
        if not np.isfinite(float(coef[field])):
            raise ValueError(f"Accepted arousal coefficient field {field} is not finite.")

    row = {
        "model_id": model_id,
        "term": term,
        "analysis_status": "SECONDARY_EXISTING_MODEL_COEFFICIENT",
        "coefficient_role_in_accepted_model": str(coef["coefficient_role"]),
        "estimate_uv_per_centered_rating_unit": float(coef["estimate"]),
        "standard_error": float(coef["standard_error"]),
        "ci95_low": float(coef["confidence_interval_low"]),
        "ci95_high": float(coef["confidence_interval_high"]),
        "test_statistic": float(coef["test_statistic"]),
        "p_value": float(coef["p_value"]),
        "model_status": str(diag["model_status"]),
        "converged": bool(diag["converged"]),
        "fit_warnings": "" if pd.isna(diag["fit_warnings"]) else str(diag["fit_warnings"]),
        "n_observations": int(diag["n_observations"]),
        "n_participants": int(diag["n_participants"]),
        "n_items": int(diag["n_items"]),
        "participant_random_intercept_variance": float(diag["participant_random_intercept_variance"]),
        "item_random_intercept_variance": float(diag["item_random_intercept_variance"]),
        "residual_variance": float(diag["residual_variance"]),
        "formula": str(model_spec["formula"]),
        "optimizer": str(model_spec["optimizer"]),
        "fit_method": str(model_spec["fit_method"]),
        "fallback_occurred": bool(diag["fallback_occurred"]),
        "refit_performed": False,
    }
    return pd.DataFrame([row]), [paths[key] for key in required_input_keys]


def plot_existing_arousal_erp_coefficient(
    table: pd.DataFrame,
    path: Path,
    *,
    root: Path,
) -> Path:
    """Plot the existing coefficient using the established Kawaii figure style."""

    import matplotlib.pyplot as plt

    if len(table) != 1:
        raise ValueError("Existing arousal ERP coefficient plot expects exactly one result row.")
    row = table.iloc[0]
    estimate = float(row["estimate_uv_per_centered_rating_unit"])
    lo = float(row["ci95_low"])
    hi = float(row["ci95_high"])

    style = load_figure_style(root=root)
    fig, ax = plt.subplots(figsize=style.figure_size("statistics"), facecolor=style.figure_facecolor)
    color = style.color("difference_waveform", "#333A40")
    ax.errorbar(
        [estimate], [0],
        xerr=np.asarray([[estimate - lo], [hi - estimate]], dtype=float),
        fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.2,
    )
    ax.axvline(
        0.0,
        color=style.color("reference", "#777777"),
        linewidth=float(style.line_widths["reference"]),
        alpha=float(style.axes["reference_alpha"]),
    )
    ax.set_yticks([0])
    ax.set_yticklabels(["Arousal (centred)"])
    ax.set_xlabel("ERP coefficient (µV per centred rating unit)")
    ax.set_title("Arousal association with ERP amplitude")
    style_axes(ax, style)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, bbox_inches="tight", facecolor=style.figure_facecolor)
    plt.close(fig)
    return path
