"""Stimulus-locked recognition ERP analyses for Kawaii Session 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..figure_style import load_figure_style, style_axes
from ..group_statistics import correct_p_values, paired_result, statistics_settings
from .config import resolve_path


def load_recognition_measurements(
    root: str | Path,
    secondary_config: Mapping[str, Any],
) -> pd.DataFrame:
    project_root = Path(root).resolve()
    path = resolve_path(project_root, secondary_config["inputs"]["recognition_measurements"])
    table = pd.read_csv(path)
    participants = [str(v) for v in secondary_config["participants"]]
    observed = sorted(table["subject"].dropna().astype(str).unique().tolist())
    if observed != sorted(participants):
        raise ValueError(
            f"Recognition measurement participant set differs from configured cohort: {observed!r}."
        )
    return table


def compute_erp_family(
    measurements: pd.DataFrame,
    family_spec: Mapping[str, Any],
    core_config: Mapping[str, Any],
    recognition_cells: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    settings = statistics_settings(core_config)
    rows: list[dict[str, Any]] = []
    comparisons = list(family_spec.get("comparisons", []) or [])

    for comparison in comparisons:
        comp = {
            "name": str(comparison["name"]),
            "condition_a": str(comparison["condition_a"]),
            "condition_b": str(comparison["condition_b"]),
        }
        for cell_spec in recognition_cells:
            roi = str(cell_spec["roi"])
            time_window = str(cell_spec["time_window"])
            cell = measurements[
                measurements["roi"].astype(str).eq(roi)
                & measurements["time_window"].astype(str).eq(time_window)
            ].copy()
            result = paired_result(cell, comp, roi, time_window, settings)
            stats_row = dict(result["stats"])
            analysis_status = str(family_spec.get("analysis_status", "")).strip().upper()
            if analysis_status not in {"SECONDARY", "EXPLORATORY"}:
                raise ValueError(
                    f"Family {family_spec.get('name', '<unnamed>')} must declare analysis_status "
                    "as SECONDARY or EXPLORATORY."
                )
            stats_row.pop("interpretation_status", None)
            stats_row["analysis_status"] = analysis_status
            stats_row["paired_dz"] = result["effect"].get("effect_size_value")
            stats_row["effect_size_metric"] = result["effect"].get("effect_size")
            stats_row["comparison_label"] = str(comparison.get("label", comp["name"]))
            stats_row["difference_direction"] = f"{comp['condition_a']} - {comp['condition_b']}"
            stats_row["cell_label"] = str(cell_spec.get("label", f"{roi} · {time_window}"))
            stats_row["hypothesis_family"] = str(family_spec["name"])
            rows.append(stats_row)

    table = pd.DataFrame(rows)
    expected_size = int(family_spec["family_size"])
    if len(table) != expected_size:
        raise ValueError(
            f"Family {family_spec['name']} produced {len(table)} tests; expected exactly {expected_size}."
        )

    correction = str(family_spec.get("correction", "holm"))
    table["raw_p"] = pd.to_numeric(table["p_value"], errors="coerce")
    table["adjusted_p"] = correct_p_values(table["raw_p"], correction)
    table["correction_method"] = correction
    table["correction_family_size"] = expected_size
    ordered = table["raw_p"].sort_values(kind="stable")
    ranks = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
    table["correction_rank"] = ranks.reindex(table.index).astype("Int64")
    alpha = float(settings.get("alpha", 0.05))
    table["reject_null"] = table["adjusted_p"].lt(alpha) & table["adjusted_p"].notna()
    return table


def validate_secondary_against_gate5(
    root: str | Path,
    secondary_config: Mapping[str, Any],
    secondary_table: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Validate reused secondary ERP statistics against current Gate 5C results."""

    project_root = Path(root).resolve()
    path = resolve_path(project_root, secondary_config["inputs"]["recognition_statistics"])
    gate5 = pd.read_csv(path)
    rows = []

    for _, row in secondary_table.iterrows():
        comparison = str(row["comparison"])
        if comparison == "hit_vs_miss":
            stored_name = "hit_vs_miss"
            sign = 1.0
        elif comparison == "false_alarm_vs_correct_rejection":
            stored_name = "correct_rejection_vs_false_alarm"
            sign = -1.0
        else:
            continue

        match = gate5[
            gate5["comparison"].astype(str).eq(stored_name)
            & gate5["roi"].astype(str).eq(str(row["roi"]))
            & gate5["time_window"].astype(str).eq(str(row["time_window"]))
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one Gate 5C comparison for {stored_name}/{row['roi']}/{row['time_window']}."
            )
        stored = match.iloc[0]
        observed = {
            "mean_difference": float(row["mean_difference"]),
            "paired_dz": float(row["paired_dz"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "raw_p": float(row["raw_p"]),
        }
        expected = {
            "mean_difference": sign * float(stored["mean_difference"]),
            "paired_dz": sign * float(stored["effect_size"]),
            "ci_low": float(stored["ci_low"]) if sign > 0 else -float(stored["ci_high"]),
            "ci_high": float(stored["ci_high"]) if sign > 0 else -float(stored["ci_low"]),
            "raw_p": float(stored["p_value"]),
        }
        errors = {key: abs(observed[key] - expected[key]) for key in observed}
        rows.append(
            {
                "comparison": comparison,
                "roi": row["roi"],
                "time_window": row["time_window"],
                **{f"abs_error_{key}": value for key, value in errors.items()},
                "status": "PASS" if max(errors.values()) <= tolerance else "FAIL",
            }
        )

    validation = pd.DataFrame(rows)
    if not validation.empty and not validation["status"].eq("PASS").all():
        raise ValueError("Independent recomputation does not reproduce current Gate 5C outputs.")
    return validation


def plot_erp_family_forest(
    table: pd.DataFrame,
    path: Path,
    *,
    root: Path,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    style = load_figure_style(root=root)
    fig, ax = plt.subplots(figsize=style.figure_size("statistics"), facecolor=style.figure_facecolor)
    display = table.reset_index(drop=True)
    y = np.arange(len(display))[::-1]
    means = display["mean_difference"].to_numpy(float)
    lo = display["ci_low"].to_numpy(float)
    hi = display["ci_high"].to_numpy(float)
    xerr = np.vstack([means - lo, hi - means])
    color = style.color("difference_waveform", "#333A40")
    ax.errorbar(means, y, xerr=xerr, fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.2)
    ax.axvline(
        0.0,
        color=style.color("reference", "#777777"),
        linewidth=float(style.line_widths["reference"]),
        alpha=float(style.axes["reference_alpha"]),
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row.comparison_label} · {row.cell_label}" for row in display.itertuples()])
    ax.set_xlabel("Mean paired difference (µV)")
    ax.set_title(title)
    style_axes(ax, style)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, bbox_inches="tight", facecolor=style.figure_facecolor)
    plt.close(fig)
    return path
