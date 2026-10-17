"""Rating-by-recognition analyses for Kawaii Session 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..figure_style import load_figure_style, style_axes
from ..group_statistics import (
    correct_p_values,
    degenerate_difference_status,
    mean_confidence_interval,
    normality_check,
    paired_test,
    selected_paired_test,
    statistics_settings,
)


def _comparison_specs(family_spec: Mapping[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for raw in family_spec.get("comparisons", []) or []:
        spec = {
            "name": str(raw.get("name", "")).strip(),
            "label": str(raw.get("label", raw.get("name", ""))).strip(),
            "rating": str(raw.get("rating", "")).strip(),
            "rating_column": str(raw.get("rating_column", "")).strip(),
            "condition_a": str(raw.get("condition_a", "")).strip(),
            "condition_b": str(raw.get("condition_b", "")).strip(),
        }
        if not all(spec.values()):
            raise ValueError(f"Incomplete rating-memory comparison specification: {raw!r}")
        specs.append(spec)
    expected_size = int(family_spec.get("family_size", -1))
    if len(specs) != expected_size:
        raise ValueError(
            f"Rating-memory family defines {len(specs)} comparisons; expected exactly {expected_size}."
        )
    return specs


def summarise_participant_rating_means(
    trials: pd.DataFrame,
    family_spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Compute participant means for each configured rating × recognition condition."""

    required_base = {"participant", "recognition_condition"}
    missing_base = sorted(required_base - set(trials.columns))
    if missing_base:
        raise ValueError(f"Rating-memory trials lack required columns: {missing_base}")

    specs = _comparison_specs(family_spec)
    rating_columns = {spec["rating"]: spec["rating_column"] for spec in specs}
    for rating, column in rating_columns.items():
        if column not in trials.columns:
            raise ValueError(f"Configured {rating!r} rating column is missing: {column!r}")

    conditions = sorted({spec["condition_a"] for spec in specs} | {spec["condition_b"] for spec in specs})
    rows: list[dict[str, Any]] = []
    for rating, column in rating_columns.items():
        values = pd.to_numeric(trials[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Source-verified {rating} ratings contain missing/non-finite values.")
        if ((values < 1) | (values > 9)).any():
            raise ValueError(f"Source-verified {rating} ratings fall outside the expected 1–9 scale.")

        work = trials.assign(_rating_value=values)
        work = work[work["recognition_condition"].astype(str).isin(conditions)].copy()
        grouped = (
            work.groupby(["participant", "recognition_condition"], sort=True)["_rating_value"]
            .agg(["count", "mean"])
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "subject": str(row.participant),
                    "rating": rating,
                    "rating_column": column,
                    "condition": str(row.recognition_condition),
                    "n_trials": int(row.count),
                    "mean_rating": float(row.mean),
                }
            )

    return pd.DataFrame(rows).sort_values(["rating", "subject", "condition"]).reset_index(drop=True)


def compute_rating_memory_family(
    trials: pd.DataFrame,
    family_spec: Mapping[str, Any],
    core_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test the four predeclared participant-level rating-memory contrasts."""

    settings = statistics_settings(core_config)
    means = summarise_participant_rating_means(trials, family_spec)
    rows: list[dict[str, Any]] = []

    for spec in _comparison_specs(family_spec):
        subset = means[
            means["rating"].eq(spec["rating"])
            & means["condition"].isin([spec["condition_a"], spec["condition_b"]])
        ].copy()
        duplicate = subset.duplicated(["subject", "condition"], keep=False)
        if bool(duplicate.any()):
            raise ValueError(f"Duplicate participant-condition rating means for {spec['name']}.")

        wide = subset.pivot(index="subject", columns="condition", values="mean_rating")
        if spec["condition_a"] not in wide.columns or spec["condition_b"] not in wide.columns:
            raise ValueError(f"Missing recognition condition for rating comparison {spec['name']}.")
        paired = wide[[spec["condition_a"], spec["condition_b"]]].dropna().sort_index()
        minimum_pairs = int(settings.get("technical_minimum_pairs", settings.get("minimum_pairs", 2)))
        if len(paired) < minimum_pairs:
            raise ValueError(
                f"Insufficient paired participants for {spec['name']}: {len(paired)} < {minimum_pairs}."
            )

        a = paired[spec["condition_a"]].astype(float)
        b = paired[spec["condition_b"]].astype(float)
        diff = a - b
        assumption = normality_check(
            diff,
            spec["name"],
            spec["rating"],
            "participant_condition_mean",
            settings,
        )
        test_name = selected_paired_test(diff, settings, assumption)
        degenerate_status, degenerate_reason = degenerate_difference_status(diff)
        if degenerate_status:
            statistic = np.nan
            raw_p = np.nan
            effect_metric = "d_z" if test_name != "wilcoxon" else "rank_biserial_correlation"
            effect_value = np.nan
            status = degenerate_status
            reason = degenerate_reason
        else:
            statistic, raw_p, effect_metric, effect_value = paired_test(diff, test_name)
            status = "computed"
            reason = ""

        ci_low, ci_high = mean_confidence_interval(
            diff,
            float(settings.get("confidence_level", 0.95)),
        )
        rows.append(
            {
                "comparison": spec["name"],
                "comparison_label": spec["label"],
                "rating": spec["rating"],
                "rating_column": spec["rating_column"],
                "condition_a": spec["condition_a"],
                "condition_b": spec["condition_b"],
                "difference_direction": f"{spec['condition_a']} - {spec['condition_b']}",
                "N": int(len(diff)),
                "mean_condition_a": float(a.mean()),
                "sd_condition_a": float(a.std(ddof=1)),
                "mean_condition_b": float(b.mean()),
                "sd_condition_b": float(b.std(ddof=1)),
                "mean_difference": float(diff.mean()),
                "sd_difference": float(diff.std(ddof=1)),
                "paired_effect_metric": effect_metric,
                "paired_effect_size": float(effect_value),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "test": test_name,
                "statistic": float(statistic),
                "raw_p": float(raw_p),
                "normality_test": assumption.get("normality_test"),
                "normality_statistic": assumption.get("statistic"),
                "normality_p": assumption.get("p_value"),
                "status": status,
                "reason": reason,
                "hypothesis_family": str(family_spec["name"]),
            }
        )

    table = pd.DataFrame(rows)
    expected_size = int(family_spec["family_size"])
    if len(table) != expected_size:
        raise ValueError(
            f"Family {family_spec['name']} produced {len(table)} tests; expected exactly {expected_size}."
        )
    correction = str(family_spec.get("correction", "holm"))
    table["adjusted_p"] = correct_p_values(table["raw_p"], correction)
    table["correction_method"] = correction
    table["correction_family_size"] = expected_size
    ordered = table["raw_p"].sort_values(kind="stable")
    ranks = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
    table["correction_rank"] = ranks.reindex(table.index).astype("Int64")
    alpha = float(settings.get("alpha", 0.05))
    table["reject_null"] = table["adjusted_p"].lt(alpha) & table["adjusted_p"].notna()
    return means, table


def plot_rating_memory_forest(
    table: pd.DataFrame,
    path: Path,
    *,
    root: Path,
) -> Path:
    """Render the four predeclared rating-memory contrasts using the Kawaii style."""

    import matplotlib.pyplot as plt

    style = load_figure_style(root=root)
    fig, ax = plt.subplots(figsize=style.figure_size("statistics"), facecolor=style.figure_facecolor)
    display = table.reset_index(drop=True)
    y = np.arange(len(display))[::-1]
    means = display["mean_difference"].to_numpy(float)
    lo = display["ci95_low"].to_numpy(float)
    hi = display["ci95_high"].to_numpy(float)
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
    ax.set_yticklabels(display["comparison_label"].astype(str))
    ax.set_xlabel("Mean paired rating difference (1–9 scale units)")
    ax.set_title("Ratings and recognition outcome")
    style_axes(ax, style)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, bbox_inches="tight", facecolor=style.figure_facecolor)
    plt.close(fig)
    return path
