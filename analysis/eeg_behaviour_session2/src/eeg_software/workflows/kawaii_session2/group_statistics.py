from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import math
import re

import numpy as np
import pandas as pd

from .cuteness_mixed import fit_cuteness_mixed_model
from .figure_style import FigureStyle, default_figure_style, load_figure_style, style_axes
from .group_aggregation import pool_participant_measurements, validate_pooled_measurements
from .paths import resolve_session2_eeg_paths
from .provenance import write_group_provenance_sidecar
from .resolved_snapshot import ResolvedRunSnapshotReference, bind_authoritative_config_provenance
from .study_definition import configured_conditions, erp_branch_settings


@dataclass(frozen=True)
class Gate5CResult:
    output_files: dict[str, Path]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PreparedPlotCell:
    """One compatible Gate 5C scientific figure cell."""

    branch: str
    roi: str
    time_window: str
    values: pd.DataFrame
    summary: pd.DataFrame
    participant_n: int
    plot_unit: str = "participant"

    @property
    def slug(self) -> str:
        return f"{_figure_slug(self.roi)}__{_figure_slug(self.time_window)}"


def run_gate5c_statistics(
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
    published_input_paths_by_branch: Mapping[str, Mapping[str, Path]] | None = None,
) -> Gate5CResult:
    """Run Gate 5C ROI/time-window statistics from Gate 5A aggregate CSVs only."""

    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    output_files: dict[str, Path] = {}
    warnings: list[str] = []
    for branch in ("recognition", "cuteness"):
        branch_outputs = run_gate5c_branch(config, branch, root=root, config_path=config_path, warnings=warnings, published_input_paths=(published_input_paths_by_branch or {}).get(branch))
        output_files.update({f"{branch}_{label}": path for label, path in branch_outputs.items()})
    return Gate5CResult(output_files=output_files, warnings=tuple(warnings))


def run_gate5c_branch(config: Mapping[str, Any], branch: str, *, root: Path | None = None, config_path: str | Path | None = None, resolved_snapshot: ResolvedRunSnapshotReference | None = None, warnings: list[str] | None = None, published_input_paths: Mapping[str, Path] | None = None) -> dict[str, Path]:
    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    warnings = warnings if warnings is not None else []
    paths = gate5c_paths(config, branch, root=root)
    inputs = dict(published_input_paths or gate5c_input_paths(config, branch, root=root))
    missing = [label for label, path in inputs.items() if not path.exists()]
    if missing:
        warnings.append(f"{branch}: missing Gate 5A input(s): {', '.join(missing)}")

    measurements = _read_csv(inputs["measurements"])
    if measurements.empty:
        measurements = pool_participant_measurements(_read_csv(inputs["measurements_parts"]))
    validate_pooled_measurements(measurements)
    participant_summary = _read_csv(inputs["participant_summary"])
    included_subjects = included_subjects_from_summary(participant_summary, branch)
    included = filter_included_measurements(measurements, included_subjects)
    settings = statistics_settings(config)
    figure_style = load_figure_style(root=root)

    stats_table, effects_table, assumptions_table, qc_rows = analyse_branch_statistics(config, branch, included, participant_summary, settings)
    stats_table = attach_technical_n(stats_table, measurements)
    stats_table = apply_multiplicity_families(stats_table, settings)

    summary = build_statistics_summary(stats_table)
    qc = build_statistics_qc_summary(branch, participant_summary, included, stats_table, assumptions_table, qc_rows, settings)

    outputs = {
        "statistics": _write_table(stats_table, paths["statistics"]),
        "effect_sizes": _write_table(effects_table, paths["effect_sizes"]),
        "assumption_checks": _write_table(assumptions_table, paths["assumption_checks"]),
        "statistics_summary": _write_table(summary, paths["statistics_summary"]),
        "statistics_qc_summary": _write_table(qc, paths["statistics_qc_summary"]),
    }
    if branch == "cuteness":
        trials = _read_csv(inputs["trial_level"])
        if not trials.empty and "participant" in trials.columns:
            trials = trials[trials["participant"].astype(str).isin(included_subjects)].copy()
        model_result = fit_cuteness_mixed_model(trials, config)
        outputs.update(
            {
                "arousal_feasibility": _write_table(model_result.arousal_feasibility, paths["arousal_feasibility"]),
                "mixed_model_specification": _write_table(model_result.specification, paths["mixed_model_specification"]),
                "mixed_model_coefficients": _write_table(model_result.coefficients, paths["mixed_model_coefficients"]),
                "mixed_model_diagnostics": _write_table(model_result.diagnostics, paths["mixed_model_diagnostics"]),
                "mixed_model_attempts": _write_table(model_result.attempt_ledger, paths["mixed_model_attempts"]),
            }
        )
    outputs.update(write_statistics_figures(branch, stats_table, effects_table, included, paths, figure_style=figure_style))
    outputs["report"] = write_statistics_report(branch, paths["report"], outputs, stats_table, assumptions_table, qc)
    _write_gate5c_provenance(config, branch, outputs, inputs, participant_summary, warnings, root=root, config_path=config_path)
    return outputs


def gate5c_input_paths(config: Mapping[str, Any], branch: str, *, root: Path | None = None) -> dict[str, Path]:
    base = resolve_session2_eeg_paths(config, root=root, create=False).output_dir / "group" / branch
    paths = {
        "measurements": base / f"{branch}_measurements_pooled.csv",
        "measurements_parts": base / f"{branch}_measurements_all.csv",
        "wide": base / f"{branch}_measurements_all_wide.csv",
        "participant_summary": base / "participant_summary.csv",
    }
    if branch == "cuteness":
        paths["trial_level"] = base / "cuteness_trial_level_all.csv"
    return paths


def gate5c_paths(config: Mapping[str, Any], branch: str, *, root: Path | None = None) -> dict[str, Path]:
    base = resolve_session2_eeg_paths(config, root=root, create=True).output_dir / "group" / branch / "statistics"
    figures = base / "figures"
    base.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    title = branch.title()
    paths = {
        "base": base,
        "figures": figures,
        "statistics": base / f"{branch}_statistics.csv",
        "effect_sizes": base / f"{branch}_effect_sizes.csv",
        "assumption_checks": base / f"{branch}_assumption_checks.csv",
        "statistics_qc_summary": base / f"{branch}_statistics_qc_summary.csv",
        "report": base / f"{title}_statistics_report.md",
        "statistics_summary": base / f"{title}_statistics_summary.csv",
    }
    if branch == "cuteness":
        paths.update(
            {
                "arousal_feasibility": base / "cuteness_arousal_feasibility.csv",
                "mixed_model_specification": base / "cuteness_mixed_model_specification.csv",
                "mixed_model_coefficients": base / "cuteness_mixed_model_coefficients.csv",
                "mixed_model_diagnostics": base / "cuteness_mixed_model_diagnostics.csv",
                "mixed_model_attempts": base / "cuteness_mixed_model_attempts.csv",
            }
        )
    return paths


def statistics_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults = {
        "alpha": 0.05,
        "multiple_comparison_correction": "holm",
        "paired_test": "paired_t",
        "fallback_paired_test": "wilcoxon",
        "normality_test": "shapiro",
        "effect_size": "auto",
        "technical_minimum_pairs": 2,
        "scientific_interpretation_minimum_n": None,
        "confidence_level": 0.95,
        "minimum_normality_n": 3,
    }
    defaults.update(dict(config.get("statistics", {}) or {}))
    validate_statistics_config(defaults)
    return defaults


def validate_statistics_config(settings: Mapping[str, Any]) -> None:
    allowed = {
        "paired_test": {"paired_t", "wilcoxon", "auto"},
        "fallback_paired_test": {"paired_t", "wilcoxon"},
        "normality_test": {"shapiro", "none"},
        "multiple_comparison_correction": {"none", "bonferroni", "holm", "fdr_bh"},
        "effect_size": {"auto", "cohens_d_paired", "rank_biserial"},
    }
    for key, values in allowed.items():
        value = str(settings.get(key, "")).strip()
        if value not in values:
            raise ValueError(f"Unsupported statistics.{key}: {value!r}. Allowed values: {', '.join(sorted(values))}.")


def included_subjects_from_summary(participant_summary: pd.DataFrame, branch: str) -> set[str]:
    include_col = f"{branch}_included"
    if participant_summary.empty or include_col not in participant_summary.columns or "subject" not in participant_summary.columns:
        return set()
    return set(participant_summary.loc[_as_bool_series(participant_summary[include_col]), "subject"].dropna().astype(str))


def filter_included_measurements(measurements: pd.DataFrame, included_subjects: set[str]) -> pd.DataFrame:
    if measurements.empty or "subject" not in measurements.columns:
        return measurements.iloc[0:0].copy()
    filtered = measurements[measurements["subject"].astype(str).isin(included_subjects)].copy()
    if "mean_amplitude_uv" in filtered.columns:
        filtered["mean_amplitude_uv"] = pd.to_numeric(filtered["mean_amplitude_uv"], errors="coerce")
    return filtered


def attach_technical_n(stats_table: pd.DataFrame, all_measurements: pd.DataFrame) -> pd.DataFrame:
    """Distinguish executable paired rows from scientifically included paired rows."""

    table = stats_table.copy()
    for index, row in table.iterrows():
        cell = all_measurements
        if {"roi", "time_window"}.issubset(cell.columns):
            cell = cell[(cell["roi"].astype(str) == str(row.get("roi"))) & (cell["time_window"].astype(str) == str(row.get("time_window")))]
        if str(row.get("analysis_type")) == "paired":
            technical_n = len(paired_condition_values(cell, str(row.get("condition_a")), str(row.get("condition_b"))))
        else:
            technical_n = int(row.get("n", 0) or 0)
        table.loc[index, "technical_n"] = technical_n
        table.loc[index, "scientifically_interpretable_n"] = int(row.get("n", 0) or 0)
    return table


def analyse_branch_statistics(
    config: Mapping[str, Any],
    branch: str,
    measurements: pd.DataFrame,
    participant_summary: pd.DataFrame,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    stats_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    assumption_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    if measurements.empty:
        qc_rows.append({"branch": branch, "comparison": "", "roi": "", "time_window": "", "status": "skipped", "reason": "no included measurements"})
        return _statistics_frame(stats_rows), _effect_frame(effect_rows), _assumption_frame(assumption_rows), qc_rows

    comparisons = branch_comparisons(config, branch, measurements)
    for comparison in comparisons:
        for roi, time_window in comparison_measurement_cells(comparison, measurements):
            cell = measurements[(measurements["roi"].astype(str) == roi) & (measurements["time_window"].astype(str) == time_window)]
            if comparison.get("kind") == "one_sample":
                result = one_sample_result(cell, comparison, roi, time_window, settings)
            else:
                result = paired_result(cell, comparison, roi, time_window, settings)
            scientific = comparison_scientific_metadata(comparison, roi, time_window)
            result["stats"].update(scientific)
            result["effect"].update(scientific)
            result["assumption"].update(scientific)
            n_value = int(result["stats"].get("n", 0) or 0)
            result["stats"].setdefault("technical_n", n_value)
            result["stats"].setdefault("scientifically_interpretable_n", n_value)
            result["stats"].setdefault("interpretation_status", "PROVISIONAL_PIPELINE_VALIDATION_UNTIL_FINAL_SAMPLE")
            result["stats"].setdefault("raw_p", result["stats"].get("p_value", math.nan))
            result["stats"]["assumption_status"] = _assumption_status(result["assumption"])
            stats_rows.append(result["stats"])
            effect_rows.append(result["effect"])
            assumption_rows.append(result["assumption"])
            if result["qc"]["status"] != "computed":
                qc_rows.append(result["qc"])

    if branch == "cuteness":
        continuous_rows = continuous_analysis_rows(measurements, settings)
        stats_rows.extend(continuous_rows["stats"])
        effect_rows.extend(continuous_rows["effects"])
        assumption_rows.extend(continuous_rows["assumptions"])
        qc_rows.extend(continuous_rows["qc"])

    return _statistics_frame(stats_rows), _effect_frame(effect_rows), _assumption_frame(assumption_rows), qc_rows


def _assumption_status(row: Mapping[str, Any]) -> str:
    if row.get("reason"):
        return "NOT_ASSESSED: " + str(row.get("reason"))
    if pd.isna(row.get("passed")):
        return "NOT_ASSESSED"
    return "DIAGNOSTIC_COMPATIBLE" if _as_bool(row.get("passed")) else "DIAGNOSTIC_FLAG"


def branch_comparisons(config: Mapping[str, Any], branch: str, measurements: pd.DataFrame) -> list[dict[str, Any]]:
    stats = statistics_settings(config)
    branch_stats = dict(stats.get(branch, {}) or {})
    if branch == "recognition":
        # Difference-wave values remain in the participant table for description and
        # plotting, but are deliberately not converted into a mathematically duplicate
        # one-sample inferential test (REC-D03 / STAT-D09).
        return [dict(item) for item in branch_stats.get("comparisons", [])]

    if str(branch_stats.get("categorical_comparisons", "automatic")) == "descriptive_only":
        return []
    configured = list(configured_conditions(erp_branch_settings(config, "cuteness")).keys())
    observed = [condition for condition in configured if condition in set(measurements["condition"].dropna().astype(str))]
    comparisons = []
    if len(observed) >= 2:
        comparisons.append({"name": f"{observed[0]}_vs_{observed[-1]}", "condition_a": observed[-1], "condition_b": observed[0], "label": f"{observed[-1]} vs {observed[0]}"})
        comparisons.extend({"name": f"{a}_vs_{b}", "condition_a": b, "condition_b": a, "label": f"{b} vs {a}"} for a, b in zip(observed, observed[1:]))
    return comparisons


def measurement_cells(measurements: pd.DataFrame) -> list[tuple[str, str]]:
    if not {"roi", "time_window"}.issubset(measurements.columns):
        return []
    cells = measurements[["roi", "time_window"]].drop_duplicates().dropna()
    return [(str(row["roi"]), str(row["time_window"])) for _, row in cells.sort_values(["roi", "time_window"]).iterrows()]


def comparison_measurement_cells(comparison: Mapping[str, Any], measurements: pd.DataFrame) -> list[tuple[str, str]]:
    """Return every predeclared cell so missing primary hypotheses remain auditable."""

    observed = set(measurement_cells(measurements))
    configured = comparison.get("cells", []) or []
    if not configured:
        return sorted(observed)
    return [(str(cell.get("roi", "")), str(cell.get("time_window", ""))) for cell in configured]


def comparison_scientific_metadata(comparison: Mapping[str, Any], roi: str, time_window: str) -> dict[str, str]:
    """Attach finite hypothesis-family metadata to an inferential result row."""

    hypothesis_id = ""
    for cell in comparison.get("cells", []) or []:
        if str(cell.get("roi", "")) == roi and str(cell.get("time_window", "")) == time_window:
            hypothesis_id = str(cell.get("hypothesis_id", ""))
            break
    return {
        "analysis_status": str(comparison.get("analysis_status", "SECONDARY")),
        "hypothesis_family": str(comparison.get("hypothesis_family", "")),
        "hypothesis_id": hypothesis_id,
    }


def paired_result(cell: pd.DataFrame, comparison: Mapping[str, str], roi: str, time_window: str, settings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    condition_a = str(comparison.get("condition_a", ""))
    condition_b = str(comparison.get("condition_b", ""))
    comparison_name = str(comparison.get("name", f"{condition_a}_vs_{condition_b}"))
    paired = paired_condition_values(cell, condition_a, condition_b)
    minimum_pairs = int(settings.get("technical_minimum_pairs", settings.get("minimum_pairs", 2)))
    if len(paired) < minimum_pairs:
        reason = f"insufficient paired observations ({len(paired)} < {minimum_pairs})"
        return empty_result("paired", comparison_name, condition_a, condition_b, roi, time_window, reason, len(paired), settings)

    a = paired["a"].astype(float)
    b = paired["b"].astype(float)
    diff = a - b
    assumption = normality_check(diff, comparison_name, roi, time_window, settings)
    test_name = selected_paired_test(diff, settings, assumption)
    degenerate_status, degenerate_reason = degenerate_difference_status(diff)
    if degenerate_status:
        statistic, p_value = math.nan, math.nan
        effect_name, effect_value = effect_size_for_test(diff, test_name)
        status, reason = degenerate_status, degenerate_reason
    else:
        statistic, p_value, effect_name, effect_value = paired_test(diff, test_name)
        status, reason = "computed", ""
    stats = descriptive_stats_row("paired", comparison_name, condition_a, condition_b, roi, time_window, a, b, diff, statistic, p_value, test_name, status=status, reason=reason, confidence_level=float(settings.get("confidence_level", 0.95)))
    stats["effect_size_metric"] = effect_name
    stats["effect_size"] = effect_value
    effect = effect_size_row("paired", comparison_name, condition_a, condition_b, roi, time_window, effect_name, effect_value, len(diff))
    qc = {"branch": "", "comparison": comparison_name, "roi": roi, "time_window": time_window, "status": status, "reason": reason}
    return {"stats": stats, "effect": effect, "assumption": assumption, "qc": qc}


def one_sample_result(cell: pd.DataFrame, comparison: Mapping[str, str], roi: str, time_window: str, settings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    condition = str(comparison.get("condition", ""))
    comparison_name = str(comparison.get("name", f"{condition}_summary"))
    values = _finite_numeric_values(
        cell.loc[cell["condition"].astype(str) == condition, "mean_amplitude_uv"],
        label=f"one-sample measurement {condition!r} at {roi!r}/{time_window!r}",
        allow_missing=False,
    )
    minimum_pairs = int(settings.get("technical_minimum_pairs", settings.get("minimum_pairs", 2)))
    if len(values) < minimum_pairs:
        reason = f"insufficient observations ({len(values)} < {minimum_pairs})"
        return empty_result("one_sample", comparison_name, condition, "zero", roi, time_window, reason, len(values), settings)
    assumption = normality_check(values, comparison_name, roi, time_window, settings)
    test_name = selected_paired_test(values, settings, assumption)
    degenerate_status, degenerate_reason = degenerate_difference_status(values)
    if degenerate_status:
        statistic, p_value = math.nan, math.nan
        effect_name, effect_value = effect_size_for_test(values, test_name)
        status, reason = degenerate_status, degenerate_reason
    else:
        statistic, p_value, effect_name, effect_value = one_sample_test(values, test_name)
        status, reason = "computed", ""
    stats = descriptive_stats_row("one_sample", comparison_name, condition, "zero", roi, time_window, values, pd.Series([0.0] * len(values)), values, statistic, p_value, test_name, status=status, reason=reason)
    effect = effect_size_row("one_sample", comparison_name, condition, "zero", roi, time_window, effect_name, effect_value, len(values))
    qc = {"branch": "", "comparison": comparison_name, "roi": roi, "time_window": time_window, "status": status, "reason": reason}
    return {"stats": stats, "effect": effect, "assumption": assumption, "qc": qc}


def paired_condition_values(cell: pd.DataFrame, condition_a: str, condition_b: str) -> pd.DataFrame:
    required = {"subject", "condition", "mean_amplitude_uv"}
    missing = sorted(required - set(cell.columns))
    if missing:
        raise ValueError(f"Paired measurement table lacks required column(s): {', '.join(missing)}")
    keep = cell[cell["condition"].astype(str).isin([condition_a, condition_b])]
    if keep.empty:
        return pd.DataFrame(columns=["subject", "a", "b"])
    for axis in ("branch", "roi", "time_window"):
        if axis in keep.columns and keep[axis].nunique(dropna=False) > 1:
            raise ValueError(
                f"Paired comparison spans multiple {axis} values; select one authoritative analysis cell before pairing."
            )
    duplicate_key = ["subject", "condition"]
    duplicate = keep.duplicated(duplicate_key, keep=False)
    if bool(duplicate.any()):
        context = [column for column in ("subject", "branch", "condition", "roi", "time_window") if column in keep.columns]
        bad = keep.loc[duplicate, context].drop_duplicates()
        raise ValueError(
            f"Duplicate participant-condition measurement before paired pivot; key={tuple(context)!r}: "
            f"{bad.head(5).to_dict(orient='records')!r}"
        )
    keep = keep.copy()
    keep["mean_amplitude_uv"] = _finite_numeric_values(
        keep["mean_amplitude_uv"],
        label=f"paired measurements {condition_a!r} versus {condition_b!r}",
        allow_missing=False,
    )
    wide = keep.pivot(index="subject", columns="condition", values="mean_amplitude_uv")
    if condition_a not in wide.columns or condition_b not in wide.columns:
        return pd.DataFrame(columns=["subject", "a", "b"])
    paired = wide[[condition_a, condition_b]].dropna().reset_index()
    paired.columns = ["subject", "a", "b"]
    return paired


def paired_test(diff: pd.Series, test_name: str) -> tuple[float, float, str, float]:
    values = _finite_numeric_values(diff, label=f"{test_name} paired differences", allow_missing=False)
    try:
        from scipy import stats

        if test_name == "wilcoxon":
            if values.nunique() < 2:
                return math.nan, math.nan, "rank_biserial_correlation", rank_biserial_correlation(values)
            statistic, p_value = stats.wilcoxon(values)
            return float(statistic), float(p_value), "rank_biserial_correlation", rank_biserial_correlation(values)
        if len(values) < 2 or values.nunique() < 2:
            return math.nan, math.nan, "d_z", cohens_d(values)
        statistic, p_value = stats.ttest_1samp(values, popmean=0.0)
        return float(statistic), float(p_value), "d_z", cohens_d(values)
    except Exception:
        return math.nan, math.nan, "d_z" if test_name != "wilcoxon" else "rank_biserial_correlation", math.nan


def one_sample_test(values: pd.Series, test_name: str) -> tuple[float, float, str, float]:
    return paired_test(values, test_name)


def degenerate_difference_status(values: pd.Series) -> tuple[str, str]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty or numeric.nunique() != 1:
        return "", ""
    value = float(numeric.iloc[0])
    if value == 0.0:
        return "degenerate_zero_difference", "All paired differences are zero; no test is estimable."
    return "degenerate_constant_nonzero_difference", "All paired differences are identical and non-zero; variance is zero, so conventional paired test is not estimable."


def effect_size_for_test(values: pd.Series, test_name: str) -> tuple[str, float]:
    if test_name == "wilcoxon":
        return "rank_biserial_correlation", rank_biserial_correlation(values)
    return "d_z", cohens_d(values)


def selected_paired_test(values: pd.Series, settings: Mapping[str, Any], assumption: Mapping[str, Any]) -> str:
    """Keep the configured primary test; assumption p-values do not select it."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    configured = str(settings.get("paired_test", "paired_t"))
    fallback = str(settings.get("fallback_paired_test", "wilcoxon"))
    if configured == "auto":
        return fallback if len(numeric) < 2 else "paired_t"
    return configured


def normality_check(values: pd.Series, comparison: str, roi: str, time_window: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    numeric = _finite_numeric_values(values, label=f"normality input for {comparison!r}", allow_missing=False)
    method = str(settings.get("normality_test", "shapiro"))
    row = {"comparison": comparison, "roi": roi, "time_window": time_window, "normality_test": method, "n": int(len(numeric)), "statistic": math.nan, "p_value": math.nan, "passed": pd.NA, "reason": ""}
    if method == "none":
        row["reason"] = "normality test disabled"
        return row
    minimum_normality_n = int(settings.get("minimum_normality_n", 3))
    if len(numeric) < minimum_normality_n:
        row["reason"] = "Normality test skipped because sample size was below the configured minimum."
        return row
    if numeric.nunique() < 2:
        row["reason"] = "zero range; Shapiro-Wilk not informative"
        return row
    try:
        from scipy import stats

        statistic, p_value = stats.shapiro(numeric)
        row["statistic"] = float(statistic)
        row["p_value"] = float(p_value)
        row["passed"] = bool(p_value >= float(settings.get("alpha", 0.05)))
    except Exception as exc:
        row["reason"] = str(exc)
    return row


def descriptive_stats_row(kind: str, comparison: str, condition_a: str, condition_b: str, roi: str, time_window: str, a: pd.Series, b: pd.Series, diff: pd.Series, statistic: float, p_value: float, test_name: str, *, status: str = "computed", reason: str = "", confidence_level: float = 0.95) -> dict[str, Any]:
    diff = pd.to_numeric(diff, errors="coerce").dropna()
    ci_low, ci_high = mean_confidence_interval(diff, confidence_level)
    return {
        "analysis_type": kind,
        "comparison": comparison,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "roi": roi,
        "time_window": time_window,
        "test": test_name,
        "n": int(len(diff)),
        "mean_a": _mean(a),
        "sd_a": _sd(a),
        "mean_b": _mean(b),
        "sd_b": _sd(b),
        "mean_difference": _mean(diff),
        "sd_difference": _sd(diff),
        "median_difference": _median(diff),
        "minimum_difference": _min(diff),
        "maximum_difference": _max(diff),
        "statistic": statistic,
        "p_value": p_value,
        "raw_p": p_value,
        "effect_estimate": _mean(diff),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": confidence_level,
        "technical_n": int(len(diff)),
        "scientifically_interpretable_n": int(len(diff)),
        "interpretation_status": "PROVISIONAL_PIPELINE_VALIDATION_UNTIL_FINAL_SAMPLE",
        "test_name": test_name,
        "status": status,
        "reason": reason,
    }


def mean_confidence_interval(values: pd.Series, confidence_level: float = 0.95) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < 2:
        return math.nan, math.nan
    if numeric.std(ddof=1) == 0:
        return float(numeric.mean()), float(numeric.mean())
    from scipy import stats

    alpha = 1.0 - confidence_level
    margin = float(stats.t.ppf(1.0 - alpha / 2.0, len(numeric) - 1) * stats.sem(numeric))
    return float(numeric.mean() - margin), float(numeric.mean() + margin)


def effect_size_row(kind: str, comparison: str, condition_a: str, condition_b: str, roi: str, time_window: str, effect_name: str, effect_value: float, n: int) -> dict[str, Any]:
    return {"analysis_type": kind, "comparison": comparison, "condition_a": condition_a, "condition_b": condition_b, "roi": roi, "time_window": time_window, "effect_size": effect_name, "effect_size_value": effect_value, "n": int(n)}


def empty_result(kind: str, comparison: str, condition_a: str, condition_b: str, roi: str, time_window: str, reason: str, n: int, settings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    stats = {
        "analysis_type": kind,
        "comparison": comparison,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "roi": roi,
        "time_window": time_window,
        "test": str(settings.get("paired_test", "paired_t")),
        "n": int(n),
        "mean_a": math.nan,
        "sd_a": math.nan,
        "mean_b": math.nan,
        "sd_b": math.nan,
        "mean_difference": math.nan,
        "sd_difference": math.nan,
        "median_difference": math.nan,
        "minimum_difference": math.nan,
        "maximum_difference": math.nan,
        "statistic": math.nan,
        "p_value": math.nan,
        "status": "skipped",
        "reason": reason,
    }
    effect = effect_size_row(kind, comparison, condition_a, condition_b, roi, time_window, "not_computed", math.nan, n)
    assumption = {"comparison": comparison, "roi": roi, "time_window": time_window, "normality_test": str(settings.get("normality_test", "shapiro")), "n": int(n), "statistic": math.nan, "p_value": math.nan, "passed": pd.NA, "reason": reason}
    qc = {"branch": "", "comparison": comparison, "roi": roi, "time_window": time_window, "status": "skipped", "reason": reason}
    return {"stats": stats, "effect": effect, "assumption": assumption, "qc": qc}


def continuous_analysis_rows(measurements: pd.DataFrame, settings: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    branch_settings = dict((settings.get("cuteness", {}) or {}).get("continuous", {}) or {})
    if not branch_settings.get("enabled", True):
        return {"stats": [], "effects": [], "assumptions": [], "qc": []}
    predictors = [column for column in branch_settings.get("predictor_columns", []) if column in measurements.columns]
    if not predictors:
        return {"stats": [], "effects": [], "assumptions": [], "qc": [{"branch": "cuteness", "comparison": "continuous", "roi": "", "time_window": "", "status": "skipped", "reason": "no configured continuous predictor columns present in Gate 5A measurements"}]}
    stats_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    assumption_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    method = str(branch_settings.get("method", "spearman"))
    for predictor in predictors:
        for keys, group in measurements.groupby(["condition", "roi", "time_window"], dropna=False, sort=True):
            condition, roi, time_window = (str(keys[0]), str(keys[1]), str(keys[2]))
            numeric = group[[predictor, "mean_amplitude_uv"]].apply(pd.to_numeric, errors="coerce")
            infinite = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
            if bool(infinite.any()):
                bad_rows, bad_columns = np.where(infinite)
                examples = [
                    {"row": str(numeric.index[row]), "column": str(numeric.columns[column]), "value": numeric.iloc[row, column]}
                    for row, column in zip(bad_rows[:5], bad_columns[:5])
                ]
                raise ValueError(
                    f"Non-finite continuous-analysis input for {predictor!r} at "
                    f"{condition!r}/{roi!r}/{time_window!r}: {examples!r}"
                )
            paired = numeric.dropna()
            comparison = f"{predictor}_vs_amplitude_{condition}"
            if len(paired) < 3:
                reason = f"insufficient observations for continuous analysis ({len(paired)} < 3)"
                stats_rows.append(empty_continuous_row(comparison, predictor, condition, roi, time_window, method, len(paired), reason))
                effect_rows.append(effect_size_row("continuous_correlation", comparison, predictor, condition, roi, time_window, "correlation_coefficient", math.nan, len(paired)))
                assumption_rows.append({"comparison": comparison, "roi": roi, "time_window": time_window, "normality_test": "not_applicable", "n": int(len(paired)), "statistic": math.nan, "p_value": math.nan, "passed": pd.NA, "reason": reason})
                qc_rows.append({"branch": "cuteness", "comparison": comparison, "roi": roi, "time_window": time_window, "status": "skipped", "reason": reason})
                continue
            statistic, p_value = correlation_test(paired[predictor], paired["mean_amplitude_uv"], method)
            stats_rows.append(
                {
                    "analysis_type": "continuous_correlation",
                    "comparison": comparison,
                    "condition_a": predictor,
                    "condition_b": condition,
                    "roi": roi,
                    "time_window": time_window,
                    "test": method,
                    "n": int(len(paired)),
                    "mean_a": _mean(paired[predictor]),
                    "sd_a": _sd(paired[predictor]),
                    "mean_b": _mean(paired["mean_amplitude_uv"]),
                    "sd_b": _sd(paired["mean_amplitude_uv"]),
                    "mean_difference": math.nan,
                    "sd_difference": math.nan,
                    "median_difference": math.nan,
                    "minimum_difference": math.nan,
                    "maximum_difference": math.nan,
                    "statistic": statistic,
                    "p_value": p_value,
                    "status": "computed",
                    "reason": "",
                }
            )
            effect_rows.append(effect_size_row("continuous_correlation", comparison, predictor, condition, roi, time_window, "correlation_coefficient", statistic, len(paired)))
            assumption_rows.append({"comparison": comparison, "roi": roi, "time_window": time_window, "normality_test": "not_applicable", "n": int(len(paired)), "statistic": math.nan, "p_value": math.nan, "passed": pd.NA, "reason": "continuous correlation"})
    return {"stats": stats_rows, "effects": effect_rows, "assumptions": assumption_rows, "qc": qc_rows}


def correlation_test(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    try:
        from scipy import stats

        if method == "pearson":
            statistic, p_value = stats.pearsonr(x, y)
        else:
            statistic, p_value = stats.spearmanr(x, y)
        return float(statistic), float(p_value)
    except Exception:
        return math.nan, math.nan


def empty_continuous_row(comparison: str, predictor: str, condition: str, roi: str, time_window: str, method: str, n: int, reason: str) -> dict[str, Any]:
    return {
        "analysis_type": "continuous_correlation",
        "comparison": comparison,
        "condition_a": predictor,
        "condition_b": condition,
        "roi": roi,
        "time_window": time_window,
        "test": method,
        "n": int(n),
        "mean_a": math.nan,
        "sd_a": math.nan,
        "mean_b": math.nan,
        "sd_b": math.nan,
        "mean_difference": math.nan,
        "sd_difference": math.nan,
        "median_difference": math.nan,
        "minimum_difference": math.nan,
        "maximum_difference": math.nan,
        "statistic": math.nan,
        "p_value": math.nan,
        "status": "skipped",
        "reason": reason,
    }


def correct_p_values(p_values: pd.Series, method: str) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    if bool(np.isinf(p.to_numpy(dtype=float, na_value=np.nan)).any()):
        raise ValueError("Non-finite infinite p-value cannot enter multiplicity correction.")
    corrected = pd.Series([math.nan] * len(p), index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty or method == "none":
        corrected.loc[valid.index] = valid
        return corrected
    m = len(valid)
    if method == "bonferroni":
        corrected.loc[valid.index] = (valid * m).clip(upper=1.0)
        return corrected
    ordered = valid.sort_values()
    if method in {"benjamini-hochberg", "bh", "fdr_bh"}:
        values = (ordered * m / pd.Series(range(1, m + 1), index=ordered.index)).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
        corrected.loc[values.index] = values
        return corrected
    adjusted = []
    for rank, (_, value) in enumerate(ordered.items()):
        adjusted.append(min(1.0, value * (m - rank)))
    values = pd.Series(adjusted, index=ordered.index).cummax().clip(upper=1.0)
    corrected.loc[values.index] = values
    return corrected


def _finite_numeric_values(values: pd.Series, *, label: str, allow_missing: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float, na_value=np.nan)
    invalid = ~np.isfinite(array)
    if allow_missing:
        invalid &= ~np.isnan(array)
    if bool(invalid.any()):
        examples = [values.iloc[index] for index in np.flatnonzero(invalid)[:5]]
        raise ValueError(f"Non-finite {label}: {examples!r}")
    return numeric.dropna().astype(float)


def apply_multiplicity_families(stats_table: pd.DataFrame, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Apply Holm within each finite PRIMARY hypothesis family only."""

    table = stats_table.copy()
    alpha = float(settings.get("alpha", 0.05))
    table["raw_p"] = pd.to_numeric(table.get("p_value"), errors="coerce")
    if bool(np.isinf(table["raw_p"].to_numpy(dtype=float, na_value=np.nan)).any()):
        raise ValueError("Non-finite infinite p-value cannot enter a multiplicity family.")
    table["adjusted_p"] = table["raw_p"]
    table["p_corrected"] = table["raw_p"]
    table["correction_method"] = "none_not_primary_family"
    table["correction_rank"] = pd.NA
    table["correction_family_size"] = pd.NA
    table["correction_valid_tests"] = pd.NA
    table["alpha"] = alpha
    primary = table.get("analysis_status", pd.Series("", index=table.index)).astype(str).eq("PRIMARY")
    families = table.get("hypothesis_family", pd.Series("", index=table.index)).fillna("").astype(str)
    for family in sorted(set(families[primary]) - {""}):
        family_indices = table.index[primary & families.eq(family)]
        valid_indices = family_indices[table.loc[family_indices, "raw_p"].notna()]
        if not len(family_indices):
            continue
        family_p = table.loc[family_indices, "raw_p"].fillna(1.0).astype(float)
        try:
            from statsmodels.stats.multitest import multipletests

            _, adjusted, _, _ = multipletests(family_p.to_numpy(), alpha=alpha, method="holm")
            adjusted_series = pd.Series(adjusted, index=family_indices)
        except Exception:
            adjusted_series = correct_p_values(family_p, "holm")
        table.loc[valid_indices, "adjusted_p"] = adjusted_series.loc[valid_indices]
        table.loc[valid_indices, "p_corrected"] = adjusted_series.loc[valid_indices]
        table.loc[family_indices, "correction_method"] = "holm"
        table.loc[family_indices, "correction_family_size"] = len(family_indices)
        table.loc[family_indices, "correction_valid_tests"] = len(valid_indices)
        ordered = family_p.sort_values(kind="stable")
        table.loc[ordered.index, "correction_rank"] = list(range(1, len(ordered) + 1))
    table["reject_null"] = table["adjusted_p"].lt(alpha) & table["adjusted_p"].notna()
    table["significant"] = table["reject_null"]
    table["multiple_comparison_correction"] = table["correction_method"]
    return table


def build_statistics_summary(stats_table: pd.DataFrame) -> pd.DataFrame:
    if stats_table.empty:
        return pd.DataFrame(columns=["comparison", "n_tests", "n_computed", "n_degenerate", "n_significant"])
    return (
        stats_table.groupby("comparison", dropna=False)
        .agg(
            n_tests=("comparison", "size"),
            n_computed=("status", lambda values: int((values == "computed").sum())),
            n_degenerate=("status", lambda values: int(pd.Series(values).astype(str).str.startswith("degenerate_").sum())),
            n_significant=("significant", lambda values: int(pd.Series(values).fillna(False).sum())),
        )
        .reset_index()
    )


def build_statistics_qc_summary(branch: str, participant_summary: pd.DataFrame, measurements: pd.DataFrame, stats_table: pd.DataFrame, assumptions_table: pd.DataFrame, qc_rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> pd.DataFrame:
    include_col = f"{branch}_included"
    included = int(_as_bool_series(participant_summary[include_col]).sum()) if include_col in participant_summary.columns else 0
    excluded = max(0, len(participant_summary) - included)
    assumption_counts = assumption_qc_counts(assumptions_table)
    degenerate_stats = int(stats_table.get("status", pd.Series(dtype=str)).astype(str).str.startswith("degenerate_").sum()) if not stats_table.empty else 0
    return pd.DataFrame(
        [
            {
                "branch": branch,
                "participants_analysed": int(measurements["subject"].nunique()) if "subject" in measurements.columns else 0,
                "participants_included_in_gate5a": included,
                "participants_excluded_in_gate5a": excluded,
                "statistical_test_selected": str(settings.get("paired_test", "paired_t")),
                "fallback_paired_test": str(settings.get("fallback_paired_test", "wilcoxon")),
                "normality_test_selected": str(settings.get("normality_test", "shapiro")),
                "minimum_normality_n": int(settings.get("minimum_normality_n", 3)),
                "multiple_comparison_method": str(settings.get("multiple_comparison_correction", "holm")),
                "alpha": float(settings.get("alpha", 0.05)),
                "technical_minimum_pairs": int(settings.get("technical_minimum_pairs", settings.get("minimum_pairs", 2))),
                "scientific_interpretation_minimum_n": settings.get("scientific_interpretation_minimum_n", None),
                "interpretation_status": "PROVISIONAL_PIPELINE_VALIDATION_UNTIL_FINAL_SAMPLE",
                "n_results": int(len(stats_table)),
                "n_computed": int((stats_table.get("status", pd.Series(dtype=str)) == "computed").sum()) if not stats_table.empty else 0,
                "n_degenerate_results": degenerate_stats,
                "missing_comparisons": [dict(row) for row in qc_rows],
                **assumption_counts,
                "normality_skipped_small_sample": assumption_counts["n_assumption_checks_insufficient_n"],
                "failed_assumption_checks": assumption_counts["n_assumption_checks_failed"],
            }
        ]
    )


def assumption_qc_counts(assumptions_table: pd.DataFrame) -> dict[str, int]:
    keys = {
        "n_assumption_checks": 0,
        "n_assumption_checks_passed": 0,
        "n_assumption_checks_failed": 0,
        "n_assumption_checks_skipped": 0,
        "n_assumption_checks_insufficient_n": 0,
        "n_assumption_checks_degenerate": 0,
    }
    if assumptions_table.empty:
        return keys
    passed = assumptions_table.get("passed", pd.Series([pd.NA] * len(assumptions_table)))
    passed_bool = passed.map(_as_bool)
    reason = assumptions_table.get("reason", pd.Series([""] * len(assumptions_table))).fillna("").astype(str)
    method = assumptions_table.get("normality_test", pd.Series([""] * len(assumptions_table))).fillna("").astype(str)
    skipped = reason.ne("") | method.isin(["none", "not_applicable"]) | passed.isna()
    keys["n_assumption_checks"] = int(len(assumptions_table))
    keys["n_assumption_checks_passed"] = int((passed_bool & ~skipped).sum())
    keys["n_assumption_checks_failed"] = int((~passed_bool & ~skipped).sum())
    keys["n_assumption_checks_skipped"] = int(skipped.sum())
    keys["n_assumption_checks_insufficient_n"] = int(reason.str.contains("below the configured minimum|insufficient observations", case=False, regex=True).sum())
    keys["n_assumption_checks_degenerate"] = int(reason.str.contains("zero range|degenerate", case=False, regex=True).sum())
    return keys


def write_statistics_figures(branch: str, stats_table: pd.DataFrame, effects_table: pd.DataFrame, measurements: pd.DataFrame, paths: Mapping[str, Path], *, figure_style: FigureStyle | None = None) -> dict[str, Path]:
    style = figure_style or default_figure_style()
    outputs: dict[str, Path] = {}
    cells = prepare_scientific_plot_cells(branch, measurements)
    for index, cell in enumerate(cells):
        mean_path = paths["figures"] / f"{branch}_mean_sem_{cell.slug}.png"
        box_path = paths["figures"] / f"{branch}_boxplots_{cell.slug}.png"
        mean_label = f"mean_sem_figure__{cell.slug}"
        box_label = f"boxplot_figure__{cell.slug}"
        outputs[mean_label] = _write_mean_sem_cell(cell, mean_path, figure_style=style)
        outputs[box_label] = _write_boxplot_cell(cell, box_path, figure_style=style)
        # Compatibility labels now point to the first explicitly cell-scoped figure,
        # never to a heterogeneous condition-only aggregate.
        if index == 0:
            outputs["mean_sem_figure"] = outputs[mean_label]
            outputs["boxplot_figure"] = outputs[box_label]
    if not effects_table.empty:
        effect_cells = effects_table.dropna(subset=["roi", "time_window"]).copy()
        for index, ((roi, time_window), cell_effects) in enumerate(
            effect_cells.groupby(["roi", "time_window"], sort=True)
        ):
            slug = f"{_figure_slug(roi)}__{_figure_slug(time_window)}"
            label = f"effect_size_figure__{slug}"
            path = paths["figures"] / f"{branch}_effect_sizes_{slug}.png"
            outputs[label] = write_effect_size_figure(branch, cell_effects, path, figure_style=style)
            if index == 0:
                outputs["effect_size_figure"] = outputs[label]
    return outputs


def prepare_scientific_plot_cells(branch: str, measurements: pd.DataFrame) -> list[PreparedPlotCell]:
    """Prepare participant-unit values without mixing ROI/time-window cells."""

    required = {"subject", "condition", "roi", "time_window", "mean_amplitude_uv"}
    if measurements.empty:
        return []
    missing = sorted(required.difference(measurements.columns))
    if missing:
        raise ValueError(f"Gate 5C figure preparation lacks required columns: {', '.join(missing)}")
    values = _participant_level_figure_values(measurements)
    cells: list[PreparedPlotCell] = []
    for (roi, time_window), cell_values in values.groupby(["roi", "time_window"], sort=True):
        cell_values = cell_values.sort_values(["condition", "subject"]).reset_index(drop=True)
        summary = (
            cell_values.groupby("condition", sort=True)["participant_value"]
            .agg(mean="mean", sem=lambda series: float(series.sem()) if len(series) > 1 else math.nan, n="count")
            .reset_index()
        )
        cells.append(
            PreparedPlotCell(
                branch=str(branch),
                roi=str(roi),
                time_window=str(time_window),
                values=cell_values,
                summary=summary,
                participant_n=int(cell_values["subject"].nunique()),
            )
        )
    return cells


def _participant_level_figure_values(measurements: pd.DataFrame) -> pd.DataFrame:
    frame = measurements.copy()
    frame["subject"] = frame["subject"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    frame["roi"] = frame["roi"].astype(str)
    frame["time_window"] = frame["time_window"].astype(str)
    frame["mean_amplitude_uv"] = pd.to_numeric(frame["mean_amplitude_uv"], errors="coerce")
    frame = frame.dropna(subset=["mean_amplitude_uv"])
    frame = frame[np.isfinite(frame["mean_amplitude_uv"].to_numpy(dtype=float))]
    key = ["subject", "condition", "roi", "time_window"]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(key, sort=True, dropna=False):
        values = group["mean_amplitude_uv"].to_numpy(dtype=float)
        if "n_epochs" in group.columns:
            weights = pd.to_numeric(group["n_epochs"], errors="coerce").to_numpy(dtype=float)
            usable = np.isfinite(weights) & (weights > 0)
        else:
            weights = np.ones(len(group), dtype=float)
            usable = np.ones(len(group), dtype=bool)
        if not usable.any():
            continue
        value = float(np.average(values[usable], weights=weights[usable]))
        rows.append(dict(zip(key, keys)) | {"participant_value": value})
    return pd.DataFrame(rows, columns=[*key, "participant_value"])


def write_mean_sem_figure(branch: str, measurements: pd.DataFrame, path: Path, *, figure_style: FigureStyle | None = None) -> Path:
    """Render one cell; heterogeneous ROI/window inputs are rejected."""

    cells = prepare_scientific_plot_cells(branch, measurements)
    if len(cells) != 1:
        raise ValueError("Mean/SEM figures require exactly one compatible ROI/time-window cell.")
    return _write_mean_sem_cell(cells[0], path, figure_style=figure_style)


def _write_mean_sem_cell(cell: PreparedPlotCell, path: Path, *, figure_style: FigureStyle | None = None) -> Path:
    import matplotlib.pyplot as plt

    summary = cell.summary
    path.parent.mkdir(parents=True, exist_ok=True)
    style = figure_style or default_figure_style()
    fig, ax = plt.subplots(figsize=style.figure_size("statistics"), facecolor=style.figure_facecolor)
    labels = [f"{style.condition_label(str(condition))}\nN={int(n)}" for condition, n in zip(summary["condition"], summary["n"])]
    ax.bar(labels, summary["mean"], yerr=summary["sem"].fillna(0.0), capsize=4, color=style.bar_color())
    ax.set_title(f"{style.branch_label(cell.branch)}: mean ± SEM · {str(cell.roi).replace('_', ' ').title()} · {_figure_window_label(cell.time_window)}")
    ax.set_ylabel("Mean amplitude (µV)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    style_axes(ax, style)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def write_boxplot_figure(branch: str, measurements: pd.DataFrame, path: Path, *, figure_style: FigureStyle | None = None) -> Path:
    cells = prepare_scientific_plot_cells(branch, measurements)
    if len(cells) != 1:
        raise ValueError("Boxplot figures require exactly one compatible ROI/time-window cell.")
    return _write_boxplot_cell(cells[0], path, figure_style=figure_style)


def _write_boxplot_cell(cell: PreparedPlotCell, path: Path, *, figure_style: FigureStyle | None = None) -> Path:
    import matplotlib.pyplot as plt

    conditions = list(cell.summary["condition"].astype(str))
    values = [cell.values.loc[cell.values["condition"] == condition, "participant_value"].dropna() for condition in conditions]
    path.parent.mkdir(parents=True, exist_ok=True)
    style = figure_style or default_figure_style()
    fig, ax = plt.subplots(figsize=style.figure_size("statistics"), facecolor=style.figure_facecolor)
    labels = [f"{style.condition_label(condition)}\nN={len(value)}" for condition, value in zip(conditions, values)]
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_title(f"{style.branch_label(cell.branch)}: participant distributions · {str(cell.roi).replace('_', ' ').title()} · {_figure_window_label(cell.time_window)}")
    ax.set_ylabel("Mean amplitude (µV)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    style_axes(ax, style)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def write_effect_size_figure(branch: str, effects_table: pd.DataFrame, path: Path, *, figure_style: FigureStyle | None = None) -> Path:
    import matplotlib.pyplot as plt

    if "effect_size_value" not in effects_table.columns:
        return path
    computed = effects_table.copy()
    cells = computed[["roi", "time_window"]].drop_duplicates() if {"roi", "time_window"}.issubset(computed.columns) else pd.DataFrame()
    if len(cells) > 1:
        raise ValueError("Effect-size figures require exactly one compatible ROI/time-window cell.")
    computed["effect_size_value"] = pd.to_numeric(computed["effect_size_value"], errors="coerce")
    computed = computed.dropna(subset=["effect_size_value"])
    if computed.empty:
        return path
    labels = (computed["comparison"].astype(str) + " | " + computed["roi"].astype(str) + " | " + computed["time_window"].astype(str)).head(30)
    values = computed["effect_size_value"].head(30)
    path.parent.mkdir(parents=True, exist_ok=True)
    style = figure_style or default_figure_style()
    fig, ax = plt.subplots(figsize=style.figure_size("effect_size", count=len(labels)), facecolor=style.figure_facecolor)
    ax.barh(labels, values, color=style.bar_color("effect_size"))
    ax.axvline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"])
    cell_label = ""
    if not cells.empty:
        cell = cells.iloc[0]
        cell_label = f" · {str(cell['roi']).replace('_', ' ').title()} · {_figure_window_label(str(cell['time_window']))}"
    ax.set_title(f"{style.branch_label(branch)}: effect sizes{cell_label}")
    ax.set_xlabel("Effect size")
    fig.tight_layout()
    style_axes(ax, style)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def _figure_slug(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(character if character.isalnum() else "_" for character in text).strip("_") or "unspecified"


def _figure_window_label(value: str) -> str:
    match = re.fullmatch(r"(\d+)_(\d+)_ms", str(value))
    if match:
        return f"{match.group(1)}–{match.group(2)} ms"
    return str(value).replace("_", " ")


def write_statistics_report(branch: str, path: Path, output_files: Mapping[str, Path], stats_table: pd.DataFrame, assumptions_table: pd.DataFrame, qc: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Gate 5C {branch.title()} Statistics",
        "",
        "## Scope",
        "- Conventional participant-level ROI/time-window statistics from Gate 5A aggregate CSVs only."
        if branch != "cuteness"
        else "- Primary cuteness inference uses the Gate 5A retained-trial table and the frozen crossed participant/item random-intercept model.",
        "- No raw EEG, epochs, or evoked FIF files were loaded.",
        "- No cluster-based permutation statistics or mass-univariate analyses were run.",
        "- Until the final scientifically eligible sample is available, group inferential outputs are provisional pipeline-validation results rather than substantive population evidence.",
        "",
        "## QC",
    ]
    if not qc.empty:
        lines.extend(f"- {key}: {value}" for key, value in qc.iloc[0].to_dict().items())
    small_sample_reason = "Normality test skipped because sample size was below the configured minimum."
    small_sample_skips = int(assumptions_table["reason"].astype(str).eq(small_sample_reason).sum()) if "reason" in assumptions_table.columns else 0
    if small_sample_skips:
        lines.append(f"- {small_sample_reason}")
    lines.extend(["", "## Results"])
    if stats_table.empty:
        lines.append("- no legacy participant-level comparison rows generated")
    else:
        lines.append(f"- rows: {len(stats_table)}")
        lines.append(f"- computed: {int((stats_table['status'] == 'computed').sum())}")
        lines.append(f"- significant after correction: {int(stats_table['significant'].fillna(False).sum())}")
    if branch == "cuteness" and "mixed_model_diagnostics" in output_files:
        model_diagnostics = _read_csv(output_files["mixed_model_diagnostics"])
        model_status = str(model_diagnostics.get("model_status", pd.Series(["UNKNOWN"])).iloc[0]) if not model_diagnostics.empty else "UNKNOWN"
        lines.append(f"- primary mixed-model status: {model_status}")
        lines.extend(["", "## Model history"])
        attempts_path = output_files.get("mixed_model_attempts")
        attempts = _read_csv(attempts_path) if attempts_path is not None else pd.DataFrame()
        if attempts.empty:
            lines.append("- attempt ledger unavailable")
        else:
            for _, attempt in attempts.sort_values("attempt_index").iterrows():
                selected = "selected" if bool(_as_bool(attempt.get("selected_for_final_result", False))) else "not selected"
                status = str(attempt.get("status", "UNKNOWN"))
                role = str(attempt.get("model_role", "unknown"))
                formula = str(attempt.get("formula", ""))
                lines.append(f"- attempt {int(attempt.get('attempt_index', 0)) + 1}: {role} — {status} — {selected} — `{formula}`")
                reason = str(attempt.get("reason", "") or "")
                warning = str(attempt.get("warnings", "") or "")
                if reason:
                    lines.append(f"  - reason: {reason}")
                if warning:
                    lines.append(f"  - warning: {warning}")
            if not model_diagnostics.empty:
                fallback = _as_bool(model_diagnostics.get("fallback_occurred", pd.Series([False])).iloc[0])
                lines.append(f"- fallback occurred: {fallback}")
                fallback_reason = str(model_diagnostics.get("fallback_reason", pd.Series([""])).iloc[0] or "")
                if fallback_reason:
                    lines.append(f"- fallback reason: {fallback_reason}")
                lines.append(f"- authoritative ledger: `{attempts_path}`")
    lines.extend(["", "## Outputs"])
    lines.extend(f"- {label}: `{path_value}`" for label, path_value in output_files.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_gate5c_provenance(
    config: Mapping[str, Any],
    branch: str,
    output_files: Mapping[str, Path],
    input_files: Mapping[str, Path],
    participant_summary: pd.DataFrame,
    warnings: Sequence[str],
    *,
    root: Path | None,
    config_path: str | Path | None = None,
) -> None:
    included, excluded, reasons = _summary_subject_sets(participant_summary, branch)
    existing_inputs = [path for path in input_files.values() if path.exists()]
    for output in output_files.values():
        write_group_provenance_sidecar(
            output,
            gate="gate5c",
            branch=branch,
            root=root,
            input_files=existing_inputs,
            included_subjects=included,
            excluded_subjects=excluded,
            exclusion_reasons=reasons,
            parameters_summary={
                "statistics": dict(statistics_settings(config)),
                "primary_model": dict(dict(dict(config.get("erp", {}) or {}).get("cuteness", {}) or {}).get("primary_model", {}) or {}) if branch == "cuteness" else {},
            },
            warnings=warnings,
            **({"config_path": config_path} if config_path is not None else {}),
        )


def _summary_subject_sets(participant_summary: pd.DataFrame, branch: str) -> tuple[list[str], list[str], dict[str, str]]:
    include_col = f"{branch}_included"
    reason_col = f"{branch}_reason"
    if participant_summary.empty or "subject" not in participant_summary.columns or include_col not in participant_summary.columns:
        return [], [], {}
    included_mask = _as_bool_series(participant_summary[include_col])
    included = sorted(participant_summary.loc[included_mask, "subject"].dropna().astype(str).unique().tolist())
    excluded = sorted(participant_summary.loc[~included_mask, "subject"].dropna().astype(str).unique().tolist())
    reasons = {}
    if reason_col in participant_summary.columns:
        for _, row in participant_summary.loc[~included_mask].iterrows():
            subject = str(row.get("subject", ""))
            if subject:
                reasons[subject] = str(row.get(reason_col, ""))
    return included, excluded, reasons


def cohens_d(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    sd = numeric.std(ddof=1)
    if len(numeric) < 2 or sd == 0 or pd.isna(sd):
        return math.nan
    return float(numeric.mean() / sd)


def rank_biserial_correlation(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    numeric = numeric[numeric != 0]
    if numeric.empty:
        return math.nan
    ranks = numeric.abs().rank(method="average")
    positive = float(ranks[numeric > 0].sum())
    negative = float(ranks[numeric < 0].sum())
    total = len(numeric) * (len(numeric) + 1) / 2.0
    return (positive - negative) / total if total else math.nan


def _statistics_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["analysis_type", "comparison", "condition_a", "condition_b", "roi", "time_window", "analysis_status", "hypothesis_family", "hypothesis_id", "test", "test_name", "n", "technical_n", "scientifically_interpretable_n", "interpretation_status", "mean_a", "sd_a", "mean_b", "sd_b", "mean_difference", "effect_estimate", "sd_difference", "median_difference", "minimum_difference", "maximum_difference", "effect_size", "effect_size_metric", "ci_low", "ci_high", "confidence_level", "assumption_status", "statistic", "p_value", "raw_p", "adjusted_p", "p_corrected", "correction_method", "correction_rank", "correction_family_size", "correction_valid_tests", "reject_null", "significant", "alpha", "multiple_comparison_correction", "status", "reason"])


def _effect_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["analysis_type", "comparison", "condition_a", "condition_b", "roi", "time_window", "analysis_status", "hypothesis_family", "hypothesis_id", "effect_size", "effect_size_value", "n"])


def _assumption_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["comparison", "roi", "time_window", "analysis_status", "hypothesis_family", "hypothesis_id", "normality_test", "n", "statistic", "p_value", "passed", "reason"])


def _as_bool_series(series: pd.Series) -> pd.Series:
    return series.map(_as_bool).astype(bool)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "included"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _mean(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").dropna().mean()) if len(pd.to_numeric(values, errors="coerce").dropna()) else math.nan


def _sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0 if len(numeric) == 1 else math.nan


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if len(numeric) else math.nan


def _min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.min()) if len(numeric) else math.nan


def _max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if len(numeric) else math.nan
