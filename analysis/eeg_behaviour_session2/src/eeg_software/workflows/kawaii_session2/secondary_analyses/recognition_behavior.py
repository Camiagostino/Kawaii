"""Behavioural recognition analyses for Kawaii Session 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from ..erp import (
    condition_mask,
    gate4a_recognition_settings,
    require_recognition_condition_reconciliation,
)
from ..figure_style import load_figure_style, style_axes, style_legend
from ..group_statistics import mean_confidence_interval, normality_check, statistics_settings
from ..source_exposure import apply_source_exposure_disposition
from ..study_definition import configured_conditions
from .config import resolve_path, sha256_file


def _manifest_sha(value: Any) -> str:
    return str(value or "").strip().split(":", 1)[-1]


def _bool_series(values: pd.Series, label: str) -> pd.Series:
    def parse(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        raise ValueError(f"Invalid boolean in {label}: {value!r}")

    return values.map(parse).astype(bool)


def _load_source_manifest(root: Path, core_config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    settings = dict(core_config.get("source_exposure_verification", {}) or {})
    manifest_path = resolve_path(root, str(settings.get("manifest_path", "")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(settings.get("expected_artifact_id", ""))
    if str(manifest.get("artifact_id", "")) != expected:
        raise ValueError(
            "Source-exposure manifest artifact_id does not match the Session 2 config."
        )
    return manifest_path, manifest


def _participant_manifest(manifest: Mapping[str, Any], subject: str) -> dict[str, Any]:
    matches = [dict(v) for v in manifest.get("participants", []) if str(v.get("subject")) == subject]
    if len(matches) != 1:
        raise ValueError(f"Expected one source-exposure manifest record for {subject}; found {len(matches)}.")
    return matches[0]


def _behavioural_log_from_manifest(root: Path, participant_manifest: Mapping[str, Any]) -> Path:
    matches = [
        dict(v)
        for v in participant_manifest.get("sources", [])
        if str(v.get("role")) == "session2_behavioral_log"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one session2_behavioral_log for {participant_manifest.get('subject')}; "
            f"found {len(matches)}."
        )
    identity = matches[0]
    path = resolve_path(root, str(identity.get("logical_path", "")))
    if not path.exists():
        raise FileNotFoundError(f"Behavioural log not found: {path}")
    if path.stat().st_size != int(identity.get("size_bytes", -1)):
        raise ValueError(f"Behavioural log size mismatch for {path}")
    if sha256_file(path) != _manifest_sha(identity.get("sha256")):
        raise ValueError(f"Behavioural log SHA-256 mismatch for {path}")
    return path


def load_source_verified_behavioural_trials(
    root: str | Path,
    core_config: Mapping[str, Any],
    secondary_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Return source-verified recognition-eligible behavioural trials.

    Old/new eligibility comes from the established source-exposure disposition;
    EEG artifact rejection is not used for behavioural memory metrics.
    """

    project_root = Path(root).resolve()
    manifest_path, manifest = _load_source_manifest(project_root, core_config)
    recognition_settings = gate4a_recognition_settings(core_config)
    conditions = configured_conditions(recognition_settings)
    participants = [str(v) for v in secondary_config["participants"]]

    trials: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    input_paths: list[Path] = [manifest_path]

    artifact = dict(manifest.get("artifact", {}) or {})
    disposition_path = resolve_path(manifest_path.parent, str(artifact.get("logical_path", "")))
    input_paths.append(disposition_path)

    for subject in participants:
        participant = _participant_manifest(manifest, subject)
        behavioural_path = _behavioural_log_from_manifest(project_root, participant)
        input_paths.append(behavioural_path)
        metadata = pd.read_csv(behavioural_path)
        result = apply_source_exposure_disposition(
            metadata,
            subject=subject,
            session="session2",
            recording_part="part1",
            config=core_config,
            root=project_root,
        )
        prepared = result.metadata.copy()
        require_recognition_condition_reconciliation(prepared, recognition_settings)

        condition_series = pd.Series(pd.NA, index=prepared.index, dtype="string")
        for condition, spec in conditions.items():
            mask = condition_mask(prepared, recognition_settings, spec, condition)
            condition_series.loc[mask] = str(condition)
        if condition_series.isna().any():
            raise ValueError(f"Unclassified source-verified recognition rows for {subject}.")

        prepared["recognition_condition"] = condition_series
        prepared["participant"] = subject
        eligible = _bool_series(prepared["recognition_eligible"], "recognition_eligible")
        eligible_rows = prepared.loc[eligible].copy().reset_index(drop=True)
        trials.append(eligible_rows)

        counts = eligible_rows["recognition_condition"].value_counts().to_dict()
        expected = dict(participant.get("expected_counts", {}) or {})
        expected_map = {
            "hit": int(expected.get("Hit", -1)),
            "miss": int(expected.get("Miss", -1)),
            "false_alarm": int(expected.get("False Alarm", -1)),
            "correct_rejection": int(expected.get("Correct Rejection", -1)),
        }
        observed_map = {name: int(counts.get(name, 0)) for name in expected_map}
        if observed_map != expected_map:
            raise ValueError(
                f"Source-verified condition counts differ from manifest for {subject}: "
                f"observed={observed_map!r}; expected={expected_map!r}"
            )
        validation_rows.append(
            {
                "subject": subject,
                "behavioural_log": str(behavioural_path.relative_to(project_root)),
                "behavioural_log_sha256": sha256_file(behavioural_path),
                "eligible_n": int(len(eligible_rows)),
                **observed_map,
                "source_exposure_artifact_id": result.summary.get("artifact_id"),
                "status": "PASS",
            }
        )

    return pd.concat(trials, ignore_index=True), pd.DataFrame(validation_rows), input_paths


def compute_signal_detection_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    required = {"participant", "recognition_condition"}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Signal-detection trials lack required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for participant, group in trials.groupby("participant", sort=True):
        counts = group["recognition_condition"].value_counts().to_dict()
        hit = int(counts.get("hit", 0))
        miss = int(counts.get("miss", 0))
        false_alarm = int(counts.get("false_alarm", 0))
        correct_rejection = int(counts.get("correct_rejection", 0))
        old_n = hit + miss
        new_n = false_alarm + correct_rejection
        if old_n <= 0 or new_n <= 0:
            raise ValueError(f"Signal-detection denominator is zero for {participant}.")
        hit_rate = hit / old_n
        false_alarm_rate = false_alarm / new_n
        correction_used = hit_rate in (0.0, 1.0) or false_alarm_rate in (0.0, 1.0)
        hit_rate_for_z = (hit + 0.5) / (old_n + 1) if hit_rate in (0.0, 1.0) else hit_rate
        fa_rate_for_z = (
            (false_alarm + 0.5) / (new_n + 1)
            if false_alarm_rate in (0.0, 1.0)
            else false_alarm_rate
        )
        z_hit = float(stats.norm.ppf(hit_rate_for_z))
        z_fa = float(stats.norm.ppf(fa_rate_for_z))
        rows.append(
            {
                "subject": str(participant),
                "hit": hit,
                "miss": miss,
                "false_alarm": false_alarm,
                "correct_rejection": correct_rejection,
                "old_n": old_n,
                "new_n": new_n,
                "hit_rate": hit_rate,
                "false_alarm_rate": false_alarm_rate,
                "accuracy": (hit + correct_rejection) / (old_n + new_n),
                "yes_response_rate": (hit + false_alarm) / (old_n + new_n),
                "loglinear_correction_used": bool(correction_used),
                "dprime": z_hit - z_fa,
                "criterion_c": -0.5 * (z_hit + z_fa),
            }
        )
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def signal_detection_group_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        ("hit_rate", "Hit rate"),
        ("false_alarm_rate", "False-alarm rate"),
        ("accuracy", "Accuracy"),
        ("yes_response_rate", "Yes-response rate"),
        ("dprime", "d-prime"),
        ("criterion_c", "Criterion c"),
    ]
    rows = []
    for column, label in columns:
        values = pd.to_numeric(metrics[column], errors="raise").astype(float)
        ci_low, ci_high = mean_confidence_interval(values, 0.95)
        rows.append(
            {
                "metric": label,
                "N": int(len(values)),
                "mean": float(values.mean()),
                "SD": float(values.std(ddof=1)),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "CI95_low": ci_low,
                "CI95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def dprime_inference(metrics: pd.DataFrame, core_config: Mapping[str, Any]) -> dict[str, Any]:
    values = pd.to_numeric(metrics["dprime"], errors="raise").astype(float)
    settings = statistics_settings(core_config)
    assumption = normality_check(values, "dprime_vs_zero", "behavioural", "source_verified", settings)
    statistic, p_value = stats.ttest_1samp(values, 0.0)
    sd = float(values.std(ddof=1))
    effect = float(values.mean() / sd) if sd else math.nan
    ci_low, ci_high = mean_confidence_interval(values, float(settings.get("confidence_level", 0.95)))
    return {
        "N": int(len(values)),
        "mean_dprime": float(values.mean()),
        "sd_dprime": sd,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_statistic": float(statistic),
        "df": int(len(values) - 1),
        "p_two_sided": float(p_value),
        "effect_size_metric": "d_z",
        "effect_size": effect,
        "normality_test": assumption.get("normality_test"),
        "normality_W": assumption.get("statistic"),
        "normality_p": assumption.get("p_value"),
        "loglinear_correction_used_any": bool(metrics["loglinear_correction_used"].any()),
    }


def plot_recognition_rates(metrics: pd.DataFrame, path: Path, *, root: Path) -> Path:
    import matplotlib.pyplot as plt

    style = load_figure_style(root=root)
    fig, ax = plt.subplots(figsize=style.figure_size("statistics"), facecolor=style.figure_facecolor)
    x = np.arange(len(metrics))
    hit = metrics["hit_rate"].to_numpy(float)
    fa = metrics["false_alarm_rate"].to_numpy(float)
    reference_color = style.color("reference", "#777777")
    for idx in range(len(metrics)):
        ax.plot([idx, idx], [fa[idx], hit[idx]], color=reference_color, alpha=0.35, linewidth=0.8)
    ax.scatter(
        x - 0.08,
        hit,
        label=style.condition_label("hit"),
        color=style.condition_color("recognition", "hit"),
        s=28,
    )
    ax.scatter(
        x + 0.08,
        fa,
        label=style.condition_label("false_alarm"),
        color=style.condition_color("recognition", "false_alarm"),
        s=28,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["subject"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Response rate")
    ax.set_ylim(0, 1)
    ax.set_title("Source-verified behavioural recognition")
    style_axes(ax, style)
    style_legend(ax, style)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, bbox_inches="tight", facecolor=style.figure_facecolor)
    plt.close(fig)
    return path
