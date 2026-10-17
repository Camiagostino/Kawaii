from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from .bad_channel_recalibration import (
    baseline_all_channel_abs_scores,
    median_correlation_scores,
    qc_only_filter_and_median_reference,
    within_window_low_robust_z,
)


FOCAL_SEVERITIES: dict[str, dict[str, float]] = {
    "broadband_high_frequency_noise": {"mild": 0.5, "moderate": 1.0, "strong": 2.0},
    "line_noise_50_hz": {"mild": 0.5, "moderate": 1.0, "strong": 2.0},
    "amplitude_scaling": {"mild": 1.5, "moderate": 2.5, "strong": 5.0},
    "partial_flatness": {"mild": 0.1, "moderate": 0.3, "strong": 0.6},
    "slow_drift_0_2_hz": {"mild": 1.0, "moderate": 2.0, "strong": 4.0},
}

GLOBAL_CONTROLS: dict[str, float] = {
    "shared_broadband_noise": 1.0,
    "montage_wide_independent_broadband_noise": 1.0,
    "shared_line_noise_50_hz": 1.0,
    "common_amplitude_scaling": 2.5,
}

CANDIDATE_METHOD_COLUMNS = (
    "production_all_channel_rule",
    "local_absolute_persistent",
    "local_robust_relative_persistent",
    "approved_independent_any",
    "line_noise_qc_flag",
    "local_absolute_plus_independent",
    "local_relative_plus_independent",
    "local_absolute_plus_any_independent_qc",
)


def inject_focal_corruption(
    data: np.ndarray,
    *,
    target_index: int,
    corruption_type: str,
    severity: str,
    sfreq: float,
    seed: int,
) -> np.ndarray:
    """Return a copy with one scientifically transparent focal corruption."""

    if corruption_type not in FOCAL_SEVERITIES or severity not in FOCAL_SEVERITIES[corruption_type]:
        raise ValueError(f"Unsupported focal corruption: {corruption_type}/{severity}")
    values = np.asarray(data, dtype=float).copy()
    if not 0 <= int(target_index) < values.shape[0]:
        raise IndexError("target_index is outside the channel axis")
    factor = FOCAL_SEVERITIES[corruption_type][severity]
    reference_rms = _reference_rms(values)
    rng = np.random.default_rng(int(seed))
    target = values[int(target_index)].copy()
    if corruption_type == "broadband_high_frequency_noise":
        noise = _band_limited_noise(rng, target.size, sfreq, 35.0, min(100.0, sfreq / 2.0 - 1.0))
        target += _normalise_rms(noise, factor * reference_rms)
    elif corruption_type == "line_noise_50_hz":
        target += _sinusoid(target.size, sfreq, 50.0, factor * reference_rms, rng)
    elif corruption_type == "amplitude_scaling":
        target *= factor
    elif corruption_type == "partial_flatness":
        length = max(1, int(round(target.size * factor)))
        start = (target.size - length) // 2
        target[start : start + length] = float(np.median(target))
    elif corruption_type == "slow_drift_0_2_hz":
        target += _sinusoid(target.size, sfreq, 0.2, factor * reference_rms, rng)
    values[int(target_index)] = target
    return values


def inject_global_perturbation(
    data: np.ndarray,
    *,
    perturbation_type: str,
    sfreq: float,
    seed: int,
) -> np.ndarray:
    """Return a copy with a montage-wide control that has no focal target."""

    if perturbation_type not in GLOBAL_CONTROLS:
        raise ValueError(f"Unsupported global perturbation: {perturbation_type}")
    values = np.asarray(data, dtype=float).copy()
    factor = GLOBAL_CONTROLS[perturbation_type]
    reference_rms = _reference_rms(values)
    rng = np.random.default_rng(int(seed))
    if perturbation_type == "shared_broadband_noise":
        common = _normalise_rms(rng.normal(size=values.shape[1]), factor * reference_rms)
        values += common[None, :]
    elif perturbation_type == "montage_wide_independent_broadband_noise":
        noise = rng.normal(size=values.shape)
        noise *= (factor * reference_rms) / np.maximum(np.sqrt(np.mean(noise**2, axis=1, keepdims=True)), np.finfo(float).tiny)
        values += noise
    elif perturbation_type == "shared_line_noise_50_hz":
        values += _sinusoid(values.shape[1], sfreq, 50.0, factor * reference_rms, rng)[None, :]
    elif perturbation_type == "common_amplitude_scaling":
        values *= factor
    return values


def evaluate_controlled_windows(
    windows: Sequence[np.ndarray],
    *,
    channels: Sequence[str],
    neighbours: Mapping[str, Sequence[str]],
    sfreq: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed candidate rules on three controlled windows."""

    names = list(map(str, channels))
    if len(windows) != 3:
        raise ValueError("controlled validation requires exactly three distributed windows")
    production_scores, local_scores, relative_scores = [], [], []
    window_rows: list[dict[str, object]] = []
    for window_index, data in enumerate(windows, start=1):
        values = np.asarray(data, dtype=float)
        if values.shape[0] != len(names):
            raise ValueError("window channel count does not match channels")
        production, _ = baseline_all_channel_abs_scores(values, names)
        filtered = qc_only_filter_and_median_reference(values, sfreq)
        local, peer_counts = median_correlation_scores(
            filtered, names, neighbours=neighbours, absolute=True, minimum_peers=2
        )
        relative = within_window_low_robust_z(local)
        production_scores.append(production)
        local_scores.append(local)
        relative_scores.append(relative)
        finite_local = np.asarray([value for value in local.values() if np.isfinite(value)])
        window_rows.append({
            "window_index": window_index,
            "local_valid_channel_count": int(len(finite_local)),
            "local_window_median": float(np.median(finite_local)) if len(finite_local) else np.nan,
            "local_broad_window": bool(len(finite_local) and np.median(finite_local) < 0.4),
            "local_relative_outlier_count": int(sum(np.isfinite(value) and value >= 3.0 for value in relative.values())),
            "production_below_0_4_count": int(sum(np.isfinite(value) and value < 0.4 for value in production.values())),
            "local_below_0_4_count": int(sum(np.isfinite(value) and value < 0.4 for value in local.values())),
        })
    independent = _approved_independent_metrics(windows, names, sfreq=sfreq)
    rows: list[dict[str, object]] = []
    for channel_index, channel in enumerate(names):
        production_failures = sum(np.isfinite(scores[channel]) and scores[channel] < 0.4 for scores in production_scores)
        local_failures = sum(np.isfinite(scores[channel]) and scores[channel] < 0.4 for scores in local_scores)
        relative_failures = sum(np.isfinite(scores[channel]) and scores[channel] >= 3.0 for scores in relative_scores)
        independent_any = bool(independent.loc[channel_index, "approved_independent_any"])
        any_independent_qc = bool(independent.loc[channel_index, "approved_independent_or_line"])
        row = {
            "channel": channel,
            "production_failed_window_count": production_failures,
            "local_failed_window_count": local_failures,
            "local_relative_failed_window_count": relative_failures,
            "production_all_channel_rule": production_failures >= 1,
            "local_absolute_persistent": local_failures >= 2,
            "local_robust_relative_persistent": relative_failures >= 2,
            "approved_independent_any": independent_any,
            "line_noise_qc_flag": bool(independent.loc[channel_index, "line_noise_qc_flag"]),
            "local_absolute_plus_independent": local_failures >= 2 and independent_any,
            "local_relative_plus_independent": relative_failures >= 2 and independent_any,
            "local_absolute_plus_any_independent_qc": local_failures >= 2 and any_independent_qc,
            "local_scores_by_window": ";".join(_format_float(scores[channel]) for scores in local_scores),
            "local_relative_z_by_window": ";".join(_format_float(scores[channel]) for scores in relative_scores),
            "finite_neighbour_count": len(neighbours.get(channel, ())),
        }
        row.update(independent.iloc[channel_index].to_dict())
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(window_rows)


def scenario_performance(
    evidence: pd.DataFrame,
    *,
    truth_type: str,
    target_channel: str | None,
    neighbours: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Summarize sensitivity, false positives, and focal localization per method."""

    rows: list[dict[str, object]] = []
    for method in CANDIDATE_METHOD_COLUMNS:
        proposed = set(evidence.loc[evidence[method].astype(bool), "channel"].astype(str))
        target_detected = bool(target_channel is not None and target_channel in proposed)
        unaffected_denominator = len(evidence) - (1 if target_channel is not None else 0)
        false_positives = len(proposed - ({target_channel} if target_channel else set()))
        specificity = 1.0 - false_positives / unaffected_denominator if unaffected_denominator else np.nan
        if truth_type == "focal":
            localization = _localization_class(proposed, str(target_channel), neighbours)
            sensitivity = float(target_detected)
            precision = float(target_detected / len(proposed)) if proposed else np.nan
        else:
            localization = "not_applicable_no_focal_target"
            sensitivity = precision = np.nan
        rows.append({
            "method": method,
            "candidate_count": len(proposed),
            "candidate_channels": ";".join(sorted(proposed)),
            "target_detected": target_detected,
            "sensitivity": sensitivity,
            "false_positive_count": false_positives,
            "false_positive_rate": false_positives / unaffected_denominator if unaffected_denominator else np.nan,
            "specificity": specificity,
            "precision": precision,
            "balanced_accuracy": (sensitivity + specificity) / 2 if np.isfinite(sensitivity) else np.nan,
            "localization_class": localization,
        })
    return pd.DataFrame(rows)


def aggregate_performance(per_scenario: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Aggregate scenario-level performance without inferential statistics."""

    rows: list[dict[str, object]] = []
    keys = [*group_columns, "method"]
    for values, group in per_scenario.groupby(keys, sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row.update({
            "scenario_count": len(group),
            "sensitivity": pd.to_numeric(group["sensitivity"], errors="coerce").mean(),
            "mean_false_positive_rate": pd.to_numeric(group["false_positive_rate"], errors="coerce").mean(),
            "median_candidate_count": pd.to_numeric(group["candidate_count"], errors="coerce").median(),
            "mean_specificity": pd.to_numeric(group["specificity"], errors="coerce").mean(),
            "mean_precision": pd.to_numeric(group["precision"], errors="coerce").mean(),
            "mean_balanced_accuracy": pd.to_numeric(group["balanced_accuracy"], errors="coerce").mean(),
        })
        if "incremental_target_detection" in group:
            row["incremental_sensitivity_vs_unchanged"] = pd.to_numeric(
                group["incremental_target_detection"], errors="coerce"
            ).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def build_blinded_review_tables(selection: pd.DataFrame, *, seed: int = 20260912) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a randomized public manifest, blank response sheet, and held-back key."""

    required = {"participant", "channel", "selection_stratum", "edge_status", "automatic_candidate_status", "figure_path"}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"Review selection is missing columns: {sorted(missing)}")
    randomized = selection.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True).copy()
    randomized.insert(0, "case_id", [f"BC{index:03d}" for index in range(1, len(randomized) + 1)])
    public = randomized[["case_id", "participant", "channel", "figure_path"]].copy()
    responses = public[["case_id", "participant", "channel"]].copy()
    responses["review_classification"] = ""
    responses["structured_reasons"] = ""
    responses["reviewer_confidence"] = ""
    responses["reviewer_notes"] = ""
    key = randomized.drop(columns=[])
    return public, responses, key


def _approved_independent_metrics(windows: Sequence[np.ndarray], channels: Sequence[str], *, sfreq: float) -> pd.DataFrame:
    stack = np.concatenate([np.asarray(window, dtype=float) for window in windows], axis=1)
    std_uv = np.nanstd(stack, axis=1) * 1e6
    ptp_uv = (np.nanmax(stack, axis=1) - np.nanmin(stack, axis=1)) * 1e6
    clipping_counts = np.sum((stack == np.nanmax(stack, axis=1, keepdims=True)) | (stack == np.nanmin(stack, axis=1, keepdims=True)), axis=1)
    clipping_fraction = clipping_counts / stack.shape[1]
    deviations = []
    for window in windows:
        log_std = np.log(np.maximum(np.nanstd(window, axis=1) * 1e6, np.finfo(float).tiny))
        centre = float(np.median(log_std))
        scale = max(1.4826 * float(np.median(np.abs(log_std - centre))), 0.05)
        deviations.append((log_std - centre) / scale)
    max_deviation = np.max(np.vstack(deviations), axis=0)
    flat = std_uv <= 0.01
    extreme = (std_uv >= 500.0) | (ptp_uv >= 5000.0)
    clipping = (clipping_counts >= 3) & (clipping_fraction >= 0.001)
    robust = max_deviation >= 5.0
    line_ratios = []
    for window in windows:
        frequencies, power = welch(window, fs=float(sfreq), nperseg=min(window.shape[1], int(round(4 * sfreq))), axis=1)
        line = (frequencies >= 49.0) & (frequencies <= 51.0)
        broad = (frequencies >= 40.0) & (frequencies <= 60.0)
        line_power = np.nanmean(power[:, line], axis=1)
        broad_power = np.nanmean(power[:, broad], axis=1)
        line_ratios.append(line_power / np.maximum(broad_power, np.finfo(float).tiny))
    maximum_line_ratio = np.nanmax(np.vstack(line_ratios), axis=0)
    line_flag = maximum_line_ratio > 0.5
    return pd.DataFrame({
        "channel": list(channels),
        "std_uv": std_uv,
        "peak_to_peak_uv": ptp_uv,
        "clipping_fraction": clipping_fraction,
        "max_robust_deviation_z": max_deviation,
        "flat_proposal": flat,
        "extreme_amplitude_proposal": extreme,
        "clipping_proposal": clipping,
        "robust_deviation_proposal": robust,
        "approved_independent_any": flat | extreme | clipping | robust,
        "maximum_window_line_noise_ratio": maximum_line_ratio,
        "line_noise_qc_flag": line_flag,
        "approved_independent_or_line": flat | extreme | clipping | robust | line_flag,
    })


def _localization_class(proposed: set[str], target: str, neighbours: Mapping[str, Sequence[str]]) -> str:
    if target not in proposed:
        return "D_missed_target"
    if proposed == {target}:
        return "A_target_only"
    extras = proposed - {target}
    neighbour_set = set(neighbours.get(target, ()))
    if extras and extras.issubset(neighbour_set):
        return "B_target_plus_neighbours"
    if len(proposed) >= 12:
        return "C_broad_montage_failure"
    return "E_target_plus_sparse_nonlocal"


def _reference_rms(values: np.ndarray) -> float:
    rms = np.sqrt(np.nanmean(np.asarray(values, dtype=float) ** 2, axis=1))
    reference = float(np.nanmedian(rms))
    return reference if np.isfinite(reference) and reference > 0 else np.finfo(float).eps


def _normalise_rms(values: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))
    return np.asarray(values, dtype=float) * (float(target_rms) / max(rms, np.finfo(float).tiny))


def _sinusoid(length: int, sfreq: float, frequency: float, rms: float, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(length, dtype=float) / float(sfreq)
    phase = float(rng.uniform(0, 2 * np.pi))
    return np.sqrt(2.0) * float(rms) * np.sin(2 * np.pi * float(frequency) * time + phase)


def _band_limited_noise(rng: np.random.Generator, length: int, sfreq: float, low: float, high: float) -> np.ndarray:
    sos = butter(4, [float(low), float(high)], btype="bandpass", fs=float(sfreq), output="sos")
    return sosfiltfilt(sos, rng.normal(size=int(length)))


def _format_float(value: float) -> str:
    return "" if not np.isfinite(value) else f"{float(value):.9g}"
