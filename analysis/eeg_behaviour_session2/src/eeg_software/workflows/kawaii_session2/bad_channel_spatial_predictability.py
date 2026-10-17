from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from mne.channels.interpolation import _make_interpolation_matrix

from .bad_channel_controlled_validation import scenario_performance
from .bad_channel_recalibration import qc_only_filter_and_median_reference


PRIMARY_CORRELATION_THRESHOLD = 0.75
PERSISTENT_WINDOW_COUNT = 2
DIAGNOSTIC_THRESHOLDS = (0.70, 0.75, 0.80)

SPATIAL_METHOD_COLUMNS = (
    "spatial_spline_corr_0_75_persistent",
    "spatial_or_approved_independent",
    "spatial_and_approved_independent",
    "spatial_context_gated_diagnostic",
)


def channel_positions_from_info(info, channels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return channel positions and an explicit defensible-geometry mask."""

    positions = []
    valid = []
    for channel in channels:
        index = info["ch_names"].index(str(channel))
        position = np.asarray(info["chs"][index]["loc"][:3], dtype=float)
        is_valid = bool(np.all(np.isfinite(position)) and np.linalg.norm(position) > 0)
        positions.append(position)
        valid.append(is_valid)
    return np.asarray(positions), np.asarray(valid, dtype=bool)


def leave_one_out_spherical_spline_weights(
    positions: np.ndarray,
    geometry_valid: Sequence[bool],
    *,
    alpha: float = 1e-5,
) -> np.ndarray:
    """Build deterministic leave-one-out MNE spherical-spline prediction weights."""

    coordinates = np.asarray(positions, dtype=float)
    valid = np.asarray(geometry_valid, dtype=bool)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(valid) != len(coordinates):
        raise ValueError("positions must be an n_channels x 3 array with a matching validity mask")
    # Zero columns are intentional for geometry-unscorable channels: they are
    # excluded as predictors without poisoning the matrix product with NaNs.
    weights = np.zeros((len(coordinates), len(coordinates)), dtype=float)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 4:
        raise ValueError("at least four coordinate-valid channels are required")
    weights[~valid] = np.nan
    for target in valid_indices:
        predictors = valid_indices[valid_indices != target]
        row = _make_interpolation_matrix(
            coordinates[predictors], coordinates[[target]], alpha=float(alpha)
        )[0]
        weights[target, predictors] = row
        weights[target, target] = 0.0
    return weights


def spatial_predictability_evidence(
    windows: Sequence[np.ndarray],
    *,
    channels: Sequence[str],
    weights: np.ndarray,
    geometry_valid: Sequence[bool],
    sfreq: float,
    correlation_threshold: float = PRIMARY_CORRELATION_THRESHOLD,
    persistent_windows: int = PERSISTENT_WINDOW_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score observed channels against leave-one-out spherical-spline predictions."""

    names = list(map(str, channels))
    valid = np.asarray(geometry_valid, dtype=bool)
    matrix = np.asarray(weights, dtype=float)
    if len(windows) != 3:
        raise ValueError("spatial benchmark requires exactly three controlled windows")
    if matrix.shape != (len(names), len(names)) or len(valid) != len(names):
        raise ValueError("weights and geometry mask must match channels")

    score_rows: list[dict[str, object]] = []
    correlations: list[np.ndarray] = []
    residual_ratios: list[np.ndarray] = []
    for window_index, window in enumerate(windows, start=1):
        values = qc_only_filter_and_median_reference(np.asarray(window, dtype=float), sfreq)
        predicted = np.full_like(values, np.nan)
        predicted[valid] = matrix[valid] @ values
        corr = np.full(len(names), np.nan)
        residual = np.full(len(names), np.nan)
        for index in np.flatnonzero(valid):
            observed = values[index]
            estimate = predicted[index]
            if np.std(observed) > 0 and np.std(estimate) > 0:
                corr[index] = float(np.corrcoef(observed, estimate)[0, 1])
            denominator = max(float(np.sqrt(np.mean(observed**2))), np.finfo(float).tiny)
            residual[index] = float(np.sqrt(np.mean((observed - estimate) ** 2)) / denominator)
            score_rows.append({
                "window_index": window_index,
                "channel": names[index],
                "spatial_prediction_correlation": corr[index],
                "normalized_residual_rms": residual[index],
            })
        correlations.append(corr)
        residual_ratios.append(residual)

    corr_stack = np.vstack(correlations)
    residual_stack = np.vstack(residual_ratios)
    failed = np.sum(np.isfinite(corr_stack) & (corr_stack < float(correlation_threshold)), axis=0)
    median_correlations = np.full(len(names), np.nan)
    median_residuals = np.full(len(names), np.nan)
    median_correlations[valid] = np.nanmedian(corr_stack[:, valid], axis=0)
    median_residuals[valid] = np.nanmedian(residual_stack[:, valid], axis=0)
    evidence = pd.DataFrame({
        "channel": names,
        "spatial_geometry_status": np.where(valid, "SCORABLE", "GEOMETRY_UNSCORABLE"),
        "spatial_failed_window_count": failed,
        "spatial_spline_corr_0_75_persistent": valid & (failed >= int(persistent_windows)),
        "median_spatial_prediction_correlation": median_correlations,
        "median_normalized_residual_rms": median_residuals,
        "spatial_correlations_by_window": [
            ";".join("" if not np.isfinite(value) else f"{value:.9g}" for value in corr_stack[:, index])
            for index in range(len(names))
        ],
    })
    return evidence, pd.DataFrame(score_rows)


def add_spatial_hybrids(
    spatial: pd.DataFrame,
    independent: pd.DataFrame,
    *,
    broad_global_context: bool,
) -> pd.DataFrame:
    """Add transparent benchmark-only hybrid and contextual interpretations."""

    merged = spatial.merge(
        independent[["channel", "approved_independent_any"]], on="channel", validate="one_to_one"
    )
    spatial_flag = merged["spatial_spline_corr_0_75_persistent"].astype(bool)
    independent_flag = merged["approved_independent_any"].astype(bool)
    merged["spatial_or_approved_independent"] = spatial_flag | independent_flag
    merged["spatial_and_approved_independent"] = spatial_flag & independent_flag
    merged["spatial_context_gated_diagnostic"] = spatial_flag & ~bool(broad_global_context)
    return merged


def spatial_scenario_performance(
    evidence: pd.DataFrame,
    *,
    truth_type: str,
    target_channel: str | None,
    neighbours: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Reuse controlled-study performance definitions for spatial methods."""

    renamed = evidence.copy()
    # scenario_performance operates on the fixed method registry; map each spatial
    # flag temporarily so localisation and denominators remain exactly comparable.
    parts = []
    for method in SPATIAL_METHOD_COLUMNS:
        method_evidence = renamed
        if method in {"spatial_spline_corr_0_75_persistent", "spatial_context_gated_diagnostic"}:
            method_evidence = renamed[renamed["spatial_geometry_status"].eq("SCORABLE")]
        proxy = pd.DataFrame({"channel": method_evidence["channel"]})
        from .bad_channel_controlled_validation import CANDIDATE_METHOD_COLUMNS

        for column in CANDIDATE_METHOD_COLUMNS:
            proxy[column] = method_evidence[method].astype(bool).to_numpy()
        row = scenario_performance(
            proxy, truth_type=truth_type, target_channel=target_channel, neighbours=neighbours
        ).iloc[[0]].copy()
        row["method"] = method
        row["evaluated_channel_count"] = len(method_evidence)
        parts.append(row)
    return pd.concat(parts, ignore_index=True)


def threshold_sensitivity_flags(
    score_rows: pd.DataFrame,
    *,
    channels: Sequence[str],
    geometry_valid: Sequence[bool],
    thresholds: Sequence[float] = DIAGNOSTIC_THRESHOLDS,
) -> pd.DataFrame:
    """Create a narrow, non-optimising threshold sensitivity table."""

    valid_by_channel = dict(zip(map(str, channels), map(bool, geometry_valid)))
    rows = []
    for threshold in thresholds:
        counts = score_rows.groupby("channel")["spatial_prediction_correlation"].apply(
            lambda values: int(np.sum(pd.to_numeric(values, errors="coerce") < float(threshold)))
        )
        for channel in map(str, channels):
            rows.append({
                "threshold": float(threshold),
                "channel": channel,
                "geometry_status": "SCORABLE" if valid_by_channel[channel] else "GEOMETRY_UNSCORABLE",
                "failed_window_count": int(counts.get(channel, 0)),
                "candidate": bool(valid_by_channel[channel] and counts.get(channel, 0) >= PERSISTENT_WINDOW_COUNT),
            })
    return pd.DataFrame(rows)
