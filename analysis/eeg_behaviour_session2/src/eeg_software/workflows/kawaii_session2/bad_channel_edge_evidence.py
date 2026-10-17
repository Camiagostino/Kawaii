from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def geometry_feature_table(
    coordinates: Mapping[str, Sequence[float]],
    neighbours: Mapping[str, Sequence[str]],
    edge_channels: Sequence[str],
) -> pd.DataFrame:
    """Describe objective distance and angular coverage of fixed neighbours."""

    edge = set(map(str, edge_channels))
    rows: list[dict[str, object]] = []
    for channel in sorted(neighbours):
        if channel not in coordinates:
            continue
        origin = np.asarray(coordinates[channel], dtype=float)[:2]
        peers = [peer for peer in neighbours[channel] if peer in coordinates]
        offsets = np.asarray([np.asarray(coordinates[peer], dtype=float)[:2] - origin for peer in peers])
        distances = np.linalg.norm(offsets, axis=1) if len(offsets) else np.asarray([], dtype=float)
        angles = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2 * np.pi) if len(offsets) else np.asarray([])
        if len(angles):
            ordered = np.sort(angles)
            gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
            maximum_gap = float(np.max(gaps))
            coverage = float(2 * np.pi - maximum_gap)
            unit = offsets / distances[:, None]
            directional_asymmetry = float(np.linalg.norm(np.mean(unit, axis=0)))
        else:
            maximum_gap = coverage = directional_asymmetry = np.nan
        rows.append({
            "channel": channel,
            "delaunay_edge_channel": channel in edge,
            "neighbour_count": len(peers),
            "mean_neighbour_distance": float(np.mean(distances)) if len(distances) else np.nan,
            "median_neighbour_distance": float(np.median(distances)) if len(distances) else np.nan,
            "maximum_neighbour_distance": float(np.max(distances)) if len(distances) else np.nan,
            "angular_coverage_degrees": float(np.degrees(coverage)) if np.isfinite(coverage) else np.nan,
            "maximum_angular_gap_degrees": float(np.degrees(maximum_gap)) if np.isfinite(maximum_gap) else np.nan,
            "directional_asymmetry": directional_asymmetry,
        })
    return pd.DataFrame(rows)


def add_participant_robust_scores(
    table: pd.DataFrame,
    metrics: Sequence[str],
    *,
    group_column: str = "participant",
) -> pd.DataFrame:
    """Add descriptive within-participant robust z scores without thresholds."""

    result = table.copy()
    for metric in metrics:
        numeric = pd.to_numeric(result[metric], errors="coerce")
        result[f"{metric}_participant_robust_z"] = np.nan
        for _, indices in result.groupby(group_column, sort=True).groups.items():
            values = numeric.loc[indices]
            finite = values[np.isfinite(values)]
            if finite.empty:
                continue
            centre = float(finite.median())
            mad = float((finite - centre).abs().median())
            scale = 1.4826 * mad
            if not np.isfinite(scale) or scale == 0:
                continue
            result.loc[indices, f"{metric}_participant_robust_z"] = (values - centre) / scale
    return result


def independent_metric_summary(
    table: pd.DataFrame,
    metrics: Sequence[str],
    *,
    edge_column: str = "delaunay_edge_channel",
) -> pd.DataFrame:
    """Return pooled descriptive distributions for edge and interior channels."""

    rows: list[dict[str, object]] = []
    for metric in metrics:
        for is_edge, group in table.groupby(edge_column, sort=True):
            values = pd.to_numeric(group[metric], errors="coerce")
            values = values[np.isfinite(values)]
            rows.append({
                "metric": metric,
                "geometry_stratum": "edge" if bool(is_edge) else "interior",
                "n": int(len(values)),
                "median": float(values.median()) if len(values) else np.nan,
                "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                "p90": float(values.quantile(0.90)) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def persistent_percentile_outliers(
    scores: pd.DataFrame,
    *,
    percentile: float = 5.0,
    minimum_windows: int = 2,
) -> pd.DataFrame:
    """Flag channels in a fixed low within-window percentile persistently."""

    if not 0 < float(percentile) < 50:
        raise ValueError("percentile must be between 0 and 50")
    rows: list[dict[str, object]] = []
    for (participant, channel), group in scores.groupby(["participant", "channel"], sort=True):
        failed = []
        for _, window in group.groupby("window_index", sort=True):
            all_window = scores[
                (scores["participant"].eq(participant))
                & (scores["window_index"].eq(window["window_index"].iloc[0]))
            ]
            finite = pd.to_numeric(all_window["score"], errors="coerce").dropna()
            value = pd.to_numeric(window["score"], errors="coerce").iloc[0]
            cutoff = float(np.percentile(finite, percentile)) if len(finite) else np.nan
            failed.append(bool(np.isfinite(value) and np.isfinite(cutoff) and value <= cutoff))
        rows.append({
            "participant": participant,
            "channel": channel,
            "percentile": float(percentile),
            "minimum_windows": int(minimum_windows),
            "failed_window_count": int(sum(failed)),
            "persistent_percentile_outlier": bool(sum(failed) >= int(minimum_windows)),
        })
    return pd.DataFrame(rows)


def descriptive_spearman_associations(
    table: pd.DataFrame,
    outcome_columns: Sequence[str],
    geometry_columns: Sequence[str],
) -> pd.DataFrame:
    """Calculate bounded descriptive Spearman coefficients without p-values."""

    rows: list[dict[str, object]] = []
    for outcome in outcome_columns:
        for geometry in geometry_columns:
            pair = table[[outcome, geometry]].apply(pd.to_numeric, errors="coerce").dropna()
            rows.append({
                "outcome": outcome,
                "geometry_feature": geometry,
                "n": int(len(pair)),
                "spearman_rho": float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) >= 3 else np.nan,
                "interpretation_boundary": "descriptive_only_no_p_value",
            })
    return pd.DataFrame(rows)
