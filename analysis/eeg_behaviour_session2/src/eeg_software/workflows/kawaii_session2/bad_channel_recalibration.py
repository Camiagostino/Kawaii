from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .raw_qc import eeg_qc_picks, qc_sample_windows


@dataclass(frozen=True)
class NeighbourGeometry:
    neighbours: dict[str, tuple[str, ...]]
    edge_channels: frozenset[str]
    omitted_channels: tuple[str, ...]


def discover_validation_candidate_ids(
    root: Path,
    config: Mapping[str, Any],
    *,
    documented_candidates: Sequence[str] = (),
) -> list[str]:
    """Discover canonical participant IDs from project records, not result folders."""

    found: set[str] = set()
    log_root = Path(root) / "workspace" / "kawaii" / "session2" / "source" / "behavioral_logs"
    if log_root.exists():
        for path in log_root.iterdir():
            match = re.match(r"^(sub\d{3}|pilot\d{2})(?:_[A-Za-z0-9]+)*_session2$", path.name)
            if match:
                found.add(match.group(1))
    participants = dict(config.get("participants", {}) or {})
    defaults = dict(participants.get("defaults", {}) or {})
    analysis = dict(config.get("analysis", {}) or {})
    for value in (*defaults.get("included", []), *defaults.get("excluded", []), *analysis.get("include_subjects", []), *analysis.get("exclude_subjects", [])):
        found.add(str(value))
    found.update(str(value) for value in documented_candidates)
    return sorted(found, key=_participant_sort_key)


def classify_validation_candidate(
    participant: str,
    *,
    raw_available: bool,
    raw_readable: bool,
    expected_channels_present: bool,
    montage_usable: bool,
    duration_sufficient: bool,
    required_windows_available: bool,
    behavioural_metadata_available: bool,
    behavioural_metadata_required: bool,
    explicitly_excluded: bool,
) -> tuple[str, bool, str]:
    """Apply the pre-QC, outcome-blind cohort eligibility rules."""

    if participant.startswith("pilot"):
        return "D", False, "pilot recording; excluded from final-study participant validation"
    if explicitly_excluded:
        return "D", False, "explicitly excluded by analysis config; exclusion remains authoritative pending reconciliation"
    if not raw_available:
        return "C", False, "Session 2 raw EEG unavailable"
    if not raw_readable:
        return "B", False, "raw EEG is available but unreadable"
    if not expected_channels_present:
        return "B", False, "expected QC channel set is incomplete"
    if not montage_usable:
        return "B", False, "montage is unusable for the geometry-dependent candidate"
    if not duration_sufficient or not required_windows_available:
        return "B", False, "recording is too short for the pre-specified distributed QC windows"
    if behavioural_metadata_required and not behavioural_metadata_available:
        return "B", False, "required behavioural metadata is unavailable"
    note = "eligible: raw structure supports outcome-blind Gate 3A/QC-method validation"
    if not behavioural_metadata_available:
        note += "; behavioural metadata absent but not structurally required for raw QC"
    return "A", True, note


def _participant_sort_key(participant: str) -> tuple[int, int, str]:
    prefix = 0 if participant.startswith("pilot") else 1
    match = re.search(r"(\d+)$", participant)
    return prefix, int(match.group(1)) if match else -1, participant


def qc_only_filter_and_median_reference(
    data: np.ndarray,
    sfreq: float,
    *,
    l_freq_hz: float = 1.0,
    h_freq_hz: float = 45.0,
    filter_order: int = 4,
) -> np.ndarray:
    """Band-pass a channels x samples window, then apply sample-wise median reference."""

    from scipy.signal import butter, sosfiltfilt

    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must have shape channels x samples")
    nyquist = float(sfreq) / 2.0
    if not 0 < float(l_freq_hz) < float(h_freq_hz) < nyquist:
        raise ValueError("QC band-pass frequencies must satisfy 0 < l_freq < h_freq < Nyquist.")
    sos = butter(int(filter_order), [float(l_freq_hz), float(h_freq_hz)], btype="bandpass", fs=float(sfreq), output="sos")
    filtered = sosfiltfilt(sos, values, axis=1)
    return filtered - np.nanmedian(filtered, axis=0, keepdims=True)


def deterministic_delaunay_neighbours(raw, channels: Sequence[str]) -> NeighbourGeometry:
    """Return deterministic 2-D montage neighbours from Delaunay adjacency."""

    from scipy.spatial import Delaunay, QhullError

    requested = [str(channel) for channel in channels]
    coordinates: dict[str, np.ndarray] = {}
    for channel in requested:
        if channel not in raw.ch_names:
            continue
        index = raw.ch_names.index(channel)
        xyz = np.asarray(raw.info["chs"][index]["loc"][:3], dtype=float)
        if np.isfinite(xyz).all() and not np.allclose(xyz, 0.0):
            coordinates[channel] = xyz[:2]
    omitted = tuple(sorted(set(requested) - set(coordinates)))
    neighbours = {channel: set() for channel in requested}
    edge_channels: set[str] = set()
    names = sorted(coordinates)
    if len(names) >= 3:
        points = np.vstack([coordinates[name] for name in names])
        try:
            triangulation = Delaunay(points)
        except QhullError:
            triangulation = None
        if triangulation is not None:
            for simplex in triangulation.simplices:
                for left, right in ((0, 1), (1, 2), (2, 0)):
                    a, b = names[int(simplex[left])], names[int(simplex[right])]
                    neighbours[a].add(b)
                    neighbours[b].add(a)
            edge_channels = {names[int(index)] for index in np.unique(triangulation.convex_hull)}
    return NeighbourGeometry(
        neighbours={channel: tuple(sorted(peers)) for channel, peers in neighbours.items()},
        edge_channels=frozenset(edge_channels),
        omitted_channels=omitted,
    )


def median_correlation_scores(
    data: np.ndarray,
    channels: Sequence[str],
    *,
    neighbours: Mapping[str, Sequence[str]] | None = None,
    absolute: bool = True,
    minimum_peers: int = 1,
) -> tuple[dict[str, float], dict[str, int]]:
    """Calculate per-channel median correlation against all or named peers."""

    values = np.asarray(data, dtype=float)
    names = [str(channel) for channel in channels]
    if values.shape[0] != len(names):
        raise ValueError("data rows and channels must have equal length")
    name_to_index = {name: index for index, name in enumerate(names)}
    scores: dict[str, float] = {}
    peer_counts: dict[str, int] = {}
    for name in names:
        peers = [peer for peer in (neighbours.get(name, ()) if neighbours is not None else names) if peer != name and peer in name_to_index]
        correlations = []
        for peer in peers:
            correlation = _finite_pearson(values[name_to_index[name]], values[name_to_index[peer]])
            if np.isfinite(correlation):
                correlations.append(abs(correlation) if absolute else correlation)
        peer_counts[name] = len(correlations)
        scores[name] = float(np.median(correlations)) if len(correlations) >= int(minimum_peers) else np.nan
    return scores, peer_counts


def baseline_all_channel_abs_scores(data: np.ndarray, channels: Sequence[str]) -> tuple[dict[str, float], dict[str, int]]:
    """Reproduce the production all-channel absolute-correlation calculation."""

    values = np.asarray(data, dtype=float)
    names = [str(channel) for channel in channels]
    if values.shape[0] != len(names):
        raise ValueError("data rows and channels must have equal length")
    if len(names) < 2:
        return ({name: np.nan for name in names}, {name: 0 for name in names})
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.abs(np.corrcoef(values))
    np.fill_diagonal(matrix, np.nan)
    scores = {}
    peer_counts = {}
    for index, name in enumerate(names):
        finite = matrix[index][np.isfinite(matrix[index])]
        peer_counts[name] = int(finite.size)
        scores[name] = float(np.median(finite)) if finite.size else np.nan
    return scores, peer_counts


def within_window_low_robust_z(scores: Mapping[str, float], *, minimum_scale: float = 0.05) -> dict[str, float]:
    """Express low channel scores relative to the same window's channel distribution."""

    finite = np.asarray([value for value in scores.values() if np.isfinite(value)], dtype=float)
    if not finite.size:
        return {channel: np.nan for channel in scores}
    centre = float(np.median(finite))
    mad = float(np.median(np.abs(finite - centre)))
    scale = max(1.4826 * mad, float(minimum_scale))
    return {channel: float((centre - value) / scale) if np.isfinite(value) else np.nan for channel, value in scores.items()}


def aggregate_failed_windows(
    scores_by_window: np.ndarray,
    *,
    threshold: float,
    minimum_failed_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return failed counts, fractions and proposals; non-finite values fail."""

    values = np.asarray(scores_by_window, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("scores_by_window must be channels x one-or-more windows")
    failed = ~np.isfinite(values) | (values < float(threshold))
    counts = failed.sum(axis=1)
    fractions = failed.mean(axis=1)
    return counts, fractions, fractions >= float(minimum_failed_fraction)


def evaluate_recalibration_candidates(raw, participant: str, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline and candidate correlation evidence without changing Raw."""

    qc_cfg = dict(config.get("preprocessing", {}).get("raw_qc", {}) or {})
    channels = eeg_qc_picks(raw, config)
    windows = qc_sample_windows(raw, qc_cfg, "channel_qc")
    geometry = deterministic_delaunay_neighbours(raw, channels)
    sfreq = float(raw.info["sfreq"])
    methods = (
        ("baseline_raw_all_abs", False, None, True),
        ("filtered_medianref_all_abs", True, None, True),
        ("filtered_medianref_local_abs", True, geometry.neighbours, True),
        ("filtered_medianref_local_signed", True, geometry.neighbours, False),
    )
    score_rows: list[dict[str, Any]] = []
    for window_index, (start, stop) in enumerate(windows, start=1):
        raw_data = raw.get_data(picks=channels, start=start, stop=stop)
        filtered_data = qc_only_filter_and_median_reference(raw_data, sfreq)
        for method, use_filtered, neighbours, absolute in methods:
            minimum_peers = 2 if neighbours is not None else 1
            if method == "baseline_raw_all_abs":
                scores, peer_counts = baseline_all_channel_abs_scores(raw_data, channels)
            else:
                scores, peer_counts = median_correlation_scores(
                    filtered_data if use_filtered else raw_data,
                    channels,
                    neighbours=neighbours,
                    absolute=absolute,
                    minimum_peers=minimum_peers,
                )
            low_z = within_window_low_robust_z(scores)
            finite_scores = np.asarray([value for value in scores.values() if np.isfinite(value)], dtype=float)
            window_median = float(np.median(finite_scores)) if finite_scores.size else np.nan
            for channel in channels:
                score_rows.append(
                    {
                        "participant": participant,
                        "window_index": window_index,
                        "window_start_s": start / sfreq,
                        "window_stop_s": stop / sfreq,
                        "channel": channel,
                        "method": method,
                        "signal_context": "unfiltered_acquisition_reference" if not use_filtered else "bandpass_1_45_hz_then_samplewise_median_reference",
                        "filter_l_freq_hz": np.nan if not use_filtered else 1.0,
                        "filter_h_freq_hz": np.nan if not use_filtered else 45.0,
                        "reference": "acquisition_CPz_absent" if not use_filtered else "samplewise_channel_median",
                        "correlation_scope": "all_other_eligible_eeg" if neighbours is None else "delaunay_montage_neighbours",
                        "absolute_correlation": bool(absolute),
                        "minimum_peers_required": minimum_peers,
                        "neighbour_definition": "not_applicable" if neighbours is None else "2D_xy_Delaunay_standard_1005",
                        "score": scores[channel],
                        "finite_peer_count": peer_counts[channel],
                        "within_window_low_robust_z": low_z[channel],
                        "window_median_score": window_median,
                        "below_0_4": (scores[channel] < 0.4) if np.isfinite(scores[channel]) else pd.NA,
                    }
                )
    scores = pd.DataFrame(score_rows)
    neighbour_rows = [
        {
            "participant": participant,
            "channel": channel,
            "coordinate_valid": channel not in geometry.omitted_channels,
            "delaunay_edge_channel": channel in geometry.edge_channels,
            "neighbour_count": len(geometry.neighbours[channel]),
            "neighbours": ";".join(geometry.neighbours[channel]),
        }
        for channel in channels
    ]
    neighbours = pd.DataFrame(neighbour_rows)
    summaries = summarize_candidate_windows(scores)
    return scores, neighbours, summaries


def summarize_candidate_windows(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-window candidate distributions and broad failures."""

    rows = []
    for keys, group in scores.groupby(["participant", "method", "window_index"], sort=True):
        participant, method, window_index = keys
        values = pd.to_numeric(group["score"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        finite = values[valid]
        quantiles = np.percentile(finite, [0, 10, 25, 50, 75, 90, 100]) if finite.size else np.full(7, np.nan)
        rows.append(
            {
                "participant": participant,
                "method": method,
                "window_index": int(window_index),
                "window_start_s": float(group["window_start_s"].iloc[0]),
                "window_stop_s": float(group["window_stop_s"].iloc[0]),
                "valid_channel_count": int(finite.size),
                "unscorable_channel_count": int((~valid).sum()),
                "min": quantiles[0],
                "p10": quantiles[1],
                "p25": quantiles[2],
                "median": quantiles[3],
                "p75": quantiles[4],
                "p90": quantiles[5],
                "max": quantiles[6],
                "below_0_4_count": int((valid & (values < 0.4)).sum()),
                "below_0_4_fraction": float((finite < 0.4).mean()) if finite.size else np.nan,
                "channel_specific_z_ge_3_count": int((pd.to_numeric(group["within_window_low_robust_z"], errors="coerce") >= 3.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def classify_global_focal_windows(
    summaries: pd.DataFrame,
    *,
    broad_median_threshold: float = 0.4,
    focal_z_count_threshold: int = 1,
) -> pd.DataFrame:
    """Classify broad and focal low-correlation evidence deterministically."""

    result = summaries.copy()
    broad = pd.to_numeric(result["median"], errors="coerce") < float(broad_median_threshold)
    focal = pd.to_numeric(result["channel_specific_z_ge_3_count"], errors="coerce") >= int(focal_z_count_threshold)
    result["broad_rule"] = f"window_median_score<{float(broad_median_threshold):g}"
    result["focal_rule"] = f"channel_specific_low_robust_z_count>={int(focal_z_count_threshold)}"
    result["broad_low_correlation"] = broad
    result["focal_low_correlation"] = focal
    result["global_focal_class"] = np.select(
        [broad & focal, broad, focal], ["mixed", "broad", "focal"], default="none"
    )
    return result


def edge_membership_summary(
    membership: pd.DataFrame,
    neighbours: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize proposal rates by edge status and montage-relative degree."""

    geometry = neighbours.copy()
    valid = geometry[geometry["coordinate_valid"].astype(bool)]
    median_degree = valid.groupby("participant")["neighbour_count"].median().rename("participant_median_degree")
    geometry = geometry.merge(median_degree, on="participant", how="left")
    geometry["degree_group"] = np.where(
        geometry["neighbour_count"] < geometry["participant_median_degree"], "low_degree", "typical_or_high_degree"
    )
    merged = membership.merge(
        geometry[["participant", "channel", "coordinate_valid", "delaunay_edge_channel", "neighbour_count", "participant_median_degree", "degree_group"]],
        on=["participant", "channel"], how="left", validate="many_to_one",
    )
    rows = []
    for grouping, column in (("edge_status", "delaunay_edge_channel"), ("degree_group", "degree_group")):
        eligible = merged[merged["scorable"].astype(bool)].copy()
        if column == "delaunay_edge_channel":
            eligible[grouping] = np.where(eligible[column].astype(bool), "edge", "interior")
        for keys, group in eligible.groupby(
            ["participant", "method", "threshold", "minimum_failed_fraction", grouping], sort=True
        ):
            participant, method, threshold, fraction, stratum = keys
            proposed = group["proposed"].astype(bool)
            rows.append({
                "participant": participant, "method": method, "threshold": float(threshold),
                "minimum_failed_fraction": float(fraction), "geometry_grouping": grouping,
                "geometry_stratum": str(stratum), "eligible_channel_count": int(len(group)),
                "proposed_count": int(proposed.sum()), "proposed_fraction": float(proposed.mean()),
                "proposed_channels": ";".join(sorted(group.loc[proposed, "channel"].astype(str))),
            })
    return merged, pd.DataFrame(rows)


def candidate_proposal_sensitivity(
    scores: pd.DataFrame,
    *,
    thresholds: Sequence[float] = (0.2, 0.25, 0.3, 0.35, 0.4),
    failed_fractions: Sequence[float] = (1 / 3, 2 / 3, 1.0),
) -> pd.DataFrame:
    """Characterize proposal counts without selecting production parameters."""

    rows = []
    for (participant, method), group in scores.groupby(["participant", "method"], sort=True):
        pivot = group.pivot(index="channel", columns="window_index", values="score").sort_index()
        pivot = pivot[np.isfinite(pivot.to_numpy(dtype=float)).any(axis=1)]
        for threshold in thresholds:
            for fraction in failed_fractions:
                counts, fractions, proposed = aggregate_failed_windows(
                    pivot.to_numpy(dtype=float),
                    threshold=float(threshold),
                    minimum_failed_fraction=float(fraction),
                )
                rows.append(
                    {
                        "participant": participant,
                        "method": method,
                        "threshold": float(threshold),
                        "minimum_failed_fraction": float(fraction),
                        "proposed_count": int(proposed.sum()),
                        "eligible_channel_count": int(len(proposed)),
                        "proposed_fraction": float(proposed.mean()),
                        "failed_window_count_distribution": ";".join(f"{value}:{int((counts == value).sum())}" for value in range(pivot.shape[1] + 1)),
                        "mean_failed_window_fraction": float(fractions.mean()),
                    }
                )
    return pd.DataFrame(rows)


def candidate_proposal_membership(
    scores: pd.DataFrame,
    *,
    threshold: float = 0.4,
    failed_fractions: Sequence[float] = (1 / 3, 2 / 3, 1.0),
) -> pd.DataFrame:
    """Return channel-level membership for descriptive method-overlap checks."""

    rows = []
    for (participant, method), group in scores.groupby(["participant", "method"], sort=True):
        pivot = group.pivot(index="channel", columns="window_index", values="score").sort_index()
        scorable = np.isfinite(pivot.to_numpy(dtype=float)).any(axis=1)
        for channel, is_scorable in zip(pivot.index, scorable):
            values = pivot.loc[channel].to_numpy(dtype=float)[None, :]
            for fraction in failed_fractions:
                if is_scorable:
                    counts, fractions, proposed = aggregate_failed_windows(values, threshold=threshold, minimum_failed_fraction=fraction)
                    failed_count = int(counts[0])
                    failed_fraction = float(fractions[0])
                    proposal = bool(proposed[0])
                else:
                    failed_count = 0
                    failed_fraction = np.nan
                    proposal = False
                rows.append(
                    {
                        "participant": participant,
                        "method": method,
                        "channel": channel,
                        "threshold": float(threshold),
                        "minimum_failed_fraction": float(fraction),
                        "scorable": bool(is_scorable),
                        "failed_window_count": failed_count,
                        "failed_window_fraction": failed_fraction,
                        "proposed": proposal,
                    }
                )
    return pd.DataFrame(rows)


def pairwise_method_overlaps(membership: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise proposal overlaps and method-unique channels."""

    from itertools import combinations

    rows = []
    for (participant, fraction), group in membership.groupby(["participant", "minimum_failed_fraction"], sort=True):
        methods = sorted(group["method"].unique())
        proposed = {
            method: set(group[(group["method"].eq(method)) & (group["proposed"].astype(bool))]["channel"].astype(str))
            for method in methods
        }
        for left, right in combinations(methods, 2):
            intersection = proposed[left] & proposed[right]
            union = proposed[left] | proposed[right]
            rows.append(
                {
                    "participant": participant,
                    "minimum_failed_fraction": float(fraction),
                    "left_method": left,
                    "right_method": right,
                    "left_count": len(proposed[left]),
                    "right_count": len(proposed[right]),
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                    "intersection_channels": ";".join(sorted(intersection)),
                    "left_unique_channels": ";".join(sorted(proposed[left] - proposed[right])),
                    "right_unique_channels": ";".join(sorted(proposed[right] - proposed[left])),
                }
            )
    return pd.DataFrame(rows)


def persistent_relative_outliers(scores: pd.DataFrame, *, z_threshold: float = 3.0, minimum_windows: int = 2) -> pd.DataFrame:
    """Describe channel-relative low-score persistence; parameters remain provisional."""

    rows = []
    for (participant, method, channel), group in scores.groupby(["participant", "method", "channel"], sort=True):
        values = pd.to_numeric(group["within_window_low_robust_z"], errors="coerce").to_numpy(dtype=float)
        failed = np.isfinite(values) & (values >= float(z_threshold))
        rows.append(
            {
                "participant": participant,
                "method": method,
                "channel": channel,
                "relative_z_threshold": float(z_threshold),
                "minimum_windows": int(minimum_windows),
                "failed_window_count": int(failed.sum()),
                "failed_window_fraction": float(failed.mean()),
                "persistent_relative_outlier": bool(failed.sum() >= int(minimum_windows)),
            }
        )
    return pd.DataFrame(rows)


def _finite_pearson(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return np.nan
    x = np.asarray(left[mask], dtype=float)
    y = np.asarray(right[mask], dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])
