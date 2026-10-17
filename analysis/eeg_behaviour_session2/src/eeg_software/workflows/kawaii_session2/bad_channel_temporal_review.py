from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from .bad_channel_recalibration import (
    median_correlation_scores,
    qc_only_filter_and_median_reference,
    within_window_low_robust_z,
)


WINDOW_LENGTH_S = 10.0
MINIMUM_VALID_FRACTION = 0.5
METRIC_FLAGS = (
    "scale_dispersion_flag",
    "peak_to_peak_flag",
    "flat_dropout_flag",
    "clipping_flag",
    "line_noise_relative_flag",
    "high_frequency_relative_flag",
    "drift_relative_flag",
    "local_absolute_flag",
    "local_robust_relative_flag",
)


def fixed_temporal_windows(
    n_times: int,
    sfreq: float,
    *,
    window_length_s: float = WINDOW_LENGTH_S,
    minimum_valid_fraction: float = MINIMUM_VALID_FRACTION,
) -> pd.DataFrame:
    """Return deterministic non-overlapping windows including the final partial window."""

    if n_times <= 0 or sfreq <= 0 or window_length_s <= 0:
        raise ValueError("n_times, sfreq, and window_length_s must be positive")
    window_samples = int(round(float(window_length_s) * float(sfreq)))
    minimum_samples = int(np.ceil(window_samples * float(minimum_valid_fraction)))
    rows = []
    for index, start in enumerate(range(0, int(n_times), window_samples), start=1):
        stop = min(start + window_samples, int(n_times))
        rows.append({
            "window_index": index,
            "start_sample": start,
            "stop_sample": stop,
            "window_start_s": start / float(sfreq),
            "window_end_s": stop / float(sfreq),
            "window_duration_s": (stop - start) / float(sfreq),
            "full_length_window": stop - start == window_samples,
            "valid_window": stop - start >= minimum_samples,
            "validity_rule": f"duration>={minimum_valid_fraction:.3g}*{window_length_s:.3g}s",
        })
    return pd.DataFrame(rows)


def temporal_window_metrics(
    data: np.ndarray,
    *,
    channels: Sequence[str],
    case_channels: Sequence[str],
    neighbours: Mapping[str, Sequence[str]],
    sfreq: float,
    window_index: int,
) -> pd.DataFrame:
    """Compute transparent channel metrics and pre-specified diagnostic flags."""

    values = np.asarray(data, dtype=float)
    names = list(map(str, channels))
    if values.shape[0] != len(names) or values.shape[1] < 8:
        raise ValueError("window data must match channels and contain at least eight samples")
    std_uv = np.std(values, axis=1) * 1e6
    median = np.median(values, axis=1, keepdims=True)
    robust_mad_uv = 1.4826 * np.median(np.abs(values - median), axis=1) * 1e6
    ptp_uv = np.ptp(values, axis=1) * 1e6
    scale_z = _positive_robust_z(np.log(np.maximum(std_uv, np.finfo(float).tiny)))

    frequencies, power = welch(
        values,
        fs=float(sfreq),
        nperseg=values.shape[1],
        axis=1,
        detrend="constant",
    )
    line_ratio = _band_ratio(power, frequencies, 49.0, 51.0, 40.0, 60.0)
    high_frequency_ratio = _band_ratio(
        power, frequencies, 35.0, min(100.0, sfreq / 2.0 - 0.1), 1.0, min(100.0, sfreq / 2.0 - 0.1)
    )
    drift_ratio = _band_ratio(power, frequencies, 0.1, 1.0, 1.0, 45.0)
    line_z = _positive_robust_z(np.log(np.maximum(line_ratio, np.finfo(float).tiny)))
    high_frequency_z = _positive_robust_z(np.log(np.maximum(high_frequency_ratio, np.finfo(float).tiny)))
    drift_z = _positive_robust_z(np.log(np.maximum(drift_ratio, np.finfo(float).tiny)))

    filtered = qc_only_filter_and_median_reference(values, sfreq)
    local, peer_counts = median_correlation_scores(
        filtered, names, neighbours=neighbours, absolute=True, minimum_peers=2
    )
    local_relative = within_window_low_robust_z(local)

    rows = []
    for channel in map(str, case_channels):
        channel_index = names.index(channel)
        target = values[channel_index]
        repeated_count, repeated_fraction = _longest_repeated_run(target)
        extreme = (target == np.max(target)) | (target == np.min(target))
        clipping_count = int(np.sum(extreme))
        clipping_fraction = clipping_count / target.size
        flags = {
            "scale_dispersion_flag": bool(std_uv[channel_index] >= 500.0 or scale_z[channel_index] >= 5.0),
            "peak_to_peak_flag": bool(ptp_uv[channel_index] >= 5000.0),
            "flat_dropout_flag": bool(std_uv[channel_index] <= 0.01 or repeated_fraction >= 0.10),
            "clipping_flag": bool(clipping_count >= 3 and clipping_fraction >= 0.001),
            "line_noise_relative_flag": bool(line_z[channel_index] >= 3.0),
            "high_frequency_relative_flag": bool(high_frequency_z[channel_index] >= 3.0),
            "drift_relative_flag": bool(drift_z[channel_index] >= 3.0),
            "local_absolute_flag": bool(np.isfinite(local.get(channel, np.nan)) and local[channel] < 0.4),
            "local_robust_relative_flag": bool(
                np.isfinite(local_relative.get(channel, np.nan)) and local_relative[channel] >= 3.0
            ),
        }
        strong_independent = any(flags[name] for name in METRIC_FLAGS[:4])
        spectral_family = any(flags[name] for name in METRIC_FLAGS[4:7])
        local_family = any(flags[name] for name in METRIC_FLAGS[7:])
        family_count = int(strong_independent) + int(spectral_family) + int(local_family)
        rows.append({
            "window_index": int(window_index),
            "channel": channel,
            "std_uv": float(std_uv[channel_index]),
            "robust_mad_uv": float(robust_mad_uv[channel_index]),
            "peak_to_peak_uv": float(ptp_uv[channel_index]),
            "scale_robust_z": float(scale_z[channel_index]),
            "longest_repeated_run_samples": repeated_count,
            "longest_repeated_run_fraction": repeated_fraction,
            "clipping_repeated_extreme_count": clipping_count,
            "clipping_fraction": clipping_fraction,
            "line_noise_ratio": float(line_ratio[channel_index]),
            "line_noise_robust_z": float(line_z[channel_index]),
            "high_frequency_ratio": float(high_frequency_ratio[channel_index]),
            "high_frequency_robust_z": float(high_frequency_z[channel_index]),
            "drift_ratio": float(drift_ratio[channel_index]),
            "drift_robust_z": float(drift_z[channel_index]),
            "local_absolute_correlation": float(local.get(channel, np.nan)),
            "local_low_robust_z": float(local_relative.get(channel, np.nan)),
            "local_neighbour_count": int(peer_counts.get(channel, 0)),
            **flags,
            "strong_independent_family": strong_independent,
            "spectral_relative_family": spectral_family,
            "local_correlation_family": local_family,
            "affected_evidence_family_count": family_count,
            "consensus_affected_window": bool(strong_independent or (spectral_family and local_family)),
            "consensus_rule": "strong_independent OR (spectral_relative AND local_correlation)",
        })
    return pd.DataFrame(rows)


def burden_summary(
    windows: pd.DataFrame,
    *,
    flag_column: str,
) -> dict[str, object]:
    """Summarize affected fraction and contiguous burden over valid fixed windows."""

    valid = windows[windows["valid_window"].astype(bool)].sort_values("window_index")
    affected = valid[flag_column].fillna(False).astype(bool).to_numpy()
    indices = valid["window_index"].to_numpy(dtype=int)
    starts = valid["window_start_s"].to_numpy(dtype=float)
    ends = valid["window_end_s"].to_numpy(dtype=float)
    selected = np.flatnonzero(affected)
    longest_count, longest_duration = _longest_contiguous(indices, starts, ends, affected)
    return {
        "valid_window_count": len(valid),
        "affected_window_count": len(selected),
        "affected_window_fraction": len(selected) / len(valid) if len(valid) else np.nan,
        "total_affected_duration_s": float(np.sum((ends - starts)[selected])) if len(selected) else 0.0,
        "longest_contiguous_affected_run_windows": longest_count,
        "longest_contiguous_affected_run_s": longest_duration,
        "first_affected_time_s": float(starts[selected[0]]) if len(selected) else np.nan,
        "last_affected_time_s": float(ends[selected[-1]]) if len(selected) else np.nan,
    }


def temporal_category(summary: Mapping[str, object]) -> str:
    """Assign a transparent diagnostic temporal descriptor, never a bad-channel label."""

    count = int(summary["affected_window_count"])
    fraction = float(summary["affected_window_fraction"])
    longest = int(summary["longest_contiguous_affected_run_windows"])
    if count == 0 or not np.isfinite(fraction):
        return "INSUFFICIENT_EVIDENCE"
    if count == 1:
        return "LOCAL_TRANSIENT"
    if fraction >= 0.50:
        return "BROADLY_PERSISTENT"
    if longest >= 3:
        return "PROLONGED"
    if count >= 3:
        return "RECURRENT"
    return "INTERMITTENT"


def metric_burden_table(case_windows: pd.DataFrame) -> pd.DataFrame:
    """Keep every metric-specific burden separate from the consensus descriptor."""

    rows = []
    for flag in (*METRIC_FLAGS, "consensus_affected_window"):
        summary = burden_summary(case_windows, flag_column=flag)
        rows.append({"metric_flag": flag, **summary})
    return pd.DataFrame(rows)


def select_review_windows(case_windows: pd.DataFrame) -> pd.DataFrame:
    """Select review panels deterministically without manual artifact cherry-picking."""

    valid = case_windows[case_windows["valid_window"].astype(bool)].sort_values("window_index").copy()
    if valid.empty:
        return pd.DataFrame(columns=[*case_windows.columns, "display_role"])
    affected = valid[valid["consensus_affected_window"].astype(bool)]
    selections: list[tuple[int, str]] = []
    if not affected.empty:
        worst = affected.sort_values(
            ["affected_evidence_family_count", "window_index"], ascending=[False, True]
        ).iloc[0]
        selections.append((int(worst["window_index"]), "worst_affected"))
        remaining = affected[affected["window_index"].ne(worst["window_index"])]
        if not remaining.empty:
            median = remaining.iloc[(len(remaining) - 1) // 2]
            selections.append((int(median["window_index"]), "median_time_affected"))
        unaffected = valid[~valid["consensus_affected_window"].astype(bool)]
        if not unaffected.empty:
            midpoint = float(valid["window_start_s"].iloc[0] + valid["window_end_s"].iloc[-1]) / 2.0
            representative = unaffected.assign(
                distance=(unaffected["window_start_s"] - midpoint).abs()
            ).sort_values(["distance", "window_index"]).iloc[0]
            selections.append((int(representative["window_index"]), "representative_unaffected"))
    else:
        positions = sorted(set([0, (len(valid) - 1) // 2, len(valid) - 1]))
        roles = ["representative_start", "representative_middle", "representative_end"]
        selections.extend((int(valid.iloc[position]["window_index"]), roles[index]) for index, position in enumerate(positions))
    rows = []
    for window_index, role in selections[:3]:
        row = valid[valid["window_index"].eq(window_index)].iloc[0].to_dict()
        row["display_role"] = role
        rows.append(row)
    return pd.DataFrame(rows)


def epoch_burden(
    event_samples: Sequence[int],
    *,
    sfreq: float,
    raw_first_samp: int,
    epoch_tmin_s: float,
    epoch_tmax_s: float,
    affected_windows: pd.DataFrame,
    target_signal: np.ndarray,
    ptp_threshold_uv: float = 330.0,
) -> dict[str, object]:
    """Map outcome-blind epoch intervals to temporal windows without rejecting epochs."""

    events = np.asarray(event_samples, dtype=int)
    affected = affected_windows[
        affected_windows["valid_window"].astype(bool)
        & affected_windows["consensus_affected_window"].astype(bool)
    ]
    overlap_flags, ptp_flags = [], []
    for event in events:
        centre_s = (int(event) - int(raw_first_samp)) / float(sfreq)
        start_s, end_s = centre_s + float(epoch_tmin_s), centre_s + float(epoch_tmax_s)
        overlaps = bool(
            np.any(
                (affected["window_start_s"].to_numpy(dtype=float) < end_s)
                & (affected["window_end_s"].to_numpy(dtype=float) > start_s)
            )
        )
        start_sample = max(0, int(np.floor(start_s * sfreq)))
        stop_sample = min(len(target_signal), int(np.ceil(end_s * sfreq)))
        ptp = np.ptp(target_signal[start_sample:stop_sample]) * 1e6 if stop_sample > start_sample else np.nan
        overlap_flags.append(overlaps)
        ptp_flags.append(bool(np.isfinite(ptp) and ptp >= float(ptp_threshold_uv)))
    either = np.asarray(overlap_flags) | np.asarray(ptp_flags)
    return {
        "epoch_mapping_available": True,
        "eligible_epoch_count": len(events),
        "consensus_window_overlap_epoch_count": int(np.sum(overlap_flags)),
        "consensus_window_overlap_epoch_fraction": float(np.mean(overlap_flags)) if len(events) else np.nan,
        "target_ptp_ge_330_epoch_count": int(np.sum(ptp_flags)),
        "target_ptp_ge_330_epoch_fraction": float(np.mean(ptp_flags)) if len(events) else np.nan,
        "either_temporal_or_ptp_epoch_count": int(np.sum(either)),
        "either_temporal_or_ptp_epoch_fraction": float(np.mean(either)) if len(events) else np.nan,
        "longest_consecutive_potentially_affected_epochs": _longest_true_run(either),
        "epoch_ptp_threshold_uv_diagnostic_only": float(ptp_threshold_uv),
    }


def plot_temporal_case(
    case_windows: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    raw_windows: Mapping[int, np.ndarray],
    sfreq: float,
    case_id: str,
    participant: str,
    channel: str,
    output_path: Path,
) -> None:
    """Render a deterministic full-recording trajectory and selected raw/PSD panels."""

    fig = plt.figure(figsize=(14, 9))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.96, bottom=0.07, hspace=0.95, wspace=0.28)
    grid = fig.add_gridspec(4, 3, height_ratios=(1.25, 0.8, 1.0, 1.0))
    trajectory = fig.add_subplot(grid[0, :])
    x = case_windows["window_start_s"].to_numpy(dtype=float) / 60.0
    for column, label in (
        ("scale_robust_z", "scale z"), ("line_noise_robust_z", "line z"),
        ("high_frequency_robust_z", "HF z"), ("drift_robust_z", "drift z"),
        ("local_low_robust_z", "local-low z"),
    ):
        trajectory.plot(x, case_windows[column], linewidth=0.8, label=label)
    trajectory.axhline(3.0, color="grey", linestyle="--", linewidth=0.8, label="relative z=3")
    trajectory.axhline(5.0, color="black", linestyle=":", linewidth=0.8, label="scale z=5")
    trajectory.set(title=f"{case_id} {participant}/{channel}: full-recording temporal evidence", xlabel="Recording time (min)", ylabel="participant-relative robust z")
    trajectory.legend(ncol=6, fontsize=7)

    markers = fig.add_subplot(grid[1, :])
    flag_names = [*METRIC_FLAGS, "consensus_affected_window"]
    image = case_windows[flag_names].fillna(False).astype(int).to_numpy().T
    extent = [x.min(), case_windows["window_end_s"].max() / 60.0, len(flag_names), 0]
    markers.imshow(image, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1, extent=extent)
    markers.set(yticks=np.arange(len(flag_names)) + 0.5, yticklabels=[name.replace("_flag", "") for name in flag_names], xlabel="Recording time (min)", title="Metric-specific and consensus affected-window markers")

    selected_rows = list(selected.itertuples())
    for column_index in range(3):
        trace_ax = fig.add_subplot(grid[2, column_index])
        psd_ax = fig.add_subplot(grid[3, column_index])
        if column_index >= len(selected_rows):
            trace_ax.axis("off"); psd_ax.axis("off")
            continue
        row = selected_rows[column_index]
        trace = np.asarray(raw_windows[int(row.window_index)], dtype=float) * 1e6
        time = np.arange(trace.size) / float(sfreq)
        trace_ax.plot(time, trace - np.median(trace), linewidth=0.65)
        role = str(row.display_role).replace("_", " ").replace("median time affected", "median affected")
        trace_ax.set(title=f"{role.title()}\n{row.window_start_s / 60:.1f} min", xlabel="Window time (s)", ylabel="demeaned uV")
        frequencies, power = welch(trace * 1e-6, fs=sfreq, nperseg=len(trace), detrend="constant")
        keep = (frequencies >= 0.1) & (frequencies <= min(100.0, sfreq / 2.0))
        psd_ax.plot(frequencies[keep], 10 * np.log10(np.maximum(power[keep], np.finfo(float).tiny)), linewidth=0.75)
        psd_ax.set(xlabel="Frequency (Hz)", ylabel="PSD (dB/Hz)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, metadata={"Software": "eeg_software temporal review"})
    plt.close(fig)


def _positive_robust_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    centre = float(np.nanmedian(array))
    scale = max(1.4826 * float(np.nanmedian(np.abs(array - centre))), 0.05)
    return (array - centre) / scale


def _band_ratio(
    power: np.ndarray,
    frequencies: np.ndarray,
    numerator_low: float,
    numerator_high: float,
    denominator_low: float,
    denominator_high: float,
) -> np.ndarray:
    numerator = (frequencies >= numerator_low) & (frequencies <= numerator_high)
    denominator = (frequencies >= denominator_low) & (frequencies <= denominator_high)
    if not numerator.any() or not denominator.any():
        return np.full(power.shape[0], np.nan)
    top = np.mean(power[:, numerator], axis=1)
    bottom = np.mean(power[:, denominator], axis=1)
    return top / np.maximum(bottom, np.finfo(float).tiny)


def _longest_repeated_run(values: np.ndarray) -> tuple[int, float]:
    if len(values) == 0:
        return 0, np.nan
    equal = np.diff(values) == 0
    longest, current = 1, 1
    for item in equal:
        current = current + 1 if item else 1
        longest = max(longest, current)
    return longest, longest / len(values)


def _longest_contiguous(
    indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    affected: np.ndarray,
) -> tuple[int, float]:
    best_count = current_count = 0
    best_duration = current_start = 0.0
    previous_index = None
    for index, start, end, flag in zip(indices, starts, ends, affected):
        if flag:
            if current_count == 0 or previous_index is None or index != previous_index + 1:
                current_count, current_start = 1, start
            else:
                current_count += 1
            duration = end - current_start
            if current_count > best_count or (current_count == best_count and duration > best_duration):
                best_count, best_duration = current_count, duration
            previous_index = index
        else:
            current_count, previous_index = 0, None
    return best_count, float(best_duration)


def _longest_true_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best
