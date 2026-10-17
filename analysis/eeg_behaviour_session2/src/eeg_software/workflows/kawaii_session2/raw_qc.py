from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChannelRoleConfig:
    """Channel role lists resolved from YAML."""

    eeg: tuple[str, ...]
    eog: tuple[str, ...]
    unused: tuple[str, ...]
    misc: tuple[str, ...]
    exclude_from_eeg_qc: tuple[str, ...]
    exclude_from_reference: tuple[str, ...]
    exclude_from_ica: tuple[str, ...]
    provisional_reference: str | None


def channel_roles_from_config(config: Mapping[str, Any]) -> ChannelRoleConfig:
    """Resolve channel roles from the authoritative YAML channel section."""

    eeg_cfg = dict(config.get("channels", {}).get("eeg", {}) or {})
    ref_cfg = dict(config.get("preprocessing", {}).get("reference", {}) or {})
    provisional = str(ref_cfg.get("provisional_reference_channel", "CPz")) if ref_cfg.get("add_provisional_cpz", False) else None
    return ChannelRoleConfig(
        eeg=tuple(_aliased_channel_list(eeg_cfg.get("expected_names", []) or [], config)),
        eog=tuple(eeg_cfg.get("eog", []) or []),
        unused=tuple(eeg_cfg.get("unused", []) or []),
        misc=tuple(eeg_cfg.get("misc", []) or []),
        exclude_from_eeg_qc=tuple(_aliased_channel_list(eeg_cfg.get("exclude_from_eeg_qc", []) or [], config)),
        exclude_from_reference=tuple(eeg_cfg.get("exclude_from_reference", []) or []),
        exclude_from_ica=tuple(eeg_cfg.get("exclude_from_ica", []) or []),
        provisional_reference=provisional,
    )


def channel_name_aliases(config: Mapping[str, Any]) -> dict[str, str]:
    """Return raw-to-canonical channel aliases applied before validation."""

    channels = dict(config.get("channels", {}) or {})
    aliases = dict(channels.get("channel_name_aliases", {}) or {})
    aliases.update(dict(channels.get("channel_name_overrides", {}) or {}))
    return {str(old): str(new) for old, new in aliases.items() if str(old) != str(new)}


def apply_channel_name_aliases(raw, config: Mapping[str, Any]):
    """Rename configured channel aliases before type/montage checks."""

    aliases = channel_name_aliases(config)
    present = {old: new for old, new in aliases.items() if old in raw.ch_names and new not in raw.ch_names}
    skipped = {old: new for old, new in aliases.items() if old in raw.ch_names and new in raw.ch_names}
    warnings = []
    if present:
        raw.rename_channels(present)
        warnings.append("Applied channel-name aliases before montage validation: " + ", ".join(f"{old}->{new}" for old, new in sorted(present.items())))
    for old, new in sorted(skipped.items()):
        warnings.append(f"Skipped channel-name alias {old}->{new} because target channel already exists.")
    return raw, warnings


def _aliased_channel_list(channels: Sequence[str], config: Mapping[str, Any]) -> list[str]:
    aliases = channel_name_aliases(config)
    return [aliases.get(str(channel), str(channel)) for channel in channels]


def channel_type_mapping(config: Mapping[str, Any]) -> dict[str, str]:
    """Return an MNE channel-type mapping from YAML roles."""

    roles = channel_roles_from_config(config)
    mapping: dict[str, str] = {}
    for channel in roles.eeg:
        mapping[channel] = "eeg"
    for channel in roles.eog:
        mapping[channel] = "eog"
    for channel in roles.unused:
        mapping[channel] = "misc"
    for channel in roles.misc:
        mapping[channel] = "misc"
    return mapping


def apply_channel_types(raw, config: Mapping[str, Any]):
    """Apply MNE channel types for known channels and return a warning list."""

    raw, alias_warnings = apply_channel_name_aliases(raw, config)
    mapping = channel_type_mapping(config)
    present_mapping = {name: kind for name, kind in mapping.items() if name in raw.ch_names}
    warnings = [f"Configured channel not present in raw: {name}" for name in sorted(set(mapping) - set(raw.ch_names))]
    if present_mapping:
        raw.set_channel_types(present_mapping, on_unit_change="ignore")
    return raw, alias_warnings + warnings


def eeg_reference_picks(raw, config: Mapping[str, Any]) -> list[str]:
    """Return EEG channels eligible for reference calculations."""

    roles = channel_roles_from_config(config)
    excluded = set(roles.exclude_from_reference) | set(roles.unused) | set(roles.misc)
    return [name for name in raw.ch_names if name not in excluded and raw.get_channel_types(picks=[name])[0] == "eeg"]


def eeg_qc_picks(raw, config: Mapping[str, Any]) -> list[str]:
    """Return EEG channels eligible for automatic QC proposals."""

    roles = channel_roles_from_config(config)
    excluded = set(roles.exclude_from_eeg_qc) | set(roles.unused) | set(roles.misc)
    if roles.provisional_reference:
        excluded.add(roles.provisional_reference)
    return [name for name in raw.ch_names if name not in excluded and raw.get_channel_types(picks=[name])[0] == "eeg"]


def add_provisional_reference_channel(raw, config: Mapping[str, Any]):
    """Add configured zero-valued provisional reference channel, usually CPz."""

    ref_cfg = dict(config.get("preprocessing", {}).get("reference", {}) or {})
    if not ref_cfg.get("add_provisional_cpz", False):
        return raw, []
    channel = str(ref_cfg.get("provisional_reference_channel", "CPz"))
    if channel in raw.ch_names:
        return raw, [f"Provisional reference channel {channel} already present; no channel added."]
    if not getattr(raw, "preload", False):
        return raw, [f"Deferred physical addition of provisional reference channel {channel} because raw is not preloaded; QC records it as provisional metadata."]

    import mne

    raw = mne.add_reference_channels(raw, ref_channels=[channel], copy=False)
    raw.set_channel_types({channel: "eeg"})
    return raw, [f"Added zero-valued provisional reference channel {channel}; assumption recorded for review."]


def validate_montage(raw, config: Mapping[str, Any]) -> tuple[object, pd.DataFrame]:
    """Validate montage availability for true EEG channels without dropping any."""

    import mne

    montage_name = str(config.get("preprocessing", {}).get("montage", "standard_1005"))
    montage = mne.channels.make_standard_montage(montage_name)
    montage_names = set(montage.ch_names)
    roles = channel_roles_from_config(config)
    ignored = set(roles.unused) | set(roles.misc) | set(roles.eog)
    if roles.provisional_reference:
        ignored.add(roles.provisional_reference)
    rows = []
    for channel, kind in zip(raw.ch_names, raw.get_channel_types()):
        is_true_eeg = kind == "eeg" and channel not in ignored
        rows.append(
            {
                "channel": channel,
                "channel_type": kind,
                "checked_for_montage": is_true_eeg,
                "in_montage": (channel in montage_names) if is_true_eeg else pd.NA,
                "montage": montage_name,
                "ignored_for_montage_reason": _montage_ignore_reason(channel, kind, roles),
            }
        )
    report = pd.DataFrame(rows)
    report["montage_applied"] = False
    return montage, report


def apply_montage_if_safe(raw, config: Mapping[str, Any]) -> tuple[object, pd.DataFrame, list[str]]:
    """Apply montage to known EEG channels and report unresolved true EEG names."""

    montage, report = validate_montage(raw, config)
    unknown = report[(report["checked_for_montage"] == True) & (report["in_montage"] == False)]  # noqa: E712
    warnings = []
    if not unknown.empty:
        warnings.append("Montage applied to known EEG channels; unresolved true EEG channels missing from montage: " + ", ".join(unknown["channel"].astype(str)))
    raw.set_montage(montage, on_missing="ignore")
    report["montage_applied"] = True
    report["montage_application"] = "partial_with_unresolved_eeg" if not unknown.empty else "complete"
    return raw, report, warnings


def raw_qc_summary(raw, subject: str, recording_part: str, config: Mapping[str, Any], alignment: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create one-row raw-level QC summary."""

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    duration_s = n_times / sfreq if sfreq else np.nan
    expected_sfreq = float(config.get("recordings", {}).get("expected_sampling_rate_hz", 500.0))
    qc_cfg = dict(config.get("preprocessing", {}).get("raw_qc", {}) or {})
    event_counts = {}
    if alignment is not None and not alignment.empty and "alignment_status" in alignment:
        event_counts = alignment["alignment_status"].value_counts(dropna=False).to_dict()
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "recording_part": recording_part,
                "sfreq_hz": sfreq,
                "expected_sfreq_hz": expected_sfreq,
                "sampling_rate_ok": abs(sfreq - expected_sfreq) <= float(config.get("recordings", {}).get("sampling_rate_tolerance_hz", 0.001)),
                "n_samples": n_times,
                "duration_s": duration_s,
                "duration_ok": duration_s >= float(qc_cfg.get("min_duration_s", 0)),
                "n_channels": len(raw.ch_names),
                "event_status_counts": event_counts,
            }
        ]
    )


def channel_qc_table(
    raw,
    subject: str,
    recording_part: str,
    config: Mapping[str, Any],
    *,
    correlation_sample_segments: Sequence[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Compute transparent proposal metrics over bounded distributed windows."""

    qc_cfg = dict(config.get("preprocessing", {}).get("raw_qc", {}) or {})
    roles = channel_roles_from_config(config)
    qc_pick_list = eeg_qc_picks(raw, config)
    qc_picks = set(qc_pick_list)
    windows = qc_sample_windows(raw, qc_cfg, "channel_qc")
    window_text = _format_qc_windows(windows)
    sfreq = float(raw.info["sfreq"])
    eligible_data = [
        np.vstack([raw.get_data(picks=[channel], start=start, stop=stop)[0] for channel in qc_pick_list])
        for start, stop in windows
    ] if qc_pick_list else []
    _, deviation_by_channel, robust_centres, robust_scales = _complementary_channel_metrics(
        eligible_data,
        qc_pick_list,
        qc_cfg,
    )
    (
        correlation_by_channel,
        dropout_by_channel,
        correlation_nonfinite_counts,
        correlation_window_count,
    ) = prep_style_correlation_metrics(
        raw,
        qc_pick_list,
        qc_cfg,
        sample_segments=correlation_sample_segments,
    )
    rows = []
    for idx, channel in enumerate(raw.ch_names):
        if channel in qc_picks:
            pick_index = qc_pick_list.index(channel)
            window_values = [data[pick_index] for data in eligible_data]
        else:
            window_values = [raw.get_data(picks=[idx], start=start, stop=stop)[0] for start, stop in windows]
        values = np.concatenate(window_values) if window_values else np.asarray([], dtype=float)
        finite_values = values[np.isfinite(values)]
        kind = raw.get_channel_types(picks=[idx])[0]
        std_uv = float(np.std(finite_values) * 1e6) if finite_values.size else np.nan
        ptp_uv = float((np.max(finite_values) - np.min(finite_values)) * 1e6) if finite_values.size else np.nan
        sampled_non_finite_count = int(values.size - finite_values.size)
        correlation_non_finite_count = int(correlation_nonfinite_counts.get(channel, 0))
        non_finite_count = max(sampled_non_finite_count, correlation_non_finite_count)
        correlation_sample_count = correlation_window_count * int(
            round(float(qc_cfg.get("poor_correlation_window_duration_s", 1.0)) * sfreq)
        )
        non_finite_denominator = max(values.size, correlation_sample_count)
        non_finite_fraction = float(non_finite_count / non_finite_denominator) if non_finite_denominator else 0.0
        non_finite = non_finite_count > 0
        is_provisional = channel == roles.provisional_reference
        exclude_from_flat_qc = is_provisional or channel not in qc_picks
        exclude_from_clipping_qc = is_provisional or channel not in qc_picks
        flat = std_uv <= float(qc_cfg.get("flat_std_uv", 0.01))
        extreme = std_uv >= float(qc_cfg.get("extreme_std_uv", 500)) or ptp_uv >= float(qc_cfg.get("extreme_peak_to_peak_uv", 5000))
        clipping_count = int(np.sum((values == np.max(finite_values)) | (values == np.min(finite_values)))) if finite_values.size else 0
        clipping_fraction = float(clipping_count / values.size) if values.size else 0.0
        clipping = clipping_count >= int(qc_cfg.get("clipping_min_repeated_samples", 3)) and clipping_fraction >= float(qc_cfg.get("clipping_fraction_threshold", 0.001))
        correlations = correlation_by_channel.get(channel, [])
        poor_correlation_fraction = float(np.mean([not np.isfinite(value) or value < float(qc_cfg.get("poor_correlation_threshold", 0.4)) for value in correlations])) if correlations else np.nan
        poor_correlation = channel in qc_picks and bool(correlations) and poor_correlation_fraction > float(qc_cfg.get("poor_correlation_bad_window_fraction", 0.01))
        dropout_windows = dropout_by_channel.get(channel, [])
        dropout_fraction = float(np.mean(dropout_windows)) if dropout_windows else np.nan
        intermittent_dropout = channel in qc_picks and bool(dropout_windows) and dropout_fraction > float(qc_cfg.get("poor_correlation_bad_window_fraction", 0.01))
        deviations = deviation_by_channel.get(channel, [])
        max_deviation = float(np.nanmax(deviations)) if deviations and np.isfinite(deviations).any() else np.nan
        robust_deviation = channel in qc_picks and np.isfinite(max_deviation) and max_deviation >= float(qc_cfg.get("robust_deviation_z_threshold", 5.0))
        metric_flags = {
            "non_finite": bool(non_finite and channel in qc_picks),
            "flat": bool(flat and channel in qc_picks and not exclude_from_flat_qc),
            "extreme_amplitude": bool(extreme and channel in qc_picks),
            "clipping": bool(clipping and channel in qc_picks and not exclude_from_clipping_qc),
            "poor_correlation": bool(poor_correlation),
            "intermittent_dropout": bool(intermittent_dropout),
            "robust_relative_deviation": bool(robust_deviation),
        }
        # The original raw/all-channel correlation rule was rejected as a
        # production proposal rule by the locked Phase 2b validation. Preserve
        # its transparent score as metric-only evidence, but do not let it alone
        # expand the human-review shortlist to the whole montage.
        shortlist_flags = {reason: flagged for reason, flagged in metric_flags.items() if reason != "poor_correlation"}
        metric_flag_reasons = [reason for reason, flagged in metric_flags.items() if flagged]
        proposal_reasons = [reason for reason, flagged in shortlist_flags.items() if flagged]
        proposed = bool(proposal_reasons)
        review_priority = "REVIEW_SUGGESTED" if proposed else "METRIC_FLAG_ONLY" if metric_flag_reasons else "NO_AUTOMATIC_FLAG"
        rows.append(
            {
                "subject": subject,
                "recording_part": recording_part,
                "channel": channel,
                "channel_type": kind,
                "role": _channel_role(channel, roles),
                "included_in_eeg_qc": channel in qc_picks,
                "is_provisional_reference_channel": is_provisional,
                "exclude_from_flat_qc": exclude_from_flat_qc,
                "exclude_from_clipping_qc": exclude_from_clipping_qc,
                "std_uv": std_uv,
                "peak_to_peak_uv": ptp_uv,
                "non_finite_sample_count": non_finite_count,
                "non_finite_fraction": non_finite_fraction,
                "clipping_fraction": clipping_fraction,
                "clipping_repeated_sample_count": clipping_count,
                "qc_start_sample": windows[0][0],
                "qc_stop_sample": windows[-1][1],
                "qc_duration_s": sum(stop - start for start, stop in windows) / sfreq if sfreq else np.nan,
                "qc_window_count": len(windows),
                "qc_sample_windows": window_text,
                "qc_time_windows_s": _format_qc_windows_seconds(windows, sfreq),
                "window_std_uv": _format_metric_values([np.nanstd(window) * 1e6 for window in window_values]),
                "minimum_window_p98_abs_correlation": float(np.nanmin(correlations)) if correlations and np.isfinite(correlations).any() else np.nan,
                "median_window_p98_abs_correlation": float(np.nanmedian(correlations)) if correlations and np.isfinite(correlations).any() else np.nan,
                "poor_correlation_window_fraction": poor_correlation_fraction,
                "poor_correlation_dropout_window_fraction": dropout_fraction,
                "poor_correlation_window_p98_abs_correlations": _format_metric_values(correlations),
                "poor_correlation_window_count": correlation_window_count,
                "poor_correlation_window_duration_s": float(qc_cfg.get("poor_correlation_window_duration_s", 1.0)),
                "poor_correlation_peer_percentile": float(qc_cfg.get("poor_correlation_peer_percentile", 0.98)),
                "poor_correlation_filter_highpass_hz": float(qc_cfg.get("poor_correlation_filter_highpass_hz", 1.0)),
                "poor_correlation_filter_lowpass_hz": float(qc_cfg.get("poor_correlation_filter_lowpass_hz", 50.0)),
                "poor_correlation_filter_method": str(qc_cfg.get("poor_correlation_filter_method", "scipy_butterworth_sosfiltfilt")),
                "poor_correlation_reference_state": str(qc_cfg.get("poor_correlation_reference_state", "acquisition_reference")),
                "poor_correlation_data_scope": str(qc_cfg.get("poor_correlation_data_scope", "full_recording")),
                "max_robust_deviation_z": max_deviation,
                "window_robust_deviation_z": _format_metric_values(deviations),
                "window_robust_log_std_median": _format_metric_values(robust_centres),
                "window_robust_log_std_scale": _format_metric_values(robust_scales),
                "flat_std_uv_threshold": float(qc_cfg.get("flat_std_uv", 0.01)),
                "extreme_std_uv_threshold": float(qc_cfg.get("extreme_std_uv", 500)),
                "extreme_peak_to_peak_uv_threshold": float(qc_cfg.get("extreme_peak_to_peak_uv", 5000)),
                "clipping_fraction_threshold": float(qc_cfg.get("clipping_fraction_threshold", 0.001)),
                "poor_correlation_threshold": float(qc_cfg.get("poor_correlation_threshold", 0.4)),
                "poor_correlation_bad_window_fraction_threshold": float(qc_cfg.get("poor_correlation_bad_window_fraction", 0.01)),
                "robust_deviation_z_threshold": float(qc_cfg.get("robust_deviation_z_threshold", 5.0)),
                "threshold_status": "provisional_proposal_thresholds_not_validated_exclusion_criteria",
                "non_finite_proposal": metric_flags["non_finite"],
                "flat_proposal": metric_flags["flat"],
                "extreme_amplitude_proposal": metric_flags["extreme_amplitude"],
                "clipping_proposal": metric_flags["clipping"],
                "poor_correlation_proposal": metric_flags["poor_correlation"],
                "intermittent_dropout_proposal": metric_flags["intermittent_dropout"],
                "robust_deviation_proposal": metric_flags["robust_relative_deviation"],
                "automatic_metric_flag_reasons": ";".join(metric_flag_reasons),
                "automatic_proposal_reasons": ";".join(proposal_reasons),
                "automatic_review_priority": review_priority,
                "proposed_bad_channel": bool(proposed),
            }
        )
    if roles.provisional_reference and roles.provisional_reference not in raw.ch_names:
        rows.append(
            {
                "subject": subject,
                "recording_part": recording_part,
                "channel": roles.provisional_reference,
                "channel_type": "eeg",
                "role": "provisional_reference",
                "included_in_eeg_qc": False,
                "is_provisional_reference_channel": True,
                "exclude_from_flat_qc": True,
                "exclude_from_clipping_qc": True,
                "std_uv": 0.0,
                "peak_to_peak_uv": 0.0,
                "non_finite_sample_count": 0,
                "non_finite_fraction": 0.0,
                "clipping_fraction": 0.0,
                "qc_start_sample": windows[0][0],
                "qc_stop_sample": windows[-1][1],
                "qc_duration_s": sum(stop - start for start, stop in windows) / sfreq if sfreq else np.nan,
                "qc_window_count": len(windows),
                "qc_sample_windows": window_text,
                "qc_time_windows_s": _format_qc_windows_seconds(windows, sfreq),
                "non_finite_proposal": False,
                "flat_proposal": False,
                "extreme_amplitude_proposal": False,
                "clipping_proposal": False,
                "poor_correlation_proposal": False,
                "intermittent_dropout_proposal": False,
                "robust_deviation_proposal": False,
                "automatic_metric_flag_reasons": "",
                "automatic_proposal_reasons": "",
                "automatic_review_priority": "NO_AUTOMATIC_FLAG",
                "proposed_bad_channel": False,
                "physical_channel_present": False,
            }
        )
    df = pd.DataFrame(rows)
    if "physical_channel_present" not in df:
        df["physical_channel_present"] = True
    else:
        df["physical_channel_present"] = df["physical_channel_present"].where(df["physical_channel_present"].notna(), True).astype(bool)
    return df


def psd_line_noise_summary(raw, subject: str, recording_part: str, config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute coarse PSD and 50 Hz summaries over distributed windows."""

    qc_cfg = dict(config.get("preprocessing", {}).get("raw_qc", {}) or {})
    line_hz = float(config.get("preprocessing", {}).get("line_frequency_hz", 50.0))
    bandwidth = float(qc_cfg.get("line_noise_bandwidth_hz", 1.0))
    fmax = float(qc_cfg.get("psd_fmax_hz", 120.0))
    picks = eeg_qc_picks(raw, config)
    if not picks:
        return pd.DataFrame()
    windows = qc_sample_windows(raw, qc_cfg, "psd")
    psds = [
        raw.compute_psd(
            picks=picks,
            fmin=1.0,
            fmax=fmax,
            tmin=start / float(raw.info["sfreq"]),
            tmax=(stop - 1) / float(raw.info["sfreq"]),
            verbose=False,
        )
        for start, stop in windows
    ]
    freqs = psds[0].freqs
    window_psd_data = [psd.get_data() for psd in psds]
    data = np.nanmean(np.stack(window_psd_data, axis=0), axis=0)
    rows = []
    line_mask = (freqs >= line_hz - bandwidth) & (freqs <= line_hz + bandwidth)
    broad_mask = (freqs >= max(1.0, line_hz - 10.0)) & (freqs <= line_hz + 10.0)
    threshold = float(qc_cfg.get("line_noise_ratio_threshold", 0.5))
    for channel_index, (channel, values) in enumerate(zip(picks, data)):
        line_power = float(np.nanmean(values[line_mask])) if np.any(line_mask) else np.nan
        broad_power = float(np.nanmean(values[broad_mask])) if np.any(broad_mask) else np.nan
        ratio = line_power / broad_power if broad_power and not np.isnan(broad_power) else np.nan
        window_ratios = []
        for window_values in window_psd_data:
            channel_values = window_values[channel_index]
            window_line_power = float(np.nanmean(channel_values[line_mask])) if np.any(line_mask) else np.nan
            window_broad_power = float(np.nanmean(channel_values[broad_mask])) if np.any(broad_mask) else np.nan
            window_ratios.append(window_line_power / window_broad_power if window_broad_power and not np.isnan(window_broad_power) else np.nan)
        maximum_window_ratio = float(np.nanmax(window_ratios)) if np.isfinite(window_ratios).any() else np.nan
        rows.append(
            {
                "subject": subject,
                "recording_part": recording_part,
                "channel": channel,
                "line_frequency_hz": line_hz,
                "line_power": line_power,
                "broadband_power_near_line": broad_power,
                "line_noise_ratio": ratio,
                "window_line_noise_ratios": _format_metric_values(window_ratios),
                "maximum_window_line_noise_ratio": maximum_window_ratio,
                "line_noise_ratio_threshold": threshold,
                "excessive_line_noise": bool(maximum_window_ratio > threshold) if not np.isnan(maximum_window_ratio) else False,
                "qc_start_sample": windows[0][0],
                "qc_stop_sample": windows[-1][1],
                "qc_duration_s": sum(stop - start for start, stop in windows) / float(raw.info["sfreq"]) if raw.info["sfreq"] else np.nan,
                "qc_window_count": len(windows),
                "qc_sample_windows": _format_qc_windows(windows),
                "qc_time_windows_s": _format_qc_windows_seconds(windows, float(raw.info["sfreq"])),
            }
        )
    return pd.DataFrame(rows)


def bad_channel_decision_table(
    channel_qc: pd.DataFrame,
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    confirmed_bad_channels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Keep automatic proposals separate from authoritative confirmed channels."""

    # The validated decision reader is the sole authority. Legacy config
    # overrides and automatic proposals must never populate raw.info["bads"].
    confirmed = set(confirmed_bad_channels or [])
    rows = []
    for _, row in channel_qc.iterrows():
        channel = str(row["channel"])
        proposed = bool(row.get("proposed_bad_channel", False))
        metric_reasons = str(row.get("automatic_metric_flag_reasons", row.get("automatic_proposal_reasons", "")) or "")
        review_priority = str(row.get("automatic_review_priority", "REVIEW_SUGGESTED" if proposed else "NO_AUTOMATIC_FLAG"))
        manual_confirmed = channel in confirmed
        rows.append(
            {
                "subject": subject,
                "recording_part": recording_part,
                "channel": channel,
                "automatic_bad_channel_proposal": proposed,
                "automatic_metric_flag_reasons": metric_reasons,
                "automatic_proposal_reasons": str(row.get("automatic_proposal_reasons", "") or ""),
                "automatic_review_priority": review_priority,
                "manual_confirmed_bad_channel": manual_confirmed,
                "final_bad_channel_for_gate3a": manual_confirmed,
                "decision": "manual_confirmed_bad" if manual_confirmed else "proposal_only" if proposed else "metric_flag_only" if review_priority == "METRIC_FLAG_ONLY" else "ok",
            }
        )
    return pd.DataFrame(rows)


def run_bad_channel_qc(
    raw,
    subject: str,
    recording_part: str,
    config: Mapping[str, Any],
    *,
    confirmed_bad_channels: Sequence[str] | None = None,
    correlation_sample_segments: Sequence[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the actual proposal → manual confirmation → warning orchestration."""

    channel_qc = channel_qc_table(
        raw,
        subject,
        recording_part,
        config,
        correlation_sample_segments=correlation_sample_segments,
    )
    decisions = bad_channel_decision_table(
        channel_qc,
        config,
        subject,
        recording_part,
        confirmed_bad_channels=confirmed_bad_channels,
    )
    spatial_warnings = bad_channel_spatial_warning_table(raw, channel_qc, decisions, config)
    raw.info["bads"] = decisions.loc[decisions["manual_confirmed_bad_channel"].astype(bool), "channel"].astype(str).tolist()
    return channel_qc, decisions, spatial_warnings


def bad_channel_spatial_warning_table(
    raw,
    channel_qc: pd.DataFrame,
    bad_channel_decisions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build proposal/confirmed spatial review warnings without exclusions."""

    qc_cfg = dict(config.get("preprocessing", {}).get("raw_qc", {}) or {})
    eligible = channel_qc.loc[channel_qc["included_in_eeg_qc"].astype(bool), "channel"].astype(str).tolist()
    positions = _finite_eeg_positions(raw, eligible)
    geometry_omitted = sorted(set(eligible) - set(positions))
    radii = {channel: float(np.linalg.norm(position[:2])) for channel, position in positions.items()}
    edge_cutoff = float(np.quantile(list(radii.values()), float(qc_cfg.get("montage_edge_quantile", 0.8)))) if radii else np.nan
    rois = {str(name): [str(channel) for channel in dict(value or {}).get("channels", []) or []] for name, value in dict(config.get("rois", {}) or {}).items()}
    scopes = {
        "automatic_proposal": bad_channel_decisions.loc[bad_channel_decisions["automatic_bad_channel_proposal"].astype(bool), "channel"].astype(str).tolist(),
        "manual_confirmed": bad_channel_decisions.loc[bad_channel_decisions["manual_confirmed_bad_channel"].astype(bool), "channel"].astype(str).tolist(),
    }
    rows = []
    for scope, affected in scopes.items():
        affected = sorted(set(affected) & set(eligible))
        fraction = len(affected) / len(eligible) if eligible else 0.0
        high_fraction = fraction >= float(qc_cfg.get("bad_channel_fraction_warn", 0.2))
        roi_parts = []
        concentrated_rois = []
        for roi, members in rois.items():
            overlap = sorted(set(affected) & set(members))
            if not overlap:
                continue
            roi_parts.append(f"{roi}:{'|'.join(overlap)}")
            roi_fraction = len(overlap) / len(set(members)) if members else 0.0
            if len(overlap) >= int(qc_cfg.get("roi_bad_count_warn", 2)) and roi_fraction >= float(qc_cfg.get("roi_bad_fraction_warn", 0.5)):
                concentrated_rois.append(roi)
        resolved_affected = [channel for channel in affected if channel in positions]
        adjacent_pairs = []
        distance_threshold = float(qc_cfg.get("spatial_cluster_distance_m", 0.06))
        for left_index, left in enumerate(resolved_affected):
            for right in resolved_affected[left_index + 1 :]:
                if float(np.linalg.norm(positions[left] - positions[right])) <= distance_threshold:
                    adjacent_pairs.append(f"{left}|{right}")
        edge_channels = [channel for channel in resolved_affected if np.isfinite(edge_cutoff) and radii[channel] >= edge_cutoff]
        affected_unresolved = sorted(set(affected) & set(geometry_omitted))
        messages = []
        if high_fraction:
            messages.append("high_bad_channel_fraction_manual_review_required")
        if adjacent_pairs:
            messages.append("spatial_cluster_manual_review_required")
        if concentrated_rois:
            messages.append("roi_concentration_manual_review_required")
        if edge_channels:
            messages.append("montage_edge_channels_manual_review_required")
        if geometry_omitted:
            messages.append("geometry_dependent_review_limited_by_unresolved_channels")
        rows.append(
            {
                "scope": scope,
                "n_eeg_qc_channels": len(eligible),
                "n_affected_channels": len(affected),
                "affected_fraction": fraction,
                "affected_channels": ";".join(affected),
                "bad_channel_fraction_warn_threshold": float(qc_cfg.get("bad_channel_fraction_warn", 0.2)),
                "high_fraction_warning": high_fraction,
                "roi_overlaps": ";".join(roi_parts),
                "concentrated_rois": ";".join(concentrated_rois),
                "roi_concentration_warning": bool(concentrated_rois),
                "adjacent_channel_pairs": ";".join(adjacent_pairs),
                "spatial_cluster_warning": bool(adjacent_pairs),
                "edge_channels": ";".join(edge_channels),
                "edge_position_warning": bool(edge_channels),
                "geometry_omitted_channels": ";".join(geometry_omitted),
                "affected_unresolved_channels": ";".join(affected_unresolved),
                "geometry_limited": bool(geometry_omitted),
                "warning_reasons": ";".join(messages),
                "automatic_exclusion": False,
                "requires_manual_review": bool(messages),
            }
        )
    return pd.DataFrame(rows)


def eog_activity_summary(raw, subject: str, recording_part: str, config: Mapping[str, Any]) -> pd.DataFrame:
    """Summarise EOG amplitudes over the distributed channel-QC windows."""

    roles = channel_roles_from_config(config)
    qc_cfg = dict(config.get("preprocessing", {}).get("raw_qc", {}) or {})
    windows = qc_sample_windows(raw, qc_cfg, "channel_qc")
    rows = []
    for channel in roles.eog:
        if channel not in raw.ch_names:
            rows.append({"subject": subject, "recording_part": recording_part, "channel": channel, "present": False})
            continue
        values = np.concatenate([raw.get_data(picks=[channel], start=start, stop=stop)[0] for start, stop in windows])
        rows.append(
            {
                "subject": subject,
                "recording_part": recording_part,
                "channel": channel,
                "present": True,
                "std_uv": float(np.nanstd(values) * 1e6),
                "peak_to_peak_uv": float((np.nanmax(values) - np.nanmin(values)) * 1e6),
                "qc_start_sample": windows[0][0],
                "qc_stop_sample": windows[-1][1],
                "qc_duration_s": sum(stop - start for start, stop in windows) / float(raw.info["sfreq"]) if raw.info["sfreq"] else np.nan,
                "qc_window_count": len(windows),
                "qc_sample_windows": _format_qc_windows(windows),
            }
        )
    return pd.DataFrame(rows)


def _channel_role(channel: str, roles: ChannelRoleConfig) -> str:
    if channel == roles.provisional_reference:
        return "provisional_reference"
    if channel in roles.eeg:
        return "eeg"
    if channel in roles.eog:
        return "eog"
    if channel in roles.unused:
        return "unused"
    if channel in roles.misc:
        return "misc"
    return "unknown"


def qc_sample_windows(raw, qc_cfg: Mapping[str, Any], prefix: str = "channel_qc") -> list[tuple[int, int]]:
    """Return deterministic, non-overlapping windows spanning the recording."""

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    if n_times <= 0:
        return [(0, 0)]
    requested_count = max(1, int(qc_cfg.get("distributed_window_count", 1)))
    duration_s = qc_cfg.get(f"{prefix}_window_duration_s", qc_cfg.get("distributed_window_duration_s"))
    if duration_s is None:
        total_duration_s = qc_cfg.get(f"{prefix}_duration_s", qc_cfg.get("channel_qc_duration_s"))
        duration_s = (float(total_duration_s) / requested_count) if total_duration_s is not None else n_times / sfreq
    window_samples = min(n_times, max(1, int(round(float(duration_s) * sfreq))))
    count = min(requested_count, max(1, n_times // window_samples))
    if count == 1:
        configured_start_s = float(qc_cfg.get(f"{prefix}_start_s", qc_cfg.get("channel_qc_start_s", 0.0)) or 0.0)
        start = min(max(0, int(round(configured_start_s * sfreq))), max(0, n_times - window_samples))
        return [(start, start + window_samples)]
    starts = np.linspace(0, n_times - window_samples, num=count)
    return [(int(round(start)), int(round(start)) + window_samples) for start in starts]


def _qc_sample_window(raw, qc_cfg: Mapping[str, Any], prefix: str) -> tuple[int, int]:
    """Compatibility wrapper returning the first distributed QC window."""

    return qc_sample_windows(raw, qc_cfg, prefix)[0]


def prep_style_correlation_metrics(
    raw,
    channels: Sequence[str],
    qc_cfg: Mapping[str, Any],
    *,
    sample_segments: Sequence[tuple[int, int]] | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[bool]], dict[str, int], int]:
    """Compute bounded-memory PREP-style correlation and dropout evidence.

    This is a QC-only transformation. It applies a deterministic 1--50 Hz
    Butterworth filter to overlapping processing blocks, evaluates
    non-overlapping 1 s windows, excludes self-correlation, and retains each
    channel's 98th-percentile absolute peer correlation. The caller's ``raw``
    object and downstream analysis data are never modified.

    PyPREP uses its own detrending and 50 Hz FIR implementation. The filter
    here is deliberately local because PyPREP is not a project dependency;
    the scientifically consequential window/statistic/threshold behavior is
    matched and all filter parameters are configuration-bound.
    """

    names = [str(channel) for channel in channels]
    scores = {channel: [] for channel in names}
    dropouts = {channel: [] for channel in names}
    nonfinite_counts = {channel: 0 for channel in names}
    if not names:
        return scores, dropouts, nonfinite_counts, 0

    sfreq = float(raw.info["sfreq"])
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("PREP-style correlation QC requires a positive finite sampling rate.")
    window_samples = int(round(float(qc_cfg.get("poor_correlation_window_duration_s", 1.0)) * sfreq))
    if window_samples < 2:
        raise ValueError("PREP-style correlation QC window must contain at least two samples.")
    percentile = float(qc_cfg.get("poor_correlation_peer_percentile", 0.98))
    if not 0 < percentile <= 1:
        raise ValueError("PREP-style correlation peer percentile must be in (0, 1].")

    segments = _validated_correlation_segments(raw, sample_segments)
    chunk_samples = max(
        window_samples,
        int(round(float(qc_cfg.get("poor_correlation_filter_chunk_duration_s", 300.0)) * sfreq)),
    )
    chunk_samples -= chunk_samples % window_samples
    padding_samples = max(0, int(round(float(qc_cfg.get("poor_correlation_filter_padding_s", 5.0)) * sfreq)))
    flat_threshold_v = float(qc_cfg.get("poor_correlation_dropout_flat_threshold_v", 1e-15))
    sos = _prep_style_correlation_filter(qc_cfg, sfreq)
    window_count = 0

    from scipy.signal import sosfiltfilt

    for segment_start, segment_stop in segments:
        core_start = segment_start
        usable_stop = segment_start + ((segment_stop - segment_start) // window_samples) * window_samples
        while core_start < usable_stop:
            core_stop = min(usable_stop, core_start + chunk_samples)
            read_start = max(segment_start, core_start - padding_samples)
            read_stop = min(segment_stop, core_stop + padding_samples)
            data = np.asarray(raw.get_data(picks=names, start=read_start, stop=read_stop), dtype=float)
            expected_shape = (len(names), read_stop - read_start)
            if data.shape != expected_shape:
                raise ValueError(
                    f"PREP-style correlation QC received data shape {data.shape}; expected {expected_shape}."
                )
            finite = np.isfinite(data)
            core_slice = slice(core_start - read_start, core_stop - read_start)
            core_finite = finite[:, core_slice]
            for index, channel in enumerate(names):
                nonfinite_counts[channel] += int(np.size(core_finite[index]) - np.count_nonzero(core_finite[index]))

            filter_input = np.where(finite, data, 0.0)
            if sos is None:
                filtered = filter_input
            else:
                filtered = sosfiltfilt(sos, filter_input, axis=1)
            filtered_core = filtered[:, core_slice]
            raw_core = data[:, core_slice]

            for offset in range(0, filtered_core.shape[1], window_samples):
                stop = offset + window_samples
                filtered_window = filtered_core[:, offset:stop]
                raw_window = raw_core[:, offset:stop]
                finite_window = np.isfinite(raw_window).all(axis=1)
                with np.errstate(invalid="ignore"):
                    window_mad = np.nanmedian(
                        np.abs(raw_window - np.nanmedian(raw_window, axis=1, keepdims=True)),
                        axis=1,
                    )
                dropout_window = (~finite_window) | (~np.isfinite(window_mad)) | (window_mad <= flat_threshold_v)
                usable = ~dropout_window
                window_scores = np.zeros(len(names), dtype=float)
                if int(np.count_nonzero(usable)) >= 2:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        matrix = np.abs(np.corrcoef(filtered_window[usable]))
                    np.fill_diagonal(matrix, np.nan)
                    usable_scores = np.asarray(
                        [
                            float(np.quantile(row[np.isfinite(row)], percentile))
                            if np.isfinite(row).any()
                            else np.nan
                            for row in matrix
                        ],
                        dtype=float,
                    )
                    window_scores[usable] = usable_scores
                elif int(np.count_nonzero(usable)) == 1:
                    window_scores[usable] = np.nan
                for index, channel in enumerate(names):
                    scores[channel].append(float(window_scores[index]))
                    dropouts[channel].append(bool(dropout_window[index]))
                window_count += 1
            core_start = core_stop

    if window_count == 0:
        raise ValueError("PREP-style correlation QC found no complete analysis windows.")
    return scores, dropouts, nonfinite_counts, window_count


def _validated_correlation_segments(raw, sample_segments: Sequence[tuple[int, int]] | None) -> list[tuple[int, int]]:
    n_times = int(raw.n_times)
    candidates = list(sample_segments) if sample_segments is not None else [(0, n_times)]
    segments: list[tuple[int, int]] = []
    for start, stop in candidates:
        start_i = max(0, int(start))
        stop_i = min(n_times, int(stop))
        if stop_i <= start_i:
            continue
        if segments and start_i < segments[-1][0]:
            raise ValueError("PREP-style correlation QC sample segments must be ordered.")
        if segments and start_i <= segments[-1][1]:
            segments[-1] = (segments[-1][0], max(segments[-1][1], stop_i))
            continue
        segments.append((start_i, stop_i))
    if not segments:
        raise ValueError("PREP-style correlation QC requires at least one non-empty sample segment.")
    return segments


def _prep_style_correlation_filter(qc_cfg: Mapping[str, Any], sfreq: float):
    from scipy.signal import butter

    method = str(qc_cfg.get("poor_correlation_filter_method", "scipy_butterworth_sosfiltfilt"))
    if method != "scipy_butterworth_sosfiltfilt":
        raise ValueError(f"Unsupported poor-correlation QC filter method: {method}")
    order = int(qc_cfg.get("poor_correlation_filter_order", 4))
    highpass = float(qc_cfg.get("poor_correlation_filter_highpass_hz", 1.0))
    lowpass = float(qc_cfg.get("poor_correlation_filter_lowpass_hz", 50.0))
    nyquist = sfreq / 2.0
    if order < 1 or highpass <= 0 or lowpass <= highpass:
        raise ValueError("Invalid PREP-style correlation QC filter parameters.")
    if highpass >= nyquist:
        raise ValueError("PREP-style correlation QC high-pass must be below Nyquist.")
    if lowpass >= nyquist:
        return butter(order, highpass, btype="highpass", fs=sfreq, output="sos")
    return butter(order, [highpass, lowpass], btype="bandpass", fs=sfreq, output="sos")


def _complementary_channel_metrics(
    window_data: Sequence[np.ndarray],
    channels: Sequence[str],
    qc_cfg: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[float], list[float]]:
    correlations = {str(channel): [] for channel in channels}
    deviations = {str(channel): [] for channel in channels}
    centres = []
    scales = []
    minimum_scale = float(qc_cfg.get("robust_deviation_min_log_scale", 0.05))
    for data in window_data:
        if len(channels) >= 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                matrix = np.abs(np.corrcoef(data))
            np.fill_diagonal(matrix, np.nan)
            correlation_scores = np.asarray(
                [float(np.median(row[np.isfinite(row)])) if np.isfinite(row).any() else np.nan for row in matrix],
                dtype=float,
            )
        else:
            correlation_scores = np.full(len(channels), np.nan)
        std_uv = np.nanstd(data, axis=1) * 1e6
        log_std = np.log10(np.maximum(std_uv, np.finfo(float).tiny))
        centre = float(np.nanmedian(log_std))
        mad = float(np.nanmedian(np.abs(log_std - centre)))
        scale = max(1.4826 * mad, minimum_scale)
        deviation_scores = (log_std - centre) / scale
        centres.append(centre)
        scales.append(scale)
        for index, channel in enumerate(channels):
            correlations[str(channel)].append(float(correlation_scores[index]))
            deviations[str(channel)].append(float(deviation_scores[index]))
    return correlations, deviations, centres, scales


def _finite_eeg_positions(raw, channels: Sequence[str]) -> dict[str, np.ndarray]:
    positions = {}
    for channel in channels:
        if channel not in raw.ch_names:
            continue
        index = raw.ch_names.index(channel)
        position = np.asarray(raw.info["chs"][index]["loc"][:3], dtype=float)
        if np.isfinite(position).all() and not np.allclose(position, 0.0):
            positions[str(channel)] = position
    return positions


def _format_qc_windows(windows: Sequence[tuple[int, int]]) -> str:
    return ";".join(f"{start}:{stop}" for start, stop in windows)


def _format_qc_windows_seconds(windows: Sequence[tuple[int, int]], sfreq: float) -> str:
    return ";".join(f"{start / sfreq:.6f}:{stop / sfreq:.6f}" for start, stop in windows) if sfreq else ""


def _format_metric_values(values: Sequence[float]) -> str:
    return ";".join("nan" if not np.isfinite(value) else f"{float(value):.9g}" for value in values)


def _montage_ignore_reason(channel: str, kind: str, roles: ChannelRoleConfig) -> str:
    if channel == roles.provisional_reference:
        return "provisional_reference_channel"
    if channel in roles.eog or kind == "eog":
        return "eog_channel"
    if channel in roles.unused:
        return "unused_misc_channel"
    if channel in roles.misc or kind == "misc":
        return "misc_channel"
    return ""
