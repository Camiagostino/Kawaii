from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def generate_gate3a_markdown_report(
    *,
    subject: str,
    recording_part: str,
    input_files: Sequence[str | Path],
    output_files: Mapping[str, str | Path],
    raw_qc_summary: pd.DataFrame,
    channel_qc: pd.DataFrame,
    bad_channel_decisions: pd.DataFrame,
    bad_channel_spatial_warnings: pd.DataFrame | None = None,
    preprocessing_decisions: pd.DataFrame,
    warnings: Sequence[str] | None = None,
    failures: Sequence[str] | None = None,
    manual_review_files: Mapping[str, str | Path] | None = None,
) -> str:
    """Create a compact Markdown QC report for one participant recording part."""

    warnings = list(warnings or [])
    failures = list(failures or [])
    raw_row = raw_qc_summary.iloc[0].to_dict() if not raw_qc_summary.empty else {}
    if not bad_channel_decisions.empty:
        proposed_bad = bad_channel_decisions[bad_channel_decisions.get("automatic_bad_channel_proposal", False) == True]  # noqa: E712
        if not channel_qc.empty and "is_provisional_reference_channel" in channel_qc:
            provisional = set(channel_qc.loc[channel_qc["is_provisional_reference_channel"].astype(bool), "channel"].astype(str))
            proposed_bad = proposed_bad[~proposed_bad["channel"].astype(str).isin(provisional)]
    else:
        proposed_bad = pd.DataFrame()
    metric_only = (
        bad_channel_decisions[bad_channel_decisions.get("automatic_review_priority", "").eq("METRIC_FLAG_ONLY")]
        if not bad_channel_decisions.empty and "automatic_review_priority" in bad_channel_decisions
        else pd.DataFrame()
    )
    confirmed_bad = bad_channel_decisions[bad_channel_decisions.get("manual_confirmed_bad_channel", False) == True] if not bad_channel_decisions.empty else pd.DataFrame()  # noqa: E712
    bad_channel_spatial_warnings = pd.DataFrame() if bad_channel_spatial_warnings is None else bad_channel_spatial_warnings

    lines = [
        f"# Gate 3A Raw QC: {subject} {recording_part}",
        "",
        "## Inputs",
    ]
    lines.extend(f"- `{path}`" for path in input_files)
    lines.extend(
        [
            "",
            "## Raw Summary",
            f"- Sampling rate: {raw_row.get('sfreq_hz', 'n/a')} Hz",
            f"- Duration: {raw_row.get('duration_s', 'n/a')} s",
            f"- Samples: {raw_row.get('n_samples', 'n/a')}",
            f"- Channels: {raw_row.get('n_channels', 'n/a')}",
            "",
            "## Channel Roles And QC",
            f"- Proposed bad channels: {', '.join(proposed_bad['channel'].astype(str)) if not proposed_bad.empty else 'none'}",
            f"- Proposal reasons: {_proposal_reason_summary(proposed_bad)}",
            f"- Metric-flag-only channels: {', '.join(metric_only['channel'].astype(str)) if not metric_only.empty else 'none'}",
            f"- Metric-only reasons: {_metric_reason_summary(metric_only)}",
            f"- Manual confirmed bad channels: {', '.join(confirmed_bad['channel'].astype(str)) if not confirmed_bad.empty else 'none'}",
            f"- Provisional reference channels: {_provisional_reference_channels(channel_qc)}",
            f"- Distributed QC sample windows: {_qc_window_summary(channel_qc, 'qc_sample_windows')}",
            f"- Distributed QC time windows (s): {_qc_window_summary(channel_qc, 'qc_time_windows_s')}",
            "- Automatic thresholds are provisional proposal thresholds, not validated exclusion criteria.",
            "",
            "## Bad-Channel Spatial And ROI Review Warnings",
        ]
    )
    if bad_channel_spatial_warnings.empty:
        lines.append("- none")
    else:
        for _, row in bad_channel_spatial_warnings.iterrows():
            lines.append(
                f"- {row.get('scope')}: affected={row.get('affected_channels', '') or 'none'}; "
                f"fraction={row.get('affected_fraction', 0)}; ROI={row.get('roi_overlaps', '') or 'none'}; "
                f"clusters={row.get('adjacent_channel_pairs', '') or 'none'}; edge={row.get('edge_channels', '') or 'none'}; "
                f"geometry omitted={row.get('geometry_omitted_channels', '') or 'none'}; "
                f"warnings={row.get('warning_reasons', '') or 'none'}; automatic exclusion=False"
            )
    lines.extend(["", "## Deterministic Preprocessing Decisions"])
    if preprocessing_decisions.empty:
        lines.append("- No preprocessing decision rows available.")
    else:
        for _, row in preprocessing_decisions.iterrows():
            lines.append(f"- {row.get('decision', 'decision')}: {row.get('value', '')} ({row.get('reason', '')})")

    lines.extend(["", "## Outputs"])
    for label, path in output_files.items():
        lines.append(f"- {label}: `{path}`")

    figure_outputs = {label: path for label, path in output_files.items() if "figure" in label}
    lines.extend(["", "## QC Figures"])
    if figure_outputs:
        for label, path in figure_outputs.items():
            lines.append(f"- {label}: `{path}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Manual Review Files"])
    if manual_review_files:
        for label, path in manual_review_files.items():
            lines.append(f"- {label}: `{path}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")

    lines.extend(["", "## Failures"])
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _provisional_reference_channels(channel_qc: pd.DataFrame) -> str:
    if channel_qc.empty or "is_provisional_reference_channel" not in channel_qc:
        return "none"
    channels = channel_qc.loc[channel_qc["is_provisional_reference_channel"].astype(bool), "channel"].astype(str).tolist()
    return ", ".join(channels) if channels else "none"


def _proposal_reason_summary(proposed_bad: pd.DataFrame) -> str:
    if proposed_bad.empty or "automatic_proposal_reasons" not in proposed_bad:
        return "none"
    return ", ".join(f"{row['channel']}={row['automatic_proposal_reasons']}" for _, row in proposed_bad.iterrows())


def _metric_reason_summary(metric_only: pd.DataFrame) -> str:
    if metric_only.empty or "automatic_metric_flag_reasons" not in metric_only:
        return "none"
    return ", ".join(f"{row['channel']}={row['automatic_metric_flag_reasons']}" for _, row in metric_only.iterrows())


def _qc_window_summary(channel_qc: pd.DataFrame, column: str) -> str:
    if channel_qc.empty or column not in channel_qc:
        return "n/a"
    values = channel_qc[column].dropna().astype(str).unique().tolist()
    return values[0] if values else "n/a"


def write_markdown_report(path: str | Path, content: str) -> Path:
    """Write a Markdown report and return its path."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path


def generate_gate3b_ica_markdown_report(
    *,
    subject: str,
    recording_part: str,
    input_files: Sequence[str | Path],
    output_files: Mapping[str, str | Path],
    ica_summary: pd.DataFrame,
    eog_suggestions: pd.DataFrame,
    pca_components: pd.DataFrame,
    warnings: Sequence[str] | None = None,
    failures: Sequence[str] | None = None,
    manual_review_files: Mapping[str, str | Path] | None = None,
) -> str:
    """Create a compact Markdown report for Gate 3B ICA diagnostics."""

    warnings = list(warnings or [])
    failures = list(failures or [])
    summary = ica_summary.iloc[0].to_dict() if not ica_summary.empty else {}
    suggested = eog_suggestions[eog_suggestions.get("suggested_ocular_component", False) == True] if not eog_suggestions.empty else pd.DataFrame()  # noqa: E712

    lines = [
        f"# Gate 3B ICA Diagnostics: {subject} {recording_part}",
        "",
        "## Policy",
        "- ICA was fitted and diagnosed only.",
        "- No ICA components were removed.",
        "- Gate 3C must use manual review decisions before applying ICA.",
        "- ICA filtering/fitting uses the task-window cropped filtered raw; Gate 3A complementary metrics use distributed full-recording windows, while PREP-style correlation QC uses alignment-derived task segments.",
        "",
        "## Inputs",
    ]
    lines.extend(f"- `{path}`" for path in input_files)
    lines.extend(
        [
            "",
            "## ICA Summary",
            f"- Method: {summary.get('method', 'n/a')}",
            f"- Fitted components: {summary.get('n_components', 'n/a')}",
            f"- Requested components: {summary.get('n_components_requested', 'n/a')}",
            f"- Effective EEG rank: {summary.get('estimated_effective_rank', 'n/a')}",
            f"- Rank reduction reason: {summary.get('ica_rank_reduction_reason', 'n/a')}",
            f"- Candidate EEG channels: {summary.get('candidate_eeg_channels', 'n/a')}",
            f"- Confirmed bad EEG channels excluded: {summary.get('confirmed_bad_eeg_channels_excluded', 'n/a')}",
            f"- Fit picks: {summary.get('n_fit_channels', 'n/a')}",
            f"- Fit sampling frequency: {summary.get('ica_fit_sampling_frequency_hz', 'n/a')} Hz",
            f"- Effective fitting sampling frequency after decimation: {summary.get('ica_effective_sampling_frequency_hz', 'n/a')} Hz",
            f"- Decimation: {summary.get('decim', 'n/a')}",
            f"- Effective ICA bandwidth: {summary.get('ica_effective_highpass_hz', 'n/a')}–{summary.get('ica_effective_lowpass_hz', 'n/a')} Hz",
            f"- Random seed: {summary.get('ica_random_state', 'n/a')}",
            f"- Fit samples available/decimated: {summary.get('ica_fit_n_samples_available', 'n/a')} / {summary.get('ica_fit_n_samples_decimated', 'n/a')}",
            f"- Fit duration: {summary.get('ica_fit_duration_s', 'n/a')} s",
            f"- Excluded from fitting: {summary.get('excluded_from_ica', 'n/a')}",
            f"- Ocular detection channels: {summary.get('ocular_detection_channels', 'n/a')}",
            f"- ICA application: {summary.get('ica_application', 'not_applied_gate3b')}",
            "",
            "## Suggested Ocular Components",
        ]
    )
    if suggested.empty:
        lines.append("- none")
    else:
        for _, row in suggested.iterrows():
            lines.append(f"- component {row.get('component')} from {row.get('eog_channel')} score={row.get('score')}")

    lines.extend(["", "## PCA Components"])
    if pca_components.empty:
        lines.append("- No PCA component summary available.")
    else:
        for _, row in pca_components.head(20).iterrows():
            lines.append(f"- PCA {row.get('pca_component')}: {row.get('explained_variance_ratio')}")

    lines.extend(["", "## Outputs"])
    for label, path in output_files.items():
        lines.append(f"- {label}: `{path}`")

    lines.extend(["", "## Manual Review Files"])
    if manual_review_files:
        for label, path in manual_review_files.items():
            lines.append(f"- {label}: `{path}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")

    lines.extend(["", "## Failures"])
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def generate_gate3c_epoch_qc_markdown_report(
    *,
    subject: str,
    recording_part: str,
    input_files: Sequence[str | Path],
    output_files: Mapping[str, str | Path],
    ica_review_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    interpolation_summary: pd.DataFrame,
    condition_counts: pd.DataFrame,
    epoch_rejection_table: pd.DataFrame,
    warnings: Sequence[str] | None = None,
    failures: Sequence[str] | None = None,
) -> str:
    """Create a compact Markdown report for Gate 3C cleaned raw and epochs."""

    warnings = list(warnings or [])
    failures = list(failures or [])
    applied_ica = ica_review_summary[ica_review_summary.get("applied", False) == True] if not ica_review_summary.empty else pd.DataFrame()  # noqa: E712
    interpolated = interpolation_summary[interpolation_summary.get("interpolated", False) == True] if not interpolation_summary.empty else pd.DataFrame()  # noqa: E712
    retained_bad_mask = interpolation_summary.get(
        "retained_bad_uninterpolated",
        pd.Series(False, index=interpolation_summary.index, dtype=bool),
    )
    retained_bad = interpolation_summary[
        retained_bad_mask == True  # noqa: E712
    ] if not interpolation_summary.empty else pd.DataFrame()
    retained = int((~epoch_rejection_table["rejected"].astype(bool)).sum()) if not epoch_rejection_table.empty else 0
    rejected = int(epoch_rejection_table["rejected"].astype(bool).sum()) if not epoch_rejection_table.empty else 0
    ptp_rejected = int(epoch_rejection_table.get("rejected_for_ptp", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    excluded_retained_bad = sorted(
        {
            channel
            for value in epoch_rejection_table.get(
                "excluded_retained_bad_channels", pd.Series(dtype=str)
            ).fillna("").astype(str)
            for channel in value.split("|")
            if channel
        }
    )
    thresholds = pd.to_numeric(epoch_rejection_table.get("threshold_uv", pd.Series(dtype=float)), errors="coerce").dropna().unique()
    threshold_uv = float(thresholds[0]) if len(thresholds) else None

    lines = [
        f"# Gate 3C Epoch QC: {subject} {recording_part}",
        "",
        "## Policy",
        "- Only manually confirmed ICA components were applied.",
        "- Only manually confirmed bad channels may be interpolated.",
        "- Explicitly dispositioned unresolved bad channels remain bad and are retained without spatial interpolation.",
        "- Epochs are image-onset only for triggers 120 and 121.",
        "- Missing EEG trials were not invented.",
        "- Gate 3C uses the task-window cropped filtered raw; Gate 3A complementary metrics use distributed full-recording windows, while PREP-style correlation QC uses alignment-derived task segments.",
        "- Structural/event invalidity is recorded separately from residual EEG PTP rejection.",
        "",
        "## Inputs",
    ]
    lines.extend(f"- `{path}`" for path in input_files)
    lines.extend(
        [
            "",
            "## ICA Application",
            f"- Applied components: {', '.join(applied_ica['component'].astype(str)) if not applied_ica.empty else 'none'}",
            "",
            "## Referencing",
        ]
    )
    if reference_summary.empty:
        lines.append("- No reference summary available.")
    else:
        row = reference_summary.iloc[0].to_dict()
        lines.append(f"- Original acquisition reference: {row.get('original_reference_channel', 'n/a')}")
        lines.append(f"- Reference evidence: {row.get('original_reference_evidence', 'n/a')}")
        lines.append(f"- CPz reconstructed: {row.get('cpz_reconstructed', 'n/a')}")
        lines.append(f"- Reconstruction stage: {row.get('cpz_reconstruction_stage', 'n/a')}")
        lines.append(f"- Reconstruction method: {row.get('cpz_reconstruction_method', 'n/a')}")
        lines.append(f"- CPz zero-valued before average reference: {row.get('cpz_zero_valued_before_average_reference', 'n/a')}")
        lines.append(
            "- EEG channel count before/after reconstruction: "
            f"{row.get('eeg_channel_count_before_reference_reconstruction', 'n/a')} / "
            f"{row.get('eeg_channel_count_after_reference_reconstruction', 'n/a')}"
        )
        lines.append(f"- Reference: {row.get('reference_type', 'n/a')}")
        lines.append(f"- Final EEG reference: {row.get('final_eeg_reference', 'n/a')}")
        lines.append(f"- Average reference applied: {row.get('average_reference_applied', 'n/a')}")
        lines.append(f"- Average reference projection: {row.get('average_reference_projection', 'n/a')}")
        lines.append(f"- Reference channels: {row.get('reference_channels', 'n/a')}")
        lines.append(f"- Excluded from reference: {row.get('excluded_from_reference', 'n/a')}")
        lines.append(f"- Montage: {row.get('montage', 'n/a')} ({row.get('montage_status', 'n/a')})")
        lines.append(f"- Unresolved coordinate channels: {row.get('unresolved_coordinate_channels', 'n/a')}")
        lines.append(f"- Operation order: {row.get('operation_order', 'n/a')}")

    lines.extend(
        [
            "",
            "## Interpolation",
            f"- Interpolated channels: {', '.join(interpolated['channel'].astype(str)) if not interpolated.empty else 'none'}",
            f"- Retained bad/uninterpolated channels: {', '.join(retained_bad['channel'].astype(str)) if not retained_bad.empty else 'none'}",
            "- Retained-bad disposition: "
            + (
                str(retained_bad.iloc[0].get("bad_channel_interpolation_disposition", "n/a"))
                if not retained_bad.empty
                else "none"
            ),
            "- Retained-bad disposition ID: "
            + (
                str(retained_bad.iloc[0].get("bad_channel_interpolation_disposition_id", "n/a"))
                if not retained_bad.empty
                else "none"
            ),
            f"- Geometry safety: {reference_summary.iloc[0].get('interpolation_geometry_safety_status', 'n/a') if not reference_summary.empty else 'n/a'}",
            "",
            "## Epoch QC",
            "- Canonical interval: -0.2 to +1.2 s, image-onset locked.",
            "- Canonical baseline: -0.2 to 0.0 s, applied once during Gate 3C construction.",
            f"- Residual EEG PTP rule: reject when maximum within-channel EEG PTP > {threshold_uv:g} µV." if threshold_uv is not None else "- Residual EEG PTP threshold unavailable.",
            f"- Residual PTP excluded retained-bad channels: {', '.join(excluded_retained_bad) if excluded_retained_bad else 'none'}",
            f"- PTP-rejected epochs: {ptp_rejected}",
            f"- Retained epochs: {retained}",
            f"- Rejected epochs: {rejected}",
            "",
            "## Condition Counts",
        ]
    )
    if condition_counts.empty:
        lines.append("- No condition counts available.")
    else:
        for _, row in condition_counts.iterrows():
            lines.append(f"- {row.get('condition')}: retained={row.get('retained_epochs')} rejected={row.get('rejected_epochs')}")

    lines.extend(["", "## Outputs"])
    for label, path in output_files.items():
        lines.append(f"- {label}: `{path}`")

    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")

    lines.extend(["", "## Failures"])
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- none")
    lines.append("")
    return "\n".join(lines)
