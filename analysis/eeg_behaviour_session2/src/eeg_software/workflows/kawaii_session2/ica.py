from __future__ import annotations

import gc
import logging
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .loading import RecordingPart
from .manual_review import ensure_manual_review_scaffold, write_ica_review_template
from .ica_identity import (
    authoritative_ica_solution_identity,
    build_canonical_ica_fit_artifact_identity,
    build_ica_fit_context_identity,
    build_ica_fit_selection_identity,
    ica_annotations_fingerprint,
    ica_decomposition_identity,
    ica_fit_data_content_sha256,
    ica_reference_state,
    resolved_ica_config_fingerprints,
    write_ica_fit_artifact_identity,
)
from .identity import identify_source_file
from .paths import subject_part_stage_dir
from .plots import (
    gate3b_ica_figure_paths,
    gate3b_ica_component_review_paths,
    plot_ica_eog_scores,
    plot_ica_component_spectra,
    plot_ica_component_topographies,
    plot_pca_explained_variance,
    save_figure_with_stage_provenance,
)
from .provenance import write_provenance_sidecar
from .resolved_snapshot import ResolvedRunSnapshotReference, bind_authoritative_config_provenance
from .resolved_config import canonical_sha256
from .reporting import generate_gate3b_ica_markdown_report, write_markdown_report
from .task_crop import (
    apply_task_crop,
    apply_task_segments,
    calculate_task_crop_window,
    calculate_task_segments,
    long_gap_settings,
    retained_image_onset_rows,
    task_crop_debug_table,
    task_crop_summary,
    task_segments_table,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ICAFitResult:
    """Gate 3B outputs for one participant recording part."""

    subject: str
    recording_part: str
    status: str
    output_files: dict[str, Path]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]


@dataclass
class ICAInstrumentationState:
    """Mutable Gate 3B instrumentation state for one recording part."""

    ica_fit_started: bool = False
    ica_fit_completed: bool = False
    ica_solution_saved: bool = False
    diagnostics_started: bool = False
    diagnostics_completed: bool = False
    ica_report_completed: bool = False


@dataclass
class ICADiagnosticSummaryCache:
    """Request-scoped cache for identity-bound ICA input summaries.

    Entries contain only successfully completed summary computations.  The
    source identity and scientific summary context form the key; descriptive
    participant/path/status fields are rebound when a cached row is emitted.
    """

    _entries: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]]

    def __init__(self) -> None:
        self._entries = {}

    def get_or_compute(
        self,
        *,
        source_identity,
        recording_part: str,
        fit_channels: Sequence[str],
        config: Mapping[str, Any],
        compute,
    ) -> dict[str, Any]:
        key = (
            source_identity.fingerprint,
            str(recording_part),
            _ica_summary_context_fingerprint(config, fit_channels),
            tuple(str(channel) for channel in fit_channels),
        )
        cached = self._entries.get(key)
        if cached is not None:
            return dict(cached)
        summary = dict(compute())
        self._entries[key] = dict(summary)
        return dict(summary)


class Gate3BSubstageError(RuntimeError):
    """Wrap an exception with the exact Gate 3B substage that failed."""

    def __init__(self, substage: str, original: Exception, elapsed_seconds: float, *, failed_operation: str = "", failure_classification: str = "UNKNOWN"):
        super().__init__(str(original))
        self.substage = substage
        self.original = original
        self.elapsed_seconds = elapsed_seconds
        self.failed_operation = failed_operation or substage
        self.failure_classification = failure_classification


def ica_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return ICA settings with conservative Gate 3B defaults."""

    prep = dict(config.get("preprocessing", {}) or {})
    ica_cfg = dict(prep.get("ica", {}) or {})
    return {
        "enabled": bool(ica_cfg.get("enabled", True)),
        "method": ica_cfg.get("method", "infomax"),
        "fit_params": dict(ica_cfg.get("fit_params", {}) or {}),
        "n_components": ica_cfg.get("n_components", "rank_aware"),
        "decim": int(ica_cfg.get("decim", 5) or 1),
        "max_iter": ica_cfg.get("max_iter", "auto"),
        "random_state": ica_cfg.get("random_state", 20260627),
        "highpass_hz": prep.get("ica_highpass_hz", 1.0),
        "lowpass_hz": prep.get("ica_lowpass_hz", prep.get("analysis_lowpass_hz", 30.0)),
        "fit_only_eeg_channels": bool(ica_cfg.get("fit_only_eeg_channels", True)),
        "ocular_detection_channels": ocular_detection_channels(config),
        "exclude_channels": channels_excluded_from_ica(config),
        "apply_components_gate3b": bool(ica_cfg.get("apply_components_gate3b", False)),
        "diagnostics": dict(ica_cfg.get("diagnostics", {}) or {}),
        "fit_subset": dict(ica_cfg.get("fit_subset", {}) or {}),
    }


def ica_diagnostic_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return low-memory Gate 3B diagnostic settings."""

    diagnostics = dict(ica_settings(config).get("diagnostics", {}) or {})
    window = list(diagnostics.get("time_window_s", [0, 60]) or [0, 60])
    if len(window) != 2:
        window = [0, 60]
    skip_properties = bool(diagnostics.get("skip_component_properties_by_default", True))
    return {
        "fail_pipeline_on_error": bool(diagnostics.get("fail_pipeline_on_error", False)),
        "generate_eog_scores": bool(diagnostics.get("generate_eog_scores", True)),
        "generate_topomaps": bool(diagnostics.get("generate_topomaps", True)),
        "generate_component_spectra": bool(diagnostics.get("generate_component_spectra", True)),
        "generate_component_properties": bool(diagnostics.get("generate_component_properties", False)),
        "generate_reports": bool(diagnostics.get("generate_reports", True)),
        "generate_figures": bool(diagnostics.get("generate_figures", True)),
        "time_window_s": (float(window[0]), float(window[1])),
        "component_review_page_size": int(diagnostics.get("component_review_page_size", 10) or 10),
        "skip_component_properties_by_default": skip_properties,
    }


def ica_fit_subset_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return representative ICA subset fitting settings."""

    subset = dict(ica_settings(config).get("fit_subset", {}) or {})
    return {
        "enabled": bool(subset.get("enabled", True)),
        "use_when_task_duration_exceeds_s": float(subset.get("use_when_task_duration_exceeds_s", 1800.0)),
        "max_duration_s": float(subset.get("max_duration_s", 1200.0)),
        "strategy": str(subset.get("strategy", "balanced_across_blocks")),
        "window_s": float(subset.get("window_s", 30.0)),
        "min_windows_per_block": int(subset.get("min_windows_per_block", 1) or 1),
        "max_windows_per_block": int(subset.get("max_windows_per_block", 3) or 3),
        "include_all_blocks": bool(subset.get("include_all_blocks", True)),
    }


def channels_excluded_from_ica(config: Mapping[str, Any]) -> list[str]:
    """Return channels that must not be included in ICA fitting."""

    channels = dict(config.get("channels", {}) or {}).get("eeg", {}) or {}
    ica_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("ica", {}) or {})
    excluded = []
    for key in ("unused", "exclude_from_ica"):
        excluded.extend(channels.get(key, []) or [])
    excluded.extend(ica_cfg.get("exclude_channels", []) or [])
    return sorted({str(channel) for channel in excluded})


def ocular_detection_channels(config: Mapping[str, Any]) -> list[str]:
    """Return EOG channels used only for ocular-component detection."""

    ica_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("ica", {}) or {})
    configured = ica_cfg.get("ocular_detection_channels")
    if configured:
        return [str(channel) for channel in configured]
    channels = dict(config.get("channels", {}) or {}).get("eeg", {}) or {}
    return [str(channel) for channel in channels.get("eog", []) or []]


def select_ica_fit_channels(
    channel_names: Sequence[str],
    channel_types: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    bad_channels: Sequence[str] | None = None,
) -> list[str]:
    """Select channels eligible for ICA fitting without using EOG/unused channels."""

    excluded = set(channels_excluded_from_ica(config))
    excluded.update(str(channel) for channel in (bad_channels or []))
    return [
        str(channel)
        for channel in channel_names
        if str(channel) not in excluded and str(channel) not in ocular_detection_channels(config) and channel_types.get(str(channel), "eeg") == "eeg"
    ]


def gate3b_output_paths(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> dict[str, Path]:
    """Return standard Gate 3B output paths for one recording part."""

    ica_dir = subject_part_stage_dir(config, subject, recording_part, "ica", root=root, create=True)
    qc_dir = ica_dir
    fig_dir = ica_dir / "figures"
    for directory in (ica_dir, qc_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "ica_dir": ica_dir,
        "qc_dir": qc_dir,
        "fig_dir": fig_dir,
        "report_dir": ica_dir,
        "ica_ready_raw_fif": ica_dir / "ica_ready_raw.fif",
        "ica_solution": ica_dir / f"{subject}_{recording_part}-ica.fif",
        "ica_fit_identity": ica_dir / "ica_fit_identity.json",
        "ica_review_template": ica_dir / "ica_review_template.yaml",
        "ica_summary": qc_dir / "component_summary.csv",
        "ica_parameters": qc_dir / "parameters.csv",
        "eog_suggestions": qc_dir / "eog_suggestions.csv",
        "pca_components": qc_dir / "pca_components.csv",
        "ica_input_summary": qc_dir / "ica_input_summary.csv",
        "participant_comparison": qc_dir / "participant_comparison.csv",
        "ica_fit_subset_windows": qc_dir / "ica_fit_subset_windows.csv",
        "ica_fit_subset_summary": qc_dir / "ica_fit_subset_summary.csv",
        "task_crop_summary": qc_dir / "task_crop_summary.csv",
        "task_crop_debug": qc_dir / "task_crop_debug.csv",
        "task_segments": qc_dir / "task_segments.csv",
        "ica_timing": qc_dir / "ica_timing.csv",
        "processing_log": qc_dir / "processing_log.csv",
        "report": ica_dir / "report.md",
    }


def stage3a_filtered_raw_path(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> Path:
    """Return the expected Stage 3A filtered raw FIF path."""

    return subject_part_stage_dir(config, subject, recording_part, "preprocessing", root=root, create=True) / "filtered_raw.fif"


def build_ica_parameter_table(
    config: Mapping[str, Any],
    *,
    fit_channels: Sequence[str],
    input_file: str | Path,
    fit_provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Represent Gate 3B ICA settings as a table for review/provenance."""

    settings = ica_settings(config)
    return pd.DataFrame(
        [
            {
                "input_file": str(input_file),
                "enabled": settings["enabled"],
                "method": settings["method"],
                "fit_params": settings["fit_params"],
                "n_components_strategy": settings["n_components"],
                "decim": settings["decim"],
                "max_iter": settings["max_iter"],
                "random_state": settings["random_state"],
                "highpass_hz": settings["highpass_hz"],
                "lowpass_hz": settings["lowpass_hz"],
                "fit_only_eeg_channels": settings["fit_only_eeg_channels"],
                "fit_channels": list(fit_channels),
                "n_fit_channels": len(fit_channels),
                "exclude_channels": settings["exclude_channels"],
                "ocular_detection_channels": settings["ocular_detection_channels"],
                "apply_components_gate3b": settings["apply_components_gate3b"],
                "fit_subset": settings["fit_subset"],
                **dict(fit_provenance or {}),
            }
        ]
    )


def estimate_ica_fit_rank(ica_raw, fit_channels: Sequence[str]) -> int:
    """Estimate effective EEG rank from the actual data supplied to ICA."""

    if not fit_channels:
        raise ValueError("ICA rank estimation requires at least one EEG fit channel.")
    import mne

    fit_raw = ica_raw.copy().pick(list(fit_channels))
    try:
        rank = mne.compute_rank(
            fit_raw,
            rank=None,
            proj=False,
            on_rank_mismatch="raise",
            verbose=False,
        )
    except Exception as error:
        raise ValueError(f"ICA effective-rank estimation failed for the actual fit data: {error}") from error
    estimated_rank = rank.get("eeg")
    if estimated_rank is None:
        raise ValueError(f"ICA effective-rank estimation returned no EEG rank: {rank!r}")
    estimated_rank = int(estimated_rank)
    if estimated_rank <= 0 or estimated_rank > len(fit_channels):
        raise ValueError(
            "ICA effective-rank estimation returned an unsafe EEG rank "
            f"{estimated_rank} for {len(fit_channels)} fit channels."
        )
    return estimated_rank


def build_ica_fit_provenance(
    ica_raw,
    config: Mapping[str, Any],
    *,
    candidate_channels: Sequence[str],
    confirmed_bad_channels: Sequence[str],
    fit_channels: Sequence[str],
    estimated_rank: int,
) -> dict[str, Any]:
    """Build explicit provenance for the actual ICA fitting data/configuration."""

    settings = ica_settings(config)
    decim = int(settings["decim"])
    n_samples = int(getattr(ica_raw, "n_times", 0) or 0)
    sfreq = float(ica_raw.info["sfreq"])
    effective_highpass = float(ica_raw.info.get("highpass", settings["highpass_hz"]))
    effective_lowpass = float(ica_raw.info.get("lowpass", settings["lowpass_hz"]))
    if estimated_rank < len(fit_channels):
        rank_reduction_reason = (
            f"data-driven effective EEG rank ({estimated_rank}) is below the "
            f"actual ICA fit-channel count ({len(fit_channels)})"
        )
    else:
        rank_reduction_reason = "none; effective EEG rank equals the actual ICA fit-channel count"
    return {
        "candidate_eeg_channels": len(candidate_channels),
        "candidate_eeg_channel_names": list(candidate_channels),
        "confirmed_bad_eeg_channels_excluded": len(confirmed_bad_channels),
        "confirmed_bad_eeg_channel_names": list(confirmed_bad_channels),
        "ica_fit_channels": len(fit_channels),
        "ica_fit_channel_names": list(fit_channels),
        "estimated_effective_rank": int(estimated_rank),
        "ica_n_components_requested": int(estimated_rank),
        "ica_n_components_strategy": settings["n_components"],
        "ica_method": settings["method"],
        "ica_fit_sampling_frequency_hz": sfreq,
        "ica_effective_sampling_frequency_hz": sfreq / decim,
        "ica_decimation": decim,
        "ica_effective_highpass_hz": effective_highpass,
        "ica_effective_lowpass_hz": effective_lowpass,
        "ica_random_state": settings["random_state"],
        "ica_fit_n_samples_available": n_samples,
        "ica_fit_n_samples_decimated": ((n_samples - 1) // decim + 1) if n_samples else 0,
        "ica_fit_duration_s": float(ica_raw.times[-1]) if n_samples else 0.0,
        "ica_rank_reduction_reason": rank_reduction_reason,
    }


def build_ica_summary_table(
    subject: str,
    recording_part: str,
    config: Mapping[str, Any],
    *,
    fit_channels: Sequence[str],
    fitted_components: int | None,
    n_iter: int | None,
    convergence_status: str,
    confirmed_ica_components: Sequence[int] | None = None,
    fit_provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Create a one-row ICA summary table without implying component removal."""

    settings = ica_settings(config)
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "recording_part": recording_part,
                "method": settings["method"],
                "n_components": fitted_components,
                "n_components_requested": (fit_provenance or {}).get("ica_n_components_requested"),
                "decim": settings["decim"],
                "n_fit_channels": len(fit_channels),
                "fit_channels": list(fit_channels),
                "excluded_from_ica": settings["exclude_channels"],
                "ocular_detection_channels": settings["ocular_detection_channels"],
                "n_iter": n_iter,
                "convergence_status": convergence_status,
                "confirmed_ica_components_from_review_not_applied_gate3b": list(confirmed_ica_components or []),
                "ica_application": "not_applied_gate3b",
                **dict(fit_provenance or {}),
            }
        ]
    )


def retained_image_onset_task_duration_s(
    alignment: pd.DataFrame,
    recording_part: str,
    *,
    sfreq: float,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Return retained image-onset span in seconds for one recording part."""

    retained = retained_image_onset_rows(alignment, recording_part, config=config)
    if retained.empty:
        raise ValueError(f"ICA subset detection failed for {recording_part}: no retained image-onset rows.")
    samples = pd.to_numeric(retained["image_onset_sample"], errors="coerce").dropna()
    if samples.empty:
        raise ValueError(f"ICA subset detection failed for {recording_part}: no valid image_onset_sample values.")
    return float(samples.max() - samples.min()) / float(sfreq)


def should_use_ica_fit_subset(alignment: pd.DataFrame, recording_part: str, config: Mapping[str, Any], *, sfreq: float) -> bool:
    """Return whether Gate 3B should fit ICA on representative task windows."""

    settings = ica_fit_subset_settings(config)
    if not bool(settings["enabled"]):
        return False
    return retained_image_onset_task_duration_s(
        alignment, recording_part, sfreq=sfreq, config=config
    ) > float(settings["use_when_task_duration_exceeds_s"])


def build_ica_fit_subset_windows(
    subject: str,
    recording_part: str,
    alignment: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    sfreq: float,
    raw_first_sample: int,
    raw_last_sample: int,
) -> pd.DataFrame:
    """Select balanced retained-trial windows for representative ICA fitting."""

    settings = ica_fit_subset_settings(config)
    if settings["strategy"] != "balanced_across_blocks":
        raise ValueError(f"ICA subset creation failed for {recording_part}: unsupported strategy {settings['strategy']!r}.")
    retained = retained_image_onset_rows(alignment, recording_part, config=config)
    if retained.empty:
        raise ValueError(f"ICA subset creation failed for {recording_part}: no retained image-onset rows.")
    retained = retained.copy()
    retained["sample"] = pd.to_numeric(retained["image_onset_sample"], errors="coerce")
    retained["trial_numeric"] = pd.to_numeric(retained.get("trial", pd.Series(dtype=float)), errors="coerce")
    retained = retained[retained["sample"].notna() & retained["trial_numeric"].notna()].copy()
    if retained.empty:
        raise ValueError(f"ICA subset creation failed for {recording_part}: retained rows lack valid samples or trials.")
    retained["sample"] = retained["sample"].astype(int)
    retained["trial_numeric"] = retained["trial_numeric"].astype(int)
    retained["block_label"] = retained["block"].astype(str) if "block" in retained else "all"
    retained = retained.sort_values(["block_label", "sample", "trial_numeric"], kind="mergesort").reset_index(drop=True)

    window_samples = max(1, int(round(float(settings["window_s"]) * float(sfreq))))
    half_window = max(0, window_samples // 2)
    max_windows_total = max(1, int(float(settings["max_duration_s"]) // float(settings["window_s"])))
    max_per_block = max(1, int(settings["max_windows_per_block"]))
    min_per_block = max(1, min(int(settings["min_windows_per_block"]), max_per_block))
    rows: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    def candidate_indices(length: int, count: int) -> list[int]:
        if length <= 0 or count <= 0:
            return []
        if count == 1:
            return [length // 2]
        return sorted({int(round(position * (length - 1) / float(count - 1))) for position in range(count)})

    def try_add(block: str, block_rows: pd.DataFrame, idx: int) -> bool:
        if len(rows) >= max_windows_total:
            return False
        center = int(block_rows.iloc[idx]["sample"])
        start = max(int(raw_first_sample), center - half_window)
        end = min(int(raw_last_sample), start + window_samples - 1)
        start = max(int(raw_first_sample), end - window_samples + 1)
        if end <= start:
            return False
        for used_start, used_end in occupied:
            if not (end < used_start or start > used_end):
                return False
        window_trials = retained[(retained["sample"] >= start) & (retained["sample"] <= end)].copy()
        if window_trials.empty:
            return False
        window_id = len(rows) + 1
        rows.append(
            {
                "subject": subject,
                "recording_part": recording_part,
                "window_id": window_id,
                "block": block,
                "start_trial": int(window_trials["trial_numeric"].min()),
                "end_trial": int(window_trials["trial_numeric"].max()),
                "start_sample": int(start),
                "end_sample": int(end),
                "start_time_s": float(start - int(raw_first_sample)) / float(sfreq),
                "end_time_s": float(end - int(raw_first_sample)) / float(sfreq),
                "duration_s": float(end - start + 1) / float(sfreq),
                "n_retained_trials": int(len(window_trials)),
            }
        )
        occupied.append((start, end))
        return True

    grouped = [(str(block), block_rows.reset_index(drop=True)) for block, block_rows in retained.groupby("block_label", sort=True)]
    if settings["include_all_blocks"] and len(grouped) > max_windows_total:
        raise ValueError(
            f"ICA subset creation failed for {recording_part}: include_all_blocks=true but {len(grouped)} blocks exceed "
            f"the maximum {max_windows_total} windows allowed by max_duration_s/window_s."
        )
    for block, block_rows in grouped:
        added = 0
        for idx in candidate_indices(len(block_rows), min_per_block):
            added += int(try_add(block, block_rows, idx))
        if settings["include_all_blocks"] and added == 0:
            raise ValueError(f"ICA subset creation failed for {recording_part}: could not select a retained-trial window for block {block}.")
    for block, block_rows in grouped:
        already = sum(1 for row in rows if row["block"] == block)
        target = min(max_per_block, len(block_rows))
        for idx in candidate_indices(len(block_rows), target):
            if already >= target or len(rows) >= max_windows_total:
                break
            if try_add(block, block_rows, idx):
                already += 1

    if not rows:
        raise ValueError(f"ICA subset creation failed for {recording_part}: no non-overlapping retained-trial windows selected.")
    return pd.DataFrame(rows).sort_values(["start_sample", "end_sample"], kind="mergesort").reset_index(drop=True)


def build_ica_fit_subset_summary(
    subject: str,
    recording_part: str,
    windows: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    full_task_duration_s: float,
    n_fit_channels: int,
    sfreq: float,
    reason: str,
) -> pd.DataFrame:
    """Summarise representative ICA subset fitting for reports and provenance."""

    settings = ica_fit_subset_settings(config)
    subset_duration_s = float(pd.to_numeric(windows["duration_s"], errors="coerce").fillna(0.0).sum()) if not windows.empty else 0.0
    n_trials = int(pd.to_numeric(windows["n_retained_trials"], errors="coerce").fillna(0).sum()) if not windows.empty else 0
    estimated_bytes = int(max(0.0, subset_duration_s) * sfreq * float(n_fit_channels) * 8.0) if sfreq else 0
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "recording_part": recording_part,
                "ica_fit_mode": "representative_subset",
                "subset_strategy": settings["strategy"],
                "subset_duration_s": subset_duration_s,
                "full_task_duration_s": float(full_task_duration_s),
                "subset_fraction": subset_duration_s / float(full_task_duration_s) if full_task_duration_s else 0.0,
                "n_blocks_covered": int(windows["block"].nunique()) if not windows.empty else 0,
                "n_windows": int(len(windows)),
                "n_trials_represented": n_trials,
                "n_eeg_fit_channels": int(n_fit_channels),
                "estimated_dense_ram_bytes": estimated_bytes,
                "reason": reason,
            }
        ]
    )


def pca_explained_variance_table(ica) -> pd.DataFrame:
    """Return PCA explained variance ratios from a fitted MNE ICA object."""

    values = getattr(ica, "pca_explained_variance_", None)
    if values is None:
        return pd.DataFrame(columns=["pca_component", "explained_variance", "explained_variance_ratio"])
    total = float(sum(values)) if len(values) else 0.0
    rows = []
    for idx, value in enumerate(values):
        rows.append(
            {
                "pca_component": idx,
                "explained_variance": float(value),
                "explained_variance_ratio": float(value) / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_ica_failure_record(
    subject: str,
    recording_part: str,
    *,
    failed_substage: str,
    error: Exception,
    state: ICAInstrumentationState | None = None,
    ica_solution_fitted: bool | None = None,
    ica_solution_saved: bool | None = None,
    elapsed_seconds: float | None = None,
    memory_configuration: Mapping[str, Any] | None = None,
    version_info: Mapping[str, str] | None = None,
    failed_operation: str = "",
    failure_classification: str = "UNKNOWN",
    filtering_started: bool = False,
    input_validation: Mapping[str, Any] | None = None,
    traceback_text: str = "",
) -> pd.DataFrame:
    """Build a detailed Gate 3B failure row."""

    if state is None:
        state = ICAInstrumentationState(
            ica_fit_completed=bool(ica_solution_fitted),
            ica_solution_saved=bool(ica_solution_saved),
        )
    versions = dict(version_info or {})
    validation = dict(input_validation or {})
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "recording_part": recording_part,
                "status": "failed",
                "failed_substage": failed_substage,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback_text,
                "failed_operation": failed_operation or failed_substage,
                "failure_classification": failure_classification,
                "filtering_started": bool(filtering_started),
                "nan_values_present": bool(validation.get("nan_values_present", False)),
                "inf_values_present": bool(validation.get("inf_values_present", False)),
                "channels_with_nan_or_inf": validation.get("channels_with_nan_or_inf", []),
                "requested_array_shape": _array_shape_from_message(str(error)),
                "requested_array_dtype": _array_dtype_from_message(str(error)),
                "current_ram_available_bytes": _available_ram_bytes(),
                "ica_fit_started": bool(state.ica_fit_started),
                "ica_fit_completed": bool(state.ica_fit_completed),
                "ica_solution_fitted_before_failure": bool(state.ica_fit_completed),
                "ica_solution_saved": bool(state.ica_solution_saved),
                "ica_solution_saved_before_failure": bool(state.ica_solution_saved),
                "diagnostics_started": bool(state.diagnostics_started),
                "diagnostics_completed": bool(state.diagnostics_completed),
                "ica_report_completed": bool(state.ica_report_completed),
                "elapsed_seconds": elapsed_seconds,
                "memory_configuration": dict(memory_configuration or {}),
                "mne_version": versions.get("mne_version", ""),
                "numpy_version": versions.get("numpy_version", ""),
                "python_version": versions.get("python_version", ""),
                "failure": str(error),
            }
        ]
    )


def gate3b_ica_solution_usable(status: str, output_files: Mapping[str, Path]) -> bool:
    """Return whether Gate 3C can consume an identity-bound Gate 3B result."""

    return (
        status == "ok"
        and "ica_solution" in output_files
        and Path(output_files["ica_solution"]).exists()
        and "ica_fit_identity" in output_files
        and Path(output_files["ica_fit_identity"]).exists()
    )


def build_ica_timing_row(substage: str, status: str, elapsed_seconds: float, *, message: str = "", exception_type: str = "", exception_message: str = "") -> dict[str, Any]:
    """Build one Gate 3B timing row."""

    return {
        "substage": substage,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "message": message,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }


def build_ica_processing_log_row(
    substage: str,
    event: str,
    message: str,
    *,
    status: str = "",
    elapsed_seconds: float | None = None,
    state: ICAInstrumentationState | None = None,
) -> dict[str, Any]:
    """Build one Gate 3B processing-log row."""

    return {
        "substage": substage,
        "event": event,
        "message": message,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "ica_fit_started": bool(state.ica_fit_started) if state else False,
        "ica_fit_completed": bool(state.ica_fit_completed) if state else False,
        "ica_solution_saved": bool(state.ica_solution_saved) if state else False,
        "diagnostics_started": bool(state.diagnostics_started) if state else False,
        "diagnostics_completed": bool(state.diagnostics_completed) if state else False,
        "ica_report_completed": bool(state.ica_report_completed) if state else False,
    }


def validate_ica_filter_input(summary: Mapping[str, Any]) -> None:
    """Raise a classified error when ICA filter input contains unsafe values."""

    if bool(summary.get("nan_values_present", False)):
        raise Gate3BSubstageError("ica_filter_input_validation", ValueError("NaN values present before ICA filtering."), 0.0, failed_operation="filter_input_nan_check", failure_classification="NAN_VALUES_PRESENT")
    if bool(summary.get("inf_values_present", False)):
        raise Gate3BSubstageError("ica_filter_input_validation", ValueError("Inf values present before ICA filtering."), 0.0, failed_operation="filter_input_inf_check", failure_classification="INF_VALUES_PRESENT")


def classify_ica_failure(error: Exception, substage: str) -> str:
    """Classify Gate 3B failures beyond the Python exception type."""

    if "ica_subset_creation" in substage:
        return "ICA_SUBSET_CREATION_FAILED"
    if "ica_subset_filter" in substage:
        return "ICA_SUBSET_FILTER_FAILED"
    if "ica_subset_fit" in substage:
        return "ICA_SUBSET_FIT_FAILED"
    if isinstance(error, MemoryError):
        return "MEMORY_LIMIT"
    if isinstance(error, ValueError):
        message = str(error).lower()
        if "task_crop_too_long" in message:
            return "TASK_CROP_TOO_LONG"
        if "nan" in message:
            return "NAN_VALUES_PRESENT"
        if "inf" in message:
            return "INF_VALUES_PRESENT"
        return "FILTER_INPUT_INVALID" if "filter" in substage else "MNE_INTERNAL_ERROR"
    if "copy" in substage:
        return "RAW_COPY_FAILED"
    if "filter" in substage:
        return "FILTER_FAILED"
    return "UNKNOWN"


def run_gate3b_for_recording_part(
    subject: str,
    recording: RecordingPart,
    config: Mapping[str, Any],
    *,
    alignment: pd.DataFrame | None = None,
    filtered_raw_path: str | Path | None = None,
    root: Path | None = None,
    config_path: str | Path | None = None,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
    ica_summary_cache: ICADiagnosticSummaryCache | None = None,
) -> ICAFitResult:
    """Fit ICA and generate diagnostics without applying component exclusions."""

    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    outputs = gate3b_output_paths(config, subject, recording.recording_part, root=root)
    review_paths = ensure_manual_review_scaffold({subject: [recording]}, config, root=root)
    review_file_paths = {key: value for key, value in review_paths.items() if key != "manual_review_dir"}
    input_raw = Path(filtered_raw_path) if filtered_raw_path else stage3a_filtered_raw_path(config, subject, recording.recording_part, root=root)
    input_files = [input_raw]
    warnings: list[str] = []
    failures: list[str] = []
    output_files: dict[str, Path] = {f"manual_review_{key}": value for key, value in review_file_paths.items()}
    failed_substage = "ica_input_loading"
    state = ICAInstrumentationState()
    diagnostics_fail_pipeline = False
    processing_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    input_validation_summary: dict[str, Any] = {}
    filtering_started = False

    try:
        if not input_raw.exists():
            raise FileNotFoundError(f"Stage 3A filtered raw FIF not found: {input_raw}")

        settings = ica_settings(config)
        if not settings["enabled"]:
            raise RuntimeError("preprocessing.ica.enabled is false; Gate 3B ICA fitting is disabled.")
        if settings["apply_components_gate3b"]:
            raise RuntimeError("Gate 3B must not apply ICA components; set apply_components_gate3b to false.")
        if alignment is None:
            raise ValueError("Gate 3B task-window crop requires the Stage 2 alignment table.")

        import mne
        from mne.preprocessing import ICA

        failed_substage = "ica_input_loading"
        raw = _run_required_ica_substage(
            substage="ica_input_loading",
            start_message="Loading filtered raw...",
            complete_message="Filtered raw loaded.",
            operation=lambda: mne.io.read_raw_fif(input_raw, preload=False, verbose=False),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        original_duration_s = float(raw.times[-1]) if len(raw.times) else 0.0
        sfreq = float(raw.info["sfreq"])
        raw_first_sample = int(raw.first_samp)
        raw_last_sample = raw_first_sample + int(raw.n_times) - 1
        subset_settings = ica_fit_subset_settings(config)
        full_task_duration_s = _run_required_ica_substage(
            substage="ica_subset_duration_detection",
            start_message="Checking ICA subset-fit threshold...",
            complete_message="ICA subset-fit threshold checked.",
            operation=lambda: retained_image_onset_task_duration_s(alignment, recording.recording_part, sfreq=sfreq),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        use_subset_fit = bool(subset_settings["enabled"]) and full_task_duration_s > float(subset_settings["use_when_task_duration_exceeds_s"])
        crop_window = None
        task_segments = []
        subset_windows = pd.DataFrame()
        if use_subset_fit:
            warnings.append(
                "Gate 3B ICA uses representative-subset fitting because retained task duration "
                f"{full_task_duration_s:.3f}s exceeds preprocessing.ica.fit_subset.use_when_task_duration_exceeds_s="
                f"{float(subset_settings['use_when_task_duration_exceeds_s']):.3f}s."
            )
        elif bool(long_gap_settings(config)["enabled"]):
            task_segments = _run_required_ica_substage(
                substage="ica_task_segment_detection",
                start_message="Detecting task segments...",
                complete_message="Task segments detected.",
                operation=lambda: calculate_task_segments(alignment, recording.recording_part, sfreq=float(raw.info["sfreq"]), n_times=int(raw.n_times), config=config, first_samp=int(raw.first_samp)),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
            segment_table = task_segments_table(task_segments)
            segment_table.to_csv(outputs["task_segments"], index=False)
            output_files["task_segments"] = outputs["task_segments"]
            _write_ica_sidecar(outputs["task_segments"], subject, recording, config_path, root, input_files, settings, warnings, review_file_paths)
            raw = _run_required_ica_substage(
                substage="ica_task_segment_application",
                start_message="Cropping raw to task segments for ICA...",
                complete_message="Task-segment raw prepared for ICA.",
                operation=lambda: apply_task_segments(raw, task_segments),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
        else:
            crop_window = _run_required_ica_substage(
                substage="ica_task_crop_detection",
                start_message="Detecting task crop window...",
                complete_message="Task crop window detected.",
                operation=lambda: calculate_task_crop_window(alignment, recording.recording_part, sfreq=float(raw.info["sfreq"]), n_times=int(raw.n_times), config=config, stage="ica", first_samp=int(raw.first_samp)),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
            crop_summary = task_crop_summary(crop_window, subject=subject, recording_part=recording.recording_part, original_duration_s=original_duration_s)
            crop_summary.to_csv(outputs["task_crop_summary"], index=False)
            output_files["task_crop_summary"] = outputs["task_crop_summary"]
            _write_ica_sidecar(outputs["task_crop_summary"], subject, recording, config_path, root, input_files, settings, warnings, review_file_paths)
            crop_debug = task_crop_debug_table(
                alignment, recording.recording_part, crop_window, config=config
            )
            crop_debug.to_csv(outputs["task_crop_debug"], index=False)
            output_files["task_crop_debug"] = outputs["task_crop_debug"]
            _write_ica_sidecar(outputs["task_crop_debug"], subject, recording, config_path, root, input_files, settings, warnings, review_file_paths)
            raw = _run_required_ica_substage(
                substage="ica_task_crop_application",
                start_message="Cropping raw to task window for ICA...",
                complete_message="Task-window raw prepared for ICA.",
                operation=lambda: apply_task_crop(raw, crop_window),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
        if not use_subset_fit:
            raw = _run_required_ica_substage(
                substage="ica_task_crop_loading",
                start_message="Loading task-window raw for ICA...",
                complete_message="Task-window raw loaded for ICA.",
                operation=lambda: raw.load_data(verbose=False),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
        fit_channels = _run_required_ica_substage(
            substage="ica_pick_selection",
            start_message="Selecting EEG picks...",
            complete_message="EEG picks complete.",
            operation=lambda: _mne_ica_fit_channels(raw, config),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        if not fit_channels:
            raise Gate3BSubstageError("ica_pick_selection", RuntimeError("No EEG channels available for ICA fitting after exclusions."), 0.0)

        input_summaries = _run_required_ica_substage(
            substage="ica_input_summary",
            start_message="Generating ICA input summaries...",
            complete_message="ICA input summaries complete.",
            operation=lambda: _build_ica_input_summaries(
                subject,
                recording.recording_part,
                raw,
                input_raw,
                fit_channels,
                config,
                root=root,
                summary_cache=ica_summary_cache,
            ),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        input_summary, participant_comparison = input_summaries
        input_summary.to_csv(outputs["ica_input_summary"], index=False)
        participant_comparison.to_csv(outputs["participant_comparison"], index=False)
        output_files["ica_input_summary"] = outputs["ica_input_summary"]
        output_files["participant_comparison"] = outputs["participant_comparison"]
        input_validation_summary = input_summary.iloc[0].to_dict() if not input_summary.empty else {}

        _run_required_ica_substage(
            substage="ica_filter_input_validation",
            start_message="Validating ICA filter input...",
            complete_message="ICA filter input valid.",
            operation=lambda: validate_ica_filter_input(input_validation_summary),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        if use_subset_fit:
            subset_payload = _run_required_ica_substage(
                substage="ica_subset_creation",
                start_message="Creating representative ICA fitting subset...",
                complete_message="Representative ICA fitting subset created.",
                operation=lambda: (
                    lambda windows: (
                        windows,
                        build_ica_fit_subset_summary(
                            subject,
                            recording.recording_part,
                            windows,
                            config,
                            full_task_duration_s=full_task_duration_s,
                            n_fit_channels=len(fit_channels),
                            sfreq=sfreq,
                            reason="long_task_recording_exceeded_ica_full_fit_duration",
                        ),
                        _build_ica_subset_raw(raw, windows, fit_channels),
                    )
                )(
                    build_ica_fit_subset_windows(
                        subject,
                        recording.recording_part,
                        alignment,
                        config,
                        sfreq=sfreq,
                        raw_first_sample=raw_first_sample,
                        raw_last_sample=raw_last_sample,
                    )
                ),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
            subset_windows, subset_summary, ica_raw = subset_payload
            subset_windows.to_csv(outputs["ica_fit_subset_windows"], index=False)
            subset_summary.to_csv(outputs["ica_fit_subset_summary"], index=False)
            output_files["ica_fit_subset_windows"] = outputs["ica_fit_subset_windows"]
            output_files["ica_fit_subset_summary"] = outputs["ica_fit_subset_summary"]
            _write_ica_sidecar(outputs["ica_fit_subset_windows"], subject, recording, config_path, root, input_files, settings, warnings, review_file_paths)
            _write_ica_sidecar(outputs["ica_fit_subset_summary"], subject, recording, config_path, root, input_files, settings, warnings, review_file_paths)
        else:
            ica_raw = _run_required_ica_substage(
                substage="ica_copy_preparation",
                start_message="Preparing ICA copy...",
                complete_message="ICA copy prepared.",
                operation=lambda: raw.copy(),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
        filtering_started = True
        _run_required_ica_substage(
            substage="ica_subset_filtering" if use_subset_fit else "ica_filtering",
            start_message="Applying ICA filter...",
            complete_message="ICA filter complete.",
            operation=lambda: _filter_ica_raw(ica_raw, fit_channels, settings),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        selection_mode, selection_window_rows = _ica_fit_selection_rows(
            use_subset_fit=use_subset_fit,
            subset_windows=subset_windows,
            task_segments=task_segments,
            crop_window=crop_window,
        )
        fit_selection = build_ica_fit_selection_identity(
            mode=selection_mode,
            source_first_sample=raw_first_sample,
            source_last_sample=raw_last_sample,
            windows=selection_window_rows,
            fit_first_sample=int(ica_raw.first_samp),
            fit_n_times=int(ica_raw.n_times),
            annotations_fingerprint=ica_annotations_fingerprint(ica_raw),
            reject_by_annotation=True,
        )
        fit_data_sha256 = ica_fit_data_content_sha256(ica_raw, fit_channels)
        input_eeg_identity = identify_source_file(
            input_raw,
            "ica_filtered_raw_input",
            logical_path="filtered_raw.fif",
        )
        resolved_config_fingerprint, scientific_fingerprint = resolved_ica_config_fingerprints(
            config, resolved_snapshot
        )
        fit_reference_state = ica_reference_state(ica_raw)
        candidate_channels = _mne_ica_candidate_channels(raw, config, bad_channels=[])
        confirmed_bad_channels = [channel for channel in raw.info.get("bads", []) if channel in candidate_channels]
        estimated_rank = _run_required_ica_substage(
            substage="ica_rank_estimation",
            start_message="Estimating effective EEG rank from ICA fit data...",
            complete_message="Effective EEG rank estimated.",
            operation=lambda: estimate_ica_fit_rank(ica_raw, fit_channels),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        fit_provenance = build_ica_fit_provenance(
            ica_raw,
            config,
            candidate_channels=candidate_channels,
            confirmed_bad_channels=confirmed_bad_channels,
            fit_channels=fit_channels,
            estimated_rank=estimated_rank,
        )
        parameter_table = build_ica_parameter_table(
            config,
            fit_channels=fit_channels,
            input_file=input_raw,
            fit_provenance=fit_provenance,
        )
        parameter_table.to_csv(outputs["ica_parameters"], index=False)
        output_files["ica_parameters"] = outputs["ica_parameters"]
        _write_ica_sidecar(
            outputs["ica_parameters"], subject, recording, config_path, root, input_files,
            settings, warnings, review_file_paths, fit_provenance=fit_provenance,
        )
        save_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("save", {}) or {})
        if save_cfg.get("ica_ready_raw_fif", True):
            _run_required_ica_substage(
                substage="ica_ready_raw_saving",
                start_message="Saving ICA-ready raw...",
                complete_message="ICA-ready raw saved.",
                operation=lambda: ica_raw.save(outputs["ica_ready_raw_fif"], overwrite=True, verbose=False),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
            output_files["ica_ready_raw_fif"] = outputs["ica_ready_raw_fif"]
            _write_ica_sidecar(outputs["ica_ready_raw_fif"], subject, recording, config_path, root, input_files, settings, warnings, review_file_paths)

        failed_substage = "ica_fitting"
        ica = _run_required_ica_substage(
            substage="ica_initialisation",
            start_message="Initialising ICA object...",
            complete_message="ICA object created.",
            operation=lambda: ICA(
                n_components=estimated_rank,
                method=settings["method"],
                fit_params=settings["fit_params"],
                max_iter=settings["max_iter"],
                random_state=settings["random_state"],
            ),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        picks = _run_required_ica_substage(
            substage="ica_mne_pick_indices",
            start_message="Selecting EEG pick indices...",
            complete_message="EEG pick indices complete.",
            operation=lambda: mne.pick_channels(ica_raw.ch_names, include=fit_channels),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        state.ica_fit_started = True
        _run_required_ica_substage(
            substage="ica_subset_fitting" if use_subset_fit else "ica_fitting",
            start_message="Beginning ICA.fit()...",
            complete_message="ICA.fit() completed successfully.",
            operation=lambda: ica.fit(ica_raw, picks=picks, decim=settings["decim"], reject_by_annotation=True, verbose=False),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        ica.exclude = []
        version_info = _version_info()
        fit_context = build_ica_fit_context_identity(
            participant=subject,
            recording_part=recording.recording_part,
            input_eeg=input_eeg_identity,
            fit_data_content_sha256=fit_data_sha256,
            fit_data_shape=(len(fit_channels), int(ica_raw.n_times)),
            fit_data_dtype="<f8",
            fit_sampling_frequency_hz=float(ica_raw.info["sfreq"]),
            selection=fit_selection,
            input_channel_names=raw.ch_names,
            input_channel_types=raw.get_channel_types(),
            bad_channel_names=raw.info.get("bads", []) or [],
            fit_channel_names=fit_channels,
            filter_parameters={
                "operation": "mne.io.BaseRaw.filter",
                "requested_highpass_hz": settings["highpass_hz"],
                "requested_lowpass_hz": settings["lowpass_hz"],
                "effective_highpass_hz": float(ica_raw.info["highpass"]),
                "effective_lowpass_hz": float(ica_raw.info["lowpass"]),
                "picks": list(fit_channels),
            },
            estimated_effective_rank=estimated_rank,
            algorithm_parameters={
                "method": str(getattr(ica, "method", settings["method"])),
                "n_components_strategy": settings["n_components"],
                "n_components_requested": int(estimated_rank),
                "n_components_effective": int(getattr(ica, "n_components_", 0) or 0),
                "decimation": int(settings["decim"]),
                "random_seed": settings["random_state"],
                "max_iter": getattr(ica, "max_iter", settings["max_iter"]),
                "fit_params": dict(getattr(ica, "fit_params", settings["fit_params"]) or {}),
                "reject_by_annotation": True,
                "mne_version": version_info["mne_version"],
                "numpy_version": version_info["numpy_version"],
            },
            reference_state=fit_reference_state,
            resolved_config_fingerprint=resolved_config_fingerprint,
            scientific_fingerprint=scientific_fingerprint,
            resolved_snapshot=resolved_snapshot,
            validation_only=False,
        )
        decomposition_id = ica_decomposition_identity(ica)
        solution_id = authoritative_ica_solution_identity(fit_context, decomposition_id)
        fit_provenance = {
            **fit_provenance,
            "ica_fit_context_schema": fit_context.schema_version,
            "ica_fit_context_id": fit_context.fingerprint,
            "ica_decomposition_id": decomposition_id,
            "ica_solution_id": solution_id,
        }
        state.ica_fit_completed = True
        processing_rows.append(build_ica_processing_log_row("ica_fitting", "state_update", "ica_fit_completed=True", status="ok", state=state))
        _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)

        if save_cfg.get("ica_solution", True):
            failed_substage = "ica_solution_saving"
            _run_required_ica_substage(
                substage="ica_solution_saving",
                start_message="Saving ICA solution...",
                complete_message="ICA solution saved.",
                operation=lambda: ica.save(outputs["ica_solution"], overwrite=True),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
            output_files["ica_solution"] = outputs["ica_solution"]
            fit_artifact = build_canonical_ica_fit_artifact_identity(
                fit_context=fit_context,
                solution_path=outputs["ica_solution"],
            )
            decomposition_id = fit_artifact.decomposition_fingerprint
            solution_id = fit_artifact.solution_fingerprint
            fit_provenance = {
                **fit_provenance,
                "ica_decomposition_id": decomposition_id,
                "ica_solution_id": solution_id,
            }
            write_ica_fit_artifact_identity(fit_artifact, outputs["ica_fit_identity"])
            output_files["ica_fit_identity"] = outputs["ica_fit_identity"]
            state.ica_solution_saved = True
            processing_rows.append(build_ica_processing_log_row("ica_solution_saving", "state_update", "ica_solution_saved=True", status="ok", state=state))
            _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
            ica_inputs = input_files + ([outputs["ica_ready_raw_fif"]] if "ica_ready_raw_fif" in output_files else [])
            _write_ica_sidecar(
                outputs["ica_solution"], subject, recording, config_path, root, ica_inputs,
                settings, warnings, review_file_paths, fit_provenance=fit_provenance,
            )
        del ica_raw
        gc.collect()

        # A newly fitted solution is necessarily pending review. Existing human
        # decisions are never copied onto a new component decomposition.
        confirmed_ica: list[int] = []
        summary = _run_required_ica_substage(
            substage="ica_summary_table",
            start_message="Generating ICA summary table...",
            complete_message="ICA summary table complete.",
            operation=lambda: build_ica_summary_table(
                subject,
                recording.recording_part,
                config,
                fit_channels=fit_channels,
                fitted_components=getattr(ica, "n_components_", None),
                n_iter=getattr(ica, "n_iter_", None),
                convergence_status="completed",
                confirmed_ica_components=confirmed_ica,
                fit_provenance=fit_provenance,
            ),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
        )
        eog_suggestions = pd.DataFrame(columns=["eog_channel", "component", "score", "score_abs", "suggested_ocular_component", "review_status", "retain_or_exclude_decision"])
        pca_components = pd.DataFrame(columns=["pca_component", "explained_variance", "explained_variance_ratio"])

        failed_substage = "ica_diagnostics_tables"
        diag_settings = ica_diagnostic_settings(config)
        diag_fail_pipeline = bool(diag_settings["fail_pipeline_on_error"])
        diagnostics_fail_pipeline = diag_fail_pipeline
        state.diagnostics_started = True
        if diag_settings["generate_eog_scores"]:
            diag_raw = _run_optional_ica_substage(
                substage="ica_diagnostic_raw_window",
                start_message="Preparing diagnostic raw window...",
                complete_message="Diagnostic raw window prepared.",
                operation=lambda: _diagnostic_raw_window(raw, config),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
                warnings=warnings,
                fail_pipeline=diag_fail_pipeline,
            )
            if diag_raw is not None:
                maybe_eog = _run_optional_ica_substage(
                    substage="ica_eog_scores",
                    start_message="Generating EOG scores...",
                    complete_message="EOG scores completed.",
                    operation=lambda: _eog_component_suggestions(ica, diag_raw, config, warnings),
                    outputs=outputs,
                    output_files=output_files,
                    processing_rows=processing_rows,
                    timing_rows=timing_rows,
                    state=state,
                    warnings=warnings,
                    fail_pipeline=diag_fail_pipeline,
                )
                if maybe_eog is not None:
                    eog_suggestions = maybe_eog
        else:
            warnings.append("Skipped ICA EOG scores because preprocessing.ica.diagnostics.generate_eog_scores is false.")
        maybe_pca = _run_optional_ica_substage(
            substage="ica_pca_components",
            start_message="Generating suggested components...",
            complete_message="Suggested components complete.",
            operation=lambda: pca_explained_variance_table(ica),
            outputs=outputs,
            output_files=output_files,
            processing_rows=processing_rows,
            timing_rows=timing_rows,
            state=state,
            warnings=warnings,
            fail_pipeline=diag_fail_pipeline,
        )
        if maybe_pca is not None:
            pca_components = maybe_pca

        for label, table in (("ica_summary", summary), ("eog_suggestions", eog_suggestions), ("pca_components", pca_components)):
            path = outputs[label]
            _run_required_ica_substage(
                substage=f"{label}_saving",
                start_message=f"Saving {label}...",
                complete_message=f"{label} saved.",
                operation=lambda table=table, path=path: table.to_csv(path, index=False),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
            )
            output_files[label] = path
            _write_ica_sidecar(
                path, subject, recording, config_path, root, input_files, settings,
                warnings, review_file_paths, fit_provenance=fit_provenance,
            )

        suggested_components = sorted(set(
            pd.to_numeric(
                eog_suggestions.loc[eog_suggestions.get("suggested_ocular_component", False) == True, "component"],  # noqa: E712
                errors="coerce",
            ).dropna().astype(int).tolist()
        )) if not eog_suggestions.empty else []
        write_ica_review_template(
            outputs["ica_review_template"],
            subject=subject,
            recording_part=recording.recording_part,
            ica_solution_id=solution_id,
            fitted_component_count=int(getattr(ica, "n_components_", 0) or 0),
            automatic_eog_suggested_components=suggested_components,
        )
        output_files["ica_review_template"] = outputs["ica_review_template"]
        _write_ica_sidecar(
            outputs["ica_review_template"], subject, recording, config_path, root, input_files,
            settings, warnings, review_file_paths, fit_provenance=fit_provenance,
        )

        failed_substage = "ica_diagnostics_figures"
        if diag_settings["generate_figures"]:
            maybe_figures = _run_optional_ica_substage(
                substage="ica_diagnostics_figures",
                start_message="Generating figures...",
                complete_message="Figures complete.",
                operation=lambda: write_ica_diagnostic_figures(
                    ica=ica,
                    raw=raw,
                    eog_suggestions=eog_suggestions,
                    pca_components=pca_components,
                    subject=subject,
                    recording=recording,
                    fig_dir=outputs["fig_dir"],
                    root=root,
                    config_path=config_path,
                    input_files=input_files,
                    warnings=warnings,
                    config=config,
                ),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
                warnings=warnings,
                fail_pipeline=diag_fail_pipeline,
            )
            if maybe_figures is not None:
                output_files.update(maybe_figures)
        else:
            warnings.append("Skipped ICA figures because preprocessing.ica.diagnostics.generate_figures is false.")
        state.diagnostics_completed = True
        processing_rows.append(build_ica_processing_log_row("ica_diagnostics", "state_update", "diagnostics_completed=True", status="ok", state=state))
        _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)

        failed_substage = "ica_report"
        if diag_settings["generate_reports"]:
            report = _run_optional_ica_substage(
                substage="ica_report_generation",
                start_message="Generating reports...",
                complete_message="Reports complete.",
                operation=lambda: generate_gate3b_ica_markdown_report(
                    subject=subject,
                    recording_part=recording.recording_part,
                    input_files=input_files,
                    output_files=output_files,
                    ica_summary=summary,
                    eog_suggestions=eog_suggestions,
                    pca_components=pca_components,
                    warnings=warnings,
                    failures=failures,
                    manual_review_files=review_file_paths,
                ),
                outputs=outputs,
                output_files=output_files,
                processing_rows=processing_rows,
                timing_rows=timing_rows,
                state=state,
                warnings=warnings,
                fail_pipeline=diag_fail_pipeline,
            )
            if report is not None:
                report_path = _run_optional_ica_substage(
                    substage="ica_report_saving",
                    start_message="Saving reports...",
                    complete_message="Report saved.",
                    operation=lambda: write_markdown_report(outputs["report"], report),
                    outputs=outputs,
                    output_files=output_files,
                    processing_rows=processing_rows,
                    timing_rows=timing_rows,
                    state=state,
                    warnings=warnings,
                    fail_pipeline=diag_fail_pipeline,
                )
                if report_path is not None:
                    output_files["ica_markdown_report"] = report_path
                    _write_ica_sidecar(report_path, subject, recording, config_path, root, input_files, settings, warnings, review_file_paths)
                    state.ica_report_completed = True
                    processing_rows.append(build_ica_processing_log_row("ica_report", "state_update", "ica_report_completed=True", status="ok", state=state))
                    _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
        else:
            warnings.append("Skipped ICA report because preprocessing.ica.diagnostics.generate_reports is false.")
        _log_gate3b_message("Gate 3B complete.")
        status = "ok"
    except Exception as error:
        if isinstance(error, Gate3BSubstageError):
            failed_substage = error.substage
            original_error = error.original
            elapsed_seconds = error.elapsed_seconds
            failed_operation = error.failed_operation
            failure_classification = error.failure_classification if error.failure_classification != "UNKNOWN" else classify_ica_failure(original_error, failed_substage)
        else:
            original_error = error
            elapsed_seconds = None
            failed_operation = failed_substage
            failure_classification = classify_ica_failure(original_error, failed_substage)
        failures.append(str(original_error))
        failures.append(_ica_failure_suggestion(original_error, config))
        failure_path = outputs["qc_dir"] / "failure.csv"
        build_ica_failure_record(
            subject,
            recording.recording_part,
            failed_substage=failed_substage,
            error=original_error,
            state=state,
            elapsed_seconds=elapsed_seconds,
            memory_configuration=_memory_configuration(config),
            version_info=_version_info(),
            failed_operation=failed_operation,
            failure_classification=failure_classification,
            filtering_started=filtering_started,
            input_validation=input_validation_summary,
            traceback_text=traceback.format_exc(),
        ).to_csv(failure_path, index=False)
        output_files["failure_record"] = failure_path
        _write_ica_sidecar(failure_path, subject, recording, config_path, root, input_files, ica_settings(config), warnings + failures, review_file_paths)
        _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
        if not diagnostics_fail_pipeline and (state.ica_solution_saved or outputs["ica_solution"].exists()) and failed_substage in {"ica_diagnostics_tables", "ica_diagnostics_figures", "ica_report", "ica_report_generation", "ica_report_saving"}:
            warnings.append(f"Gate 3B optional {failed_substage} failed after ICA solution was saved; keeping ICA usable: {original_error}")
            status = "ok"
            output_files["ica_solution"] = outputs["ica_solution"]
        else:
            status = "failed"

    return ICAFitResult(
        subject=subject,
        recording_part=recording.recording_part,
        status=status,
        output_files=output_files,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )


def write_ica_diagnostic_figures(
    *,
    ica,
    raw,
    eog_suggestions: pd.DataFrame,
    pca_components: pd.DataFrame,
    subject: str,
    recording: RecordingPart,
    fig_dir: Path,
    root: Path | None,
    config_path: str | Path | None,
    input_files: Sequence[str | Path],
    warnings: list[str],
    config: Mapping[str, Any],
) -> dict[str, Path]:
    """Generate reusable Gate 3B ICA diagnostic figures where MNE supports them."""

    outputs: dict[str, Path] = {}
    figure_paths = gate3b_ica_figure_paths(fig_dir, subject, recording.recording_part)
    settings = ica_settings(config)
    diag_settings = ica_diagnostic_settings(config)
    n_components = int(getattr(ica, "n_components_", 0) or 0)
    fail_pipeline = bool(diag_settings["fail_pipeline_on_error"])
    diag_raw = _diagnostic_raw_window(raw, config)

    if diag_settings["generate_topomaps"]:
        _log_gate3b_message("Generating topomaps...")
        review_page_size = int(diag_settings["component_review_page_size"])
        review_paths = gate3b_ica_component_review_paths(
            fig_dir, subject, recording.recording_part, n_components, page_size=review_page_size
        )
        for page, picks in enumerate(_component_review_pages(n_components, page_size=review_page_size), start=1):
            _save_mne_figure(
                outputs,
                f"component_maps_page_{page:02d}",
                lambda picks=picks: plot_ica_component_topographies(
                    ica, raw, picks=picks,
                    title=f"{subject} {recording.recording_part} ICA component topographies",
                ),
                review_paths["component_maps_pages"][page - 1],
                subject,
                recording,
                root,
                config_path,
                input_files,
                warnings,
                fail_pipeline=fail_pipeline,
            )
        if "component_maps_page_01" in outputs:
            outputs["component_maps_figure"] = outputs["component_maps_page_01"]
        _log_gate3b_message("Topomap generation complete.")
    else:
        warnings.append("Skipped ICA topomaps because preprocessing.ica.diagnostics.generate_topomaps is false.")
    review_page_size = int(diag_settings["component_review_page_size"])
    review_paths = gate3b_ica_component_review_paths(
        fig_dir, subject, recording.recording_part, n_components, page_size=review_page_size
    )
    for page, picks in enumerate(_component_review_pages(n_components, page_size=review_page_size), start=1):
        _save_mne_figure(
            outputs,
            f"component_sources_page_{page:02d}",
            lambda picks=picks: ica.plot_sources(diag_raw, picks=picks, show=False),
            review_paths["component_sources_pages"][page - 1],
            subject,
            recording,
            root,
            config_path,
            input_files,
            warnings,
            fail_pipeline=fail_pipeline,
        )
    if "component_sources_page_01" in outputs:
        outputs["component_sources_figure"] = outputs["component_sources_page_01"]
    if diag_settings["generate_component_spectra"]:
        for page, picks in enumerate(_component_review_pages(n_components, page_size=review_page_size), start=1):
            _save_mne_figure(
                outputs,
                f"component_spectra_page_{page:02d}",
                lambda picks=picks: plot_ica_component_spectra(
                    ica, diag_raw, picks=picks,
                    title=f"{subject} {recording.recording_part} ICA source spectra",
                ),
                review_paths["component_spectra_pages"][page - 1],
                subject, recording, root, config_path, input_files, warnings,
                fail_pipeline=fail_pipeline,
            )
    else:
        warnings.append("Skipped ICA component spectra because preprocessing.ica.diagnostics.generate_component_spectra is false.")
    _save_mne_figure(
        outputs,
        "eog_scores_figure",
        lambda: plot_ica_eog_scores(eog_suggestions, title=f"{subject} {recording.recording_part} ICA EOG scores"),
        figure_paths["eog_scores_figure"],
        subject,
        recording,
        root,
        config_path,
        input_files,
        warnings,
        fail_pipeline=fail_pipeline,
    )
    _save_mne_figure(
        outputs,
        "pca_explained_variance_figure",
        lambda: plot_pca_explained_variance(pca_components, title=f"{subject} {recording.recording_part} ICA PCA variance"),
        figure_paths["pca_explained_variance_figure"],
        subject,
        recording,
        root,
        config_path,
        input_files,
        warnings,
        fail_pipeline=fail_pipeline,
    )

    if not bool(diag_settings["generate_component_properties"]) or bool(diag_settings["skip_component_properties_by_default"]):
        warnings.append("Skipped ICA component property plots for Gate 3B diagnostics.")
        return outputs
    _log_gate3b_message("Generating component properties...")
    for component in range(n_components):
        property_path = fig_dir / f"component_{component:02d}_properties.png"
        _save_mne_figure(
            outputs,
            f"component_{component:02d}_properties_figure",
            lambda component=component: ica.plot_properties(diag_raw, picks=[component], show=False),
            property_path,
            subject,
            recording,
            root,
            config_path,
            input_files,
            warnings,
            fail_pipeline=fail_pipeline,
        )
    _log_gate3b_message("Component properties complete.")
    return outputs


def _component_review_pages(n_components: int, *, page_size: int = 10) -> list[list[int]]:
    """Return non-overlapping component index pages covering every fitted IC."""

    if n_components < 0 or page_size <= 0:
        raise ValueError("n_components must be non-negative and page_size must be positive.")
    return [list(range(start, min(start + page_size, n_components))) for start in range(0, n_components, page_size)] or [[]]


def _mne_ica_fit_channels(raw, config: Mapping[str, Any]) -> list[str]:
    return _mne_ica_candidate_channels(raw, config, bad_channels=raw.info.get("bads", []))


def _ica_fit_selection_rows(
    *,
    use_subset_fit: bool,
    subset_windows: pd.DataFrame,
    task_segments: Sequence[Any],
    crop_window: Any | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return the exact source-coordinate windows used to construct fit Raw."""

    if use_subset_fit:
        rows = []
        for _, row in subset_windows.sort_values("window_id", kind="mergesort").iterrows():
            block = row.get("block")
            rows.append({
                "start_sample": int(row["start_sample"]),
                "end_sample": int(row["end_sample"]),
                "start_trial": int(row["start_trial"]),
                "end_trial": int(row["end_trial"]),
                "block": None if pd.isna(block) else str(block),
            })
        return "representative_subset", rows
    if task_segments:
        return "task_segments", [
            {
                "start_sample": int(segment.start_sample),
                "end_sample": int(segment.end_sample),
                "start_trial": int(segment.start_trial),
                "end_trial": int(segment.end_trial),
                "block": None,
            }
            for segment in task_segments
        ]
    if crop_window is None:
        raise ValueError("ICA fit selection has no subset, task segments, or crop window.")
    return "task_crop", [{
        "start_sample": int(crop_window.crop_start_sample),
        "end_sample": int(crop_window.crop_end_sample),
        "start_trial": int(crop_window.first_retained_trial),
        "end_trial": int(crop_window.last_retained_trial),
        "block": None,
    }]


def _mne_ica_candidate_channels(raw, config: Mapping[str, Any], *, bad_channels: Sequence[str] | None = None) -> list[str]:
    channel_types = {name: raw.get_channel_types(picks=[name])[0] for name in raw.ch_names}
    selected = select_ica_fit_channels(raw.ch_names, channel_types, config, bad_channels=bad_channels)
    return [channel for channel in selected if channel in raw.ch_names]


def _filter_ica_raw(ica_raw, fit_channels: Sequence[str], settings: Mapping[str, Any]):
    return ica_raw.filter(l_freq=settings["highpass_hz"], h_freq=settings["lowpass_hz"], picks=list(fit_channels), verbose=False)


def _build_ica_subset_raw(raw, windows: pd.DataFrame, fit_channels: Sequence[str]):
    """Return an EEG-only Raw made from selected representative ICA windows."""

    if windows.empty:
        raise ValueError("ICA subset creation failed: no subset windows were selected.")
    import mne

    cropped = []
    sfreq = float(raw.info["sfreq"])
    raw_first = int(raw.first_samp)
    raw_last = raw_first + int(raw.n_times) - 1
    for _, row in windows.sort_values("window_id", kind="mergesort").iterrows():
        start_sample = int(row["start_sample"])
        end_sample = int(row["end_sample"])
        if start_sample < raw_first or end_sample > raw_last:
            raise ValueError(
                "ICA subset creation failed: selected window "
                f"{start_sample}-{end_sample} falls outside raw samples {raw_first}-{raw_last}."
            )
        tmin = float(start_sample - raw_first) / sfreq
        tmax = float(end_sample - raw_first) / sfreq
        cropped.append(raw.copy().pick(list(fit_channels)).crop(tmin=tmin, tmax=tmax, include_tmax=True))
    subset = cropped[0] if len(cropped) == 1 else mne.concatenate_raws(cropped, preload=False, verbose=False)
    subset.load_data(verbose=False)
    return subset


def _build_ica_input_summaries(
    subject: str,
    recording_part: str,
    raw,
    input_raw: Path,
    fit_channels: Sequence[str],
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    summary_cache: ICADiagnosticSummaryCache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise ICA inputs for current and comparison participants."""

    cache = summary_cache if summary_cache is not None else ICADiagnosticSummaryCache()
    current = _ica_summary_for_input(
        subject,
        recording_part,
        raw,
        input_raw,
        fit_channels,
        config,
        status="current",
        root=root,
        cache=cache,
    )
    rows = [current]
    comparison_subjects = _comparison_subjects(config, subject)
    import mne

    for other_subject in comparison_subjects:
        other_path = stage3a_filtered_raw_path(config, other_subject, recording_part, root=root)
        if not other_path.exists():
            rows.append({"subject": other_subject, "recording_part": recording_part, "status": "missing_filtered_raw", "input_file": str(other_path)})
            continue
        try:
            other_raw = mne.io.read_raw_fif(other_path, preload=False, verbose=False)
            other_channels = _mne_ica_fit_channels(other_raw, config)
            rows.append(
                _ica_summary_for_input(
                    other_subject,
                    recording_part,
                    other_raw,
                    other_path,
                    other_channels,
                    config,
                    status="comparison",
                    root=root,
                    cache=cache,
                )
            )
        except Exception as error:
            rows.append({"subject": other_subject, "recording_part": recording_part, "status": "summary_failed", "input_file": str(other_path), "summary_error": f"{type(error).__name__}: {error}"})
    return pd.DataFrame([current]), pd.DataFrame(rows)


def _ica_summary_context_fingerprint(config: Mapping[str, Any], fit_channels: Sequence[str]) -> str:
    """Fingerprint the configuration and fit-channel context used by a summary."""

    settings = ica_settings(config)
    return canonical_sha256(
        {
            # These settings define the input-channel selection context.  Do
            # not include diagnostic presentation/output options that cannot
            # change the numerical input summary.
            "fit_only_eeg_channels": settings["fit_only_eeg_channels"],
            "exclude_channels": settings["exclude_channels"],
            "ocular_detection_channels": settings["ocular_detection_channels"],
            "fit_channels": [str(channel) for channel in fit_channels],
        }
    )


def _ica_summary_for_input(
    subject: str,
    recording_part: str,
    raw,
    input_file: Path,
    fit_channels: Sequence[str],
    config: Mapping[str, Any],
    *,
    status: str,
    root: Path | None,
    cache: ICADiagnosticSummaryCache,
) -> dict[str, Any]:
    """Return one summary row, reusing only an identity-bound completed core."""

    source_identity = identify_source_file(
        input_file,
        "ica_filtered_raw_input",
        logical_root=root,
    )
    summary = cache.get_or_compute(
        source_identity=source_identity,
        recording_part=recording_part,
        fit_channels=fit_channels,
        config=config,
        compute=lambda: _summarise_raw_for_ica(
            subject,
            recording_part,
            raw,
            input_file,
            fit_channels,
            status=status,
        ),
    )
    summary.update(
        {
            "subject": subject,
            "recording_part": recording_part,
            "status": status,
            "input_file": str(input_file),
            "file_size_bytes": input_file.stat().st_size if input_file.exists() else pd.NA,
        }
    )
    return summary


def _comparison_subjects(config: Mapping[str, Any], subject: str) -> list[str]:
    participants = dict(config.get("participants", {}) or {})
    defaults = dict(participants.get("defaults", {}) or {})
    included = [str(item) for item in defaults.get("included", []) or []]
    return [item for item in included if item != subject]


def _summarise_raw_for_ica(subject: str, recording_part: str, raw, input_file: Path, fit_channels: Sequence[str], *, status: str) -> dict[str, Any]:
    import numpy as np

    picks = [channel for channel in fit_channels if channel in raw.ch_names]
    sfreq = float(raw.info.get("sfreq", 0.0) or 0.0)
    n_times = int(getattr(raw, "n_times", 0) or 0)
    chunk_samples = max(1, int(sfreq * 30.0)) if sfreq else max(1, min(n_times, 15000))
    channel_stats = {channel: {"sum": 0.0, "sumsq": 0.0, "n": 0, "nan": 0, "inf": 0, "min": None, "max": None, "zero": 0} for channel in picks}
    total_nan = 0
    total_inf = 0
    total_samples = 0
    total_zero = 0
    dtype_names: set[str] = set()
    global_min = None
    global_max = None
    channels_with_nan_or_inf: set[str] = set()
    flat_channels: list[str] = []
    clipping_channels: set[str] = set()

    for start in range(0, n_times, chunk_samples):
        stop = min(n_times, start + chunk_samples)
        data = raw.get_data(picks=picks, start=start, stop=stop)
        dtype_names.add(str(data.dtype))
        finite = np.isfinite(data)
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)
        total_nan += int(nan_mask.sum())
        total_inf += int(inf_mask.sum())
        total_samples += int(data.size)
        total_zero += int((data == 0).sum())
        finite_values = data[finite]
        if finite_values.size:
            chunk_min = float(finite_values.min())
            chunk_max = float(finite_values.max())
            global_min = chunk_min if global_min is None else min(global_min, chunk_min)
            global_max = chunk_max if global_max is None else max(global_max, chunk_max)
        for idx, channel in enumerate(picks):
            row = data[idx]
            row_finite = row[np.isfinite(row)]
            stats = channel_stats[channel]
            stats["nan"] += int(np.isnan(row).sum())
            stats["inf"] += int(np.isinf(row).sum())
            stats["zero"] += int((row == 0).sum())
            if stats["nan"] or stats["inf"]:
                channels_with_nan_or_inf.add(channel)
            if row_finite.size:
                stats["sum"] += float(row_finite.sum())
                stats["sumsq"] += float((row_finite * row_finite).sum())
                stats["n"] += int(row_finite.size)
                rmin = float(row_finite.min())
                rmax = float(row_finite.max())
                stats["min"] = rmin if stats["min"] is None else min(stats["min"], rmin)
                stats["max"] = rmax if stats["max"] is None else max(stats["max"], rmax)
                if max(abs(rmin), abs(rmax)) > 500e-6:
                    clipping_channels.add(channel)

    variances: dict[str, float] = {}
    for channel, stats in channel_stats.items():
        if stats["n"]:
            mean = stats["sum"] / stats["n"]
            variance = max(0.0, stats["sumsq"] / stats["n"] - mean * mean)
            variances[channel] = variance
            if variance == 0.0:
                flat_channels.append(channel)

    return {
        "subject": subject,
        "recording_part": recording_part,
        "status": status,
        "input_file": str(input_file),
        "file_size_bytes": input_file.stat().st_size if input_file.exists() else pd.NA,
        "duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
        "n_samples": n_times,
        "sfreq": sfreq,
        "channel_count": len(raw.ch_names),
        "eeg_channel_count": len(picks),
        "eeg_channel_names": list(picks),
        "bad_channels": list(raw.info.get("bads", []) or []),
        "data_dtype": sorted(dtype_names),
        "preload_state": bool(getattr(raw, "preload", False)),
        "memory_mapping_state": str(type(getattr(raw, "_data", None)).__name__) if getattr(raw, "preload", False) else "not_preloaded",
        "n_annotations": len(getattr(raw, "annotations", [])),
        "event_count": len(getattr(raw, "annotations", [])),
        "nan_count": total_nan,
        "inf_count": total_inf,
        "nan_values_present": total_nan > 0,
        "inf_values_present": total_inf > 0,
        "channels_with_nan_or_inf": sorted(channels_with_nan_or_inf),
        "signal_min": global_min,
        "signal_max": global_max,
        "channel_variance_min": min(variances.values()) if variances else pd.NA,
        "channel_variance_max": max(variances.values()) if variances else pd.NA,
        "flat_channels": flat_channels,
        "constant_channel_count": len(flat_channels),
        "zero_fraction": float(total_zero / total_samples) if total_samples else 0.0,
        "clipping_channels": sorted(clipping_channels),
        "clipping_channel_count": len(clipping_channels),
    }


def _eog_component_suggestions(ica, raw, config: Mapping[str, Any], warnings: list[str]) -> pd.DataFrame:
    rows = []
    for eog_channel in ocular_detection_channels(config):
        if eog_channel not in raw.ch_names:
            warnings.append(f"Ocular detection channel missing from raw data: {eog_channel}")
            continue
        try:
            components, scores = ica.find_bads_eog(raw, ch_name=eog_channel, verbose=False)
        except Exception as error:
            warnings.append(f"Could not compute EOG correlations for {eog_channel}: {error}")
            continue
        suggested = {int(component) for component in components}
        for component, score in enumerate(scores):
            rows.append(
                {
                    "eog_channel": eog_channel,
                    "component": component,
                    "score": float(score),
                    "score_abs": abs(float(score)),
                    "suggested_ocular_component": component in suggested,
                    "review_status": "suggested_for_manual_review" if component in suggested else "not_suggested",
                    "retain_or_exclude_decision": "pending_manual_review",
                }
            )
    return pd.DataFrame(rows, columns=["eog_channel", "component", "score", "score_abs", "suggested_ocular_component", "review_status", "retain_or_exclude_decision"])


def _diagnostic_raw_window(raw, config: Mapping[str, Any]):
    """Return a short raw view for optional ICA diagnostics."""

    start, stop = ica_diagnostic_settings(config)["time_window_s"]
    tmin = max(0.0, float(start))
    raw_tmax = float(raw.times[-1]) if len(raw.times) else tmin
    tmax = min(max(tmin, float(stop)), raw_tmax)
    return raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)


def _handle_optional_ica_diagnostic_error(substage: str, error: Exception, warnings: list[str], *, fail_pipeline: bool) -> None:
    message = f"Optional Gate 3B {substage} failed: {type(error).__name__}: {error}"
    if fail_pipeline:
        raise error
    warnings.append(message)


def _log_gate3b_message(message: str) -> None:
    LOGGER.info(message)
    print(f"[KAWAII EEG] {message}")


def _write_ica_instrumentation_logs(outputs: Mapping[str, Path], output_files: dict[str, Path], processing_rows: list[dict[str, Any]], timing_rows: list[dict[str, Any]]) -> None:
    if processing_rows:
        pd.DataFrame(processing_rows).to_csv(outputs["processing_log"], index=False)
        output_files["processing_log"] = outputs["processing_log"]
    if timing_rows:
        pd.DataFrame(timing_rows).to_csv(outputs["ica_timing"], index=False)
        output_files["ica_timing"] = outputs["ica_timing"]


def _run_required_ica_substage(
    *,
    substage: str,
    start_message: str,
    complete_message: str,
    operation,
    outputs: Mapping[str, Path],
    output_files: dict[str, Path],
    processing_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    state: ICAInstrumentationState,
):
    _log_gate3b_message(start_message)
    processing_rows.append(build_ica_processing_log_row(substage, "start", start_message, status="running", state=state))
    _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as error:
        elapsed = time.perf_counter() - started
        timing_rows.append(build_ica_timing_row(substage, "failed", elapsed, message=start_message, exception_type=type(error).__name__, exception_message=str(error)))
        processing_rows.append(build_ica_processing_log_row(substage, "failure", str(error), status="failed", elapsed_seconds=elapsed, state=state))
        _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
        raise Gate3BSubstageError(substage, error, elapsed) from error
    elapsed = time.perf_counter() - started
    _log_gate3b_message(complete_message)
    timing_rows.append(build_ica_timing_row(substage, "ok", elapsed, message=complete_message))
    processing_rows.append(build_ica_processing_log_row(substage, "complete", complete_message, status="ok", elapsed_seconds=elapsed, state=state))
    _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
    return result


def _run_optional_ica_substage(
    *,
    substage: str,
    start_message: str,
    complete_message: str,
    operation,
    outputs: Mapping[str, Path],
    output_files: dict[str, Path],
    processing_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    state: ICAInstrumentationState,
    warnings: list[str],
    fail_pipeline: bool,
):
    _log_gate3b_message(start_message)
    processing_rows.append(build_ica_processing_log_row(substage, "start", start_message, status="running", state=state))
    _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as error:
        elapsed = time.perf_counter() - started
        timing_rows.append(build_ica_timing_row(substage, "warning" if not fail_pipeline else "failed", elapsed, message=start_message, exception_type=type(error).__name__, exception_message=str(error)))
        processing_rows.append(build_ica_processing_log_row(substage, "warning" if not fail_pipeline else "failure", str(error), status="warning" if not fail_pipeline else "failed", elapsed_seconds=elapsed, state=state))
        _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
        if fail_pipeline:
            raise Gate3BSubstageError(substage, error, elapsed) from error
        warnings.append(f"Optional Gate 3B {substage} failed: {type(error).__name__}: {error}")
        return None
    elapsed = time.perf_counter() - started
    _log_gate3b_message(complete_message)
    timing_rows.append(build_ica_timing_row(substage, "ok", elapsed, message=complete_message))
    processing_rows.append(build_ica_processing_log_row(substage, "complete", complete_message, status="ok", elapsed_seconds=elapsed, state=state))
    _write_ica_instrumentation_logs(outputs, output_files, processing_rows, timing_rows)
    return result


def _memory_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    preprocessing = dict(config.get("preprocessing", {}) or {})
    return {
        "memory": dict(preprocessing.get("memory", {}) or {}),
        "ica": {
            key: value
            for key, value in ica_settings(config).items()
            if key not in {"fit_params", "diagnostics"}
        },
        "ica_diagnostics": ica_diagnostic_settings(config),
    }


def _version_info() -> dict[str, str]:
    versions = {"python_version": sys.version.replace("\n", " "), "platform": platform.platform()}
    try:
        import mne

        versions["mne_version"] = str(mne.__version__)
    except Exception:
        versions["mne_version"] = ""
    try:
        import numpy as np

        versions["numpy_version"] = str(np.__version__)
    except Exception:
        versions["numpy_version"] = ""
    return versions


def _array_shape_from_message(message: str) -> str:
    import re

    match = re.search(r"shape\s+(\([^)]+\))", message)
    return match.group(1) if match else ""


def _array_dtype_from_message(message: str) -> str:
    import re

    match = re.search(r"(?:dtype|data type)\s+([A-Za-z0-9_]+)", message)
    return match.group(1) if match else ""


def _available_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _save_mne_figure(
    outputs: dict[str, Path],
    label: str,
    builder,
    path: Path,
    subject: str,
    recording: RecordingPart,
    root: Path | None,
    config_path: str | Path | None,
    input_files: Sequence[str | Path],
    warnings: list[str],
    *,
    fail_pipeline: bool = False,
) -> None:
    try:
        figures = builder()
        if not isinstance(figures, list):
            figures = [figures]
        for idx, figure in enumerate(figures):
            target = path if idx == 0 else path.with_name(f"{path.stem}_{idx + 1:02d}{path.suffix}")
            saved = save_figure_with_stage_provenance(
                figure,
                target,
                pipeline_stage="gate3b_ica_diagnostic_figure",
                subject=subject,
                recording_part=recording.recording_part,
                root=root,
                config_path=config_path,
                input_files=input_files,
                parameters_summary={"figure_type": label, "ica_application": "not_applied_gate3b"},
                warnings=warnings,
            )
            outputs[label if idx == 0 else f"{label}_{idx + 1:02d}"] = saved
    except Exception as error:
        if fail_pipeline:
            raise
        warnings.append(f"Could not generate {label}: {error}")


def _write_ica_sidecar(
    output_file: Path,
    subject: str,
    recording: RecordingPart,
    config_path: str | Path | None,
    root: Path | None,
    input_files: Sequence[str | Path],
    settings: Mapping[str, Any],
    warnings: Sequence[str],
    manual_review_files: Mapping[str, Path],
    fit_provenance: Mapping[str, Any] | None = None,
) -> None:
    write_provenance_sidecar(
        output_file,
        pipeline_stage="gate3b_ica",
        subject=subject,
        recording_part=recording.recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary={
            "ica_method": settings.get("method"),
            "n_components": settings.get("n_components"),
            "decim": settings.get("decim"),
            "max_iter": settings.get("max_iter"),
            "random_state": settings.get("random_state"),
            "ica_highpass_hz": settings.get("highpass_hz"),
            "ica_lowpass_hz": settings.get("lowpass_hz"),
            "fit_only_eeg_channels": settings.get("fit_only_eeg_channels"),
            "fit_subset": settings.get("fit_subset"),
            "exclude_channels": settings.get("exclude_channels"),
            "ocular_detection_channels": settings.get("ocular_detection_channels"),
            "ica_application": "not_applied_gate3b",
            "task_crop_scope": "Gate 3B uses task-window filtered raw for standard ICA fitting; unusually long recordings use representative EEG-only ICA subset fitting. Gate 3A raw QC remains full-recording.",
            "manual_review_files": {key: str(value) for key, value in manual_review_files.items()},
            **dict(fit_provenance or {}),
        },
        warnings=warnings,
    )


def _ica_failure_suggestion(error: Exception, config: Mapping[str, Any]) -> str:
    settings = ica_settings(config)
    return (
        "ICA failed. Suggested lower-memory settings: increase preprocessing.ica.decim "
        f"above {settings.get('decim')}, verify the estimated effective EEG rank, "
        "disable preprocessing.save.ica_ready_raw_fif, and verify only EEG picks are used. "
        f"Original error type: {type(error).__name__}."
    )
