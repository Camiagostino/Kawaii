from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .loading import RecordingPart
from .manual_review import (
    bad_channel_review_context_identity,
    bad_channel_source_identity,
    manual_review_paths,
    read_bad_channel_review_decision,
    read_confirmed_ica_components,
    read_review_notes,
)
from .paths import subject_part_stage_dir
from .plots import (
    gate3a_figure_paths,
    plot_bad_channel_summary,
    plot_channel_variance,
    plot_correlation_qc_summary,
    plot_line_noise,
    plot_montage,
    plot_psd,
    plot_raw_overview,
    save_figure_with_provenance,
)
from .provenance import write_provenance_sidecar
from .resolved_snapshot import ResolvedRunSnapshotReference, bind_authoritative_config_provenance
from .raw_qc import (
    apply_channel_types,
    apply_montage_if_safe,
    eog_activity_summary,
    psd_line_noise_summary,
    raw_qc_summary,
    run_bad_channel_qc,
)
from .reporting import generate_gate3a_markdown_report, write_markdown_report
from .task_crop import (
    apply_task_crop,
    apply_task_segments,
    calculate_task_crop_window,
    calculate_task_segments,
    long_gap_settings,
    task_crop_debug_table,
    task_crop_settings,
    task_crop_summary,
    task_segments_table,
)
from .temporary_resources import TemporaryResourceOwner


class Gate3AFilterMemoryError(MemoryError):
    """Gate 3A filtering failed in the memory-sensitive filtering path."""


@dataclass(frozen=True)
class PreprocessingPartResult:
    """Gate 3A outputs for one participant recording part."""

    subject: str
    recording_part: str
    status: str
    output_files: dict[str, Path]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]


def run_gate3a_for_recording_part(
    subject: str,
    recording: RecordingPart,
    config: Mapping[str, Any],
    *,
    alignment: pd.DataFrame | None = None,
    root: Path | None = None,
    config_path: str | Path | None = None,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
    execution_run_id: str | None = None,
) -> PreprocessingPartResult:
    """Run MNE raw loading, raw/channel QC, and deterministic filtering.

    This function intentionally stops before ICA fitting, epoching, evoked
    creation, or scientific analyses.
    """

    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    part_dir = subject_part_stage_dir(config, subject, recording.recording_part, "preprocessing", root=root, create=True)
    qc_dir = subject_part_stage_dir(config, subject, recording.recording_part, "raw_qc", root=root, create=True)
    fig_dir = qc_dir / "figures"
    for directory in (part_dir, qc_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    failures: list[str] = []
    output_files: dict[str, Path] = {}
    input_files = [recording.vhdr_path, recording.vmrk_path, recording.eeg_path]
    review_paths = manual_review_paths(config, root=root)
    review_file_paths = {key: value for key, value in review_paths.items() if key != "manual_review_dir"}
    stage_context = "loading"
    temporary_owner: TemporaryResourceOwner | None = None

    try:
        import mne

        memory_cfg = gate3a_memory_settings(config, root=root)
        temporary_owner = TemporaryResourceOwner.create(
            memory_cfg["memmap_dir"],
            stage="gate3a",
            participant=subject,
            recording_part=recording.recording_part,
            run_id=execution_run_id,
        )
        raw = mne.io.read_raw_brainvision(recording.vhdr_path, preload=memory_cfg["mne_preload_mode"], verbose=False)
        raw, channel_warnings = apply_channel_types(raw, config)
        warnings.extend(channel_warnings)
        raw, montage_report, montage_warnings = apply_montage_if_safe(raw, config)
        warnings.extend(montage_warnings)

        part_alignment = _alignment_for_part(alignment, recording.recording_part)
        stage_context = "qc"
        raw_summary = raw_qc_summary(raw, subject, recording.recording_part, config, alignment=part_alignment)
        psd_summary = psd_line_noise_summary(raw, subject, recording.recording_part, config)
        eog_summary = eog_activity_summary(raw, subject, recording.recording_part, config)
        reviewed_eeg_channels = [
            name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"
        ]
        bad_source_id = bad_channel_source_identity(recording)
        bad_context_id = bad_channel_review_context_identity(
            recording, config, reviewed_eeg_channels, source_recording_id=bad_source_id
        )
        bad_review_decision = read_bad_channel_review_decision(
            config,
            subject,
            recording.recording_part,
            root=root,
            expected_source_recording_id=bad_source_id,
            expected_review_context_id=bad_context_id,
            expected_reviewed_channels=reviewed_eeg_channels,
        )
        confirmed_bad_channels = (
            list(bad_review_decision.confirmed_bad_channels) if bad_review_decision.complete else []
        )
        confirmed_ica_components = read_confirmed_ica_components(config, subject, recording.recording_part, root=root)
        review_notes = read_review_notes(config, subject, recording.recording_part, root=root)
        correlation_segments = bad_channel_correlation_sample_segments(
            raw,
            part_alignment,
            recording.recording_part,
            config,
        )
        channel_qc, bad_decisions, bad_channel_warnings = run_bad_channel_qc(
            raw,
            subject,
            recording.recording_part,
            config,
            confirmed_bad_channels=confirmed_bad_channels,
            correlation_sample_segments=correlation_segments,
        )
        for _, warning_row in bad_channel_warnings.iterrows():
            reasons = str(warning_row.get("warning_reasons", "") or "")
            if reasons:
                warnings.append(f"Bad-channel QC review warning ({warning_row.get('scope')}): {reasons}.")
        overrides = manual_overrides_for_part(config, subject, recording.recording_part)
        overrides["manual_review_files"] = {key: str(value) for key, value in review_paths.items() if key != "manual_review_dir"}
        overrides["confirmed_bad_channels_from_review"] = confirmed_bad_channels
        overrides["bad_channel_review_state"] = bad_review_decision.decision_state
        overrides["bad_channel_decision_id"] = bad_review_decision.decision_id
        overrides["bad_channel_source_recording_id"] = bad_source_id
        overrides["bad_channel_review_context_id"] = bad_context_id
        overrides["bad_channel_legacy_ambiguous"] = bad_review_decision.legacy_ambiguous
        overrides["confirmed_ica_components_from_review_not_applied_gate3a"] = confirmed_ica_components
        if review_notes:
            overrides["manual_review_notes"] = review_notes
        decisions = deterministic_preprocessing_decisions(config, warnings=warnings, manual_overrides=overrides, raw=raw, memory_settings=memory_cfg)
        qc_row = channel_qc[channel_qc["included_in_eeg_qc"].astype(bool)].iloc[0] if bool(channel_qc["included_in_eeg_qc"].astype(bool).any()) else {}
        decisions = _append_decision_rows(
            decisions,
            [
                {"decision": "bad_channel_qc_input", "value": "unfiltered_raw_with_isolated_qc_only_1_50_hz_correlation_representation", "reason": "Gate 3A amplitude/deviation QC uses acquisition raw; PREP-style correlation uses a temporary QC-only filtered representation and never mutates downstream data."},
                {"decision": "bad_channel_qc_sample_windows", "value": qc_row.get("qc_sample_windows", ""), "reason": "Deterministic distributed windows used for bounded whole-recording coverage."},
                {"decision": "bad_channel_correlation_sample_segments", "value": [f"{start}:{stop}" for start, stop in correlation_segments], "reason": "Alignment-derived task segments, including configured padding, omit long idle gaps from correlation QC."},
                {"decision": "bad_channel_qc_metrics", "value": ["std_uv", "peak_to_peak_uv", "clipping_fraction", "p98_abs_interchannel_correlation_1_50_hz_1_s", "intermittent_dropout", "robust_log_std_deviation_z"], "reason": "Separate transparent automatic proposal metrics; no composite exclusion score."},
                {"decision": "automatic_bad_channel_metric_flags", "value": bad_decisions.loc[bad_decisions["automatic_review_priority"].ne("NO_AUTOMATIC_FLAG"), "channel"].astype(str).tolist(), "reason": "All transparent metric flags, including non-discriminating correlation-only evidence."},
                {"decision": "automatic_bad_channel_proposals", "value": bad_decisions.loc[bad_decisions["automatic_bad_channel_proposal"].astype(bool), "channel"].astype(str).tolist(), "reason": "Review shortlist only; poor-correlation-only results remain metric evidence and require human interpretation."},
                {"decision": "manual_confirmed_bad_channels", "value": bad_decisions.loc[bad_decisions["manual_confirmed_bad_channel"].astype(bool), "channel"].astype(str).tolist(), "reason": "Only these channels are passed downstream as bad."},
                {"decision": "bad_channel_spatial_review_warnings", "value": bad_channel_warnings[["scope", "warning_reasons"]].to_dict(orient="records"), "reason": "Informational only; never automatic exclusion."},
            ],
        )

        stage_context = "filtering"
        filter_metadata_rows: list[dict[str, Any]] = []
        filtered_raw = apply_gate3a_filtering(
            raw,
            config,
            decisions,
            raw_path=recording.vhdr_path,
            memmap_dir=Path(memory_cfg["memmap_dir"]),
            warnings=warnings,
            alignment=part_alignment,
            recording_part=recording.recording_part,
            subject=subject,
            metadata_rows=filter_metadata_rows,
            temporary_owner=temporary_owner,
        )
        decisions = _append_decision_rows(decisions, filter_metadata_rows)
        stage_context = "writing"

        tables = {
            "raw_qc_summary": raw_summary,
            "channel_qc": channel_qc,
            "montage_validation": montage_report,
            "psd_line_noise_summary": psd_summary,
            "eog_activity_summary": eog_summary,
            "bad_channel_decisions": bad_decisions,
            "bad_channel_spatial_warnings": bad_channel_warnings,
        }
        for label, table in tables.items():
            path = qc_dir / f"{label}.csv"
            table.to_csv(path, index=False)
            output_files[label] = path
            _write_sidecar(path, subject, recording, config_path, root, input_files, warnings, manual_review_files=review_file_paths)
        decisions_path = part_dir / "processing_decisions.csv"
        decisions.to_csv(decisions_path, index=False)
        output_files["preprocessing_decisions"] = decisions_path
        _write_sidecar(decisions_path, subject, recording, config_path, root, input_files, warnings, manual_review_files=review_file_paths)

        figure_cfg = dict(dict(dict(config.get("preprocessing", {}) or {}).get("raw_qc", {}) or {}).get("figures", {}) or {})
        if bool(figure_cfg.get("enabled", True)):
            stage_context = "qc_figure"
            figure_paths = write_gate3a_qc_figures(
                raw=raw,
                channel_qc=channel_qc,
                bad_channel_decisions=bad_decisions,
                psd_summary=psd_summary,
                subject=subject,
                recording=recording,
                fig_dir=fig_dir,
                root=root,
                config_path=config_path,
                input_files=input_files,
                warnings=warnings,
                config=config,
            )
            output_files.update(figure_paths)
            stage_context = "writing"
        else:
            warnings.append("Gate 3A QC figure generation skipped because preprocessing.raw_qc.figures.enabled is false.")

        save_cfg = dict(config.get("preprocessing", {}).get("save", {}) or {})
        if save_cfg.get("raw_fif", True):
            raw_path = part_dir / "raw_original.fif"
            raw.save(raw_path, overwrite=True, verbose=False)
            output_files["raw_fif"] = raw_path
            _write_sidecar(raw_path, subject, recording, config_path, root, input_files, warnings, manual_review_files=review_file_paths)
        if save_cfg.get("filtered_raw_fif", True):
            filt_path = part_dir / "filtered_raw.fif"
            filtered_raw.save(filt_path, overwrite=True, verbose=False)
            output_files["filtered_raw_fif"] = filt_path
            _write_sidecar(filt_path, subject, recording, config_path, root, input_files, warnings, manual_review_files=review_file_paths)
        del filtered_raw
        gc.collect()

        report = generate_gate3a_markdown_report(
            subject=subject,
            recording_part=recording.recording_part,
            input_files=input_files,
            output_files=output_files,
            raw_qc_summary=raw_summary,
            channel_qc=channel_qc,
            bad_channel_decisions=bad_decisions,
            bad_channel_spatial_warnings=bad_channel_warnings,
            preprocessing_decisions=decisions,
            warnings=warnings,
            failures=failures,
            manual_review_files=review_file_paths,
        )
        report_path = write_markdown_report(qc_dir / "report.md", report)
        output_files["markdown_report"] = report_path
        _write_sidecar(report_path, subject, recording, config_path, root, input_files, warnings, manual_review_files=review_file_paths)
        status = "ok" if not failures else "failed"
    except Exception as error:
        failures.append(str(error))
        failure_path = qc_dir / "failure.csv"
        status = "failed_gate3a_filter_memory" if _is_gate3a_filter_memory_failure(error, stage_context) else f"failed_{stage_context}"
        pd.DataFrame(
            [
                {
                    "subject": subject,
                    "recording_part": recording.recording_part,
                    "status": status,
                    "failure_stage": stage_context,
                    "failure": str(error),
                }
            ]
        ).to_csv(failure_path, index=False)
        output_files["failure_record"] = failure_path
        _write_sidecar(failure_path, subject, recording, config_path, root, input_files, warnings + failures, manual_review_files=review_file_paths)
    finally:
        gc.collect()
        if temporary_owner is not None:
            temporary_owner.cleanup(warnings)

    return PreprocessingPartResult(
        subject=subject,
        recording_part=recording.recording_part,
        status=status,
        output_files=output_files,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )


def deterministic_preprocessing_decisions(
    config: Mapping[str, Any],
    *,
    warnings: list[str] | None = None,
    manual_overrides: Mapping[str, Any] | None = None,
    raw=None,
    memory_settings: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return deterministic filtering/reference decisions for Gate 3A.

    Recognition image epochs remain configured as -0.2 to 1.2 s so later gates
    can examine late positivity, slow waves, and exploratory effects without
    changing preprocessing outputs.
    """

    prep = dict(config.get("preprocessing", {}) or {})
    notch_cfg = dict(prep.get("notch_filter", {}) or {})
    ref_cfg = dict(prep.get("reference", {}) or {})
    memory = dict(memory_settings or gate3a_memory_settings(config))
    rows = [
        {
            "decision": "gate3a_status",
            "value": "running",
            "reason": "Stage status is finalised by the entry script after Gate 3A completes.",
        },
        {
            "decision": "analysis_highpass_hz",
            "value": prep.get("analysis_highpass_hz", 0.1),
            "reason": "Gate 3A analysis-filtered raw output",
        },
        {
            "decision": "analysis_lowpass_hz",
            "value": prep.get("analysis_lowpass_hz", 30.0),
            "reason": "Gate 3A analysis-filtered raw output",
        },
        {
            "decision": "notch_filter_applied",
            "value": bool(notch_cfg.get("enabled", False)),
            "reason": "Notch is disabled by default unless explicitly configured or later justified by PSD QC.",
        },
        {
            "decision": "average_reference_applied",
            "value": bool(ref_cfg.get("apply_average_reference_gate3a", False)),
            "reason": "Average reference is conservative/deferred unless explicitly enabled before ICA.",
        },
        {
            "decision": "provisional_reference_channel",
            "value": ref_cfg.get("provisional_reference_channel", "CPz") if ref_cfg.get("add_provisional_cpz", False) else "",
            "reason": "Acquisition reference channel is metadata-only in Gate 3A and is reconstructed in Gate 3C after ICA application.",
        },
        {
            "decision": "save_raw_original_fif",
            "value": bool(dict(prep.get("save", {}) or {}).get("raw_fif", True)),
            "reason": "Raw FIF persistence is configurable to manage storage.",
        },
        {
            "decision": "save_raw_filtered_fif",
            "value": bool(dict(prep.get("save", {}) or {}).get("filtered_raw_fif", True)),
            "reason": "Filtered raw FIF is normally needed by ICA/epoching.",
        },
        {
            "decision": "mne_preload_mode",
            "value": memory.get("mne_preload_mode"),
            "reason": "Gate 3A raw loading memory mode.",
        },
        {
            "decision": "use_memmap_for_filtering",
            "value": memory.get("use_memmap_for_filtering"),
            "reason": "Use file-backed preload for full-record filtering when configured.",
        },
        {
            "decision": "memmap_dir",
            "value": memory.get("memmap_dir"),
            "reason": "Directory for temporary MNE memmap preload files.",
        },
    ]
    if raw is not None:
        n_channels = len(getattr(raw, "ch_names", []) or [])
        n_samples = int(getattr(raw, "n_times", 0) or 0)
        rows.extend(
            [
                {"decision": "n_channels", "value": n_channels, "reason": "Loaded raw channel count for memory logging."},
                {"decision": "n_samples", "value": n_samples, "reason": "Loaded raw sample count for memory logging."},
                {"decision": "estimated_float64_recording_mib", "value": round(n_channels * n_samples * 8 / (1024**2), 2), "reason": "Approximate dense float64 recording size avoided by default QC path."},
            ]
        )
    for warning in warnings or []:
        rows.append({"decision": "warning", "value": warning, "reason": "Gate 3A QC warning"})
    for key, value in dict(manual_overrides or {}).items():
        rows.append({"decision": f"manual_override_{key}", "value": value, "reason": "Manual override recorded for this part"})
    return pd.DataFrame(rows)


def raw_fif_output_paths(base_dir: str | Path, subject: str, recording_part: str) -> dict[str, Path]:
    """Return explicit raw FIF output names for one recording part."""

    directory = Path(base_dir)
    return {
        "raw_fif": directory / f"{subject}_{recording_part}_raw_original.fif",
        "filtered_raw_fif": directory / f"{subject}_{recording_part}_raw_filtered.fif",
    }


def write_gate3a_qc_figures(
    *,
    raw,
    channel_qc: pd.DataFrame,
    bad_channel_decisions: pd.DataFrame | None = None,
    psd_summary: pd.DataFrame,
    subject: str,
    recording: RecordingPart,
    fig_dir: Path,
    root: Path | None,
    config_path: str | Path | None,
    input_files: list[Path],
    warnings: list[str],
    config: Mapping[str, Any],
) -> dict[str, Path]:
    """Generate reusable Gate 3A QC figures, recording warnings on failure."""

    outputs: dict[str, Path] = {}
    figure_paths = gate3a_figure_paths(fig_dir, subject, recording.recording_part)
    qc_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("raw_qc", {}) or {})
    figure_builders = {
        "psd_figure": lambda: plot_psd(
            raw,
            title=f"{subject} {recording.recording_part} PSD",
            fmin=float(qc_cfg.get("psd_plot_fmin_hz", 1.0)),
            fmax=float(qc_cfg.get("psd_plot_fmax_hz", 80.0)),
            start=float(qc_cfg.get("psd_start_s", 0.0)),
            duration=float(qc_cfg.get("psd_duration_s", 120.0)) if qc_cfg.get("psd_duration_s", 120.0) is not None else None,
        ),
        "raw_overview_figure": lambda: plot_raw_overview(
            raw,
            title=f"{subject} {recording.recording_part} raw overview",
            duration=float(qc_cfg.get("raw_overview_duration_s", 20.0)),
            start=float(qc_cfg.get("raw_overview_start_s", 0.0)),
            eeg_scaling_uv=float(qc_cfg.get("raw_overview_eeg_scaling_uv", 100.0)),
        ),
        "channel_variance_figure": lambda: plot_channel_variance(channel_qc, title=f"{subject} {recording.recording_part} channel variance"),
        "line_noise_figure": lambda: plot_line_noise(psd_summary, title=f"{subject} {recording.recording_part} line noise"),
        "montage_figure": lambda: plot_montage(raw, title=f"{subject} {recording.recording_part} montage"),
        "bad_channel_figure": lambda: plot_bad_channel_summary(
            channel_qc,
            bad_channel_decisions=bad_channel_decisions,
            title=f"{subject} {recording.recording_part} EEG channel SD with proposal status",
        ),
        "correlation_qc_figure": lambda: plot_correlation_qc_summary(
            channel_qc,
            title=f"{subject} {recording.recording_part} poor-correlation proposal evidence",
        ),
    }
    figure_cfg = dict(qc_cfg.get("figures", {}) or {})
    fail_on_error = bool(figure_cfg.get("fail_pipeline_on_figure_error", False))
    for label, builder in figure_builders.items():
        try:
            fig = builder()
            outputs[label] = save_figure_with_provenance(
                fig,
                figure_paths[label],
                subject=subject,
                recording_part=recording.recording_part,
                root=root,
                config_path=config_path,
                input_files=input_files,
                parameters_summary={
                    "figure_type": label,
                    "brainvision_file_base": recording.file_base,
                    "raw_overview_channels": "eeg_only" if label == "raw_overview_figure" else None,
                    "raw_overview_eeg_scaling_uv": float(qc_cfg.get("raw_overview_eeg_scaling_uv", 100.0)) if label == "raw_overview_figure" else None,
                    "raw_overview_remove_dc": True if label == "raw_overview_figure" else None,
                    "raw_overview_display_clipping": "none" if label == "raw_overview_figure" else None,
                },
                warnings=warnings,
            )
        except Exception as error:
            warnings.append(f"Could not generate {label}: {error}")
            if fail_on_error:
                raise RuntimeError(f"Gate 3A QC figure generation failed for {label}: {error}") from error
    return outputs


def apply_gate3a_filtering(
    raw,
    config: Mapping[str, Any],
    decisions: pd.DataFrame,
    *,
    raw_path: str | Path | None = None,
    memmap_dir: Path | None = None,
    warnings: list[str] | None = None,
    alignment: pd.DataFrame | None = None,
    recording_part: str = "",
    subject: str = "",
    metadata_rows: list[dict[str, Any]] | None = None,
    temporary_owner: TemporaryResourceOwner | None = None,
):
    """Apply configured analysis filter after task crop/channel picking."""

    prep = dict(config.get("preprocessing", {}) or {})
    memory = gate3a_memory_settings(config)
    metadata_rows = metadata_rows if metadata_rows is not None else []
    try:
        if not raw_path:
            raise RuntimeError("Gate 3A filtering requires raw_path so crop/pick can be materialised safely.")
        if not memory["use_memmap_for_filtering"]:
            raise RuntimeError("Gate 3A filtering requires preprocessing.memory.use_memmap_for_filtering for disk-backed preload.")
        if _task_crop_enabled_for_gate3a(config) and (alignment is None or alignment.empty):
            raise RuntimeError("Gate 3A task-cropped filtering requires the Stage 2 alignment table.")

        import mne

        if temporary_owner is None:
            raise RuntimeError("Gate 3A filtering requires explicit run-scoped temporary-resource ownership.")
        crop_fif = temporary_owner.allocate("task_crop_input_raw.fif")
        preload_file = temporary_owner.allocate("filtering_memmap.dat")

        filtered_input = raw.copy()
        crop_window = None
        if _task_crop_enabled_for_gate3a(config):
            sfreq = float(filtered_input.info["sfreq"])
            if bool(long_gap_settings(config)["enabled"]):
                segments = calculate_task_segments(alignment, recording_part, sfreq=sfreq, n_times=int(filtered_input.n_times), config=config, first_samp=int(filtered_input.first_samp))
                segment_table = task_segments_table(segments)
                filtered_input = apply_task_segments(filtered_input, segments)
                metadata_rows.append({"decision": "gate3a_task_segment_count", "value": len(segments), "reason": "Long-gap task segments used before Gate 3A filtering."})
                metadata_rows.append({"decision": "gate3a_task_segment_total_duration_s", "value": float(segment_table["duration_s"].sum()) if not segment_table.empty else 0.0, "reason": "Total retained task-segment duration after excluding long idle gaps."})
                if warnings is not None:
                    warnings.append(
                        "Gate 3A filtering uses long-gap task segments before analysis filtering: "
                        f"{len(segments)} segments, total duration {float(segment_table['duration_s'].sum()) if not segment_table.empty else 0.0} s."
                    )
            else:
                crop_window = calculate_task_crop_window(
                    alignment,
                    recording_part,
                    sfreq=sfreq,
                    n_times=int(filtered_input.n_times),
                    config=config,
                    stage="ica",
                    first_samp=int(filtered_input.first_samp),
                )
                crop_summary = task_crop_summary(crop_window, subject=subject, recording_part=recording_part, original_duration_s=float(filtered_input.times[-1]) if len(filtered_input.times) else 0.0)
                crop_debug = task_crop_debug_table(
                    alignment, recording_part, crop_window, config=config
                )
                filtered_input = apply_task_crop(filtered_input, crop_window)
                _metadata_from_table(metadata_rows, "gate3a_task_crop", crop_summary)
                metadata_rows.append({"decision": "gate3a_task_crop_debug_rows", "value": len(crop_debug), "reason": "Retained image-onset rows used to define the Gate 3A filtering crop."})
                if warnings is not None:
                    row = crop_summary.iloc[0].to_dict()
                    warnings.append(
                        "Gate 3A filtering uses task-cropped raw before analysis filtering: "
                        f"samples {row.get('crop_start_sample')}-{row.get('crop_end_sample')}, duration {row.get('cropped_duration_s')} s."
                    )

        keep_channels = gate3a_filter_channel_names(filtered_input, config)
        dropped_channels = [ch for ch in filtered_input.ch_names if ch not in keep_channels]
        filtered_input.pick(keep_channels)
        filtered_input.info["bads"] = [ch for ch in raw.info.get("bads", []) if ch in filtered_input.ch_names]
        filtered_input.save(crop_fif, overwrite=True, verbose=False)
        filtered = mne.io.read_raw_fif(crop_fif, preload=str(preload_file), verbose=False)
        filtered.info["bads"] = list(filtered_input.info.get("bads", []) or [])

        metadata_rows.extend(
            gate3a_filter_memory_decisions(
                filtered,
                memmap_path=preload_file,
                used_memmap=True,
                dropped_channels=dropped_channels,
                crop_window=crop_window,
            )
        )
        if warnings is not None:
            warnings.append(f"Gate 3A filtering used cropped/picked temporary FIF: {crop_fif}")
            warnings.append(f"Gate 3A filtering used memmap preload file: {preload_file}")
            if dropped_channels:
                warnings.append(f"Gate 3A filtered raw dropped unused/non-analysis channels before filtering: {', '.join(dropped_channels)}")

        notch_cfg = dict(prep.get("notch_filter", {}) or {})
        if notch_cfg.get("enabled", False):
            filtered.notch_filter(freqs=notch_cfg.get("frequencies_hz", [prep.get("line_frequency_hz", 50.0)]), picks="data", verbose=False)
        filtered.filter(
            l_freq=prep.get("analysis_highpass_hz", 0.1),
            h_freq=prep.get("analysis_lowpass_hz", 30.0),
            picks="data",
            verbose=False,
        )
        ref_cfg = dict(prep.get("reference", {}) or {})
        if ref_cfg.get("apply_average_reference_gate3a", False):
            raise ValueError("Average reference must remain disabled in Gate 3A; it is applied in Gate 3C after ICA and CPz reconstruction.")
        return filtered
    except MemoryError as error:
        raise Gate3AFilterMemoryError(f"Gate 3A filtering memory failure after task crop/channel picking: {error}") from error
    except Exception as error:
        if "unable to allocate" in str(error).lower():
            raise Gate3AFilterMemoryError(f"Gate 3A filtering memory failure after task crop/channel picking: {error}") from error
        raise


def gate3a_memory_settings(config: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    prep = dict(config.get("preprocessing", {}) or {})
    memory = dict(prep.get("memory", {}) or {})
    memmap_dir = Path(memory.get("memmap_dir", "workspace/kawaii/session2/tmp/memmap"))
    if root is not None and not memmap_dir.is_absolute():
        memmap_dir = root / memmap_dir
    return {
        "mne_preload_mode": memory.get("mne_preload_mode", False),
        "use_memmap_for_filtering": bool(memory.get("use_memmap_for_filtering", True)),
        "memmap_dir": str(memmap_dir),
        "cleanup_memmap_after_success": bool(memory.get("cleanup_memmap_after_success", True)),
    }


def gate3a_filter_channel_names(raw, config: Mapping[str, Any]) -> list[str]:
    """Return channels carried into the analysis-filtered Gate 3A raw."""

    channels_cfg = dict(dict(config.get("channels", {}) or {}).get("eeg", {}) or {})
    unused = {str(ch) for ch in channels_cfg.get("unused", []) or []}
    misc = {str(ch) for ch in channels_cfg.get("misc", []) or []}
    eog_keep = {str(ch) for ch in channels_cfg.get("eog", []) or []}
    ref_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("reference", {}) or {})
    provisional = str(ref_cfg.get("provisional_reference_channel", "CPz"))
    keep = []
    for channel in raw.ch_names:
        if channel in unused or channel in misc:
            continue
        kind = raw.get_channel_types(picks=[channel])[0]
        if kind == "eeg" or (kind == "eog" and channel in eog_keep) or channel == provisional:
            keep.append(channel)
    return keep


def gate3a_filter_memory_decisions(raw, *, memmap_path: Path, used_memmap: bool, dropped_channels: list[str], crop_window=None) -> list[dict[str, Any]]:
    """Return processing-decision rows documenting Gate 3A filter memory use."""

    n_channels = len(getattr(raw, "ch_names", []) or [])
    n_samples = int(getattr(raw, "n_times", 0) or 0)
    rows = [
        {"decision": "gate3a_filter_memmap_used", "value": bool(used_memmap), "reason": "Disk-backed preload used for Gate 3A analysis filtering."},
        {"decision": "gate3a_filter_memmap_path", "value": str(memmap_path), "reason": "Temporary disk-backed preload path for filtering."},
        {"decision": "gate3a_filter_channel_count_after_picking", "value": n_channels, "reason": "Channel count after dropping unused/non-analysis channels before filtering."},
        {"decision": "gate3a_filter_sample_count_after_crop", "value": n_samples, "reason": "Sample count after task crop before filtering."},
        {"decision": "gate3a_filter_estimated_dense_ram_mib", "value": round(n_channels * n_samples * 8 / (1024**2), 2), "reason": "Estimated dense float64 size of the filtered analysis raw."},
        {"decision": "gate3a_filter_dropped_channels", "value": dropped_channels, "reason": "Unused/non-analysis channels omitted from filtered raw but retained in raw QC documentation."},
        {"decision": "gate3a_filtered_raw_scope", "value": "task_cropped_channel_picked_analysis_raw", "reason": "Complementary raw-QC metrics use distributed full-recording windows; PREP-style correlation QC and the filtered analysis raw use alignment-derived task scope."},
    ]
    if crop_window is not None:
        rows.extend(
            [
                {"decision": "gate3a_filter_crop_start_sample", "value": crop_window.crop_start_sample, "reason": "Original sample coordinate for task crop start."},
                {"decision": "gate3a_filter_crop_end_sample", "value": crop_window.crop_end_sample, "reason": "Original sample coordinate for task crop end."},
                {"decision": "gate3a_filter_crop_duration_s", "value": round(crop_window.crop_end_s - crop_window.crop_start_s, 6), "reason": "Duration of task-cropped data passed to Gate 3A filtering."},
            ]
        )
    return rows


def _metadata_from_table(rows: list[dict[str, Any]], prefix: str, table: pd.DataFrame) -> None:
    if table.empty:
        return
    for key, value in table.iloc[0].to_dict().items():
        rows.append({"decision": f"{prefix}_{key}", "value": value, "reason": "Gate 3A task crop metadata for analysis-filtered raw."})


def _append_decision_rows(decisions: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return decisions
    return pd.concat([decisions, pd.DataFrame(rows)], ignore_index=True)


def _task_crop_enabled_for_gate3a(config: Mapping[str, Any]) -> bool:
    settings = task_crop_settings(config)
    return bool(settings.get("enabled_for_ica", True) or settings.get("enabled_for_epoching", True))


def _is_gate3a_filter_memory_failure(error: Exception, stage_context: str) -> bool:
    return isinstance(error, Gate3AFilterMemoryError) or (stage_context == "filtering" and "unable to allocate" in str(error).lower())


def manual_overrides_for_part(config: Mapping[str, Any], subject: str, recording_part: str) -> dict[str, Any]:
    """Return manual overrides relevant to one participant recording part."""

    overrides = dict(config.get("manual_overrides", {}) or {})
    result: dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, MappingABC):
            subject_value = value.get(subject, {})
            if isinstance(subject_value, MappingABC):
                result[key] = subject_value.get(recording_part, subject_value.get("all", {}))
            else:
                result[key] = subject_value
        else:
            result[key] = value
    return result


def _alignment_for_part(alignment: pd.DataFrame | None, recording_part: str) -> pd.DataFrame | None:
    if alignment is None or alignment.empty or "recording_part" not in alignment:
        return alignment
    return alignment[alignment["recording_part"].astype(str).eq(str(recording_part))].copy()


def bad_channel_correlation_sample_segments(
    raw,
    alignment: pd.DataFrame | None,
    recording_part: str,
    config: Mapping[str, Any],
) -> list[tuple[int, int]]:
    """Resolve the configured correlation-QC scope in raw-relative samples."""

    qc_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("raw_qc", {}) or {})
    scope = str(qc_cfg.get("poor_correlation_data_scope", "task_segments"))
    if scope == "full_recording":
        return [(0, int(raw.n_times))]
    if scope != "task_segments":
        raise ValueError(f"Unsupported poor-correlation data scope: {scope}")
    if alignment is None or alignment.empty:
        raise ValueError("Task-segment correlation QC requires the current Stage 2 alignment table.")
    segments = calculate_task_segments(
        alignment,
        recording_part,
        sfreq=float(raw.info["sfreq"]),
        n_times=int(raw.n_times),
        config=config,
        first_samp=int(raw.first_samp),
    )
    if not segments:
        raise ValueError("Task-segment correlation QC found no retained task segments.")
    return [
        (
            int(segment.start_sample - int(raw.first_samp)),
            int(segment.end_sample - int(raw.first_samp) + 1),
        )
        for segment in segments
    ]


def _write_sidecar(
    output_file: Path,
    subject: str,
    recording: RecordingPart,
    config_path: str | Path | None,
    root: Path | None,
    input_files: list[Path],
    warnings: list[str],
    manual_review_files: Mapping[str, Path] | None = None,
) -> None:
    write_provenance_sidecar(
        output_file,
        pipeline_stage="gate3a_preprocessing",
        subject=subject,
        recording_part=recording.recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary={
            "brainvision_file_base": recording.file_base,
            "preprocessing_scope": "raw_loading_channel_qc_analysis_filter_only",
            "gate3a_raw_qc_scope": "distributed_raw_metrics_plus_task_segment_prep_style_correlation_qc",
            "gate3a_filtered_raw_scope": "task_cropped_channel_picked_analysis_raw_when_task_crop_enabled",
            "ica": "not_run",
            "epoching": "not_run",
            "manual_review_files": {key: str(value) for key, value in dict(manual_review_files or {}).items()},
        },
        warnings=warnings,
    )
