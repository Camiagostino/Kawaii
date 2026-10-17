from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from eeg_software.project import ensure_directory

from .loading import RecordingPart
from .metadata import MetadataResult, load_metadata_for_recording_part
from .paths import subject_part_stage_dir, subject_dir
from .provenance import write_provenance_sidecar
from .identity import identify_behavioral_inputs, identify_brainvision_dataset
from .resolved_config import canonical_sha256
from .resolved_snapshot import ResolvedRunSnapshotReference, bind_authoritative_config_provenance
from .stage_identity import (
    AlignmentArtifactIdentity,
    DatasetContextIdentity,
    alignment_identity_path,
    build_alignment_artifact_identity,
    build_dataset_context_identity,
    write_alignment_artifact_identity,
)
from .study_definition import KawaiiSession2StudyDefinition, KawaiiTriggerDefinition
from .task_crop import calculate_task_segments, image_onset_gap_summary_table, image_onset_timing_debug_table, task_segments_table


@dataclass(frozen=True)
class AlignmentResult:
    """Trial-level marker alignment plus validation reports."""

    trial_alignment: pd.DataFrame
    coverage_summary: pd.DataFrame
    unexpected_marker_report: pd.DataFrame
    trigger_interval_report: pd.DataFrame
    alignment_csv: Path | None = None
    coverage_csv: Path | None = None
    unexpected_marker_csv: Path | None = None
    trigger_interval_csv: Path | None = None
    alignment_identity_path: Path | None = None
    alignment_identity: AlignmentArtifactIdentity | None = None


def build_trial_alignment(
    subject: str,
    recordings: Sequence[RecordingPart],
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
    write_outputs: bool = True,
) -> AlignmentResult:
    """Build validated trial-level alignment tables from BrainVision markers.

    Trial IDs are reconstructed using the configured Kawaii study rule. No
    alignment step depends on row order in behavioural logs or schedules.
    """

    trigger_definition = KawaiiTriggerDefinition.from_config(config)
    alignment_frames = []
    unexpected_frames = []
    interval_frames = []
    dataset_contexts: list[DatasetContextIdentity] = []
    metadata_input_files: list[Path] = []
    for recording in recordings:
        markers = read_marker_events(recording, trigger_definition=trigger_definition)
        marker_trials, marker_issues = reconstruct_marker_trials(
            recording, markers, trigger_definition=trigger_definition
        )
        metadata = load_metadata_for_recording_part(subject, recording.recording_part, config, file_base=recording.file_base, root=root)
        dataset_contexts.append(
            _dataset_context_for_recording(
                subject,
                recording,
                metadata,
                config,
                resolved_snapshot=resolved_snapshot,
            )
        )
        if metadata.source_path is not None:
            metadata_input_files.append(metadata.source_path)
        aligned = merge_marker_trials_with_metadata(marker_trials, metadata.dataframe, metadata_source=metadata.metadata_source, behavioural_log_available=metadata.behavioural_log_available)
        alignment_frames.append(aligned)
        unexpected_frames.append(build_unexpected_marker_report(recording, markers, marker_issues))
        interval_frames.append(build_trigger_interval_report(recording, markers))

    trial_alignment = pd.concat(alignment_frames, ignore_index=True, sort=False) if alignment_frames else pd.DataFrame()
    trial_alignment = _mark_cross_part_duplicates(trial_alignment)
    trial_alignment = apply_duplicate_trial_policy(trial_alignment)
    coverage_summary = build_coverage_summary(subject, trial_alignment)
    unexpected_marker_report = pd.concat(unexpected_frames, ignore_index=True, sort=False) if unexpected_frames else pd.DataFrame()
    trigger_interval_report = pd.concat(interval_frames, ignore_index=True, sort=False) if interval_frames else pd.DataFrame()

    csv_paths = (None, None, None, None)
    if write_outputs:
        csv_paths = write_alignment_outputs(
            subject,
            trial_alignment,
            coverage_summary,
            unexpected_marker_report,
            trigger_interval_report,
            config,
            recordings=recordings,
            root=root,
            config_path=config_path,
            resolved_snapshot=resolved_snapshot,
            dataset_contexts=dataset_contexts,
            metadata_input_files=metadata_input_files,
        )

    identity_path = alignment_identity_path(csv_paths[0]) if csv_paths[0] is not None else None
    artifact_identity = None
    if identity_path is not None and identity_path.is_file():
        from .stage_identity import read_alignment_artifact_identity

        artifact_identity = read_alignment_artifact_identity(identity_path)

    return AlignmentResult(
        trial_alignment=trial_alignment,
        coverage_summary=coverage_summary,
        unexpected_marker_report=unexpected_marker_report,
        trigger_interval_report=trigger_interval_report,
        alignment_csv=csv_paths[0],
        coverage_csv=csv_paths[1],
        unexpected_marker_csv=csv_paths[2],
        trigger_interval_csv=csv_paths[3],
        alignment_identity_path=identity_path,
        alignment_identity=artifact_identity,
    )


def read_marker_events(
    recording: RecordingPart,
    *,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Read BrainVision marker events from a validated recording part."""

    rows = []
    marker_re = re.compile(r"^Mk(?P<index>\d+)=(?P<body>.*)$")
    code_re = re.compile(r"S\s*(?P<code>\d+)")
    for line in recording.vmrk_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = marker_re.match(line.strip())
        if not match:
            continue
        fields = match.group("body").split(",")
        if len(fields) < 3:
            continue
        marker_type = fields[0].strip()
        description = fields[1].strip()
        try:
            sample = int(float(fields[2].strip()))
        except ValueError:
            sample = pd.NA
        code_match = code_re.search(description)
        code = int(code_match.group("code")) if code_match else pd.NA
        rows.append(
            {
                "subject": recording.subject,
                "recording_part": recording.recording_part,
                "marker_index": int(match.group("index")),
                "marker_type": marker_type,
                "description": description,
                "code": code,
                "sample": sample,
                "time_s": float(sample) / recording.sampling_rate_hz if pd.notna(sample) else pd.NA,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["subject", "recording_part", "marker_index", "marker_type", "description", "code", "sample", "time_s", "marker_category"])
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    df["marker_category"] = df["code"].apply(definition.category)
    return df.sort_values(["sample", "marker_index"], kind="mergesort").reset_index(drop=True)


def reconstruct_marker_trials(
    recording: RecordingPart,
    markers: pd.DataFrame,
    *,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct trial rows from EEG marker windows, tolerating gaps.

    Image onset markers define trial windows when present. Trial IDs are derived
    only from the kawaii/arousal question trigger pair; when that pair is
    incomplete, the image-onset trial remains visible with an incomplete status.
    """

    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    stim = markers[markers["marker_type"].str.lower().eq("stimulus") & markers["code"].notna()].copy()
    stim = stim.sort_values(["sample", "marker_index"], kind="mergesort").reset_index(drop=True)
    rows = []
    issues = []
    image_positions = stim.index[stim["marker_category"].eq("image_onset")].tolist()
    anchor_positions = image_positions or stim.index[stim["marker_category"].eq("kawaii_question")].tolist()

    for seq, anchor_pos in enumerate(anchor_positions, start=1):
        next_anchor_pos = anchor_positions[seq] if seq < len(anchor_positions) else len(stim)
        window = stim.loc[anchor_pos: next_anchor_pos - 1].copy()
        image = _first_event(window, "image_onset")
        k_event = _first_event(window, "kawaii_question")
        a_event = _first_event(window, "arousal_question")
        recognition_response = _first_event(window, "recognition_response")
        kawaii_response = _first_event(window, "kawaii_response")
        arousal_response = _first_event(window, "arousal_response")

        trial = pd.NA
        if k_event is not None and a_event is not None:
            trial = definition.trial_reconstruction.reconstruct(
                int(k_event["code"]), int(a_event["code"])
            )

        sequence_warnings = []
        if image is None:
            sequence_warnings.append("missing_image_onset")
        if k_event is None:
            sequence_warnings.append("missing_kawaii_question")
        if a_event is None:
            sequence_warnings.append("missing_arousal_question")
        if recognition_response is None:
            sequence_warnings.append("missing_recognition_response")
        if kawaii_response is None:
            sequence_warnings.append("missing_kawaii_response")
        if arousal_response is None:
            sequence_warnings.append("missing_arousal_response")
        selected_events = {
            "image_onset": image,
            "recognition_response": recognition_response,
            "kawaii_question": k_event,
            "kawaii_response": kawaii_response,
            "arousal_question": a_event,
            "arousal_response": arousal_response,
        }
        for category, event in selected_events.items():
            if int(window["marker_category"].eq(category).sum()) > 1:
                sequence_warnings.append(f"ambiguous_{category}")
            if event is not None and pd.isna(event.get("sample")):
                sequence_warnings.append(f"malformed_{category}_sample")
        if sequence_warnings:
            issue_marker = image if image is not None else k_event if k_event is not None else window.iloc[0]
            issues.append(_issue_row(recording, issue_marker, "incomplete_trial", "; ".join(sequence_warnings), trial=None if pd.isna(trial) else int(trial)))

        rows.append(
            {
                "subject": recording.subject,
                "recording_part": recording.recording_part,
                "marker_trial_sequence": seq,
                "trial": trial,
                "eeg_trial_complete": not sequence_warnings,
                "sequence_warning": ";".join(sequence_warnings),
                "image_onset_code": _event_value(image, "code"),
                "image_onset_sample": _event_value(image, "sample"),
                "recognition_response_code": _event_value(recognition_response, "code"),
                "recognition_response_sample": _event_value(recognition_response, "sample"),
                "kawaii_question_code": _event_value(k_event, "code"),
                "kawaii_question_sample": _event_value(k_event, "sample"),
                "kawaii_response_code": _event_value(kawaii_response, "code"),
                "kawaii_response_sample": _event_value(kawaii_response, "sample"),
                "arousal_question_code": _event_value(a_event, "code"),
                "arousal_question_sample": _event_value(a_event, "sample"),
                "arousal_response_code": _event_value(arousal_response, "code"),
                "arousal_response_sample": _event_value(arousal_response, "sample"),
                "trial_start_sample": _event_value(image, "sample", default=_event_value(k_event, "sample")),
                "trial_anchor_sample": _event_value(k_event, "sample", default=_event_value(image, "sample")),
                "trial_end_sample": _event_value(arousal_response, "sample", default=_event_value(a_event, "sample", default=_event_value(image, "sample"))),
            }
        )

    marker_trials = pd.DataFrame(rows)
    if marker_trials.empty:
        marker_trials = pd.DataFrame(columns=["subject", "recording_part", "trial"])
    else:
        trial_present = marker_trials["trial"].notna()
        marker_trials["duplicate_trial_within_part"] = False
        marker_trials.loc[trial_present, "duplicate_trial_within_part"] = marker_trials.loc[trial_present].duplicated(subset=["recording_part", "trial"], keep=False)
        marker_trials["duplicate_trial_index_within_part"] = pd.NA
        marker_trials.loc[trial_present, "duplicate_trial_index_within_part"] = marker_trials.loc[trial_present].groupby(["recording_part", "trial"]).cumcount() + 1
    return marker_trials, pd.DataFrame(issues)


def merge_marker_trials_with_metadata(marker_trials: pd.DataFrame, metadata: pd.DataFrame, *, metadata_source: str, behavioural_log_available: bool) -> pd.DataFrame:
    """Merge marker-derived trials with metadata by trial ID, never by row order."""

    if marker_trials.empty and metadata.empty:
        return pd.DataFrame()
    marker_trials = marker_trials.copy()
    metadata = metadata.copy()
    for frame in (marker_trials, metadata):
        if "trial" in frame.columns:
            frame["trial"] = pd.to_numeric(frame["trial"], errors="coerce").astype("Int64")
        if "recording_part" in frame.columns:
            frame["recording_part"] = frame["recording_part"].astype(str)
    metadata_key = ["recording_part", "trial"]
    if metadata.duplicated(metadata_key, keep=False).any():
        raise ValueError("Trial metadata contains duplicate recording-part/trial keys; alignment is ambiguous.")
    aligned = marker_trials.merge(
        metadata,
        on=metadata_key,
        how="outer",
        suffixes=("", "_metadata"),
        validate="many_to_one",
    )
    aligned["metadata_source"] = _series_or_default(aligned, "metadata_source", metadata_source)
    aligned["behavioural_log_available"] = _series_or_default(aligned, "behavioural_log_available", behavioural_log_available)
    aligned["metadata_matched"] = aligned["metadata_part"].notna() if "metadata_part" in aligned.columns else False
    aligned["marker_trial_present"] = aligned["marker_trial_sequence"].notna() if "marker_trial_sequence" in aligned.columns else False
    aligned["eeg_trial_complete"] = aligned["eeg_trial_complete"].fillna(False) if "eeg_trial_complete" in aligned.columns else False
    aligned["duplicate_trial_within_part"] = aligned["duplicate_trial_within_part"].fillna(False) if "duplicate_trial_within_part" in aligned.columns else False
    aligned["alignment_status"] = "ALIGNED"
    aligned.loc[~aligned["marker_trial_present"] & aligned["metadata_matched"], "alignment_status"] = "MISSING_EEG_TRIAL"
    aligned.loc[aligned["marker_trial_present"] & ~aligned["metadata_matched"], "alignment_status"] = "MARKER_ONLY"
    aligned.loc[aligned["marker_trial_present"] & ~aligned["eeg_trial_complete"], "alignment_status"] = "INCOMPLETE_TRIAL"
    aligned.loc[aligned["duplicate_trial_within_part"], "alignment_status"] = "DUPLICATE_TRIALS"
    return aligned


def apply_duplicate_trial_policy(alignment: pd.DataFrame) -> pd.DataFrame:
    """Annotate duplicate trials and future epoch-retention decisions.

    Duplicate rows are never dropped or heuristically selected. Every candidate
    in an ambiguous trial group remains visible and is excluded from epoching.
    """

    if alignment.empty:
        return alignment
    out = alignment.copy()
    out["is_duplicate_trial"] = _bool_series(out, "duplicate_trial_within_part") | _bool_series(out, "duplicate_trial_across_parts")
    out["duplicate_group_id"] = pd.NA
    trial_present = out["trial"].notna()
    subject_values = out["subject"].fillna("") if "subject" in out else pd.Series("", index=out.index)
    out.loc[out["is_duplicate_trial"] & trial_present, "duplicate_group_id"] = (
        subject_values.loc[out["is_duplicate_trial"] & trial_present]
        + ":trial"
        + out.loc[out["is_duplicate_trial"] & trial_present, "trial"].astype("Int64").astype(str)
    )
    out["duplicate_resolution"] = "not_duplicate"
    out["retain_for_epoching"] = _valid_epoch_candidate(out)
    out["exclusion_reason"] = ""
    out.loc[~out["retain_for_epoching"], "exclusion_reason"] = out.loc[~out["retain_for_epoching"]].apply(_default_exclusion_reason, axis=1)

    duplicate_groups = out.loc[out["is_duplicate_trial"] & out["duplicate_group_id"].notna(), "duplicate_group_id"].dropna().unique()
    for group_id in duplicate_groups:
        group_mask = out["duplicate_group_id"].eq(group_id)
        out.loc[group_mask, "alignment_status"] = "DUPLICATE_TRIALS"
        out.loc[group_mask, "duplicate_resolution"] = "unresolved"
        out.loc[group_mask, "retain_for_epoching"] = False
        out.loc[group_mask, "exclusion_reason"] = "unresolved_duplicate_trial"
    return out


def build_coverage_summary(subject: str, alignment: pd.DataFrame) -> pd.DataFrame:
    """Summarise marker coverage and classify gaps/duplicates."""

    rows = []
    if alignment.empty:
        return pd.DataFrame([{"subject": subject, "recording_part": pd.NA, "coverage_classification": "no_trials"}])
    for recording_part, part_df in alignment.groupby("recording_part", dropna=False):
        marker_present = part_df["marker_trial_present"].fillna(False) if "marker_trial_present" in part_df else pd.Series(False, index=part_df.index)
        observed = set(pd.to_numeric(part_df.loc[marker_present, "trial"], errors="coerce").dropna().astype(int))
        expected = set(pd.to_numeric(part_df.loc[part_df["metadata_matched"], "trial"], errors="coerce").dropna().astype(int))
        if expected:
            min_expected = min(expected)
            max_expected = max(expected)
            full_expected = set(range(min_expected, max_expected + 1))
            missing = sorted(full_expected - observed)
            extra = sorted(observed - full_expected)
        else:
            min_expected = min(observed) if observed else pd.NA
            max_expected = max(observed) if observed else pd.NA
            missing = []
            extra = []
        marker_rows = part_df[marker_present].copy()
        duplicates = int(_bool_series(marker_rows, "is_duplicate_trial").sum())
        incomplete = int((~marker_rows["eeg_trial_complete"]).sum()) if "eeg_trial_complete" in marker_rows else 0
        overlap = int(part_df.get("duplicate_trial_across_parts", pd.Series(False, index=part_df.index)).fillna(False).sum())
        classification = classify_coverage(observed, missing, extra, duplicates, incomplete, bool(expected), overlap=overlap, recording_part=str(recording_part))
        rows.append(
            {
                "subject": subject,
                "recording_part": recording_part,
                "metadata_source": _first_non_null(part_df, "metadata_source"),
                "behavioural_log_available": bool(part_df["behavioural_log_available"].fillna(False).any()),
                "n_trials_observed": len(observed),
                "min_trial_observed": min(observed) if observed else pd.NA,
                "max_trial_observed": max(observed) if observed else pd.NA,
                "min_trial_expected": min_expected,
                "max_trial_expected": max_expected,
                "n_missing_trials": len(missing),
                "missing_trials": " ".join(str(x) for x in missing),
                "n_extra_trials": len(extra),
                "extra_trials": " ".join(str(x) for x in extra),
                "n_duplicate_trial_rows": duplicates,
                "n_retained_for_epoching": int(_bool_series(part_df, "retain_for_epoching").sum()),
                "n_excluded_from_epoching": int((~_bool_series(part_df, "retain_for_epoching")).sum()),
                "exclusion_reasons": _summarise_exclusion_reasons(part_df),
                "n_overlap_trial_rows": overlap,
                "n_incomplete_trial_rows": incomplete,
                "coverage_classification": classification,
            }
        )
    return pd.DataFrame(rows)


def classify_coverage(
    observed: set[int],
    missing: list[int],
    extra: list[int],
    duplicates: int,
    incomplete: int,
    has_metadata: bool,
    *,
    overlap: int = 0,
    recording_part: str = "",
) -> str:
    if not observed:
        return "UNRESOLVED_ALIGNMENT"
    if not has_metadata:
        return "UNRESOLVED_ALIGNMENT"
    if overlap:
        return "OVERLAP"
    if duplicates:
        return "DUPLICATE_TRIALS"
    if incomplete:
        return "INCOMPLETE_TRIAL"
    if extra:
        return "UNRESOLVED_ALIGNMENT"
    if not missing:
        return "RESUMED" if _is_resumed_part(recording_part) else "COMPLETE"
    if observed and missing:
        if max(missing) < min(observed):
            return "LATE_START"
        if min(missing) > max(observed):
            return "EARLY_STOP"
    return "GAP"


def build_unexpected_marker_report(recording: RecordingPart, markers: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    unexpected = markers[(markers["marker_type"].str.lower().eq("stimulus")) & (markers["marker_category"].eq("unexpected"))].copy()
    unexpected["issue_type"] = "unexpected_marker"
    unexpected["message"] = "Stimulus marker code is not part of the configured Session 2 EEG trigger families."
    if issues.empty:
        return unexpected
    return pd.concat([unexpected, issues], ignore_index=True, sort=False)


def build_trigger_interval_report(recording: RecordingPart, markers: pd.DataFrame) -> pd.DataFrame:
    stim = markers[markers["marker_type"].str.lower().eq("stimulus") & markers["sample"].notna()].copy()
    stim = stim.sort_values(["sample", "marker_index"], kind="mergesort").reset_index(drop=True)
    if stim.empty:
        return pd.DataFrame()
    stim["next_code"] = stim["code"].shift(-1)
    stim["next_category"] = stim["marker_category"].shift(-1)
    stim["next_sample"] = stim["sample"].shift(-1)
    stim["interval_s"] = (stim["next_sample"] - stim["sample"]) / recording.sampling_rate_hz
    return stim[
        [
            "subject",
            "recording_part",
            "marker_index",
            "code",
            "marker_category",
            "sample",
            "next_code",
            "next_category",
            "next_sample",
            "interval_s",
        ]
    ].iloc[:-1].copy()


def write_alignment_outputs(
    subject: str,
    trial_alignment: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    unexpected_marker_report: pd.DataFrame,
    trigger_interval_report: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    recordings: Sequence[RecordingPart] | None = None,
    root: Path | None = None,
    config_path: str | Path | None = None,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
    dataset_contexts: Sequence[DatasetContextIdentity] | None = None,
    metadata_input_files: Sequence[Path] | None = None,
) -> tuple[Path, Path, Path, Path]:
    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    if recordings and len(recordings) == 1:
        out_dir = ensure_directory(subject_part_stage_dir(config, subject, recordings[0].recording_part, "alignment", root=root, create=True))
    else:
        out_dir = ensure_directory(subject_dir(config, subject, root=root, create=True) / "alignment")
    alignment_csv = out_dir / "trial_alignment.csv"
    coverage_csv = out_dir / "coverage_summary.csv"
    unexpected_csv = out_dir / "unexpected_marker_report.csv"
    intervals_csv = out_dir / "trigger_interval_report.csv"
    timing_csv = out_dir / "image_onset_timing_debug.csv"
    gap_csv = out_dir / "image_onset_gap_summary.csv"
    segments_csv = out_dir / "task_segments.csv"
    trial_alignment.to_csv(alignment_csv, index=False)
    coverage_summary.to_csv(coverage_csv, index=False)
    unexpected_marker_report.to_csv(unexpected_csv, index=False)
    trigger_interval_report.to_csv(intervals_csv, index=False)
    timing_debug, gap_summary, task_segments = build_alignment_timing_debug_outputs(subject, trial_alignment, config, recordings=recordings or [])
    timing_debug.to_csv(timing_csv, index=False)
    gap_summary.to_csv(gap_csv, index=False)
    task_segments.to_csv(segments_csv, index=False)
    contexts = list(dataset_contexts or [])
    metadata_paths = list(metadata_input_files or [])
    if recordings and not contexts:
        for recording in recordings:
            metadata = load_metadata_for_recording_part(
                subject,
                recording.recording_part,
                config,
                file_base=recording.file_base,
                root=root,
            )
            contexts.append(
                _dataset_context_for_recording(
                    subject,
                    recording,
                    metadata,
                    config,
                    resolved_snapshot=resolved_snapshot,
                )
            )
            if metadata.source_path is not None:
                metadata_paths.append(metadata.source_path)
    input_files = _alignment_input_files(recordings or [], metadata_paths)
    parameters = {
        "trial_reconstruction": KawaiiTriggerDefinition.from_config(config).trial_reconstruction.formula,
        "duplicate_policy": "retain exactly one valid duplicate candidate only when safely resolvable; otherwise exclude all duplicate candidates",
        "missing_eeg_trials": "not invented",
    }
    if contexts:
        artifact_identity = build_alignment_artifact_identity(
            contexts=contexts,
            alignment=trial_alignment,
            output_path=alignment_csv,
        )
        identity_path = write_alignment_artifact_identity(
            artifact_identity,
            alignment_identity_path(alignment_csv),
        )
        parameters = {
            **parameters,
            "alignment_identity_path": str(identity_path),
            "alignment_identity_fingerprint": artifact_identity.fingerprint,
        }
    for output in (alignment_csv, coverage_csv, unexpected_csv, intervals_csv, timing_csv, gap_csv, segments_csv):
        write_provenance_sidecar(
            output,
            pipeline_stage="alignment",
            subject=subject,
            root=root,
            config_path=config_path,
            input_files=input_files,
            parameters_summary=parameters,
            warnings=[],
        )
    return alignment_csv, coverage_csv, unexpected_csv, intervals_csv


def build_alignment_timing_debug_outputs(
    subject: str,
    trial_alignment: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    recordings: Sequence[RecordingPart],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build image-onset timing debug, gap summary, and task segment tables."""

    timing_frames = []
    gap_frames = []
    segment_frames = []
    recordings_by_part = {recording.recording_part: recording for recording in recordings}
    if trial_alignment.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for recording_part, part_df in trial_alignment.groupby("recording_part", dropna=False):
        recording = recordings_by_part.get(str(recording_part))
        sfreq = float(recording.sampling_rate_hz if recording is not None else 500.0)
        timing = image_onset_timing_debug_table(
            part_df, str(recording_part), sfreq=sfreq, config=config
        )
        matlab_duration = _matlab_task_duration_s(part_df)
        gap = image_onset_gap_summary_table(timing, matlab_duration_s=matlab_duration)
        gap["subject"] = subject
        gap["recording_part"] = str(recording_part)
        timing["subject"] = subject
        segments = []
        if recording is not None:
            max_alignment_sample = _max_alignment_sample(part_df)
            n_times = max(_recording_n_times(recording), max_alignment_sample + 1)
            segments = calculate_task_segments(part_df, str(recording_part), sfreq=sfreq, n_times=n_times, config=config, first_samp=0)
        segment_table = task_segments_table(segments)
        if not segment_table.empty:
            segment_table.insert(0, "recording_part", str(recording_part))
            segment_table.insert(0, "subject", subject)
        timing_frames.append(timing)
        gap_frames.append(gap)
        segment_frames.append(segment_table)
    return (
        pd.concat(timing_frames, ignore_index=True, sort=False) if timing_frames else pd.DataFrame(),
        pd.concat(gap_frames, ignore_index=True, sort=False) if gap_frames else pd.DataFrame(),
        pd.concat(segment_frames, ignore_index=True, sort=False) if segment_frames else pd.DataFrame(),
    )


def marker_category(
    code: Any, trigger_definition: KawaiiTriggerDefinition | None = None
) -> str:
    """Compatibility adapter to the Kawaii trigger-definition owner."""

    if pd.isna(code):
        return "non_numeric"
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    return definition.category(code)


def _first_event(df: pd.DataFrame, category: str) -> pd.Series | None:
    matches = df[df["marker_category"].eq(category)]
    return None if matches.empty else matches.iloc[0]


def _last_event(df: pd.DataFrame, category: str) -> pd.Series | None:
    matches = df[df["marker_category"].eq(category)]
    return None if matches.empty else matches.iloc[-1]


def _event_value(event: pd.Series | None, key: str, default: Any = pd.NA) -> Any:
    if event is None:
        return default
    return event.get(key, default)


def _issue_row(recording: RecordingPart, marker: pd.Series, issue_type: str, message: str, trial: int | None = None) -> dict[str, Any]:
    return {
        "subject": recording.subject,
        "recording_part": recording.recording_part,
        "trial": trial,
        "marker_index": marker.get("marker_index", pd.NA),
        "code": marker.get("code", pd.NA),
        "sample": marker.get("sample", pd.NA),
        "marker_category": marker.get("marker_category", pd.NA),
        "issue_type": issue_type,
        "message": message,
    }


def _mark_cross_part_duplicates(alignment: pd.DataFrame) -> pd.DataFrame:
    if alignment.empty or "trial" not in alignment:
        return alignment
    out = alignment.copy()
    marker_present = out["marker_trial_present"].fillna(False) if "marker_trial_present" in out else pd.Series(False, index=out.index)
    out["duplicate_trial_across_parts"] = False
    out.loc[marker_present, "duplicate_trial_across_parts"] = out.loc[marker_present].duplicated(subset=["trial"], keep=False)
    out.loc[out["duplicate_trial_across_parts"], "alignment_status"] = "DUPLICATE_TRIALS"
    return out


def _valid_epoch_candidate(df: pd.DataFrame) -> pd.Series:
    return (
        _bool_series(df, "marker_trial_present")
        & _bool_series(df, "metadata_matched")
        & _bool_series(df, "eeg_trial_complete")
        & df["trial"].notna()
    )


def _default_exclusion_reason(row: pd.Series) -> str:
    if not bool(row.get("marker_trial_present", False)):
        return "missing_eeg_trial"
    if not bool(row.get("metadata_matched", False)):
        return "missing_metadata"
    if not bool(row.get("eeg_trial_complete", False)):
        return "incomplete_trial"
    if pd.isna(row.get("trial")):
        return "missing_trial_id"
    return ""


def _summarise_exclusion_reasons(df: pd.DataFrame) -> str:
    if "exclusion_reason" not in df:
        return ""
    reasons = [str(reason) for reason in df["exclusion_reason"].dropna().unique() if str(reason)]
    return " ".join(sorted(reasons))


def _alignment_input_files(
    recordings: Sequence[RecordingPart], metadata_input_files: Sequence[Path] = ()
) -> list[Path]:
    files: list[Path] = []
    for recording in recordings:
        files.extend([recording.vhdr_path, recording.vmrk_path, recording.eeg_path])
    files.extend(metadata_input_files)
    return list(dict.fromkeys(files))


def _dataset_context_for_recording(
    subject: str,
    recording: RecordingPart,
    metadata: MetadataResult,
    config: Mapping[str, Any],
    *,
    resolved_snapshot: ResolvedRunSnapshotReference | None,
) -> DatasetContextIdentity:
    brainvision = identify_brainvision_dataset(
        recording.vhdr_path,
        recording.vmrk_path,
        recording.eeg_path,
    )
    behavioral_inputs = None
    if metadata.source_path is not None:
        behavioral_inputs = identify_behavioral_inputs(
            behavioral_log_path=metadata.source_path if metadata.behavioural_log_available else None,
            schedule_path=metadata.source_path if not metadata.behavioural_log_available else None,
        )
    resolved_fingerprint, scientific_fingerprint = _config_fingerprints(
        config, resolved_snapshot
    )
    return build_dataset_context_identity(
        participant=subject,
        session="session2",
        recording_part=recording.recording_part,
        brainvision=brainvision,
        behavioral_inputs=behavioral_inputs,
        resolved_config_fingerprint=resolved_fingerprint,
        scientific_fingerprint=scientific_fingerprint,
        resolved_snapshot=resolved_snapshot,
    )


def _config_fingerprints(
    config: Mapping[str, Any],
    resolved_snapshot: ResolvedRunSnapshotReference | None,
) -> tuple[str, str]:
    if resolved_snapshot is not None:
        return (
            resolved_snapshot.resolved_config_fingerprint,
            resolved_snapshot.scientific_fingerprint,
        )
    study = KawaiiSession2StudyDefinition.from_config(config).to_primitive()
    return (
        canonical_sha256({"effective_config": dict(config), "study_definition": study}),
        canonical_sha256(study),
    )


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df:
        return df[column].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def _first_non_null(df: pd.DataFrame, column: str) -> Any:
    if column not in df:
        return pd.NA
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else pd.NA


def _series_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in df:
        return df[column].fillna(default)
    return pd.Series(default, index=df.index)


def _is_resumed_part(recording_part: str) -> bool:
    match = re.fullmatch(r"part(\d+)", str(recording_part))
    if match:
        return int(match.group(1)) > 1
    return bool(re.search(r"_\d+$", str(recording_part)))


def _matlab_task_duration_s(part_df: pd.DataFrame) -> float | None:
    """Return MATLAB/log task duration when absolute timing columns exist."""

    candidate_pairs = [
        ("image_onset_time_s", "image_onset_time_s"),
        ("trial_start_time_s", "trial_end_time_s"),
        ("task_time_s", "task_time_s"),
        ("time_s", "time_s"),
    ]
    for start_col, end_col in candidate_pairs:
        if start_col in part_df and end_col in part_df:
            start = pd.to_numeric(part_df[start_col], errors="coerce").dropna()
            end = pd.to_numeric(part_df[end_col], errors="coerce").dropna()
            if not start.empty and not end.empty:
                duration = float(end.max() - start.min())
                if duration >= 0:
                    return duration
    return None


def _recording_n_times(recording: RecordingPart) -> int:
    """Estimate recording sample count for segment clipping without loading EEG."""

    if recording.channel_names and recording.eeg_path.exists():
        bytes_per_sample = 2
        n_channels = max(1, len(recording.channel_names))
        estimated = int(recording.eeg_path.stat().st_size / (bytes_per_sample * n_channels))
        if estimated > 0:
            return estimated
    return 1


def _max_alignment_sample(part_df: pd.DataFrame) -> int:
    sample_cols = [col for col in part_df.columns if col.endswith("_sample")]
    values = []
    for col in sample_cols:
        values.append(pd.to_numeric(part_df[col], errors="coerce"))
    if not values:
        return 0
    samples = pd.concat(values, ignore_index=True).dropna()
    return int(samples.max()) if not samples.empty else 0
