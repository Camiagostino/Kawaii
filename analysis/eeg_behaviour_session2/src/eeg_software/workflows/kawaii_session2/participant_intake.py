"""Fail-closed standard participant intake and source-exposure registration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from .config import load_session2_eeg_config
from .identity import identify_source_file, source_file_identity_from_primitive
from .loading import read_brainvision_header, read_brainvision_marker_header
from .source_exposure import (
    AMBIGUOUS_RUN4_EXPOSURE,
    DISPOSITION_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    VERIFIED_NEW,
    VERIFIED_OLD,
    apply_source_exposure_disposition,
    build_source_exposure_participant_record,
    participant_manifest_identity_payload,
    source_exposure_artifact_identity,
)
from .subject_selection import normalize_subject_id, subject_sort_key


class ParticipantIntakeError(ValueError):
    """A standard intake cannot establish a safe source history."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ParticipantIntakeSummary:
    subject: str
    source_identity: str
    brainvision_file_base: str
    sampling_rate_hz: float
    channel_count: int
    session1_trials: int
    session1_seen_images: int
    session2_trials: int
    verified_old: int
    verified_new: int
    ambiguous: int
    mismatches: int
    recognition_eligible: int
    continuous_cuteness_old_new_eligible: int
    hit: int
    miss: int
    false_alarm: int
    correct_rejection: int
    recognition_delay_days: int
    protocol_disposition: str
    source_files: tuple[Mapping[str, Any], ...]

    def to_primitive(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "status": "PASS_SOURCE_READY",
            "source_identity": self.source_identity,
            "brainvision_file_base": self.brainvision_file_base,
            "sampling_rate_hz": self.sampling_rate_hz,
            "channel_count": self.channel_count,
            "session1_trials": self.session1_trials,
            "session1_seen_images": self.session1_seen_images,
            "session2_trials": self.session2_trials,
            "verified_old": self.verified_old,
            "verified_new": self.verified_new,
            "ambiguous": self.ambiguous,
            "mismatches": self.mismatches,
            "recognition_eligible": self.recognition_eligible,
            "continuous_cuteness_old_new_eligible": self.continuous_cuteness_old_new_eligible,
            "Hit": self.hit,
            "Miss": self.miss,
            "False Alarm": self.false_alarm,
            "Correct Rejection": self.correct_rejection,
            "recognition_delay_days": self.recognition_delay_days,
            "protocol_disposition": self.protocol_disposition,
            "source_files": [dict(value) for value in self.source_files],
        }


@dataclass(frozen=True)
class ParticipantIntakeResult:
    old_artifact_id: str
    new_artifact_id: str
    total_rows: int
    participants: tuple[ParticipantIntakeSummary, ...]
    manifest_path: Path
    disposition_path: Path
    config_path: Path
    persisted: bool

    def to_primitive(self) -> dict[str, Any]:
        return {
            "status": "PASS_SOURCE_READY",
            "persisted": self.persisted,
            "old_artifact_id": self.old_artifact_id,
            "new_artifact_id": self.new_artifact_id,
            "total_rows": self.total_rows,
            "manifest_path": str(self.manifest_path),
            "disposition_path": str(self.disposition_path),
            "config_path": str(self.config_path),
            "participants": [value.to_primitive() for value in self.participants],
        }

    def render_text(self) -> str:
        lines = [
            "PARTICIPANT INTAKE PASS / SOURCE-READY",
            f"artifact: {self.old_artifact_id} -> {self.new_artifact_id}",
            f"disposition rows: {self.total_rows}",
        ]
        for value in self.participants:
            lines.append(
                f"{value.subject}: old={value.verified_old}, new={value.verified_new}, "
                f"H/M/FA/CR={value.hit}/{value.miss}/{value.false_alarm}/{value.correct_rejection}, "
                f"eligible={value.recognition_eligible}, delay={value.recognition_delay_days} day(s), "
                f"protocol={value.protocol_disposition}"
            )
        return "\n".join(lines)


def run_standard_participant_intake(
    *,
    root: Path,
    config_path: str | Path = "config/session2_eeg.yaml",
    subjects: Sequence[str],
    persist: bool = True,
) -> ParticipantIntakeResult:
    """Validate and register clean standard participants in one transaction."""

    project_root = Path(root).resolve()
    selected_config = Path(config_path)
    if not selected_config.is_absolute():
        selected_config = project_root / selected_config
    config = load_session2_eeg_config(root=project_root, config_path=selected_config)
    clean_subjects = tuple(
        sorted(
            {normalize_subject_id(value, config=config) for value in subjects},
            key=subject_sort_key,
        )
    )
    if not clean_subjects:
        raise ParticipantIntakeError("At least one participant is required.", reason_code="missing_participant")

    settings = dict(config.get("source_exposure_verification", {}) or {})
    manifest_path = _configured_path(settings.get("manifest_path"), project_root)
    manifest = _read_existing_manifest(manifest_path)
    disposition_path = _existing_disposition_path(manifest_path, manifest)
    old_artifact_id = source_exposure_artifact_identity(manifest["participants"])
    if manifest.get("artifact_id") != old_artifact_id:
        raise ParticipantIntakeError(
            "Existing source-exposure manifest has a stale canonical artifact identity.",
            reason_code="stale_expected_artifact_id",
        )
    if str(settings.get("expected_artifact_id", "")) != old_artifact_id:
        raise ParticipantIntakeError(
            "Configured expected_artifact_id does not match the existing canonical manifest.",
            reason_code="stale_expected_artifact_id",
        )
    _verify_existing_table_identity(disposition_path, manifest)
    existing = pd.read_csv(disposition_path, dtype=str, keep_default_na=False)
    existing_semantics = _semantic_rows(existing)
    existing_subjects = {str(item.get("subject")) for item in manifest["participants"]}
    duplicate = sorted(existing_subjects & set(clean_subjects))
    if duplicate:
        raise ParticipantIntakeError(
            f"Participant(s) already registered: {duplicate!r}.",
            reason_code="participant_already_registered",
        )

    records: list[dict[str, Any]] = []
    new_rows: list[pd.DataFrame] = []
    summaries: list[ParticipantIntakeSummary] = []
    for subject in clean_subjects:
        record, rows, summary = _build_standard_participant(project_root, config, subject)
        records.append(record)
        new_rows.append(rows)
        summaries.append(summary)

    participants = sorted(
        [dict(value) for value in manifest["participants"]] + records,
        key=lambda value: subject_sort_key(str(value["subject"])),
    )
    new_artifact_id = source_exposure_artifact_identity(participants)
    combined = pd.concat([existing, *new_rows], ignore_index=True)
    combined["_trial_sort"] = pd.to_numeric(combined["trial"], errors="raise").astype(int)
    combined = (
        combined.sort_values(["subject", "_trial_sort"], kind="stable")
        .drop(columns="_trial_sort")
        .reset_index(drop=True)
    )
    combined["source_exposure_artifact_id"] = new_artifact_id
    if not _semantic_rows(combined.loc[combined["subject"].isin(existing_subjects)]).equals(existing_semantics):
        raise ParticipantIntakeError(
            "Existing participant semantic dispositions changed during append.",
            reason_code="existing_disposition_changed",
        )

    result = ParticipantIntakeResult(
        old_artifact_id=old_artifact_id,
        new_artifact_id=new_artifact_id,
        total_rows=int(len(combined)),
        participants=tuple(summaries),
        manifest_path=manifest_path,
        disposition_path=disposition_path,
        config_path=selected_config,
        persisted=bool(persist),
    )
    if not persist:
        return result

    _persist_and_validate(
        result=result,
        config=config,
        config_text=selected_config.read_text(encoding="utf-8"),
        participants=participants,
        disposition=combined,
        existing_semantics=existing_semantics,
        existing_subjects=existing_subjects,
        project_root=project_root,
    )
    return result


def _build_standard_participant(
    root: Path, config: Mapping[str, Any], subject: str
) -> tuple[dict[str, Any], pd.DataFrame, ParticipantIntakeSummary]:
    session1_root = root / "workspace" / "kawaii" / "session1" / "outputs"
    task_parent = session1_root / "task_logs"
    task_dir = task_parent / f"{subject}_session1"
    related = sorted(path for path in task_parent.glob(f"{subject}*_session1") if path.is_dir())
    if related != [task_dir]:
        raise ParticipantIntakeError(
            f"{subject}: standard intake requires exactly one canonical Session 1 family; found {[path.name for path in related]!r}.",
            reason_code="exception_restart_or_overlap",
        )

    session1 = {
        "execution": task_dir / f"{subject}_session1_matlab_trial_log_analysis.csv",
        "seen": task_dir / f"{subject}_session1_matlab_seen_images.csv",
        "run": task_dir / f"{subject}_session1_matlab_run_log.csv",
        "raw": task_dir / f"{subject}_session1_matlab_task_log_raw.csv",
        "timing": task_dir / f"{subject}_session1_matlab_timing_log.csv",
        "pool": session1_root / "session2_inputs" / f"{subject}_session1_recognition_pool.csv",
        "pool_summary": session1_root / "session2_inputs" / f"{subject}_session1_recognition_pool_summary.txt",
    }
    session2_dir = root / "workspace" / "kawaii" / "session2" / "source" / "behavioral_logs" / f"{subject}_session2"
    session2 = {
        "analysis": session2_dir / f"{subject}_session2_trial_log_analysis.csv",
        "raw": session2_dir / f"{subject}_session2_task_log_raw.csv",
        "status": session2_dir / f"{subject}_session2_status.csv",
        "trigger": session2_dir / f"{subject}_session2_trigger_log.csv",
        "timing": session2_dir / f"{subject}_session2_timing_log.csv",
        "block": session2_dir / f"{subject}_session2_block_log.csv",
        "config": session2_dir / f"{subject}_session2_config_used.m",
    }
    _require_files(subject, [*session1.values(), *session2.values()])

    execution = pd.read_csv(session1["execution"])
    raw1 = pd.read_csv(session1["raw"])
    timing1 = pd.read_csv(session1["timing"])
    run_log = pd.read_csv(session1["run"])
    seen = pd.read_csv(session1["seen"])
    pool = pd.read_csv(session1["pool"])
    _validate_session1(subject, execution, raw1, timing1, run_log, seen, pool)

    status = pd.read_csv(session2["status"])
    analysis = pd.read_csv(session2["analysis"])
    raw2 = pd.read_csv(session2["raw"])
    timing2 = pd.read_csv(session2["timing"])
    trigger = pd.read_csv(session2["trigger"])
    block = pd.read_csv(session2["block"])
    _validate_session2(subject, status, analysis, raw2, timing2, trigger, block, session2["config"])
    _validate_cross_session(subject, seen, pool, analysis)

    brainvision, acquisition_sources = _validate_brainvision(root, config, subject)
    delay, protocol = _derive_protocol(subject, session1, session2)
    expected = _expected_counts(analysis)
    source_specs = [
        (session1["execution"], "session1_execution_log", f"workspace/kawaii/session1/outputs/task_logs/{subject}_session1/{subject}_session1_matlab_trial_log_analysis.csv"),
        (session1["seen"], "session1_seen_images", f"workspace/kawaii/session1/outputs/task_logs/{subject}_session1/{subject}_session1_matlab_seen_images.csv"),
        (session1["pool"], "session1_recognition_pool", f"workspace/kawaii/session1/outputs/session2_inputs/{subject}_session1_recognition_pool.csv"),
        (session2["analysis"], "session2_behavioral_log", f"workspace/kawaii/session2/source/behavioral_logs/{subject}_session2/{subject}_session2_trial_log_analysis.csv"),
    ]
    # Preserve the historical standard-participant source-identity scope above:
    # execution + independently written seen list + recognition pool + Session 2
    # analysis log.  The wider readiness family is still fingerprinted and
    # returned by the intake result, but does not redefine that canonical
    # scientific identity.
    readiness_specs = [
        *source_specs,
        (session1["run"], "session1_run_log", f"workspace/kawaii/session1/outputs/task_logs/{subject}_session1/{subject}_session1_matlab_run_log.csv"),
        (session1["raw"], "session1_raw_log", f"workspace/kawaii/session1/outputs/task_logs/{subject}_session1/{subject}_session1_matlab_task_log_raw.csv"),
        (session1["timing"], "session1_timing_log", f"workspace/kawaii/session1/outputs/task_logs/{subject}_session1/{subject}_session1_matlab_timing_log.csv"),
        (session1["pool_summary"], "session1_recognition_pool_summary", f"workspace/kawaii/session1/outputs/session2_inputs/{subject}_session1_recognition_pool_summary.txt"),
        (session2["raw"], "session2_raw_log", f"workspace/kawaii/session2/source/behavioral_logs/{subject}_session2/{subject}_session2_task_log_raw.csv"),
        (session2["status"], "session2_status", f"workspace/kawaii/session2/source/behavioral_logs/{subject}_session2/{subject}_session2_status.csv"),
        (session2["trigger"], "session2_trigger_log", f"workspace/kawaii/session2/source/behavioral_logs/{subject}_session2/{subject}_session2_trigger_log.csv"),
        (session2["timing"], "session2_timing_log", f"workspace/kawaii/session2/source/behavioral_logs/{subject}_session2/{subject}_session2_timing_log.csv"),
        (session2["block"], "session2_block_log", f"workspace/kawaii/session2/source/behavioral_logs/{subject}_session2/{subject}_session2_block_log.csv"),
        (session2["config"], "session2_config_used", f"workspace/kawaii/session2/source/behavioral_logs/{subject}_session2/{subject}_session2_config_used.m"),
    ]
    source_files = [
        identify_source_file(path, role, logical_path=logical).to_primitive()
        for path, role, logical in source_specs
    ]
    readiness_files = [
        identify_source_file(path, role, logical_path=logical).to_primitive()
        for path, role, logical in readiness_specs
    ]
    record = build_source_exposure_participant_record(
        subject,
        source_files,
        expected,
        protocol=protocol,
    )
    rows = _disposition_rows(analysis, record["source_identity"])
    rows.insert(0, "schema_version", DISPOSITION_SCHEMA_VERSION)
    rows["source_exposure_artifact_id"] = "PENDING_GLOBAL_ARTIFACT_ID"
    summary = ParticipantIntakeSummary(
        subject=subject,
        source_identity=record["source_identity"],
        brainvision_file_base=brainvision["file_base"],
        sampling_rate_hz=brainvision["sampling_rate_hz"],
        channel_count=brainvision["channel_count"],
        session1_trials=len(execution),
        session1_seen_images=len(seen),
        session2_trials=len(analysis),
        verified_old=expected[VERIFIED_OLD],
        verified_new=expected[VERIFIED_NEW],
        ambiguous=expected[AMBIGUOUS_RUN4_EXPOSURE],
        mismatches=0,
        recognition_eligible=expected["recognition_eligible"],
        continuous_cuteness_old_new_eligible=expected["continuous_cuteness_old_new_eligible"],
        hit=expected["Hit"],
        miss=expected["Miss"],
        false_alarm=expected["False Alarm"],
        correct_rejection=expected["Correct Rejection"],
        recognition_delay_days=delay,
        protocol_disposition="STANDARD",
        source_files=tuple([*readiness_files, *acquisition_sources]),
    )
    return record, rows, summary


def _validate_session1(subject, execution, raw, timing, runs, seen, pool) -> None:
    _require_scope(execution, subject, 1, "Session 1 execution")
    _require_exact_trials(execution, 400, f"{subject} Session 1 execution")
    _require_exact_trials(raw, 400, f"{subject} Session 1 raw log")
    _require_exact_trials(timing, 400, f"{subject} Session 1 timing")
    if not execution.fillna("").astype(str).equals(raw.fillna("").astype(str)):
        raise ParticipantIntakeError(
            f"{subject}: Session 1 raw and analysis logs conflict.",
            reason_code="source_evidence_conflict",
        )
    required_runs = {1, 2, 3, 4}
    observed_runs = set(pd.to_numeric(runs.get("run"), errors="coerce").dropna().astype(int))
    trial_runs = set(pd.to_numeric(execution.get("run"), errors="coerce").dropna().astype(int))
    if observed_runs != required_runs or trial_runs != required_runs or len(runs) != 4:
        raise ParticipantIntakeError(
            f"{subject}: missing/unsaved or extra Session 1 run; run log={sorted(observed_runs)!r}, trials={sorted(trial_runs)!r}.",
            reason_code="exception_missing_or_unsaved_run",
        )
    run_counts = execution.groupby("run").size().astype(int).to_dict()
    declared = dict(zip(pd.to_numeric(runs["run"]).astype(int), pd.to_numeric(runs["n_trials"]).astype(int)))
    if run_counts != declared or sum(declared.values()) != 400:
        raise ParticipantIntakeError(
            f"{subject}: Session 1 run counts are incomplete or conflicting.",
            reason_code="exception_missing_or_unsaved_run",
        )
    if seen["image_filename"].astype(str).duplicated().any():
        raise ParticipantIntakeError(f"{subject}: duplicate seen image.", reason_code="source_evidence_conflict")
    feedback = execution.loc[execution["condition"].astype(str).eq("image_feedback")].copy()
    images = feedback["feedback_image"].fillna("").astype(str).str.strip()
    empty = images.eq("")
    if not feedback.loc[empty, "feedback_type"].astype(str).eq("too_slow").all():
        raise ParticipantIntakeError(
            f"{subject}: missing Session 1 exposure is not explicitly a too-slow trial.",
            reason_code="ambiguous_exposure_history",
        )
    nonempty = images.loc[~empty]
    if nonempty.duplicated().any() or set(nonempty) != set(seen["image_filename"].astype(str)):
        raise ParticipantIntakeError(
            f"{subject}: execution and seen-image evidence conflict.",
            reason_code="source_evidence_conflict",
        )
    if len(pool) != 400 or pool["image_filename"].astype(str).duplicated().any():
        raise ParticipantIntakeError(
            f"{subject}: recognition pool is missing or contains duplicate items.",
            reason_code="pool_missing_or_duplicate_item",
        )
    _require_scope(seen, subject, 1, "Session 1 seen-image log")
    _require_scope(pool.rename(columns={"session1": "session"}), subject, 1, "Session 1 recognition pool")


def _validate_session2(subject, status, analysis, raw, timing, trigger, block, config_path) -> None:
    if len(status) != 1:
        raise ParticipantIntakeError(f"{subject}: Session 2 status is not singular.", reason_code="protocol_metadata_ambiguous")
    row = status.iloc[0]
    if str(row.get("task_status", "")).strip().lower() != "finished":
        raise ParticipantIntakeError(f"{subject}: Session 2 did not finish.", reason_code="early_termination")
    if str(row.get("language", "")).strip().lower() != "de" or str(row.get("run_type", "")).strip().lower() != "experiment" or str(row.get("environment", "")).strip().lower() != "eeg":
        raise ParticipantIntakeError(
            f"{subject}: Session 2 protocol metadata is not standard German experiment/eeg.",
            reason_code="protocol_metadata_ambiguous",
        )
    _require_scope(analysis, subject, 2, "Session 2 analysis")
    _require_exact_trials(analysis, 400, f"{subject} Session 2 analysis")
    _require_exact_trials(raw, 400, f"{subject} Session 2 raw log")
    _require_exact_trials(timing, 400, f"{subject} Session 2 timing")
    shared = ["subject_id", "session", "block", "trial", "image_filename", "old_new_status", "recognition_response"]
    if not analysis[shared].fillna("").astype(str).equals(raw[shared].fillna("").astype(str)):
        raise ParticipantIntakeError(
            f"{subject}: Session 2 raw and analysis logs conflict.",
            reason_code="source_evidence_conflict",
        )
    if analysis["image_filename"].astype(str).duplicated().any():
        raise ParticipantIntakeError(f"{subject}: duplicate Session 2 image.", reason_code="session2_duplicate_item")
    block_counts = analysis.groupby("block").size().astype(int).to_dict()
    if block_counts != {value: 50 for value in range(1, 9)}:
        raise ParticipantIntakeError(
            f"{subject}: Session 2 is not eight complete 50-trial blocks: {block_counts!r}.",
            reason_code="early_termination",
        )
    block_pairs = block.groupby(["block", "status"]).size().to_dict()
    expected_pairs = {(value, state): 1 for value in range(1, 9) for state in ("started", "finished")}
    if block_pairs != expected_pairs:
        raise ParticipantIntakeError(f"{subject}: block completion evidence is incomplete.", reason_code="early_termination")
    if len(trigger) != 2400 or not trigger.groupby("trial").size().eq(6).all():
        raise ParticipantIntakeError(f"{subject}: trigger log is not complete at six events per trial.", reason_code="incomplete_source_family")
    config_text = Path(config_path).read_text(encoding="utf-8", errors="replace")
    if not re.search(r"max_total_trials\s*=\s*400\s*;", config_text) or not re.search(r"n_blocks\s*=\s*8\s*;", config_text):
        raise ParticipantIntakeError(f"{subject}: saved Session 2 config is not the standard 400/8 protocol.", reason_code="protocol_metadata_ambiguous")


def _validate_cross_session(subject: str, seen: pd.DataFrame, pool: pd.DataFrame, session2: pd.DataFrame) -> None:
    pool_items = pool["image_filename"].astype(str)
    session2_items = session2["image_filename"].astype(str)
    if set(pool_items) != set(session2_items):
        raise ParticipantIntakeError(
            f"{subject}: recognition-pool and Session 2 item universes differ.",
            reason_code="session2_missing_or_conflicting_item",
        )
    old = pool["old_new_status"].astype(str).str.lower().eq("old")
    shown = pd.to_numeric(pool["was_shown_in_session1"], errors="coerce").eq(1)
    seen_set = set(seen["image_filename"].astype(str))
    if not old.equals(shown) or set(pool.loc[old, "image_filename"].astype(str)) != seen_set:
        raise ParticipantIntakeError(
            f"{subject}: seen-image and recognition-pool exposure evidence conflicts.",
            reason_code="pool_seen_image_mismatch",
        )
    pool_labels = dict(zip(pool_items, pool["old_new_status"].astype(str).str.lower()))
    session2_labels = dict(zip(session2_items, session2["old_new_status"].astype(str).str.lower()))
    if pool_labels != session2_labels:
        raise ParticipantIntakeError(
            f"{subject}: recognition-pool and Session 2 old/new labels conflict.",
            reason_code="source_session2_old_new_mismatch",
        )


def _validate_brainvision(root: Path, config: Mapping[str, Any], subject: str):
    match = re.search(r"(\d+)$", subject)
    if match is None:
        raise ParticipantIntakeError(f"Cannot resolve EEG family for {subject}.", reason_code="incomplete_source_family")
    recordings = dict(config.get("recordings", {}) or {})
    template = str(recordings.get("filename_template", "kawaii{subject_number:04d}"))
    base = template.format(subject=subject, subject_number=int(match.group(1)))
    raw_dir_value = dict(config.get("paths", {}) or {}).get("raw_eeg_dir", "workspace/kawaii/session2/source/eeg")
    raw_dir = Path(str(raw_dir_value))
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    families = sorted(raw_dir.glob(f"{base}*.vhdr"))
    if families != [raw_dir / f"{base}.vhdr"]:
        raise ParticipantIntakeError(
            f"{subject}: standard intake requires one unsuffixed BrainVision family; found {[path.stem for path in families]!r}.",
            reason_code="exception_restart_or_overlap",
        )
    vhdr, vmrk, eeg = (raw_dir / f"{base}{suffix}" for suffix in (".vhdr", ".vmrk", ".eeg"))
    _require_files(subject, [vhdr, vmrk, eeg])
    header = read_brainvision_header(vhdr)
    marker = read_brainvision_marker_header(vmrk)
    if header.common_infos.get("DataFile") != eeg.name or header.common_infos.get("MarkerFile") != vmrk.name or marker.common_infos.get("DataFile") != eeg.name:
        raise ParticipantIntakeError(f"{subject}: BrainVision family references conflict.", reason_code="incomplete_source_family")
    expected_rate = float(recordings.get("expected_sampling_rate_hz", 500.0))
    tolerance = float(recordings.get("sampling_rate_tolerance_hz", 0.001))
    if header.sampling_rate_hz is None or abs(header.sampling_rate_hz - expected_rate) > tolerance:
        raise ParticipantIntakeError(f"{subject}: BrainVision sampling rate is not {expected_rate:g} Hz.", reason_code="acquisition_mismatch")
    declared_channels = int(header.common_infos.get("NumberOfChannels", 0))
    if declared_channels != 64 or len(header.channel_names) != declared_channels:
        raise ParticipantIntakeError(f"{subject}: BrainVision channel inventory is not the expected 64 channels.", reason_code="acquisition_mismatch")
    identities = tuple(
        identify_source_file(path, f"brainvision_{path.suffix[1:]}", logical_path=f"workspace/kawaii/session2/source/eeg/{path.name}").to_primitive()
        for path in (vhdr, vmrk, eeg)
    )
    return {"file_base": base, "sampling_rate_hz": header.sampling_rate_hz, "channel_count": declared_channels}, identities


def _derive_protocol(subject: str, session1: Mapping[str, Path], session2: Mapping[str, Path]):
    first_dates = {_utc_date(path) for path in session1.values()}
    second_dates = {_utc_date(path) for path in session2.values() if path.suffix.lower() != ".m"}
    if len(first_dates) != 1 or len(second_dates) != 1:
        raise ParticipantIntakeError(
            f"{subject}: session dates cannot be inferred consistently from source-family timestamps.",
            reason_code="protocol_metadata_ambiguous",
        )
    first = next(iter(first_dates))
    second = next(iter(second_dates))
    delay = (second - first).days
    if delay != 1:
        raise ParticipantIntakeError(
            f"{subject}: recognition delay is {delay} days, requiring tailored protocol review.",
            reason_code="exception_protocol_deviation",
        )
    protocol = {
        "recognition_delay_days": delay,
        "protocol_deviation": False,
        "protocol_disposition": "STANDARD",
        "recognition_delay_evidence": {
            "session1_utc_date": first.isoformat(),
            "session2_utc_date": second.isoformat(),
            "method": "consistent_source_family_filesystem_utc_dates",
        },
    }
    return delay, protocol


def _disposition_rows(session2: pd.DataFrame, source_identity: str) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "subject": session2["subject_id"].astype(str),
            "session": "session2",
            "trial": pd.to_numeric(session2["trial"]).astype(int).astype(str),
            "image_filename": session2["image_filename"].astype(str),
            "original_old_new_status": session2["old_new_status"].astype(str).str.lower(),
            "recognition_response": session2["recognition_response"].astype(str).str.lower(),
        }
    )
    output["source_exposure_status"] = output["original_old_new_status"].map({"old": VERIFIED_OLD, "new": VERIFIED_NEW})
    output["source_verified_old_new_status"] = output["original_old_new_status"]
    output["recognition_eligible"] = True
    output["continuous_cuteness_old_new_eligible"] = True
    output["exposure_exclusion_reason"] = ""
    output["source_exposure_identity"] = source_identity
    return output[
        [
            "subject", "session", "trial", "image_filename", "original_old_new_status",
            "source_exposure_status", "source_verified_old_new_status", "recognition_eligible",
            "continuous_cuteness_old_new_eligible", "exposure_exclusion_reason",
            "source_exposure_identity", "recognition_response",
        ]
    ]


def _expected_counts(session2: pd.DataFrame) -> dict[str, int]:
    old = session2["old_new_status"].astype(str).str.lower().eq("old")
    new = session2["old_new_status"].astype(str).str.lower().eq("new")
    response = session2["recognition_response"].astype(str).str.lower()
    if not (old | new).all() or not response.isin(["yes", "no"]).all():
        raise ParticipantIntakeError("Malformed old/new or recognition response values.", reason_code="source_evidence_conflict")
    return {
        VERIFIED_OLD: int(old.sum()),
        VERIFIED_NEW: int(new.sum()),
        AMBIGUOUS_RUN4_EXPOSURE: 0,
        "recognition_eligible": int(len(session2)),
        "continuous_cuteness_old_new_eligible": int(len(session2)),
        "Hit": int((old & response.eq("yes")).sum()),
        "Miss": int((old & response.eq("no")).sum()),
        "False Alarm": int((new & response.eq("yes")).sum()),
        "Correct Rejection": int((new & response.eq("no")).sum()),
    }


def _persist_and_validate(*, result, config, config_text, participants, disposition, existing_semantics, existing_subjects, project_root) -> None:
    config_path = result.config_path
    manifest_path = result.manifest_path
    table_path = result.disposition_path
    new_config_text = _updated_config_text(config_text, [value.subject for value in result.participants], result.new_artifact_id)
    with tempfile.TemporaryDirectory(prefix="kawaii-intake-") as directory:
        staging = Path(directory)
        staged_table = staging / table_path.name
        staged_manifest = staging / manifest_path.name
        disposition.to_csv(staged_table, index=False, lineterminator="\n")
        artifact = identify_source_file(staged_table, "source_exposure_disposition_table", logical_path=table_path.name).to_primitive()
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "disposition_schema_version": DISPOSITION_SCHEMA_VERSION,
            "artifact_id": result.new_artifact_id,
            "artifact": artifact,
            "participants": participants,
        }
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validation_config = deepcopy(dict(config))
        validation_settings = dict(validation_config.get("source_exposure_verification", {}) or {})
        validation_settings["required_subjects"] = sorted(existing_subjects | {value.subject for value in result.participants}, key=subject_sort_key)
        validation_settings["manifest_path"] = str(staged_manifest)
        validation_settings["expected_artifact_id"] = result.new_artifact_id
        validation_config["source_exposure_verification"] = validation_settings
        for participant in participants:
            subject = str(participant["subject"])
            scoped = disposition.loc[disposition["subject"].astype(str).eq(subject)]
            metadata = scoped[["trial", "image_filename", "original_old_new_status"]].rename(columns={"original_old_new_status": "old_new_status"})
            metadata.insert(0, "subject_id", subject)
            metadata.insert(1, "session", 2)
            apply_source_exposure_disposition(metadata, subject=subject, session=2, recording_part="part1", config=validation_config, root=project_root)

        backups = {path: path.read_bytes() for path in (table_path, manifest_path, config_path)}
        try:
            _atomic_write_bytes(table_path, staged_table.read_bytes())
            _atomic_write_bytes(manifest_path, staged_manifest.read_bytes())
            _atomic_write_bytes(config_path, new_config_text.encode("utf-8"))
            committed = pd.read_csv(table_path, dtype=str, keep_default_na=False)
            if not _semantic_rows(committed.loc[committed["subject"].isin(existing_subjects)]).equals(existing_semantics):
                raise ParticipantIntakeError("Committed append altered existing dispositions.", reason_code="existing_disposition_changed")
            committed_config = load_session2_eeg_config(root=project_root, config_path=config_path)
            for value in result.participants:
                scoped = committed.loc[committed["subject"].eq(value.subject)]
                metadata = scoped[["trial", "image_filename", "original_old_new_status"]].rename(columns={"original_old_new_status": "old_new_status"})
                metadata.insert(0, "subject_id", value.subject)
                metadata.insert(1, "session", 2)
                apply_source_exposure_disposition(metadata, subject=value.subject, session=2, recording_part="part1", config=committed_config, root=project_root)
        except Exception:
            for path, content in backups.items():
                _atomic_write_bytes(path, content)
            raise


def _updated_config_text(text: str, subjects: Sequence[str], artifact_id: str) -> str:
    match = re.search(r"(?m)^(\s*required_subjects:\s*)\[([^\]]*)\][ \t]*$", text)
    if match is None:
        raise ParticipantIntakeError("Config required_subjects is not in the supported inline form.", reason_code="unsupported_config_layout")
    existing = [value.strip() for value in match.group(2).split(",") if value.strip()]
    required = sorted(set(existing) | set(subjects), key=subject_sort_key)
    text = text[:match.start()] + f"{match.group(1)}[{', '.join(required)}]" + text[match.end():]
    text, count = re.subn(
        r"(?m)^(\s*expected_artifact_id:\s*)sha256:[0-9a-f]{64}[ \t]*$",
        rf"\g<1>{artifact_id}",
        text,
        count=1,
    )
    if count != 1:
        raise ParticipantIntakeError("Config expected_artifact_id could not be updated uniquely.", reason_code="unsupported_config_layout")
    for subject in subjects:
        text = _append_yaml_list_value(text, ("participants", "defaults", "included"), subject)
        text = _append_yaml_list_value(text, ("analysis", "include_subjects"), subject)
    return text


def _append_yaml_list_value(text: str, path: tuple[str, ...], value: str) -> str:
    lines = text.splitlines(keepends=True)
    start = 0
    indent = 0
    for key in path:
        found = None
        pattern = re.compile(r"^" + " " * indent + re.escape(key) + r":\s*(?:#.*)?$")
        for index in range(start, len(lines)):
            stripped = lines[index].rstrip("\r\n")
            current_indent = len(stripped) - len(stripped.lstrip(" "))
            if index > start and current_indent < indent:
                break
            if pattern.fullmatch(stripped):
                found = index
                break
        if found is None:
            raise ParticipantIntakeError(f"Config list path {'.'.join(path)} is absent.", reason_code="unsupported_config_layout")
        start = found + 1
        indent += 2
    end = start
    values = []
    item_pattern = re.compile(r"^" + " " * indent + r"-\s*(\S+)\s*$")
    while end < len(lines):
        stripped = lines[end].rstrip("\r\n")
        if not stripped.strip():
            end += 1
            continue
        current_indent = len(stripped) - len(stripped.lstrip(" "))
        if current_indent < indent:
            break
        item = item_pattern.fullmatch(stripped)
        if item:
            values.append(item.group(1))
        end += 1
    if value in values:
        return text
    newline = "\r\n" if "\r\n" in text else "\n"
    lines.insert(end, " " * indent + f"- {value}" + newline)
    return "".join(lines)


def _read_existing_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParticipantIntakeError(f"Cannot read existing manifest: {error}", reason_code="missing_source_exposure_artifact") from error
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION or value.get("disposition_schema_version") != DISPOSITION_SCHEMA_VERSION or not isinstance(value.get("participants"), list):
        raise ParticipantIntakeError("Existing manifest schema is unsupported.", reason_code="malformed_source_exposure_artifact")
    return value


def _existing_disposition_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    logical = str(dict(manifest.get("artifact", {}) or {}).get("logical_path", ""))
    if not logical:
        raise ParticipantIntakeError("Manifest lacks disposition logical path.", reason_code="malformed_source_exposure_artifact")
    path = Path(logical)
    return path if path.is_absolute() else manifest_path.parent / path


def _verify_existing_table_identity(path: Path, manifest: Mapping[str, Any]) -> None:
    try:
        expected = source_file_identity_from_primitive(manifest.get("artifact"))
        current = identify_source_file(path, expected.role, logical_path=expected.logical_path)
    except Exception as error:
        raise ParticipantIntakeError(f"Existing disposition identity cannot be verified: {error}", reason_code="stale_source_exposure_artifact") from error
    if current.identity_payload() != expected.identity_payload():
        raise ParticipantIntakeError("Existing disposition table is stale.", reason_code="stale_source_exposure_artifact")


def _semantic_rows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in frame.columns if column != "source_exposure_artifact_id"]
    output = frame[columns].copy()
    output["_trial_sort"] = pd.to_numeric(output["trial"], errors="raise").astype(int)
    return (
        output.sort_values(["subject", "_trial_sort"], kind="stable")
        .drop(columns="_trial_sort")
        .reset_index(drop=True)
        .astype(str)
    )


def _require_files(subject: str, paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise ParticipantIntakeError(f"{subject}: incomplete source family; missing {missing!r}.", reason_code="incomplete_source_family")


def _require_scope(frame: pd.DataFrame, subject: str, session: int, label: str) -> None:
    if "subject_id" not in frame or "session" not in frame or set(frame["subject_id"].astype(str)) != {subject} or set(pd.to_numeric(frame["session"], errors="coerce").dropna().astype(int)) != {session}:
        raise ParticipantIntakeError(f"{label} scope is inconsistent.", reason_code="source_scope_mismatch")


def _require_exact_trials(frame: pd.DataFrame, expected: int, label: str) -> None:
    if "trial" not in frame:
        raise ParticipantIntakeError(f"{label} lacks trial identity.", reason_code="incomplete_source_family")
    trials = pd.to_numeric(frame["trial"], errors="coerce")
    if len(frame) != expected or trials.isna().any() or trials.duplicated().any() or set(trials.astype(int)) != set(range(1, expected + 1)):
        raise ParticipantIntakeError(f"{label} is incomplete or has duplicate/conflicting trials.", reason_code="early_termination")


def _configured_path(value: Any, root: Path) -> Path:
    if not value:
        raise ParticipantIntakeError("Source-exposure manifest path is not configured.", reason_code="missing_source_exposure_configuration")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _utc_date(path: Path):
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
