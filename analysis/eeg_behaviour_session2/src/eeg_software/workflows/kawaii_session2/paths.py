from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from eeg_software.project import ensure_directory, get_project_root


@dataclass(frozen=True)
class Session2EEGPaths:
    """Resolved project paths used by the future Session 2 EEG pipeline."""

    project_root: Path
    raw_eeg_dir: Path
    session2_logs_dir: Path
    session2_schedules_dir: Path
    output_dir: Path
    derivatives_dir: Path
    brainvision_working_dir: Path
    alignment_dir: Path
    preprocessing_dir: Path
    ica_dir: Path
    epochs_dir: Path
    raw_qc_dir: Path
    ica_qc_dir: Path
    epoch_qc_dir: Path
    report_dir: Path
    manual_review_dir: Path
    qc_dir: Path


def subject_dir(config: Mapping[str, Any], subject: str, *, root: Path | None = None, create: bool = False) -> Path:
    paths = resolve_session2_eeg_paths(config, root=root, create=create)
    directory = paths.output_dir / "subjects" / subject
    if create:
        ensure_directory(directory)
    return directory


def subject_part_dir(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None, create: bool = False) -> Path:
    directory = subject_dir(config, subject, root=root, create=create) / recording_part
    if create:
        ensure_directory(directory)
    return directory


def subject_part_stage_dir(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    stage: str,
    *,
    root: Path | None = None,
    create: bool = False,
) -> Path:
    directory = subject_part_dir(config, subject, recording_part, root=root, create=create) / stage
    if create:
        ensure_directory(directory)
    return directory


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def resolve_session2_eeg_paths(config: Mapping[str, Any], root: Path | None = None, create: bool = False) -> Session2EEGPaths:
    """Resolve path settings from the Session 2 EEG config.

    TODO: Later pipeline stages can extend this with subject-specific derivative
    and report paths once their on-disk outputs are defined.
    """

    project_root = root or get_project_root()
    path_config = config.get("paths", {})
    if not isinstance(path_config, Mapping):
        path_config = {}

    resolved = Session2EEGPaths(
        project_root=project_root,
        raw_eeg_dir=_resolve_path(path_config.get("raw_eeg_dir", "workspace/kawaii/session2/source/eeg"), project_root),
        session2_logs_dir=_resolve_path(path_config.get("session2_logs_dir", "workspace/kawaii/session2/source/behavioral_logs"), project_root),
        session2_schedules_dir=_resolve_path(path_config.get("session2_schedules_dir", "workspace/kawaii/session2/source/schedules"), project_root),
        output_dir=_resolve_path(path_config.get("output_dir", "workspace/kawaii/session2/analysis"), project_root),
        derivatives_dir=_resolve_path(path_config.get("derivatives_dir", "workspace/kawaii/session2/analysis/derivatives"), project_root),
        brainvision_working_dir=_resolve_path(path_config.get("brainvision_working_dir", "workspace/kawaii/session2/analysis/derivatives/brainvision_working_copies"), project_root),
        alignment_dir=_resolve_path(path_config.get("alignment_dir", "workspace/kawaii/session2/analysis/derivatives/alignment"), project_root),
        preprocessing_dir=_resolve_path(path_config.get("preprocessing_dir", "workspace/kawaii/session2/analysis/derivatives/preprocessing"), project_root),
        ica_dir=_resolve_path(path_config.get("ica_dir", "workspace/kawaii/session2/analysis/derivatives/ica"), project_root),
        epochs_dir=_resolve_path(path_config.get("epochs_dir", "workspace/kawaii/session2/analysis/derivatives/epochs"), project_root),
        raw_qc_dir=_resolve_path(path_config.get("raw_qc_dir", "workspace/kawaii/session2/analysis/qc/raw"), project_root),
        ica_qc_dir=_resolve_path(path_config.get("ica_qc_dir", "workspace/kawaii/session2/analysis/qc/ica"), project_root),
        epoch_qc_dir=_resolve_path(path_config.get("epoch_qc_dir", "workspace/kawaii/session2/analysis/qc/epochs"), project_root),
        report_dir=_resolve_path(path_config.get("report_dir", "workspace/kawaii/session2/analysis/reports"), project_root),
        manual_review_dir=_resolve_path(path_config.get("manual_review_dir", "workspace/kawaii/session2/manual_review"), project_root),
        qc_dir=_resolve_path(path_config.get("qc_dir", "workspace/kawaii/session2/analysis/qc"), project_root),
    )

    if create:
        for path in (
            resolved.output_dir,
            resolved.derivatives_dir,
            resolved.brainvision_working_dir,
            resolved.alignment_dir,
            resolved.preprocessing_dir,
            resolved.ica_dir,
            resolved.epochs_dir,
            resolved.raw_qc_dir,
            resolved.ica_qc_dir,
            resolved.epoch_qc_dir,
            resolved.report_dir,
            resolved.manual_review_dir,
            resolved.qc_dir,
        ):
            ensure_directory(path)

    return resolved
