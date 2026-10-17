from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from .paths import resolve_session2_eeg_paths


@dataclass(frozen=True)
class MetadataResult:
    """Trial metadata recovered from behavioural logs or schedules."""

    dataframe: pd.DataFrame
    metadata_source: str
    behavioural_log_available: bool
    source_path: Path | None


def metadata_part_for_recording(subject: str, recording_part: str, file_base: str | None = None) -> str:
    """Infer the behavioural-log subject/part name for a recording part.

    Examples
    --------
    ``pilot04`` + file base ``kawaii0004_2`` maps to ``pilot04_2`` when no
    explicit per-part metadata configuration is present.
    """

    suffix = _numeric_suffix(file_base or recording_part)
    if suffix and subject.endswith(f"_{suffix}"):
        return subject
    return f"{subject}_{suffix}" if suffix else subject


def load_metadata_for_recording_part(
    subject: str,
    recording_part: str,
    config: Mapping[str, Any],
    *,
    file_base: str | None = None,
    root: Path | None = None,
) -> MetadataResult:
    """Load trial metadata for one recording part.

    Each participant part may independently configure its behavioural log part,
    schedule source, trigger map label, and metadata source. MATLAB trial logs
    are preferred when enabled and available. If unavailable or disabled, the
    Session 2 recognition schedule is used as a fallback. This module does not
    align triggers by row order; callers should merge using marker-derived trial
    IDs.
    """

    paths = resolve_session2_eeg_paths(config, root=root, create=True)
    source_config = _metadata_source_config(subject, recording_part, config)
    metadata_part = str(source_config.get("behavioural_log_part") or metadata_part_for_recording(subject, recording_part, file_base=file_base))
    behavioural_log_enabled = bool(source_config.get("behavioural_log_enabled", True))
    log_path = paths.session2_logs_dir / f"{metadata_part}_session2" / f"{metadata_part}_session2_trial_log_analysis.csv"
    if behavioural_log_enabled and log_path.exists():
        df = pd.read_csv(log_path)
        return MetadataResult(
            dataframe=_normalise_log_metadata(df, metadata_part, recording_part, "matlab_log"),
            metadata_source="matlab_log",
            behavioural_log_available=True,
            source_path=log_path,
        )

    schedule_subject = str(source_config.get("schedule_subject") or re.sub(r"_\d+$", "", metadata_part))
    schedule_source_label = str(source_config.get("metadata_source_if_schedule", "schedule_plus_eeg_markers"))
    schedule_path = paths.session2_schedules_dir / f"{schedule_subject}_session2_recognition_schedule.csv"
    if schedule_path.exists():
        df = pd.read_csv(schedule_path)
        return MetadataResult(
            dataframe=_normalise_schedule_metadata(df, metadata_part, recording_part, schedule_source_label),
            metadata_source=schedule_source_label,
            behavioural_log_available=False,
            source_path=schedule_path,
        )

    return MetadataResult(
        dataframe=pd.DataFrame(columns=_metadata_columns()),
        metadata_source="markers_only",
        behavioural_log_available=False,
        source_path=None,
    )


def _normalise_log_metadata(df: pd.DataFrame, metadata_part: str, recording_part: str, source: str) -> pd.DataFrame:
    out = df.copy()
    out["trial"] = _validated_trial_ids(out, source)
    out["recording_part"] = recording_part
    out["metadata_part"] = metadata_part
    out["metadata_source"] = source
    out["behavioural_log_available"] = True
    return _select_metadata_columns(out)


def _normalise_schedule_metadata(df: pd.DataFrame, metadata_part: str, recording_part: str, source: str) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "trigger_image_onset": "image_onset_trigger",
        "trigger_kawaii_question": "kawaii_onset_trigger",
        "trigger_arousal_question": "arousal_onset_trigger",
    }
    out = out.rename(columns={old: new for old, new in rename_map.items() if old in out.columns})
    out["trial"] = _validated_trial_ids(out, source)
    out["recording_part"] = recording_part
    out["metadata_part"] = metadata_part
    out["metadata_source"] = source
    out["behavioural_log_available"] = False
    return _select_metadata_columns(out)


def _metadata_source_config(subject: str, recording_part: str, config: Mapping[str, Any]) -> dict[str, Any]:
    recordings = config.get("recordings", {})
    if not isinstance(recordings, Mapping):
        return {}
    participant_recordings = recordings.get("participant_recordings", {})
    if not isinstance(participant_recordings, Mapping):
        return {}
    subject_config = participant_recordings.get(subject, {})
    if not isinstance(subject_config, Mapping):
        return {}
    part_config = subject_config.get(recording_part, {})
    if not isinstance(part_config, Mapping):
        return {}
    return dict(part_config)


def _select_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in _metadata_columns():
        if column not in df.columns:
            df[column] = pd.NA
    result = df[_metadata_columns()].copy()
    if result["trial"].isna().any():
        raise ValueError("Trial metadata contains missing or malformed trial IDs.")
    result["trial"] = result["trial"].astype(int)
    key = ["recording_part", "trial"]
    if result.duplicated(key, keep=False).any():
        duplicates = result.loc[result.duplicated(key, keep=False), key]
        raise ValueError(
            "Trial metadata contains duplicate recording-part/trial keys: "
            + ", ".join(
                f"{row.recording_part}/trial{row.trial}"
                for row in duplicates.drop_duplicates().itertuples(index=False)
            )
        )
    return result


def _validated_trial_ids(df: pd.DataFrame, source: str) -> pd.Series:
    """Return integer trial IDs without silently dropping malformed records."""

    if "trial" not in df.columns:
        raise ValueError(f"{source} trial metadata is missing the trial column.")
    numeric = pd.to_numeric(df["trial"], errors="coerce")
    invalid = numeric.isna() | ~numeric.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    invalid |= numeric.map(lambda value: not float(value).is_integer() if pd.notna(value) else True)
    if invalid.any():
        rows = [str(index) for index in df.index[invalid].tolist()]
        raise ValueError(
            f"{source} trial metadata contains missing or malformed trial IDs at row(s): "
            + ", ".join(rows)
        )
    return numeric.astype("Int64")


def _metadata_columns() -> list[str]:
    return [
        "recording_part",
        "metadata_part",
        "metadata_source",
        "behavioural_log_available",
        "subject_id",
        "session",
        "mode",
        "language",
        "block",
        "trial",
        "trial_in_block",
        "image_filename",
        "image_path",
        "old_new_status",
        "correct_recognition_response",
        "recognition_response",
        "recognition_correct",
        "kawaii_rating",
        "arousal_rating",
        "image_onset_trigger",
        "recognition_response_trigger",
        "kawaii_onset_trigger",
        "kawaii_response_trigger",
        "arousal_onset_trigger",
        "arousal_response_trigger",
        "M_Kawaii",
        "M_Arousal",
        "species",
        "mammal_status",
        "animal_class",
        "source_feedback_type",
        "source_run",
        "source_trial",
    ]


def _numeric_suffix(recording_part: str) -> str:
    match = re.search(r"_(\d+)$", recording_part)
    return match.group(1) if match else ""
