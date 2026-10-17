from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .loading import RecordingPart
from .manual_review_persistence import (
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    ManualReviewConflictError,
    ManualReviewPersistenceError,
    update_text_store,
    update_yaml_store,
    write_yaml_document,
)
from .paths import resolve_session2_eeg_paths

BAD_CHANNELS_FILE = "bad_channels.yaml"
ICA_COMPONENTS_FILE = "ica_components.yaml"
PARTICIPANT_NOTES_FILE = "participant_notes.md"
ICA_REVIEW_STATUSES = frozenset({"not_reviewed", "in_progress", "complete"})
BAD_CHANNEL_REVIEW_STATUSES = frozenset({"not_reviewed", "in_progress", "complete"})
BAD_CHANNEL_DECISION_SCHEMA_VERSION = 1
ICA_DECISION_SCHEMA_VERSION = 1
CANONICAL_SESSION = "session2"
_EXPECTED_CONTEXT_UNSPECIFIED = object()
_EXPECTED_DECISION_UNSPECIFIED = object()


@dataclass(frozen=True)
class BadChannelReviewDecision:
    participant: str
    recording_part: str
    review_status: str
    confirmed_bad_channels: tuple[str, ...]
    source_recording_id: str | None
    review_context_id: str | None
    reviewed_channels: tuple[str, ...]
    reviewer: str
    reviewed_at_utc: str | None
    decision_id: str | None
    decision_schema_version: int
    notes: str
    legacy_ambiguous: bool = False

    @property
    def complete(self) -> bool:
        return self.review_status == "complete"

    @property
    def session(self) -> str:
        return CANONICAL_SESSION

    @property
    def decision_state(self) -> str:
        if self.review_status == "not_reviewed":
            return "NOT_REVIEWED"
        if self.review_status == "in_progress":
            return "IN_PROGRESS"
        return "REVIEWED_WITH_BAD_CHANNELS" if self.confirmed_bad_channels else "REVIEWED_NO_BAD_CHANNELS"

    @property
    def semantic_decision_id(self) -> str | None:
        """Stable scientific decision identity, independent of audit metadata."""

        if not self.complete:
            return None
        return _bad_channel_semantic_decision_identity({
            "decision_schema_version": self.decision_schema_version,
            "participant": self.participant,
            "session": self.session,
            "recording_part": self.recording_part,
            "review_status": self.review_status,
            "source_recording_id": self.source_recording_id,
            "review_context_id": self.review_context_id,
            "reviewed_channels": list(self.reviewed_channels),
            "confirmed_bad_channels": list(self.confirmed_bad_channels),
        })

    @property
    def audit_record_id(self) -> str | None:
        """Persisted v1 record fingerprint, including reviewer/time/notes."""

        return self.decision_id


@dataclass(frozen=True)
class ICAReviewDecision:
    participant: str
    session: str
    recording_part: str
    review_status: str
    confirmed_ica_components: tuple[int, ...]
    ica_solution_id: str | None
    fitted_component_count: int | None
    bad_channel_decision_id: str | None
    reviewer: str
    reviewed_at_utc: str | None
    decision_id: str | None
    audit_record_id: str | None
    decision_schema_version: int
    notes: str
    legacy_ambiguous: bool = False

    @property
    def complete(self) -> bool:
        return self.review_status == "complete" and not self.legacy_ambiguous

    @property
    def decision_state(self) -> str:
        if self.legacy_ambiguous:
            return "LEGACY_UNVERIFIED"
        if self.review_status == "not_reviewed":
            return "NOT_REVIEWED"
        if self.review_status == "in_progress":
            return "IN_PROGRESS"
        return "REVIEWED_WITH_ICA_COMPONENTS" if self.confirmed_ica_components else "REVIEWED_NO_ICA_COMPONENTS"

    @property
    def semantic_decision_id(self) -> str | None:
        return self.decision_id


def manual_review_paths(config: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Path]:
    """Return configured manual-review file paths."""

    paths = resolve_session2_eeg_paths(config, root=root, create=False)
    return {
        "manual_review_dir": paths.manual_review_dir,
        "bad_channels": paths.manual_review_dir / BAD_CHANNELS_FILE,
        "ica_components": paths.manual_review_dir / ICA_COMPONENTS_FILE,
        "participant_notes": paths.manual_review_dir / PARTICIPANT_NOTES_FILE,
    }


def ensure_manual_review_scaffold(
    subject_recordings: Mapping[str, Sequence[RecordingPart] | Sequence[str]],
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Path]:
    """Create/update manual-review scaffold files without overwriting decisions."""

    paths = manual_review_paths(config, root=root)
    paths["manual_review_dir"].mkdir(parents=True, exist_ok=True)
    requested = {
        str(subject): tuple(_part_label(part) for part in recordings)
        for subject, recordings in subject_recordings.items()
    }

    def add_bad_entries(data: dict[str, Any]) -> None:
        for subject, part_labels in requested.items():
            _ensure_bad_channel_entries(data, subject, part_labels)

    def add_ica_entries(data: dict[str, Any]) -> None:
        for subject, part_labels in requested.items():
            _ensure_ica_entries(data, subject, part_labels)

    update_yaml_store(
        paths["bad_channels"],
        mutate=add_bad_entries,
        validate_store=_validate_bad_channel_store,
        validate_committed=lambda data, _result, path: _validate_requested_scaffold(data, requested, path),
        lock_timeout_s=lock_timeout_s,
    )
    update_yaml_store(
        paths["ica_components"],
        mutate=add_ica_entries,
        validate_store=_validate_ica_store,
        validate_committed=lambda data, _result, path: _validate_requested_scaffold(data, requested, path),
        lock_timeout_s=lock_timeout_s,
    )
    _ensure_notes_file(
        paths["participant_notes"], subject_recordings, lock_timeout_s=lock_timeout_s
    )
    return paths


def read_confirmed_bad_channels(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    root: Path | None = None,
    expected_source_recording_id: str | None = None,
    expected_review_context_id: str | None = None,
    expected_reviewed_channels: Sequence[str] | None = None,
    require_complete: bool = False,
) -> list[str]:
    """Return channels only from a validated, completed human decision."""

    decision = read_bad_channel_review_decision(
        config,
        subject,
        recording_part,
        root=root,
        expected_source_recording_id=expected_source_recording_id,
        expected_review_context_id=expected_review_context_id,
        expected_reviewed_channels=expected_reviewed_channels,
        require_complete=require_complete,
    )
    return list(decision.confirmed_bad_channels) if decision.complete else []


def read_bad_channel_review_decision(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    root: Path | None = None,
    expected_source_recording_id: str | None = None,
    expected_review_context_id: str | None = None,
    expected_reviewed_channels: Sequence[str] | None = None,
    require_complete: bool = False,
) -> BadChannelReviewDecision:
    """Read and validate one participant/part/source/context-bound decision."""

    path = manual_review_paths(config, root=root)["bad_channels"]
    data = _load_yaml_mapping(path)
    entry = data.get(subject, {}).get(recording_part)
    decision = _bad_channel_decision_from_entry(entry, subject, recording_part, path)
    if require_complete and not decision.complete:
        legacy = " legacy ambiguous entry" if decision.legacy_ambiguous else ""
        raise ValueError(
            f"Bad-channel review is {decision.review_status!r}, not complete, for {subject} {recording_part};"
            f" Gate 3C application is blocked.{legacy}"
        )
    if decision.complete:
        if expected_source_recording_id is not None and decision.source_recording_id != expected_source_recording_id:
            raise ValueError(
                f"Stale bad-channel review for {subject} {recording_part}: decision source "
                f"{decision.source_recording_id!r} does not match current source {expected_source_recording_id!r}."
            )
        if expected_review_context_id is not None and decision.review_context_id != expected_review_context_id:
            raise ValueError(
                f"Stale bad-channel review context for {subject} {recording_part}: decision context "
                f"{decision.review_context_id!r} does not match current context {expected_review_context_id!r}."
            )
        if expected_reviewed_channels is not None and decision.reviewed_channels != tuple(map(str, expected_reviewed_channels)):
            raise ValueError(
                f"Bad-channel reviewed-channel inventory mismatch for {subject} {recording_part}: "
                f"decision={list(decision.reviewed_channels)!r}, current={list(map(str, expected_reviewed_channels))!r}."
            )
    return decision


def bad_channel_source_identity(recording: RecordingPart) -> str:
    """Fingerprint the complete BrainVision source triplet and recording metadata."""

    digest = hashlib.sha256()
    for value in (
        "kawaii_bad_channel_source_v1",
        recording.subject,
        recording.recording_part,
        recording.file_base,
        f"{float(recording.sampling_rate_hz):.12g}",
        *map(str, recording.channel_names),
    ):
        digest.update(str(value).encode("utf-8") + b"\0")
    for label, path in (("vhdr", recording.vhdr_path), ("vmrk", recording.vmrk_path)):
        resolved = Path(path)
        digest.update(label.encode("ascii") + b"\0" + resolved.name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(resolved.read_bytes()).digest())
    eeg_path = Path(recording.eeg_path)
    digest.update(eeg_path.name.encode("utf-8") + b"\0")
    digest.update(_sha256_file(eeg_path))
    return "sha256:" + digest.hexdigest()


def bad_channel_review_context_identity(
    recording: RecordingPart,
    config: Mapping[str, Any],
    reviewed_channels: Sequence[str],
    *,
    source_recording_id: str | None = None,
) -> str:
    """Bind review to source, EEG inventory, and bad-channel QC context."""

    preprocessing = dict(config.get("preprocessing", {}) or {})
    payload = {
        "identity_version": "kawaii_bad_channel_review_context_v1",
        "source_recording_id": source_recording_id or bad_channel_source_identity(recording),
        "reviewed_channels": list(map(str, reviewed_channels)),
        "channel_configuration": dict(config.get("channels", {}) or {}),
        "montage": preprocessing.get("montage"),
        "line_frequency_hz": preprocessing.get("line_frequency_hz"),
        "raw_qc": dict(preprocessing.get("raw_qc", {}) or {}),
        "methodology": "conservative_human_in_loop_supporting_diagnostics_only",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_bad_channel_review_decision(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    review_status: str,
    confirmed_bad_channels: Sequence[str],
    source_recording_id: str | None,
    review_context_id: str | None,
    reviewed_channels: Sequence[str],
    reviewer: str = "",
    reviewed_at_utc: str | None = None,
    notes: str = "",
    root: Path | None = None,
    expected_current_semantic_decision_id: str | None | object = _EXPECTED_DECISION_UNSPECIFIED,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> BadChannelReviewDecision:
    """Replace one scoped decision through the shared YAML persistence seam."""

    entry = {
        "decision_schema_version": BAD_CHANNEL_DECISION_SCHEMA_VERSION,
        "participant": subject,
        "recording_part": recording_part,
        "review_status": str(review_status),
        "source_recording_id": source_recording_id,
        "review_context_id": review_context_id,
        "reviewed_channels": list(map(str, reviewed_channels)),
        "confirmed_bad_channels": list(map(str, confirmed_bad_channels)),
        "reviewer": str(reviewer),
        "reviewed_at_utc": reviewed_at_utc,
        "notes": str(notes),
    }
    if review_status == "complete":
        entry["decision_id"] = _bad_channel_decision_identity(entry)
    else:
        entry["decision_id"] = None
    path = manual_review_paths(config, root=root)["bad_channels"]
    decision = _bad_channel_decision_from_entry(entry, subject, recording_part, path)

    def replace_scoped(data: dict[str, Any]) -> BadChannelReviewDecision:
        current_entry = data.get(subject, {}).get(recording_part)
        current = _bad_channel_decision_from_entry(current_entry, subject, recording_part, path)
        _validate_scoped_replacement(
            current_status=current.review_status,
            current_complete=current.complete,
            current_legacy=current.legacy_ambiguous,
            current_semantic_id=current.semantic_decision_id,
            incoming_semantic_id=decision.semantic_decision_id,
            expected_current_semantic_id=expected_current_semantic_decision_id,
            label=f"bad-channel decision for {subject} {recording_part}",
        )
        data.setdefault(subject, {})[recording_part] = dict(entry)
        return decision

    return update_yaml_store(
        path,
        mutate=replace_scoped,
        validate_store=_validate_bad_channel_store,
        validate_committed=lambda data, result, committed_path: _validate_committed_bad_decision(
            data, result, entry, subject, recording_part, committed_path
        ),
        lock_timeout_s=lock_timeout_s,
    )


def _bad_channel_decision_from_entry(
    entry: Any,
    subject: str,
    recording_part: str,
    path: Path,
) -> BadChannelReviewDecision:
    if entry is None:
        return BadChannelReviewDecision(
            subject, recording_part, "not_reviewed", (), None, None, (), "", None, None,
            BAD_CHANNEL_DECISION_SCHEMA_VERSION, "",
        )
    if not isinstance(entry, Mapping):
        raise ValueError(f"Bad-channel review entry must be a mapping for {subject} {recording_part}: {path}")
    if "review_status" not in entry:
        legacy_channels = _validated_channel_list(
            entry.get("confirmed_bad_channels", []), subject, recording_part, "confirmed_bad_channels"
        )
        if legacy_channels:
            raise ValueError(
                f"Legacy bad-channel entry for {subject} {recording_part} contains channels but no explicit "
                "review_status; it cannot be treated as authoritative."
            )
        return BadChannelReviewDecision(
            subject, recording_part, "not_reviewed", (), None, None, (), "", None, None,
            BAD_CHANNEL_DECISION_SCHEMA_VERSION, str(entry.get("notes", "") or ""), True,
        )

    status = str(entry.get("review_status"))
    if status not in BAD_CHANNEL_REVIEW_STATUSES:
        raise ValueError(f"Invalid bad-channel review_status {status!r} for {subject} {recording_part}.")
    entry_subject = str(entry.get("participant", subject))
    entry_part = str(entry.get("recording_part", recording_part))
    if entry_subject != subject:
        raise ValueError(f"Bad-channel review participant mismatch: expected {subject}, found {entry_subject}.")
    if entry_part != recording_part:
        raise ValueError(f"Bad-channel review part mismatch: expected {recording_part}, found {entry_part}.")
    confirmed = _validated_channel_list(
        entry.get("confirmed_bad_channels", []), subject, recording_part, "confirmed_bad_channels"
    )
    reviewed = _validated_channel_list(
        entry.get("reviewed_channels", []), subject, recording_part, "reviewed_channels"
    )
    if status != "complete" and confirmed:
        raise ValueError(
            f"Incomplete bad-channel review for {subject} {recording_part} cannot contain confirmed channels."
        )
    unknown = [channel for channel in confirmed if channel not in reviewed]
    if unknown:
        raise ValueError(
            f"Confirmed bad channels are absent from the reviewed channel inventory for {subject} "
            f"{recording_part}: {unknown}."
        )
    schema_version = entry.get("decision_schema_version", BAD_CHANNEL_DECISION_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f"Bad-channel decision_schema_version must be an integer for {subject} {recording_part}.")
    if schema_version != BAD_CHANNEL_DECISION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bad-channel decision schema version {schema_version} for {subject} {recording_part}; "
            f"expected {BAD_CHANNEL_DECISION_SCHEMA_VERSION}."
        )
    source_id = entry.get("source_recording_id")
    source_id = str(source_id) if source_id not in (None, "") else None
    context_id = entry.get("review_context_id")
    context_id = str(context_id) if context_id not in (None, "") else None
    reviewer = str(entry.get("reviewer", "") or "")
    reviewed_at = entry.get("reviewed_at_utc")
    reviewed_at = str(reviewed_at) if reviewed_at not in (None, "") else None
    decision_id = entry.get("decision_id")
    decision_id = str(decision_id) if decision_id not in (None, "") else None
    if status == "complete":
        missing = [
            name for name, value in (
                ("decision_schema_version", entry.get("decision_schema_version")),
                ("participant", entry.get("participant")),
                ("recording_part", entry.get("recording_part")),
                ("source_recording_id", source_id),
                ("review_context_id", context_id),
                ("reviewed_channels", reviewed),
                ("reviewer", reviewer),
                ("reviewed_at_utc", reviewed_at),
                ("decision_id", decision_id),
            ) if not value
        ]
        if missing:
            raise ValueError(
                f"Completed bad-channel review for {subject} {recording_part} is missing required fields: {missing}."
            )
        _validate_sha256_identity(source_id, "source_recording_id", subject, recording_part)
        _validate_sha256_identity(context_id, "review_context_id", subject, recording_part)
        _validate_utc_timestamp(reviewed_at, subject, recording_part)
        expected_decision_id = _bad_channel_decision_identity(entry)
        if decision_id != expected_decision_id:
            raise ValueError(
                f"Bad-channel decision fingerprint mismatch for {subject} {recording_part}: "
                f"stored {decision_id!r}, computed {expected_decision_id!r}."
            )
    return BadChannelReviewDecision(
        entry_subject,
        entry_part,
        status,
        tuple(confirmed),
        source_id,
        context_id,
        tuple(reviewed),
        reviewer,
        reviewed_at,
        decision_id,
        schema_version,
        str(entry.get("notes", "") or ""),
    )


def _bad_channel_decision_identity(entry: Mapping[str, Any]) -> str:
    fields = {
        key: entry.get(key)
        for key in (
            "decision_schema_version", "participant", "session", "recording_part", "review_status",
            "source_recording_id", "review_context_id", "reviewed_channels",
            "confirmed_bad_channels", "reviewer", "reviewed_at_utc", "notes",
        )
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bad_channel_semantic_decision_identity(entry: Mapping[str, Any]) -> str:
    fields = {
        key: entry.get(key)
        for key in (
            "decision_schema_version", "participant", "recording_part", "review_status",
            "source_recording_id", "review_context_id", "reviewed_channels",
            "confirmed_bad_channels",
        )
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.digest()


def _validated_channel_list(
    value: Any,
    subject: str,
    recording_part: str,
    field: str,
) -> list[str]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list for {subject} {recording_part}.")
    channels = []
    for channel in value:
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError(f"{field} values must be non-empty channel names; got {channel!r}.")
        channels.append(channel.strip())
    duplicates = sorted({channel for channel in channels if channels.count(channel) > 1})
    if duplicates:
        raise ValueError(f"Duplicate channel decisions are not allowed in {field}: {duplicates}.")
    return channels


def _validate_utc_timestamp(
    value: str,
    subject: str,
    recording_part: str,
    *,
    kind: str = "Bad-channel",
) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{kind} reviewed_at_utc must be ISO-8601 for {subject} {recording_part}."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{kind} reviewed_at_utc must use UTC for {subject} {recording_part}.")


def _validate_sha256_identity(
    value: str,
    field: str,
    subject: str,
    recording_part: str,
    *,
    kind: str = "bad-channel",
) -> None:
    prefix = "sha256:"
    hexadecimal = value[len(prefix):] if value.startswith(prefix) else ""
    if len(hexadecimal) != 64 or any(character not in "0123456789abcdef" for character in hexadecimal.lower()):
        raise ValueError(
            f"Completed {kind} {field} must be a full SHA-256 identity for {subject} {recording_part}."
        )


def read_confirmed_ica_components(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> list[int]:
    """Read confirmed ICA components from the manual-review scaffold.

    Gate 3B never applies these values; Gate 3C consumes only completed reviews.
    """

    decision = read_ica_review_decision(config, subject, recording_part, root=root)
    return list(decision.confirmed_ica_components) if decision.complete else []


def read_ica_review_decision(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    root: Path | None = None,
    expected_solution_id: str | None = None,
    expected_component_count: int | None = None,
    expected_bad_channel_decision_id: str | None | object = _EXPECTED_CONTEXT_UNSPECIFIED,
    require_complete: bool = False,
) -> ICAReviewDecision:
    """Read and validate one audited, context-bound ICA human decision."""

    path = manual_review_paths(config, root=root)["ica_components"]
    data = _load_yaml_mapping(path)
    entry = data.get(subject, {}).get(recording_part)
    decision = _ica_decision_from_entry(entry, subject, recording_part, path)
    if require_complete and not decision.complete:
        legacy = " legacy/unverified entry" if decision.legacy_ambiguous else ""
        raise ValueError(
            f"ICA review is {decision.review_status!r}, not complete or authoritative, for {subject} "
            f"{recording_part}; Gate 3C application is blocked.{legacy}"
        )
    if expected_solution_id is not None and decision.complete and decision.ica_solution_id != expected_solution_id:
        raise ValueError(
            f"Stale ICA review for {subject} {recording_part}: decision solution {decision.ica_solution_id!r} "
            f"does not match loaded solution {expected_solution_id!r}."
        )
    if expected_component_count is not None and decision.complete and decision.fitted_component_count != int(expected_component_count):
        raise ValueError(
            f"ICA component-count mismatch for {subject} {recording_part}: decision records "
            f"{decision.fitted_component_count}, loaded solution has {int(expected_component_count)}."
        )
    if (
        expected_bad_channel_decision_id is not _EXPECTED_CONTEXT_UNSPECIFIED
        and decision.complete
        and decision.bad_channel_decision_id != expected_bad_channel_decision_id
    ):
        raise ValueError(
            f"Stale ICA review for {subject} {recording_part}: bound bad-channel decision "
            f"{decision.bad_channel_decision_id!r} does not match current authoritative decision "
            f"{expected_bad_channel_decision_id!r}."
        )
    if decision.complete:
        _validate_component_range(decision.confirmed_ica_components, int(expected_component_count or decision.fitted_component_count or 0))
    return decision


def write_ica_review_decision(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    review_status: str,
    confirmed_ica_components: Sequence[int],
    ica_solution_id: str | None,
    fitted_component_count: int | None,
    bad_channel_decision_id: str | None,
    reviewer: str = "",
    reviewed_at_utc: str | None = None,
    notes: str = "",
    root: Path | None = None,
    expected_current_semantic_decision_id: str | None | object = _EXPECTED_DECISION_UNSPECIFIED,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> ICAReviewDecision:
    """Persist one exact ICA human decision through the shared YAML writer."""

    entry: dict[str, Any] = {
        "decision_schema_version": ICA_DECISION_SCHEMA_VERSION,
        "participant": subject,
        "session": CANONICAL_SESSION,
        "recording_part": recording_part,
        "review_status": str(review_status),
        "ica_solution_id": ica_solution_id,
        "fitted_component_count": fitted_component_count,
        "bad_channel_decision_id": bad_channel_decision_id,
        "confirmed_ica_components": list(confirmed_ica_components),
        "reviewer": str(reviewer),
        "reviewed_at_utc": reviewed_at_utc,
        "notes": str(notes),
        "decision_id": None,
        "audit_record_id": None,
    }
    if review_status == "complete":
        entry["decision_id"] = _ica_semantic_decision_identity(entry)
        entry["audit_record_id"] = _ica_audit_record_identity(entry)
    path = manual_review_paths(config, root=root)["ica_components"]
    decision = _ica_decision_from_entry(entry, subject, recording_part, path)

    def replace_scoped(data: dict[str, Any]) -> ICAReviewDecision:
        current_entry = data.get(subject, {}).get(recording_part)
        current = _ica_decision_from_entry(current_entry, subject, recording_part, path)
        _validate_scoped_replacement(
            current_status=current.review_status,
            current_complete=current.complete,
            current_legacy=current.legacy_ambiguous,
            current_semantic_id=current.semantic_decision_id,
            incoming_semantic_id=decision.semantic_decision_id,
            expected_current_semantic_id=expected_current_semantic_decision_id,
            label=f"ICA decision for {subject} {recording_part}",
        )
        data.setdefault(subject, {})[recording_part] = dict(entry)
        return decision

    return update_yaml_store(
        path,
        mutate=replace_scoped,
        validate_store=_validate_ica_store,
        validate_committed=lambda data, result, committed_path: _validate_committed_ica_decision(
            data, result, entry, subject, recording_part, committed_path
        ),
        lock_timeout_s=lock_timeout_s,
    )


def _ica_decision_from_entry(
    entry: Any,
    subject: str,
    recording_part: str,
    path: Path,
) -> ICAReviewDecision:
    empty = ICAReviewDecision(
        subject, CANONICAL_SESSION, recording_part, "not_reviewed", (), None, None, None,
        "", None, None, None, ICA_DECISION_SCHEMA_VERSION, "",
    )
    if entry is None:
        return empty
    if not isinstance(entry, Mapping):
        raise ValueError(f"ICA review entry must be a mapping for {subject} {recording_part}: {path}")
    status = str(entry.get("review_status", "not_reviewed"))
    if status not in ICA_REVIEW_STATUSES:
        raise ValueError(f"Invalid ICA review_status {status!r} for {subject} {recording_part}.")
    entry_subject = str(entry.get("participant", subject))
    entry_session = str(entry.get("session", CANONICAL_SESSION))
    entry_part = str(entry.get("recording_part", recording_part))
    if entry_subject != subject:
        raise ValueError(f"ICA review participant mismatch: expected {subject}, found {entry_subject}.")
    if entry_session != CANONICAL_SESSION:
        raise ValueError(f"ICA review session mismatch: expected {CANONICAL_SESSION}, found {entry_session}.")
    if entry_part != recording_part:
        raise ValueError(f"ICA review part mismatch: expected {recording_part}, found {entry_part}.")
    components = _validated_component_list(entry.get("confirmed_ica_components", []), subject, recording_part)
    solution_id = _optional_string(entry.get("ica_solution_id"))
    bad_decision_id = _optional_string(entry.get("bad_channel_decision_id"))
    component_count = entry.get("fitted_component_count")
    if component_count is not None and (
        isinstance(component_count, bool) or not isinstance(component_count, int) or component_count < 0
    ):
        raise ValueError(f"fitted_component_count must be a non-negative integer for {subject} {recording_part}.")
    if status != "complete" and components:
        raise ValueError(
            f"Incomplete ICA review for {subject} {recording_part} cannot contain confirmed exclusions; "
            "use a separate draft record until review_status is complete."
        )
    schema = entry.get("decision_schema_version")
    if status == "complete" and schema is None:
        # Historical completed entries lack attribution and a decision fingerprint.
        # Preserve their exact selection as evidence, but never authorize it.
        return ICAReviewDecision(
            entry_subject, entry_session, entry_part, status, tuple(components), solution_id,
            component_count, bad_decision_id, str(entry.get("reviewer", "") or ""),
            _optional_string(entry.get("reviewed_at_utc")), None, None,
            ICA_DECISION_SCHEMA_VERSION, str(entry.get("notes", "") or ""), True,
        )
    if schema is None:
        schema = ICA_DECISION_SCHEMA_VERSION
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != ICA_DECISION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ICA decision schema version {schema!r} for {subject} {recording_part}; "
            f"expected {ICA_DECISION_SCHEMA_VERSION}."
        )
    reviewer = str(entry.get("reviewer", "") or "")
    reviewed_at = _optional_string(entry.get("reviewed_at_utc"))
    decision_id = _optional_string(entry.get("decision_id"))
    audit_record_id = _optional_string(entry.get("audit_record_id"))
    if status == "complete":
        missing = [
            name for name, value in (
                ("participant", entry.get("participant")),
                ("session", entry.get("session")),
                ("recording_part", entry.get("recording_part")),
                ("ica_solution_id", solution_id),
                ("fitted_component_count", component_count if component_count is not None else ""),
                ("bad_channel_decision_id", bad_decision_id),
                ("reviewer", reviewer),
                ("reviewed_at_utc", reviewed_at),
                ("decision_id", decision_id),
                ("audit_record_id", audit_record_id),
            ) if value == "" or value is None
        ]
        if missing:
            raise ValueError(
                f"Completed ICA review for {subject} {recording_part} is missing required fields: {missing}."
            )
        _validate_sha256_identity(solution_id, "ica_solution_id", subject, recording_part, kind="ICA")
        _validate_sha256_identity(bad_decision_id, "bad_channel_decision_id", subject, recording_part, kind="ICA")
        _validate_utc_timestamp(reviewed_at, subject, recording_part, kind="ICA")
        _validate_component_range(components, int(component_count))
        expected_decision = _ica_semantic_decision_identity(entry)
        if decision_id != expected_decision:
            raise ValueError(
                f"ICA semantic decision fingerprint mismatch for {subject} {recording_part}: "
                f"stored {decision_id!r}, computed {expected_decision!r}."
            )
        expected_audit = _ica_audit_record_identity(entry)
        if audit_record_id != expected_audit:
            raise ValueError(
                f"ICA audit-record fingerprint mismatch for {subject} {recording_part}: "
                f"stored {audit_record_id!r}, computed {expected_audit!r}."
            )
    return ICAReviewDecision(
        entry_subject, entry_session, entry_part, status, tuple(components), solution_id,
        component_count, bad_decision_id, reviewer, reviewed_at, decision_id, audit_record_id,
        schema, str(entry.get("notes", "") or ""),
    )


def _ica_semantic_decision_identity(entry: Mapping[str, Any]) -> str:
    fields = {
        key: entry.get(key)
        for key in (
            "decision_schema_version", "participant", "session", "recording_part", "review_status",
            "ica_solution_id", "fitted_component_count", "bad_channel_decision_id",
            "confirmed_ica_components",
        )
    }
    return _mapping_identity(fields)


def _ica_audit_record_identity(entry: Mapping[str, Any]) -> str:
    return _mapping_identity({
        "decision_id": entry.get("decision_id"),
        "reviewer": entry.get("reviewer"),
        "reviewed_at_utc": entry.get("reviewed_at_utc"),
        "notes": entry.get("notes"),
    })


def _mapping_identity(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def ica_solution_identity(ica, subject: str, recording_part: str) -> str:
    """Return the legacy pre-WS3 decomposition/scope ID for inspection only.

    Authoritative new review binding uses the persisted fit-context-aware
    solution fingerprint from :mod:`ica_identity`. This compatibility helper
    intentionally cannot attest input EEG, fit settings, or configuration.
    """

    digest = hashlib.sha256()
    for text in ("kawaii_ica_solution_v1", subject, recording_part, str(getattr(ica, "method", ""))):
        digest.update(text.encode("utf-8") + b"\0")
    for name in getattr(ica, "ch_names", []) or []:
        digest.update(str(name).encode("utf-8") + b"\0")
    digest.update(str(int(getattr(ica, "n_components_", 0) or 0)).encode("ascii") + b"\0")
    for attribute in ("unmixing_matrix_", "mixing_matrix_", "pca_components_", "pre_whitener_"):
        value = getattr(ica, attribute, None)
        if value is None:
            digest.update(attribute.encode("ascii") + b":none\0")
            continue
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(attribute.encode("ascii") + repr(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def write_ica_review_template(
    path: Path,
    *,
    subject: str,
    recording_part: str,
    ica_solution_id: str,
    fitted_component_count: int,
    automatic_eog_suggested_components: Sequence[int] = (),
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Path:
    """Write a solution-bound template without changing the human decision file."""

    payload = {
        "participant": subject,
        "recording_part": recording_part,
        "review_status": "not_reviewed",
        "ica_solution_id": ica_solution_id,
        "fitted_component_count": int(fitted_component_count),
        "automatic_eog_suggested_components": sorted(set(int(value) for value in automatic_eog_suggested_components)),
        "confirmed_ica_components": [],
        "notes": "",
    }
    return write_yaml_document(path, payload, lock_timeout_s=lock_timeout_s)


def read_review_notes(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> str:
    """Read YAML notes from the bad-channel and ICA review files."""

    paths = manual_review_paths(config, root=root)
    notes = []
    for key in ("bad_channels", "ica_components"):
        data = _load_yaml_mapping(paths[key])
        note = data.get(subject, {}).get(recording_part, {}).get("notes", "")
        if note:
            notes.append(str(note))
    return "\n".join(notes)


def _ensure_bad_channel_entries(data: dict[str, Any], subject: str, part_labels: Sequence[str]) -> None:
    subject_entry = data.setdefault(subject, {})
    for part in part_labels:
        subject_entry.setdefault(part, {
            "decision_schema_version": BAD_CHANNEL_DECISION_SCHEMA_VERSION,
            "participant": subject,
            "recording_part": part,
            "review_status": "not_reviewed",
            "source_recording_id": None,
            "review_context_id": None,
            "reviewed_channels": [],
            "confirmed_bad_channels": [],
            "reviewer": "",
            "reviewed_at_utc": None,
            "decision_id": None,
            "notes": "",
        })


def _ensure_ica_entries(data: dict[str, Any], subject: str, part_labels: Sequence[str]) -> None:
    subject_entry = data.setdefault(subject, {})
    for part in part_labels:
        subject_entry.setdefault(part, {
            "decision_schema_version": ICA_DECISION_SCHEMA_VERSION,
            "participant": subject,
            "session": CANONICAL_SESSION,
            "recording_part": part,
            "review_status": "not_reviewed",
            "ica_solution_id": None,
            "fitted_component_count": None,
            "bad_channel_decision_id": None,
            "confirmed_ica_components": [],
            "reviewer": "",
            "reviewed_at_utc": None,
            "decision_id": None,
            "audit_record_id": None,
            "notes": "",
        })


def _validated_component_list(value: Any, subject: str, recording_part: str) -> list[int]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError(f"confirmed_ica_components must be a list for {subject} {recording_part}.")
    components = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, int) or component < 0:
            raise ValueError(f"ICA component indices must be non-negative integers; got {component!r}.")
        components.append(component)
    duplicates = sorted({component for component in components if components.count(component) > 1})
    if duplicates:
        raise ValueError(f"Duplicate ICA component indices are not allowed: {duplicates}.")
    return components


def _validate_component_range(components: Sequence[int], n_components: int) -> None:
    for component in components:
        if component >= n_components:
            raise ValueError(f"ICA component index {component} is out of range for {n_components} fitted components.")


def _ensure_notes_file(
    path: Path,
    subject_recordings: Mapping[str, Sequence[RecordingPart] | Sequence[str]],
    *,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> None:
    def add_missing_headings(current: str) -> str:
        existing = current or "# Session 2 EEG Participant Notes\n\n"
        additions = []
        for subject, recordings in subject_recordings.items():
            if f"## {subject}" not in existing:
                additions.append(f"## {subject}\n")
            for part in recordings:
                part_label = _part_label(part)
                heading = f"### {subject} {part_label}"
                if heading not in existing:
                    additions.append(f"{heading}\n\n- Notes:\n\n")
        if not additions:
            return existing
        return existing.rstrip() + "\n\n" + "\n".join(additions)

    update_text_store(path, mutate=add_missing_headings, lock_timeout_s=lock_timeout_s)


def _validate_scoped_replacement(
    *,
    current_status: str,
    current_complete: bool,
    current_legacy: bool,
    current_semantic_id: str | None,
    incoming_semantic_id: str | None,
    expected_current_semantic_id: str | None | object,
    label: str,
) -> None:
    if current_legacy and current_status == "complete":
        raise ManualReviewConflictError(
            f"Cannot replace legacy/unverified completed {label}; human re-review is required."
        )
    if expected_current_semantic_id is not _EXPECTED_DECISION_UNSPECIFIED:
        if expected_current_semantic_id is not None and not isinstance(
            expected_current_semantic_id, str
        ):
            raise ValueError("Expected current semantic decision identity must be text or None.")
        if expected_current_semantic_id != current_semantic_id:
            raise ManualReviewConflictError(
                f"Stale manual-review update for {label}: expected current semantic decision "
                f"{expected_current_semantic_id!r}, found {current_semantic_id!r}."
            )
        return
    if current_complete and incoming_semantic_id != current_semantic_id:
        raise ManualReviewConflictError(
            f"Refusing to replace existing authoritative {label} without its expected current "
            f"semantic decision identity {current_semantic_id!r}."
        )


def _validate_bad_channel_store(data: Mapping[str, Any], path: Path) -> None:
    _validate_store_shape(data, path)
    for subject, parts in data.items():
        for recording_part, entry in parts.items():
            if isinstance(entry, Mapping) and "review_status" not in entry:
                _validated_channel_list(
                    entry.get("confirmed_bad_channels", []),
                    subject,
                    recording_part,
                    "confirmed_bad_channels",
                )
                continue
            _bad_channel_decision_from_entry(entry, subject, recording_part, path)


def _validate_ica_store(data: Mapping[str, Any], path: Path) -> None:
    _validate_store_shape(data, path)
    for subject, parts in data.items():
        for recording_part, entry in parts.items():
            if isinstance(entry, Mapping) and "review_status" not in entry:
                _validated_component_list(
                    entry.get("confirmed_ica_components", []), subject, recording_part
                )
                continue
            _ica_decision_from_entry(entry, subject, recording_part, path)


def _validate_store_shape(data: Mapping[str, Any], path: Path) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"Manual-review store must contain a mapping: {path}")
    for subject, parts in data.items():
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError(f"Manual-review participant keys must be non-empty text: {path}")
        if not isinstance(parts, Mapping):
            raise ValueError(f"Manual-review participant entry must be a mapping for {subject}: {path}")
        for recording_part in parts:
            if not isinstance(recording_part, str) or not recording_part.strip():
                raise ValueError(
                    f"Manual-review recording-part keys must be non-empty text for {subject}: {path}"
                )


def _validate_requested_scaffold(
    data: Mapping[str, Any],
    requested: Mapping[str, Sequence[str]],
    path: Path,
) -> None:
    for subject, parts in requested.items():
        stored = data.get(subject)
        if not isinstance(stored, Mapping):
            raise ManualReviewPersistenceError(
                f"Committed scaffold is missing participant {subject!r}: {path}"
            )
        missing = [part for part in parts if part not in stored]
        if missing:
            raise ManualReviewPersistenceError(
                f"Committed scaffold is missing parts for {subject!r}: {missing!r}: {path}"
            )


def _validate_committed_bad_decision(
    data: Mapping[str, Any],
    expected: BadChannelReviewDecision,
    expected_entry: Mapping[str, Any],
    subject: str,
    recording_part: str,
    path: Path,
) -> None:
    entry = data.get(subject, {}).get(recording_part)
    if not isinstance(entry, Mapping) or dict(entry) != dict(expected_entry):
        raise ManualReviewPersistenceError(
            f"Committed bad-channel decision does not exactly match the intended record for "
            f"{subject} {recording_part}: {path}"
        )
    committed = _bad_channel_decision_from_entry(entry, subject, recording_part, path)
    if (
        committed.semantic_decision_id != expected.semantic_decision_id
        or committed.audit_record_id != expected.audit_record_id
    ):
        raise ManualReviewPersistenceError(
            f"Committed bad-channel decision identities do not match the intended record for "
            f"{subject} {recording_part}: {path}"
        )


def _validate_committed_ica_decision(
    data: Mapping[str, Any],
    expected: ICAReviewDecision,
    expected_entry: Mapping[str, Any],
    subject: str,
    recording_part: str,
    path: Path,
) -> None:
    entry = data.get(subject, {}).get(recording_part)
    if not isinstance(entry, Mapping) or dict(entry) != dict(expected_entry):
        raise ManualReviewPersistenceError(
            f"Committed ICA decision does not exactly match the intended record for "
            f"{subject} {recording_part}: {path}"
        )
    committed = _ica_decision_from_entry(entry, subject, recording_part, path)
    if (
        committed.semantic_decision_id != expected.semantic_decision_id
        or committed.audit_record_id != expected.audit_record_id
    ):
        raise ManualReviewPersistenceError(
            f"Committed ICA decision identities do not match the intended record for "
            f"{subject} {recording_part}: {path}"
        )


def _part_label(part: RecordingPart | str) -> str:
    return part.recording_part if isinstance(part, RecordingPart) else str(part)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Manual review file must contain a mapping: {path}")
    return loaded
