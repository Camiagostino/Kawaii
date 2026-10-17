from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .manual_review_persistence import DEFAULT_LOCK_TIMEOUT_SECONDS, update_yaml_store
from .paths import resolve_session2_eeg_paths
from .study_definition import configured_rois, erp_branch_settings, plotting_channel_exclusions


DISPOSITION_FILE = "bad_channel_interpolation_dispositions.yaml"
DISPOSITION_SCHEMA_VERSION = 1
RETAIN_BAD_UNINTERPOLATED = "retain_bad_uninterpolated_due_to_unresolved_geometry"
INFO_MARKER_PREFIX = "KAWAII_RETAINED_BAD_UNINTERPOLATED_V1="


@dataclass(frozen=True)
class BadChannelInterpolationDisposition:
    participant: str
    recording_part: str
    approval_status: str
    disposition: str | None
    channels: tuple[str, ...]
    bad_channel_decision_id: str | None
    bad_channel_review_context_id: str | None
    scientific_fingerprint: str | None
    reviewer: str
    reviewed_at_utc: str | None
    notes: str
    decision_id: str | None

    @property
    def approved(self) -> bool:
        return self.approval_status == "approved"


def disposition_store_path(config: Mapping[str, Any], *, root: Path | None = None) -> Path:
    paths = resolve_session2_eeg_paths(config, root=root, create=False)
    return paths.manual_review_dir / DISPOSITION_FILE


def read_bad_channel_interpolation_disposition(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    root: Path | None = None,
    expected_bad_channel_decision_id: str | None = None,
    expected_bad_channel_review_context_id: str | None = None,
    expected_scientific_fingerprint: str | None = None,
    expected_confirmed_bad_channels: Sequence[str] | None = None,
) -> BadChannelInterpolationDisposition:
    path = disposition_store_path(config, root=root)
    if not path.exists():
        return _empty_disposition(subject, recording_part)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read bad-channel interpolation disposition store {path}: {error}") from error
    _validate_store(data, path)
    entry = data.get(subject, {}).get(recording_part)
    decision = _decision_from_entry(entry, subject, recording_part, path)
    if not decision.approved:
        return decision
    expected = {
        "bad-channel decision": (decision.bad_channel_decision_id, expected_bad_channel_decision_id),
        "bad-channel review context": (
            decision.bad_channel_review_context_id,
            expected_bad_channel_review_context_id,
        ),
        "scientific configuration": (decision.scientific_fingerprint, expected_scientific_fingerprint),
    }
    for label, (stored, current) in expected.items():
        if current is not None and stored != current:
            raise ValueError(
                f"Stale or mismatched bad-channel interpolation disposition for {subject} "
                f"{recording_part}: stored {label} {stored!r} does not match current {current!r}."
            )
    if expected_confirmed_bad_channels is not None:
        confirmed = set(map(str, expected_confirmed_bad_channels))
        invalid = [channel for channel in decision.channels if channel not in confirmed]
        if invalid:
            raise ValueError(
                "Bad-channel interpolation disposition names channels that are not in the current "
                f"human-confirmed bad-channel decision: {invalid}."
            )
    return decision


def write_bad_channel_interpolation_disposition(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    approval_status: str,
    disposition: str | None,
    channels: Sequence[str],
    bad_channel_decision_id: str | None,
    bad_channel_review_context_id: str | None,
    scientific_fingerprint: str | None,
    reviewer: str = "",
    reviewed_at_utc: str | None = None,
    notes: str = "",
    root: Path | None = None,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> BadChannelInterpolationDisposition:
    entry = {
        "decision_schema_version": DISPOSITION_SCHEMA_VERSION,
        "participant": str(subject),
        "session": "session2",
        "recording_part": str(recording_part),
        "approval_status": str(approval_status),
        "disposition": disposition,
        "channels": list(map(str, channels)),
        "bad_channel_decision_id": bad_channel_decision_id,
        "bad_channel_review_context_id": bad_channel_review_context_id,
        "scientific_fingerprint": scientific_fingerprint,
        "reviewer": str(reviewer),
        "reviewed_at_utc": reviewed_at_utc,
        "notes": str(notes),
    }
    entry["decision_id"] = _decision_identity(entry) if approval_status == "approved" else None
    path = disposition_store_path(config, root=root)
    decision = _decision_from_entry(entry, subject, recording_part, path)

    def replace(data: dict[str, Any]) -> BadChannelInterpolationDisposition:
        data.setdefault(str(subject), {})[str(recording_part)] = dict(entry)
        return decision

    return update_yaml_store(
        path,
        mutate=replace,
        validate_store=_validate_store,
        validate_committed=lambda data, result, committed_path: _validate_committed(
            data, result, entry, committed_path
        ),
        lock_timeout_s=lock_timeout_s,
    )


def validate_retained_bad_channel_analysis_safety(
    config: Mapping[str, Any], channels: Sequence[str]
) -> None:
    retained = set(map(str, channels))
    if not retained:
        return
    for branch in ("recognition", "cuteness"):
        settings = erp_branch_settings(config, branch)
        roi_channels = {
            channel for roi in configured_rois(settings).values() for channel in roi
        }
        overlap = sorted(retained & roi_channels)
        if overlap:
            raise ValueError(
                f"Retained-uninterpolated bad channel enters configured {branch} ROI(s): {overlap}."
            )
        plotting_excluded = set(plotting_channel_exclusions(settings))
        missing = sorted(retained - plotting_excluded)
        if missing:
            raise ValueError(
                f"Retained-uninterpolated bad channel is not excluded from {branch} all-channel "
                f"figures: {missing}."
            )
    primary = (
        dict(dict(config.get("erp", {}) or {}).get("cuteness", {}) or {})
        .get("primary_model", {})
        or {}
    )
    primary_roi = set(map(str, dict(primary.get("roi", {}) or {}).get("channels", []) or []))
    overlap = sorted(retained & primary_roi)
    if overlap:
        raise ValueError(
            f"Retained-uninterpolated bad channel enters the primary continuous-cuteness ROI: {overlap}."
        )


def validate_downstream_retained_bad_channel_safety(info: Any, settings: Mapping[str, Any]) -> tuple[str, ...]:
    """Fail if retained bad data could enter a configured downstream analysis."""

    retained = retained_bad_channel_marker(info)
    if not retained:
        return ()
    retained_set = set(retained)
    roi_channels = {
        channel for roi in configured_rois(settings).values() for channel in roi
    }
    primary = dict(settings.get("primary_model", {}) or {})
    primary_roi = set(map(str, dict(primary.get("roi", {}) or {}).get("channels", []) or []))
    overlap = sorted(retained_set & (roi_channels | primary_roi))
    if overlap:
        raise ValueError(f"Retained-uninterpolated bad channel enters a downstream ROI: {overlap}.")
    missing_plot_exclusion = sorted(retained_set - set(plotting_channel_exclusions(settings)))
    if missing_plot_exclusion:
        raise ValueError(
            "Retained-uninterpolated bad channel is not excluded from downstream all-channel "
            f"figures: {missing_plot_exclusion}."
        )
    return retained


def bind_retained_bad_channel_marker(
    info: Any, channels: Sequence[str], disposition_id: str
) -> None:
    retained = list(dict.fromkeys(map(str, channels)))
    missing_bad = [channel for channel in retained if channel not in info.get("bads", [])]
    if missing_bad:
        raise ValueError(
            f"Cannot mark retained-uninterpolated channels that are not still bad: {missing_bad}."
        )
    payload = json.dumps(
        {"channels": retained, "disposition_id": str(disposition_id)},
        sort_keys=True,
        separators=(",", ":"),
    )
    prior = str(info.get("description") or "")
    lines = [line for line in prior.splitlines() if not line.startswith(INFO_MARKER_PREFIX)]
    lines.append(INFO_MARKER_PREFIX + payload)
    info["description"] = "\n".join(line for line in lines if line)


def retained_bad_channel_marker(info: Any, *, require_bad: bool = True) -> tuple[str, ...]:
    markers = [
        line[len(INFO_MARKER_PREFIX) :]
        for line in str(info.get("description") or "").splitlines()
        if line.startswith(INFO_MARKER_PREFIX)
    ]
    if not markers:
        return ()
    if len(markers) != 1:
        raise ValueError("Ambiguous retained-bad-channel execution markers in MNE Info.")
    try:
        payload = json.loads(markers[0])
    except json.JSONDecodeError as error:
        raise ValueError("Invalid retained-bad-channel execution marker in MNE Info.") from error
    channels = payload.get("channels")
    disposition_id = payload.get("disposition_id")
    if not isinstance(channels, list) or not channels or len(channels) != len(set(channels)):
        raise ValueError("Invalid retained-bad-channel channel list in MNE Info.")
    if not isinstance(disposition_id, str) or not disposition_id.startswith("sha256:"):
        raise ValueError("Invalid retained-bad-channel disposition identity in MNE Info.")
    if require_bad and any(channel not in info["ch_names"] for channel in channels):
        raise ValueError("Retained-bad-channel execution marker names an absent channel.")
    if require_bad and any(channel not in info.get("bads", []) for channel in channels):
        raise ValueError("A retained-uninterpolated channel is no longer marked bad downstream.")
    return tuple(map(str, channels))




def _empty_disposition(subject: str, recording_part: str) -> BadChannelInterpolationDisposition:
    return BadChannelInterpolationDisposition(
        str(subject), str(recording_part), "not_approved", None, (), None, None, None, "", None, "", None
    )


def _decision_from_entry(
    entry: Any, subject: str, recording_part: str, path: Path
) -> BadChannelInterpolationDisposition:
    if entry is None:
        return _empty_disposition(subject, recording_part)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Bad-channel interpolation disposition must be a mapping: {path}")
    status = entry.get("approval_status")
    if status not in {"not_approved", "approved"}:
        raise ValueError(f"Invalid bad-channel interpolation approval status for {subject} {recording_part}.")
    channels = entry.get("channels", [])
    if not isinstance(channels, list) or any(not isinstance(value, str) or not value for value in channels):
        raise ValueError(f"Bad-channel interpolation disposition channels must be non-empty text: {path}")
    if len(channels) != len(set(channels)):
        raise ValueError(f"Duplicate bad-channel interpolation disposition channels: {path}")
    if int(entry.get("decision_schema_version", 0)) != DISPOSITION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported bad-channel interpolation disposition schema: {path}")
    if entry.get("participant") != subject or entry.get("recording_part") != recording_part:
        raise ValueError(f"Bad-channel interpolation disposition scope mismatch: {path}")
    if entry.get("session") != "session2":
        raise ValueError(f"Bad-channel interpolation disposition session mismatch: {path}")
    if status == "approved":
        if entry.get("disposition") != RETAIN_BAD_UNINTERPOLATED or not channels:
            raise ValueError(f"Approved bad-channel interpolation disposition is incomplete: {path}")
        for field in (
            "bad_channel_decision_id",
            "bad_channel_review_context_id",
            "scientific_fingerprint",
            "reviewer",
            "reviewed_at_utc",
        ):
            if not isinstance(entry.get(field), str) or not entry.get(field):
                raise ValueError(f"Approved bad-channel interpolation disposition lacks {field}: {path}")
        _validate_timestamp(entry["reviewed_at_utc"], path)
        expected_id = _decision_identity(entry)
        if entry.get("decision_id") != expected_id:
            raise ValueError(f"Bad-channel interpolation disposition identity mismatch: {path}")
    return BadChannelInterpolationDisposition(
        str(subject), str(recording_part), str(status), entry.get("disposition"), tuple(channels),
        entry.get("bad_channel_decision_id"), entry.get("bad_channel_review_context_id"),
        entry.get("scientific_fingerprint"), str(entry.get("reviewer", "")),
        entry.get("reviewed_at_utc"), str(entry.get("notes", "")), entry.get("decision_id"),
    )


def _decision_identity(entry: Mapping[str, Any]) -> str:
    fields = {
        key: entry.get(key)
        for key in (
            "decision_schema_version", "participant", "session", "recording_part",
            "approval_status", "disposition", "channels", "bad_channel_decision_id",
            "bad_channel_review_context_id", "scientific_fingerprint", "reviewer",
            "reviewed_at_utc", "notes",
        )
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_timestamp(value: str, path: Path) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid reviewed_at_utc in bad-channel interpolation disposition: {path}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"reviewed_at_utc must include a timezone: {path}")


def _validate_store(data: Mapping[str, Any], path: Path) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"Bad-channel interpolation disposition store must be a mapping: {path}")
    for subject, parts in data.items():
        if not isinstance(subject, str) or not isinstance(parts, Mapping):
            raise ValueError(f"Invalid bad-channel interpolation disposition store shape: {path}")
        for recording_part, entry in parts.items():
            if not isinstance(recording_part, str):
                raise ValueError(f"Invalid bad-channel interpolation disposition part key: {path}")
            _decision_from_entry(entry, subject, recording_part, path)


def _validate_committed(
    data: Mapping[str, Any],
    expected: BadChannelInterpolationDisposition,
    expected_entry: Mapping[str, Any],
    path: Path,
) -> None:
    entry = data.get(expected.participant, {}).get(expected.recording_part)
    if not isinstance(entry, Mapping) or dict(entry) != dict(expected_entry):
        raise ValueError(f"Committed bad-channel interpolation disposition differs from intended record: {path}")
    if _decision_from_entry(entry, expected.participant, expected.recording_part, path) != expected:
        raise ValueError(f"Committed bad-channel interpolation disposition failed round-trip validation: {path}")
