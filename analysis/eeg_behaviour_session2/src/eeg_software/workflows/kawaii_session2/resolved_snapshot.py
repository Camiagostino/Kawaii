from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .resolved_config import canonical_json, canonical_sha256
from .workflow import Session2EEGRunRequest


RESOLVED_RUN_SNAPSHOT_SCHEMA_VERSION = "kawaii_session2_resolved_run_snapshot_v1"
REQUESTED_SNAPSHOT_STATE = "requested_resolved"
REALIZATION_IN_PROGRESS_SNAPSHOT_STATE = "realization_in_progress"
REALIZED_SNAPSHOT_STATE = "realized_complete"
_SNAPSHOT_STATES = {
    REQUESTED_SNAPSHOT_STATE,
    REALIZATION_IN_PROGRESS_SNAPSHOT_STATE,
    REALIZED_SNAPSHOT_STATE,
}


class ResolvedRunSnapshotError(ValueError):
    """Raised when a persisted authoritative snapshot is invalid."""


@dataclass(frozen=True)
class PersistedResolvedRunSnapshot:
    """Validated persisted snapshot payload.

    The digest identifies snapshot content only. It is not a dataset or run
    identity and intentionally contains no input-data identity.
    """

    path: Path
    payload: Mapping[str, Any]

    @property
    def snapshot_state(self) -> str:
        return str(self.payload["snapshot_state"])

    @property
    def run_request(self) -> Mapping[str, Any]:
        return _mapping(self.payload["run_request"], "run_request")

    @property
    def resolved_config(self) -> Mapping[str, Any]:
        return _mapping(self.run_request["resolved_config"], "run_request.resolved_config")

    @property
    def scientific_fingerprint(self) -> str:
        return str(self.resolved_config["scientific_fingerprint"])

    @property
    def resolved_config_fingerprint(self) -> str:
        return str(self.resolved_config["resolved_config_fingerprint"])

    @property
    def source_config(self) -> Mapping[str, Any]:
        return _mapping(self.resolved_config["source"], "run_request.resolved_config.source")

    @property
    def snapshot_content_sha256(self) -> str:
        return str(self.payload["snapshot_content_sha256"])


@dataclass(frozen=True)
class ResolvedRunSnapshotReference:
    """Small immutable reference threaded through existing provenance APIs."""

    path: Path
    schema_version: str
    snapshot_state: str
    snapshot_content_sha256: str
    source_config_sha256: str
    scientific_fingerprint: str
    resolved_config_fingerprint: str

    @classmethod
    def from_snapshot(cls, snapshot: PersistedResolvedRunSnapshot) -> "ResolvedRunSnapshotReference":
        return cls(
            path=snapshot.path.resolve(),
            schema_version=str(snapshot.payload["schema_version"]),
            snapshot_state=snapshot.snapshot_state,
            snapshot_content_sha256=snapshot.snapshot_content_sha256,
            source_config_sha256=str(snapshot.source_config["sha256"]),
            scientific_fingerprint=snapshot.scientific_fingerprint,
            resolved_config_fingerprint=snapshot.resolved_config_fingerprint,
        )


@dataclass(frozen=True)
class AuthoritativeConfigProvenance(os.PathLike[str]):
    """Path-like compatibility adapter carrying an authoritative snapshot."""

    source_config_path: Path
    snapshot: ResolvedRunSnapshotReference

    def __fspath__(self) -> str:
        return str(self.source_config_path)


def resolved_run_snapshot_path(
    request: Session2EEGRunRequest, snapshot_state: str
) -> Path:
    """Return the stable requested or realized snapshot path."""

    _validate_snapshot_state(snapshot_state)
    if snapshot_state == REQUESTED_SNAPSHOT_STATE:
        filename = "resolved_run_request.requested.json"
    elif snapshot_state == REALIZATION_IN_PROGRESS_SNAPSHOT_STATE:
        realized_count = len(request.realized.recording_parts)
        filename = f"resolved_run_request.realized.in-progress-{realized_count:03d}.json"
    else:
        filename = "resolved_run_request.realized.json"
    return request.output_destination / "provenance" / filename


def build_resolved_run_snapshot(
    request: Session2EEGRunRequest, snapshot_state: str
) -> dict[str, Any]:
    """Build authoritative persisted content only from the run request."""

    _validate_snapshot_state(snapshot_state)
    _validate_state_against_request(request, snapshot_state)
    content = {
        "schema_version": RESOLVED_RUN_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_state": snapshot_state,
        "authority": "Session2EEGRunRequest",
        "run_request": request.to_primitive(),
    }
    return {**content, "snapshot_content_sha256": canonical_sha256(content)}


def write_resolved_run_snapshot(
    request: Session2EEGRunRequest,
    snapshot_state: str,
    *,
    path: str | Path | None = None,
) -> ResolvedRunSnapshotReference:
    """Atomically persist and validate one requested or realized state."""

    target = Path(path) if path is not None else resolved_run_snapshot_path(request, snapshot_state)
    payload = build_resolved_run_snapshot(request, snapshot_state)
    _atomic_write_text(target, canonical_json(payload) + "\n")
    return ResolvedRunSnapshotReference.from_snapshot(read_resolved_run_snapshot(target))


def read_resolved_run_snapshot(path: str | Path) -> PersistedResolvedRunSnapshot:
    """Read and fail closed on malformed or internally inconsistent content."""

    snapshot_path = Path(path)
    try:
        parsed = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResolvedRunSnapshotError(f"Could not read resolved run snapshot {snapshot_path}: {error}") from error
    if not isinstance(parsed, dict):
        raise ResolvedRunSnapshotError("Resolved run snapshot root must be a mapping.")
    _validate_snapshot_payload(parsed)
    return PersistedResolvedRunSnapshot(path=snapshot_path.resolve(), payload=parsed)


def bind_authoritative_config_provenance(
    config_path: str | Path | os.PathLike[str] | None,
    snapshot: ResolvedRunSnapshotReference | None,
) -> str | Path | os.PathLike[str] | None:
    """Attach a snapshot to the selected source path for legacy call funnels."""

    if snapshot is None or isinstance(config_path, AuthoritativeConfigProvenance):
        return config_path
    if config_path is None:
        persisted = read_resolved_run_snapshot(snapshot.path)
        selected = persisted.run_request.get("selected_config_path")
        if not isinstance(selected, str) or not selected:
            raise ResolvedRunSnapshotError("Snapshot lacks selected_config_path for provenance binding.")
        config_path = selected
    return AuthoritativeConfigProvenance(Path(config_path).resolve(), snapshot)


def split_authoritative_config_provenance(
    config_path: str | Path | os.PathLike[str] | None,
) -> tuple[str | Path | os.PathLike[str] | None, ResolvedRunSnapshotReference | None]:
    if isinstance(config_path, AuthoritativeConfigProvenance):
        return config_path.source_config_path, config_path.snapshot
    return config_path, None


def _validate_snapshot_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESOLVED_RUN_SNAPSHOT_SCHEMA_VERSION:
        raise ResolvedRunSnapshotError("Unsupported or missing resolved run snapshot schema_version.")
    state = payload.get("snapshot_state")
    if state not in _SNAPSHOT_STATES:
        raise ResolvedRunSnapshotError(f"Invalid resolved run snapshot state: {state!r}.")
    if payload.get("authority") != "Session2EEGRunRequest":
        raise ResolvedRunSnapshotError("Resolved run snapshot authority must be Session2EEGRunRequest.")
    run_request = _mapping(payload.get("run_request"), "run_request")
    resolved = _mapping(run_request.get("resolved_config"), "run_request.resolved_config")
    study = _mapping(resolved.get("study_definition"), "run_request.resolved_config.study_definition")
    execution = _mapping(resolved.get("execution"), "run_request.resolved_config.execution")
    effective = _mapping(resolved.get("effective_config"), "run_request.resolved_config.effective_config")
    source = _mapping(resolved.get("source"), "run_request.resolved_config.source")
    requested = _mapping(run_request.get("requested"), "run_request.requested")
    realized = _mapping(run_request.get("realized"), "run_request.realized")
    gate3 = _mapping(run_request.get("gate3"), "run_request.gate3")
    workflow = _mapping(run_request.get("workflow"), "run_request.workflow")

    for label, value in (
        ("source.path", source.get("path")),
        ("source.sha256", source.get("sha256")),
        ("workflow.name", workflow.get("name")),
        ("workflow.version", workflow.get("version")),
        ("gate3.execution_mode", gate3.get("execution_mode")),
    ):
        if not isinstance(value, str) or not value:
            raise ResolvedRunSnapshotError(f"Resolved run snapshot requires non-empty {label}.")
    if not isinstance(requested.get("participants"), (list, type(None))):
        raise ResolvedRunSnapshotError("Requested participants must be a list or null.")
    if not isinstance(requested.get("recording_parts"), (list, type(None))):
        raise ResolvedRunSnapshotError("Requested recording parts must be a list or null.")
    if not isinstance(realized.get("participants"), list) or not isinstance(realized.get("recording_parts"), list):
        raise ResolvedRunSnapshotError("Realized participants and recording parts must be lists.")
    if realized["participants"] != execution.get("subjects"):
        raise ResolvedRunSnapshotError("Realized participants do not match resolved execution subjects.")
    if bool(gate3.get("validation_only_skip_ica")) != bool(execution.get("validation_only_skip_ica")):
        raise ResolvedRunSnapshotError("Validation-only state disagrees between request and resolved execution.")
    if state == REALIZED_SNAPSHOT_STATE:
        part_subjects = [
            _mapping(item, "run_request.realized.recording_parts item").get("participant")
            for item in realized["recording_parts"]
        ]
        if part_subjects != realized["participants"]:
            raise ResolvedRunSnapshotError("Complete realized snapshot must represent every selected participant in order.")

    scientific_expected = canonical_sha256(study)
    if resolved.get("scientific_fingerprint") != scientific_expected:
        raise ResolvedRunSnapshotError("Persisted scientific fingerprint does not match the study definition.")
    resolved_identity = {
        "schema_version": resolved.get("schema_version"),
        "effective_config": effective,
        "study_definition": study,
        "execution": execution,
    }
    if resolved.get("resolved_config_fingerprint") != canonical_sha256(resolved_identity):
        raise ResolvedRunSnapshotError("Persisted resolved-config fingerprint does not match effective content.")
    content = {key: value for key, value in payload.items() if key != "snapshot_content_sha256"}
    if payload.get("snapshot_content_sha256") != canonical_sha256(content):
        raise ResolvedRunSnapshotError("Persisted snapshot content digest does not match its content.")


def _validate_state_against_request(request: Session2EEGRunRequest, snapshot_state: str) -> None:
    represented = {item.participant for item in request.realized.recording_parts}
    if snapshot_state == REQUESTED_SNAPSHOT_STATE and represented:
        raise ResolvedRunSnapshotError("Requested snapshot must be written before recording-part realization.")
    if snapshot_state == REALIZATION_IN_PROGRESS_SNAPSHOT_STATE and not represented:
        raise ResolvedRunSnapshotError("In-progress snapshot requires at least one realized participant entry.")
    if snapshot_state == REALIZED_SNAPSHOT_STATE and represented != set(request.realized.participants):
        raise ResolvedRunSnapshotError("Complete realized snapshot requires every selected participant.")


def _validate_snapshot_state(snapshot_state: str) -> None:
    if snapshot_state not in _SNAPSHOT_STATES:
        raise ResolvedRunSnapshotError(f"Unknown resolved run snapshot state: {snapshot_state!r}.")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResolvedRunSnapshotError(f"Resolved run snapshot field {label} must be a mapping.")
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
