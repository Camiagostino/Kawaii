from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Sequence

from .resolved_config import canonical_json, canonical_sha256


SOURCE_FILE_IDENTITY_SCHEMA_VERSION = "kawaii_source_file_identity_v1"
BRAINVISION_DATASET_IDENTITY_SCHEMA_VERSION = "kawaii_brainvision_dataset_identity_v1"
BEHAVIORAL_INPUT_IDENTITY_SCHEMA_VERSION = "kawaii_behavioral_input_identity_v1"
DATASET_SOURCE_MANIFEST_SCHEMA_VERSION = "kawaii_dataset_source_manifest_v1"
BRAINVISION_WORKING_LINEAGE_SCHEMA_VERSION = "kawaii_brainvision_working_lineage_v1"
BRAINVISION_FILENAME_CORRECTION_TRANSFORMATION = (
    "kawaii_brainvision_filename_reference_correction_v1"
)
BRAINVISION_WORKING_LINEAGE_FILENAME = "brainvision_working_lineage.json"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:/")
_BRAINVISION_ROLES = (
    "brainvision_header",
    "brainvision_marker",
    "brainvision_eeg",
)


class SourceIdentityError(ValueError):
    """Raised when an authoritative source identity cannot be constructed."""


@dataclass(frozen=True)
class SourceFileIdentity:
    """Immutable byte identity plus non-authoritative logical location metadata."""

    role: str
    logical_path: str
    size_bytes: int
    sha256: str
    schema_version: str = SOURCE_FILE_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_FILE_IDENTITY_SCHEMA_VERSION:
            raise SourceIdentityError(
                f"Unsupported source-file identity schema: {self.schema_version!r}."
            )
        _require_nonempty_token(self.role, "source role")
        _validate_logical_path(self.logical_path)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise SourceIdentityError("Source file size_bytes must be a non-negative integer.")
        _validate_sha256(self.sha256, "source file sha256")

    @property
    def content_fingerprint(self) -> str:
        """Return the direct SHA-256 identity of the file bytes."""

        return self.sha256

    @property
    def fingerprint(self) -> str:
        """Bind source role and bytes while excluding descriptive path metadata."""

        return canonical_sha256(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "logical_path": self.logical_path,
            "fingerprint": self.fingerprint,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_primitive())


@dataclass(frozen=True)
class BrainVisionDatasetIdentity:
    """Content-bound identity for one structurally linked BrainVision triplet."""

    header: SourceFileIdentity
    marker: SourceFileIdentity
    eeg: SourceFileIdentity
    linkage_status: str
    linkage_issues: tuple[str, ...]
    fingerprint: str
    schema_version: str = BRAINVISION_DATASET_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BRAINVISION_DATASET_IDENTITY_SCHEMA_VERSION:
            raise SourceIdentityError(
                f"Unsupported BrainVision identity schema: {self.schema_version!r}."
            )
        sources = self.sources
        _require_unique_roles(sources)
        roles = tuple(source.role for source in sources)
        if roles != _BRAINVISION_ROLES:
            raise SourceIdentityError(
                f"BrainVision identity roles must be {_BRAINVISION_ROLES!r}, got {roles!r}."
            )
        if self.linkage_status not in {
            "validated",
            "filename_references_require_correction",
        }:
            raise SourceIdentityError(
                f"Invalid BrainVision linkage status: {self.linkage_status!r}."
            )
        if not isinstance(self.linkage_issues, tuple) or any(
            not isinstance(issue, str) or not issue for issue in self.linkage_issues
        ):
            raise SourceIdentityError("BrainVision linkage_issues must be a tuple of strings.")
        if self.linkage_status == "validated" and self.linkage_issues:
            raise SourceIdentityError("Validated BrainVision identity cannot contain linkage issues.")
        if self.linkage_status == "filename_references_require_correction" and not self.linkage_issues:
            raise SourceIdentityError("Repairable BrainVision identity must describe its linkage issues.")
        expected = canonical_sha256(self.identity_payload())
        if self.fingerprint != expected:
            raise SourceIdentityError("BrainVision dataset fingerprint does not match its members.")

    @property
    def sources(self) -> tuple[SourceFileIdentity, SourceFileIdentity, SourceFileIdentity]:
        return self.header, self.marker, self.eeg

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "linkage_status": self.linkage_status,
            "linkage_issues": list(self.linkage_issues),
            "sources": [source.identity_payload() for source in self.sources],
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "linkage_status": self.linkage_status,
            "linkage_issues": list(self.linkage_issues),
            "sources": [source.to_primitive() for source in self.sources],
            "fingerprint": self.fingerprint,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_primitive())


@dataclass(frozen=True)
class BrainVisionWorkingLineage:
    """Immutable original-to-working identity for one corrected BrainVision set."""

    participant: str
    session: str
    recording_part: str
    transformation: str
    source_dataset: BrainVisionDatasetIdentity
    working_dataset: BrainVisionDatasetIdentity
    fingerprint: str
    schema_version: str = BRAINVISION_WORKING_LINEAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BRAINVISION_WORKING_LINEAGE_SCHEMA_VERSION:
            raise SourceIdentityError(
                f"Unsupported BrainVision working-lineage schema: {self.schema_version!r}."
            )
        _require_nonempty_token(self.participant, "participant")
        _require_nonempty_token(self.session, "session")
        _require_nonempty_token(self.recording_part, "recording_part")
        if self.transformation != BRAINVISION_FILENAME_CORRECTION_TRANSFORMATION:
            raise SourceIdentityError(
                f"Unsupported BrainVision working transformation: {self.transformation!r}."
            )
        if not isinstance(self.source_dataset, BrainVisionDatasetIdentity):
            raise SourceIdentityError("Working lineage source_dataset is malformed.")
        if not isinstance(self.working_dataset, BrainVisionDatasetIdentity):
            raise SourceIdentityError("Working lineage working_dataset is malformed.")
        if self.working_dataset.linkage_status != "validated":
            raise SourceIdentityError("Working BrainVision dataset linkage must be validated.")
        expected = canonical_sha256(self.identity_payload())
        if self.fingerprint != expected:
            raise SourceIdentityError("BrainVision working-lineage fingerprint is inconsistent.")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {
                "participant": self.participant,
                "session": self.session,
                "recording_part": self.recording_part,
            },
            "transformation": self.transformation,
            "source_dataset": self.source_dataset.identity_payload(),
            "working_dataset": self.working_dataset.identity_payload(),
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {
                "participant": self.participant,
                "session": self.session,
                "recording_part": self.recording_part,
            },
            "transformation": self.transformation,
            "source_dataset": self.source_dataset.to_primitive(),
            "working_dataset": self.working_dataset.to_primitive(),
            "fingerprint": self.fingerprint,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_primitive())


@dataclass(frozen=True)
class BehavioralInputIdentity:
    """Identity of the behavioral log and/or schedule available as source input."""

    behavioral_log: SourceFileIdentity | None
    schedule: SourceFileIdentity | None
    fingerprint: str
    schema_version: str = BEHAVIORAL_INPUT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BEHAVIORAL_INPUT_IDENTITY_SCHEMA_VERSION:
            raise SourceIdentityError(
                f"Unsupported behavioral-input identity schema: {self.schema_version!r}."
            )
        sources = self.sources
        if not sources:
            raise SourceIdentityError(
                "Behavioral input identity requires a behavioral log, a schedule, or both."
            )
        _require_unique_roles(sources)
        expected_roles = {"behavioral_log", "recognition_schedule"}
        unexpected = sorted(source.role for source in sources if source.role not in expected_roles)
        if unexpected:
            raise SourceIdentityError(f"Unexpected behavioral input role(s): {unexpected!r}.")
        expected = canonical_sha256(self.identity_payload())
        if self.fingerprint != expected:
            raise SourceIdentityError("Behavioral input fingerprint does not match its members.")

    @property
    def sources(self) -> tuple[SourceFileIdentity, ...]:
        return tuple(
            source
            for source in (self.behavioral_log, self.schedule)
            if source is not None
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [source.identity_payload() for source in self.sources],
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [source.to_primitive() for source in self.sources],
            "fingerprint": self.fingerprint,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_primitive())


@dataclass(frozen=True)
class DatasetSourceManifest:
    """Scoped, content-bound manifest of uniquely role-labelled source files."""

    participant: str
    session: str
    recording_part: str
    sources: tuple[SourceFileIdentity, ...]
    fingerprint: str
    schema_version: str = DATASET_SOURCE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_SOURCE_MANIFEST_SCHEMA_VERSION:
            raise SourceIdentityError(
                f"Unsupported dataset source manifest schema: {self.schema_version!r}."
            )
        _require_nonempty_token(self.participant, "participant")
        _require_nonempty_token(self.session, "session")
        _require_nonempty_token(self.recording_part, "recording_part")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise SourceIdentityError("Dataset source manifest requires at least one source.")
        if any(not isinstance(source, SourceFileIdentity) for source in self.sources):
            raise SourceIdentityError("Dataset source manifest contains a malformed source identity.")
        _require_unique_roles(self.sources)
        if tuple(sorted(self.sources, key=lambda source: source.role)) != self.sources:
            raise SourceIdentityError("Dataset source manifest sources must be sorted by role.")
        expected = canonical_sha256(self.identity_payload())
        if self.fingerprint != expected:
            raise SourceIdentityError("Dataset source manifest fingerprint does not match its content.")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {
                "participant": self.participant,
                "session": self.session,
                "recording_part": self.recording_part,
            },
            "sources": [source.identity_payload() for source in self.sources],
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {
                "participant": self.participant,
                "session": self.session,
                "recording_part": self.recording_part,
            },
            "sources": [source.to_primitive() for source in self.sources],
            "fingerprint": self.fingerprint,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_primitive())


def hash_file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a prefixed SHA-256 digest using bounded streaming reads."""

    digest, _ = _hash_file(path, chunk_size=chunk_size)
    return digest


def identify_source_file(
    path: str | Path,
    role: str,
    *,
    logical_path: str | Path | None = None,
    logical_root: str | Path | None = None,
    chunk_size: int = 1024 * 1024,
) -> SourceFileIdentity:
    """Identify one required file without making its machine path authoritative."""

    source_path = _require_file(path)
    clean_role = _require_nonempty_token(role, "source role")
    logical = _logical_path(source_path, logical_path=logical_path, logical_root=logical_root)
    digest, size = _hash_file(source_path, chunk_size=chunk_size)
    return SourceFileIdentity(
        role=clean_role,
        logical_path=logical,
        size_bytes=size,
        sha256=digest,
    )


def identify_brainvision_dataset(
    vhdr_path: str | Path,
    vmrk_path: str | Path,
    eeg_path: str | Path,
    *,
    logical_root: str | Path | None = None,
    allow_filename_reference_correction: bool = False,
) -> BrainVisionDatasetIdentity:
    """Identify a triplet, requiring linkage unless correction is explicit."""

    vhdr = _require_file(vhdr_path)
    vmrk = _require_file(vmrk_path)
    eeg = _require_file(eeg_path)
    _require_distinct_paths((vhdr, vmrk, eeg), "BrainVision triplet")
    _require_suffix(vhdr, ".vhdr", "BrainVision header")
    _require_suffix(vmrk, ".vmrk", "BrainVision marker")
    _require_suffix(eeg, ".eeg", "BrainVision binary")
    linkage_issues = _brainvision_linkage_issues(vhdr, vmrk, eeg)
    if linkage_issues and not allow_filename_reference_correction:
        raise SourceIdentityError(linkage_issues[0])
    if linkage_issues and len({vhdr.stem, vmrk.stem, eeg.stem}) != 1:
        raise SourceIdentityError(
            "Repairable BrainVision members must share one file stem."
        )
    linkage_status = (
        "filename_references_require_correction" if linkage_issues else "validated"
    )

    header = identify_source_file(vhdr, _BRAINVISION_ROLES[0], logical_root=logical_root)
    marker = identify_source_file(vmrk, _BRAINVISION_ROLES[1], logical_root=logical_root)
    binary = identify_source_file(eeg, _BRAINVISION_ROLES[2], logical_root=logical_root)
    payload = {
        "schema_version": BRAINVISION_DATASET_IDENTITY_SCHEMA_VERSION,
        "linkage_status": linkage_status,
        "linkage_issues": list(linkage_issues),
        "sources": [source.identity_payload() for source in (header, marker, binary)],
    }
    return BrainVisionDatasetIdentity(
        header=header,
        marker=marker,
        eeg=binary,
        linkage_status=linkage_status,
        linkage_issues=linkage_issues,
        fingerprint=canonical_sha256(payload),
    )


def build_brainvision_working_lineage(
    *,
    participant: str,
    session: str,
    recording_part: str,
    source_dataset: BrainVisionDatasetIdentity,
    working_dataset: BrainVisionDatasetIdentity,
    transformation: str = BRAINVISION_FILENAME_CORRECTION_TRANSFORMATION,
) -> BrainVisionWorkingLineage:
    """Build a deterministic corrected-working lineage record."""

    clean_participant = _require_nonempty_token(participant, "participant")
    clean_session = _require_nonempty_token(session, "session")
    clean_part = _require_nonempty_token(recording_part, "recording_part")
    payload = {
        "schema_version": BRAINVISION_WORKING_LINEAGE_SCHEMA_VERSION,
        "scope": {
            "participant": clean_participant,
            "session": clean_session,
            "recording_part": clean_part,
        },
        "transformation": transformation,
        "source_dataset": source_dataset.identity_payload(),
        "working_dataset": working_dataset.identity_payload(),
    }
    return BrainVisionWorkingLineage(
        participant=clean_participant,
        session=clean_session,
        recording_part=clean_part,
        transformation=transformation,
        source_dataset=source_dataset,
        working_dataset=working_dataset,
        fingerprint=canonical_sha256(payload),
    )


def brainvision_working_lineage_path(working_dir: str | Path) -> Path:
    """Return the stable lineage path for one subject/part working directory."""

    return Path(working_dir) / BRAINVISION_WORKING_LINEAGE_FILENAME


def write_brainvision_working_lineage(
    lineage: BrainVisionWorkingLineage,
    path: str | Path,
) -> Path:
    """Atomically persist lineage and verify the completed record."""

    if not isinstance(lineage, BrainVisionWorkingLineage):
        raise SourceIdentityError("lineage must be a BrainVisionWorkingLineage.")
    target = Path(path)
    _atomic_write_text(target, lineage.canonical_json() + "\n")
    persisted = read_brainvision_working_lineage(target)
    if persisted != lineage:
        raise SourceIdentityError("Persisted BrainVision working lineage did not round-trip.")
    return target


def read_brainvision_working_lineage(path: str | Path) -> BrainVisionWorkingLineage:
    """Read and validate a complete working-lineage record."""

    lineage_path = Path(path)
    if not lineage_path.exists():
        raise SourceIdentityError(f"BrainVision working lineage is missing: {lineage_path}")
    if not lineage_path.is_file():
        raise SourceIdentityError(f"BrainVision working lineage is not a file: {lineage_path}")
    try:
        payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceIdentityError(
            f"Could not read BrainVision working lineage {lineage_path}: {error}"
        ) from error
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "scope",
            "transformation",
            "source_dataset",
            "working_dataset",
            "fingerprint",
        },
        "BrainVision working lineage",
    )
    scope = payload["scope"]
    _require_exact_keys(
        scope,
        {"participant", "session", "recording_part"},
        "BrainVision working lineage scope",
    )
    return BrainVisionWorkingLineage(
        participant=scope["participant"],
        session=scope["session"],
        recording_part=scope["recording_part"],
        transformation=payload["transformation"],
        source_dataset=_brainvision_identity_from_primitive(payload["source_dataset"]),
        working_dataset=_brainvision_identity_from_primitive(payload["working_dataset"]),
        fingerprint=payload["fingerprint"],
        schema_version=payload["schema_version"],
    )


def validate_brainvision_working_lineage(
    lineage: BrainVisionWorkingLineage,
    *,
    participant: str,
    session: str,
    recording_part: str,
    source_dataset: BrainVisionDatasetIdentity,
    working_dataset: BrainVisionDatasetIdentity,
) -> None:
    """Fail closed unless scope and both current dataset identities match."""

    if not isinstance(lineage, BrainVisionWorkingLineage):
        raise SourceIdentityError("lineage must be a BrainVisionWorkingLineage.")
    expected_scope = (participant, session, recording_part)
    actual_scope = (lineage.participant, lineage.session, lineage.recording_part)
    if actual_scope != expected_scope:
        raise SourceIdentityError(
            f"BrainVision working lineage scope mismatch: stored={actual_scope!r}, expected={expected_scope!r}."
        )
    if lineage.source_dataset.fingerprint != source_dataset.fingerprint:
        raise SourceIdentityError(
            "Current original BrainVision dataset does not match working lineage."
        )
    if lineage.working_dataset.fingerprint != working_dataset.fingerprint:
        raise SourceIdentityError(
            "Current working BrainVision dataset does not match working lineage."
        )


def identify_behavioral_inputs(
    *,
    behavioral_log_path: str | Path | None = None,
    schedule_path: str | Path | None = None,
    logical_root: str | Path | None = None,
) -> BehavioralInputIdentity:
    """Identify the available behavioral log and/or recognition schedule."""

    if behavioral_log_path is None and schedule_path is None:
        raise SourceIdentityError(
            "At least one behavioral log or schedule path is required."
        )
    provided = [path for path in (behavioral_log_path, schedule_path) if path is not None]
    if len(provided) == 2:
        _require_distinct_paths(tuple(Path(path) for path in provided), "behavioral inputs")
    behavioral_log = (
        identify_source_file(
            behavioral_log_path,
            "behavioral_log",
            logical_root=logical_root,
        )
        if behavioral_log_path is not None
        else None
    )
    schedule = (
        identify_source_file(
            schedule_path,
            "recognition_schedule",
            logical_root=logical_root,
        )
        if schedule_path is not None
        else None
    )
    sources = tuple(source for source in (behavioral_log, schedule) if source is not None)
    payload = {
        "schema_version": BEHAVIORAL_INPUT_IDENTITY_SCHEMA_VERSION,
        "sources": [source.identity_payload() for source in sources],
    }
    return BehavioralInputIdentity(
        behavioral_log=behavioral_log,
        schedule=schedule,
        fingerprint=canonical_sha256(payload),
    )


def source_file_identity_from_primitive(value: Any) -> SourceFileIdentity:
    """Reconstruct and validate a serialized source-file identity."""

    return _source_identity_from_primitive(value)


def brainvision_dataset_identity_from_primitive(value: Any) -> BrainVisionDatasetIdentity:
    """Reconstruct and validate a serialized BrainVision dataset identity."""

    return _brainvision_identity_from_primitive(value)


def behavioral_input_identity_from_primitive(value: Any) -> BehavioralInputIdentity:
    """Reconstruct and validate a serialized behavioral-input identity."""

    _require_exact_keys(
        value,
        {"schema_version", "sources", "fingerprint"},
        "behavioral input identity",
    )
    sources = value["sources"]
    if not isinstance(sources, list) or not sources:
        raise SourceIdentityError("Behavioral input identity requires at least one source.")
    parsed = tuple(_source_identity_from_primitive(source) for source in sources)
    by_role = {source.role: source for source in parsed}
    if len(by_role) != len(parsed):
        raise SourceIdentityError("Behavioral input identity contains duplicate source roles.")
    unexpected = sorted(set(by_role) - {"behavioral_log", "recognition_schedule"})
    if unexpected:
        raise SourceIdentityError(f"Unexpected behavioral input role(s): {unexpected!r}.")
    return BehavioralInputIdentity(
        behavioral_log=by_role.get("behavioral_log"),
        schedule=by_role.get("recognition_schedule"),
        fingerprint=value["fingerprint"],
        schema_version=value["schema_version"],
    )


def build_source_manifest(
    *,
    participant: str,
    session: str | int,
    recording_part: str,
    sources: Sequence[SourceFileIdentity] = (),
    brainvision: BrainVisionDatasetIdentity | None = None,
    behavioral_inputs: BehavioralInputIdentity | None = None,
) -> DatasetSourceManifest:
    """Compose source identities with descriptive participant/session/part scope."""

    combined = list(sources)
    if brainvision is not None:
        if not isinstance(brainvision, BrainVisionDatasetIdentity):
            raise SourceIdentityError("brainvision must be a BrainVisionDatasetIdentity.")
        combined.extend(brainvision.sources)
    if behavioral_inputs is not None:
        if not isinstance(behavioral_inputs, BehavioralInputIdentity):
            raise SourceIdentityError("behavioral_inputs must be a BehavioralInputIdentity.")
        combined.extend(behavioral_inputs.sources)
    if not combined:
        raise SourceIdentityError("Dataset source manifest requires at least one source.")
    if any(not isinstance(source, SourceFileIdentity) for source in combined):
        raise SourceIdentityError("Dataset source manifest contains a malformed source identity.")
    _require_unique_roles(combined)
    ordered = tuple(sorted(combined, key=lambda source: source.role))
    clean_participant = _require_nonempty_token(participant, "participant")
    clean_session = _require_nonempty_token(str(session), "session")
    clean_part = _require_nonempty_token(recording_part, "recording_part")
    payload = {
        "schema_version": DATASET_SOURCE_MANIFEST_SCHEMA_VERSION,
        "scope": {
            "participant": clean_participant,
            "session": clean_session,
            "recording_part": clean_part,
        },
        "sources": [source.identity_payload() for source in ordered],
    }
    return DatasetSourceManifest(
        participant=clean_participant,
        session=clean_session,
        recording_part=clean_part,
        sources=ordered,
        fingerprint=canonical_sha256(payload),
    )


def _hash_file(path: str | Path, *, chunk_size: int) -> tuple[str, int]:
    source_path = _require_file(path)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise SourceIdentityError("Hash chunk_size must be a positive integer.")
    digest = hashlib.sha256()
    size = 0
    try:
        with source_path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise SourceIdentityError(f"Could not read source file {source_path}: {error}") from error
    return "sha256:" + digest.hexdigest(), size


def _require_file(path: str | Path) -> Path:
    source_path = Path(path)
    if not source_path.exists():
        raise SourceIdentityError(f"Required source file does not exist: {source_path}")
    if not source_path.is_file():
        raise SourceIdentityError(f"Source identity requires a file, got: {source_path}")
    return source_path


def _logical_path(
    source_path: Path,
    *,
    logical_path: str | Path | None,
    logical_root: str | Path | None,
) -> str:
    if logical_path is not None and logical_root is not None:
        raise SourceIdentityError("Specify logical_path or logical_root, not both.")
    if logical_path is not None:
        return _normalise_logical_path(logical_path)
    if logical_root is not None:
        try:
            relative = source_path.resolve().relative_to(Path(logical_root).resolve())
        except ValueError as error:
            raise SourceIdentityError(
                f"Source file {source_path} is outside logical_root {logical_root}."
            ) from error
        return _normalise_logical_path(relative)
    return _normalise_logical_path(source_path.name)


def _normalise_logical_path(value: str | Path) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or text.startswith("/") or _WINDOWS_ABSOLUTE_PATTERN.match(text):
        raise SourceIdentityError(
            f"logical_path must be a non-empty relative path, got {value!r}."
        )
    path = PurePosixPath(text)
    if any(part in ("", ".", "..") for part in path.parts):
        raise SourceIdentityError(f"logical_path contains an unsafe segment: {value!r}.")
    return path.as_posix()


def _validate_logical_path(value: str) -> None:
    if _normalise_logical_path(value) != value:
        raise SourceIdentityError(f"logical_path is not normalized: {value!r}.")


def _brainvision_linkage_issues(vhdr: Path, vmrk: Path, eeg: Path) -> tuple[str, ...]:
    try:
        header_common = _read_brainvision_common_infos(vhdr)
        marker_common = _read_brainvision_common_infos(vmrk)
    except OSError as error:
        raise SourceIdentityError(f"Could not read BrainVision text headers: {error}") from error
    required = (
        ("header DataFile", header_common.get("DataFile"), eeg.name),
        ("header MarkerFile", header_common.get("MarkerFile"), vmrk.name),
        ("marker DataFile", marker_common.get("DataFile"), eeg.name),
    )
    issues = []
    for label, actual, expected in required:
        if not actual:
            raise SourceIdentityError(f"BrainVision {label} reference is missing.")
        if _reference_basename(actual) != expected:
            issues.append(
                f"BrainVision {label} references {_reference_basename(actual)!r}, expected {expected!r}."
            )
    return tuple(issues)


def _read_brainvision_common_infos(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    common: dict[str, str] = {}
    in_common = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_common = line == "[Common Infos]"
            continue
        if in_common and "=" in line:
            key, value = line.split("=", 1)
            common[key.strip()] = value.strip()
    return common


def _reference_basename(value: str) -> str:
    return PurePosixPath(str(value).replace("\\", "/")).name


def _brainvision_identity_from_primitive(value: Any) -> BrainVisionDatasetIdentity:
    _require_exact_keys(
        value,
        {
            "schema_version",
            "linkage_status",
            "linkage_issues",
            "sources",
            "fingerprint",
        },
        "BrainVision dataset identity",
    )
    sources = value["sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise SourceIdentityError("BrainVision dataset identity requires exactly three sources.")
    parsed = tuple(_source_identity_from_primitive(source) for source in sources)
    issues = value["linkage_issues"]
    if not isinstance(issues, list):
        raise SourceIdentityError("BrainVision linkage_issues must be a list.")
    return BrainVisionDatasetIdentity(
        header=parsed[0],
        marker=parsed[1],
        eeg=parsed[2],
        linkage_status=value["linkage_status"],
        linkage_issues=tuple(issues),
        fingerprint=value["fingerprint"],
        schema_version=value["schema_version"],
    )


def _source_identity_from_primitive(value: Any) -> SourceFileIdentity:
    _require_exact_keys(
        value,
        {
            "schema_version",
            "role",
            "size_bytes",
            "sha256",
            "logical_path",
            "fingerprint",
        },
        "source file identity",
    )
    identity = SourceFileIdentity(
        role=value["role"],
        logical_path=value["logical_path"],
        size_bytes=value["size_bytes"],
        sha256=value["sha256"],
        schema_version=value["schema_version"],
    )
    if value["fingerprint"] != identity.fingerprint:
        raise SourceIdentityError("Source file identity fingerprint is inconsistent.")
    return identity


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise SourceIdentityError(f"{label} must be a mapping.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SourceIdentityError(
            f"{label} fields are invalid; missing={missing!r}, extra={extra!r}."
        )


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
    except OSError as error:
        raise SourceIdentityError(f"Could not write BrainVision lineage {path}: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _require_unique_roles(sources: Iterable[SourceFileIdentity]) -> None:
    roles = [source.role for source in sources]
    duplicates = sorted({role for role in roles if roles.count(role) > 1})
    if duplicates:
        raise SourceIdentityError(f"Duplicate source role(s): {duplicates!r}.")


def _require_distinct_paths(paths: Sequence[Path], label: str) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise SourceIdentityError(f"{label} requires distinct source files.")


def _require_suffix(path: Path, suffix: str, label: str) -> None:
    if path.suffix.lower() != suffix:
        raise SourceIdentityError(f"{label} must use the {suffix} suffix: {path}")


def _require_nonempty_token(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceIdentityError(f"{label} must be a non-empty string.")
    clean = value.strip()
    if value != clean:
        raise SourceIdentityError(f"{label} must not contain surrounding whitespace.")
    return clean


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise SourceIdentityError(f"{label} must be a prefixed lowercase SHA-256 digest.")
