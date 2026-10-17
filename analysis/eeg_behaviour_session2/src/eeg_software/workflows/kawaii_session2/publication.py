"""Versioned, manifest-final transactional publication semantics.

Artifacts are copied into an immutable generation directory.  The only
canonical finality record (``current.json``) is atomically replaced after that
entire generation validates.  This is semantic atomicity, not an assertion of
atomic multi-file replacement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import threading
import time
from typing import Any, Iterator
from uuid import uuid4


PUBLICATION_MANIFEST_SCHEMA = "kawaii_publication_manifest_v1"
PUBLICATION_LOCK_SCHEMA = "kawaii_publication_lock_v1"
DEFAULT_PUBLICATION_LOCK_TIMEOUT_SECONDS = 10.0


class PublicationError(RuntimeError):
    """Base error for fail-closed publication or consumption."""


class PublicationValidationError(PublicationError):
    """Candidate or current publication failed validation."""


class PublicationConflictError(PublicationError):
    """Canonical generation changed after the candidate observed it."""


class PublicationCancelled(PublicationError):
    """Cancellation was observed at a safe publication boundary."""


class PublicationLockTimeoutError(PublicationError):
    """The bounded per-scope publication lock could not be acquired."""


class PublicationState(str, Enum):
    STAGING = "STAGING"
    READY = "READY"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class PublicationScope:
    stage: str
    session: str = "session2"
    participant: str | None = None
    recording_part: str | None = None
    branch: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "stage": self.stage,
            "session": self.session,
            "participant": self.participant,
            "recording_part": self.recording_part,
            "branch": self.branch,
        }

    @property
    def identity(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class PublishedArtifact:
    label: str
    path: Path
    sha256: str
    size_bytes: int
    required: bool


@dataclass(frozen=True)
class ValidatedPublication:
    scope: PublicationScope
    run_id: str
    generation_id: str
    predecessor_generation_id: str | None
    context_reference: str
    identity_references: Mapping[str, str]
    artifacts: tuple[PublishedArtifact, ...]
    manifest_path: Path

    def artifact_path(self, label: str) -> Path | None:
        return next((artifact.path for artifact in self.artifacts if artifact.label == label), None)


def validate_publication_scientific_context(
    publication: ValidatedPublication,
    expected_scientific_fingerprint: str,
) -> None:
    """Validate stage science independently of cohort/config namespace changes."""

    observed: set[str] = set()
    provenance_count = 0
    for artifact in publication.artifacts:
        if not artifact.label.endswith("__provenance"):
            continue
        provenance_count += 1
        try:
            value = json.loads(artifact.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PublicationValidationError(
                f"Published provenance cannot establish scientific compatibility: {artifact.label}: {error}"
            ) from error
        authority = value.get("configuration_authority")
        if not isinstance(authority, Mapping):
            raise PublicationValidationError(
                f"Published provenance lacks configuration authority: {artifact.label}."
            )
        fingerprint = authority.get("scientific_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise PublicationValidationError(
                f"Published provenance lacks a scientific fingerprint: {artifact.label}."
            )
        observed.add(fingerprint)
    if provenance_count == 0:
        raise PublicationValidationError(
            "Published stage has no provenance artifacts for scientific compatibility validation."
        )
    if observed != {str(expected_scientific_fingerprint)}:
        raise PublicationValidationError(
            "Published stage scientific context is stale or internally inconsistent."
        )


def publication_scope_root(canonical_directory: str | Path, scope: PublicationScope) -> Path:
    return Path(canonical_directory) / ".kawaii-publication" / scope.identity


def current_manifest_path(canonical_directory: str | Path, scope: PublicationScope) -> Path:
    return publication_scope_root(canonical_directory, scope) / "current.json"


def publish_artifact_set(
    canonical_directory: str | Path,
    scope: PublicationScope,
    *,
    run_id: str,
    context_reference: str,
    required_artifacts: Mapping[str, str | Path],
    optional_artifacts: Mapping[str, str | Path] | None = None,
    expected_predecessor_generation_id: str | None | object = ...,
    validation_only: bool = False,
    cancelled: Callable[[], bool] | None = None,
    lock_timeout_s: float = DEFAULT_PUBLICATION_LOCK_TIMEOUT_SECONDS,
    fault_injector: Callable[[str], None] | None = None,
    identity_references: Mapping[str, str] | None = None,
) -> ValidatedPublication:
    """Publish one complete immutable generation, with the manifest last."""

    if validation_only:
        raise PublicationValidationError("Validation-only execution cannot publish canonical finality.")
    if not run_id or not context_reference:
        raise PublicationValidationError("Publication run and context identities must be non-empty.")
    if not required_artifacts:
        raise PublicationValidationError("A publication requires at least one required artifact.")
    cancellation = cancelled or (lambda: False)
    inject = fault_injector or (lambda _boundary: None)
    _check_cancelled(cancellation)
    inject("before_staging")

    root = publication_scope_root(canonical_directory, scope)
    root.mkdir(parents=True, exist_ok=True)
    initial_generation = _read_generation_id(current_manifest_path(canonical_directory, scope))
    expected = initial_generation if expected_predecessor_generation_id is ... else expected_predecessor_generation_id
    staging = root / "staging" / f"{_safe_component(run_id)}--{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    ownership = staging / "ownership.json"
    _write_json_atomic(ownership, {
        "schema": "kawaii_publication_staging_ownership_v1",
        "run_id": run_id,
        "scope_identity": scope.identity,
        "directory": str(staging.resolve()),
    })
    staged_files: list[Path] = [ownership]
    try:
        artifact_rows: list[dict[str, Any]] = []
        combined = [(label, path, True) for label, path in required_artifacts.items()]
        combined.extend((label, path, False) for label, path in (optional_artifacts or {}).items())
        if len({label for label, _, _ in combined}) != len(combined):
            raise PublicationValidationError("Required and optional publication labels must be unique.")
        for index, (label, source_value, required) in enumerate(sorted(combined)):
            _check_cancelled(cancellation)
            source = Path(source_value)
            if not source.is_file():
                if required:
                    raise PublicationValidationError(f"Required publication artifact is missing: {label}: {source}")
                continue
            if source.stat().st_size <= 0:
                raise PublicationValidationError(f"Publication artifact is empty: {label}: {source}")
            destination = staging / f"{index:03d}--{_safe_component(label)}--{_safe_component(source.name)}"
            staged_files.append(destination)
            shutil.copyfile(source, destination)
            inject(f"artifact_staged:{label}")
            if _sha256_file(source) != _sha256_file(destination):
                raise PublicationValidationError(f"Staged artifact identity changed during copy: {label}")
            artifact_rows.append({
                "label": label,
                "relative_path": destination.name,
                "sha256": _sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "required": required,
                "context_reference": context_reference,
                "run_id": run_id,
            })
        if {row["label"] for row in artifact_rows if row["required"]} != set(required_artifacts):
            raise PublicationValidationError("The staged generation is missing required artifacts.")
        references = dict(sorted((str(key), str(value)) for key, value in (identity_references or {}).items()))
        generation_payload = {
            "scope": scope.as_dict(),
            "scope_identity": scope.identity,
            "context_reference": context_reference,
            "identity_references": references,
            "artifacts": artifact_rows,
        }
        generation_id = _canonical_sha256(generation_payload)
        ready_manifest = {
            "schema": PUBLICATION_MANIFEST_SCHEMA,
            "state": PublicationState.READY.value,
            "scope": scope.as_dict(),
            "scope_identity": scope.identity,
            "run_id": run_id,
            "context_reference": context_reference,
            "identity_references": references,
            "generation_id": generation_id,
            "predecessor_generation_id": expected,
            "validation_only": False,
            "artifacts": artifact_rows,
        }
        ready_path = staging / "generation.json"
        _write_json_atomic(ready_path, ready_manifest)
        staged_files.append(ready_path)
        _validate_generation_directory(staging, ready_manifest)
        inject("after_validation_before_promotion")
        _check_cancelled(cancellation)

        with _publication_lock(root, scope, timeout_s=lock_timeout_s):
            current_path = root / "current.json"
            actual = _read_generation_id(current_path)
            if actual == generation_id:
                existing = validate_current_publication(canonical_directory, scope, expected_context_reference=context_reference)
                _remove_owned_staging(staging, staged_files)
                return existing
            if actual != expected:
                raise PublicationConflictError(
                    f"Publication generation conflict for {scope.as_dict()}: expected {expected!r}, found {actual!r}."
                )
            _check_cancelled(cancellation)
            generation_dir = root / "generations" / generation_id
            generation_dir.parent.mkdir(parents=True, exist_ok=True)
            if generation_dir.exists():
                _validate_generation_directory(generation_dir, ready_manifest)
                _remove_owned_staging(staging, staged_files)
            else:
                os.replace(staging, generation_dir)
            inject("after_generation_promotion_before_finality")
            _check_cancelled(cancellation)
            final_manifest = dict(ready_manifest)
            final_manifest.update({
                "state": PublicationState.PUBLISHED.value,
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
                "generation_manifest": str((generation_dir / "generation.json").resolve()),
            })
            final_artifacts = []
            for row in artifact_rows:
                final_row = dict(row)
                final_row["path"] = str((generation_dir / row["relative_path"]).resolve())
                final_artifacts.append(final_row)
            final_manifest["artifacts"] = final_artifacts
            inject("before_finality_publication")
            _write_json_atomic(current_path, final_manifest)
            inject("after_finality_publication")
            return validate_current_publication(
                canonical_directory,
                scope,
                expected_context_reference=context_reference,
                expected_generation_id=generation_id,
                expected_run_id=run_id,
            )
    except Exception:
        if staging.exists():
            _remove_owned_staging(staging, staged_files)
        raise


def validate_current_publication(
    canonical_directory: str | Path,
    scope: PublicationScope,
    *,
    expected_context_reference: str | None = None,
    expected_generation_id: str | None = None,
    expected_run_id: str | None = None,
    expected_identity_references: Mapping[str, str] | None = None,
) -> ValidatedPublication:
    path = current_manifest_path(canonical_directory, scope)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationValidationError(f"Canonical publication finality is missing or invalid: {path}: {error}") from error
    if value.get("schema") != PUBLICATION_MANIFEST_SCHEMA or value.get("state") != PublicationState.PUBLISHED.value:
        raise PublicationValidationError(f"Canonical publication is not final/published: {path}")
    if value.get("validation_only") is not False or value.get("scope") != scope.as_dict() or value.get("scope_identity") != scope.identity:
        raise PublicationValidationError(f"Canonical publication scope/finality mismatch: {path}")
    if expected_context_reference is not None and value.get("context_reference") != expected_context_reference:
        raise PublicationValidationError("Canonical publication context is stale or mismatched.")
    if expected_generation_id is not None and value.get("generation_id") != expected_generation_id:
        raise PublicationValidationError("Canonical publication generation is stale or mismatched.")
    if expected_run_id is not None and value.get("run_id") != expected_run_id:
        raise PublicationValidationError("Canonical publication run identity is mismatched.")
    references = value.get("identity_references", {})
    if not isinstance(references, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in references.items()
    ):
        raise PublicationValidationError("Canonical publication identity references are invalid.")
    if expected_identity_references is not None and references != dict(expected_identity_references):
        raise PublicationValidationError("Canonical publication dependency identities are stale or mismatched.")
    generation_dir = publication_scope_root(canonical_directory, scope) / "generations" / str(value.get("generation_id"))
    artifacts: list[PublishedArtifact] = []
    rows = value.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise PublicationValidationError("Canonical publication declares no artifacts.")
    labels: set[str] = set()
    for row in rows:
        label = row.get("label")
        if not isinstance(label, str) or label in labels:
            raise PublicationValidationError("Canonical publication has invalid/duplicate artifact labels.")
        labels.add(label)
        artifact_path = Path(str(row.get("path", "")))
        if artifact_path.parent.resolve() != generation_dir.resolve():
            raise PublicationValidationError(f"Published artifact escapes its immutable generation: {label}")
        if row.get("context_reference") != value.get("context_reference") or row.get("run_id") != value.get("run_id"):
            raise PublicationValidationError(f"Published artifact identity is mixed: {label}")
        if not artifact_path.is_file() or artifact_path.stat().st_size != row.get("size_bytes"):
            raise PublicationValidationError(f"Published artifact is missing/truncated: {label}")
        digest = _sha256_file(artifact_path)
        if digest != row.get("sha256"):
            raise PublicationValidationError(f"Published artifact bytes changed: {label}")
        artifacts.append(PublishedArtifact(label, artifact_path, digest, artifact_path.stat().st_size, bool(row.get("required"))))
    generation_payload = {
        "scope": value["scope"],
        "scope_identity": value["scope_identity"],
        "context_reference": value["context_reference"],
        "identity_references": references,
        "artifacts": [{key: row[key] for key in ("label", "relative_path", "sha256", "size_bytes", "required", "context_reference", "run_id")} for row in rows],
    }
    if _canonical_sha256(generation_payload) != value.get("generation_id"):
        raise PublicationValidationError("Canonical publication generation identity does not match its artifact set.")
    return ValidatedPublication(
        scope=scope,
        run_id=str(value["run_id"]),
        generation_id=str(value["generation_id"]),
        predecessor_generation_id=value.get("predecessor_generation_id"),
        context_reference=str(value["context_reference"]),
        identity_references=references,
        artifacts=tuple(artifacts),
        manifest_path=path,
    )


def validate_publication_generation(
    canonical_directory: str | Path,
    scope: PublicationScope,
    generation_id: str,
    *,
    expected_context_reference: str | None = None,
) -> ValidatedPublication:
    """Validate one immutable generation without consulting ``current.json``.

    This is intentionally separate from :func:`validate_current_publication`.
    Ordinary consumers must continue to resolve canonical state through the
    current pointer; the two-pass reviewed-ICA route uses this function only
    after a solution-bound human decision has selected an exact generation.
    """

    if not isinstance(generation_id, str) or re.fullmatch(r"[0-9a-f]{64}", generation_id) is None:
        raise PublicationValidationError("Immutable publication generation ID must be a SHA-256 hex digest.")
    generation_dir = publication_scope_root(canonical_directory, scope) / "generations" / generation_id
    path = generation_dir / "generation.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationValidationError(
            f"Immutable publication generation is missing or invalid: {path}: {error}"
        ) from error
    if value.get("schema") != PUBLICATION_MANIFEST_SCHEMA or value.get("state") != PublicationState.READY.value:
        raise PublicationValidationError(f"Immutable publication generation is not ready: {path}")
    if value.get("validation_only") is not False or value.get("scope") != scope.as_dict() or value.get("scope_identity") != scope.identity:
        raise PublicationValidationError(f"Immutable publication generation scope/finality mismatch: {path}")
    if value.get("generation_id") != generation_id:
        raise PublicationValidationError("Immutable publication generation identity is mismatched.")
    if expected_context_reference is not None and value.get("context_reference") != expected_context_reference:
        raise PublicationValidationError("Immutable publication generation context is stale or mismatched.")
    references = value.get("identity_references", {})
    if not isinstance(references, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in references.items()
    ):
        raise PublicationValidationError("Immutable publication dependency identities are invalid.")
    rows = value.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise PublicationValidationError("Immutable publication generation declares no artifacts.")
    labels: set[str] = set()
    artifacts: list[PublishedArtifact] = []
    for row in rows:
        label = row.get("label")
        if not isinstance(label, str) or label in labels:
            raise PublicationValidationError("Immutable publication generation has invalid/duplicate artifact labels.")
        labels.add(label)
        relative_path = row.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise PublicationValidationError(f"Immutable publication artifact has an invalid path: {label}")
        artifact_path = generation_dir / relative_path
        if artifact_path.parent.resolve() != generation_dir.resolve():
            raise PublicationValidationError(f"Immutable publication artifact escapes its generation: {label}")
        if row.get("context_reference") != value.get("context_reference") or row.get("run_id") != value.get("run_id"):
            raise PublicationValidationError(f"Immutable publication artifact identity is mixed: {label}")
        if not artifact_path.is_file() or artifact_path.stat().st_size != row.get("size_bytes"):
            raise PublicationValidationError(f"Immutable publication artifact is missing/truncated: {label}")
        digest = _sha256_file(artifact_path)
        if digest != row.get("sha256"):
            raise PublicationValidationError(f"Immutable publication artifact bytes changed: {label}")
        artifacts.append(PublishedArtifact(label, artifact_path, digest, artifact_path.stat().st_size, bool(row.get("required"))))
    generation_payload = {
        "scope": value["scope"],
        "scope_identity": value["scope_identity"],
        "context_reference": value["context_reference"],
        "identity_references": references,
        "artifacts": [
            {key: row[key] for key in ("label", "relative_path", "sha256", "size_bytes", "required", "context_reference", "run_id")}
            for row in rows
        ],
    }
    if _canonical_sha256(generation_payload) != generation_id:
        raise PublicationValidationError("Immutable publication generation identity does not match its artifact set.")
    return ValidatedPublication(
        scope=scope,
        run_id=str(value["run_id"]),
        generation_id=generation_id,
        predecessor_generation_id=value.get("predecessor_generation_id"),
        context_reference=str(value["context_reference"]),
        identity_references=references,
        artifacts=tuple(artifacts),
        manifest_path=path,
    )


def validate_immutable_participant_publication_chain(
    canonical_directory: str | Path,
    publication: ValidatedPublication,
) -> tuple[ValidatedPublication, ...]:
    """Validate exact immutable participant ancestors, never current pointers."""

    if publication.scope.participant is None or publication.scope.recording_part is None:
        raise PublicationValidationError(
            "Immutable participant publication-chain validation requires participant and recording-part scope."
        )
    validated: list[ValidatedPublication] = []
    visited: set[tuple[str, str]] = set()

    def visit(value: ValidatedPublication) -> None:
        marker = (value.scope.stage, value.generation_id)
        if marker in visited:
            return
        visited.add(marker)
        for key, upstream_generation_id in value.identity_references.items():
            prefix = "publication:"
            if not key.startswith(prefix):
                continue
            fields = key.split(":", 2)
            if len(fields) != 3 or fields[1] not in {"gate3a", "gate3b", "gate3c"}:
                raise PublicationValidationError(
                    f"Unsupported participant publication dependency reference: {key!r}."
                )
            upstream_scope = PublicationScope(
                stage=fields[1],
                session=value.scope.session,
                participant=value.scope.participant,
                recording_part=value.scope.recording_part,
            )
            if fields[2] != upstream_scope.identity:
                raise PublicationValidationError(
                    f"Participant publication dependency scope identity is mismatched: {key!r}."
                )
            upstream = validate_publication_generation(
                canonical_directory,
                upstream_scope,
                upstream_generation_id,
                expected_context_reference=value.context_reference,
            )
            validated.append(upstream)
            visit(upstream)

    visit(publication)
    return tuple(validated)


def validate_participant_publication_chain(
    canonical_directory: str | Path,
    publication: ValidatedPublication,
    *,
    immutable_upstream_canonical_directory: str | Path | None = None,
) -> tuple[ValidatedPublication, ...]:
    """Validate every current participant-stage generation bound upstream.

    Participant Gate 3/4 publications share one part directory and bind their
    generated predecessors using ``publication:<stage>:<scope-id>`` identity
    references.  A downstream manifest is not coherent merely because its own
    bytes still validate: each referenced generation must also remain the
    current generation for the same participant/part and resolved context.
    """

    if publication.scope.participant is None or publication.scope.recording_part is None:
        raise PublicationValidationError(
            "Participant publication-chain validation requires participant and recording-part scope."
        )
    canonical = Path(canonical_directory)
    immutable_upstream = _validate_split_publication_roots(
        canonical,
        immutable_upstream_canonical_directory,
    )
    expected_manifest = (
        publication_scope_root(canonical, publication.scope)
        / "generations"
        / publication.generation_id
        / "generation.json"
    )
    expected_current = current_manifest_path(canonical, publication.scope)
    if publication.manifest_path.resolve() not in {
        expected_manifest.resolve(),
        expected_current.resolve(),
    }:
        raise PublicationValidationError(
            "Participant publication was not validated from the supplied canonical directory."
        )

    validated: list[ValidatedPublication] = []
    visited: set[tuple[str, str, Path]] = set()

    def visit(
        value: ValidatedPublication,
        value_directory: Path,
        *,
        exact_lineage: bool,
    ) -> set[tuple[str, str]]:
        marker = (value.scope.stage, value.generation_id, value_directory.resolve())
        if marker in visited:
            return {(value.scope.stage, value.generation_id)}
        visited.add(marker)
        lineage = {(value.scope.stage, value.generation_id)}
        for key, generation_id in value.identity_references.items():
            if key.startswith("immutable-publication:"):
                fields = key.split(":", 2)
                if len(fields) != 3 or fields[1] not in {"gate3a", "gate3b", "gate3c"}:
                    raise PublicationValidationError(
                        f"Unsupported immutable participant publication dependency reference: {key!r}."
                    )
                upstream_scope = PublicationScope(
                    stage=fields[1],
                    session=value.scope.session,
                    participant=value.scope.participant,
                    recording_part=value.scope.recording_part,
                )
                if fields[2] != upstream_scope.identity:
                    raise PublicationValidationError(
                        f"Immutable participant publication dependency scope identity is mismatched: {key!r}."
                    )
                upstream_directory = (
                    immutable_upstream
                    if immutable_upstream is not None and fields[1] in {"gate3a", "gate3b"}
                    else canonical
                )
                upstream = validate_publication_generation(
                    upstream_directory,
                    upstream_scope,
                    generation_id,
                )
                validated.append(upstream)
                ancestors = visit(
                    upstream,
                    upstream_directory,
                    exact_lineage=True,
                )
                if fields[1] == "gate3b":
                    gate3a_scope = PublicationScope(
                        stage="gate3a",
                        session=value.scope.session,
                        participant=value.scope.participant,
                        recording_part=value.scope.recording_part,
                    )
                    direct_gate3a = value.identity_references.get(
                        f"immutable-publication:gate3a:{gate3a_scope.identity}"
                    )
                    if direct_gate3a is not None and not any(
                        ancestor_stage == "gate3a" and ancestor_generation == direct_gate3a
                        for ancestor_stage, ancestor_generation in ancestors
                    ):
                        raise PublicationValidationError(
                            "Immutable Gate 3A and Gate 3B publication references do not describe one lineage."
                        )
                lineage.update(ancestors)
                continue
            if not key.startswith("publication:"):
                continue
            fields = key.split(":", 2)
            if len(fields) != 3 or fields[1] not in {"gate3a", "gate3b", "gate3c"}:
                raise PublicationValidationError(
                    f"Unsupported participant publication dependency reference: {key!r}."
                )
            upstream_scope = PublicationScope(
                stage=fields[1],
                session=value.scope.session,
                participant=value.scope.participant,
                recording_part=value.scope.recording_part,
            )
            if fields[2] != upstream_scope.identity:
                raise PublicationValidationError(
                    f"Participant publication dependency scope identity is mismatched: {key!r}."
                )
            if exact_lineage:
                upstream = validate_publication_generation(
                    value_directory,
                    upstream_scope,
                    generation_id,
                    expected_context_reference=value.context_reference,
                )
            else:
                upstream = validate_current_publication(
                    value_directory,
                    upstream_scope,
                    expected_context_reference=value.context_reference,
                    expected_generation_id=generation_id,
                )
            validated.append(upstream)
            lineage.update(
                visit(
                    upstream,
                    value_directory,
                    exact_lineage=exact_lineage,
                )
            )
        return lineage

    visit(publication, canonical, exact_lineage=False)
    unique: list[ValidatedPublication] = []
    emitted: set[tuple[str, str, Path]] = set()
    for item in validated:
        item_marker = (item.scope.stage, item.generation_id, item.manifest_path.resolve())
        if item_marker not in emitted:
            unique.append(item)
            emitted.add(item_marker)
    return tuple(unique)


def _validate_split_publication_roots(
    canonical_directory: Path,
    immutable_upstream_canonical_directory: str | Path | None,
) -> Path | None:
    if immutable_upstream_canonical_directory is None:
        return None
    corrected_supplied = Path(canonical_directory)
    upstream_supplied = Path(immutable_upstream_canonical_directory)
    if _path_has_symlink_component(corrected_supplied) or _path_has_symlink_component(upstream_supplied):
        raise PublicationValidationError(
            "Split-root publication validation rejects symbolic-link root aliases."
        )
    corrected = corrected_supplied.resolve(strict=False)
    upstream = upstream_supplied.resolve(strict=False)
    if corrected == upstream or corrected in upstream.parents or upstream in corrected.parents:
        raise PublicationValidationError(
            "Corrected and immutable-upstream publication roots must be distinct and non-nested."
        )
    return upstream


def _path_has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _validate_generation_directory(directory: Path, manifest: Mapping[str, Any]) -> None:
    for row in manifest["artifacts"]:
        path = directory / row["relative_path"]
        if path.parent.resolve() != directory.resolve() or not path.is_file():
            raise PublicationValidationError(f"Staged generation artifact is missing/unsafe: {row['label']}")
        if path.stat().st_size != row["size_bytes"] or _sha256_file(path) != row["sha256"]:
            raise PublicationValidationError(f"Staged generation artifact failed identity validation: {row['label']}")


@contextmanager
def _publication_lock(root: Path, scope: PublicationScope, *, timeout_s: float) -> Iterator[None]:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s < 0:
        raise ValueError("Publication lock timeout must be a non-negative number of seconds.")
    lock_path = root / ".publication.lock"
    token = uuid4().hex
    metadata = {
        "schema": PUBLICATION_LOCK_SCHEMA,
        "token": token,
        "scope": scope.as_dict(),
        "host": socket.gethostname(),
        "process_id": os.getpid(),
        "thread_id": threading.get_ident(),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    deadline = time.monotonic() + float(timeout_s)
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PublicationLockTimeoutError(
                    f"Timed out acquiring publication lock {lock_path}; it was not broken automatically."
                )
            time.sleep(min(0.05, remaining))
    try:
        os.write(descriptor, (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                lock_path.unlink()
        except (OSError, json.JSONDecodeError):
            pass


def _read_generation_id(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationValidationError(f"Existing canonical finality is unreadable: {path}: {error}") from error
    if value.get("schema") != PUBLICATION_MANIFEST_SCHEMA or value.get("state") != PublicationState.PUBLISHED.value:
        raise PublicationValidationError(f"Existing canonical finality is invalid: {path}")
    return str(value.get("generation_id"))


def _remove_owned_staging(directory: Path, known_files: list[Path]) -> None:
    if not directory.exists():
        return
    ownership = directory / "ownership.json"
    try:
        value = json.loads(ownership.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if value.get("schema") != "kawaii_publication_staging_ownership_v1" or value.get("directory") != str(directory.resolve()):
        return
    for path in sorted(set(known_files), key=lambda item: str(item)):
        if path.parent.resolve() == directory.resolve() and (path.is_file() or path.is_symlink()):
            path.unlink()
    try:
        directory.rmdir()
    except OSError:
        pass  # foreign/unknown content is deliberately retained


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise PublicationCancelled("Publication cancelled before canonical finality.")


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise PublicationValidationError(f"Unsafe empty publication path component: {value!r}")
    return result[:120]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
