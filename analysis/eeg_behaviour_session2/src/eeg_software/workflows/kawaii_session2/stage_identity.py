from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .identity import (
    BehavioralInputIdentity,
    BrainVisionDatasetIdentity,
    SourceFileIdentity,
    behavioral_input_identity_from_primitive,
    brainvision_dataset_identity_from_primitive,
    identify_source_file,
    source_file_identity_from_primitive,
)
from .resolved_config import canonical_json, canonical_sha256
from .resolved_snapshot import ResolvedRunSnapshotReference


DATASET_CONTEXT_IDENTITY_SCHEMA_VERSION = "kawaii_dataset_context_identity_v1"
ALIGNMENT_ARTIFACT_IDENTITY_SCHEMA_VERSION = "kawaii_alignment_artifact_identity_v1"
ALIGNMENT_ALGORITHM_IDENTITY = "kawaii_session2_marker_behavior_alignment_v1"
ALIGNMENT_IDENTITY_FILENAME = "trial_alignment.identity.json"
EPOCH_ROW_IDENTITY_SCHEMA_VERSION = "kawaii_epoch_row_identity_v1"
EPOCH_ROW_CONTENT_IDENTITY_SCHEMA_VERSION = "kawaii_epoch_row_content_identity_v2"
EPOCH_METADATA_BINDING_SCHEMA_VERSION = "kawaii_epoch_metadata_binding_v1"
TIMING_QC_INPUT_IDENTITY_SCHEMA_VERSION = "kawaii_timing_qc_input_identity_v1"
PARTICIPANT_ERP_ARTIFACT_IDENTITY_SCHEMA_VERSION = "kawaii_participant_erp_artifact_identity_v1"
PARTICIPANT_ERP_IDENTITY_FILENAME = "participant_erp.identity.json"
PARTICIPANT_ERP_AUTHORITATIVE_FINALITY = "authoritative_final"

EPOCH_PARTICIPANT_COLUMN = "epoch_participant"
EPOCH_SESSION_COLUMN = "epoch_session"
EPOCH_RECORDING_PART_COLUMN = "epoch_recording_part"
EPOCH_ROW_KEY_COLUMN = "epoch_row_key"
EPOCH_ROW_CONTENT_COLUMN = "epoch_row_content_fingerprint"
EPOCH_BINDING_COLUMNS = (
    EPOCH_PARTICIPANT_COLUMN,
    EPOCH_SESSION_COLUMN,
    EPOCH_RECORDING_PART_COLUMN,
    EPOCH_ROW_KEY_COLUMN,
    EPOCH_ROW_CONTENT_COLUMN,
)

_SHA256_PREFIX = "sha256:"


class StageIdentityError(ValueError):
    """Raised when a migrated stage identity is missing or inconsistent."""


@dataclass(frozen=True)
class DatasetContextIdentity:
    """Exact dataset and configuration context for one participant recording part."""

    participant: str
    session: str
    recording_part: str
    brainvision: BrainVisionDatasetIdentity
    behavioral_inputs: BehavioralInputIdentity | None
    resolved_config_fingerprint: str
    scientific_fingerprint: str
    resolved_snapshot_schema_version: str | None
    resolved_snapshot_state: str | None
    resolved_snapshot_content_sha256: str | None
    source_config_sha256: str | None
    resolved_snapshot_path: str | None
    fingerprint: str
    schema_version: str = DATASET_CONTEXT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_CONTEXT_IDENTITY_SCHEMA_VERSION:
            raise StageIdentityError(f"Unsupported dataset-context schema: {self.schema_version!r}.")
        _require_token(self.participant, "participant")
        if self.session != normalise_session(self.session):
            raise StageIdentityError(f"Dataset-context session is not canonical: {self.session!r}.")
        _require_token(self.recording_part, "recording_part")
        if not isinstance(self.brainvision, BrainVisionDatasetIdentity):
            raise StageIdentityError("Dataset context requires a BrainVisionDatasetIdentity.")
        if self.behavioral_inputs is not None and not isinstance(
            self.behavioral_inputs, BehavioralInputIdentity
        ):
            raise StageIdentityError("Dataset-context behavioral_inputs is malformed.")
        _require_sha256(self.resolved_config_fingerprint, "resolved_config_fingerprint")
        _require_sha256(self.scientific_fingerprint, "scientific_fingerprint")
        snapshot_values = (
            self.resolved_snapshot_schema_version,
            self.resolved_snapshot_state,
            self.resolved_snapshot_content_sha256,
            self.source_config_sha256,
        )
        if any(value is not None for value in snapshot_values):
            if any(value is None for value in snapshot_values):
                raise StageIdentityError("Dataset-context resolved snapshot reference is incomplete.")
            _require_sha256(str(self.resolved_snapshot_content_sha256), "snapshot content sha256")
            _require_source_file_sha256(str(self.source_config_sha256), "source config sha256")
        elif self.resolved_snapshot_path is not None:
            raise StageIdentityError("Dataset-context snapshot path cannot exist without snapshot identity.")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise StageIdentityError("Dataset-context fingerprint is inconsistent.")

    def identity_payload(self) -> dict[str, Any]:
        snapshot = None
        if self.resolved_snapshot_content_sha256 is not None:
            snapshot = {
                "schema_version": self.resolved_snapshot_schema_version,
                "snapshot_state": self.resolved_snapshot_state,
                "snapshot_content_sha256": self.resolved_snapshot_content_sha256,
                "source_config_sha256": self.source_config_sha256,
            }
        return {
            "schema_version": self.schema_version,
            "scope": {
                "participant": self.participant,
                "session": self.session,
                "recording_part": self.recording_part,
            },
            "brainvision": self.brainvision.identity_payload(),
            "behavioral_inputs": (
                self.behavioral_inputs.identity_payload()
                if self.behavioral_inputs is not None
                else None
            ),
            "resolved_config_fingerprint": self.resolved_config_fingerprint,
            "scientific_fingerprint": self.scientific_fingerprint,
            "resolved_snapshot": snapshot,
        }

    def to_primitive(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["brainvision"] = self.brainvision.to_primitive()
        value["behavioral_inputs"] = (
            self.behavioral_inputs.to_primitive()
            if self.behavioral_inputs is not None
            else None
        )
        if value["resolved_snapshot"] is not None:
            value["resolved_snapshot"] = {
                **value["resolved_snapshot"],
                "path": self.resolved_snapshot_path,
            }
        value["fingerprint"] = self.fingerprint
        return value

    def canonical_json(self) -> str:
        return canonical_json(self.to_primitive())


@dataclass(frozen=True)
class AlignmentArtifactIdentity:
    """Content-bound contract for a persisted trial-alignment table."""

    participant: str
    session: str
    contexts: tuple[DatasetContextIdentity, ...]
    algorithm: str
    alignment_payload_fingerprint: str
    output_file: SourceFileIdentity
    fingerprint: str
    schema_version: str = ALIGNMENT_ARTIFACT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ALIGNMENT_ARTIFACT_IDENTITY_SCHEMA_VERSION:
            raise StageIdentityError(f"Unsupported alignment identity schema: {self.schema_version!r}.")
        _require_token(self.participant, "participant")
        if self.session != normalise_session(self.session):
            raise StageIdentityError(f"Alignment session is not canonical: {self.session!r}.")
        if not isinstance(self.contexts, tuple) or not self.contexts:
            raise StageIdentityError("Alignment identity requires at least one dataset context.")
        if any(not isinstance(context, DatasetContextIdentity) for context in self.contexts):
            raise StageIdentityError("Alignment identity contains a malformed dataset context.")
        parts = [context.recording_part for context in self.contexts]
        if len(parts) != len(set(parts)) or parts != sorted(parts):
            raise StageIdentityError("Alignment contexts must have unique, sorted recording parts.")
        if any(
            context.participant != self.participant or context.session != self.session
            for context in self.contexts
        ):
            raise StageIdentityError("Alignment contexts do not share the artifact scope.")
        if self.algorithm != ALIGNMENT_ALGORITHM_IDENTITY:
            raise StageIdentityError(f"Unsupported alignment algorithm identity: {self.algorithm!r}.")
        _require_sha256(self.alignment_payload_fingerprint, "alignment payload fingerprint")
        if not isinstance(self.output_file, SourceFileIdentity) or self.output_file.role != "trial_alignment":
            raise StageIdentityError("Alignment output identity must have role 'trial_alignment'.")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise StageIdentityError("Alignment artifact fingerprint is inconsistent.")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {"participant": self.participant, "session": self.session},
            "contexts": [context.identity_payload() for context in self.contexts],
            "algorithm": self.algorithm,
            "alignment_payload_fingerprint": self.alignment_payload_fingerprint,
            "output_file": self.output_file.identity_payload(),
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {"participant": self.participant, "session": self.session},
            "contexts": [context.to_primitive() for context in self.contexts],
            "algorithm": self.algorithm,
            "alignment_payload_fingerprint": self.alignment_payload_fingerprint,
            "output_file": self.output_file.to_primitive(),
            "fingerprint": self.fingerprint,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_primitive())


@dataclass(frozen=True)
class EpochRowIdentity:
    """Stable scope/event key and content identity for one epoch row."""

    participant: str
    session: str
    recording_part: str
    trial: str
    event_sample: int
    event_code: int
    row_key: str
    content_fingerprint: str
    schema_version: str = EPOCH_ROW_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_token(self.participant, "participant")
        if self.session != normalise_session(self.session):
            raise StageIdentityError(f"Epoch-row session is not canonical: {self.session!r}.")
        _require_token(self.recording_part, "recording_part")
        _require_token(self.trial, "trial")
        if isinstance(self.event_sample, bool) or not isinstance(self.event_sample, int):
            raise StageIdentityError("Epoch event_sample must be an integer.")
        if isinstance(self.event_code, bool) or not isinstance(self.event_code, int):
            raise StageIdentityError("Epoch event_code must be an integer.")
        _require_sha256(self.row_key, "epoch row key")
        _require_sha256(self.content_fingerprint, "epoch row content fingerprint")
        if self.row_key != canonical_sha256(self.key_payload()):
            raise StageIdentityError("Epoch row key is inconsistent.")

    def key_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "participant": self.participant,
            "session": self.session,
            "recording_part": self.recording_part,
            "trial": self.trial,
            "event_sample": self.event_sample,
            "event_code": self.event_code,
        }


@dataclass(frozen=True)
class EpochMetadataBinding:
    """Ordered, content-bound relationship between epochs and metadata rows."""

    participant: str
    session: str
    recording_part: str
    rows: tuple[EpochRowIdentity, ...]
    fingerprint: str
    schema_version: str = EPOCH_METADATA_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EPOCH_METADATA_BINDING_SCHEMA_VERSION:
            raise StageIdentityError(f"Unsupported epoch binding schema: {self.schema_version!r}.")
        if not self.rows:
            raise StageIdentityError("Epoch metadata binding requires at least one row.")
        if any(
            row.participant != self.participant
            or row.session != self.session
            or row.recording_part != self.recording_part
            for row in self.rows
        ):
            raise StageIdentityError("Epoch binding rows do not share one scope.")
        keys = [row.row_key for row in self.rows]
        if len(keys) != len(set(keys)):
            raise StageIdentityError("Epoch metadata binding contains duplicate stable row keys.")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise StageIdentityError("Epoch metadata binding fingerprint is inconsistent.")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {
                "participant": self.participant,
                "session": self.session,
                "recording_part": self.recording_part,
            },
            "rows": [
                {"row_key": row.row_key, "content_fingerprint": row.content_fingerprint}
                for row in self.rows
            ],
        }


@dataclass(frozen=True)
class TimingQCInputIdentity:
    """Exact behavioral/alignment/config context consumed by timing QC."""

    context: DatasetContextIdentity
    alignment_artifact_fingerprint: str
    behavioral_payload_fingerprint: str
    alignment_payload_fingerprint: str
    timing_settings_fingerprint: str
    fingerprint: str
    schema_version: str = TIMING_QC_INPUT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TIMING_QC_INPUT_IDENTITY_SCHEMA_VERSION:
            raise StageIdentityError(f"Unsupported timing-QC input schema: {self.schema_version!r}.")
        if self.context.behavioral_inputs is None:
            raise StageIdentityError("Timing-QC context requires content-bound behavioral input.")
        for value, label in (
            (self.alignment_artifact_fingerprint, "alignment artifact fingerprint"),
            (self.behavioral_payload_fingerprint, "behavioral payload fingerprint"),
            (self.alignment_payload_fingerprint, "alignment payload fingerprint"),
            (self.timing_settings_fingerprint, "timing settings fingerprint"),
        ):
            _require_sha256(value, label)
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise StageIdentityError("Timing-QC input fingerprint is inconsistent.")

    @property
    def participant(self) -> str:
        return self.context.participant

    @property
    def session(self) -> str:
        return self.context.session

    @property
    def recording_part(self) -> str:
        return self.context.recording_part

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context": self.context.identity_payload(),
            "alignment_artifact_fingerprint": self.alignment_artifact_fingerprint,
            "behavioral_payload_fingerprint": self.behavioral_payload_fingerprint,
            "alignment_payload_fingerprint": self.alignment_payload_fingerprint,
            "timing_settings_fingerprint": self.timing_settings_fingerprint,
        }


@dataclass(frozen=True)
class ParticipantERPArtifactIdentity:
    """Content-bound Gate 4 result consumed by Gate 5 aggregation."""

    participant: str
    session: str
    recording_part: str
    branch: str
    epoch_metadata_binding_fingerprint: str
    configuration_fingerprint: str
    branch_settings_fingerprint: str
    output_files: tuple[tuple[str, SourceFileIdentity], ...]
    finality: str
    fingerprint: str
    schema_version: str = PARTICIPANT_ERP_ARTIFACT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARTICIPANT_ERP_ARTIFACT_IDENTITY_SCHEMA_VERSION:
            raise StageIdentityError(
                f"Unsupported participant ERP artifact schema: {self.schema_version!r}."
            )
        _require_token(self.participant, "participant ERP participant")
        if self.session != normalise_session(self.session):
            raise StageIdentityError(
                f"Participant ERP session is not canonical: {self.session!r}."
            )
        _require_token(self.recording_part, "participant ERP recording part")
        if self.branch not in {"recognition", "cuteness"}:
            raise StageIdentityError(f"Unsupported participant ERP branch: {self.branch!r}.")
        for value, label in (
            (self.epoch_metadata_binding_fingerprint, "epoch metadata binding fingerprint"),
            (self.configuration_fingerprint, "participant ERP configuration fingerprint"),
            (self.branch_settings_fingerprint, "participant ERP branch-settings fingerprint"),
        ):
            _require_sha256(value, label)
        if self.finality != PARTICIPANT_ERP_AUTHORITATIVE_FINALITY:
            raise StageIdentityError(
                f"Participant ERP artifact is not authoritative final: {self.finality!r}."
            )
        labels = [label for label, _ in self.output_files]
        if not labels or labels != sorted(labels) or len(labels) != len(set(labels)):
            raise StageIdentityError(
                "Participant ERP output labels must be non-empty, unique, and sorted."
            )
        for label, identity in self.output_files:
            _require_token(label, "participant ERP output label")
            expected_role = f"participant_erp_{self.branch}_{label}"
            if not isinstance(identity, SourceFileIdentity) or identity.role != expected_role:
                raise StageIdentityError(
                    f"Participant ERP output {label!r} does not have role {expected_role!r}."
                )
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise StageIdentityError("Participant ERP artifact fingerprint is inconsistent.")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {
                "participant": self.participant,
                "session": self.session,
                "recording_part": self.recording_part,
                "branch": self.branch,
            },
            "epoch_metadata_binding_fingerprint": self.epoch_metadata_binding_fingerprint,
            "configuration_fingerprint": self.configuration_fingerprint,
            "branch_settings_fingerprint": self.branch_settings_fingerprint,
            "output_files": [
                {"label": label, "file": identity.identity_payload()}
                for label, identity in self.output_files
            ],
            "finality": self.finality,
        }

    def to_primitive(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["output_files"] = [
            {"label": label, "file": identity.to_primitive()}
            for label, identity in self.output_files
        ]
        value["fingerprint"] = self.fingerprint
        return value

    def canonical_json(self) -> str:
        return canonical_json(self.to_primitive())


def build_dataset_context_identity(
    *,
    participant: str,
    session: str | int,
    recording_part: str,
    brainvision: BrainVisionDatasetIdentity,
    behavioral_inputs: BehavioralInputIdentity | None,
    resolved_config_fingerprint: str,
    scientific_fingerprint: str,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
) -> DatasetContextIdentity:
    clean_participant = _require_token(participant, "participant")
    clean_session = normalise_session(session)
    clean_part = _require_token(recording_part, "recording_part")
    snapshot_values: dict[str, str | None] = {
        "resolved_snapshot_schema_version": None,
        "resolved_snapshot_state": None,
        "resolved_snapshot_content_sha256": None,
        "source_config_sha256": None,
        "resolved_snapshot_path": None,
    }
    if resolved_snapshot is not None:
        snapshot_values = {
            "resolved_snapshot_schema_version": resolved_snapshot.schema_version,
            "resolved_snapshot_state": resolved_snapshot.snapshot_state,
            "resolved_snapshot_content_sha256": resolved_snapshot.snapshot_content_sha256,
            "source_config_sha256": resolved_snapshot.source_config_sha256,
            "resolved_snapshot_path": str(resolved_snapshot.path),
        }
    provisional = {
        "schema_version": DATASET_CONTEXT_IDENTITY_SCHEMA_VERSION,
        "scope": {
            "participant": clean_participant,
            "session": clean_session,
            "recording_part": clean_part,
        },
        "brainvision": brainvision.identity_payload(),
        "behavioral_inputs": behavioral_inputs.identity_payload() if behavioral_inputs else None,
        "resolved_config_fingerprint": resolved_config_fingerprint,
        "scientific_fingerprint": scientific_fingerprint,
        "resolved_snapshot": (
            {
                "schema_version": snapshot_values["resolved_snapshot_schema_version"],
                "snapshot_state": snapshot_values["resolved_snapshot_state"],
                "snapshot_content_sha256": snapshot_values["resolved_snapshot_content_sha256"],
                "source_config_sha256": snapshot_values["source_config_sha256"],
            }
            if resolved_snapshot is not None
            else None
        ),
    }
    return DatasetContextIdentity(
        participant=clean_participant,
        session=clean_session,
        recording_part=clean_part,
        brainvision=brainvision,
        behavioral_inputs=behavioral_inputs,
        resolved_config_fingerprint=resolved_config_fingerprint,
        scientific_fingerprint=scientific_fingerprint,
        fingerprint=canonical_sha256(provisional),
        **snapshot_values,
    )


def build_alignment_artifact_identity(
    *,
    contexts: Sequence[DatasetContextIdentity],
    alignment: pd.DataFrame,
    output_path: str | Path,
) -> AlignmentArtifactIdentity:
    ordered = tuple(sorted(contexts, key=lambda context: context.recording_part))
    if not ordered:
        raise StageIdentityError("Cannot build alignment identity without dataset contexts.")
    output = identify_source_file(output_path, "trial_alignment", logical_path=Path(output_path).name)
    # The persisted CSV is the authoritative representation consumed later.
    # Hash that representation rather than the pre-serialization DataFrame so
    # pandas' CSV dtype inference cannot make an unchanged file appear stale.
    try:
        persisted_alignment = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        # A zero-row, zero-column DataFrame serializes to an empty file.  Treat
        # that one unparseable CSV representation as the same empty table; the
        # output-file identity below still binds its exact bytes.
        persisted_alignment = pd.DataFrame()
    payload_fingerprint = dataframe_content_fingerprint(persisted_alignment)
    payload = {
        "schema_version": ALIGNMENT_ARTIFACT_IDENTITY_SCHEMA_VERSION,
        "scope": {"participant": ordered[0].participant, "session": ordered[0].session},
        "contexts": [context.identity_payload() for context in ordered],
        "algorithm": ALIGNMENT_ALGORITHM_IDENTITY,
        "alignment_payload_fingerprint": payload_fingerprint,
        "output_file": output.identity_payload(),
    }
    return AlignmentArtifactIdentity(
        participant=ordered[0].participant,
        session=ordered[0].session,
        contexts=ordered,
        algorithm=ALIGNMENT_ALGORITHM_IDENTITY,
        alignment_payload_fingerprint=payload_fingerprint,
        output_file=output,
        fingerprint=canonical_sha256(payload),
    )


def alignment_identity_path(alignment_path: str | Path) -> Path:
    return Path(alignment_path).with_name(ALIGNMENT_IDENTITY_FILENAME)


def write_alignment_artifact_identity(
    identity: AlignmentArtifactIdentity, path: str | Path
) -> Path:
    if not isinstance(identity, AlignmentArtifactIdentity):
        raise StageIdentityError("identity must be an AlignmentArtifactIdentity.")
    target = Path(path)
    _atomic_write_text(target, identity.canonical_json() + "\n")
    if read_alignment_artifact_identity(target) != identity:
        raise StageIdentityError("Persisted alignment identity did not round-trip.")
    return target


def read_alignment_artifact_identity(path: str | Path) -> AlignmentArtifactIdentity:
    source = Path(path)
    if not source.is_file():
        raise StageIdentityError(f"Alignment identity is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageIdentityError(f"Could not read alignment identity {source}: {error}") from error
    _exact_keys(
        value,
        {
            "schema_version",
            "scope",
            "contexts",
            "algorithm",
            "alignment_payload_fingerprint",
            "output_file",
            "fingerprint",
        },
        "alignment identity",
    )
    _exact_keys(value["scope"], {"participant", "session"}, "alignment identity scope")
    contexts_value = value["contexts"]
    if not isinstance(contexts_value, list):
        raise StageIdentityError("Alignment identity contexts must be a list.")
    return AlignmentArtifactIdentity(
        participant=value["scope"]["participant"],
        session=value["scope"]["session"],
        contexts=tuple(_dataset_context_from_primitive(item) for item in contexts_value),
        algorithm=value["algorithm"],
        alignment_payload_fingerprint=value["alignment_payload_fingerprint"],
        output_file=source_file_identity_from_primitive(value["output_file"]),
        fingerprint=value["fingerprint"],
        schema_version=value["schema_version"],
    )


def validate_alignment_artifact_identity(
    identity: AlignmentArtifactIdentity,
    *,
    expected_contexts: Sequence[DatasetContextIdentity],
    alignment_path: str | Path | None = None,
    alignment: pd.DataFrame | None = None,
) -> None:
    expected = tuple(sorted(expected_contexts, key=lambda context: context.recording_part))
    if tuple(context.fingerprint for context in identity.contexts) != tuple(
        context.fingerprint for context in expected
    ):
        raise StageIdentityError("Alignment dataset context does not match the expected current inputs.")
    if alignment_path is not None:
        current = identify_source_file(
            alignment_path, "trial_alignment", logical_path=Path(alignment_path).name
        )
        if current.fingerprint != identity.output_file.fingerprint:
            raise StageIdentityError("Current alignment file content does not match its identity contract.")
    if alignment is not None and dataframe_content_fingerprint(alignment) != identity.alignment_payload_fingerprint:
        raise StageIdentityError("Current alignment table payload does not match its identity contract.")


def read_validated_alignment(
    alignment_path: str | Path,
    *,
    expected_contexts: Sequence[DatasetContextIdentity],
) -> tuple[pd.DataFrame, AlignmentArtifactIdentity]:
    path = Path(alignment_path)
    identity = read_alignment_artifact_identity(alignment_identity_path(path))
    alignment = pd.read_csv(path)
    validate_alignment_artifact_identity(
        identity,
        expected_contexts=expected_contexts,
        alignment_path=path,
        alignment=alignment,
    )
    return alignment, identity


def bind_epoch_metadata_rows(
    metadata: pd.DataFrame,
    *,
    participant: str,
    session: str | int,
    recording_part: str,
) -> pd.DataFrame:
    """Return metadata with explicit stable row keys and content fingerprints."""

    if metadata.empty:
        raise StageIdentityError("Cannot bind empty epoch metadata.")
    clean_participant = _require_token(participant, "participant")
    clean_session = normalise_session(session)
    clean_part = _require_token(recording_part, "recording_part")
    result = metadata.copy().reset_index(drop=True)
    for column in EPOCH_BINDING_COLUMNS:
        if column in result:
            result = result.drop(columns=[column])
    _validate_frame_scope(result, clean_participant, clean_session, clean_part, allow_absent=True)
    sample_column = _first_column(result, ("event_sample_original", "event_sample", "image_onset_sample"))
    code_column = _first_column(result, ("event_code", "image_onset_code"))
    if "trial" not in result or sample_column is None or code_column is None:
        raise StageIdentityError(
            "Epoch metadata requires trial plus event sample/code fields for stable row identity."
        )
    result[EPOCH_PARTICIPANT_COLUMN] = clean_participant
    result[EPOCH_SESSION_COLUMN] = clean_session
    result[EPOCH_RECORDING_PART_COLUMN] = clean_part
    keys: list[str] = []
    contents: list[str] = []
    for _, row in result.iterrows():
        trial = _stable_trial(row["trial"])
        event_sample = _required_int(row[sample_column], "event sample")
        event_code = _required_int(row[code_column], "event code")
        key_payload = {
            "schema_version": EPOCH_ROW_IDENTITY_SCHEMA_VERSION,
            "participant": clean_participant,
            "session": clean_session,
            "recording_part": clean_part,
            "trial": trial,
            "event_sample": event_sample,
            "event_code": event_code,
        }
        keys.append(canonical_sha256(key_payload))
        contents.append(canonical_sha256(_row_content_payload(row)))
    if len(keys) != len(set(keys)):
        raise StageIdentityError("Epoch metadata contains duplicate stable row keys.")
    result[EPOCH_ROW_KEY_COLUMN] = keys
    result[EPOCH_ROW_CONTENT_COLUMN] = contents
    return result


def epoch_metadata_binding(metadata: pd.DataFrame) -> EpochMetadataBinding:
    participant, session, part = _epoch_metadata_scope(metadata)
    rebound = bind_epoch_metadata_rows(
        metadata.drop(columns=list(EPOCH_BINDING_COLUMNS)),
        participant=participant,
        session=session,
        recording_part=part,
    )
    return _build_epoch_metadata_binding(
        metadata,
        rebound,
        participant=participant,
        session=session,
        recording_part=part,
        verify_content=True,
    )


def _epoch_metadata_scope(metadata: pd.DataFrame) -> tuple[str, str, str]:
    missing = [column for column in EPOCH_BINDING_COLUMNS if column not in metadata]
    if missing:
        raise StageIdentityError(
            "Legacy epoch metadata lacks required identity binding columns: " + ", ".join(missing)
        )
    if metadata.empty:
        raise StageIdentityError("Epoch metadata binding cannot be empty.")
    participants = _unique_tokens(metadata[EPOCH_PARTICIPANT_COLUMN], "epoch participant")
    sessions = {normalise_session(value) for value in metadata[EPOCH_SESSION_COLUMN].tolist()}
    parts = _unique_tokens(metadata[EPOCH_RECORDING_PART_COLUMN], "epoch recording part")
    if len(participants) != 1 or len(sessions) != 1 or len(parts) != 1:
        raise StageIdentityError("Epoch metadata binding must contain exactly one participant/session/part scope.")
    return next(iter(participants)), next(iter(sessions)), next(iter(parts))


def _build_epoch_metadata_binding(
    metadata: pd.DataFrame,
    rebound: pd.DataFrame,
    *,
    participant: str,
    session: str,
    recording_part: str,
    verify_content: bool,
) -> EpochMetadataBinding:
    rows: list[EpochRowIdentity] = []
    sample_column = _first_column(rebound, ("event_sample_original", "event_sample", "image_onset_sample"))
    code_column = _first_column(rebound, ("event_code", "image_onset_code"))
    assert sample_column is not None and code_column is not None
    for index, row in rebound.iterrows():
        stored_key = str(metadata.iloc[index][EPOCH_ROW_KEY_COLUMN])
        stored_content = str(metadata.iloc[index][EPOCH_ROW_CONTENT_COLUMN])
        if stored_key != row[EPOCH_ROW_KEY_COLUMN]:
            raise StageIdentityError(f"Epoch row key is stale or tampered at row {index}.")
        if verify_content and stored_content != row[EPOCH_ROW_CONTENT_COLUMN]:
            raise StageIdentityError(f"Epoch row content identity is stale or tampered at row {index}.")
        rows.append(
            EpochRowIdentity(
                participant=participant,
                session=session,
                recording_part=recording_part,
                trial=_stable_trial(row["trial"]),
                event_sample=_required_int(row[sample_column], "event sample"),
                event_code=_required_int(row[code_column], "event code"),
                row_key=stored_key,
                content_fingerprint=stored_content,
            )
        )
    payload = {
        "schema_version": EPOCH_METADATA_BINDING_SCHEMA_VERSION,
        "scope": {
            "participant": participant,
            "session": session,
            "recording_part": recording_part,
        },
        "rows": [
            {"row_key": row.row_key, "content_fingerprint": row.content_fingerprint}
            for row in rows
        ],
    }
    return EpochMetadataBinding(
        participant=participant,
        session=session,
        recording_part=recording_part,
        rows=tuple(rows),
        fingerprint=canonical_sha256(payload),
    )


def validate_epoch_metadata_binding(
    embedded_metadata: pd.DataFrame | None,
    metadata: pd.DataFrame,
    *,
    participant: str | None = None,
    session: str | int | None = None,
    recording_part: str | None = None,
    allow_legacy_serialization_compatibility: bool = False,
) -> EpochMetadataBinding:
    if embedded_metadata is None:
        raise StageIdentityError("Epochs lack embedded metadata identity; legacy pairing is not authoritative.")
    embedded_frame = embedded_metadata.reset_index(drop=True)
    external_frame = metadata.reset_index(drop=True)
    try:
        embedded = epoch_metadata_binding(embedded_frame)
        external = epoch_metadata_binding(external_frame)
    except StageIdentityError as error:
        if (
            not allow_legacy_serialization_compatibility
            or "Epoch row content identity is stale or tampered" not in str(error)
        ):
            raise
        embedded, external = _legacy_serialized_epoch_metadata_bindings(
            embedded_frame,
            external_frame,
        )
    expected_scope = (
        participant or embedded.participant,
        normalise_session(session if session is not None else embedded.session),
        recording_part or embedded.recording_part,
    )
    if (embedded.participant, embedded.session, embedded.recording_part) != expected_scope:
        raise StageIdentityError("Embedded epoch metadata scope does not match the requested scope.")
    if (external.participant, external.session, external.recording_part) != expected_scope:
        raise StageIdentityError("External epoch metadata scope does not match the requested scope.")
    embedded_keys = [row.row_key for row in embedded.rows]
    external_keys = [row.row_key for row in external.rows]
    if set(embedded_keys) != set(external_keys):
        missing = sorted(set(embedded_keys) - set(external_keys))
        extra = sorted(set(external_keys) - set(embedded_keys))
        raise StageIdentityError(f"Epoch metadata key set mismatch; missing={missing!r}, extra={extra!r}.")
    if embedded_keys != external_keys:
        raise StageIdentityError("Epoch metadata row order does not match embedded epoch order.")
    if [row.content_fingerprint for row in embedded.rows] != [
        row.content_fingerprint for row in external.rows
    ]:
        raise StageIdentityError("Epoch metadata row content does not match embedded epoch metadata.")
    return embedded


def _legacy_serialized_epoch_metadata_bindings(
    embedded_metadata: pd.DataFrame,
    external_metadata: pd.DataFrame,
) -> tuple[EpochMetadataBinding, EpochMetadataBinding]:
    """Validate the bounded MNE-FIF/CSV representation drift of legacy rows.

    Early Gate 3C publications hashed the in-memory frame before MNE encoded
    metadata as JSON (10-decimal float precision) and before pandas inferred
    CSV scalar types.  The original pre-serialization float values cannot be
    recovered from those publications.  This compatibility path therefore
    requires two independently published copies with identical stored tokens,
    exact stable keys, and content that differs only under the current narrow
    storage canonicalization.  Callers must additionally have validated the
    enclosing immutable publication artifact identities.
    """

    embedded_scope = _epoch_metadata_scope(embedded_metadata)
    external_scope = _epoch_metadata_scope(external_metadata)
    embedded_rebound = bind_epoch_metadata_rows(
        embedded_metadata.drop(columns=list(EPOCH_BINDING_COLUMNS)),
        participant=embedded_scope[0],
        session=embedded_scope[1],
        recording_part=embedded_scope[2],
    )
    external_rebound = bind_epoch_metadata_rows(
        external_metadata.drop(columns=list(EPOCH_BINDING_COLUMNS)),
        participant=external_scope[0],
        session=external_scope[1],
        recording_part=external_scope[2],
    )
    embedded = _build_epoch_metadata_binding(
        embedded_metadata,
        embedded_rebound,
        participant=embedded_scope[0],
        session=embedded_scope[1],
        recording_part=embedded_scope[2],
        verify_content=False,
    )
    external = _build_epoch_metadata_binding(
        external_metadata,
        external_rebound,
        participant=external_scope[0],
        session=external_scope[1],
        recording_part=external_scope[2],
        verify_content=False,
    )
    if embedded_scope != external_scope:
        raise StageIdentityError("Legacy epoch metadata serialization copies have mismatched scope.")
    if len(embedded_metadata) != len(external_metadata):
        raise StageIdentityError("Legacy epoch metadata serialization copies have mismatched row counts.")

    embedded_payloads = embedded_metadata.drop(columns=list(EPOCH_BINDING_COLUMNS))
    external_payloads = external_metadata.drop(columns=list(EPOCH_BINDING_COLUMNS))
    for index in range(len(embedded_metadata)):
        embedded_stored = str(embedded_metadata.iloc[index][EPOCH_ROW_CONTENT_COLUMN])
        external_stored = str(external_metadata.iloc[index][EPOCH_ROW_CONTENT_COLUMN])
        embedded_current = str(embedded_rebound.iloc[index][EPOCH_ROW_CONTENT_COLUMN])
        external_current = str(external_rebound.iloc[index][EPOCH_ROW_CONTENT_COLUMN])
        if embedded_stored != external_stored:
            raise StageIdentityError(
                f"Legacy epoch row content tokens disagree between FIF and CSV at row {index}."
            )
        if embedded_current != external_current:
            raise StageIdentityError(
                f"Epoch row content differs beyond allowed FIF/CSV serialization at row {index}."
            )
        if embedded_stored == embedded_current:
            continue
        embedded_payload = embedded_payloads.iloc[index]
        external_payload = external_payloads.iloc[index]
        if embedded_payload.equals(external_payload):
            raise StageIdentityError(
                f"Epoch row content identity is stale or tampered at row {index}; "
                "no legacy serialization difference explains it."
            )
    return embedded, external


def build_timing_qc_input_identity(
    *,
    context: DatasetContextIdentity,
    alignment_artifact: AlignmentArtifactIdentity,
    behavioral_inputs: BehavioralInputIdentity,
    behavioral: pd.DataFrame,
    alignment: pd.DataFrame,
    settings: Mapping[str, Any],
    alignment_path: str | Path | None = None,
) -> TimingQCInputIdentity:
    if context.behavioral_inputs is None or behavioral_inputs.fingerprint != context.behavioral_inputs.fingerprint:
        raise StageIdentityError("Current behavioral source content does not match the dataset context.")
    matching = [item for item in alignment_artifact.contexts if item.recording_part == context.recording_part]
    if len(matching) != 1 or matching[0].fingerprint != context.fingerprint:
        raise StageIdentityError("Alignment identity does not contain the expected dataset context.")
    current_alignment_fingerprint = dataframe_content_fingerprint(alignment)
    if current_alignment_fingerprint != alignment_artifact.alignment_payload_fingerprint:
        # Early alignment identities hashed the in-memory DataFrame before it
        # was serialized to CSV.  Permit that bounded legacy mismatch only
        # when the exact persisted CSV still matches the artifact's immutable
        # size/SHA-256 identity.  The timing identity below binds the actually
        # consumed, reread table content.
        if alignment_path is None:
            raise StageIdentityError("Alignment table does not match the alignment artifact identity.")
        validate_alignment_artifact_identity(
            alignment_artifact,
            expected_contexts=alignment_artifact.contexts,
            alignment_path=alignment_path,
        )
    _validate_frame_scope(
        behavioral,
        context.participant,
        context.session,
        context.recording_part,
        behavioral=True,
    )
    _validate_frame_scope(
        alignment,
        context.participant,
        context.session,
        context.recording_part,
        alignment=True,
    )
    payload = {
        "schema_version": TIMING_QC_INPUT_IDENTITY_SCHEMA_VERSION,
        "context": context.identity_payload(),
        "alignment_artifact_fingerprint": alignment_artifact.fingerprint,
        "behavioral_payload_fingerprint": dataframe_content_fingerprint(behavioral),
        "alignment_payload_fingerprint": current_alignment_fingerprint,
        "timing_settings_fingerprint": canonical_sha256(_json_safe(settings)),
    }
    return TimingQCInputIdentity(
        context=context,
        alignment_artifact_fingerprint=alignment_artifact.fingerprint,
        behavioral_payload_fingerprint=payload["behavioral_payload_fingerprint"],
        alignment_payload_fingerprint=current_alignment_fingerprint,
        timing_settings_fingerprint=payload["timing_settings_fingerprint"],
        fingerprint=canonical_sha256(payload),
    )


def validate_timing_qc_input_identity(
    identity: TimingQCInputIdentity | None,
    *,
    behavioral: pd.DataFrame,
    alignment: pd.DataFrame,
    subject: str,
    recording_part: str,
    settings: Mapping[str, Any],
) -> None:
    if not isinstance(identity, TimingQCInputIdentity):
        raise StageIdentityError(
            "Timing-QC requires a content-bound input identity; legacy unlabeled inputs are rejected."
        )
    if identity.participant != subject or identity.recording_part != recording_part:
        raise StageIdentityError("Timing-QC requested participant/part does not match its input identity.")
    if dataframe_content_fingerprint(behavioral) != identity.behavioral_payload_fingerprint:
        raise StageIdentityError("Timing-QC behavioral content changed after input identity construction.")
    if dataframe_content_fingerprint(alignment) != identity.alignment_payload_fingerprint:
        raise StageIdentityError("Timing-QC alignment content changed after input identity construction.")
    if canonical_sha256(_json_safe(settings)) != identity.timing_settings_fingerprint:
        raise StageIdentityError("Timing-QC settings do not match the content-bound input identity.")
    _validate_frame_scope(
        behavioral,
        identity.participant,
        identity.session,
        identity.recording_part,
        behavioral=True,
    )
    _validate_frame_scope(
        alignment,
        identity.participant,
        identity.session,
        identity.recording_part,
        alignment=True,
    )


def configuration_mapping_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a deterministic identity for the effective configuration mapping."""

    return canonical_sha256(_json_safe(config))


def participant_erp_identity_path(branch_directory: str | Path) -> Path:
    return Path(branch_directory) / PARTICIPANT_ERP_IDENTITY_FILENAME


def build_participant_erp_artifact_identity(
    *,
    participant: str,
    session: str | int,
    recording_part: str,
    branch: str,
    epoch_metadata_binding_fingerprint: str,
    configuration_fingerprint: str,
    branch_settings: Mapping[str, Any],
    output_paths: Mapping[str, str | Path],
) -> ParticipantERPArtifactIdentity:
    clean_session = normalise_session(session)
    clean_branch = _require_token(branch, "participant ERP branch")
    files = tuple(
        (
            str(label),
            identify_source_file(
                path,
                f"participant_erp_{clean_branch}_{label}",
                logical_path=f"{clean_branch}/{Path(path).name}",
            ),
        )
        for label, path in sorted(output_paths.items())
    )
    payload = {
        "schema_version": PARTICIPANT_ERP_ARTIFACT_IDENTITY_SCHEMA_VERSION,
        "scope": {
            "participant": participant,
            "session": clean_session,
            "recording_part": recording_part,
            "branch": clean_branch,
        },
        "epoch_metadata_binding_fingerprint": epoch_metadata_binding_fingerprint,
        "configuration_fingerprint": configuration_fingerprint,
        "branch_settings_fingerprint": canonical_sha256(_json_safe(branch_settings)),
        "output_files": [
            {"label": label, "file": identity.identity_payload()}
            for label, identity in files
        ],
        "finality": PARTICIPANT_ERP_AUTHORITATIVE_FINALITY,
    }
    return ParticipantERPArtifactIdentity(
        participant=participant,
        session=clean_session,
        recording_part=recording_part,
        branch=clean_branch,
        epoch_metadata_binding_fingerprint=epoch_metadata_binding_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        branch_settings_fingerprint=payload["branch_settings_fingerprint"],
        output_files=files,
        finality=PARTICIPANT_ERP_AUTHORITATIVE_FINALITY,
        fingerprint=canonical_sha256(payload),
    )


def write_participant_erp_artifact_identity(
    identity: ParticipantERPArtifactIdentity, path: str | Path
) -> Path:
    if not isinstance(identity, ParticipantERPArtifactIdentity):
        raise StageIdentityError("Expected a ParticipantERPArtifactIdentity.")
    destination = Path(path)
    _atomic_write_text(destination, identity.canonical_json() + "\n")
    return destination


def read_participant_erp_artifact_identity(
    path: str | Path,
) -> ParticipantERPArtifactIdentity:
    source = Path(path)
    if not source.is_file():
        raise StageIdentityError(
            "Legacy participant ERP outputs lack an authoritative identity; regenerate Gate 4."
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        _exact_keys(
            value,
            {
                "schema_version",
                "scope",
                "epoch_metadata_binding_fingerprint",
                "configuration_fingerprint",
                "branch_settings_fingerprint",
                "output_files",
                "finality",
                "fingerprint",
            },
            "participant ERP artifact",
        )
        _exact_keys(
            value["scope"],
            {"participant", "session", "recording_part", "branch"},
            "participant ERP scope",
        )
        output_files = []
        if not isinstance(value["output_files"], list):
            raise StageIdentityError("Participant ERP output_files must be a list.")
        for item in value["output_files"]:
            _exact_keys(item, {"label", "file"}, "participant ERP output")
            output_files.append(
                (str(item["label"]), source_file_identity_from_primitive(item["file"]))
            )
        return ParticipantERPArtifactIdentity(
            participant=value["scope"]["participant"],
            session=value["scope"]["session"],
            recording_part=value["scope"]["recording_part"],
            branch=value["scope"]["branch"],
            epoch_metadata_binding_fingerprint=value["epoch_metadata_binding_fingerprint"],
            configuration_fingerprint=value["configuration_fingerprint"],
            branch_settings_fingerprint=value["branch_settings_fingerprint"],
            output_files=tuple(output_files),
            finality=value["finality"],
            fingerprint=value["fingerprint"],
            schema_version=value["schema_version"],
        )
    except StageIdentityError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StageIdentityError(
            f"Participant ERP identity is malformed or incomplete: {source}."
        ) from error


def validate_participant_erp_artifact_identity(
    identity: ParticipantERPArtifactIdentity,
    *,
    expected_participant: str,
    expected_recording_part: str,
    expected_branch: str,
    expected_configuration_fingerprint: str,
    expected_branch_settings: Mapping[str, Any],
    current_output_paths: Mapping[str, str | Path],
) -> None:
    expected_scope = (
        expected_participant,
        normalise_session("session2"),
        expected_recording_part,
        expected_branch,
    )
    actual_scope = (
        identity.participant,
        identity.session,
        identity.recording_part,
        identity.branch,
    )
    if actual_scope != expected_scope:
        raise StageIdentityError(
            "Participant ERP artifact scope does not match its Gate 5 location/request."
        )
    if identity.configuration_fingerprint != expected_configuration_fingerprint:
        raise StageIdentityError("Participant ERP configuration fingerprint mismatch.")
    if identity.branch_settings_fingerprint != canonical_sha256(
        _json_safe(expected_branch_settings)
    ):
        raise StageIdentityError("Participant ERP branch-settings fingerprint mismatch.")
    expected_labels = sorted(str(label) for label in current_output_paths)
    stored = dict(identity.output_files)
    if sorted(stored) != expected_labels:
        raise StageIdentityError(
            "Participant ERP identity does not cover the exact required Gate 5 inputs."
        )
    for label, path in current_output_paths.items():
        stored_identity = stored[str(label)]
        current = identify_source_file(
            path,
            stored_identity.role,
            logical_path=stored_identity.logical_path,
        )
        if current.fingerprint != stored_identity.fingerprint:
            raise StageIdentityError(
                f"Participant ERP output content mismatch for {label!r}."
            )


def read_validated_participant_erp_artifact(
    identity_path: str | Path,
    **expected: Any,
) -> ParticipantERPArtifactIdentity:
    identity = read_participant_erp_artifact_identity(identity_path)
    validate_participant_erp_artifact_identity(identity, **expected)
    return identity


def dataframe_content_fingerprint(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise StageIdentityError("Content fingerprinting requires a pandas DataFrame.")
    columns = [str(column) for column in frame.columns]
    if len(columns) != len(set(columns)):
        raise StageIdentityError("DataFrame content identity requires unique column names.")
    payload = {
        "columns": columns,
        "rows": [
            [_normalise_scalar(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }
    return canonical_sha256(payload)


def normalise_session(value: str | int | float) -> str:
    text = str(value).strip().lower()
    if text.startswith("session"):
        text = text[7:]
    if text.startswith("ses-"):
        text = text[4:]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise StageIdentityError("session must be non-empty.")
    return f"session{text}"


def _dataset_context_from_primitive(value: Any) -> DatasetContextIdentity:
    _exact_keys(
        value,
        {
            "schema_version",
            "scope",
            "brainvision",
            "behavioral_inputs",
            "resolved_config_fingerprint",
            "scientific_fingerprint",
            "resolved_snapshot",
            "fingerprint",
        },
        "dataset context",
    )
    _exact_keys(value["scope"], {"participant", "session", "recording_part"}, "dataset context scope")
    snapshot = value["resolved_snapshot"]
    snapshot_values = {
        "resolved_snapshot_schema_version": None,
        "resolved_snapshot_state": None,
        "resolved_snapshot_content_sha256": None,
        "source_config_sha256": None,
        "resolved_snapshot_path": None,
    }
    if snapshot is not None:
        _exact_keys(
            snapshot,
            {"schema_version", "snapshot_state", "snapshot_content_sha256", "source_config_sha256", "path"},
            "dataset context resolved snapshot",
        )
        snapshot_values = {
            "resolved_snapshot_schema_version": snapshot["schema_version"],
            "resolved_snapshot_state": snapshot["snapshot_state"],
            "resolved_snapshot_content_sha256": snapshot["snapshot_content_sha256"],
            "source_config_sha256": snapshot["source_config_sha256"],
            "resolved_snapshot_path": snapshot["path"],
        }
    return DatasetContextIdentity(
        participant=value["scope"]["participant"],
        session=value["scope"]["session"],
        recording_part=value["scope"]["recording_part"],
        brainvision=brainvision_dataset_identity_from_primitive(value["brainvision"]),
        behavioral_inputs=(
            behavioral_input_identity_from_primitive(value["behavioral_inputs"])
            if value["behavioral_inputs"] is not None
            else None
        ),
        resolved_config_fingerprint=value["resolved_config_fingerprint"],
        scientific_fingerprint=value["scientific_fingerprint"],
        fingerprint=value["fingerprint"],
        schema_version=value["schema_version"],
        **snapshot_values,
    )


def _row_content_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "schema_version": EPOCH_ROW_CONTENT_IDENTITY_SCHEMA_VERSION,
        "values": {
            str(column): _normalise_epoch_scalar(str(column), value)
            for column, value in row.items()
            if str(column) not in EPOCH_BINDING_COLUMNS
        },
    }


def _normalise_epoch_scalar(column: str, value: Any) -> Any:
    """Canonicalize scalar values across MNE JSON and pandas CSV storage."""

    normalized = _normalise_scalar(value)
    if normalized == "":
        return None
    if isinstance(normalized, float):
        # MNE persists DataFrame metadata through pandas ``to_json``, whose
        # default double precision is 10 decimal places.
        rounded = round(normalized, 10)
        return int(rounded) if rounded.is_integer() else rounded
    if column == "block" and isinstance(normalized, str):
        # MNE preserves a textual block value in FIF metadata, while pandas
        # infers the same CSV column as numeric on read.
        try:
            numeric = float(normalized)
        except ValueError:
            return normalized
        if math.isfinite(numeric):
            return int(numeric) if numeric.is_integer() else round(numeric, 10)
    return normalized


def _normalise_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return int(numeric) if numeric.is_integer() else numeric
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return str(value) if not isinstance(value, str) else value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _normalise_scalar(value)


def _validate_frame_scope(
    frame: pd.DataFrame,
    participant: str,
    session: str,
    recording_part: str,
    *,
    behavioral: bool = False,
    alignment: bool = False,
    allow_absent: bool = False,
) -> None:
    if behavioral:
        if "subject_id" not in frame or "session" not in frame:
            raise StageIdentityError("Behavioral input lacks subject_id/session scope columns.")
        participants = _unique_tokens(frame["subject_id"], "behavioral subject")
        sessions = {normalise_session(value) for value in frame["session"].dropna().tolist()}
        if participants != {participant} or sessions != {session}:
            raise StageIdentityError("Behavioral content scope does not match requested participant/session.")
    if alignment:
        if "recording_part" not in frame:
            raise StageIdentityError("Alignment input lacks recording_part scope.")
        selected = frame[frame["recording_part"].astype(str) == recording_part]
        if selected.empty:
            raise StageIdentityError("Alignment content does not contain the requested recording part.")
        if "subject" in selected:
            participants = _unique_tokens(selected["subject"], "alignment subject")
            if participants != {participant}:
                raise StageIdentityError("Alignment content subject does not match requested participant.")
    if allow_absent:
        subject_columns = [column for column in ("subject", "subject_id") if column in frame]
        for column in subject_columns:
            values = _unique_tokens(frame[column].dropna(), f"epoch {column}")
            if values and values != {participant}:
                raise StageIdentityError("Epoch metadata participant scope is inconsistent.")
        if "recording_part" in frame:
            values = _unique_tokens(frame["recording_part"].dropna(), "epoch recording part")
            if values and values != {recording_part}:
                raise StageIdentityError("Epoch metadata recording-part scope is inconsistent.")
        if "session" in frame:
            values = {normalise_session(value) for value in frame["session"].dropna().tolist()}
            if values and values != {session}:
                raise StageIdentityError("Epoch metadata session scope is inconsistent.")


def _stable_trial(value: Any) -> str:
    normal = _normalise_scalar(value)
    if normal is None:
        raise StageIdentityError("Epoch trial identity cannot be missing.")
    return str(normal)


def _required_int(value: Any, label: str) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not float(numeric).is_integer():
        raise StageIdentityError(f"Epoch {label} must be an integer.")
    return int(numeric)


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in frame), None)


def _unique_tokens(values: Sequence[Any] | pd.Series, label: str) -> set[str]:
    tokens = {str(value).strip() for value in values if str(value).strip() and not pd.isna(value)}
    if any(not token for token in tokens):
        raise StageIdentityError(f"{label} contains an empty value.")
    return tokens


def _require_token(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageIdentityError(f"{label} must be a non-empty string.")
    return value.strip()


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX) or len(value) != 71:
        raise StageIdentityError(f"{label} must be a prefixed SHA-256 digest.")
    try:
        int(value[len(_SHA256_PREFIX) :], 16)
    except ValueError as error:
        raise StageIdentityError(f"{label} must be a prefixed SHA-256 digest.") from error


def _require_source_file_sha256(value: str, label: str) -> None:
    """Validate the legacy bare digest used by ConfigSourceIdentity.

    Resolved/scientific/content identities are canonical ``sha256:`` values,
    while the selected YAML file identity is deliberately persisted as the
    bare digest returned by ``hashlib.sha256(...).hexdigest()``.
    """

    if not isinstance(value, str) or len(value) != 64:
        raise StageIdentityError(f"{label} must be a bare SHA-256 file digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise StageIdentityError(f"{label} must be a bare SHA-256 file digest.") from error


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise StageIdentityError(f"{label} must be a mapping.")
    actual = set(value)
    if actual != expected:
        raise StageIdentityError(
            f"{label} fields are invalid; missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}."
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
        raise StageIdentityError(f"Could not write stage identity {path}: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
