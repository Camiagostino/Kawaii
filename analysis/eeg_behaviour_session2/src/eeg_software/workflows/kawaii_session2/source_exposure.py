from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .identity import identify_source_file, source_file_identity_from_primitive
from .resolved_config import canonical_sha256
from .stage_identity import bind_epoch_metadata_rows


MANIFEST_SCHEMA_VERSION = "kawaii_source_exposure_manifest_v1"
DISPOSITION_SCHEMA_VERSION = "kawaii_source_exposure_disposition_v1"

VERIFIED_OLD = "VERIFIED_OLD"
VERIFIED_NEW = "VERIFIED_NEW"
AMBIGUOUS_RUN4_EXPOSURE = "AMBIGUOUS_RUN4_EXPOSURE"
SOURCE_MISSING = "SOURCE_MISSING"
SOURCE_MISMATCH = "SOURCE_MISMATCH"

ALLOWED_EXPOSURE_STATES = {
    VERIFIED_OLD,
    VERIFIED_NEW,
    AMBIGUOUS_RUN4_EXPOSURE,
    SOURCE_MISSING,
    SOURCE_MISMATCH,
}
VERIFIED_STATES = {VERIFIED_OLD, VERIFIED_NEW}
AUTHORIZED_INELIGIBLE_STATES = {AMBIGUOUS_RUN4_EXPOSURE}

DISPOSITION_COLUMNS = (
    "original_old_new_status",
    "source_exposure_status",
    "source_verified_old_new_status",
    "recognition_eligible",
    "continuous_cuteness_old_new_eligible",
    "exposure_exclusion_reason",
    "source_exposure_identity",
    "source_exposure_artifact_id",
    "recognition_delay_days",
    "protocol_deviation",
    "protocol_disposition",
)


class SourceExposureValidationError(ValueError):
    """Raised when old/new source verification is absent or untrustworthy."""

    reason_code = "source_exposure_validation_failed"

    def __init__(self, message: str, *, reason_code: str = "source_exposure_validation_failed") -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class SourceExposureResult:
    metadata: pd.DataFrame
    summary: dict[str, Any]
    input_paths: tuple[Path, ...]
    required: bool


def build_source_exposure_participant_record(
    subject: str,
    sources: Sequence[Mapping[str, Any]],
    expected_counts: Mapping[str, int],
    *,
    protocol: Mapping[str, Any],
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one participant record with the canonical historical identity."""

    source_values = [dict(item) for item in sources]
    identity_payload = {
        "subject": str(subject),
        "session": "session2",
        "recording_parts": ["part1"],
        "sources": [
            {
                "schema_version": item["schema_version"],
                "role": item["role"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in source_values
        ],
        "protocol": dict(protocol),
        "notes": [str(value) for value in notes],
    }
    return {
        **identity_payload,
        "source_identity": canonical_sha256(identity_payload),
        "sources": source_values,
        "expected_counts": {str(key): int(value) for key, value in expected_counts.items()},
    }


def participant_manifest_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical participant payload used by the global artifact ID."""

    return {
        "subject": value.get("subject"),
        "session": value.get("session"),
        "recording_parts": list(value.get("recording_parts", []) or []),
        "source_identity": value.get("source_identity"),
        "sources": [
            source_file_identity_from_primitive(item).identity_payload()
            for item in value.get("sources", [])
        ],
        "expected_counts": dict(value.get("expected_counts", {}) or {}),
        "protocol": dict(value.get("protocol", {}) or {}),
        "notes": list(value.get("notes", []) or []),
    }


def source_exposure_artifact_identity(participants: Sequence[Mapping[str, Any]]) -> str:
    """Return the canonical historical global source-exposure artifact ID."""

    return canonical_sha256(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "disposition_schema_version": DISPOSITION_SCHEMA_VERSION,
            "participants": [
                participant_manifest_identity_payload(item) for item in participants
            ],
        }
    )


def source_exposure_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("source_exposure_verification", {}) or {})


def source_exposure_required(subject: str, config: Mapping[str, Any]) -> bool:
    settings = source_exposure_settings(config)
    required = {str(value) for value in settings.get("required_subjects", []) or []}
    return bool(settings.get("enabled", False)) and str(subject) in required


def apply_source_exposure_disposition(
    metadata: pd.DataFrame,
    *,
    subject: str,
    session: str | int,
    recording_part: str,
    config: Mapping[str, Any],
    root: Path | None = None,
) -> SourceExposureResult:
    """Attach and validate provenance-bound old/new trial dispositions.

    Participants outside the explicitly governed current-cohort set are left
    unchanged. Governed participants fail closed unless every supplied row has
    exactly one valid, identity-bound disposition.
    """

    if not source_exposure_required(subject, config):
        return SourceExposureResult(metadata.copy(), {}, (), False)

    settings = source_exposure_settings(config)
    manifest_path = _configured_path(settings.get("manifest_path"), root, "manifest_path")
    manifest = _read_manifest(manifest_path)
    table_path = _resolve_artifact_path(manifest_path, manifest)
    _validate_artifact_identity(table_path, manifest)

    artifact_id = _manifest_artifact_id(manifest)
    expected_id = str(settings.get("expected_artifact_id", "")).strip()
    if not expected_id or artifact_id != expected_id:
        raise SourceExposureValidationError(
            "Source-exposure manifest identity is stale or differs from the configured execution identity.",
            reason_code="stale_source_exposure_identity",
        )

    participant = _participant_manifest(manifest, subject)
    expected_session = _normalise_session(session)
    if str(participant.get("session")) != expected_session:
        raise SourceExposureValidationError(
            f"Source-exposure participant/session scope mismatch for {subject!r}.",
            reason_code="source_exposure_scope_mismatch",
        )
    allowed_parts = {str(value) for value in participant.get("recording_parts", []) or []}
    if allowed_parts and str(recording_part) not in allowed_parts:
        raise SourceExposureValidationError(
            f"Source-exposure recording-part scope mismatch for {subject!r}/{recording_part!r}.",
            reason_code="source_exposure_scope_mismatch",
        )

    table = pd.read_csv(table_path, dtype={"subject": "string", "session": "string", "image_filename": "string"})
    _validate_table_schema(table)
    scoped = table.loc[
        table["subject"].astype(str).eq(str(subject))
        & table["session"].map(_normalise_session).eq(expected_session)
    ].copy()
    if scoped.empty:
        raise SourceExposureValidationError(
            f"No source-exposure dispositions exist for {subject!r}/{expected_session!r}.",
            reason_code="missing_source_exposure_disposition",
        )
    _validate_disposition_rows(scoped, participant, artifact_id)

    clean = _validate_metadata_scope(metadata, subject, expected_session, recording_part)
    existing = clean[[column for column in DISPOSITION_COLUMNS if column in clean]].copy()
    clean = clean.drop(columns=[column for column in DISPOSITION_COLUMNS if column in clean])
    clean["trial"] = _integer_trials(clean["trial"], "metadata")
    clean["image_filename"] = _normalised_items(clean["image_filename"], "metadata")
    scoped["trial"] = _integer_trials(scoped["trial"], "disposition")
    scoped["image_filename"] = _normalised_items(scoped["image_filename"], "disposition")

    merge_columns = ["trial", "image_filename", *DISPOSITION_COLUMNS[:-3]]
    merged = clean.merge(
        scoped[merge_columns],
        on=["trial", "image_filename"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[~merged["_merge"].eq("both"), ["trial", "image_filename"]].to_dict("records")
        raise SourceExposureValidationError(
            f"Required source-exposure disposition is missing for metadata row(s): {missing!r}.",
            reason_code="missing_source_exposure_disposition",
        )
    merged = merged.drop(columns="_merge")

    protocol = dict(participant.get("protocol", {}) or {})
    merged["recognition_delay_days"] = protocol.get("recognition_delay_days", 1)
    merged["protocol_deviation"] = bool(protocol.get("protocol_deviation", False))
    merged["protocol_disposition"] = str(protocol.get("protocol_disposition", "STANDARD"))

    original = _normalised_old_new(merged["old_new_status"], "Session 2 old_new_status")
    recorded_original = _normalised_old_new(merged["original_old_new_status"], "recorded original_old_new_status")
    if not original.equals(recorded_original):
        raise SourceExposureValidationError(
            "Source-derived old/new disposition conflicts with the preserved Session 2 label.",
            reason_code="source_session2_old_new_mismatch",
        )
    verified = merged["source_exposure_status"].isin(VERIFIED_STATES)
    derived = merged["source_verified_old_new_status"].astype("string").str.strip().str.lower()
    if not original.loc[verified].equals(derived.loc[verified]):
        raise SourceExposureValidationError(
            "Source-derived old/new status mismatches the Session 2 label.",
            reason_code="source_session2_old_new_mismatch",
        )

    if not existing.empty:
        _validate_idempotent_existing(existing.reset_index(drop=True), merged[list(existing.columns)].reset_index(drop=True))

    summary = _build_summary(scoped, merged, participant, artifact_id)
    source_paths = tuple(
        [manifest_path, table_path]
        + [Path(str(item["logical_path"])) for item in participant.get("sources", []) if Path(str(item["logical_path"])).exists()]
    )
    return SourceExposureResult(merged, summary, source_paths, True)


def prepare_recognition_inputs(
    epochs,
    metadata: pd.DataFrame,
    *,
    subject: str,
    session: str | int,
    recording_part: str,
    config: Mapping[str, Any],
    root: Path | None = None,
):
    result = apply_source_exposure_disposition(
        metadata,
        subject=subject,
        session=session,
        recording_part=recording_part,
        config=config,
        root=root,
    )
    if not result.required:
        return epochs, result
    eligible = _strict_bool_series(result.metadata["recognition_eligible"], "recognition_eligible")
    selected_indices = result.metadata.index[eligible].tolist()
    if not selected_indices:
        raise SourceExposureValidationError(
            "No source-verified recognition-eligible trials remain.",
            reason_code="no_source_verified_recognition_trials",
        )
    selected_epochs = epochs[selected_indices]
    selected_metadata = result.metadata.loc[eligible].reset_index(drop=True)
    rebound = bind_epoch_metadata_rows(
        selected_metadata,
        participant=subject,
        session=session,
        recording_part=recording_part,
    )
    selected_epochs.metadata = rebound.copy()
    return selected_epochs, SourceExposureResult(rebound, result.summary, result.input_paths, True)


def prepare_cuteness_inputs(
    epochs,
    metadata: pd.DataFrame,
    *,
    subject: str,
    session: str | int,
    recording_part: str,
    config: Mapping[str, Any],
    root: Path | None = None,
):
    result = apply_source_exposure_disposition(
        metadata,
        subject=subject,
        session=session,
        recording_part=recording_part,
        config=config,
        root=root,
    )
    if not result.required:
        return epochs, result
    rebound = bind_epoch_metadata_rows(
        result.metadata.reset_index(drop=True),
        participant=subject,
        session=session,
        recording_part=recording_part,
    )
    prepared_epochs = epochs.copy()
    prepared_epochs.metadata = rebound.copy()
    return prepared_epochs, SourceExposureResult(rebound, result.summary, result.input_paths, True)


def append_source_exposure_validation(validation: pd.DataFrame, result: SourceExposureResult) -> pd.DataFrame:
    if not result.required:
        return validation
    row = pd.DataFrame([{"check": "source_exposure_disposition", "status": "ok", "detail": result.summary}])
    return pd.concat([validation, row], ignore_index=True)


def attach_source_exposure_summary(table: pd.DataFrame, result: SourceExposureResult) -> pd.DataFrame:
    if not result.required:
        return table
    output = table.copy()
    for key, value in result.summary.items():
        output[f"source_exposure_{key}"] = value
    return output


def validate_continuous_cuteness_disposition(table: pd.DataFrame) -> None:
    if "continuous_cuteness_old_new_eligible" not in table:
        return
    eligible = _strict_bool_series(
        table["continuous_cuteness_old_new_eligible"],
        "continuous_cuteness_old_new_eligible",
    )
    model_columns = [column for column in ("model_eligible_reduced", "model_eligible_full") if column in table]
    for column in model_columns:
        marked = _strict_bool_series(table[column], column)
        if bool((marked & ~eligible).any()):
            raise SourceExposureValidationError(
                f"Continuous-cuteness group preparation restored source-ineligible row(s) in {column!r}.",
                reason_code="continuous_cuteness_disposition_restored",
            )


def _configured_path(value: Any, root: Path | None, label: str) -> Path:
    if value is None or not str(value).strip():
        raise SourceExposureValidationError(
            f"Source-exposure {label} is not configured.",
            reason_code="missing_source_exposure_configuration",
        )
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(root or ".") / path
    return path.resolve()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceExposureValidationError(
            f"Cannot read source-exposure manifest {path}: {error}",
            reason_code="missing_source_exposure_disposition",
        ) from error
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SourceExposureValidationError(
            "Malformed or unsupported source-exposure manifest schema.",
            reason_code="malformed_source_exposure_disposition",
        )
    return value


def _resolve_artifact_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    artifact = dict(manifest.get("artifact", {}) or {})
    logical = str(artifact.get("logical_path", "")).strip()
    if not logical:
        raise SourceExposureValidationError(
            "Source-exposure manifest lacks an artifact logical path.",
            reason_code="malformed_source_exposure_disposition",
        )
    path = Path(logical)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _validate_artifact_identity(path: Path, manifest: Mapping[str, Any]) -> None:
    try:
        expected = source_file_identity_from_primitive(manifest.get("artifact"))
        current = identify_source_file(path, expected.role, logical_path=expected.logical_path)
    except Exception as error:
        raise SourceExposureValidationError(
            f"Source-exposure artifact identity cannot be verified: {error}",
            reason_code="stale_source_exposure_identity",
        ) from error
    if current.identity_payload() != expected.identity_payload():
        raise SourceExposureValidationError(
            "Source-exposure artifact content identity is stale or mismatched.",
            reason_code="stale_source_exposure_identity",
        )


def _manifest_artifact_id(manifest: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": manifest.get("schema_version"),
        "disposition_schema_version": manifest.get("disposition_schema_version"),
        "participants": [participant_manifest_identity_payload(item) for item in manifest.get("participants", [])],
    }
    actual = canonical_sha256(payload)
    if actual != manifest.get("artifact_id"):
        raise SourceExposureValidationError(
            "Source-exposure manifest artifact_id is internally inconsistent.",
            reason_code="stale_source_exposure_identity",
        )
    return actual


def _participant_manifest(manifest: Mapping[str, Any], subject: str) -> dict[str, Any]:
    matches = [dict(item) for item in manifest.get("participants", []) if str(item.get("subject")) == str(subject)]
    if len(matches) != 1:
        raise SourceExposureValidationError(
            f"Expected exactly one participant manifest for {subject!r}; found {len(matches)}.",
            reason_code="source_exposure_scope_mismatch",
        )
    participant = matches[0]
    payload = {
        "subject": participant.get("subject"),
        "session": participant.get("session"),
        "recording_parts": list(participant.get("recording_parts", []) or []),
        "sources": [source_file_identity_from_primitive(item).identity_payload() for item in participant.get("sources", [])],
        "protocol": dict(participant.get("protocol", {}) or {}),
        "notes": list(participant.get("notes", []) or []),
    }
    if canonical_sha256(payload) != participant.get("source_identity"):
        raise SourceExposureValidationError(
            f"Participant source identity is stale for {subject!r}.",
            reason_code="stale_source_exposure_identity",
        )
    return participant


def _validate_table_schema(table: pd.DataFrame) -> None:
    required = {"schema_version", "subject", "session", "trial", "image_filename", *DISPOSITION_COLUMNS[:-3]}
    missing = sorted(required - set(table.columns))
    if missing:
        raise SourceExposureValidationError(
            f"Source-exposure disposition table lacks column(s): {missing!r}.",
            reason_code="malformed_source_exposure_disposition",
        )
    if not table["schema_version"].astype(str).eq(DISPOSITION_SCHEMA_VERSION).all():
        raise SourceExposureValidationError(
            "Source-exposure disposition table contains an unsupported schema version.",
            reason_code="malformed_source_exposure_disposition",
        )


def _validate_disposition_rows(table: pd.DataFrame, participant: Mapping[str, Any], artifact_id: str) -> None:
    key = ["subject", "session", "trial", "image_filename"]
    if table.duplicated(key, keep=False).any() or table.duplicated(["subject", "session", "image_filename"], keep=False).any():
        raise SourceExposureValidationError(
            "Source-exposure disposition contains duplicate trial/item mapping.",
            reason_code="duplicate_source_exposure_mapping",
        )
    statuses = table["source_exposure_status"].astype(str)
    malformed = sorted(set(statuses) - ALLOWED_EXPOSURE_STATES)
    if malformed:
        raise SourceExposureValidationError(
            f"Malformed source-exposure status value(s): {malformed!r}.",
            reason_code="malformed_source_exposure_disposition",
        )
    if not table["source_exposure_artifact_id"].astype(str).eq(artifact_id).all():
        raise SourceExposureValidationError(
            "Disposition rows are bound to a stale/wrong verification artifact identity.",
            reason_code="stale_source_exposure_identity",
        )
    if not table["source_exposure_identity"].astype(str).eq(str(participant.get("source_identity"))).all():
        raise SourceExposureValidationError(
            "Disposition rows are bound to a stale/wrong participant source identity.",
            reason_code="stale_source_exposure_identity",
        )
    recognition = _strict_bool_series(table["recognition_eligible"], "recognition_eligible")
    continuous = _strict_bool_series(
        table["continuous_cuteness_old_new_eligible"],
        "continuous_cuteness_old_new_eligible",
    )
    verified = statuses.isin(VERIFIED_STATES)
    ambiguous = statuses.isin(AUTHORIZED_INELIGIBLE_STATES)
    unexpected = ~(verified | ambiguous)
    derived = table["source_verified_old_new_status"].astype("string").str.strip().str.lower()
    expected_derived = statuses.map({VERIFIED_OLD: "old", VERIFIED_NEW: "new"}).astype("string")
    if not derived.loc[verified].equals(expected_derived.loc[verified]):
        raise SourceExposureValidationError(
            "Verified source-exposure row has a malformed source-derived old/new value.",
            reason_code="malformed_source_exposure_disposition",
        )
    if derived.loc[~verified].notna().any() and derived.loc[~verified].ne("").any():
        raise SourceExposureValidationError(
            "Ambiguous/unresolved exposure was silently converted to old or new.",
            reason_code="ambiguous_source_exposure_converted",
        )
    if bool((~recognition.loc[verified] | ~continuous.loc[verified]).any()):
        raise SourceExposureValidationError(
            "Verified exposure row is inconsistently marked ineligible.",
            reason_code="malformed_source_exposure_disposition",
        )
    if bool((recognition.loc[ambiguous] | continuous.loc[ambiguous]).any()):
        raise SourceExposureValidationError(
            "Unresolved ambiguity is incorrectly marked eligible.",
            reason_code="ambiguous_source_exposure_eligible",
        )
    if bool(unexpected.any()):
        raise SourceExposureValidationError(
            "Unexpected missing/mismatched source evidence cannot enter old/new-dependent execution.",
            reason_code="unresolved_source_exposure_invalid",
        )
    reasons = table["exposure_exclusion_reason"].astype("string").fillna("").str.strip()
    if reasons.loc[ambiguous].eq("").any():
        raise SourceExposureValidationError(
            "Authorized ineligible exposure rows require an explicit exclusion reason.",
            reason_code="malformed_source_exposure_disposition",
        )
    observed = statuses.value_counts().to_dict()
    expected = dict(participant.get("expected_counts", {}) or {})
    for status in (VERIFIED_OLD, VERIFIED_NEW, AMBIGUOUS_RUN4_EXPOSURE):
        if int(observed.get(status, 0)) != int(expected.get(status, 0)):
            raise SourceExposureValidationError(
                f"Source-exposure count differs from the authorized manifest for {status}.",
                reason_code="source_exposure_count_mismatch",
            )


def _validate_metadata_scope(metadata: pd.DataFrame, subject: str, session: str, recording_part: str) -> pd.DataFrame:
    required = {"trial", "image_filename", "old_new_status"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise SourceExposureValidationError(
            f"Old/new-dependent analysis metadata lacks column(s): {missing!r}.",
            reason_code="missing_source_exposure_disposition",
        )
    output = metadata.copy().reset_index(drop=True)
    for column in ("subject_id", "epoch_identity_participant"):
        if column in output and not output[column].dropna().astype(str).eq(str(subject)).all():
            raise SourceExposureValidationError(
                f"Metadata participant identity in {column!r} does not match {subject!r}.",
                reason_code="source_exposure_scope_mismatch",
            )
    for column in ("session", "epoch_identity_session"):
        if column in output and not output[column].dropna().map(_normalise_session).eq(session).all():
            raise SourceExposureValidationError(
                f"Metadata session identity in {column!r} does not match {session!r}.",
                reason_code="source_exposure_scope_mismatch",
            )
    for column in ("recording_part", "epoch_identity_recording_part"):
        if column in output and not output[column].dropna().astype(str).eq(str(recording_part)).all():
            raise SourceExposureValidationError(
                f"Metadata recording-part identity in {column!r} does not match {recording_part!r}.",
                reason_code="source_exposure_scope_mismatch",
            )
    if output.duplicated(["trial"], keep=False).any() or output.duplicated(["image_filename"], keep=False).any():
        raise SourceExposureValidationError(
            "Old/new-dependent metadata contains duplicate trial/item keys.",
            reason_code="duplicate_source_exposure_mapping",
        )
    return output


def _build_summary(table: pd.DataFrame, current: pd.DataFrame, participant: Mapping[str, Any], artifact_id: str) -> dict[str, Any]:
    statuses = table["source_exposure_status"].astype(str)
    current_statuses = current["source_exposure_status"].astype(str)
    recognition = _strict_bool_series(current["recognition_eligible"], "recognition_eligible")
    continuous = _strict_bool_series(current["continuous_cuteness_old_new_eligible"], "continuous_cuteness_old_new_eligible")
    reasons = current.loc[~recognition, "exposure_exclusion_reason"].astype(str).value_counts().to_dict()
    protocol = dict(participant.get("protocol", {}) or {})
    return {
        "artifact_id": artifact_id,
        "source_identity": participant.get("source_identity"),
        "verified_old": int(statuses.eq(VERIFIED_OLD).sum()),
        "verified_new": int(statuses.eq(VERIFIED_NEW).sum()),
        "ambiguous_ineligible": int(statuses.isin(AUTHORIZED_INELIGIBLE_STATES).sum()),
        "mismatches": int(statuses.eq(SOURCE_MISMATCH).sum()),
        "artifact_recognition_eligible": int(_strict_bool_series(table["recognition_eligible"], "recognition_eligible").sum()),
        "artifact_continuous_cuteness_eligible": int(_strict_bool_series(table["continuous_cuteness_old_new_eligible"], "continuous_cuteness_old_new_eligible").sum()),
        "current_input_rows": int(len(current)),
        "current_input_verified_old": int(current_statuses.eq(VERIFIED_OLD).sum()),
        "current_input_verified_new": int(current_statuses.eq(VERIFIED_NEW).sum()),
        "current_input_ambiguous": int(current_statuses.isin(AUTHORIZED_INELIGIBLE_STATES).sum()),
        "current_input_recognition_eligible": int(recognition.sum()),
        "current_input_continuous_cuteness_eligible": int(continuous.sum()),
        "exclusion_reasons": json.dumps(reasons, sort_keys=True, separators=(",", ":")),
        "recognition_delay_days": int(protocol.get("recognition_delay_days", 1)),
        "protocol_deviation": bool(protocol.get("protocol_deviation", False)),
        "protocol_disposition": str(protocol.get("protocol_disposition", "STANDARD")),
    }


def _validate_idempotent_existing(existing: pd.DataFrame, reapplied: pd.DataFrame) -> None:
    left = existing.astype("string").fillna("")
    right = reapplied.astype("string").fillna("")
    if list(left.columns) != list(right.columns) or not left.equals(right):
        raise SourceExposureValidationError(
            "Repeated source-exposure application found conflicting existing disposition fields.",
            reason_code="conflicting_source_exposure_disposition",
        )


def _strict_bool_series(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalised = values.astype("string").str.strip().str.lower()
    malformed = ~normalised.isin(["true", "false"])
    if bool(malformed.any()):
        raise SourceExposureValidationError(
            f"Malformed boolean disposition value in {label!r}.",
            reason_code="malformed_source_exposure_disposition",
        )
    return normalised.eq("true")


def _integer_trials(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | numeric.map(lambda value: not float(value).is_integer() if pd.notna(value) else True)
    if bool(invalid.any()):
        raise SourceExposureValidationError(
            f"Malformed trial identity in {label}.",
            reason_code="malformed_source_exposure_disposition",
        )
    return numeric.astype(int)


def _normalised_items(values: pd.Series, label: str) -> pd.Series:
    output = values.astype("string").str.strip().str.lower()
    if output.isna().any() or output.eq("").any():
        raise SourceExposureValidationError(
            f"Missing item identity in {label}.",
            reason_code="malformed_source_exposure_disposition",
        )
    return output


def _normalised_old_new(values: pd.Series, label: str) -> pd.Series:
    output = values.astype("string").str.strip().str.lower()
    if not output.isin(["old", "new"]).all():
        raise SourceExposureValidationError(
            f"Malformed {label}; expected only old/new.",
            reason_code="malformed_source_exposure_disposition",
        )
    return output


def _normalise_session(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("session"):
        text = text[7:]
    try:
        return f"session{int(float(text))}"
    except (TypeError, ValueError):
        return str(value).strip().lower()
