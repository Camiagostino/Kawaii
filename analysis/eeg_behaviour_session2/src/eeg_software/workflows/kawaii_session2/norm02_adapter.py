"""Read-only source-authenticated adapter for original-cohort Kawaii NORM-02.

This intentionally cohort-pinned adapter is a distinct reproduction mode, not the
reusable archive interface or a new Gate 3C/4 processing pipeline. It authenticates
current immutable publications, reuses existing metadata/source-exposure
validators, and returns accepted ROI traces for the separate aggregation core.
It neither writes outputs nor refits models.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import re
from pathlib import Path
from typing import Mapping
import json

import numpy as np
import pandas as pd

from .norm02_descriptive import AcceptedPartTraces, Norm02ValidationError, accepted_roi_traces_uv

EXPECTED_SUBJECT_COUNTS: Mapping[str, int] = {
    "sub005": 398, "sub009": 204, "sub010": 399, "sub011": 400,
    "sub012": 371, "sub013": 396, "sub014": 400, "sub015": 400,
}
EXPECTED_TOTAL = 2968
AAPS_COUNT = 400
PINNED_AAPS_REFERENCE_SHA256 = "54975215718700947a1f83495950974b91d31bd816543ce368e34e143fa05b10"


@dataclass(frozen=True)
class Norm02BoundSources:
    participant: str
    recording_part: str
    epochs_fif: Path
    epoch_metadata_csv: Path
    gate3c_generation: str
    gate4b_generation: str
    epochs_sha256: str
    metadata_sha256: str
    source_binding_sha256: str
    interpolation_summary_csv: Path | None = None
    interpolation_summary_sha256: str = ""


def corrected_part_root(repository: Path, participant: str, recording_part: str = "part1") -> Path:
    """Never silently fall back to original, R11 V1 or R11 V2 analyses."""
    if participant not in EXPECTED_SUBJECT_COUNTS or recording_part != "part1":
        raise Norm02ValidationError("NORM-02 requires exactly the accepted R11 V3 participant part1 scope")
    root = Path(repository).resolve(strict=True)
    return root / "workspace/kawaii/session2/corrections/r11_qc_v3/subjects" / participant / recording_part


def resolve_published_sources(repository: Path, participant: str, recording_part: str = "part1") -> Norm02BoundSources:
    """Reuse the production validator: checks published finality AND every artifact hash."""
    from .publication import PublicationScope, validate_current_publication

    part_dir = corrected_part_root(repository, participant, recording_part)
    gate3_scope = PublicationScope(stage="gate3c", participant=participant, recording_part=recording_part)
    gate4_scope = PublicationScope(stage="gate4b", branch="cuteness", participant=participant, recording_part=recording_part)
    gate3 = validate_current_publication(part_dir, gate3_scope)
    gate4 = validate_current_publication(part_dir, gate4_scope)
    required_reference = f"immutable-publication:gate3c:{gate3_scope.identity}"
    if gate4.identity_references.get(required_reference) != gate3.generation_id:
        raise Norm02ValidationError("Current Gate 4B cuteness publication does not bind the current Gate 3C generation")
    if (gate3.scope, gate4.scope) != (gate3_scope, gate4_scope):
        raise Norm02ValidationError("Invalid publication scope")
    index = {artifact.label: artifact for artifact in gate3.artifacts}
    if any(label not in index for label in ("epochs_fif", "epoch_metadata_csv", "interpolation_summary")):
        raise Norm02ValidationError("Current Gate 3C publication lacks the epoch pair or interpolation summary")
    epoch = index["epochs_fif"]
    metadata = index["epoch_metadata_csv"]
    interpolation = index["interpolation_summary"]
    payload = {
        "schema": "kawaii_norm02_publication_source_binding_v1",
        "scope": gate3_scope.as_dict(),
        "gate3c_generation": gate3.generation_id,
        "gate4b_generation": gate4.generation_id,
        "gate3c_context": gate3.context_reference,
        "epochs_sha256": epoch.sha256,
        "metadata_sha256": metadata.sha256,
        "interpolation_summary_sha256": interpolation.sha256,
    }
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return Norm02BoundSources(
        participant, recording_part, epoch.path, metadata.path,
        gate3.generation_id, gate4.generation_id, epoch.sha256, metadata.sha256, digest,
        interpolation.path, interpolation.sha256,
    )



def authenticated_interpolated_roi_channels(epochs: object, sources: Norm02BoundSources) -> tuple[str, ...]:
    """Verify *published* Gate 3C interpolation evidence for still-flagged ROI channels.

    The recorded bad flag is preserved (Gate 3C used reset_bads=False).
    A source-bound, byte-hash-verified summary is mandatory whenever an ROI
    channel remains marked bad. Missing/contradictory provenance fails closed.
    No interpolation, MNE metadata mutation or EEG signal reading here.
    """
    from .norm02_descriptive import ROI

    flagged = set(epochs.info["bads"]) & set(ROI)
    if not flagged:
        return ()
    if sources.interpolation_summary_csv is None or not re.fullmatch(
        r"[0-9a-fA-F]{64}", sources.interpolation_summary_sha256
    ):
        raise Norm02ValidationError("Missing authenticated Gate 3C interpolation summary identity")
    try:
        raw = Path(sources.interpolation_summary_csv).read_bytes()
    except OSError as exc:
        raise Norm02ValidationError("Published Gate 3C interpolation summary is inaccessible") from exc
    if sha256(raw).hexdigest() != sources.interpolation_summary_sha256.lower():
        raise Norm02ValidationError("Gate 3C interpolation summary source hash mismatch")
    try:
        table = pd.read_csv(BytesIO(raw))
    except (ValueError, pd.errors.ParserError) as exc:
        raise Norm02ValidationError("Cannot parse authenticated Gate 3C interpolation summary") from exc
    columns = (
        "channel", "manual_confirmed_bad_channel", "interpolated",
        "retained_bad_uninterpolated", "interpolation_method",
        "interpolation_geometry_safety_status", "bad_channel_decision_state",
        "bad_channel_decision_id",
    )
    if any(column not in table.columns for column in columns):
        raise Norm02ValidationError("Gate 3C interpolation summary lacks required disposition columns")
    rows = table.loc[table["channel"].isin(ROI)]
    if len(rows) != len(ROI) or set(rows["channel"]) != set(ROI):
        raise Norm02ValidationError("Gate 3C interpolation summary has missing/duplicate ROI channels")
    authorised = []
    for channel in sorted(flagged):
        row = rows.loc[rows["channel"] == channel].iloc[0]
        # Strict boolean types: never treat a non-empty 'False' string as True.
        def bool_is(name: str, expected: bool) -> bool:
            value = row[name]
            return isinstance(value, (bool, np.bool_)) and bool(value) is expected

        safe = (
            bool_is("manual_confirmed_bad_channel", True)
            and bool_is("interpolated", True)
            and bool_is("retained_bad_uninterpolated", False)
            and row["interpolation_method"] == "mne_interpolate_bads"
            and row["interpolation_geometry_safety_status"] == "safe_unresolved_non_targets_excluded"
            and row["bad_channel_decision_state"] == "REVIEWED_WITH_BAD_CHANNELS"
            and isinstance(row["bad_channel_decision_id"], str)
            and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", row["bad_channel_decision_id"]) is not None
        )
        if not safe:
            raise Norm02ValidationError(
                f"Marked bad ROI {channel} lacks confirmed safe published interpolation disposition"
            )
        authorised.append(channel)
    return tuple(authorised)


def load_accepted_roi_part(
    repository: Path,
    participant: str,
    *,
    config: Mapping[str, object],
    allow_sub012_exposure_ambiguous: bool = False,
) -> AcceptedPartTraces:
    """Reuse Gate 4's loader, identity and exposure contracts; return ROI, not a new FIF.

    Sub012 remains STOP by default until its descriptive (not old/new) inclusion
    has been explicitly resolved. This flag records a conscious inclusion choice;
    it does not invent an exposure label or make inference about old/new status.
    """
    if participant == "sub012" and not allow_sub012_exposure_ambiguous:
        raise Norm02ValidationError("sub012 descriptive exposure-ambiguous inclusion requires explicit disposition")
    sources = resolve_published_sources(repository, participant)

    # Imported lazily so source-only and mocked adapter tests require no MNE runtime.
    from .erp import (
        load_gate3c_epochs_and_metadata,
        _validate_and_rebind_gate3c_epoch_metadata,
        prepare_cuteness_inputs,
        gate4b_cuteness_settings,
        validate_epochs_metadata,
        apply_erp_baseline,
    )
    from .bad_channel_disposition import validate_downstream_retained_bad_channel_safety

    epochs, metadata = load_gate3c_epochs_and_metadata({
        "epochs_fif": sources.epochs_fif,
        "epoch_metadata_csv": sources.epoch_metadata_csv,
    })
    epochs, metadata, _binding = _validate_and_rebind_gate3c_epoch_metadata(
        epochs, metadata, participant=participant, session="session2", recording_part="part1",
        allow_legacy_serialization_compatibility=True,  # same mode as published Gate 4
    )
    epochs, exposure = prepare_cuteness_inputs(
        epochs, metadata, subject=participant, session="session2", recording_part="part1",
        config=config, root=Path(repository),
    )
    if not exposure.required:
        raise Norm02ValidationError("Accepted cohort source-exposure disposition must be enabled and authoritative")
    metadata = exposure.metadata
    settings = gate4b_cuteness_settings(config)
    validate_downstream_retained_bad_channel_safety(epochs.info, settings)
    validate_epochs_metadata(epochs, metadata, settings, participant=participant, session="session2", recording_part="part1")
    apply_erp_baseline(epochs, settings)  # validates only; does not apply baseline again
    expected = EXPECTED_SUBJECT_COUNTS[participant]
    if len(epochs) != expected or len(metadata) != expected:
        raise Norm02ValidationError(f"Corrected R11 V3 participant retention mismatch: {participant} != {expected}")
    if not all(key in metadata.columns for key in ("event_sample", "trial", "image_filename", "M_Kawaii", "kawaii_rating")):
        raise Norm02ValidationError("Gate 3C accepted cuteness metadata is missing mandatory NORM-02 columns")
    # Do not drop uncertain old/new rows from a descriptive rating comparison.
    if participant == "sub012":
        if "source_exposure_status" not in metadata:
            raise Norm02ValidationError("sub012 source-exposure status is unavailable after authoritative disposition")
        unknown = int(metadata["source_exposure_status"].astype("string").eq("AMBIGUOUS_RUN4_EXPOSURE").fillna(False).sum())
        if unknown != 190:
            raise Norm02ValidationError(f"sub012 source-exposure ambiguity count changed: observed {unknown}, expected 190")
    safe_interpolated = authenticated_interpolated_roi_channels(epochs, sources)
    traces = accepted_roi_traces_uv(
        epochs, authenticated_interpolated_roi=safe_interpolated
    )
    return AcceptedPartTraces(
        participant=participant,
        recording_part="part1",
        times_s=np.asarray(epochs.times, dtype=float).copy(),
        traces_uv=traces,
        metadata=metadata.copy(),
        source_binding_sha256=sources.source_binding_sha256,
    )


def load_aaps_reference(path: Path) -> dict[str, float]:
    """Read a separately pinned 400-image reference, not self-derived epoch data."""
    reference = pd.read_csv(path, dtype={"image_filename": str})
    if list(reference.columns) != ["image_filename", "M_Kawaii"]:
        raise Norm02ValidationError("Pinned AAPS reference must contain exactly image_filename,M_Kawaii")
    if len(reference) != AAPS_COUNT or reference.image_filename.isna().any() or reference.image_filename.duplicated().any():
        raise Norm02ValidationError("Pinned AAPS reference must contain exactly 400 distinct images")
    result: dict[str, float] = {}
    for image, raw in zip(reference.image_filename, reference.M_Kawaii):
        from .norm02_descriptive import normative_category
        normative_category(raw)
        result[str(image)] = float(raw)
    if sum(value <= 5.0 for value in result.values()) != 200 or sum(value > 6.5 for value in result.values()) != 200:
        raise Norm02ValidationError("Pinned AAPS scores do not reproduce the accepted 200/200 split")
    if sha256(Path(path).read_bytes()).hexdigest() != PINNED_AAPS_REFERENCE_SHA256:
        raise Norm02ValidationError("AAPS reference bytes differ from the verified pinned source table")
    return result


def validate_full_cohort(parts: Mapping[str, AcceptedPartTraces]) -> None:
    """Require all eight accepted participants and all accepted retained epochs."""
    if set(parts) != set(EXPECTED_SUBJECT_COUNTS):
        raise Norm02ValidationError("NORM-02 requires the entire eight-participant accepted cohort")
    for participant, expected in EXPECTED_SUBJECT_COUNTS.items():
        part = parts[participant]
        if part.participant != participant or part.recording_part != "part1" or len(part.metadata) != expected:
            raise Norm02ValidationError(f"Mismatched participant/retention count for {participant}")
    if sum(len(part.metadata) for part in parts.values()) != EXPECTED_TOTAL:
        raise Norm02ValidationError("Accepted eight-participant cohort total differs from 2,968")
