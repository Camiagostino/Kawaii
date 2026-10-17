from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from io import StringIO
import json
from pathlib import Path

import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.identity import (
    identify_behavioral_inputs,
    identify_brainvision_dataset,
)
from eeg_software.workflows.kawaii_session2.response_timing_qc import build_trial_level_qc
from eeg_software.workflows.kawaii_session2.resolved_snapshot import ResolvedRunSnapshotReference
from eeg_software.workflows.kawaii_session2.resolved_config import canonical_sha256
from eeg_software.workflows.kawaii_session2.stage_identity import (
    ALIGNMENT_IDENTITY_FILENAME,
    DatasetContextIdentity,
    EPOCH_ROW_CONTENT_COLUMN,
    StageIdentityError,
    alignment_identity_path,
    bind_epoch_metadata_rows,
    build_alignment_artifact_identity,
    build_dataset_context_identity,
    build_timing_qc_input_identity,
    epoch_metadata_binding,
    read_alignment_artifact_identity,
    read_validated_alignment,
    validate_alignment_artifact_identity,
    validate_epoch_metadata_binding,
    write_alignment_artifact_identity,
)


def _write_triplet(root: Path, *, eeg: bytes = b"samples") -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    vhdr = root / "recording.vhdr"
    vmrk = root / "recording.vmrk"
    eeg_path = root / "recording.eeg"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\nDataFile=recording.eeg\nMarkerFile=recording.vmrk\n",
        encoding="utf-8",
    )
    vmrk.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "[Common Infos]\nDataFile=recording.eeg\n",
        encoding="utf-8",
    )
    eeg_path.write_bytes(eeg)
    return vhdr, vmrk, eeg_path


def _context(
    root: Path,
    *,
    participant: str = "sub005",
    part: str = "part1",
    behavioral_content: str = "subject_id,session,trial\nsub005,2,1\n",
    schedule: bool = False,
    eeg: bytes = b"samples",
    config_fingerprint: str = "sha256:" + "1" * 64,
    scientific_fingerprint: str = "sha256:" + "2" * 64,
    snapshot: ResolvedRunSnapshotReference | None = None,
) -> tuple[DatasetContextIdentity, Path]:
    vhdr, vmrk, eeg_path = _write_triplet(root / "eeg", eeg=eeg)
    source = root / ("schedule.csv" if schedule else "behavior.csv")
    source.write_text(behavioral_content, encoding="utf-8")
    behavioral = identify_behavioral_inputs(
        behavioral_log_path=None if schedule else source,
        schedule_path=source if schedule else None,
    )
    return (
        build_dataset_context_identity(
            participant=participant,
            session="session2",
            recording_part=part,
            brainvision=identify_brainvision_dataset(vhdr, vmrk, eeg_path),
            behavioral_inputs=behavioral,
            resolved_config_fingerprint=config_fingerprint,
            scientific_fingerprint=scientific_fingerprint,
            resolved_snapshot=snapshot,
        ),
        source,
    )


def _alignment(participant: str = "sub005", part: str = "part1") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": participant,
                "recording_part": part,
                "trial": 1,
                "block": 1,
                "image_onset_sample": 100,
                "recognition_response_sample": 550,
                "image_onset_code": 120,
                "recognition_response_code": 150,
            }
        ]
    )


def _alignment_artifact(
    root: Path,
    context: DatasetContextIdentity,
    alignment: pd.DataFrame | None = None,
):
    table = _alignment(context.participant, context.recording_part) if alignment is None else alignment
    path = root / "trial_alignment.csv"
    table.to_csv(path, index=False)
    return table, path, build_alignment_artifact_identity(
        contexts=[context], alignment=table, output_path=path
    )


def _epoch_rows(
    *, participant: str = "sub005", part: str = "part1", condition_2: str = "new"
) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "subject": participant,
                "recording_part": part,
                "trial": 1,
                "event_sample": 100,
                "event_code": 120,
                "condition": "old",
            },
            {
                "subject": participant,
                "recording_part": part,
                "trial": 2,
                "event_sample": 200,
                "event_code": 121,
                "condition": condition_2,
            },
        ]
    )
    return bind_epoch_metadata_rows(
        frame,
        participant=participant,
        session="session2",
        recording_part=part,
    )


def _timing_settings() -> dict:
    return {
        "sampling_rate_hz": 500.0,
        "response_types": {
            "recognition": {
                "behavioural_rt": "recognition_rt_s",
                "timeout": "recognition_timeout",
                "onset_sample": "image_onset_sample",
                "response_sample": "recognition_response_sample",
                "onset_code": "image_onset_code",
                "response_code": "recognition_response_code",
            }
        },
    }


def _behavioral_frame(*, participant: str = "sub005", session: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": participant,
                "session": session,
                "block": 1,
                "trial": 1,
                "recognition_rt_s": 0.8,
                "recognition_timeout": 0,
            }
        ]
    )


def test_dataset_context_is_deterministic_immutable_and_scope_bound(tmp_path: Path) -> None:
    first, _ = _context(tmp_path / "first")
    second, _ = _context(tmp_path / "second")
    assert first.fingerprint == second.fingerprint
    with pytest.raises(FrozenInstanceError):
        first.participant = "sub999"  # type: ignore[misc]
    with pytest.raises(StageIdentityError, match="fingerprint"):
        replace(
            first,
            participant="sub006",
            fingerprint="sha256:" + "0" * 64,
        )


@pytest.mark.parametrize("field,value", [("participant", "sub006"), ("part", "part2")])
def test_dataset_context_participant_or_part_changes_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    baseline, _ = _context(tmp_path / "baseline")
    kwargs = {field: value}
    changed, _ = _context(tmp_path / "changed", **kwargs)
    assert changed.fingerprint != baseline.fingerprint


def test_behavioral_content_change_alters_dataset_context(tmp_path: Path) -> None:
    before, _ = _context(tmp_path / "before")
    after, _ = _context(
        tmp_path / "after",
        behavioral_content="subject_id,session,trial\nsub005,2,99\n",
    )
    assert after.fingerprint != before.fingerprint


def test_brainvision_content_change_alters_dataset_context(tmp_path: Path) -> None:
    before, _ = _context(tmp_path / "before", eeg=b"first")
    after, _ = _context(tmp_path / "after", eeg=b"replacement")
    assert after.fingerprint != before.fingerprint


def test_config_fingerprint_change_alters_dataset_context(tmp_path: Path) -> None:
    before, _ = _context(tmp_path / "before")
    after, _ = _context(
        tmp_path / "after", config_fingerprint="sha256:" + "9" * 64
    )
    assert after.fingerprint != before.fingerprint


def test_absolute_root_and_snapshot_path_are_not_identity_authority(tmp_path: Path) -> None:
    snapshot_a = ResolvedRunSnapshotReference(
        path=tmp_path / "machine-a" / "snapshot.json",
        schema_version="snapshot-v1",
        snapshot_state="requested_resolved",
        snapshot_content_sha256="sha256:" + "3" * 64,
        source_config_sha256="4" * 64,
        scientific_fingerprint="sha256:" + "2" * 64,
        resolved_config_fingerprint="sha256:" + "1" * 64,
    )
    snapshot_b = replace(snapshot_a, path=tmp_path / "machine-b" / "snapshot.json")
    first, _ = _context(tmp_path / "machine-a" / "data", snapshot=snapshot_a)
    second, _ = _context(tmp_path / "machine-b" / "data", snapshot=snapshot_b)
    assert first.fingerprint == second.fingerprint
    assert first.to_primitive()["resolved_snapshot"]["path"] != second.to_primitive()["resolved_snapshot"]["path"]


def test_alignment_contract_round_trips_and_matching_inputs_validate(tmp_path: Path) -> None:
    context, _ = _context(tmp_path / "inputs")
    table, path, artifact = _alignment_artifact(tmp_path, context)
    contract_path = write_alignment_artifact_identity(artifact, alignment_identity_path(path))
    assert contract_path.name == ALIGNMENT_IDENTITY_FILENAME
    assert read_alignment_artifact_identity(contract_path) == artifact
    validate_alignment_artifact_identity(
        artifact, expected_contexts=[context], alignment_path=path, alignment=table
    )
    loaded, loaded_identity = read_validated_alignment(path, expected_contexts=[context])
    pd.testing.assert_frame_equal(loaded, table)
    assert loaded_identity == artifact
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("source_kind", ["behavior", "schedule"])
def test_same_path_behavioral_or_schedule_replacement_invalidates_alignment(
    tmp_path: Path, source_kind: str
) -> None:
    context, source = _context(tmp_path / "inputs", schedule=source_kind == "schedule")
    table, _, artifact = _alignment_artifact(tmp_path, context)
    source.write_text("subject_id,session,trial\nsub005,2,99\n", encoding="utf-8")
    replacement_inputs = identify_behavioral_inputs(
        behavioral_log_path=source if source_kind == "behavior" else None,
        schedule_path=source if source_kind == "schedule" else None,
    )
    replacement = build_dataset_context_identity(
        participant=context.participant,
        session=context.session,
        recording_part=context.recording_part,
        brainvision=context.brainvision,
        behavioral_inputs=replacement_inputs,
        resolved_config_fingerprint=context.resolved_config_fingerprint,
        scientific_fingerprint=context.scientific_fingerprint,
    )
    with pytest.raises(StageIdentityError, match="context"):
        validate_alignment_artifact_identity(
            artifact, expected_contexts=[replacement], alignment=table
        )


def test_brainvision_source_replacement_invalidates_alignment(tmp_path: Path) -> None:
    context, _ = _context(tmp_path / "inputs")
    table, _, artifact = _alignment_artifact(tmp_path, context)
    eeg = tmp_path / "inputs" / "eeg" / "recording.eeg"
    eeg.write_bytes(b"replacement")
    changed = build_dataset_context_identity(
        participant=context.participant,
        session=context.session,
        recording_part=context.recording_part,
        brainvision=identify_brainvision_dataset(eeg.with_suffix(".vhdr"), eeg.with_suffix(".vmrk"), eeg),
        behavioral_inputs=context.behavioral_inputs,
        resolved_config_fingerprint=context.resolved_config_fingerprint,
        scientific_fingerprint=context.scientific_fingerprint,
    )
    with pytest.raises(StageIdentityError, match="context"):
        validate_alignment_artifact_identity(artifact, expected_contexts=[changed], alignment=table)


@pytest.mark.parametrize(
    "change",
    ["participant", "part", "config"],
)
def test_alignment_rejects_scope_or_config_mismatch(tmp_path: Path, change: str) -> None:
    context, _ = _context(tmp_path / "baseline")
    table, _, artifact = _alignment_artifact(tmp_path, context)
    kwargs: dict[str, str] = {}
    if change == "participant":
        kwargs["participant"] = "sub006"
    elif change == "part":
        kwargs["part"] = "part2"
    else:
        kwargs["config_fingerprint"] = "sha256:" + "8" * 64
    changed, _ = _context(tmp_path / "changed", **kwargs)
    with pytest.raises(StageIdentityError, match="context"):
        validate_alignment_artifact_identity(artifact, expected_contexts=[changed], alignment=table)


@pytest.mark.parametrize("mutation", ["malformed", "incomplete", "tampered"])
def test_alignment_contract_malformed_incomplete_or_tampered_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    context, _ = _context(tmp_path / "inputs")
    _, path, artifact = _alignment_artifact(tmp_path, context)
    contract = write_alignment_artifact_identity(artifact, alignment_identity_path(path))
    if mutation == "malformed":
        contract.write_text("{not-json", encoding="utf-8")
    else:
        payload = json.loads(contract.read_text(encoding="utf-8"))
        if mutation == "incomplete":
            payload.pop("contexts")
        else:
            payload["alignment_payload_fingerprint"] = "sha256:" + "f" * 64
        contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StageIdentityError):
        read_alignment_artifact_identity(contract)


def test_legacy_alignment_without_contract_fails_closed(tmp_path: Path) -> None:
    context, _ = _context(tmp_path / "inputs")
    table = _alignment()
    path = tmp_path / "trial_alignment.csv"
    table.to_csv(path, index=False)
    with pytest.raises(StageIdentityError, match="missing"):
        read_validated_alignment(path, expected_contexts=[context])


def test_epoch_metadata_binding_validates_exact_ordered_content() -> None:
    embedded = _epoch_rows()
    external = embedded.copy()
    binding = validate_epoch_metadata_binding(
        embedded,
        external,
        participant="sub005",
        session="session2",
        recording_part="part1",
    )
    assert len(binding.rows) == 2
    assert binding.fingerprint == epoch_metadata_binding(embedded).fingerprint


def _serialization_sensitive_epoch_row() -> tuple[pd.DataFrame, dict[str, object]]:
    payload = {
        "subject": "sub005",
        "recording_part": "part1",
        "trial": 1,
        "event_sample": 22570,
        "event_code": 121,
        "block": "1",
        "M_Kawaii": 2.34146341463414,
        "M_Arousal": 4.78048780487804,
        "sequence_warning": "",
        "exclusion_reason": "",
    }
    return pd.DataFrame([payload]), payload


def _serialized_epoch_copies(bound: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # This is MNE's exact metadata persistence boundary: reset_index(),
    # pandas JSON records at its default 10-decimal precision, then records.
    records = json.loads(bound.reset_index().to_json(orient="records"))
    embedded = pd.DataFrame.from_records(records).set_index("index").reset_index(drop=True)
    csv = StringIO()
    embedded.to_csv(csv, index=False)
    csv.seek(0)
    external = pd.read_csv(csv)
    return embedded, external


def test_epoch_content_identity_is_stable_across_mne_json_and_csv_serialization() -> None:
    raw, _ = _serialization_sensitive_epoch_row()
    bound = bind_epoch_metadata_rows(
        raw,
        participant="sub005",
        session="session2",
        recording_part="part1",
    )
    embedded, external = _serialized_epoch_copies(bound)

    binding = validate_epoch_metadata_binding(
        embedded,
        external,
        participant="sub005",
        session="session2",
        recording_part="part1",
    )

    assert binding.rows[0].content_fingerprint == bound.loc[0, EPOCH_ROW_CONTENT_COLUMN]


def test_validated_legacy_publication_serialization_is_accepted_and_rebound() -> None:
    raw, legacy_payload = _serialization_sensitive_epoch_row()
    legacy = bind_epoch_metadata_rows(
        raw,
        participant="sub005",
        session="session2",
        recording_part="part1",
    )
    legacy.loc[0, EPOCH_ROW_CONTENT_COLUMN] = canonical_sha256(legacy_payload)
    embedded, external = _serialized_epoch_copies(legacy)

    with pytest.raises(StageIdentityError, match="stale or tampered"):
        validate_epoch_metadata_binding(embedded, external)

    binding = validate_epoch_metadata_binding(
        embedded,
        external,
        participant="sub005",
        session="session2",
        recording_part="part1",
        allow_legacy_serialization_compatibility=True,
    )

    assert binding.rows[0].content_fingerprint == canonical_sha256(legacy_payload)


def test_legacy_serialization_compatibility_still_rejects_genuine_row_mutation() -> None:
    raw, legacy_payload = _serialization_sensitive_epoch_row()
    legacy = bind_epoch_metadata_rows(
        raw,
        participant="sub005",
        session="session2",
        recording_part="part1",
    )
    legacy.loc[0, EPOCH_ROW_CONTENT_COLUMN] = canonical_sha256(legacy_payload)
    embedded, external = _serialized_epoch_copies(legacy)
    external.loc[0, "M_Arousal"] = 9.0

    with pytest.raises(StageIdentityError, match="beyond allowed FIF/CSV serialization"):
        validate_epoch_metadata_binding(
            embedded,
            external,
            allow_legacy_serialization_compatibility=True,
        )


def test_same_length_reordered_epoch_metadata_fails_closed() -> None:
    embedded = _epoch_rows()
    external = embedded.iloc[::-1].reset_index(drop=True)
    with pytest.raises(StageIdentityError, match="order"):
        validate_epoch_metadata_binding(embedded, external)


@pytest.mark.parametrize("participant,part", [("sub999", "part1"), ("sub005", "part2")])
def test_rescoped_epoch_metadata_is_rejected(participant: str, part: str) -> None:
    embedded = _epoch_rows()
    foreign = _epoch_rows(participant=participant, part=part)
    with pytest.raises(StageIdentityError, match="scope|key set"):
        validate_epoch_metadata_binding(embedded, foreign)


def test_duplicate_epoch_stable_keys_are_rejected() -> None:
    duplicate = pd.DataFrame(
        [
            {"subject": "sub005", "recording_part": "part1", "trial": 1, "event_sample": 100, "event_code": 120},
            {"subject": "sub005", "recording_part": "part1", "trial": 1, "event_sample": 100, "event_code": 120},
        ]
    )
    with pytest.raises(StageIdentityError, match="duplicate"):
        bind_epoch_metadata_rows(
            duplicate,
            participant="sub005",
            session="session2",
            recording_part="part1",
        )


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_missing_or_extra_epoch_key_is_rejected(change: str) -> None:
    embedded = _epoch_rows()
    if change == "missing":
        external = embedded.iloc[:1].copy()
    else:
        raw = embedded.drop(columns=[column for column in embedded if column.startswith("epoch_")])
        extra = raw.iloc[[1]].copy()
        extra["trial"] = 3
        extra["event_sample"] = 300
        raw = pd.concat([raw, extra], ignore_index=True)
        external = bind_epoch_metadata_rows(
            raw,
            participant="sub005",
            session="session2",
            recording_part="part1",
        )
    with pytest.raises(StageIdentityError, match="key set"):
        validate_epoch_metadata_binding(embedded, external)


def test_same_key_with_conflicting_epoch_metadata_content_is_rejected() -> None:
    embedded = _epoch_rows()
    conflicting = _epoch_rows(condition_2="old")
    with pytest.raises(StageIdentityError, match="content"):
        validate_epoch_metadata_binding(embedded, conflicting)


def test_legacy_epoch_metadata_without_binding_fails_closed() -> None:
    legacy = pd.DataFrame(
        [{"subject": "sub005", "recording_part": "part1", "trial": 1, "event_sample": 100, "event_code": 120}]
    )
    with pytest.raises(StageIdentityError, match="Legacy"):
        validate_epoch_metadata_binding(legacy, legacy)


def test_matching_timing_inputs_validate_and_preserve_scientific_output(tmp_path: Path) -> None:
    context, source = _context(tmp_path / "inputs")
    behavioral = _behavioral_frame()
    behavioral.to_csv(source, index=False)
    current_behavior = identify_behavioral_inputs(behavioral_log_path=source)
    context = build_dataset_context_identity(
        participant=context.participant,
        session=context.session,
        recording_part=context.recording_part,
        brainvision=context.brainvision,
        behavioral_inputs=current_behavior,
        resolved_config_fingerprint=context.resolved_config_fingerprint,
        scientific_fingerprint=context.scientific_fingerprint,
    )
    alignment, _, artifact = _alignment_artifact(tmp_path, context)
    settings = _timing_settings()
    identity = build_timing_qc_input_identity(
        context=context,
        alignment_artifact=artifact,
        behavioral_inputs=current_behavior,
        behavioral=behavioral,
        alignment=alignment,
        settings=settings,
    )
    output = build_trial_level_qc(
        behavioral,
        alignment,
        None,
        subject="sub005",
        recording_part="part1",
        settings=settings,
        input_identity=identity,
    )
    assert output.loc[0, "qc_status"] == "valid"
    assert output.loc[0, "behavioural_rt_ms"] == 800.0
    assert output.loc[0, "eeg_marker_rt_ms"] == 900.0
    assert output.loc[0, "eeg_minus_behavioural_ms"] == pytest.approx(100.0)


@pytest.mark.parametrize("subject,part", [("sub999", "part1"), ("sub005", "part2")])
def test_timing_qc_rejects_caller_subject_or_part(
    tmp_path: Path, subject: str, part: str
) -> None:
    context, source = _context(tmp_path / "inputs")
    behavioral = _behavioral_frame()
    behavioral.to_csv(source, index=False)
    current_behavior = identify_behavioral_inputs(behavioral_log_path=source)
    context = build_dataset_context_identity(
        participant=context.participant,
        session=context.session,
        recording_part=context.recording_part,
        brainvision=context.brainvision,
        behavioral_inputs=current_behavior,
        resolved_config_fingerprint=context.resolved_config_fingerprint,
        scientific_fingerprint=context.scientific_fingerprint,
    )
    alignment, _, artifact = _alignment_artifact(tmp_path, context)
    settings = _timing_settings()
    identity = build_timing_qc_input_identity(
        context=context,
        alignment_artifact=artifact,
        behavioral_inputs=current_behavior,
        behavioral=behavioral,
        alignment=alignment,
        settings=settings,
    )
    with pytest.raises(StageIdentityError, match="participant/part"):
        build_trial_level_qc(
            behavioral,
            alignment,
            None,
            subject=subject,
            recording_part=part,
            settings=settings,
            input_identity=identity,
        )


def test_timing_qc_rejects_behavioral_session_mismatch(tmp_path: Path) -> None:
    context, source = _context(tmp_path / "inputs")
    behavioral = _behavioral_frame(session=9)
    behavioral.to_csv(source, index=False)
    current = identify_behavioral_inputs(behavioral_log_path=source)
    context = build_dataset_context_identity(
        participant=context.participant,
        session=context.session,
        recording_part=context.recording_part,
        brainvision=context.brainvision,
        behavioral_inputs=current,
        resolved_config_fingerprint=context.resolved_config_fingerprint,
        scientific_fingerprint=context.scientific_fingerprint,
    )
    alignment, _, artifact = _alignment_artifact(tmp_path, context)
    with pytest.raises(StageIdentityError, match="scope"):
        build_timing_qc_input_identity(
            context=context,
            alignment_artifact=artifact,
            behavioral_inputs=current,
            behavioral=behavioral,
            alignment=alignment,
            settings=_timing_settings(),
        )


def test_timing_qc_rejects_behavioral_source_replacement(tmp_path: Path) -> None:
    context, source = _context(tmp_path / "inputs")
    alignment, _, artifact = _alignment_artifact(tmp_path, context)
    source.write_text("subject_id,session,trial\nsub005,2,99\n", encoding="utf-8")
    replacement = identify_behavioral_inputs(behavioral_log_path=source)
    with pytest.raises(StageIdentityError, match="behavioral source"):
        build_timing_qc_input_identity(
            context=context,
            alignment_artifact=artifact,
            behavioral_inputs=replacement,
            behavioral=_behavioral_frame(),
            alignment=alignment,
            settings=_timing_settings(),
        )


def test_timing_qc_rejects_alignment_identity_mismatch(tmp_path: Path) -> None:
    context, _ = _context(tmp_path / "inputs")
    alignment, _, artifact = _alignment_artifact(tmp_path, context)
    changed = alignment.copy()
    changed.loc[0, "trial"] = 99
    assert context.behavioral_inputs is not None
    with pytest.raises(StageIdentityError, match="Alignment table"):
        build_timing_qc_input_identity(
            context=context,
            alignment_artifact=artifact,
            behavioral_inputs=context.behavioral_inputs,
            behavioral=_behavioral_frame(),
            alignment=changed,
            settings=_timing_settings(),
        )


def test_legacy_timing_inputs_without_identity_fail_closed() -> None:
    with pytest.raises(StageIdentityError, match="legacy"):
        build_trial_level_qc(
            _behavioral_frame(),
            _alignment(),
            None,
            subject="sub005",
            recording_part="part1",
            settings=_timing_settings(),
        )
