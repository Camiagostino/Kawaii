from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.cuteness_mixed import center_cuteness_trials
from eeg_software.workflows.kawaii_session2.identity import identify_source_file
from eeg_software.workflows.kawaii_session2.resolved_config import canonical_sha256
from eeg_software.workflows.kawaii_session2.source_exposure import (
    AMBIGUOUS_RUN4_EXPOSURE,
    DISPOSITION_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    SOURCE_MISSING,
    SourceExposureValidationError,
    VERIFIED_NEW,
    VERIFIED_OLD,
    apply_source_exposure_disposition,
)


def _row(trial: int, item: str, status: str, original: str, *, eligible: bool | None = None) -> dict:
    verified = status in {VERIFIED_OLD, VERIFIED_NEW}
    if eligible is None:
        eligible = verified
    return {
        "schema_version": DISPOSITION_SCHEMA_VERSION,
        "subject": "sub999",
        "session": "session2",
        "trial": trial,
        "image_filename": item,
        "original_old_new_status": original,
        "source_exposure_status": status,
        "source_verified_old_new_status": "old" if status == VERIFIED_OLD else "new" if status == VERIFIED_NEW else pd.NA,
        "recognition_eligible": eligible,
        "continuous_cuteness_old_new_eligible": eligible,
        "exposure_exclusion_reason": "ambiguous_run4_exposure" if status == AMBIGUOUS_RUN4_EXPOSURE else "",
    }


def _fixture(
    tmp_path: Path,
    rows: list[dict],
    *,
    protocol: dict | None = None,
    schema_version: str = MANIFEST_SCHEMA_VERSION,
) -> tuple[dict, pd.DataFrame, Path]:
    evidence_path = tmp_path / "source.csv"
    evidence_path.write_text("source\nverified\n", encoding="utf-8")
    source = identify_source_file(evidence_path, "session1_source", logical_path="source.csv").to_primitive()
    protocol_value = protocol or {
        "recognition_delay_days": 1,
        "protocol_deviation": False,
        "protocol_disposition": "STANDARD",
    }
    participant_identity_payload = {
        "subject": "sub999",
        "session": "session2",
        "recording_parts": ["part1"],
        "sources": [{key: source[key] for key in ("schema_version", "role", "size_bytes", "sha256")}],
        "protocol": protocol_value,
        "notes": [],
    }
    source_identity = canonical_sha256(participant_identity_payload)
    counts = {
        VERIFIED_OLD: sum(row["source_exposure_status"] == VERIFIED_OLD for row in rows),
        VERIFIED_NEW: sum(row["source_exposure_status"] == VERIFIED_NEW for row in rows),
        AMBIGUOUS_RUN4_EXPOSURE: sum(row["source_exposure_status"] == AMBIGUOUS_RUN4_EXPOSURE for row in rows),
    }
    participant = {
        **participant_identity_payload,
        "source_identity": source_identity,
        "sources": [source],
        "expected_counts": counts,
    }
    artifact_payload = {
        "schema_version": schema_version,
        "disposition_schema_version": DISPOSITION_SCHEMA_VERSION,
        "participants": [
            {
                "subject": "sub999",
                "session": "session2",
                "recording_parts": ["part1"],
                "source_identity": source_identity,
                "sources": [{key: source[key] for key in ("schema_version", "role", "size_bytes", "sha256")}],
                "expected_counts": counts,
                "protocol": protocol_value,
                "notes": [],
            }
        ],
    }
    artifact_id = canonical_sha256(artifact_payload)
    table = pd.DataFrame(rows)
    table["source_exposure_identity"] = source_identity
    table["source_exposure_artifact_id"] = artifact_id
    table_path = tmp_path / "disposition.csv"
    table.to_csv(table_path, index=False)
    artifact = identify_source_file(table_path, "source_exposure_disposition_table", logical_path=table_path.name).to_primitive()
    manifest = {
        **artifact_payload,
        "artifact_id": artifact_id,
        "artifact": artifact,
        "participants": [participant],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "source_exposure_verification": {
            "enabled": True,
            "required_subjects": ["sub999"],
            "manifest_path": str(manifest_path),
            "expected_artifact_id": artifact_id,
        }
    }
    metadata = pd.DataFrame(
        [
            {
                "subject_id": "sub999",
                "session": 2,
                "recording_part": "part1",
                "trial": row["trial"],
                "image_filename": row["image_filename"],
                "old_new_status": row["original_old_new_status"],
                "recognition_response": "yes",
                "kawaii_rating": 7,
            }
            for row in rows
        ]
    )
    return config, metadata, table_path


def _apply(config: dict, metadata: pd.DataFrame, tmp_path: Path):
    return apply_source_exposure_disposition(
        metadata,
        subject="sub999",
        session="session2",
        recording_part="part1",
        config=config,
        root=tmp_path,
    )


def test_matching_verified_old_and_new_are_eligible_and_original_is_preserved(tmp_path: Path) -> None:
    rows = [_row(1, "old.jpg", VERIFIED_OLD, "old"), _row(2, "new.jpg", VERIFIED_NEW, "new")]
    config, metadata, _ = _fixture(tmp_path, rows)
    result = _apply(config, metadata, tmp_path)
    assert result.metadata["original_old_new_status"].tolist() == ["old", "new"]
    assert result.metadata["old_new_status"].tolist() == ["old", "new"]
    assert result.metadata["recognition_eligible"].tolist() == [True, True]
    assert result.metadata["continuous_cuteness_old_new_eligible"].tolist() == [True, True]


def test_source_session2_mismatch_fails_closed(tmp_path: Path) -> None:
    config, metadata, _ = _fixture(tmp_path, [_row(1, "a.jpg", VERIFIED_OLD, "old")])
    metadata.loc[0, "old_new_status"] = "new"
    with pytest.raises(SourceExposureValidationError, match="conflicts|mismatch") as error:
        _apply(config, metadata, tmp_path)
    assert error.value.reason_code == "source_session2_old_new_mismatch"


def test_duplicate_source_mapping_fails_closed(tmp_path: Path) -> None:
    rows = [_row(1, "a.jpg", VERIFIED_OLD, "old"), _row(1, "a.jpg", VERIFIED_OLD, "old")]
    config, metadata, _ = _fixture(tmp_path, rows)
    metadata = metadata.iloc[:1]
    with pytest.raises(SourceExposureValidationError) as error:
        _apply(config, metadata, tmp_path)
    assert error.value.reason_code == "duplicate_source_exposure_mapping"


def test_missing_required_verification_fails_closed(tmp_path: Path) -> None:
    config, metadata, _ = _fixture(tmp_path, [_row(1, "a.jpg", VERIFIED_OLD, "old")])
    metadata.loc[0, ["trial", "image_filename"]] = [2, "missing.jpg"]
    with pytest.raises(SourceExposureValidationError) as error:
        _apply(config, metadata, tmp_path)
    assert error.value.reason_code == "missing_source_exposure_disposition"


@pytest.mark.parametrize("column,value", [("subject_id", "sub998"), ("session", 1), ("recording_part", "part2")])
def test_wrong_participant_session_or_part_identity_fails_closed(tmp_path: Path, column: str, value: object) -> None:
    config, metadata, _ = _fixture(tmp_path, [_row(1, "a.jpg", VERIFIED_OLD, "old")])
    metadata.loc[0, column] = value
    with pytest.raises(SourceExposureValidationError) as error:
        _apply(config, metadata, tmp_path)
    assert error.value.reason_code == "source_exposure_scope_mismatch"


def test_authorized_ambiguity_is_analysis_specific_and_not_converted(tmp_path: Path) -> None:
    rows = [_row(1, "old.jpg", VERIFIED_OLD, "old"), _row(2, "ambiguous.jpg", AMBIGUOUS_RUN4_EXPOSURE, "new")]
    config, metadata, _ = _fixture(tmp_path, rows)
    result = _apply(config, metadata, tmp_path)
    ambiguous = result.metadata.iloc[1]
    assert ambiguous["recognition_eligible"] is False or ambiguous["recognition_eligible"] == False
    assert ambiguous["continuous_cuteness_old_new_eligible"] is False or ambiguous["continuous_cuteness_old_new_eligible"] == False
    assert pd.isna(ambiguous["source_verified_old_new_status"])
    assert len(result.metadata) == 2  # categorical cuteness/timing inputs are not globally filtered
    assert result.metadata["kawaii_rating"].notna().all()


def test_group_recentering_cannot_restore_source_ineligible_continuous_trial() -> None:
    trials = pd.DataFrame(
        [
            {
                "participant": "sub999", "item_id": "old.jpg", "epoch_retained": True,
                "roi_complete": True, "cuteness_raw": 7.0, "arousal_raw": 5.0,
                "ERP_mean_uv": 1.0, "mean_amplitude_uv": 1.0, "old_new_old": 1.0,
                "continuous_cuteness_old_new_eligible": True, "exposure_exclusion_reason": "",
            },
            {
                "participant": "sub999", "item_id": "ambiguous.jpg", "epoch_retained": True,
                "roi_complete": True, "cuteness_raw": 6.0, "arousal_raw": 4.0,
                "ERP_mean_uv": 2.0, "mean_amplitude_uv": 2.0, "old_new_old": 0.0,
                "continuous_cuteness_old_new_eligible": False,
                "exposure_exclusion_reason": "ambiguous_run4_exposure",
            },
        ]
    )
    prepared = center_cuteness_trials(trials)
    assert prepared["model_eligible_reduced"].tolist() == [True, False]
    assert "ambiguous_run4_exposure" in prepared.loc[1, "exclusion_reason"]


def test_retained_count_provenance_and_protocol_flag_are_preserved(tmp_path: Path) -> None:
    rows = [_row(1, "old.jpg", VERIFIED_OLD, "old"), _row(2, "amb.jpg", AMBIGUOUS_RUN4_EXPOSURE, "new")]
    protocol = {"recognition_delay_days": 2, "protocol_deviation": True, "protocol_disposition": "INCLUDED_FLAGGED"}
    config, metadata, _ = _fixture(tmp_path, rows, protocol=protocol)
    result = _apply(config, metadata, tmp_path)
    assert result.summary["artifact_recognition_eligible"] == 1
    assert result.summary["artifact_continuous_cuteness_eligible"] == 1
    assert result.summary["ambiguous_ineligible"] == 1
    assert result.summary["recognition_delay_days"] == 2
    assert result.summary["protocol_deviation"] is True
    assert result.summary["protocol_disposition"] == "INCLUDED_FLAGGED"
    assert result.metadata["recognition_eligible"].sum() == 1


def test_repeated_application_is_idempotent(tmp_path: Path) -> None:
    config, metadata, _ = _fixture(tmp_path, [_row(1, "a.jpg", VERIFIED_OLD, "old")])
    once = _apply(config, metadata, tmp_path)
    twice = _apply(config, once.metadata, tmp_path)
    pd.testing.assert_frame_equal(once.metadata, twice.metadata)
    assert once.summary == twice.summary


@pytest.mark.parametrize(
    "row_change,reason_code",
    [
        ({"source_exposure_status": "VERIFIED_MAYBE"}, "malformed_source_exposure_disposition"),
        ({"source_exposure_status": AMBIGUOUS_RUN4_EXPOSURE, "source_verified_old_new_status": "new", "recognition_eligible": False, "continuous_cuteness_old_new_eligible": False, "exposure_exclusion_reason": "ambiguous"}, "ambiguous_source_exposure_converted"),
        ({"source_exposure_status": AMBIGUOUS_RUN4_EXPOSURE, "source_verified_old_new_status": pd.NA, "recognition_eligible": True, "continuous_cuteness_old_new_eligible": True, "exposure_exclusion_reason": "ambiguous"}, "ambiguous_source_exposure_eligible"),
        ({"source_exposure_status": SOURCE_MISSING, "source_verified_old_new_status": pd.NA, "recognition_eligible": False, "continuous_cuteness_old_new_eligible": False, "exposure_exclusion_reason": "missing"}, "unresolved_source_exposure_invalid"),
    ],
)
def test_malformed_or_unresolved_dispositions_fail_closed(tmp_path: Path, row_change: dict, reason_code: str) -> None:
    row = _row(1, "a.jpg", VERIFIED_OLD, "old")
    row.update(row_change)
    config, metadata, _ = _fixture(tmp_path, [row])
    with pytest.raises(SourceExposureValidationError) as error:
        _apply(config, metadata, tmp_path)
    assert error.value.reason_code == reason_code


def test_stale_artifact_identity_and_schema_fail_closed(tmp_path: Path) -> None:
    config, metadata, table_path = _fixture(tmp_path, [_row(1, "a.jpg", VERIFIED_OLD, "old")])
    table_path.write_text(table_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SourceExposureValidationError) as error:
        _apply(config, metadata, tmp_path)
    assert error.value.reason_code == "stale_source_exposure_identity"

    other = tmp_path / "schema"
    other.mkdir()
    bad_config, bad_metadata, _ = _fixture(other, [_row(1, "a.jpg", VERIFIED_OLD, "old")], schema_version="future")
    with pytest.raises(SourceExposureValidationError) as schema_error:
        _apply(bad_config, bad_metadata, other)
    assert schema_error.value.reason_code == "malformed_source_exposure_disposition"
