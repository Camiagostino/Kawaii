from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path

import pytest

from eeg_software.workflows.kawaii_session2 import loading
from eeg_software.workflows.kawaii_session2.identity import (
    BRAINVISION_FILENAME_CORRECTION_TRANSFORMATION,
    BRAINVISION_WORKING_LINEAGE_SCHEMA_VERSION,
    SourceIdentityError,
    brainvision_working_lineage_path,
    build_brainvision_working_lineage,
    identify_brainvision_dataset,
    read_brainvision_working_lineage,
    validate_brainvision_working_lineage,
    write_brainvision_working_lineage,
)
from eeg_software.workflows.kawaii_session2.loading import discover_recording_parts


CHANNELS = ["Cz", "Fz", "LO1", "ECG1"]


def _config() -> dict:
    return {
        "paths": {
            "raw_eeg_dir": "raw_eeg",
            "brainvision_working_dir": "derivatives/brainvision_working_copies",
            "derivatives_dir": "derivatives",
            "output_dir": "out",
            "qc_dir": "out/qc",
            "session2_logs_dir": "logs",
            "session2_schedules_dir": "schedules",
        },
        "recordings": {
            "filename_template": "kawaii{subject_number:04d}",
            "expected_sampling_rate_hz": 500.0,
            "sampling_rate_tolerance_hz": 0.001,
            "participant_recordings": {},
        },
        "channels": {
            "eeg": {
                "expected_count": len(CHANNELS),
                "expected_names": ["Cz", "Fz"],
                "eog": ["LO1"],
                "unused": ["ECG1"],
                "misc": [],
            }
        },
    }


def _write_triplet(
    directory: Path,
    *,
    stem: str = "kawaii0005",
    corrected_references: bool = False,
    eeg_bytes: bytes = b"original-eeg-content",
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    vhdr = directory / f"{stem}.vhdr"
    vmrk = directory / f"{stem}.vmrk"
    eeg = directory / f"{stem}.eeg"
    datafile = eeg.name if corrected_references else "wrong.eeg"
    markerfile = vmrk.name if corrected_references else "wrong.vmrk"
    channel_lines = "\n".join(
        f"Ch{index}={name},,0.1,uV"
        for index, name in enumerate(CHANNELS, start=1)
    )
    vhdr.write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Header File Version 1.0",
                "[Common Infos]",
                f"DataFile={datafile}",
                f"MarkerFile={markerfile}",
                "DataFormat=BINARY",
                f"NumberOfChannels={len(CHANNELS)}",
                "SamplingInterval=2000",
                "[Channel Infos]",
                channel_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    vmrk.write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Marker File, Version 1.0",
                "[Common Infos]",
                f"DataFile={datafile}",
                "[Marker Infos]",
                "Mk1=New Segment,,1,1,0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    eeg.write_bytes(eeg_bytes)
    return vhdr, vmrk, eeg


def _source_paths(root: Path) -> tuple[Path, Path, Path]:
    raw = root / "raw_eeg"
    return tuple(raw / f"kawaii0005.{suffix}" for suffix in ("vhdr", "vmrk", "eeg"))


def _create_working(root: Path):
    _write_triplet(root / "raw_eeg")
    return discover_recording_parts("sub005", _config(), root=root)[0]


def _current_identities(root: Path, part):
    source = identify_brainvision_dataset(
        *_source_paths(root),
        logical_root=root / "raw_eeg",
        allow_filename_reference_correction=True,
    )
    working = identify_brainvision_dataset(
        part.vhdr_path,
        part.vmrk_path,
        part.eeg_path,
        logical_root=part.vhdr_path.parent,
    )
    return source, working


def test_fresh_working_artifact_persists_valid_lineage(tmp_path: Path) -> None:
    part = _create_working(tmp_path)
    lineage_path = part.filename_validation.lineage_path
    assert part.filename_validation.working_copy_action == "created"
    assert lineage_path == brainvision_working_lineage_path(part.vhdr_path.parent)
    assert lineage_path is not None and lineage_path.exists()
    lineage = read_brainvision_working_lineage(lineage_path)
    source, working = _current_identities(tmp_path, part)

    validate_brainvision_working_lineage(
        lineage,
        participant="sub005",
        session="session2",
        recording_part="part1",
        source_dataset=source,
        working_dataset=working,
    )
    assert lineage.source_dataset.linkage_status == "filename_references_require_correction"
    assert lineage.working_dataset.linkage_status == "validated"
    assert source.header.sha256 != working.header.sha256
    assert source.eeg.sha256 == working.eeg.sha256


def test_unchanged_source_and_working_artifact_reuse_validated_lineage(tmp_path: Path) -> None:
    first = _create_working(tmp_path)
    lineage_path = first.filename_validation.lineage_path
    assert lineage_path is not None
    before = lineage_path.read_bytes()

    second = discover_recording_parts("sub005", _config(), root=tmp_path)[0]

    assert second.filename_validation.working_copy_action == "reused_validated"
    assert lineage_path.read_bytes() == before
    assert any("Reused lineage-validated" in warning for warning in second.warnings)


@pytest.mark.parametrize("member", ["vhdr", "vmrk", "eeg"])
def test_same_path_source_replacement_regenerates_stale_working_copy(
    tmp_path: Path,
    member: str,
) -> None:
    first = _create_working(tmp_path)
    before = read_brainvision_working_lineage(first.filename_validation.lineage_path)
    source = dict(zip(("vhdr", "vmrk", "eeg"), _source_paths(tmp_path), strict=True))[member]
    if member == "eeg":
        source.unlink()
        source.write_bytes(b"replacement-eeg-content")
    else:
        source.write_text(source.read_text(encoding="utf-8") + "; source replacement\n", encoding="utf-8")

    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]
    after = read_brainvision_working_lineage(current.filename_validation.lineage_path)

    assert current.filename_validation.working_copy_action == "regenerated"
    assert any("stale or invalid" in warning for warning in current.warnings)
    assert after.source_dataset.fingerprint != before.source_dataset.fingerprint
    if member == "eeg":
        assert current.eeg_path.read_bytes() == b"replacement-eeg-content"


@pytest.mark.parametrize("member", ["vhdr", "vmrk", "eeg"])
def test_working_member_tampering_is_rejected_and_regenerated(
    tmp_path: Path,
    member: str,
) -> None:
    first = _create_working(tmp_path)
    selected = {
        "vhdr": first.vhdr_path,
        "vmrk": first.vmrk_path,
        "eeg": first.eeg_path,
    }[member]
    if member == "eeg":
        selected.unlink()
        selected.write_bytes(b"tampered-working-eeg")
    else:
        selected.write_text(selected.read_text(encoding="utf-8") + "; tampered\n", encoding="utf-8")

    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]
    lineage = read_brainvision_working_lineage(current.filename_validation.lineage_path)
    source, working = _current_identities(tmp_path, current)

    assert current.filename_validation.working_copy_action == "regenerated"
    validate_brainvision_working_lineage(
        lineage,
        participant="sub005",
        session="session2",
        recording_part="part1",
        source_dataset=source,
        working_dataset=working,
    )


def test_malformed_lineage_fails_validation_and_triggers_explicit_regeneration(tmp_path: Path) -> None:
    first = _create_working(tmp_path)
    lineage_path = first.filename_validation.lineage_path
    lineage_path.write_text("{", encoding="utf-8")
    with pytest.raises(SourceIdentityError, match="Could not read"):
        read_brainvision_working_lineage(lineage_path)

    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]

    assert current.filename_validation.working_copy_action == "regenerated"
    assert read_brainvision_working_lineage(lineage_path).participant == "sub005"


@pytest.mark.parametrize("tamper", ["missing_field", "fingerprint"])
def test_incomplete_or_tampered_lineage_fails_closed_and_regenerates(
    tmp_path: Path,
    tamper: str,
) -> None:
    first = _create_working(tmp_path)
    lineage_path = first.filename_validation.lineage_path
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    if tamper == "missing_field":
        payload.pop("transformation")
    else:
        payload["fingerprint"] = "sha256:" + "0" * 64
    lineage_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SourceIdentityError, match="fields are invalid|fingerprint is inconsistent"):
        read_brainvision_working_lineage(lineage_path)
    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]

    assert current.filename_validation.working_copy_action == "regenerated"
    assert read_brainvision_working_lineage(lineage_path).participant == "sub005"


def test_legacy_working_copy_without_lineage_is_regenerated_not_trusted(tmp_path: Path) -> None:
    first = _create_working(tmp_path)
    lineage_path = first.filename_validation.lineage_path
    lineage_path.unlink()

    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]

    assert current.filename_validation.working_copy_action == "regenerated"
    assert lineage_path.exists()
    assert any("legacy working copy" in warning for warning in current.warnings)


@pytest.mark.parametrize(
    ("participant", "recording_part"),
    [("sub999", "part1"), ("sub005", "part2")],
)
def test_lineage_scope_mismatch_is_rejected_and_regenerated(
    tmp_path: Path,
    participant: str,
    recording_part: str,
) -> None:
    first = _create_working(tmp_path)
    source, working = _current_identities(tmp_path, first)
    wrong = build_brainvision_working_lineage(
        participant=participant,
        session="session2",
        recording_part=recording_part,
        source_dataset=source,
        working_dataset=working,
    )
    with pytest.raises(SourceIdentityError, match="scope mismatch"):
        validate_brainvision_working_lineage(
            wrong,
            participant="sub005",
            session="session2",
            recording_part="part1",
            source_dataset=source,
            working_dataset=working,
        )
    write_brainvision_working_lineage(wrong, first.filename_validation.lineage_path)

    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]

    assert current.filename_validation.working_copy_action == "regenerated"


def test_lineage_referencing_other_working_dataset_is_rejected(tmp_path: Path) -> None:
    first = _create_working(tmp_path)
    source, working = _current_identities(tmp_path, first)
    other_paths = _write_triplet(
        tmp_path / "other",
        corrected_references=True,
        eeg_bytes=b"different-working-content",
    )
    other_working = identify_brainvision_dataset(*other_paths, logical_root=tmp_path / "other")
    wrong = build_brainvision_working_lineage(
        participant="sub005",
        session="session2",
        recording_part="part1",
        source_dataset=source,
        working_dataset=other_working,
    )
    with pytest.raises(SourceIdentityError, match="working BrainVision dataset"):
        validate_brainvision_working_lineage(
            wrong,
            participant="sub005",
            session="session2",
            recording_part="part1",
            source_dataset=source,
            working_dataset=working,
        )
    write_brainvision_working_lineage(wrong, first.filename_validation.lineage_path)

    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]

    assert current.filename_validation.working_copy_action == "regenerated"


def test_working_linkage_tampering_is_rejected_and_regenerated(tmp_path: Path) -> None:
    first = _create_working(tmp_path)
    first.vhdr_path.write_text(
        first.vhdr_path.read_text(encoding="utf-8").replace(
            "DataFile=kawaii0005.eeg",
            "DataFile=other.eeg",
        ),
        encoding="utf-8",
    )
    with pytest.raises(SourceIdentityError, match="header DataFile"):
        identify_brainvision_dataset(first.vhdr_path, first.vmrk_path, first.eeg_path)

    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]

    assert current.filename_validation.working_copy_action == "regenerated"
    assert "DataFile=kawaii0005.eeg" in current.vhdr_path.read_text(encoding="utf-8")


def test_repairable_source_identity_is_absolute_root_independent(tmp_path: Path) -> None:
    first_paths = _write_triplet(tmp_path / "machine-a")
    second_paths = _write_triplet(tmp_path / "machine-b")

    first = identify_brainvision_dataset(
        *first_paths,
        logical_root=tmp_path / "machine-a",
        allow_filename_reference_correction=True,
    )
    second = identify_brainvision_dataset(
        *second_paths,
        logical_root=tmp_path / "machine-b",
        allow_filename_reference_correction=True,
    )

    assert first.fingerprint == second.fingerprint
    assert first.linkage_status == "filename_references_require_correction"


def test_lineage_serialization_and_fingerprint_are_deterministic_and_atomic(tmp_path: Path) -> None:
    part = _create_working(tmp_path)
    source, working = _current_identities(tmp_path, part)
    first = build_brainvision_working_lineage(
        participant="sub005",
        session="session2",
        recording_part="part1",
        source_dataset=source,
        working_dataset=working,
    )
    second = build_brainvision_working_lineage(
        participant="sub005",
        session="session2",
        recording_part="part1",
        source_dataset=source,
        working_dataset=working,
    )
    path = tmp_path / "lineage" / "lineage.json"

    write_brainvision_working_lineage(first, path)
    before = path.read_bytes()
    write_brainvision_working_lineage(second, path)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert path.read_bytes() == before
    assert read_brainvision_working_lineage(path) == first
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_lineage_is_immutable_and_contains_no_run_or_config_identity(tmp_path: Path) -> None:
    part = _create_working(tmp_path)
    lineage = read_brainvision_working_lineage(part.filename_validation.lineage_path)
    payload = lineage.to_primitive()

    with pytest.raises(FrozenInstanceError):
        lineage.participant = "sub999"  # type: ignore[misc]
    assert lineage.schema_version == BRAINVISION_WORKING_LINEAGE_SCHEMA_VERSION
    assert lineage.transformation == BRAINVISION_FILENAME_CORRECTION_TRANSFORMATION
    assert "config" not in json.dumps(payload).lower()
    assert "run_request" not in payload


def test_mtime_only_changes_do_not_invalidate_lineage(tmp_path: Path) -> None:
    first = _create_working(tmp_path)
    for path in (*_source_paths(tmp_path), first.vhdr_path, first.vmrk_path, first.eeg_path):
        stat = path.stat()
        os.utime(path, (stat.st_atime + 10, stat.st_mtime + 10))

    current = discover_recording_parts("sub005", _config(), root=tmp_path)[0]

    assert current.filename_validation.working_copy_action == "reused_validated"


def test_failed_working_generation_does_not_publish_success_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_triplet(tmp_path / "raw_eeg")
    working_dir = tmp_path / "derivatives" / "brainvision_working_copies" / "sub005" / "part1"
    lineage_path = brainvision_working_lineage_path(working_dir)

    def fail_copy(*args, **kwargs):
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(loading, "_replace_eeg_link_or_copy", fail_copy)
    with pytest.raises(OSError, match="synthetic copy failure"):
        discover_recording_parts("sub005", _config(), root=tmp_path)

    assert not lineage_path.exists()
