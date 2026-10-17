from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

from eeg_software.workflows.kawaii_session2 import (
    BEHAVIORAL_INPUT_IDENTITY_SCHEMA_VERSION,
    BRAINVISION_DATASET_IDENTITY_SCHEMA_VERSION,
    DATASET_SOURCE_MANIFEST_SCHEMA_VERSION,
    SOURCE_FILE_IDENTITY_SCHEMA_VERSION,
    SourceFileIdentity,
    SourceIdentityError,
    build_source_manifest,
    hash_file_sha256,
    identify_behavioral_inputs,
    identify_brainvision_dataset,
    identify_source_file,
)


def _write_brainvision_triplet(root: Path, stem: str = "recording") -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    vhdr = root / f"{stem}.vhdr"
    vmrk = root / f"{stem}.vmrk"
    eeg = root / f"{stem}.eeg"
    vhdr.write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Header File Version 1.0",
                "[Common Infos]",
                f"DataFile={eeg.name}",
                f"MarkerFile={vmrk.name}",
                "DataFormat=BINARY",
                "NumberOfChannels=1",
                "SamplingInterval=2000",
                "[Channel Infos]",
                "Ch1=Cz,,0.1,uV",
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
                f"DataFile={eeg.name}",
                "[Marker Infos]",
                "Mk1=New Segment,,1,1,0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    eeg.write_bytes(b"tiny synthetic eeg bytes")
    return vhdr, vmrk, eeg


def test_identical_file_bytes_produce_identical_content_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"same bytes")

    first = identify_source_file(source, "test_source", logical_path="inputs/source.bin")
    second = identify_source_file(source, "test_source", logical_path="inputs/source.bin")

    assert first == second
    assert first.sha256 == "sha256:" + hashlib.sha256(b"same bytes").hexdigest()
    assert first.fingerprint == second.fingerprint


def test_same_path_replacement_changes_content_identity_even_at_equal_size(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"aaaa")
    first = identify_source_file(source, "test_source")

    source.unlink()
    source.write_bytes(b"bbbb")
    second = identify_source_file(source, "test_source")

    assert first.size_bytes == second.size_bytes == 4
    assert first.sha256 != second.sha256
    assert first.fingerprint != second.fingerprint


def test_absolute_location_and_logical_path_do_not_change_content_fingerprint(tmp_path: Path) -> None:
    first_path = tmp_path / "machine-a" / "source.bin"
    second_path = tmp_path / "machine-b" / "renamed.bin"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_bytes(b"portable bytes")
    second_path.write_bytes(b"portable bytes")

    first = identify_source_file(first_path, "test_source", logical_path="input/a.bin")
    second = identify_source_file(second_path, "test_source", logical_path="input/b.bin")

    assert first.content_fingerprint == second.content_fingerprint
    assert first.fingerprint == second.fingerprint
    assert first.logical_path != second.logical_path
    assert first.to_primitive() != second.to_primitive()


def test_logical_paths_are_relative_normalized_metadata(tmp_path: Path) -> None:
    root = tmp_path / "project"
    source = root / "inputs" / "source.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"data")

    identity = identify_source_file(source, "test_source", logical_root=root)
    windows_spelling = identify_source_file(
        source,
        "test_source",
        logical_path="inputs\\source.bin",
    )

    assert identity.logical_path == windows_spelling.logical_path == "inputs/source.bin"
    with pytest.raises(SourceIdentityError, match="relative path"):
        identify_source_file(source, "test_source", logical_path="C:\\machine\\source.bin")
    with pytest.raises(SourceIdentityError, match="outside logical_root"):
        identify_source_file(source, "test_source", logical_root=tmp_path / "elsewhere")


def test_hashing_streams_without_path_read_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "large-enough-for-chunks.bin"
    content = b"0123456789" * 10
    source.write_bytes(content)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(AssertionError("read_bytes must not be used")),
    )

    digest = hash_file_sha256(source, chunk_size=7)
    identity = identify_source_file(source, "streamed", chunk_size=9)

    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    assert digest == expected
    assert identity.sha256 == expected
    assert identity.size_bytes == len(content)


@pytest.mark.parametrize("target", ["missing", "directory"])
def test_source_file_identity_fails_closed_for_non_files(tmp_path: Path, target: str) -> None:
    path = tmp_path / target
    if target == "directory":
        path.mkdir()

    with pytest.raises(SourceIdentityError, match="does not exist|requires a file"):
        identify_source_file(path, "test_source")


def test_brainvision_dataset_identity_is_deterministic_and_path_independent(tmp_path: Path) -> None:
    first_paths = _write_brainvision_triplet(tmp_path / "machine-a")
    second_paths = _write_brainvision_triplet(tmp_path / "machine-b")

    first = identify_brainvision_dataset(*first_paths, logical_root=tmp_path / "machine-a")
    repeated = identify_brainvision_dataset(*first_paths, logical_root=tmp_path / "machine-a")
    relocated = identify_brainvision_dataset(*second_paths, logical_root=tmp_path / "machine-b")

    assert first == repeated
    assert first.fingerprint == relocated.fingerprint
    assert first.schema_version == BRAINVISION_DATASET_IDENTITY_SCHEMA_VERSION
    assert [source.role for source in first.sources] == [
        "brainvision_header",
        "brainvision_marker",
        "brainvision_eeg",
    ]


@pytest.mark.parametrize("member", ["vhdr", "vmrk", "eeg"])
def test_changing_any_brainvision_member_changes_dataset_identity(
    tmp_path: Path,
    member: str,
) -> None:
    vhdr, vmrk, eeg = _write_brainvision_triplet(tmp_path)
    baseline = identify_brainvision_dataset(vhdr, vmrk, eeg)
    selected = {"vhdr": vhdr, "vmrk": vmrk, "eeg": eeg}[member]
    if member == "eeg":
        selected.write_bytes(selected.read_bytes() + b" changed")
    else:
        selected.write_text(selected.read_text(encoding="utf-8") + "; changed\n", encoding="utf-8")

    changed = identify_brainvision_dataset(vhdr, vmrk, eeg)

    assert changed.fingerprint != baseline.fingerprint


@pytest.mark.parametrize(
    ("reference", "replacement", "message"),
    [
        ("DataFile=recording.eeg", "DataFile=other.eeg", "header DataFile"),
        ("MarkerFile=recording.vmrk", "MarkerFile=other.vmrk", "header MarkerFile"),
        ("marker:DataFile=recording.eeg", "DataFile=other.eeg", "marker DataFile"),
    ],
)
def test_brainvision_linkage_mismatch_fails_closed(
    tmp_path: Path,
    reference: str,
    replacement: str,
    message: str,
) -> None:
    vhdr, vmrk, eeg = _write_brainvision_triplet(tmp_path)
    target = vmrk if reference.startswith("marker:") else vhdr
    needle = reference.removeprefix("marker:")
    target.write_text(
        target.read_text(encoding="utf-8").replace(needle, replacement),
        encoding="utf-8",
    )

    with pytest.raises(SourceIdentityError, match=message):
        identify_brainvision_dataset(vhdr, vmrk, eeg)


def test_behavioral_log_and_schedule_identities_are_deterministic(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "trials.csv"
    schedule = tmp_path / "schedules" / "schedule.csv"
    log.parent.mkdir()
    schedule.parent.mkdir()
    log.write_text("trial,response\n1,yes\n", encoding="utf-8")
    schedule.write_text("trial,image\n1,a.png\n", encoding="utf-8")

    first = identify_behavioral_inputs(
        behavioral_log_path=log,
        schedule_path=schedule,
        logical_root=tmp_path,
    )
    second = identify_behavioral_inputs(
        behavioral_log_path=log,
        schedule_path=schedule,
        logical_root=tmp_path,
    )

    assert first == second
    assert first.behavioral_log is not None
    assert first.schedule is not None
    assert first.schema_version == BEHAVIORAL_INPUT_IDENTITY_SCHEMA_VERSION


@pytest.mark.parametrize("member", ["behavioral_log", "schedule"])
def test_behavioral_composite_changes_when_any_input_content_changes(
    tmp_path: Path,
    member: str,
) -> None:
    log = tmp_path / "trials.csv"
    schedule = tmp_path / "schedule.csv"
    log.write_text("trial,response\n1,yes\n", encoding="utf-8")
    schedule.write_text("trial,image\n1,a.png\n", encoding="utf-8")
    baseline = identify_behavioral_inputs(
        behavioral_log_path=log,
        schedule_path=schedule,
    )
    selected = log if member == "behavioral_log" else schedule
    selected.write_text(selected.read_text(encoding="utf-8") + "2,changed\n", encoding="utf-8")

    changed = identify_behavioral_inputs(
        behavioral_log_path=log,
        schedule_path=schedule,
    )

    assert changed.fingerprint != baseline.fingerprint


def test_behavioral_input_identity_rejects_empty_or_conflicting_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("trial\n1\n", encoding="utf-8")

    with pytest.raises(SourceIdentityError, match="At least one"):
        identify_behavioral_inputs()
    with pytest.raises(SourceIdentityError, match="distinct source files"):
        identify_behavioral_inputs(
            behavioral_log_path=source,
            schedule_path=source,
        )


def test_source_manifest_is_deterministic_sorted_and_scope_bound(tmp_path: Path) -> None:
    vhdr, vmrk, eeg = _write_brainvision_triplet(tmp_path / "brainvision")
    log = tmp_path / "trials.csv"
    log.write_text("trial,response\n1,yes\n", encoding="utf-8")
    brainvision = identify_brainvision_dataset(vhdr, vmrk, eeg, logical_root=tmp_path)
    behavioral = identify_behavioral_inputs(
        behavioral_log_path=log,
        logical_root=tmp_path,
    )

    first = build_source_manifest(
        participant="sub005",
        session=2,
        recording_part="part1",
        brainvision=brainvision,
        behavioral_inputs=behavioral,
    )
    reordered = build_source_manifest(
        participant="sub005",
        session="2",
        recording_part="part1",
        sources=tuple(reversed((*brainvision.sources, *behavioral.sources))),
    )
    other_scope = build_source_manifest(
        participant="sub005",
        session=2,
        recording_part="part2",
        brainvision=brainvision,
        behavioral_inputs=behavioral,
    )

    assert first == reordered
    assert first.fingerprint != other_scope.fingerprint
    assert first.schema_version == DATASET_SOURCE_MANIFEST_SCHEMA_VERSION
    assert [source.role for source in first.sources] == sorted(
        source.role for source in first.sources
    )


def test_source_manifest_rejects_duplicate_roles_and_malformed_scope(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"content")
    identity = identify_source_file(source, "duplicate")

    with pytest.raises(SourceIdentityError, match="Duplicate source role"):
        build_source_manifest(
            participant="sub005",
            session=2,
            recording_part="part1",
            sources=(identity, identity),
        )
    with pytest.raises(SourceIdentityError, match="participant"):
        build_source_manifest(
            participant="",
            session=2,
            recording_part="part1",
            sources=(identity,),
        )


def test_identity_serialization_is_canonical_and_schema_versioned(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"content")
    identity = identify_source_file(source, "test_source", logical_path="inputs/source.bin")

    first = identity.canonical_json()
    second = identity.canonical_json()
    payload = json.loads(first)

    assert first == second
    assert first == json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert payload["schema_version"] == SOURCE_FILE_IDENTITY_SCHEMA_VERSION
    assert payload["fingerprint"] == identity.fingerprint


def test_identity_objects_are_immutable_and_reject_malformed_digests(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"content")
    identity = identify_source_file(source, "test_source")

    with pytest.raises(FrozenInstanceError):
        identity.role = "changed"  # type: ignore[misc]
    with pytest.raises(SourceIdentityError, match="SHA-256"):
        SourceFileIdentity(
            role="test_source",
            logical_path="source.bin",
            size_bytes=7,
            sha256="not-a-digest",
        )
