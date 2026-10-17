from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeg_software.workflows.kawaii_session2.temporary_resources import (
    TemporaryResourceOwner,
    TemporaryResourceOwnershipError,
)


def _owner(root: Path, stage: str, run_id: str) -> TemporaryResourceOwner:
    return TemporaryResourceOwner.create(
        root,
        stage=stage,
        participant="sub006",
        recording_part="part1",
        run_id=run_id,
    )


def test_gate3a_success_cleans_owned_crop_and_memmap(tmp_path: Path) -> None:
    owner = _owner(tmp_path, "gate3a", "success")
    crop = owner.allocate("task_crop_input_raw.fif")
    memmap = owner.allocate("filtering_memmap.dat")
    crop.write_bytes(b"crop")
    memmap.write_bytes(b"memmap")
    owner.cleanup([])
    assert not crop.exists() and not memmap.exists() and not owner.directory.exists()


def test_gate3a_exception_finally_cleans_owned_resources(tmp_path: Path) -> None:
    owner = _owner(tmp_path, "gate3a", "failure")
    resource = owner.allocate("filtering_memmap.dat")
    with pytest.raises(RuntimeError):
        try:
            resource.write_bytes(b"temporary")
            raise RuntimeError("injected Gate 3A error")
        finally:
            owner.cleanup([])
    assert not resource.exists()


def test_gate3c_success_cleans_owned_memmap(tmp_path: Path) -> None:
    owner = _owner(tmp_path, "gate3c", "success")
    memmap = owner.allocate("raw_preload.dat")
    memmap.write_bytes(b"memmap")
    owner.cleanup([])
    assert not memmap.exists()


def test_gate3c_exception_finally_cleans_owned_memmap(tmp_path: Path) -> None:
    owner = _owner(tmp_path, "gate3c", "failure")
    memmap = owner.allocate("raw_preload.dat")
    try:
        memmap.write_bytes(b"memmap")
        raise ValueError("injected")
    except ValueError:
        pass
    finally:
        owner.cleanup([])
    assert not memmap.exists()


def test_same_scope_concurrent_runs_use_distinct_locations(tmp_path: Path) -> None:
    first = _owner(tmp_path, "gate3a", "run-one")
    second = _owner(tmp_path, "gate3a", "run-two")
    assert first.directory != second.directory
    assert first.allocate("filtering_memmap.dat") != second.allocate("filtering_memmap.dat")


def test_one_run_cannot_delete_another_runs_resources(tmp_path: Path) -> None:
    first = _owner(tmp_path, "gate3a", "run-one")
    second = _owner(tmp_path, "gate3a", "run-two")
    other = second.allocate("filtering_memmap.dat")
    other.write_bytes(b"other")
    first.cleanup([])
    assert other.read_bytes() == b"other"


def test_foreign_temp_files_remain_untouched(tmp_path: Path) -> None:
    foreign = tmp_path / "historic_gate3a_filtering_memmap.dat"
    foreign.write_bytes(b"audit residue")
    owner = _owner(tmp_path, "gate3a", "run")
    owned = owner.allocate("filtering_memmap.dat")
    owned.write_bytes(b"owned")
    owner.cleanup([])
    assert foreign.read_bytes() == b"audit residue"


def test_restart_does_not_consume_abandoned_foreign_owned_state(tmp_path: Path) -> None:
    abandoned = _owner(tmp_path, "gate3c", "abandoned")
    old = abandoned.allocate("raw_preload.dat")
    old.write_bytes(b"abandoned")
    restarted = _owner(tmp_path, "gate3c", "restart")
    restarted.allocate("raw_preload.dat").write_bytes(b"current")
    restarted.cleanup([])
    assert old.read_bytes() == b"abandoned"


def test_tampered_ownership_fails_closed_without_cleanup(tmp_path: Path) -> None:
    owner = _owner(tmp_path, "gate3a", "tamper")
    resource = owner.allocate("task_crop_input_raw.fif")
    resource.write_bytes(b"scientific-equivalent-input")
    metadata = json.loads(owner.ownership_path.read_text(encoding="utf-8"))
    metadata["run_id"] = "foreign"
    owner.ownership_path.write_text(json.dumps(metadata), encoding="utf-8")
    warnings: list[str] = []
    owner.cleanup(warnings)
    assert resource.read_bytes() == b"scientific-equivalent-input"
    assert "ownership mismatch" in warnings[0].lower()


def test_path_traversal_is_rejected_and_scientific_bytes_unchanged(tmp_path: Path) -> None:
    scientific = tmp_path / "scientific-output.fif"
    scientific.write_bytes(b"unchanged")
    owner = _owner(tmp_path, "gate3a", "safe")
    with pytest.raises(TemporaryResourceOwnershipError):
        owner.allocate("../scientific-output.fif")
    owner.cleanup([])
    assert scientific.read_bytes() == b"unchanged"
