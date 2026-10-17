from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest
import yaml

from eeg_software.workflows.kawaii_session2.freeze import write_manual_review_snapshot
from eeg_software.workflows.kawaii_session2.manual_review import (
    ensure_manual_review_scaffold,
    manual_review_paths,
    read_bad_channel_review_decision,
    read_ica_review_decision,
    write_bad_channel_review_decision,
    write_ica_review_decision,
    write_ica_review_template,
)
from eeg_software.workflows.kawaii_session2.manual_review_persistence import (
    ManualReviewConflictError,
    ManualReviewLockTimeoutError,
    ManualReviewPersistenceError,
    store_lock_path,
)
import eeg_software.workflows.kawaii_session2.manual_review_persistence as persistence


SOURCE_ID = "sha256:" + "1" * 64
CONTEXT_ID = "sha256:" + "2" * 64
BAD_AUTHORITY_ID = "sha256:" + "3" * 64
SOLUTION_A = "sha256:" + "a" * 64
SOLUTION_B = "sha256:" + "b" * 64
REVIEWED_AT = "2026-09-20T10:00:00Z"


def _config() -> dict:
    return {
        "paths": {
            "output_dir": "workspace/kawaii/session2/analysis",
            "manual_review_dir": "workspace/kawaii/session2/manual_review",
        }
    }


def _write_bad(root: Path, subject: str, channels: list[str], **overrides):
    values = {
        "review_status": "complete",
        "confirmed_bad_channels": channels,
        "source_recording_id": SOURCE_ID,
        "review_context_id": CONTEXT_ID,
        "reviewed_channels": ["A", "B", "C"],
        "reviewer": "reviewer",
        "reviewed_at_utc": REVIEWED_AT,
    }
    values.update(overrides)
    return write_bad_channel_review_decision(
        _config(), subject, "part1", root=root, **values
    )


def _write_ica(root: Path, subject: str, components: list[int], **overrides):
    values = {
        "review_status": "complete",
        "confirmed_ica_components": components,
        "ica_solution_id": SOLUTION_A,
        "fitted_component_count": 4,
        "bad_channel_decision_id": BAD_AUTHORITY_ID,
        "reviewer": "reviewer",
        "reviewed_at_utc": REVIEWED_AT,
    }
    values.update(overrides)
    return write_ica_review_decision(
        _config(), subject, "part1", root=root, **values
    )


def test_bad_channel_compare_and_swap_and_idempotence(tmp_path: Path) -> None:
    reviewed_zero = _write_bad(tmp_path, "subA", [])
    same_semantics = _write_bad(
        tmp_path,
        "subA",
        [],
        reviewer="second-reviewer",
        reviewed_at_utc="2026-09-20T11:00:00Z",
    )
    assert same_semantics.semantic_decision_id == reviewed_zero.semantic_decision_id
    assert same_semantics.audit_record_id != reviewed_zero.audit_record_id
    with pytest.raises(ManualReviewConflictError, match="without its expected current"):
        _write_bad(tmp_path, "subA", ["A"])
    replacement = _write_bad(
        tmp_path,
        "subA",
        ["A"],
        expected_current_semantic_decision_id=reviewed_zero.semantic_decision_id,
    )
    with pytest.raises(ManualReviewConflictError, match="Stale manual-review update"):
        _write_bad(
            tmp_path,
            "subA",
            ["B"],
            expected_current_semantic_decision_id=reviewed_zero.semantic_decision_id,
        )
    assert read_bad_channel_review_decision(
        _config(), "subA", "part1", root=tmp_path
    ).semantic_decision_id == replacement.semantic_decision_id


def test_ica_compare_and_swap_binds_solution_and_reviewed_zero(tmp_path: Path) -> None:
    reviewed_zero = _write_ica(tmp_path, "subA", [])
    repeated = _write_ica(tmp_path, "subA", [], notes="audit-only revision")
    assert repeated.semantic_decision_id == reviewed_zero.semantic_decision_id
    assert repeated.audit_record_id != reviewed_zero.audit_record_id
    with pytest.raises(ManualReviewConflictError):
        _write_ica(tmp_path, "subA", [1], ica_solution_id=SOLUTION_B)
    changed_solution = _write_ica(
        tmp_path,
        "subA",
        [1],
        ica_solution_id=SOLUTION_B,
        expected_current_semantic_decision_id=reviewed_zero.semantic_decision_id,
    )
    assert changed_solution.ica_solution_id == SOLUTION_B
    assert read_ica_review_decision(
        _config(), "subA", "part1", root=tmp_path, require_complete=True
    ).semantic_decision_id == changed_solution.semantic_decision_id


def test_concurrent_disjoint_bad_channel_updates_both_survive(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)

    def write(subject: str, channel: str):
        barrier.wait()
        return _write_bad(tmp_path, subject, [channel])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item: write(*item), [("subA", "A"), ("subB", "B")]))
    assert len(results) == 2
    assert read_bad_channel_review_decision(
        _config(), "subA", "part1", root=tmp_path
    ).confirmed_bad_channels == ("A",)
    assert read_bad_channel_review_decision(
        _config(), "subB", "part1", root=tmp_path
    ).confirmed_bad_channels == ("B",)


def test_concurrent_same_scope_bad_channel_update_has_one_explicit_conflict(tmp_path: Path) -> None:
    initial = _write_bad(tmp_path, "subA", [])
    barrier = threading.Barrier(2)

    def replace(channel: str):
        barrier.wait()
        try:
            return _write_bad(
                tmp_path,
                "subA",
                [channel],
                expected_current_semantic_decision_id=initial.semantic_decision_id,
            )
        except ManualReviewConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(replace, ["A", "B"]))
    assert sum(isinstance(value, ManualReviewConflictError) for value in outcomes) == 1
    final = read_bad_channel_review_decision(_config(), "subA", "part1", root=tmp_path)
    assert final.confirmed_bad_channels in {("A",), ("B",)}
    yaml.safe_load(manual_review_paths(_config(), root=tmp_path)["bad_channels"].read_text())


def test_concurrent_disjoint_ica_updates_both_survive(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)

    def write(subject: str, component: int):
        barrier.wait()
        return _write_ica(tmp_path, subject, [component])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item: write(*item), [("subA", 0), ("subB", 1)]))
    assert len(results) == 2
    assert read_ica_review_decision(
        _config(), "subA", "part1", root=tmp_path
    ).confirmed_ica_components == (0,)
    assert read_ica_review_decision(
        _config(), "subB", "part1", root=tmp_path
    ).confirmed_ica_components == (1,)


def test_concurrent_same_scope_ica_update_has_one_explicit_conflict(tmp_path: Path) -> None:
    initial = _write_ica(tmp_path, "subA", [])
    barrier = threading.Barrier(2)

    def replace(component: int):
        barrier.wait()
        try:
            return _write_ica(
                tmp_path,
                "subA",
                [component],
                expected_current_semantic_decision_id=initial.semantic_decision_id,
            )
        except ManualReviewConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(replace, [1, 2]))
    assert sum(isinstance(value, ManualReviewConflictError) for value in outcomes) == 1
    assert read_ica_review_decision(
        _config(), "subA", "part1", root=tmp_path
    ).confirmed_ica_components in {(1,), (2,)}


def test_lock_contention_fails_bounded_without_mutation(tmp_path: Path) -> None:
    prior = _write_bad(tmp_path, "subA", [])
    path = manual_review_paths(_config(), root=tmp_path)["bad_channels"]
    before = path.read_bytes()
    lock = store_lock_path(path)
    lock.write_text(json.dumps({"token": "foreign", "owner": "test"}), encoding="utf-8")
    with pytest.raises(ManualReviewLockTimeoutError, match="was not broken automatically"):
        _write_bad(
            tmp_path,
            "subA",
            ["A"],
            expected_current_semantic_decision_id=prior.semantic_decision_id,
            lock_timeout_s=0.01,
        )
    assert path.read_bytes() == before
    assert lock.exists()


def test_failures_before_promotion_preserve_prior_canonical_store(tmp_path: Path) -> None:
    prior = _write_bad(tmp_path, "subA", [])
    path = manual_review_paths(_config(), root=tmp_path)["bad_channels"]
    before = path.read_bytes()

    def fail_partial_temporary_write(handle, content: str) -> None:
        handle.write(content[: max(1, len(content) // 2)])
        raise OSError("injected incomplete temporary write")

    with patch.object(
        persistence, "_write_text_content", side_effect=fail_partial_temporary_write
    ):
        with pytest.raises(ManualReviewPersistenceError, match="temporary manual-review file"):
            _write_bad(
                tmp_path,
                "subA",
                ["A"],
                expected_current_semantic_decision_id=prior.semantic_decision_id,
            )
    assert path.read_bytes() == before
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []

    observed_temp: list[Path] = []

    def fail_before_replace(temporary: Path, _path: Path) -> None:
        assert temporary.exists()
        observed_temp.append(temporary)
        raise ManualReviewPersistenceError("injected before replace")

    with patch.object(persistence, "_replace_canonical", side_effect=fail_before_replace):
        with pytest.raises(ManualReviewPersistenceError, match="before replace"):
            _write_bad(
                tmp_path,
                "subA",
                ["A"],
                expected_current_semantic_decision_id=prior.semantic_decision_id,
            )
    assert path.read_bytes() == before
    assert observed_temp and not observed_temp[0].exists()

    with patch.object(persistence.os, "replace", side_effect=OSError("injected replace failure")):
        with pytest.raises(ManualReviewPersistenceError, match="atomically persist"):
            _write_bad(
                tmp_path,
                "subA",
                ["A"],
                expected_current_semantic_decision_id=prior.semantic_decision_id,
            )
    assert path.read_bytes() == before


def test_failed_ica_promotion_preserves_prior_authoritative_decision(tmp_path: Path) -> None:
    prior = _write_ica(tmp_path, "subA", [])
    path = manual_review_paths(_config(), root=tmp_path)["ica_components"]
    before = path.read_bytes()
    with patch.object(persistence, "_replace_canonical", side_effect=OSError("replace failed")):
        with pytest.raises(ManualReviewPersistenceError):
            _write_ica(
                tmp_path,
                "subA",
                [1],
                expected_current_semantic_decision_id=prior.semantic_decision_id,
            )
    assert path.read_bytes() == before
    assert read_ica_review_decision(
        _config(), "subA", "part1", root=tmp_path
    ).confirmed_ica_components == ()


def test_post_replace_validation_failure_leaves_complete_new_store(tmp_path: Path) -> None:
    prior = _write_ica(tmp_path, "subA", [])
    path = manual_review_paths(_config(), root=tmp_path)["ica_components"]
    original_reader = persistence._read_yaml_mapping
    calls = 0

    def fail_round_trip(target: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            return {}
        return original_reader(target)

    with patch.object(persistence, "_read_yaml_mapping", side_effect=fail_round_trip):
        with pytest.raises(ManualReviewPersistenceError, match="does not exactly match"):
            _write_ica(
                tmp_path,
                "subA",
                [1],
                expected_current_semantic_decision_id=prior.semantic_decision_id,
            )
    committed = read_ica_review_decision(_config(), "subA", "part1", root=tmp_path)
    assert committed.confirmed_ica_components == (1,)
    yaml.safe_load(path.read_text(encoding="utf-8"))


def test_malformed_current_store_fails_closed(tmp_path: Path) -> None:
    path = manual_review_paths(_config(), root=tmp_path)["bad_channels"]
    path.parent.mkdir(parents=True)
    path.write_text("[not-a-mapping]\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ManualReviewPersistenceError, match="must contain a mapping"):
        _write_bad(tmp_path, "subA", ["A"])
    assert path.read_bytes() == before


def test_scaffold_is_concurrent_idempotent_and_preserves_notes(tmp_path: Path) -> None:
    paths = manual_review_paths(_config(), root=tmp_path)
    paths["manual_review_dir"].mkdir(parents=True)
    paths["participant_notes"].write_text(
        "# Session 2 EEG Participant Notes\n\nHuman text must stay.\n", encoding="utf-8"
    )
    barrier = threading.Barrier(2)

    def create(subject: str):
        barrier.wait()
        return ensure_manual_review_scaffold(
            {subject: ["part1"]}, _config(), root=tmp_path
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(create, ["subA", "subB"]))
    bad = yaml.safe_load(paths["bad_channels"].read_text(encoding="utf-8"))
    ica = yaml.safe_load(paths["ica_components"].read_text(encoding="utf-8"))
    assert set(bad) == {"subA", "subB"}
    assert set(ica) == {"subA", "subB"}
    notes = paths["participant_notes"].read_text(encoding="utf-8")
    assert "Human text must stay." in notes
    assert "### subA part1" in notes and "### subB part1" in notes


def test_template_and_snapshot_failed_replace_preserve_prior_complete_file(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text("prior: template\n", encoding="utf-8")
    with patch.object(persistence, "_replace_canonical", side_effect=OSError("replace failed")):
        with pytest.raises(ManualReviewPersistenceError):
            write_ica_review_template(
                template,
                subject="subA",
                recording_part="part1",
                ica_solution_id=SOLUTION_A,
                fitted_component_count=4,
            )
    assert template.read_text(encoding="utf-8") == "prior: template\n"

    snapshot = (
        tmp_path
        / "workspace/kawaii/session2/analysis/subjects/subA/part1/provenance/manual_review_snapshot.yaml"
    )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("prior: snapshot\n", encoding="utf-8")
    with patch.object(persistence, "_replace_canonical", side_effect=OSError("replace failed")):
        with pytest.raises(ManualReviewPersistenceError):
            write_manual_review_snapshot(_config(), "subA", "part1", root=tmp_path)
    assert snapshot.read_text(encoding="utf-8") == "prior: snapshot\n"
