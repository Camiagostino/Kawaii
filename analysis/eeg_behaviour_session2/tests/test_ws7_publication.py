from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from eeg_software.workflows.kawaii_session2.publication import (
    PublicationCancelled,
    PublicationConflictError,
    PublicationScope,
    PublicationValidationError,
    current_manifest_path,
    publication_scope_root,
    publish_artifact_set,
    validate_current_publication,
    validate_publication_scientific_context,
)


def _file(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _scope(part: str = "part1") -> PublicationScope:
    return PublicationScope("gate3a", participant="sub006", recording_part=part)


def _publish(tmp_path: Path, scope: PublicationScope | None = None, **kwargs):
    source = _file(tmp_path / f"source-{kwargs.get('run_id', 'run')}.bin", b"artifact")
    return publish_artifact_set(
        tmp_path / "canonical",
        scope or _scope(),
        run_id=kwargs.pop("run_id", "run-1"),
        context_reference=kwargs.pop("context_reference", "config-identity"),
        required_artifacts=kwargs.pop("required_artifacts", {"data": source}),
        **kwargs,
    )


def test_successful_coherent_candidate_becomes_current(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    consumed = validate_current_publication(tmp_path / "canonical", _scope())
    assert consumed.generation_id == published.generation_id
    assert consumed.artifact_path("data").read_bytes() == b"artifact"


def test_required_artifact_missing_does_not_publish(tmp_path: Path) -> None:
    with pytest.raises(PublicationValidationError, match="missing"):
        _publish(tmp_path, required_artifacts={"data": tmp_path / "absent"})
    assert not current_manifest_path(tmp_path / "canonical", _scope()).exists()


def test_staged_artifact_corruption_does_not_publish(tmp_path: Path) -> None:
    def corrupt(boundary: str) -> None:
        if boundary == "artifact_staged:data":
            staged = next(publication_scope_root(tmp_path / "canonical", _scope()).glob("staging/*/*--data--*"))
            staged.write_bytes(b"corrupt")

    with pytest.raises(PublicationValidationError, match="identity changed"):
        _publish(tmp_path, fault_injector=corrupt)
    assert not current_manifest_path(tmp_path / "canonical", _scope()).exists()


def test_consumer_rejects_mixed_context_identity(tmp_path: Path) -> None:
    _publish(tmp_path)
    with pytest.raises(PublicationValidationError, match="context"):
        validate_current_publication(
            tmp_path / "canonical", _scope(), expected_context_reference="different-config"
        )


def test_scientific_context_validation_allows_cohort_namespace_change(tmp_path: Path) -> None:
    provenance = tmp_path / "epochs.provenance.json"
    provenance.write_text(
        json.dumps({"configuration_authority": {"scientific_fingerprint": "sha256:" + "a" * 64}}),
        encoding="utf-8",
    )
    published = _publish(
        tmp_path,
        context_reference="old-whole-config-context",
        required_artifacts={"epochs_fif__provenance": provenance},
    )
    validate_publication_scientific_context(published, "sha256:" + "a" * 64)
    with pytest.raises(PublicationValidationError, match="scientific context"):
        validate_publication_scientific_context(published, "sha256:" + "b" * 64)


def test_scientific_context_validation_fails_closed_without_authority(tmp_path: Path) -> None:
    provenance = tmp_path / "epochs.provenance.json"
    provenance.write_text(json.dumps({"configuration_authority": None}), encoding="utf-8")
    published = _publish(tmp_path, required_artifacts={"epochs_fif__provenance": provenance})
    with pytest.raises(PublicationValidationError, match="lacks configuration authority"):
        validate_publication_scientific_context(published, "sha256:" + "a" * 64)


def test_failure_before_promotion_preserves_previous_generation(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    newer = _file(tmp_path / "new.bin", b"new")

    def fail(boundary: str) -> None:
        if boundary == "after_validation_before_promotion":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        _publish(tmp_path, run_id="run-2", required_artifacts={"data": newer}, fault_injector=fail)
    assert validate_current_publication(tmp_path / "canonical", _scope()).generation_id == first.generation_id


def test_failure_after_generation_promotion_exposes_no_partial_generation(tmp_path: Path) -> None:
    def fail(boundary: str) -> None:
        if boundary == "after_generation_promotion_before_finality":
            raise RuntimeError("injected promotion boundary")

    with pytest.raises(RuntimeError, match="promotion boundary"):
        _publish(tmp_path, fault_injector=fail)
    with pytest.raises(PublicationValidationError):
        validate_current_publication(tmp_path / "canonical", _scope())


def test_finality_manifest_is_published_last(tmp_path: Path) -> None:
    observations: list[tuple[str, bool]] = []

    def observe(boundary: str) -> None:
        if boundary in {"before_finality_publication", "after_finality_publication"}:
            observations.append((boundary, current_manifest_path(tmp_path / "canonical", _scope()).exists()))

    _publish(tmp_path, fault_injector=observe)
    assert observations == [("before_finality_publication", False), ("after_finality_publication", True)]


def test_consumer_rejects_incomplete_publication(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    published.artifact_path("data").unlink()
    with pytest.raises(PublicationValidationError, match="missing/truncated"):
        validate_current_publication(tmp_path / "canonical", _scope())


def test_consumer_rejects_mutated_artifact_bytes(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    published.artifact_path("data").write_bytes(b"mutated!")
    with pytest.raises(PublicationValidationError, match="bytes changed"):
        validate_current_publication(tmp_path / "canonical", _scope())


def test_stale_sibling_is_not_consumable_in_new_generation(tmp_path: Path) -> None:
    first_a = _file(tmp_path / "first-a", b"a1")
    first_b = _file(tmp_path / "first-b", b"b1")
    first = _publish(tmp_path, required_artifacts={"a": first_a, "stale": first_b})
    second_a = _file(tmp_path / "second-a", b"a2")
    second = _publish(
        tmp_path,
        run_id="run-2",
        required_artifacts={"a": second_a},
        expected_predecessor_generation_id=first.generation_id,
    )
    assert second.artifact_path("stale") is None
    assert {artifact.label for artifact in validate_current_publication(tmp_path / "canonical", _scope()).artifacts} == {"a"}


def test_same_scope_compare_and_swap_conflicts_deterministically(tmp_path: Path) -> None:
    _publish(tmp_path, expected_predecessor_generation_id=None)
    with pytest.raises(PublicationConflictError, match="conflict"):
        _publish(tmp_path, run_id="run-2", expected_predecessor_generation_id=None)


def test_disjoint_scopes_publish_independently(tmp_path: Path) -> None:
    one = _publish(tmp_path, _scope("part1"), run_id="part1")
    two = _publish(tmp_path, _scope("part2"), run_id="part2")
    assert one.manifest_path != two.manifest_path
    assert validate_current_publication(tmp_path / "canonical", _scope("part1"))
    assert validate_current_publication(tmp_path / "canonical", _scope("part2"))


def test_cancellation_does_not_publish_candidate(tmp_path: Path) -> None:
    with pytest.raises(PublicationCancelled):
        _publish(tmp_path, cancelled=lambda: True)
    assert not current_manifest_path(tmp_path / "canonical", _scope()).exists()


def test_cancelled_replacement_preserves_previous_generation(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    with pytest.raises(PublicationCancelled):
        _publish(tmp_path, run_id="cancelled-replacement", cancelled=lambda: True)
    assert validate_current_publication(tmp_path / "canonical", _scope()).generation_id == first.generation_id


def test_validation_only_candidate_cannot_become_canonical(tmp_path: Path) -> None:
    with pytest.raises(PublicationValidationError, match="Validation-only"):
        _publish(tmp_path, validation_only=True)
    assert not current_manifest_path(tmp_path / "canonical", _scope()).exists()


def test_foreign_staging_resource_is_never_removed(tmp_path: Path) -> None:
    foreign: list[Path] = []

    def inject(boundary: str) -> None:
        if boundary == "artifact_staged:data":
            staging = next(publication_scope_root(tmp_path / "canonical", _scope()).glob("staging/*"))
            foreign.append(_file(staging / "foreign.keep", b"foreign"))
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        _publish(tmp_path, fault_injector=inject)
    assert foreign[0].read_bytes() == b"foreign"


def test_wrong_scope_finality_is_rejected(tmp_path: Path) -> None:
    _publish(tmp_path)
    source = current_manifest_path(tmp_path / "canonical", _scope())
    wrong = current_manifest_path(tmp_path / "canonical", _scope("part2"))
    wrong.parent.mkdir(parents=True)
    wrong.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(PublicationValidationError, match="scope"):
        validate_current_publication(tmp_path / "canonical", _scope("part2"))


def test_identity_references_are_generation_bound_and_exact(tmp_path: Path) -> None:
    references = {"publication:gate3a:scope": "generation-a", "manual_review": "decision-1"}
    published = _publish(tmp_path, identity_references=references)
    consumed = validate_current_publication(
        tmp_path / "canonical",
        _scope(),
        expected_identity_references=references,
    )
    assert consumed.identity_references == references
    with pytest.raises(PublicationValidationError, match="dependency"):
        validate_current_publication(
            tmp_path / "canonical",
            _scope(),
            expected_identity_references={"publication:gate3a:scope": "new-generation"},
        )


def test_consumer_during_candidate_staging_observes_previous_generation(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    staged = Event()
    release = Event()
    replacement = _file(tmp_path / "replacement.bin", b"replacement")

    def pause(boundary: str) -> None:
        if boundary == "after_validation_before_promotion":
            staged.set()
            assert release.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _publish,
            tmp_path,
            run_id="run-2",
            required_artifacts={"data": replacement},
            fault_injector=pause,
        )
        assert staged.wait(timeout=5)
        observed = validate_current_publication(tmp_path / "canonical", _scope())
        assert observed.generation_id == first.generation_id
        release.set()
        second = future.result(timeout=5)
    assert second.generation_id != first.generation_id


@pytest.mark.parametrize(
    "scope",
    (_scope(), PublicationScope("gate5a")),
    ids=("participant-scope", "group-scope"),
)
def test_concurrent_same_predecessor_has_one_complete_cas_winner(
    tmp_path: Path, scope: PublicationScope
) -> None:
    first = _publish(tmp_path, scope)
    barrier = Barrier(2)

    def candidate(run_id: str, payload: bytes):
        source = _file(tmp_path / f"{run_id}.bin", payload)

        def synchronize(boundary: str) -> None:
            if boundary == "after_validation_before_promotion":
                barrier.wait(timeout=5)

        return _publish(
            tmp_path,
            scope,
            run_id=run_id,
            required_artifacts={"data": source},
            expected_predecessor_generation_id=first.generation_id,
            fault_injector=synchronize,
        )

    outcomes = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(candidate, "candidate-a", b"a"),
            executor.submit(candidate, "candidate-b", b"b"),
        ]
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except PublicationConflictError as error:
                outcomes.append(error)
    winners = [item for item in outcomes if not isinstance(item, Exception)]
    conflicts = [item for item in outcomes if isinstance(item, PublicationConflictError)]
    assert len(winners) == 1 and len(conflicts) == 1
    current = validate_current_publication(tmp_path / "canonical", scope)
    assert current.generation_id == winners[0].generation_id


def test_promoted_orphan_is_not_adopted_and_previous_remains_current(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    replacement = _file(tmp_path / "orphan-candidate.bin", b"orphan")

    def stop_after_promotion(boundary: str) -> None:
        if boundary == "after_generation_promotion_before_finality":
            raise RuntimeError("leave promoted orphan")

    with pytest.raises(RuntimeError, match="promoted orphan"):
        _publish(
            tmp_path,
            run_id="orphan-run",
            required_artifacts={"data": replacement},
            expected_predecessor_generation_id=first.generation_id,
            fault_injector=stop_after_promotion,
        )
    root = publication_scope_root(tmp_path / "canonical", _scope())
    assert len(list((root / "generations").iterdir())) == 2
    assert validate_current_publication(tmp_path / "canonical", _scope()).generation_id == first.generation_id


def test_post_finality_corruption_is_reported_and_never_falls_back(tmp_path: Path) -> None:
    _publish(tmp_path)
    replacement = _file(tmp_path / "post-finality.bin", b"replacement")

    def corrupt_after_finality(boundary: str) -> None:
        if boundary == "after_finality_publication":
            manifest = json.loads(
                current_manifest_path(tmp_path / "canonical", _scope()).read_text(encoding="utf-8")
            )
            Path(manifest["artifacts"][0]["path"]).write_bytes(b"corrupt")

    with pytest.raises(PublicationValidationError, match="truncated|bytes changed"):
        _publish(
            tmp_path,
            run_id="post-finality-corruption",
            required_artifacts={"data": replacement},
            fault_injector=corrupt_after_finality,
        )
    with pytest.raises(PublicationValidationError, match="truncated|bytes changed"):
        validate_current_publication(tmp_path / "canonical", _scope())
