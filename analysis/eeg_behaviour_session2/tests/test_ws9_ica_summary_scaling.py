from pathlib import Path

import mne
import pandas as pd
import pytest

import eeg_software.workflows.kawaii_session2.ica as ica_module


def test_pre_ws9_ica_diagnostic_rescans_each_comparison_input(monkeypatch, tmp_path: Path) -> None:
    """Characterize the pre-WS9 quadratic summary scan before memoization."""

    subjects = ["sub001", "sub002", "sub003", "sub004"]
    config = {"participants": {"defaults": {"included": subjects}}}
    paths = {subject: tmp_path / f"{subject}.fif" for subject in subjects}
    for path in paths.values():
        path.write_bytes(b"synthetic filtered raw placeholder")

    calls: list[tuple[str, str]] = []

    def fake_summary(subject, recording_part, raw, input_file, fit_channels, *, status):
        calls.append((str(subject), str(input_file)))
        return {
            "subject": subject,
            "recording_part": recording_part,
            "status": status,
            "input_file": str(input_file),
        }

    class FakeRaw:
        ch_names = ["Cz"]

    class NoReuseCache:
        """Explicitly model the pre-WS9 helper with no memoized reuse."""

        def get_or_compute(self, **kwargs):
            return dict(kwargs["compute"]())

    monkeypatch.setattr(ica_module, "_summarise_raw_for_ica", fake_summary)
    monkeypatch.setattr(ica_module, "_mne_ica_fit_channels", lambda raw, config: ["Cz"])
    monkeypatch.setattr(
        ica_module,
        "stage3a_filtered_raw_path",
        lambda config, subject, recording_part, root=None: paths[str(subject)],
    )
    monkeypatch.setattr(
        mne.io,
        "read_raw_fif",
        lambda path, preload=False, verbose=False: FakeRaw(),
        raising=False,
    )

    for subject in subjects:
        ica_module._build_ica_input_summaries(
            subject,
            "part1",
            FakeRaw(),
            paths[subject],
            ["Cz"],
            config,
            root=tmp_path,
            summary_cache=NoReuseCache(),
        )

    assert len(calls) == len(subjects) ** 2


def test_ws9_ica_diagnostic_cache_reuses_unique_identities_and_preserves_order(monkeypatch, tmp_path: Path) -> None:
    subjects = ["sub001", "sub002", "sub003", "sub004"]
    config = {"participants": {"defaults": {"included": subjects}}}
    paths = {subject: tmp_path / f"{subject}.fif" for subject in subjects}
    for index, path in enumerate(paths.values(), start=1):
        path.write_bytes(f"synthetic filtered raw {index}".encode("ascii"))

    calls: list[str] = []

    def fake_summary(subject, recording_part, raw, input_file, fit_channels, *, status):
        calls.append(str(input_file))
        return {
            "subject": subject,
            "recording_part": recording_part,
            "status": status,
            "input_file": str(input_file),
            "summary_value": f"summary:{input_file}",
        }

    class FakeRaw:
        ch_names = ["Cz"]

    monkeypatch.setattr(ica_module, "_summarise_raw_for_ica", fake_summary)
    monkeypatch.setattr(ica_module, "_mne_ica_fit_channels", lambda raw, config: ["Cz"])
    monkeypatch.setattr(
        ica_module,
        "stage3a_filtered_raw_path",
        lambda config, subject, recording_part, root=None: paths[str(subject)],
    )
    monkeypatch.setattr(
        mne.io,
        "read_raw_fif",
        lambda path, preload=False, verbose=False: FakeRaw(),
        raising=False,
    )

    cache = ica_module.ICADiagnosticSummaryCache()
    for subject in subjects:
        current, comparison = ica_module._build_ica_input_summaries(
            subject,
            "part1",
            FakeRaw(),
            paths[subject],
            ["Cz"],
            config,
            root=tmp_path,
            summary_cache=cache,
        )
        assert current.iloc[0]["subject"] == subject
        assert list(comparison["subject"]) == [subject] + [item for item in subjects if item != subject]

    assert len(calls) == len(subjects)

    cached_current, cached_comparison = ica_module._build_ica_input_summaries(
        subjects[0],
        "part1",
        FakeRaw(),
        paths[subjects[0]],
        ["Cz"],
        config,
        root=tmp_path,
        summary_cache=cache,
    )
    fresh_current, fresh_comparison = ica_module._build_ica_input_summaries(
        subjects[0],
        "part1",
        FakeRaw(),
        paths[subjects[0]],
        ["Cz"],
        config,
        root=tmp_path,
    )
    pd.testing.assert_frame_equal(cached_current, fresh_current)
    pd.testing.assert_frame_equal(cached_comparison, fresh_comparison)
    assert len(calls) == len(subjects) + len(subjects)

    reordered = {"participants": {"defaults": {"included": list(reversed(subjects))}}}
    _, reordered_comparison = ica_module._build_ica_input_summaries(
        subjects[-1],
        "part1",
        FakeRaw(),
        paths[subjects[-1]],
        ["Cz"],
        reordered,
        root=tmp_path,
        summary_cache=cache,
    )
    assert list(reordered_comparison["subject"]) == [subjects[-1], "sub003", "sub002", "sub001"]
    assert len(calls) == len(subjects) + len(subjects)


def test_ws9_ica_diagnostic_cache_invalidates_content_scope_and_context(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "source_a.fif"
    alias = tmp_path / "source_alias.fif"
    path.write_bytes(b"same source bytes")
    alias.write_bytes(path.read_bytes())
    config = {"participants": {"defaults": {"included": ["sub001"]}}}
    calls: list[str] = []

    def fake_summary(subject, recording_part, raw, input_file, fit_channels, *, status):
        calls.append(str(input_file))
        return {"summary_value": len(calls)}

    class FakeRaw:
        ch_names = ["Cz", "Fz"]

    monkeypatch.setattr(ica_module, "_summarise_raw_for_ica", fake_summary)
    cache = ica_module.ICADiagnosticSummaryCache()

    def build(input_file, *, part="part1", fit_channels=("Cz",), current_config=config):
        return ica_module._build_ica_input_summaries(
            "sub001",
            part,
            FakeRaw(),
            input_file,
            list(fit_channels),
            current_config,
            root=tmp_path,
            summary_cache=cache,
        )

    build(path)
    build(alias)
    assert len(calls) == 1  # path and mtime are not part of the identity key

    path.write_bytes(b"changed source bytes")
    build(path)
    build(path, part="part2")
    build(path, fit_channels=("Fz",))
    changed_config = {
        "participants": config["participants"],
        "preprocessing": {"ica": {"exclude_channels": ["Fz"]}},
    }
    build(path, current_config=changed_config)
    assert len(calls) == 5


def test_ws9_ica_diagnostic_cache_does_not_poison_retry_after_failure(tmp_path: Path) -> None:
    path = tmp_path / "source.fif"
    path.write_bytes(b"source")
    identity = ica_module.identify_source_file(path, "ica_filtered_raw_input")
    cache = ica_module.ICADiagnosticSummaryCache()
    config = {"participants": {"defaults": {"included": ["sub001"]}}}
    calls = 0

    def failing_compute():
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic summary failure")

    with pytest.raises(RuntimeError, match="synthetic summary failure"):
        cache.get_or_compute(
            source_identity=identity,
            recording_part="part1",
            fit_channels=["Cz"],
            config=config,
            compute=failing_compute,
        )

    result = cache.get_or_compute(
        source_identity=identity,
        recording_part="part1",
        fit_channels=["Cz"],
        config=config,
        compute=lambda: {"summary_value": "retry"},
    )
    assert result["summary_value"] == "retry"
    assert calls == 1
