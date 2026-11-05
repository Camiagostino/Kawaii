"""Synthetic contract tests: unified participant/normative orchestration only.

No real study input, EEG processing, ICA or model fitting is exercised.
"""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from eeg_software.workflows.kawaii_session2 import cuteness_rating_workflow as workflow


def _root(tmp_path):
    root = tmp_path / "analysis_checkout"
    (root / "tools").mkdir(parents=True)
    for file in ("run_kawaii_norm02_exploratory.py", "run_kawaii_cuteness_report_figures.py"):
        (root / "tools" / file).write_text("# synthetic tool stub\n")
    pub = root / "workspace/kawaii/session2/corrections/r11_qc_v3"
    pub.mkdir(parents=True)
    return root, pub


def _fake_calls(monkeypatch):
    observed = []

    def call(root, name, argv):
        observed.append((name, list(argv)))
        if name == "run_kawaii_norm02_exploratory.py":
            out = Path(argv[argv.index("--output") + 1])
            out.mkdir()
            with zipfile.ZipFile(out.with_suffix(".zip"), "w") as z:
                z.writestr("MANIFEST.json", '{"synthetic": true}')
        elif name == "run_kawaii_cuteness_report_figures.py":
            out = Path(argv[argv.index("--output-dir") + 1])
            out.mkdir()
            (out / "MANIFEST.json").write_text('{"synthetic": true}', encoding="utf-8")
            (out / "mock_figure.png").write_bytes(b"synthetic")

    monkeypatch.setattr(workflow, "_call_script", call)
    monkeypatch.setattr(workflow, "_checkout_head", lambda root: "f" * 40)
    return observed


def test_three_rating_sources_are_explicit():
    parser = ArgumentParser()
    workflow.add_arguments(parser)
    for choice in workflow.CHOICES:
        opts = parser.parse_args(["--rating-source", choice, "--output-dir", "example"])
        assert opts.rating_source == choice
    with pytest.raises(SystemExit):
        parser.parse_args(["--rating-source", "wrong", "--output-dir", "example"])


def test_participant_uses_published_figures_not_epoch_aggregation(tmp_path, monkeypatch):
    root, pub = _root(tmp_path)
    calls = _fake_calls(monkeypatch)
    dest = tmp_path / "participant_results"
    out = workflow.execute(root=root, output_dir=dest, rating_source="participant")
    assert out == dest
    assert [name for name, _ in calls] == ["run_kawaii_cuteness_report_figures.py"]
    fig_args = calls[0][1]
    assert fig_args[fig_args.index("--rating-source") + 1] == "participant"
    assert Path(fig_args[fig_args.index("--source-root") + 1]) == pub
    assert "--normative-archive" not in fig_args
    assert json.loads((dest / "MANIFEST.json").read_text())["normative_mode"] is None


def test_normative_reuses_verified_archive_without_regenerating(tmp_path, monkeypatch):
    root, _ = _root(tmp_path)
    archive = tmp_path / "authorised_summary.zip"
    archive.write_bytes(b"synthetic archive bytes")
    calls = _fake_calls(monkeypatch)
    dest = tmp_path / "from_archive"
    workflow.execute(root=root, output_dir=dest, rating_source="normative",
                     normative_archive=archive, acknowledge_sub012_exploratory=True)
    assert [name for name, _ in calls] == ["run_kawaii_cuteness_report_figures.py"]
    fig_args = calls[0][1]
    assert Path(fig_args[fig_args.index("--normative-archive") + 1]) == archive
    assert "--source-root" not in fig_args
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert manifest["normative_mode"] == "existing_archive"
    assert manifest["normative_archive_sha256"] == workflow._sha256(archive)


def test_normative_calculates_only_from_accepted_gate3c(tmp_path, monkeypatch):
    root, _ = _root(tmp_path)
    reference = tmp_path / "approved_crosswalk.csv"
    reference.write_text("synthetic", encoding="utf-8")
    calls = _fake_calls(monkeypatch)
    dest = tmp_path / "calculated_normative"
    workflow.execute(root=root, output_dir=dest, rating_source="normative",
                     aaps_reference=reference, acknowledge_sub012_exploratory=True)
    assert [name for name, _ in calls] == [
        "run_kawaii_norm02_exploratory.py", "run_kawaii_cuteness_report_figures.py"
    ]
    normargs = calls[0][1]
    assert normargs[normargs.index("--source-head") + 1] == "f" * 40
    assert "--acknowledge-sub012-exploratory" in normargs
    assert normargs[normargs.index("--aaps-reference") + 1] == str(reference)
    assert (dest / "normative_summary.zip").is_file()
    assert (dest / "figures/MANIFEST.json").is_file()
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert manifest["normative_mode"] == "authenticated_corrected_epochs"
    assert manifest["normative_checkout_head"] == "f" * 40


def test_both_keeps_provenance_and_figures_separate(tmp_path, monkeypatch):
    root, _ = _root(tmp_path)
    calls = _fake_calls(monkeypatch)
    dest = tmp_path / "both_results"
    workflow.execute(root=root, output_dir=dest, rating_source="both",
                     acknowledge_sub012_exploratory=True)
    assert len(calls) == 2
    fig_args = calls[-1][1]
    assert "--source-root" in fig_args and "--normative-archive" in fig_args
    assert fig_args[fig_args.index("--rating-source") + 1] == "both"
    assert json.loads((dest / "MANIFEST.json").read_text())["separate_rating_authorities"]


def test_sub012_requires_explicit_acknowledgement_even_with_archive(tmp_path):
    root, _ = _root(tmp_path)
    path = tmp_path / "any.zip"
    path.write_bytes(b"synthetic")
    for choice in ("normative", "both"):
        with pytest.raises(ValueError, match="acknowledge-sub012"):
            workflow.execute(root=root, output_dir=tmp_path / choice,
                             rating_source=choice, normative_archive=path)


def test_invalid_mixed_modes_fail_closed(tmp_path):
    root, _ = _root(tmp_path)
    archive = tmp_path / "summary.zip"
    archive.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="Normative-only options"):
        workflow.execute(root=root, output_dir=tmp_path / "bad", rating_source="participant",
                         normative_archive=archive)
    with pytest.raises(ValueError, match="only used"):
        workflow.execute(root=root, output_dir=tmp_path / "bad", rating_source="normative",
                         normative_archive=archive, aaps_reference=archive,
                         acknowledge_sub012_exploratory=True)
    with pytest.raises(ValueError, match="source-root"):
        workflow.execute(root=root, output_dir=tmp_path / "bad", rating_source="normative",
                         source_root=root, acknowledge_sub012_exploratory=True)


def test_output_must_be_fresh_and_outside_repository(tmp_path, monkeypatch):
    root, _ = _root(tmp_path)
    _fake_calls(monkeypatch)
    with pytest.raises(ValueError, match="outside"):
        workflow.execute(root=root, output_dir=root / "outputs" / "new", rating_source="participant")
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        workflow.execute(root=root, output_dir=existing, rating_source="participant")


def test_failed_renderer_does_not_publish_partial_results(tmp_path, monkeypatch):
    root, _ = _root(tmp_path)
    dest = tmp_path / "failed_output"

    def fail(root, name, argv):
        if name == "run_kawaii_norm02_exploratory.py":
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir()
            with zipfile.ZipFile(output.with_suffix(".zip"), "w") as z:
                z.writestr("MANIFEST.json", "{}")
        else:
            raise subprocess.CalledProcessError(3, argv)

    monkeypatch.setattr(workflow, "_call_script", fail)
    monkeypatch.setattr(workflow, "_checkout_head", lambda root: "d" * 40)
    with pytest.raises(subprocess.CalledProcessError):
        workflow.execute(root=root, output_dir=dest, rating_source="normative",
                         acknowledge_sub012_exploratory=True)
    assert not dest.exists()


def test_missing_normative_script_blocks_without_output(tmp_path):
    root, _ = _root(tmp_path)
    (root / "tools/run_kawaii_norm02_exploratory.py").unlink()
    dest = tmp_path / "should_not_exist"
    with pytest.raises(FileNotFoundError, match="implementation missing"):
        workflow.execute(root=root, output_dir=dest, rating_source="normative",
                         acknowledge_sub012_exploratory=True)
    assert not dest.exists()


def test_application_cli_registers_cuteness_entry_point():
    # The full internal checkout has further coordinator dependencies not
    # bundled in this source-only review fixture. Verify the reviewed adapter
    # wiring and let the owner run the real installed CLI help in WSL.
    source = (Path(__file__).resolve().parents[1] /
              "src/eeg_software/workflows/kawaii_session2/application_cli.py").read_text()
    assert 'subcommands.add_parser(\n        "cuteness"' in source
    assert 'if args.command == "cuteness":' in source
    assert 'return run_cuteness(args)' in source
