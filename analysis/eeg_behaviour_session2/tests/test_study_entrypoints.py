"""Operator entry point contracts; no EEG processing or result publication."""

from pathlib import Path

import pytest

from eeg_software.workflows.kawaii_session2 import application_cli
from eeg_software.workflows.kawaii_session2.secondary_analyses import cli as secondary_cli
from eeg_software.workflows.kawaii_session2.secondary_analyses import preflight


def test_study_behaviour_defaults_to_two_source_behaviour_stages(monkeypatch):
    captured = []
    monkeypatch.setattr(secondary_cli, "main", lambda args: captured.append(args) or 0)
    assert application_cli.main(["behaviour", "--root", "/tmp/kawaii"]) == 0
    assert len(captured) == 1
    assert captured[0][-3:] == ["--stages", "recognition-behavior", "rating-memory"]


def test_study_secondary_preserves_explicit_erp_choice(monkeypatch):
    captured = []
    monkeypatch.setattr(secondary_cli, "main", lambda args: captured.append(args) or 0)
    assert application_cli.main([
        "secondary", "--stages", "true-vs-false-erp", "--check-inputs"
    ]) == 0
    assert captured[0][-3:] == ["--stages", "true-vs-false-erp", "--check-inputs"]


def test_study_secondary_default_is_behaviour_only(monkeypatch):
    captured = []
    monkeypatch.setattr(secondary_cli, "main", lambda args: captured.append(args) or 0)
    assert application_cli.main(["secondary"]) == 0
    assert captured[0][-3:] == ["--stages", "recognition-behavior", "rating-memory"]


def test_secondary_preflight_branch_does_not_execute_analysis(monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(preflight, "check_secondary_inputs", lambda **kw: {
        "stages": ["recognition-behavior"], "verified_file_count": 4, "warnings": [],
    })
    monkeypatch.setattr(secondary_cli, "run_secondary_analyses", lambda **kw: observed.append("ran"))
    assert secondary_cli.main(["--check-inputs", "--stages", "recognition-behavior"]) == 0
    assert "INPUT_PREFLIGHT_PASS" in capsys.readouterr().out
    assert observed == []


def test_preflight_rejects_unknown_or_empty_stages(tmp_path):
    with pytest.raises(ValueError, match="Unknown or empty"):
        preflight.check_secondary_inputs(root=tmp_path, stages=())
    with pytest.raises(ValueError, match="Unknown or empty"):
        preflight.check_secondary_inputs(root=tmp_path, stages=("random-stage",))


def test_preflight_reports_historical_erp_path_without_remapping(tmp_path, monkeypatch):
    core = tmp_path / "core.yaml"
    sec = tmp_path / "secondary.yaml"
    style = tmp_path / "style.yaml"
    measurement = tmp_path / "analysis" / "group" / "recognition.csv"
    for path in [core, sec, style, measurement]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy", encoding="utf-8")
    config = {
        "figure_style_path": str(style),
        "inputs": {
            "recognition_measurements": "analysis/group/recognition.csv",
        },
    }
    monkeypatch.setattr(preflight, "load_session2_eeg_config", lambda *a, **kw: {})
    monkeypatch.setattr(preflight, "load_secondary_analysis_config", lambda *a, **kw: config)
    monkeypatch.setattr(preflight, "verify_figure_style", lambda *a, **kw: style)
    import eeg_software.workflows.kawaii_session2.secondary_analyses.recognition_erp as erp
    monkeypatch.setattr(erp, "load_recognition_measurements", lambda *a, **kw: None)
    summary = preflight.check_secondary_inputs(
        root=tmp_path,
        core_config_path="core.yaml",
        secondary_config_path="secondary.yaml",
        stages=("true-vs-false-erp",),
    )
    assert summary["verified_file_count"] == 4
    assert len(summary["warnings"]) == 1
    assert "not remapped" not in summary["warnings"][0].lower() or "No input was remapped" in summary["warnings"][0]
    assert measurement.read_text() == "dummy"
