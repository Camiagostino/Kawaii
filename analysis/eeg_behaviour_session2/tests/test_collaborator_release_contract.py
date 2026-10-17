"""Collaborator-release configuration and import contracts (no study data)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import yaml
from eeg_software.workflows.kawaii_session2.secondary_analyses.config import load_secondary_analysis_config, verify_figure_style
ROOT=Path(__file__).resolve().parents[1]
SCIENTIFIC_FIELDS=['schema_version', 'analysis_scope', 'primary_analysis_unchanged', 'core_config_path', 'figure_style_path', 'figure_style_sha256', 'recognition_cells', 'families', 'existing_arousal_coefficient', 'response_locked_inference']
SCIENTIFIC_SHA256="fd3f849bb4da037b51287534134eb32c93e30ae5bbbb708cfa67f85fff9f1a3c"

def test_collaborator_secondary_has_synthetic_identity_and_explicit_inputs():
    cfg=load_secondary_analysis_config(ROOT, "config/kawaii_session2_secondary_analyses.example.yaml")
    assert cfg["participants"] == ["synthetic001","synthetic002"]
    assert cfg["output_dir"] == "outputs/session2/secondary_analyses"
    assert len(cfg["inputs"]) == 6
    assert all(value.startswith("USER_SUPPLIED/validated_group/") for value in cfg["inputs"].values())
    # Real operators may legitimately have a Git-ignored local config. The
    # contract is that it must never be tracked, not that it cannot exist.
    if (ROOT / ".git").exists():
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "config/session2_eeg.yaml"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode != 0, "Local cohort config must not be tracked"

def test_secondary_scientific_settings_are_frozen():
    cfg=load_secondary_analysis_config(ROOT, "config/kawaii_session2_secondary_analyses.example.yaml")
    obj={key:cfg[key] for key in SCIENTIFIC_FIELDS}
    sha=hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert sha == SCIENTIFIC_SHA256
    assert verify_figure_style(ROOT,cfg).is_file()

def test_collaborator_project_root_marker_is_present():
    from eeg_software.project import find_project_root
    assert find_project_root(ROOT / "src" / "eeg_software" / "project.py") == ROOT
