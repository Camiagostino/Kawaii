from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.figure_style import FIGURE_STYLE_SCHEMA_VERSION, load_figure_style
from eeg_software.workflows.kawaii_session2.resolved_config import resolve_session2_eeg_config

EXPECTED_SANITIZED_SCIENTIFIC_FINGERPRINT = "sha256:c80d23f7839b77766d25f47aef4030d6c41d134440247416caf9b7151dcbaef8"


def test_example_config_loads_with_stable_fingerprint(tmp_path: Path) -> None:
    # The application expects an operator-local config/session2_eeg.yaml.
    # The test supplies one in an isolated temporary project, never the checkout.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copyfile(ROOT / "config/session2_eeg.example.yaml", config_dir / "session2_eeg.yaml")
    resolved = resolve_session2_eeg_config(root=tmp_path)
    assert resolved.scientific_fingerprint == EXPECTED_SANITIZED_SCIENTIFIC_FINGERPRINT
    assert resolved.execution.subjects == ("synthetic001", "synthetic002")


def test_example_config_contains_no_cohort_authority() -> None:
    config = yaml.safe_load((ROOT / "config/session2_eeg.example.yaml").read_text(encoding="utf-8"))
    assert config["source_exposure_verification"]["enabled"] is False
    assert config["source_exposure_verification"]["required_subjects"] == []
    assert config["recordings"]["participant_recordings"] == {}
    assert config["analysis"]["exclude_subjects"] == []
    assert config["analysis"]["exclusion_reasons"] == {}


def test_figure_style_loads() -> None:
    style = load_figure_style(root=ROOT)
    assert FIGURE_STYLE_SCHEMA_VERSION == "kawaii_session2_figure_style_v1"
    assert style.dpi > 0
