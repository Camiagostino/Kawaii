from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_supported_python_and_dependency_profiles_are_declared() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["requires-python"] == ">=3.11,<3.12"
    assert any(requirement.startswith("numpy>=") for requirement in project["dependencies"])
    assert any(requirement.startswith("mne>=") for requirement in project["dependencies"])

    optional = project["optional-dependencies"]
    assert optional["test"] == ["pytest>=9.1,<10"]
    assert "acquisition" not in optional

    runtime_lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert runtime_lines == ["-e ."]


def test_wsl_entry_constraints_record_provenance_and_critical_versions() -> None:
    text = (ROOT / "requirements" / "constraints-wsl2-py311-entry.txt").read_text(encoding="utf-8")

    for evidence in (
        "Ubuntu 24.04.5 LTS",
        "Python: 3.11.16",
        "uv: 0.12.6",
        "Source commit tested: b8e57d71dc13d00daf3a202b0c73389cfaf6f2b5",
        "pytest==9.1.1",
        "numpy==2.4.6",
        "pandas==3.0.5",
        "scipy==1.17.1",
        "statsmodels==0.15.0",
        "matplotlib==3.11.2",
        "mne==1.13.2",
        "psychopy==2026.2.4",
    ):
        assert evidence in text
