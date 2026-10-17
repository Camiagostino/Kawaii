from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.project import find_project_root  # noqa: E402
from eeg_software.workflows.kawaii_session2.paths import subject_part_stage_dir  # noqa: E402


class TestProjectRoot(unittest.TestCase):
    def test_find_project_root_from_repo_file(self) -> None:
        root = find_project_root(Path(__file__))
        self.assertTrue((root / "README.md").exists())
        self.assertTrue((root / "pyproject.toml").exists())

    def test_subject_part_stage_dirs_are_subject_centred(self) -> None:
        config = {"paths": {"output_dir": "workspace/kawaii/session2/analysis"}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_qc = subject_part_stage_dir(config, "sub005", "part1", "raw_qc", root=root)
            provenance = subject_part_stage_dir(config, "sub005", "part1", "provenance", root=root)

            self.assertEqual(raw_qc, root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / "sub005" / "part1" / "raw_qc")
            self.assertEqual(provenance, root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / "sub005" / "part1" / "provenance")


if __name__ == "__main__":
    unittest.main()
