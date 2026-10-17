from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.provenance import PIPELINE_VERSION, write_group_provenance_sidecar, write_provenance_sidecar  # noqa: E402


class TestProvenance(unittest.TestCase):
    def test_provenance_json_can_be_created_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            input_file = root / "input.csv"
            output_file = root / "output.csv"
            config.write_text("answer: 42\n", encoding="utf-8")
            input_file.write_text("x\n1\n", encoding="utf-8")
            output_file.write_text("y\n2\n", encoding="utf-8")

            sidecar = write_provenance_sidecar(
                output_file,
                pipeline_stage="alignment",
                subject="sub999",
                recording_part="part1",
                root=root,
                config_path=config,
                input_files=[input_file],
                parameters_summary={"mode": "test"},
                warnings=["synthetic"],
            )

            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["pipeline_stage"], "alignment")
            self.assertEqual(payload["subject"], "sub999")
            self.assertEqual(payload["recording_part"], "part1")
            self.assertIn("config_hash", payload)
            self.assertIsNotNone(payload["config_hash"])
            self.assertEqual(payload["input_files"], [str(input_file)])
            self.assertEqual(payload["output_file"], str(output_file))
            self.assertIn("git_commit", payload)
            self.assertIn("mne_version", payload)
            self.assertIn("numpy_version", payload)
            self.assertIn("python_version", payload)
            self.assertIn("preprocessing_configuration_hash", payload)

    def test_subject_centred_outputs_collect_provenance_in_part_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_file = root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / "sub005" / "part1" / "raw_qc" / "channel_qc.csv"
            output_file.parent.mkdir(parents=True)
            output_file.write_text("x\n", encoding="utf-8")

            sidecar = write_provenance_sidecar(
                output_file,
                pipeline_stage="gate3a_preprocessing",
                subject="sub005",
                recording_part="part1",
                root=root,
                input_files=[],
            )

            self.assertEqual(sidecar.parent, root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / "sub005" / "part1" / "provenance")
            self.assertIn("raw_qc__channel_qc", sidecar.name)

    def test_pipeline_version_is_v1_freeze_label(self) -> None:
        self.assertEqual(PIPELINE_VERSION, "session2_eeg_v1.0")

    def test_group_provenance_sidecar_records_subject_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_file = root / "workspace" / "kawaii" / "session2" / "analysis" / "group" / "recognition" / "recognition_statistics.csv"
            output_file.parent.mkdir(parents=True)
            output_file.write_text("x\n", encoding="utf-8")

            sidecar = write_group_provenance_sidecar(
                output_file,
                gate="gate5c",
                branch="recognition",
                root=root,
                included_subjects=["sub005", "sub006"],
                excluded_subjects=["sub008"],
                exclusion_reasons={"sub008": "missing behavioural log"},
            )

            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["pipeline_stage"], "gate5c_recognition_group")
            self.assertEqual(payload["subject"], "group")
            self.assertEqual(payload["pipeline_version"], "session2_eeg_v1.0")
            self.assertEqual(payload["parameters_summary"]["included_subjects"], ["sub005", "sub006"])
            self.assertEqual(payload["parameters_summary"]["exclusion_reasons"]["sub008"], "missing behavioural log")


if __name__ == "__main__":
    unittest.main()
