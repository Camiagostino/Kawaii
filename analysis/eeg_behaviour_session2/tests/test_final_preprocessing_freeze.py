from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.freeze import (  # noqa: E402
    build_manual_review_snapshot,
    build_validation_manifest,
    expected_preprocessing_outputs,
    manual_review_snapshot_path,
    missing_expected_outputs,
    write_manual_review_snapshot,
    write_validation_manifest,
)
from eeg_software.workflows.kawaii_session2.ica import gate3b_output_paths  # noqa: E402
from eeg_software.workflows.kawaii_session2.ica_identity import ica_fit_identity_path  # noqa: E402
from eeg_software.workflows.kawaii_session2.manual_review import (  # noqa: E402
    write_bad_channel_review_decision,
    write_ica_review_decision,
)
from eeg_software.workflows.kawaii_session2.provenance import PIPELINE_VERSION, build_provenance_record, write_provenance_sidecar  # noqa: E402


class TestFinalPreprocessingFreeze(unittest.TestCase):
    def test_ica_solution_filename_follows_mne_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = gate3b_output_paths(_config(), "sub005", "part1", root=Path(tmp))

            self.assertEqual(outputs["ica_solution"].name, "sub005_part1-ica.fif")

    def test_manual_review_snapshot_records_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config()
            review = root / "workspace" / "kawaii" / "session2" / "manual_review"
            review.mkdir(parents=True)
            bad_decision = write_bad_channel_review_decision(
                config,
                "sub005",
                "part1",
                root=root,
                review_status="complete",
                confirmed_bad_channels=["Fz"],
                source_recording_id="sha256:" + "1" * 64,
                review_context_id="sha256:" + "2" * 64,
                reviewed_channels=["Fz", "Cz"],
                reviewer="synthetic-reviewer",
                reviewed_at_utc="2026-09-12T12:00:00+00:00",
                notes="bad note",
            )
            solution_id = "sha256:" + "3" * 64
            write_ica_review_decision(
                config, "sub005", "part1", root=root, review_status="complete",
                confirmed_ica_components=[1], ica_solution_id=solution_id,
                fitted_component_count=2,
                bad_channel_decision_id=bad_decision.semantic_decision_id,
                reviewer="synthetic-reviewer", reviewed_at_utc="2026-09-12T12:00:00Z",
                notes="ica note",
            )

            path = write_manual_review_snapshot(config, "sub005", "part1", root=root, config_path=None)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))

            self.assertEqual(path, manual_review_snapshot_path(config, "sub005", "part1", root=root))
            self.assertEqual(payload["confirmed_bad_channels"], ["Fz"])
            self.assertEqual(payload["confirmed_ica_components"], [1])
            self.assertEqual(payload["ica_review_status"], "complete")
            self.assertEqual(payload["ica_solution_id"], solution_id)
            self.assertEqual(payload["ica_decision_state"], "REVIEWED_WITH_ICA_COMPONENTS")
            self.assertIsNotNone(payload["ica_decision_id"])
            self.assertIsNotNone(payload["ica_audit_record_id"])
            self.assertIn("bad note", payload["notes"])
            self.assertEqual(payload["pipeline_version"], PIPELINE_VERSION)

    def test_manual_review_snapshot_labels_stale_ica_binding_without_becoming_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config()
            write_ica_review_decision(
                config, "sub005", "part1", root=root, review_status="complete",
                confirmed_ica_components=[], ica_solution_id="sha256:" + "3" * 64,
                fitted_component_count=2, bad_channel_decision_id="sha256:" + "4" * 64,
                reviewer="synthetic-reviewer", reviewed_at_utc="2026-09-12T12:00:00Z",
            )
            identity_path = ica_fit_identity_path(
                gate3b_output_paths(config, "sub005", "part1", root=root)["ica_solution"]
            )
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            identity_path.write_text("{}", encoding="utf-8")
            current = SimpleNamespace(
                solution_fingerprint="sha256:" + "9" * 64,
                fit_context=SimpleNamespace(algorithm_parameters={"n_components_effective": 2}),
            )
            with patch("eeg_software.workflows.kawaii_session2.freeze.read_ica_fit_artifact_identity", return_value=current):
                payload = build_manual_review_snapshot(config, "sub005", "part1", root=root)
            self.assertEqual(payload["ica_decision_state"], "STALE_OR_MISMATCHED")
            self.assertEqual(payload["confirmed_ica_components"], [])
            self.assertNotEqual(payload["ica_solution_id"], payload["current_ica_solution_id"])

    def test_validation_manifest_marks_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config()
            part = root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / "sub005" / "part1"
            for rel in (
                "alignment/trial_alignment.csv",
                "raw_qc/raw_qc_summary.csv",
                "raw_qc/channel_qc.csv",
                "raw_qc/report.md",
                "preprocessing/filtered_raw.fif",
                "ica/sub005_part1-ica.fif",
                "ica/ica_review_template.yaml",
                "ica/report.md",
                "ica/component_summary.csv",
                "epochs/all_image_epochs-epo.fif",
                "epochs/block01-epo.fif",
                "epochs/block02-epo.fif",
                "epochs/block_processing_summary.csv",
                "epochs/epoch_metadata.csv",
                "epochs/rejection_table.csv",
                "epochs/report.md",
                "provenance/manual_review_snapshot.yaml",
                "provenance/example.provenance.json",
            ):
                target = part / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x", encoding="utf-8")
            (part / "epochs" / "block_processing_summary.csv").write_text(
                "block,block_index,n_epochs_created\n1,1,50\n2,2,50\n",
                encoding="utf-8",
            )

            path = write_validation_manifest(config, "sub005", "part1", root=root)
            manifest = path.read_text(encoding="utf-8")

            self.assertIn("alignment,trial_alignment.csv,True,ok", manifest)
            self.assertIn("epochs,all_image_epochs-epo.fif,True,ok", manifest)
            self.assertIn("epochs,block01-epo.fif,True,ok", manifest)
            self.assertIn("epochs,block02-epo.fif,True,ok", manifest)
            self.assertNotIn("image_onset_epochs-epo.fif", manifest)
            self.assertEqual(missing_expected_outputs(config, "sub005", "part1", root=root), [])

    def test_validation_manifest_requires_only_available_block_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config()
            part = root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / "sub005" / "part1"
            for rel in (
                "alignment/trial_alignment.csv",
                "raw_qc/raw_qc_summary.csv",
                "raw_qc/channel_qc.csv",
                "raw_qc/report.md",
                "preprocessing/filtered_raw.fif",
                "ica/sub005_part1-ica.fif",
                "ica/ica_review_template.yaml",
                "ica/report.md",
                "ica/component_summary.csv",
                "epochs/all_image_epochs-epo.fif",
                "epochs/block01-epo.fif",
                "epochs/block_processing_summary.csv",
                "epochs/epoch_metadata.csv",
                "epochs/rejection_table.csv",
                "epochs/report.md",
                "provenance/manual_review_snapshot.yaml",
                "provenance/example.provenance.json",
            ):
                target = part / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x", encoding="utf-8")
            (part / "epochs" / "block_processing_summary.csv").write_text(
                "block,block_index,n_epochs_created\n1,1,50\n",
                encoding="utf-8",
            )

            manifest = write_validation_manifest(config, "sub005", "part1", root=root).read_text(encoding="utf-8")

            self.assertIn("epochs,block01-epo.fif,True,ok", manifest)
            self.assertNotIn("block02-epo.fif", manifest)
            self.assertEqual(missing_expected_outputs(config, "sub005", "part1", root=root), [])

    def test_validation_manifest_reports_missing_outputs(self) -> None:
        manifest = build_validation_manifest([{"stage": "ica", "path": Path("missing-ica.fif"), "filename": "missing-ica.fif", "notes": ""}])

        self.assertEqual(manifest.iloc[0]["status"], "missing")
        self.assertFalse(bool(manifest.iloc[0]["exists"]))

    def test_provenance_records_freeze_fields_and_snapshot_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / "sub005" / "part1" / "raw_qc" / "report.md"
            output.parent.mkdir(parents=True)
            output.write_text("report", encoding="utf-8")
            snapshot = root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / "sub005" / "part1" / "provenance" / "manual_review_snapshot.yaml"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("subject: sub005\n", encoding="utf-8")

            sidecar = write_provenance_sidecar(output, pipeline_stage="gate3a_preprocessing", subject="sub005", recording_part="part1", root=root)
            payload = json.loads(sidecar.read_text(encoding="utf-8"))

            self.assertEqual(payload["pipeline_version"], PIPELINE_VERSION)
            self.assertIn("mne_version", payload)
            self.assertIn("numpy_version", payload)
            self.assertIn("python_version", payload)
            self.assertIn("preprocessing_configuration_hash", payload)
            self.assertEqual(payload["manual_review_snapshot_path"], str(snapshot))

    def test_build_provenance_uses_unavailable_git_fallback(self) -> None:
        record = build_provenance_record(pipeline_stage="test", subject="sub005", output_file="out.txt", root=Path("Z:/definitely/not/a/repo"))

        self.assertEqual(record["git_commit"], "unavailable")


def _config() -> dict:
    return {
        "paths": {
            "output_dir": "workspace/kawaii/session2/analysis",
            "manual_review_dir": "workspace/kawaii/session2/manual_review",
        }
    }


if __name__ == "__main__":
    unittest.main()
