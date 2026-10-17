from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.manual_review import (  # noqa: E402
    ensure_manual_review_scaffold,
    manual_review_paths,
    read_confirmed_bad_channels,
    read_confirmed_ica_components,
    read_ica_review_decision,
    ica_solution_identity,
    write_ica_review_decision,
    write_ica_review_template,
    write_bad_channel_review_decision,
)

SOLUTION_A = "sha256:" + "a" * 64
SOLUTION_B = "sha256:" + "b" * 64
BAD_DECISION_A = "sha256:" + "c" * 64
BAD_DECISION_B = "sha256:" + "d" * 64
REVIEWED_AT = "2026-09-12T12:00:00+00:00"


def _config(root: Path) -> dict:
    return {
        "paths": {
            "manual_review_dir": "manual_review",
            "raw_eeg_dir": "raw",
            "session2_logs_dir": "logs",
            "session2_schedules_dir": "schedules",
            "output_dir": "out",
            "derivatives_dir": "out/derivatives",
            "brainvision_working_dir": "out/derivatives/brainvision",
            "alignment_dir": "out/derivatives/alignment",
            "preprocessing_dir": "out/derivatives/preprocessing",
            "raw_qc_dir": "out/qc/raw",
            "report_dir": "out/reports",
            "qc_dir": "out/qc",
        }
    }


class TestManualReviewScaffold(unittest.TestCase):
    def test_creates_manual_review_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)

            paths = ensure_manual_review_scaffold({"sub005": ["part1"]}, config, root=root)

            self.assertTrue(paths["bad_channels"].exists())
            self.assertTrue(paths["ica_components"].exists())
            self.assertTrue(paths["participant_notes"].exists())

    def test_preserves_existing_decisions_and_adds_missing_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            paths = manual_review_paths(config, root=root)
            paths["manual_review_dir"].mkdir(parents=True)
            paths["bad_channels"].write_text(
                yaml.safe_dump({"sub005": {"part1": {"confirmed_bad_channels": ["Fz"], "notes": "keep"}}}, sort_keys=False),
                encoding="utf-8",
            )

            ensure_manual_review_scaffold({"sub005": ["part1", "part2"]}, config, root=root)
            data = yaml.safe_load(paths["bad_channels"].read_text(encoding="utf-8"))

            self.assertEqual(data["sub005"]["part1"]["confirmed_bad_channels"], ["Fz"])
            self.assertEqual(data["sub005"]["part1"]["notes"], "keep")
            self.assertIn("part2", data["sub005"])
            self.assertEqual(data["sub005"]["part2"]["confirmed_bad_channels"], [])

    def test_reads_confirmed_bad_channels_and_ica_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            paths = manual_review_paths(config, root=root)
            paths["manual_review_dir"].mkdir(parents=True)
            write_bad_channel_review_decision(
                config, "sub005", "part1", root=root, review_status="complete",
                confirmed_bad_channels=["Fz"], reviewed_channels=["Fz", "Cz"],
                source_recording_id="sha256:" + "1" * 64, review_context_id="sha256:" + "2" * 64,
                reviewer="synthetic-reviewer", reviewed_at_utc="2026-09-12T12:00:00+00:00",
            )
            write_ica_review_decision(
                config, "sub005", "part1", root=root, review_status="complete",
                confirmed_ica_components=[0, 2], ica_solution_id=SOLUTION_A,
                fitted_component_count=3, bad_channel_decision_id=BAD_DECISION_A,
                reviewer="synthetic-reviewer", reviewed_at_utc=REVIEWED_AT,
            )

            self.assertEqual(read_confirmed_bad_channels(config, "sub005", "part1", root=root), ["Fz"])
            self.assertEqual(read_confirmed_ica_components(config, "sub005", "part1", root=root), [0, 2])

    def test_missing_and_completed_zero_exclusion_are_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            missing = read_ica_review_decision(config, "sub005", "part1", root=root)
            self.assertEqual(missing.review_status, "not_reviewed")
            self.assertFalse(missing.complete)
            with self.assertRaisesRegex(ValueError, "not complete"):
                read_ica_review_decision(config, "sub005", "part1", root=root, require_complete=True)

            write_ica_review_decision(
                config, "sub005", "part1", root=root, review_status="complete",
                confirmed_ica_components=[], ica_solution_id=SOLUTION_A,
                fitted_component_count=4, bad_channel_decision_id=BAD_DECISION_A,
                reviewer="synthetic-reviewer", reviewed_at_utc=REVIEWED_AT, notes="retain all",
            )
            complete = read_ica_review_decision(
                config, "sub005", "part1", root=root, expected_solution_id=SOLUTION_A,
                expected_component_count=4, expected_bad_channel_decision_id=BAD_DECISION_A,
                require_complete=True,
            )
            self.assertTrue(complete.complete)
            self.assertEqual(complete.confirmed_ica_components, ())
            self.assertEqual(complete.decision_state, "REVIEWED_NO_ICA_COMPONENTS")

    def test_solution_scope_staleness_and_component_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = _config(root); paths = manual_review_paths(config, root=root)
            write_ica_review_decision(
                config, "sub005", "part1", root=root, review_status="complete",
                confirmed_ica_components=[1, 3], ica_solution_id=SOLUTION_A,
                fitted_component_count=4, bad_channel_decision_id=BAD_DECISION_A,
                reviewer="reviewer", reviewed_at_utc=REVIEWED_AT,
            )
            decision = read_ica_review_decision(
                config, "sub005", "part1", root=root, expected_solution_id=SOLUTION_A,
                expected_component_count=4, expected_bad_channel_decision_id=BAD_DECISION_A,
                require_complete=True,
            )
            self.assertEqual(decision.confirmed_ica_components, (1, 3))
            with self.assertRaisesRegex(ValueError, "Stale ICA review"):
                read_ica_review_decision(config, "sub005", "part1", root=root, expected_solution_id=SOLUTION_B, expected_component_count=4)
            with self.assertRaisesRegex(ValueError, "component-count mismatch"):
                read_ica_review_decision(config, "sub005", "part1", root=root, expected_solution_id=SOLUTION_A, expected_component_count=5)
            with self.assertRaisesRegex(ValueError, "bound bad-channel decision"):
                read_ica_review_decision(
                    config, "sub005", "part1", root=root, expected_solution_id=SOLUTION_A,
                    expected_component_count=4, expected_bad_channel_decision_id=BAD_DECISION_B,
                )
            with self.assertRaisesRegex(ValueError, "bound bad-channel decision"):
                read_ica_review_decision(
                    config, "sub005", "part1", root=root, expected_solution_id=SOLUTION_A,
                    expected_component_count=4, expected_bad_channel_decision_id=None,
                )
            base = yaml.safe_load(paths["ica_components"].read_text(encoding="utf-8"))["sub005"]["part1"]
            for field, value, message in (
                ("participant", "sub006", "participant mismatch"),
                ("session", "session1", "session mismatch"),
                ("recording_part", "part2", "part mismatch"),
                ("confirmed_ica_components", [1, 1], "Duplicate"),
                ("confirmed_ica_components", [-1], "non-negative"),
                ("confirmed_ica_components", [4], "out of range"),
            ):
                entry = dict(base); entry[field] = value
                paths["ica_components"].write_text(
                    yaml.safe_dump({"sub005": {"part1": entry}}), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, message):
                    read_ica_review_decision(config, "sub005", "part1", root=root, expected_solution_id=SOLUTION_A, expected_component_count=4)

    def test_in_progress_cannot_masquerade_as_confirmed_and_parts_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = _config(root); paths = manual_review_paths(config, root=root)
            paths["manual_review_dir"].mkdir(parents=True)
            payload = {"sub005": {
                "part1": {"participant": "sub005", "recording_part": "part1", "review_status": "in_progress", "confirmed_ica_components": [1]},
            }}
            paths["ica_components"].write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete ICA review"):
                read_ica_review_decision(config, "sub005", "part1", root=root)
            payload["sub005"]["part1"]["confirmed_ica_components"] = []
            paths["ica_components"].write_text(yaml.safe_dump(payload), encoding="utf-8")
            write_ica_review_decision(
                config, "sub005", "part2", root=root, review_status="complete",
                confirmed_ica_components=[0], ica_solution_id=SOLUTION_B,
                fitted_component_count=2, bad_channel_decision_id=BAD_DECISION_B,
                reviewer="reviewer", reviewed_at_utc=REVIEWED_AT,
            )
            part2 = read_ica_review_decision(config, "sub005", "part2", root=root, expected_solution_id=SOLUTION_B, expected_component_count=2)
            self.assertEqual(part2.confirmed_ica_components, (0,))

    def test_ica_audit_identity_detects_tampering_but_metadata_does_not_change_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = _config(root); paths = manual_review_paths(config, root=root)
            first = write_ica_review_decision(
                config, "sub005", "part1", root=root, review_status="complete",
                confirmed_ica_components=[1], ica_solution_id=SOLUTION_A,
                fitted_component_count=3, bad_channel_decision_id=BAD_DECISION_A,
                reviewer="reviewer-a", reviewed_at_utc=REVIEWED_AT, notes="first",
            )
            second = write_ica_review_decision(
                config, "sub005", "part1", root=root, review_status="complete",
                confirmed_ica_components=[1], ica_solution_id=SOLUTION_A,
                fitted_component_count=3, bad_channel_decision_id=BAD_DECISION_A,
                reviewer="reviewer-b", reviewed_at_utc="2026-09-12T13:00:00Z", notes="second",
            )
            self.assertEqual(first.decision_id, second.decision_id)
            self.assertNotEqual(first.audit_record_id, second.audit_record_id)
            payload = yaml.safe_load(paths["ica_components"].read_text(encoding="utf-8"))
            payload["sub005"]["part1"]["confirmed_ica_components"] = [2]
            paths["ica_components"].write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantic decision fingerprint mismatch"):
                read_ica_review_decision(config, "sub005", "part1", root=root)

    def test_legacy_completed_ica_entry_is_preserved_but_not_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = _config(root); paths = manual_review_paths(config, root=root)
            paths["manual_review_dir"].mkdir(parents=True)
            paths["ica_components"].write_text(yaml.safe_dump({"sub005": {"part1": {
                "participant": "sub005", "recording_part": "part1", "review_status": "complete",
                "ica_solution_id": SOLUTION_A, "fitted_component_count": 4,
                "confirmed_ica_components": [1], "notes": "historical exact choice",
            }}}), encoding="utf-8")
            decision = read_ica_review_decision(config, "sub005", "part1", root=root)
            self.assertTrue(decision.legacy_ambiguous)
            self.assertFalse(decision.complete)
            self.assertEqual(decision.confirmed_ica_components, (1,))
            self.assertEqual(decision.decision_state, "LEGACY_UNVERIFIED")
            with self.assertRaisesRegex(ValueError, "legacy/unverified"):
                read_ica_review_decision(config, "sub005", "part1", root=root, require_complete=True)

    def test_solution_identity_is_deterministic_and_bound_to_part_and_decomposition(self) -> None:
        ica = SimpleNamespace(
            method="infomax", ch_names=["Cz", "Fz"], n_components_=2,
            unmixing_matrix_=np.eye(2), mixing_matrix_=np.eye(2),
            pca_components_=np.eye(2), pre_whitener_=np.ones((2, 1)),
        )
        first = ica_solution_identity(ica, "sub005", "part1")
        self.assertEqual(first, ica_solution_identity(ica, "sub005", "part1"))
        self.assertNotEqual(first, ica_solution_identity(ica, "sub005", "part2"))
        ica.unmixing_matrix_[0, 0] = 2.0
        self.assertNotEqual(first, ica_solution_identity(ica, "sub005", "part1"))

    def test_review_template_separates_suggestions_from_confirmed_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_ica_review_template(
                Path(tmp) / "template.yaml", subject="sub005", recording_part="part1",
                ica_solution_id="sha256:test", fitted_component_count=4,
                automatic_eog_suggested_components=[2, 0, 2],
            )
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["automatic_eog_suggested_components"], [0, 2])
            self.assertEqual(payload["confirmed_ica_components"], [])
            self.assertEqual(payload["review_status"], "not_reviewed")

    def test_notes_file_generation_adds_headings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)

            paths = ensure_manual_review_scaffold({"sub005": ["part1"], "sub006": ["part1"]}, config, root=root)
            text = paths["participant_notes"].read_text(encoding="utf-8")

            self.assertIn("## sub005", text)
            self.assertIn("### sub006 part1", text)


if __name__ == "__main__":
    unittest.main()
