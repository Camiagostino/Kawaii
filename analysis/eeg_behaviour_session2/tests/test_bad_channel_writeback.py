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

from eeg_software.workflows.kawaii_session2.epoching import build_interpolation_table  # noqa: E402
from eeg_software.workflows.kawaii_session2.raw_qc import bad_channel_decision_table  # noqa: E402
from eeg_software.workflows.kawaii_session2.manual_review import (  # noqa: E402
    bad_channel_review_context_identity,
    bad_channel_source_identity,
    manual_review_paths,
    read_bad_channel_review_decision,
    read_confirmed_bad_channels,
    write_bad_channel_review_decision,
)

SOURCE_ID = "sha256:" + "1" * 64
CONTEXT_ID = "sha256:" + "2" * 64


def _config() -> dict:
    return {
        "paths": {
            "manual_review_dir": "manual_review", "raw_eeg_dir": "raw", "session2_logs_dir": "logs",
            "session2_schedules_dir": "schedules", "output_dir": "out", "derivatives_dir": "out/derivatives",
            "brainvision_working_dir": "out/derivatives/brainvision", "alignment_dir": "out/alignment",
            "preprocessing_dir": "out/preprocessing", "raw_qc_dir": "out/qc/raw", "report_dir": "out/reports",
            "qc_dir": "out/qc",
        },
        "preprocessing": {"montage": "standard_1005", "line_frequency_hz": 50.0, "raw_qc": {"window_s": 40}},
    }


class TestBadChannelWriteback(unittest.TestCase):
    def _write(self, root: Path, channels=(), **overrides):
        values = {
            "review_status": "complete",
            "confirmed_bad_channels": list(channels),
            "source_recording_id": SOURCE_ID,
            "review_context_id": CONTEXT_ID,
            "reviewed_channels": ["A", "B", "C"],
            "reviewer": "synthetic-reviewer",
            "reviewed_at_utc": "2026-09-12T12:00:00+00:00",
            "notes": "synthetic only",
        }
        values.update(overrides)
        return write_bad_channel_review_decision(_config(), "subA", "part1", root=root, **values)

    def test_reviewed_zero_reviewed_with_bads_and_not_reviewed_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zero = self._write(root)
            self.assertEqual(zero.decision_state, "REVIEWED_NO_BAD_CHANNELS")
            self.assertEqual(read_confirmed_bad_channels(_config(), "subA", "part1", root=root), [])
            with_bads = self._write(
                root, ["A", "B"],
                expected_current_semantic_decision_id=zero.semantic_decision_id,
            )
            self.assertEqual(with_bads.decision_state, "REVIEWED_WITH_BAD_CHANNELS")
            self.assertEqual(read_confirmed_bad_channels(_config(), "subA", "part1", root=root), ["A", "B"])
            missing = read_bad_channel_review_decision(_config(), "subA", "part2", root=root)
            self.assertEqual(missing.decision_state, "NOT_REVIEWED")
            with self.assertRaisesRegex(ValueError, "not complete"):
                read_bad_channel_review_decision(_config(), "subA", "part2", root=root, require_complete=True)

    def test_participant_part_source_context_and_inventory_mismatches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, ["A"])
            kwargs = dict(
                root=root, expected_source_recording_id=SOURCE_ID,
                expected_review_context_id=CONTEXT_ID, expected_reviewed_channels=["A", "B", "C"],
                require_complete=True,
            )
            decision = read_bad_channel_review_decision(_config(), "subA", "part1", **kwargs)
            self.assertEqual(decision.confirmed_bad_channels, ("A",))
            with self.assertRaisesRegex(ValueError, "Stale bad-channel review"):
                read_bad_channel_review_decision(_config(), "subA", "part1", **{**kwargs, "expected_source_recording_id": "sha256:" + "3" * 64})
            with self.assertRaisesRegex(ValueError, "Stale bad-channel review context"):
                read_bad_channel_review_decision(_config(), "subA", "part1", **{**kwargs, "expected_review_context_id": "sha256:" + "4" * 64})
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                read_bad_channel_review_decision(_config(), "subA", "part1", **{**kwargs, "expected_reviewed_channels": ["A", "B"]})

            path = manual_review_paths(_config(), root=root)["bad_channels"]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            entry = payload["subA"]["part1"]
            for field, value, message in (
                ("participant", "subB", "participant mismatch"),
                ("recording_part", "part2", "part mismatch"),
            ):
                changed = dict(entry); changed[field] = value
                payload["subA"]["part1"] = changed
                path.write_text(yaml.safe_dump(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_bad_channel_review_decision(_config(), "subA", "part1", root=root)
                payload["subA"]["part1"] = entry

    def test_unknown_duplicate_and_tampered_decisions_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "absent from the reviewed"):
                self._write(root, ["X"])
            with self.assertRaisesRegex(ValueError, "Duplicate channel decisions"):
                self._write(root, ["A", "A"])
            valid = self._write(root, ["A"])
            path = manual_review_paths(_config(), root=root)["bad_channels"]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["subA"]["part1"]["confirmed_bad_channels"] = ["B"]
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                read_bad_channel_review_decision(_config(), "subA", "part1", root=root)
            self.assertIsNotNone(valid.decision_id)

    def test_unsupported_schema_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, ["A"])
            path = manual_review_paths(_config(), root=root)["bad_channels"]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["subA"]["part1"]["decision_schema_version"] = 999
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported bad-channel decision schema version"):
                read_bad_channel_review_decision(_config(), "subA", "part1", root=root)

    def test_legacy_empty_is_not_reviewed_and_legacy_nonempty_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = manual_review_paths(_config(), root=root)["bad_channels"]
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump({"subA": {"part1": {"confirmed_bad_channels": []}}}), encoding="utf-8")
            legacy = read_bad_channel_review_decision(_config(), "subA", "part1", root=root)
            self.assertTrue(legacy.legacy_ambiguous)
            self.assertEqual(legacy.decision_state, "NOT_REVIEWED")
            with self.assertRaisesRegex(ValueError, "not complete"):
                read_bad_channel_review_decision(_config(), "subA", "part1", root=root, require_complete=True)
            path.write_text(yaml.safe_dump({"subA": {"part1": {"confirmed_bad_channels": ["A"]}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be treated as authoritative"):
                read_bad_channel_review_decision(_config(), "subA", "part1", root=root)

    def test_repeated_read_is_idempotent_and_revised_write_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self._write(root, ["A", "B"])
            first = read_confirmed_bad_channels(_config(), "subA", "part1", root=root)
            second = read_confirmed_bad_channels(_config(), "subA", "part1", root=root)
            self.assertEqual(first, second)
            self._write(
                root, ["B", "C"], notes="revised synthetic decision",
                expected_current_semantic_decision_id=original.semantic_decision_id,
            )
            reloaded = read_confirmed_bad_channels(_config(), "subA", "part1", root=root)
            self.assertEqual(reloaded, ["B", "C"])
            self.assertNotIn("A", reloaded)

    def test_semantic_identity_is_stable_across_attribution_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write(root, ["A"], reviewer="reviewer-a", notes="first")
            second = self._write(
                root, ["A"], reviewer="reviewer-b",
                reviewed_at_utc="2026-09-12T13:00:00Z", notes="second",
            )
            self.assertEqual(first.semantic_decision_id, second.semantic_decision_id)
            self.assertNotEqual(first.audit_record_id, second.audit_record_id)
            changed = self._write(
                root, ["B"], expected_current_semantic_decision_id=second.semantic_decision_id
            )
            self.assertNotEqual(first.semantic_decision_id, changed.semantic_decision_id)

    def test_only_confirmed_channels_reach_interpolation_and_audit_trace(self) -> None:
        reviewed_zero = build_interpolation_table(
            ["A", "B", "C"], confirmed_bad_channels=[], automatic_bad_channel_proposals=["A", "C"],
            decision_state="REVIEWED_NO_BAD_CHANNELS", decision_id="sha256:zero", decision_file="manual/bad_channels.yaml",
        )
        self.assertFalse(reviewed_zero["interpolated"].any())
        applied = build_interpolation_table(
            ["A", "B", "C"], confirmed_bad_channels=["B"], automatic_bad_channel_proposals=["A"],
            decision_state="REVIEWED_WITH_BAD_CHANNELS", decision_id="sha256:one", decision_file="manual/bad_channels.yaml",
            runtime_provenance={"interpolated_targets": ["B"]},
        )
        self.assertEqual(applied.loc[applied["interpolated"], "channel"].tolist(), ["B"])
        self.assertFalse(bool(applied.loc[applied["channel"].eq("A"), "interpolated"].iloc[0]))
        self.assertEqual(set(applied["bad_channel_decision_id"]), {"sha256:one"})

    def test_legacy_config_override_is_not_authoritative(self) -> None:
        config = _config()
        config["manual_overrides"] = {"bad_channels": {"subA": {"part1": ["A"]}}}
        channel_qc = __import__("pandas").DataFrame([
            {"channel": "A", "proposed_bad_channel": True, "automatic_proposal_reasons": "synthetic"},
            {"channel": "B", "proposed_bad_channel": False, "automatic_proposal_reasons": ""},
        ])
        proposals_only = bad_channel_decision_table(channel_qc, config, "subA", "part1")
        self.assertFalse(proposals_only["manual_confirmed_bad_channel"].any())
        confirmed = bad_channel_decision_table(
            channel_qc, config, "subA", "part1", confirmed_bad_channels=["B"]
        )
        self.assertEqual(
            confirmed.loc[confirmed["manual_confirmed_bad_channel"], "channel"].tolist(), ["B"]
        )

    def test_source_and_context_identities_detect_material_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vhdr, vmrk, eeg = root / "x.vhdr", root / "x.vmrk", root / "x.eeg"
            vhdr.write_text("header", encoding="utf-8")
            vmrk.write_text("markers", encoding="utf-8")
            eeg.write_bytes(b"1234")
            recording = SimpleNamespace(
                subject="subA", recording_part="part1", file_base="x", vhdr_path=vhdr,
                vmrk_path=vmrk, eeg_path=eeg, sampling_rate_hz=500.0, channel_names=("A", "B", "C"),
            )
            source = bad_channel_source_identity(recording)
            self.assertEqual(source, bad_channel_source_identity(recording))
            context = bad_channel_review_context_identity(recording, _config(), ["A", "B", "C"])
            self.assertNotEqual(context, bad_channel_review_context_identity(recording, _config(), ["A", "B"]))
            vmrk.write_text("changed markers", encoding="utf-8")
            self.assertNotEqual(source, bad_channel_source_identity(recording))


if __name__ == "__main__":
    unittest.main()
