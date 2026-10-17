from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.alignment import apply_duplicate_trial_policy  # noqa: E402


class TestDuplicateTrialPolicy(unittest.TestCase):
    def test_unresolved_duplicate_rows_remain_visible_and_excluded(self) -> None:
        alignment = pd.DataFrame(
            [
                {
                    "subject": "sub999",
                    "recording_part": "part1",
                    "trial": 42,
                    "marker_trial_present": True,
                    "metadata_matched": True,
                    "eeg_trial_complete": True,
                    "duplicate_trial_within_part": True,
                    "duplicate_trial_across_parts": False,
                },
                {
                    "subject": "sub999",
                    "recording_part": "part1",
                    "trial": 42,
                    "marker_trial_present": True,
                    "metadata_matched": True,
                    "eeg_trial_complete": True,
                    "duplicate_trial_within_part": True,
                    "duplicate_trial_across_parts": False,
                },
            ]
        )

        result = apply_duplicate_trial_policy(alignment)

        self.assertEqual(len(result), 2)
        self.assertTrue(result["is_duplicate_trial"].all())
        self.assertEqual(set(result["duplicate_group_id"]), {"sub999:trial42"})
        self.assertEqual(set(result["duplicate_resolution"]), {"unresolved"})
        self.assertFalse(result["retain_for_epoching"].any())
        self.assertEqual(set(result["exclusion_reason"]), {"unresolved_duplicate_trial"})

    def test_single_valid_duplicate_candidate_is_not_heuristically_retained(self) -> None:
        alignment = pd.DataFrame(
            [
                {
                    "subject": "sub999",
                    "recording_part": "part1",
                    "trial": 42,
                    "marker_trial_present": True,
                    "metadata_matched": True,
                    "eeg_trial_complete": True,
                    "duplicate_trial_within_part": True,
                    "duplicate_trial_across_parts": False,
                },
                {
                    "subject": "sub999",
                    "recording_part": "part1",
                    "trial": 42,
                    "marker_trial_present": True,
                    "metadata_matched": True,
                    "eeg_trial_complete": False,
                    "duplicate_trial_within_part": True,
                    "duplicate_trial_across_parts": False,
                },
            ]
        )

        result = apply_duplicate_trial_policy(alignment)

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["duplicate_resolution"]), {"unresolved"})
        self.assertFalse(result["retain_for_epoching"].any())
        self.assertEqual(set(result["exclusion_reason"]), {"unresolved_duplicate_trial"})


if __name__ == "__main__":
    unittest.main()
