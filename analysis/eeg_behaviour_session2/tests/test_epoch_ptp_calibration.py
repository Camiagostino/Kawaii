from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.epoch_ptp_calibration import (  # noqa: E402
    candidate_thresholds,
    distribution_summary,
    max_eeg_ptp_uv,
    participant_inventory,
    threshold_impacts,
)


class TestEpochPTPCalibration(unittest.TestCase):
    def test_max_eeg_ptp_uses_maximum_channel_and_converts_to_uv(self) -> None:
        data = np.array(
            [
                [[-1e-6, 1e-6, 0.0], [-4e-6, 3e-6, 1e-6]],
                [[-2e-6, 2e-6, 0.0], [-1e-6, 1e-6, 0.0]],
            ]
        )

        values, channels = max_eeg_ptp_uv(data, ["Fz", "Cz"])

        np.testing.assert_allclose(values, [7.0, 4.0])
        self.assertEqual(channels, ["Cz", "Fz"])

    def test_max_eeg_ptp_rejects_nonfinite_values_before_calculation(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            data = np.zeros((1, 2, 3), dtype=float)
            data[0, 1, 1] = value
            with self.assertRaisesRegex(ValueError, "must be finite"):
                max_eeg_ptp_uv(data, ["Fz", "Cz"])

    def test_distribution_summary_reports_requested_quantiles_and_mad(self) -> None:
        summary = distribution_summary([1.0, 2.0, 3.0, 4.0, 10.0])

        self.assertEqual(summary["n_epochs"], 5)
        self.assertEqual(summary["median_uv"], 3.0)
        self.assertEqual(summary["mad_uv"], 1.0)
        self.assertEqual(summary["maximum_uv"], 10.0)
        self.assertIn("p97_5_uv", summary)

    def test_candidate_impacts_are_epoch_specific_and_reconcile(self) -> None:
        metrics = pd.DataFrame(
            {
                "participant": ["sub005", "sub005", "sub006", "sub006"],
                "ptp_max_uv": [10.0, 20.0, 30.0, 100.0],
            }
        )
        candidates = pd.DataFrame(
            [{"candidate": "test", "threshold_uv": 50.0, "derivation": "synthetic"}]
        )

        pooled, participants = threshold_impacts(metrics, candidates)

        self.assertEqual(int(pooled.iloc[0]["epochs_above_threshold"]), 1)
        self.assertEqual(int(pooled.iloc[0]["epochs_retained"]), 3)
        self.assertEqual(int(participants["epochs_above_threshold"].sum()), 1)
        self.assertEqual(int(participants["epochs_retained"].sum()), 3)

    def test_candidates_are_distribution_derived_without_final_selection(self) -> None:
        candidates = candidate_thresholds(np.arange(1.0, 101.0))

        self.assertEqual(
            set(candidates["candidate"]),
            {"pooled_p97_5", "pooled_p99", "pooled_median_plus_6_robust_sigma", "reference_330_uv"},
        )
        self.assertEqual(float(candidates.loc[candidates["candidate"].eq("reference_330_uv"), "threshold_uv"].iloc[0]), 330.0)
        self.assertNotIn("selected", candidates.columns)

    def test_inventory_groups_pilot_files_and_suffixed_logs_under_canonical_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "workspace" / "kawaii" / "session2" / "source" / "eeg"
            raw_dir.mkdir(parents=True)
            (raw_dir / "kawaii0003.vhdr").touch()
            log_dir = root / "workspace" / "kawaii" / "session2" / "source" / "behavioral_logs" / "pilot00_debug_session2"
            log_dir.mkdir(parents=True)
            (log_dir / "pilot00_debug_session2_trial_log_analysis.csv").touch()
            (log_dir / "pilot00_debug_session2_trigger_log.csv").touch()
            config = {
                "participants": {"defaults": {"excluded": ["pilot03"]}},
                "recordings": {"participant_recordings": {}},
                "analysis": {},
            }

            inventory = participant_inventory(root, config).set_index("participant")

            self.assertIn("pilot00", inventory.index)
            self.assertEqual(inventory.loc["pilot00", "behaviour_event_data_present"], "YES")
            self.assertIn("pilot03", inventory.index)
            self.assertNotIn("sub003", inventory.index)


if __name__ == "__main__":
    unittest.main()
