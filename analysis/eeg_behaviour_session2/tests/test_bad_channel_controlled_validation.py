from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.bad_channel_controlled_validation import (  # noqa: E402
    CANDIDATE_METHOD_COLUMNS,
    aggregate_performance,
    build_blinded_review_tables,
    inject_focal_corruption,
    inject_global_perturbation,
    scenario_performance,
)


class TestBadChannelControlledValidation(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.data = rng.normal(scale=1e-5, size=(4, 2000))

    def test_focal_corruptions_are_deterministic_and_change_only_target(self) -> None:
        for kind in ("broadband_high_frequency_noise", "line_noise_50_hz", "amplitude_scaling", "partial_flatness", "slow_drift_0_2_hz"):
            first = inject_focal_corruption(self.data, target_index=1, corruption_type=kind, severity="moderate", sfreq=500, seed=11)
            second = inject_focal_corruption(self.data, target_index=1, corruption_type=kind, severity="moderate", sfreq=500, seed=11)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first[[0, 2, 3]], self.data[[0, 2, 3]])
            self.assertFalse(np.array_equal(first[1], self.data[1]))

    def test_global_controls_change_all_channels_without_focal_identity(self) -> None:
        for kind in ("shared_broadband_noise", "montage_wide_independent_broadband_noise", "shared_line_noise_50_hz", "common_amplitude_scaling"):
            changed = inject_global_perturbation(self.data, perturbation_type=kind, sfreq=500, seed=13)
            self.assertEqual(changed.shape, self.data.shape)
            self.assertTrue(np.all(np.any(changed != self.data, axis=1)))

    def test_performance_counts_target_and_false_positives(self) -> None:
        evidence = pd.DataFrame({"channel": ["A", "B", "C"], **{
            method: [True, True, False] for method in CANDIDATE_METHOD_COLUMNS
        }})
        result = scenario_performance(evidence, truth_type="focal", target_channel="A", neighbours={"A": ("B",)})
        self.assertTrue(result["target_detected"].all())
        self.assertTrue(np.allclose(result["false_positive_rate"], 0.5))
        self.assertTrue(result["localization_class"].eq("B_target_plus_neighbours").all())

    def test_global_control_performance_has_false_positive_rate_but_no_sensitivity(self) -> None:
        evidence = pd.DataFrame({"channel": ["A", "B", "C"], **{
            method: [True, False, False] for method in CANDIDATE_METHOD_COLUMNS
        }})
        result = scenario_performance(evidence, truth_type="global", target_channel=None, neighbours={})
        self.assertTrue(result["sensitivity"].isna().all())
        self.assertTrue(np.allclose(result["false_positive_rate"], 1 / 3))
        self.assertTrue(result["localization_class"].eq("not_applicable_no_focal_target").all())

    def test_edge_interior_aggregation_remains_stratified(self) -> None:
        rows = []
        for edge, detected in (("edge", False), ("interior", True)):
            for method in CANDIDATE_METHOD_COLUMNS:
                rows.append({
                    "target_edge_status": edge, "method": method, "sensitivity": float(detected),
                    "false_positive_rate": 0.0, "candidate_count": int(detected), "specificity": 1.0,
                    "precision": float(detected), "balanced_accuracy": 0.5 + 0.5 * float(detected),
                    "incremental_target_detection": bool(detected),
                })
        result = aggregate_performance(pd.DataFrame(rows), ["target_edge_status"])
        edge = result[(result["target_edge_status"].eq("edge")) & result["method"].eq(CANDIDATE_METHOD_COLUMNS[0])].iloc[0]
        interior = result[(result["target_edge_status"].eq("interior")) & result["method"].eq(CANDIDATE_METHOD_COLUMNS[0])].iloc[0]
        self.assertEqual(float(edge["sensitivity"]), 0.0)
        self.assertEqual(float(interior["sensitivity"]), 1.0)

    def test_blinded_manifest_hides_algorithmic_fields_and_is_seeded(self) -> None:
        selection = pd.DataFrame({
            "participant": ["sub005", "sub006"], "channel": ["T9", "Cz"],
            "selection_stratum": ["candidate", "ordinary"], "edge_status": [True, False],
            "automatic_candidate_status": [True, False], "figure_path": ["a.png", "b.png"],
        })
        public1, responses1, key1 = build_blinded_review_tables(selection, seed=4)
        public2, _, _ = build_blinded_review_tables(selection, seed=4)
        pd.testing.assert_frame_equal(public1, public2)
        self.assertNotIn("edge_status", public1)
        self.assertNotIn("automatic_candidate_status", public1)
        self.assertTrue(responses1["review_classification"].eq("").all())
        self.assertIn("selection_stratum", key1)


if __name__ == "__main__":
    unittest.main()
