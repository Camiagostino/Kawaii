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

from eeg_software.workflows.kawaii_session2.bad_channel_edge_evidence import (  # noqa: E402
    add_participant_robust_scores,
    geometry_feature_table,
    persistent_percentile_outliers,
)


class TestBadChannelEdgeEvidence(unittest.TestCase):
    def test_geometry_features_capture_one_sided_edge_coverage(self) -> None:
        coordinates = {"C": (0, 0), "L": (-1, 0), "R": (1, 0), "U": (0, 1), "E": (2, 0)}
        neighbours = {"C": ("L", "R", "U"), "E": ("R", "U")}
        table = geometry_feature_table(coordinates, neighbours, edge_channels=["E"])
        centre = table[table["channel"].eq("C")].iloc[0]
        edge = table[table["channel"].eq("E")].iloc[0]
        self.assertFalse(bool(centre["delaunay_edge_channel"]))
        self.assertTrue(bool(edge["delaunay_edge_channel"]))
        self.assertGreater(float(edge["maximum_angular_gap_degrees"]), 180.0)
        self.assertGreater(float(edge["directional_asymmetry"]), float(centre["directional_asymmetry"]))

    def test_participant_robust_scores_are_within_participant(self) -> None:
        table = pd.DataFrame({"participant": ["a"] * 5 + ["b"] * 5, "metric": [1, 2, 3, 4, 20, 101, 102, 103, 104, 105]})
        scored = add_participant_robust_scores(table, ["metric"])
        self.assertAlmostEqual(float(scored.loc[2, "metric_participant_robust_z"]), 0.0)
        self.assertAlmostEqual(float(scored.loc[7, "metric_participant_robust_z"]), 0.0)
        self.assertGreater(float(scored.loc[4, "metric_participant_robust_z"]), 5.0)

    def test_percentile_persistence_is_window_relative_and_deterministic(self) -> None:
        rows = []
        for window, values in enumerate(([0.1, 0.8, 0.9], [0.2, 0.7, 0.8], [0.9, 0.8, 0.7]), start=1):
            for channel, value in zip(("A", "B", "C"), values):
                rows.append({"participant": "sub", "window_index": window, "channel": channel, "score": value})
        result = persistent_percentile_outliers(pd.DataFrame(rows), percentile=34, minimum_windows=2)
        a = result[result["channel"].eq("A")].iloc[0]
        self.assertEqual(int(a["failed_window_count"]), 2)
        self.assertTrue(bool(a["persistent_percentile_outlier"]))
        with self.assertRaises(ValueError):
            persistent_percentile_outliers(pd.DataFrame(rows), percentile=0)


if __name__ == "__main__":
    unittest.main()
