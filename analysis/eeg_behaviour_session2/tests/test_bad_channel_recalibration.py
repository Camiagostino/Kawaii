from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.bad_channel_recalibration import (  # noqa: E402
    aggregate_failed_windows,
    baseline_all_channel_abs_scores,
    candidate_proposal_membership,
    candidate_proposal_sensitivity,
    classify_global_focal_windows,
    classify_validation_candidate,
    deterministic_delaunay_neighbours,
    discover_validation_candidate_ids,
    edge_membership_summary,
    median_correlation_scores,
    pairwise_method_overlaps,
    qc_only_filter_and_median_reference,
    within_window_low_robust_z,
)


class TestBadChannelRecalibration(unittest.TestCase):
    def test_global_focal_classifier_distinguishes_all_four_states(self) -> None:
        summaries = pd.DataFrame({
            "median": [0.7, 0.7, 0.2, 0.2],
            "channel_specific_z_ge_3_count": [0, 1, 0, 2],
        })

        result = classify_global_focal_windows(summaries)

        self.assertEqual(result["global_focal_class"].tolist(), ["none", "focal", "broad", "mixed"])

    def test_cohort_classification_preserves_pilots_and_explicit_exclusions(self) -> None:
        suitable = dict(
            raw_available=True, raw_readable=True, expected_channels_present=True, montage_usable=True,
            duration_sufficient=True, required_windows_available=True, behavioural_metadata_available=True,
            behavioural_metadata_required=False,
        )

        self.assertEqual(classify_validation_candidate("pilot03", explicitly_excluded=False, **suitable)[:2], ("D", False))
        self.assertEqual(classify_validation_candidate("sub008", explicitly_excluded=True, **suitable)[:2], ("D", False))
        self.assertEqual(classify_validation_candidate("sub009", explicitly_excluded=False, **suitable)[:2], ("A", True))

    def test_cohort_discovery_is_deterministic_and_collapses_recording_suffixes(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            log_root = root / "workspace" / "kawaii" / "session2" / "source" / "behavioral_logs"
            for name in ("sub010_session2", "pilot04_2_session2", "pilot04_session2", "junk"):
                (log_root / name).mkdir(parents=True)
            config = {"participants": {"defaults": {"included": ["sub005"], "excluded": ["pilot03"]}}, "analysis": {"exclude_subjects": ["sub008"]}}

            discovered = discover_validation_candidate_ids(root, config, documented_candidates=("sub014",))

        self.assertEqual(discovered, ["pilot03", "pilot04", "sub005", "sub008", "sub010", "sub014"])

    def test_edge_summary_uses_edge_and_low_degree_strata(self) -> None:
        membership = pd.DataFrame({
            "participant": ["sub999"] * 4, "channel": ["A", "B", "C", "D"], "method": ["local"] * 4,
            "threshold": [0.4] * 4, "minimum_failed_fraction": [2 / 3] * 4,
            "scorable": [True] * 4, "proposed": [True, False, False, False],
        })
        neighbours = pd.DataFrame({
            "participant": ["sub999"] * 4, "channel": ["A", "B", "C", "D"],
            "coordinate_valid": [True] * 4, "delaunay_edge_channel": [True, True, False, False],
            "neighbour_count": [2, 3, 4, 5],
        })

        merged, summary = edge_membership_summary(membership, neighbours)

        self.assertEqual(merged.loc[merged["channel"].eq("A"), "degree_group"].iloc[0], "low_degree")
        edge = summary[(summary["geometry_grouping"].eq("edge_status")) & (summary["geometry_stratum"].eq("edge"))].iloc[0]
        self.assertEqual((int(edge["proposed_count"]), int(edge["eligible_channel_count"])), (1, 2))

    def test_baseline_reproduces_absolute_matrix_diagonal_exclusion(self) -> None:
        samples = np.linspace(-1.0, 1.0, 101)
        data = np.vstack([samples, -samples, np.sin(np.linspace(0, 4 * np.pi, 101))])

        scores, counts = baseline_all_channel_abs_scores(data, ["A", "B", "C"])

        matrix = np.abs(np.corrcoef(data))
        np.fill_diagonal(matrix, np.nan)
        self.assertEqual(counts, {"A": 2, "B": 2, "C": 2})
        np.testing.assert_allclose([scores[name] for name in ("A", "B", "C")], np.nanmedian(matrix, axis=1))

    def test_local_signed_and_absolute_scores_treat_inversion_explicitly(self) -> None:
        samples = np.linspace(-1.0, 1.0, 101)
        data = np.vstack([samples, -samples])
        neighbours = {"A": ["B"], "B": ["A"]}

        absolute, _ = median_correlation_scores(data, ["A", "B"], neighbours=neighbours, absolute=True)
        signed, _ = median_correlation_scores(data, ["A", "B"], neighbours=neighbours, absolute=False)

        self.assertAlmostEqual(absolute["A"], 1.0)
        self.assertAlmostEqual(signed["A"], -1.0)

    def test_non_finite_and_constant_peers_do_not_create_finite_evidence(self) -> None:
        data = np.vstack([np.ones(20), np.arange(20, dtype=float), np.full(20, np.nan)])

        scores, counts = median_correlation_scores(data, ["flat", "ramp", "nan"])

        self.assertTrue(all(np.isnan(value) for value in scores.values()))
        self.assertEqual(counts, {"flat": 0, "ramp": 0, "nan": 0})

    def test_global_low_correlation_is_separate_from_channel_relative_outlier(self) -> None:
        rng = np.random.default_rng(42)
        common = np.sin(np.linspace(0, 12 * np.pi, 2000))
        coherent = np.vstack([common + rng.normal(scale=0.02, size=common.size) for _ in range(7)])
        one_bad = coherent.copy()
        one_bad[-1] = rng.normal(size=common.size)
        global_low = rng.normal(size=(7, common.size))

        one_bad_scores, _ = median_correlation_scores(one_bad, [f"C{i}" for i in range(7)])
        global_scores, _ = median_correlation_scores(global_low, [f"C{i}" for i in range(7)])
        one_bad_z = within_window_low_robust_z(one_bad_scores)
        global_z = within_window_low_robust_z(global_scores)

        self.assertGreater(one_bad_z["C6"], 3.0)
        self.assertEqual(sum(value >= 3.0 for value in one_bad_z.values()), 1)
        self.assertEqual(sum(value >= 3.0 for value in global_z.values()), 0)
        self.assertLess(np.median(list(global_scores.values())), 0.1)

    def test_persistence_rules_distinguish_one_two_and_three_failed_windows(self) -> None:
        values = np.asarray([[0.2, 0.8, 0.8], [0.2, 0.2, 0.8], [0.2, 0.2, 0.2], [np.nan, 0.8, 0.8]])

        counts, fractions, any_window = aggregate_failed_windows(values, threshold=0.4, minimum_failed_fraction=1 / 3)
        _, _, repeated = aggregate_failed_windows(values, threshold=0.4, minimum_failed_fraction=2 / 3)
        _, _, persistent = aggregate_failed_windows(values, threshold=0.4, minimum_failed_fraction=1.0)

        np.testing.assert_array_equal(counts, [1, 2, 3, 1])
        np.testing.assert_allclose(fractions, [1 / 3, 2 / 3, 1.0, 1 / 3])
        np.testing.assert_array_equal(any_window, [True, True, True, True])
        np.testing.assert_array_equal(repeated, [False, True, True, False])
        np.testing.assert_array_equal(persistent, [False, False, True, False])

    def test_qc_filter_then_median_reference_is_shape_preserving_and_robustly_centred(self) -> None:
        sfreq = 200.0
        time = np.arange(2000) / sfreq
        data = np.vstack([np.sin(2 * np.pi * 10 * time) + offset for offset in (0.0, 1.0, 10.0)])

        transformed = qc_only_filter_and_median_reference(data, sfreq)

        self.assertEqual(transformed.shape, data.shape)
        np.testing.assert_allclose(np.median(transformed, axis=0), 0.0, atol=1e-12)

    def test_delaunay_neighbours_are_deterministic_and_omit_invalid_coordinates(self) -> None:
        names = ["A", "B", "C", "D", "invalid"]
        xy = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        chs = []
        for index, (x, y) in enumerate(xy):
            loc = np.zeros(12)
            if names[index] != "invalid":
                loc[:3] = [x + 0.1, y + 0.1, 0.5]
            chs.append({"loc": loc})
        raw = SimpleNamespace(ch_names=names, info={"chs": chs})

        first = deterministic_delaunay_neighbours(raw, names)
        second = deterministic_delaunay_neighbours(raw, list(reversed(names)))

        self.assertEqual(first.neighbours, second.neighbours)
        self.assertEqual(first.omitted_channels, ("invalid",))
        self.assertEqual(first.neighbours["invalid"], ())
        self.assertTrue(all(len(first.neighbours[name]) >= 2 for name in names[:-1]))

    def test_insufficient_geometry_returns_no_invented_neighbours(self) -> None:
        chs = []
        for x in (0.1, 0.2):
            loc = np.zeros(12)
            loc[:3] = [x, 0.1, 0.5]
            chs.append({"loc": loc})
        raw = SimpleNamespace(ch_names=["A", "B"], info={"chs": chs})

        geometry = deterministic_delaunay_neighbours(raw, ["A", "B"])

        self.assertEqual(geometry.neighbours, {"A": (), "B": ()})

    def test_unscorable_geometry_is_excluded_from_candidate_denominator_and_overlap(self) -> None:
        rows = []
        for method, channel, values in (
            ("local", "valid", [0.2, 0.2, 0.8]),
            ("local", "invalid", [np.nan, np.nan, np.nan]),
            ("all", "valid", [0.2, 0.8, 0.8]),
            ("all", "invalid", [0.8, 0.8, 0.8]),
        ):
            for window, value in enumerate(values, start=1):
                rows.append({"participant": "sub999", "method": method, "channel": channel, "window_index": window, "score": value})
        scores = pd.DataFrame(rows)

        sensitivity = candidate_proposal_sensitivity(scores, thresholds=[0.4], failed_fractions=[2 / 3])
        local = sensitivity[sensitivity["method"].eq("local")].iloc[0]
        membership = candidate_proposal_membership(scores, threshold=0.4, failed_fractions=[1 / 3])
        overlaps = pairwise_method_overlaps(membership)

        self.assertEqual(int(local["eligible_channel_count"]), 1)
        invalid_local = membership[(membership["method"].eq("local")) & (membership["channel"].eq("invalid"))].iloc[0]
        self.assertFalse(bool(invalid_local["scorable"]))
        self.assertFalse(bool(invalid_local["proposed"]))
        self.assertEqual(len(overlaps), 1)


if __name__ == "__main__":
    unittest.main()
