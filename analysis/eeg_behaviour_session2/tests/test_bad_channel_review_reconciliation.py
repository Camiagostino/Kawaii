from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.bad_channel_review_reconciliation import (  # noqa: E402
    build_reconciliation,
    reconciliation_summaries,
    split_reason_tokens,
    substantive_review_rows,
    validate_completed_review,
)


class TestBadChannelReviewReconciliation(unittest.TestCase):
    def setUp(self) -> None:
        self.public = pd.DataFrame({
            "case_id": ["BC001", "BC002"],
            "participant": ["sub005", "sub006"],
            "channel": ["A", "B"],
            "figure_path": ["a.png", "b.png"],
        })
        self.responses = pd.DataFrame({
            "case_id": ["BC001", "BC002", ""],
            "participant": ["sub005", "sub006", ""],
            "channel": ["A", "B", ""],
            "review_classification": ["NORMAL", "QUESTIONABLE", ""],
            "structured_reasons": ["otherwise_normal", "drift\nfocal_transient", ""],
            "reviewer_confidence": ["high", "medium", ""],
            "reviewer_notes": ["ordinary", "intermittent", ""],
        })

    def test_blank_trailing_rows_are_not_review_cases(self) -> None:
        substantive, blank_count = substantive_review_rows(self.responses)
        self.assertEqual(len(substantive), 2)
        self.assertEqual(blank_count, 1)

    def test_reason_parser_accepts_semicolons_and_newlines_verbatim(self) -> None:
        self.assertEqual(split_reason_tokens("drift; focal_transient\nline_noise"), (
            "drift", "focal_transient", "line_noise"
        ))

    def test_validation_accepts_complete_allowed_review(self) -> None:
        substantive, checks = validate_completed_review(self.responses, self.public)
        self.assertEqual(len(substantive), 2)
        self.assertTrue(checks["passed"].all())
        self.assertEqual(
            substantive.loc[substantive["case_id"].eq("BC002"), "structured_reason_tokens"].iloc[0],
            "drift;focal_transient",
        )

    def test_validation_flags_invalid_reason_without_rewriting_it(self) -> None:
        altered = self.responses.copy()
        altered.loc[1, "structured_reasons"] = "possibly drift"
        substantive, checks = validate_completed_review(altered, self.public)
        row = substantive[substantive["case_id"].eq("BC002")].iloc[0]
        self.assertEqual(row["structured_reasons"], "possibly drift")
        self.assertEqual(row["invalid_reason_tokens"], "possibly drift")
        check = checks[checks["check"].eq("structured_reasons_allowed")].iloc[0]
        self.assertFalse(bool(check["passed"]))

    def test_reconciliation_preserves_rating_and_joins_all_evidence(self) -> None:
        substantive, _ = validate_completed_review(self.responses.iloc[:2], self.public)
        key = self.public.assign(
            selection_stratum=["ordinary", "candidate"], edge_status=["edge", "interior"],
            automatic_candidate_status=["none", "local_abs=True"],
        )
        relative = pd.DataFrame({
            "participant": ["sub005", "sub006"], "channel": ["A", "B"],
            "absolute_0_40_two_windows": [False, True], "robust_z3_two_windows": [False, False],
            "mean_local_score": [0.8, 0.3], "minimum_local_score": [0.7, 0.2],
            "maximum_within_window_low_robust_z": [0.1, 2.0],
        })
        independent = pd.DataFrame({
            "participant": ["sub005", "sub006"], "channel": ["A", "B"],
            "flat_proposal": [False, False], "extreme_amplitude_proposal": [False, True],
            "clipping_proposal": [False, False], "robust_deviation_proposal": [False, False],
            "std_uv": [10.0, 600.0], "peak_to_peak_uv": [100.0, 6000.0],
            "max_robust_deviation_z": [0.0, 2.0], "clipping_fraction": [0.0, 0.0],
            "maximum_window_line_noise_ratio": [0.2, 0.3],
        })
        spatial = self.public[["case_id", "participant", "channel"]].assign(
            geometry_status=["SCORABLE", "GEOMETRY_UNSCORABLE"],
            median_spatial_prediction_correlation=[0.9, float("nan")],
            median_normalized_residual_rms=[0.2, float("nan")], failed_window_count=[0, float("nan")],
            spatial_primary_candidate=[False, False],
        )
        scores = pd.DataFrame({
            "participant": ["sub005"] * 3 + ["sub006"] * 3,
            "channel": ["A"] * 3 + ["B"] * 3,
            "method": ["baseline_raw_all_abs"] * 6,
            "window_index": [1, 2, 3, 1, 2, 3],
            "score": [0.8, 0.7, 0.9, 0.2, 0.3, 0.5],
        })
        result = build_reconciliation(
            substantive, held_back_key=key, relative=relative, independent=independent,
            spatial=spatial, expanded_scores=scores,
        )
        self.assertEqual(result.loc[1, "review_classification"], "QUESTIONABLE")
        self.assertTrue(bool(result.loc[1, "approved_independent_any"]))
        self.assertTrue(bool(result.loc[1, "production_candidate"]))
        self.assertEqual(result.loc[1, "geometry_aware_edge_status"], "GEOMETRY_UNSCORABLE")
        self.assertEqual(result.loc[0, "review_provenance"], "ASSISTED_BLINDED_HUMAN_REVIEW_NOT_INDEPENDENT_EXPERT_GROUND_TRUTH")

    def test_descriptive_summary_uses_ratings_as_strata_not_truth(self) -> None:
        frame = pd.DataFrame({
            "review_classification": ["NORMAL", "QUESTIONABLE"],
            "edge_status": ["edge", "interior"],
            "geometry_aware_edge_status": ["edge", "interior"],
            "production_candidate": [True, True],
            "absolute_0_40_two_windows": [False, True],
            "robust_z3_two_windows": [False, False],
            "approved_independent_any": [False, True],
            "spatial_primary_candidate": [False, True],
            "geometry_status": ["SCORABLE", "SCORABLE"],
            "case_id": ["BC001", "BC002"],
            "structured_reasons": ["otherwise_normal", "drift"],
            "selection_stratum": ["ordinary", "candidate"],
        })
        summaries = reconciliation_summaries(frame)
        normal_local = summaries["algorithm_by_rating"].query(
            "review_classification == 'NORMAL' and algorithm_evidence == 'local_absolute'"
        ).iloc[0]
        self.assertEqual(int(normal_local["flagged_case_count"]), 0)
        self.assertNotIn("sensitivity", summaries["algorithm_by_rating"].columns)


if __name__ == "__main__":
    unittest.main()
