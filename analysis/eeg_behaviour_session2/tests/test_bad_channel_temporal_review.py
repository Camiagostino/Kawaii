from __future__ import annotations

import hashlib
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

from eeg_software.workflows.kawaii_session2.bad_channel_temporal_review import (  # noqa: E402
    METRIC_FLAGS,
    burden_summary,
    epoch_burden,
    fixed_temporal_windows,
    plot_temporal_case,
    select_review_windows,
    temporal_category,
    temporal_window_metrics,
)


class TestBadChannelTemporalReview(unittest.TestCase):
    def test_fixed_windows_cover_recording_and_mark_short_final_window(self) -> None:
        windows = fixed_temporal_windows(2600, 100.0, window_length_s=10.0)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows["start_sample"].tolist(), [0, 1000, 2000])
        self.assertEqual(windows["stop_sample"].tolist(), [1000, 2000, 2600])
        self.assertTrue(windows["valid_window"].all())
        short = fixed_temporal_windows(2200, 100.0, window_length_s=10.0)
        self.assertFalse(bool(short.iloc[-1]["valid_window"]))
        self.assertEqual(float(short.iloc[-1]["window_duration_s"]), 2.0)

    def test_metrics_are_deterministic_and_target_scoped(self) -> None:
        rng = np.random.default_rng(9)
        data = rng.normal(scale=1e-5, size=(5, 2500))
        channels = ["A", "B", "C", "D", "E"]
        neighbours = {name: tuple(other for other in channels if other != name) for name in channels}
        first = temporal_window_metrics(
            data, channels=channels, case_channels=["A", "C"], neighbours=neighbours,
            sfreq=250.0, window_index=1,
        )
        second = temporal_window_metrics(
            data, channels=channels, case_channels=["A", "C"], neighbours=neighbours,
            sfreq=250.0, window_index=1,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["channel"].tolist(), ["A", "C"])
        self.assertTrue(set(METRIC_FLAGS).issubset(first.columns))

    def test_flat_run_and_extreme_scale_are_transparently_flagged(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.normal(scale=1e-5, size=(5, 2500))
        data[0] *= 100
        data[0, 1000:1500] = 0.0
        channels = ["A", "B", "C", "D", "E"]
        neighbours = {name: tuple(other for other in channels if other != name) for name in channels}
        result = temporal_window_metrics(
            data, channels=channels, case_channels=["A"], neighbours=neighbours,
            sfreq=250.0, window_index=1,
        ).iloc[0]
        self.assertTrue(bool(result["flat_dropout_flag"]))
        self.assertTrue(bool(result["scale_dispersion_flag"]))
        self.assertTrue(bool(result["consensus_affected_window"]))

    def test_burden_and_categories_use_contiguous_windows(self) -> None:
        windows = pd.DataFrame({
            "window_index": [1, 2, 3, 4, 5],
            "window_start_s": [0, 10, 20, 30, 40],
            "window_end_s": [10, 20, 30, 40, 50],
            "valid_window": [True] * 5,
            "flag": [False, True, True, True, False],
        })
        summary = burden_summary(windows, flag_column="flag")
        self.assertEqual(summary["affected_window_count"], 3)
        self.assertEqual(summary["longest_contiguous_affected_run_windows"], 3)
        self.assertEqual(summary["longest_contiguous_affected_run_s"], 30.0)
        self.assertEqual(temporal_category(summary), "BROADLY_PERSISTENT")
        windows["flag"] = [True, False, True, False, True]
        self.assertEqual(temporal_category(burden_summary(windows, flag_column="flag")), "BROADLY_PERSISTENT")
        windows["flag"] = [True, False, False, False, False]
        self.assertEqual(temporal_category(burden_summary(windows, flag_column="flag")), "LOCAL_TRANSIENT")

        longer = pd.concat([windows, windows.assign(
            window_index=windows["window_index"] + 5,
            window_start_s=windows["window_start_s"] + 50,
            window_end_s=windows["window_end_s"] + 50,
        )], ignore_index=True)
        longer["flag"] = [True, False, True, False, True, False, False, False, False, False]
        self.assertEqual(temporal_category(burden_summary(longer, flag_column="flag")), "RECURRENT")
        longer["flag"] = [False, True, True, True, False, False, False, False, False, False]
        self.assertEqual(temporal_category(burden_summary(longer, flag_column="flag")), "PROLONGED")
        longer["flag"] = [True, False, False, False, False, False, False, False, False, True]
        self.assertEqual(temporal_category(burden_summary(longer, flag_column="flag")), "INTERMITTENT")
        longer["flag"] = False
        self.assertEqual(temporal_category(burden_summary(longer, flag_column="flag")), "INSUFFICIENT_EVIDENCE")

    def test_epoch_mapping_quantifies_overlap_without_rejection(self) -> None:
        windows = pd.DataFrame({
            "window_start_s": [0.0, 10.0], "window_end_s": [10.0, 20.0],
            "valid_window": [True, True], "consensus_affected_window": [False, True],
        })
        signal = np.zeros(2500)
        signal[1200:1210] = 500e-6
        result = epoch_burden(
            [500, 1200], sfreq=100.0, raw_first_samp=0, epoch_tmin_s=-0.2,
            epoch_tmax_s=1.2, affected_windows=windows, target_signal=signal,
        )
        self.assertEqual(result["eligible_epoch_count"], 2)
        self.assertEqual(result["consensus_window_overlap_epoch_count"], 1)
        self.assertEqual(result["target_ptp_ge_330_epoch_count"], 1)
        self.assertNotIn("rejected", result)

    def _figure_frame(self) -> pd.DataFrame:
        rows = []
        for index in range(1, 7):
            row = {
                "window_index": index, "window_start_s": (index - 1) * 10.0,
                "window_end_s": index * 10.0, "start_sample": (index - 1) * 1000,
                "stop_sample": index * 1000, "valid_window": True,
                "scale_robust_z": float(index), "line_noise_robust_z": 0.0,
                "high_frequency_robust_z": 0.0, "drift_robust_z": 0.0,
                "local_low_robust_z": 0.0, "affected_evidence_family_count": 0,
                "consensus_affected_window": index in {2, 5},
            }
            for flag in METRIC_FLAGS:
                row[flag] = flag == "scale_dispersion_flag" and index in {2, 5}
            rows.append(row)
        return pd.DataFrame(rows)

    def test_window_selection_and_figure_are_deterministic(self) -> None:
        frame = self._figure_frame()
        first = select_review_windows(frame)
        second = select_review_windows(frame)
        pd.testing.assert_frame_equal(first, second)
        rng = np.random.default_rng(2)
        raw_windows = {int(row.window_index): rng.normal(size=1000) * 1e-5 for row in first.itertuples()}
        with tempfile.TemporaryDirectory() as directory:
            one = Path(directory) / "one.png"
            two = Path(directory) / "two.png"
            kwargs = dict(
                case_windows=frame, selected=first, raw_windows=raw_windows, sfreq=100.0,
                case_id="BC001", participant="sub005", channel="A",
            )
            plot_temporal_case(output_path=one, **kwargs)
            plot_temporal_case(output_path=two, **kwargs)
            self.assertEqual(hashlib.sha256(one.read_bytes()).hexdigest(), hashlib.sha256(two.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
