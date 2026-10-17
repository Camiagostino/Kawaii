from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.epoching import build_event_epoch_mapping, retained_alignment_for_epoching  # noqa: E402
from eeg_software.workflows.kawaii_session2.task_crop import (  # noqa: E402
    apply_task_crop,
    calculate_task_crop_window,
    calculate_task_segments,
    image_onset_gap_summary_table,
    image_onset_timing_debug_table,
    segment_event_mapping_table,
    task_crop_debug_table,
    task_crop_summary,
)


def _config(*, before: float = 30.0, after: float = 30.0, ica: bool = True, epoching: bool = True, max_duration: float = 3600.0, fail_long: bool = True, long_gap: bool = True) -> dict:
    return {
        "preprocessing": {
            "task_crop": {
                "enabled_for_ica": ica,
                "enabled_for_epoching": epoching,
                "padding_before_s": before,
                "padding_after_s": after,
                "max_expected_duration_s": max_duration,
                "fail_if_crop_too_long": fail_long,
                "long_gap_handling": {"enabled": long_gap, "split_on_gap_s": 60, "keep_only_segments_with_retained_trials": True},
            }
        },
        "epochs": {"canonical": {"tmax_s": 1.2}},
    }


def _alignment(n_trials: int = 4) -> pd.DataFrame:
    rows = []
    for trial in range(1, n_trials + 1):
        sample = 1000 + (trial * 100)
        rows.append(
            {
                "subject": "sub006",
                "recording_part": "part1",
                "trial": trial,
                "image_onset_sample": sample,
                "image_onset_code": 120 if trial % 2 else 121,
                "kawaii_response_sample": sample + 20,
                "recognition_response_sample": sample + 50,
                "trial_end_sample": sample + 80,
                "retain_for_epoching": True,
                "exclusion_reason": "",
                "is_duplicate_trial": False,
                "duplicate_resolution": "not_duplicate",
                "eeg_trial_complete": True,
            }
        )
    return pd.DataFrame(rows)


class TestTaskCrop(unittest.TestCase):
    def test_task_window_uses_retained_image_onset_extents_not_row_order(self) -> None:
        alignment = _alignment(4).iloc[[3, 0, 2, 1]].reset_index(drop=True)

        window = calculate_task_crop_window(alignment, "part1", sfreq=100.0, n_times=10000, config=_config(before=2, after=3), stage="ica")

        self.assertEqual(window.first_retained_image_onset_sample, 1100)
        self.assertEqual(window.last_retained_image_onset_sample, 1400)
        self.assertEqual(window.crop_start_sample, 900)
        self.assertEqual(window.crop_end_sample, 1820)
        self.assertEqual(window.first_retained_trial, 1)
        self.assertEqual(window.last_retained_trial, 4)
        self.assertEqual(window.n_retained_trials, 4)

    def test_task_window_ignores_non_image_response_samples(self) -> None:
        alignment = _alignment(2)
        alignment.loc[alignment["trial"].eq(2), "recognition_response_sample"] = 9000
        alignment.loc[alignment["trial"].eq(2), "trial_end_sample"] = 9500

        window = calculate_task_crop_window(alignment, "part1", sfreq=100.0, n_times=10000, config=_config(before=0, after=0), stage="ica")

        self.assertEqual(window.first_retained_image_onset_sample, 1100)
        self.assertEqual(window.last_retained_image_onset_sample, 1200)
        self.assertEqual(window.crop_end_sample, 1320)
        self.assertLess(window.crop_end_sample, 9000)

    def test_task_crop_padding_is_clamped_to_recording_bounds(self) -> None:
        alignment = pd.DataFrame(
            [
                {
                    "recording_part": "part1",
                    "trial": 1,
                    "image_onset_sample": 20,
                    "image_onset_code": 120,
                    "trial_end_sample": 970,
                    "retain_for_epoching": True,
                }
            ]
        )

        window = calculate_task_crop_window(alignment, "part1", sfreq=100.0, n_times=1000, config=_config(before=30, after=30), stage="ica")

        self.assertEqual(window.crop_start_sample, 0)
        self.assertEqual(window.crop_end_sample, 999)

    def test_crop_seconds_are_relative_to_raw_first_sample(self) -> None:
        alignment = _alignment(1)

        window = calculate_task_crop_window(alignment, "part1", sfreq=100.0, n_times=5000, config=_config(before=1, after=1), stage="ica", first_samp=1000)

        self.assertEqual(window.original_first_sample, 1000)
        self.assertEqual(window.crop_start_sample, 1000)
        self.assertEqual(window.crop_start_s, 0.0)
        self.assertEqual(window.crop_end_sample, 1320)
        self.assertEqual(window.crop_end_s, 3.2)

    def test_sub006_like_long_recording_uses_short_ica_task_window(self) -> None:
        alignment = _alignment(400)

        window = calculate_task_crop_window(alignment, "part1", sfreq=500.0, n_times=3_000_000, config=_config(before=30, after=30), stage="ica")
        summary = task_crop_summary(window, subject="sub006", recording_part="part1", original_duration_s=(3_000_000 - 1) / 500.0)

        self.assertTrue(bool(summary.iloc[0]["task_crop_enabled"]))
        self.assertLess(float(summary.iloc[0]["cropped_duration_s"]), 200.0)
        self.assertGreater(float(summary.iloc[0]["original_duration_s"]), 5000.0)
        self.assertEqual(int(summary.iloc[0]["n_retained_trials_covered"]), 400)
        self.assertEqual(int(summary.iloc[0]["n_samples_after_crop"]), window.crop_end_sample - window.crop_start_sample + 1)
        self.assertIn("last_retained_image_onset_time_s", summary.columns)

    def test_crop_fails_when_duration_exceeds_configured_limit(self) -> None:
        alignment = _alignment(2)
        alignment.loc[alignment["trial"].eq(2), "image_onset_sample"] = 200000

        with self.assertRaisesRegex(ValueError, "TASK_CROP_TOO_LONG"):
            calculate_task_crop_window(alignment, "part1", sfreq=100.0, n_times=300000, config=_config(before=0, after=0, max_duration=60, long_gap=False), stage="ica")

    def test_image_onset_timing_debug_table_uses_eeg_marker_source(self) -> None:
        timing = image_onset_timing_debug_table(_alignment(3), "part1", sfreq=100.0)

        self.assertEqual(timing["event_source"].unique().tolist(), ["eeg_marker_120_121"])
        self.assertEqual(timing["event_code"].astype(int).tolist(), [120, 121, 120])
        self.assertEqual(timing["delta_from_previous_image_s"].dropna().round(3).tolist(), [1.0, 1.0])

    def test_gap_summary_counts_long_gaps(self) -> None:
        alignment = _alignment(3)
        alignment.loc[alignment["trial"].eq(2), "image_onset_sample"] = 10000
        alignment.loc[alignment["trial"].eq(3), "image_onset_sample"] = 50000
        timing = image_onset_timing_debug_table(alignment, "part1", sfreq=100.0)

        summary = image_onset_gap_summary_table(timing, matlab_duration_s=100.0).iloc[0]

        self.assertEqual(int(summary["n_gaps_gt_60s"]), 2)
        self.assertEqual(int(summary["n_gaps_gt_300s"]), 1)
        self.assertIn("1", str(summary["trials_surrounding_long_gaps"]))

    def test_task_segments_split_on_long_gaps_and_exclude_idle_span(self) -> None:
        alignment = _alignment(3)
        alignment.loc[alignment["trial"].eq(2), "image_onset_sample"] = 10000
        alignment.loc[alignment["trial"].eq(3), "image_onset_sample"] = 50000

        segments = calculate_task_segments(alignment, "part1", sfreq=100.0, n_times=60000, config=_config(before=1, after=1), first_samp=0)

        self.assertEqual(len(segments), 3)
        self.assertLess(sum(segment.duration_s for segment in segments), 10.0)
        self.assertEqual([segment.start_trial for segment in segments], [1, 2, 3])

    def test_segment_event_mapping_keeps_all_retained_trials_epochable(self) -> None:
        alignment = _alignment(3)
        alignment.loc[alignment["trial"].eq(2), "image_onset_sample"] = 10000
        alignment.loc[alignment["trial"].eq(3), "image_onset_sample"] = 50000
        segments = calculate_task_segments(alignment, "part1", sfreq=100.0, n_times=60000, config=_config(before=1, after=1), first_samp=0)

        mapping = segment_event_mapping_table(alignment, "part1", segments, sfreq=100.0)

        self.assertEqual(mapping["trial"].astype(int).tolist(), [1, 2, 3])
        self.assertEqual(mapping["event_sample_original"].astype(int).tolist(), [1100, 10000, 50000])
        self.assertTrue(mapping["event_sample_segmented"].is_monotonic_increasing)

    def test_crop_detection_fails_without_retained_event_information(self) -> None:
        with self.assertRaisesRegex(ValueError, "no retained 120/121 image-onset alignment rows"):
            calculate_task_crop_window(_alignment(2).assign(retain_for_epoching=False), "part1", sfreq=500.0, n_times=10000, config=_config(), stage="ica")

        with self.assertRaisesRegex(ValueError, "missing image_onset_sample"):
            calculate_task_crop_window(_alignment(2).drop(columns=["image_onset_sample"]), "part1", sfreq=500.0, n_times=10000, config=_config(), stage="epoching")

    def test_string_false_rows_are_not_retained_for_crop(self) -> None:
        alignment = pd.concat([_alignment(1), _alignment(1).assign(trial=2, image_onset_sample=9000, retain_for_epoching="False")], ignore_index=True)

        window = calculate_task_crop_window(alignment, "part1", sfreq=100.0, n_times=10000, config=_config(before=0, after=0), stage="ica")

        self.assertEqual(window.first_retained_image_onset_sample, 1100)
        self.assertLess(window.crop_end_sample, 9000)

    def test_task_crop_debug_table_reports_original_and_after_crop_coordinates(self) -> None:
        alignment = _alignment(2)
        window = calculate_task_crop_window(alignment, "part1", sfreq=100.0, n_times=10000, config=_config(before=1, after=1), stage="ica")

        debug = task_crop_debug_table(alignment, "part1", window)

        self.assertEqual(debug["trial"].astype(int).tolist(), [1, 2])
        self.assertEqual(debug["event_code"].astype(int).tolist(), [120, 121])
        self.assertEqual(debug["event_sample_original"].astype(int).tolist(), [1100, 1200])
        self.assertEqual(debug["event_sample_after_crop"].astype(int).tolist(), [100, 200])
        self.assertTrue(debug["included_in_crop"].astype(bool).all())

    def test_cropped_raw_preserves_original_event_sample_coordinates(self) -> None:
        try:
            import mne
            import numpy as np
        except Exception as error:  # pragma: no cover - depends on local test environment
            self.skipTest(f"MNE is unavailable: {error}")

        raw = mne.io.RawArray(np.zeros((1, 10000)), mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"]), verbose=False)
        alignment = _alignment(2)
        window = calculate_task_crop_window(alignment, "part1", sfreq=100.0, n_times=raw.n_times, config=_config(before=1, after=1), stage="epoching")

        cropped = apply_task_crop(raw, window)
        retained = retained_alignment_for_epoching(alignment, "part1")
        mapping = build_event_epoch_mapping(pd.DataFrame(columns=["event_sample", "event_code"]), retained)

        self.assertEqual(cropped.first_samp, window.crop_start_sample)
        self.assertEqual(cropped.last_samp, window.crop_end_sample)
        self.assertEqual(mapping["event_sample"].astype(int).tolist(), [1100, 1200])
        self.assertTrue(all(cropped.first_samp <= sample <= cropped.last_samp for sample in mapping["event_sample"].astype(int)))

    def test_complete_participant_mapping_keeps_all_400_retained_trials(self) -> None:
        alignment = _alignment(400)
        retained = retained_alignment_for_epoching(alignment, "part1")

        mapping = build_event_epoch_mapping(pd.DataFrame(columns=["event_sample", "event_code"]), retained)

        self.assertEqual(len(mapping), 400)
        self.assertEqual(mapping["trial"].astype(int).tolist(), list(range(1, 401)))
        self.assertEqual(mapping["mapping_status"].unique().tolist(), ["from_retained_alignment_no_observed_events"])


if __name__ == "__main__":
    unittest.main()
