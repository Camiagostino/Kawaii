from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.bad_channel_adjudication import (  # noqa: E402
    AVAILABLE_FROM_EXISTING_EPOCH_OUTPUT,
    NOT_RECONSTRUCTABLE_WITH_CURRENT_EVIDENCE,
    RECONSTRUCTABLE_OUTCOME_BLIND_FROM_EVENTS,
    abnormality_evidence,
    boundary_valid_event_samples,
    corroboration_category,
    evidence_families,
    image_event_samples_from_raw,
    local_handling_plausibility,
    mapping_provenance,
    review_disposition,
    task_segment_burden,
    temporal_extent,
    whole_channel_concern,
)
from eeg_software.workflows.kawaii_session2.study_definition import KawaiiTriggerDefinition  # noqa: E402


class TestBadChannelAdjudication(unittest.TestCase):
    def test_image_event_reconstruction_is_trigger_only_and_deterministic(self) -> None:
        raw = mne.io.RawArray(np.zeros((1, 1000)), mne.create_info(["Cz"], 100.0, "eeg"), verbose=False)
        raw.set_annotations(mne.Annotations([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], ["Stimulus/S 120", "Stimulus/S 151", "Stimulus/S 121", "Stimulus/S 320"]))
        np.testing.assert_array_equal(image_event_samples_from_raw(raw), [100, 300])
        np.testing.assert_array_equal(image_event_samples_from_raw(raw), [100, 300])
        alternative = KawaiiTriggerDefinition.from_config(
            {"triggers": {"image_onset": {"old": 320, "new": 321}}}
        )
        np.testing.assert_array_equal(
            image_event_samples_from_raw(raw, trigger_definition=alternative), [400]
        )

    def test_existing_mapping_reads_only_retained_event_locations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.csv"
            pd.DataFrame({
                "event_sample": [30, 10, 20],
                "retain_for_epoching": [True, True, False],
                "condition": ["new", "old", "new"],
            }).to_csv(path, index=False)
            from eeg_software.workflows.kawaii_session2.bad_channel_adjudication import existing_epoch_samples
            np.testing.assert_array_equal(existing_epoch_samples(path), [10, 30])

    def test_boundary_filter_and_provenance(self) -> None:
        valid = boundary_valid_event_samples(
            [10, 50, 990], raw_first_samp=0, n_times=1000, sfreq=100.0,
            epoch_tmin_s=-0.2, epoch_tmax_s=1.2,
        )
        np.testing.assert_array_equal(valid, [50])
        with tempfile.TemporaryDirectory() as directory:
            present = Path(directory) / "mapping.csv"
            present.touch()
            self.assertEqual(mapping_provenance(present, 0), AVAILABLE_FROM_EXISTING_EPOCH_OUTPUT)
            self.assertEqual(mapping_provenance(Path(directory) / "absent.csv", 3), RECONSTRUCTABLE_OUTCOME_BLIND_FROM_EVENTS)
            self.assertEqual(
                mapping_provenance(Path(directory) / "absent.csv", 0),
                NOT_RECONSTRUCTABLE_WITH_CURRENT_EVIDENCE,
            )

    def test_axis_categories_remain_separate(self) -> None:
        self.assertEqual(abnormality_evidence("CLEARLY_ABNORMAL"), "CLEAR")
        self.assertEqual(temporal_extent("RECURRENT"), "RECURRENT_SPARSE")
        self.assertEqual(task_segment_burden(0), "NEGLIGIBLE")
        self.assertEqual(task_segment_burden(0.049), "LOW")
        self.assertEqual(task_segment_burden(0.05), "MODERATE")
        self.assertEqual(task_segment_burden(0.20), "HIGH")
        self.assertEqual(whole_channel_concern("CLEAR", "PROLONGED"), "SUPPORTED")
        self.assertEqual(whole_channel_concern("WEAK_QUESTIONABLE", "RECURRENT_SPARSE"), "POSSIBLE")
        self.assertEqual(local_handling_plausibility("ISOLATED", "LOW", "NOT_SUPPORTED"), "YES")
        self.assertEqual(review_disposition("NONE", "NOT_SUPPORTED", "NOT_NEEDED"), "RETAIN_CHANNEL")
        self.assertEqual(
            review_disposition("CLEAR", "SUPPORTED", "NO"),
            "WHOLE_CHANNEL_INTERPOLATION_CONCERN",
        )

    def test_evidence_families_are_transparent_not_votes(self) -> None:
        case = {
            "prior_approved_independent_evidence": True,
            "prior_local_absolute_candidate": False,
            "prior_local_robust_relative_candidate": False,
            "prior_spatial_primary_candidate": True,
        }
        counts = {"drift_relative_flag": 2, "local_absolute_flag": 0}
        self.assertEqual(
            evidence_families(case, counts),
            ("TIME_DOMAIN_INDEPENDENT", "SPECTRAL_RELATIVE", "SPATIAL_PREDICTABILITY"),
        )
        self.assertEqual(corroboration_category(()), "NONE")
        self.assertEqual(corroboration_category(("LOCAL_CORRELATION",)), "ONE_EVIDENCE_TYPE")
        self.assertEqual(
            corroboration_category(("LOCAL_CORRELATION", "SPATIAL_PREDICTABILITY")),
            "MULTIPLE_EVIDENCE_TYPES",
        )


if __name__ == "__main__":
    unittest.main()
