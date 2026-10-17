from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.reporting import generate_gate3a_markdown_report, generate_gate3c_epoch_qc_markdown_report  # noqa: E402


class TestGate3AReporting(unittest.TestCase):
    def test_markdown_report_can_be_generated_from_synthetic_qc_records(self) -> None:
        report = generate_gate3a_markdown_report(
            subject="sub999",
            recording_part="part1",
            input_files=["input.vhdr", "input.vmrk", "input.eeg"],
            output_files={"filtered_raw_fif": "filtered_raw.fif"},
            raw_qc_summary=pd.DataFrame([{"sfreq_hz": 500, "duration_s": 10, "n_samples": 5000, "n_channels": 4}]),
            channel_qc=pd.DataFrame([{"channel": "Cz"}]),
            bad_channel_decisions=pd.DataFrame([{"channel": "Cz", "automatic_bad_channel_proposal": True, "manual_confirmed_bad_channel": False}]),
            bad_channel_spatial_warnings=pd.DataFrame(
                [
                    {
                        "scope": "automatic_proposal",
                        "affected_channels": "Cz;Fz",
                        "affected_fraction": 0.5,
                        "roi_overlaps": "central:Cz;frontal:Fz",
                        "adjacent_channel_pairs": "Cz|Fz",
                        "edge_channels": "",
                        "geometry_omitted_channels": "T1;T2",
                        "warning_reasons": "spatial_cluster_manual_review_required",
                    }
                ]
            ),
            preprocessing_decisions=pd.DataFrame([{"decision": "notch_filter_applied", "value": False, "reason": "default"}]),
            warnings=["synthetic warning"],
            failures=[],
        )

        self.assertIn("# Gate 3A Raw QC: sub999 part1", report)
        self.assertIn("filtered_raw.fif", report)
        self.assertIn("synthetic warning", report)
        self.assertIn("Bad-Channel Spatial And ROI Review Warnings", report)
        self.assertIn("automatic exclusion=False", report)

    def test_gate3c_report_exposes_reference_reconstruction_and_geometry_safety(self) -> None:
        report = generate_gate3c_epoch_qc_markdown_report(
            subject="sub999",
            recording_part="part1",
            input_files=["filtered_raw.fif", "solution-ica.fif"],
            output_files={"epochs_fif": "epochs.fif"},
            ica_review_summary=pd.DataFrame([{"component": 0, "applied": True}]),
            reference_summary=pd.DataFrame(
                [
                    {
                        "reference_type": "average",
                        "original_reference_channel": "CPz",
                        "original_reference_evidence": "BrainVision acquisition evidence",
                        "cpz_reconstructed": True,
                        "cpz_reconstruction_stage": "gate3c_after_ica_application_before_average_reference",
                        "cpz_reconstruction_method": "mne.add_reference_channels(copy=False)",
                        "cpz_zero_valued_before_average_reference": True,
                        "eeg_channel_count_before_reference_reconstruction": 64,
                        "eeg_channel_count_after_reference_reconstruction": 65,
                        "final_eeg_reference": "average",
                        "average_reference_applied": True,
                        "average_reference_projection": False,
                        "reference_channels": ["Fz", "Cz", "CPz"],
                        "excluded_from_reference": ["LO1"],
                        "montage": "standard_1005",
                        "montage_status": "partial_with_unresolved_eeg",
                        "unresolved_coordinate_channels": ["T1", "T2"],
                        "interpolation_geometry_safety_status": "not_required",
                        "operation_order": ["ica_application", "cpz_reconstruction", "average_reference"],
                    }
                ]
            ),
            interpolation_summary=pd.DataFrame(
                [
                    {
                        "channel": "T2",
                        "interpolated": False,
                        "retained_bad_uninterpolated": True,
                        "bad_channel_interpolation_disposition": "retain_bad_uninterpolated_due_to_unresolved_geometry",
                        "bad_channel_interpolation_disposition_id": "sha256:synthetic",
                    }
                ]
            ),
            condition_counts=pd.DataFrame(),
            epoch_rejection_table=pd.DataFrame(
                [
                    {
                        "rejected": False,
                        "rejected_for_ptp": False,
                        "threshold_uv": 330.0,
                        "excluded_retained_bad_channels": "T2",
                    }
                ]
            ),
        )

        self.assertIn("CPz reconstructed: True", report)
        self.assertIn("EEG channel count before/after reconstruction: 64 / 65", report)
        self.assertIn("Unresolved coordinate channels: ['T1', 'T2']", report)
        self.assertIn("Geometry safety: not_required", report)
        self.assertIn("Interpolated channels: none", report)
        self.assertIn("Retained bad/uninterpolated channels: T2", report)
        self.assertIn("Retained-bad disposition: retain_bad_uninterpolated_due_to_unresolved_geometry", report)
        self.assertIn("Retained-bad disposition ID: sha256:synthetic", report)
        self.assertIn("Residual PTP excluded retained-bad channels: T2", report)
        self.assertNotIn("Only manually confirmed bad channels were interpolated", report)


if __name__ == "__main__":
    unittest.main()
