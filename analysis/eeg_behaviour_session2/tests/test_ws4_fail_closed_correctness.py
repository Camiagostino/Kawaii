from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.alignment import (
    merge_marker_trials_with_metadata,
    reconstruct_marker_trials,
)
from eeg_software.workflows.kawaii_session2.epoching import (
    build_event_epoch_mapping,
    merge_residual_qc_with_rejection_table,
    retained_alignment_for_epoching,
)
from eeg_software.workflows.kawaii_session2.metadata import _normalise_log_metadata


def test_valid_unique_metadata_is_preserved_exactly() -> None:
    source = pd.DataFrame(
        [
            {"trial": 1, "block": 1, "image_filename": "a.jpg"},
            {"trial": 2, "block": 1, "image_filename": "b.jpg"},
        ]
    )

    result = _normalise_log_metadata(source, "sub001", "part1", "matlab_log")

    assert result["trial"].tolist() == [1, 2]
    assert result["image_filename"].tolist() == ["a.jpg", "b.jpg"]


@pytest.mark.parametrize("trial", [None, "bad", 1.5, float("nan"), float("inf"), float("-inf")])
def test_malformed_trial_id_cannot_disappear_silently(trial: object) -> None:
    source = pd.DataFrame([{"trial": 1}, {"trial": trial}])

    with pytest.raises(ValueError, match="missing or malformed trial IDs"):
        _normalise_log_metadata(source, "sub001", "part1", "matlab_log")


def test_duplicate_metadata_key_is_rejected_instead_of_first_row_winning() -> None:
    source = pd.DataFrame(
        [
            {"trial": 1, "image_filename": "first.jpg"},
            {"trial": 1, "image_filename": "conflict.jpg"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate recording-part/trial keys"):
        _normalise_log_metadata(source, "sub001", "part1", "matlab_log")


def test_direct_alignment_merge_rejects_duplicate_metadata_key() -> None:
    markers = pd.DataFrame(
        [{"recording_part": "part1", "trial": 1, "marker_trial_sequence": 1}]
    )
    metadata = pd.DataFrame(
        [
            {"recording_part": "part1", "trial": 1, "metadata_part": "sub001"},
            {"recording_part": "part1", "trial": 1, "metadata_part": "sub001"},
        ]
    )

    with pytest.raises(ValueError, match="alignment is ambiguous"):
        merge_marker_trials_with_metadata(
            markers,
            metadata,
            metadata_source="matlab_log",
            behavioural_log_available=True,
        )


def test_valid_unique_alignment_merge_is_unchanged() -> None:
    markers = pd.DataFrame(
        [{"recording_part": "part1", "trial": 1, "marker_trial_sequence": 1, "eeg_trial_complete": True}]
    )
    metadata = pd.DataFrame(
        [{"recording_part": "part1", "trial": 1, "metadata_part": "sub001", "image_filename": "a.jpg"}]
    )

    result = merge_marker_trials_with_metadata(
        markers,
        metadata,
        metadata_source="matlab_log",
        behavioural_log_available=True,
    )

    assert len(result) == 1
    assert result.iloc[0]["alignment_status"] == "ALIGNED"
    assert result.iloc[0]["image_filename"] == "a.jpg"


def test_recognised_marker_with_malformed_sample_is_not_analysis_ready() -> None:
    markers = pd.DataFrame(
        [
            _marker(1, 100, 120, "image_onset"),
            _marker(2, 110, 150, "recognition_response"),
            _marker(3, 120, 100, "kawaii_question"),
            _marker(4, 130, 11, "kawaii_response"),
            _marker(5, 140, 200, "arousal_question"),
            {**_marker(6, 150, 21, "arousal_response"), "sample": pd.NA},
        ]
    )

    trials, issues = reconstruct_marker_trials(
        SimpleNamespace(subject="sub001", recording_part="part1"), markers
    )

    assert not bool(trials.iloc[0]["eeg_trial_complete"])
    assert "malformed_arousal_response_sample" in trials.iloc[0]["sequence_warning"]
    assert "malformed_arousal_response_sample" in issues.iloc[0]["message"]


def test_multiple_recognised_markers_make_trial_incomplete_and_visible() -> None:
    markers = pd.DataFrame(
        [
            _marker(1, 100, 120, "image_onset"),
            _marker(2, 110, 150, "recognition_response"),
            _marker(3, 120, 100, "kawaii_question"),
            _marker(4, 121, 100, "kawaii_question"),
            _marker(5, 130, 11, "kawaii_response"),
            _marker(6, 140, 200, "arousal_question"),
            _marker(7, 150, 21, "arousal_response"),
        ]
    )

    trials, issues = reconstruct_marker_trials(
        SimpleNamespace(subject="sub001", recording_part="part1"), markers
    )

    assert len(trials) == 1
    assert not bool(trials.iloc[0]["eeg_trial_complete"])
    assert "ambiguous_kawaii_question" in trials.iloc[0]["sequence_warning"]
    assert "ambiguous_kawaii_question" in issues.iloc[0]["message"]


def test_duplicate_trial_and_event_keys_fail_before_epoch_mapping() -> None:
    alignment = pd.DataFrame(
        [
            _retained_alignment_row(1, 100, 120),
            _retained_alignment_row(1, 200, 121),
        ]
    )
    with pytest.raises(ValueError, match="Retained alignment.*duplicate keys"):
        retained_alignment_for_epoching(alignment, "part1")

    retained = pd.DataFrame([_retained_event_row(1, 100, 120)])
    duplicate_events = pd.DataFrame(
        [{"event_sample": 100, "event_code": 120}, {"event_sample": 100, "event_code": 120}]
    )
    with pytest.raises(ValueError, match="Observed image event contains duplicate keys"):
        build_event_epoch_mapping(duplicate_events, retained)


def test_duplicate_residual_qc_key_is_rejected_instead_of_collapsed() -> None:
    rejection = pd.DataFrame(
        [{"subject": "sub001", "recording_part": "part1", "trial": 1, "condition": "old", "event_sample": 100, "event_code": 120, "rejected": False, "rejection_reason": ""}]
    )
    qc_row = {"subject": "sub001", "recording_part": "part1", "trial": 1, "condition": "old", "event_sample": 100, "event_code": 120, "ptp_uv": 20.0, "max_ptp_channel": "Fz", "threshold_uv": 330.0, "rejected_for_ptp": False, "rejected_for_nonfinite": False, "eeg_qc_status": "retained"}
    residual = pd.DataFrame([qc_row, {**qc_row, "ptp_uv": 400.0, "rejected_for_ptp": True}])

    with pytest.raises(ValueError, match="Residual EEG QC contains duplicate keys"):
        merge_residual_qc_with_rejection_table(rejection, residual)


def test_duplicate_epoch_rejection_key_is_rejected_before_qc_merge() -> None:
    rejection_row = {"subject": "sub001", "recording_part": "part1", "trial": 1, "condition": "old", "event_sample": 100, "event_code": 120, "rejected": False, "rejection_reason": ""}
    rejection = pd.DataFrame([rejection_row, rejection_row])
    residual = pd.DataFrame(
        [{"subject": "sub001", "recording_part": "part1", "trial": 1, "condition": "old", "event_sample": 100, "event_code": 120, "ptp_uv": 20.0, "max_ptp_channel": "Fz", "threshold_uv": 330.0, "rejected_for_ptp": False, "rejected_for_nonfinite": False, "eeg_qc_status": "retained"}]
    )

    with pytest.raises(ValueError, match="Epoch rejection table contains duplicate keys"):
        merge_residual_qc_with_rejection_table(rejection, residual)


def _marker(index: int, sample: int, code: int, category: str) -> dict[str, object]:
    return {
        "marker_index": index,
        "marker_type": "Stimulus",
        "description": f"S {code}",
        "code": code,
        "sample": sample,
        "marker_category": category,
    }


def _retained_alignment_row(trial: int, sample: int, code: int) -> dict[str, object]:
    return {
        "subject": "sub001",
        "recording_part": "part1",
        "trial": trial,
        "image_onset_sample": sample,
        "image_onset_code": code,
        "retain_for_epoching": True,
    }


def _retained_event_row(trial: int, sample: int, code: int) -> dict[str, object]:
    return {
        **_retained_alignment_row(trial, sample, code),
        "event_sample": sample,
        "event_code": code,
        "condition": "old" if code == 120 else "new",
    }
