from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.cuteness_mixed import (
    center_cuteness_trials,
    validate_cuteness_model_inputs,
)
from eeg_software.workflows.kawaii_session2.group_aggregation import (
    pool_participant_measurements,
    validate_part_level_measurements,
    validate_pooled_measurements,
)
from eeg_software.workflows.kawaii_session2.group_statistics import (
    apply_multiplicity_families,
    attach_technical_n,
    continuous_analysis_rows,
    paired_condition_values,
)
from eeg_software.workflows.kawaii_session2.erp import extract_erp_measurements


class _Evoked:
    ch_names = ["Cz"]
    times = np.array([0.5])

    def __init__(self, value: float) -> None:
        self.data = np.array([[value]], dtype=float)


def _part_measurements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "sub001",
                "recording_part": "part1",
                "branch": "recognition",
                "condition": "hit",
                "roi": "frontal",
                "time_window": "300_500_ms",
                "mean_amplitude_uv": 2.0,
                "n_epochs": 2,
            },
            {
                "subject": "sub001",
                "recording_part": "part2",
                "branch": "recognition",
                "condition": "hit",
                "roi": "frontal",
                "time_window": "300_500_ms",
                "mean_amplitude_uv": 8.0,
                "n_epochs": 8,
            },
        ]
    )


def _paired_measurements() -> pd.DataFrame:
    rows = []
    for subject, hit, cr in (("sub001", 4.0, 1.0), ("sub002", 6.0, 2.0)):
        for condition, value in (("hit", hit), ("correct_rejection", cr)):
            rows.append(
                {
                    "subject": subject,
                    "branch": "recognition",
                    "condition": condition,
                    "roi": "frontal",
                    "time_window": "300_500_ms",
                    "mean_amplitude_uv": value,
                }
            )
    return pd.DataFrame(rows)


def test_single_part_measurement_passes_unchanged() -> None:
    source = _part_measurements().iloc[[0]].copy()

    pooled = pool_participant_measurements(source)

    assert pooled.loc[0, "mean_amplitude_uv"] == 2.0
    assert pooled.loc[0, "n_epochs"] == 2
    assert pooled.loc[0, "recording_parts"] == "part1"


def test_finite_participant_erp_measurement_is_unchanged() -> None:
    result = extract_erp_measurements(
        subject="sub001",
        recording_part="part1",
        branch="cuteness",
        evokeds={"high": _Evoked(2e-6)},
        condition_counts=pd.DataFrame([{"condition": "high", "retained_epochs": 3}]),
        settings={
            "rois": {"central": {"channels": ["Cz"]}},
            "measurement_time_windows": {
                "500_ms": {"start_ms": 500, "end_ms": 500}
            },
        },
        metadata_source="test",
        processing_version="test",
    )

    assert result.loc[0, "mean_amplitude_uv"] == pytest.approx(2.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_erp_data_cannot_become_participant_measurement(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite ERP values cannot define measurement"):
        extract_erp_measurements(
            subject="sub001",
            recording_part="part1",
            branch="cuteness",
            evokeds={"high": _Evoked(value)},
            condition_counts=pd.DataFrame([{"condition": "high", "retained_epochs": 3}]),
            settings={
                "rois": {"central": {"channels": ["Cz"]}},
                "measurement_time_windows": {
                    "500_ms": {"start_ms": 500, "end_ms": 500}
                },
            },
            metadata_source="test",
            processing_version="test",
        )


def test_distinct_parts_keep_retained_trial_weighting() -> None:
    pooled = pool_participant_measurements(_part_measurements())

    assert pooled.loc[0, "mean_amplitude_uv"] == pytest.approx(6.8)
    assert pooled.loc[0, "n_epochs"] == 10
    assert pooled.loc[0, "recording_parts"] == "part1;part2"


@pytest.mark.parametrize("conflicting_value", [2.0, 99.0])
def test_duplicate_same_part_measurement_key_is_rejected(conflicting_value: float) -> None:
    source = _part_measurements().iloc[[0]].copy()
    duplicate = source.copy()
    duplicate.loc[duplicate.index[0], "mean_amplitude_uv"] = conflicting_value

    with pytest.raises(ValueError, match="Duplicate part-level measurement key.*recording_part"):
        pool_participant_measurements(pd.concat([source, duplicate], ignore_index=True))


def test_duplicate_post_pool_participant_cell_is_rejected() -> None:
    pooled = pool_participant_measurements(_part_measurements())

    with pytest.raises(ValueError, match="Duplicate pooled participant-level measurement key.*time_window"):
        validate_pooled_measurements(pd.concat([pooled, pooled], ignore_index=True))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_participant_measurement_fails_before_pooling(value: float) -> None:
    source = _part_measurements().iloc[[0]].copy()
    source.loc[source.index[0], "mean_amplitude_uv"] = value

    with pytest.raises(ValueError, match="Non-finite part-level mean_amplitude_uv"):
        validate_part_level_measurements(source)


def test_valid_paired_values_are_unchanged() -> None:
    paired = paired_condition_values(_paired_measurements(), "hit", "correct_rejection")

    assert paired.to_dict(orient="records") == [
        {"subject": "sub001", "a": 4.0, "b": 1.0},
        {"subject": "sub002", "a": 6.0, "b": 2.0},
    ]


def test_duplicate_paired_condition_rejected_before_pivot_and_technical_n() -> None:
    measurements = _paired_measurements()
    duplicate = measurements.iloc[[0]].copy()
    invalid = pd.concat([measurements, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate participant-condition measurement before paired pivot"):
        paired_condition_values(invalid, "hit", "correct_rejection")

    stats = pd.DataFrame(
        [
            {
                "analysis_type": "paired",
                "condition_a": "hit",
                "condition_b": "correct_rejection",
                "roi": "frontal",
                "time_window": "300_500_ms",
                "n": 2,
            }
        ]
    )
    with pytest.raises(ValueError, match="Duplicate participant-condition measurement before paired pivot"):
        attach_technical_n(stats, invalid)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_pair_member_cannot_silently_change_n(value: float) -> None:
    measurements = _paired_measurements()
    measurements.loc[0, "mean_amplitude_uv"] = value

    with pytest.raises(ValueError, match="Non-finite paired measurements"):
        paired_condition_values(measurements, "hit", "correct_rejection")


def test_missing_arousal_remains_explicit_and_selects_reduced_eligibility() -> None:
    trials = pd.DataFrame(
        {
            "participant": ["sub001", "sub001"],
            "part_or_run": ["part1", "part1"],
            "epoch_retained": [True, True],
            "cuteness_raw": [2.0, 4.0],
            "arousal_raw": [np.nan, 3.0],
            "ERP_mean_uv": [1.0, 2.0],
            "mean_amplitude_uv": [1.0, 2.0],
            "roi_complete": [True, True],
            "item_id": ["a.png", "b.png"],
            "old_new_old": [0.0, 1.0],
        }
    )

    centered = center_cuteness_trials(trials)

    assert bool(centered.loc[0, "model_eligible_reduced"])
    assert not bool(centered.loc[0, "model_eligible_full"])
    assert pd.isna(centered.loc[0, "arousal_centered"])


@pytest.mark.parametrize("column", ["ERP_mean_uv", "cuteness_raw", "arousal_raw"])
@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_infinite_model_outcome_or_predictor_is_rejected(column: str, value: float) -> None:
    trials = pd.DataFrame(
        {
            "participant": ["sub001"],
            "epoch_retained": [True],
            "cuteness_raw": [2.0],
            "arousal_raw": [3.0],
            "ERP_mean_uv": [1.0],
            "roi_complete": [True],
            "item_id": ["a.png"],
            "old_new_old": [0.0],
        }
    )
    trials.loc[0, column] = value

    with pytest.raises(ValueError, match="Infinite cuteness model input"):
        center_cuteness_trials(trials)


def test_nan_in_manually_model_eligible_outcome_is_rejected() -> None:
    trials = pd.DataFrame(
        {
            "model_eligible_reduced": [True],
            "model_eligible_full": [False],
            "ERP_mean_uv": [math.nan],
            "cuteness_centered": [0.0],
            "old_new_old": [1.0],
            "arousal_centered": [math.nan],
        }
    )

    with pytest.raises(ValueError, match="Non-finite cuteness model ERP_mean_uv"):
        validate_cuteness_model_inputs(trials)


def test_infinite_continuous_predictor_is_not_silently_dropped() -> None:
    measurements = _paired_measurements().iloc[:3].copy()
    measurements["rating"] = [1.0, math.inf, 3.0]
    settings = {"cuteness": {"continuous": {"enabled": True, "predictor_columns": ["rating"]}}}

    with pytest.raises(ValueError, match="Non-finite continuous-analysis input"):
        continuous_analysis_rows(measurements, settings)


def test_infinite_pvalue_cannot_enter_holm_family() -> None:
    stats = pd.DataFrame(
        {
            "analysis_status": ["PRIMARY", "PRIMARY"],
            "hypothesis_family": ["recognition_primary", "recognition_primary"],
            "p_value": [0.01, math.inf],
        }
    )

    with pytest.raises(ValueError, match="infinite p-value"):
        apply_multiplicity_families(stats, {"alpha": 0.05})
