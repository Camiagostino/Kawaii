from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.identity import identify_behavioral_inputs, identify_brainvision_dataset
from eeg_software.workflows.kawaii_session2.response_timing_qc import DEFAULT_RESPONSE_TYPES, build_trial_level_qc, response_timing_semantics, summarise_rt_medians, summarise_valid, validate_trial_level, write_report
from eeg_software.workflows.kawaii_session2.stage_identity import (
    build_alignment_artifact_identity,
    build_dataset_context_identity,
    build_timing_qc_input_identity,
)


def _settings():
    return {"sampling_rate_hz": 500.0, "response_types": DEFAULT_RESPONSE_TYPES}


def _input_identity(
    tmp_path: Path,
    behaviour: pd.DataFrame,
    alignment: pd.DataFrame,
    settings: dict,
):
    vhdr = tmp_path / "recording.vhdr"
    vmrk = tmp_path / "recording.vmrk"
    eeg = tmp_path / "recording.eeg"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\nDataFile=recording.eeg\nMarkerFile=recording.vmrk\n",
        encoding="utf-8",
    )
    vmrk.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "[Common Infos]\nDataFile=recording.eeg\n",
        encoding="utf-8",
    )
    eeg.write_bytes(b"samples")
    behavioral_path = tmp_path / "behavior.csv"
    alignment_path = tmp_path / "trial_alignment.csv"
    behaviour.to_csv(behavioral_path, index=False)
    alignment.to_csv(alignment_path, index=False)
    behavioral_inputs = identify_behavioral_inputs(behavioral_log_path=behavioral_path)
    context = build_dataset_context_identity(
        participant="sub001",
        session="session2",
        recording_part="part1",
        brainvision=identify_brainvision_dataset(vhdr, vmrk, eeg),
        behavioral_inputs=behavioral_inputs,
        resolved_config_fingerprint="sha256:" + "1" * 64,
        scientific_fingerprint="sha256:" + "2" * 64,
    )
    artifact = build_alignment_artifact_identity(
        contexts=[context], alignment=alignment, output_path=alignment_path
    )
    return build_timing_qc_input_identity(
        context=context,
        alignment_artifact=artifact,
        behavioral_inputs=behavioral_inputs,
        behavioral=behaviour,
        alignment=alignment,
        settings=settings,
    )


def test_maps_response_types_and_converts_seconds_to_ms(tmp_path: Path):
    behaviour = pd.DataFrame([{"subject_id": "sub001", "session": 2, "block": 1, "trial": 1,
        "recognition_rt_s": .8, "recognition_timeout": 0, "kawaii_rt_s": 1.0, "kawaii_timeout": 0,
        "arousal_rt_s": 1.2, "arousal_timeout": 0}])
    alignment = pd.DataFrame([{"recording_part": "part1", "trial": 1,
        "image_onset_sample": 100, "recognition_response_sample": 550, "image_onset_code": 120, "recognition_response_code": 150,
        "kawaii_question_sample": 600, "kawaii_response_sample": 1150, "kawaii_question_code": 100, "kawaii_response_code": 15,
        "arousal_question_sample": 1200, "arousal_response_sample": 1850, "arousal_question_code": 200, "arousal_response_code": 25}])
    settings = _settings()
    identity = _input_identity(tmp_path, behaviour, alignment, settings)
    out = build_trial_level_qc(behaviour, alignment, None, subject="sub001", recording_part="part1", settings=settings, input_identity=identity)
    assert out.response_type.tolist() == ["recognition", "kawaii", "arousal"]
    assert out.eeg_marker_rt_ms.tolist() == [900.0, 1100.0, 1300.0]
    assert out.eeg_minus_behavioural_ms.tolist() == pytest.approx([100.0, 100.0, 100.0])
    assert out.qc_status.eq("valid").all()
    assert out.qc_reason.str.contains("missing_trigger_log").all()
    assert out.behavioral_rt_s.tolist() == pytest.approx([.8, 1.0, 1.2])
    assert out.eeg_response_marker_latency_s.tolist() == pytest.approx([.9, 1.1, 1.3])
    assert out.marker_semantics.eq("post_key_release_eeg_response_marker").all()
    assert out.response_locked_inference.eq("NOT_VALIDATED_NOT_APPROVED").all()


def test_explicitly_retains_timeout_missing_and_negative_rows(tmp_path: Path):
    behaviour = pd.DataFrame([{"subject_id": "sub001", "session": 2, "block": 1, "trial": 1,
        "recognition_rt_s": None, "recognition_timeout": 1, "kawaii_rt_s": 1, "kawaii_timeout": 0, "arousal_rt_s": 1, "arousal_timeout": 0}])
    alignment = pd.DataFrame([{"recording_part": "part1", "trial": 1, "image_onset_sample": 200,
        "recognition_response_sample": 100, "kawaii_question_sample": None, "kawaii_response_sample": 300,
        "arousal_question_sample": 300, "arousal_response_sample": None}])
    settings = _settings()
    identity = _input_identity(tmp_path, behaviour, alignment, settings)
    out = build_trial_level_qc(behaviour, alignment, None, subject="sub001", recording_part="part1", settings=settings, input_identity=identity)
    assert out.qc_status.eq("excluded").all()
    assert "timeout" in out.iloc[0].qc_reason and "negative_eeg_marker_rt" in out.iloc[0].qc_reason
    assert "missing_onset_marker" in out.iloc[1].qc_reason
    assert "missing_response_marker" in out.iloc[2].qc_reason


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (-0.1, "negative_behavioural_rt"),
        (float("nan"), "nonfinite_behavioural_rt"),
        (float("inf"), "nonfinite_behavioural_rt"),
        (float("-inf"), "nonfinite_behavioural_rt"),
    ],
)
def test_invalid_behavioural_rt_is_explicitly_excluded(tmp_path: Path, value: float, reason: str):
    behaviour = pd.DataFrame([{"subject_id": "sub001", "session": 2, "block": 1, "trial": 1,
        "recognition_rt_s": value, "recognition_timeout": 0, "kawaii_rt_s": 1.0, "kawaii_timeout": 0,
        "arousal_rt_s": 1.2, "arousal_timeout": 0}])
    alignment = pd.DataFrame([{"recording_part": "part1", "trial": 1,
        "image_onset_sample": 100, "recognition_response_sample": 550,
        "kawaii_question_sample": 600, "kawaii_response_sample": 1150,
        "arousal_question_sample": 1200, "arousal_response_sample": 1850}])
    settings = _settings()
    identity = _input_identity(tmp_path, behaviour, alignment, settings)

    out = build_trial_level_qc(behaviour, alignment, None, subject="sub001", recording_part="part1", settings=settings, input_identity=identity)

    recognition = out[out["response_type"].eq("recognition")].iloc[0]
    assert recognition.qc_status == "excluded"
    assert reason in recognition.qc_reason
    assert out.loc[out["response_type"].isin(["kawaii", "arousal"]), "qc_status"].eq("valid").all()
    summary = summarise_valid(out)
    assert int(summary["n"].sum()) == 2


def test_zero_behavioural_rt_remains_valid_and_all_invalid_summary_is_empty(tmp_path: Path):
    behaviour = pd.DataFrame([{"subject_id": "sub001", "session": 2, "block": 1, "trial": 1,
        "recognition_rt_s": 0.0, "recognition_timeout": 0, "kawaii_rt_s": -1.0, "kawaii_timeout": 0,
        "arousal_rt_s": float("inf"), "arousal_timeout": 0}])
    alignment = pd.DataFrame([{"recording_part": "part1", "trial": 1,
        "image_onset_sample": 100, "recognition_response_sample": 550,
        "kawaii_question_sample": 600, "kawaii_response_sample": 1150,
        "arousal_question_sample": 1200, "arousal_response_sample": 1850}])
    settings = _settings()
    identity = _input_identity(tmp_path, behaviour, alignment, settings)
    out = build_trial_level_qc(behaviour, alignment, None, subject="sub001", recording_part="part1", settings=settings, input_identity=identity)

    recognition = out[out["response_type"].eq("recognition")].iloc[0]
    assert recognition.qc_status == "valid"
    assert recognition.behavioural_rt_ms == 0.0
    assert out.loc[out["response_type"].isin(["kawaii", "arousal"]), "qc_status"].eq("excluded").all()

    all_invalid = out.assign(qc_status="excluded")
    assert summarise_valid(all_invalid).empty
    assert summarise_rt_medians(all_invalid).empty


def test_validation_rejects_duplicate_valid_match():
    row = {"subject_id": "sub001", "session": 2, "recording_part": "part1", "block": 1, "trial": 1,
           "response_type": "recognition", "qc_status": "valid", "behavioural_rt_ms": 10, "eeg_marker_rt_ms": 20}
    with pytest.raises(ValueError, match="one-to-one"):
        validate_trial_level(pd.DataFrame([row, row]))


def test_report_medians_use_valid_rows_only_and_reconcile_counts():
    table = pd.DataFrame([
        {"response_type": "recognition", "qc_status": "valid", "behavioural_rt_ms": 800, "eeg_marker_rt_ms": 900, "eeg_minus_behavioural_ms": 100},
        {"response_type": "recognition", "qc_status": "valid", "behavioural_rt_ms": 1000, "eeg_marker_rt_ms": 1200, "eeg_minus_behavioural_ms": 200},
        {"response_type": "recognition", "qc_status": "excluded", "behavioural_rt_ms": 9999, "eeg_marker_rt_ms": 9999, "eeg_minus_behavioural_ms": 9999},
    ])
    summary = summarise_rt_medians(table).iloc[0]
    assert summary["Valid trials"] == 2
    assert summary["Behavioural RT median (ms)"] == 900
    assert summary["EEG-derived RT median (ms)"] == 1050
    assert summary["Difference median (ms)"] == 150


def test_verified_acquisition_semantics_prohibit_response_locked_correction():
    semantics = response_timing_semantics()
    sequence = semantics["sequence"]
    assert sequence.index("key_down_detected") < sequence.index("key_release") < sequence.index("eeg_response_marker")
    assert semantics["behavioral_rt_anchor"] == "key_down"
    assert semantics["difference_interpretation"] == "composite_timing_discrepancy"
    assert semantics["trigger_delay_interpretation"] == "trigger_write_overhead_only"
    assert semantics["fixed_correction_applied"] is False
    assert semantics["participant_median_correction_applied"] is False
    assert semantics["response_locked_inference"] == "NOT_VALIDATED_NOT_APPROVED"
    assert semantics["stimulus_locked_erp_status"] == "UNAFFECTED"


def test_report_does_not_require_optional_tabulate_dependency(tmp_path, monkeypatch):
    behaviour = pd.DataFrame([{"subject_id": "sub001", "session": 2, "block": 1, "trial": 1,
        "recognition_rt_s": .8, "recognition_timeout": 0, "kawaii_rt_s": 1.0, "kawaii_timeout": 0,
        "arousal_rt_s": 1.2, "arousal_timeout": 0}])
    alignment = pd.DataFrame([{"recording_part": "part1", "trial": 1,
        "image_onset_sample": 100, "recognition_response_sample": 550,
        "kawaii_question_sample": 600, "kawaii_response_sample": 1150,
        "arousal_question_sample": 1200, "arousal_response_sample": 1850}])
    settings = _settings()
    identity = _input_identity(tmp_path, behaviour, alignment, settings)
    table = build_trial_level_qc(behaviour, alignment, None, subject="sub001", recording_part="part1", settings=settings, input_identity=identity)
    monkeypatch.setattr(pd.DataFrame, "to_markdown", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optional dependency used")))

    report = write_report(table, [], tmp_path / "report.md")

    assert report.exists()
    assert "| Response type |" in report.read_text(encoding="utf-8")
