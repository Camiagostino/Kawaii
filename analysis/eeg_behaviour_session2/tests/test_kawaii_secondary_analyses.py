from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.config import load_session2_eeg_config
from eeg_software.workflows.kawaii_session2.secondary_analyses.arousal_erp import (
    extract_existing_arousal_erp,
)
from eeg_software.workflows.kawaii_session2.secondary_analyses.config import (
    load_secondary_analysis_config,
)
from eeg_software.workflows.kawaii_session2.secondary_analyses.recognition_behavior import (
    compute_signal_detection_metrics,
    dprime_inference,
    load_source_verified_behavioural_trials,
)
from eeg_software.workflows.kawaii_session2.secondary_analyses.rating_memory import (
    compute_rating_memory_family,
    summarise_participant_rating_means,
)
from eeg_software.workflows.kawaii_session2.secondary_analyses.recognition_erp import (
    compute_erp_family,
    load_recognition_measurements,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_behaviour() -> pd.DataFrame:
    rows = []
    counts = {
        "s1": {"hit": 8, "miss": 2, "false_alarm": 2, "correct_rejection": 8},
        "s2": {"hit": 7, "miss": 3, "false_alarm": 1, "correct_rejection": 9},
    }
    for subject, mapping in counts.items():
        for condition, n in mapping.items():
            rows.extend({"participant": subject, "recognition_condition": condition} for _ in range(n))
    return pd.DataFrame(rows)


def test_signal_detection_uses_standard_rates_when_no_extreme_rate() -> None:
    result = compute_signal_detection_metrics(_synthetic_behaviour()).set_index("subject")
    assert result.loc["s1", "hit_rate"] == pytest.approx(0.8)
    assert result.loc["s1", "false_alarm_rate"] == pytest.approx(0.2)
    assert not bool(result.loc["s1", "loglinear_correction_used"])
    assert result.loc["s1", "dprime"] > 0


def test_signal_detection_uses_loglinear_only_for_exact_zero_or_one() -> None:
    rows = []
    rows.extend({"participant": "s1", "recognition_condition": "hit"} for _ in range(10))
    rows.extend({"participant": "s1", "recognition_condition": "correct_rejection"} for _ in range(10))
    result = compute_signal_detection_metrics(pd.DataFrame(rows)).iloc[0]
    assert bool(result["loglinear_correction_used"])
    assert np.isfinite(result["dprime"])
    assert np.isfinite(result["criterion_c"])


def test_secondary_config_reuses_existing_figure_style_authority() -> None:
    config = load_secondary_analysis_config(ROOT, "config/kawaii_session2_secondary_analyses.example.yaml")
    assert config["figure_style_path"] == "config/kawaii_session2_figure_style.yaml"
    assert config["output_dir"] == "outputs/session2/secondary_analyses"


def test_erp_family_uses_exact_declared_family_size_and_holm() -> None:
    rows = []
    subjects = [f"s{i}" for i in range(1, 5)]
    cells = [
        {"roi": "frontal", "time_window": "300_500_ms", "label": "F"},
        {"roi": "frontocentral", "time_window": "300_500_ms", "label": "FC"},
        {"roi": "centroparietal", "time_window": "500_800_ms", "label": "CP"},
        {"roi": "parietal", "time_window": "500_800_ms", "label": "P"},
    ]
    for subject_index, subject in enumerate(subjects):
        for cell_index, cell in enumerate(cells):
            baseline = float(subject_index + cell_index)
            offsets = {
                "hit": 0.40 + 0.03 * subject_index,
                "miss": 0.00,
                "false_alarm": 0.20 - 0.01 * subject_index,
                "correct_rejection": 0.00,
            }
            for condition, offset in offsets.items():
                rows.append({
                    "subject": subject,
                    "condition": condition,
                    "roi": cell["roi"],
                    "time_window": cell["time_window"],
                    "mean_amplitude_uv": baseline + offset + (subject_index * 0.01 * (cell_index + 1)),
                })
    measurements = pd.DataFrame(rows)
    family = {
        "name": "recognition_secondary_memory_outcome",
        "correction": "holm",
        "family_size": 8,
        "analysis_status": "SECONDARY",
        "comparisons": [
            {"name": "hit_vs_miss", "condition_a": "hit", "condition_b": "miss", "label": "Hit vs Miss"},
            {"name": "false_alarm_vs_correct_rejection", "condition_a": "false_alarm", "condition_b": "correct_rejection", "label": "FA vs CR"},
        ],
    }
    core = {"statistics": {"paired_test": "paired_t", "fallback_paired_test": "wilcoxon", "normality_test": "shapiro", "multiple_comparison_correction": "holm", "effect_size": "auto", "alpha": 0.05, "technical_minimum_pairs": 2, "confidence_level": 0.95, "minimum_normality_n": 3}}
    result = compute_erp_family(measurements, family, core, cells)
    assert len(result) == 8
    assert set(result["correction_family_size"]) == {8}
    assert set(result["correction_method"]) == {"holm"}
    assert set(result["analysis_status"]) == {"SECONDARY"}
    assert "interpretation_status" not in result.columns



def _synthetic_rating_trials() -> pd.DataFrame:
    rows = []
    for subject_index, subject in enumerate(["s1", "s2", "s3", "s4"]):
        condition_offsets = {
            "hit": (0.45 + 0.10 * subject_index, 0.14 + 0.04 * subject_index),
            "miss": (0.0, 0.0),
            "false_alarm": (0.24 + 0.04 * subject_index, 0.31 + 0.06 * subject_index),
            "correct_rejection": (0.0, 0.0),
        }
        for condition, (cute_offset, arousal_offset) in condition_offsets.items():
            for trial_index in range(3):
                rows.append({
                    "participant": subject,
                    "recognition_condition": condition,
                    "kawaii_rating": 4.0 + cute_offset + 0.1 * subject_index + 0.02 * trial_index,
                    "arousal_rating": 3.0 + arousal_offset + 0.05 * subject_index + 0.03 * trial_index,
                })
    return pd.DataFrame(rows)


def _synthetic_rating_family() -> dict:
    return {
        "name": "rating_memory_association",
        "correction": "holm",
        "family_size": 4,
        "comparisons": [
            {"name": "cuteness_hit_vs_miss", "label": "Cuteness · Hit vs Miss", "rating": "cuteness", "rating_column": "kawaii_rating", "condition_a": "hit", "condition_b": "miss"},
            {"name": "cuteness_false_alarm_vs_correct_rejection", "label": "Cuteness · FA vs CR", "rating": "cuteness", "rating_column": "kawaii_rating", "condition_a": "false_alarm", "condition_b": "correct_rejection"},
            {"name": "arousal_hit_vs_miss", "label": "Arousal · Hit vs Miss", "rating": "arousal", "rating_column": "arousal_rating", "condition_a": "hit", "condition_b": "miss"},
            {"name": "arousal_false_alarm_vs_correct_rejection", "label": "Arousal · FA vs CR", "rating": "arousal", "rating_column": "arousal_rating", "condition_a": "false_alarm", "condition_b": "correct_rejection"},
        ],
    }


def test_rating_memory_summary_is_participant_weighted() -> None:
    means = summarise_participant_rating_means(_synthetic_rating_trials(), _synthetic_rating_family())
    assert len(means) == 32
    row = means[(means["subject"] == "s1") & (means["rating"] == "cuteness") & (means["condition"] == "hit")].iloc[0]
    assert row["n_trials"] == 3
    assert row["mean_rating"] == pytest.approx(4.47)


def test_rating_memory_family_has_exact_four_test_holm_family() -> None:
    core = {"statistics": {"paired_test": "paired_t", "fallback_paired_test": "wilcoxon", "normality_test": "shapiro", "multiple_comparison_correction": "holm", "effect_size": "auto", "alpha": 0.05, "technical_minimum_pairs": 2, "confidence_level": 0.95, "minimum_normality_n": 3}}
    _, table = compute_rating_memory_family(_synthetic_rating_trials(), _synthetic_rating_family(), core)
    assert len(table) == 4
    assert set(table["correction_family_size"]) == {4}
    assert set(table["correction_method"]) == {"holm"}
    expected = {
        "cuteness_hit_vs_miss": 0.6,
        "cuteness_false_alarm_vs_correct_rejection": 0.3,
        "arousal_hit_vs_miss": 0.2,
        "arousal_false_alarm_vs_correct_rejection": 0.4,
    }
    for row in table.itertuples():
        assert row.mean_difference == pytest.approx(expected[row.comparison])


def test_secondary_config_declares_rating_memory_family() -> None:
    config = load_secondary_analysis_config(ROOT, "config/kawaii_session2_secondary_analyses.example.yaml")
    family = config["families"]["rating_memory_association"]
    assert family["implementation_status"] == "IMPLEMENTED"
    assert family["family_size"] == 4
    assert [item["rating"] for item in family["comparisons"]].count("cuteness") == 2
    assert [item["rating"] for item in family["comparisons"]].count("arousal") == 2


def test_secondary_config_declares_existing_arousal_coefficient_without_refit() -> None:
    config = load_secondary_analysis_config(ROOT, "config/kawaii_session2_secondary_analyses.example.yaml")
    spec = config["existing_arousal_coefficient"]
    assert spec["implementation_status"] == "IMPLEMENTED"
    assert spec["coefficient"] == "arousal_centered"
    assert spec["model_id"] == "cuteness_primary_continuous_500_800_centroparietal"
    assert spec["refit_allowed"] is False


def test_existing_arousal_erp_extraction_is_read_only_and_preserves_diagnostics(tmp_path: Path) -> None:
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir(parents=True)
    pd.DataFrame([{
        "model_id":"m1","hypothesis_family":"cuteness_primary_continuous","hypothesis_id":np.nan,
        "analysis_status":"ADJUSTMENT","term":"arousal_centered","coefficient_role":"ADJUSTMENT_COEFFICIENT",
        "estimate":0.02,"standard_error":0.05,"confidence_interval_low":-0.08,
        "confidence_interval_high":0.12,"test_statistic":0.4,"p_value":0.69,
    }]).to_csv(stats_dir/"coefficients.csv",index=False)
    pd.DataFrame([{
        "model_id":"m1","model_status":"FIT_WARNING","converged":True,"fit_warnings":"boundary warning",
        "n_observations":100,"n_participants":4,"n_items":25,
        "participant_random_intercept_variance":1.2,"item_random_intercept_variance":0.01,
        "residual_variance":4.0,"arousal_included":True,"fallback_occurred":False,
    }]).to_csv(stats_dir/"diagnostics.csv",index=False)
    pd.DataFrame([{
        "model_id":"m1","formula":"ERP_mean_uv ~ cuteness_centered + old_new_old + arousal_centered",
        "fixed_effects":"cuteness_centered;old_new_old;arousal_centered","optimizer":"lbfgs",
        "fit_method":"maximum_likelihood","arousal_included":True,
    }]).to_csv(stats_dir/"specification.csv",index=False)
    cfg={"inputs":{
        "cuteness_mixed_model_coefficients":"stats/coefficients.csv",
        "cuteness_mixed_model_diagnostics":"stats/diagnostics.csv",
        "cuteness_mixed_model_specification":"stats/specification.csv",
    },"existing_arousal_coefficient":{"model_id":"m1","coefficient":"arousal_centered","refit_allowed":False}}
    table, inputs = extract_existing_arousal_erp(tmp_path,cfg)
    row=table.iloc[0]
    assert row["estimate_uv_per_centered_rating_unit"] == pytest.approx(0.02)
    assert row["p_value"] == pytest.approx(0.69)
    assert row["model_status"] == "FIT_WARNING"
    assert row["n_observations"] == 100
    assert row["n_participants"] == 4
    assert not bool(row["refit_performed"])
    assert len(inputs)==3
