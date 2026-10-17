from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2 import cuteness_mixed  # noqa: E402
from eeg_software.workflows.kawaii_session2.cuteness_mixed import center_cuteness_trials, fit_cuteness_mixed_model  # noqa: E402
from eeg_software.workflows.kawaii_session2.group_statistics import (  # noqa: E402
    prepare_scientific_plot_cells,
    write_mean_sem_figure,
    write_statistics_report,
    write_statistics_figures,
)
from eeg_software.workflows.kawaii_session2.group_workflow import _statistics_status  # noqa: E402
from eeg_software.workflows.kawaii_session2.operations import ArtifactReference, OperationStatus  # noqa: E402


def _model_trials(*, missing_arousal: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(808)
    rows = []
    for participant_index in range(4):
        participant = f"sub{participant_index + 1:03d}"
        for item_index in range(4):
            cuteness = float(rng.integers(1, 10))
            arousal = float(np.clip(1.0 + 0.45 * cuteness + rng.normal(0, 1.2), 1, 9))
            rows.append(
                {
                    "participant": participant,
                    "item_id": f"item{item_index}",
                    "cuteness_raw": cuteness,
                    "arousal_raw": np.nan if missing_arousal and participant_index == 0 and item_index == 0 else arousal,
                    "ERP_mean_uv": cuteness + arousal * 0.1,
                    "mean_amplitude_uv": cuteness + arousal * 0.1,
                    "roi_complete": True,
                    "old_new_old": float(item_index % 2),
                    "epoch_retained": True,
                }
            )
    return center_cuteness_trials(pd.DataFrame(rows))


def _attempt(status: str, *, reason: str = "", warnings: str = "") -> dict:
    return {
        "status": status,
        "reason": reason,
        "warnings": warnings,
        "fitted": None,
        "data": pd.DataFrame(),
        "variance_components": [],
        "variance_component_names": [],
    }


def _settings() -> dict:
    return {
        "primary_model": {
            "full_formula": "ERP_mean_uv ~ cuteness_centered + old_new_old + arousal_centered",
            "reduced_formula": "ERP_mean_uv ~ cuteness_centered + old_new_old",
        }
    }


@pytest.mark.parametrize(
    ("statuses", "expected_roles", "selected", "expected_fallback"),
    [
        (("FIT_OK",), ("full",), 0, False),
        (("FIT_WARNING",), ("full",), 0, False),
        (("NOT_FIT_SINGULAR", "FIT_OK"), ("full", "reduced"), 1, True),
        (("NOT_FIT_SINGULAR", "NOT_FIT_NONCONVERGENT"), ("full", "reduced"), None, True),
    ],
)
def test_model_attempt_ledger_preserves_order_selection_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    statuses: tuple[str, ...],
    expected_roles: tuple[str, ...],
    selected: int | None,
    expected_fallback: bool,
) -> None:
    calls = iter(statuses)

    def fake_fit(data, formula, model, *, arousal_included):
        status = next(calls)
        return _attempt(status, reason="synthetic failure" if status.startswith("NOT_") else "", warnings="synthetic warning" if status == "FIT_WARNING" else "") | {"data": data}

    monkeypatch.setattr(cuteness_mixed, "_fit_crossed_model", fake_fit)
    result = fit_cuteness_mixed_model(_model_trials(), _settings())

    ledger = result.attempt_ledger
    assert ledger["model_role"].tolist() == list(expected_roles)
    assert ledger["attempt_index"].tolist() == list(range(len(expected_roles)))
    assert ledger["selected_for_final_result"].tolist() == [index == selected for index in range(len(expected_roles))]
    assert result.selected_attempt_index == selected
    assert result.fallback_occurred is expected_fallback
    assert ledger.loc[ledger["status"] == "FIT_WARNING", "warnings"].tolist() == (["synthetic warning"] if "FIT_WARNING" in statuses else [])
    assert list(ledger["formula"]) == [
        _settings()["primary_model"]["full_formula"] if role == "full" else _settings()["primary_model"]["reduced_formula"]
        for role in expected_roles
    ]


def test_prefit_reduced_only_path_does_not_invent_full_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    def fake_fit(data, formula, model, *, arousal_included):
        called.append((formula, arousal_included))
        return _attempt("FIT_OK") | {"data": data}

    monkeypatch.setattr(cuteness_mixed, "_fit_crossed_model", fake_fit)
    result = fit_cuteness_mixed_model(_model_trials(missing_arousal=True), _settings())

    assert called == [(_settings()["primary_model"]["reduced_formula"], False)]
    assert result.attempt_ledger["model_role"].tolist() == ["reduced"]
    assert result.fallback_occurred is False
    assert result.selected_attempt_index == 0


def test_model_attempt_ledger_serialization_and_report_linkage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fit(data, formula, model, *, arousal_included):
        return _attempt("NOT_FIT_SINGULAR", reason="synthetic singularity") | {"data": data}

    monkeypatch.setattr(cuteness_mixed, "_fit_crossed_model", fake_fit)
    result = fit_cuteness_mixed_model(_model_trials(), _settings())
    ledger_path = tmp_path / "cuteness_mixed_model_attempts.csv"
    diagnostics_path = tmp_path / "cuteness_mixed_model_diagnostics.csv"
    report_path = tmp_path / "report.md"
    result.attempt_ledger.to_csv(ledger_path, index=False)
    result.diagnostics.to_csv(diagnostics_path, index=False)

    write_statistics_report(
        "cuteness",
        report_path,
        {"mixed_model_diagnostics": diagnostics_path, "mixed_model_attempts": ledger_path},
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    report = report_path.read_text(encoding="utf-8")
    assert {"attempt_index", "formula", "status", "selected_for_final_result"}.issubset(result.attempt_ledger.columns)
    assert "## Model history" in report
    assert "authoritative ledger" in report
    assert "synthetic singularity" in report


def test_missing_required_model_ledger_blocks_gate5c_completion(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics.csv"
    pd.DataFrame([{"model_status": "FIT_OK"}]).to_csv(diagnostics, index=False)
    status, reason, diagnostic, _ = _statistics_status(
        object(),
        (ArtifactReference("mixed_model_diagnostics", diagnostics),),
        "cuteness",
    )
    assert status is OperationStatus.FAILED
    assert reason == "statistics_incomplete"
    assert "attempt ledger" in diagnostic


def test_cell_scoped_figure_preparation_uses_participant_unit_and_unique_cells() -> None:
    measurements = pd.DataFrame(
        [
            {"subject": "sub001", "condition": "hit", "roi": "frontal", "time_window": "early", "mean_amplitude_uv": 1.0, "n_epochs": 10},
            {"subject": "sub001", "condition": "hit", "roi": "frontal", "time_window": "early", "mean_amplitude_uv": 3.0, "n_epochs": 30},
            {"subject": "sub002", "condition": "hit", "roi": "frontal", "time_window": "early", "mean_amplitude_uv": 5.0, "n_epochs": 20},
            {"subject": "sub001", "condition": "hit", "roi": "frontal", "time_window": "late", "mean_amplitude_uv": 10.0, "n_epochs": 10},
            {"subject": "sub002", "condition": "hit", "roi": "frontal", "time_window": "late", "mean_amplitude_uv": 14.0, "n_epochs": 20},
            {"subject": "sub001", "condition": "hit", "roi": "parietal", "time_window": "early", "mean_amplitude_uv": 20.0, "n_epochs": 10},
            {"subject": "sub002", "condition": "hit", "roi": "parietal", "time_window": "early", "mean_amplitude_uv": 24.0, "n_epochs": 20},
        ]
    )
    cells = prepare_scientific_plot_cells("recognition", measurements)

    assert {(cell.roi, cell.time_window) for cell in cells} == {
        ("frontal", "early"),
        ("frontal", "late"),
        ("parietal", "early"),
    }
    early = next(cell for cell in cells if cell.roi == "frontal" and cell.time_window == "early")
    assert len(early.values) == 2
    assert early.participant_n == 2
    assert early.summary.loc[0, "mean"] == pytest.approx(((1 * 10 + 3 * 30) / 40 + 5) / 2)
    assert early.summary.loc[0, "n"] == 2
    assert early.summary.loc[0, "sem"] == pytest.approx(np.std([(1 * 10 + 3 * 30) / 40, 5], ddof=1) / np.sqrt(2))


def test_cell_scoped_outputs_do_not_collide_and_mixed_input_is_rejected(tmp_path: Path) -> None:
    measurements = pd.DataFrame(
        [
            {"subject": "sub001", "condition": "hit", "roi": "frontal", "time_window": "early", "mean_amplitude_uv": 1.0},
            {"subject": "sub002", "condition": "hit", "roi": "frontal", "time_window": "early", "mean_amplitude_uv": 2.0},
            {"subject": "sub001", "condition": "hit", "roi": "parietal", "time_window": "late", "mean_amplitude_uv": 4.0},
            {"subject": "sub002", "condition": "hit", "roi": "parietal", "time_window": "late", "mean_amplitude_uv": 5.0},
        ]
    )
    paths = {"figures": tmp_path / "figures"}
    outputs = write_statistics_figures("recognition", pd.DataFrame(), pd.DataFrame(), measurements, paths)
    figure_paths = [path for label, path in outputs.items() if "figure" in label and "__" in label]
    assert len(figure_paths) == len(set(figure_paths))
    assert all(path.exists() for path in figure_paths)
    assert any("frontal__early" in path.name for path in figure_paths)
    assert any("parietal__late" in path.name for path in figure_paths)

    with pytest.raises(ValueError, match="exactly one compatible"):
        write_mean_sem_figure("recognition", measurements, tmp_path / "mixed.png")
