from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Mapping
import math
import warnings

import numpy as np
import pandas as pd

from .bad_channel_disposition import validate_downstream_retained_bad_channel_safety
from .source_exposure import validate_continuous_cuteness_disposition


MODEL_STATES = {
    "FIT_OK",
    "FIT_WARNING",
    "NOT_FIT_INSUFFICIENT_DATA",
    "NOT_FIT_RANK_DEFICIENT",
    "NOT_FIT_SINGULAR",
    "NOT_FIT_NONCONVERGENT",
    "NOT_FIT_UNSUPPORTED_STRUCTURE",
}

ATTEMPT_LEDGER_COLUMNS = [
    "attempt_index",
    "model_role",
    "formula",
    "arousal_included",
    "status",
    "reason_code",
    "reason",
    "warnings",
    "convergence_warning_state",
    "success",
    "selected_for_final_result",
    "n_observations",
    "n_participants",
    "n_items",
]


@dataclass(frozen=True)
class CutenessModelResult:
    trial_table: pd.DataFrame
    arousal_feasibility: pd.DataFrame
    specification: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    attempt_ledger: pd.DataFrame
    selected_attempt_index: int | None
    fallback_occurred: bool
    fallback_reason: str


def primary_model_settings(config_or_settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen Phase 2C model settings from full config or ERP settings."""

    if "primary_model" in config_or_settings:
        return dict(config_or_settings.get("primary_model", {}) or {})
    erp = dict(config_or_settings.get("erp", {}) or {})
    return dict(dict(erp.get("cuteness", {}) or {}).get("primary_model", {}) or {})


def build_cuteness_trial_table(
    *,
    participant: str,
    recording_part: str,
    epochs,
    metadata: pd.DataFrame,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """Build one auditable primary-cell row per retained canonical epoch."""

    if len(epochs) != len(metadata):
        raise ValueError(f"Epoch/metadata mismatch for cuteness trial table: epochs={len(epochs)} metadata={len(metadata)}.")
    if hasattr(epochs, "info"):
        validate_downstream_retained_bad_channel_safety(epochs.info, settings)
    model = primary_model_settings(settings)
    window = dict(model.get("window", {}) or {})
    roi = dict(model.get("roi", {}) or {})
    requested = [str(channel) for channel in roi.get("channels", []) or []]
    available = [channel for channel in requested if channel in getattr(epochs, "ch_names", [])]
    missing = [channel for channel in requested if channel not in getattr(epochs, "ch_names", [])]
    roi_complete = bool(requested) and not missing
    start_ms = float(window.get("start_ms", 500.0))
    end_ms = float(window.get("end_ms", 800.0))
    times = np.asarray(getattr(epochs, "times", []), dtype=float)
    time_mask = (times >= start_ms / 1000.0) & (times <= end_ms / 1000.0)

    amplitude = np.full(len(metadata), np.nan, dtype=float)
    if roi_complete and bool(time_mask.any()):
        data = epochs.get_data(picks=available, copy=True)
        measurement_data = np.asarray(data[:, :, time_mask], dtype=float)
        if not np.isfinite(measurement_data).all():
            raise ValueError(
                f"Non-finite EEG values cannot define cuteness ERP measurements for "
                f"{participant!r}/{recording_part!r}."
            )
        amplitude = measurement_data.mean(axis=(1, 2)) * 1_000_000.0

    rating_column = str(model.get("rating_column", "kawaii_rating"))
    arousal_column = str(model.get("arousal_column", "arousal_rating"))
    item_column = str(model.get("item_column", "image_filename"))
    configured_old_new_column = str(model.get("old_new_column", "old_new_status"))
    old_new_column = (
        "source_verified_old_new_status"
        if "source_verified_old_new_status" in metadata.columns
        else configured_old_new_column
    )
    table = pd.DataFrame(
        {
            "participant": str(participant),
            "part_or_run": str(recording_part),
            "trial_id": metadata.get("trial", pd.Series([pd.NA] * len(metadata))).reset_index(drop=True),
            "event_sample": metadata.get("event_sample", metadata.get("image_onset_sample", pd.Series([pd.NA] * len(metadata)))).reset_index(drop=True),
            "event_code": metadata.get("event_code", metadata.get("image_onset_code", pd.Series([pd.NA] * len(metadata)))).reset_index(drop=True),
            "item_id": metadata.get(item_column, pd.Series([pd.NA] * len(metadata))).reset_index(drop=True),
            "item_source_column": item_column,
            "old_new": metadata.get(old_new_column, pd.Series([pd.NA] * len(metadata))).astype("string").str.strip().str.lower().reset_index(drop=True),
            "old_new_source_column": old_new_column,
            "cuteness_raw": pd.to_numeric(metadata.get(rating_column, pd.Series([pd.NA] * len(metadata))), errors="coerce").reset_index(drop=True),
            "cuteness_source_column": rating_column,
            "arousal_raw": pd.to_numeric(metadata.get(arousal_column, pd.Series([pd.NA] * len(metadata))), errors="coerce").reset_index(drop=True),
            "arousal_source_column": arousal_column,
            "epoch_retained": True,
            "epoch_input_scope": "phase2a4_retained_canonical_gate3c",
            "window": str(window.get("name", "500_800_ms")),
            "window_start_ms": start_ms,
            "window_end_ms": end_ms,
            "window_interval": str(window.get("interval", "closed_inclusive")),
            "n_time_samples": int(time_mask.sum()),
            "roi": str(roi.get("name", "centroparietal")),
            "roi_channels": ";".join(requested),
            "available_roi_channels": ";".join(available),
            "missing_roi_channels": ";".join(missing),
            "roi_channel_count": len(available),
            "roi_complete": roi_complete,
            "mean_amplitude_uv": amplitude,
            "ERP_mean_uv": amplitude,
            "analysis_status": str(model.get("analysis_status", "PRIMARY")),
            "hypothesis_family": str(model.get("hypothesis_family", "cuteness_primary_continuous")),
            "hypothesis_id": str(model.get("hypothesis_id", "CUT_PRIMARY_CENTERED_CUTENESS_500_800_CENTROPARIETAL")),
            "model_id": str(model.get("model_id", "cuteness_primary_continuous_500_800_centroparietal")),
        }
    )
    for column in (
        "original_old_new_status",
        "source_exposure_status",
        "source_verified_old_new_status",
        "recognition_eligible",
        "continuous_cuteness_old_new_eligible",
        "exposure_exclusion_reason",
        "source_exposure_identity",
        "source_exposure_artifact_id",
        "recognition_delay_days",
        "protocol_deviation",
        "protocol_disposition",
    ):
        if column in metadata:
            table[column] = metadata[column].reset_index(drop=True)
    table["old_new_old"] = table["old_new"].map({"new": 0.0, "old": 1.0})

    return center_cuteness_trials(table, scope="participant_recording_part")


def center_cuteness_trials(trials: pd.DataFrame, *, scope: str = "participant_all_parts") -> pd.DataFrame:
    """Recompute participant-mean centered predictors across the supplied retained trials."""

    table = trials.copy()
    if table.empty:
        return table
    _reject_infinite_cuteness_values(table)
    retained = _bool_series(table["epoch_retained"]) if "epoch_retained" in table.columns else pd.Series(True, index=table.index)
    cuteness_centering = retained & _finite_mask(table["cuteness_raw"]) & _finite_mask(table["ERP_mean_uv"]) & _bool_series(table["roi_complete"])
    arousal_centering = cuteness_centering & _finite_mask(table["arousal_raw"])
    table["cuteness_centering_eligible"] = cuteness_centering
    table["arousal_centering_eligible"] = arousal_centering
    table["participant_cuteness_mean"] = table["cuteness_raw"].where(cuteness_centering).groupby(table["participant"]).transform("mean")
    table["participant_arousal_mean"] = table["arousal_raw"].where(arousal_centering).groupby(table["participant"]).transform("mean")
    table["cuteness_centered"] = (table["cuteness_raw"] - table["participant_cuteness_mean"]).where(cuteness_centering)
    table["arousal_centered"] = (table["arousal_raw"] - table["participant_arousal_mean"]).where(arousal_centering)
    table["centering_scope"] = scope
    usable_item = table["item_id"].notna() & table["item_id"].astype("string").str.strip().ne("").fillna(False)
    source_eligible = (
        _bool_series(table["continuous_cuteness_old_new_eligible"])
        if "continuous_cuteness_old_new_eligible" in table.columns
        else pd.Series(True, index=table.index)
    )
    table["model_eligible_reduced"] = cuteness_centering & usable_item & _finite_mask(table["old_new_old"]) & source_eligible
    table["model_eligible_full"] = table["model_eligible_reduced"] & _finite_mask(table["arousal_centered"])
    table["exclusion_reason"] = table.apply(_trial_exclusion_reason, axis=1)
    validate_cuteness_model_inputs(table)
    validate_continuous_cuteness_disposition(table)
    return table


def validate_cuteness_model_inputs(trials: pd.DataFrame) -> None:
    """Reject corrupted infinities and non-finite rows marked model-eligible."""

    if trials.empty:
        return
    _reject_infinite_cuteness_values(trials)
    contracts = (
        ("model_eligible_reduced", ("ERP_mean_uv", "cuteness_centered", "old_new_old")),
        ("model_eligible_full", ("ERP_mean_uv", "cuteness_centered", "old_new_old", "arousal_centered")),
    )
    for eligibility_column, numeric_columns in contracts:
        if eligibility_column not in trials.columns:
            continue
        eligible = _bool_series(trials[eligibility_column])
        for column in numeric_columns:
            if column not in trials.columns:
                raise ValueError(f"Cuteness model input lacks required column {column!r} for {eligibility_column}.")
            invalid = eligible & ~_finite_mask(trials[column])
            if bool(invalid.any()):
                raise ValueError(
                    f"Non-finite cuteness model {column} in {eligibility_column} row(s): "
                    f"{trials.index[invalid].tolist()[:5]!r}"
                )


def _reject_infinite_cuteness_values(trials: pd.DataFrame) -> None:
    for column in (
        "ERP_mean_uv",
        "mean_amplitude_uv",
        "cuteness_raw",
        "arousal_raw",
        "cuteness_centered",
        "arousal_centered",
        "old_new_old",
    ):
        if column not in trials.columns:
            continue
        numeric = pd.to_numeric(trials[column], errors="coerce")
        infinite = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
        if bool(infinite.any()):
            raise ValueError(
                f"Infinite cuteness model input {column!r} at row(s) "
                f"{trials.index[infinite].tolist()[:5]!r}."
            )


def _finite_mask(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)), index=values.index)


def tag_categorical_cuteness_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    """Label categorical cuteness products without promoting them to inference."""

    tagged = measurements.copy()
    tagged["analysis_status"] = "DESCRIPTIVE"
    tagged["window_status"] = tagged.get("time_window", pd.Series(dtype=str)).astype(str).map(
        {"300_500_ms": "EXPLORATORY", "500_800_ms": "DESCRIPTIVE"}
    ).fillna("DESCRIPTIVE")
    tagged["hypothesis_family"] = "cuteness_descriptive_bins"
    tagged["hypothesis_id"] = ""
    return tagged


def evaluate_arousal_feasibility(trials: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen outcome-independent arousal feasibility procedure."""

    validate_cuteness_model_inputs(trials)
    reduced = trials[_bool_series(trials.get("model_eligible_reduced", pd.Series(False, index=trials.index)))].copy()
    full = trials[_bool_series(trials.get("model_eligible_full", pd.Series(False, index=trials.index)))].copy()
    reduced_participants = set(reduced.get("participant", pd.Series(dtype=str)).dropna().astype(str))
    full_participants = set(full.get("participant", pd.Series(dtype=str)).dropna().astype(str))
    reduced_items = set(reduced.get("item_id", pd.Series(dtype=str)).dropna().astype(str))
    full_items = set(full.get("item_id", pd.Series(dtype=str)).dropna().astype(str))
    completeness_ok = len(full) == len(reduced) and full_participants == reduced_participants and full_items == reduced_items and len(full) > 0
    participant_loss = _groups_with_row_loss(reduced, full, "participant")
    item_loss = _groups_with_row_loss(reduced, full, "item_id")

    variation = full.groupby("participant")["arousal_centered"].agg(["count", "nunique"]) if not full.empty else pd.DataFrame()
    zero_variation_participants = [] if variation.empty else variation.index[(variation["count"] < 2) | (variation["nunique"] < 2)].astype(str).tolist()
    arousal_variance = float(pd.to_numeric(full.get("arousal_centered", pd.Series(dtype=float)), errors="coerce").var(ddof=1)) if len(full) > 1 else math.nan
    variation_ok = bool(len(full) > 1 and np.isfinite(arousal_variance) and arousal_variance > 0 and not zero_variation_participants)

    design_columns = ["cuteness_centered", "old_new_old", "arousal_centered"]
    design = full[design_columns].apply(pd.to_numeric, errors="coerce").dropna() if set(design_columns).issubset(full.columns) else pd.DataFrame()
    matrix = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)]) if len(design) else np.empty((0, 4))
    rank = int(np.linalg.matrix_rank(matrix)) if len(matrix) else 0
    expected_rank = int(matrix.shape[1]) if len(matrix) else 4
    condition_number = float(np.linalg.cond(matrix)) if len(matrix) else math.nan
    rank_ok = bool(len(matrix) and rank == expected_rank and np.isfinite(condition_number))
    correlation = float(design["cuteness_centered"].corr(design["arousal_centered"])) if len(design) > 1 else math.nan
    vif = _variance_inflation_factors(design) if len(design) else {}
    collinearity_ok = rank_ok

    failed = []
    if not completeness_ok:
        failed.append("DATA_COMPLETENESS")
    if not variation_ok:
        failed.append("WITHIN_PARTICIPANT_VARIATION")
    if not rank_ok:
        failed.append("DESIGN_MATRIX_IDENTIFIABILITY")
    if not collinearity_ok:
        failed.append("CUTENESS_AROUSAL_COLLINEARITY")
    return pd.DataFrame(
        [
            {
                "eligible_primary_rows": len(reduced),
                "rows_with_usable_arousal": len(full),
                "rows_lost_if_arousal_included": len(reduced) - len(full),
                "participants_reduced": len(reduced_participants),
                "participants_with_arousal": len(full_participants),
                "participants_affected": ";".join(participant_loss),
                "items_reduced": len(reduced_items),
                "items_with_arousal": len(full_items),
                "items_affected": ";".join(item_loss),
                "completeness_ok": completeness_ok,
                "participants_with_usable_within_variation": max(0, len(full_participants) - len(zero_variation_participants)),
                "participants_without_usable_within_variation": ";".join(zero_variation_participants),
                "centered_arousal_variance": arousal_variance,
                "variation_ok": variation_ok,
                "design_rank": rank,
                "design_columns": expected_rank,
                "design_condition_number": condition_number,
                "design_identifiable": rank_ok,
                "centered_cuteness_arousal_correlation": correlation,
                "vif_cuteness_centered": vif.get("cuteness_centered", math.nan),
                "vif_old_new_old": vif.get("old_new_old", math.nan),
                "vif_arousal_centered": vif.get("arousal_centered", math.nan),
                "collinearity_diagnostics_acceptable": collinearity_ok,
                "prefit_arousal_feasible": not failed,
                "failed_criteria": ";".join(failed),
                "decision_rule": "no row/participant/item loss; nonzero within-participant variation; full-rank finite design; diagnostics reported; full-model estimability",
                "outcome_significance_used": False,
            }
        ]
    )


def fit_cuteness_mixed_model(trials: pd.DataFrame, settings: Mapping[str, Any]) -> CutenessModelResult:
    """Fit the frozen crossed participant/item model and retain every attempt."""

    validate_cuteness_model_inputs(trials)
    model = primary_model_settings(settings)
    feasibility = evaluate_arousal_feasibility(trials)
    prefit_full = bool(feasibility.loc[0, "prefit_arousal_feasible"])
    full_formula = str(model.get("full_formula", "ERP_mean_uv ~ cuteness_centered + old_new_old + arousal_centered"))
    reduced_formula = str(model.get("reduced_formula", "ERP_mean_uv ~ cuteness_centered + old_new_old"))
    if prefit_full:
        full_data = trials[_bool_series(trials["model_eligible_full"])].copy()
        attempt = _fit_crossed_model(full_data, full_formula, model, arousal_included=True)
        attempts = [_decorate_attempt(attempt, role="full", formula=full_formula, arousal_included=True)]
        if attempt["status"] in {"FIT_OK", "FIT_WARNING"}:
            feasibility.loc[0, "model_estimability_ok"] = True
            feasibility.loc[0, "arousal_included"] = True
            feasibility.loc[0, "arousal_decision_reason"] = "All frozen criteria passed, including full-model estimability."
            return _assemble_result(trials, feasibility, model, full_formula, True, attempt, attempts, selected_attempt_index=0, fallback_reason="")
        reduced_data = trials[_bool_series(trials["model_eligible_reduced"])].copy()
        reduced_attempt = _fit_crossed_model(reduced_data, reduced_formula, model, arousal_included=False)
        attempts.append(_decorate_attempt(reduced_attempt, role="reduced", formula=reduced_formula, arousal_included=False))
        feasibility.loc[0, "model_estimability_ok"] = False
        feasibility.loc[0, "failed_criteria"] = _append_criterion(str(feasibility.loc[0, "failed_criteria"]), "MODEL_ESTIMABILITY")
        if reduced_attempt["status"] in {"FIT_OK", "FIT_WARNING"}:
            feasibility.loc[0, "arousal_included"] = False
            feasibility.loc[0, "arousal_decision_reason"] = "Full model failed while the prespecified reduced model was estimable; MODEL_ESTIMABILITY failed for arousal."
            return _assemble_result(trials, feasibility, model, reduced_formula, False, reduced_attempt, attempts, selected_attempt_index=1, fallback_reason=_fallback_reason(attempts))
        feasibility.loc[0, "arousal_included"] = False
        feasibility.loc[0, "arousal_decision_reason"] = "Neither full nor reduced crossed-intercept model was estimable."
        return _assemble_result(trials, feasibility, model, reduced_formula, False, reduced_attempt, attempts, selected_attempt_index=None, fallback_reason=_fallback_reason(attempts))

    reduced_data = trials[_bool_series(trials.get("model_eligible_reduced", pd.Series(False, index=trials.index)))].copy()
    attempt = _fit_crossed_model(reduced_data, reduced_formula, model, arousal_included=False)
    attempts = [_decorate_attempt(attempt, role="reduced", formula=reduced_formula, arousal_included=False)]
    feasibility.loc[0, "model_estimability_ok"] = attempt["status"] in {"FIT_OK", "FIT_WARNING"}
    feasibility.loc[0, "arousal_included"] = False
    feasibility.loc[0, "arousal_decision_reason"] = "Arousal excluded because frozen pre-fit criterion failed: " + str(feasibility.loc[0, "failed_criteria"])
    return _assemble_result(
        trials,
        feasibility,
        model,
        reduced_formula,
        False,
        attempt,
        attempts,
        selected_attempt_index=0 if attempt["status"] in {"FIT_OK", "FIT_WARNING"} else None,
        fallback_reason="prefit_arousal_infeasible:" + str(feasibility.loc[0, "failed_criteria"]),
    )


def _fit_crossed_model(data: pd.DataFrame, formula: str, model: Mapping[str, Any], *, arousal_included: bool) -> dict[str, Any]:
    n_participants = data["participant"].nunique() if "participant" in data else 0
    n_items = data["item_id"].nunique() if "item_id" in data else 0
    repeated_items = data.groupby("item_id")["participant"].nunique() if {"item_id", "participant"}.issubset(data.columns) else pd.Series(dtype=int)
    if len(data) < 8 or n_participants < 2 or n_items < 2:
        return _failed_attempt("NOT_FIT_INSUFFICIENT_DATA", "At least 8 rows, 2 participants, and 2 items are required.", data)
    if not bool((repeated_items >= 2).any()):
        return _failed_attempt("NOT_FIT_UNSUPPORTED_STRUCTURE", "Item IDs do not repeat across participants; crossed item effects are unsupported.", data)
    fixed_columns = ["cuteness_centered", "old_new_old"] + (["arousal_centered"] if arousal_included else [])
    required_numeric = ["ERP_mean_uv", *fixed_columns]
    numeric = data[required_numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"Non-finite values cannot enter cuteness mixed model columns {required_numeric!r}.")
    matrix = np.column_stack([np.ones(len(data)), data[fixed_columns].to_numpy(dtype=float)])
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        return _failed_attempt("NOT_FIT_RANK_DEFICIENT", "Fixed-effect design matrix is rank deficient.", data)

    try:
        import statsmodels.formula.api as smf

        fit_data = data.copy()
        fit_data["_crossed_group"] = 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = smf.mixedlm(
                formula,
                fit_data,
                groups=fit_data["_crossed_group"],
                re_formula="0",
                vc_formula={"participant": "0 + C(participant)", "item": "0 + C(item_id)"},
            ).fit(reml=False, method=str(model.get("optimizer", "lbfgs")), disp=False)
        warning_text = "; ".join(str(item.message) for item in caught)
        if not bool(getattr(fitted, "converged", False)):
            return _failed_attempt("NOT_FIT_NONCONVERGENT", warning_text or "Optimizer did not converge.", data)
        variance_components = np.asarray(getattr(fitted, "vcomp", []), dtype=float)
        if len(variance_components) < 2 or not np.isfinite(variance_components).all() or bool((variance_components <= 1e-10).any()):
            return _failed_attempt("NOT_FIT_SINGULAR", warning_text or "A crossed random-intercept variance component was zero or non-finite.", data)
        status = "FIT_WARNING" if warning_text else "FIT_OK"
        variance_names = list(getattr(getattr(fitted.model, "exog_vc", None), "names", []))
        return {"status": status, "reason": "", "warnings": warning_text, "fitted": fitted, "data": data, "variance_components": variance_components, "variance_component_names": variance_names}
    except np.linalg.LinAlgError as exc:
        return _failed_attempt("NOT_FIT_SINGULAR", str(exc), data)
    except Exception as exc:
        return _failed_attempt("NOT_FIT_UNSUPPORTED_STRUCTURE", str(exc), data)


def _assemble_result(
    trials: pd.DataFrame,
    feasibility: pd.DataFrame,
    model: Mapping[str, Any],
    formula: str,
    arousal_included: bool,
    attempt: Mapping[str, Any],
    attempts: list[dict[str, Any]],
    *,
    selected_attempt_index: int | None,
    fallback_reason: str,
) -> CutenessModelResult:
    status = str(attempt["status"])
    data = attempt.get("data", pd.DataFrame())
    fitted = attempt.get("fitted")
    attempt_rows = []
    for index, row in enumerate(attempts):
        copied = dict(row)
        copied["attempt_index"] = index
        copied["selected_for_final_result"] = copied["attempt_index"] == selected_attempt_index
        attempt_rows.append(copied)
    attempt_ledger = pd.DataFrame(attempt_rows, columns=ATTEMPT_LEDGER_COLUMNS)
    coefficient_columns = ["model_id", "hypothesis_family", "hypothesis_id", "analysis_status", "term", "coefficient_role", "estimate", "standard_error", "confidence_interval_low", "confidence_interval_high", "test_statistic", "p_value"]
    coefficient_rows: list[dict[str, Any]] = []
    if fitted is not None and status in {"FIT_OK", "FIT_WARNING"}:
        confidence = fitted.conf_int()
        for term, estimate in fitted.fe_params.items():
            standard_error = float(fitted.bse_fe[term])
            coefficient_rows.append(
                {
                    "model_id": model.get("model_id", ""),
                    "hypothesis_family": model.get("hypothesis_family", "cuteness_primary_continuous"),
                    "hypothesis_id": model.get("hypothesis_id", "") if term == "cuteness_centered" else "",
                    "analysis_status": "PRIMARY" if term == "cuteness_centered" else "ADJUSTMENT",
                    "term": term,
                    "coefficient_role": "PRIMARY_COEFFICIENT" if term == "cuteness_centered" else "ADJUSTMENT_COEFFICIENT",
                    "estimate": float(estimate),
                    "standard_error": standard_error,
                    "confidence_interval_low": float(confidence.loc[term, 0]),
                    "confidence_interval_high": float(confidence.loc[term, 1]),
                    "test_statistic": float(estimate / standard_error) if standard_error else math.nan,
                    "p_value": float(fitted.pvalues[term]),
                }
            )
    coefficients = pd.DataFrame(coefficient_rows, columns=coefficient_columns)
    specification = pd.DataFrame(
        [
            {
                "model_id": model.get("model_id", ""),
                "formula": formula,
                "full_formula": model.get("full_formula", ""),
                "reduced_formula": model.get("reduced_formula", ""),
                "dependent_variable": "ERP_mean_uv",
                "primary_predictor": "cuteness_centered",
                "fixed_effects": "cuteness_centered;old_new_old" + (";arousal_centered" if arousal_included else ""),
                "random_effects": "crossed variance-component random intercepts: participant;item_id",
                "random_slopes": "none",
                "interactions": "none",
                "old_new_reference": model.get("old_new_reference", "new"),
                "backend": "statsmodels MixedLM.from_formula",
                "backend_version": version("statsmodels"),
                "crossed_effects_implementation": "single constant group; vc_formula participant=0+C(participant), item=0+C(item_id); re_formula=0",
                "optimizer": model.get("optimizer", "lbfgs"),
                "fit_method": model.get("fit_method", "maximum_likelihood"),
                "arousal_included": arousal_included,
                "analysis_status": "PRIMARY",
                "hypothesis_family": model.get("hypothesis_family", "cuteness_primary_continuous"),
                "hypothesis_id": model.get("hypothesis_id", ""),
                "selected_attempt_index": selected_attempt_index,
                "attempt_count": len(attempts),
                "fallback_occurred": len(attempts) > 1,
                "fallback_reason": fallback_reason,
            }
        ]
    )
    variance_components = attempt.get("variance_components", [])
    variance_names = attempt.get("variance_component_names", [])
    named_variances = dict(zip(variance_names, variance_components))
    diagnostics = pd.DataFrame(
        [
            {
                "model_id": model.get("model_id", ""),
                "model_status": status,
                "converged": status in {"FIT_OK", "FIT_WARNING"},
                "fit_warnings": attempt.get("warnings", ""),
                "failure_reason": attempt.get("reason", ""),
                "n_observations": len(data),
                "n_participants": data["participant"].nunique() if "participant" in data else 0,
                "n_items": data["item_id"].nunique() if "item_id" in data else 0,
                "participant_random_intercept": True,
                "item_random_intercept": True,
                "variance_components": ";".join(str(float(value)) for value in variance_components),
                "participant_random_intercept_variance": float(named_variances.get("participant", math.nan)),
                "item_random_intercept_variance": float(named_variances.get("item", math.nan)),
                "residual_variance": float(getattr(fitted, "scale", math.nan)) if fitted is not None else math.nan,
                "arousal_included": arousal_included,
                "arousal_decision_reason": feasibility.loc[0, "arousal_decision_reason"],
                "selected_attempt_index": selected_attempt_index,
                "attempt_count": len(attempts),
                "fallback_occurred": len(attempts) > 1,
                "fallback_reason": fallback_reason,
            }
        ]
    )
    return CutenessModelResult(
        trial_table=trials,
        arousal_feasibility=feasibility,
        specification=specification,
        coefficients=coefficients,
        diagnostics=diagnostics,
        attempt_ledger=attempt_ledger,
        selected_attempt_index=selected_attempt_index,
        fallback_occurred=len(attempts) > 1,
        fallback_reason=fallback_reason,
    )


def _trial_exclusion_reason(row: pd.Series) -> str:
    reasons = []
    if "continuous_cuteness_old_new_eligible" in row.index and not bool(row.get("continuous_cuteness_old_new_eligible")):
        source_reason = str(row.get("exposure_exclusion_reason", "")).strip()
        reasons.append(source_reason or "source_exposure_ineligible")
    if not bool(row.get("roi_complete", False)):
        reasons.append("missing_primary_roi_channels")
    if pd.isna(row.get("mean_amplitude_uv")):
        reasons.append("missing_primary_erp_amplitude")
    if pd.isna(row.get("cuteness_raw")):
        reasons.append("missing_cuteness")
    if pd.isna(row.get("item_id")) or str(row.get("item_id", "")).strip() == "":
        reasons.append("missing_item_id")
    if pd.isna(row.get("old_new_old")):
        reasons.append("invalid_old_new")
    if pd.isna(row.get("arousal_raw")):
        reasons.append("missing_arousal_full_model_only")
    return ";".join(reasons)


def _groups_with_row_loss(reduced: pd.DataFrame, full: pd.DataFrame, column: str) -> list[str]:
    if column not in reduced.columns:
        return []
    reduced_counts = reduced.groupby(column, dropna=False).size()
    full_counts = full.groupby(column, dropna=False).size() if column in full.columns else pd.Series(dtype=int)
    return sorted(str(value) for value, count in reduced_counts.items() if int(full_counts.get(value, 0)) < int(count))


def _variance_inflation_factors(design: pd.DataFrame) -> dict[str, float]:
    values = design.to_numpy(dtype=float)
    result: dict[str, float] = {}
    for index, column in enumerate(design.columns):
        y = values[:, index]
        others = np.delete(values, index, axis=1)
        x = np.column_stack([np.ones(len(others)), others])
        fitted, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ fitted
        total = y - y.mean()
        denominator = float(total @ total)
        r_squared = 1.0 - float(residual @ residual) / denominator if denominator > 0 else 1.0
        result[str(column)] = math.inf if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)
    return result


def _failed_attempt(status: str, reason: str, data: pd.DataFrame) -> dict[str, Any]:
    return {"status": status, "reason": reason, "warnings": "", "fitted": None, "data": data, "variance_components": [], "variance_component_names": []}


def _decorate_attempt(
    attempt: Mapping[str, Any],
    *,
    role: str,
    formula: str,
    arousal_included: bool,
) -> dict[str, Any]:
    data = attempt.get("data", pd.DataFrame())
    status = str(attempt.get("status", "UNKNOWN"))
    warnings_text = str(attempt.get("warnings", "") or "")
    success = status in {"FIT_OK", "FIT_WARNING"}
    return {
        "attempt_index": 0,
        "model_role": role,
        "formula": formula,
        "arousal_included": bool(arousal_included),
        "status": status,
        "reason_code": "completed" if success else status,
        "reason": str(attempt.get("reason", "") or ""),
        "warnings": warnings_text,
        "convergence_warning_state": "FIT_WARNING" if status == "FIT_WARNING" else ("NONE" if success else "NOT_CONVERGED_OR_NOT_FIT"),
        "success": success,
        "selected_for_final_result": False,
        "n_observations": int(len(data)),
        "n_participants": int(data["participant"].nunique()) if "participant" in data else 0,
        "n_items": int(data["item_id"].nunique()) if "item_id" in data else 0,
    }


def _fallback_reason(attempts: list[Mapping[str, Any]]) -> str:
    if len(attempts) < 2:
        return ""
    full = attempts[0]
    reduced = attempts[1]
    return (
        "full_model_failed:"
        + str(full.get("reason_code", full.get("status", "UNKNOWN")))
        + ";reduced_model_attempted:"
        + str(reduced.get("status", "UNKNOWN"))
    )


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.map(lambda value: False if pd.isna(value) else str(value).strip().lower() in {"true", "1", "yes"}).astype(bool)


def _append_criterion(existing: str, criterion: str) -> str:
    values = [value for value in str(existing).split(";") if value and value.lower() != "nan"]
    if criterion not in values:
        values.append(criterion)
    return ";".join(values)
