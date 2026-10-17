from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


ALLOWED_CLASSIFICATIONS = {
    "NORMAL",
    "QUESTIONABLE",
    "CLEARLY_ABNORMAL",
    "INSUFFICIENT_EVIDENCE",
}
ALLOWED_REASONS = {
    "high_frequency_noise",
    "line_noise",
    "amplitude_variance",
    "flatness_dropout",
    "drift",
    "discontinuity",
    "focal_transient",
    "otherwise_normal",
    "other_explain_in_notes",
}
RESPONSE_COLUMNS = (
    "case_id",
    "participant",
    "channel",
    "review_classification",
    "structured_reasons",
    "reviewer_confidence",
    "reviewer_notes",
)


def substantive_review_rows(responses: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Separate completed case rows from trailing entirely blank CSV rows."""

    values = responses.copy()
    blank = values[list(RESPONSE_COLUMNS)].apply(
        lambda column: column.fillna("").astype(str).str.strip().eq("")
    ).all(axis=1)
    return values.loc[~blank].copy().reset_index(drop=True), int(blank.sum())


def split_reason_tokens(value: object) -> tuple[str, ...]:
    """Parse semicolon- or newline-separated reasons without changing their text."""

    if pd.isna(value):
        return ()
    normalized = str(value).replace("\r\n", "\n").replace("\r", "\n").replace(";", "\n")
    return tuple(token.strip() for token in normalized.split("\n") if token.strip())


def validate_completed_review(
    responses: pd.DataFrame,
    public_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate completeness, schema, identifiers, ratings, and held-back-field absence."""

    missing_columns = sorted(set(RESPONSE_COLUMNS) - set(responses.columns))
    extra_columns = sorted(set(responses.columns) - set(RESPONSE_COLUMNS))
    if missing_columns:
        raise ValueError(f"completed review is missing required columns: {missing_columns}")
    substantive, ignored_blank_rows = substantive_review_rows(responses)
    expected_ids = set(public_manifest["case_id"].astype(str))
    actual_ids = set(substantive["case_id"].astype(str))
    duplicated = sorted(substantive.loc[substantive["case_id"].duplicated(False), "case_id"].astype(str).unique())
    mismatched_identity = substantive.merge(
        public_manifest[["case_id", "participant", "channel"]],
        on="case_id",
        suffixes=("_response", "_manifest"),
        how="inner",
    )
    mismatched_identity = mismatched_identity[
        mismatched_identity["participant_response"].ne(mismatched_identity["participant_manifest"])
        | mismatched_identity["channel_response"].ne(mismatched_identity["channel_manifest"])
    ]
    classifications = substantive["review_classification"].fillna("").astype(str).str.strip()
    invalid_classifications = sorted(set(classifications) - ALLOWED_CLASSIFICATIONS)
    missing_confidence = sorted(
        substantive.loc[
            substantive["reviewer_confidence"].fillna("").astype(str).str.strip().eq(""), "case_id"
        ].astype(str)
    )
    token_rows = []
    for row in substantive.itertuples():
        tokens = split_reason_tokens(row.structured_reasons)
        invalid = tuple(token for token in tokens if token not in ALLOWED_REASONS)
        token_rows.append({
            "case_id": row.case_id,
            "structured_reason_tokens": ";".join(tokens),
            "invalid_reason_tokens": ";".join(invalid),
            "structured_reasons_valid": not invalid and bool(tokens),
        })
    token_table = pd.DataFrame(token_rows)
    substantive = substantive.merge(token_table, on="case_id", validate="one_to_one")

    checks = pd.DataFrame([
        _check("required_columns_present", not missing_columns, ";".join(missing_columns)),
        _check("no_unexpected_or_held_back_columns", not extra_columns, ";".join(extra_columns)),
        _check("exact_case_ids_BC001_to_BC020", actual_ids == expected_ids, _set_difference(expected_ids, actual_ids)),
        _check("no_duplicate_case_ids", not duplicated, ";".join(duplicated)),
        _check("participant_channel_match_public_manifest", mismatched_identity.empty, ";".join(mismatched_identity["case_id"].astype(str))),
        _check("classifications_allowed", not invalid_classifications, ";".join(invalid_classifications)),
        _check("structured_reasons_allowed", bool(substantive["structured_reasons_valid"].all()), _invalid_reason_detail(substantive)),
        _check("confidence_recorded", not missing_confidence, ";".join(missing_confidence)),
        _check("held_back_algorithm_fields_absent", not (set(responses.columns) & {
            "selection_stratum", "edge_status", "automatic_candidate_status",
            "mean_local_score", "robust_z3_two_windows", "spatial_primary_candidate",
        }), "response schema contains no held-back algorithm fields"),
        _check("trailing_blank_rows_ignored_not_ratings", True, str(ignored_blank_rows)),
    ])
    return substantive, checks


def build_reconciliation(
    responses: pd.DataFrame,
    *,
    held_back_key: pd.DataFrame,
    relative: pd.DataFrame,
    independent: pd.DataFrame,
    spatial: pd.DataFrame,
    expanded_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Join assisted ratings to held-back algorithm evidence without changing ratings."""

    key_columns = [
        "case_id", "participant", "channel", "selection_stratum", "edge_status",
        "automatic_candidate_status",
    ]
    local_columns = [
        "participant", "channel", "absolute_0_40_two_windows", "robust_z3_two_windows",
        "mean_local_score", "minimum_local_score", "maximum_within_window_low_robust_z",
    ]
    independent_columns = [
        "participant", "channel", "flat_proposal", "extreme_amplitude_proposal",
        "clipping_proposal", "robust_deviation_proposal", "std_uv", "peak_to_peak_uv",
        "max_robust_deviation_z", "clipping_fraction", "maximum_window_line_noise_ratio",
    ]
    result = responses.merge(
        held_back_key[key_columns], on=["case_id", "participant", "channel"], validate="one_to_one"
    ).merge(
        relative[local_columns], on=["participant", "channel"], validate="one_to_one"
    ).merge(
        independent[independent_columns], on=["participant", "channel"], validate="one_to_one"
    ).merge(
        spatial, on=["case_id", "participant", "channel"], validate="one_to_one"
    )
    independent_flags = [
        "flat_proposal", "extreme_amplitude_proposal", "clipping_proposal", "robust_deviation_proposal"
    ]
    result["approved_independent_any"] = result[independent_flags].astype(bool).any(axis=1)
    result["geometry_aware_edge_status"] = np.where(
        result["geometry_status"].eq("GEOMETRY_UNSCORABLE"),
        "GEOMETRY_UNSCORABLE",
        result["edge_status"],
    )
    production = _production_evidence(expanded_scores)
    result = result.merge(production, on=["participant", "channel"], validate="one_to_one")

    recurrence = relative.groupby("channel", as_index=False).agg(
        local_absolute_recurrence_participant_count=("absolute_0_40_two_windows", "sum"),
        robust_relative_recurrence_participant_count=("robust_z3_two_windows", "sum"),
    )
    result = result.merge(recurrence, on="channel", validate="many_to_one")
    result.insert(3, "review_provenance", "ASSISTED_BLINDED_HUMAN_REVIEW_NOT_INDEPENDENT_EXPERT_GROUND_TRUTH")
    return result.sort_values("case_id").reset_index(drop=True)


def reconciliation_summaries(reconciliation: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create compact descriptive summaries; human ratings are not ground truth."""

    rating_order = ["NORMAL", "QUESTIONABLE", "CLEARLY_ABNORMAL", "INSUFFICIENT_EVIDENCE"]
    flags = {
        "production_all_channel": "production_candidate",
        "local_absolute": "absolute_0_40_two_windows",
        "local_robust_relative": "robust_z3_two_windows",
        "approved_independent": "approved_independent_any",
        "spatial_predictability": "spatial_primary_candidate",
    }
    algorithm_rows = []
    for rating in rating_order:
        group = reconciliation[reconciliation["review_classification"].eq(rating)]
        for label, column in flags.items():
            scorable = group
            if label == "spatial_predictability":
                scorable = group[group["geometry_status"].eq("SCORABLE")]
            algorithm_rows.append({
                "review_classification": rating,
                "algorithm_evidence": label,
                "review_case_count": len(group),
                "algorithm_scorable_case_count": len(scorable),
                "flagged_case_count": int(scorable[column].fillna(False).astype(bool).sum()),
                "flagged_case_fraction": (
                    float(scorable[column].fillna(False).astype(bool).mean()) if len(scorable) else np.nan
                ),
            })
    distribution = (
        reconciliation.groupby(["review_classification", "geometry_aware_edge_status"], as_index=False)
        .size()
        .rename(columns={"size": "case_count"})
    )
    distribution["classification_total"] = distribution.groupby("review_classification")["case_count"].transform("sum")
    distribution["within_classification_fraction"] = distribution["case_count"] / distribution["classification_total"]
    reason_rows = []
    for row in reconciliation.itertuples():
        for token in split_reason_tokens(row.structured_reasons):
            reason_rows.append({
                "case_id": row.case_id,
                "review_classification": row.review_classification,
                "reason_token_verbatim": token,
                "reason_token_allowed": token in ALLOWED_REASONS,
            })
    reasons = pd.DataFrame(reason_rows)
    reason_summary = (
        reasons.groupby(["reason_token_verbatim", "reason_token_allowed"], as_index=False)
        .size()
        .rename(columns={"size": "case_count"})
    )
    selection = (
        reconciliation.groupby(["selection_stratum", "review_classification"], as_index=False)
        .size()
        .rename(columns={"size": "case_count"})
    )
    return {
        "algorithm_by_rating": pd.DataFrame(algorithm_rows),
        "rating_distribution": distribution,
        "reason_summary": reason_summary,
        "selection_by_rating": selection,
    }


def _production_evidence(scores: pd.DataFrame) -> pd.DataFrame:
    baseline = scores[scores["method"].eq("baseline_raw_all_abs")].copy()
    baseline["score"] = pd.to_numeric(baseline["score"], errors="coerce")
    rows = []
    for (participant, channel), group in baseline.groupby(["participant", "channel"], sort=True):
        ordered = group.sort_values("window_index")
        values = ordered["score"].to_numpy(dtype=float)
        failed = int(np.sum(np.isfinite(values) & (values < 0.4)))
        rows.append({
            "participant": participant,
            "channel": channel,
            "production_scores_by_window": ";".join("" if not np.isfinite(value) else f"{value:.9g}" for value in values),
            "production_failed_window_count": failed,
            "production_candidate": failed >= 1,
        })
    return pd.DataFrame(rows)


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


def _set_difference(expected: Iterable[str], actual: Iterable[str]) -> str:
    expected_set, actual_set = set(expected), set(actual)
    return f"missing={sorted(expected_set - actual_set)};unexpected={sorted(actual_set - expected_set)}"


def _invalid_reason_detail(responses: pd.DataFrame) -> str:
    invalid = responses[responses["invalid_reason_tokens"].ne("")]
    return ";".join(f"{row.case_id}:{row.invalid_reason_tokens}" for row in invalid.itertuples())
