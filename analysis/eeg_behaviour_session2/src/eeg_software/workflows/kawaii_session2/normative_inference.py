"""Exploratory Kawaii NORM-INF-01/02: source-eligible ROI inference.

Pure statistical operations. This module does not open EEG or determine source
eligibility. Its caller must use the authenticated corrected-epoch adapter.
No model selection is performed based on outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROI = ("Cz", "CP1", "CP2", "Pz")
WINDOW_S = (0.5, 0.8)
MIN_CELL = 10
SCORE_CENTRE = 5.0
FORMULA = "erp_mean_uv ~ aaps_c + C(verified_old_new)"
VARIANCE_COMPONENTS = {
    "participant": "0 + C(participant)",
    "image": "0 + C(image_filename)",
}

class InferenceStop(ValueError):
    """An explicit invalid input or unidentifiable inference boundary."""


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise InferenceStop(f"Missing required trial columns: {sorted(missing)}")


def validate_trials(raw: pd.DataFrame, *, include_sub012: bool = False) -> pd.DataFrame:
    """Fail closed: *all supplied rows* must have source-verified old/new labels.

    Row-level eligibility filtering is permitted only in the authenticated
    source adapter, never by silently dropping CSV rows here.
    """
    columns = (
        "participant", "recording_part", "trial", "event_sample", "image_filename",
        "M_Kawaii", "erp_mean_uv", "verified_old_new", "source_eligible", "source_binding_sha256",
    )
    _require_columns(raw, columns)
    if raw.empty:
        raise InferenceStop("No verified input trials")
    data = raw.copy().reset_index(drop=True)
    if data[list(columns)].isna().any().any():
        raise InferenceStop("Null accepted-trial identity, score, ERP or source disposition")
    if data.duplicated(["participant", "recording_part", "trial"]).any() or data.duplicated(
        ["participant", "recording_part", "event_sample"]
    ).any():
        raise InferenceStop("Duplicate authenticated trial/event identity")
    if data["participant"].astype(str).str.strip().eq("").any() or data["image_filename"].astype(str).str.strip().eq("").any():
        raise InferenceStop("Blank participant or image identity")
    if data["source_eligible"].map(type).ne(bool).any() or not data["source_eligible"].all():
        raise InferenceStop("Input contains non-eligible or non-Boolean source disposition")
    if not data["verified_old_new"].isin(["old", "new"]).all():
        raise InferenceStop("Unverified old/new labels; no label may be imputed")
    if not include_sub012 and "sub012" in set(data["participant"]):
        raise InferenceStop("sub012 not authorised in seven-person primary analysis")
    for column in ("M_Kawaii", "erp_mean_uv", "trial", "event_sample"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise InferenceStop(f"Non-finite numeric input: {column}")
    if not data["M_Kawaii"].between(1, 9).all():
        raise InferenceStop("AAPS score outside 1–9")
    if (data["trial"] <= 0).any() or (data["event_sample"] <= 0).any():
        raise InferenceStop("Invalid trial/event identity")
    if not np.equal(data["trial"], np.floor(data["trial"])).all():
        raise InferenceStop("Non-integral trial identity")
    # Within an item, AAPS score must be identical in every participant.
    if data.groupby("image_filename")["M_Kawaii"].nunique().gt(1).any():
        raise InferenceStop("AAPS score varies for the same stimulus identity")
    if data.groupby("participant")["source_binding_sha256"].nunique().gt(1).any():
        raise InferenceStop("Participant source digest varies across rows")
    import re
    if not data["source_binding_sha256"].astype(str).map(
        lambda x: re.fullmatch(r"[0-9a-fA-F]{64}", x) is not None
    ).all():
        raise InferenceStop("Invalid authenticated source binding")
    data["aaps_c"] = data["M_Kawaii"] - SCORE_CENTRE
    data["aaps_category"] = np.select(
        [data.M_Kawaii <= 5.0, data.M_Kawaii > 6.5],
        ["lower_aaps", "higher_aaps"], default="unclassified",
    )
    if (data.aaps_category == "unclassified").any():
        raise InferenceStop("Score in unrepresented 5.0–6.5 gap; categorical membership undefined")
    return data


def category_contrasts(table: pd.DataFrame, *, min_cell: int = MIN_CELL) -> tuple[pd.DataFrame, dict]:
    """NORM-INF-01: mean within participant, then participant-level inference."""
    if min_cell != MIN_CELL:
        raise InferenceStop("Unreviewed minimum-trial change")
    data = validate_trials(table, include_sub012="sub012" in set(table.participant))
    group = data.groupby(["participant", "aaps_category"], sort=True)["erp_mean_uv"].agg(["mean", "count"]).unstack()
    categories = ("lower_aaps", "higher_aaps")
    if any((metric, category) not in group for metric in ("mean", "count") for category in categories):
        raise InferenceStop("Both categories must be present for all participants")
    if group[("count", "lower_aaps")].isna().any() or group[("count", "higher_aaps")].isna().any():
        raise InferenceStop("Participant missing AAPS category")
    if (group.loc[:, [("count", cat) for cat in categories]] < min_cell).any().any():
        raise InferenceStop("Participant/category below predefined 10-trial minimum")
    contrasts = pd.DataFrame({
        "participant": group.index.astype(str),
        "lower_n": group[("count", "lower_aaps")].to_numpy(dtype=int),
        "higher_n": group[("count", "higher_aaps")].to_numpy(dtype=int),
        "lower_uv": group[("mean", "lower_aaps")].to_numpy(dtype=float),
        "higher_uv": group[("mean", "higher_aaps")].to_numpy(dtype=float),
    })
    contrasts["higher_minus_lower_uv"] = contrasts.higher_uv - contrasts.lower_uv
    values = contrasts.higher_minus_lower_uv.to_numpy(dtype=float)
    n = len(values)
    if n < 3 or np.std(values, ddof=1) <= 0:
        raise InferenceStop("At least 3 participants with nondegenerate contrasts are required")
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    ttest = stats.ttest_1samp(values, 0, alternative="two-sided")
    # Exact sign-flip sensitivity (all 2^n assignments, no pseudoreplication).
    if n > 16:
        raise InferenceStop("Exact sensitivity enumeration capped at 16 participants")
    null_means = np.asarray([np.mean(values * np.asarray(signs)) for signs in product((-1,1),repeat=n)])
    signflip_p = float(np.mean(np.abs(null_means) >= abs(mean) - 1e-12))
    result = {
        "endpoint":"NORM-INF-01", "estimand":"participant mean (higher AAPS - lower AAPS), µV",
        "n_participants":n, "n_trials":len(data), "n_items":int(data.image_filename.nunique()),
        "contrast_mean_uv":mean, "contrast_median_uv":float(np.median(values)),
        "positive_participants":int(np.sum(values>0)), "negative_participants":int(np.sum(values<0)),
        "t":float(ttest.statistic), "df":n-1, "p_unadjusted":float(ttest.pvalue),
        "ci_95_uv":[float(ci[0]),float(ci[1])], "signflip_p_sensitivity":signflip_p,
        "cohort":"eight_verified_sensitivity" if "sub012" in set(data.participant) else "seven_primary",
        "min_cell":min_cell,
    }
    return contrasts, result


def continuous_mixed_model(table: pd.DataFrame) -> dict:
    """NORM-INF-02: ML crossed participant/item intercepts; one L-BFGS fit.

    AAPS score is an image-level covariate; statistical support for its slope
    relies on the crossed random-intercept assumptions. No optimiser search.
    """
    data = validate_trials(table, include_sub012="sub012" in set(table.participant))
    if data.participant.nunique() < 3 or data.image_filename.nunique() < 10:
        raise InferenceStop("Insufficient independent participant/item levels")
    if data.aaps_c.std() < 0.01:
        raise InferenceStop("AAPS score has insufficient variation")
    import statsmodels.formula.api as smf
    frame = data.copy()
    frame["crossed_group"] = "all"
    model = smf.mixedlm(
        FORMULA, data=frame, groups=frame["crossed_group"],
        re_formula="0", vc_formula=VARIANCE_COMPONENTS, use_sparse=True,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = model.fit(reml=False, method="lbfgs", maxiter=800, disp=False)
    note = [f"{type(item.message).__name__}: {item.message}" for item in captured]
    if not bool(result.converged):
        raise InferenceStop("Prespecified crossed mixed-effects fit did not converge; no alternate optimiser used")
    term = "aaps_c"
    estimate, se, p = (float(result.fe_params[term]),float(result.bse_fe[term]),float(result.pvalues[term]))
    if not np.isfinite([estimate,se,p]).all() or se <= 0:
        raise InferenceStop("Crossed model returned an invalid AAPS coefficient or uncertainty")
    ci = result.conf_int().loc[term].to_list()
    if not np.isfinite(ci).all():
        raise InferenceStop("Crossed model coefficient CI invalid")
    variance = dict(zip(model.exog_vc.names, (float(x) for x in result.vcomp)))
    if any(not np.isfinite(x) or x < 0 for x in variance.values()):
        raise InferenceStop("Crossed model variance estimates invalid")
    return {
        "endpoint":"NORM-INF-02", "estimand":"µV per +1 AAPS M_Kawaii point, adjusted for verified old/new",
        "cohort":"eight_verified_sensitivity" if "sub012" in set(data.participant) else "seven_primary",
        "n_participants":int(data.participant.nunique()), "n_items":int(data.image_filename.nunique()), "n_trials":len(data),
        "formula":FORMULA, "random_intercepts":VARIANCE_COMPONENTS,
        "fit":"ML, L-BFGS, single predeclared optimiser", "converged":True,
        "coefficient_uv_per_point":estimate, "standard_error":se,
        "ci_95_uv_per_point":[float(ci[0]),float(ci[1])],
        "p_unadjusted":p, "variance_components":variance,
        "residual_variance":float(result.scale), "warnings":note,
        "aaps_center":SCORE_CENTRE,
    }


def holm_two(p_categorical: float, p_continuous: float) -> dict:
    """Holm family for exactly the two planned primary endpoints."""
    p = np.asarray([p_categorical,p_continuous], dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise InferenceStop("Cannot adjust invalid/missing primary p-values")
    order = np.argsort(p)
    adj = np.empty(2)
    adj[order[0]] = min(1.,2*float(p[order[0]]))
    adj[order[1]] = max(adj[order[0]],float(p[order[1]]))
    return {"NORM-INF-01":float(adj[0]),"NORM-INF-02":float(adj[1]),"method":"Holm across two exploratory co-primary hypotheses"}
