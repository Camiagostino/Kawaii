from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd

from .study_definition import KawaiiTriggerDefinition


AVAILABLE_FROM_EXISTING_EPOCH_OUTPUT = "AVAILABLE_FROM_EXISTING_EPOCH_OUTPUT"
RECONSTRUCTABLE_OUTCOME_BLIND_FROM_EVENTS = "RECONSTRUCTABLE_OUTCOME_BLIND_FROM_EVENTS"
NOT_RECONSTRUCTABLE_WITH_CURRENT_EVIDENCE = "NOT_RECONSTRUCTABLE_WITH_CURRENT_EVIDENCE"


def image_event_samples_from_raw(
    raw, *, trigger_definition: KawaiiTriggerDefinition | None = None
) -> np.ndarray:
    """Extract image-onset trigger samples without reading task outcomes."""

    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    image_codes = set(definition.image_event_id.values())
    samples: list[int] = []
    for description, mne_code in event_ids.items():
        match = re.search(r"(\d+)$", str(description))
        trigger_code = int(match.group(1)) if match else int(mne_code)
        if trigger_code in image_codes:
            samples.extend(events[events[:, 2] == int(mne_code), 0].astype(int).tolist())
    return np.asarray(sorted(samples), dtype=int)


def existing_epoch_samples(mapping_path: Path) -> np.ndarray:
    """Read only event location and epoch-eligibility fields from an existing mapping."""

    mapping = pd.read_csv(mapping_path, usecols=["event_sample", "retain_for_epoching"])
    retained = mapping["retain_for_epoching"].astype(str).str.lower().eq("true")
    return (
        pd.to_numeric(mapping.loc[retained, "event_sample"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .to_numpy()
    )


def boundary_valid_event_samples(
    event_samples: Sequence[int],
    *,
    raw_first_samp: int,
    n_times: int,
    sfreq: float,
    epoch_tmin_s: float,
    epoch_tmax_s: float,
) -> np.ndarray:
    """Keep reconstructed events whose complete epoch interval is present in the raw."""

    events = np.asarray(event_samples, dtype=int)
    start = events + int(np.floor(float(epoch_tmin_s) * float(sfreq)))
    stop = events + int(np.ceil(float(epoch_tmax_s) * float(sfreq)))
    raw_stop = int(raw_first_samp) + int(n_times)
    return events[(start >= int(raw_first_samp)) & (stop <= raw_stop)]


def mapping_provenance(mapping_path: Path, reconstructed_event_count: int) -> str:
    if mapping_path.exists():
        return AVAILABLE_FROM_EXISTING_EPOCH_OUTPUT
    if int(reconstructed_event_count) > 0:
        return RECONSTRUCTABLE_OUTCOME_BLIND_FROM_EVENTS
    return NOT_RECONSTRUCTABLE_WITH_CURRENT_EVIDENCE


def temporal_extent(temporal_category: str) -> str:
    return {
        "LOCAL_TRANSIENT": "ISOLATED",
        "INTERMITTENT": "INTERMITTENT",
        "RECURRENT": "RECURRENT_SPARSE",
        "PROLONGED": "PROLONGED",
        "BROADLY_PERSISTENT": "BROADLY_PERSISTENT",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT",
    }[str(temporal_category)]


def task_segment_burden(fraction: float | None) -> str:
    """Describe diagnostic event burden; boundaries are not production thresholds."""

    if fraction is None or not np.isfinite(float(fraction)):
        return "UNKNOWN"
    value = float(fraction)
    if value == 0:
        return "NEGLIGIBLE"
    if value < 0.05:
        return "LOW"
    if value < 0.20:
        return "MODERATE"
    return "HIGH"


def evidence_families(
    case: Mapping[str, object],
    metric_affected_counts: Mapping[str, int],
) -> tuple[str, ...]:
    """Return distinct diagnostic families, not votes or automatic authority."""

    families = []
    time_domain_flags = (
        "scale_dispersion_flag", "peak_to_peak_flag", "flat_dropout_flag", "clipping_flag"
    )
    spectral_flags = (
        "line_noise_relative_flag", "high_frequency_relative_flag", "drift_relative_flag"
    )
    local_flags = ("local_absolute_flag", "local_robust_relative_flag")
    if _truthy(case.get("prior_approved_independent_evidence")) or any(
        int(metric_affected_counts.get(flag, 0)) > 0 for flag in time_domain_flags
    ):
        families.append("TIME_DOMAIN_INDEPENDENT")
    if any(int(metric_affected_counts.get(flag, 0)) > 0 for flag in spectral_flags):
        families.append("SPECTRAL_RELATIVE")
    if (
        _truthy(case.get("prior_local_absolute_candidate"))
        or _truthy(case.get("prior_local_robust_relative_candidate"))
        or any(int(metric_affected_counts.get(flag, 0)) > 0 for flag in local_flags)
    ):
        families.append("LOCAL_CORRELATION")
    if _truthy(case.get("prior_spatial_primary_candidate")):
        families.append("SPATIAL_PREDICTABILITY")
    return tuple(families)


def corroboration_category(families: Sequence[str]) -> str:
    if not families:
        return "NONE"
    if len(families) == 1:
        return "ONE_EVIDENCE_TYPE"
    return "MULTIPLE_EVIDENCE_TYPES"


def abnormality_evidence(assisted_rating: str) -> str:
    return {
        "NORMAL": "NONE",
        "QUESTIONABLE": "WEAK_QUESTIONABLE",
        "CLEARLY_ABNORMAL": "CLEAR",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT",
    }[str(assisted_rating)]


def whole_channel_concern(abnormality: str, extent: str) -> str:
    """Dry-run synthesis for review support; never an interpolation decision."""

    if abnormality == "NONE":
        return "NOT_SUPPORTED"
    if extent == "INSUFFICIENT" or abnormality == "INSUFFICIENT":
        return "INSUFFICIENT"
    if abnormality == "CLEAR" and extent in {"PROLONGED", "BROADLY_PERSISTENT"}:
        return "SUPPORTED"
    if extent in {"RECURRENT_SPARSE", "PROLONGED", "BROADLY_PERSISTENT"}:
        return "POSSIBLE"
    return "NOT_SUPPORTED"


def local_handling_plausibility(extent: str, burden: str, concern: str) -> str:
    if extent == "INSUFFICIENT" or burden == "UNKNOWN":
        return "INSUFFICIENT"
    if concern == "SUPPORTED" or burden == "HIGH":
        return "NO"
    if extent in {"ISOLATED", "INTERMITTENT"}:
        return "YES"
    if extent == "RECURRENT_SPARSE" and burden in {"NEGLIGIBLE", "LOW"}:
        return "YES"
    return "PARTIAL"


def residual_insufficiency(concern: str) -> str:
    if concern == "INSUFFICIENT":
        return "YES"
    if concern == "POSSIBLE":
        return "PARTIAL"
    return "NO"


def review_disposition(abnormality: str, concern: str, local_handling: str) -> str:
    """Suggest the next human-review posture, never an automatic cleaning action."""

    if abnormality == "NONE":
        return "RETAIN_CHANNEL"
    if concern == "SUPPORTED":
        return "WHOLE_CHANNEL_INTERPOLATION_CONCERN"
    if concern in {"POSSIBLE", "INSUFFICIENT"}:
        return "ESCALATE_FOR_CLOSER_HUMAN_REVIEW"
    if local_handling == "YES":
        return "RETAIN_CHANNEL_PLUS_LOCAL_SEGMENT_EPOCH_HANDLING"
    return "RETAIN_CHANNEL"


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
