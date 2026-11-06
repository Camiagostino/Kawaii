"""Isolated, descriptive NORM-02 cuteness ERP aggregation core.

Scientific boundary: accepts *already validated* Gate 3C epoch-derived ROI
traces (microvolts), never filters, baselines, interpolates or changes epochs.
This module does not load FIF files or establish source/publication authority.
The caller must validate the exact published epoch/metadata binding first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping, Sequence
import re

import numpy as np
import pandas as pd

ROI = ("Cz", "CP1", "CP2", "Pz")
NORMATIVE = ("lower_aaps", "higher_aaps")
SUBJECTIVE = ("low", "medium_low", "medium_high", "high")
DEFAULT_MINIMUM_TRIALS = 10


class Norm02ValidationError(ValueError):
    """Fail closed on unverified/ambiguous trial metadata or signal shape."""


@dataclass(frozen=True)
class AcceptedPartTraces:
    """Traces produced by a separate source-validated Gate 3C loader.

    traces_uv: shape (retained_epochs, time_samples), already spatially
        averaged over the exact Cz, CP1, CP2, Pz ROI in µV.
    metadata: row-aligned, accepted Gate 3C metadata; never behavioural-log
        rows or the smaller old/new-dependent model subset.
    """

    participant: str
    recording_part: str
    times_s: np.ndarray
    traces_uv: np.ndarray
    metadata: pd.DataFrame
    source_binding_sha256: str


@dataclass(frozen=True)
class Norm02Result:
    waveforms: pd.DataFrame
    counts: pd.DataFrame
    contributions: pd.DataFrame
    summary: pd.DataFrame


def normative_category(value: object) -> str:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise Norm02ValidationError(f"Non-numeric normative cuteness score: {value!r}") from exc
    if not np.isfinite(number):
        raise Norm02ValidationError("Normative cuteness score must be finite")
    if 1 <= number <= 5:
        return NORMATIVE[0]
    if 6.5 < number <= 9:
        return NORMATIVE[1]
    raise Norm02ValidationError(f"Score {number} is outside the verified two-pool AAPS score ranges")


def subjective_category(value: object) -> str:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise Norm02ValidationError(f"Missing/non-numeric subjective rating: {value!r}") from exc
    if not np.isfinite(number) or not number.is_integer() or not 1 <= number <= 9:
        raise Norm02ValidationError(f"Subjective cuteness rating is not an integer on 1–9: {value!r}")
    rating = int(number)
    if rating <= 2:
        return "low"
    if rating <= 4:
        return "medium_low"
    if rating <= 6:
        return "medium_high"
    return "high"


def _checked_part(part: AcceptedPartTraces, *, reference: Mapping[str, float]) -> tuple[np.ndarray, pd.DataFrame]:
    participant = str(part.participant).strip()
    if (
        not participant
        or participant in {".", ".."}
        or any(character in participant for character in ("/", "\\"))
        or any(ord(character) < 32 for character in participant)
        or not re.fullmatch(r"part\d+", part.recording_part)
    ):
        raise Norm02ValidationError("Invalid participant/part identity")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", part.source_binding_sha256):
        raise Norm02ValidationError("Missing/invalid source-binding digest; validation must precede aggregation")
    times = np.asarray(part.times_s, dtype=float)
    data = np.asarray(part.traces_uv, dtype=float)
    if times.ndim != 1 or len(times) < 3 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise Norm02ValidationError("Invalid epoch time vector")
    if abs(times[0] + .2) > 0.002 or abs(times[-1] - 1.2) > 0.002:
        raise Norm02ValidationError("Unexpected non-canonical epoch interval")
    if data.ndim != 2 or data.shape != (len(part.metadata), len(times)) or not len(data):
        raise Norm02ValidationError("ROI trace shape/epoch metadata count mismatch")
    if not np.isfinite(data).all():
        raise Norm02ValidationError("Non-finite ROI data")
    required = ["image_filename", "M_Kawaii", "kawaii_rating", "trial", "event_sample"]
    missing = [x for x in required if x not in part.metadata]
    if missing:
        raise Norm02ValidationError(f"Missing accepted metadata columns: {missing}")
    md = part.metadata.copy().reset_index(drop=True)
    trial = pd.to_numeric(md["trial"], errors="coerce")
    event = pd.to_numeric(md["event_sample"], errors="coerce")
    if trial.isna().any() or event.isna().any() or trial.duplicated().any() or event.duplicated().any():
        raise Norm02ValidationError("Missing or duplicate trial/event identities within part")
    if not (trial.gt(0).all() and event.gt(0).all()):
        raise Norm02ValidationError("Invalid trial/event index")
    if len(reference) == 0:
        raise Norm02ValidationError("AAPS reference table is empty")
    normative = []
    subjective = []
    for item, score, self_rating in zip(md["image_filename"], md["M_Kawaii"], md["kawaii_rating"]):
        item = str(item).strip()
        if not item:
            raise Norm02ValidationError("Source image ID must be non-empty")
        if item not in reference:
            raise Norm02ValidationError(f"Image absent from pinned AAPS crosswalk: {item}")
        try:
            observed = float(score)
        except (TypeError, ValueError) as exc:
            raise Norm02ValidationError(f"Invalid metadata normative score for {item}") from exc
        if not np.isfinite(observed) or abs(observed - float(reference[item])) > 1e-8:
            raise Norm02ValidationError(f"AAPS score mismatch for {item}")
        normative.append(normative_category(reference[item]))
        subjective.append(subjective_category(self_rating))
    md["norm02_normative"] = normative
    md["norm02_subjective"] = subjective
    return data, md


def aggregate_descriptive(
    parts: Sequence[AcceptedPartTraces],
    *,
    aaps_reference: Mapping[str, float],
    minimum_trials: int = DEFAULT_MINIMUM_TRIALS,
) -> Norm02Result:
    """Within-participant nave-weighted means, then equal participant weights.

    A participant/category with <minimum_trials trials is not a contributor;
    all retained trials are counted in QC regardless of contributor status.
    Not a test of hypotheses, and not a statistical model.
    """
    if not isinstance(minimum_trials, int) or minimum_trials < 1:
        raise Norm02ValidationError("minimum_trials must be a positive integer")
    if not parts:
        raise Norm02ValidationError("No accepted participant parts")
    keys = [(part.participant, part.recording_part) for part in parts]
    if len(keys) != len(set(keys)):
        raise Norm02ValidationError("Duplicate participant/part")
    checked = []
    template_times = None
    for part in parts:
        data, md = _checked_part(part, reference=aaps_reference)
        if template_times is None:
            template_times = np.asarray(part.times_s, dtype=float)
        elif not np.array_equal(template_times, np.asarray(part.times_s, dtype=float)):
            raise Norm02ValidationError("Inconsistent time grids across participant parts")
        checked.append((part, data, md))
    assert template_times is not None
    subjects = sorted({part.participant for part in parts})
    count_rows = []
    contribution_rows = []
    group_traces = {}
    for family, categories, column in (
        ("normative", NORMATIVE, "norm02_normative"),
        ("subjective", SUBJECTIVE, "norm02_subjective"),
    ):
        for subject in subjects:
            for category in categories:
                participant_traces = []
                participant_weights = []
                part_counts = []
                for part, data, md in checked:
                    if part.participant != subject:
                        continue
                    mask = md[column].eq(category).to_numpy(dtype=bool)
                    n = int(mask.sum())
                    part_counts.append(n)
                    if n:
                        participant_traces.append(data[mask].mean(axis=0))
                        participant_weights.append(n)
                count = sum(part_counts)
                contributes = count >= minimum_trials
                count_rows.append(dict(family=family, category=category, participant=subject,
                                       retained_trials=count, eligible_for_group=contributes,
                                       n_parts=len(part_counts)))
                if contributes:
                    trace = np.average(np.stack(participant_traces), axis=0, weights=participant_weights)
                    group_traces[(family, category, subject)] = trace
                    contribution_rows.append(dict(family=family, category=category, participant=subject,
                                                  contributed_trials=count, parts_with_trials=len(participant_weights)))
    waveform_rows = []
    summary_rows = []
    counts = pd.DataFrame(count_rows)
    for family, categories in (("normative", NORMATIVE), ("subjective", SUBJECTIVE)):
        for category in categories:
            contributors = [group_traces[(family, category, sub)] for sub in subjects
                            if (family, category, sub) in group_traces]
            total = int(counts.loc[(counts.family == family) & (counts.category == category), "retained_trials"].sum())
            contributed = int(counts.loc[(counts.family == family) & (counts.category == category) & (counts.eligible_for_group), "retained_trials"].sum())
            summary_rows.append(dict(family=family, category=category, n_participants=len(contributors),
                                     retained_trials=total, contributed_trials=contributed,
                                     excluded_trials=total-contributed, minimum_trials=minimum_trials))
            if not contributors:
                continue
            mean = np.mean(np.stack(contributors), axis=0)  # deliberately equal participant weight
            for t, amp in zip(template_times, mean):
                waveform_rows.append(dict(family=family, category=category,
                                          time_s=float(t), mean_amplitude_uv=float(amp),
                                          n_participants=len(contributors)))
    return Norm02Result(waveforms=pd.DataFrame(waveform_rows),
                        counts=counts, contributions=pd.DataFrame(contribution_rows),
                        summary=pd.DataFrame(summary_rows))


def accepted_roi_traces_uv(
    epochs: object, *, authenticated_interpolated_roi: Collection[str] = (),
) -> np.ndarray:
    """Take the existing unmodified MNE Epochs' four-channel ROI, in µV.

    Requires exact four-channel availability. An MNE bad flag may persist after
    approved Gate 3C interpolation (reset_bads=False): permit that flag ONLY
    if the source adapter has authenticated the published interpolation record.
    The caller MUST validate publication identity and epoch/metadata binding.
    """
    baseline = getattr(epochs, "baseline", None)
    if baseline is None or len(baseline) != 2 or any(
        abs(float(a) - b) >= 1e-6 for a, b in zip(baseline, (-0.2, 0.0))
    ):
        raise Norm02ValidationError("Gate 3C epochs lack the already-applied canonical -0.2-to-0 s baseline")
    channels = list(getattr(epochs, "ch_names", []))
    if any(channel not in channels for channel in ROI):
        raise Norm02ValidationError("Accepted epochs lack a required centroparietal ROI channel")
    bads = set(getattr(epochs, "info", {}).get("bads", []))
    authenticated = set(authenticated_interpolated_roi)
    if not authenticated.issubset(set(ROI) & bads):
        raise Norm02ValidationError("Authenticated interpolation channels do not match marked-bad ROI")
    unresolved = (bads & set(ROI)) - authenticated
    if unresolved:
        raise Norm02ValidationError(
            f"ROI contains a bad channel without authenticated Gate 3C interpolation: {sorted(unresolved)}"
        )
    data = np.asarray(epochs.get_data(picks=list(ROI)), dtype=float)
    if data.ndim != 3 or data.shape[1] != 4 or not np.isfinite(data).all():
        raise Norm02ValidationError("Bad MNE epoch data/ROI dimensions")
    return data.mean(axis=1) * 1e6
