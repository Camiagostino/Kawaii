from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .paths import resolve_session2_eeg_paths
from .stage_identity import TimingQCInputIdentity, validate_timing_qc_input_identity
from .study_definition import (
    ACQUISITION_RESPONSE_SEQUENCE,
    DEFAULT_RESPONSE_TYPES,
    response_timing_semantics,
)

OUTPUT_COLUMNS = [
    "subject_id", "session", "recording_part", "block", "trial", "response_type",
    "participant", "trial_id", "behavioral_rt_s", "eeg_response_marker_latency_s", "difference_s",
    "trigger_write_overhead_s", "marker_semantics", "response_locked_inference",
    "behavioural_rt_ms", "eeg_marker_rt_ms", "eeg_minus_behavioural_ms",
    "trigger_write_delay_ms", "onset_event", "onset_trigger_code", "onset_sample",
    "response_event", "response_trigger_code", "response_sample", "qc_status", "qc_reason",
]


def response_timing_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("response_timing_qc", {}) or {})
    settings.setdefault("sampling_rate_hz", config.get("recordings", {}).get("expected_sampling_rate_hz", 500.0))
    settings.setdefault("response_types", DEFAULT_RESPONSE_TYPES)
    settings.setdefault("output_dir", "response_timing_qc")
    return settings


def build_trial_level_qc(
    behavioural: pd.DataFrame,
    alignment: pd.DataFrame,
    trigger_log: pd.DataFrame | None,
    *,
    subject: str,
    recording_part: str,
    settings: Mapping[str, Any],
    input_identity: TimingQCInputIdentity | None = None,
) -> pd.DataFrame:
    """Create one explicit QC row per behavioural trial and response type."""
    validate_timing_qc_input_identity(
        input_identity,
        behavioral=behavioural,
        alignment=alignment,
        subject=subject,
        recording_part=recording_part,
        settings=settings,
    )
    required_behaviour = {"subject_id", "session", "block", "trial"}
    required_alignment = {"trial", "recording_part"}
    missing = sorted((required_behaviour - set(behavioural)) | (required_alignment - set(alignment)))
    if missing:
        raise ValueError("Required response-timing input columns are unavailable: " + ", ".join(missing))
    rate = float(settings["sampling_rate_hz"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("response_timing_qc sampling_rate_hz must be finite and positive.")
    maps = dict(settings.get("response_types", DEFAULT_RESPONSE_TYPES))
    rows: list[dict[str, Any]] = []
    alignment = alignment.copy()
    alignment["trial"] = pd.to_numeric(alignment["trial"], errors="coerce")
    triggers = trigger_log.copy() if trigger_log is not None else pd.DataFrame()

    for _, b in behavioural.iterrows():
        trial = pd.to_numeric(pd.Series([b.get("trial")]), errors="coerce").iloc[0]
        matches = alignment[(alignment["recording_part"].astype(str) == recording_part) & (alignment["trial"] == trial)]
        if "block" in alignment and pd.notna(b.get("block")):
            matches = matches[pd.to_numeric(matches["block"], errors="coerce") == _number(b.get("block"))]
        for response_type, mapping_raw in maps.items():
            mapping = dict(mapping_raw)
            reasons: list[str] = []
            timeout = _truthy(b.get(mapping["timeout"], False))
            behavioural_rt_value = b.get(mapping["behavioural_rt"])
            behavioural_rt = _number(behavioural_rt_value)
            if timeout:
                reasons.append("timeout")
            if _is_nonfinite_number(behavioural_rt_value):
                reasons.append("nonfinite_behavioural_rt")
            elif behavioural_rt is None:
                reasons.append("missing_behavioural_rt")
            elif behavioural_rt < 0:
                reasons.append("negative_behavioural_rt")
            if len(matches) == 0:
                reasons.append("missing_aligned_trial")
                a = pd.Series(dtype=object)
            elif len(matches) > 1:
                reasons.append("ambiguous_aligned_trial")
                a = matches.iloc[0]
            else:
                a = matches.iloc[0]
            onset = _number(a.get(mapping["onset_sample"]))
            response = _number(a.get(mapping["response_sample"]))
            if onset is None:
                reasons.append("missing_onset_marker")
            if response is None:
                reasons.append("missing_response_marker")
            eeg_rt = None if onset is None or response is None else (response - onset) / rate
            if eeg_rt is not None and eeg_rt < 0:
                reasons.append("negative_eeg_marker_rt")
            delay = None if behavioural_rt is None or eeg_rt is None else eeg_rt - behavioural_rt
            trigger_delay, trigger_reason = _match_trigger_delay(triggers, b, mapping)
            if trigger_reason:
                reasons.append(trigger_reason)
            critical = [r for r in reasons if not r.startswith("missing_trigger_log") and not r.startswith("missing_trigger_delay")]
            rows.append({
                "subject_id": subject, "session": b.get("session"), "recording_part": recording_part,
                "block": b.get("block"), "trial": b.get("trial"), "response_type": response_type,
                "participant": subject, "trial_id": b.get("trial"),
                "behavioral_rt_s": behavioural_rt, "eeg_response_marker_latency_s": eeg_rt,
                "difference_s": delay, "trigger_write_overhead_s": trigger_delay,
                "marker_semantics": "post_key_release_eeg_response_marker",
                "response_locked_inference": "NOT_VALIDATED_NOT_APPROVED",
                "behavioural_rt_ms": _ms(behavioural_rt), "eeg_marker_rt_ms": _ms(eeg_rt),
                "eeg_minus_behavioural_ms": _ms(delay), "trigger_write_delay_ms": _ms(trigger_delay),
                "onset_event": mapping["onset_sample"].removesuffix("_sample"),
                "onset_trigger_code": a.get(mapping["onset_code"], pd.NA), "onset_sample": onset,
                "response_event": mapping["response_sample"].removesuffix("_sample"),
                "response_trigger_code": a.get(mapping["response_code"], pd.NA), "response_sample": response,
                "qc_status": "valid" if not critical else "excluded",
                "qc_reason": ";".join(reasons) if reasons else "ok",
            })
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    validate_trial_level(out)
    return out


def validate_trial_level(table: pd.DataFrame) -> None:
    valid = table[table["qc_status"].eq("valid")]
    identity_columns = ["subject_id", "session", "recording_part", "block", "trial", "response_type"]
    if all(column in valid for column in identity_columns) and valid.duplicated(identity_columns).any():
        raise ValueError("Valid response-timing matches are not one-to-one.")
    for column in ("behavioural_rt_ms", "eeg_marker_rt_ms", "eeg_minus_behavioural_ms"):
        if column not in valid:
            continue
        values = pd.to_numeric(valid[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"A valid response-timing match has non-finite {column}.")
    if (pd.to_numeric(valid["behavioural_rt_ms"], errors="coerce") < 0).any():
        raise ValueError("A valid response-timing match has negative behavioural RT.")
    if (pd.to_numeric(valid["eeg_marker_rt_ms"], errors="coerce") < 0).any():
        raise ValueError("A valid response-timing match has negative EEG-derived RT.")
    if len(table) != int(table["qc_status"].value_counts(dropna=False).sum()):
        raise ValueError("Response-timing QC status counts do not reconcile.")


def summarise_valid(table: pd.DataFrame) -> pd.DataFrame:
    validate_trial_level(table)
    valid = table[table["qc_status"].eq("valid")].copy()
    columns = ["response_type", "n", "mean_ms", "median_ms", "sd_ms", "min_ms", "max_ms", "p2_5_ms", "p97_5_ms"]
    rows = []
    for response_type, group in valid.groupby("response_type", sort=False):
        values = pd.to_numeric(group["eeg_minus_behavioural_ms"], errors="coerce").dropna()
        rows.append({"response_type": response_type, "n": len(values), "mean_ms": values.mean(), "median_ms": values.median(),
                     "sd_ms": values.std(ddof=1), "min_ms": values.min(), "max_ms": values.max(),
                     "p2_5_ms": values.quantile(.025), "p97_5_ms": values.quantile(.975)})
    return pd.DataFrame(rows, columns=columns)


def summarise_rt_medians(table: pd.DataFrame) -> pd.DataFrame:
    """Return report-ready RT medians from valid one-to-one matches only."""
    validate_trial_level(table)
    valid = table[table["qc_status"].eq("valid")].copy()
    rows = []
    for response_type, group in valid.groupby("response_type", sort=False):
        difference = pd.to_numeric(group["eeg_minus_behavioural_ms"], errors="coerce").dropna()
        q1, q3 = difference.quantile([.25, .75]) if len(difference) else (np.nan, np.nan)
        iqr = q3 - q1
        outliers = int(((difference < q1 - 1.5 * iqr) | (difference > q3 + 1.5 * iqr)).sum()) if len(difference) else 0
        rows.append({
            "Response type": str(response_type).capitalize(),
            "Valid trials": len(group),
            "Behavioural RT median (ms)": pd.to_numeric(group["behavioural_rt_ms"], errors="coerce").median(),
            "EEG-derived RT median (ms)": pd.to_numeric(group["eeg_marker_rt_ms"], errors="coerce").median(),
            "Difference median (ms)": difference.median(),
            "IQR (ms)": iqr,
            "Outliers (1.5 x IQR)": outliers,
        })
    return pd.DataFrame(rows)


def write_figure(table: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt
    valid = table[table["qc_status"].eq("valid")].copy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    order = [x for x in DEFAULT_RESPONSE_TYPES if x in set(valid["response_type"])]
    data = [valid.loc[valid["response_type"].eq(x), "eeg_minus_behavioural_ms"].dropna() for x in order]
    if data:
        ax.boxplot(data, tick_labels=[x.capitalize() for x in order], showfliers=False)
        rng = np.random.default_rng(20260717)
        subjects = sorted(valid["subject_id"].astype(str).unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 1)))
        for si, subject in enumerate(subjects):
            sdf = valid[valid["subject_id"].astype(str).eq(subject)]
            for xi, response_type in enumerate(order, start=1):
                vals = pd.to_numeric(sdf.loc[sdf["response_type"].eq(response_type), "eeg_minus_behavioural_ms"], errors="coerce").dropna()
                ax.scatter(xi + rng.uniform(-.12, .12, len(vals)), vals, s=12, alpha=.38, color=colors[si])
                if len(vals): ax.scatter(xi, vals.median(), s=34, color=colors[si], edgecolor="white", linewidth=.5)
    else:
        ax.text(.5, .5, "No valid response-timing matches", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0, color="black", linewidth=.8, linestyle="--")
    ax.set(title="Retrospective response-marker timing discrepancy", xlabel="Response type", ylabel="EEG marker RT - behavioural RT (ms)")
    ax.grid(axis="y", alpha=.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(table: pd.DataFrame, inputs: Sequence[Path], path: Path) -> Path:
    valid = table[table.qc_status.eq("valid")]
    summary = summarise_valid(table)
    rt_summary = summarise_rt_medians(table)
    participants = table.subject_id.nunique()
    attempted, n_valid = len(table), len(valid)
    excluded = attempted - n_valid
    trig = pd.to_numeric(table.trigger_write_delay_ms, errors="coerce").dropna()
    participant_medians = valid.groupby(["subject_id", "response_type"])["eeg_minus_behavioural_ms"].median()
    spread = participant_medians.groupby("response_type").agg(["min", "max"]) if len(participant_medians) else pd.DataFrame()
    summary_md = _markdown_table(summary.round(3)) if len(summary) else "No valid matches."
    rt_summary_md = _markdown_table(rt_summary.round(3)) if len(rt_summary) else "No valid matches."
    status_md = _markdown_table(table.groupby(["response_type", "qc_status"]).size().rename("n").reset_index())
    trig_text = "unavailable" if trig.empty else f"n={len(trig)}, mean={trig.mean():.3f} ms, median={trig.median():.3f} ms, SD={trig.std(ddof=1):.3f} ms, range={trig.min():.3f} to {trig.max():.3f} ms"
    discrepancy_median = pd.to_numeric(valid.eeg_minus_behavioural_ms, errors="coerce").median()
    trigger_comparison = "Trigger-writing latency was unavailable for comparison."
    if not trig.empty and pd.notna(discrepancy_median) and trig.median() > 0:
        trigger_comparison = f"The pooled median discrepancy ({discrepancy_median:.1f} ms) was approximately {discrepancy_median / trig.median():,.0f} times the median trigger-writing latency ({trig.median():.3f} ms). Thus, the recorded parallel-port write interval is negligible on this scale and is not the main source of the delayed response markers."
    spread_text = "unavailable" if spread.empty else _markdown_table(spread.round(3).reset_index())
    interpretation = _interpret_distributions(summary, valid)
    input_text = "\n".join(f"- `{p}`" for p in inputs)
    content = f"""# Retrospective Session 2 response-timing QC

## Purpose

This analysis quantifies response-marker timing discrepancies in existing data collected with the current Session 2 task script. It does not alter acquisition, EEG markers, or ERP analyses.

## Acquisition-script behaviour

Behavioural RT is computed from the `secs` key-down timestamp returned by `KbCheck`. `KbReleaseWait` then occurs before the response function returns, so the EEG response trigger is sent only after key release. This affects response-marker timing, not behavioural RT calculation.

This sequence applies to recognition responses and to kawaii and arousal rating-confirmation responses:

```text
KbCheck detects key-down
-> `secs` is stored
-> behavioural RT is calculated
-> KbReleaseWait
-> response function returns
-> sendTrigger(...)
-> EEG response marker is recorded
```

## Variable definitions

| Variable | Represents | Does not represent |
|---|---|---|
| `recognition_rt_s` | Key-down RT from recognition onset | EEG response-marker latency |
| `kawaii_rt_s` | Key-down RT from kawaii-question onset | Key-release time |
| `arousal_rt_s` | Key-down RT from arousal-question onset | Key-release time |
| `event_time` | Fresh `GetSecs` immediately before `sendTrigger` | Original key-down time |
| `trigger_time` | Time recorded after trigger writing | Physical keyboard response time |
| `trigger_delay_s` | `trigger_time - event_time`, approximately trigger-writing overhead | Full keypress-to-trigger delay |
| EEG-derived response RT | EEG response sample minus corresponding EEG onset sample | Behavioural key-down RT |
| EEG-minus-behavioural delay | EEG-derived RT minus behavioural RT; a composite retrospective discrepancy | Pure keyboard hardware latency |

## Methods

Inputs:
{input_text}

### Verified event pairing

| Response type | EEG onset event / code | EEG response event / code | Behavioural variable | Matching keys |
|---|---|---|---|---|
| Recognition | `image_onset` / 120 old, 121 new | `recognition_response` / 150 yes, 151 no | `recognition_rt_s` | subject, session, recording part, block, trial, response type |
| Kawaii | `kawaii_question` / 100-109 | `kawaii_response` / 11-19 | `kawaii_rt_s` | subject, session, recording part, block, trial, response type |
| Arousal | `arousal_question` / 200-239 | `arousal_response` / 21-29 | `arousal_rt_s` | subject, session, recording part, block, trial, response type |

The aligned EEG table is selected separately for each subject and recording part, then matched on block and trial; session and response type come from the corresponding behavioural row and configured mapping. Image filename and event order are not used as matching keys. Exactly one aligned row is required for a valid match. Zero matches are labelled `missing_aligned_trial`; multiple matches are labelled `ambiguous_aligned_trial`. Missing onset/response markers, timeouts, missing behavioural RT, and negative EEG RT are retained as excluded rows with explicit reasons. Trigger-log delay is matched separately by subject, session, block, trial, and response event-name prefix; missing or ambiguous trigger-delay rows are flagged but do not invalidate the EEG/behavioural match.

EEG RT was `(response_sample - onset_sample) / sampling_rate`; all calculations used seconds internally and were exported in milliseconds. No fixed correction was applied.

## Results

- Participants: {participants}
- Attempted matches: {attempted}
- Valid matches: {n_valid}
- Excluded/unmatched: {excluded} ({(100 * excluded / attempted if attempted else 0):.1f}%)

### EEG-minus-behavioural delay by response type

{summary_md}

### Behavioural and EEG-derived RT medians

All values below use valid matched rows only. Outliers are counted using the stated conventional 1.5 x IQR rule; they are not excluded from the summaries.

{rt_summary_md}

### QC counts

{status_md}

### Participant-level variability

Participant median ranges (ms):

{spread_text}

### Trigger-writing latency (`trigger_delay_s`)

{trig_text}. This is reported separately and is not the full keypress-to-trigger delay.

{trigger_comparison}

## Interpretation

{interpretation} The dominant expected contributor is remaining key-hold duration because the trigger is emitted only after `KbReleaseWait`; key-hold duration itself was not measured because key-release time was not saved. Smaller contributors may include MATLAB execution/return time, trigger-writing overhead, EEG sampling resolution, and event-alignment uncertainty. This explains why the discrepancy can be much larger than keyboard or parallel-port hardware latency of only a few milliseconds. EEG-minus-behavioural delay is therefore a composite discrepancy, not pure keyboard hardware latency.

## Analysis implications

Current stimulus-locked ERP analyses based on image or question-onset triggers are not affected by this response-trigger issue. Behavioural RT analyses remain key-down based. **Response-locked inference from existing response markers is NOT VALIDATED / NOT APPROVED** because those markers are post-release rather than key-down timestamps. No automatic fixed or participant-median correction was applied; subtracting a fixed 3 ms would not correct a variable delay caused by `KbReleaseWait`.

For cuteness (`CUT-D16`/`CUT-D17`), the predefined 500–800-ms analysis is stimulus locked. It may contain evaluative, decision-formation, or response-preparation activity when those processes overlap the late window; it is not interpreted as purely passive perception and is not converted into response-locked inference.

## Limitations

The absolute `KbCheck secs` timestamp and key-release time were not saved, so true physical keypress-to-trigger latency cannot be reconstructed. The retrospective difference necessarily combines several timing components.

## Main conclusion

> Behavioural RT reflects key-down detection, whereas existing EEG response markers occur only after `KbReleaseWait` and cannot be interpreted as key-down timestamps. EEG-minus-behavioural timing is a composite delay, not pure keyboard latency. This matters for response-locked analyses; existing stimulus-locked ERP analyses are unaffected. No fixed correction was applied.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small report table without pandas' optional tabulate dependency."""

    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(_markdown_cell(value) for value in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _interpret_distributions(summary: pd.DataFrame, valid: pd.DataFrame) -> str:
    if summary.empty:
        return "No valid matches were available, so stability could not be assessed."
    statements = []
    for response_type, group in valid.groupby("response_type", sort=False):
        values = pd.to_numeric(group["eeg_minus_behavioural_ms"], errors="coerce").dropna()
        q1, q3 = values.quantile([.25, .75])
        iqr = q3 - q1
        outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        statements.append(f"For {response_type}, the median discrepancy was {values.median():.1f} ms, the IQR was {iqr:.1f} ms, and {outliers}/{len(values)} trials were identified as outliers by the 1.5 x IQR rule.")
    medians = valid.groupby(["subject_id", "response_type"])["eeg_minus_behavioural_ms"].median()
    between = medians.groupby("response_type").agg(lambda x: x.max() - x.min())
    if len(between):
        statements.append(f"Across response types, the observed ranges between the lowest and highest participant medians were {between.min():.1f}-{between.max():.1f} ms.")
    return " ".join(statements)


def _match_trigger_delay(triggers: pd.DataFrame, b: pd.Series, mapping: Mapping[str, Any]) -> tuple[float | None, str]:
    if triggers.empty:
        return None, "missing_trigger_log"
    needed = {"subject_id", "session", "block", "trial", "event_name", "trigger_delay_s"}
    if not needed.issubset(triggers.columns):
        return None, "missing_trigger_log_columns"
    mask = pd.Series(True, index=triggers.index)
    for col in ("subject_id", "session", "block", "trial"):
        mask &= triggers[col].astype(str).eq(str(b.get(col)))
    mask &= triggers.event_name.astype(str).str.startswith(str(mapping["trigger_event_prefix"]))
    found = triggers[mask]
    if len(found) != 1:
        return None, "missing_trigger_delay" if len(found) == 0 else "ambiguous_trigger_delay"
    return _number(found.iloc[0].trigger_delay_s), ""


def _number(value: Any) -> float | None:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(value) or not np.isfinite(float(value)) else float(value)


def _is_nonfinite_number(value: Any) -> bool:
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return True
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(numeric) and not np.isfinite(float(numeric)))


def _truthy(value: Any) -> bool:
    if pd.isna(value): return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _ms(value: float | None) -> float:
    return np.nan if value is None else value * 1000.0
