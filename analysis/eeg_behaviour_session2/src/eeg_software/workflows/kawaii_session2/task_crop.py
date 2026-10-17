from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .study_definition import KawaiiTriggerDefinition


@dataclass(frozen=True)
class TaskCropWindow:
    """Task-relevant crop window in original raw sample coordinates."""

    enabled: bool
    original_first_sample: int
    original_last_sample: int
    sfreq: float
    crop_start_sample: int
    crop_end_sample: int
    crop_start_s: float
    crop_end_s: float
    padding_before_s: float
    padding_after_s: float
    epoch_tmax_s: float
    first_retained_image_onset_sample: int
    last_retained_image_onset_sample: int
    first_retained_trial: int
    last_retained_trial: int
    n_retained_trials: int


@dataclass(frozen=True)
class TaskSegment:
    """One retained-trial task segment in original sample coordinates."""

    segment_id: int
    start_trial: int
    end_trial: int
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    duration_s: float
    n_trials: int


def task_crop_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return task-window crop settings."""

    cfg = dict(dict(config.get("preprocessing", {}) or {}).get("task_crop", {}) or {})
    return {
        "enabled_for_ica": bool(cfg.get("enabled_for_ica", True)),
        "enabled_for_epoching": bool(cfg.get("enabled_for_epoching", True)),
        "padding_before_s": float(cfg.get("padding_before_s", 30.0)),
        "padding_after_s": float(cfg.get("padding_after_s", 30.0)),
        "max_expected_duration_s": float(cfg.get("max_expected_duration_s", 3600.0)),
        "fail_if_crop_too_long": bool(cfg.get("fail_if_crop_too_long", True)),
        "long_gap_handling": dict(cfg.get("long_gap_handling", {}) or {}),
    }


def long_gap_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(task_crop_settings(config).get("long_gap_handling", {}) or {})
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "split_on_gap_s": float(cfg.get("split_on_gap_s", 60.0)),
        "keep_only_segments_with_retained_trials": bool(cfg.get("keep_only_segments_with_retained_trials", True)),
    }


def calculate_task_crop_window(
    alignment: pd.DataFrame,
    recording_part: str,
    *,
    sfreq: float,
    n_times: int,
    config: Mapping[str, Any],
    stage: str,
    first_samp: int = 0,
) -> TaskCropWindow:
    """Calculate a padded task crop window from retained alignment samples."""

    settings = task_crop_settings(config)
    enabled_key = "enabled_for_ica" if stage == "ica" else "enabled_for_epoching"
    enabled = bool(settings[enabled_key])
    original_first = int(first_samp)
    original_last = original_first + max(0, int(n_times) - 1)
    if not enabled:
        return TaskCropWindow(
            enabled=False,
            original_first_sample=original_first,
            original_last_sample=original_last,
            sfreq=float(sfreq),
            crop_start_sample=original_first,
            crop_end_sample=original_last,
            crop_start_s=0.0,
            crop_end_s=float(original_last - original_first) / float(sfreq),
            padding_before_s=0.0,
            padding_after_s=0.0,
            epoch_tmax_s=0.0,
            first_retained_image_onset_sample=original_first,
            last_retained_image_onset_sample=original_last,
            first_retained_trial=0,
            last_retained_trial=0,
            n_retained_trials=0,
        )
    retained = retained_image_onset_rows(
        alignment,
        recording_part,
        trigger_definition=KawaiiTriggerDefinition.from_config(config),
    )
    if retained.empty:
        raise ValueError(f"Task crop detection failed for {recording_part}: no retained 120/121 image-onset alignment rows.")
    if "image_onset_sample" not in retained or retained["image_onset_sample"].isna().all():
        raise ValueError(f"Task crop detection failed for {recording_part}: missing image_onset_sample.")
    image_samples = pd.to_numeric(retained["image_onset_sample"], errors="coerce").dropna()
    if image_samples.empty:
        raise ValueError(f"Task crop detection failed for {recording_part}: no valid retained image-onset samples.")
    trials = pd.to_numeric(retained.get("trial", pd.Series(dtype=float)), errors="coerce").dropna()
    if trials.empty:
        raise ValueError(f"Task crop detection failed for {recording_part}: missing retained trial numbers.")

    first_image = int(image_samples.min())
    last_image = int(image_samples.max())
    pad_before = float(settings["padding_before_s"])
    pad_after = float(settings["padding_after_s"])
    epoch_tmax = _recognition_epoch_tmax_s(config)
    start = max(original_first, int(first_image - round(pad_before * float(sfreq))))
    end = min(original_last, int(last_image + round((epoch_tmax + pad_after) * float(sfreq))))
    if end <= start:
        raise ValueError(f"Task crop detection failed for {recording_part}: invalid crop samples {start}-{end}.")
    cropped_duration = float(end - start) / float(sfreq)
    max_expected = float(settings["max_expected_duration_s"])
    if bool(settings["fail_if_crop_too_long"]) and cropped_duration > max_expected and not bool(long_gap_settings(config)["enabled"]):
        raise ValueError(
            "TASK_CROP_TOO_LONG: "
            f"{recording_part} crop duration {cropped_duration:.3f}s exceeds max_expected_duration_s={max_expected:.3f}; "
            f"raw_samples={original_first}-{original_last}; image_onset_samples={first_image}-{last_image}; crop_samples={start}-{end}."
        )
    return TaskCropWindow(
        enabled=True,
        original_first_sample=original_first,
        original_last_sample=original_last,
        sfreq=float(sfreq),
        crop_start_sample=start,
        crop_end_sample=end,
        crop_start_s=float(start - original_first) / float(sfreq),
        crop_end_s=float(end - original_first) / float(sfreq),
        padding_before_s=pad_before,
        padding_after_s=pad_after,
        epoch_tmax_s=epoch_tmax,
        first_retained_image_onset_sample=first_image,
        last_retained_image_onset_sample=last_image,
        first_retained_trial=int(trials.min()),
        last_retained_trial=int(trials.max()),
        n_retained_trials=int(len(retained)),
    )


def image_onset_timing_debug_table(
    alignment: pd.DataFrame,
    recording_part: str,
    *,
    sfreq: float,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return retained image-onset timing rows for crop debugging."""

    trigger_definition = KawaiiTriggerDefinition.from_config(config)
    retained = retained_image_onset_rows(
        alignment, recording_part, trigger_definition=trigger_definition
    )
    if retained.empty:
        return pd.DataFrame(columns=["trial", "event_code", "sample", "time_s", "delta_from_previous_image_s", "block", "metadata_source", "recording_part", "event_source"])
    rows = retained.copy()
    rows["trial"] = pd.to_numeric(rows["trial"], errors="coerce").astype("Int64")
    rows["event_code"] = pd.to_numeric(rows["image_onset_code"], errors="coerce").astype("Int64")
    rows["sample"] = pd.to_numeric(rows["image_onset_sample"], errors="coerce").astype("Int64")
    rows = rows.sort_values(["sample", "trial"], kind="mergesort").reset_index(drop=True)
    rows["time_s"] = rows["sample"].astype(float) / float(sfreq)
    rows["delta_from_previous_image_s"] = rows["time_s"].diff()
    rows["block"] = rows["block"] if "block" in rows else pd.NA
    rows["metadata_source"] = rows["metadata_source"] if "metadata_source" in rows else pd.NA
    event_codes = "_".join(str(code) for code in trigger_definition.image_event_id.values())
    rows["event_source"] = f"eeg_marker_{event_codes}"
    return rows[["trial", "event_code", "sample", "time_s", "delta_from_previous_image_s", "block", "metadata_source", "recording_part", "event_source"]].copy()


def image_onset_gap_summary_table(timing: pd.DataFrame, *, matlab_duration_s: float | None = None) -> pd.DataFrame:
    """Summarise retained image-onset gaps and MATLAB agreement status."""

    if timing.empty:
        return pd.DataFrame([{"status": "NO_RETAINED_IMAGE_ONSETS"}])
    deltas = pd.to_numeric(timing["delta_from_previous_image_s"], errors="coerce").dropna()
    first = timing.iloc[0]
    last = timing.iloc[-1]
    max_gap = float(deltas.max()) if not deltas.empty else 0.0
    task_duration = float(last["time_s"] - first["time_s"])
    matlab_flag = ""
    if matlab_duration_s is not None:
        if task_duration > 3600 and matlab_duration_s < 3600:
            matlab_flag = "EEG_MARKER_TIMING_DISAGREES_WITH_MATLAB_LOG"
        elif task_duration > 3600 and matlab_duration_s >= 3600:
            matlab_flag = "TRUE_LONG_RECORDING_OR_LONG_PAUSES"
    long_gap_rows = timing[pd.to_numeric(timing["delta_from_previous_image_s"], errors="coerce").fillna(0) > 60].copy()
    surrounding = []
    for idx in long_gap_rows.index.tolist():
        for near in (idx - 1, idx):
            if 0 <= near < len(timing):
                surrounding.append(str(int(timing.loc[near, "trial"])))
    return pd.DataFrame(
        [
            {
                "first_trial": int(first["trial"]),
                "first_sample": int(first["sample"]),
                "first_time_s": float(first["time_s"]),
                "last_trial": int(last["trial"]),
                "last_sample": int(last["sample"]),
                "last_time_s": float(last["time_s"]),
                "task_duration_s": task_duration,
                "matlab_task_duration_s": matlab_duration_s if matlab_duration_s is not None else pd.NA,
                "median_image_interval_s": float(deltas.median()) if not deltas.empty else pd.NA,
                "max_image_interval_s": max_gap,
                "trials_surrounding_long_gaps": " ".join(sorted(set(surrounding), key=int)) if surrounding else "",
                "block_boundaries": _block_boundaries(timing),
                "n_gaps_gt_10s": int((deltas > 10).sum()),
                "n_gaps_gt_30s": int((deltas > 30).sum()),
                "n_gaps_gt_60s": int((deltas > 60).sum()),
                "n_gaps_gt_300s": int((deltas > 300).sum()),
                "timing_flag": matlab_flag,
            }
        ]
    )


def calculate_task_segments(
    alignment: pd.DataFrame,
    recording_part: str,
    *,
    sfreq: float,
    n_times: int,
    config: Mapping[str, Any],
    first_samp: int = 0,
) -> list[TaskSegment]:
    """Split retained image-onset trials into task segments around long gaps."""

    timing = image_onset_timing_debug_table(
        alignment, recording_part, sfreq=sfreq, config=config
    )
    if timing.empty:
        return []
    settings = task_crop_settings(config)
    gap_settings = long_gap_settings(config)
    split_on = float(gap_settings["split_on_gap_s"])
    original_first = int(first_samp)
    original_last = original_first + max(0, int(n_times) - 1)
    pad_before = float(settings["padding_before_s"])
    pad_after = float(settings["padding_after_s"])
    epoch_tmax = _recognition_epoch_tmax_s(config)
    timing = timing.copy()
    timing["segment_break"] = pd.to_numeric(timing["delta_from_previous_image_s"], errors="coerce").fillna(0) > split_on
    timing["segment_id"] = timing["segment_break"].cumsum().astype(int) + 1
    segments: list[TaskSegment] = []
    for segment_id, segment_df in timing.groupby("segment_id", sort=True):
        first = segment_df.iloc[0]
        last = segment_df.iloc[-1]
        start_sample = max(original_first, int(int(first["sample"]) - round(pad_before * float(sfreq))))
        end_sample = min(original_last, int(int(last["sample"]) + round((epoch_tmax + pad_after) * float(sfreq))))
        segments.append(
            TaskSegment(
                segment_id=int(segment_id),
                start_trial=int(first["trial"]),
                end_trial=int(last["trial"]),
                start_sample=start_sample,
                end_sample=end_sample,
                start_s=float(start_sample - original_first) / float(sfreq),
                end_s=float(end_sample - original_first) / float(sfreq),
                duration_s=float(end_sample - start_sample) / float(sfreq),
                n_trials=int(len(segment_df)),
            )
        )
    return segments


def task_segments_table(segments: list[TaskSegment]) -> pd.DataFrame:
    return pd.DataFrame([segment.__dict__ for segment in segments])


def apply_task_segments(raw, segments: list[TaskSegment]):
    """Return raw cropped to task segments with long idle gaps removed."""

    if not segments:
        raise ValueError("Task segment crop failed: no task segments were available.")
    if len(segments) == 1:
        segment = segments[0]
        return raw.crop(tmin=segment.start_s, tmax=segment.end_s, include_tmax=True)
    import mne

    cropped = [raw.copy().crop(tmin=segment.start_s, tmax=segment.end_s, include_tmax=True) for segment in segments]
    return mne.concatenate_raws(cropped, preload=getattr(raw, "preload", False), verbose=False)


def segment_event_mapping_table(
    alignment: pd.DataFrame,
    recording_part: str,
    segments: list[TaskSegment],
    *,
    sfreq: float = 1.0,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Map original retained image-onset samples into concatenated segment coordinates."""

    timing = image_onset_timing_debug_table(
        alignment, recording_part, sfreq=sfreq, config=config
    )
    if timing.empty or not segments:
        return pd.DataFrame()
    ordered_segments = list(segments)
    for index, segment in enumerate(ordered_segments):
        if int(segment.end_sample) < int(segment.start_sample):
            raise ValueError(
                f"Task segment {segment.segment_id} has invalid inclusive sample bounds."
            )
        if int(segment.end_trial) < int(segment.start_trial):
            raise ValueError(
                f"Task segment {segment.segment_id} has invalid inclusive trial bounds."
            )
        if index:
            previous = ordered_segments[index - 1]
            if int(segment.segment_id) <= int(previous.segment_id):
                raise ValueError("Task segments are not in unique increasing segment order.")
            if int(segment.start_trial) <= int(previous.end_trial):
                raise ValueError("Task segment trial ranges overlap or are out of order.")
            if int(segment.start_sample) < int(previous.start_sample):
                raise ValueError("Task segment sample ranges are out of order.")

    timing = timing.copy()
    timing["trial"] = pd.to_numeric(timing["trial"], errors="coerce").astype("Int64")
    timing["sample"] = pd.to_numeric(timing["sample"], errors="coerce").astype("Int64")
    timing["event_code"] = pd.to_numeric(timing["event_code"], errors="coerce").astype("Int64")
    if timing[["trial", "sample", "event_code"]].isna().any(axis=None):
        raise ValueError("Retained task events contain missing trial, sample, or event-code values.")
    if timing.duplicated(["trial"], keep=False).any():
        raise ValueError("Retained task events contain duplicate trial keys; mapping is ambiguous.")

    rows = []
    offset = 0
    mapped_trials: set[int] = set()
    segmented_first_sample = int(ordered_segments[0].start_sample)
    for segment in ordered_segments:
        # Padding may overlap adjacent segments. Trial bounds, rather than padded
        # sample containment, identify which concatenated copy is authoritative.
        seg_rows = timing[
            timing["trial"].astype(int).between(
                int(segment.start_trial), int(segment.end_trial), inclusive="both"
            )
        ].copy()
        for _, row in seg_rows.iterrows():
            trial = int(row["trial"])
            original_sample = int(row["sample"])
            if trial in mapped_trials:
                raise ValueError(
                    f"Retained task trial {trial} maps to duplicate task segments."
                )
            if not int(segment.start_sample) <= original_sample <= int(segment.end_sample):
                raise ValueError(
                    f"Retained task trial {trial} falls outside its intended segment sample bounds."
                )
            mapped_trials.add(trial)
            rows.append(
                {
                    "trial": trial,
                    "event_code": int(row["event_code"]),
                    "event_sample_original": original_sample,
                    "segment_id": segment.segment_id,
                    "event_sample_segmented": int(
                        segmented_first_sample
                        + offset
                        + (original_sample - int(segment.start_sample))
                    ),
                }
            )
        offset += int(segment.end_sample - segment.start_sample + 1)
    expected_trials = set(timing["trial"].astype(int).tolist())
    missing_trials = sorted(expected_trials - mapped_trials)
    if missing_trials:
        raise ValueError(
            "Retained task events are missing an intended task-segment assignment: "
            + ", ".join(str(value) for value in missing_trials)
        )
    mapped = pd.DataFrame(rows)
    if mapped.duplicated(["trial"], keep=False).any():
        raise ValueError("Segment event mapping contains duplicate trial keys.")
    return mapped


def apply_task_crop(raw, window: TaskCropWindow):
    """Crop raw to the task window while preserving MNE sample coordinates."""

    if not window.enabled:
        return raw
    return raw.crop(tmin=window.crop_start_s, tmax=window.crop_end_s, include_tmax=True)


def task_crop_summary(window: TaskCropWindow, *, subject: str, recording_part: str, original_duration_s: float) -> pd.DataFrame:
    """Return a one-row task crop summary table."""

    cropped_duration = max(0.0, window.crop_end_s - window.crop_start_s)
    raw_duration = max(0.0, float(window.original_last_sample - window.original_first_sample) / window.sfreq)
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "recording_part": recording_part,
                "task_crop_enabled": window.enabled,
                "raw_first_samp": window.original_first_sample,
                "raw_last_samp": window.original_last_sample,
                "raw_duration_s": raw_duration if original_duration_s is None else original_duration_s,
                "crop_start_sample": window.crop_start_sample,
                "crop_end_sample": window.crop_end_sample,
                "crop_start_time_s": window.crop_start_s,
                "crop_end_time_s": window.crop_end_s,
                "cropped_duration_s": cropped_duration,
                "padding_before_s": window.padding_before_s,
                "padding_after_s": window.padding_after_s,
                "epoch_tmax_s": window.epoch_tmax_s,
                "first_retained_image_onset_sample": window.first_retained_image_onset_sample,
                "last_retained_image_onset_sample": window.last_retained_image_onset_sample,
                "first_retained_image_onset_time_s": float(window.first_retained_image_onset_sample - window.original_first_sample) / window.sfreq,
                "last_retained_image_onset_time_s": float(window.last_retained_image_onset_sample - window.original_first_sample) / window.sfreq,
                "first_retained_trial": window.first_retained_trial,
                "last_retained_trial": window.last_retained_trial,
                "n_retained_trials_covered": window.n_retained_trials,
                "n_samples_after_crop": int(window.crop_end_sample - window.crop_start_sample + 1),
                "original_duration_s": raw_duration if original_duration_s is None else original_duration_s,
                "original_first_sample": window.original_first_sample,
                "original_last_sample": window.original_last_sample,
                "crop_start_s": window.crop_start_s,
                "crop_end_s": window.crop_end_s,
                "last_retained_event_sample": window.last_retained_image_onset_sample,
            }
        ]
    )


def task_crop_debug_table(
    alignment: pd.DataFrame,
    recording_part: str,
    window: TaskCropWindow,
    *,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return per-retained-image-onset crop coordinates for debugging."""

    retained = retained_image_onset_rows(
        alignment,
        recording_part,
        trigger_definition=KawaiiTriggerDefinition.from_config(config),
    )
    rows = []
    for _, row in retained.sort_values(["image_onset_sample", "trial"], kind="mergesort").iterrows():
        sample = int(pd.to_numeric(pd.Series([row.get("image_onset_sample")]), errors="coerce").iloc[0])
        rows.append(
            {
                "trial": row.get("trial"),
                "event_code": int(row.get("image_onset_code")),
                "event_sample_original": sample,
                "event_time_original_s": float(sample - window.original_first_sample) / window.sfreq,
                "event_sample_after_crop": int(sample - window.crop_start_sample),
                "event_time_after_crop_s": float(sample - window.crop_start_sample) / window.sfreq,
                "included_in_crop": bool(window.crop_start_sample <= sample <= window.crop_end_sample),
            }
        )
    return pd.DataFrame(rows)


def retained_image_onset_rows(
    alignment: pd.DataFrame,
    recording_part: str,
    *,
    config: Mapping[str, Any] | None = None,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Return retained study-defined image-onset rows for one recording part."""

    retained = retained_task_rows(alignment, recording_part)
    if retained.empty:
        return retained
    if "image_onset_code" not in retained:
        raise ValueError(f"Task crop detection failed for {recording_part}: missing image_onset_code.")
    if "image_onset_sample" not in retained:
        raise ValueError(f"Task crop detection failed for {recording_part}: missing image_onset_sample.")
    definition = trigger_definition or KawaiiTriggerDefinition.from_config(config)
    codes = pd.to_numeric(retained["image_onset_code"], errors="coerce")
    samples = pd.to_numeric(retained["image_onset_sample"], errors="coerce")
    return retained[codes.isin(definition.image_event_id.values()) & samples.notna()].copy()


def retained_task_rows(alignment: pd.DataFrame, recording_part: str) -> pd.DataFrame:
    """Return retained alignment rows for one recording part."""

    if alignment.empty:
        return pd.DataFrame()
    df = alignment.copy()
    if "recording_part" in df:
        df = df[df["recording_part"].astype(str).eq(str(recording_part))].copy()
    if "retain_for_epoching" not in df:
        raise ValueError("Task crop detection failed: alignment is missing retain_for_epoching.")
    return df[df["retain_for_epoching"].map(_truthy)].copy()


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _recognition_epoch_tmax_s(config: Mapping[str, Any]) -> float:
    epochs = dict(config.get("epochs", {}) or {})
    canonical = dict(epochs.get("canonical", {}) or {})
    return float(canonical.get("tmax_s", 1.2))


def _block_boundaries(timing: pd.DataFrame) -> str:
    if "block" not in timing or timing["block"].isna().all():
        return ""
    rows = []
    for block, block_df in timing.groupby("block", dropna=True, sort=True):
        first = block_df.iloc[0]
        last = block_df.iloc[-1]
        rows.append(f"{block}:trials{int(first['trial'])}-{int(last['trial'])}")
    return "; ".join(rows)
