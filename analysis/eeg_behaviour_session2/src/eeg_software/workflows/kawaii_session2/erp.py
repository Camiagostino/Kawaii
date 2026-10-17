from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import math
import re
import time

import numpy as np
import pandas as pd

from .bad_channel_disposition import validate_downstream_retained_bad_channel_safety
from .stage_identity import (
    bind_epoch_metadata_rows,
    build_participant_erp_artifact_identity,
    configuration_mapping_fingerprint,
    epoch_metadata_binding,
    participant_erp_identity_path,
    validate_epoch_metadata_binding,
    write_participant_erp_artifact_identity,
)

from .cuteness_mixed import build_cuteness_trial_table, tag_categorical_cuteness_measurements
from .figure_style import (
    FigureStyle,
    default_figure_style,
    load_figure_style,
    shade_primary_windows,
    style_axes,
    style_legend,
    style_topography_figure,
)
from .paths import subject_part_dir, subject_part_stage_dir
from .plots import save_figure_with_stage_provenance
from .provenance import write_provenance_sidecar
from .reporting import write_markdown_report
from .source_exposure import (
    append_source_exposure_validation,
    attach_source_exposure_summary,
    prepare_cuteness_inputs,
    prepare_recognition_inputs,
)
from .study_definition import (
    configured_conditions,
    configured_minimum_trials,
    configured_rois,
    configured_time_windows,
    erp_branch_settings,
    plotting_channel_exclusion_rationale,
    plotting_channel_exclusions,
    plotting_safe_evokeds,
    recognition_roi_status,
    recognition_scientific_metadata,
)



class RecognitionConditionReconciliationError(ValueError):
    """Raised when retained recognition rows cannot be reconciled to exactly one frozen condition."""

    reason_code = "recognition_condition_unclassified_rows"

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "recognition_condition_unclassified_rows",
        unclassified_rows: Sequence[Any] = (),
        overlapping_rows: Sequence[Any] = (),
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.unclassified_rows = tuple(unclassified_rows)
        self.overlapping_rows = tuple(overlapping_rows)


@dataclass(frozen=True)
class ERPResult:
    """Participant-level ERP outputs for one subject recording part."""

    subject: str
    recording_part: str
    branch: str
    status: str
    output_files: dict[str, Path]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    info_messages: tuple[str, ...] = ()
    metrics: dict[str, Any] | None = None


def gate4a_recognition_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return participant-level recognition ERP settings."""

    return erp_branch_settings(config, "recognition")


def gate4b_cuteness_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return participant-level cuteness ERP settings."""

    return erp_branch_settings(config, "cuteness")


def erp_output_paths(config: Mapping[str, Any], subject: str, recording_part: str, branch: str, *, root: Path | None = None) -> dict[str, Path]:
    """Return reusable ERP output paths for a branch such as recognition/cuteness."""

    base = subject_part_stage_dir(config, subject, recording_part, "erp", root=root, create=True) / branch
    fig_dir = base / "figures"
    waveform_dir = base / "waveforms"
    topography_dir = base / "topographies"
    fig_dir.mkdir(parents=True, exist_ok=True)
    waveform_dir.mkdir(parents=True, exist_ok=True)
    topography_dir.mkdir(parents=True, exist_ok=True)
    branch_settings = erp_branch_settings(config, branch)
    difference_cfg = dict(branch_settings.get("difference_wave", {}) or {})
    paths = {
        "erp_dir": base,
        "fig_dir": fig_dir,
        "waveform_dir": waveform_dir,
        "topography_dir": topography_dir,
        "erp_parameters": base / "erp_parameters.csv",
        "condition_counts": base / "condition_counts.csv",
        "measurements": base / f"{branch}_measurements.csv",
        "measurements_wide": base / f"{branch}_measurements_wide.csv",
        "measurement_dictionary": base / "measurement_dictionary.csv",
        "qc_summary": base / f"{branch}_qc_summary.csv",
        "continuous_summary": base / f"{branch}_continuous_summary.csv",
        "participant_erps_fif": base / str(branch_settings.get("evoked_filename", "participant_erps-ave.fif")),
        "difference_wave_fif": base / str(difference_cfg.get("filename", "difference_wave-ave.fif")),
        "processing_log": base / "processing_log.csv",
        "report": base / "report.md",
        "failure": base / "failure.csv",
        "participant_erp_identity": participant_erp_identity_path(base),
        "waveforms": fig_dir / str(dict(branch_settings.get("figures", {}) or {}).get("waveform_filename", "waveforms.png")),
        "difference_waveform": waveform_dir / "difference_wave.png",
        "roi_waveforms": waveform_dir / "roi_waveforms.png",
        "trial_count_summary": fig_dir / "trial_count_summary.png",
    }
    paths[f"{branch}_measurements"] = paths["measurements"]
    paths[f"{branch}_measurements_wide"] = paths["measurements_wide"]
    paths[f"{branch}_qc_summary"] = paths["qc_summary"]
    paths[f"{branch}_continuous_summary"] = paths["continuous_summary"]
    if branch == "recognition":
        paths["recognition_measurements"] = paths["measurements"]
        paths["recognition_measurements_wide"] = paths["measurements_wide"]
        paths["recognition_contrasts"] = base / "recognition_contrasts.csv"
        paths["recognition_qc_summary"] = paths["qc_summary"]
    if branch == "cuteness":
        paths["cuteness_measurements"] = paths["measurements"]
        paths["cuteness_measurements_wide"] = paths["measurements_wide"]
        paths["cuteness_qc_summary"] = paths["qc_summary"]
        paths["cuteness_trial_level"] = base / "cuteness_trial_level.csv"
    return paths


def gate3c_erp_input_paths(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> dict[str, Path]:
    """Return Gate 3C products used as stable Gate 4 ERP inputs."""

    settings = erp_branch_settings(config, "recognition")
    part_dir = subject_part_dir(config, subject, recording_part, root=root, create=False)
    return {
        "epochs_fif": part_dir / str(settings.get("input_epochs_file", "epochs/all_image_epochs-epo.fif")),
        "epoch_metadata_csv": part_dir / str(settings.get("input_metadata_file", "epochs/epoch_metadata.csv")),
    }


def discover_gate3c_recording_parts(config: Mapping[str, Any], subject: str, *, root: Path | None = None) -> list[str]:
    """Discover recording parts with completed Gate 3C epoch inputs."""

    base = subject_part_dir(config, subject, "part1", root=root, create=False).parent
    if not base.exists():
        return []
    parts = []
    for part_dir in sorted(path for path in base.glob("part*") if path.is_dir()):
        inputs = gate3c_erp_input_paths(config, subject, part_dir.name, root=root)
        if inputs["epochs_fif"].exists() and inputs["epoch_metadata_csv"].exists():
            parts.append(part_dir.name)
    return parts


def load_gate3c_epochs_and_metadata(input_paths: Mapping[str, Path]):
    """Load cleaned Gate 3C epochs and the exported epoch metadata table."""

    import mne

    epochs_path = Path(input_paths["epochs_fif"])
    metadata_path = Path(input_paths["epoch_metadata_csv"])
    if not epochs_path.exists():
        raise FileNotFoundError(f"Gate 3C epochs FIF not found: {epochs_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Gate 3C epoch metadata CSV not found: {metadata_path}")
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    metadata = pd.read_csv(metadata_path)
    return epochs, metadata


def _validate_and_rebind_gate3c_epoch_metadata(
    epochs,
    metadata: pd.DataFrame,
    *,
    participant: str,
    session: str | int,
    recording_part: str,
    allow_legacy_serialization_compatibility: bool,
):
    """Validate published Gate 3C copies, then create one stable in-memory binding."""

    binding = validate_epoch_metadata_binding(
        getattr(epochs, "metadata", None),
        metadata,
        participant=participant,
        session=session,
        recording_part=recording_part,
        allow_legacy_serialization_compatibility=allow_legacy_serialization_compatibility,
    )
    rebound = bind_epoch_metadata_rows(
        metadata,
        participant=participant,
        session=session,
        recording_part=recording_part,
    )
    epochs.metadata = rebound.copy()
    return epochs, rebound, binding


def validate_epochs_metadata(
    epochs,
    metadata: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    participant: str | None = None,
    session: str | int | None = None,
    recording_part: str | None = None,
) -> pd.DataFrame:
    """Validate that epochs, metadata, and configured ERP conditions agree."""

    rows = []
    n_epochs = int(len(epochs))
    n_metadata = int(len(metadata))
    rows.append({"check": "epoch_metadata_row_count", "status": "ok" if n_epochs == n_metadata else "failed", "detail": f"epochs={n_epochs}; metadata_rows={n_metadata}"})
    if n_epochs != n_metadata:
        raise ValueError(f"Epoch count mismatch: epochs={n_epochs}, epoch_metadata rows={n_metadata}.")

    expected_tmin = float(settings.get("epoch_tmin_s", -0.2))
    expected_tmax = float(settings.get("epoch_tmax_s", 1.2))
    actual_tmin = float(getattr(epochs, "tmin", expected_tmin))
    actual_tmax = float(getattr(epochs, "tmax", expected_tmax))
    if not (abs(actual_tmin - expected_tmin) < 1e-9 and abs(actual_tmax - expected_tmax) < 1e-9):
        raise ValueError(f"Non-canonical epoch interval: {(actual_tmin, actual_tmax)}; expected {(expected_tmin, expected_tmax)}.")

    embedded = getattr(epochs, "metadata", None)
    if embedded is not None:
        embedded_n = int(len(embedded))
        rows.append({"check": "embedded_metadata_row_count", "status": "ok" if embedded_n == n_epochs else "failed", "detail": f"embedded_metadata_rows={embedded_n}; epochs={n_epochs}"})
        if embedded_n != n_epochs:
            raise ValueError(f"Embedded epoch metadata count mismatch: embedded={embedded_n}, epochs={n_epochs}.")
    binding = validate_epoch_metadata_binding(
        embedded,
        metadata,
        participant=participant,
        session=session,
        recording_part=recording_part,
    )
    rows.append(
        {
            "check": "epoch_metadata_identity_binding",
            "status": "ok",
            "detail": binding.fingerprint,
        }
    )

    rows.append({"check": "metadata_columns", "status": "ok", "detail": ", ".join(available_metadata_columns(metadata))})
    rows.append({"check": "recognition_values", "status": "ok", "detail": available_recognition_values(metadata, settings)})

    missing_fields = missing_condition_metadata_fields(metadata, settings)
    rows.append(
        {
            "check": "condition_metadata_fields",
            "status": "ok" if not missing_fields else "failed",
            "detail": "; ".join(f"{condition}: {', '.join(fields)}" for condition, fields in missing_fields.items()) if missing_fields else "all configured condition metadata fields present",
        }
    )
    if missing_fields:
        raise ValueError(
            "Configured recognition ERP condition derivation requires unavailable metadata field(s): "
            + "; ".join(f"{condition}: {', '.join(fields)}" for condition, fields in missing_fields.items())
            + f". Available metadata columns: {', '.join(available_metadata_columns(metadata))}."
        )

    reconciliation = require_recognition_condition_reconciliation(metadata, settings)
    if reconciliation.get("applies"):
        rows.append(
            {
                "check": "recognition_condition_reconciliation",
                "status": "warning"
                if reconciliation["status"] == "qc_only_global_behavioural_metadata_incomplete"
                else "ok",
                "detail": {
                    "status": reconciliation["status"],
                    "reason_code": reconciliation["reason_code"],
                    "classified_rows": reconciliation["classified_rows"],
                    "total_rows": reconciliation["total_rows"],
                    "unclassified_rows": reconciliation["unclassified_rows"],
                    "overlapping_rows": reconciliation["overlapping_rows"],
                },
            }
        )

    condition_column = resolve_condition_column(metadata, settings)
    rows.append({"check": "condition_column", "status": "ok", "detail": condition_column})
    counts = condition_counts_from_metadata(metadata, settings)
    missing = [condition for condition, count in counts.items() if count == 0]
    rows.append({"check": "derived_condition_counts", "status": "ok" if not missing else "warning", "detail": counts})
    rows.append({"check": "required_conditions", "status": "ok" if not missing else "warning", "detail": ", ".join(missing) if missing else "all configured conditions present"})
    return pd.DataFrame(rows)


def resolve_condition_column(metadata: pd.DataFrame, settings: Mapping[str, Any]) -> str:
    """Choose the configured behavioural metadata column used for conditions."""

    candidates = [str(settings.get("condition_column", ""))]
    candidates.extend(str(col) for col in settings.get("fallback_condition_columns", []) or [])
    for column in candidates:
        if column and column in metadata.columns:
            return column
    raise ValueError(f"None of the configured condition columns exist in epoch metadata: {candidates}.")


def missing_conditions(metadata: pd.DataFrame, settings: Mapping[str, Any], *, condition_column: str | None = None) -> list[str]:
    """Return configured conditions with no matching metadata rows."""

    missing = []
    for condition, spec in configured_conditions(settings).items():
        if not condition_mask(metadata, settings, spec, condition).any():
            missing.append(condition)
    return missing


def available_metadata_columns(metadata: pd.DataFrame) -> list[str]:
    """Return available epoch metadata columns for validation/reporting."""

    return [str(column) for column in metadata.columns]


def available_recognition_values(metadata: pd.DataFrame, settings: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return nonblank recognition-related values observed in epoch metadata."""

    columns: set[str] = set()
    for candidate in [settings.get("condition_column", ""), *(settings.get("fallback_condition_columns", []) or [])]:
        if candidate:
            columns.add(str(candidate))
    for spec in configured_conditions(settings).values():
        columns.update(str(column) for column in dict(spec.get("criteria", {}) or {}))
    for candidate in ("old_new_status", "recognition_response", "recognition_response_metadata", "recognition_correct", "recognition_correct_metadata", "correct_recognition_response", "condition"):
        columns.add(candidate)

    values: dict[str, list[str]] = {}
    for column in sorted(columns):
        if column not in metadata.columns:
            values[column] = []
            continue
        series = metadata[column].dropna().astype(str).str.strip()
        values[column] = sorted(value for value in series.unique().tolist() if value)
    return values


def missing_condition_metadata_fields(metadata: pd.DataFrame, settings: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return configured condition criteria columns that are absent from metadata."""

    missing: dict[str, list[str]] = {}
    for condition, spec in configured_conditions(settings).items():
        criteria = dict(spec.get("criteria", {}) or {})
        missing_columns = [str(column) for column in criteria if str(column) not in metadata.columns]
        range_spec = dict(spec.get("range", {}) or {})
        if range_spec:
            range_column = str(range_spec.get("column", settings.get("condition_column", "")))
            if range_column and range_column not in metadata.columns:
                missing_columns.append(range_column)
        if missing_columns:
            missing[condition] = missing_columns
    return missing


def blank_condition_metadata_fields(metadata: pd.DataFrame, settings: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return configured condition criteria columns that exist but contain no values."""

    blank: dict[str, list[str]] = {}
    for condition, spec in configured_conditions(settings).items():
        criteria = dict(spec.get("criteria", {}) or {})
        range_spec = dict(spec.get("range", {}) or {})
        if range_spec:
            criteria[str(range_spec.get("column", settings.get("condition_column", "")))] = "numeric"
        empty_columns = []
        for column in criteria:
            column = str(column)
            if column in metadata.columns:
                series = metadata[column].dropna().astype(str).str.strip()
                if not bool(series.ne("").any()):
                    empty_columns.append(column)
        if empty_columns:
            blank[condition] = empty_columns
    return blank



def recognition_condition_reconciliation(
    metadata: pd.DataFrame,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile every retained recognition row to exactly one configured condition.

    Whole-column missing/blank behavioural metadata remains an explicit QC-only
    state handled by the existing participant-status logic. Partial malformed,
    missing, or overlapping rows are not allowed to disappear silently from
    condition-specific ERP analyses.
    """

    if str(settings.get("branch", "")).strip().lower() != "recognition":
        return {
            "applies": False,
            "status": "not_applicable",
            "reason_code": "",
            "unclassified_rows": [],
            "overlapping_rows": [],
            "classified_rows": int(len(metadata)),
            "total_rows": int(len(metadata)),
        }

    conditions = configured_conditions(settings)
    if not conditions:
        return {
            "applies": True,
            "status": "ok",
            "reason_code": "",
            "unclassified_rows": [],
            "overlapping_rows": [],
            "classified_rows": int(len(metadata)),
            "total_rows": int(len(metadata)),
        }

    masks = pd.DataFrame(
        {
            condition: condition_mask(metadata, settings, spec, condition).astype(bool)
            for condition, spec in conditions.items()
        },
        index=metadata.index,
    )
    match_count = masks.sum(axis=1)
    unclassified = metadata.index[match_count.eq(0)].tolist()
    overlapping = metadata.index[match_count.gt(1)].tolist()

    # Preserve the already-frozen whole-log QC semantics: if a required
    # behavioural criterion column is completely blank/missing-in-values, the
    # existing QC-only path remains authoritative instead of treating every row
    # as an individual malformed-row failure.
    globally_blank_fields = blank_condition_metadata_fields(metadata, settings)
    if globally_blank_fields and len(unclassified) == len(metadata) and not overlapping:
        return {
            "applies": True,
            "status": "qc_only_global_behavioural_metadata_incomplete",
            "reason_code": "behavioural_metadata_incomplete",
            "unclassified_rows": unclassified,
            "overlapping_rows": [],
            "classified_rows": 0,
            "total_rows": int(len(metadata)),
        }

    if overlapping:
        return {
            "applies": True,
            "status": "failed",
            "reason_code": "recognition_condition_overlapping_rows",
            "unclassified_rows": unclassified,
            "overlapping_rows": overlapping,
            "classified_rows": int(match_count.eq(1).sum()),
            "total_rows": int(len(metadata)),
        }

    if unclassified:
        return {
            "applies": True,
            "status": "failed",
            "reason_code": "recognition_condition_unclassified_rows",
            "unclassified_rows": unclassified,
            "overlapping_rows": [],
            "classified_rows": int(match_count.eq(1).sum()),
            "total_rows": int(len(metadata)),
        }

    return {
        "applies": True,
        "status": "ok",
        "reason_code": "",
        "unclassified_rows": [],
        "overlapping_rows": [],
        "classified_rows": int(len(metadata)),
        "total_rows": int(len(metadata)),
    }


def require_recognition_condition_reconciliation(
    metadata: pd.DataFrame,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when retained recognition rows are partially unclassified/overlapping."""

    reconciliation = recognition_condition_reconciliation(metadata, settings)
    if reconciliation["status"] == "failed":
        reason_code = str(reconciliation["reason_code"])
        unclassified = list(reconciliation["unclassified_rows"])
        overlapping = list(reconciliation["overlapping_rows"])
        raise RecognitionConditionReconciliationError(
            f"{reason_code}: retained recognition metadata must map each row to exactly one "
            f"configured condition; unclassified_rows={unclassified}; overlapping_rows={overlapping}.",
            reason_code=reason_code,
            unclassified_rows=unclassified,
            overlapping_rows=overlapping,
        )
    return reconciliation


def condition_counts_from_metadata(metadata: pd.DataFrame, settings: Mapping[str, Any]) -> dict[str, int]:
    """Count rows matching each derived recognition condition."""

    counts: dict[str, int] = {}
    for condition, spec in configured_conditions(settings).items():
        counts[condition] = int(condition_mask(metadata, settings, spec, condition).sum())
    return counts


def apply_erp_baseline(epochs, settings: Mapping[str, Any]):
    """Validate the canonical Gate 3C baseline without applying it twice."""

    baseline = tuple(float(value) for value in settings.get("baseline_s", [-0.2, 0.0]))
    existing = getattr(epochs, "baseline", None)
    if existing is None:
        raise ValueError("Gate 4 requires canonical Gate 3C epochs with baseline already applied.")
    existing = tuple(float(value) for value in existing)
    # FIF stores baseline bounds as single-precision values; tolerate only that
    # serialization-scale rounding while still rejecting a scientifically different baseline.
    if len(existing) != 2 or any(abs(actual - expected) >= 1e-6 for actual, expected in zip(existing, baseline)):
        raise ValueError(f"Gate 4 baseline mismatch: epochs={existing}; canonical={baseline}.")
    return epochs


def split_epochs_by_condition(
    epochs,
    metadata: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    participant: str | None = None,
    session: str | int | None = None,
    recording_part: str | None = None,
) -> dict[str, Any]:
    """Split epochs into configured conditions using behavioural metadata."""

    validate_epoch_metadata_binding(
        getattr(epochs, "metadata", None),
        metadata,
        participant=participant,
        session=session,
        recording_part=recording_part,
    )
    require_recognition_condition_reconciliation(metadata, settings)
    splits = {}
    for condition, spec in configured_conditions(settings).items():
        mask = condition_mask(metadata, settings, spec, condition)
        indices = metadata.index[mask].tolist()
        splits[condition] = epochs[indices] if indices else []
    return splits


def condition_mask(metadata: pd.DataFrame, settings: Mapping[str, Any], spec: Mapping[str, Any], condition: str) -> pd.Series:
    """Return rows matching a configured condition specification."""

    range_spec = dict(spec.get("range", {}) or {})
    if range_spec:
        column = str(range_spec.get("column", settings.get("condition_column", "")))
        if column not in metadata.columns:
            raise ValueError(f"Condition {condition!r} requires missing metadata column(s): {column}.")
        values = pd.to_numeric(metadata[column], errors="coerce")
        mask = pd.Series(True, index=metadata.index)
        if "min" in range_spec:
            mask &= values >= float(range_spec["min"])
        if "max" in range_spec:
            mask &= values <= float(range_spec["max"])
        return mask.fillna(False)

    criteria = dict(spec.get("criteria", {}) or {})
    if criteria:
        mask = pd.Series(True, index=metadata.index)
        missing_columns = [column for column in criteria if column not in metadata.columns]
        if missing_columns:
            raise ValueError(f"Condition {condition!r} requires missing metadata column(s): {', '.join(missing_columns)}.")
        for column, expected in criteria.items():
            mask &= _normalised_condition_text(metadata[column]).eq(_normalise_condition_value(expected))
        return mask
    column = resolve_condition_column(metadata, settings)
    value = str(spec.get("value", condition))
    return _normalised_condition_text(metadata[column]).eq(_normalise_condition_value(value))


def _normalised_condition_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _normalise_condition_value(value: Any) -> str:
    return str(value).strip().lower()


def compute_participant_erps(condition_epochs: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """Average participant epochs for each configured ERP condition."""

    picks = erp_picks(settings)
    evokeds = {}
    for condition, epochs in condition_epochs.items():
        if len(epochs) == 0:
            continue
        if hasattr(epochs, "info"):
            validate_downstream_retained_bad_channel_safety(epochs.info, settings)
        evoked = epochs.average(picks=picks)
        label = str(configured_conditions(settings).get(condition, {}).get("label", condition))
        try:
            evoked.comment = label
        except Exception:
            pass
        evokeds[condition] = evoked
    return evokeds


def compute_difference_wave(evokeds: Mapping[str, Any], settings: Mapping[str, Any]):
    """Compute the configured participant-level difference wave."""

    diff = dict(settings.get("difference_wave", {}) or {})
    if not bool(diff.get("enabled", False)):
        return None
    minuend = str(diff.get("minuend", "hit"))
    subtrahend = str(diff.get("subtrahend", "correct_rejection"))
    if minuend not in evokeds or subtrahend not in evokeds:
        return None
    result = evokeds[minuend].copy()
    result.data = evokeds[minuend].data - evokeds[subtrahend].data
    result.comment = str(diff.get("label", f"{minuend} - {subtrahend}"))
    return result


def evokeds_with_difference(evokeds: Mapping[str, Any], difference, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return condition evokeds plus the configured difference wave when present."""

    combined = dict(evokeds)
    if difference is not None:
        combined[str(dict(settings.get("difference_wave", {}) or {}).get("name", "difference_wave"))] = difference
    return combined


def build_condition_counts(condition_epochs: Mapping[str, Any], metadata: pd.DataFrame, settings: Mapping[str, Any], epochs) -> pd.DataFrame:
    """Summarise retained/dropped epochs by configured condition."""

    minimum_trials = configured_minimum_trials(settings)
    rows = []
    for condition, split in condition_epochs.items():
        retained = int(len(split))
        available = int(condition_mask(metadata, settings, configured_conditions(settings).get(condition, {}), condition).sum())
        rejected = max(0, available - retained)
        denominator = max(available, 1)
        threshold = int(minimum_trials.get(condition, 0) or 0)
        rows.append(
            {
                "condition": condition,
                "label": configured_conditions(settings).get(condition, {}).get("label", condition),
                "available_before_rejection": available,
                "retained_epochs": retained,
                "rejected_epochs": rejected,
                "percent_retained": 100.0 * retained / denominator,
                "percent_rejected": 100.0 * rejected / denominator,
                "minimum_trials": threshold,
                "minimum_trial_threshold_met": retained >= threshold,
                "condition_column": resolve_condition_column(metadata, settings),
                "epoch_input_scope": "canonical_retained_gate3c",
                "primary_inference_condition": condition in {"hit", "correct_rejection"},
            }
        )
    diff = dict(settings.get("difference_wave", {}) or {})
    if bool(diff.get("enabled", False)):
        name = str(diff.get("name", "difference_wave"))
        minuend = str(diff.get("minuend", ""))
        subtrahend = str(diff.get("subtrahend", ""))
        retained = min(int(len(condition_epochs.get(minuend, []))), int(len(condition_epochs.get(subtrahend, []))))
        threshold = int(minimum_trials.get("difference_wave", 0) or 0)
        rows.append(
            {
                "condition": name,
                "label": str(diff.get("label", name)),
                "available_before_rejection": retained,
                "retained_epochs": retained,
                "rejected_epochs": 0,
                "percent_retained": 100.0 if retained else 0.0,
                "percent_rejected": 0.0,
                "minimum_trials": threshold,
                "minimum_trial_threshold_met": retained >= threshold,
                "condition_column": "difference_wave",
                "epoch_input_scope": "derived_from_canonical_retained_gate3c",
                "primary_inference_condition": False,
            }
        )
    return pd.DataFrame(rows)


def roi_membership(configured_channels: Sequence[str], available_channel_names: Sequence[str]) -> dict[str, Any]:
    """Describe effective ROI membership and enforce the configured strict-majority rule."""

    requested = [str(channel) for channel in configured_channels]
    available_names = set(str(channel) for channel in available_channel_names)
    available = [channel for channel in requested if channel in available_names]
    missing = [channel for channel in requested if channel not in available_names]
    required = math.floor(len(requested) / 2) + 1 if requested else 1
    return {
        "requested_channels": requested,
        "available_channels": available,
        "missing_channels": missing,
        "channel_count": len(available),
        "required_channel_count": required,
        "roi_complete": bool(requested) and len(available) >= required,
    }


def extract_erp_measurements(
    *,
    subject: str,
    recording_part: str,
    branch: str,
    evokeds: Mapping[str, Any],
    condition_counts: pd.DataFrame,
    settings: Mapping[str, Any],
    metadata_source: str,
    processing_version: str,
) -> pd.DataFrame:
    """Extract mean-amplitude measurements for each condition, ROI, and window."""

    rois = configured_rois(settings)
    windows = configured_time_windows(settings)
    for evoked in evokeds.values():
        if hasattr(evoked, "info"):
            validate_downstream_retained_bad_channel_safety(evoked.info, settings)
    counts = condition_counts.set_index("condition") if not condition_counts.empty else pd.DataFrame()
    columns = [
        "subject",
        "recording_part",
        "branch",
        "condition",
        "roi",
        "time_window",
        "window_start_ms",
        "window_end_ms",
        "n_epochs",
        "mean_amplitude_uv",
        "baseline_window",
        "metadata_source",
        "processing_version",
    ]
    if branch == "recognition":
        columns.extend(
            [
                "participant",
                "contrast",
                "roi_status",
                "analysis_status",
                "hypothesis_family",
                "hypothesis_id",
                "trial_count",
                "requested_channels",
                "available_channels",
                "missing_channels",
                "channel_count",
                "required_channel_count",
                "roi_complete",
                "exact_channel_list",
                "n_time_samples",
                "window_interval",
                "amplitude_metric",
            ]
        )
    rows = []
    for condition, evoked in evokeds.items():
        for roi, configured_channels in rois.items():
            membership = roi_membership(configured_channels, getattr(evoked, "ch_names", []))
            channels = list(membership["available_channels"])
            channel_indices = [evoked.ch_names.index(channel) for channel in channels]
            for window_name, window in windows.items():
                start_s = float(window["start_ms"]) / 1000.0
                end_s = float(window["end_ms"]) / 1000.0
                time_mask = (evoked.times >= start_s) & (evoked.times <= end_s)
                if not bool(time_mask.any()) or not channels or (branch == "recognition" and not membership["roi_complete"]):
                    mean_uv = float("nan")
                else:
                    measurement_data = np.asarray(evoked.data[channel_indices][:, time_mask], dtype=float)
                    if not np.isfinite(measurement_data).all():
                        raise ValueError(
                            f"Non-finite ERP values cannot define measurement "
                            f"{subject!r}/{recording_part!r}/{branch!r}/{condition!r}/{roi!r}/{window_name!r}."
                        )
                    mean_uv = float(measurement_data.mean() * 1_000_000.0)
                n_epochs = int(counts.loc[condition, "retained_epochs"]) if condition in counts.index else pd.NA
                row = {
                        "subject": subject,
                        "recording_part": recording_part,
                        "branch": branch,
                        "condition": condition,
                        "roi": roi,
                        "time_window": window_name,
                        "window_start_ms": window["start_ms"],
                        "window_end_ms": window["end_ms"],
                        "n_epochs": n_epochs,
                        "mean_amplitude_uv": mean_uv,
                        "baseline_window": list(settings.get("baseline_s", [])),
                        "metadata_source": metadata_source,
                        "processing_version": processing_version,
                    }
                if branch == "recognition":
                    scientific = recognition_scientific_metadata(settings, condition, roi, window_name)
                    row.update(
                        {
                            "participant": subject,
                            **scientific,
                            "roi_status": recognition_roi_status(roi),
                            "trial_count": n_epochs,
                            "requested_channels": ";".join(membership["requested_channels"]),
                            "available_channels": ";".join(membership["available_channels"]),
                            "missing_channels": ";".join(membership["missing_channels"]),
                            "channel_count": membership["channel_count"],
                            "required_channel_count": membership["required_channel_count"],
                            "roi_complete": membership["roi_complete"],
                            "exact_channel_list": ";".join(membership["available_channels"]),
                            "n_time_samples": int(time_mask.sum()),
                            "window_interval": "closed_inclusive",
                            "amplitude_metric": "mean_amplitude_uv",
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def extract_recognition_measurements(
    *,
    subject: str,
    recording_part: str,
    evokeds: Mapping[str, Any],
    condition_counts: pd.DataFrame,
    settings: Mapping[str, Any],
    metadata_source: str,
    processing_version: str,
) -> pd.DataFrame:
    """Compatibility wrapper for Gate 4A recognition measurements."""

    return extract_erp_measurements(
        subject=subject,
        recording_part=recording_part,
        branch="recognition",
        evokeds=evokeds,
        condition_counts=condition_counts,
        settings=settings,
        metadata_source=metadata_source,
        processing_version=processing_version,
    )


def build_wide_erp_measurements(measurements: pd.DataFrame, *, subject: str, recording_part: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Build one-row-per-participant wide ERP measurement table."""

    row: dict[str, Any] = {"subject": subject, "recording_part": recording_part}
    if "branch" in measurements.columns and not measurements.empty:
        row["branch"] = str(measurements["branch"].iloc[0])
    if measurements.empty:
        return pd.DataFrame([row])
    for _, item in measurements.iterrows():
        column = wide_measurement_column(str(item["condition"]), str(item["roi"]), str(item["time_window"]), settings)
        row[column] = item["mean_amplitude_uv"]
    return pd.DataFrame([row])


def build_wide_recognition_measurements(measurements: pd.DataFrame, *, subject: str, recording_part: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Compatibility wrapper for Gate 4A wide recognition measurements."""

    return build_wide_erp_measurements(measurements, subject=subject, recording_part=recording_part, settings=settings)


def build_recognition_contrast_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    """Build paired Hit/CR participant rows without adding a second inferential test."""

    columns = [
        "participant", "subject", "recording_part", "window", "window_start_ms", "window_end_ms", "roi",
        "roi_status", "analysis_status", "hypothesis_family", "hypothesis_id", "channel_count",
        "exact_channel_list", "requested_channels", "available_channels", "missing_channels", "roi_complete",
        "hit_mean_uv", "cr_mean_uv", "hit_minus_cr_uv", "hit_trial_count", "cr_trial_count",
    ]
    required = {"condition", "roi", "time_window", "mean_amplitude_uv"}
    if measurements.empty or not required.issubset(measurements.columns):
        return pd.DataFrame(columns=columns)
    source = measurements[measurements["condition"].astype(str).isin(["hit", "correct_rejection"])].copy()
    duplicate_key = ["subject", "recording_part", "roi", "time_window", "condition"]
    if set(duplicate_key).issubset(source.columns):
        duplicate = source.duplicated(duplicate_key, keep=False)
        if bool(duplicate.any()):
            bad = source.loc[duplicate, duplicate_key].drop_duplicates()
            raise ValueError(
                f"Duplicate recognition measurement before participant contrast; key={tuple(duplicate_key)!r}: "
                f"{bad.head(5).to_dict(orient='records')!r}"
            )
    rows: list[dict[str, Any]] = []
    keys = ["subject", "recording_part", "roi", "time_window"]
    for key, cell in source.groupby(keys, dropna=False, sort=True):
        by_condition = cell.set_index("condition")
        if not {"hit", "correct_rejection"}.issubset(by_condition.index):
            continue
        hit = by_condition.loc["hit"]
        cr = by_condition.loc["correct_rejection"]
        hit_value = pd.to_numeric(pd.Series([hit.get("mean_amplitude_uv")]), errors="coerce").iloc[0]
        cr_value = pd.to_numeric(pd.Series([cr.get("mean_amplitude_uv")]), errors="coerce").iloc[0]
        if bool(np.isinf([hit_value, cr_value]).any()):
            raise ValueError(f"Infinite recognition contrast measurement at participant cell {key!r}.")
        row = {
            "participant": key[0], "subject": key[0], "recording_part": key[1], "roi": key[2], "window": key[3],
            "window_start_ms": hit.get("window_start_ms", pd.NA), "window_end_ms": hit.get("window_end_ms", pd.NA),
            "roi_status": hit.get("roi_status", ""), "analysis_status": hit.get("analysis_status", ""),
            "hypothesis_family": hit.get("hypothesis_family", ""), "hypothesis_id": hit.get("hypothesis_id", ""),
            "channel_count": hit.get("channel_count", pd.NA), "exact_channel_list": hit.get("exact_channel_list", ""),
            "requested_channels": hit.get("requested_channels", ""), "available_channels": hit.get("available_channels", ""),
            "missing_channels": hit.get("missing_channels", ""), "roi_complete": hit.get("roi_complete", pd.NA),
            "hit_mean_uv": hit_value, "cr_mean_uv": cr_value,
            "hit_minus_cr_uv": hit_value - cr_value if pd.notna(hit_value) and pd.notna(cr_value) else float("nan"),
            "hit_trial_count": hit.get("trial_count", hit.get("n_epochs", pd.NA)),
            "cr_trial_count": cr.get("trial_count", cr.get("n_epochs", pd.NA)),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_measurement_dictionary(measurements: pd.DataFrame, wide_measurements: pd.DataFrame, *, branch: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Build a compact data dictionary for long and wide participant measurement exports."""

    rows: list[dict[str, Any]] = []
    descriptions = {
        "subject": "Participant identifier.",
        "recording_part": "Recording part identifier.",
        "branch": "ERP analysis branch.",
        "condition": "Participant-level ERP condition.",
        "roi": "Region of interest used for the measurement.",
        "time_window": "Configured latency window identifier.",
        "window_start_ms": "Latency-window start relative to image onset.",
        "window_end_ms": "Latency-window end relative to image onset.",
        "n_epochs": "Retained epochs contributing to the condition evoked.",
        "mean_amplitude_uv": "Mean ERP amplitude over the ROI and time window.",
        "baseline_window": "Baseline interval applied before averaging.",
        "metadata_source": "Source label for behavioural epoch metadata.",
        "processing_version": "Participant-level ERP measurement processing version.",
        "participant": "Participant identifier (explicit scientific-table alias of subject).",
        "contrast": "Contrast supported by the condition/cell, where applicable.",
        "roi_status": "A priori temporal role of the configured ROI.",
        "analysis_status": "Machine-readable PRIMARY, SECONDARY, DESCRIPTIVE, or EXPLORATORY status.",
        "hypothesis_family": "Predefined scientific hypothesis family.",
        "hypothesis_id": "Finite predefined hypothesis identifier, blank for non-primary rows.",
        "trial_count": "Retained post-Phase-2A-4 epochs contributing to the condition evoked.",
        "requested_channels": "Configured ROI channels in declared order.",
        "available_channels": "Configured ROI channels available in the participant evoked.",
        "missing_channels": "Configured ROI channels absent from the participant evoked.",
        "channel_count": "Number of available ROI channels contributing to the measurement.",
        "required_channel_count": "Strict-majority minimum channel count for an interpretable recognition ROI.",
        "roi_complete": "Whether the recognition ROI passes the explicit strict-majority completeness rule.",
        "exact_channel_list": "Exact ordered channel list used for the measurement.",
        "n_time_samples": "Number of evoked samples included in the closed latency interval.",
        "window_interval": "Window endpoint convention; recognition uses closed/inclusive endpoints.",
        "amplitude_metric": "Participant-level dependent measure and unit.",
    }
    units = {
        "window_start_ms": "ms",
        "window_end_ms": "ms",
        "n_epochs": "epochs",
        "trial_count": "epochs",
        "channel_count": "channels",
        "required_channel_count": "channels",
        "n_time_samples": "samples",
        "mean_amplitude_uv": "uV",
    }
    for column in measurements.columns:
        rows.append(
            {
                "column_name": column,
                "description": descriptions.get(str(column), f"{branch} long-format measurement field."),
                "units": units.get(str(column), ""),
                "ROI": "varies by row" if column in {"roi", "mean_amplitude_uv"} else "",
                "time_window": "varies by row" if column in {"time_window", "mean_amplitude_uv"} else "",
                "source": f"{branch}_measurements.csv",
            }
        )
    for column in wide_measurements.columns:
        roi = ""
        time_window = ""
        if column not in {"subject", "recording_part", "branch"}:
            parts = str(column).split("_")
            if len(parts) >= 3:
                roi = parts[1]
                time_window = "_".join(parts[2:])
        rows.append(
            {
                "column_name": column,
                "description": "Wide-format participant mean ERP amplitude." if column not in {"subject", "recording_part", "branch"} else descriptions.get(str(column), f"{branch} wide-format identifier field."),
                "units": "uV" if column not in {"subject", "recording_part", "branch"} else "",
                "ROI": roi,
                "time_window": time_window,
                "source": f"{branch}_measurements_wide.csv",
            }
        )
    return pd.DataFrame(rows, columns=["column_name", "description", "units", "ROI", "time_window", "source"])


def wide_measurement_column(condition: str, roi: str, time_window: str, settings: Mapping[str, Any]) -> str:
    """Return stable wide-format column names such as Hit_Frontocentral_300_500."""

    condition_label = _wide_condition_label(condition, settings)
    roi_label = _wide_token(roi)
    window_label = re.sub(r"_?ms$", "", str(time_window), flags=re.IGNORECASE)
    window_label = re.sub(r"[^0-9A-Za-z]+", "_", window_label).strip("_")
    return f"{condition_label}_{roi_label}_{window_label}"


def _wide_condition_label(condition: str, settings: Mapping[str, Any]) -> str:
    diff = dict(settings.get("difference_wave", {}) or {})
    if condition == str(diff.get("name", "difference_wave")):
        return "HitMinusCR" if condition == "hit_minus_correct_rejection" else _wide_token(str(diff.get("label", condition)))
    label = str(configured_conditions(settings).get(condition, {}).get("label", condition))
    return _wide_token(label)


def _wide_token(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", str(value))
    return "".join(part[:1].upper() + part[1:] for part in parts)


def build_erp_qc_summary(
    *,
    subject: str,
    recording_part: str,
    branch: str,
    settings: Mapping[str, Any],
    condition_counts: pd.DataFrame,
    missing_conditions_list: Sequence[str],
    difference_wave_generated: bool,
    metadata: pd.DataFrame | None = None,
    processing_info: Mapping[str, Any] | None = None,
    participant_status: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Summarise participant-level ERP QC for one branch."""

    threshold_flags = {}
    if not condition_counts.empty:
        threshold_flags = {
            str(row["condition"]): bool(row["minimum_trial_threshold_met"])
            for _, row in condition_counts.iterrows()
            if "minimum_trial_threshold_met" in row
        }
    status_info = dict(participant_status or {})
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "recording_part": recording_part,
                "branch": branch,
                "conditions_available": [condition for condition in configured_conditions(settings) if condition not in set(missing_conditions_list)],
                "conditions_missing": list(missing_conditions_list),
                "available_metadata_columns": available_metadata_columns(metadata) if metadata is not None else [],
                "available_recognition_values": available_recognition_values(metadata, settings) if metadata is not None else {},
                "derived_condition_counts": condition_counts_from_metadata(metadata, settings) if metadata is not None else {},
                "blank_condition_metadata_fields": blank_condition_metadata_fields(metadata, settings) if metadata is not None else {},
                "trial_counts": condition_counts.to_dict(orient="records"),
                "rois_analysed": list(configured_rois(settings).keys()),
                "time_windows_analysed": list(configured_time_windows(settings).keys()),
                "baseline": list(settings.get("baseline_s", [])),
                "difference_wave_generated": bool(difference_wave_generated),
                "minimum_trial_flags": threshold_flags,
                "processing_info": dict(processing_info or {}),
                "participant_reporting_status": status_info.get("status", ""),
                "participant_reporting_reason_code": status_info.get("reason_code", ""),
                "participant_reporting_reason": status_info.get("reason", ""),
                "preprocessing_status": status_info.get("preprocessing", "Completed successfully"),
                "recognition_erp_status": status_info.get("recognition_erp", ""),
                "cuteness_erp_status": status_info.get("cuteness_erp", ""),
                "eligible_for_future_analyses": status_info.get("eligible_for_future_analyses", ""),
            }
        ]
    )


def build_recognition_qc_summary(
    *,
    subject: str,
    recording_part: str,
    settings: Mapping[str, Any],
    condition_counts: pd.DataFrame,
    missing_conditions_list: Sequence[str],
    difference_wave_generated: bool,
    metadata: pd.DataFrame | None = None,
    participant_status: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper for Gate 4A recognition QC."""

    return build_erp_qc_summary(
        subject=subject,
        recording_part=recording_part,
        branch="recognition",
        settings=settings,
        condition_counts=condition_counts,
        missing_conditions_list=missing_conditions_list,
        difference_wave_generated=difference_wave_generated,
        metadata=metadata,
        participant_status=participant_status,
    )


def build_continuous_summary(
    *,
    subject: str,
    recording_part: str,
    branch: str,
    metadata: pd.DataFrame,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """Summarise continuous behavioural predictors for participant-level ERP follow-up."""

    continuous = dict(settings.get("continuous", {}) or {})
    columns = [str(col) for col in continuous.get("columns", []) or []]
    if not columns:
        column = continuous.get("column")
        if column:
            columns = [str(column)]
    rows = []
    for column in columns:
        if column not in metadata.columns:
            rows.append({"subject": subject, "recording_part": recording_part, "branch": branch, "predictor": column, "available": False, "n_trials": 0, "mean": pd.NA, "sd": pd.NA, "min": pd.NA, "max": pd.NA})
            continue
        numeric = pd.to_numeric(metadata[column], errors="coerce")
        if bool(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any()):
            raise ValueError(
                f"Infinite continuous predictor {column!r} cannot enter participant summary "
                f"for {subject!r}/{recording_part!r}/{branch!r}."
            )
        values = numeric.dropna()
        rows.append(
            {
                "subject": subject,
                "recording_part": recording_part,
                "branch": branch,
                "predictor": column,
                "available": bool(len(values)),
                "n_trials": int(len(values)),
                "mean": float(values.mean()) if len(values) else pd.NA,
                "sd": float(values.std()) if len(values) > 1 else 0.0 if len(values) else pd.NA,
                "min": float(values.min()) if len(values) else pd.NA,
                "max": float(values.max()) if len(values) else pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=["subject", "recording_part", "branch", "predictor", "available", "n_trials", "mean", "sd", "min", "max"])


def classify_participant_erp_status(
    *,
    branch: str,
    settings: Mapping[str, Any],
    metadata: pd.DataFrame,
    condition_counts: pd.DataFrame,
    evokeds: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe participant-level ERP reporting status without changing analysis results."""

    branch_label = "Recognition ERP" if branch == "recognition" else "Cuteness ERP" if branch == "cuteness" else f"{branch} ERP"
    other_label = "Cuteness ERP" if branch == "recognition" else "Recognition ERP" if branch == "cuteness" else "Other ERP"
    branch_status = "Completed" if evokeds else "Skipped"
    missing_fields = missing_condition_metadata_fields(metadata, settings)
    blank_fields = blank_condition_metadata_fields(metadata, settings)
    reconciliation = recognition_condition_reconciliation(metadata, settings)
    missing_conditions_list = missing_conditions(metadata, settings)
    retained_total = int(pd.to_numeric(condition_counts.get("retained_epochs", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not condition_counts.empty else 0
    available_total = int(pd.to_numeric(condition_counts.get("available_before_rejection", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not condition_counts.empty else 0

    status = "Completed"
    reason_code = "analysis_completed"
    reason = "ERP conditions were derived and participant-level ERP outputs were generated."
    eligible = "Yes"

    if reconciliation.get("status") == "failed":
        status = "QC-only"
        reason_code = str(reconciliation.get("reason_code") or "recognition_condition_unreconciled_rows")
        reason = (
            "Retained recognition metadata contain row(s) that do not map to exactly one "
            "configured recognition condition."
        )
        eligible = "No (recognition metadata require reconciliation)."
    elif missing_fields:
        status = "QC-only"
        reason_code = "behavioural_metadata_missing"
        reason = "Missing behavioural metadata required for behavioural-condition derivation."
        eligible = "No (unless behavioural metadata become available)."
    elif blank_fields:
        status = "QC-only"
        reason_code = "behavioural_metadata_incomplete"
        reason = "Behavioural metadata required for behavioural-condition derivation are present but incomplete or blank."
        eligible = "No (unless behavioural metadata become complete)."
    elif available_total == 0 and configured_conditions(settings):
        status = "QC-only"
        reason_code = "behavioural_metadata_inconsistent"
        reason = "Behavioural metadata are present but do not match configured ERP condition definitions."
        eligible = "No (unless behavioural metadata or condition configuration are resolved)."
    elif available_total > 0 and retained_total == 0:
        status = "QC-only"
        reason_code = "zero_usable_trials"
        reason = "Participant has zero usable trials after applying existing ERP condition and retention rules."
        eligible = "No (zero usable participant-level ERP trials)."
    elif missing_conditions_list:
        status = "Completed with QC flags"
        reason_code = "behavioural_metadata_incomplete"
        reason = "Some configured behavioural ERP conditions have no matching trials."
        eligible = "Review QC flags before future analyses."

    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "preprocessing": "Completed successfully",
        "recognition_erp": branch_status if branch == "recognition" else f"Not run in this {branch_label} branch.",
        "cuteness_erp": branch_status if branch == "cuteness" else f"Not run in this {branch_label} branch.",
        "eligible_for_future_analyses": eligible,
        "branch_label": branch_label,
        "other_branch_label": other_label,
    }


def build_erp_parameters(settings: Mapping[str, Any], epochs, evokeds: Mapping[str, Any]) -> pd.DataFrame:
    """Record participant-level ERP analysis parameters."""

    first_evoked = next(iter(evokeds.values()), None)
    included_channels = list(getattr(first_evoked, "ch_names", []) or [])
    sfreq = 0.0
    if hasattr(epochs, "info") and hasattr(epochs.info, "get"):
        sfreq = float(epochs.info.get("sfreq") or 0.0)
    return pd.DataFrame(
        [
            {
                "branch": settings.get("branch", "recognition"),
                "baseline_s": list(settings.get("baseline_s", [-0.2, 0.0])),
                "picks": erp_picks(settings),
                "configured_channels": list(settings.get("channels", []) or []),
                "included_channels": included_channels,
                "n_included_channels": len(included_channels),
                "plotting_excluded_channels": plotting_channel_exclusions(settings),
                "plotting_exclusion_rationale": plotting_channel_exclusion_rationale(settings),
                "latency_window_s": list(settings.get("latency_window_s", [])),
                "measurement_time_windows": configured_time_windows(settings),
                "topography_time_windows": configured_time_windows(settings, key="topography_time_windows"),
                "rois": configured_rois(settings),
                "minimum_trials": configured_minimum_trials(settings),
                "difference_wave": dict(settings.get("difference_wave", {}) or {}),
                "conditions": list(configured_conditions(settings).keys()),
                "sfreq_hz": sfreq,
            }
        ]
    )


def build_processing_log(
    subject: str,
    recording_part: str,
    settings: Mapping[str, Any],
    validation: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    duration_s: float | None = None,
    figures_generated: int | None = None,
    topographies_generated: int | None = None,
    measurement_rows: int | None = None,
) -> pd.DataFrame:
    """Create a concise participant-level ERP processing log."""

    return pd.DataFrame(
        [
            {"step": "load_gate3c_epochs", "detail": f"subject={subject}; recording_part={recording_part}"},
            {"step": "validate_inputs", "detail": validation.to_dict(orient="records")},
            {"step": "baseline", "detail": f"baseline_s={list(settings.get('baseline_s', []))}"},
            {"step": "condition_split", "detail": counts.to_dict(orient="records")},
            {"step": "average", "detail": f"conditions={list(configured_conditions(settings).keys())}"},
            {"step": "measurements", "detail": f"rois={list(configured_rois(settings).keys())}; windows={list(configured_time_windows(settings).keys())}"},
            {"step": "difference_wave", "detail": dict(settings.get("difference_wave", {}) or {})},
            {"step": "plotting_channel_exclusions", "detail": {"channels": plotting_channel_exclusions(settings), "rationale": plotting_channel_exclusion_rationale(settings)}},
            {
                "step": "processing_summary",
                "detail": {
                    "analysis_duration_s": duration_s,
                    "number_of_conditions_analysed": len([row for _, row in counts.iterrows() if int(row.get("retained_epochs", 0) or 0) > 0]) if not counts.empty else 0,
                    "number_of_figures_generated": figures_generated,
                    "number_of_topographies": topographies_generated,
                    "number_of_measurement_rows_exported": measurement_rows,
                },
            },
        ]
    )


def save_participant_erps(evokeds: Mapping[str, Any], path: Path) -> Path:
    """Save participant evoked objects for later Gate 5 use."""

    import mne

    path.parent.mkdir(parents=True, exist_ok=True)
    mne.write_evokeds(path, list(evokeds.values()), overwrite=True)
    return path


def write_erp_waveform_figure(evokeds: Mapping[str, Any], settings: Mapping[str, Any], path: Path, *, subject: str, recording_part: str, root: Path | None, config_path: str | Path | None, input_files: Sequence[str | Path], warnings: Sequence[str], figure_style: FigureStyle | None = None) -> Path | None:
    """Write condition-overlaid participant ERP waveforms."""

    figures = dict(settings.get("figures", {}) or {})
    if not bool(figures.get("enabled", True)) or not evokeds:
        return None
    import mne

    combine = figures.get("waveform_combine", "mean")
    style = figure_style or default_figure_style()
    plot_evokeds = plotting_safe_evokeds(evokeds, settings)
    branch = str(settings.get("branch", "recognition"))
    display_evokeds = {style.condition_label(condition): evoked for condition, evoked in plot_evokeds.items()}
    colors = {style.condition_label(condition): style.condition_color(branch, condition) for condition in plot_evokeds}
    colors = {condition: color for condition, color in colors.items() if color is not None}
    plot_styles = {
        style.condition_label(condition): {
            "linestyle": style.condition_line_style(branch, condition),
            "linewidth": style.condition_line_width(branch, condition),
        }
        for condition in plot_evokeds
    }
    try:
        plot = mne.viz.plot_compare_evokeds(display_evokeds, picks=erp_picks(settings), combine=combine, colors=colors or None, styles=plot_styles, show=False)
    except TypeError:
        # Compatibility with test doubles and older MNE versions without ``colors``.
        plot = mne.viz.plot_compare_evokeds(display_evokeds, picks=erp_picks(settings), combine=combine, show=False)
    figure = plot[0] if isinstance(plot, list) else plot
    figure.set_size_inches(*style.figure_size("waveform"))
    figure.set_facecolor(style.figure_facecolor)
    for ax in figure.axes:
        style_axes(ax, style)
    return save_figure_with_stage_provenance(
        figure,
        path,
        pipeline_stage=f"gate4_{settings.get('branch', 'erp')}_erp_figure",
        subject=subject,
        recording_part=recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary={"figure_type": "waveforms", "combine": combine, "branch": settings.get("branch", "recognition"), "plotting_excluded_channels": plotting_channel_exclusions(settings)},
        warnings=warnings,
        dpi=style.dpi,
        facecolor=style.figure_facecolor,
    )


def write_difference_waveform_figure(difference, settings: Mapping[str, Any], path: Path, *, subject: str, recording_part: str, root: Path | None, config_path: str | Path | None, input_files: Sequence[str | Path], warnings: Sequence[str], figure_style: FigureStyle | None = None) -> Path | None:
    """Write the participant-level recognition difference waveform."""

    if difference is None:
        return None
    import matplotlib.pyplot as plt

    safe_difference = plotting_safe_evokeds({"difference": difference}, settings)["difference"]
    validate_downstream_retained_bad_channel_safety(difference.info, settings)
    style = figure_style or default_figure_style()
    fig, ax = plt.subplots(figsize=style.figure_size("difference_waveform"), facecolor=style.figure_facecolor)
    data_uv = safe_difference.data.mean(axis=0) * 1_000_000.0
    ax.plot(safe_difference.times * 1000.0, data_uv, color=style.color("difference_waveform"), linewidth=style.line_widths["difference_waveform"], label=difference.comment)
    reference_alpha = float(style.axes["reference_alpha"])
    ax.axhline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=reference_alpha)
    ax.axvline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=reference_alpha)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{subject} {recording_part} · {style.condition_label('hit_minus_correct_rejection')}")
    style_axes(ax, style)
    ax.title.set_fontsize(style.fonts["difference_title_size"])
    style_legend(ax, style)
    return save_figure_with_stage_provenance(
        fig,
        path,
        pipeline_stage=f"gate4_{settings.get('branch', 'erp')}_erp_figure",
        subject=subject,
        recording_part=recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary={"figure_type": "difference_waveform", "branch": settings.get("branch", "recognition")},
        warnings=warnings,
        dpi=style.dpi,
        facecolor=style.figure_facecolor,
    )


def write_roi_waveform_figure(evokeds: Mapping[str, Any], settings: Mapping[str, Any], path: Path, *, subject: str, recording_part: str, root: Path | None, config_path: str | Path | None, input_files: Sequence[str | Path], warnings: Sequence[str], figure_style: FigureStyle | None = None) -> Path | None:
    """Write ROI-averaged participant waveforms."""

    rois = configured_rois(settings)
    if not rois or not evokeds:
        return None
    import matplotlib.pyplot as plt

    evokeds = plotting_safe_evokeds(evokeds, settings)
    style = figure_style or default_figure_style()
    branch = str(settings.get("branch", "recognition"))
    reference_alpha = float(style.axes["reference_alpha"])
    evokeds = {
        condition: evoked
        for condition, evoked in evokeds.items()
        if not (branch == "recognition" and condition == "hit_minus_correct_rejection")
    }
    rois = {
        roi: channels
        for roi, channels in rois.items()
        if any(any(channel in evoked.ch_names for channel in channels) for evoked in evokeds.values())
    }
    if not rois:
        return None
    fig, axes = plt.subplots(len(rois), 1, figsize=style.figure_size("participant_roi_waveform", count=len(rois)), sharex=True, facecolor=style.figure_facecolor)
    if len(rois) == 1:
        axes = [axes]
    plotted_values: list[np.ndarray] = []
    for ax, (roi, channels) in zip(axes, rois.items()):
        for condition, evoked in evokeds.items():
            available = [channel for channel in channels if channel in evoked.ch_names]
            if not available:
                continue
            picks = [evoked.ch_names.index(channel) for channel in available]
            values = evoked.data[picks].mean(axis=0) * 1_000_000.0
            plotted_values.append(np.asarray(values, dtype=float))
            ax.plot(evoked.times * 1000.0, values, linewidth=style.condition_line_width(branch, condition, roi=True), linestyle=style.condition_line_style(branch, condition), label=style.condition_label(condition), color=style.condition_color(branch, condition))
        ax.axhline(0.0, color=style.color("reference"), linewidth=style.line_widths["participant_roi_reference"], alpha=reference_alpha)
        ax.axvline(0.0, color=style.color("reference"), linewidth=style.line_widths["participant_roi_reference"], alpha=reference_alpha)
        ax.set_ylabel(f"{str(roi).replace('_', ' ').title()}\n(µV)")
        shade_primary_windows(ax, settings, style, roi=str(roi))
        style_axes(ax, style)
        ax.title.set_fontsize(style.fonts["panel_title_size"])
        if not bool(style.legend["shared_roi"]):
            style_legend(ax, style)
    if plotted_values:
        minimum = min(0.0, min(float(np.nanmin(values)) for values in plotted_values))
        maximum = max(0.0, max(float(np.nanmax(values)) for values in plotted_values))
        span = maximum - minimum
        padding = 0.08 * span if span > 0 else 1.0
        for ax in axes:
            ax.set_ylim(minimum - padding, maximum + padding)
    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"{subject} {recording_part} · {style.branch_label(branch)} ROI waveforms", fontsize=style.fonts["title_size"])
    if bool(style.legend["shared_roi"]):
        handles_by_label: dict[str, Any] = {}
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            handles_by_label.update(zip(labels, handles))
        labels = list(handles_by_label)
        handles = [handles_by_label[label] for label in labels]
        fig.legend(
            handles,
            labels,
            loc=str(style.legend["shared_location"]),
            bbox_to_anchor=tuple(style.legend["shared_bbox_to_anchor"]),
            ncol=min(int(style.legend["shared_columns"]), max(1, len(labels))),
            frameon=bool(style.legend["frameon"]),
            fontsize=style.fonts["legend_size"],
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    else:
        fig.tight_layout()
    return save_figure_with_stage_provenance(
        fig,
        path,
        pipeline_stage=f"gate4_{settings.get('branch', 'erp')}_erp_figure",
        subject=subject,
        recording_part=recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary={"figure_type": "roi_waveforms", "rois": configured_rois(settings), "branch": settings.get("branch", "recognition"), "plotting_excluded_channels": plotting_channel_exclusions(settings)},
        warnings=warnings,
        dpi=style.dpi,
        facecolor=style.figure_facecolor,
    )


def write_trial_count_summary_figure(condition_counts: pd.DataFrame, settings: Mapping[str, Any], path: Path, *, subject: str, recording_part: str, root: Path | None, config_path: str | Path | None, input_files: Sequence[str | Path], warnings: Sequence[str], figure_style: FigureStyle | None = None) -> Path | None:
    """Write retained trial counts and minimum-threshold flags."""

    if condition_counts.empty:
        return None
    import matplotlib.pyplot as plt

    df = condition_counts.copy()
    style = figure_style or default_figure_style()
    colours = df["minimum_trial_threshold_met"].astype(bool).map({True: style.bar_color("threshold_met"), False: style.bar_color("threshold_failed")})
    fig, ax = plt.subplots(figsize=style.figure_size("participant_trial_count"), facecolor=style.figure_facecolor)
    labels = [style.condition_label(condition) for condition in df["condition"].astype(str)]
    ax.bar(labels, pd.to_numeric(df["retained_epochs"], errors="coerce"), color=colours)
    ax.set_ylabel("Retained epochs")
    ax.set_xlabel("Condition")
    ax.set_title(f"{subject} {recording_part} {settings.get('branch', 'ERP')} trial counts")
    ax.tick_params(axis="x", rotation=25)
    style_axes(ax, style)
    fig.tight_layout()
    return save_figure_with_stage_provenance(
        fig,
        path,
        pipeline_stage=f"gate4_{settings.get('branch', 'erp')}_erp_figure",
        subject=subject,
        recording_part=recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary={"figure_type": "trial_count_summary", "minimum_trials": configured_minimum_trials(settings), "branch": settings.get("branch", "recognition")},
        warnings=warnings,
        dpi=style.dpi,
        facecolor=style.figure_facecolor,
    )


def write_topography_figures(evokeds: Mapping[str, Any], settings: Mapping[str, Any], topography_dir: Path, *, subject: str, recording_part: str, root: Path | None, config_path: str | Path | None, input_files: Sequence[str | Path], warnings: Sequence[str], figure_style: FigureStyle | None = None) -> dict[str, Path]:
    """Write configured participant-level topographic maps."""

    outputs: dict[str, Path] = {}
    style = figure_style or default_figure_style()
    evokeds = plotting_safe_evokeds(evokeds, settings)
    windows = configured_time_windows(settings, key="topography_time_windows")
    requested = [str(item) for item in settings.get("topography_conditions", []) or []]
    if not requested:
        requested = ["hit", "correct_rejection", str(dict(settings.get("difference_wave", {}) or {}).get("name", "difference_wave"))]
    limits = _participant_topography_limits(requested, windows, evokeds, style)
    for condition in requested:
        if condition not in evokeds:
            continue
        evoked = evokeds[condition]
        for window_name, window in windows.items():
            start_s = float(window["start_ms"]) / 1000.0
            end_s = float(window["end_ms"]) / 1000.0
            midpoint = (start_s + end_s) / 2.0
            duration = max(0.001, end_s - start_s)
            try:
                is_difference = "minus" in condition or "difference" in condition
                kwargs = {
                    "cmap": style.topography_colormap(difference=is_difference),
                    "contours": int(style.topography["contour_count"]),
                }
                if condition in limits:
                    kwargs["vlim"] = limits[condition]
                fig = evoked.plot_topomap(times=[midpoint], average=duration, show=False, time_unit="s", **kwargs)
            except TypeError:
                try:
                    fig = evoked.plot_topomap(times=[midpoint], average=duration, show=False, time_unit="s")
                except TypeError:
                    fig = evoked.plot_topomap(times=[midpoint], show=False, time_unit="s")
            if fig.axes:
                window_label = str(window.get("label", str(window_name).replace("_", " "))).replace("-", "–", 1)
                fig.axes[0].set_title("")
                fig.set_size_inches(*style.figure_size("topography"), forward=True)
                fig.suptitle(
                    f"{style.condition_label(condition)} · {window_label}",
                    y=0.98,
                    fontsize=style.fonts["compact_legend_size"],
                )
            style_topography_figure(fig, style)
            path = topography_dir / f"{condition}_{window_name}.png"
            outputs[f"topography_{condition}_{window_name}"] = save_figure_with_stage_provenance(
                fig,
                path,
                pipeline_stage=f"gate4_{settings.get('branch', 'erp')}_erp_figure",
                subject=subject,
                recording_part=recording_part,
                root=root,
                config_path=config_path,
                input_files=input_files,
                parameters_summary={"figure_type": "topography", "condition": condition, "time_window": window, "branch": settings.get("branch", "recognition")},
                warnings=warnings,
                dpi=style.dpi,
                facecolor=style.figure_facecolor,
            )
    return outputs


def _participant_topography_limits(
    conditions: Sequence[str],
    windows: Mapping[str, Mapping[str, Any]],
    evokeds: Mapping[str, Any],
    style: FigureStyle,
) -> dict[str, tuple[float, float]]:
    if not bool(style.topography["symmetric_voltage_limits"]):
        return {}
    maxima: dict[str, float] = {}
    for condition in conditions:
        if condition not in evokeds:
            continue
        maximum = 0.0
        times = np.asarray(evokeds[condition].times, dtype=float)
        data = np.asarray(evokeds[condition].data, dtype=float)
        for window in windows.values():
            start = float(window["start_ms"]) / 1000.0
            end = float(window["end_ms"]) / 1000.0
            mask = (times >= start) & (times <= end)
            if mask.any():
                maximum = max(maximum, float(np.nanmax(np.abs(data[:, mask].mean(axis=1)))) * 1_000_000.0)
        maxima[condition] = maximum
    differences = [condition for condition in maxima if "minus" in condition or "difference" in condition]
    ordinary = [condition for condition in maxima if condition not in differences]
    limits: dict[str, tuple[float, float]] = {}
    if ordinary:
        common = max(maxima[condition] for condition in ordinary)
        for condition in ordinary:
            limit = common if bool(style.topography["common_condition_scale"]) else maxima[condition]
            if limit > 0:
                limits[condition] = (-limit, limit)
    for condition in differences:
        limit = maxima[condition]
        if limit > 0 and bool(style.topography["center_difference_at_zero"]):
            limits[condition] = (-limit, limit)
    return limits


def generate_erp_report(*, subject: str, recording_part: str, branch: str, input_files: Sequence[str | Path], output_files: Mapping[str, Path], parameters: pd.DataFrame, counts: pd.DataFrame, validation: pd.DataFrame, warnings: Sequence[str], failures: Sequence[str], participant_status: Mapping[str, Any] | None = None) -> str:
    """Create a compact participant-level ERP Markdown report."""

    branch_label = str(branch).replace("_", " ").title()
    lines = [
        f"# Gate 4 {branch_label} ERP: {subject} {recording_part}",
        "",
        "## Policy",
        "- Participant-level ERP products only.",
        "- Gate 3C epochs were loaded; epochs were not recreated.",
        "- No group analyses, inferential statistics, decoding, or behavioural accuracy/statistics were run.",
        "",
        "## Inputs",
    ]
    lines.extend(f"- `{path}`" for path in input_files)
    row = parameters.iloc[0].to_dict() if not parameters.empty else {}
    status_info = dict(participant_status or {})
    lines.extend(
        [
            "",
            "## Participant Status",
            f"- Status: {status_info.get('status', 'n/a')}",
            f"- Reason: {status_info.get('reason', 'n/a')}",
            f"- Preprocessing: {status_info.get('preprocessing', 'Completed successfully')}",
            f"- Recognition ERP: {status_info.get('recognition_erp', 'n/a')}",
            f"- Cuteness ERP: {status_info.get('cuteness_erp', 'n/a')}",
            f"- Eligible for future analyses: {status_info.get('eligible_for_future_analyses', 'n/a')}",
            "",
            "## Parameters",
            f"- Branch: {branch}",
            f"- Baseline: {row.get('baseline_s', 'n/a')}",
            f"- Latency window: {row.get('latency_window_s', 'n/a')}",
            f"- Channels included: {row.get('n_included_channels', 'n/a')}",
            "",
            "## Validation",
        ]
    )
    for _, item in validation.iterrows():
        lines.append(f"- {item.get('check')}: {item.get('status')} ({item.get('detail')})")
    lines.extend(["", "## Condition Counts"])
    for _, item in counts.iterrows():
        lines.append(f"- {item.get('condition')}: retained={item.get('retained_epochs')} dropped_total={item.get('dropped_epochs_total')}")
    lines.extend(["", "## Outputs"])
    for label, path in output_files.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    lines.extend(["", "## Failures"])
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def validate_erp_outputs(output_files: Mapping[str, Path], *, branch: str) -> list[str]:
    """Return missing required participant ERP output labels."""

    required = [
        "erp_parameters",
        "condition_counts",
        f"{branch}_measurements",
        f"{branch}_measurements_wide",
        "measurement_dictionary",
        f"{branch}_qc_summary",
        "processing_log",
        "report",
        "participant_erp_identity",
    ]
    return [label for label in required if label not in output_files or not Path(output_files[label]).exists()]


def validate_gate4a_outputs(output_files: Mapping[str, Path]) -> list[str]:
    """Return missing required Gate 4A participant output labels."""

    missing = validate_erp_outputs(output_files, branch="recognition")
    if "recognition_contrasts" not in output_files or not Path(output_files["recognition_contrasts"]).exists():
        missing.append("recognition_contrasts")
    return missing


def _write_participant_erp_identity(
    *,
    subject: str,
    recording_part: str,
    branch: str,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    metadata: pd.DataFrame,
    output_files: dict[str, Path],
    destination: Path,
) -> Path:
    required_labels = [
        "condition_counts",
        f"{branch}_measurements",
        f"{branch}_measurements_wide",
        f"{branch}_qc_summary",
    ]
    if branch == "cuteness":
        required_labels.append("cuteness_trial_level")
    missing = [label for label in required_labels if label not in output_files]
    if missing:
        raise RuntimeError(
            "Cannot publish participant ERP identity without Gate 5 input(s): "
            + ", ".join(missing)
        )
    binding = epoch_metadata_binding(metadata.reset_index(drop=True))
    identity = build_participant_erp_artifact_identity(
        participant=subject,
        session="session2",
        recording_part=recording_part,
        branch=branch,
        epoch_metadata_binding_fingerprint=binding.fingerprint,
        configuration_fingerprint=configuration_mapping_fingerprint(config),
        branch_settings=settings,
        output_paths={label: output_files[label] for label in required_labels},
    )
    return write_participant_erp_artifact_identity(identity, destination)


def run_gate4a_recognition_for_recording_part(
    subject: str,
    recording_part: str,
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    published_input_paths: Mapping[str, Path] | None = None,
) -> ERPResult:
    """Run participant-level Gate 4A recognition ERP analysis."""

    branch = "recognition"
    settings = gate4a_recognition_settings(config)
    figure_style = load_figure_style(root=root)
    outputs = erp_output_paths(config, subject, recording_part, branch, root=root)
    inputs = dict(published_input_paths or gate3c_erp_input_paths(config, subject, recording_part, root=root))
    input_files = [inputs["epochs_fif"], inputs["epoch_metadata_csv"]]
    output_files: dict[str, Path] = {}
    warnings: list[str] = []
    info_messages: list[str] = []
    failures: list[str] = []
    start_time = time.perf_counter()

    try:
        epochs, metadata = load_gate3c_epochs_and_metadata(inputs)
        epochs, metadata, _ = _validate_and_rebind_gate3c_epoch_metadata(
            epochs,
            metadata,
            participant=subject,
            session="session2",
            recording_part=recording_part,
            allow_legacy_serialization_compatibility=published_input_paths is not None,
        )
        epochs, exposure = prepare_recognition_inputs(
            epochs,
            metadata,
            subject=subject,
            session="session2",
            recording_part=recording_part,
            config=config,
            root=root,
        )
        metadata = exposure.metadata
        input_files.extend(path for path in exposure.input_paths if path not in input_files)
        validation = validate_epochs_metadata(
            epochs,
            metadata,
            settings,
            participant=subject,
            session="session2",
            recording_part=recording_part,
        )
        validation = append_source_exposure_validation(validation, exposure)
        epochs = apply_erp_baseline(epochs, settings)
        condition_epochs = split_epochs_by_condition(
            epochs,
            metadata,
            settings,
            participant=subject,
            session="session2",
            recording_part=recording_part,
        )
        evokeds = compute_participant_erps(condition_epochs, settings)
        difference_wave = compute_difference_wave(evokeds, settings)
        all_evokeds = evokeds_with_difference(evokeds, difference_wave, settings)
        counts = build_condition_counts(condition_epochs, metadata, settings, epochs)
        participant_status = classify_participant_erp_status(branch=branch, settings=settings, metadata=metadata, condition_counts=counts, evokeds=evokeds)
        missing = missing_conditions(metadata, settings)
        if missing:
            warnings.append(f"Recognition ERP condition(s) unavailable from behavioural metadata and retained as QC flags: {', '.join(missing)}")
        blank_fields = blank_condition_metadata_fields(metadata, settings)
        if blank_fields:
            warnings.append(
                "Configured recognition response metadata field(s) contain no usable values: "
                + "; ".join(f"{condition}: {', '.join(fields)}" for condition, fields in blank_fields.items())
            )
        if plotting_channel_exclusions(settings):
            info_messages.append(
                "Gate 4A figures exclude channel(s) with unresolved/overlapping montage positions: "
                + ", ".join(plotting_channel_exclusions(settings))
                + f". {plotting_channel_exclusion_rationale(settings)}"
            )
        if difference_wave is None and bool(dict(settings.get("difference_wave", {}) or {}).get("enabled", False)):
            warnings.append("Configured recognition difference wave could not be generated because one or both source evokeds are unavailable.")
        parameters = build_erp_parameters(settings, epochs, evokeds)
        metadata_source = _metadata_source(metadata)
        measurements = extract_recognition_measurements(
            subject=subject,
            recording_part=recording_part,
            evokeds=all_evokeds,
            condition_counts=counts,
            settings=settings,
            metadata_source=metadata_source,
            processing_version="gate4a2_recognition_measurements",
        )
        wide_measurements = build_wide_recognition_measurements(measurements, subject=subject, recording_part=recording_part, settings=settings)
        contrast_measurements = build_recognition_contrast_measurements(measurements)
        measurement_dictionary = build_measurement_dictionary(measurements, wide_measurements, branch=branch, settings=settings)
        qc_summary = build_recognition_qc_summary(
            subject=subject,
            recording_part=recording_part,
            settings=settings,
            condition_counts=counts,
            missing_conditions_list=missing,
            difference_wave_generated=difference_wave is not None,
            metadata=metadata,
            participant_status=participant_status,
        )
        qc_summary = attach_source_exposure_summary(qc_summary, exposure)

        for label, table in (
            ("erp_parameters", parameters),
            ("condition_counts", counts),
            ("recognition_measurements", measurements),
            ("recognition_measurements_wide", wide_measurements),
            ("recognition_contrasts", contrast_measurements),
            ("measurement_dictionary", measurement_dictionary),
            ("recognition_qc_summary", qc_summary),
        ):
            path = outputs[label]
            table.to_csv(path, index=False)
            output_files[label] = path
            _write_erp_sidecar(path, subject, recording_part, branch, config_path, root, input_files, warnings, settings)

        if bool(settings.get("save_evokeds_fif", True)) and evokeds:
            save_participant_erps(evokeds, outputs["participant_erps_fif"])
            output_files["participant_erps_fif"] = outputs["participant_erps_fif"]
            _write_erp_sidecar(outputs["participant_erps_fif"], subject, recording_part, branch, config_path, root, input_files, warnings, settings)
            if difference_wave is not None:
                save_participant_erps({"difference_wave": difference_wave}, outputs["difference_wave_fif"])
                output_files["difference_wave_fif"] = outputs["difference_wave_fif"]
                _write_erp_sidecar(outputs["difference_wave_fif"], subject, recording_part, branch, config_path, root, input_files, warnings, settings)

        try:
            waveform = write_erp_waveform_figure(
                all_evokeds,
                settings,
                outputs["waveforms"],
                subject=subject,
                recording_part=recording_part,
                root=root,
                config_path=config_path,
                input_files=input_files,
                warnings=warnings,
                figure_style=figure_style,
            )
            if waveform is not None:
                output_files["waveforms"] = waveform
            diff_waveform = write_difference_waveform_figure(
                difference_wave,
                settings,
                outputs["difference_waveform"],
                subject=subject,
                recording_part=recording_part,
                root=root,
                config_path=config_path,
                input_files=input_files,
                warnings=warnings,
                figure_style=figure_style,
            )
            if diff_waveform is not None:
                output_files["difference_waveform"] = diff_waveform
            roi_waveforms = write_roi_waveform_figure(
                all_evokeds,
                settings,
                outputs["roi_waveforms"],
                subject=subject,
                recording_part=recording_part,
                root=root,
                config_path=config_path,
                input_files=input_files,
                warnings=warnings,
                figure_style=figure_style,
            )
            if roi_waveforms is not None:
                output_files["roi_waveforms"] = roi_waveforms
            trial_counts = write_trial_count_summary_figure(
                counts,
                settings,
                outputs["trial_count_summary"],
                subject=subject,
                recording_part=recording_part,
                root=root,
                config_path=config_path,
                input_files=input_files,
                warnings=warnings,
                figure_style=figure_style,
            )
            if trial_counts is not None:
                output_files["trial_count_summary"] = trial_counts
            output_files.update(
                write_topography_figures(
                    all_evokeds,
                    settings,
                    outputs["topography_dir"],
                    subject=subject,
                    recording_part=recording_part,
                    root=root,
                    config_path=config_path,
                    input_files=input_files,
                    warnings=warnings,
                    figure_style=figure_style,
                )
            )
        except Exception as error:
            if bool(dict(settings.get("figures", {}) or {}).get("fail_pipeline_on_figure_error", False)):
                raise
            warnings.append(f"Could not generate recognition ERP waveform figure: {error}")

        topography_count = len([label for label in output_files if label.startswith("topography_")])
        figure_count = len([label for label in output_files if label in {"waveforms", "difference_waveform", "roi_waveforms", "trial_count_summary"} or label.startswith("topography_")])
        duration_s = round(time.perf_counter() - start_time, 3)
        processing_log = build_processing_log(
            subject,
            recording_part,
            settings,
            validation,
            counts,
            duration_s=duration_s,
            figures_generated=figure_count,
            topographies_generated=topography_count,
            measurement_rows=len(measurements),
        )
        path = outputs["processing_log"]
        processing_log.to_csv(path, index=False)
        output_files["processing_log"] = path
        _write_erp_sidecar(path, subject, recording_part, branch, config_path, root, input_files, warnings, settings)

        report = generate_erp_report(
            subject=subject,
            recording_part=recording_part,
            branch=branch,
            input_files=input_files,
            output_files=output_files,
            parameters=parameters,
            counts=counts,
            validation=validation,
            warnings=warnings,
            failures=failures,
            participant_status=participant_status,
        )
        report_path = write_markdown_report(outputs["report"], report)
        output_files["report"] = report_path
        _write_erp_sidecar(report_path, subject, recording_part, branch, config_path, root, input_files, warnings, settings)

        output_files["participant_erp_identity"] = _write_participant_erp_identity(
            subject=subject,
            recording_part=recording_part,
            branch=branch,
            config=config,
            settings=settings,
            metadata=metadata,
            output_files=output_files,
            destination=outputs["participant_erp_identity"],
        )

        missing = validate_gate4a_outputs(output_files)
        if missing:
            raise RuntimeError(f"Missing Gate 4A output(s): {', '.join(missing)}")
        status = "ok"
    except Exception as error:
        failures.append(str(error))
        failure_reason_code = str(getattr(error, "reason_code", "participant_erp_failed"))
        failure = pd.DataFrame(
            [
                {
                    "subject": subject,
                    "recording_part": recording_part,
                    "branch": branch,
                    "status": "failed",
                    "reason_code": failure_reason_code,
                    "failure": str(error),
                }
            ]
        )
        failure.to_csv(outputs["failure"], index=False)
        output_files["failure"] = outputs["failure"]
        _write_erp_sidecar(outputs["failure"], subject, recording_part, branch, config_path, root, input_files, warnings + failures, settings)
        status = "failed"

    metrics = {
        "duration_s": round(time.perf_counter() - start_time, 3),
        "conditions_analysed": _count_csv_rows(output_files.get("condition_counts")),
        "figures_generated": len([label for label in output_files if label in {"waveforms", "difference_waveform", "roi_waveforms", "trial_count_summary"} or label.startswith("topography_")]),
        "topographies_generated": len([label for label in output_files if label.startswith("topography_")]),
        "measurement_rows_exported": _count_csv_rows(output_files.get("recognition_measurements")),
        "difference_wave_generated": "difference_wave_fif" in output_files,
        "reporting_status": locals().get("participant_status", {}).get("status", status),
        "reporting_reason": locals().get("participant_status", {}).get("reason", "; ".join(failures) if failures else ""),
        "reporting_reason_code": locals().get(
            "participant_status", {}
        ).get(
            "reason_code",
            locals().get("failure_reason_code", "failed" if failures else ""),
        ),
    }
    return ERPResult(subject=subject, recording_part=recording_part, branch=branch, status=status, output_files=output_files, warnings=tuple(warnings), failures=tuple(failures), info_messages=tuple(info_messages), metrics=metrics)


def validate_gate4b_outputs(output_files: Mapping[str, Path]) -> list[str]:
    """Return missing required Gate 4B participant output labels."""

    missing = validate_erp_outputs(output_files, branch="cuteness")
    if "cuteness_trial_level" not in output_files or not Path(output_files["cuteness_trial_level"]).exists():
        missing.append("cuteness_trial_level")
    return missing


def run_gate4b_cuteness_for_recording_part(
    subject: str,
    recording_part: str,
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    published_input_paths: Mapping[str, Path] | None = None,
) -> ERPResult:
    """Run participant-level Gate 4B cuteness ERP analysis."""

    branch = "cuteness"
    settings = gate4b_cuteness_settings(config)
    figure_style = load_figure_style(root=root)
    outputs = erp_output_paths(config, subject, recording_part, branch, root=root)
    inputs = dict(published_input_paths or gate3c_erp_input_paths(config, subject, recording_part, root=root))
    input_files = [inputs["epochs_fif"], inputs["epoch_metadata_csv"]]
    output_files: dict[str, Path] = {}
    warnings: list[str] = []
    info_messages: list[str] = []
    failures: list[str] = []
    start_time = time.perf_counter()

    try:
        epochs, metadata = load_gate3c_epochs_and_metadata(inputs)
        epochs, metadata, _ = _validate_and_rebind_gate3c_epoch_metadata(
            epochs,
            metadata,
            participant=subject,
            session="session2",
            recording_part=recording_part,
            allow_legacy_serialization_compatibility=published_input_paths is not None,
        )
        epochs, exposure = prepare_cuteness_inputs(
            epochs,
            metadata,
            subject=subject,
            session="session2",
            recording_part=recording_part,
            config=config,
            root=root,
        )
        metadata = exposure.metadata
        input_files.extend(path for path in exposure.input_paths if path not in input_files)
        validation = validate_epochs_metadata(
            epochs,
            metadata,
            settings,
            participant=subject,
            session="session2",
            recording_part=recording_part,
        )
        validation = append_source_exposure_validation(validation, exposure)
        epochs = apply_erp_baseline(epochs, settings)
        condition_epochs = split_epochs_by_condition(
            epochs,
            metadata,
            settings,
            participant=subject,
            session="session2",
            recording_part=recording_part,
        )
        evokeds = compute_participant_erps(condition_epochs, settings)
        all_evokeds = evokeds_with_difference(evokeds, None, settings)
        counts = build_condition_counts(condition_epochs, metadata, settings, epochs)
        participant_status = classify_participant_erp_status(branch=branch, settings=settings, metadata=metadata, condition_counts=counts, evokeds=evokeds)
        missing = missing_conditions(metadata, settings)
        if missing:
            warnings.append(f"Cuteness ERP condition(s) unavailable from behavioural metadata and retained as QC flags: {', '.join(missing)}")
        if plotting_channel_exclusions(settings):
            info_messages.append(
                "Gate 4B figures exclude channel(s) with unresolved/overlapping montage positions: "
                + ", ".join(plotting_channel_exclusions(settings))
                + f". {plotting_channel_exclusion_rationale(settings)}"
            )

        parameters = build_erp_parameters(settings, epochs, evokeds)
        metadata_source = _metadata_source(metadata)
        measurements = extract_erp_measurements(
            subject=subject,
            recording_part=recording_part,
            branch=branch,
            evokeds=all_evokeds,
            condition_counts=counts,
            settings=settings,
            metadata_source=metadata_source,
            processing_version="gate4b_cuteness",
        )
        measurements = tag_categorical_cuteness_measurements(measurements)
        trial_level = build_cuteness_trial_table(
            participant=subject,
            recording_part=recording_part,
            epochs=epochs,
            metadata=metadata,
            settings=settings,
        )
        wide_measurements = build_wide_erp_measurements(measurements, subject=subject, recording_part=recording_part, settings=settings)
        measurement_dictionary = build_measurement_dictionary(measurements, wide_measurements, branch=branch, settings=settings)
        continuous_summary = build_continuous_summary(subject=subject, recording_part=recording_part, branch=branch, metadata=metadata, settings=settings)
        qc_summary = build_erp_qc_summary(
            subject=subject,
            recording_part=recording_part,
            branch=branch,
            settings=settings,
            condition_counts=counts,
            missing_conditions_list=missing,
            difference_wave_generated=False,
            metadata=metadata,
            participant_status=participant_status,
        )
        qc_summary = attach_source_exposure_summary(qc_summary, exposure)

        for label, table in (
            ("erp_parameters", parameters),
            ("condition_counts", counts),
            ("cuteness_measurements", measurements),
            ("cuteness_measurements_wide", wide_measurements),
            ("measurement_dictionary", measurement_dictionary),
            ("cuteness_qc_summary", qc_summary),
            ("cuteness_continuous_summary", continuous_summary),
            ("cuteness_trial_level", trial_level),
        ):
            path = outputs[label]
            table.to_csv(path, index=False)
            output_files[label] = path
            _write_erp_sidecar(path, subject, recording_part, branch, config_path, root, input_files, warnings, settings)

        if bool(settings.get("save_evokeds_fif", True)) and evokeds:
            save_participant_erps(evokeds, outputs["participant_erps_fif"])
            output_files["participant_erps_fif"] = outputs["participant_erps_fif"]
            _write_erp_sidecar(outputs["participant_erps_fif"], subject, recording_part, branch, config_path, root, input_files, warnings, settings)

        try:
            waveform = write_erp_waveform_figure(all_evokeds, settings, outputs["waveforms"], subject=subject, recording_part=recording_part, root=root, config_path=config_path, input_files=input_files, warnings=warnings, figure_style=figure_style)
            if waveform is not None:
                output_files["waveforms"] = waveform
            roi_waveforms = write_roi_waveform_figure(all_evokeds, settings, outputs["roi_waveforms"], subject=subject, recording_part=recording_part, root=root, config_path=config_path, input_files=input_files, warnings=warnings, figure_style=figure_style)
            if roi_waveforms is not None:
                output_files["roi_waveforms"] = roi_waveforms
            trial_counts = write_trial_count_summary_figure(counts, settings, outputs["trial_count_summary"], subject=subject, recording_part=recording_part, root=root, config_path=config_path, input_files=input_files, warnings=warnings, figure_style=figure_style)
            if trial_counts is not None:
                output_files["trial_count_summary"] = trial_counts
            output_files.update(write_topography_figures(all_evokeds, settings, outputs["topography_dir"], subject=subject, recording_part=recording_part, root=root, config_path=config_path, input_files=input_files, warnings=warnings, figure_style=figure_style))
        except Exception as error:
            if bool(dict(settings.get("figures", {}) or {}).get("fail_pipeline_on_figure_error", False)):
                raise
            warnings.append(f"Could not generate cuteness ERP figure: {error}")

        topography_count = len([label for label in output_files if label.startswith("topography_")])
        figure_count = len([label for label in output_files if label in {"waveforms", "roi_waveforms", "trial_count_summary"} or label.startswith("topography_")])
        duration_s = round(time.perf_counter() - start_time, 3)
        processing_log = build_processing_log(
            subject,
            recording_part,
            settings,
            validation,
            counts,
            duration_s=duration_s,
            figures_generated=figure_count,
            topographies_generated=topography_count,
            measurement_rows=len(measurements),
        )
        path = outputs["processing_log"]
        processing_log.to_csv(path, index=False)
        output_files["processing_log"] = path
        _write_erp_sidecar(path, subject, recording_part, branch, config_path, root, input_files, warnings, settings)

        report = generate_erp_report(subject=subject, recording_part=recording_part, branch=branch, input_files=input_files, output_files=output_files, parameters=parameters, counts=counts, validation=validation, warnings=warnings, failures=failures, participant_status=participant_status)
        report_path = write_markdown_report(outputs["report"], report)
        output_files["report"] = report_path
        _write_erp_sidecar(report_path, subject, recording_part, branch, config_path, root, input_files, warnings, settings)

        output_files["participant_erp_identity"] = _write_participant_erp_identity(
            subject=subject,
            recording_part=recording_part,
            branch=branch,
            config=config,
            settings=settings,
            metadata=metadata,
            output_files=output_files,
            destination=outputs["participant_erp_identity"],
        )

        missing_outputs = validate_gate4b_outputs(output_files)
        if missing_outputs:
            raise RuntimeError(f"Missing Gate 4B output(s): {', '.join(missing_outputs)}")
        status = "ok"
    except Exception as error:
        failures.append(str(error))
        failure_reason_code = str(getattr(error, "reason_code", "participant_erp_failed"))
        failure = pd.DataFrame(
            [
                {
                    "subject": subject,
                    "recording_part": recording_part,
                    "branch": branch,
                    "status": "failed",
                    "reason_code": failure_reason_code,
                    "failure": str(error),
                }
            ]
        )
        failure.to_csv(outputs["failure"], index=False)
        output_files["failure"] = outputs["failure"]
        _write_erp_sidecar(outputs["failure"], subject, recording_part, branch, config_path, root, input_files, warnings + failures, settings)
        status = "failed"

    metrics = {
        "duration_s": round(time.perf_counter() - start_time, 3),
        "conditions_analysed": _count_csv_rows(output_files.get("condition_counts")),
        "figures_generated": len([label for label in output_files if label in {"waveforms", "roi_waveforms", "trial_count_summary"} or label.startswith("topography_")]),
        "topographies_generated": len([label for label in output_files if label.startswith("topography_")]),
        "measurement_rows_exported": _count_csv_rows(output_files.get("cuteness_measurements")),
        "difference_wave_generated": False,
        "reporting_status": locals().get("participant_status", {}).get("status", status),
        "reporting_reason": locals().get("participant_status", {}).get("reason", "; ".join(failures) if failures else ""),
        "reporting_reason_code": locals().get(
            "participant_status", {}
        ).get(
            "reason_code",
            locals().get("failure_reason_code", "failed" if failures else ""),
        ),
    }
    return ERPResult(subject=subject, recording_part=recording_part, branch=branch, status=status, output_files=output_files, warnings=tuple(warnings), failures=tuple(failures), info_messages=tuple(info_messages), metrics=metrics)


def erp_summary_lines(title: str, results: Sequence[ERPResult]) -> list[str]:
    """Return concise console-summary lines for participant ERP runs."""

    successful = [result for result in results if result.status == "ok"]
    qc_only = [result for result in successful if str((result.metrics or {}).get("reporting_status", "")).lower() == "qc-only" or "participant_erps_fif" not in result.output_files]
    with_warnings = [result for result in results if result.warnings]
    difference_waves = [result for result in results if bool((result.metrics or {}).get("difference_wave_generated"))]
    overall = "Completed successfully" if results and all(result.status == "ok" for result in results) else "Completed with failures"
    return [
        "======================================",
        title,
        f"Participants processed: {len(results)}",
        "Successful analyses: " + (", ".join(f"{item.subject} {item.recording_part}" for item in successful) if successful else "none"),
        "QC-only participants: " + (", ".join(f"{item.subject} {item.recording_part} ({(item.metrics or {}).get('reporting_reason', 'reason unavailable')})" for item in qc_only) if qc_only else "none"),
        "Participants with warnings: " + (", ".join(f"{item.subject} {item.recording_part}" for item in with_warnings) if with_warnings else "none"),
        f"Difference waves generated: {len(difference_waves)}",
        f"Overall status: {overall}",
        "======================================",
    ]


def erp_picks(settings: Mapping[str, Any]) -> Any:
    channels = list(settings.get("channels", []) or [])
    if channels:
        return channels
    return settings.get("picks", "eeg")


def _count_csv_rows(path: str | Path | None) -> int:
    if path is None:
        return 0
    csv_path = Path(path)
    if not csv_path.exists():
        return 0
    try:
        with csv_path.open("r", encoding="utf-8") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except Exception:
        return 0


def _write_erp_sidecar(
    output_file: Path,
    subject: str,
    recording_part: str,
    branch: str,
    config_path: str | Path | None,
    root: Path | None,
    input_files: Sequence[str | Path],
    warnings: Sequence[str],
    settings: Mapping[str, Any],
) -> None:
    write_provenance_sidecar(
        output_file,
        pipeline_stage=f"gate4a_{branch}_erp",
        subject=subject,
        recording_part=recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary={
            "scope": "participant_level_recognition_erp",
            "branch": branch,
            "baseline_s": list(settings.get("baseline_s", [])),
            "conditions": configured_conditions(settings),
            "roi_definitions": configured_rois(settings),
            "measurement_time_windows": configured_time_windows(settings),
            "topography_time_windows": configured_time_windows(settings, key="topography_time_windows"),
            "minimum_trials": configured_minimum_trials(settings),
            "difference_wave": dict(settings.get("difference_wave", {}) or {}),
            "latency_window_s": list(settings.get("latency_window_s", [])),
            "group_analysis": "not_run",
            "statistics": "not_run",
            "decoding": "not_run",
        },
        warnings=warnings,
    )


def _metadata_source(metadata: pd.DataFrame) -> str:
    for column in ("metadata_source", "metadata_source_metadata"):
        if column in metadata and metadata[column].notna().any():
            values = sorted(set(metadata[column].dropna().astype(str)))
            return ";".join(values)
    return "epoch_metadata.csv"
