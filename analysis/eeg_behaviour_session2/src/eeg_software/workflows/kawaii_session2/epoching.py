from __future__ import annotations

import gc
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bad_channel_disposition import (
    bind_retained_bad_channel_marker,
    disposition_store_path,
    read_bad_channel_interpolation_disposition,
    retained_bad_channel_marker,
    validate_retained_bad_channel_analysis_safety,
)
from .ica import (
    gate3b_output_paths,
    ica_settings,
    select_ica_fit_channels,
    stage3a_filtered_raw_path,
)
from .ica_identity import (
    ica_fit_identity_path,
    resolved_ica_config_fingerprints,
    validate_ica_fit_artifact_identity,
)
from .identity import identify_source_file
from .epoch_ptp_calibration import max_eeg_ptp_uv
from .loading import RecordingPart
from .manual_review import (
    bad_channel_review_context_identity,
    bad_channel_source_identity,
    manual_review_paths,
    read_bad_channel_review_decision,
    read_ica_review_decision,
    read_review_notes,
)
from .paths import subject_part_stage_dir
from .plots import (
    gate3c_epoch_figure_paths,
    plot_epoch_counts,
    plot_epoch_rejection_summary,
    plot_erp_preview,
    plot_ica_before_after,
    plot_interpolation_summary,
    save_figure_with_stage_provenance,
)
from .provenance import write_provenance_sidecar
from .resolved_snapshot import ResolvedRunSnapshotReference, bind_authoritative_config_provenance
from .raw_qc import add_provisional_reference_channel
from .reporting import generate_gate3c_epoch_qc_markdown_report, write_markdown_report
from .study_definition import KawaiiTriggerDefinition
from .stage_identity import bind_epoch_metadata_rows, normalise_session
from .task_crop import (
    apply_task_crop,
    apply_task_segments,
    calculate_task_crop_window,
    calculate_task_segments,
    long_gap_settings,
    segment_event_mapping_table,
    task_crop_debug_table,
    task_crop_summary,
    task_segments_table,
)
from .temporary_resources import TemporaryResourceOwner


@dataclass(frozen=True)
class Gate3CResult:
    """Gate 3C outputs for one participant recording part."""

    subject: str
    recording_part: str
    status: str
    output_files: dict[str, Path]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    ica_decision_state: str | None = None
    confirmed_ica_components: tuple[int, ...] = ()
    bad_channel_decision_state: str | None = None


@dataclass(frozen=True)
class ICASuggestionEvidence:
    """Diagnostic-only Gate 3B suggestion evidence used by Gate 3C reporting."""

    components: tuple[int, ...]
    status: str
    source_path: Path | None
    sha256: str | None = None
    size_bytes: int | None = None
    generation_id: str | None = None
    observed_sha256: str | None = None
    observed_size_bytes: int | None = None

    def provenance(self) -> dict[str, Any]:
        return {
            "automatic_eog_suggestion_status": self.status,
            "automatic_eog_suggestion_source_path": (
                None if self.source_path is None else str(self.source_path)
            ),
            "automatic_eog_suggestion_sha256": self.sha256,
            "automatic_eog_suggestion_size_bytes": self.size_bytes,
            "automatic_eog_suggestion_gate3b_generation_id": self.generation_id,
            "automatic_eog_suggestion_observed_sha256": self.observed_sha256,
            "automatic_eog_suggestion_observed_size_bytes": self.observed_size_bytes,
        }


@dataclass(frozen=True)
class Gate3CBlock:
    """One experimental block processed independently by Gate 3C."""

    block_id: str
    block_index: int
    start_trial: int
    end_trial: int
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    duration_s: float
    n_retained_trials: int


def validate_ica_components(components: Sequence[Any], n_components: int) -> list[int]:
    """Validate manually confirmed ICA components before any application."""

    validated: list[int] = []
    for component in components:
        if isinstance(component, bool) or not isinstance(component, int):
            raise ValueError(f"ICA component indices must be integers; got {component!r}.")
        if component < 0:
            raise ValueError(f"ICA component indices must be non-negative; got {component}.")
        if component >= n_components:
            raise ValueError(f"ICA component index {component} is out of range for {n_components} fitted components.")
        validated.append(component)
    duplicates = sorted({component for component in validated if validated.count(component) > 1})
    if duplicates:
        raise ValueError(f"Duplicate ICA component indices are not allowed: {duplicates}.")
    return validated


def ica_required_for_epoching(config: Mapping[str, Any]) -> bool:
    """Return whether Gate 3C must stop when Gate 3B ICA is unavailable."""

    ica_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("ica", {}) or {})
    return bool(ica_cfg.get("required_for_epoching", True))


def gate3c_memory_settings(config: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Return Gate 3C memory settings with disk-backed preload defaults."""

    memory = dict(dict(config.get("preprocessing", {}) or {}).get("memory", {}) or {})
    memmap_dir = Path(memory.get("gate3c_memmap_dir") or memory.get("memmap_dir") or "workspace/kawaii/session2/tmp/memmap")
    if root is not None and not memmap_dir.is_absolute():
        memmap_dir = Path(root) / memmap_dir
    return {
        "use_memmap": bool(memory.get("gate3c_use_memmap", True)),
        "memmap_dir": memmap_dir,
        "cleanup_after_success": bool(memory.get("cleanup_gate3c_memmap_after_success", True)),
    }


def epoch_runtime_settings(config: Mapping[str, Any]) -> dict[str, bool]:
    """Return Gate 3C epoch persistence settings."""

    epochs = dict(config.get("epochs", {}) or {})
    return {
        "preload": bool(epochs.get("preload", False)),
        "save_epochs_fif": bool(epochs.get("save_epochs_fif", True)),
    }


def residual_epoch_qc_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provenance-visible, study-level residual EEG QC settings."""

    qc = dict(dict(config.get("epochs", {}) or {}).get("residual_eeg_qc", {}) or {})
    threshold = qc.get("peak_to_peak_threshold_uv")
    if threshold is None:
        raise ValueError("epochs.residual_eeg_qc.peak_to_peak_threshold_uv must be configured.")
    return {
        "enabled": bool(qc.get("enabled", True)),
        "threshold_uv": float(threshold),
        "comparison": "greater_than",
        "channels": "eeg_only",
        "metric": "maximum_within-channel_peak_to_peak_across_epoch_uv",
    }


def _gate3c_block_padding_settings(config: Mapping[str, Any]) -> dict[str, float]:
    """Return block crop padding in seconds using the existing task-crop settings."""

    task_crop = dict(dict(config.get("preprocessing", {}) or {}).get("task_crop", {}) or {})
    return {
        "padding_before_s": float(task_crop.get("padding_before_s", 30.0)),
        "padding_after_s": float(task_crop.get("padding_after_s", 30.0)),
    }


def should_apply_ica_components(components: Sequence[int]) -> bool:
    """Return whether Gate 3C should call ICA.apply."""

    return bool(list(components))


def should_apply_interpolation(config: Mapping[str, Any], confirmed_bad_channels: Sequence[str]) -> bool:
    """Return whether Gate 3C should call interpolate_bads."""

    interp_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("interpolation", {}) or {})
    return bool(interp_cfg.get("enabled_gate3c", True)) and bool(list(confirmed_bad_channels))


def build_ica_review_summary(
    *,
    n_components: int,
    suggested_components: Sequence[int] | None = None,
    confirmed_components: Sequence[int] | None = None,
    reviewer_notes: str = "",
    review_status: str = "not_reviewed",
    ica_solution_id: str | None = None,
) -> pd.DataFrame:
    """Build the required component-level ICA review/application table."""

    suggested = set(int(component) for component in (suggested_components or []))
    confirmed = set(validate_ica_components(list(confirmed_components or []), n_components))
    return pd.DataFrame(
        [
            {
                "component": component,
                "suggested_by_pipeline": component in suggested,
                "confirmed_by_reviewer": component in confirmed,
                "applied": component in confirmed,
                "review_status": review_status,
                "ica_solution_id": ica_solution_id,
                "reviewer_notes": reviewer_notes if component in confirmed else "",
            }
            for component in range(n_components)
        ]
    )


def reference_channels_for_average(
    channel_names: Sequence[str],
    channel_types: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    confirmed_bad_channels: Sequence[str] | None = None,
) -> list[str]:
    """Return EEG channels included in the Gate 3C average reference."""

    channels = dict(config.get("channels", {}) or {}).get("eeg", {}) or {}
    excluded = set(str(ch) for ch in channels.get("exclude_from_reference", []) or [])
    excluded.update(str(ch) for ch in channels.get("unused", []) or [])
    excluded.update(str(ch) for ch in (confirmed_bad_channels or []))
    return [str(ch) for ch in channel_names if channel_types.get(str(ch), "eeg") == "eeg" and str(ch) not in excluded]


def build_reference_summary(
    channel_names: Sequence[str],
    channel_types: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    confirmed_bad_channels: Sequence[str] | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Summarise exact average-reference channel inclusion/exclusion."""

    ref_channels = reference_channels_for_average(channel_names, channel_types, config, confirmed_bad_channels=confirmed_bad_channels)
    channels = dict(config.get("channels", {}) or {}).get("eeg", {}) or {}
    excluded = sorted(set(channels.get("exclude_from_reference", []) or []) | set(channels.get("unused", []) or []) | set(confirmed_bad_channels or []))
    return pd.DataFrame(
        [
            {
                "reference_type": "average",
                "reference_channels": ref_channels,
                "n_reference_channels": len(ref_channels),
                "excluded_from_reference": excluded,
                "confirmed_bad_channels_excluded": list(confirmed_bad_channels or []),
                **dict(runtime_provenance or {}),
            }
        ]
    )


def build_interpolation_table(
    channel_names: Sequence[str],
    *,
    confirmed_bad_channels: Sequence[str] | None = None,
    automatic_bad_channel_proposals: Sequence[str] | None = None,
    automatic_proposal_reasons: Mapping[str, str] | None = None,
    method: str = "mne_interpolate_bads",
    decision_state: str = "NOT_REVIEWED",
    decision_id: str | None = None,
    semantic_decision_id: str | None = None,
    audit_record_id: str | None = None,
    decision_file: str | None = None,
    retained_bad_channel_disposition: str | None = None,
    retained_bad_channel_disposition_id: str | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Describe actual interpolation/disposition outcomes for each channel."""

    confirmed = set(str(channel) for channel in (confirmed_bad_channels or []))
    automatic = set(str(channel) for channel in (automatic_bad_channel_proposals or []))
    proposal_reasons = {str(channel): str(reasons) for channel, reasons in dict(automatic_proposal_reasons or {}).items()}
    runtime = dict(runtime_provenance or {})
    interpolated_targets = set(
        str(channel) for channel in runtime.get("interpolated_targets", []) or []
    )
    retained_bad = set(
        str(channel)
        for channel in runtime.get("retained_uninterpolated_bad_channels", []) or []
    )
    invalid_interpolated = sorted(interpolated_targets - confirmed)
    invalid_retained = sorted(retained_bad - confirmed)
    if invalid_interpolated or invalid_retained or interpolated_targets & retained_bad:
        raise ValueError(
            "Interpolation runtime provenance is inconsistent with the confirmed-bad decision: "
            f"unconfirmed interpolated={invalid_interpolated}, "
            f"unconfirmed retained={invalid_retained}, "
            f"both interpolated and retained={sorted(interpolated_targets & retained_bad)}."
        )
    donors = set(str(channel) for channel in runtime.get("interpolation_donor_channels", []) or [])
    excluded_unresolved = set(
        str(channel)
        for channel in runtime.get(
            "unresolved_channels_excluded_from_interpolation_donor_geometry", []
        ) or []
    )
    rows = []
    for channel in channel_names:
        channel = str(channel)
        interpolated = channel in interpolated_targets
        retained_uninterpolated = channel in retained_bad
        rows.append(
            {
                "channel": channel,
                "manual_confirmed_bad_channel": channel in confirmed,
                "automatic_bad_channel_proposal": channel in automatic,
                "automatic_proposal_reasons": proposal_reasons.get(channel, ""),
                "interpolated": interpolated,
                "interpolation_method": method if interpolated else "",
                "bad_channel_decision_state": decision_state,
                "bad_channel_decision_id": decision_id,
                "bad_channel_semantic_decision_id": semantic_decision_id,
                "bad_channel_audit_record_id": audit_record_id or decision_id,
                "bad_channel_decision_file": decision_file,
                "interpolation_geometry_safety_status": runtime.get(
                    "interpolation_geometry_safety_status", ""
                ),
                "interpolation_donor": channel in donors,
                "excluded_from_interpolation_donor_geometry": channel in excluded_unresolved,
                "retained_bad_uninterpolated": retained_uninterpolated,
                "retained_unchanged_during_interpolation": (
                    channel in excluded_unresolved or retained_uninterpolated
                ),
                "bad_channel_interpolation_disposition": (
                    retained_bad_channel_disposition if retained_uninterpolated else ""
                ),
                "bad_channel_interpolation_disposition_id": (
                    retained_bad_channel_disposition_id if retained_uninterpolated else ""
                ),
                "proposal_confirmation_outcome": (
                    "automatic_proposed_manual_confirmed"
                    if channel in automatic and channel in confirmed
                    else "automatic_proposed_not_confirmed"
                    if channel in automatic
                    else "manual_identified_not_automatic"
                    if channel in confirmed
                    else "not_proposed_not_confirmed"
                ),
            }
        )
    return pd.DataFrame(rows)


def unresolved_eeg_coordinate_channels(raw) -> list[str]:
    """Return EEG channels without finite, non-zero sensor coordinates."""

    import numpy as np

    unresolved = []
    channel_types = raw.get_channel_types()
    for index, (channel, kind) in enumerate(zip(raw.ch_names, channel_types)):
        if kind != "eeg":
            continue
        position = np.asarray(raw.info["chs"][index]["loc"][:3], dtype=float)
        if not np.isfinite(position).all() or np.allclose(position, 0.0):
            unresolved.append(str(channel))
    return unresolved


def validate_interpolation_geometry(raw, confirmed_bad_channels: Sequence[str]) -> list[str]:
    """Validate targets and resolved donors for fail-closed EEG interpolation."""

    requested = list(dict.fromkeys(str(channel) for channel in confirmed_bad_channels))
    if not requested:
        return unresolved_eeg_coordinate_channels(raw)
    missing = [channel for channel in requested if channel not in raw.ch_names]
    if missing:
        raise ValueError(
            "Unsafe EEG interpolation blocked because confirmed bad targets are absent: "
            f"{', '.join(missing)}."
        )
    channel_types = dict(zip(raw.ch_names, raw.get_channel_types(), strict=True))
    non_eeg = [channel for channel in requested if channel_types[channel] != "eeg"]
    if non_eeg:
        raise ValueError(
            "Unsafe EEG interpolation blocked because confirmed bad targets are not EEG channels: "
            f"{', '.join(non_eeg)}."
        )
    unresolved = unresolved_eeg_coordinate_channels(raw)
    unresolved_targets = [channel for channel in requested if channel in unresolved]
    if unresolved_targets:
        raise ValueError(
            "Unsafe EEG interpolation blocked because confirmed bad targets have unresolved "
            f"sensor coordinates: {', '.join(unresolved_targets)}. Provide verified acquisition "
            "cap/channel coordinates; coordinates will not be guessed."
        )
    resolved_donors = [
        channel
        for channel, kind in channel_types.items()
        if kind == "eeg" and channel not in unresolved and channel not in requested
    ]
    if not resolved_donors:
        raise ValueError(
            "Unsafe EEG interpolation blocked because no resolved non-target EEG donor channels "
            f"are available for: {', '.join(requested)}."
        )
    return unresolved


def interpolate_confirmed_bad_channels(raw, confirmed_bad_channels: Sequence[str]) -> dict[str, Any]:
    """Interpolate confirmed targets using only EEG channels with valid geometry.

    Unresolved non-target EEG channels stay in the original recording and are
    never relabelled.  Interpolation runs on a resolved-geometry working copy;
    only the confirmed target sample arrays are copied back.
    """

    requested = list(dict.fromkeys(str(channel) for channel in confirmed_bad_channels))
    unresolved = validate_interpolation_geometry(raw, requested)
    if not requested:
        return {
            "interpolated_targets": [],
            "interpolation_donor_channels": [],
            "unresolved_channels_excluded_from_interpolation_donor_geometry": [],
            "unresolved_non_target_channels_retained_unchanged": [],
            "unresolved_non_targets_retained_unchanged": True,
            "interpolation_geometry_safety_status": "not_required",
            "interpolation_execution_strategy": "not_required",
        }
    if not getattr(raw, "preload", False):
        raise ValueError("EEG interpolation requires preloaded data for targeted sample replacement.")

    channel_types = dict(zip(raw.ch_names, raw.get_channel_types(), strict=True))
    resolved_eeg = [
        channel
        for channel in raw.ch_names
        if channel_types[channel] == "eeg" and channel not in unresolved
    ]
    donors = [channel for channel in resolved_eeg if channel not in requested]
    excluded = [channel for channel in unresolved if channel not in requested]
    working = raw.copy().pick(resolved_eeg)
    working.info["bads"] = list(requested)
    working.interpolate_bads(reset_bads=False, verbose=False)
    for channel in requested:
        target_index = raw.ch_names.index(channel)
        np.copyto(raw._data[target_index], working.get_data(picks=[channel])[0])

    return {
        "interpolated_targets": list(requested),
        "interpolation_donor_channels": donors,
        "unresolved_channels_excluded_from_interpolation_donor_geometry": excluded,
        "unresolved_non_target_channels_retained_unchanged": excluded,
        "unresolved_non_targets_retained_unchanged": True,
        "interpolation_execution_strategy": "resolved_eeg_working_copy_target_copyback",
        "interpolation_geometry_safety_status": (
            "safe_unresolved_non_targets_excluded"
            if excluded
            else "safe_complete_geometry"
        ),
    }


def apply_gate3c_signal_transforms(
    raw,
    config: Mapping[str, Any],
    *,
    ica=None,
    validated_ica: Sequence[int] = (),
    confirmed_bad_channels: Sequence[str] = (),
    retained_uninterpolated_bad_channels: Sequence[str] = (),
    retained_bad_channel_disposition_id: str | None = None,
    operation_log: list[str] | None = None,
):
    """Apply ICA, reconstruct the acquisition reference, reference, and interpolate."""

    import mne
    import numpy as np

    operations = operation_log if operation_log is not None else []
    retained = list(dict.fromkeys(map(str, retained_uninterpolated_bad_channels)))
    confirmed = list(dict.fromkeys(map(str, confirmed_bad_channels)))
    invalid_retained = [channel for channel in retained if channel not in confirmed]
    if invalid_retained:
        raise ValueError(
            "Retained-uninterpolated disposition names channels that are not human-confirmed bad: "
            f"{invalid_retained}."
        )
    if retained and not retained_bad_channel_disposition_id:
        raise ValueError("Retained-uninterpolated bad channels require an explicit disposition identity.")
    reference_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("reference", {}) or {})
    reference_channel = str(reference_cfg.get("acquisition_reference_channel", reference_cfg.get("provisional_reference_channel", "CPz")))
    reconstruct_reference = bool(reference_cfg.get("reconstruct_after_ica_gate3c", reference_cfg.get("add_provisional_cpz", False)))
    if reconstruct_reference and reference_channel in raw.ch_names:
        raise ValueError(
            f"Authoritative Gate 3C input unexpectedly contains {reference_channel} before ICA/reference reconstruction."
        )
    _mark_unused_channels_misc(raw, config)
    ica_applied = bool(ica is not None and should_apply_ica_components(validated_ica))
    if ica is not None:
        # Replace, never accumulate, state carried by a reused ICA object.
        ica.exclude = list(validated_ica)
    if ica_applied:
        operations.append("ica_application")
        raw = ica.apply(raw, exclude=list(validated_ica), verbose=False)

    eeg_count_before = sum(kind == "eeg" for kind in raw.get_channel_types())
    reference_present_before = reference_channel in raw.ch_names
    reference_reconstructed = False
    reference_zero_before_average = False
    if reconstruct_reference and not reference_present_before:
        operations.append("cpz_reconstruction")
        raw, _ = add_provisional_reference_channel(raw, config)
        if reference_channel not in raw.ch_names:
            raise RuntimeError(
                f"Failed to reconstruct acquisition reference channel {reference_channel}; Gate 3C data must be preloaded."
            )
        reference_reconstructed = True
        reference_zero_before_average = bool(np.allclose(raw.get_data(picks=[reference_channel]), 0.0))
        montage_name = str(dict(config.get("preprocessing", {}) or {}).get("montage", "standard_1005"))
        raw.set_montage(mne.channels.make_standard_montage(montage_name), on_missing="ignore")
    eeg_count_after_reconstruction = sum(kind == "eeg" for kind in raw.get_channel_types())

    _mark_unused_channels_misc(raw, config)
    stable_names = tuple(raw.ch_names)
    stable_types = tuple(raw.get_channel_types())
    raw.info["bads"] = [channel for channel in confirmed if channel in raw.ch_names]
    missing_retained = [channel for channel in retained if channel not in raw.ch_names]
    if missing_retained:
        raise ValueError(f"Retained-uninterpolated bad channels are absent: {missing_retained}.")
    unresolved_before_reference = unresolved_eeg_coordinate_channels(raw)
    resolved_retained = [channel for channel in retained if channel not in unresolved_before_reference]
    if resolved_retained:
        raise ValueError(
            "Retained-uninterpolated disposition is allowed only for unresolved bad-channel targets; "
            f"resolved channels were named: {resolved_retained}."
        )
    apply_reference = bool(reference_cfg.get("apply_average_reference_gate3c", True))
    reference_channels = reference_channels_for_average(
        raw.ch_names,
        {name: raw.get_channel_types(picks=[name])[0] for name in raw.ch_names},
        config,
        confirmed_bad_channels=confirmed,
    )
    unsafe_reference_channels = [channel for channel in retained if channel in reference_channels]
    if unsafe_reference_channels:
        raise ValueError(
            "Average-reference safety invariant failed: retained bad channels would contribute to "
            f"the reference: {unsafe_reference_channels}."
        )
    if apply_reference:
        if reference_channel not in raw.ch_names:
            raise ValueError(f"Acquisition reference channel {reference_channel} is absent immediately before average reference.")
        operations.append("average_reference")
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

    interpolation_targets = [channel for channel in confirmed if channel not in retained]
    retained_samples = raw.get_data(picks=retained).copy() if retained else None
    apply_interpolation = should_apply_interpolation(config, interpolation_targets)
    unresolved = unresolved_eeg_coordinate_channels(raw)
    interpolation_geometry_status = "not_required"
    interpolation_provenance = {
        "interpolated_targets": [],
        "interpolation_donor_channels": [],
        "unresolved_channels_excluded_from_interpolation_donor_geometry": [],
        "unresolved_non_target_channels_retained_unchanged": [],
        "unresolved_non_targets_retained_unchanged": True,
        "interpolation_execution_strategy": "not_required",
    }
    if apply_interpolation:
        interpolation_provenance = interpolate_confirmed_bad_channels(
            raw, interpolation_targets
        )
        operations.append("interpolation")
        interpolation_geometry_status = str(
            interpolation_provenance["interpolation_geometry_safety_status"]
        )
    if retained:
        if not np.array_equal(retained_samples, raw.get_data(picks=retained)):
            raise RuntimeError("Retained-uninterpolated bad-channel samples changed during interpolation.")
        if any(channel not in raw.info["bads"] for channel in retained):
            raise RuntimeError("Retained-uninterpolated channel lost its bad-channel state.")
        if tuple(raw.ch_names) != stable_names or tuple(raw.get_channel_types()) != stable_types:
            raise RuntimeError("Retained-uninterpolated disposition changed channel names/order/types.")
        bind_retained_bad_channel_marker(raw.info, retained, str(retained_bad_channel_disposition_id))
        operations.append("retain_bad_uninterpolated_due_to_unresolved_geometry")

    return raw, {
        "original_reference_channel": reference_channel,
        "original_reference_evidence": str(reference_cfg.get("acquisition_reference_evidence", "configured acquisition reference")),
        "original_reference_channel_present_before_reconstruction": reference_present_before,
        "cpz_reconstructed": reference_reconstructed,
        "cpz_reconstruction_stage": "gate3c_after_ica_application_before_average_reference" if reference_reconstructed else "not_reconstructed_configuration_disabled",
        "cpz_reconstruction_method": "mne.add_reference_channels(copy=False)",
        "cpz_zero_valued_before_average_reference": reference_zero_before_average,
        "eeg_channel_count_before_reference_reconstruction": eeg_count_before,
        "eeg_channel_count_after_reference_reconstruction": eeg_count_after_reconstruction,
        "final_eeg_reference": "average" if apply_reference else "acquisition_reference_unchanged",
        "average_reference_applied": apply_reference,
        "average_reference_projection": False,
        "average_reference_channels": reference_channels,
        "montage": str(dict(config.get("preprocessing", {}) or {}).get("montage", "standard_1005")),
        "unresolved_coordinate_channels": unresolved,
        "montage_status": "partial_with_unresolved_eeg" if unresolved else "complete",
        "interpolation_requested": apply_interpolation,
        "retained_uninterpolated_bad_channels": retained,
        "retained_bad_channel_disposition_id": retained_bad_channel_disposition_id,
        "retained_bad_channels_excluded_from_average_reference": bool(
            not set(retained) & set(reference_channels)
        ),
        "interpolation_geometry_safety_status": interpolation_geometry_status,
        **interpolation_provenance,
        "operation_order": list(operations),
        "ica_applied": ica_applied,
    }


def image_epoch_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return configured image-onset epoch settings for Gate 3C."""

    epochs = dict(config.get("epochs", {}) or {})
    canonical = dict(epochs.get("canonical", {}) or {})
    if not canonical:
        raise ValueError("epochs.canonical must define the shared Session 2 image epoch.")
    image_event_id = KawaiiTriggerDefinition.from_config(config).image_event_id
    return {
        "event": "image_onset",
        "event_codes": list(image_event_id.values()),
        "tmin": canonical.get("tmin_s", -0.2),
        "tmax": canonical.get("tmax_s", 1.2),
        "baseline": tuple(canonical.get("baseline_s", [-0.2, 0.0])),
    }


def residual_epoch_ptp_table(epochs, config: Mapping[str, Any]) -> pd.DataFrame:
    """Calculate the frozen calibration metric for every epoch entering EEG QC."""

    import mne
    settings = residual_epoch_qc_settings(config)
    retained = retained_bad_channel_marker(epochs.info)
    eeg_indices = mne.pick_types(
        epochs.info, eeg=True, eog=False, meg=False, exclude=list(retained)
    )
    if not len(eeg_indices):
        raise ValueError("Residual epoch PTP QC requires at least one EEG channel.")
    eeg_names = [str(epochs.ch_names[index]) for index in eeg_indices]
    eeg_data = epochs.get_data(picks=eeg_indices)
    finite_epochs = np.isfinite(eeg_data).all(axis=(1, 2))
    ptp_uv = np.full(len(eeg_data), np.nan, dtype=float)
    max_channels = ["" for _ in range(len(eeg_data))]
    if finite_epochs.any():
        finite_ptp, finite_channels = max_eeg_ptp_uv(eeg_data[finite_epochs], eeg_names)
        ptp_uv[finite_epochs] = finite_ptp
        for index, channel in zip(np.flatnonzero(finite_epochs), finite_channels):
            max_channels[int(index)] = channel
    rejected_nonfinite = ~finite_epochs
    rejected_ptp = finite_epochs & (ptp_uv > float(settings["threshold_uv"]))
    rejected = rejected_nonfinite | rejected_ptp
    metadata = epochs.metadata.reset_index(drop=True).copy() if epochs.metadata is not None else pd.DataFrame(index=range(len(epochs)))
    table = metadata.copy()
    table["source_epoch_index"] = range(len(table))
    table["ptp_uv"] = ptp_uv
    table["max_ptp_channel"] = max_channels
    table["threshold_uv"] = float(settings["threshold_uv"])
    table["excluded_retained_bad_channels"] = "|".join(retained)
    table["rejected_for_ptp"] = rejected_ptp
    table["rejected_for_nonfinite"] = rejected_nonfinite
    table["structural_invalid"] = False
    table["eeg_qc_status"] = [
        "rejected_nonfinite_eeg" if nonfinite else "rejected_ptp" if ptp else "retained"
        for nonfinite, ptp in zip(rejected_nonfinite, rejected_ptp)
    ]
    table["final_status"] = ["rejected" if value else "retained" for value in rejected]
    table["rejection_reason"] = [
        "nonfinite_eeg" if nonfinite else "residual_eeg_ptp_above_threshold" if ptp else ""
        for nonfinite, ptp in zip(rejected_nonfinite, rejected_ptp)
    ]
    return table


def apply_residual_epoch_ptp_qc(epochs, config: Mapping[str, Any]):
    """Drop only epochs whose calibrated residual EEG PTP exceeds the threshold."""

    settings = residual_epoch_qc_settings(config)
    table = residual_epoch_ptp_table(epochs, config)
    if not settings["enabled"]:
        table["rejected_for_ptp"] = False
        finite = ~table["rejected_for_nonfinite"].astype(bool)
        table.loc[finite, "eeg_qc_status"] = "disabled"
        table.loc[finite, "final_status"] = "retained"
        table.loc[finite, "rejection_reason"] = ""
    rejected = table[table["final_status"].eq("rejected")].sort_values("source_epoch_index", ascending=False)
    for row in rejected.itertuples(index=False):
        reason = "NONFINITE_EEG" if bool(row.rejected_for_nonfinite) else "RESIDUAL_EEG_PTP"
        epochs.drop([int(row.source_epoch_index)], reason=reason, verbose=False)
    return epochs, table


def retained_alignment_for_epoching(
    alignment: pd.DataFrame,
    recording_part: str,
    *,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Return alignment rows eligible for epoching without inventing trials."""

    if alignment.empty:
        return pd.DataFrame()
    df = alignment.copy()
    if "recording_part" in df:
        df = df[df["recording_part"].astype(str).eq(str(recording_part))].copy()
    if "retain_for_epoching" not in df:
        df["retain_for_epoching"] = False
    df = df[df["retain_for_epoching"].fillna(False).astype(bool)].copy()
    _validate_unique_keys(df, ["recording_part", "trial"], "Retained alignment")
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    event_id = dict(definition.image_event_id)
    df = df[
        df["image_onset_sample"].notna()
        & df["image_onset_code"].isin(event_id.values())
    ].copy()
    df["event_sample"] = pd.to_numeric(df["image_onset_sample"], errors="coerce").astype("Int64")
    df["event_code"] = pd.to_numeric(df["image_onset_code"], errors="coerce").astype("Int64")
    _validate_unique_keys(df, ["event_sample", "event_code"], "Retained alignment event")
    df["condition"] = df["event_code"].map(
        {int(code): str(label) for label, code in event_id.items()}
    )
    if "block" not in df:
        df["block"] = pd.NA
    return df.sort_values(["event_sample", "trial"], kind="mergesort").reset_index(drop=True)


def detect_gate3c_blocks(
    retained_alignment: pd.DataFrame,
    *,
    sfreq: float,
    raw_first_sample: int,
    raw_last_sample: int,
    config: Mapping[str, Any],
) -> list[Gate3CBlock]:
    """Detect experimental blocks from retained Stage 2 alignment rows."""

    if retained_alignment.empty:
        raise ValueError("Gate 3C block detection failed: no retained image-onset alignment rows.")
    if "block" not in retained_alignment or retained_alignment["block"].isna().all():
        raise ValueError("Gate 3C block detection failed: retained alignment is missing experimental block labels.")
    settings = _gate3c_block_padding_settings(config)
    pad_before = float(settings["padding_before_s"])
    pad_after = float(settings["padding_after_s"])
    epoch_tmax = float(image_epoch_settings(config)["tmax"])
    blocks: list[Gate3CBlock] = []
    rows = retained_alignment.copy()
    rows["event_sample"] = pd.to_numeric(rows["event_sample"], errors="coerce")
    rows["trial"] = pd.to_numeric(rows["trial"], errors="coerce")
    rows = rows[rows["event_sample"].notna() & rows["trial"].notna()].copy()
    for block_index, (block_id, block_rows) in enumerate(rows.groupby("block", dropna=True, sort=True), start=1):
        block_rows = block_rows.sort_values(["event_sample", "trial"], kind="mergesort")
        first_sample = int(block_rows["event_sample"].min())
        last_sample = int(block_rows["event_sample"].max())
        start = max(int(raw_first_sample), int(first_sample - round(pad_before * float(sfreq))))
        end = min(int(raw_last_sample), int(last_sample + round((epoch_tmax + pad_after) * float(sfreq))))
        if end <= start:
            raise ValueError(f"Gate 3C block detection failed: invalid crop for block {block_id}: {start}-{end}.")
        blocks.append(
            Gate3CBlock(
                block_id=str(block_id),
                block_index=int(block_index),
                start_trial=int(block_rows["trial"].min()),
                end_trial=int(block_rows["trial"].max()),
                start_sample=int(start),
                end_sample=int(end),
                start_s=float(start - int(raw_first_sample)) / float(sfreq),
                end_s=float(end - int(raw_first_sample)) / float(sfreq),
                duration_s=float(end - start) / float(sfreq),
                n_retained_trials=int(len(block_rows)),
            )
        )
    if not blocks:
        raise ValueError("Gate 3C block detection failed: no available experimental blocks.")
    return blocks


def block_processing_summary_table(blocks: Sequence[Gate3CBlock]) -> pd.DataFrame:
    """Return a reviewable block-processing summary before block execution."""

    return pd.DataFrame([block.__dict__ for block in blocks])


def _authoritative_gate3c_task_segments(
    recording: RecordingPart,
    alignment: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    filtered_raw,
) -> list[Any]:
    """Recreate the Gate 3A segment geometry from the source recording header."""

    if not Path(recording.vhdr_path).is_file():
        raise FileNotFoundError(
            "Gate 3C coordinate reconciliation requires the authoritative BrainVision "
            f"header: {recording.vhdr_path}"
        )
    import mne

    source_raw = mne.io.read_raw_brainvision(recording.vhdr_path, preload=False, verbose=False)
    try:
        source_sfreq = float(source_raw.info["sfreq"])
        filtered_sfreq = float(filtered_raw.info["sfreq"])
        if not np.isclose(source_sfreq, filtered_sfreq, rtol=0.0, atol=1e-9):
            raise ValueError(
                "Gate 3C coordinate reconciliation found a sampling-rate mismatch "
                f"between source ({source_sfreq}) and filtered FIF ({filtered_sfreq})."
            )
        if not np.isclose(
            source_sfreq, float(recording.sampling_rate_hz), rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "Gate 3C coordinate reconciliation found a sampling-rate mismatch "
                "between the validated recording metadata and source header."
            )
        segments = calculate_task_segments(
            alignment,
            recording.recording_part,
            sfreq=source_sfreq,
            n_times=int(source_raw.n_times),
            config=config,
            first_samp=int(source_raw.first_samp),
        )
    finally:
        source_raw.close()
    if not segments:
        raise ValueError(
            "Gate 3C coordinate reconciliation could not recreate any authoritative task segments."
        )
    expected_n_times = sum(
        int(segment.end_sample) - int(segment.start_sample) + 1
        for segment in segments
    )
    if int(filtered_raw.n_times) != expected_n_times:
        raise ValueError(
            "Gate 3C coordinate reconciliation found that the filtered FIF length does not "
            f"match the inclusive task-segment geometry ({filtered_raw.n_times} != {expected_n_times})."
        )
    if int(filtered_raw.first_samp) != int(segments[0].start_sample):
        raise ValueError(
            "Gate 3C coordinate reconciliation found that the filtered FIF first sample does "
            "not match the first authoritative task segment."
        )
    return segments


def build_event_epoch_mapping(
    events: pd.DataFrame,
    retained_alignment: pd.DataFrame,
    *,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Map image events to retained alignment rows without inventing trials."""

    if retained_alignment.empty:
        return pd.DataFrame(columns=["epoch_index", "event_sample", "event_code", "trial", "condition", "mapping_status"])
    if events.empty:
        return _event_epoch_mapping_from_retained_alignment(
            retained_alignment,
            mapping_status="from_retained_alignment_no_observed_events",
            trigger_definition=trigger_definition,
        )
    left = events.copy()
    right = retained_alignment.copy()
    for frame in (left, right):
        frame["event_sample"] = pd.to_numeric(frame["event_sample"], errors="coerce").astype("Int64")
        frame["event_code"] = pd.to_numeric(frame["event_code"], errors="coerce").astype("Int64")
    event_key = ["event_sample", "event_code"]
    _validate_unique_keys(left, event_key, "Observed image event")
    _validate_unique_keys(right, event_key, "Retained alignment event")
    mapped = left.merge(
        right,
        on=event_key,
        how="inner",
        suffixes=("", "_alignment"),
        validate="one_to_one",
    )
    if mapped.empty:
        return _event_epoch_mapping_from_retained_alignment(
            retained_alignment,
            mapping_status="from_retained_alignment_no_exact_event_match",
            trigger_definition=trigger_definition,
        )
    mapped = mapped.sort_values(["event_sample", "event_code"], kind="mergesort").reset_index(drop=True)
    mapped["epoch_index"] = range(len(mapped))
    mapped["mapping_status"] = "mapped_retained_alignment_trial"
    return mapped


def _event_epoch_mapping_from_retained_alignment(
    retained_alignment: pd.DataFrame,
    *,
    mapping_status: str,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Build epoch events from validated retained alignment rows."""

    mapping = retained_alignment.copy()
    if "event_sample" not in mapping:
        mapping["event_sample"] = pd.to_numeric(mapping["image_onset_sample"], errors="coerce").astype("Int64")
    if "event_code" not in mapping:
        mapping["event_code"] = pd.to_numeric(mapping["image_onset_code"], errors="coerce").astype("Int64")
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    mapping = mapping[
        mapping["event_sample"].notna()
        & mapping["event_code"].isin(definition.image_event_id.values())
    ].copy()
    _validate_unique_keys(mapping, ["event_sample", "event_code"], "Retained alignment event")
    mapping = mapping.sort_values(["event_sample", "event_code", "trial"], kind="mergesort").reset_index(drop=True)
    mapping["epoch_index"] = range(len(mapping))
    mapping["mapping_status"] = mapping_status
    if "mne_event_code" not in mapping:
        mapping["mne_event_code"] = pd.NA
    return mapping


def _event_epoch_mapping_from_segmented_events(
    segmented_events: pd.DataFrame,
    retained_alignment: pd.DataFrame,
    *,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Build epoch mapping with segmented raw samples and original sample provenance."""

    if segmented_events.empty:
        return _event_epoch_mapping_from_retained_alignment(
            retained_alignment,
            mapping_status="from_segmented_events_empty",
            trigger_definition=trigger_definition,
        )
    left = segmented_events.copy()
    left["event_sample_original"] = pd.to_numeric(left["event_sample_original"], errors="coerce").astype("Int64")
    left["event_sample"] = pd.to_numeric(left["event_sample_segmented"], errors="coerce").astype("Int64")
    left["event_code"] = pd.to_numeric(left["event_code"], errors="coerce").astype("Int64")
    right = retained_alignment.copy()
    right["event_sample_original"] = pd.to_numeric(right["event_sample"], errors="coerce").astype("Int64")
    right["event_code"] = pd.to_numeric(right["event_code"], errors="coerce").astype("Int64")
    event_key = ["event_sample_original", "event_code"]
    _validate_unique_keys(left, event_key, "Segmented image event")
    _validate_unique_keys(right, event_key, "Retained alignment event")
    mapped = left.merge(
        right,
        on=event_key,
        how="inner",
        suffixes=("", "_alignment"),
        validate="one_to_one",
    )
    mapped = mapped.sort_values(["event_sample", "event_code"], kind="mergesort").reset_index(drop=True)
    mapped["epoch_index"] = range(len(mapped))
    mapped["mapping_status"] = "mapped_retained_alignment_trial_segmented"
    return mapped


def reconcile_gate3c_segment_coordinates(
    retained_alignment: pd.DataFrame,
    segments: Sequence[Any],
    observed_events: pd.DataFrame,
    *,
    recording_part: str,
    sfreq: float,
    annotation_tolerance_samples: int = 1,
    trigger_definition: KawaiiTriggerDefinition | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Bind original alignment events to saved concatenated-FIF coordinates.

    Segment geometry predicts where each retained stimulus must land after
    inclusive crops are concatenated. The saved FIF annotations are the final
    authority for MNE crop/epoch coordinates; the prediction is accepted only
    when exactly one same-code annotation is within the documented one-sample
    serialization tolerance.
    """

    if retained_alignment.empty:
        raise ValueError("Gate 3C coordinate reconciliation has no retained alignment events.")
    if not segments:
        raise ValueError("Gate 3C coordinate reconciliation has no task-segment geometry.")
    if int(annotation_tolerance_samples) < 0:
        raise ValueError("Gate 3C annotation tolerance must be non-negative.")

    retained = retained_alignment.copy()
    if "event_sample" not in retained:
        retained["event_sample"] = pd.to_numeric(
            retained["image_onset_sample"], errors="coerce"
        ).astype("Int64")
    if "event_code" not in retained:
        retained["event_code"] = pd.to_numeric(
            retained["image_onset_code"], errors="coerce"
        ).astype("Int64")
    retained["trial"] = pd.to_numeric(retained["trial"], errors="coerce").astype("Int64")
    original_samples = pd.to_numeric(
        retained["event_sample"], errors="coerce"
    ).astype("Int64")
    if "event_sample_original" in retained:
        incoming_original = pd.to_numeric(
            retained["event_sample_original"], errors="coerce"
        ).astype("Int64")
        incoming_present = retained["event_sample_original"].notna()
        if (
            incoming_present
            & (incoming_original.isna() | incoming_original.ne(original_samples))
        ).any():
            raise ValueError(
                "Gate 3C coordinate reconciliation found an inconsistent original-coordinate "
                "column event_sample_original."
            )
    retained["event_sample_original"] = original_samples
    retained["event_code"] = pd.to_numeric(
        retained["event_code"], errors="coerce"
    ).astype("Int64")
    retained_key = ["trial", "event_sample_original", "event_code"]
    _validate_unique_keys(retained, retained_key, "Retained original-coordinate event")

    predicted = segment_event_mapping_table(
        retained,
        recording_part,
        list(segments),
        sfreq=sfreq,
        config=config,
    ).rename(columns={"event_sample_segmented": "event_sample_segmented_predicted"})
    _validate_unique_keys(predicted, retained_key, "Predicted concatenated-coordinate event")
    if len(predicted) != len(retained):
        raise ValueError(
            "Gate 3C coordinate reconciliation did not map every retained alignment event."
        )

    authoritative_derived_columns = (
        "segment_id",
        "event_sample_segmented_predicted",
    )
    for column in authoritative_derived_columns:
        if column not in retained:
            continue
        comparison = retained[retained_key + [column]].merge(
            predicted[retained_key + [column]],
            on=retained_key,
            how="left",
            validate="one_to_one",
            suffixes=("_incoming", "_authoritative"),
        )
        incoming_column = f"{column}_incoming"
        authoritative_column = f"{column}_authoritative"
        incoming_present = comparison[incoming_column].notna()
        incoming_values = pd.to_numeric(
            comparison[incoming_column], errors="coerce"
        ).astype("Int64")
        authoritative_values = pd.to_numeric(
            comparison[authoritative_column], errors="coerce"
        ).astype("Int64")
        inconsistent = incoming_present & (
            incoming_values.isna() | incoming_values.ne(authoritative_values)
        )
        if inconsistent.any():
            trials = comparison.loc[inconsistent, "trial"].astype(int).tolist()
            raise ValueError(
                "Gate 3C coordinate reconciliation found inconsistent authoritative "
                f"derived column {column} for trial(s) {trials}."
            )
    retained = retained.drop(columns=list(authoritative_derived_columns), errors="ignore")
    predicted_columns = set(authoritative_derived_columns)
    mapped = retained.merge(
        predicted,
        on=retained_key,
        how="left",
        validate="one_to_one",
    )
    if any(column not in mapped for column in predicted_columns) or mapped[
        "event_sample_segmented_predicted"
    ].isna().any():
        raise ValueError("Gate 3C coordinate reconciliation has an incomplete segment mapping.")

    observed = observed_events.copy()
    if observed.empty:
        raise ValueError(
            "Gate 3C coordinate reconciliation found a missing saved stimulus annotation."
        )
    for column in ("event_sample", "event_code"):
        if column not in observed:
            raise ValueError(f"Observed saved stimulus annotations are missing {column}.")
        observed[column] = pd.to_numeric(observed[column], errors="coerce").astype("Int64")
    if observed[["event_sample", "event_code"]].isna().any(axis=None):
        raise ValueError("Observed saved stimulus annotations contain missing coordinates or codes.")
    _validate_unique_keys(observed, ["event_sample", "event_code"], "Observed saved stimulus annotation")

    used_observed_indices: set[Any] = set()
    actual_samples: list[int] = []
    annotation_deltas: list[int] = []
    observed_mne_codes: list[Any] = []
    tolerance = int(annotation_tolerance_samples)
    for row in mapped.itertuples(index=False):
        predicted_sample = int(row.event_sample_segmented_predicted)
        event_code = int(row.event_code)
        candidates = observed[
            observed["event_code"].astype(int).eq(event_code)
            & observed["event_sample"].astype(int).sub(predicted_sample).abs().le(tolerance)
        ]
        if candidates.empty:
            raise ValueError(
                "Gate 3C coordinate reconciliation found a missing saved stimulus annotation "
                f"for trial {int(row.trial)}, code {event_code}, predicted sample {predicted_sample}."
            )
        if len(candidates) != 1:
            raise ValueError(
                "Gate 3C coordinate reconciliation found an ambiguous saved stimulus annotation "
                f"for trial {int(row.trial)}, code {event_code}, predicted sample {predicted_sample}."
            )
        observed_index = candidates.index[0]
        if observed_index in used_observed_indices:
            raise ValueError(
                "Gate 3C coordinate reconciliation would reuse a saved stimulus annotation; "
                "mapping is ambiguous."
            )
        used_observed_indices.add(observed_index)
        candidate = candidates.iloc[0]
        actual_sample = int(candidate["event_sample"])
        actual_samples.append(actual_sample)
        annotation_deltas.append(actual_sample - predicted_sample)
        observed_mne_codes.append(candidate.get("mne_event_code", pd.NA))

    mapped["event_sample"] = pd.array(actual_samples, dtype="Int64")
    mapped["saved_annotation_delta_samples"] = pd.array(annotation_deltas, dtype="Int64")
    mapped["mne_event_code"] = observed_mne_codes
    mapped["coordinate_frame"] = "saved_filtered_fif_annotation_samples"
    mapped["coordinate_mapping_status"] = (
        "validated_original_to_concatenated_saved_annotation"
    )
    mapped["mapping_status"] = "mapped_retained_alignment_trial_saved_annotation"
    mapped = mapped.sort_values(["event_sample", "event_code", "trial"], kind="mergesort").reset_index(drop=True)
    mapped["epoch_index"] = range(len(mapped))
    return mapped


def validate_gate3c_internal_seam_safety(
    reconciled_events: pd.DataFrame,
    segments: Sequence[Any],
    *,
    sfreq: float,
    config: Mapping[str, Any],
) -> None:
    """Fail if an accepted epoch would cross an internal concatenation seam."""

    if reconciled_events.empty:
        raise ValueError("Gate 3C seam-safety validation has no reconciled events.")
    if not segments:
        raise ValueError("Gate 3C seam-safety validation has no task segments.")
    required = {"trial", "segment_id", "event_sample"}
    missing = sorted(required - set(reconciled_events.columns))
    if missing:
        raise ValueError(
            "Gate 3C seam-safety validation is missing column(s): "
            + ", ".join(missing)
        )

    ordered_segments = list(segments)
    segment_bounds: dict[int, tuple[int, int, int]] = {}
    offset = 0
    concatenated_first = int(ordered_segments[0].start_sample)
    for index, segment in enumerate(ordered_segments):
        segment_id = int(segment.segment_id)
        if segment_id in segment_bounds:
            raise ValueError(
                f"Gate 3C seam-safety validation found duplicate segment ID {segment_id}."
            )
        length = int(segment.end_sample) - int(segment.start_sample) + 1
        if length <= 0:
            raise ValueError(
                f"Gate 3C seam-safety validation found invalid segment {segment_id} bounds."
            )
        start = concatenated_first + offset
        end = start + length - 1
        segment_bounds[segment_id] = (index, start, end)
        offset += length

    settings = image_epoch_settings(config)
    epoch_start_offset = int(round(float(settings["tmin"]) * float(sfreq)))
    epoch_end_offset = int(round(float(settings["tmax"]) * float(sfreq)))
    last_index = len(ordered_segments) - 1
    for row in reconciled_events.itertuples(index=False):
        segment_id = int(row.segment_id)
        if segment_id not in segment_bounds:
            raise ValueError(
                "Gate 3C seam-safety validation found an event with unknown segment ID "
                f"{segment_id}."
            )
        segment_index, segment_start, segment_end = segment_bounds[segment_id]
        event_sample = int(row.event_sample)
        trial = int(row.trial)
        if not segment_start <= event_sample <= segment_end:
            raise ValueError(
                f"Gate 3C trial {trial} event lies outside its concatenated segment."
            )
        requested_start = event_sample + epoch_start_offset
        requested_end = event_sample + epoch_end_offset
        if segment_index > 0 and requested_start < segment_start:
            raise ValueError(
                f"Gate 3C trial {trial} epoch would cross the preceding internal concatenation seam."
            )
        if segment_index < last_index and requested_end > segment_end:
            raise ValueError(
                f"Gate 3C trial {trial} epoch would cross the following internal concatenation seam."
            )


def should_use_segmentwise_gate3c(task_segments: Sequence[Any]) -> bool:
    """Return whether Gate 3C should process task segments independently."""

    return len(list(task_segments or [])) > 1


def event_epoch_mapping_for_task_segment(
    retained_alignment: pd.DataFrame,
    segment: Any,
    *,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Build epoch mapping for one task segment in original sample coordinates."""

    if retained_alignment.empty:
        return pd.DataFrame(columns=["epoch_index", "event_sample", "event_code", "trial", "condition", "segment_id", "event_sample_original", "mapping_status"])
    mapping = retained_alignment.copy()
    mapping["event_sample"] = pd.to_numeric(mapping["event_sample"], errors="coerce").astype("Int64")
    mapping["event_code"] = pd.to_numeric(mapping["event_code"], errors="coerce").astype("Int64")
    start = int(segment.start_sample)
    end = int(segment.end_sample)
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    mapping = mapping[
        mapping["event_sample"].notna()
        & mapping["event_sample"].between(start, end)
        & mapping["event_code"].isin(definition.image_event_id.values())
    ].copy()
    mapping = mapping.sort_values(["event_sample", "event_code", "trial"], kind="mergesort").reset_index(drop=True)
    mapping["epoch_index"] = range(len(mapping))
    mapping["segment_id"] = int(segment.segment_id)
    mapping["event_sample_original"] = mapping["event_sample"]
    mapping["mapping_status"] = "mapped_retained_alignment_trial_segmentwise"
    if "mne_event_code" not in mapping:
        mapping["mne_event_code"] = pd.NA
    return mapping


def event_epoch_mapping_for_block(
    retained_alignment: pd.DataFrame,
    block: Gate3CBlock,
    *,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Build epoch mapping for one experimental block in original sample coordinates."""

    if retained_alignment.empty:
        return pd.DataFrame(columns=["epoch_index", "event_sample", "event_code", "trial", "condition", "block", "event_sample_original", "mapping_status"])
    mapping = retained_alignment.copy()
    mapping["event_sample"] = pd.to_numeric(mapping["event_sample"], errors="coerce").astype("Int64")
    mapping["event_code"] = pd.to_numeric(mapping["event_code"], errors="coerce").astype("Int64")
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    mapping = mapping[
        mapping["block"].astype(str).eq(str(block.block_id))
        & mapping["event_sample"].notna()
        & mapping["event_code"].isin(definition.image_event_id.values())
    ].copy()
    mapping = mapping.sort_values(["event_sample", "event_code", "trial"], kind="mergesort").reset_index(drop=True)
    mapping["epoch_index"] = range(len(mapping))
    mapping["block"] = str(block.block_id)
    mapping["block_index"] = int(block.block_index)
    if "event_sample_original" not in mapping:
        mapping["event_sample_original"] = mapping["event_sample"]
    mapping["mapping_status"] = "mapped_retained_alignment_trial_blockwise"
    if "segment_id" not in mapping:
        mapping["segment_id"] = pd.NA
    if "mne_event_code" not in mapping:
        mapping["mne_event_code"] = pd.NA
    return mapping


def validate_epoch_boundaries(
    event_mapping: pd.DataFrame,
    *,
    raw,
    subject: str,
    recording_part: str,
    block: Any,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop events whose requested epoch samples are outside the cropped raw."""

    columns = [
        "subject",
        "recording_part",
        "block",
        "trial",
        "event_code",
        "event_sample",
        "event_sample_original",
        "requested_epoch_start",
        "requested_epoch_end",
        "available_start",
        "available_end",
        "reason",
    ]
    if event_mapping.empty:
        return event_mapping.copy(), pd.DataFrame(columns=columns)

    settings = image_epoch_settings(config)
    sfreq = float(raw.info["sfreq"])
    available_start = int(raw.first_samp)
    available_end = int(raw.first_samp) + int(raw.n_times) - 1
    valid_rows = []
    rejected_rows = []
    for _, row in event_mapping.iterrows():
        event_sample = _nullable_int(row.get("event_sample"))
        event_code = _nullable_int(row.get("event_code"))
        requested_start = pd.NA
        requested_end = pd.NA
        reason = ""
        if pd.isna(event_sample):
            reason = "event_sample_missing"
        else:
            requested_start = int(event_sample) + int(round(float(settings["tmin"]) * sfreq))
            requested_end = int(event_sample) + int(round(float(settings["tmax"]) * sfreq))
            if requested_start < available_start:
                reason = "epoch_start_before_available_data"
            elif requested_end > available_end:
                reason = "epoch_end_after_available_data"
            elif requested_end < requested_start:
                reason = "epoch_window_invalid"
            elif requested_end - requested_start + 1 <= 0:
                reason = "epoch_window_zero_length"

        if reason:
            rejected = row.copy()
            rejected["epoch_index"] = pd.NA
            rejected["mapping_status"] = f"rejected_{reason}"
            valid_rows.append(rejected)
            rejected_rows.append(
                {
                    "subject": row.get("subject", subject),
                    "recording_part": row.get("recording_part", recording_part),
                    "block": row.get("block", block),
                    "trial": row.get("trial", pd.NA),
                    "event_code": event_code,
                    "event_sample": event_sample,
                    "event_sample_original": row.get(
                        "event_sample_original", event_sample
                    ),
                    "requested_epoch_start": requested_start,
                    "requested_epoch_end": requested_end,
                    "available_start": available_start,
                    "available_end": available_end,
                    "reason": reason,
                }
            )
        else:
            valid_rows.append(row)

    checked = pd.DataFrame(valid_rows).reset_index(drop=True)
    valid = _epochable_event_mapping(checked).copy()
    valid["epoch_index"] = range(len(valid))
    rejected = pd.DataFrame(rejected_rows, columns=columns)
    return valid, rejected


def build_epoch_rejection_table(
    alignment: pd.DataFrame,
    event_epoch_mapping: pd.DataFrame,
    recording_part: str,
    boundary_rejections: pd.DataFrame | None = None,
    *,
    trigger_definition: KawaiiTriggerDefinition | None = None,
) -> pd.DataFrame:
    """Summarise retained/rejected trial rows and rejection reasons."""

    if alignment.empty:
        return pd.DataFrame(columns=["trial", "recording_part", "rejected", "rejection_reason"])
    df = alignment.copy()
    if "recording_part" in df:
        df = df[df["recording_part"].astype(str).eq(str(recording_part))].copy()
    if "retain_for_epoching" not in df:
        df["retain_for_epoching"] = False
    mapped_keys = set()
    if not event_epoch_mapping.empty:
        event_epoch_mapping = _epochable_event_mapping(event_epoch_mapping)
        sample_col = "event_sample_original" if "event_sample_original" in event_epoch_mapping else "event_sample"
        mapped_keys = set(zip(event_epoch_mapping[sample_col].astype("Int64"), event_epoch_mapping["event_code"].astype("Int64")))
    boundary_rejection_reasons = {}
    if boundary_rejections is not None and not boundary_rejections.empty:
        for _, rejected_event in boundary_rejections.iterrows():
            key = (
                _nullable_int(
                    rejected_event.get(
                        "event_sample_original", rejected_event.get("event_sample")
                    )
                ),
                _nullable_int(rejected_event.get("event_code")),
            )
            boundary_rejection_reasons[key] = str(rejected_event.get("reason", "epoch_boundary_outside_data") or "epoch_boundary_outside_data")
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    condition_by_code = {
        int(code): str(label) for label, code in definition.image_event_id.items()
    }
    rows = []
    for _, row in df.iterrows():
        retained = _truthy(row.get("retain_for_epoching", False))
        key = (_nullable_int(row.get("image_onset_sample")), _nullable_int(row.get("image_onset_code")))
        event_mapped = key in mapped_keys
        rejected = not retained or not event_mapped
        reason = str(row.get("exclusion_reason", "") or "")
        if retained and not event_mapped:
            reason = boundary_rejection_reasons.get(key, "event_not_found_for_retained_trial")
        rows.append(
            {
                "subject": row.get("subject"),
                "recording_part": row.get("recording_part", recording_part),
                "trial": row.get("trial"),
                "condition": condition_by_code.get(row.get("image_onset_code"), pd.NA),
                "event_sample": row.get("image_onset_sample"),
                "event_code": row.get("image_onset_code"),
                "rejected": rejected,
                "rejection_reason": reason,
                "retain_for_epoching": retained,
                "is_duplicate_trial": _truthy(row.get("is_duplicate_trial", False)),
                "duplicate_resolution": row.get("duplicate_resolution", ""),
                "eeg_trial_complete": _truthy(row.get("eeg_trial_complete", False)),
            }
        )
    return pd.DataFrame(rows)


def merge_residual_qc_with_rejection_table(
    epoch_rejection_table: pd.DataFrame,
    residual_qc: pd.DataFrame,
) -> pd.DataFrame:
    """Merge EEG QC by stable trial identity and validate both coordinate frames.

    ``epoch_rejection_table.event_sample`` is the original-source coordinate,
    whereas ``residual_qc.event_sample`` is the coordinate used to create the
    saved epoch.  The latter may differ after task-segment concatenation.  The
    merge therefore uses the coordinate-neutral participant/part/trial identity
    only after proving exact row coverage and agreement of event, condition, and
    original-to-saved coordinate provenance.
    """

    table = epoch_rejection_table.copy()
    identity_columns = ["subject", "recording_part", "trial"]
    assertion_columns = ["event_code", "condition", "event_sample"]
    _validate_unique_keys(table, identity_columns, "Epoch rejection table")
    missing_table_columns = [
        column for column in assertion_columns if column not in table.columns
    ]
    if missing_table_columns:
        raise ValueError(
            "Epoch rejection table is missing QC-lineage column(s): "
            + ", ".join(missing_table_columns)
            + "."
        )
    if table[assertion_columns].isna().any(axis=None):
        raise ValueError("Epoch rejection table contains missing QC-lineage values.")

    table["structural_invalid"] = table["rejected"].astype(bool)
    table["event_sample_original"] = pd.to_numeric(
        table["event_sample"], errors="coerce"
    ).astype("Int64")
    if table["event_sample_original"].isna().any():
        raise ValueError("Epoch rejection table contains an invalid original event sample.")
    for column, default in (
        ("ptp_uv", pd.NA),
        ("max_ptp_channel", ""),
        ("threshold_uv", pd.NA),
        ("rejected_for_ptp", False),
        ("rejected_for_nonfinite", False),
        ("eeg_qc_status", "not_evaluated_structural_invalid"),
    ):
        table[column] = default

    expected = table.loc[~table["structural_invalid"], identity_columns].copy()
    if residual_qc.empty:
        if not expected.empty:
            raise ValueError(
                "Residual EEG QC is missing rows for structurally valid epoch candidates."
            )
    else:
        residual_qc = residual_qc.copy()
        required_qc_columns = [
            *identity_columns,
            "event_code",
            "condition",
            "event_sample",
            "ptp_uv",
            "max_ptp_channel",
            "threshold_uv",
            "rejected_for_ptp",
            "rejected_for_nonfinite",
            "eeg_qc_status",
        ]
        missing_qc_columns = [
            column for column in required_qc_columns if column not in residual_qc.columns
        ]
        if missing_qc_columns:
            raise ValueError(
                "Residual EEG QC is missing QC-lineage column(s): "
                + ", ".join(missing_qc_columns)
                + "."
            )
        _validate_unique_keys(residual_qc, identity_columns, "Residual EEG QC")

        coverage = expected.merge(
            residual_qc[identity_columns],
            on=identity_columns,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        missing_rows = coverage.loc[coverage["_merge"].eq("left_only"), identity_columns]
        orphan_rows = coverage.loc[coverage["_merge"].eq("right_only"), identity_columns]
        if not missing_rows.empty:
            raise ValueError(
                "Residual EEG QC is missing structurally valid trial row(s): "
                + _format_identity_rows(missing_rows, identity_columns)
                + "."
            )
        if not orphan_rows.empty:
            raise ValueError(
                "Residual EEG QC contains orphaned trial row(s): "
                + _format_identity_rows(orphan_rows, identity_columns)
                + "."
            )

        qc = residual_qc.copy()
        qc["event_sample_corrected"] = pd.to_numeric(
            qc["event_sample"], errors="coerce"
        ).astype("Int64")
        if qc["event_sample_corrected"].isna().any():
            raise ValueError("Residual EEG QC contains invalid corrected event samples.")
        if "event_sample_original" in qc.columns:
            qc["event_sample_original_qc"] = pd.to_numeric(
                qc["event_sample_original"], errors="coerce"
            ).astype("Int64")
            if qc["event_sample_original_qc"].isna().any():
                raise ValueError("Residual EEG QC contains invalid original event samples.")
        else:
            # Legacy/unsegmented epochs may use one coordinate frame, but a
            # shifted coordinate without an explicit original binding is unsafe.
            qc["event_sample_original_qc"] = qc["event_sample_corrected"]

        _validate_unique_keys(
            qc,
            ["event_sample_original_qc", "event_code"],
            "Residual EEG QC original-coordinate event",
        )
        _validate_unique_keys(
            qc,
            ["event_sample_corrected", "event_code"],
            "Residual EEG QC corrected-coordinate event",
        )

        qc["rejected_for_ptp"] = _strict_qc_boolean_series(
            qc["rejected_for_ptp"], "rejected_for_ptp"
        )
        qc["rejected_for_nonfinite"] = _strict_qc_boolean_series(
            qc["rejected_for_nonfinite"], "rejected_for_nonfinite"
        )
        overlapping = qc["rejected_for_ptp"] & qc["rejected_for_nonfinite"]
        if overlapping.any():
            raise ValueError(
                "Residual EEG QC contains overlapping PTP and non-finite rejection categories."
            )
        expected_status = pd.Series("retained", index=qc.index, dtype="object")
        expected_status.loc[qc["rejected_for_ptp"]] = "rejected_ptp"
        expected_status.loc[qc["rejected_for_nonfinite"]] = "rejected_nonfinite_eeg"
        actual_status = qc["eeg_qc_status"].fillna("").astype(str)
        status_ok = actual_status.eq(expected_status)
        status_ok |= expected_status.eq("retained") & actual_status.eq("disabled")
        if not status_ok.all():
            raise ValueError("Residual EEG QC contains an inconsistent EEG QC status.")
        expected_final_status = pd.Series("retained", index=qc.index, dtype="object")
        expected_final_status.loc[
            qc["rejected_for_ptp"] | qc["rejected_for_nonfinite"]
        ] = "rejected"
        if "final_status" in qc.columns and not qc["final_status"].fillna("").astype(
            str
        ).eq(expected_final_status).all():
            raise ValueError("Residual EEG QC contains an inconsistent final status.")
        if "rejection_reason" in qc.columns:
            expected_reason = pd.Series("", index=qc.index, dtype="object")
            expected_reason.loc[qc["rejected_for_ptp"]] = (
                "residual_eeg_ptp_above_threshold"
            )
            expected_reason.loc[qc["rejected_for_nonfinite"]] = "nonfinite_eeg"
            if not qc["rejection_reason"].fillna("").astype(str).eq(
                expected_reason
            ).all():
                raise ValueError(
                    "Residual EEG QC contains an inconsistent rejection reason."
                )
        if "structural_invalid" in qc.columns and _strict_qc_boolean_series(
            qc["structural_invalid"], "structural_invalid"
        ).any():
            raise ValueError(
                "Residual EEG QC contains a structurally invalid trial row."
            )
        ptp_values = pd.to_numeric(qc["ptp_uv"], errors="coerce")
        thresholds = pd.to_numeric(qc["threshold_uv"], errors="coerce")
        invalid_ptp_rejection = qc["rejected_for_ptp"] & (
            ptp_values.isna() | thresholds.isna() | ptp_values.le(thresholds)
        )
        if invalid_ptp_rejection.any():
            raise ValueError("Residual EEG QC contains an invalid PTP rejection decision.")
        # A retained epoch must not exceed an enabled PTP threshold.  The
        # explicit "disabled" state is the documented exception: amplitude
        # may be measured while PTP rejection is intentionally turned off.
        retained_above_threshold = (
            actual_status.eq("retained")
            & ptp_values.notna()
            & thresholds.notna()
            & ptp_values.gt(thresholds)
        )
        if retained_above_threshold.any():
            raise ValueError(
                "Residual EEG QC retained an epoch above its enabled PTP threshold."
            )

        has_predicted = "event_sample_segmented_predicted" in qc.columns
        has_delta = "saved_annotation_delta_samples" in qc.columns
        shifted_samples = qc["event_sample_corrected"].ne(
            qc["event_sample_original_qc"]
        )
        if (has_predicted != has_delta) or (
            shifted_samples.any() and not (has_predicted and has_delta)
        ):
            raise ValueError(
                "Residual EEG QC shifted coordinate mapping requires complete "
                "predicted-coordinate and saved-annotation-delta evidence."
            )

        if has_predicted:
            predicted = pd.to_numeric(
                qc["event_sample_segmented_predicted"], errors="coerce"
            ).astype("Int64")
            if predicted.isna().any():
                raise ValueError(
                    "Residual EEG QC contains an incomplete predicted coordinate mapping."
                )
            if "saved_annotation_delta_samples" not in qc.columns:
                raise ValueError(
                    "Residual EEG QC predicted coordinates lack saved-annotation deltas."
                )
            deltas = pd.to_numeric(
                qc["saved_annotation_delta_samples"], errors="coerce"
            ).astype("Int64")
            if deltas.isna().any() or not qc["event_sample_corrected"].eq(
                predicted + deltas
            ).all():
                raise ValueError(
                    "Residual EEG QC contains an inconsistent predicted-to-saved coordinate mapping."
                )

        provenance_columns = [
            column
            for column in (
                "segment_id",
                "event_sample_segmented_predicted",
                "saved_annotation_delta_samples",
                "coordinate_frame",
                "coordinate_mapping_status",
                "mapping_status",
                "source_epoch_index",
            )
            if column in qc.columns
        ]
        merge_columns = [
            *identity_columns,
            "event_code",
            "condition",
            "event_sample_original_qc",
            "event_sample_corrected",
            *provenance_columns,
            "ptp_uv",
            "max_ptp_channel",
            "threshold_uv",
            "rejected_for_ptp",
            "rejected_for_nonfinite",
            "eeg_qc_status",
        ]
        table = table.merge(
            qc[merge_columns],
            on=identity_columns,
            how="left",
            suffixes=("", "_qc"),
            validate="one_to_one",
        )
        valid = ~table["structural_invalid"]
        event_code_matches = pd.to_numeric(
            table.loc[valid, "event_code_qc"], errors="coerce"
        ).astype("Int64").eq(
            pd.to_numeric(table.loc[valid, "event_code"], errors="coerce").astype("Int64")
        )
        condition_matches = table.loc[valid, "condition_qc"].astype(str).eq(
            table.loc[valid, "condition"].astype(str)
        )
        original_matches = table.loc[valid, "event_sample_original_qc"].eq(
            table.loc[valid, "event_sample_original"]
        )
        if not event_code_matches.all():
            raise ValueError("Residual EEG QC event-code identity does not match trial authority.")
        if not condition_matches.all():
            raise ValueError("Residual EEG QC condition identity does not match trial authority.")
        if not original_matches.all():
            raise ValueError(
                "Residual EEG QC original coordinate does not match trial authority."
            )
        table = table.drop(
            columns=["event_code_qc", "condition_qc", "event_sample_original_qc"]
        )
        for column in (
            "ptp_uv",
            "max_ptp_channel",
            "threshold_uv",
            "rejected_for_ptp",
            "rejected_for_nonfinite",
            "eeg_qc_status",
        ):
            incoming = f"{column}_qc"
            if incoming in table:
                table[column] = table[incoming].combine_first(table[column])
                table = table.drop(columns=[incoming])

    table["rejected_for_ptp"] = table["rejected_for_ptp"].map(lambda value: bool(value) if pd.notna(value) else False)
    table["rejected_for_nonfinite"] = table["rejected_for_nonfinite"].map(lambda value: bool(value) if pd.notna(value) else False)
    nonfinite_mask = ~table["structural_invalid"] & table["rejected_for_nonfinite"]
    table.loc[nonfinite_mask, "rejected"] = True
    table.loc[nonfinite_mask, "rejection_reason"] = "nonfinite_eeg"
    ptp_mask = ~table["structural_invalid"] & table["rejected_for_ptp"]
    table.loc[ptp_mask, "rejected"] = True
    table.loc[ptp_mask, "rejection_reason"] = "residual_eeg_ptp_above_threshold"
    table["final_status"] = table["rejected"].astype(bool).map({True: "rejected", False: "retained"})
    return table


def participant_retention_summary(epoch_rejection_table: pd.DataFrame) -> pd.DataFrame:
    """Summarise structural validity and residual EEG retention for one run."""

    if epoch_rejection_table.empty:
        return pd.DataFrame()
    structural_invalid = epoch_rejection_table["structural_invalid"].astype(bool)
    ptp_rejected = epoch_rejection_table["rejected_for_ptp"].astype(bool)
    nonfinite_rejected = epoch_rejection_table.get(
        "rejected_for_nonfinite", pd.Series(False, index=epoch_rejection_table.index)
    ).astype(bool)
    retained = ~epoch_rejection_table["rejected"].astype(bool)
    thresholds = pd.to_numeric(epoch_rejection_table["threshold_uv"], errors="coerce").dropna().unique()
    reason_counts = epoch_rejection_table.loc[epoch_rejection_table["rejected"].astype(bool), "rejection_reason"].fillna("unknown").value_counts()
    return pd.DataFrame(
        [
            {
                "participant": epoch_rejection_table["subject"].dropna().astype(str).iloc[0] if epoch_rejection_table["subject"].notna().any() else "",
                "recording_part": epoch_rejection_table["recording_part"].dropna().astype(str).iloc[0] if epoch_rejection_table["recording_part"].notna().any() else "",
                "candidate_stimulus_trials": int(len(epoch_rejection_table)),
                "structurally_valid_trials": int((~structural_invalid).sum()),
                "epochs_entering_eeg_qc": int((~structural_invalid).sum()),
                "ptp_rejected_epochs": int(ptp_rejected.sum()),
                "nonfinite_eeg_rejected_epochs": int(nonfinite_rejected.sum()),
                "retained_epochs": int(retained.sum()),
                "retention_percent": float(100.0 * retained.sum() / (~structural_invalid).sum()) if int((~structural_invalid).sum()) else 0.0,
                "final_retention_percent_of_candidate_trials": float(100.0 * retained.sum() / len(epoch_rejection_table)),
                "threshold_uv": float(thresholds[0]) if len(thresholds) else pd.NA,
                "rejection_reason_counts": ";".join(f"{reason}={count}" for reason, count in reason_counts.items()),
            }
        ]
    )


def condition_count_table(epoch_rejection_table: pd.DataFrame) -> pd.DataFrame:
    """Count retained and rejected epochs/trials by old/new condition."""

    if epoch_rejection_table.empty:
        return pd.DataFrame(columns=["condition", "retained_epochs", "rejected_epochs"])
    rows = []
    for condition, df in epoch_rejection_table.groupby("condition", dropna=False):
        structural = df.get("structural_invalid", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        ptp = df.get("rejected_for_ptp", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        nonfinite = df.get("rejected_for_nonfinite", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        entering_qc = int((~structural).sum())
        retained = int((~df["rejected"].astype(bool)).sum())
        rows.append(
            {
                "condition": condition,
                "candidate_trials": int(len(df)),
                "structurally_invalid_trials": int(structural.sum()),
                "epochs_entering_eeg_qc": entering_qc,
                "ptp_rejected_epochs": int(ptp.sum()),
                "nonfinite_eeg_rejected_epochs": int(nonfinite.sum()),
                "retained_epochs": retained,
                "rejected_epochs": int(df["rejected"].astype(bool).sum()),
                "retention_percent": float(100.0 * retained / entering_qc) if entering_qc else 0.0,
            }
        )
    return pd.DataFrame(rows)


def duplicate_incomplete_summary(epoch_rejection_table: pd.DataFrame) -> pd.DataFrame:
    """Summarise duplicate and incomplete trial handling for epoch QC."""

    if epoch_rejection_table.empty:
        return pd.DataFrame([{"n_duplicate_rows": 0, "n_incomplete_rows": 0, "n_unresolved_duplicate_rows": 0}])
    return pd.DataFrame(
        [
            {
                "n_duplicate_rows": int(epoch_rejection_table["is_duplicate_trial"].fillna(False).astype(bool).sum()),
                "n_incomplete_rows": int((~epoch_rejection_table["eeg_trial_complete"].fillna(False).astype(bool)).sum()),
                "n_unresolved_duplicate_rows": int(epoch_rejection_table["rejection_reason"].fillna("").eq("unresolved_duplicate_trial").sum()),
            }
        ]
    )


def gate3c_output_paths(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> dict[str, Path]:
    """Return standard Gate 3C output paths."""

    clean_dir = subject_part_stage_dir(config, subject, recording_part, "preprocessing", root=root, create=True)
    epoch_dir = subject_part_stage_dir(config, subject, recording_part, "epochs", root=root, create=True)
    qc_dir = epoch_dir
    fig_dir = epoch_dir / "figures"
    for directory in (clean_dir, epoch_dir, qc_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "clean_dir": clean_dir,
        "epoch_dir": epoch_dir,
        "qc_dir": qc_dir,
        "fig_dir": fig_dir,
        "report_dir": epoch_dir,
        "ica_cleaned_raw_fif": clean_dir / "ica_cleaned_raw.fif",
        "epochs_fif": epoch_dir / "all_image_epochs-epo.fif",
        "legacy_epochs_fif": epoch_dir / "image_onset_epochs-epo.fif",
        "epoch_metadata_csv": qc_dir / "epoch_metadata.csv",
        "event_epoch_mapping": qc_dir / "event_epoch_mapping.csv",
        "epoch_rejection_table": qc_dir / "rejection_table.csv",
        "residual_epoch_qc": qc_dir / "residual_epoch_qc.csv",
        "participant_retention_summary": qc_dir / "participant_retention_summary.csv",
        "epoch_boundary_rejections": qc_dir / "epoch_boundary_rejections.csv",
        "condition_counts": qc_dir / "condition_counts.csv",
        "duplicate_incomplete_summary": qc_dir / "duplicate_incomplete_summary.csv",
        "ica_review_summary": qc_dir / "ica_review_summary.csv",
        "reference_summary": qc_dir / "reference_summary.csv",
        "interpolation_summary": qc_dir / "interpolation_summary.csv",
        "processing_log": qc_dir / "processing_log.csv",
        "task_crop_summary": epoch_dir / "task_crop_summary.csv",
        "task_crop_debug": epoch_dir / "task_crop_debug.csv",
        "task_segments": epoch_dir / "task_segments.csv",
        "segment_event_mapping": epoch_dir / "segment_event_mapping.csv",
        "segment_processing_summary": epoch_dir / "segment_processing_summary.csv",
        "block_processing_summary": epoch_dir / "block_processing_summary.csv",
        "report": epoch_dir / "report.md",
    }


def gate3c_block_epochs_path(outputs: Mapping[str, Path], block: Gate3CBlock) -> Path:
    """Return the per-block epochs FIF path requested for Gate 3C."""

    return Path(outputs["epoch_dir"]) / f"block{int(block.block_index):02d}-epo.fif"


def stage3b_ica_solution_path(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> Path:
    """Return the expected Gate 3B ICA solution path."""

    return gate3b_output_paths(config, subject, recording_part, root=root)["ica_solution"]


def run_gate3c_for_recording_part(
    subject: str,
    recording: RecordingPart,
    config: Mapping[str, Any],
    *,
    alignment: pd.DataFrame,
    filtered_raw_path: str | Path | None = None,
    gate3a_bad_channel_decisions_path: str | Path | None = None,
    ica_solution_path: str | Path | None = None,
    ica_fit_identity_path_override: str | Path | None = None,
    gate3b_eog_suggestions_path: str | Path | None = None,
    gate3b_eog_suggestions_sha256: str | None = None,
    gate3b_eog_suggestions_size_bytes: int | None = None,
    gate3b_generation_id: str | None = None,
    ica_bypass_reason: str = "",
    allow_incomplete_ica_review_for_validation: bool = False,
    allow_incomplete_bad_channel_review_for_validation: bool = False,
    allow_reviewed_ica_context_reuse: bool = False,
    replay_root_provenance: Mapping[str, str] | None = None,
    root: Path | None = None,
    config_path: str | Path | None = None,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
    execution_run_id: str | None = None,
) -> Gate3CResult:
    """Apply reviewed ICA, reference/interpolate, and create image-onset epochs."""

    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    outputs = gate3c_output_paths(config, subject, recording.recording_part, root=root)
    input_raw = Path(filtered_raw_path) if filtered_raw_path else stage3a_filtered_raw_path(config, subject, recording.recording_part, root=root)
    input_ica = None if ica_bypass_reason else (Path(ica_solution_path) if ica_solution_path else stage3b_ica_solution_path(config, subject, recording.recording_part, root=root))
    input_ica_identity = Path(ica_fit_identity_path_override) if ica_fit_identity_path_override else (ica_fit_identity_path(input_ica) if input_ica is not None else None)
    input_files = [input_raw, input_ica] if input_ica is not None and input_ica.exists() else [input_raw]
    warnings: list[str] = []
    failures: list[str] = []
    output_files: dict[str, Path] = {}
    memory_settings = gate3c_memory_settings(config, root=root)
    temporary_owner: TemporaryResourceOwner | None = None
    stage_context = "setup"
    replay_root_fields = (
        {"replay_root_separation": dict(replay_root_provenance)}
        if replay_root_provenance is not None
        else {}
    )
    if replay_root_provenance is not None and gate3a_bad_channel_decisions_path is None:
        raise ValueError(
            "Isolated replay requires the immutable Gate 3A bad-channel decision artifact."
        )

    try:
        temporary_owner = TemporaryResourceOwner.create(
            memory_settings["memmap_dir"],
            stage="gate3c",
            participant=subject,
            recording_part=recording.recording_part,
            run_id=execution_run_id,
        )
        if not input_raw.exists():
            raise FileNotFoundError(f"Stage 3A filtered raw FIF not found: {input_raw}")
        ica_required = ica_required_for_epoching(config)
        if input_ica is not None and not input_ica.exists() and ica_required:
            raise FileNotFoundError(f"Gate 3B ICA solution not found: {input_ica}")

        import mne

        raw_before = mne.io.read_raw_fif(input_raw, preload=False, verbose=False)
        raw = mne.io.read_raw_fif(input_raw, preload=False, verbose=False)
        crop_window = None
        trigger_definition = KawaiiTriggerDefinition.from_config(config)
        retained_alignment = retained_alignment_for_epoching(
            alignment,
            recording.recording_part,
            trigger_definition=trigger_definition,
        )
        stage_context = "segment_coordinate_reconciliation"
        task_segments = _authoritative_gate3c_task_segments(
            recording,
            alignment,
            config,
            filtered_raw=raw,
        )
        observed_events = _image_events_from_raw(
            raw, trigger_definition=trigger_definition
        )
        retained_alignment = reconcile_gate3c_segment_coordinates(
            retained_alignment,
            task_segments,
            observed_events,
            recording_part=recording.recording_part,
            sfreq=float(raw.info["sfreq"]),
            trigger_definition=trigger_definition,
            config=config,
        )
        validate_gate3c_internal_seam_safety(
            retained_alignment,
            task_segments,
            sfreq=float(raw.info["sfreq"]),
            config=config,
        )
        segment_table = task_segments_table(task_segments)
        segmented_event_mapping = retained_alignment[
            [
                "trial",
                "event_code",
                "event_sample_original",
                "segment_id",
                "event_sample_segmented_predicted",
                "event_sample",
                "saved_annotation_delta_samples",
                "coordinate_frame",
                "coordinate_mapping_status",
            ]
        ].copy()
        segment_table.to_csv(outputs["task_segments"], index=False)
        segmented_event_mapping.to_csv(outputs["segment_event_mapping"], index=False)
        output_files["task_segments"] = outputs["task_segments"]
        output_files["segment_event_mapping"] = outputs["segment_event_mapping"]
        coordinate_provenance = {
            "coordinate_frame": "saved_filtered_fif_annotation_samples",
            "coordinate_source": "authoritative_source_geometry_plus_saved_fif_annotations",
            "annotation_tolerance_samples": 1,
            "original_coordinate_column": "event_sample_original",
            "predicted_concatenated_coordinate_column": "event_sample_segmented_predicted",
            "epoch_coordinate_column": "event_sample",
            "inclusive_segment_boundaries": True,
            "internal_concatenation_seam_safety": "validated_fail_closed",
            **replay_root_fields,
        }
        _write_gate3c_sidecar(
            outputs["task_segments"],
            subject,
            recording,
            config_path,
            root,
            input_files,
            warnings,
            coordinate_provenance,
        )
        _write_gate3c_sidecar(
            outputs["segment_event_mapping"],
            subject,
            recording,
            config_path,
            root,
            input_files,
            warnings,
            coordinate_provenance,
        )
        stage_context = "block_detection"
        gate3c_blocks = detect_gate3c_blocks(
            retained_alignment,
            sfreq=float(raw.info["sfreq"]),
            raw_first_sample=int(raw.first_samp),
            raw_last_sample=int(raw.first_samp) + int(raw.n_times) - 1,
            config=config,
        )
        block_detection = block_processing_summary_table(gate3c_blocks)
        block_detection.to_csv(outputs["block_processing_summary"], index=False)
        output_files["block_processing_summary"] = outputs["block_processing_summary"]
        _write_gate3c_sidecar(outputs["block_processing_summary"], subject, recording, config_path, root, input_files, warnings, {"preprocessing_mode": "blockwise_epoching", "blockwise_gate3c": True, "block_detection": "reconciled_retained_alignment_block_labels", **coordinate_provenance})
        blockwise_gate3c = True
        segmentwise_gate3c = False
        warnings.append("Gate 3C using block-wise preprocessing; Raw objects are cropped and processed per experimental block.")
        stage_context = "setup"
        suggestion_evidence = _read_suggested_ica_components(
            config,
            subject,
            recording.recording_part,
            root=root,
            historical_artifact_path=gate3b_eog_suggestions_path,
            historical_sha256=gate3b_eog_suggestions_sha256,
            historical_size_bytes=gate3b_eog_suggestions_size_bytes,
            historical_generation_id=gate3b_generation_id,
            isolated_replay=replay_root_provenance is not None,
        )
        suggested = list(suggestion_evidence.components)
        if suggestion_evidence.source_path is not None:
            input_files.append(suggestion_evidence.source_path)
        if suggestion_evidence.status != "AVAILABLE_VERIFIED":
            warnings.append(
                "Gate 3B automatic EOG suggestion evidence is "
                f"{suggestion_evidence.status}; manual ICA exclusions remain authoritative."
            )
        notes = read_review_notes(config, subject, recording.recording_part, root=root)
        reviewed_eeg_channels = [
            name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"
        ]
        bad_source_id = bad_channel_source_identity(recording)
        bad_context_id = bad_channel_review_context_identity(
            recording, config, reviewed_eeg_channels, source_recording_id=bad_source_id
        )
        bad_review_decision = read_bad_channel_review_decision(
            config,
            subject,
            recording.recording_part,
            root=root,
            expected_source_recording_id=bad_source_id,
            expected_review_context_id=bad_context_id,
            expected_reviewed_channels=reviewed_eeg_channels,
            require_complete=not allow_incomplete_bad_channel_review_for_validation,
        )
        if not bad_review_decision.complete and allow_incomplete_bad_channel_review_for_validation:
            warnings.append(
                "Gate 3C proceeding with zero bad-channel interpolation under explicit validation-only "
                "incomplete-review bypass."
            )
        confirmed_bad = list(bad_review_decision.confirmed_bad_channels) if bad_review_decision.complete else []
        resolved_config_fingerprint, scientific_fingerprint = resolved_ica_config_fingerprints(
            config, resolved_snapshot
        )
        bad_disposition = read_bad_channel_interpolation_disposition(
            config,
            subject,
            recording.recording_part,
            root=root,
            expected_bad_channel_decision_id=bad_review_decision.semantic_decision_id,
            expected_bad_channel_review_context_id=bad_context_id,
            expected_scientific_fingerprint=scientific_fingerprint,
            expected_confirmed_bad_channels=confirmed_bad,
        )
        retained_uninterpolated_bad = list(bad_disposition.channels) if bad_disposition.approved else []
        validate_retained_bad_channel_analysis_safety(config, retained_uninterpolated_bad)
        disposition_path = disposition_store_path(config, root=root)
        if bad_disposition.approved:
            input_files.append(disposition_path)
        if input_ica is not None and input_ica.exists():
            ica = mne.preprocessing.read_ica(input_ica)
            n_components = int(getattr(ica, "n_components_", 0) or 0)
            fit_settings = ica_settings(config)
            raw_channel_types = raw.get_channel_types()
            channel_type_map = dict(zip(raw.ch_names, raw_channel_types, strict=True))
            expected_fit_channels = select_ica_fit_channels(
                raw.ch_names,
                channel_type_map,
                config,
                bad_channels=raw.info.get("bads", []) or [],
            )
            fit_identity = validate_ica_fit_artifact_identity(
                input_ica_identity,
                ica=ica,
                solution_path=input_ica,
                expected_participant=subject,
                expected_recording_part=recording.recording_part,
                expected_input_eeg=identify_source_file(
                    input_raw,
                    "ica_filtered_raw_input",
                    logical_path="filtered_raw.fif",
                ),
                expected_resolved_config_fingerprint=(
                    None if allow_reviewed_ica_context_reuse else resolved_config_fingerprint
                ),
                expected_scientific_fingerprint=scientific_fingerprint,
                expected_input_channel_names=raw.ch_names,
                expected_input_channel_types=raw_channel_types,
                expected_bad_channel_names=raw.info.get("bads", []) or [],
                expected_fit_channel_names=expected_fit_channels,
                expected_filter_parameters={
                    "operation": "mne.io.BaseRaw.filter",
                    "requested_highpass_hz": fit_settings["highpass_hz"],
                    "requested_lowpass_hz": fit_settings["lowpass_hz"],
                    "picks": expected_fit_channels,
                },
                expected_algorithm_parameters={
                    "method": fit_settings["method"],
                    "n_components_strategy": fit_settings["n_components"],
                    "decimation": fit_settings["decim"],
                    "random_seed": fit_settings["random_state"],
                    "fit_params": fit_settings["fit_params"],
                    "reject_by_annotation": True,
                },
                require_authoritative=True,
            )
            solution_id = fit_identity.solution_fingerprint
            input_files.append(input_ica_identity)
            review_decision = read_ica_review_decision(
                config,
                subject,
                recording.recording_part,
                root=root,
                expected_solution_id=solution_id,
                expected_component_count=n_components,
                expected_bad_channel_decision_id=bad_review_decision.semantic_decision_id,
                require_complete=not allow_incomplete_ica_review_for_validation,
            )
            if not review_decision.complete and allow_incomplete_ica_review_for_validation:
                warnings.append(
                    "Gate 3C proceeding with zero ICA exclusions under explicit validation-only incomplete-review bypass."
                )
            confirmed_ica = list(review_decision.confirmed_ica_components) if review_decision.complete else []
            validated_ica = validate_ica_components(confirmed_ica, n_components)
        else:
            if ica_bypass_reason:
                warnings.append(f"Gate 3C proceeding without ICA because explicit validation bypass was requested: {ica_bypass_reason}.")
            else:
                warnings.append("Gate 3C proceeding without ICA because preprocessing.ica.required_for_epoching is false and no ICA solution was available.")
            ica = None
            n_components = 0
            validated_ica = []
            solution_id = None
            review_decision = read_ica_review_decision(config, subject, recording.recording_part, root=root)
        review_provenance = {
            "ica_review_status": review_decision.review_status,
            "ica_decision_state": review_decision.decision_state,
            "ica_review_complete": review_decision.complete,
            "ica_decision_schema_version": review_decision.decision_schema_version,
            "ica_decision_id": review_decision.decision_id,
            "ica_audit_record_id": review_decision.audit_record_id,
            "ica_solution_id": solution_id,
            "decision_ica_solution_id": review_decision.ica_solution_id,
            "decision_fitted_component_count": review_decision.fitted_component_count,
            "decision_bad_channel_decision_id": review_decision.bad_channel_decision_id,
            "ica_reviewer": review_decision.reviewer,
            "ica_reviewed_at_utc": review_decision.reviewed_at_utc,
            "ica_legacy_ambiguous": review_decision.legacy_ambiguous,
            "automatic_eog_suggested_components": suggested,
            **suggestion_evidence.provenance(),
            "human_confirmed_ica_components": validated_ica,
            "ica_decision_file": str(manual_review_paths(config, root=root)["ica_components"]),
            "incomplete_review_validation_bypass": bool(allow_incomplete_ica_review_for_validation),
            **replay_root_fields,
        }
        ica_review = build_ica_review_summary(
            n_components=n_components,
            suggested_components=suggested,
            confirmed_components=validated_ica,
            reviewer_notes=notes,
            review_status=review_decision.review_status,
            ica_solution_id=solution_id,
        )
        ica_review = _annotate_ica_review_with_suggestion_evidence(ica_review, suggestion_evidence)
        ica_review.to_csv(outputs["ica_review_summary"], index=False)
        output_files["ica_review_summary"] = outputs["ica_review_summary"]
        _write_gate3c_sidecar(outputs["ica_review_summary"], subject, recording, config_path, root, input_files, warnings, {"applied_ica_components": validated_ica, **review_provenance})

        review_provenance.update({
            "bad_channel_review_status": bad_review_decision.review_status,
            "bad_channel_decision_state": bad_review_decision.decision_state,
            "bad_channel_review_complete": bad_review_decision.complete,
            "bad_channel_decision_id": bad_review_decision.decision_id,
            "bad_channel_semantic_decision_id": bad_review_decision.semantic_decision_id,
            "bad_channel_audit_record_id": bad_review_decision.audit_record_id,
            "bad_channel_source_recording_id": bad_source_id,
            "decision_bad_channel_source_recording_id": bad_review_decision.source_recording_id,
            "bad_channel_review_context_id": bad_context_id,
            "decision_bad_channel_review_context_id": bad_review_decision.review_context_id,
            "bad_channel_reviewed_channels": list(bad_review_decision.reviewed_channels),
            "bad_channel_reviewer": bad_review_decision.reviewer,
            "bad_channel_reviewed_at_utc": bad_review_decision.reviewed_at_utc,
            "human_confirmed_bad_channels": confirmed_bad,
            "bad_channel_decision_file": str(manual_review_paths(config, root=root)["bad_channels"]),
            "incomplete_bad_channel_review_validation_bypass": bool(
                allow_incomplete_bad_channel_review_for_validation
            ),
            "bad_channel_interpolation_disposition_status": bad_disposition.approval_status,
            "bad_channel_interpolation_disposition": bad_disposition.disposition,
            "bad_channel_interpolation_disposition_id": bad_disposition.decision_id,
            "retained_uninterpolated_bad_channels": retained_uninterpolated_bad,
            "bad_channel_interpolation_disposition_file": str(disposition_path),
        })
        ica_applied = False
        ica_skip_reason = ""
        segment_processing_summary = pd.DataFrame()
        block_processing_summary = pd.DataFrame()
        reference_provenance: dict[str, Any] = {}
        epoch_boundary_rejections = _empty_epoch_boundary_rejections()
        if blockwise_gate3c:
            stage_context = "blockwise_preprocessing"
            raw_before, raw, epochs, event_mapping, block_processing_summary, ica_applied, epoch_boundary_rejections, reference_provenance = _run_blockwise_gate3c_processing(
                input_raw=input_raw,
                subject=subject,
                recording=recording,
                blocks=gate3c_blocks,
                retained_alignment=retained_alignment,
                config=config,
                ica=ica,
                validated_ica=validated_ica,
                confirmed_bad=confirmed_bad,
                retained_uninterpolated_bad=retained_uninterpolated_bad,
                retained_bad_channel_disposition_id=bad_disposition.decision_id,
                input_files=input_files,
                outputs=outputs,
                output_files=output_files,
                warnings=warnings,
                root=root,
                config_path=config_path,
                replay_root_fields=replay_root_fields,
            )
            if ica is not None and not should_apply_ica_components(validated_ica):
                ica_skip_reason = "review_complete_zero_exclusions" if review_decision.complete else "validation_bypass_incomplete_review"
                warnings.append("Gate 3C skipped ICA application because the completed review confirmed zero exclusions." if review_decision.complete else "Gate 3C skipped ICA application under the explicit validation-only incomplete-review bypass.")
            elif ica_bypass_reason:
                ica_skip_reason = ica_bypass_reason
        elif segmentwise_gate3c:
            stage_context = "segmentwise_preprocessing"
            raw_before, raw, epochs, event_mapping, segment_processing_summary, ica_applied, reference_provenance = _run_segmentwise_gate3c_processing(
                input_raw=input_raw,
                subject=subject,
                recording=recording,
                task_segments=task_segments,
                retained_alignment=retained_alignment,
                config=config,
                ica=ica,
                validated_ica=validated_ica,
                confirmed_bad=confirmed_bad,
                retained_uninterpolated_bad=retained_uninterpolated_bad,
                retained_bad_channel_disposition_id=bad_disposition.decision_id,
                input_files=input_files,
                outputs=outputs,
                output_files=output_files,
                warnings=warnings,
                root=root,
                config_path=config_path,
                replay_root_fields=replay_root_fields,
            )
            if ica is not None and not should_apply_ica_components(validated_ica):
                ica_skip_reason = "review_complete_zero_exclusions" if review_decision.complete else "validation_bypass_incomplete_review"
                warnings.append("Gate 3C skipped ICA application because the completed review confirmed zero exclusions." if review_decision.complete else "Gate 3C skipped ICA application under the explicit validation-only incomplete-review bypass.")
            elif ica_bypass_reason:
                ica_skip_reason = ica_bypass_reason
        else:
            stage_context = "ica_reference_interpolation"
            raw = _ensure_gate3c_preloaded(raw, input_raw, subject, recording.recording_part, memory_settings, temporary_owner, warnings, crop_window=crop_window, task_segments=task_segments)
            raw, reference_provenance = apply_gate3c_signal_transforms(
                raw,
                config,
                ica=ica,
                validated_ica=validated_ica,
                confirmed_bad_channels=confirmed_bad,
                retained_uninterpolated_bad_channels=retained_uninterpolated_bad,
                retained_bad_channel_disposition_id=bad_disposition.decision_id,
            )
            ica_applied = bool(reference_provenance.get("ica_applied", False))
            if ica is not None and not ica_applied:
                ica_skip_reason = "review_complete_zero_exclusions" if review_decision.complete else "validation_bypass_incomplete_review"
                warnings.append("Gate 3C skipped ICA application because the completed review confirmed zero exclusions." if review_decision.complete else "Gate 3C skipped ICA application under the explicit validation-only incomplete-review bypass.")
            elif ica_bypass_reason:
                ica_skip_reason = ica_bypass_reason

        _mark_unused_channels_misc(raw, config)
        channel_types = {name: raw.get_channel_types(picks=[name])[0] for name in raw.ch_names}
        reference_summary = build_reference_summary(
            raw.ch_names,
            channel_types,
            config,
            confirmed_bad_channels=confirmed_bad,
            runtime_provenance=reference_provenance,
        )

        interp_cfg = dict(dict(config.get("preprocessing", {}) or {}).get("interpolation", {}) or {})
        gate3a_proposals = _read_gate3a_bad_channel_proposals(
            config,
            subject,
            recording.recording_part,
            root=root,
            path_override=gate3a_bad_channel_decisions_path,
        )
        proposed_channels = gate3a_proposals.loc[gate3a_proposals.get("automatic_bad_channel_proposal", False).astype(bool), "channel"].astype(str).tolist() if not gate3a_proposals.empty else []
        proposal_reasons = (
            dict(zip(gate3a_proposals["channel"].astype(str), gate3a_proposals.get("automatic_proposal_reasons", "").fillna("").astype(str)))
            if not gate3a_proposals.empty
            else {}
        )
        interpolation = build_interpolation_table(
            raw.ch_names,
            confirmed_bad_channels=confirmed_bad,
            automatic_bad_channel_proposals=proposed_channels,
            automatic_proposal_reasons=proposal_reasons,
            method=str(interp_cfg.get("method", "mne_interpolate_bads")),
            decision_state=bad_review_decision.decision_state,
            decision_id=bad_review_decision.decision_id,
            semantic_decision_id=bad_review_decision.semantic_decision_id,
            audit_record_id=bad_review_decision.audit_record_id,
            decision_file=str(manual_review_paths(config, root=root)["bad_channels"]),
            retained_bad_channel_disposition=bad_disposition.disposition,
            retained_bad_channel_disposition_id=bad_disposition.decision_id,
            runtime_provenance=reference_provenance,
        )

        if not blockwise_gate3c and not segmentwise_gate3c:
            raw.save(outputs["ica_cleaned_raw_fif"], overwrite=True, verbose=False)
            output_files["ica_cleaned_raw_fif"] = outputs["ica_cleaned_raw_fif"]
            _write_gate3c_sidecar(
                outputs["ica_cleaned_raw_fif"],
                subject,
                recording,
                config_path,
                root,
                input_files,
                warnings,
                {
                    "applied_ica_components": validated_ica,
                    "confirmed_bad_channels": confirmed_bad,
                    "ica_applied": ica_applied,
                    "ica_skip_reason": ica_skip_reason,
                    "ica_status": "skipped_configured" if ica_bypass_reason else "not_available",
                    "gate3c_memory": _serialisable_memory_settings(memory_settings),
                    **review_provenance,
                    **reference_provenance,
                },
            )

            events_df = _image_events_from_raw(
                raw, trigger_definition=trigger_definition
            )
            if not segmented_event_mapping.empty:
                event_mapping = _event_epoch_mapping_from_segmented_events(
                    segmented_event_mapping,
                    retained_alignment,
                    trigger_definition=trigger_definition,
                )
            else:
                event_mapping = build_event_epoch_mapping(
                    events_df,
                    retained_alignment,
                    trigger_definition=trigger_definition,
                )
            original_event_mapping = event_mapping.copy()
            event_mapping, epoch_boundary_rejections = validate_epoch_boundaries(
                event_mapping,
                raw=raw,
                subject=subject,
                recording_part=recording.recording_part,
                block=pd.NA,
                config=config,
            )
            if not epoch_boundary_rejections.empty:
                rejected_mapping = original_event_mapping[
                    original_event_mapping["event_sample"].astype("Int64").isin(epoch_boundary_rejections["event_sample"].astype("Int64"))
                ].copy()
                rejection_reasons = {
                    _nullable_int(row.get("event_sample")): str(row.get("reason", "epoch_boundary_outside_data") or "epoch_boundary_outside_data")
                    for _, row in epoch_boundary_rejections.iterrows()
                }
                rejected_mapping["epoch_index"] = pd.NA
                rejected_mapping["mapping_status"] = rejected_mapping["event_sample"].map(lambda sample: f"rejected_{rejection_reasons.get(_nullable_int(sample), 'epoch_boundary_outside_data')}")
                event_mapping = pd.concat([event_mapping, rejected_mapping], ignore_index=True)
        stage_context = "epoching"
        if not blockwise_gate3c and not segmentwise_gate3c:
            epochs = _create_image_epochs(raw, event_mapping, retained_alignment, config)

        stage_context = "residual_epoch_qc"
        epochs, residual_qc = apply_residual_epoch_ptp_qc(epochs, config)
        if blockwise_gate3c and epochs.metadata is not None and "block" in epochs.metadata:
            for block in gate3c_blocks:
                block_indices = [index for index, matches in enumerate(epochs.metadata["block"].astype(str).eq(str(block.block_id)).tolist()) if matches]
                block_path = gate3c_block_epochs_path(outputs, block)
                if block_indices:
                    epochs[block_indices].save(block_path, overwrite=True)
                elif block_path.exists():
                    block_path.unlink()
                    output_files.pop(f"block{int(block.block_index):02d}_epochs_fif", None)
        if "subject" not in residual_qc:
            residual_qc["subject"] = subject
        if "recording_part" not in residual_qc:
            residual_qc["recording_part"] = recording.recording_part
        epoch_rejections = build_epoch_rejection_table(
            alignment,
            event_mapping,
            recording.recording_part,
            boundary_rejections=epoch_boundary_rejections,
            trigger_definition=trigger_definition,
        )
        epoch_rejections = merge_residual_qc_with_rejection_table(epoch_rejections, residual_qc)
        counts = condition_count_table(epoch_rejections)
        duplicate_summary = duplicate_incomplete_summary(epoch_rejections)
        retention_summary = participant_retention_summary(epoch_rejections)

        for label, table in (
            ("reference_summary", reference_summary),
            ("interpolation_summary", interpolation),
            ("event_epoch_mapping", event_mapping),
            ("epoch_rejection_table", epoch_rejections),
            ("residual_epoch_qc", residual_qc),
            ("participant_retention_summary", retention_summary),
            ("epoch_boundary_rejections", epoch_boundary_rejections),
            ("condition_counts", counts),
            ("duplicate_incomplete_summary", duplicate_summary),
        ):
            path = outputs[label]
            table.to_csv(path, index=False)
            output_files[label] = path
            _write_gate3c_sidecar(
                path,
                subject,
                recording,
                config_path,
                root,
                input_files,
                warnings,
                {"applied_ica_components": validated_ica, **review_provenance, **residual_epoch_qc_settings(config), **reference_provenance, **coordinate_provenance},
            )
        if segmentwise_gate3c:
            segment_processing_summary.to_csv(outputs["segment_processing_summary"], index=False)
            output_files["segment_processing_summary"] = outputs["segment_processing_summary"]
            _write_gate3c_sidecar(outputs["segment_processing_summary"], subject, recording, config_path, root, input_files, warnings, {"segmentwise_gate3c": True, **reference_provenance})
        if blockwise_gate3c:
            block_processing_summary.to_csv(outputs["block_processing_summary"], index=False)
            output_files["block_processing_summary"] = outputs["block_processing_summary"]
            _write_gate3c_sidecar(outputs["block_processing_summary"], subject, recording, config_path, root, input_files, warnings, {"preprocessing_mode": "blockwise_epoching", "blockwise_gate3c": True, **reference_provenance, **coordinate_provenance})

        epoch_settings = {**image_epoch_settings(config), **epoch_runtime_settings(config), **residual_epoch_qc_settings(config)}
        if epoch_settings["save_epochs_fif"]:
            epochs.save(outputs["epochs_fif"], overwrite=True)
            output_files["epochs_fif"] = outputs["epochs_fif"]
            cleaned_inputs = [path for key, path in output_files.items() if key == "ica_cleaned_raw_fif" or key.startswith("ica_cleaned_raw_segment_")]
            _write_gate3c_sidecar(outputs["epochs_fif"], subject, recording, config_path, root, input_files + cleaned_inputs, warnings, {"applied_ica_components": validated_ica, **review_provenance, "preprocessing_mode": "blockwise_epoching" if blockwise_gate3c else "legacy_single_raw_epoching", "blockwise_gate3c": blockwise_gate3c, "segmentwise_gate3c": segmentwise_gate3c, **epoch_settings, **residual_epoch_qc_settings(config), **reference_provenance, **coordinate_provenance})

        metadata = epochs.metadata.copy() if epochs.metadata is not None else pd.DataFrame()
        metadata.to_csv(outputs["epoch_metadata_csv"], index=False)
        output_files["epoch_metadata_csv"] = outputs["epoch_metadata_csv"]
        _write_gate3c_sidecar(outputs["epoch_metadata_csv"], subject, recording, config_path, root, input_files, warnings, {"applied_ica_components": validated_ica, **review_provenance, **residual_epoch_qc_settings(config), **coordinate_provenance})

        processing_log = _processing_log(
            subject,
            recording.recording_part,
            validated_ica,
            confirmed_bad,
            reference_summary,
            interpolation,
            int(len(epochs)),
            ica_applied=ica_applied,
            ica_skip_reason=ica_skip_reason,
            epoch_settings=epoch_settings,
            memory_settings=memory_settings,
            review_provenance=review_provenance,
        )
        processing_log.to_csv(outputs["processing_log"], index=False)
        output_files["processing_log"] = outputs["processing_log"]
        _write_gate3c_sidecar(outputs["processing_log"], subject, recording, config_path, root, input_files, warnings, {"applied_ica_components": validated_ica, **review_provenance, **reference_provenance, **coordinate_provenance})

        stage_context = "plotting"
        output_files.update(
            write_gate3c_figures(
                raw_before=raw_before,
                raw_after=raw,
                epochs=epochs,
                condition_counts=counts,
                epoch_rejection_table=epoch_rejections,
                interpolation_table=interpolation,
                subject=subject,
                recording=recording,
                fig_dir=outputs["fig_dir"],
                root=root,
                config_path=config_path,
                input_files=input_files,
                warnings=warnings,
                config=config,
                provenance_parameters=replay_root_fields,
            )
        )

        report = generate_gate3c_epoch_qc_markdown_report(
            subject=subject,
            recording_part=recording.recording_part,
            input_files=input_files,
            output_files=output_files,
            ica_review_summary=ica_review,
            reference_summary=reference_summary,
            interpolation_summary=interpolation,
            condition_counts=counts,
            epoch_rejection_table=epoch_rejections,
            warnings=warnings,
            failures=failures,
        )
        report_path = write_markdown_report(outputs["report"], report)
        output_files["epoch_qc_report"] = report_path
        _write_gate3c_sidecar(report_path, subject, recording, config_path, root, input_files, warnings, {"applied_ica_components": validated_ica, **review_provenance, **coordinate_provenance})
        del epochs, raw, raw_before
        gc.collect()
        status = "ok"
    except Exception as error:
        failures.append(str(error))
        failure_path = outputs["qc_dir"] / "failure.csv"
        status = f"failed_{stage_context}"
        pd.DataFrame([{"subject": subject, "recording_part": recording.recording_part, "status": status, "failure_stage": stage_context, "failure": str(error)}]).to_csv(failure_path, index=False)
        output_files["failure_record"] = failure_path
        _write_gate3c_sidecar(failure_path, subject, recording, config_path, root, input_files, warnings + failures, {"status": status, "failure_stage": stage_context, **replay_root_fields})
    finally:
        for variable_name in ("raw", "raw_before"):
            candidate = locals().get(variable_name)
            close = getattr(candidate, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    warnings.append(f"Could not close Gate 3C temporary-backed raw object: {error}")
        gc.collect()
        if temporary_owner is not None:
            temporary_owner.cleanup(warnings)

    return Gate3CResult(
        subject=subject,
        recording_part=recording.recording_part,
        status=status,
        output_files=output_files,
        warnings=tuple(warnings),
        failures=tuple(failures),
        ica_decision_state=getattr(locals().get("review_decision"), "decision_state", None),
        confirmed_ica_components=tuple(locals().get("validated_ica", ())),
        bad_channel_decision_state=getattr(locals().get("bad_review_decision"), "decision_state", None),
    )


def write_gate3c_figures(
    *,
    raw_before,
    raw_after,
    epochs,
    condition_counts: pd.DataFrame,
    epoch_rejection_table: pd.DataFrame,
    interpolation_table: pd.DataFrame,
    subject: str,
    recording: RecordingPart,
    fig_dir: Path,
    root: Path | None,
    config_path: str | Path | None,
    input_files: Sequence[str | Path],
    warnings: list[str],
    config: Mapping[str, Any],
    provenance_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate reusable Gate 3C QC figures."""

    outputs: dict[str, Path] = {}
    fig_cfg = dict(dict(config.get("epochs", {}) or {}).get("qc_figures", {}) or {})
    if not bool(fig_cfg.get("enabled", True)):
        return outputs
    fail_on_error = bool(fig_cfg.get("fail_pipeline_on_figure_error", False))
    erp_preview_enabled = bool(fig_cfg.get("erp_preview_enabled", False))
    ica_duration = fig_cfg.get("ica_before_after_duration_s", 20)
    ica_start = fig_cfg.get("ica_before_after_start_s", 0)
    figure_paths = gate3c_epoch_figure_paths(fig_dir, subject, recording.recording_part)
    builders = {
        "epoch_count_figure": lambda: plot_epoch_counts(condition_counts, title=f"{subject} {recording.recording_part} epoch counts"),
        "epoch_rejection_figure": lambda: plot_epoch_rejection_summary(epoch_rejection_table, title=f"{subject} {recording.recording_part} epoch rejections"),
        "ica_before_after_figure": lambda: plot_ica_before_after(raw_before, raw_after, title=f"{subject} {recording.recording_part} ICA before/after", start=float(ica_start), duration=None if ica_duration is None else float(ica_duration)),
        "interpolation_summary_figure": lambda: plot_interpolation_summary(interpolation_table, title=f"{subject} {recording.recording_part} interpolation"),
    }
    if erp_preview_enabled:
        builders["erp_preview_figure"] = lambda: plot_erp_preview(epochs, title=f"{subject} {recording.recording_part} ERP preview")
    for label, builder in builders.items():
        try:
            figure = builder()
            outputs[label] = save_figure_with_stage_provenance(
                figure,
                figure_paths[label],
                pipeline_stage="gate3c_epoch_qc_figure",
                subject=subject,
                recording_part=recording.recording_part,
                root=root,
                config_path=config_path,
                input_files=input_files,
                parameters_summary={"figure_type": label, **dict(provenance_parameters or {})},
                warnings=warnings,
            )
        except Exception as error:
            if fail_on_error:
                raise
            warnings.append(f"Could not generate {label}: {error}")
    return outputs


def _run_segmentwise_gate3c_processing(
    *,
    input_raw: Path,
    subject: str,
    recording: RecordingPart,
    task_segments: Sequence[Any],
    retained_alignment: pd.DataFrame,
    config: Mapping[str, Any],
    ica,
    validated_ica: Sequence[int],
    confirmed_bad: Sequence[str],
    input_files: Sequence[str | Path],
    outputs: Mapping[str, Path],
    output_files: dict[str, Path],
    warnings: list[str],
    root: Path | None,
    config_path: str | Path | None,
    replay_root_fields: Mapping[str, Any] | None = None,
    retained_uninterpolated_bad: Sequence[str] = (),
    retained_bad_channel_disposition_id: str | None = None,
):
    """Run Gate 3C preprocessing and epoching independently for task segments."""

    import mne

    trigger_definition = KawaiiTriggerDefinition.from_config(config)
    epoch_list = []
    mapping_rows = []
    summary_rows = []
    raw_before_preview = None
    raw_after_preview = None
    ica_applied = False
    reference_provenance: dict[str, Any] = {}

    for segment in task_segments:
        raw_segment = mne.io.read_raw_fif(input_raw, preload=False, verbose=False)
        raw_segment = raw_segment.crop(tmin=float(segment.start_s), tmax=float(segment.end_s), include_tmax=True)
        before_segment = raw_segment.copy()
        raw_segment.load_data(verbose=False)
        raw_segment, segment_reference_provenance = apply_gate3c_signal_transforms(
            raw_segment,
            config,
            ica=ica,
            validated_ica=validated_ica,
            confirmed_bad_channels=confirmed_bad,
            retained_uninterpolated_bad_channels=retained_uninterpolated_bad,
            retained_bad_channel_disposition_id=retained_bad_channel_disposition_id,
        )
        ica_applied = ica_applied or bool(segment_reference_provenance.get("ica_applied", False))
        if not reference_provenance:
            reference_provenance = dict(segment_reference_provenance)

        segment_path = outputs["clean_dir"] / f"ica_cleaned_segment_{int(segment.segment_id):02d}_raw.fif"
        raw_segment.save(segment_path, overwrite=True, verbose=False)
        output_files[f"ica_cleaned_raw_segment_{int(segment.segment_id):02d}"] = segment_path
        _write_gate3c_sidecar(
            segment_path,
            subject,
            recording,
            config_path,
            root,
            input_files,
            warnings,
            {
                "segment_id": int(segment.segment_id),
                "segment_start_sample": int(segment.start_sample),
                "segment_end_sample": int(segment.end_sample),
                "applied_ica_components": list(validated_ica),
                "confirmed_bad_channels": list(confirmed_bad),
                "segmentwise_gate3c": True,
                **segment_reference_provenance,
                **dict(replay_root_fields or {}),
            },
        )

        segment_mapping = event_epoch_mapping_for_task_segment(
            retained_alignment,
            segment,
            trigger_definition=trigger_definition,
        )
        if not segment_mapping.empty:
            segment_epochs = _create_image_epochs(raw_segment, segment_mapping, retained_alignment, config)
            epoch_list.append(segment_epochs)
            mapping_rows.append(segment_mapping)
        if raw_before_preview is None:
            raw_before_preview = before_segment
            raw_after_preview = raw_segment.copy()
        summary_rows.append(
            {
                "subject": subject,
                "recording_part": recording.recording_part,
                "segment_id": int(segment.segment_id),
                "start_trial": int(segment.start_trial),
                "end_trial": int(segment.end_trial),
                "start_sample": int(segment.start_sample),
                "end_sample": int(segment.end_sample),
                "duration_s": float(segment.duration_s),
                "n_retained_trials": int(segment.n_trials),
                "n_epochs_created": int(len(segment_mapping)),
                "ica_applied": bool(segment_reference_provenance.get("ica_applied", False)),
                "cpz_reconstructed": bool(segment_reference_provenance.get("cpz_reconstructed", False)),
                "average_reference_applied": bool(segment_reference_provenance.get("average_reference_applied", False)),
                "interpolation_applied": bool(segment_reference_provenance.get("interpolation_requested", False)),
                "interpolation_geometry_safety_status": segment_reference_provenance.get("interpolation_geometry_safety_status", ""),
                "operation_order": segment_reference_provenance.get("operation_order", []),
                "cleaned_raw_file": str(segment_path),
            }
        )

    if not epoch_list:
        raise ValueError("Gate 3C segment-wise epoching failed: no segment epochs were created.")
    epochs = epoch_list[0] if len(epoch_list) == 1 else mne.concatenate_epochs(epoch_list, add_offset=False, verbose=False)
    event_mapping = pd.concat(mapping_rows, ignore_index=True) if mapping_rows else pd.DataFrame()
    event_mapping["epoch_index"] = range(len(event_mapping))
    segment_summary = pd.DataFrame(summary_rows)
    if raw_before_preview is None or raw_after_preview is None:
        raise ValueError("Gate 3C segment-wise preprocessing failed: no preview raw segment was available.")
    return raw_before_preview, raw_after_preview, epochs, event_mapping, segment_summary, ica_applied, reference_provenance


def _run_blockwise_gate3c_processing(
    *,
    input_raw: Path,
    subject: str,
    recording: RecordingPart,
    blocks: Sequence[Gate3CBlock],
    retained_alignment: pd.DataFrame,
    config: Mapping[str, Any],
    ica,
    validated_ica: Sequence[int],
    confirmed_bad: Sequence[str],
    input_files: Sequence[str | Path],
    outputs: Mapping[str, Path],
    output_files: dict[str, Path],
    warnings: list[str],
    root: Path | None,
    config_path: str | Path | None,
    replay_root_fields: Mapping[str, Any] | None = None,
    retained_uninterpolated_bad: Sequence[str] = (),
    retained_bad_channel_disposition_id: str | None = None,
):
    """Run the standard Gate 3C architecture: preprocess and epoch one block at a time."""

    import mne

    trigger_definition = KawaiiTriggerDefinition.from_config(config)
    block_epoch_paths: list[Path] = []
    mapping_rows = []
    boundary_rejection_rows = []
    summary_rows = []
    raw_before_preview = None
    raw_after_preview = None
    ica_applied = False
    reference_provenance: dict[str, Any] = {}

    for block in blocks:
        raw_block = mne.io.read_raw_fif(input_raw, preload=False, verbose=False)
        raw_block = raw_block.crop(tmin=float(block.start_s), tmax=float(block.end_s), include_tmax=True)
        before_block = _raw_qc_preview(raw_block, config)
        raw_block.load_data(verbose=False)
        raw_block, block_reference_provenance = apply_gate3c_signal_transforms(
            raw_block,
            config,
            ica=ica,
            validated_ica=validated_ica,
            confirmed_bad_channels=confirmed_bad,
            retained_uninterpolated_bad_channels=retained_uninterpolated_bad,
            retained_bad_channel_disposition_id=retained_bad_channel_disposition_id,
        )
        ica_applied = ica_applied or bool(block_reference_provenance.get("ica_applied", False))
        if not reference_provenance:
            reference_provenance = dict(block_reference_provenance)

        block_mapping = event_epoch_mapping_for_block(
            retained_alignment,
            block,
            trigger_definition=trigger_definition,
        )
        block_mapping, block_boundary_rejections = validate_epoch_boundaries(
            block_mapping,
            raw=raw_block,
            subject=subject,
            recording_part=recording.recording_part,
            block=block.block_id,
            config=config,
        )
        if not block_boundary_rejections.empty:
            boundary_rejection_rows.append(block_boundary_rejections)
            rejected_mapping = event_epoch_mapping_for_block(
                retained_alignment,
                block,
                trigger_definition=trigger_definition,
            )
            rejected_mapping = rejected_mapping[
                rejected_mapping["event_sample"].astype("Int64").isin(block_boundary_rejections["event_sample"].astype("Int64"))
            ].copy()
            rejection_reasons = {
                _nullable_int(row.get("event_sample")): str(row.get("reason", "epoch_boundary_outside_data") or "epoch_boundary_outside_data")
                for _, row in block_boundary_rejections.iterrows()
            }
            rejected_mapping["epoch_index"] = pd.NA
            rejected_mapping["mapping_status"] = rejected_mapping["event_sample"].map(lambda sample: f"rejected_{rejection_reasons.get(_nullable_int(sample), 'epoch_boundary_outside_data')}")
            mapping_rows.append(rejected_mapping)
        if block_mapping.empty:
            warnings.append(f"Gate 3C block {block.block_id} contained no retained image-onset epochs; skipping block epochs.")
            n_epochs = 0
        else:
            block_epochs = _create_image_epochs(raw_block, block_mapping, retained_alignment, config)
            block_path = gate3c_block_epochs_path(outputs, block)
            block_epochs.save(block_path, overwrite=True)
            output_files[f"block{int(block.block_index):02d}_epochs_fif"] = block_path
            _write_gate3c_sidecar(
                block_path,
                subject,
                recording,
                config_path,
                root,
                input_files,
                warnings,
                {
                    "preprocessing_mode": "blockwise_epoching",
                    "blockwise_gate3c": True,
                    "block": block.block_id,
                    "block_index": int(block.block_index),
                    "coordinate_frame": "saved_filtered_fif_annotation_samples",
                    "coordinate_source": "validated_original_to_concatenated_saved_annotation",
                    "applied_ica_components": list(validated_ica),
                    "confirmed_bad_channels": list(confirmed_bad),
                    **block_reference_provenance,
                    **dict(replay_root_fields or {}),
                },
            )
            block_epoch_paths.append(block_path)
            mapping_rows.append(block_mapping)
            n_epochs = int(len(block_mapping))
            del block_epochs
        if raw_before_preview is None:
            raw_before_preview = before_block
            raw_after_preview = _raw_qc_preview(raw_block, config)
        summary_rows.append(
            {
                "subject": subject,
                "recording_part": recording.recording_part,
                "block": block.block_id,
                "block_index": int(block.block_index),
                "start_trial": int(block.start_trial),
                "end_trial": int(block.end_trial),
                "start_sample": int(block.start_sample),
                "end_sample": int(block.end_sample),
                "coordinate_frame": "saved_filtered_fif_annotation_samples",
                "duration_s": float(block.duration_s),
                "n_retained_trials": int(block.n_retained_trials),
                "n_epochs_created": int(n_epochs),
                "ica_applied": bool(block_reference_provenance.get("ica_applied", False)),
                "cpz_reconstructed": bool(block_reference_provenance.get("cpz_reconstructed", False)),
                "average_reference_applied": bool(block_reference_provenance.get("average_reference_applied", False)),
                "interpolation_applied": bool(block_reference_provenance.get("interpolation_requested", False)),
                "interpolation_geometry_safety_status": block_reference_provenance.get("interpolation_geometry_safety_status", ""),
                "operation_order": block_reference_provenance.get("operation_order", []),
                "block_epochs_file": str(gate3c_block_epochs_path(outputs, block)) if n_epochs else "",
            }
        )
        del raw_block
        gc.collect()

    if not block_epoch_paths:
        raise ValueError("Gate 3C block-wise epoching failed: no block epochs were created.")
    epochs = _read_combined_epochs_from_files(block_epoch_paths)
    event_mapping = pd.concat(mapping_rows, ignore_index=True) if mapping_rows else pd.DataFrame()
    event_mapping["epoch_index"] = pd.NA
    epochable_mask = event_mapping["event_sample"].notna() & event_mapping["event_code"].notna()
    if "mapping_status" in event_mapping:
        epochable_mask &= ~event_mapping["mapping_status"].fillna("").astype(str).str.startswith("rejected_")
    event_mapping.loc[epochable_mask, "epoch_index"] = range(int(epochable_mask.sum()))
    epoch_boundary_rejections = pd.concat(boundary_rejection_rows, ignore_index=True) if boundary_rejection_rows else _empty_epoch_boundary_rejections()
    block_summary = pd.DataFrame(summary_rows)
    if raw_before_preview is None or raw_after_preview is None:
        raise ValueError("Gate 3C block-wise preprocessing failed: no preview raw block was available.")
    return raw_before_preview, raw_after_preview, epochs, event_mapping, block_summary, ica_applied, epoch_boundary_rejections, reference_provenance


def _empty_epoch_boundary_rejections() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "subject",
            "recording_part",
            "block",
            "trial",
            "event_code",
            "event_sample",
            "event_sample_original",
            "requested_epoch_start",
            "requested_epoch_end",
            "available_start",
            "available_end",
            "reason",
        ]
    )


def _read_combined_epochs_from_files(epoch_paths: Sequence[Path]):
    import mne

    paths = [Path(path) for path in epoch_paths]
    if not paths:
        raise ValueError("No block epoch files are available for combined Gate 3C epochs.")
    if len(paths) == 1:
        return mne.read_epochs(paths[0], preload=False, verbose=False)
    epochs = [mne.read_epochs(path, preload=False, verbose=False) for path in paths]
    try:
        return mne.concatenate_epochs(epochs, add_offset=False, verbose=False)
    finally:
        del epochs
        gc.collect()


def _raw_qc_preview(raw, config: Mapping[str, Any]):
    """Create a short in-memory Raw preview for Gate 3C figures."""

    import mne

    fig_cfg = dict(dict(config.get("epochs", {}) or {}).get("qc_figures", {}) or {})
    duration = fig_cfg.get("ica_before_after_duration_s", 20)
    start_s = float(fig_cfg.get("ica_before_after_start_s", 0) or 0)
    sfreq = float(raw.info["sfreq"])
    start = max(0, int(round(start_s * sfreq)))
    if duration is None:
        stop = int(raw.n_times)
    else:
        stop = min(int(raw.n_times), start + max(1, int(round(float(duration) * sfreq))))
    if stop <= start:
        start = 0
        stop = min(int(raw.n_times), max(1, int(round(20.0 * sfreq))))
    data = raw.get_data(start=start, stop=stop)
    info = raw.info.copy()
    preview = mne.io.RawArray(data, info, first_samp=int(raw.first_samp) + start, verbose=False)
    preview.info["bads"] = list(raw.info.get("bads", []) or [])
    return preview


def _ensure_gate3c_preloaded(
    raw,
    input_raw: Path,
    subject: str,
    recording_part: str,
    memory_settings: Mapping[str, Any],
    temporary_owner: TemporaryResourceOwner,
    warnings: list[str],
    *,
    crop_window=None,
    task_segments=None,
):
    """Return a data-resident raw object, using disk memmap when configured."""

    if bool(getattr(raw, "preload", False)):
        return raw
    import mne

    if task_segments:
        reloaded = mne.io.read_raw_fif(input_raw, preload=False, verbose=False)
        reloaded = apply_task_segments(reloaded, task_segments)
        warnings.append("Gate 3C applied task segments before preload so long idle gaps are excluded.")
        reloaded.load_data(verbose=False)
        reloaded.info["bads"] = list(getattr(raw, "info", {}).get("bads", []) or [])
        return reloaded

    if crop_window is not None:
        reloaded = mne.io.read_raw_fif(input_raw, preload=False, verbose=False)
        reloaded = apply_task_crop(reloaded, crop_window)
        warnings.append("Gate 3C cropped raw before preload so only the task window is loaded.")
        reloaded.load_data(verbose=False)
        reloaded.info["bads"] = list(getattr(raw, "info", {}).get("bads", []) or [])
        return reloaded

    if bool(memory_settings.get("use_memmap", True)):
        memmap_path = temporary_owner.allocate("raw_preload.dat")
        warnings.append(f"Gate 3C loaded raw data using disk memmap: {memmap_path}")
        reloaded = mne.io.read_raw_fif(input_raw, preload=str(memmap_path), verbose=False)
    else:
        warnings.append("Gate 3C loaded raw data into memory because gate3c_use_memmap is false.")
        reloaded = mne.io.read_raw_fif(input_raw, preload=True, verbose=False)
    reloaded.info["bads"] = list(getattr(raw, "info", {}).get("bads", []) or [])
    return reloaded


def _serialisable_memory_settings(memory_settings: Mapping[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in memory_settings.items()}


def _read_suggested_ica_components(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    root: Path | None = None,
    historical_artifact_path: str | Path | None = None,
    historical_sha256: str | None = None,
    historical_size_bytes: int | None = None,
    historical_generation_id: str | None = None,
    isolated_replay: bool = False,
) -> ICASuggestionEvidence:
    historical_artifact_bound = historical_artifact_path is not None
    if isolated_replay and not historical_artifact_bound:
        return ICASuggestionEvidence((), "NOT_AVAILABLE", None)

    path = Path(historical_artifact_path) if historical_artifact_bound else gate3b_output_paths(
        config, subject, recording_part, root=root
    )["eog_suggestions"]
    if not path.is_file():
        return ICASuggestionEvidence(
            (),
            "NOT_REVALIDATED" if historical_artifact_bound else "NOT_AVAILABLE",
            path if historical_artifact_bound else None,
            historical_sha256,
            historical_size_bytes,
            historical_generation_id,
        )
    try:
        payload = path.read_bytes()
    except OSError:
        return ICASuggestionEvidence(
            (),
            "NOT_REVALIDATED",
            path,
            historical_sha256,
            historical_size_bytes,
            historical_generation_id,
        )

    observed_sha256 = hashlib.sha256(payload).hexdigest()
    observed_size_bytes = len(payload)
    if historical_artifact_bound:
        historical_identity_verified = (
            _is_sha256_hex(historical_sha256)
            and historical_size_bytes is not None
            and observed_size_bytes == historical_size_bytes
            and observed_sha256 == str(historical_sha256).lower()
            and _is_sha256_hex(historical_generation_id)
        )
        if not historical_identity_verified:
            return ICASuggestionEvidence(
                (),
                "NOT_REVALIDATED",
                path,
                historical_sha256,
                historical_size_bytes,
                historical_generation_id,
                observed_sha256,
                observed_size_bytes,
            )

    try:
        table = pd.read_csv(io.BytesIO(payload))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return ICASuggestionEvidence(
            (),
            "NOT_REVALIDATED",
            path,
            historical_sha256,
            historical_size_bytes,
            historical_generation_id,
            observed_sha256,
            observed_size_bytes,
        )
    required = {"suggested_ocular_component", "component"}
    if table.empty or not required.issubset(table.columns):
        return ICASuggestionEvidence(
            (),
            "NOT_REVALIDATED",
            path,
            historical_sha256,
            historical_size_bytes,
            historical_generation_id,
            observed_sha256,
            observed_size_bytes,
        )
    components = tuple(
        sorted(
            set(
                pd.to_numeric(
                    table.loc[table["suggested_ocular_component"].astype(bool), "component"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )
        )
    )
    status = "AVAILABLE_VERIFIED" if historical_artifact_bound else "AVAILABLE_CURRENT_OUTPUT"
    return ICASuggestionEvidence(
        components,
        status,
        path,
        historical_sha256,
        historical_size_bytes,
        historical_generation_id,
        observed_sha256,
        observed_size_bytes,
    )


def _is_sha256_hex(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _annotate_ica_review_with_suggestion_evidence(
    table: pd.DataFrame,
    evidence: ICASuggestionEvidence,
) -> pd.DataFrame:
    result = table.copy()
    result["automatic_suggestion_evidence_status"] = evidence.status
    result["automatic_suggestion_source_path"] = (
        "" if evidence.source_path is None else str(evidence.source_path)
    )
    result["automatic_suggestion_sha256"] = evidence.sha256 or ""
    result["automatic_suggestion_gate3b_generation_id"] = evidence.generation_id or ""
    result["automatic_suggestion_observed_sha256"] = evidence.observed_sha256 or ""
    result["automatic_suggestion_observed_size_bytes"] = evidence.observed_size_bytes
    return result


def _read_gate3a_bad_channel_proposals(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    root: Path | None = None,
    path_override: str | Path | None = None,
) -> pd.DataFrame:
    """Read Gate 3A proposal decisions for proposal→confirmation provenance."""

    path = (
        Path(path_override)
        if path_override is not None
        else subject_part_stage_dir(config, subject, recording_part, "raw_qc", root=root, create=False)
        / "bad_channel_decisions.csv"
    )
    if not path.exists():
        return pd.DataFrame(columns=["channel", "automatic_bad_channel_proposal", "automatic_proposal_reasons"])
    table = pd.read_csv(path)
    required = {"channel", "automatic_bad_channel_proposal"}
    if not required.issubset(table.columns):
        return pd.DataFrame(columns=["channel", "automatic_bad_channel_proposal", "automatic_proposal_reasons"])
    if "automatic_proposal_reasons" not in table:
        table["automatic_proposal_reasons"] = ""
    return table


def _mark_unused_channels_misc(raw, config: Mapping[str, Any]) -> None:
    channels = dict(config.get("channels", {}) or {}).get("eeg", {}) or {}
    unused = [channel for channel in channels.get("unused", []) or [] if channel in raw.ch_names]
    if unused:
        raw.set_channel_types({channel: "misc" for channel in unused}, verbose=False)


def _image_events_from_raw(
    raw, *, trigger_definition: KawaiiTriggerDefinition | None = None
) -> pd.DataFrame:
    import re
    import mne

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    definition = trigger_definition or KawaiiTriggerDefinition.from_config()
    image_codes = set(definition.image_event_id.values())
    rows = []
    for description, event_code in event_id.items():
        match = re.search(r"(\d+)$", str(description))
        trigger_code = int(match.group(1)) if match else int(event_code)
        if trigger_code not in image_codes:
            continue
        for sample, _, code in events[events[:, 2] == event_code]:
            rows.append({"event_sample": int(sample), "event_code": trigger_code, "mne_event_code": int(code)})
    return pd.DataFrame(rows).sort_values(["event_sample", "event_code"], kind="mergesort").reset_index(drop=True) if rows else pd.DataFrame(columns=["event_sample", "event_code", "mne_event_code"])


def _create_image_epochs(raw, event_mapping: pd.DataFrame, retained_alignment: pd.DataFrame, config: Mapping[str, Any]):
    import mne

    settings = image_epoch_settings(config)
    runtime = epoch_runtime_settings(config)
    event_mapping = _epochable_event_mapping(event_mapping)
    if event_mapping.empty:
        raise ValueError("No retained image-onset events available for epoching after alignment/event mapping.")
    _validate_unique_keys(event_mapping, ["event_sample", "event_code"], "Epoch event mapping")
    events = event_mapping[["event_sample", "event_code"]].copy()
    events.insert(1, "previous_event", 0)
    event_array = events[["event_sample", "previous_event", "event_code"]].astype(int).to_numpy()
    present_codes = sorted(set(event_array[:, 2].astype(int).tolist()))
    configured_event_id = KawaiiTriggerDefinition.from_config(config).image_event_id
    event_id = {label: code for label, code in configured_event_id.items() if code in present_codes}
    if not event_id:
        raise ValueError(f"No configured image-onset event codes found in event mapping: {present_codes}")
    metadata = event_mapping.copy()
    retained_meta = retained_alignment.drop(columns=[col for col in retained_alignment.columns if col.endswith("_alignment")], errors="ignore").copy()
    if "event_sample_original" in metadata:
        if "event_sample_original" not in retained_meta:
            retained_meta["event_sample_original"] = pd.to_numeric(
                retained_meta["event_sample"], errors="coerce"
            ).astype("Int64")
        else:
            retained_meta["event_sample_original"] = pd.to_numeric(
                retained_meta["event_sample_original"], errors="coerce"
            ).astype("Int64")
        merge_key = ["event_sample_original", "event_code"]
        _validate_unique_keys(retained_meta, merge_key, "Retained epoch metadata")
        metadata = metadata.merge(
            retained_meta,
            on=merge_key,
            how="left",
            suffixes=("", "_metadata"),
            validate="one_to_one",
        )
    else:
        merge_key = ["event_sample", "event_code"]
        _validate_unique_keys(retained_meta, merge_key, "Retained epoch metadata")
        metadata = metadata.merge(
            retained_meta,
            on=merge_key,
            how="left",
            suffixes=("", "_metadata"),
            validate="one_to_one",
        )
    participants = {
        str(value).strip()
        for column in ("subject", "subject_id")
        if column in metadata
        for value in metadata[column].dropna().tolist()
        if str(value).strip()
    }
    parts = {
        str(value).strip()
        for value in metadata.get("recording_part", pd.Series(dtype=object)).dropna().tolist()
        if str(value).strip()
    }
    sessions = {
        normalise_session(value)
        for value in metadata.get("session", pd.Series(dtype=object)).dropna().tolist()
    }
    if len(participants) != 1 or len(parts) != 1 or len(sessions) > 1:
        raise ValueError(
            "Epoch metadata must identify exactly one participant/recording part/session before binding."
        )
    metadata = bind_epoch_metadata_rows(
        metadata,
        participant=next(iter(participants)),
        session=next(iter(sessions), "session2"),
        recording_part=next(iter(parts)),
    )
    return mne.Epochs(
        raw,
        event_array,
        event_id=event_id,
        tmin=settings["tmin"],
        tmax=settings["tmax"],
        baseline=settings["baseline"],
        metadata=metadata,
        preload=runtime["preload"],
        reject_by_annotation=False,
        verbose=False,
    )


def _epochable_event_mapping(event_mapping: pd.DataFrame) -> pd.DataFrame:
    """Return only event-mapping rows that should be passed to MNE epoching."""

    if event_mapping.empty:
        return event_mapping.copy()
    mapping = event_mapping.copy()
    if "mapping_status" in mapping:
        mapping = mapping[~mapping["mapping_status"].fillna("").astype(str).str.startswith("rejected_")].copy()
    mapping = mapping[mapping["event_sample"].notna() & mapping["event_code"].notna()].copy()
    return mapping.reset_index(drop=True)


def _validate_unique_keys(frame: pd.DataFrame, key_columns: Sequence[str], label: str) -> None:
    """Reject malformed or duplicate authoritative table keys before joins."""

    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing key column(s): {', '.join(missing)}.")
    if frame.empty:
        return
    if frame[list(key_columns)].isna().any(axis=None):
        raise ValueError(f"{label} contains missing key values.")
    if frame.duplicated(list(key_columns), keep=False).any():
        raise ValueError(f"{label} contains duplicate keys; mapping is ambiguous.")


def _format_identity_rows(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Format a bounded identity preview for fail-closed diagnostics."""

    values = []
    for row in frame[list(columns)].head(5).itertuples(index=False, name=None):
        values.append("/".join(str(value) for value in row))
    suffix = " ..." if len(frame) > 5 else ""
    return ", ".join(values) + suffix


def _strict_qc_boolean_series(series: pd.Series, label: str) -> pd.Series:
    """Return boolean QC decisions without truthifying arbitrary strings."""

    def convert(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised == "true":
                return True
            if normalised == "false":
                return False
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if int(value) in (0, 1):
                return bool(value)
        raise ValueError(f"Residual EEG QC contains an invalid {label} value: {value!r}.")

    return series.map(convert).astype(bool)


def _processing_log(
    subject: str,
    recording_part: str,
    applied_ica: Sequence[int],
    confirmed_bad: Sequence[str],
    reference_summary: pd.DataFrame,
    interpolation_table: pd.DataFrame,
    n_epochs: int,
    *,
    ica_applied: bool,
    ica_skip_reason: str,
    epoch_settings: Mapping[str, Any],
    memory_settings: Mapping[str, Any],
    review_provenance: Mapping[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"step": "ica_review", "detail": dict(review_provenance)},
            {"step": "apply_ica", "detail": f"ica_applied={ica_applied}; applied_components={list(applied_ica)}; skip_reason={ica_skip_reason}"},
            {
                "step": "reconstruct_acquisition_reference",
                "detail": {
                    key: reference_summary.iloc[0].get(key)
                    for key in (
                        "original_reference_channel",
                        "original_reference_evidence",
                        "cpz_reconstructed",
                        "cpz_reconstruction_stage",
                        "cpz_reconstruction_method",
                        "cpz_zero_valued_before_average_reference",
                        "eeg_channel_count_before_reference_reconstruction",
                        "eeg_channel_count_after_reference_reconstruction",
                        "operation_order",
                    )
                }
                if not reference_summary.empty
                else {},
            },
            {"step": "average_reference", "detail": reference_summary.iloc[0].to_dict() if not reference_summary.empty else {}},
            {"step": "interpolation", "detail": f"manual_confirmed_bad_channels={list(confirmed_bad)}; interpolated={interpolation_table.loc[interpolation_table['interpolated'].astype(bool), 'channel'].astype(str).tolist() if not interpolation_table.empty else []}"},
            {"step": "epoching", "detail": f"subject={subject}; recording_part={recording_part}; image_onset_epochs={n_epochs}; settings={dict(epoch_settings)}"},
            {"step": "memory", "detail": _serialisable_memory_settings(memory_settings)},
        ]
    )


def _write_gate3c_sidecar(
    output_file: Path,
    subject: str,
    recording: RecordingPart,
    config_path: str | Path | None,
    root: Path | None,
    input_files: Sequence[str | Path],
    warnings: Sequence[str],
    parameters: Mapping[str, Any],
) -> None:
    write_provenance_sidecar(
        output_file,
        pipeline_stage="gate3c_ica_cleaning_epoching",
        subject=subject,
        recording_part=recording.recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary={
            "scope": "apply_reviewed_ica_reference_interpolate_image_onset_epoch_qc",
            "task_crop_scope": "Gate 3C preprocesses and epochs independent experimental blocks from the filtered raw; Gate 3A raw QC remains full-recording.",
            "preprocessing_mode": "blockwise_epoching",
            "recognition_analysis": "not_run",
            "cuteness_analysis": "not_run",
            "group_analysis": "not_run",
            **dict(parameters),
        },
        warnings=warnings,
    )


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _nullable_int(value: Any) -> Any:
    return pd.to_numeric(pd.Series([value]), errors="coerce").astype("Int64").iloc[0]
