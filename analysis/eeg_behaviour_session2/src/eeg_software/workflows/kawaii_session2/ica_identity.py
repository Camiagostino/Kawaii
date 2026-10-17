from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .identity import (
    SourceFileIdentity,
    identify_source_file,
    source_file_identity_from_primitive,
)
from .resolved_config import canonical_json, canonical_sha256
from .resolved_snapshot import ResolvedRunSnapshotReference
from .study_definition import KawaiiSession2StudyDefinition


ICA_FIT_SELECTION_IDENTITY_SCHEMA_VERSION = "kawaii_ica_fit_selection_identity_v1"
ICA_FIT_CONTEXT_IDENTITY_SCHEMA_VERSION = "kawaii_ica_fit_context_identity_v1"
ICA_DECOMPOSITION_IDENTITY_SCHEMA_VERSION = "kawaii_ica_decomposition_identity_v2"
ICA_SOLUTION_IDENTITY_SCHEMA_VERSION = "kawaii_authoritative_ica_solution_identity_v1"
ICA_FIT_ARTIFACT_IDENTITY_SCHEMA_VERSION = "kawaii_ica_fit_artifact_identity_v1"
ICA_FIT_IDENTITY_FILENAME = "ica_fit_identity.json"

ICA_AUTHORITATIVE_FINALITY = "fit_complete_pending_review"
ICA_VALIDATION_ONLY_FINALITY = "validation_only_fit"

_SHA256_PREFIX = "sha256:"


class ICAFitIdentityError(ValueError):
    """Raised when an ICA fit identity is absent, malformed, or inconsistent."""


@dataclass(frozen=True)
class ICAFitWindowIdentity:
    """One immutable source-coordinate window supplied to an ICA fit."""

    window_id: int
    start_sample: int
    end_sample: int
    start_trial: int | None = None
    end_trial: int | None = None
    block: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.window_id, "window_id"),
            (self.start_sample, "start_sample"),
            (self.end_sample, "end_sample"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ICAFitIdentityError(f"ICA fit-window {label} must be an integer.")
        if self.window_id < 1:
            raise ICAFitIdentityError("ICA fit-window window_id must be positive.")
        if self.end_sample < self.start_sample:
            raise ICAFitIdentityError("ICA fit-window end_sample precedes start_sample.")
        for value, label in (
            (self.start_trial, "start_trial"),
            (self.end_trial, "end_trial"),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ICAFitIdentityError(f"ICA fit-window {label} must be an integer or null.")
        if (self.start_trial is None) != (self.end_trial is None):
            raise ICAFitIdentityError("ICA fit-window trial bounds must both be present or both be null.")
        if self.start_trial is not None and self.end_trial < self.start_trial:
            raise ICAFitIdentityError("ICA fit-window end_trial precedes start_trial.")
        if self.block is not None and (not isinstance(self.block, str) or not self.block.strip()):
            raise ICAFitIdentityError("ICA fit-window block must be a non-empty string or null.")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_trial": self.start_trial,
            "end_trial": self.end_trial,
            "block": self.block,
        }


@dataclass(frozen=True)
class ICAFitSelectionIdentity:
    """Exact crop/subset selection and annotation state supplied to ICA.fit."""

    mode: str
    source_first_sample: int
    source_last_sample: int
    windows: tuple[ICAFitWindowIdentity, ...]
    fit_first_sample: int
    fit_n_times: int
    reject_by_annotation: bool
    annotations_fingerprint: str
    fingerprint: str
    schema_version: str = ICA_FIT_SELECTION_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ICA_FIT_SELECTION_IDENTITY_SCHEMA_VERSION:
            raise ICAFitIdentityError(f"Unsupported ICA fit-selection schema: {self.schema_version!r}.")
        _require_token(self.mode, "ICA fit-selection mode")
        for value, label in (
            (self.source_first_sample, "source_first_sample"),
            (self.source_last_sample, "source_last_sample"),
            (self.fit_first_sample, "fit_first_sample"),
            (self.fit_n_times, "fit_n_times"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ICAFitIdentityError(f"ICA fit-selection {label} must be an integer.")
        if self.source_last_sample < self.source_first_sample:
            raise ICAFitIdentityError("ICA fit-selection source sample bounds are invalid.")
        if self.fit_n_times <= 0:
            raise ICAFitIdentityError("ICA fit-selection fit_n_times must be positive.")
        if not isinstance(self.windows, tuple) or not self.windows:
            raise ICAFitIdentityError("ICA fit selection requires at least one source window.")
        if any(not isinstance(window, ICAFitWindowIdentity) for window in self.windows):
            raise ICAFitIdentityError("ICA fit selection contains a malformed window.")
        window_ids = [window.window_id for window in self.windows]
        if window_ids != list(range(1, len(self.windows) + 1)):
            raise ICAFitIdentityError("ICA fit-selection windows must have contiguous ordered IDs.")
        if any(
            window.start_sample < self.source_first_sample
            or window.end_sample > self.source_last_sample
            for window in self.windows
        ):
            raise ICAFitIdentityError("ICA fit-selection window falls outside source sample bounds.")
        if not isinstance(self.reject_by_annotation, bool):
            raise ICAFitIdentityError("ICA fit-selection reject_by_annotation must be boolean.")
        _require_sha256(self.annotations_fingerprint, "ICA annotations fingerprint")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ICAFitIdentityError("ICA fit-selection fingerprint is inconsistent.")

    @property
    def spec_fingerprint(self) -> str:
        """Identity of the source-coordinate selection, excluding realized Raw details."""

        return canonical_sha256(self.spec_payload())

    def spec_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "source_first_sample": self.source_first_sample,
            "source_last_sample": self.source_last_sample,
            "windows": [window.to_primitive() for window in self.windows],
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            **self.spec_payload(),
            "fit_first_sample": self.fit_first_sample,
            "fit_n_times": self.fit_n_times,
            "reject_by_annotation": self.reject_by_annotation,
            "annotations_fingerprint": self.annotations_fingerprint,
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "spec_fingerprint": self.spec_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ICAFitContextIdentity:
    """Content-bound identity of every material input to one authoritative ICA fit."""

    participant: str
    session: str
    recording_part: str
    input_eeg: SourceFileIdentity
    fit_data_content_sha256: str
    fit_data_shape: tuple[int, int]
    fit_data_dtype: str
    fit_sampling_frequency_hz: float
    selection: ICAFitSelectionIdentity
    input_channel_names: tuple[str, ...]
    input_channel_types: tuple[str, ...]
    bad_channel_names: tuple[str, ...]
    fit_channel_names: tuple[str, ...]
    filter_parameters_json: str
    estimated_effective_rank: int
    algorithm_parameters_json: str
    reference_state_json: str
    resolved_config_fingerprint: str
    scientific_fingerprint: str
    resolved_snapshot_json: str
    validation_only: bool
    finality: str
    fingerprint: str
    schema_version: str = ICA_FIT_CONTEXT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ICA_FIT_CONTEXT_IDENTITY_SCHEMA_VERSION:
            raise ICAFitIdentityError(f"Unsupported ICA fit-context schema: {self.schema_version!r}.")
        _require_token(self.participant, "ICA participant")
        if self.session != "session2":
            raise ICAFitIdentityError(f"ICA session must be 'session2', got {self.session!r}.")
        _require_token(self.recording_part, "ICA recording part")
        if not isinstance(self.input_eeg, SourceFileIdentity) or self.input_eeg.role != "ica_filtered_raw_input":
            raise ICAFitIdentityError("ICA context input_eeg must have role 'ica_filtered_raw_input'.")
        _require_sha256(self.fit_data_content_sha256, "ICA fit-data content digest")
        if (
            not isinstance(self.fit_data_shape, tuple)
            or len(self.fit_data_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.fit_data_shape)
        ):
            raise ICAFitIdentityError("ICA fit-data shape must contain two positive integers.")
        _require_token(self.fit_data_dtype, "ICA fit-data dtype")
        if not isinstance(self.fit_sampling_frequency_hz, (int, float)) or not np.isfinite(self.fit_sampling_frequency_hz) or self.fit_sampling_frequency_hz <= 0:
            raise ICAFitIdentityError("ICA fit sampling frequency must be positive and finite.")
        if not isinstance(self.selection, ICAFitSelectionIdentity):
            raise ICAFitIdentityError("ICA context fit selection is malformed.")
        for values, label in (
            (self.input_channel_names, "input_channel_names"),
            (self.input_channel_types, "input_channel_types"),
            (self.bad_channel_names, "bad_channel_names"),
            (self.fit_channel_names, "fit_channel_names"),
        ):
            _require_token_tuple(values, f"ICA {label}", allow_empty=label == "bad_channel_names")
        if len(self.input_channel_names) != len(self.input_channel_types):
            raise ICAFitIdentityError("ICA input channel names/types have different lengths.")
        if len(set(self.input_channel_names)) != len(self.input_channel_names):
            raise ICAFitIdentityError("ICA input channel inventory contains duplicates.")
        if len(set(self.fit_channel_names)) != len(self.fit_channel_names):
            raise ICAFitIdentityError("ICA fit channel inventory contains duplicates.")
        if any(channel not in self.input_channel_names for channel in self.fit_channel_names):
            raise ICAFitIdentityError("ICA fit channel is absent from the input channel inventory.")
        if any(channel not in self.input_channel_names for channel in self.bad_channel_names):
            raise ICAFitIdentityError("ICA bad channel is absent from the input channel inventory.")
        if self.fit_data_shape[0] != len(self.fit_channel_names):
            raise ICAFitIdentityError("ICA fit-data channel dimension does not match ordered fit channels.")
        if self.fit_data_shape[1] != self.selection.fit_n_times:
            raise ICAFitIdentityError("ICA fit-data sample dimension does not match the fit selection.")
        if isinstance(self.estimated_effective_rank, bool) or not isinstance(self.estimated_effective_rank, int) or self.estimated_effective_rank <= 0:
            raise ICAFitIdentityError("ICA estimated effective rank must be a positive integer.")
        if self.estimated_effective_rank > len(self.fit_channel_names):
            raise ICAFitIdentityError("ICA estimated effective rank exceeds the fit-channel count.")
        filter_parameters = _validated_json_mapping(self.filter_parameters_json, "ICA filter parameters")
        algorithm_parameters = _validated_json_mapping(self.algorithm_parameters_json, "ICA algorithm parameters")
        _validated_json_mapping(self.reference_state_json, "ICA reference state")
        snapshot = _validated_json_value(self.resolved_snapshot_json, "ICA resolved snapshot")
        if snapshot is not None and not isinstance(snapshot, dict):
            raise ICAFitIdentityError("ICA resolved snapshot must be a mapping or null.")
        _require_exact_keys(
            filter_parameters,
            {"operation", "requested_highpass_hz", "requested_lowpass_hz", "effective_highpass_hz", "effective_lowpass_hz", "picks"},
            "ICA filter parameters",
        )
        _require_exact_keys(
            algorithm_parameters,
            {
                "method", "n_components_strategy", "n_components_requested",
                "n_components_effective", "decimation", "random_seed", "max_iter",
                "fit_params", "reject_by_annotation", "mne_version", "numpy_version",
            },
            "ICA algorithm parameters",
        )
        if algorithm_parameters["n_components_requested"] != self.estimated_effective_rank:
            raise ICAFitIdentityError("ICA requested components do not match estimated effective rank.")
        if algorithm_parameters["n_components_effective"] <= 0:
            raise ICAFitIdentityError("ICA effective component count must be positive.")
        if tuple(filter_parameters["picks"]) != self.fit_channel_names:
            raise ICAFitIdentityError("ICA filter picks do not match ordered fit channels.")
        _require_sha256(self.resolved_config_fingerprint, "ICA resolved-config fingerprint")
        _require_sha256(self.scientific_fingerprint, "ICA scientific fingerprint")
        if not isinstance(self.validation_only, bool):
            raise ICAFitIdentityError("ICA validation_only must be boolean.")
        expected_finality = ICA_VALIDATION_ONLY_FINALITY if self.validation_only else ICA_AUTHORITATIVE_FINALITY
        if self.finality != expected_finality:
            raise ICAFitIdentityError(
                f"ICA finality {self.finality!r} is inconsistent with validation_only={self.validation_only}."
            )
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ICAFitIdentityError("ICA fit-context fingerprint is inconsistent.")

    @property
    def filter_parameters(self) -> dict[str, Any]:
        return json.loads(self.filter_parameters_json)

    @property
    def algorithm_parameters(self) -> dict[str, Any]:
        return json.loads(self.algorithm_parameters_json)

    @property
    def reference_state(self) -> dict[str, Any]:
        return json.loads(self.reference_state_json)

    @property
    def resolved_snapshot(self) -> dict[str, Any] | None:
        return json.loads(self.resolved_snapshot_json)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": {
                "participant": self.participant,
                "session": self.session,
                "recording_part": self.recording_part,
            },
            "input_eeg": self.input_eeg.identity_payload(),
            "fit_data": {
                "content_sha256": self.fit_data_content_sha256,
                "shape": list(self.fit_data_shape),
                "dtype": self.fit_data_dtype,
                "sampling_frequency_hz": self.fit_sampling_frequency_hz,
            },
            "selection": self.selection.identity_payload(),
            "channels": {
                "input_names": list(self.input_channel_names),
                "input_types": list(self.input_channel_types),
                "bad_names": list(self.bad_channel_names),
                "fit_names": list(self.fit_channel_names),
            },
            "filter": self.filter_parameters,
            "estimated_effective_rank": self.estimated_effective_rank,
            "ica": self.algorithm_parameters,
            "reference_state": self.reference_state,
            "configuration": {
                "resolved_config_fingerprint": self.resolved_config_fingerprint,
                "scientific_fingerprint": self.scientific_fingerprint,
                "resolved_snapshot": self.resolved_snapshot,
            },
            "authority": {
                "validation_only": self.validation_only,
                "finality": self.finality,
            },
        }

    def to_primitive(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["input_eeg"] = self.input_eeg.to_primitive()
        value["selection"] = self.selection.to_primitive()
        value["fingerprint"] = self.fingerprint
        return value


@dataclass(frozen=True)
class ICAFitArtifactIdentity:
    """Persisted authoritative bridge joining fit context and decomposition."""

    fit_context: ICAFitContextIdentity
    decomposition_fingerprint: str
    solution_fingerprint: str
    solution_file: SourceFileIdentity
    fingerprint: str
    schema_version: str = ICA_FIT_ARTIFACT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ICA_FIT_ARTIFACT_IDENTITY_SCHEMA_VERSION:
            raise ICAFitIdentityError(f"Unsupported ICA fit-artifact schema: {self.schema_version!r}.")
        if not isinstance(self.fit_context, ICAFitContextIdentity):
            raise ICAFitIdentityError("ICA fit artifact contains a malformed fit context.")
        _require_sha256(self.decomposition_fingerprint, "ICA decomposition fingerprint")
        expected_solution = authoritative_ica_solution_identity(
            self.fit_context, self.decomposition_fingerprint
        )
        if self.solution_fingerprint != expected_solution:
            raise ICAFitIdentityError("ICA solution fingerprint is inconsistent with context/decomposition.")
        if not isinstance(self.solution_file, SourceFileIdentity) or self.solution_file.role != "ica_solution_fif":
            raise ICAFitIdentityError("ICA solution file identity must have role 'ica_solution_fif'.")
        if self.fingerprint != canonical_sha256(self.identity_payload()):
            raise ICAFitIdentityError("ICA fit-artifact fingerprint is inconsistent.")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fit_context": self.fit_context.identity_payload(),
            "fit_context_fingerprint": self.fit_context.fingerprint,
            "decomposition_fingerprint": self.decomposition_fingerprint,
            "solution_fingerprint": self.solution_fingerprint,
            "solution_file": self.solution_file.identity_payload(),
        }

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fit_context": self.fit_context.to_primitive(),
            "fit_context_fingerprint": self.fit_context.fingerprint,
            "decomposition_fingerprint": self.decomposition_fingerprint,
            "solution_fingerprint": self.solution_fingerprint,
            "solution_file": self.solution_file.to_primitive(),
            "fingerprint": self.fingerprint,
        }


def build_ica_fit_selection_identity(
    *,
    mode: str,
    source_first_sample: int,
    source_last_sample: int,
    windows: Sequence[Mapping[str, Any] | ICAFitWindowIdentity],
    fit_first_sample: int,
    fit_n_times: int,
    annotations_fingerprint: str,
    reject_by_annotation: bool = True,
) -> ICAFitSelectionIdentity:
    parsed_windows = tuple(
        window
        if isinstance(window, ICAFitWindowIdentity)
        else ICAFitWindowIdentity(
            window_id=index,
            start_sample=_required_int(window.get("start_sample"), "window start_sample"),
            end_sample=_required_int(window.get("end_sample"), "window end_sample"),
            start_trial=_optional_int(window.get("start_trial"), "window start_trial"),
            end_trial=_optional_int(window.get("end_trial"), "window end_trial"),
            block=_optional_token(window.get("block")),
        )
        for index, window in enumerate(windows, start=1)
    )
    provisional = {
        "schema_version": ICA_FIT_SELECTION_IDENTITY_SCHEMA_VERSION,
        "mode": str(mode),
        "source_first_sample": int(source_first_sample),
        "source_last_sample": int(source_last_sample),
        "windows": [window.to_primitive() for window in parsed_windows],
        "fit_first_sample": int(fit_first_sample),
        "fit_n_times": int(fit_n_times),
        "reject_by_annotation": bool(reject_by_annotation),
        "annotations_fingerprint": annotations_fingerprint,
    }
    return ICAFitSelectionIdentity(
        mode=str(mode),
        source_first_sample=int(source_first_sample),
        source_last_sample=int(source_last_sample),
        windows=parsed_windows,
        fit_first_sample=int(fit_first_sample),
        fit_n_times=int(fit_n_times),
        reject_by_annotation=bool(reject_by_annotation),
        annotations_fingerprint=annotations_fingerprint,
        fingerprint=canonical_sha256(provisional),
    )


def build_ica_fit_context_identity(
    *,
    participant: str,
    recording_part: str,
    input_eeg: SourceFileIdentity,
    fit_data_content_sha256: str,
    fit_data_shape: Sequence[int],
    fit_data_dtype: str,
    fit_sampling_frequency_hz: float,
    selection: ICAFitSelectionIdentity,
    input_channel_names: Sequence[str],
    input_channel_types: Sequence[str],
    bad_channel_names: Sequence[str],
    fit_channel_names: Sequence[str],
    filter_parameters: Mapping[str, Any],
    estimated_effective_rank: int,
    algorithm_parameters: Mapping[str, Any],
    reference_state: Mapping[str, Any],
    resolved_config_fingerprint: str,
    scientific_fingerprint: str,
    resolved_snapshot: ResolvedRunSnapshotReference | Mapping[str, Any] | None = None,
    validation_only: bool = False,
    session: str = "session2",
) -> ICAFitContextIdentity:
    filter_json = _canonical_mapping_json(filter_parameters, "ICA filter parameters")
    algorithm_json = _canonical_mapping_json(algorithm_parameters, "ICA algorithm parameters")
    reference_json = _canonical_mapping_json(reference_state, "ICA reference state")
    snapshot_json = canonical_json(_resolved_snapshot_primitive(resolved_snapshot))
    finality = ICA_VALIDATION_ONLY_FINALITY if validation_only else ICA_AUTHORITATIVE_FINALITY
    values = {
        "participant": str(participant),
        "session": str(session),
        "recording_part": str(recording_part),
        "input_eeg": input_eeg,
        "fit_data_content_sha256": fit_data_content_sha256,
        "fit_data_shape": tuple(int(value) for value in fit_data_shape),
        "fit_data_dtype": str(fit_data_dtype),
        "fit_sampling_frequency_hz": float(fit_sampling_frequency_hz),
        "selection": selection,
        "input_channel_names": tuple(str(value) for value in input_channel_names),
        "input_channel_types": tuple(str(value) for value in input_channel_types),
        "bad_channel_names": tuple(str(value) for value in bad_channel_names),
        "fit_channel_names": tuple(str(value) for value in fit_channel_names),
        "filter_parameters_json": filter_json,
        "estimated_effective_rank": int(estimated_effective_rank),
        "algorithm_parameters_json": algorithm_json,
        "reference_state_json": reference_json,
        "resolved_config_fingerprint": resolved_config_fingerprint,
        "scientific_fingerprint": scientific_fingerprint,
        "resolved_snapshot_json": snapshot_json,
        "validation_only": bool(validation_only),
        "finality": finality,
    }
    provisional = ICAFitContextIdentity(
        **values,
        fingerprint=_context_fingerprint_from_values(values),
    )
    return provisional


def ica_fit_data_content_sha256(raw, fit_channel_names: Sequence[str], *, chunk_samples: int = 100_000) -> str:
    """Hash the exact ordered float64 EEG samples supplied to ICA.fit."""

    if isinstance(chunk_samples, bool) or not isinstance(chunk_samples, int) or chunk_samples <= 0:
        raise ICAFitIdentityError("ICA fit-data hash chunk_samples must be a positive integer.")
    channels = tuple(str(channel) for channel in fit_channel_names)
    if not channels:
        raise ICAFitIdentityError("ICA fit-data identity requires at least one channel.")
    if any(channel not in raw.ch_names for channel in channels):
        raise ICAFitIdentityError("ICA fit-data identity includes a channel absent from Raw.")
    digest = hashlib.sha256()
    n_times = int(raw.n_times)
    # Channel-major traversal matches a contiguous (channels, samples) array
    # while keeping peak memory bounded and the digest independent of chunking.
    for channel in channels:
        for start in range(0, n_times, chunk_samples):
            data = np.asarray(
                raw.get_data(picks=[channel], start=start, stop=min(n_times, start + chunk_samples)),
                dtype="<f8",
                order="C",
            )
            digest.update(np.ascontiguousarray(data).tobytes())
    return _SHA256_PREFIX + digest.hexdigest()


def ica_annotations_fingerprint(raw) -> str:
    """Fingerprint the annotation state used by reject_by_annotation."""

    annotations = getattr(raw, "annotations", ())
    rows = []
    for onset, duration, description in zip(
        getattr(annotations, "onset", ()),
        getattr(annotations, "duration", ()),
        getattr(annotations, "description", ()),
    ):
        rows.append({
            "onset": float(onset),
            "duration": float(duration),
            "description": str(description),
        })
    return canonical_sha256(rows)


def ica_reference_state(raw) -> dict[str, Any]:
    """Return deterministic material reference/projection state from the fit Raw."""

    projectors = []
    for projection in raw.info.get("projs", []) or []:
        data = projection.get("data", {}) or {}
        array = np.asarray(data.get("data", []), dtype="<f8")
        projectors.append({
            "active": bool(projection.get("active", False)),
            "kind": int(projection.get("kind", 0) or 0),
            "description": str(projection.get("desc", "") or ""),
            "column_names": [str(value) for value in data.get("col_names", []) or []],
            "nrow": int(data.get("nrow", array.shape[0] if array.ndim else 0) or 0),
            "ncol": int(data.get("ncol", array.shape[-1] if array.ndim else 0) or 0),
            "data_sha256": _SHA256_PREFIX + hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        })
    return {
        "custom_ref_applied": int(raw.info.get("custom_ref_applied", 0) or 0),
        "projectors": projectors,
    }


def ica_decomposition_identity(ica) -> str:
    """Return a save/load-stable identity of the fitted decomposition only."""

    arrays: dict[str, Any] = {}
    for attribute in ("unmixing_matrix_", "mixing_matrix_", "pca_components_", "pre_whitener_"):
        value = getattr(ica, attribute, None)
        if value is None:
            arrays[attribute] = None
            continue
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        arrays[attribute] = {
            "shape": list(array.shape),
            "sha256": _SHA256_PREFIX + hashlib.sha256(array.tobytes()).hexdigest(),
        }
    payload = {
        "schema_version": ICA_DECOMPOSITION_IDENTITY_SCHEMA_VERSION,
        "method": str(getattr(ica, "method", "")),
        "channel_names": [str(value) for value in (getattr(ica, "ch_names", []) or [])],
        "n_components_effective": int(getattr(ica, "n_components_", 0) or 0),
        "arrays": arrays,
    }
    return canonical_sha256(payload)


def _matches_canonical_fif_decomposition(ica, solution_path: str | Path) -> bool:
    """Verify a loaded ICA against an independent read of the exact FIF.

    MNE persists the unmixing matrix but reconstructs ``mixing_matrix_`` as its
    pseudoinverse on read.  Existing v1 fit artifacts were fingerprinted from
    the fitted in-memory object, so that derived matrix can differ from a read
    by machine-roundoff even though every persisted decomposition value is
    identical.  This compatibility check deliberately keeps the persisted
    arrays exact and permits only that derived roundoff.
    """

    try:
        import mne

        canonical = mne.preprocessing.read_ica(solution_path, verbose=False)
    except Exception:
        return False

    if str(getattr(ica, "method", "")) != str(getattr(canonical, "method", "")):
        return False
    if list(getattr(ica, "ch_names", []) or []) != list(getattr(canonical, "ch_names", []) or []):
        return False
    if int(getattr(ica, "n_components_", 0) or 0) != int(
        getattr(canonical, "n_components_", 0) or 0
    ):
        return False

    for attribute in ("unmixing_matrix_", "pca_components_", "pre_whitener_"):
        current = getattr(ica, attribute, None)
        expected = getattr(canonical, attribute, None)
        if (current is None) != (expected is None):
            return False
        if current is not None:
            current_array = np.ascontiguousarray(np.asarray(current, dtype="<f8"))
            expected_array = np.ascontiguousarray(np.asarray(expected, dtype="<f8"))
            if current_array.shape != expected_array.shape or not np.array_equal(
                current_array, expected_array
            ):
                return False

    current_mixing = getattr(ica, "mixing_matrix_", None)
    expected_mixing = getattr(canonical, "mixing_matrix_", None)
    if (current_mixing is None) != (expected_mixing is None):
        return False
    if current_mixing is None:
        return True
    current_array = np.asarray(current_mixing, dtype=np.float64)
    expected_array = np.asarray(expected_mixing, dtype=np.float64)
    if current_array.shape != expected_array.shape:
        return False
    if not np.isfinite(current_array).all() or not np.isfinite(expected_array).all():
        return False
    scale = max(1.0, float(np.max(np.abs(expected_array), initial=0.0)))
    epsilon = np.finfo(np.float64).eps
    return bool(
        np.allclose(
            current_array,
            expected_array,
            rtol=64.0 * epsilon,
            atol=64.0 * epsilon * scale,
        )
    )


def authoritative_ica_solution_identity(
    fit_context: ICAFitContextIdentity,
    decomposition_fingerprint: str,
) -> str:
    """Bind one decomposition to the exact authoritative fit context."""

    if not isinstance(fit_context, ICAFitContextIdentity):
        raise ICAFitIdentityError("Authoritative ICA solution identity requires ICAFitContextIdentity.")
    _require_sha256(decomposition_fingerprint, "ICA decomposition fingerprint")
    return canonical_sha256({
        "schema_version": ICA_SOLUTION_IDENTITY_SCHEMA_VERSION,
        "fit_context_fingerprint": fit_context.fingerprint,
        "decomposition_fingerprint": decomposition_fingerprint,
    })


def build_ica_fit_artifact_identity(
    *,
    fit_context: ICAFitContextIdentity,
    ica,
    solution_path: str | Path,
) -> ICAFitArtifactIdentity:
    solution_file = identify_source_file(
        solution_path,
        "ica_solution_fif",
        logical_path=Path(solution_path).name,
    )
    decomposition = ica_decomposition_identity(ica)
    solution = authoritative_ica_solution_identity(fit_context, decomposition)
    payload = {
        "schema_version": ICA_FIT_ARTIFACT_IDENTITY_SCHEMA_VERSION,
        "fit_context": fit_context.identity_payload(),
        "fit_context_fingerprint": fit_context.fingerprint,
        "decomposition_fingerprint": decomposition,
        "solution_fingerprint": solution,
        "solution_file": solution_file.identity_payload(),
    }
    return ICAFitArtifactIdentity(
        fit_context=fit_context,
        decomposition_fingerprint=decomposition,
        solution_fingerprint=solution,
        solution_file=solution_file,
        fingerprint=canonical_sha256(payload),
    )


def build_canonical_ica_fit_artifact_identity(
    *,
    fit_context: ICAFitContextIdentity,
    solution_path: str | Path,
) -> ICAFitArtifactIdentity:
    """Build an identity from MNE's canonical representation of a saved FIF."""

    import mne

    canonical_ica = mne.preprocessing.read_ica(solution_path, verbose=False)
    return build_ica_fit_artifact_identity(
        fit_context=fit_context,
        ica=canonical_ica,
        solution_path=solution_path,
    )


def ica_fit_identity_path(solution_path: str | Path) -> Path:
    return Path(solution_path).with_name(ICA_FIT_IDENTITY_FILENAME)


def write_ica_fit_artifact_identity(
    artifact: ICAFitArtifactIdentity,
    path: str | Path,
) -> Path:
    if not isinstance(artifact, ICAFitArtifactIdentity):
        raise ICAFitIdentityError("Cannot persist malformed ICA fit artifact identity.")
    target = Path(path)
    _atomic_write_text(target, canonical_json(artifact.to_primitive()) + "\n")
    validated = read_ica_fit_artifact_identity(target)
    if validated.fingerprint != artifact.fingerprint:
        raise ICAFitIdentityError("Persisted ICA fit identity failed round-trip validation.")
    return target


def read_ica_fit_artifact_identity(path: str | Path) -> ICAFitArtifactIdentity:
    identity_path = Path(path)
    if not identity_path.exists():
        raise ICAFitIdentityError(
            f"Legacy/unverified ICA solution lacks required fit-context identity: {identity_path}. "
            "Regenerate ICA before authoritative review binding or use decomposition inspection only."
        )
    try:
        value = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ICAFitIdentityError(f"Could not read ICA fit identity {identity_path}: {error}") from error
    _require_exact_keys(
        value,
        {
            "schema_version", "fit_context", "fit_context_fingerprint",
            "decomposition_fingerprint", "solution_fingerprint", "solution_file", "fingerprint",
        },
        "ICA fit artifact identity",
    )
    context = _context_from_primitive(value["fit_context"])
    if value["fit_context_fingerprint"] != context.fingerprint:
        raise ICAFitIdentityError("Persisted ICA fit-context reference is inconsistent.")
    solution_file = source_file_identity_from_primitive(value["solution_file"])
    return ICAFitArtifactIdentity(
        fit_context=context,
        decomposition_fingerprint=value["decomposition_fingerprint"],
        solution_fingerprint=value["solution_fingerprint"],
        solution_file=solution_file,
        fingerprint=value["fingerprint"],
        schema_version=value["schema_version"],
    )


def validate_ica_solution_file_binding(
    identity_path: str | Path,
    *,
    solution_path: str | Path,
    expected_participant: str,
    expected_recording_part: str,
    expected_resolved_config_fingerprint: str | None = None,
) -> ICAFitArtifactIdentity:
    """Validate the persisted authoritative identity against ICA solution bytes.

    This is the publication-boundary subset of full ICA validation.  Gate 3C
    still loads the decomposition and performs the complete context/review
    validation before applying it.
    """

    artifact = read_ica_fit_artifact_identity(identity_path)
    context = artifact.fit_context
    if context.participant != expected_participant:
        raise ICAFitIdentityError(
            f"ICA participant mismatch: expected {expected_participant!r}, found {context.participant!r}."
        )
    if context.recording_part != expected_recording_part:
        raise ICAFitIdentityError(
            f"ICA recording-part mismatch: expected {expected_recording_part!r}, found {context.recording_part!r}."
        )
    if context.validation_only or context.finality != ICA_AUTHORITATIVE_FINALITY:
        raise ICAFitIdentityError("Validation-only/non-authoritative ICA context cannot be published.")
    _expect_equal(
        context.resolved_config_fingerprint,
        expected_resolved_config_fingerprint,
        "resolved-config fingerprint",
    )
    current_solution_file = identify_source_file(
        solution_path,
        "ica_solution_fif",
        logical_path=artifact.solution_file.logical_path,
    )
    if current_solution_file.identity_payload() != artifact.solution_file.identity_payload():
        raise ICAFitIdentityError("ICA solution FIF content does not match the persisted identity.")
    return artifact


def validate_ica_fit_artifact_identity(
    path: str | Path,
    *,
    ica,
    solution_path: str | Path,
    expected_participant: str,
    expected_recording_part: str,
    expected_input_eeg: SourceFileIdentity | None = None,
    expected_resolved_config_fingerprint: str | None = None,
    expected_scientific_fingerprint: str | None = None,
    expected_input_channel_names: Sequence[str] | None = None,
    expected_input_channel_types: Sequence[str] | None = None,
    expected_bad_channel_names: Sequence[str] | None = None,
    expected_fit_channel_names: Sequence[str] | None = None,
    expected_selection_spec_fingerprint: str | None = None,
    expected_filter_parameters: Mapping[str, Any] | None = None,
    expected_algorithm_parameters: Mapping[str, Any] | None = None,
    expected_reference_state: Mapping[str, Any] | None = None,
    require_authoritative: bool = True,
) -> ICAFitArtifactIdentity:
    """Validate persisted ICA context against current expected inputs and solution."""

    artifact = read_ica_fit_artifact_identity(path)
    context = artifact.fit_context
    if context.participant != expected_participant:
        raise ICAFitIdentityError(
            f"ICA participant mismatch: expected {expected_participant!r}, found {context.participant!r}."
        )
    if context.recording_part != expected_recording_part:
        raise ICAFitIdentityError(
            f"ICA recording-part mismatch: expected {expected_recording_part!r}, found {context.recording_part!r}."
        )
    if require_authoritative and (context.validation_only or context.finality != ICA_AUTHORITATIVE_FINALITY):
        raise ICAFitIdentityError("Validation-only/non-authoritative ICA context cannot bind authoritative review.")
    current_solution_file = identify_source_file(
        solution_path,
        "ica_solution_fif",
        logical_path=artifact.solution_file.logical_path,
    )
    if current_solution_file.identity_payload() != artifact.solution_file.identity_payload():
        raise ICAFitIdentityError("ICA solution FIF content does not match the persisted identity.")
    current_decomposition = ica_decomposition_identity(ica)
    if current_decomposition != artifact.decomposition_fingerprint:
        if not _matches_canonical_fif_decomposition(ica, solution_path):
            raise ICAFitIdentityError("Loaded ICA decomposition does not match the persisted identity.")
        # Preserve the reviewed legacy solution ID.  The exact FIF identity
        # above anchors the immutable solution bytes; the independent reload
        # proves that the supplied decomposition is exactly those bytes, apart
        # from MNE's derived pseudoinverse roundoff.
        current_decomposition = artifact.decomposition_fingerprint
    current_solution = authoritative_ica_solution_identity(context, current_decomposition)
    if current_solution != artifact.solution_fingerprint:
        raise ICAFitIdentityError("Loaded ICA authoritative solution fingerprint is inconsistent.")
    if expected_input_eeg is not None and expected_input_eeg.identity_payload() != context.input_eeg.identity_payload():
        raise ICAFitIdentityError("Current ICA input EEG content identity does not match the fit context.")
    _expect_equal(context.resolved_config_fingerprint, expected_resolved_config_fingerprint, "resolved-config fingerprint")
    _expect_equal(context.scientific_fingerprint, expected_scientific_fingerprint, "scientific fingerprint")
    _expect_tuple(context.input_channel_names, expected_input_channel_names, "input channel inventory/order")
    _expect_tuple(context.input_channel_types, expected_input_channel_types, "input channel types/order")
    _expect_tuple(context.bad_channel_names, expected_bad_channel_names, "bad-channel state/order")
    _expect_tuple(context.fit_channel_names, expected_fit_channel_names, "fit channel inventory/order")
    _expect_equal(context.selection.spec_fingerprint, expected_selection_spec_fingerprint, "fit-selection fingerprint")
    _expect_mapping_subset(context.filter_parameters, expected_filter_parameters, "filter parameters")
    _expect_mapping_subset(context.algorithm_parameters, expected_algorithm_parameters, "ICA algorithm parameters")
    if expected_reference_state is not None and canonical_json(context.reference_state) != canonical_json(_json_safe(expected_reference_state)):
        raise ICAFitIdentityError("ICA reference state does not match the expected current state.")
    return artifact


def resolved_ica_config_fingerprints(
    config: Mapping[str, Any],
    resolved_snapshot: ResolvedRunSnapshotReference | None,
) -> tuple[str, str]:
    """Return the same WS2 configuration/science identities used by stage contracts."""

    if resolved_snapshot is not None:
        return (
            resolved_snapshot.resolved_config_fingerprint,
            resolved_snapshot.scientific_fingerprint,
        )
    study = KawaiiSession2StudyDefinition.from_config(config).to_primitive()
    return (
        canonical_sha256({"effective_config": dict(config), "study_definition": study}),
        canonical_sha256(study),
    )


def _context_fingerprint_from_values(values: Mapping[str, Any]) -> str:
    temporary = object.__new__(ICAFitContextIdentity)
    for key, value in values.items():
        object.__setattr__(temporary, key, value)
    object.__setattr__(temporary, "schema_version", ICA_FIT_CONTEXT_IDENTITY_SCHEMA_VERSION)
    return canonical_sha256(ICAFitContextIdentity.identity_payload(temporary))


def _context_from_primitive(value: Any) -> ICAFitContextIdentity:
    _require_exact_keys(
        value,
        {
            "schema_version", "scope", "input_eeg", "fit_data", "selection", "channels",
            "filter", "estimated_effective_rank", "ica", "reference_state", "configuration",
            "authority", "fingerprint",
        },
        "ICA fit context",
    )
    scope = value["scope"]
    fit_data = value["fit_data"]
    channels = value["channels"]
    configuration = value["configuration"]
    authority = value["authority"]
    _require_exact_keys(scope, {"participant", "session", "recording_part"}, "ICA fit-context scope")
    _require_exact_keys(fit_data, {"content_sha256", "shape", "dtype", "sampling_frequency_hz"}, "ICA fit data")
    _require_exact_keys(channels, {"input_names", "input_types", "bad_names", "fit_names"}, "ICA channels")
    _require_exact_keys(configuration, {"resolved_config_fingerprint", "scientific_fingerprint", "resolved_snapshot"}, "ICA configuration")
    _require_exact_keys(authority, {"validation_only", "finality"}, "ICA authority")
    selection = _selection_from_primitive(value["selection"])
    context = ICAFitContextIdentity(
        participant=scope["participant"],
        session=scope["session"],
        recording_part=scope["recording_part"],
        input_eeg=source_file_identity_from_primitive(value["input_eeg"]),
        fit_data_content_sha256=fit_data["content_sha256"],
        fit_data_shape=tuple(fit_data["shape"]),
        fit_data_dtype=fit_data["dtype"],
        fit_sampling_frequency_hz=fit_data["sampling_frequency_hz"],
        selection=selection,
        input_channel_names=tuple(channels["input_names"]),
        input_channel_types=tuple(channels["input_types"]),
        bad_channel_names=tuple(channels["bad_names"]),
        fit_channel_names=tuple(channels["fit_names"]),
        filter_parameters_json=canonical_json(value["filter"]),
        estimated_effective_rank=value["estimated_effective_rank"],
        algorithm_parameters_json=canonical_json(value["ica"]),
        reference_state_json=canonical_json(value["reference_state"]),
        resolved_config_fingerprint=configuration["resolved_config_fingerprint"],
        scientific_fingerprint=configuration["scientific_fingerprint"],
        resolved_snapshot_json=canonical_json(configuration["resolved_snapshot"]),
        validation_only=authority["validation_only"],
        finality=authority["finality"],
        fingerprint=value["fingerprint"],
        schema_version=value["schema_version"],
    )
    return context


def _selection_from_primitive(value: Any) -> ICAFitSelectionIdentity:
    _require_exact_keys(
        value,
        {
            "schema_version", "mode", "source_first_sample", "source_last_sample", "windows",
            "fit_first_sample", "fit_n_times", "reject_by_annotation", "annotations_fingerprint",
            "spec_fingerprint", "fingerprint",
        },
        "ICA fit selection",
    )
    if not isinstance(value["windows"], list):
        raise ICAFitIdentityError("ICA fit-selection windows must be a list.")
    windows = []
    for row in value["windows"]:
        _require_exact_keys(
            row,
            {"window_id", "start_sample", "end_sample", "start_trial", "end_trial", "block"},
            "ICA fit window",
        )
        windows.append(ICAFitWindowIdentity(**row))
    selection = ICAFitSelectionIdentity(
        mode=value["mode"],
        source_first_sample=value["source_first_sample"],
        source_last_sample=value["source_last_sample"],
        windows=tuple(windows),
        fit_first_sample=value["fit_first_sample"],
        fit_n_times=value["fit_n_times"],
        reject_by_annotation=value["reject_by_annotation"],
        annotations_fingerprint=value["annotations_fingerprint"],
        fingerprint=value["fingerprint"],
        schema_version=value["schema_version"],
    )
    if value["spec_fingerprint"] != selection.spec_fingerprint:
        raise ICAFitIdentityError("ICA fit-selection spec fingerprint is inconsistent.")
    return selection


def _resolved_snapshot_primitive(
    value: ResolvedRunSnapshotReference | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, ResolvedRunSnapshotReference):
        return {
            "schema_version": value.schema_version,
            "snapshot_state": value.snapshot_state,
            "snapshot_content_sha256": value.snapshot_content_sha256,
            "source_config_sha256": value.source_config_sha256,
        }
    if not isinstance(value, Mapping):
        raise ICAFitIdentityError("ICA resolved snapshot must be a snapshot reference, mapping, or null.")
    return _json_safe(value)


def _canonical_mapping_json(value: Mapping[str, Any], label: str) -> str:
    if not isinstance(value, Mapping):
        raise ICAFitIdentityError(f"{label} must be a mapping.")
    return canonical_json(_json_safe(value))


def _validated_json_mapping(value: str, label: str) -> dict[str, Any]:
    parsed = _validated_json_value(value, label)
    if not isinstance(parsed, dict):
        raise ICAFitIdentityError(f"{label} must encode a mapping.")
    if canonical_json(parsed) != value:
        raise ICAFitIdentityError(f"{label} must use canonical JSON.")
    return parsed


def _validated_json_value(value: str, label: str) -> Any:
    if not isinstance(value, str):
        raise ICAFitIdentityError(f"{label} must be canonical JSON text.")
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ICAFitIdentityError(f"{label} contains malformed JSON.") from error


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ICAFitIdentityError(f"ICA identity contains unsupported value type {type(value).__name__}.")


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if expected is not None and actual != expected:
        raise ICAFitIdentityError(f"ICA {label} does not match the expected current value.")


def _expect_tuple(actual: tuple[str, ...], expected: Sequence[str] | None, label: str) -> None:
    if expected is not None and actual != tuple(str(value) for value in expected):
        raise ICAFitIdentityError(f"ICA {label} does not match the expected current value.")


def _expect_mapping_subset(actual: Mapping[str, Any], expected: Mapping[str, Any] | None, label: str) -> None:
    if expected is None:
        return
    safe_expected = _json_safe(expected)
    for key, expected_value in safe_expected.items():
        if key not in actual or not _json_subset_matches(actual[key], expected_value):
            raise ICAFitIdentityError(f"ICA {label} field {key!r} does not match the expected current value.")


def _json_subset_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, Mapping) and all(
            key in actual and _json_subset_matches(actual[key], value)
            for key, value in expected.items()
        )
    return canonical_json(actual) == canonical_json(expected)


def _require_token(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ICAFitIdentityError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_token_tuple(value: tuple[str, ...], label: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ICAFitIdentityError(f"{label} must be a{' possibly empty' if allow_empty else ' non-empty'} tuple.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ICAFitIdentityError(f"{label} contains an empty/non-string value.")


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX) or len(value) != 71:
        raise ICAFitIdentityError(f"{label} must be a prefixed SHA-256 digest.")
    try:
        int(value[len(_SHA256_PREFIX):], 16)
    except ValueError as error:
        raise ICAFitIdentityError(f"{label} must be a prefixed SHA-256 digest.") from error


def _required_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ICAFitIdentityError(f"ICA {label} must be an integer.")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ICAFitIdentityError(f"ICA {label} must be an integer.") from error
    if isinstance(value, float) and not value.is_integer():
        raise ICAFitIdentityError(f"ICA {label} must be an integer.")
    return converted


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    try:
        if bool(np.isnan(value)):
            return None
    except TypeError:
        pass
    return _required_int(value, label)


def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ICAFitIdentityError(f"{label} must be a mapping.")
    actual = set(value)
    if actual != expected:
        raise ICAFitIdentityError(
            f"{label} fields are invalid; missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}."
        )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except OSError as error:
        raise ICAFitIdentityError(f"Could not persist ICA fit identity {path}: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
