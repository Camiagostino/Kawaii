from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from eeg_software.project import read_yaml_config
from eeg_software.project import get_project_root

from .config import CONFIG_NAME
from .paths import Session2EEGPaths, resolve_session2_eeg_paths
from .study_definition import (
    DEFAULT_IMAGE_EVENT_ID,
    DEFAULT_RESIDUAL_EEG_QC_SEMANTICS,
    KawaiiSession2StudyDefinition,
)
from .subject_selection import resolve_subjects


RESOLVED_CONFIG_SCHEMA_VERSION = "kawaii_session2_resolved_config_v2"

# These values are currently implicit in consumer helpers.  Materialising them
# here makes the resolved representation complete without changing those
# consumers in this batch.
_CENTRAL_RESOLVED_DEFAULTS: dict[str, Any] = {
    "epochs": {
        "canonical": {"event_codes": list(DEFAULT_IMAGE_EVENT_ID.values())},
        "residual_eeg_qc": dict(DEFAULT_RESIDUAL_EEG_QC_SEMANTICS),
    },
    "response_timing_qc": {"sampling_rate_hz": 500.0},
}


class ConfigurationValidationError(ValueError):
    """Raised when Session 2 configuration cannot be resolved safely."""


@dataclass(frozen=True)
class RuntimeConfigOverrides:
    """Allowlisted runtime choices applied after source configuration."""

    subjects: tuple[str, ...] | None = None
    recording_parts: tuple[str, ...] | None = None
    allow_excluded: bool = False
    validation_only_skip_ica: bool = False
    corrected_output_root: Path | None = None

    @classmethod
    def from_values(
        cls,
        *,
        subjects: Sequence[str] | None = None,
        recording_parts: Sequence[str] | None = None,
        allow_excluded: bool = False,
        validation_only_skip_ica: bool = False,
        corrected_output_root: str | Path | None = None,
    ) -> "RuntimeConfigOverrides":
        return cls(
            subjects=None if subjects is None else tuple(str(value) for value in subjects),
            recording_parts=None if recording_parts is None else tuple(str(value) for value in recording_parts),
            allow_excluded=bool(allow_excluded),
            validation_only_skip_ica=bool(validation_only_skip_ica),
            corrected_output_root=(
                None if corrected_output_root is None else Path(corrected_output_root)
            ),
        )


@dataclass(frozen=True)
class ConfigSourceIdentity:
    """Identity of the selected source file, separate from effective identity."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class ResolvedExecutionConfig:
    """Explicit participant/part and validation-mode choices for one run."""

    subjects: tuple[str, ...]
    recording_parts: tuple[str, ...]
    recording_part_selection: str
    allow_excluded: bool
    validation_only_skip_ica: bool
    final_preprocessing_eligible: bool


@dataclass(frozen=True)
class ResolvedSession2EEGConfig:
    """Validated, immutable effective Session 2 configuration."""

    schema_version: str
    source: ConfigSourceIdentity
    project_root: Path
    values: Mapping[str, Any]
    study_definition: KawaiiSession2StudyDefinition
    execution: ResolvedExecutionConfig
    resolved_paths: Mapping[str, Path]
    scientific_fingerprint: str
    resolved_config_fingerprint: str

    def to_mutable_config(self) -> dict[str, Any]:
        """Return a legacy mapping adapter for consumers not yet migrated."""

        value = _deep_thaw(self.values)
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise TypeError("Resolved configuration root must be a mapping.")
        return value

    def scientific_payload(self) -> dict[str, Any]:
        return self.study_definition.to_primitive()

    def resolved_identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "effective_config": self.to_mutable_config(),
            "study_definition": self.study_definition.to_primitive(),
            "execution": _execution_primitive(self.execution),
        }

    def to_primitive(self) -> dict[str, Any]:
        """Return the complete JSON-safe snapshot representation."""

        return {
            "schema_version": self.schema_version,
            "source": {
                "path": _portable_path(self.source.path, self.project_root),
                "sha256": self.source.sha256,
            },
            "project_root": self.project_root.resolve().as_posix(),
            "effective_config": self.to_mutable_config(),
            "study_definition": self.study_definition.to_primitive(),
            "execution": _execution_primitive(self.execution),
            "resolved_paths": {
                key: _portable_path(path, self.project_root)
                for key, path in sorted(self.resolved_paths.items())
            },
            "scientific_fingerprint": self.scientific_fingerprint,
            "resolved_config_fingerprint": self.resolved_config_fingerprint,
        }

    def canonical_json(self) -> str:
        """Serialize the complete snapshot deterministically as canonical JSON."""

        return canonical_json(self.to_primitive())


_OPEN_MAPPING_PATHS = {
    ("channels", "channel_name_aliases"),
    ("channels", "channel_name_overrides"),
    ("participants", "eeg_file_map"),
    ("analysis", "exclusion_reasons"),
    ("manual_overrides", "bad_channels"),
    ("manual_overrides", "ignore_channels"),
    ("manual_overrides", "exclude_recording_parts"),
    ("manual_overrides", "exclude_trials"),
    ("manual_overrides", "preprocessing_thresholds"),
    ("manual_overrides", "bad_segments"),
    ("manual_overrides", "ica_components"),
    ("manual_overrides", "notes"),
}

_DYNAMIC_MAPPING_PATHS = {
    ("rois",),
    ("erp", "recognition", "measurement_time_windows"),
    ("erp", "recognition", "topography_time_windows"),
    ("erp", "recognition", "rois"),
    ("erp", "recognition", "conditions"),
    ("erp", "cuteness", "measurement_time_windows"),
    ("erp", "cuteness", "topography_time_windows"),
    ("erp", "cuteness", "conditions"),
    ("response_timing_qc", "response_types"),
}

_REPLACE_MAPPING_PATHS = _OPEN_MAPPING_PATHS | _DYNAMIC_MAPPING_PATHS | {
    ("recordings", "participant_recordings"),
}

_REQUIRED_PATHS = (
    ("recordings", "expected_sampling_rate_hz"),
    ("recordings", "sampling_rate_tolerance_hz"),
    ("preprocessing", "analysis_highpass_hz"),
    ("preprocessing", "analysis_lowpass_hz"),
    ("preprocessing", "ica_highpass_hz"),
    ("preprocessing", "ica_lowpass_hz"),
    ("preprocessing", "reference", "acquisition_reference_channel"),
    ("preprocessing", "reference", "reconstruct_after_ica_gate3c"),
    ("preprocessing", "reference", "apply_average_reference_gate3a"),
    ("preprocessing", "reference", "apply_average_reference_gate3c"),
    ("preprocessing", "interpolation", "apply_only_manual_confirmed_bad_channels"),
    ("preprocessing", "ica", "method"),
    ("preprocessing", "ica", "n_components"),
    ("preprocessing", "ica", "decim"),
    ("preprocessing", "ica", "random_state"),
    ("epochs", "canonical", "event"),
    ("epochs", "canonical", "tmin_s"),
    ("epochs", "canonical", "tmax_s"),
    ("epochs", "canonical", "baseline_s"),
    ("epochs", "residual_eeg_qc", "peak_to_peak_threshold_uv"),
    ("erp", "recognition", "conditions"),
    ("erp", "recognition", "rois"),
    ("erp", "recognition", "measurement_time_windows"),
    ("erp", "recognition", "scientific_hierarchy"),
    ("erp", "cuteness", "primary_model"),
    ("triggers",),
    ("channels", "eeg"),
    ("analysis", "include_subjects"),
    ("analysis", "exclude_subjects"),
    ("response_timing_qc", "behavioral_rt_anchor"),
    ("response_timing_qc", "eeg_response_marker_anchor"),
    ("response_timing_qc", "response_locked_inference"),
    ("statistics", "multiple_comparison_correction"),
    ("statistics", "recognition", "comparisons"),
)


def resolve_session2_eeg_config(
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    overrides: RuntimeConfigOverrides | None = None,
) -> ResolvedSession2EEGConfig:
    """Validate and resolve one authoritative Session 2 configuration.

    Precedence is: schema/default config, selected source configuration,
    allowlisted runtime overrides, then root/path and participant/part
    resolution. Source identity remains separate from effective identities.
    """

    project_root = (root or get_project_root()).resolve()
    selected_path = _selected_config_path(project_root, config_path)
    source_config = read_yaml_config(selected_path)
    default_path = project_root / "config" / CONFIG_NAME
    default_config = read_yaml_config(default_path)
    schema_defaults = _deep_merge(default_config, _CENTRAL_RESOLVED_DEFAULTS)

    _validate_required_paths(source_config)
    _validate_value_tree(source_config, schema_defaults)
    effective = _deep_merge(schema_defaults, source_config)
    if not _has_path(source_config, ("response_timing_qc", "sampling_rate_hz")):
        effective["response_timing_qc"]["sampling_rate_hz"] = effective["recordings"]["expected_sampling_rate_hz"]
    _validate_value_tree(effective, schema_defaults)
    _validate_material_semantics(effective)
    study_definition = KawaiiSession2StudyDefinition.from_config(effective)

    runtime = overrides or RuntimeConfigOverrides()
    effective = deepcopy(effective)
    if runtime.validation_only_skip_ica:
        ica = effective["preprocessing"]["ica"]
        ica["enabled"] = False
        ica["required_for_epoching"] = False
    if runtime.corrected_output_root is not None:
        _route_generated_paths(effective, runtime.corrected_output_root, project_root)

    subjects = tuple(
        resolve_subjects(
            None if runtime.subjects is None else list(runtime.subjects),
            effective,
            allow_excluded=runtime.allow_excluded,
        )
    )
    parts = _normalise_recording_parts(runtime.recording_parts)
    execution = ResolvedExecutionConfig(
        subjects=subjects,
        recording_parts=parts,
        recording_part_selection="discover" if runtime.recording_parts is None else "explicit",
        allow_excluded=runtime.allow_excluded,
        validation_only_skip_ica=runtime.validation_only_skip_ica,
        final_preprocessing_eligible=not runtime.validation_only_skip_ica,
    )
    resolved_paths = _path_mapping(resolve_session2_eeg_paths(effective, root=project_root, create=False))
    scientific_payload = study_definition.to_primitive()
    resolved_payload = {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "effective_config": effective,
        "study_definition": scientific_payload,
        "execution": _execution_primitive(execution),
    }

    return ResolvedSession2EEGConfig(
        schema_version=RESOLVED_CONFIG_SCHEMA_VERSION,
        source=ConfigSourceIdentity(path=selected_path, sha256=_sha256_file(selected_path)),
        project_root=project_root,
        values=_deep_freeze(effective),
        study_definition=study_definition,
        execution=execution,
        resolved_paths=MappingProxyType(resolved_paths),
        scientific_fingerprint=canonical_sha256(scientific_payload),
        resolved_config_fingerprint=canonical_sha256(resolved_payload),
    )


def _route_generated_paths(
    config: dict[str, Any], corrected_output_root: Path, project_root: Path
) -> None:
    """Route every generated/scratch path into one isolated replay namespace.

    Source and manual-review authorities deliberately remain unchanged.  The
    override is applied only after source configuration validation so it is an
    explicit execution choice, not a silent change to the scientific source.
    """

    output = corrected_output_root
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve(strict=False)
    paths = config.setdefault("paths", {})
    routed = {
        "output_dir": output,
        "derivatives_dir": output / "derivatives",
        "brainvision_working_dir": output / "derivatives" / "brainvision_working_copies",
        "alignment_dir": output / "derivatives" / "alignment",
        "preprocessing_dir": output / "derivatives" / "preprocessing",
        "ica_dir": output / "derivatives" / "ica",
        "epochs_dir": output / "derivatives" / "epochs",
        "raw_qc_dir": output / "qc" / "raw",
        "ica_qc_dir": output / "qc" / "ica",
        "epoch_qc_dir": output / "qc" / "epochs",
        "report_dir": output / "reports",
        "qc_dir": output / "qc",
    }
    paths.update({key: str(value) for key, value in routed.items()})
    memory = config.setdefault("preprocessing", {}).setdefault("memory", {})
    scratch = output / ".scratch" / "memmap"
    memory["memmap_dir"] = str(scratch)
    memory["gate3c_memmap_dir"] = str(scratch)


def _selected_config_path(project_root: Path, config_path: str | Path | None) -> Path:
    path = Path(config_path) if config_path is not None else Path("config") / CONFIG_NAME
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _validate_required_paths(config: Mapping[str, Any]) -> None:
    missing = [".".join(path) for path in _REQUIRED_PATHS if not _has_path(config, path)]
    if missing:
        raise ConfigurationValidationError("Missing required configuration key(s): " + ", ".join(missing))


def _has_path(config: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return False
        value = value[key]
    return True


def _validate_value_tree(value: Any, template: Any, path: tuple[str, ...] = ()) -> None:
    dotted = ".".join(path) or "<root>"
    if isinstance(template, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigurationValidationError(f"{dotted} must be a mapping, got {type(value).__name__}.")
        if _is_open_mapping(path):
            return
        if path == ("recordings", "participant_recordings"):
            _validate_participant_recordings(value, template, path)
            return
        if path in _DYNAMIC_MAPPING_PATHS:
            prototype = _merged_mapping_values(template)
            for key, item in value.items():
                _validate_value_tree(item, prototype, path + (str(key),))
            return
        unknown = sorted(str(key) for key in value if key not in template)
        if unknown:
            raise ConfigurationValidationError(f"Unknown configuration key(s) under {dotted}: {', '.join(unknown)}")
        for key, item in value.items():
            _validate_value_tree(item, template[key], path + (str(key),))
        return

    if isinstance(template, list):
        if not isinstance(value, list):
            raise ConfigurationValidationError(f"{dotted} must be a list, got {type(value).__name__}.")
        if not template:
            return
        item_template = _merged_list_templates(template)
        for index, item in enumerate(value):
            _validate_value_tree(item, item_template, path + (str(index),))
        return

    if template is None:
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ConfigurationValidationError(f"{dotted} must be a scalar or null, got {type(value).__name__}.")
        return
    if isinstance(template, bool):
        if type(value) is not bool:
            raise ConfigurationValidationError(f"{dotted} must be a boolean, got {type(value).__name__}.")
        return
    if isinstance(template, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ConfigurationValidationError(f"{dotted} must be a finite number, got {value!r}.")
        return
    if not isinstance(value, type(template)):
        raise ConfigurationValidationError(f"{dotted} must be {type(template).__name__}, got {type(value).__name__}.")


def _is_open_mapping(path: tuple[str, ...]) -> bool:
    if path in _OPEN_MAPPING_PATHS:
        return True
    return bool(path and path[-1] == "criteria" and "conditions" in path)


def _validate_participant_recordings(value: Mapping[str, Any], template: Mapping[str, Any], path: tuple[str, ...]) -> None:
    participant_template = _merged_mapping_values(template)
    part_template = _merged_mapping_values(participant_template)
    for subject, parts in value.items():
        if not isinstance(parts, Mapping):
            raise ConfigurationValidationError(f"{'.'.join(path + (str(subject),))} must be a mapping.")
        for part, spec in parts.items():
            _validate_value_tree(spec, part_template, path + (str(subject), str(part)))


def _merged_mapping_values(mapping: Mapping[str, Any]) -> Any:
    values = list(mapping.values())
    if not values:
        return {}
    merged = deepcopy(values[0])
    for value in values[1:]:
        merged = _merge_schema_templates(merged, value)
    return merged


def _merged_list_templates(values: list[Any]) -> Any:
    merged = deepcopy(values[0])
    for value in values[1:]:
        merged = _merge_schema_templates(merged, value)
    return merged


def _merge_schema_templates(left: Any, right: Any) -> Any:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        merged = deepcopy(dict(left))
        for key, value in right.items():
            merged[key] = _merge_schema_templates(merged[key], value) if key in merged else deepcopy(value)
        return merged
    return left


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any], path: tuple[str, ...] = ()) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        child_path = path + (str(key),)
        if child_path in _REPLACE_MAPPING_PATHS:
            result[key] = deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value, child_path)
        else:
            result[key] = deepcopy(value)
    return result


def _validate_material_semantics(config: Mapping[str, Any]) -> None:
    recordings = config["recordings"]
    if float(recordings["expected_sampling_rate_hz"]) <= 0:
        raise ConfigurationValidationError("recordings.expected_sampling_rate_hz must be positive.")
    if float(recordings["sampling_rate_tolerance_hz"]) < 0:
        raise ConfigurationValidationError("recordings.sampling_rate_tolerance_hz must be non-negative.")

    prep = config["preprocessing"]
    _validate_band(prep["analysis_highpass_hz"], prep["analysis_lowpass_hz"], "preprocessing.analysis")
    _validate_band(prep["ica_highpass_hz"], prep["ica_lowpass_hz"], "preprocessing.ica")
    raw_qc = prep["raw_qc"]
    _validate_choice(raw_qc["poor_correlation_method"], {"prep_style_p98_v1"}, "preprocessing.raw_qc.poor_correlation_method")
    _validate_choice(
        raw_qc["poor_correlation_filter_method"],
        {"scipy_butterworth_sosfiltfilt"},
        "preprocessing.raw_qc.poor_correlation_filter_method",
    )
    _validate_choice(
        raw_qc["poor_correlation_reference_state"],
        {"acquisition_reference"},
        "preprocessing.raw_qc.poor_correlation_reference_state",
    )
    _validate_choice(
        raw_qc["poor_correlation_data_scope"],
        {"task_segments", "full_recording"},
        "preprocessing.raw_qc.poor_correlation_data_scope",
    )
    _validate_band(
        raw_qc["poor_correlation_filter_highpass_hz"],
        raw_qc["poor_correlation_filter_lowpass_hz"],
        "preprocessing.raw_qc.poor_correlation",
    )
    for key in ("poor_correlation_window_duration_s", "poor_correlation_filter_chunk_duration_s"):
        if float(raw_qc[key]) <= 0:
            raise ConfigurationValidationError(f"preprocessing.raw_qc.{key} must be positive.")
    if float(raw_qc["poor_correlation_filter_padding_s"]) < 0:
        raise ConfigurationValidationError("preprocessing.raw_qc.poor_correlation_filter_padding_s must be non-negative.")
    if not 0 < float(raw_qc["poor_correlation_peer_percentile"]) <= 1:
        raise ConfigurationValidationError("preprocessing.raw_qc.poor_correlation_peer_percentile must be in (0, 1].")
    if not 0 <= float(raw_qc["poor_correlation_threshold"]) <= 1:
        raise ConfigurationValidationError("preprocessing.raw_qc.poor_correlation_threshold must be in [0, 1].")
    if not 0 <= float(raw_qc["poor_correlation_bad_window_fraction"]) <= 1:
        raise ConfigurationValidationError("preprocessing.raw_qc.poor_correlation_bad_window_fraction must be in [0, 1].")
    order = raw_qc["poor_correlation_filter_order"]
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ConfigurationValidationError("preprocessing.raw_qc.poor_correlation_filter_order must be a positive integer.")
    decim = prep["ica"]["decim"]
    if isinstance(decim, bool) or not isinstance(decim, int) or decim < 1:
        raise ConfigurationValidationError("preprocessing.ica.decim must be a positive integer.")

    canonical = config["epochs"]["canonical"]
    tmin = float(canonical["tmin_s"])
    tmax = float(canonical["tmax_s"])
    baseline = canonical["baseline_s"]
    if not tmin < tmax:
        raise ConfigurationValidationError("epochs.canonical.tmin_s must be less than tmax_s.")
    if len(baseline) != 2 or not tmin <= float(baseline[0]) <= float(baseline[1]) <= tmax:
        raise ConfigurationValidationError("epochs.canonical.baseline_s must be an ordered two-value window inside the epoch.")
    if float(config["epochs"]["residual_eeg_qc"]["peak_to_peak_threshold_uv"]) <= 0:
        raise ConfigurationValidationError("epochs.residual_eeg_qc.peak_to_peak_threshold_uv must be positive.")
    residual_qc = config["epochs"]["residual_eeg_qc"]
    _validate_choice(residual_qc["comparison"], {"greater_than"}, "epochs.residual_eeg_qc.comparison")
    _validate_choice(residual_qc["channels"], {"eeg_only"}, "epochs.residual_eeg_qc.channels")
    _validate_choice(
        residual_qc["metric"],
        {"maximum_within-channel_peak_to_peak_across_epoch_uv"},
        "epochs.residual_eeg_qc.metric",
    )

    statistics = config["statistics"]
    if not 0 < float(statistics["alpha"]) < 1:
        raise ConfigurationValidationError("statistics.alpha must be between 0 and 1.")
    if not 0 < float(statistics["confidence_level"]) < 1:
        raise ConfigurationValidationError("statistics.confidence_level must be between 0 and 1.")
    statistical_choices = {
        "paired_test": {"auto", "paired_t", "wilcoxon"},
        "fallback_paired_test": {"paired_t", "wilcoxon"},
        "normality_test": {"none", "shapiro"},
        "multiple_comparison_correction": {"none", "bonferroni", "holm", "fdr_bh"},
        "effect_size": {"auto", "cohens_d_paired", "rank_biserial"},
    }
    for key, allowed in statistical_choices.items():
        _validate_choice(statistics[key], allowed, f"statistics.{key}")
    for key in ("technical_minimum_pairs", "minimum_normality_n"):
        value = statistics[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationValidationError(f"statistics.{key} must be a positive integer.")

    timing_rate = config["response_timing_qc"]["sampling_rate_hz"]
    if float(timing_rate) <= 0:
        raise ConfigurationValidationError("response_timing_qc.sampling_rate_hz must be positive.")

    _validate_trigger_codes(config["triggers"])
    image_codes = set(config["epochs"]["canonical"]["event_codes"])
    trigger_image_codes = set(config["triggers"]["image_onset"].values())
    if image_codes != trigger_image_codes:
        raise ConfigurationValidationError(
            "epochs.canonical.event_codes must match the triggers.image_onset codes."
        )
    _validate_primary_cells(config)
    _validate_nonempty_string(config["preprocessing"]["reference"]["acquisition_reference_channel"], "preprocessing.reference.acquisition_reference_channel")


def _validate_band(highpass: Any, lowpass: Any, label: str) -> None:
    if float(highpass) < 0 or float(lowpass) <= float(highpass):
        raise ConfigurationValidationError(f"{label} filter band must satisfy 0 <= highpass < lowpass.")


def _validate_trigger_codes(triggers: Mapping[str, Any]) -> None:
    codes: list[int] = []
    for name, spec in triggers.items():
        if not isinstance(spec, Mapping):
            raise ConfigurationValidationError(f"triggers.{name} must be a mapping.")
        for key, value in spec.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationValidationError(f"triggers.{name}.{key} must be an integer.")
        if "min" in spec or "max" in spec:
            if set(spec) != {"min", "max"} or int(spec["min"]) > int(spec["max"]):
                raise ConfigurationValidationError(f"triggers.{name} must define an ordered min/max range.")
            codes.extend(range(int(spec["min"]), int(spec["max"]) + 1))
        else:
            codes.extend(int(value) for value in spec.values())
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    if duplicates:
        raise ConfigurationValidationError(f"Trigger codes/ranges overlap: {duplicates}.")


def _validate_primary_cells(config: Mapping[str, Any]) -> None:
    recognition = config["erp"]["recognition"]
    rois = recognition["rois"]
    windows = recognition["measurement_time_windows"]
    for index, cell in enumerate(recognition["scientific_hierarchy"]["primary_cells"]):
        if cell["roi"] not in rois:
            raise ConfigurationValidationError(f"erp.recognition.scientific_hierarchy.primary_cells.{index}.roi is unknown.")
        if cell["time_window"] not in windows:
            raise ConfigurationValidationError(f"erp.recognition.scientific_hierarchy.primary_cells.{index}.time_window is unknown.")


def _validate_nonempty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationValidationError(f"{path} must be a non-empty string.")


def _validate_choice(value: Any, allowed: set[str], path: str) -> None:
    if value not in allowed:
        raise ConfigurationValidationError(
            f"{path} must be one of {', '.join(sorted(allowed))}, got {value!r}."
        )


def _normalise_recording_parts(parts: tuple[str, ...] | None) -> tuple[str, ...]:
    if parts is None:
        return ()
    normalised = tuple(str(part).strip() for part in parts)
    if any(not part for part in normalised):
        raise ConfigurationValidationError("Recording-part selectors must be non-empty strings.")
    if len(set(normalised)) != len(normalised):
        raise ConfigurationValidationError("Recording-part selectors must be unique.")
    return normalised


def _scientific_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the Batch 2 private helper."""

    return KawaiiSession2StudyDefinition.from_config(config).to_primitive()


def _path_mapping(paths: Session2EEGPaths) -> dict[str, Path]:
    return {field.name: Path(getattr(paths, field.name)).resolve() for field in fields(paths)}


def _execution_primitive(execution: ResolvedExecutionConfig) -> dict[str, Any]:
    return {
        "subjects": list(execution.subjects),
        "recording_parts": list(execution.recording_parts),
        "recording_part_selection": execution.recording_part_selection,
        "allow_excluded": execution.allow_excluded,
        "validation_only_skip_ica": execution.validation_only_skip_ica,
        "final_preprocessing_eligible": execution.final_preprocessing_eligible,
    }


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return the shared deterministic JSON representation for WS2 identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    """Hash a value through the shared canonical JSON implementation."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# Batch 2 private compatibility aliases. New persistence code uses the public
# names above so canonicalization has one implementation.
_canonical_json = canonical_json
_sha256_json = canonical_sha256


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
