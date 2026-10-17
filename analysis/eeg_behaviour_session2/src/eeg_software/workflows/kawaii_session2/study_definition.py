from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return deepcopy(value)


SCIENTIFIC_DEFINITION_IDENTITY_VERSION = "kawaii_session2_scientific_definition_v3"

ACQUISITION_RESPONSE_SEQUENCE = (
    "key_down_detected",
    "secs_stored",
    "behavioral_rt_calculated",
    "KbReleaseWait",
    "key_release",
    "response_function_returns",
    "sendTrigger",
    "eeg_response_marker",
)

_DEFAULT_RESPONSE_TYPES = {
    "recognition": {
        "behavioural_rt": "recognition_rt_s",
        "timeout": "recognition_timeout",
        "onset_sample": "image_onset_sample",
        "response_sample": "recognition_response_sample",
        "onset_code": "image_onset_code",
        "response_code": "recognition_response_code",
        "trigger_event_prefix": "recognition_response",
    },
    "kawaii": {
        "behavioural_rt": "kawaii_rt_s",
        "timeout": "kawaii_timeout",
        "onset_sample": "kawaii_question_sample",
        "response_sample": "kawaii_response_sample",
        "onset_code": "kawaii_question_code",
        "response_code": "kawaii_response_code",
        "trigger_event_prefix": "kawaii_rating_response",
    },
    "arousal": {
        "behavioural_rt": "arousal_rt_s",
        "timeout": "arousal_timeout",
        "onset_sample": "arousal_question_sample",
        "response_sample": "arousal_response_sample",
        "onset_code": "arousal_question_code",
        "response_code": "arousal_response_code",
        "trigger_event_prefix": "arousal_rating_response",
    },
}
DEFAULT_RESPONSE_TYPES: Mapping[str, Any] = _deep_freeze(_DEFAULT_RESPONSE_TYPES)

_DEFAULT_TRIGGER_FAMILIES = {
    "image_onset": {"old": 120, "new": 121},
    "recognition_response": {"yes": 150, "no": 151},
    "kawaii_question_trial_digit": {"min": 100, "max": 109},
    "kawaii_rating_response": {"min": 11, "max": 19},
    "arousal_question_trial_group": {"min": 200, "max": 239},
    "arousal_rating_response": {"min": 21, "max": 29},
}
DEFAULT_IMAGE_EVENT_ID: Mapping[str, int] = MappingProxyType(
    dict(_DEFAULT_TRIGGER_FAMILIES["image_onset"])
)
DEFAULT_RESIDUAL_EEG_QC_SEMANTICS: Mapping[str, str] = MappingProxyType(
    {
        "comparison": "greater_than",
        "channels": "eeg_only",
        "metric": "maximum_within-channel_peak_to_peak_across_epoch_uv",
    }
)

_TRIGGER_CATEGORY_FAMILIES = (
    ("image_onset", "image_onset"),
    ("recognition_response", "recognition_response"),
    ("kawaii_question", "kawaii_question_trial_digit"),
    ("kawaii_response", "kawaii_rating_response"),
    ("arousal_question", "arousal_question_trial_group"),
    ("arousal_response", "arousal_rating_response"),
)

_DEFAULT_TIMING_VALUES = {
    "behavioral_rt_anchor": "key_down",
    "eeg_response_marker_anchor": "post_key_release_trigger",
    "discrepancy_interpretation": "composite_timing_discrepancy",
    "trigger_delay_interpretation": "trigger_write_overhead_only",
    "fixed_marker_correction_s": None,
    "participant_median_correction": False,
    "response_locked_inference": "NOT_VALIDATED_NOT_APPROVED",
    "stimulus_locked_erp_status": "UNAFFECTED",
    "response_types": _DEFAULT_RESPONSE_TYPES,
}


@dataclass(frozen=True)
class ERPPrimaryCell:
    roi: str
    time_window: str
    hypothesis_id: str


@dataclass(frozen=True)
class ERPBranchStudyDefinition:
    """Immutable scientific definition for one Kawaii ERP branch."""

    name: str
    values: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any], branch: str) -> "ERPBranchStudyDefinition":
        erp = _mapping(config.get("erp"))
        branch_values = _deep_thaw(_mapping(erp.get(branch)))
        branch_values.pop("plotting", None)
        branch_values.pop("figures", None)
        return cls(name=str(branch), values=_deep_freeze(branch_values))

    def to_primitive(self) -> dict[str, Any]:
        return _deep_thaw(self.values)

    @property
    def primary_cells(self) -> tuple[ERPPrimaryCell, ...]:
        hierarchy = _mapping(self.values.get("scientific_hierarchy"))
        return tuple(
            ERPPrimaryCell(
                roi=str(item.get("roi", "")),
                time_window=str(item.get("time_window", "")),
                hypothesis_id=str(item.get("hypothesis_id", "")),
            )
            for item in hierarchy.get("primary_cells", ())
        )

    @property
    def primary_model(self) -> Mapping[str, Any]:
        return _mapping(self.values.get("primary_model"))


@dataclass(frozen=True)
class KawaiiTrialReconstructionDefinition:
    question_family: str
    question_base: int
    arousal_family: str
    arousal_base: int
    arousal_multiplier: int = 10
    trial_offset: int = 1

    @property
    def formula(self) -> str:
        return (
            f"{self.arousal_multiplier} * (arousal_question_trigger - {self.arousal_base}) + "
            f"(kawaii_question_trigger - {self.question_base}) + {self.trial_offset}"
        )

    def reconstruct(self, kawaii_question_code: int, arousal_question_code: int) -> int:
        return (
            self.arousal_multiplier * (int(arousal_question_code) - self.arousal_base)
            + (int(kawaii_question_code) - self.question_base)
            + self.trial_offset
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "question_family": self.question_family,
            "question_base": self.question_base,
            "arousal_family": self.arousal_family,
            "arousal_base": self.arousal_base,
            "arousal_multiplier": self.arousal_multiplier,
            "trial_offset": self.trial_offset,
            "formula": self.formula,
        }


@dataclass(frozen=True)
class KawaiiTriggerDefinition:
    """Kawaii marker meanings, distinct from generic marker parsing."""

    families: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "KawaiiTriggerDefinition":
        configured = _mapping((config or {}).get("triggers"))
        families = configured or _DEFAULT_TRIGGER_FAMILIES
        return cls(families=_deep_freeze(families))

    @property
    def image_event_id(self) -> Mapping[str, int]:
        return MappingProxyType(
            {str(label): int(code) for label, code in _mapping(self.families.get("image_onset")).items()}
        )

    @property
    def trial_reconstruction(self) -> KawaiiTrialReconstructionDefinition:
        question = _mapping(self.families.get("kawaii_question_trial_digit"))
        arousal = _mapping(self.families.get("arousal_question_trial_group"))
        return KawaiiTrialReconstructionDefinition(
            question_family="kawaii_question_trial_digit",
            question_base=int(question["min"]),
            arousal_family="arousal_question_trial_group",
            arousal_base=int(arousal["min"]),
        )

    def category(self, code: Any) -> str:
        if code is None:
            return "non_numeric"
        try:
            numeric = int(code)
        except (TypeError, ValueError):
            return "non_numeric"
        for category, family_name in _TRIGGER_CATEGORY_FAMILIES:
            if _code_in_family(numeric, _mapping(self.families.get(family_name))):
                return category
        return "unexpected"

    def to_primitive(self) -> dict[str, Any]:
        return _deep_thaw(self.families)

    def semantics_primitive(self) -> dict[str, Any]:
        return {
            "category_families": {
                category: family for category, family in _TRIGGER_CATEGORY_FAMILIES
            },
            "trial_reconstruction": self.trial_reconstruction.to_primitive(),
        }


@dataclass(frozen=True)
class KawaiiTimingStudyDefinition:
    """Verified scientific interpretation of behavioural and EEG timing."""

    values: Mapping[str, Any]
    acquisition_sequence: tuple[str, ...] = ACQUISITION_RESPONSE_SEQUENCE

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "KawaiiTimingStudyDefinition":
        timing = _mapping((config or {}).get("response_timing_qc"))
        source = timing or _DEFAULT_TIMING_VALUES
        scientific = _deep_thaw(source)
        scientific.pop("output_dir", None)
        return cls(values=_deep_freeze(scientific))

    def to_primitive(self) -> dict[str, Any]:
        return _deep_thaw(self.values)

    def semantics_primitive(self) -> dict[str, Any]:
        return {
            "sequence": list(self.acquisition_sequence),
            "behavioral_rt_anchor": self.values["behavioral_rt_anchor"],
            "eeg_response_marker_anchor": self.values["eeg_response_marker_anchor"],
            "difference_interpretation": self.values["discrepancy_interpretation"],
            "trigger_delay_interpretation": self.values["trigger_delay_interpretation"],
            "fixed_correction_applied": self.values.get("fixed_marker_correction_s") is not None,
            "participant_median_correction_applied": bool(self.values["participant_median_correction"]),
            "response_locked_inference": self.values["response_locked_inference"],
            "stimulus_locked_erp_status": self.values["stimulus_locked_erp_status"],
        }


@dataclass(frozen=True)
class KawaiiSession2StudyDefinition:
    """Immutable scientific meaning for the Kawaii Session 2 EEG study.

    This is derived from the validated effective configuration. Execution
    selection, storage, and presentation remain outside this object.
    """

    recordings: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    epochs: Mapping[str, Any]
    recognition: ERPBranchStudyDefinition
    cuteness: ERPBranchStudyDefinition
    triggers: KawaiiTriggerDefinition
    channels: Mapping[str, Any]
    rois: Mapping[str, Any]
    timing: KawaiiTimingStudyDefinition
    statistics: Mapping[str, Any]
    manual_overrides: Mapping[str, Any]
    identity_version: str = SCIENTIFIC_DEFINITION_IDENTITY_VERSION

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "KawaiiSession2StudyDefinition":
        preprocessing = _deep_thaw(_mapping(config.get("preprocessing")))
        preprocessing.pop("memory", None)
        preprocessing.pop("save", None)
        ica = _mapping(preprocessing.get("ica"))
        ica.pop("diagnostics", None)
        preprocessing["ica"] = ica
        raw_qc = _mapping(preprocessing.get("raw_qc"))
        raw_qc.pop("figures", None)
        for key in (
            "raw_overview_duration_s",
            "raw_overview_start_s",
            "raw_overview_eeg_scaling_uv",
            "psd_plot_fmin_hz",
            "psd_plot_fmax_hz",
        ):
            raw_qc.pop(key, None)
        preprocessing["raw_qc"] = raw_qc

        epochs = _mapping(config.get("epochs"))
        epoch_science = {
            "canonical": _deep_thaw(epochs.get("canonical")),
            "residual_eeg_qc": _deep_thaw(epochs.get("residual_eeg_qc")),
        }
        return cls(
            recordings=_deep_freeze(_mapping(config.get("recordings"))),
            preprocessing=_deep_freeze(preprocessing),
            epochs=_deep_freeze(epoch_science),
            recognition=ERPBranchStudyDefinition.from_config(config, "recognition"),
            cuteness=ERPBranchStudyDefinition.from_config(config, "cuteness"),
            triggers=KawaiiTriggerDefinition.from_config(config),
            channels=_deep_freeze(_mapping(config.get("channels"))),
            rois=_deep_freeze(_mapping(config.get("rois"))),
            timing=KawaiiTimingStudyDefinition.from_config(config),
            statistics=_deep_freeze(_mapping(config.get("statistics"))),
            manual_overrides=_deep_freeze(_mapping(config.get("manual_overrides"))),
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "identity_version": self.identity_version,
            "recordings": _deep_thaw(self.recordings),
            "preprocessing": _deep_thaw(self.preprocessing),
            "epochs": _deep_thaw(self.epochs),
            "erp": {
                "recognition": self.recognition.to_primitive(),
                "cuteness": self.cuteness.to_primitive(),
            },
            "triggers": self.triggers.to_primitive(),
            "channels": _deep_thaw(self.channels),
            "rois": _deep_thaw(self.rois),
            "response_timing_qc": self.timing.to_primitive(),
            "statistics": _deep_thaw(self.statistics),
            "manual_overrides": _deep_thaw(self.manual_overrides),
            "task_semantics": {
                "triggers": self.triggers.semantics_primitive(),
                "response_timing": self.timing.semantics_primitive(),
            },
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_primitive(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


def erp_branch_settings(config: Mapping[str, Any], branch: str) -> dict[str, Any]:
    """Return legacy ERP settings with science sourced from the study layer."""

    erp = dict(_mapping(config.get("erp")))
    common = _deep_thaw(_mapping(erp.get("common")))
    raw_branch = _deep_thaw(_mapping(erp.get(branch)))
    definition = ERPBranchStudyDefinition.from_config(config, branch)
    scientific_settings = definition.to_primitive()
    figures = dict(_mapping(common.get("figures")))
    figures.update(dict(_mapping(raw_branch.get("figures"))))
    settings = {**common, **scientific_settings}
    if "plotting" in raw_branch:
        settings["plotting"] = raw_branch["plotting"]
    settings["figures"] = figures
    settings["conditions"] = dict(_mapping(scientific_settings.get("conditions")))
    canonical = _mapping(_mapping(config.get("epochs")).get("canonical"))
    settings["epoch_tmin_s"] = float(canonical.get("tmin_s", -0.2))
    settings["epoch_tmax_s"] = float(canonical.get("tmax_s", 1.2))
    settings["baseline_s"] = list(canonical.get("baseline_s", [-0.2, 0.0]))
    if "rois" not in settings:
        settings["rois"] = _deep_thaw(_mapping(config.get("rois")))
    settings["branch"] = branch
    return settings


def configured_rois(settings: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        str(name): list(_mapping(spec).get("channels", ()) or ())
        for name, spec in _mapping(settings.get("rois")).items()
    }


def configured_time_windows(
    settings: Mapping[str, Any], key: str = "measurement_time_windows"
) -> dict[str, dict[str, float]]:
    windows = {}
    for name, spec in _mapping(settings.get(key)).items():
        row = _mapping(spec)
        windows[str(name)] = {
            "label": str(row.get("label", name)),
            "start_ms": float(row.get("start_ms")),
            "end_ms": float(row.get("end_ms")),
        }
    return windows


def configured_minimum_trials(settings: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value or 0)
        for key, value in _mapping(settings.get("minimum_trials")).items()
    }


def configured_conditions(settings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(key): _deep_thaw(value)
        for key, value in _mapping(settings.get("conditions")).items()
    }


def recognition_scientific_metadata(
    settings: Mapping[str, Any], condition: str, roi: str, time_window: str
) -> dict[str, str]:
    hierarchy = _mapping(settings.get("scientific_hierarchy"))
    primary_cells = {
        (str(item.get("roi", "")), str(item.get("time_window", ""))): str(
            item.get("hypothesis_id", "")
        )
        for item in hierarchy.get("primary_cells", ())
    }
    cell = (str(roi), str(time_window))
    primary_contrast = str(hierarchy.get("primary_contrast", "hit_vs_correct_rejection"))
    difference_name = str(
        _mapping(settings.get("difference_wave")).get("name", "hit_minus_correct_rejection")
    )
    if condition == difference_name:
        return {
            "contrast": difference_name,
            "analysis_status": str(hierarchy.get("difference_wave_status", "DESCRIPTIVE")),
            "hypothesis_family": "recognition_descriptive_difference_wave",
            "hypothesis_id": "",
        }
    if condition in {"miss", "false_alarm"}:
        return {
            "contrast": "",
            "analysis_status": str(hierarchy.get("error_condition_status", "SECONDARY")),
            "hypothesis_family": "recognition_secondary_error_conditions",
            "hypothesis_id": "",
        }
    if condition in {"hit", "correct_rejection"} and cell in primary_cells:
        return {
            "contrast": primary_contrast,
            "analysis_status": "PRIMARY",
            "hypothesis_family": str(
                hierarchy.get("primary_hypothesis_family", "recognition_primary_old_new")
            ),
            "hypothesis_id": primary_cells[cell],
        }
    return {
        "contrast": primary_contrast if condition in {"hit", "correct_rejection"} else "",
        "analysis_status": str(hierarchy.get("crossover_status", "EXPLORATORY")),
        "hypothesis_family": "recognition_exploratory_crossover",
        "hypothesis_id": "",
    }


def recognition_roi_status(roi: str) -> str:
    if roi in {"frontal", "frontocentral"}:
        return "PRIMARY_EARLY"
    if roi in {"centroparietal", "parietal"}:
        return "PRIMARY_LATE"
    return "EXPLORATORY"


def plotting_channel_exclusions(settings: Mapping[str, Any]) -> list[str]:
    plotting = _mapping(settings.get("plotting"))
    return [str(channel) for channel in plotting.get("exclude_channels", ()) or ()]


def plotting_channel_exclusion_rationale(settings: Mapping[str, Any]) -> str:
    plotting = _mapping(settings.get("plotting"))
    return str(plotting.get("exclude_channels_rationale", ""))


def plotting_safe_evokeds(
    evokeds: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Return evoked copies with configured visualization-only exclusions."""

    excluded = plotting_channel_exclusions(settings)
    if not excluded:
        return dict(evokeds)
    safe: dict[str, Any] = {}
    for condition, evoked in evokeds.items():
        copied = evoked.copy() if hasattr(evoked, "copy") else evoked
        present = [channel for channel in excluded if channel in getattr(copied, "ch_names", ())]
        if present and hasattr(copied, "drop_channels"):
            copied.drop_channels(present)
        safe[condition] = copied
    return safe


def response_timing_semantics(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return verified acquisition semantics from the Kawaii study owner."""

    semantics = KawaiiTimingStudyDefinition.from_config(config).semantics_primitive()
    semantics["sequence"] = tuple(semantics["sequence"])
    return semantics


def _code_in_family(code: int, family: Mapping[str, Any]) -> bool:
    if "min" in family or "max" in family:
        return int(family["min"]) <= code <= int(family["max"])
    return code in {int(value) for value in family.values()}
