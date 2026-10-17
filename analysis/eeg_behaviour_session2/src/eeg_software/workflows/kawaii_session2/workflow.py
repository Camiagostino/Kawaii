from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Sequence

from .resolved_config import (
    ResolvedSession2EEGConfig,
    RuntimeConfigOverrides,
    resolve_session2_eeg_config,
)
from .study_definition import KawaiiSession2StudyDefinition


RUN_REQUEST_SCHEMA_VERSION = "kawaii_session2_run_request_v1"
WORKFLOW_DEFINITION_VERSION = "kawaii_session2_workflow_v1"


@dataclass(frozen=True)
class WorkflowStageDefinition:
    """One ordered workflow stage and its conceptual inputs."""

    name: str
    consumes: tuple[str, ...]

    def to_primitive(self) -> dict[str, Any]:
        return {"name": self.name, "consumes": list(self.consumes)}


@dataclass(frozen=True)
class KawaiiSession2WorkflowDefinition:
    """Small, declarative boundary around the established stage sequence."""

    stages: tuple[WorkflowStageDefinition, ...]
    name: str = "kawaii_session2_eeg"
    version: str = WORKFLOW_DEFINITION_VERSION

    def to_primitive(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "stages": [stage.to_primitive() for stage in self.stages],
        }


KAWAII_SESSION2_WORKFLOW = KawaiiSession2WorkflowDefinition(
    stages=(
        WorkflowStageDefinition("alignment", ("study_definition", "recording_parts", "metadata")),
        WorkflowStageDefinition("gate3a_preprocessing_qc", ("resolved_config", "alignment", "manual_review")),
        WorkflowStageDefinition("gate3b_ica_fit_review", ("resolved_config", "gate3a_outputs", "manual_review")),
        WorkflowStageDefinition("gate3c_ica_application_epoch_qc", ("resolved_config", "alignment", "gate3a_outputs", "gate3b_review")),
        WorkflowStageDefinition("gate4a_recognition_erp", ("study_definition", "gate3c_epochs")),
        WorkflowStageDefinition("gate4b_cuteness_erp", ("study_definition", "gate3c_epochs")),
        WorkflowStageDefinition("gate5a_group_aggregation", ("resolved_config", "gate4_outputs")),
        WorkflowStageDefinition("gate5b_group_figures", ("study_definition", "gate5a_outputs")),
        WorkflowStageDefinition("gate5c_group_statistics", ("study_definition", "gate5a_outputs")),
        WorkflowStageDefinition("response_timing_qc", ("study_definition", "alignment", "behavioral_logs")),
    )
)


@dataclass(frozen=True)
class Gate3ExecutionOptions:
    """Non-scientific Gate 3 orchestration switches supplied by the CLI."""

    skip_ica: bool = False
    validation_only_skip_ica: bool = False
    skip_epoching: bool = False
    use_existing_reviewed_ica: bool = False
    allow_incomplete_ica_review_for_validation: bool = False
    clean_subject_output: bool = False
    reviewed_upstream_root: Path | None = None
    corrected_output_root: Path | None = None

    def __post_init__(self) -> None:
        if self.use_existing_reviewed_ica and (self.skip_ica or self.validation_only_skip_ica):
            raise ValueError("--use-existing-reviewed-ica cannot be combined with an ICA skip option.")
        separated = (self.reviewed_upstream_root is not None, self.corrected_output_root is not None)
        if separated[0] != separated[1]:
            raise ValueError("Corrected replay requires both reviewed upstream and corrected output roots.")
        if any(separated):
            if not self.use_existing_reviewed_ica:
                raise ValueError("Separated replay roots require --use-existing-reviewed-ica.")
            if self.clean_subject_output:
                raise ValueError("Corrected replay refuses --clean-subject-output.")
            _validate_separated_roots(
                Path(self.reviewed_upstream_root), Path(self.corrected_output_root)
            )

    @property
    def isolated_replay(self) -> bool:
        return self.reviewed_upstream_root is not None

    @property
    def execution_mode(self) -> str:
        if self.skip_ica and self.validation_only_skip_ica:
            return "gate3a_only_with_validation_config"
        if self.validation_only_skip_ica:
            return "validation_without_ica"
        if self.skip_ica:
            return "gate3a_only"
        if self.skip_epoching:
            return "preprocessing_and_ica_only"
        if self.use_existing_reviewed_ica:
            return "existing_reviewed_ica"
        return "final_preprocessing"

    def to_primitive(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "skip_ica": self.skip_ica,
            "validation_only_skip_ica": self.validation_only_skip_ica,
            "skip_epoching": self.skip_epoching,
            "use_existing_reviewed_ica": self.use_existing_reviewed_ica,
            "allow_incomplete_ica_review_for_validation": self.allow_incomplete_ica_review_for_validation,
            "clean_subject_output": self.clean_subject_output,
            "isolated_replay": self.isolated_replay,
            "reviewed_upstream_root": (
                None if self.reviewed_upstream_root is None else Path(self.reviewed_upstream_root).resolve(strict=False).as_posix()
            ),
            "corrected_output_root": (
                None if self.corrected_output_root is None else Path(self.corrected_output_root).resolve(strict=False).as_posix()
            ),
        }


@dataclass(frozen=True)
class RequestedDataSelection:
    """Selectors as requested by a caller, before resolution/discovery."""

    participants: tuple[str, ...] | None
    recording_parts: tuple[str, ...] | None

    def to_primitive(self) -> dict[str, Any]:
        return {
            "participants": None if self.participants is None else list(self.participants),
            "recording_parts": None if self.recording_parts is None else list(self.recording_parts),
        }


@dataclass(frozen=True)
class RealizedParticipantParts:
    participant: str
    recording_parts: tuple[str, ...]

    def to_primitive(self) -> dict[str, Any]:
        return {"participant": self.participant, "recording_parts": list(self.recording_parts)}


@dataclass(frozen=True)
class RealizedDataSelection:
    """Participants selected by config rules and parts actually discovered."""

    participants: tuple[str, ...]
    recording_parts: tuple[RealizedParticipantParts, ...] = ()

    def parts_for(self, participant: str) -> tuple[str, ...]:
        for item in self.recording_parts:
            if item.participant == participant:
                return item.recording_parts
        return ()

    def to_primitive(self) -> dict[str, Any]:
        return {
            "participants": list(self.participants),
            "recording_parts": [item.to_primitive() for item in self.recording_parts],
        }


@dataclass(frozen=True)
class Session2EEGRunRequest:
    """Immutable request to orchestrate one resolved Kawaii Session 2 run.

    This is execution state, not a persisted run/dataset identity. The resolved
    configuration and its study definition are retained by reference as the
    authoritative objects. Legacy mappings are detached adapters only.
    """

    resolved_config: ResolvedSession2EEGConfig
    requested: RequestedDataSelection
    realized: RealizedDataSelection
    gate3: Gate3ExecutionOptions
    workflow: KawaiiSession2WorkflowDefinition = KAWAII_SESSION2_WORKFLOW
    schema_version: str = RUN_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.gate3.validation_only_skip_ica != self.resolved_config.execution.validation_only_skip_ica:
            raise ValueError("Gate 3 validation mode must match the resolved execution configuration.")
        if self.realized.participants != self.resolved_config.execution.subjects:
            raise ValueError("Realized participants must match resolved participant selection.")
        unknown = [
            item.participant
            for item in self.realized.recording_parts
            if item.participant not in self.realized.participants
        ]
        if unknown:
            raise ValueError(f"Realized recording parts contain unselected participant(s): {', '.join(unknown)}")

    @property
    def study_definition(self) -> KawaiiSession2StudyDefinition:
        return self.resolved_config.study_definition

    @property
    def selected_config_path(self) -> Path:
        return self.resolved_config.source.path

    @property
    def project_root(self) -> Path:
        return self.resolved_config.project_root

    @property
    def output_destination(self) -> Path:
        return self.resolved_config.resolved_paths["output_dir"]

    @property
    def reviewed_upstream_root(self) -> Path | None:
        value = self.gate3.reviewed_upstream_root
        return None if value is None else Path(value).resolve(strict=False)

    def reviewed_upstream_part_dir(self, participant: str, recording_part: str) -> Path:
        if self.reviewed_upstream_root is None:
            raise ValueError("No reviewed upstream root is configured for this run.")
        return self.reviewed_upstream_root / "subjects" / participant / recording_part

    @property
    def data_root(self) -> Path:
        return self.resolved_config.resolved_paths["raw_eeg_dir"]

    @property
    def execution_mode(self) -> str:
        return self.gate3.execution_mode

    def to_legacy_config(self) -> dict[str, Any]:
        """Return a detached adapter for mapping-based stage APIs."""

        return self.resolved_config.to_mutable_config()

    def with_realized_recording_parts(
        self, participant: str, recording_parts: Sequence[str]
    ) -> "Session2EEGRunRequest":
        """Return a new request recording parts actually selected/discovered."""

        participant = str(participant)
        if participant not in self.realized.participants:
            raise ValueError(f"Cannot realize parts for unselected participant {participant!r}.")
        parts = tuple(str(part) for part in recording_parts)
        if any(not part for part in parts) or len(set(parts)) != len(parts):
            raise ValueError("Realized recording parts must be unique non-empty strings.")
        by_participant = {item.participant: item for item in self.realized.recording_parts}
        by_participant[participant] = RealizedParticipantParts(participant, parts)
        ordered = tuple(
            by_participant[selected]
            for selected in self.realized.participants
            if selected in by_participant
        )
        return replace(self, realized=replace(self.realized, recording_parts=ordered))

    def to_primitive(self) -> dict[str, Any]:
        """Return a deterministic, complete in-memory request representation."""

        return {
            "schema_version": self.schema_version,
            "workflow": self.workflow.to_primitive(),
            "resolved_config": self.resolved_config.to_primitive(),
            "requested": self.requested.to_primitive(),
            "realized": self.realized.to_primitive(),
            "gate3": self.gate3.to_primitive(),
            "selected_config_path": self.selected_config_path.as_posix(),
            "project_root": self.project_root.as_posix(),
            "data_root": self.data_root.as_posix(),
            "output_destination": self.output_destination.as_posix(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_primitive(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


def resolve_session2_eeg_run_request(
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    participants: Sequence[str] | None = None,
    recording_parts: Sequence[str] | None = None,
    allow_excluded: bool = False,
    gate3: Gate3ExecutionOptions | None = None,
    workflow: KawaiiSession2WorkflowDefinition = KAWAII_SESSION2_WORKFLOW,
) -> Session2EEGRunRequest:
    """Resolve one authoritative configuration and wrap its execution request."""

    gate3_options = gate3 or Gate3ExecutionOptions()
    requested_participants = None if participants is None else tuple(str(value) for value in participants)
    requested_parts = None if recording_parts is None else tuple(str(value) for value in recording_parts)
    if gate3_options.isolated_replay and requested_participants is None:
        raise ValueError("Corrected replay requires an explicit participant selection.")
    if gate3_options.isolated_replay and requested_parts != ("part1",):
        raise ValueError(
            "This bounded corrected replay requires the exact explicit recording-part selection --parts part1."
        )
    resolved = resolve_session2_eeg_config(
        root=root,
        config_path=config_path,
        overrides=RuntimeConfigOverrides.from_values(
            subjects=requested_participants,
            recording_parts=requested_parts,
            allow_excluded=allow_excluded,
            validation_only_skip_ica=gate3_options.validation_only_skip_ica,
            corrected_output_root=gate3_options.corrected_output_root,
        ),
    )
    return Session2EEGRunRequest(
        resolved_config=resolved,
        requested=RequestedDataSelection(requested_participants, requested_parts),
        realized=RealizedDataSelection(resolved.execution.subjects),
        gate3=gate3_options,
        workflow=workflow,
    )


def _validate_separated_roots(upstream: Path, corrected: Path) -> None:
    upstream_resolved = upstream.resolve(strict=False)
    corrected_resolved = corrected.resolve(strict=False)
    if upstream_resolved == corrected_resolved:
        raise ValueError("Reviewed upstream and corrected output roots must be distinct.")
    if upstream_resolved in corrected_resolved.parents or corrected_resolved in upstream_resolved.parents:
        raise ValueError("Reviewed upstream and corrected output roots must not be nested.")
