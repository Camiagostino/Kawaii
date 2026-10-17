"""Versioned application contracts for Session 2 EEG operations.

These contracts describe automation-visible execution state. They do not
replace persisted scientific or provenance identities and they make no claim
about transactional artifact publication (owned by Phase 4 WS7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable
from uuid import uuid4

from .loading import RecordingPart
from .workflow import Session2EEGRunRequest


OPERATION_CONTRACT_VERSION = "kawaii_session2_operation_v1"


class Operation(str, Enum):
    GATE3A_DECISION_COMPUTE = "gate3a_decision_compute"
    GATE3A_PART = "gate3a_part"
    GATE3B_PART = "gate3b_part"
    GATE3C_PART = "gate3c_part"
    GATE4A_RECOGNITION_CONDITION_COMPUTE = "gate4a_recognition_condition_compute"
    GATE4A_RECOGNITION_PART = "gate4a_recognition_part"
    GATE4B_CUTENESS_PART = "gate4b_cuteness_part"
    PARTICIPANT_GATE3 = "participant_gate3"
    PARTICIPANT_GATE4A = "participant_gate4a"
    PARTICIPANT_GATE4B = "participant_gate4b"
    GATE5A_GROUP_AGGREGATION = "gate5a_group_aggregation"
    GATE5B_GROUP_FIGURES = "gate5b_group_figures"
    GATE5C_GROUP_STATISTICS = "gate5c_group_statistics"
    GROUP_WORKFLOW = "group_workflow"
    RESPONSE_TIMING_QC = "response_timing_qc"
    COORDINATED = "coordinated_operations"


class OperationStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_WARNINGS = "succeeded_with_warnings"
    REVIEW_REQUIRED = "review_required"
    INVALID_INPUT = "invalid_input"
    STALE_INPUT = "stale_input"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ProgressPhase(str, Enum):
    STARTED = "started"
    INPUT_VALIDATED = "input_validated"
    COMPUTE_STARTED = "compute_started"
    COMPUTE_COMPLETED = "compute_completed"
    ARTIFACT_STEP_STARTED = "artifact_step_started"
    ARTIFACT_STEP_COMPLETED = "artifact_step_completed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PARTICIPANT_STARTED = "participant_started"
    PARTICIPANT_COMPLETED = "participant_completed"
    PART_STARTED = "part_started"
    PART_COMPLETED = "part_completed"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"


class ParticipantWorkflow(str, Enum):
    GATE3 = "gate3"
    RECOGNITION = "recognition"
    CUTENESS = "cuteness"

    @property
    def operation(self) -> Operation:
        return {
            ParticipantWorkflow.GATE3: Operation.PARTICIPANT_GATE3,
            ParticipantWorkflow.RECOGNITION: Operation.PARTICIPANT_GATE4A,
            ParticipantWorkflow.CUTENESS: Operation.PARTICIPANT_GATE4B,
        }[self]


class GroupWorkflow(str, Enum):
    PREPARE = "prepare"
    FIGURES = "figures"
    STATISTICS = "statistics"

    @property
    def operation(self) -> Operation:
        return {
            GroupWorkflow.PREPARE: Operation.GATE5A_GROUP_AGGREGATION,
            GroupWorkflow.FIGURES: Operation.GATE5B_GROUP_FIGURES,
            GroupWorkflow.STATISTICS: Operation.GATE5C_GROUP_STATISTICS,
        }[self]


@dataclass(frozen=True)
class ParticipantWorkflowRequest:
    """Exact validated participant workflow scope; discovery remains coordinator-owned."""

    workflow_request: Session2EEGRunRequest
    workflow: ParticipantWorkflow
    request_id: str = field(default_factory=lambda: uuid4().hex)
    session: str = "session2"
    contract_version: str = OPERATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Participant workflow request_id must be non-empty.")
        if self.session != "session2":
            raise ValueError("The Kawaii participant workflow requires session='session2'.")

    @property
    def participants(self) -> tuple[str, ...]:
        return self.workflow_request.realized.participants

    @property
    def requested_recording_parts(self) -> tuple[str, ...] | None:
        return self.workflow_request.requested.recording_parts


@dataclass(frozen=True)
class GroupWorkflowRequest:
    """Validated Gate 5 application scope using the authoritative run request."""

    workflow_request: Session2EEGRunRequest
    workflows: tuple[GroupWorkflow, ...]
    branch: str | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    session: str = "session2"
    contract_version: str = OPERATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Group workflow request_id must be non-empty.")
        if self.session != "session2":
            raise ValueError("The Kawaii group workflow requires session='session2'.")
        if not self.workflows or len(set(self.workflows)) != len(self.workflows):
            raise ValueError("Group workflows must be a non-empty unique sequence.")
        if self.branch not in (None, "recognition", "cuteness"):
            raise ValueError("Group workflow branch must be recognition, cuteness, or omitted.")
        if self.branch is not None and GroupWorkflow.PREPARE in self.workflows:
            raise ValueError("Gate 5A preparation is an atomic two-branch compatibility workflow.")

    @property
    def operation(self) -> Operation:
        return self.workflows[0].operation if len(self.workflows) == 1 else Operation.GROUP_WORKFLOW

    @property
    def participants(self) -> tuple[str, ...]:
        return self.workflow_request.realized.participants


@dataclass(frozen=True)
class TimingWorkflowRequest:
    """Validated timing-QC scope; exact input identities are discovered fail-closed."""

    workflow_request: Session2EEGRunRequest
    request_id: str = field(default_factory=lambda: uuid4().hex)
    session: str = "session2"
    contract_version: str = OPERATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Timing workflow request_id must be non-empty.")
        if self.session != "session2":
            raise ValueError("The Kawaii timing workflow requires session='session2'.")

    @property
    def participants(self) -> tuple[str, ...]:
        return self.workflow_request.realized.participants

    @property
    def requested_recording_parts(self) -> tuple[str, ...] | None:
        return self.workflow_request.requested.recording_parts


@dataclass(frozen=True)
class OperationRequest:
    operation: Operation
    workflow_request: Session2EEGRunRequest
    participant: str | None = None
    recording_part: str | None = None
    branch: str | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    session: str = "session2"
    contract_version: str = OPERATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Operation request_id must be non-empty.")
        if self.session != "session2":
            raise ValueError("The Kawaii Session 2 coordinator requires session='session2'.")
        participant_operations = {
            Operation.GATE3A_PART,
            Operation.GATE3B_PART,
            Operation.GATE3C_PART,
            Operation.GATE4A_RECOGNITION_PART,
            Operation.GATE4B_CUTENESS_PART,
        }
        if self.operation in participant_operations:
            if not self.participant or not self.recording_part:
                raise ValueError(f"{self.operation.value} requires participant and recording_part.")
            if self.participant not in self.workflow_request.realized.participants:
                raise ValueError(f"Participant {self.participant!r} is not selected by the workflow request.")
        if self.operation in {
            Operation.GATE4A_RECOGNITION_CONDITION_COMPUTE,
            Operation.GATE4A_RECOGNITION_PART,
        } and self.branch not in (None, "recognition"):
            raise ValueError("Gate 4A operations require the recognition branch.")
        if self.operation is Operation.GATE4B_CUTENESS_PART and self.branch not in (None, "cuteness"):
            raise ValueError("Gate 4B operations require the cuteness branch.")


@dataclass(frozen=True)
class OperationInputs:
    """Explicit in-process inputs for the representative Batch 1 operations."""

    recording: RecordingPart | None = None
    alignment: Any | None = None
    metadata: Any | None = None
    filtered_raw_path: Path | None = None
    gate3a_bad_channel_decisions_path: Path | None = None
    ica_solution_path: Path | None = None
    ica_fit_identity_path: Path | None = None
    gate3b_eog_suggestions_path: Path | None = None
    gate3b_eog_suggestions_sha256: str | None = None
    gate3b_eog_suggestions_size_bytes: int | None = None
    gate3b_generation_id: str | None = None
    epochs_path: Path | None = None
    epoch_metadata_path: Path | None = None
    ica_bypass_reason: str = ""
    resolved_snapshot: Any | None = None
    ica_summary_cache: Any | None = None
    allow_incomplete_ica_review_for_validation: bool = False
    reviewed_ica_upstream_publications: tuple[Any, ...] = ()
    reviewed_ica_upstream_canonical_directory: Path | None = None
    allow_reviewed_ica_context_reuse: bool = False


@dataclass(frozen=True)
class OperationStateDetails:
    ica_decision_state: str | None = None
    confirmed_ica_components: tuple[int, ...] = ()
    bad_channel_decision_state: str | None = None


@dataclass(frozen=True)
class ArtifactReference:
    label: str
    path: Path
    produced: bool = True


@dataclass(frozen=True)
class ProgressEvent:
    request_id: str
    sequence: int
    operation: Operation
    phase: ProgressPhase
    participant: str | None = None
    recording_part: str | None = None
    branch: str | None = None
    session: str = "session2"
    message: str | None = None


@dataclass(frozen=True)
class OperationResult:
    operation: Operation
    request_id: str
    status: OperationStatus
    reason_code: str
    diagnostic: str
    participant: str | None = None
    recording_part: str | None = None
    branch: str | None = None
    session: str = "session2"
    context_reference: str | None = None
    state_details: OperationStateDetails | None = None
    warnings: tuple[str, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    computation_completed: bool = False
    publication_completed: bool = False
    payload: Any | None = None
    children: tuple["OperationResult", ...] = ()
    developer_details: str | None = None
    contract_version: str = OPERATION_CONTRACT_VERSION

    @property
    def successful(self) -> bool:
        return self.status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS}


@dataclass(frozen=True)
class OperationRejected(Exception):
    """Expected fail-closed outcome carrying a stable client-facing status."""

    status: OperationStatus
    reason_code: str
    diagnostic: str

    def __post_init__(self) -> None:
        if self.status not in {
            OperationStatus.REVIEW_REQUIRED,
            OperationStatus.INVALID_INPUT,
            OperationStatus.STALE_INPUT,
        }:
            raise ValueError("OperationRejected requires an expected fail-closed status.")

    def __str__(self) -> str:
        return self.diagnostic


class CancellationToken:
    """Minimal cooperative token checked only at safe orchestration boundaries."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled()


class OperationCancelled(Exception):
    """Internal cooperative-cancellation signal converted by the coordinator."""


ProgressSink = Callable[[ProgressEvent], None]
