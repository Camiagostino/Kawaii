"""Shared application coordinator for tracked CLI and future backend callers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
import traceback
from typing import Any

from .erp import (
    condition_counts_from_metadata,
    gate4a_recognition_settings,
    run_gate4a_recognition_for_recording_part,
    run_gate4b_cuteness_for_recording_part,
)
from .epoching import run_gate3c_for_recording_part
from .group_aggregation import run_gate5a_group_aggregation
from .group_figures import run_gate5b_branch, run_gate5b_group_figures
from .group_statistics import run_gate5c_branch, run_gate5c_statistics
from .ica import run_gate3b_for_recording_part
from .ica_identity import ICAFitIdentityError, validate_ica_solution_file_binding
from .identity import SourceIdentityError
from .operations import (
    ArtifactReference,
    CancellationToken,
    Operation,
    OperationCancelled,
    OperationInputs,
    OperationRejected,
    OperationRequest,
    OperationResult,
    OperationStateDetails,
    OperationStatus,
    ParticipantWorkflowRequest,
    GroupWorkflowRequest,
    TimingWorkflowRequest,
    ProgressEvent,
    ProgressPhase,
    ProgressSink,
)
from .preprocessing import deterministic_preprocessing_decisions, run_gate3a_for_recording_part
from .paths import subject_part_dir
from .provenance import provenance_sidecar_path
from .publication import (
    PublicationCancelled,
    PublicationError,
    PublicationScope,
    PublicationValidationError,
    publish_artifact_set,
    validate_current_publication,
    validate_immutable_participant_publication_chain,
    validate_participant_publication_chain,
    validate_publication_scientific_context,
    validate_publication_generation,
)
from .resolved_config import ConfigurationValidationError
from .stage_identity import StageIdentityError
from .stage_identity import configuration_mapping_fingerprint, read_validated_participant_erp_artifact
from .study_definition import erp_branch_settings


class Session2EEGCoordinator:
    """Authoritative non-CLI application layer for migrated WS6 operations.

    Batch 1 established representative Gate 3A/Gate 4A operations, Batch 2
    added complete participant Gate 3/4 workflows, and Batch 3 adds Gate 5 and
    timing workflows. Existing scientific runners retain artifact behavior for
    compatibility; the coordinator owns routing and automation-visible state.
    """

    def __init__(
        self,
        *,
        progress_sink: ProgressSink | None = None,
        gate3a_runner: Callable[..., Any] = run_gate3a_for_recording_part,
        gate3b_runner: Callable[..., Any] = run_gate3b_for_recording_part,
        gate3c_runner: Callable[..., Any] = run_gate3c_for_recording_part,
        gate4a_runner: Callable[..., Any] = run_gate4a_recognition_for_recording_part,
        gate4b_runner: Callable[..., Any] = run_gate4b_cuteness_for_recording_part,
        gate5a_runner: Callable[..., Any] = run_gate5a_group_aggregation,
        gate5b_runner: Callable[..., Any] = run_gate5b_group_figures,
        gate5c_runner: Callable[..., Any] = run_gate5c_statistics,
        gate5b_branch_runner: Callable[..., Any] = run_gate5b_branch,
        gate5c_branch_runner: Callable[..., Any] = run_gate5c_branch,
    ) -> None:
        self._progress_sink = progress_sink
        self._gate3a_runner = gate3a_runner
        self._gate3b_runner = gate3b_runner
        self._gate3c_runner = gate3c_runner
        self._gate4a_runner = gate4a_runner
        self._gate4b_runner = gate4b_runner
        self._gate5a_runner = gate5a_runner
        self._gate5b_runner = gate5b_runner
        self._gate5c_runner = gate5c_runner
        self._gate5b_branch_runner = gate5b_branch_runner
        self._gate5c_branch_runner = gate5c_branch_runner

    def execute_participant(
        self,
        request: ParticipantWorkflowRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> OperationResult:
        """Execute one tracked participant workflow without CLI ownership."""

        from .participant_workflow import execute_participant_workflow

        return execute_participant_workflow(self, request, cancellation=cancellation)

    def execute_group(
        self,
        request: GroupWorkflowRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> OperationResult:
        """Execute tracked Gate 5 workflow stages without CLI ownership."""

        from .group_workflow import execute_group_workflow

        return execute_group_workflow(self, request, cancellation=cancellation)

    def execute_timing(
        self,
        request: TimingWorkflowRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> OperationResult:
        """Execute tracked response-timing QC without CLI ownership."""

        from .timing_workflow import execute_timing_workflow

        return execute_timing_workflow(self, request, cancellation=cancellation)

    def execute(
        self,
        request: OperationRequest,
        *,
        inputs: OperationInputs | None = None,
        cancellation: CancellationToken | None = None,
    ) -> OperationResult:
        inputs = inputs or OperationInputs()
        cancellation = cancellation or CancellationToken()
        sequence = 0

        def emit(phase: ProgressPhase, message: str | None = None) -> None:
            nonlocal sequence
            sequence += 1
            if self._progress_sink is not None:
                self._progress_sink(
                    ProgressEvent(
                        request_id=request.request_id,
                        sequence=sequence,
                        operation=request.operation,
                        phase=phase,
                        participant=request.participant,
                        recording_part=request.recording_part,
                        branch=request.branch,
                        session=request.session,
                        message=message,
                    )
                )

        emit(ProgressPhase.STARTED)
        try:
            cancellation.raise_if_cancelled()
            self._validate_inputs(request, inputs)
            inputs = self._bind_current_participant_inputs(request, inputs)
            emit(ProgressPhase.INPUT_VALIDATED)
            cancellation.raise_if_cancelled()
            emit(ProgressPhase.COMPUTE_STARTED)
            writes_artifacts = request.operation in {
                Operation.GATE3A_PART,
                Operation.GATE3B_PART,
                Operation.GATE3C_PART,
                Operation.GATE4A_RECOGNITION_PART,
                Operation.GATE4B_CUTENESS_PART,
            }
            if writes_artifacts:
                emit(ProgressPhase.ARTIFACT_STEP_STARTED, "Existing compatibility runner owns compute and artifact writes.")
            value = self._invoke(request, inputs)
            emit(ProgressPhase.COMPUTE_COMPLETED)
            if writes_artifacts:
                emit(ProgressPhase.ARTIFACT_STEP_COMPLETED)
            result = self._normalise_result(request, value, writes_artifacts=writes_artifacts)
            if cancellation.cancelled:
                emit(ProgressPhase.CANCELLED)
                return replace(
                    result,
                    status=OperationStatus.CANCELLED,
                    reason_code="operation_cancelled_after_core",
                    diagnostic=(
                        "Cancellation was observed after the existing runner returned; "
                        "reported artifacts and completion flags describe work already performed."
                    ),
                )
            if request.operation in {
                Operation.GATE3A_PART,
                Operation.GATE3B_PART,
                Operation.GATE3C_PART,
                Operation.GATE4A_RECOGNITION_PART,
                Operation.GATE4B_CUTENESS_PART,
            } and result.successful:
                result = self._publish_participant_stage(request, inputs, result, cancellation)
            emit(ProgressPhase.COMPLETED, result.status.value)
            return result
        except OperationCancelled:
            emit(ProgressPhase.CANCELLED)
            return self._result(
                request,
                OperationStatus.CANCELLED,
                "operation_cancelled",
                "Cancellation was observed at a safe orchestration boundary.",
            )
        except PublicationCancelled:
            emit(ProgressPhase.CANCELLED)
            return self._result(
                request,
                OperationStatus.CANCELLED,
                "publication_cancelled",
                "Cancellation was observed before canonical publication finality.",
                computation_completed=True,
                publication_completed=False,
            )
        except PublicationError as error:
            emit(ProgressPhase.COMPLETED, "publication_invalid_or_stale")
            return self._result(
                request,
                OperationStatus.STALE_INPUT,
                "publication_invalid_or_stale",
                str(error),
                computation_completed=True,
                publication_completed=False,
            )
        except OperationRejected as error:
            emit(ProgressPhase.COMPLETED, error.reason_code)
            return self._result(request, error.status, error.reason_code, error.diagnostic)
        except ConfigurationValidationError as error:
            emit(ProgressPhase.COMPLETED, "configuration_invalid")
            return self._result(request, OperationStatus.INVALID_INPUT, "configuration_invalid", str(error))
        except (SourceIdentityError, StageIdentityError, ICAFitIdentityError) as error:
            emit(ProgressPhase.COMPLETED, "identity_stale_or_invalid")
            return self._result(request, OperationStatus.STALE_INPUT, "identity_stale_or_invalid", str(error))
        except Exception as error:  # coordinator boundary: preserve details, never claim success
            emit(ProgressPhase.FAILED, type(error).__name__)
            return self._result(
                request,
                OperationStatus.FAILED,
                "unexpected_exception",
                f"Unexpected {type(error).__name__}: {error}",
                developer_details="".join(traceback.format_exception(error)),
            )

    def coordinate(
        self,
        requests: Sequence[OperationRequest],
        *,
        inputs: Sequence[OperationInputs] | None = None,
        cancellation: CancellationToken | None = None,
    ) -> OperationResult:
        if not requests:
            raise ValueError("At least one child operation is required.")
        child_inputs = tuple(inputs or (OperationInputs() for _ in requests))
        if len(child_inputs) != len(requests):
            raise ValueError("Each child request requires one OperationInputs value.")
        token = cancellation or CancellationToken()
        children = tuple(
            self.execute(request, inputs=item_inputs, cancellation=token)
            for request, item_inputs in zip(requests, child_inputs)
        )
        status = _aggregate_status(children)
        parent_id = "coordinated:" + ",".join(child.request_id for child in children)
        return OperationResult(
            operation=Operation.COORDINATED,
            request_id=parent_id,
            status=status,
            reason_code="children_complete" if status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS} else "child_operation_not_successful",
            diagnostic=f"{len(children)} child operation(s); parent status reflects every child.",
            session=children[0].session,
            context_reference=(
                children[0].context_reference
                if len({child.context_reference for child in children}) == 1
                else None
            ),
            warnings=tuple(warning for child in children for warning in child.warnings),
            artifacts=tuple(artifact for child in children for artifact in child.artifacts),
            computation_completed=all(child.computation_completed for child in children),
            publication_completed=all(child.publication_completed for child in children),
            children=children,
        )

    @staticmethod
    def _validate_inputs(request: OperationRequest, inputs: OperationInputs) -> None:
        if request.operation in {Operation.GATE3A_PART, Operation.GATE3B_PART, Operation.GATE3C_PART}:
            if inputs.recording is None:
                raise OperationRejected(OperationStatus.INVALID_INPUT, "recording_required", f"{request.operation.value} requires an explicit validated RecordingPart.")
            if (inputs.recording.subject, inputs.recording.recording_part) != (request.participant, request.recording_part):
                raise OperationRejected(OperationStatus.INVALID_INPUT, "recording_scope_mismatch", "RecordingPart identity does not match the operation request.")
        if request.operation is Operation.GATE4A_RECOGNITION_CONDITION_COMPUTE and inputs.metadata is None:
            raise OperationRejected(OperationStatus.INVALID_INPUT, "metadata_required", "Recognition condition computation requires explicit metadata.")
        if (
            inputs.reviewed_ica_upstream_publications
            or inputs.reviewed_ica_upstream_canonical_directory is not None
            or inputs.allow_reviewed_ica_context_reuse
        ):
            if (
                request.operation is not Operation.GATE3C_PART
                or not request.workflow_request.gate3.use_existing_reviewed_ica
                or not inputs.reviewed_ica_upstream_publications
                or inputs.reviewed_ica_upstream_canonical_directory is None
                or not inputs.allow_reviewed_ica_context_reuse
            ):
                raise OperationRejected(
                    OperationStatus.INVALID_INPUT,
                    "reviewed_ica_reuse_contract_invalid",
                    "Immutable reviewed-ICA lineage inputs are valid only for the explicit Gate 3C reuse route.",
                )

    @staticmethod
    def _bind_current_participant_inputs(
        request: OperationRequest,
        inputs: OperationInputs,
    ) -> OperationInputs:
        """Resolve generated inputs only through current validated publications."""

        if (
            request.workflow_request.gate3.validation_only_skip_ica
            and request.operation in {Operation.GATE3B_PART, Operation.GATE3C_PART}
        ):
            return inputs
        stages = {
            Operation.GATE3B_PART: ("gate3a",),
            Operation.GATE3C_PART: ("gate3a", "gate3b"),
            Operation.GATE4A_RECOGNITION_PART: ("gate3c",),
            Operation.GATE4B_CUTENESS_PART: ("gate3c",),
        }.get(request.operation)
        if stages is None:
            return inputs
        canonical = subject_part_dir(
            request.workflow_request.to_legacy_config(),
            str(request.participant),
            str(request.recording_part),
            root=request.workflow_request.project_root,
            create=False,
        )
        context = request.workflow_request.resolved_config.resolved_config_fingerprint
        if inputs.reviewed_ica_upstream_publications:
            upstream_canonical = Path(inputs.reviewed_ica_upstream_canonical_directory).resolve()
            supplied = {publication.scope.stage: publication for publication in inputs.reviewed_ica_upstream_publications}
            if set(supplied) != set(stages):
                raise PublicationValidationError(
                    "Reviewed-ICA reuse requires exact immutable Gate 3A and Gate 3B publications."
                )
            publications = {}
            for stage in stages:
                candidate = supplied[stage]
                scope = PublicationScope(
                    stage=stage,
                    session=request.session,
                    participant=request.participant,
                    recording_part=request.recording_part,
                )
                if candidate.scope != scope:
                    raise PublicationValidationError(
                        f"Reviewed-ICA immutable {stage} publication scope is mismatched."
                    )
                publication = validate_publication_generation(
                    upstream_canonical,
                    scope,
                    candidate.generation_id,
                    expected_context_reference=candidate.context_reference,
                )
                validate_immutable_participant_publication_chain(upstream_canonical, publication)
                publications[stage] = publication
            return Session2EEGCoordinator._bind_participant_publication_paths(inputs, publications)
        publications = {}
        for stage in stages:
            stage_scope = PublicationScope(
                stage=stage,
                session=request.session,
                participant=request.participant,
                recording_part=request.recording_part,
            )
            allow_scientific_cross_context = (
                stage == "gate3c"
                and request.operation
                in {Operation.GATE4A_RECOGNITION_PART, Operation.GATE4B_CUTENESS_PART}
            )
            publication = validate_current_publication(
                canonical,
                stage_scope,
                expected_context_reference=(None if allow_scientific_cross_context else context),
            )
            validate_participant_publication_chain(
                canonical,
                publication,
                immutable_upstream_canonical_directory=(
                    request.workflow_request.reviewed_upstream_part_dir(
                        str(request.participant), str(request.recording_part)
                    )
                    if request.workflow_request.gate3.isolated_replay
                    else None
                ),
            )
            if allow_scientific_cross_context and publication.context_reference != context:
                validate_publication_scientific_context(
                    publication,
                    request.workflow_request.resolved_config.scientific_fingerprint,
                )
            publications[stage] = publication

        return Session2EEGCoordinator._bind_participant_publication_paths(inputs, publications)

    @staticmethod
    def _bind_participant_publication_paths(inputs: OperationInputs, publications) -> OperationInputs:
        replacements: dict[str, Path] = {}
        if "gate3a" in publications:
            replacements["filtered_raw_path"] = Session2EEGCoordinator._required_published_artifact(
                publications["gate3a"], "filtered_raw_fif"
            )
            bad_decisions = publications["gate3a"].artifact_path("bad_channel_decisions")
            if bad_decisions is not None:
                replacements["gate3a_bad_channel_decisions_path"] = Path(bad_decisions)
        if "gate3b" in publications:
            replacements["ica_solution_path"] = Session2EEGCoordinator._required_published_artifact(
                publications["gate3b"], "ica_solution"
            )
            replacements["ica_fit_identity_path"] = Session2EEGCoordinator._required_published_artifact(
                publications["gate3b"], "ica_fit_identity"
            )
            suggestions = publications["gate3b"].artifact_path("eog_suggestions")
            if suggestions is not None:
                replacements["gate3b_eog_suggestions_path"] = Path(suggestions)
        if "gate3c" in publications:
            replacements["epochs_path"] = Session2EEGCoordinator._required_published_artifact(
                publications["gate3c"], "epochs_fif"
            )
            replacements["epoch_metadata_path"] = Session2EEGCoordinator._required_published_artifact(
                publications["gate3c"], "epoch_metadata_csv"
            )
        for field, path in replacements.items():
            supplied = getattr(inputs, field)
            if supplied is not None and Path(supplied).resolve() != path.resolve():
                raise PublicationValidationError(
                    f"Explicit {field} does not match the current published generation."
                )
        gate3b = publications.get("gate3b")
        suggestion_artifact = next(
            (
                artifact
                for artifact in getattr(gate3b, "artifacts", ())
                if getattr(artifact, "label", None) == "eog_suggestions"
            ),
            None,
        )
        return replace(
            inputs,
            **replacements,
            gate3b_eog_suggestions_sha256=(
                None if suggestion_artifact is None else str(suggestion_artifact.sha256)
            ),
            gate3b_eog_suggestions_size_bytes=(
                None if suggestion_artifact is None else int(suggestion_artifact.size_bytes)
            ),
            gate3b_generation_id=(
                None if gate3b is None else str(gate3b.generation_id)
            ),
        )

    @staticmethod
    def _required_published_artifact(publication, label: str) -> Path:
        path = publication.artifact_path(label)
        if path is None:
            raise PublicationValidationError(
                f"Current {publication.scope.stage} publication lacks required artifact {label!r}."
            )
        return Path(path)

    def _invoke(self, request: OperationRequest, inputs: OperationInputs) -> Any:
        workflow = request.workflow_request
        config = workflow.to_legacy_config()
        if request.operation is Operation.GATE3A_DECISION_COMPUTE:
            return deterministic_preprocessing_decisions(config)
        if request.operation is Operation.GATE4A_RECOGNITION_CONDITION_COMPUTE:
            return condition_counts_from_metadata(inputs.metadata, gate4a_recognition_settings(config))
        if request.operation is Operation.GATE3A_PART:
            return self._gate3a_runner(
                request.participant,
                inputs.recording,
                config,
                alignment=inputs.alignment,
                root=workflow.project_root,
                config_path=workflow.selected_config_path,
                resolved_snapshot=inputs.resolved_snapshot,
                execution_run_id=request.request_id,
            )
        if request.operation is Operation.GATE3B_PART:
            gate3b_kwargs = dict(
                alignment=inputs.alignment,
                filtered_raw_path=inputs.filtered_raw_path,
                root=workflow.project_root,
                config_path=workflow.selected_config_path,
                resolved_snapshot=inputs.resolved_snapshot,
            )
            if inputs.ica_summary_cache is not None:
                gate3b_kwargs["ica_summary_cache"] = inputs.ica_summary_cache
            return self._gate3b_runner(
                request.participant,
                inputs.recording,
                config,
                **gate3b_kwargs,
            )
        if request.operation is Operation.GATE3C_PART:
            return self._gate3c_runner(
                request.participant,
                inputs.recording,
                config,
                alignment=inputs.alignment,
                filtered_raw_path=inputs.filtered_raw_path,
                gate3a_bad_channel_decisions_path=inputs.gate3a_bad_channel_decisions_path,
                ica_solution_path=inputs.ica_solution_path,
                ica_fit_identity_path_override=inputs.ica_fit_identity_path,
                gate3b_eog_suggestions_path=inputs.gate3b_eog_suggestions_path,
                gate3b_eog_suggestions_sha256=inputs.gate3b_eog_suggestions_sha256,
                gate3b_eog_suggestions_size_bytes=inputs.gate3b_eog_suggestions_size_bytes,
                gate3b_generation_id=inputs.gate3b_generation_id,
                ica_bypass_reason=inputs.ica_bypass_reason,
                allow_incomplete_ica_review_for_validation=inputs.allow_incomplete_ica_review_for_validation,
                allow_reviewed_ica_context_reuse=inputs.allow_reviewed_ica_context_reuse,
                replay_root_provenance=(
                    {
                        "historical_upstream_read_root": str(
                            request.workflow_request.reviewed_upstream_root
                        ),
                        "corrected_output_write_root": str(
                            request.workflow_request.output_destination.resolve(strict=False)
                        ),
                        "manual_review_authority_root": str(
                            request.workflow_request.resolved_config.resolved_paths["manual_review_dir"]
                        ),
                    }
                    if request.workflow_request.gate3.isolated_replay
                    else None
                ),
                root=workflow.project_root,
                config_path=workflow.selected_config_path,
                resolved_snapshot=inputs.resolved_snapshot,
                execution_run_id=request.request_id,
            )
        if request.operation is Operation.GATE4A_RECOGNITION_PART:
            return self._gate4a_runner(
                request.participant,
                request.recording_part,
                config,
                root=workflow.project_root,
                config_path=workflow.selected_config_path,
                published_input_paths=(
                    {
                        "epochs_fif": inputs.epochs_path,
                        "epoch_metadata_csv": inputs.epoch_metadata_path,
                    }
                    if inputs.epochs_path is not None and inputs.epoch_metadata_path is not None
                    else None
                ),
            )
        if request.operation is Operation.GATE4B_CUTENESS_PART:
            return self._gate4b_runner(
                request.participant,
                request.recording_part,
                config,
                root=workflow.project_root,
                config_path=workflow.selected_config_path,
                published_input_paths=(
                    {
                        "epochs_fif": inputs.epochs_path,
                        "epoch_metadata_csv": inputs.epoch_metadata_path,
                    }
                    if inputs.epochs_path is not None and inputs.epoch_metadata_path is not None
                    else None
                ),
            )
        raise OperationRejected(OperationStatus.INVALID_INPUT, "unsupported_operation", f"Operation {request.operation.value!r} is not routed by the coordinator.")

    @staticmethod
    def _publish_participant_stage(
        request: OperationRequest,
        inputs: OperationInputs,
        result: OperationResult,
        cancellation: CancellationToken,
    ) -> OperationResult:
        """Publish and rebind one complete participant-stage artifact set."""

        validation_only = (
            request.workflow_request.gate3.validation_only_skip_ica
            or (
                request.operation is Operation.GATE3C_PART
                and bool(
                    inputs.ica_bypass_reason
                    or inputs.allow_incomplete_ica_review_for_validation
                )
            )
        )
        if validation_only:
            return replace(
                result,
                reason_code="completed_validation_only_nonfinal",
                diagnostic=(
                    f"{request.operation.value} validation-only artifacts are deliberately non-final and were not "
                    "published as the canonical generation."
                ),
                publication_completed=False,
            )
        sources = {artifact.label: artifact.path for artifact in result.artifacts if artifact.produced}
        if not sources:
            raise PublicationError(
                f"{request.operation.value} reported success without required publication artifacts."
            )
        required_labels = {
            Operation.GATE3A_PART: {"filtered_raw_fif"},
            Operation.GATE3B_PART: {"ica_solution", "ica_fit_identity"},
            Operation.GATE3C_PART: {"epochs_fif", "epoch_metadata_csv"},
            Operation.GATE4A_RECOGNITION_PART: {
                "condition_counts",
                "recognition_measurements",
                "recognition_measurements_wide",
                "recognition_qc_summary",
                "participant_erp_identity",
            },
            Operation.GATE4B_CUTENESS_PART: {
                "condition_counts",
                "cuteness_measurements",
                "cuteness_measurements_wide",
                "cuteness_qc_summary",
                "cuteness_trial_level",
                "participant_erp_identity",
            },
        }[request.operation]
        missing_labels = sorted(required_labels.difference(sources))
        if missing_labels:
            raise PublicationError(
                f"{request.operation.value} lacks required publication artifact(s): "
                + ", ".join(missing_labels)
            )
        context = request.workflow_request.resolved_config.resolved_config_fingerprint
        config = request.workflow_request.to_legacy_config()
        if request.operation is Operation.GATE3B_PART:
            validate_ica_solution_file_binding(
                sources["ica_fit_identity"],
                solution_path=sources["ica_solution"],
                expected_participant=str(request.participant),
                expected_recording_part=str(request.recording_part),
                expected_resolved_config_fingerprint=context,
            )
        if request.operation in {Operation.GATE4A_RECOGNITION_PART, Operation.GATE4B_CUTENESS_PART}:
            branch = "recognition" if request.operation is Operation.GATE4A_RECOGNITION_PART else "cuteness"
            identity_labels = {
                "condition_counts": sources["condition_counts"],
                f"{branch}_measurements": sources[f"{branch}_measurements"],
                f"{branch}_measurements_wide": sources[f"{branch}_measurements_wide"],
                f"{branch}_qc_summary": sources[f"{branch}_qc_summary"],
            }
            if branch == "cuteness":
                identity_labels["cuteness_trial_level"] = sources["cuteness_trial_level"]
            read_validated_participant_erp_artifact(
                sources["participant_erp_identity"],
                expected_participant=str(request.participant),
                expected_recording_part=str(request.recording_part),
                expected_branch=branch,
                expected_configuration_fingerprint=configuration_mapping_fingerprint(config),
                expected_branch_settings=erp_branch_settings(config, branch),
                current_output_paths=identity_labels,
            )
        provenance_stages = {
            Operation.GATE3A_PART: ("gate3a_preprocessing", "gate3a_qc_figure"),
            Operation.GATE3B_PART: ("gate3b_ica", "gate3b_ica_diagnostic_figure"),
            Operation.GATE3C_PART: ("gate3c_ica_cleaning_epoching", "gate3c_epoch_qc_figure"),
            Operation.GATE4A_RECOGNITION_PART: ("gate4a_recognition_erp", "gate4_recognition_erp_figure"),
            Operation.GATE4B_CUTENESS_PART: ("gate4a_cuteness_erp", "gate4_cuteness_erp_figure"),
        }
        for label, path in tuple(sources.items()):
            ordinary_stage, figure_stage = provenance_stages[request.operation]
            pipeline_stage = figure_stage if Path(path).suffix.lower() == ".png" else ordinary_stage
            provenance = provenance_sidecar_path(path, pipeline_stage)
            if provenance.is_file():
                sources[f"{label}__provenance"] = provenance
        stage_name = {
            Operation.GATE3A_PART: "gate3a",
            Operation.GATE3B_PART: "gate3b",
            Operation.GATE3C_PART: "gate3c",
            Operation.GATE4A_RECOGNITION_PART: "gate4a",
            Operation.GATE4B_CUTENESS_PART: "gate4b",
        }[request.operation]
        scope = PublicationScope(
            stage=stage_name,
            session=request.session,
            participant=request.participant,
            recording_part=request.recording_part,
            branch=request.branch if stage_name in {"gate4a", "gate4b"} else None,
        )
        canonical_directory = subject_part_dir(
            config,
            str(request.participant),
            str(request.recording_part),
            root=request.workflow_request.project_root,
            create=True,
        )
        upstream_stages = {
            Operation.GATE3A_PART: (),
            Operation.GATE3B_PART: ("gate3a",),
            Operation.GATE3C_PART: ("gate3a", "gate3b"),
            Operation.GATE4A_RECOGNITION_PART: ("gate3c",),
            Operation.GATE4B_CUTENESS_PART: ("gate3c",),
        }[request.operation]
        identity_references: dict[str, str] = {}
        reviewed_ica_upstreams = {
            publication.scope.stage: publication
            for publication in inputs.reviewed_ica_upstream_publications
        }
        for upstream_stage in upstream_stages:
            upstream_scope = PublicationScope(
                stage=upstream_stage,
                session=request.session,
                participant=request.participant,
                recording_part=request.recording_part,
            )
            if reviewed_ica_upstreams:
                upstream = reviewed_ica_upstreams.get(upstream_stage)
                if upstream is None or upstream.scope != upstream_scope:
                    raise PublicationValidationError(
                        f"Reviewed-ICA publication dependency is missing or mismatched: {upstream_stage}."
                    )
                upstream = validate_publication_generation(
                    Path(inputs.reviewed_ica_upstream_canonical_directory).resolve(),
                    upstream_scope,
                    upstream.generation_id,
                    expected_context_reference=upstream.context_reference,
                )
                validate_immutable_participant_publication_chain(
                    Path(inputs.reviewed_ica_upstream_canonical_directory).resolve(), upstream
                )
                prefix = "immutable-publication"
            else:
                allow_scientific_cross_context = (
                    upstream_stage == "gate3c"
                    and request.operation
                    in {Operation.GATE4A_RECOGNITION_PART, Operation.GATE4B_CUTENESS_PART}
                )
                upstream = validate_current_publication(
                    canonical_directory,
                    upstream_scope,
                    expected_context_reference=(None if allow_scientific_cross_context else context),
                )
                if allow_scientific_cross_context and upstream.context_reference != context:
                    validate_participant_publication_chain(
                        canonical_directory,
                        upstream,
                        immutable_upstream_canonical_directory=(
                            request.workflow_request.reviewed_upstream_part_dir(
                                str(request.participant), str(request.recording_part)
                            )
                            if request.workflow_request.gate3.isolated_replay
                            else None
                        ),
                    )
                    validate_publication_scientific_context(
                        upstream,
                        request.workflow_request.resolved_config.scientific_fingerprint,
                    )
                    prefix = "immutable-publication"
                else:
                    prefix = "publication"
            identity_references[f"{prefix}:{upstream_stage}:{upstream_scope.identity}"] = upstream.generation_id
        published = publish_artifact_set(
            canonical_directory,
            scope,
            run_id=request.request_id,
            context_reference=context,
            required_artifacts=sources,
            cancelled=lambda: cancellation.cancelled,
            identity_references=identity_references,
        )
        rebound = tuple(
            ArtifactReference(artifact.label, artifact.path)
            for artifact in published.artifacts
        )
        rebound += (ArtifactReference("publication_finality_manifest", published.manifest_path),)
        return replace(
            result,
            artifacts=rebound,
            publication_completed=True,
        )

    def _normalise_result(self, request: OperationRequest, value: Any, *, writes_artifacts: bool) -> OperationResult:
        if not writes_artifacts:
            return self._result(
                request,
                OperationStatus.SUCCEEDED,
                "compute_completed",
                "Write-free domain computation completed.",
                computation_completed=True,
                payload=value,
            )
        warnings = tuple(str(item) for item in getattr(value, "warnings", ()) or ())
        failures = tuple(str(item) for item in getattr(value, "failures", ()) or ())
        artifacts = tuple(
            ArtifactReference(str(label), path)
            for label, path in (getattr(value, "output_files", {}) or {}).items()
        )
        low_level_status = str(getattr(value, "status", "")).lower()
        diagnostic = "; ".join(failures) or low_level_status
        diagnostic_lower = diagnostic.lower()
        if low_level_status == "ok":
            status = OperationStatus.SUCCEEDED_WITH_WARNINGS if warnings else OperationStatus.SUCCEEDED
            reason = "completed_with_warnings" if warnings else "completed"
        elif "stale" in low_level_status or "stale" in diagnostic_lower or "mismatch" in diagnostic_lower:
            status, reason = OperationStatus.STALE_INPUT, "stale_or_mismatched_input"
        elif (
            "review" in low_level_status
            or "review is" in diagnostic_lower
            or "review-required" in diagnostic_lower
            or "application is blocked" in diagnostic_lower
        ):
            status, reason = OperationStatus.REVIEW_REQUIRED, "manual_review_required"
        elif "invalid" in low_level_status or "validation" in low_level_status:
            status, reason = OperationStatus.INVALID_INPUT, "invalid_input"
        else:
            status, reason = OperationStatus.FAILED, "core_reported_failure"
        state_details = OperationStateDetails(
            ica_decision_state=getattr(value, "ica_decision_state", None),
            confirmed_ica_components=tuple(getattr(value, "confirmed_ica_components", ()) or ()),
            bad_channel_decision_state=getattr(value, "bad_channel_decision_state", None),
        )
        if status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS}:
            if state_details.ica_decision_state == "REVIEWED_NO_ICA_COMPONENTS":
                reason = "completed_reviewed_zero" + ("_with_warnings" if warnings else "")
            elif state_details.ica_decision_state == "REVIEWED_WITH_ICA_COMPONENTS":
                reason = "completed_reviewed_with_exclusions" + ("_with_warnings" if warnings else "")
        diagnostic = diagnostic or status.value
        return self._result(
            request,
            status,
            reason,
            diagnostic,
            warnings=warnings,
            artifacts=artifacts,
            computation_completed=status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS},
            publication_completed=status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS},
            payload=value,
            state_details=state_details,
        )

    @staticmethod
    def _result(
        request: OperationRequest,
        status: OperationStatus,
        reason_code: str,
        diagnostic: str,
        **kwargs: Any,
    ) -> OperationResult:
        return OperationResult(
            operation=request.operation,
            request_id=request.request_id,
            status=status,
            reason_code=reason_code,
            diagnostic=diagnostic,
            participant=request.participant,
            recording_part=request.recording_part,
            branch=request.branch,
            session=request.session,
            context_reference=request.workflow_request.resolved_config.resolved_config_fingerprint,
            **kwargs,
        )


def _aggregate_status(children: Sequence[OperationResult]) -> OperationStatus:
    statuses = {child.status for child in children}
    for status in (
        OperationStatus.FAILED,
        OperationStatus.STALE_INPUT,
        OperationStatus.INVALID_INPUT,
        OperationStatus.REVIEW_REQUIRED,
        OperationStatus.CANCELLED,
    ):
        if status in statuses:
            return status
    if OperationStatus.SUCCEEDED_WITH_WARNINGS in statuses:
        return OperationStatus.SUCCEEDED_WITH_WARNINGS
    return OperationStatus.SUCCEEDED
