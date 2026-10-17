"""Tracked Gate 5 application routing and automation-visible status policy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import traceback
from typing import Any, TYPE_CHECKING

import pandas as pd

from .group_aggregation import validate_gate4_publication_for_group
from .group_figures import Gate5BResult
from .group_statistics import Gate5CResult
from .ica_identity import ICAFitIdentityError
from .identity import SourceIdentityError
from .operations import (
    ArtifactReference,
    CancellationToken,
    GroupWorkflow,
    GroupWorkflowRequest,
    Operation,
    OperationResult,
    OperationStatus,
    ProgressEvent,
    ProgressPhase,
)
from .paths import resolve_session2_eeg_paths, subject_part_dir
from .publication import (
    PublicationCancelled,
    PublicationError,
    PublicationScope,
    publish_artifact_set,
    validate_current_publication,
)
from .resolved_config import ConfigurationValidationError
from .stage_identity import StageIdentityError

if TYPE_CHECKING:
    from .coordinator import Session2EEGCoordinator


def execute_group_workflow(
    coordinator: "Session2EEGCoordinator",
    request: GroupWorkflowRequest,
    *,
    cancellation: CancellationToken | None = None,
) -> OperationResult:
    """Execute requested Gate 5 stages through the shared coordinator boundary."""

    token = cancellation or CancellationToken()
    sequence = 0

    def emit(phase: ProgressPhase, operation: Operation, message: str | None = None) -> None:
        nonlocal sequence
        sequence += 1
        if coordinator._progress_sink is not None:
            coordinator._progress_sink(
                ProgressEvent(
                    request_id=request.request_id,
                    sequence=sequence,
                    operation=operation,
                    phase=phase,
                    branch=request.branch,
                    session=request.session,
                    message=message,
                )
            )

    emit(ProgressPhase.STARTED, request.operation)
    children: list[OperationResult] = []
    for workflow in request.workflows:
        operation = workflow.operation
        if token.cancelled:
            children.append(_cancelled(request, operation, "Cancelled before the requested Gate 5 stage."))
            emit(ProgressPhase.CANCELLED, operation)
            break
        emit(ProgressPhase.STAGE_STARTED, operation)
        child = _execute_stage(coordinator, request, workflow, token)
        if token.cancelled:
            child = replace(
                child,
                status=OperationStatus.CANCELLED,
                reason_code="operation_cancelled_after_core",
                diagnostic=(
                    "Cancellation was observed after the existing Gate 5 runner returned; "
                    "reported artifacts describe work already performed."
                ),
            )
        children.append(child)
        emit(ProgressPhase.STAGE_COMPLETED, operation, child.status.value)
        if token.cancelled:
            emit(ProgressPhase.CANCELLED, operation)
            break

    result = children[0] if len(children) == 1 else _aggregate(request, tuple(children))
    emit(ProgressPhase.COMPLETED, result.operation, result.status.value)
    return result


def _execute_stage(
    coordinator: "Session2EEGCoordinator",
    request: GroupWorkflowRequest,
    workflow: GroupWorkflow,
    token: CancellationToken,
) -> OperationResult:
    run_request = request.workflow_request
    config = run_request.to_legacy_config()
    kwargs = {
        "root": run_request.project_root,
        "config_path": run_request.selected_config_path,
    }
    try:
        context = run_request.resolved_config.resolved_config_fingerprint
        published_inputs_by_branch: dict[str, dict[str, Path]] | None = None
        gate5a_publication = None
        if workflow in {GroupWorkflow.FIGURES, GroupWorkflow.STATISTICS}:
            gate5a_publication = _validated_gate5a_publication(request)
            published_inputs_by_branch = {
                branch: _gate5a_branch_inputs(gate5a_publication, branch)
                for branch in ("recognition", "cuteness")
                if request.branch in (None, branch)
            }
        if workflow is GroupWorkflow.PREPARE:
            value = coordinator._gate5a_runner(
                config,
                publication_context_reference=context,
                publication_scientific_fingerprint=(
                    run_request.resolved_config.scientific_fingerprint
                ),
                **(
                    {"immutable_upstream_root": run_request.reviewed_upstream_root}
                    if run_request.gate3.isolated_replay
                    else {}
                ),
                **kwargs,
            )
        elif workflow is GroupWorkflow.FIGURES:
            if request.branch is None:
                value = coordinator._gate5b_runner(
                    config,
                    published_input_paths_by_branch=published_inputs_by_branch,
                    publication_context_reference=context,
                    publication_scientific_fingerprint=(
                        run_request.resolved_config.scientific_fingerprint
                    ),
                    expected_participant_publications=gate5a_publication.identity_references,
                    **(
                        {"immutable_upstream_root": run_request.reviewed_upstream_root}
                        if run_request.gate3.isolated_replay
                        else {}
                    ),
                    **kwargs,
                )
            else:
                warnings: list[str] = []
                info: list[str] = []
                outputs = coordinator._gate5b_branch_runner(
                    config,
                    request.branch,
                    warnings=warnings,
                    info_messages=info,
                    published_input_paths=published_inputs_by_branch[request.branch],
                    publication_context_reference=context,
                    publication_scientific_fingerprint=(
                        run_request.resolved_config.scientific_fingerprint
                    ),
                    expected_participant_publications=gate5a_publication.identity_references,
                    **(
                        {"immutable_upstream_root": run_request.reviewed_upstream_root}
                        if run_request.gate3.isolated_replay
                        else {}
                    ),
                    **kwargs,
                )
                value = Gate5BResult(outputs, tuple(warnings), tuple(info))
        else:
            if request.branch is None:
                value = coordinator._gate5c_runner(
                    config,
                    published_input_paths_by_branch=published_inputs_by_branch,
                    **kwargs,
                )
            else:
                warnings = []
                outputs = coordinator._gate5c_branch_runner(
                    config,
                    request.branch,
                    warnings=warnings,
                    published_input_paths=published_inputs_by_branch[request.branch],
                    **kwargs,
                )
                value = Gate5CResult(outputs, tuple(warnings))
        result = _normalise(request, workflow, value)
        if result.successful:
            if token.cancelled:
                raise PublicationCancelled("Group publication cancelled before canonical finality.")
            result = _publish_group_result(
                request,
                workflow,
                result,
                value,
                gate5a_publication,
                token,
            )
        return result
    except ConfigurationValidationError as error:
        return _result(request, workflow.operation, OperationStatus.INVALID_INPUT, "configuration_invalid", str(error))
    except (SourceIdentityError, StageIdentityError, ICAFitIdentityError) as error:
        return _result(request, workflow.operation, OperationStatus.STALE_INPUT, "identity_stale_or_invalid", str(error))
    except PublicationCancelled as error:
        return _result(request, workflow.operation, OperationStatus.CANCELLED, "publication_cancelled", str(error), computation_completed=True, publication_completed=False)
    except PublicationError as error:
        return _result(request, workflow.operation, OperationStatus.STALE_INPUT, "publication_invalid_or_stale", str(error), computation_completed=True, publication_completed=False)
    except (FileNotFoundError, ValueError) as error:
        return _result(request, workflow.operation, OperationStatus.INVALID_INPUT, "required_group_input_invalid", str(error))
    except Exception as error:
        return _result(
            request,
            workflow.operation,
            OperationStatus.FAILED,
            "unexpected_group_workflow_exception",
            f"Unexpected {type(error).__name__}: {error}",
            developer_details="".join(traceback.format_exception(error)),
        )


def _gate5a_branch_inputs(publication, branch: str) -> dict[str, Path]:
    labels = {
        "measurements": f"{branch}_measurements_pooled",
        "measurements_parts": f"{branch}_measurements_all",
        "wide": f"{branch}_measurements_all_wide",
        "participant_summary": f"{branch}_participant_summary",
    }
    if branch == "cuteness":
        labels["trial_level"] = "cuteness_trial_level_all"
    paths = {name: publication.artifact_path(label) for name, label in labels.items()}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise PublicationError(
            f"Current Gate 5A publication lacks required {branch} inputs: {', '.join(missing)}."
        )
    return {name: Path(path) for name, path in paths.items()}


def _validated_gate5a_publication(request: GroupWorkflowRequest):
    """Validate current Gate 5A without crossing request-context namespaces."""

    run_request = request.workflow_request
    config = run_request.to_legacy_config()
    publication = validate_current_publication(
        resolve_session2_eeg_paths(config, root=run_request.project_root).output_dir / "group",
        PublicationScope(stage="gate5a", session=request.session),
    )
    _validate_gate5a_participant_dependencies(request, publication)
    return publication


def _validate_gate5a_participant_dependencies(request: GroupWorkflowRequest, publication) -> None:
    """Reject Gate 5A after any bound Gate 4/current-chain replacement."""

    config = request.workflow_request.to_legacy_config()
    scientific_fingerprint = request.workflow_request.resolved_config.scientific_fingerprint
    if not publication.identity_references:
        raise PublicationError("Gate 5A publication has no participant dependency bindings.")
    for key, generation_id in publication.identity_references.items():
        fields = key.split(":")
        if len(fields) != 5 or fields[1] not in {"gate4a", "gate4b"}:
            raise PublicationError(f"Gate 5A has an invalid participant dependency reference: {key!r}.")
        _, stage, branch, participant, recording_part = fields
        expected_branch = "recognition" if stage == "gate4a" else "cuteness"
        if branch != expected_branch:
            raise PublicationError(f"Gate 5A participant dependency branch is mismatched: {key!r}.")
        part_directory = subject_part_dir(
            config,
            participant,
            recording_part,
            root=request.workflow_request.project_root,
            create=False,
        )
        validate_gate4_publication_for_group(
            config,
            part_directory,
            participant,
            recording_part,
            branch,
            scientific_fingerprint,
            expected_generation_id=generation_id,
            session=request.session,
            **(
                {
                    "immutable_upstream_canonical_directory": request.workflow_request.reviewed_upstream_part_dir(
                        participant, recording_part
                    )
                }
                if request.workflow_request.gate3.isolated_replay
                else {}
            ),
        )


def _publish_group_result(
    request: GroupWorkflowRequest,
    workflow: GroupWorkflow,
    result: OperationResult,
    value: Any,
    gate5a_publication,
    token: CancellationToken,
) -> OperationResult:
    context = request.workflow_request.resolved_config.resolved_config_fingerprint
    sources = {artifact.label: artifact.path for artifact in result.artifacts if artifact.produced}
    if not sources:
        raise PublicationError(f"{workflow.value} reported success without publication artifacts.")
    for label, path in tuple(sources.items()):
        sidecar = Path(path).with_suffix(Path(path).suffix + ".provenance.json")
        if sidecar.is_file():
            sources[f"{label}__provenance"] = sidecar
    stage = {
        GroupWorkflow.PREPARE: "gate5a",
        GroupWorkflow.FIGURES: "gate5b",
        GroupWorkflow.STATISTICS: "gate5c",
    }[workflow]
    scope = PublicationScope(stage=stage, session=request.session, branch=request.branch)
    references = dict(getattr(value, "publication_identity_references", {}) or {})
    if gate5a_publication is not None:
        references[f"publication:gate5a:{gate5a_publication.scope.identity}"] = gate5a_publication.generation_id
        references.update(gate5a_publication.identity_references)
    canonical = resolve_session2_eeg_paths(
        request.workflow_request.to_legacy_config(),
        root=request.workflow_request.project_root,
    ).output_dir / "group"
    published = publish_artifact_set(
        canonical,
        scope,
        run_id=f"{request.request_id}:{workflow.value}",
        context_reference=context,
        required_artifacts=sources,
        cancelled=lambda: token.cancelled,
        identity_references=references,
    )
    artifacts = tuple(ArtifactReference(item.label, item.path) for item in published.artifacts)
    artifacts += (ArtifactReference("publication_finality_manifest", published.manifest_path),)
    return replace(
        result,
        artifacts=artifacts,
        publication_completed=True,
    )


def _normalise(request: GroupWorkflowRequest, workflow: GroupWorkflow, value: Any) -> OperationResult:
    warnings = tuple(str(item) for item in getattr(value, "warnings", ()) or ())
    artifacts = tuple(
        ArtifactReference(str(label), Path(path))
        for label, path in (getattr(value, "output_files", {}) or {}).items()
    )
    if not artifacts:
        return _result(
            request,
            workflow.operation,
            OperationStatus.FAILED,
            "required_group_artifacts_missing",
            "The Gate 5 runner returned no output artifacts.",
            warnings=warnings,
            payload=value,
        )

    status = OperationStatus.SUCCEEDED_WITH_WARNINGS if warnings else OperationStatus.SUCCEEDED
    reason = "completed_with_warnings" if warnings else "completed"
    diagnostic = f"{workflow.value} completed."

    if workflow is GroupWorkflow.PREPARE:
        incomplete = tuple(
            item for item in getattr(value, "participant_outcomes", ())
            if getattr(item, "status", "") == "missing_required_input"
        )
        material_warnings = tuple(
            item for item in warnings
            if any(
                marker in item
                for marker in (
                    "no recording parts with Gate 4 outputs",
                    "missing Gate 4 output(s)",
                    "missing trial-level model input",
                    "missing required wide identifier column(s)",
                    "missing measurement axis column",
                )
            )
        )
        if incomplete or material_warnings:
            status = OperationStatus.FAILED
            reason = "group_aggregation_incomplete"
            details = [f"{item.branch} {item.subject}: {item.reason}" for item in incomplete]
            details.extend(material_warnings)
            diagnostic = "Required Gate 4 participant input was unavailable: " + "; ".join(details)
        else:
            exclusions = tuple(
                item for item in getattr(value, "participant_outcomes", ())
                if getattr(item, "status", "") in {"configured_exclusion", "scientific_exclusion"}
            )
            diagnostic = (
                "Gate 5A aggregation completed; "
                f"{len(exclusions)} configured/scientific branch exclusion(s) remained non-fatal."
            )
    elif workflow is GroupWorkflow.FIGURES:
        missing = tuple(item for item in warnings if "missing Gate 5A input(s)" in item)
        if missing:
            status, reason = OperationStatus.INVALID_INPUT, "required_group_input_missing"
            diagnostic = "; ".join(missing)
        elif warnings:
            status, reason = OperationStatus.FAILED, "required_group_figure_incomplete"
            diagnostic = "; ".join(warnings)
        else:
            info = tuple(str(item) for item in getattr(value, "info_messages", ()) or ())
            diagnostic = (
                "Gate 5B required figures completed; legitimate QC exclusions remained non-fatal."
                if info
                else "Gate 5B required figures completed."
            )
    else:
        missing = tuple(item for item in warnings if "missing Gate 5A input(s)" in item)
        if missing:
            status, reason = OperationStatus.INVALID_INPUT, "required_group_input_missing"
            diagnostic = "; ".join(missing)
        else:
            status, reason, diagnostic, model_warnings = _statistics_status(value, artifacts, request.branch)
            warnings = (*warnings, *model_warnings)

    return _result(
        request,
        workflow.operation,
        status,
        reason,
        diagnostic,
        warnings=warnings,
        artifacts=artifacts,
        computation_completed=status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS},
        publication_completed=status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS},
        payload=value,
    )


def _statistics_status(
    value: Any,
    artifacts: tuple[ArtifactReference, ...],
    branch: str | None,
) -> tuple[OperationStatus, str, str, tuple[str, ...]]:
    by_label = {item.label: item.path for item in artifacts}
    model_warnings: tuple[str, ...] = ()
    diagnostic_label = (
        "mixed_model_diagnostics" if branch == "cuteness" else "cuteness_mixed_model_diagnostics"
    )
    path = by_label.get(diagnostic_label)
    if path is not None and path.is_file():
        table = pd.read_csv(path)
        model_status = (
            str(table.get("model_status", pd.Series(["UNKNOWN"])).iloc[0])
            if not table.empty
            else "UNKNOWN"
        )
        if model_status == "FIT_WARNING":
            warning = f"cuteness mixed model completed with {model_status}."
            model_warnings = (warning,)
        elif model_status != "FIT_OK":
            return (
                OperationStatus.FAILED,
                "statistical_model_incomplete",
                f"Cuteness mixed model reported {model_status}.",
                (),
            )
    if branch in {"cuteness", None}:
        ledger_label = "mixed_model_attempts" if branch == "cuteness" else "cuteness_mixed_model_attempts"
        ledger_path = by_label.get(ledger_label)
        if ledger_path is None or not ledger_path.is_file():
            return (
                OperationStatus.FAILED,
                "statistics_incomplete",
                "Cuteness mixed-model attempt ledger is missing.",
                (),
            )
        ledger = pd.read_csv(ledger_path)
        required = {"attempt_index", "model_role", "formula", "status", "success", "selected_for_final_result"}
        if ledger.empty or not required.issubset(ledger.columns):
            return (
                OperationStatus.FAILED,
                "statistics_incomplete",
                "Cuteness mixed-model attempt ledger is incomplete.",
                (),
            )

    statistic_labels = (
        ()
        if branch == "cuteness"
        else ("statistics",)
        if branch == "recognition"
        else ("recognition_statistics",)
    )
    for label in statistic_labels:
        stats_path = by_label.get(label)
        if stats_path is None or not stats_path.is_file():
            continue
        table = pd.read_csv(stats_path)
        if table.empty:
            return OperationStatus.FAILED, "statistics_incomplete", "Required recognition statistics were not computed.", ()
        primary = table[
            table.get("analysis_status", pd.Series("", index=table.index)).astype(str).eq("PRIMARY")
        ]
        incomplete = primary[
            ~primary.get("status", pd.Series("", index=primary.index)).astype(str).eq("computed")
        ]
        if not incomplete.empty:
            return (
                OperationStatus.FAILED,
                "statistics_incomplete",
                "One or more prespecified primary statistics were not computed.",
                (),
            )
    if model_warnings:
        return (
            OperationStatus.SUCCEEDED_WITH_WARNINGS,
            "completed_with_model_warning",
            model_warnings[0],
            model_warnings,
        )
    return OperationStatus.SUCCEEDED, "completed", "Gate 5C statistics completed.", ()


def _result(
    request: GroupWorkflowRequest,
    operation: Operation,
    status: OperationStatus,
    reason_code: str,
    diagnostic: str,
    **kwargs: Any,
) -> OperationResult:
    return OperationResult(
        operation=operation,
        request_id=request.request_id,
        status=status,
        reason_code=reason_code,
        diagnostic=diagnostic,
        branch=request.branch,
        session=request.session,
        context_reference=request.workflow_request.resolved_config.resolved_config_fingerprint,
        **kwargs,
    )


def _cancelled(request: GroupWorkflowRequest, operation: Operation, diagnostic: str) -> OperationResult:
    return _result(request, operation, OperationStatus.CANCELLED, "operation_cancelled", diagnostic)


def _aggregate(request: GroupWorkflowRequest, children: tuple[OperationResult, ...]) -> OperationResult:
    status = _aggregate_status(children)
    return _result(
        request,
        Operation.GROUP_WORKFLOW,
        status,
        "children_complete" if status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS} else "child_operation_not_successful",
        f"{len(children)} requested Gate 5 stage(s); parent status reflects every child.",
        warnings=tuple(item for child in children for item in child.warnings),
        artifacts=tuple(item for child in children for item in child.artifacts),
        computation_completed=all(child.computation_completed for child in children),
        publication_completed=all(child.publication_completed for child in children),
        children=children,
    )


def _aggregate_status(children: tuple[OperationResult, ...]) -> OperationStatus:
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
    return (
        OperationStatus.SUCCEEDED_WITH_WARNINGS
        if OperationStatus.SUCCEEDED_WITH_WARNINGS in statuses
        else OperationStatus.SUCCEEDED
    )
