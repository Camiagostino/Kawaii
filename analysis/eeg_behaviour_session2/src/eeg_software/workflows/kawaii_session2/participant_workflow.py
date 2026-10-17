"""Tracked participant application routing for Session 2 EEG.

Scientific stage functions remain authoritative for computation and artifacts.
This module owns validated discovery, ordering, structured status aggregation,
and the compatibility manifests previously owned by launcher scripts.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import re
import shutil
import traceback
from typing import Any, TYPE_CHECKING

from .alignment import build_trial_alignment
from .epoching import ica_required_for_epoching
from .erp import discover_gate3c_recording_parts
from .freeze import missing_expected_outputs, write_manual_review_snapshot, write_validation_manifest
from .ica import ICADiagnosticSummaryCache, gate3b_ica_solution_usable
from .loading import discover_recording_parts
from .ica_identity import validate_ica_solution_file_binding
from .identity import identify_source_file
from .manual_review import (
    ensure_manual_review_scaffold,
    read_bad_channel_review_decision,
    read_ica_review_decision,
)
from .operations import (
    ArtifactReference,
    CancellationToken,
    Operation,
    OperationInputs,
    OperationRequest,
    OperationResult,
    OperationStatus,
    ParticipantWorkflow,
    ParticipantWorkflowRequest,
    ProgressEvent,
    ProgressPhase,
)
from .paths import resolve_session2_eeg_paths, subject_dir, subject_part_dir
from .publication import (
    PublicationScope,
    PublicationValidationError,
    validate_current_publication,
    validate_immutable_participant_publication_chain,
    validate_participant_publication_chain,
    validate_publication_generation,
    validate_publication_scientific_context,
)
from .r11_replay_preflight import validate_r11_replay_preflight
from .resolved_snapshot import (
    REALIZATION_IN_PROGRESS_SNAPSHOT_STATE,
    REALIZED_SNAPSHOT_STATE,
    REQUESTED_SNAPSHOT_STATE,
    bind_authoritative_config_provenance,
    write_resolved_run_snapshot,
)
from .subject_selection import normalize_subject_id

if TYPE_CHECKING:
    from .coordinator import Session2EEGCoordinator


def execute_participant_workflow(
    coordinator: "Session2EEGCoordinator",
    request: ParticipantWorkflowRequest,
    *,
    cancellation: CancellationToken | None = None,
) -> OperationResult:
    token = cancellation or CancellationToken()
    sequence = 0

    def emit(
        phase: ProgressPhase,
        *,
        participant: str | None = None,
        recording_part: str | None = None,
        message: str | None = None,
    ) -> None:
        nonlocal sequence
        sequence += 1
        if coordinator._progress_sink is not None:
            coordinator._progress_sink(
                ProgressEvent(
                    request_id=request.request_id,
                    sequence=sequence,
                    operation=request.workflow.operation,
                    phase=phase,
                    participant=participant,
                    recording_part=recording_part,
                    branch=_branch_for(request.workflow),
                    session=request.session,
                    message=message,
                )
            )

    emit(ProgressPhase.STARTED)
    try:
        if token.cancelled:
            return _cancelled_parent(request, "Cancellation was requested before participant execution.")
        if request.workflow is ParticipantWorkflow.GATE3:
            result = _execute_gate3(coordinator, request, token, emit)
        else:
            result = _execute_gate4(coordinator, request, token, emit)
        emit(ProgressPhase.COMPLETED, message=result.status.value)
        return result
    except Exception as error:
        emit(ProgressPhase.FAILED, message=type(error).__name__)
        return _workflow_result(
            request,
            (),
            status=OperationStatus.FAILED,
            reason_code="unexpected_participant_workflow_exception",
            diagnostic=f"Unexpected {type(error).__name__}: {error}",
            developer_details="".join(traceback.format_exception(error)),
        )


def _execute_gate3(
    coordinator: "Session2EEGCoordinator",
    request: ParticipantWorkflowRequest,
    token: CancellationToken,
    emit,
) -> OperationResult:
    run_request = request.workflow_request
    config = run_request.to_legacy_config()
    root = run_request.project_root
    selected_config = run_request.selected_config_path
    if run_request.gate3.isolated_replay:
        preflight = validate_r11_replay_preflight(
            upstream_root=run_request.gate3.reviewed_upstream_root,
            corrected_output_root=run_request.gate3.corrected_output_root,
            participant_parts=tuple(
                (participant, recording_part)
                for participant in run_request.realized.participants
                for recording_part in run_request.requested.recording_parts
            ),
            manual_review_root=run_request.resolved_config.resolved_paths["manual_review_dir"],
            immutable_source_roots={
                key: run_request.resolved_config.resolved_paths[key]
                for key in ("raw_eeg_dir", "session2_logs_dir", "session2_schedules_dir")
            },
        )
        if not preflight.go:
            return _workflow_result(
                request,
                (),
                status=OperationStatus.INVALID_INPUT,
                reason_code="r11_replay_preflight_no_go",
                diagnostic=preflight.render_text(),
            )
    resolved_snapshot = write_resolved_run_snapshot(run_request, REQUESTED_SNAPSHOT_STATE)
    snapshot_state = REQUESTED_SNAPSHOT_STATE
    participant_results: list[OperationResult] = []
    ica_summary_cache = ICADiagnosticSummaryCache()

    for participant in run_request.realized.participants:
        if token.cancelled:
            participant_results.append(_cancelled_result(request, participant, "Cancelled before participant."))
            break
        emit(ProgressPhase.PARTICIPANT_STARTED, participant=participant)
        if run_request.gate3.clean_subject_output:
            clean_generated_subject_outputs(config, participant, root=root)
        parts = discover_recording_parts(participant, config, root=root)
        requested_parts = run_request.requested.recording_parts
        if requested_parts is not None:
            by_name = {part.recording_part: part for part in parts}
            missing = [part for part in requested_parts if part not in by_name]
            if missing:
                participant_results.append(
                    _invalid_part_selection(request, participant, missing)
                )
                continue
            parts = [by_name[name] for name in requested_parts]
        run_request = run_request.with_realized_recording_parts(
            participant, [part.recording_part for part in parts]
        )
        snapshot_state = (
            REALIZED_SNAPSHOT_STATE
            if len(run_request.realized.recording_parts) == len(run_request.realized.participants)
            else REALIZATION_IN_PROGRESS_SNAPSHOT_STATE
        )
        resolved_snapshot = write_resolved_run_snapshot(run_request, snapshot_state)
        authoritative_config = bind_authoritative_config_provenance(selected_config, resolved_snapshot)
        if not run_request.gate3.isolated_replay:
            ensure_manual_review_scaffold({participant: parts}, config, root=root)
            for part in parts:
                write_manual_review_snapshot(
                    config,
                    participant,
                    part.recording_part,
                    root=root,
                    config_path=authoritative_config,
                )
        alignment = build_trial_alignment(
            participant,
            parts,
            config,
            root=root,
            config_path=selected_config,
            resolved_snapshot=resolved_snapshot,
            write_outputs=True,
        )
        part_results: list[OperationResult] = []
        for part in parts:
            if token.cancelled:
                part_results.append(_cancelled_result(request, participant, "Cancelled before recording part.", part.recording_part))
                break
            emit(ProgressPhase.PART_STARTED, participant=participant, recording_part=part.recording_part)
            part_result = _execute_gate3_part(
                coordinator,
                request,
                run_request,
                participant,
                part,
                alignment,
                resolved_snapshot,
                ica_summary_cache,
                token,
                emit,
            )
            part_results.append(part_result)
            emit(
                ProgressPhase.PART_COMPLETED,
                participant=participant,
                recording_part=part.recording_part,
                message=part_result.status.value,
            )
        _write_subject_manifest(config, participant, root=root)
        participant_result = _aggregate(
            request.workflow.operation,
            f"{request.request_id}:{participant}",
            request.workflow_request,
            tuple(part_results),
            participant=participant,
            branch=None,
        )
        participant_results.append(participant_result)
        emit(ProgressPhase.PARTICIPANT_COMPLETED, participant=participant, message=participant_result.status.value)
        if token.cancelled:
            break

    if snapshot_state != REALIZED_SNAPSHOT_STATE:
        write_resolved_run_snapshot(run_request, REALIZED_SNAPSHOT_STATE)
    return _workflow_result(request, tuple(participant_results))


def _execute_gate3_part(
    coordinator: "Session2EEGCoordinator",
    request: ParticipantWorkflowRequest,
    run_request,
    participant: str,
    part,
    alignment,
    resolved_snapshot,
    ica_summary_cache: ICADiagnosticSummaryCache,
    token: CancellationToken,
    emit,
) -> OperationResult:
    config = run_request.to_legacy_config()
    root = run_request.project_root
    options = run_request.gate3
    stage_status = {
        "subject": participant,
        "recording_part": part.recording_part,
        "gate3a_status": "not_run",
        "gate3b_ica_status": "not_run",
        "gate3c_epoching_status": "not_run",
        "final_status": "running",
        "message": "",
    }
    children: list[OperationResult] = []

    reviewed_ica_lineage = None
    if options.use_existing_reviewed_ica:
        try:
            reviewed_ica_lineage = _resolve_reviewed_ica_lineage(
                run_request, participant, part.recording_part
            )
            reviewed_gate3a, _reviewed_gate3b = reviewed_ica_lineage
            filtered_publication_path = _required_publication_artifact(
                reviewed_gate3a, "filtered_raw_fif"
            )
            gate3a = _synthetic_stage_result(
                request,
                Operation.GATE3A_PART,
                participant,
                part.recording_part,
                OperationStatus.SUCCEEDED,
                "reviewed_ica_immutable_upstream_reused",
                "Reused the exact immutable Gate 3A generation bound to the reviewed ICA; Gate 3A was not rerun.",
                artifacts=(
                    ArtifactReference("filtered_raw_fif", filtered_publication_path),
                    ArtifactReference("publication_generation_manifest", reviewed_gate3a.manifest_path),
                ),
            )
        except (PublicationValidationError, ValueError) as error:
            failed = _synthetic_stage_result(
                request,
                Operation.GATE3B_PART,
                participant,
                part.recording_part,
                OperationStatus.STALE_INPUT,
                "existing_reviewed_ica_invalid_or_stale",
                str(error),
            )
            children.append(failed)
            stage_status.update(
                gate3a_status="not_run_reviewed_ica_validation_failed",
                gate3b_ica_status="stale_reviewed_ica",
                gate3c_epoching_status="stopped_required_ica_failed",
                final_status="failed",
                message=str(error),
            )
            return _finish_gate3_part(
                request, run_request, participant, part.recording_part, children, stage_status
            )
    else:
        gate3a = _execute_stage(
            coordinator,
            request,
            Operation.GATE3A_PART,
            participant,
            part.recording_part,
            OperationInputs(recording=part, alignment=alignment.trial_alignment, resolved_snapshot=resolved_snapshot),
            token,
            emit,
        )
    children.append(gate3a)
    stage_status["gate3a_status"] = _compatibility_status(gate3a)
    if gate3a.successful and not options.validation_only_skip_ica and not options.use_existing_reviewed_ica:
        try:
            published_gate3a = validate_current_publication(
                subject_part_dir(
                    config,
                    participant,
                    part.recording_part,
                    root=root,
                    create=False,
                ),
                PublicationScope(
                    stage="gate3a",
                    participant=participant,
                    recording_part=part.recording_part,
                ),
                expected_context_reference=run_request.resolved_config.resolved_config_fingerprint,
                expected_run_id=gate3a.request_id,
            )
            validate_participant_publication_chain(
                subject_part_dir(config, participant, part.recording_part, root=root, create=False),
                published_gate3a,
            )
            filtered_publication_path = published_gate3a.artifact_path("filtered_raw_fif")
            if filtered_publication_path is None:
                raise PublicationValidationError(
                    "Gate 3A finality does not declare the required filtered_raw_fif artifact."
                )
        except PublicationValidationError as error:
            gate3a = replace(
                gate3a,
                status=OperationStatus.STALE_INPUT,
                reason_code="publication_invalid_or_stale",
                diagnostic=str(error),
                publication_completed=False,
            )
            children[-1] = gate3a
            stage_status["gate3a_status"] = "stale_publication"
    if not gate3a.successful or not _artifact_path(gate3a, "filtered_raw_fif"):
        stage_status.update(
            gate3b_ica_status="skipped_gate3a_failed",
            gate3c_epoching_status="skipped_gate3a_failed",
            final_status="failed",
            message=gate3a.diagnostic or "Gate 3A filtered raw FIF unavailable.",
        )
        return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)

    filtered_raw = (
        _artifact_path(gate3a, "filtered_raw_fif")
        if options.validation_only_skip_ica
        else filtered_publication_path
    )
    if options.skip_ica:
        stage_status.update(
            gate3b_ica_status="skipped_configured",
            gate3c_epoching_status="skipped_ica_not_run",
            final_status="gate3a_only_ok",
            message="--skip-ica was used.",
        )
        return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)

    if options.validation_only_skip_ica:
        gate3c = _execute_stage(
            coordinator,
            request,
            Operation.GATE3C_PART,
            participant,
            part.recording_part,
            OperationInputs(
                recording=part,
                alignment=alignment.trial_alignment,
                filtered_raw_path=filtered_raw,
                ica_bypass_reason="ica_disabled_for_validation",
                resolved_snapshot=resolved_snapshot,
            ),
            token,
            emit,
        )
        children.append(gate3c)
        stage_status.update(
            gate3b_ica_status="skipped_configured_validation",
            gate3c_epoching_status=_compatibility_status(gate3c),
            final_status="ok" if gate3c.successful else "failed",
            message=(
                "NOT FINAL PREPROCESSING: ICA disabled only for validation."
                if gate3c.successful
                else gate3c.diagnostic
            ),
        )
        return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)

    if token.cancelled:
        children.append(_cancelled_result(request, participant, "Cancelled between Gate 3A and Gate 3B.", part.recording_part, Operation.GATE3B_PART))
        stage_status.update(final_status="cancelled", message="Cancelled between Gate 3A and Gate 3B.")
        return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)

    if options.use_existing_reviewed_ica:
        assert reviewed_ica_lineage is not None
        _reviewed_gate3a, existing_publication = reviewed_ica_lineage
        existing = _required_publication_artifact(existing_publication, "ica_solution")
        existing_identity = _required_publication_artifact(existing_publication, "ica_fit_identity")
        gate3b = _synthetic_stage_result(
            request,
            Operation.GATE3B_PART,
            participant,
            part.recording_part,
            OperationStatus.SUCCEEDED,
            "existing_reviewed_solution_validated",
            "Exact decision-bound ICA solution and immutable upstream lineage validated; ICA was not refit.",
            artifacts=(
                ArtifactReference("ica_solution", existing),
                ArtifactReference("ica_fit_identity", existing_identity),
                ArtifactReference("publication_generation_manifest", existing_publication.manifest_path),
            ),
        )
    else:
        gate3b = _execute_stage(
            coordinator,
            request,
            Operation.GATE3B_PART,
            participant,
            part.recording_part,
            OperationInputs(
                recording=part,
                alignment=alignment.trial_alignment,
                filtered_raw_path=filtered_raw,
                resolved_snapshot=resolved_snapshot,
                ica_summary_cache=ica_summary_cache,
            ),
            token,
            emit,
        )
        if gate3b.successful:
            try:
                gate3b_publication = validate_current_publication(
                    subject_part_dir(config, participant, part.recording_part, root=root, create=False),
                    PublicationScope(stage="gate3b", participant=participant, recording_part=part.recording_part),
                    expected_context_reference=run_request.resolved_config.resolved_config_fingerprint,
                    expected_run_id=gate3b.request_id,
                )
                validate_participant_publication_chain(
                    subject_part_dir(config, participant, part.recording_part, root=root, create=False),
                    gate3b_publication,
                )
                if gate3b_publication.artifact_path("ica_solution") is None or gate3b_publication.artifact_path("ica_fit_identity") is None:
                    raise PublicationValidationError(
                        "Gate 3B finality lacks ICA solution or fit identity."
                    )
            except PublicationValidationError as error:
                gate3b = replace(
                    gate3b,
                    status=OperationStatus.STALE_INPUT,
                    reason_code="publication_invalid_or_stale",
                    diagnostic=str(error),
                    publication_completed=False,
                )
    children.append(gate3b)
    stage_status["gate3b_ica_status"] = _compatibility_status(gate3b)
    if options.skip_epoching:
        stage_status.update(
            gate3c_epoching_status="skipped_configured",
            final_status="gate3b_ok_epoching_skipped" if gate3b.successful else "failed",
            message="--skip-epoching was used.",
        )
        return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)
    if not gate3b.successful:
        stage_status.update(
            gate3c_epoching_status="stopped_required_ica_failed",
            final_status="failed",
            message=gate3b.diagnostic,
        )
        return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)

    ica_solution = _artifact_path(gate3b, "ica_solution")
    ica_fit_identity = _artifact_path(gate3b, "ica_fit_identity")
    low_level_gate3b = gate3b.payload
    if not options.use_existing_reviewed_ica and not gate3b_ica_solution_usable(
        str(getattr(low_level_gate3b, "status", "")),
        getattr(low_level_gate3b, "output_files", {}) or {},
    ):
        if ica_required_for_epoching(config):
            failed = _synthetic_stage_result(
                request,
                Operation.GATE3C_PART,
                participant,
                part.recording_part,
                OperationStatus.FAILED,
                "required_ica_unavailable",
                "Required Gate 3B ICA solution unavailable.",
            )
            children.append(failed)
            stage_status.update(
                gate3c_epoching_status="stopped_required_ica_failed",
                final_status="failed",
                message=failed.diagnostic,
            )
            return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)
        ica_solution = None

    if token.cancelled:
        children.append(_cancelled_result(request, participant, "Cancelled between Gate 3B and Gate 3C.", part.recording_part, Operation.GATE3C_PART))
        stage_status.update(final_status="cancelled", message="Cancelled between Gate 3B and Gate 3C.")
        return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)

    gate3c = _execute_stage(
        coordinator,
        request,
        Operation.GATE3C_PART,
        participant,
        part.recording_part,
        OperationInputs(
            recording=part,
            alignment=alignment.trial_alignment,
            filtered_raw_path=filtered_raw,
            ica_solution_path=ica_solution,
            ica_fit_identity_path=ica_fit_identity,
            allow_incomplete_ica_review_for_validation=options.allow_incomplete_ica_review_for_validation,
            resolved_snapshot=resolved_snapshot,
            reviewed_ica_upstream_publications=(
                reviewed_ica_lineage if options.use_existing_reviewed_ica else ()
            ),
            reviewed_ica_upstream_canonical_directory=(
                run_request.reviewed_upstream_part_dir(participant, part.recording_part)
                if options.isolated_replay
                else subject_part_dir(config, participant, part.recording_part, root=root, create=False)
            ) if options.use_existing_reviewed_ica else None,
            allow_reviewed_ica_context_reuse=options.use_existing_reviewed_ica,
        ),
        token,
        emit,
    )
    if gate3c.successful:
        try:
            gate3c_publication = validate_current_publication(
                subject_part_dir(config, participant, part.recording_part, root=root, create=False),
                PublicationScope(stage="gate3c", participant=participant, recording_part=part.recording_part),
                expected_context_reference=run_request.resolved_config.resolved_config_fingerprint,
                expected_run_id=gate3c.request_id,
            )
            validate_participant_publication_chain(
                subject_part_dir(config, participant, part.recording_part, root=root, create=False),
                gate3c_publication,
                immutable_upstream_canonical_directory=(
                    run_request.reviewed_upstream_part_dir(participant, part.recording_part)
                    if options.isolated_replay
                    else None
                ),
            )
            if gate3c_publication.artifact_path("epochs_fif") is None or gate3c_publication.artifact_path("epoch_metadata_csv") is None:
                raise PublicationValidationError(
                    "Gate 3C finality lacks epochs_fif or epoch_metadata_csv."
                )
        except PublicationValidationError as error:
            gate3c = replace(
                gate3c,
                status=OperationStatus.STALE_INPUT,
                reason_code="publication_invalid_or_stale",
                diagnostic=str(error),
                publication_completed=False,
            )
    children.append(gate3c)
    stage_status.update(
        gate3c_epoching_status=_compatibility_status(gate3c),
        final_status="ok" if gate3c.successful else gate3c.status.value,
        message="" if gate3c.successful else gate3c.diagnostic,
    )
    manifest = write_validation_manifest(config, participant, part.recording_part, root=root)
    gate3c = replace(gate3c, artifacts=gate3c.artifacts + (ArtifactReference("validation_manifest", manifest),))
    children[-1] = gate3c
    missing = missing_expected_outputs(config, participant, part.recording_part, root=root)
    if gate3c.successful and missing:
        gate3c = replace(
            gate3c,
            status=OperationStatus.FAILED,
            reason_code="required_artifacts_missing",
            diagnostic="Missing expected preprocessing outputs: " + "; ".join(missing),
            publication_completed=False,
        )
        children[-1] = gate3c
        stage_status.update(final_status="failed", message=gate3c.diagnostic)
    return _finish_gate3_part(request, run_request, participant, part.recording_part, children, stage_status)


def _resolve_reviewed_ica_lineage(run_request, participant: str, recording_part: str):
    """Resolve the one immutable Gate 3B generation authorized by human review."""

    config = run_request.to_legacy_config()
    root = run_request.project_root
    canonical = (
        run_request.reviewed_upstream_part_dir(participant, recording_part)
        if run_request.gate3.isolated_replay
        else subject_part_dir(config, participant, recording_part, root=root, create=False)
    )
    scope = PublicationScope(stage="gate3b", participant=participant, recording_part=recording_part)
    bad_decision = read_bad_channel_review_decision(
        config, participant, recording_part, root=root, require_complete=True
    )
    initial_review = read_ica_review_decision(
        config, participant, recording_part, root=root, require_complete=True
    )
    reviewed_generation_id = _reviewed_gate3b_generation_id(initial_review)
    matches = []
    for publication in _validated_publication_history(canonical, scope):
        solution = _required_publication_artifact(publication, "ica_solution")
        identity_path = _required_publication_artifact(publication, "ica_fit_identity")
        identity = validate_ica_solution_file_binding(
            identity_path,
            solution_path=solution,
            expected_participant=participant,
            expected_recording_part=recording_part,
        )
        if (
            identity.solution_fingerprint == initial_review.ica_solution_id
            and (
                reviewed_generation_id is None
                or publication.generation_id == reviewed_generation_id
            )
        ):
            matches.append((publication, identity))
    if len(matches) != 1:
        raise PublicationValidationError(
            f"Expected exactly one immutable Gate 3B generation matching the reviewed ICA solution for "
            f"{participant} {recording_part}; found {len(matches)}."
        )

    gate3b, identity = matches[0]
    if identity.fit_context.scientific_fingerprint != run_request.resolved_config.scientific_fingerprint:
        raise PublicationValidationError(
            "Reviewed ICA scientific configuration is stale or mismatched with the current invocation."
        )
    component_count = int(identity.fit_context.algorithm_parameters["n_components_effective"])
    read_ica_review_decision(
        config,
        participant,
        recording_part,
        root=root,
        expected_solution_id=identity.solution_fingerprint,
        expected_component_count=component_count,
        expected_bad_channel_decision_id=bad_decision.semantic_decision_id,
        require_complete=True,
    )
    ancestors = validate_immutable_participant_publication_chain(canonical, gate3b)
    gate3a_matches = [publication for publication in ancestors if publication.scope.stage == "gate3a"]
    if len(gate3a_matches) != 1:
        raise PublicationValidationError(
            "Reviewed Gate 3B must bind exactly one immutable Gate 3A generation."
        )
    gate3a = gate3a_matches[0]
    filtered_raw = _required_publication_artifact(gate3a, "filtered_raw_fif")
    filtered_identity = identify_source_file(
        filtered_raw, "ica_filtered_raw_input", logical_path="filtered_raw.fif"
    )
    if filtered_identity.identity_payload() != identity.fit_context.input_eeg.identity_payload():
        raise PublicationValidationError(
            "Reviewed ICA fit identity does not match its immutable Gate 3A filtered EEG input."
        )
    return gate3a, gate3b


def _reviewed_gate3b_generation_id(decision) -> str | None:
    """Read the optional audit-bound canonical generation marker from v1 notes."""

    notes = str(getattr(decision, "notes", "") or "")
    marker = "Reviewed canonical Gate 3B generation"
    matches = re.findall(r"Reviewed canonical Gate 3B generation ([0-9a-f]{64})\b", notes)
    if marker in notes and len(set(matches)) != 1:
        raise PublicationValidationError(
            "ICA review notes contain an invalid or ambiguous canonical Gate 3B generation marker."
        )
    return matches[0] if matches else None


def _validated_publication_history(canonical: Path, scope: PublicationScope):
    """Return current and predecessor generations; promoted orphans are excluded."""

    current = validate_current_publication(canonical, scope)
    history = [current]
    seen = {current.generation_id}
    predecessor = current.predecessor_generation_id
    while predecessor is not None:
        if predecessor in seen:
            raise PublicationValidationError(
                f"Publication predecessor cycle detected for {scope.identity}."
            )
        generation = validate_publication_generation(canonical, scope, predecessor)
        history.append(generation)
        seen.add(generation.generation_id)
        predecessor = generation.predecessor_generation_id
    return tuple(history)


def _required_publication_artifact(publication, label: str) -> Path:
    path = publication.artifact_path(label)
    if path is None:
        raise PublicationValidationError(
            f"{publication.scope.stage} publication lacks required artifact {label!r}."
        )
    return Path(path)


def _execute_gate4(
    coordinator: "Session2EEGCoordinator",
    request: ParticipantWorkflowRequest,
    token: CancellationToken,
    emit,
) -> OperationResult:
    run_request = request.workflow_request
    config = run_request.to_legacy_config()
    branch = _branch_for(request.workflow)
    operation = request.workflow.operation
    stage_operation = (
        Operation.GATE4A_RECOGNITION_PART
        if request.workflow is ParticipantWorkflow.RECOGNITION
        else Operation.GATE4B_CUTENESS_PART
    )
    participant_results: list[OperationResult] = []
    for participant in run_request.realized.participants:
        if token.cancelled:
            participant_results.append(_cancelled_result(request, participant, "Cancelled before participant.", operation=operation))
            break
        emit(ProgressPhase.PARTICIPANT_STARTED, participant=participant)
        parts = list(
            request.requested_recording_parts
            or discover_gate3c_recording_parts(config, participant, root=run_request.project_root)
        )
        if not parts:
            result = _synthetic_stage_result(
                request,
                operation,
                participant,
                None,
                OperationStatus.INVALID_INPUT,
                "gate3c_inputs_unavailable",
                "No Gate 3C epoch inputs found for participant.",
                branch=branch,
            )
            status_path = _write_gate4_status(
                config,
                participant,
                "",
                branch,
                "failed",
                result.diagnostic,
                root=run_request.project_root,
            )
            participant_results.append(
                replace(result, artifacts=(ArtifactReference(f"gate4_{branch}_status", status_path),))
            )
            continue
        part_results: list[OperationResult] = []
        for recording_part in parts:
            if token.cancelled:
                part_results.append(_cancelled_result(request, participant, "Cancelled before recording part.", recording_part, stage_operation))
                break
            emit(ProgressPhase.PART_STARTED, participant=participant, recording_part=recording_part)
            try:
                gate3c_publication = validate_current_publication(
                    subject_part_dir(config, participant, recording_part, root=run_request.project_root, create=False),
                    PublicationScope(stage="gate3c", participant=participant, recording_part=recording_part),
                )
                validate_participant_publication_chain(
                    subject_part_dir(config, participant, recording_part, root=run_request.project_root, create=False),
                    gate3c_publication,
                    immutable_upstream_canonical_directory=(
                        run_request.reviewed_upstream_part_dir(participant, recording_part)
                        if run_request.gate3.isolated_replay
                        else None
                    ),
                )
                if (
                    gate3c_publication.context_reference
                    != run_request.resolved_config.resolved_config_fingerprint
                ):
                    validate_publication_scientific_context(
                        gate3c_publication,
                        run_request.resolved_config.scientific_fingerprint,
                    )
                epochs_path = gate3c_publication.artifact_path("epochs_fif")
                epoch_metadata_path = gate3c_publication.artifact_path("epoch_metadata_csv")
                if epochs_path is None or epoch_metadata_path is None:
                    raise PublicationValidationError(
                        "Current Gate 3C publication lacks epochs_fif or epoch_metadata_csv."
                    )
            except PublicationValidationError as error:
                part_results.append(
                    _synthetic_stage_result(
                        request,
                        stage_operation,
                        participant,
                        recording_part,
                        OperationStatus.STALE_INPUT,
                        "gate3c_publication_invalid_or_stale",
                        str(error),
                        branch=branch,
                    )
                )
                continue
            result = _execute_stage(
                coordinator,
                request,
                stage_operation,
                participant,
                recording_part,
                OperationInputs(epochs_path=epochs_path, epoch_metadata_path=epoch_metadata_path),
                token,
                emit,
                branch=branch,
            )
            status_path = _write_gate4_status(
                config,
                participant,
                recording_part,
                branch,
                _compatibility_status(result),
                result.diagnostic if not result.successful else "",
                root=run_request.project_root,
            )
            result = replace(result, artifacts=result.artifacts + (ArtifactReference(f"gate4{branch}_status", status_path),))
            part_results.append(result)
            emit(ProgressPhase.PART_COMPLETED, participant=participant, recording_part=recording_part, message=result.status.value)
        participant_result = _aggregate(
            operation,
            f"{request.request_id}:{participant}",
            request.workflow_request,
            tuple(part_results),
            participant=participant,
            branch=branch,
        )
        participant_results.append(participant_result)
        emit(ProgressPhase.PARTICIPANT_COMPLETED, participant=participant, message=participant_result.status.value)
        if token.cancelled:
            break
    return _workflow_result(request, tuple(participant_results))


def _execute_stage(
    coordinator,
    workflow_request: ParticipantWorkflowRequest,
    operation: Operation,
    participant: str,
    recording_part: str,
    inputs: OperationInputs,
    token: CancellationToken,
    emit,
    *,
    branch: str | None = None,
) -> OperationResult:
    emit(ProgressPhase.STAGE_STARTED, participant=participant, recording_part=recording_part, message=operation.value)
    result = coordinator.execute(
        OperationRequest(
            operation,
            workflow_request.workflow_request,
            participant=participant,
            recording_part=recording_part,
            branch=branch,
            request_id=f"{workflow_request.request_id}:{participant}:{recording_part}:{operation.value}",
        ),
        inputs=inputs,
        cancellation=token,
    )
    emit(ProgressPhase.STAGE_COMPLETED, participant=participant, recording_part=recording_part, message=f"{operation.value}:{result.status.value}")
    return result


def _finish_gate3_part(
    request: ParticipantWorkflowRequest,
    run_request,
    participant: str,
    recording_part: str,
    children: list[OperationResult],
    stage_status: dict[str, str],
) -> OperationResult:
    status_path = _write_stage_status(
        run_request.to_legacy_config(), participant, recording_part, stage_status, root=run_request.project_root
    )
    result = _aggregate(
        Operation.PARTICIPANT_GATE3,
        f"{request.request_id}:{participant}:{recording_part}",
        request.workflow_request,
        tuple(children),
        participant=participant,
        recording_part=recording_part,
        branch=None,
    )
    return replace(result, artifacts=result.artifacts + (ArtifactReference("stage_status", status_path),))


def _aggregate(
    operation: Operation,
    request_id: str,
    run_request,
    children: tuple[OperationResult, ...],
    *,
    participant: str | None,
    recording_part: str | None = None,
    branch: str | None,
) -> OperationResult:
    if not children:
        status = OperationStatus.INVALID_INPUT
    else:
        status = _aggregate_status(children)
    return OperationResult(
        operation=operation,
        request_id=request_id,
        status=status,
        reason_code=(
            "children_complete"
            if status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS}
            else "required_child_not_successful"
        ),
        diagnostic=f"{len(children)} required child operation(s); aggregate reflects every child.",
        participant=participant,
        recording_part=recording_part,
        branch=branch,
        context_reference=run_request.resolved_config.resolved_config_fingerprint,
        warnings=tuple(warning for child in children for warning in child.warnings),
        artifacts=tuple(artifact for child in children for artifact in child.artifacts),
        computation_completed=bool(children) and all(child.computation_completed for child in children),
        publication_completed=bool(children) and all(child.publication_completed for child in children),
        children=children,
    )


def _workflow_result(
    request: ParticipantWorkflowRequest,
    children: tuple[OperationResult, ...],
    *,
    status: OperationStatus | None = None,
    reason_code: str | None = None,
    diagnostic: str | None = None,
    developer_details: str | None = None,
) -> OperationResult:
    if status is None:
        status = _aggregate_status(children) if children else OperationStatus.INVALID_INPUT
    return OperationResult(
        operation=request.workflow.operation,
        request_id=request.request_id,
        status=status,
        reason_code=reason_code or ("participants_complete" if status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS} else "participant_not_successful"),
        diagnostic=diagnostic or f"{len(children)} participant result(s); aggregate reflects every required participant and part.",
        branch=_branch_for(request.workflow),
        context_reference=request.workflow_request.resolved_config.resolved_config_fingerprint,
        warnings=tuple(warning for child in children for warning in child.warnings),
        artifacts=tuple(artifact for child in children for artifact in child.artifacts),
        computation_completed=bool(children) and all(child.computation_completed for child in children),
        publication_completed=bool(children) and all(child.publication_completed for child in children),
        children=children,
        developer_details=developer_details,
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
    if OperationStatus.SUCCEEDED_WITH_WARNINGS in statuses:
        return OperationStatus.SUCCEEDED_WITH_WARNINGS
    return OperationStatus.SUCCEEDED


def _synthetic_stage_result(
    request: ParticipantWorkflowRequest,
    operation: Operation,
    participant: str,
    recording_part: str | None,
    status: OperationStatus,
    reason_code: str,
    diagnostic: str,
    *,
    branch: str | None = None,
    artifacts: tuple[ArtifactReference, ...] = (),
) -> OperationResult:
    return OperationResult(
        operation=operation,
        request_id=f"{request.request_id}:{participant}:{recording_part or 'none'}:{operation.value}",
        status=status,
        reason_code=reason_code,
        diagnostic=diagnostic,
        participant=participant,
        recording_part=recording_part,
        branch=branch,
        context_reference=request.workflow_request.resolved_config.resolved_config_fingerprint,
        artifacts=artifacts,
        computation_completed=status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS},
        publication_completed=status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS},
    )


def _cancelled_result(
    request: ParticipantWorkflowRequest,
    participant: str,
    diagnostic: str,
    recording_part: str | None = None,
    operation: Operation | None = None,
) -> OperationResult:
    return _synthetic_stage_result(
        request,
        operation or request.workflow.operation,
        participant,
        recording_part,
        OperationStatus.CANCELLED,
        "operation_cancelled",
        diagnostic,
        branch=_branch_for(request.workflow),
    )


def _cancelled_parent(request: ParticipantWorkflowRequest, diagnostic: str) -> OperationResult:
    return _workflow_result(
        request,
        (),
        status=OperationStatus.CANCELLED,
        reason_code="operation_cancelled",
        diagnostic=diagnostic,
    )


def _invalid_part_selection(request: ParticipantWorkflowRequest, participant: str, missing: list[str]) -> OperationResult:
    return _synthetic_stage_result(
        request,
        request.workflow.operation,
        participant,
        None,
        OperationStatus.INVALID_INPUT,
        "recording_part_not_discovered",
        "Requested recording part(s) not discovered: " + ", ".join(missing),
        branch=_branch_for(request.workflow),
    )


def _artifact_path(result: OperationResult, label: str) -> Path | None:
    return next((artifact.path for artifact in result.artifacts if artifact.label == label), None)


def _compatibility_status(result: OperationResult) -> str:
    if result.payload is not None and hasattr(result.payload, "status"):
        return str(result.payload.status)
    return result.status.value


def _branch_for(workflow: ParticipantWorkflow) -> str | None:
    return None if workflow is ParticipantWorkflow.GATE3 else workflow.value


def _write_gate4_status(
    config: dict,
    participant: str,
    recording_part: str,
    branch: str,
    status_value: str,
    message: str,
    *,
    root: Path,
) -> Path:
    gate = "gate4a" if branch == "recognition" else "gate4b"
    filename = f"{gate}_{branch}_status.csv"
    field = f"{gate}_{branch}_status"
    base = subject_dir(config, participant, root=root, create=True)
    logs_dir = base / (recording_part or gate) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "recording_part", field, "message"])
        writer.writeheader()
        writer.writerow({"subject": participant, "recording_part": recording_part, field: status_value, "message": message})
    return path


def _write_stage_status(
    config: dict,
    participant: str,
    recording_part: str,
    row: dict[str, str],
    *,
    root: Path,
) -> Path:
    logs_dir = subject_dir(config, participant, root=root, create=True) / recording_part / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "stage_status.csv"
    fields = ["subject", "recording_part", "gate3a_status", "gate3b_ica_status", "gate3c_epoching_status", "final_status", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _write_subject_manifest(config: dict, participant: str, *, root: Path) -> Path:
    base = subject_dir(config, participant, root=root, create=True)
    path = base / "README.md"
    path.write_text(
        f"# {participant} Session 2 EEG Outputs\n\n"
        "Each `part*/` folder is self-contained for frozen preprocessing inspection. "
        "See each part's `README.md` and `validation_manifest.csv` for stage-specific "
        "file descriptions and completeness checks.\n",
        encoding="utf-8",
    )
    for part_dir in sorted(base.glob("part*")):
        if part_dir.is_dir():
            _write_part_manifest(part_dir, participant, part_dir.name)
    return path


def _write_part_manifest(part_dir: Path, participant: str, recording_part: str) -> Path:
    path = part_dir / "README.md"
    content = f"""# {participant} {recording_part} EEG Outputs

This folder is the frozen preprocessing record for one Session 2 EEG recording part. Recognition ERP, cuteness ERP, group analyses, and statistics are not produced here.

## Folder Structure

- `alignment/`: trigger/trial alignment tables and coverage summaries for this recording part.
- `raw_qc/`: Gate 3A raw QC CSVs, `report.md`, and `figures/` for raw/PSD/variance/montage diagnostics.
- `preprocessing/`: Gate 3A preprocessing artefacts such as `filtered_raw.fif` and `processing_decisions.csv`.
- `ica/`: Gate 3B ICA parameters, component summaries, MNE-named `*-ica.fif` solution files, `report.md`, timing logs, and `figures/`.
- `epochs/`: Gate 3C block-wise epoch FIFs (`block01-epo.fif`, etc. for available experimental blocks), the canonical concatenated `all_image_epochs-epo.fif`, epoch metadata/rejection tables, `report.md`, and `figures/`.
- `provenance/`: provenance JSON records plus `manual_review_snapshot.yaml`, the immutable manual-review decisions used for this run.
- `logs/`: authoritative `stage_status.csv` and runtime log-style outputs.

## Manual Review Workflow

Editable manual-review files are shared at `../../../manual_review/` and are not removed by clean reruns. This part stores the decisions actually used in `provenance/manual_review_snapshot.yaml` for reproducibility if the editable review files change later.

## Expected Outputs

`validation_manifest.csv` lists the expected frozen preprocessing outputs and whether each one exists, including alignment, raw QC, filtered raw, ICA, block-wise epoch outputs, provenance JSON, and the manual-review snapshot. Missing experimental blocks are not invented or required; per-block epoch FIFs are validated only for blocks listed in `epochs/block_processing_summary.csv`.

Gate 3C preprocessing mode is `blockwise_epoching`: each available experimental block is cropped, ICA-cleaned, average-referenced, interpolated, and epoched independently. `all_image_epochs-epo.fif` is the canonical final epoch file created by concatenating block-level Epochs objects, not by concatenating Raw recordings.

## Provenance

Provenance JSON records include pipeline version, configuration hash, git commit when available, Python/MNE/NumPy versions, input files, output file, warnings, and the manual-review snapshot path.
"""
    path.write_text(content, encoding="utf-8")
    return path


def validate_generated_subject_cleanup_target(
    candidate: Path,
    *,
    subject: str,
    output_root: Path,
    family_root: Path,
    forbidden_roots: tuple[Path, ...] = (),
) -> Path:
    resolved_output = output_root.resolve()
    resolved_family = family_root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_family.relative_to(resolved_output)
        resolved_candidate.relative_to(resolved_output)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing cleanup outside output hierarchy: candidate={resolved_candidate}, "
            f"family={resolved_family}, output={resolved_output}"
        ) from exc
    if resolved_family == resolved_output and resolved_candidate == resolved_output:
        raise RuntimeError(f"Refusing to clean output root itself: {resolved_candidate}")
    if resolved_candidate.parent != resolved_family:
        raise RuntimeError(
            "Refusing cleanup target that is not an exact child of its generated artifact family: "
            f"candidate={resolved_candidate}, family={resolved_family}"
        )
    if resolved_candidate.name != subject:
        raise RuntimeError(
            f"Refusing cleanup target for a different subject: candidate={resolved_candidate}, "
            f"expected_subject={subject!r}"
        )
    for forbidden in forbidden_roots:
        resolved_forbidden = forbidden.resolve()
        if resolved_candidate == resolved_forbidden or resolved_forbidden in resolved_candidate.parents:
            raise RuntimeError(f"Refusing cleanup of protected path: {resolved_candidate}")
    return resolved_candidate


def generated_subject_cleanup_targets(config: dict, subject: str, *, root: Path) -> tuple[Path, ...]:
    canonical_subject = normalize_subject_id(subject, config=config)
    if canonical_subject != str(subject):
        raise ValueError(
            f"Cleanup requires an exact canonical subject identifier; got {subject!r}, expected {canonical_subject!r}."
        )
    paths = resolve_session2_eeg_paths(config, root=root, create=True)
    output_root = paths.output_dir.resolve()
    project_root = Path(root).resolve()
    try:
        output_root.relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing cleanup because output_dir is outside the project root: {output_root}") from exc
    if output_root == project_root:
        raise RuntimeError(f"Refusing cleanup because output_dir is the project root: {output_root}")
    family_roots = (
        output_root / "subjects",
        paths.alignment_dir,
        paths.preprocessing_dir,
        paths.ica_dir,
        paths.epochs_dir,
        paths.raw_qc_dir,
        paths.ica_qc_dir,
        paths.epoch_qc_dir,
        paths.report_dir,
        output_root,
    )
    forbidden_roots = (
        paths.manual_review_dir,
        paths.raw_eeg_dir,
        project_root / "config",
        project_root / "docs",
        project_root / "src",
        project_root / "tests",
    )
    validated: list[Path] = []
    for family_root in family_roots:
        target = validate_generated_subject_cleanup_target(
            Path(family_root) / canonical_subject,
            subject=canonical_subject,
            output_root=output_root,
            family_root=Path(family_root),
            forbidden_roots=forbidden_roots,
        )
        if target not in validated:
            validated.append(target)
    return tuple(validated)


def clean_generated_subject_outputs(config: dict, subject: str, *, root: Path) -> list[Path]:
    candidates = generated_subject_cleanup_targets(config, subject, root=root)
    removed: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            shutil.rmtree(candidate)
            removed.append(candidate)
    return removed
