"""Tracked response-timing application workflow with WS3 identity binding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import traceback
from typing import Any, TYPE_CHECKING

import pandas as pd

from .identity import (
    BehavioralInputIdentity,
    SourceFileIdentity,
    SourceIdentityError,
    identify_behavioral_inputs,
    identify_source_file,
)
from .operations import (
    ArtifactReference,
    CancellationToken,
    Operation,
    OperationResult,
    OperationStatus,
    ProgressEvent,
    ProgressPhase,
    TimingWorkflowRequest,
)
from .paths import resolve_session2_eeg_paths, subject_dir
from .publication import (
    PublicationCancelled,
    PublicationError,
    PublicationScope,
    ValidatedPublication,
    publish_artifact_set,
    validate_current_publication,
    validate_participant_publication_chain,
    validate_publication_scientific_context,
)
from .response_timing_qc import (
    build_trial_level_qc,
    response_timing_settings,
    write_figure,
    write_report,
)
from .stage_identity import (
    AlignmentArtifactIdentity,
    StageIdentityError,
    TimingQCInputIdentity,
    alignment_identity_path,
    build_timing_qc_input_identity,
    read_alignment_artifact_identity,
    validate_alignment_artifact_identity,
)

if TYPE_CHECKING:
    from .coordinator import Session2EEGCoordinator


@dataclass(frozen=True)
class TimingInputSelection:
    """Exact content identities and paths selected for one timing part."""

    behavioral_log_path: Path
    behavioral_inputs: BehavioralInputIdentity
    trigger_log_path: Path | None
    trigger_log_identity: SourceFileIdentity | None
    alignment_path: Path
    alignment_artifact: AlignmentArtifactIdentity
    timing_input_identity: TimingQCInputIdentity
    metadata_parts: tuple[str, ...]


@dataclass(frozen=True)
class TimingPartPayload:
    """Typed in-process timing result retaining its exact selected identity."""

    table: pd.DataFrame
    inputs: TimingInputSelection
    gate3c_publication: ValidatedPublication


def execute_timing_workflow(
    coordinator: "Session2EEGCoordinator",
    request: TimingWorkflowRequest,
    *,
    cancellation: CancellationToken | None = None,
) -> OperationResult:
    """Reproduce the compatibility timing assembly behind a typed boundary."""

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
                    operation=Operation.RESPONSE_TIMING_QC,
                    phase=phase,
                    participant=participant,
                    recording_part=recording_part,
                    session=request.session,
                    message=message,
                )
            )

    emit(ProgressPhase.STARTED)
    run_request = request.workflow_request
    config = run_request.to_legacy_config()
    root = run_request.project_root
    settings = response_timing_settings(config)
    paths = resolve_session2_eeg_paths(config, root=root)
    frames: list[pd.DataFrame] = []
    input_paths: list[Path] = []
    children: list[OperationResult] = []

    for participant in request.participants:
        if token.cancelled:
            children.append(_cancelled(request, participant=participant))
            emit(ProgressPhase.CANCELLED, participant=participant)
            break
        emit(ProgressPhase.PARTICIPANT_STARTED, participant=participant)
        behaviour_path = (
            paths.session2_logs_dir
            / f"{participant}_session2"
            / f"{participant}_session2_trial_log_analysis.csv"
        )
        trigger_path = (
            paths.session2_logs_dir
            / f"{participant}_session2"
            / f"{participant}_session2_trigger_log.csv"
        )
        if not behaviour_path.is_file():
            child = _result(
                request,
                OperationStatus.INVALID_INPUT,
                "behavioral_log_missing",
                f"Required behavioral timing log is unavailable: {behaviour_path}",
                participant=participant,
            )
            children.append(child)
            emit(ProgressPhase.PARTICIPANT_COMPLETED, participant=participant, message=child.status.value)
            continue

        behaviour = pd.read_csv(behaviour_path)
        trigger = pd.read_csv(trigger_path) if trigger_path.is_file() else None
        input_paths.append(behaviour_path)
        if trigger_path.is_file():
            input_paths.append(trigger_path)

        part_dirs = _selected_part_dirs(request, config, participant, root)
        if isinstance(part_dirs, OperationResult):
            children.append(part_dirs)
            emit(ProgressPhase.PARTICIPANT_COMPLETED, participant=participant, message=part_dirs.status.value)
            continue

        part_children: list[OperationResult] = []
        for part_dir in part_dirs:
            if token.cancelled:
                part_children.append(_cancelled(request, participant=participant, recording_part=part_dir.name))
                emit(ProgressPhase.CANCELLED, participant=participant, recording_part=part_dir.name)
                break
            emit(ProgressPhase.PART_STARTED, participant=participant, recording_part=part_dir.name)
            child, frame, used = _execute_part(
                request,
                participant,
                part_dir,
                behaviour,
                behaviour_path,
                trigger,
                trigger_path if trigger_path.is_file() else None,
                settings,
                config,
                root,
            )
            part_children.append(child)
            if frame is not None:
                frames.append(frame)
            input_paths.extend(used)
            emit(
                ProgressPhase.PART_COMPLETED,
                participant=participant,
                recording_part=part_dir.name,
                message=child.status.value,
            )

        participant_result = _aggregate_scope(
            request,
            tuple(part_children),
            participant=participant,
            diagnostic=f"Timing-QC recording parts for {participant}.",
        )
        children.append(participant_result)
        emit(ProgressPhase.PARTICIPANT_COMPLETED, participant=participant, message=participant_result.status.value)
        if token.cancelled:
            break

    status = _aggregate_status(tuple(children)) if children else OperationStatus.FAILED
    output_artifacts: tuple[ArtifactReference, ...] = ()
    publication_completed = False
    if frames and not token.cancelled:
        table = pd.concat(frames, ignore_index=True)
        output_dir = paths.output_dir / str(settings["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "response_timing_trial_level.csv"
        fig_path = output_dir / "response_timing_qc_figure.png"
        report_path = output_dir / "response_timing_qc_report.md"
        try:
            table.to_csv(csv_path, index=False)
            write_figure(table, fig_path)
            write_report(table, list(dict.fromkeys(input_paths)), report_path)
            output_artifacts = tuple(
                ArtifactReference(label, path)
                for label, path in (
                    ("response_timing_trial_level", csv_path),
                    ("response_timing_qc_figure", fig_path),
                    ("response_timing_qc_report", report_path),
                )
            )
            publication_completed = False
        except Exception as error:
            children.append(
                _result(
                    request,
                    OperationStatus.FAILED,
                    "timing_output_failure",
                    f"Unexpected {type(error).__name__} while writing timing outputs: {error}",
                    developer_details="".join(traceback.format_exception(error)),
                )
            )
            status = OperationStatus.FAILED

    if not frames and status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS}:
        status = OperationStatus.FAILED
    reason = (
        "completed" if status is OperationStatus.SUCCEEDED
        else "completed_with_warnings" if status is OperationStatus.SUCCEEDED_WITH_WARNINGS
        else "timing_workflow_incomplete"
    )
    result = _result(
        request,
        status,
        reason,
        f"{len(frames)} timing part result(s); workflow status reflects every requested participant/part.",
        artifacts=output_artifacts,
        computation_completed=bool(frames) and status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS},
        publication_completed=publication_completed and status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS},
        payload=pd.concat(frames, ignore_index=True) if frames else None,
        children=tuple(children),
    )
    if result.successful and output_artifacts:
        try:
            if token.cancelled:
                raise PublicationCancelled("Timing publication cancelled before canonical finality.")
            identity_references = _timing_identity_references(tuple(children))
            output_dir = paths.output_dir / str(settings["output_dir"])
            scope = PublicationScope(stage="timing", session=request.session)
            published = publish_artifact_set(
                output_dir,
                scope,
                run_id=request.request_id,
                context_reference=run_request.resolved_config.resolved_config_fingerprint,
                required_artifacts={artifact.label: artifact.path for artifact in output_artifacts},
                cancelled=lambda: token.cancelled,
                identity_references=identity_references,
            )
            artifacts = tuple(ArtifactReference(item.label, item.path) for item in published.artifacts)
            artifacts += (ArtifactReference("publication_finality_manifest", published.manifest_path),)
            result = replace(
                result,
                artifacts=artifacts,
                publication_completed=True,
            )
        except PublicationCancelled as error:
            result = replace(result, status=OperationStatus.CANCELLED, reason_code="publication_cancelled", diagnostic=str(error), publication_completed=False)
        except PublicationError as error:
            result = replace(result, status=OperationStatus.STALE_INPUT, reason_code="publication_invalid_or_stale", diagnostic=str(error), publication_completed=False)
    emit(ProgressPhase.COMPLETED, message=result.status.value)
    return result


def _timing_identity_references(children: tuple[OperationResult, ...]) -> dict[str, str]:
    references: dict[str, str] = {}

    def visit(result: OperationResult) -> None:
        payload = result.payload
        if isinstance(payload, TimingPartPayload):
            identity = payload.inputs.timing_input_identity
            key = f"timing_input:{identity.participant}:{identity.session}:{identity.recording_part}"
            references[key] = identity.fingerprint
            publication = payload.gate3c_publication
            publication_key = (
                f"immutable-publication:gate3c:{publication.scope.identity}"
            )
            references[publication_key] = publication.generation_id
        for child in result.children:
            visit(child)

    for child in children:
        visit(child)
    if not references:
        raise PublicationError("Timing publication has no validated WS3 timing input identities.")
    return dict(sorted(references.items()))


def _selected_part_dirs(
    request: TimingWorkflowRequest,
    config: dict[str, Any],
    participant: str,
    root: Path,
) -> list[Path] | OperationResult:
    available = sorted(
        (path for path in subject_dir(config, participant, root=root).glob("part*") if path.is_dir()),
        key=lambda path: path.name,
    )
    requested = request.requested_recording_parts
    if requested is None:
        if available:
            return available
        return _result(
            request,
            OperationStatus.INVALID_INPUT,
            "recording_parts_missing",
            f"Required aligned EEG timing input unavailable for {participant}: no recording-part directory.",
            participant=participant,
        )
    by_name = {path.name: path for path in available}
    missing = [part for part in requested if part not in by_name]
    if missing:
        return _result(
            request,
            OperationStatus.INVALID_INPUT,
            "requested_recording_part_missing",
            f"Requested timing recording part(s) unavailable for {participant}: {', '.join(missing)}.",
            participant=participant,
        )
    return [by_name[part] for part in requested]


def _execute_part(
    request: TimingWorkflowRequest,
    participant: str,
    part_dir: Path,
    behaviour: pd.DataFrame,
    behaviour_path: Path,
    trigger: pd.DataFrame | None,
    trigger_path: Path | None,
    settings: dict[str, Any],
    config: dict[str, Any],
    root: Path,
) -> tuple[OperationResult, pd.DataFrame | None, list[Path]]:
    part = part_dir.name
    shared = subject_dir(config, participant, root=root) / "alignment" / "trial_alignment.csv"
    local = part_dir / "alignment" / "trial_alignment.csv"
    alignment_path = local if local.is_file() else shared
    used = [behaviour_path]
    if trigger_path is not None:
        used.append(trigger_path)
    if not alignment_path.is_file():
        return (
            _result(
                request,
                OperationStatus.INVALID_INPUT,
                "alignment_missing",
                f"Required aligned EEG timing input unavailable: {alignment_path}",
                participant=participant,
                recording_part=part,
            ),
            None,
            used,
        )
    used.append(alignment_path)
    try:
        alignment = pd.read_csv(alignment_path)
        identity_path = alignment_identity_path(alignment_path)
        artifact_identity = read_alignment_artifact_identity(identity_path)
        validate_alignment_artifact_identity(
            artifact_identity,
            expected_contexts=artifact_identity.contexts,
            alignment_path=alignment_path,
        )
        contexts = [
            item for item in artifact_identity.contexts
            if (item.participant, item.session, item.recording_part) == (participant, request.session, part)
        ]
        if len(contexts) != 1:
            raise StageIdentityError(
                f"Alignment identity does not contain exactly one context for {participant} {part}."
            )
        context = contexts[0]
        current = request.workflow_request.resolved_config
        if context.scientific_fingerprint != current.scientific_fingerprint:
            raise StageIdentityError("Timing alignment scientific context does not match the resolved configuration.")
        # Timing consumes the immutable alignment identity and timing science.
        # Cohort membership and source-exposure registration are downstream
        # namespaces and must not stale an otherwise scientifically current
        # participant alignment.
        gate3c_publication = _validate_timing_gate3c_publication(
            request,
            participant,
            part_dir,
            context.resolved_config_fingerprint,
        )

        selected_behaviour = behaviour
        part_alignment = alignment[alignment["recording_part"].astype(str).eq(part)]
        expected = ()
        if "source_part" in behaviour and behaviour["source_part"].notna().any():
            expected_values = (
                part_alignment.get("metadata_part", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
            )
            expected = tuple(str(value) for value in expected_values)
            if expected:
                selected_behaviour = behaviour[
                    behaviour["source_part"].astype(str).isin(expected)
                ]
        if selected_behaviour.empty:
            raise ValueError(
                f"No behavioral timing rows matched alignment metadata_part for {participant} {part}."
            )
        behavioral_inputs = identify_behavioral_inputs(behavioral_log_path=behaviour_path)
        input_identity = build_timing_qc_input_identity(
            context=context,
            alignment_artifact=artifact_identity,
            behavioral_inputs=behavioral_inputs,
            behavioral=selected_behaviour,
            alignment=alignment,
            settings=settings,
            alignment_path=alignment_path,
        )
        trigger_identity = (
            identify_source_file(trigger_path, "response_timing_trigger_log")
            if trigger_path is not None
            else None
        )
        table = build_trial_level_qc(
            selected_behaviour,
            alignment,
            trigger,
            subject=participant,
            recording_part=part,
            settings=settings,
            input_identity=input_identity,
        )
        artifacts = [
            ArtifactReference("behavioral_log", behaviour_path, produced=False),
            ArtifactReference("alignment", alignment_path, produced=False),
            ArtifactReference("alignment_identity", identity_path, produced=False),
        ]
        if trigger_path is not None:
            artifacts.append(ArtifactReference("trigger_log", trigger_path, produced=False))
        return (
            _result(
                request,
                OperationStatus.SUCCEEDED,
                "completed",
                f"Timing-QC computation completed for {participant} {part}.",
                participant=participant,
                recording_part=part,
                artifacts=tuple(artifacts),
                computation_completed=True,
                payload=TimingPartPayload(
                    table=table,
                    inputs=TimingInputSelection(
                        behavioral_log_path=behaviour_path,
                        behavioral_inputs=behavioral_inputs,
                        trigger_log_path=trigger_path,
                        trigger_log_identity=trigger_identity,
                        alignment_path=alignment_path,
                        alignment_artifact=artifact_identity,
                        timing_input_identity=input_identity,
                        metadata_parts=expected,
                    ),
                    gate3c_publication=gate3c_publication,
                ),
            ),
            table,
            used,
        )
    except (PublicationError, SourceIdentityError, StageIdentityError) as error:
        return (
            _result(
                request,
                OperationStatus.STALE_INPUT,
                "timing_identity_stale_or_invalid",
                str(error),
                participant=participant,
                recording_part=part,
            ),
            None,
            used,
        )
    except (FileNotFoundError, ValueError) as error:
        return (
            _result(
                request,
                OperationStatus.INVALID_INPUT,
                "timing_input_invalid",
                str(error),
                participant=participant,
                recording_part=part,
            ),
            None,
            used,
        )
    except Exception as error:
        return (
            _result(
                request,
                OperationStatus.FAILED,
                "unexpected_timing_exception",
                f"Unexpected {type(error).__name__}: {error}",
                participant=participant,
                recording_part=part,
                developer_details="".join(traceback.format_exception(error)),
            ),
            None,
            used,
        )


def _validate_timing_gate3c_publication(
    request: TimingWorkflowRequest,
    participant: str,
    part_dir: Path,
    alignment_context_reference: str,
):
    """Validate timing's participant upstream without downstream-context equality."""

    publication = validate_current_publication(
        part_dir,
        PublicationScope(
            stage="gate3c",
            session=request.session,
            participant=participant,
            recording_part=part_dir.name,
        ),
    )
    ancestors = validate_participant_publication_chain(
        part_dir,
        publication,
        immutable_upstream_canonical_directory=(
            request.workflow_request.reviewed_upstream_part_dir(participant, part_dir.name)
            if request.workflow_request.gate3.isolated_replay
            else None
        ),
    )
    ancestor_stages = {item.scope.stage for item in ancestors}
    if not {"gate3a", "gate3b"}.issubset(ancestor_stages):
        raise PublicationError(
            "Current Gate 3C publication lacks exact immutable Gate 3A/Gate 3B lineage."
        )
    validate_publication_scientific_context(
        publication,
        request.workflow_request.resolved_config.scientific_fingerprint,
    )
    if publication.context_reference != alignment_context_reference:
        raise PublicationError(
            "Timing alignment configuration authority does not match its Gate 3C publication."
        )
    return publication


def _result(
    request: TimingWorkflowRequest,
    status: OperationStatus,
    reason_code: str,
    diagnostic: str,
    *,
    participant: str | None = None,
    recording_part: str | None = None,
    **kwargs: Any,
) -> OperationResult:
    return OperationResult(
        operation=Operation.RESPONSE_TIMING_QC,
        request_id=request.request_id,
        status=status,
        reason_code=reason_code,
        diagnostic=diagnostic,
        participant=participant,
        recording_part=recording_part,
        session=request.session,
        context_reference=request.workflow_request.resolved_config.resolved_config_fingerprint,
        **kwargs,
    )


def _cancelled(
    request: TimingWorkflowRequest,
    *,
    participant: str | None = None,
    recording_part: str | None = None,
) -> OperationResult:
    return _result(
        request,
        OperationStatus.CANCELLED,
        "operation_cancelled",
        "Cancellation was observed at a safe timing-workflow boundary.",
        participant=participant,
        recording_part=recording_part,
    )


def _aggregate_scope(
    request: TimingWorkflowRequest,
    children: tuple[OperationResult, ...],
    *,
    participant: str,
    diagnostic: str,
) -> OperationResult:
    if not children:
        return _result(
            request,
            OperationStatus.INVALID_INPUT,
            "recording_parts_missing",
            diagnostic,
            participant=participant,
        )
    status = _aggregate_status(children)
    return _result(
        request,
        status,
        "children_complete" if status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS} else "child_operation_not_successful",
        diagnostic,
        participant=participant,
        warnings=tuple(item for child in children for item in child.warnings),
        artifacts=tuple(item for child in children for item in child.artifacts),
        computation_completed=all(child.computation_completed for child in children),
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
