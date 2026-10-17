"""Tracked CLI adapter foundation for the Session 2 EEG coordinator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .coordinator import Session2EEGCoordinator
from .operations import (
    GroupWorkflow,
    GroupWorkflowRequest,
    Operation,
    OperationResult,
    OperationStatus,
    ParticipantWorkflow,
    ParticipantWorkflowRequest,
    ProgressEvent,
    TimingWorkflowRequest,
)
from .workflow import Gate3ExecutionOptions, resolve_session2_eeg_run_request
from .participant_intake import ParticipantIntakeError, run_standard_participant_intake
from .r11_replay_preflight import validate_r11_replay_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kawaii-session2-eeg")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("operations", help="List coordinator operations.")
    participant = subcommands.add_parser("participant", help="Run one tracked participant workflow.")
    participant.add_argument("workflow", choices=("intake", *(item.value for item in ParticipantWorkflow)))
    participant.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing config and data roots.")
    participant.add_argument("--config", default="config/session2_eeg.yaml")
    participant.add_argument("--subjects", nargs="+", default=None)
    participant.add_argument("--parts", nargs="+", default=None)
    participant.add_argument("--allow-excluded", action="store_true")
    participant.add_argument("--skip-ica", action="store_true")
    participant.add_argument("--skip-ica-for-validation", action="store_true")
    participant.add_argument("--skip-epoching", action="store_true")
    participant.add_argument("--use-existing-reviewed-ica", action="store_true")
    participant.add_argument("--allow-incomplete-ica-review-for-validation", action="store_true")
    participant.add_argument("--clean-subject-output", action="store_true")
    participant.add_argument("--reviewed-upstream-root", type=Path, default=None)
    participant.add_argument("--corrected-output-root", type=Path, default=None)
    participant.add_argument(
        "--r11-replay-preflight",
        action="store_true",
        help="Run only the read-only metadata preflight; never execute participant stages.",
    )
    participant.add_argument("--dry-run", action="store_true", help="Validate intake without persisting artifacts/config.")
    group = subcommands.add_parser("group", help="Run tracked Gate 5 group workflow.")
    group.add_argument("workflow", choices=tuple(item.value for item in GroupWorkflow))
    group.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing config and data roots.")
    group.add_argument("--config", default="config/session2_eeg.yaml")
    group.add_argument("--branch", choices=("recognition", "cuteness"), default=None)
    timing = subcommands.add_parser("timing-qc", help="Run tracked retrospective response-timing QC.")
    timing.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing config and data roots.")
    timing.add_argument("--config", default="config/session2_eeg.yaml")
    timing.add_argument("--subjects", nargs="+", default=None)
    timing.add_argument("--parts", nargs="+", default=None)
    timing.add_argument("--allow-excluded", action="store_true")
    # Study-level entry points for behavioural and other secondary analyses.
    # The existing EEG entry point remains available for backwards compatibility.
    behaviour = subcommands.add_parser(
        "behaviour", help="Run behavioural recognition and rating-memory analyses."
    )
    secondary = subcommands.add_parser(
        "secondary", help="Run selected secondary behavioural/ERP analyses."
    )
    for subparser in (behaviour, secondary):
        subparser.add_argument("--root", type=Path, default=Path.cwd())
        subparser.add_argument("--config", default="config/session2_eeg.yaml")
        subparser.add_argument(
            "--secondary-config", default="config/kawaii_session2_secondary_analyses.yaml"
        )
        subparser.add_argument(
            "--check-inputs", action="store_true",
            help="Validate selected existing inputs without writing outputs.",
        )
    # Do not let this convenience wrapper change the frozen analysis family.
    secondary.add_argument(
        "--stages", nargs="+",
        choices=("recognition-behavior", "recognition-outcome-erp", "true-vs-false-erp", "rating-memory", "arousal-erp"),
        default=("recognition-behavior", "rating-memory"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "operations":
        for operation in Operation:
            if operation is not Operation.COORDINATED:
                print(operation.value)
        print("Participant, group, and response-timing execution use the tracked coordinator.")
        return 0
    if args.command in {"behaviour", "secondary"}:
        # Route through the existing implementation; no statistical code is duplicated.
        from .secondary_analyses.cli import main as secondary_main

        stages = (
            ("recognition-behavior", "rating-memory")
            if args.command == "behaviour" else args.stages
        )
        secondary_args = [
            "--root", str(args.root),
            "--config", str(args.config),
            "--secondary-config", str(args.secondary_config),
            "--stages", *stages,
        ]
        if args.check_inputs:
            secondary_args.append("--check-inputs")
        return secondary_main(secondary_args)
    if args.command == "group":
        workflow = GroupWorkflow(args.workflow)
        if workflow is GroupWorkflow.PREPARE and args.branch is not None:
            build_parser().error("Gate 5A preparation is an atomic two-branch workflow; omit --branch.")
        resolved = resolve_session2_eeg_run_request(root=args.root, config_path=args.config)
        request = GroupWorkflowRequest(resolved, (workflow,), branch=args.branch)
        result = Session2EEGCoordinator(progress_sink=_render_progress).execute_group(request)
        _render_result(result)
        return exit_code_for_result(result)
    if args.command == "timing-qc":
        resolved = resolve_session2_eeg_run_request(
            root=args.root,
            config_path=args.config,
            participants=args.subjects,
            recording_parts=args.parts,
            allow_excluded=args.allow_excluded,
        )
        request = TimingWorkflowRequest(resolved)
        result = Session2EEGCoordinator(progress_sink=_render_progress).execute_timing(request)
        _render_result(result)
        return exit_code_for_result(result)

    if args.workflow == "intake":
        if not args.subjects:
            build_parser().error("'participant intake' requires --subjects.")
        if any(
            (
                args.parts,
                args.allow_excluded,
                args.skip_ica,
                args.skip_ica_for_validation,
                args.skip_epoching,
                args.use_existing_reviewed_ica,
                args.allow_incomplete_ica_review_for_validation,
                args.clean_subject_output,
                args.reviewed_upstream_root,
                args.corrected_output_root,
                args.r11_replay_preflight,
            )
        ):
            build_parser().error("EEG/Gate 3 modifiers are not valid for 'participant intake'.")
        try:
            result = run_standard_participant_intake(
                root=args.root,
                config_path=args.config,
                subjects=args.subjects,
                persist=not args.dry_run,
            )
        except ParticipantIntakeError as error:
            print(f"INTAKE FAILED {error.reason_code}: {error}")
            return 3
        print(result.render_text())
        print("INTAKE_RESULT_JSON " + json.dumps(result.to_primitive(), sort_keys=True))
        return 0

    workflow = ParticipantWorkflow(args.workflow)
    reviewed_upstream_root = _resolve_execution_root(args.root, args.reviewed_upstream_root)
    corrected_output_root = _resolve_execution_root(args.root, args.corrected_output_root)
    gate3 = Gate3ExecutionOptions(
        skip_ica=args.skip_ica,
        validation_only_skip_ica=args.skip_ica_for_validation,
        skip_epoching=args.skip_epoching,
        use_existing_reviewed_ica=args.use_existing_reviewed_ica,
        allow_incomplete_ica_review_for_validation=args.allow_incomplete_ica_review_for_validation,
        clean_subject_output=args.clean_subject_output,
        reviewed_upstream_root=reviewed_upstream_root,
        corrected_output_root=corrected_output_root,
    )
    if workflow is not ParticipantWorkflow.GATE3 and gate3 != Gate3ExecutionOptions():
        build_parser().error("Gate 3 execution modifiers are valid only for 'participant gate3'.")
    resolved = resolve_session2_eeg_run_request(
        root=args.root,
        config_path=args.config,
        participants=args.subjects,
        recording_parts=args.parts,
        allow_excluded=args.allow_excluded,
        gate3=gate3,
    )
    if args.r11_replay_preflight:
        if workflow is not ParticipantWorkflow.GATE3 or not gate3.isolated_replay:
            build_parser().error("--r11-replay-preflight requires participant gate3 and both replay roots.")
        result = validate_r11_replay_preflight(
            upstream_root=gate3.reviewed_upstream_root,
            corrected_output_root=gate3.corrected_output_root,
            participant_parts=tuple(
                (participant, recording_part)
                for participant in resolved.realized.participants
                for recording_part in resolved.requested.recording_parts
            ),
            manual_review_root=resolved.resolved_config.resolved_paths["manual_review_dir"],
            immutable_source_roots={
                key: resolved.resolved_config.resolved_paths[key]
                for key in ("raw_eeg_dir", "session2_logs_dir", "session2_schedules_dir")
            },
        )
        print(result.render_text())
        print("R11_PREFLIGHT_RESULT_JSON " + json.dumps(result.to_primitive(), sort_keys=True))
        return 0 if result.go else 3
    request = ParticipantWorkflowRequest(resolved, workflow)
    result = Session2EEGCoordinator(progress_sink=_render_progress).execute_participant(request)
    _render_result(result)
    return exit_code_for_result(result)


def _resolve_execution_root(project_root: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return (value if value.is_absolute() else Path(project_root) / value).resolve(strict=False)


def exit_code_for_result(result: OperationResult) -> int:
    if result.status in {OperationStatus.SUCCEEDED, OperationStatus.SUCCEEDED_WITH_WARNINGS}:
        return 0
    if result.status is OperationStatus.REVIEW_REQUIRED:
        return 2
    if result.status in {OperationStatus.INVALID_INPUT, OperationStatus.STALE_INPUT}:
        return 3
    if result.status is OperationStatus.CANCELLED:
        return 4
    return 1


def _render_progress(event: ProgressEvent) -> None:
    scope = " ".join(value for value in (event.participant, event.recording_part, event.branch) if value)
    message = f" {event.message}" if event.message else ""
    print(f"[{event.sequence}] {event.phase.value} {event.operation.value} {scope}{message}".rstrip())


def _render_result(result: OperationResult) -> None:
    print(f"RESULT {result.operation.value} {result.status.value} {result.reason_code}")
    if result.diagnostic:
        print(result.diagnostic)


if __name__ == "__main__":
    raise SystemExit(main())
