"""Command-line interface for Kawaii Session 2 secondary analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from .workflow import SUPPORTED_STAGES, run_secondary_analyses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kawaii-session2-secondary-analyses")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="config/session2_eeg.yaml")
    parser.add_argument(
        "--secondary-config",
        default="config/kawaii_session2_secondary_analyses.yaml",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=SUPPORTED_STAGES,
        default=SUPPORTED_STAGES,
    )
    parser.add_argument(
        "--check-inputs", action="store_true",
        help="Read-only validation of required source files for the chosen stages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_inputs:
        from .preflight import check_secondary_inputs

        summary = check_secondary_inputs(
            root=args.root,
            core_config_path=args.config,
            secondary_config_path=args.secondary_config,
            stages=args.stages,
        )
        print(f"INPUT_PREFLIGHT_PASS stages={','.join(summary['stages'])}")
        print(f"INPUT_PREFLIGHT_VERIFIED_FILES {summary['verified_file_count']}")
        for warning in summary["warnings"]:
            print(f"INPUT_PREFLIGHT_WARNING {warning}")
        return 0
    result = run_secondary_analyses(
        root=args.root,
        core_config_path=args.config,
        secondary_config_path=args.secondary_config,
        stages=args.stages,
    )
    print(f"SECONDARY_ANALYSES_MANIFEST {result.manifest_path}")
    for label, path in sorted(result.output_files.items()):
        print(f"OUTPUT {label} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
