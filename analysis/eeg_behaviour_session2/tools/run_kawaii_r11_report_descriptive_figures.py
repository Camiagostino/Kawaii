"""Generate report-only descriptive figures from the current R11 group publication."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_software.workflows.kawaii_session2.report_descriptive_figures import (  # noqa: E402
    render_r11_report_descriptive_figures,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "workspace/kawaii/session2/corrections/r11_qc_v3",
        help="Root containing the current published R11 group generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "workspace/kawaii/session2/report_figures/r11_qc_v3",
        help="Non-canonical destination for the report figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this workflow's named files in an existing output directory.",
    )
    args = parser.parse_args()
    result = render_r11_report_descriptive_figures(
        args.source_root,
        args.output_dir,
        project_root=ROOT,
        overwrite=args.overwrite,
    )
    print(f"Rendered {len(result['figures'])} figures from group generation {result['source'].generation_id}")
    print(f"Manifest: {result['manifest']}")
    print("Canonical EEG analysis outputs modified: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
