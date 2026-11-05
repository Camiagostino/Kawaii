"""Portable direct-script entry point for the unified Session 2 descriptive cuteness command.

Normally use: kawaii-session2-eeg cuteness --rating-source participant|normative|both
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_software.workflows.kawaii_session2.cuteness_rating_workflow import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
