from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_software.workflows.kawaii_session2.config import load_session2_eeg_config  # noqa: E402
from eeg_software.workflows.kawaii_session2.metadata import load_metadata_for_recording_part  # noqa: E402
from eeg_software.workflows.kawaii_session2.source_exposure import apply_source_exposure_disposition  # noqa: E402


EXPECTED = {
    "sub005": {"verified_old": 200, "verified_new": 200, "ambiguous": 0, "Hit": 64, "Miss": 136, "False Alarm": 31, "Correct Rejection": 169},
    "sub009": {"verified_old": 102, "verified_new": 102, "ambiguous": 0, "Hit": 47, "Miss": 55, "False Alarm": 24, "Correct Rejection": 78},
    "sub010": {"verified_old": 200, "verified_new": 200, "ambiguous": 0, "Hit": 140, "Miss": 60, "False Alarm": 51, "Correct Rejection": 149},
    "sub011": {"verified_old": 200, "verified_new": 200, "ambiguous": 0, "Hit": 77, "Miss": 123, "False Alarm": 39, "Correct Rejection": 161},
    "sub012": {"verified_old": 98, "verified_new": 98, "ambiguous": 204, "Hit": 76, "Miss": 22, "False Alarm": 39, "Correct Rejection": 59},
    "sub013": {"verified_old": 198, "verified_new": 202, "ambiguous": 0, "Hit": 53, "Miss": 145, "False Alarm": 16, "Correct Rejection": 186},
    "sub014": {"verified_old": 200, "verified_new": 200, "ambiguous": 0, "Hit": 111, "Miss": 89, "False Alarm": 24, "Correct Rejection": 176},
}


def main() -> int:
    config = load_session2_eeg_config(root=ROOT)
    report: dict[str, dict] = {}
    for subject, expected in EXPECTED.items():
        source = load_metadata_for_recording_part(subject, "part1", config, root=ROOT)
        original = source.dataframe.copy()
        result = apply_source_exposure_disposition(
            original,
            subject=subject,
            session="session2",
            recording_part="part1",
            config=config,
            root=ROOT,
        )
        metadata = result.metadata
        eligible = metadata.loc[metadata["recognition_eligible"].astype(bool)].copy()
        old_new = eligible["source_verified_old_new_status"].astype(str).str.lower()
        response = eligible["recognition_response"].astype(str).str.lower()
        recognition = {
            "Hit": int((old_new.eq("old") & response.eq("yes")).sum()),
            "Miss": int((old_new.eq("old") & response.eq("no")).sum()),
            "False Alarm": int((old_new.eq("new") & response.eq("yes")).sum()),
            "Correct Rejection": int((old_new.eq("new") & response.eq("no")).sum()),
        }
        observed = {
            "verified_old": result.summary["verified_old"],
            "verified_new": result.summary["verified_new"],
            "ambiguous": result.summary["ambiguous_ineligible"],
            **recognition,
        }
        if observed != expected:
            raise RuntimeError(f"STOP: {subject} acceptance mismatch: observed={observed!r}; expected={expected!r}")
        if len(metadata) != len(original):
            raise RuntimeError(f"STOP: {subject} exposure annotation globally removed rows.")
        if int(metadata["kawaii_rating"].notna().sum()) != int(original["kawaii_rating"].notna().sum()):
            raise RuntimeError(f"STOP: {subject} exposure annotation changed categorical cuteness availability.")
        report[subject] = {
            **observed,
            "recognition_eligible_total": int(metadata["recognition_eligible"].sum()),
            "continuous_cuteness_old_new_eligible_total": int(metadata["continuous_cuteness_old_new_eligible"].sum()),
            "categorical_cuteness_rows_preserved": int(len(metadata)),
            "timing_rows_preserved_by_exposure_layer": int(len(metadata)),
            "recognition_delay_days": result.summary["recognition_delay_days"],
            "protocol_deviation": result.summary["protocol_deviation"],
            "protocol_disposition": result.summary["protocol_disposition"],
            "artifact_id": result.summary["artifact_id"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
