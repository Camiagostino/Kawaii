from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from eeg_software.workflows.kawaii_session2.identity import identify_source_file  # noqa: E402
from eeg_software.workflows.kawaii_session2.source_exposure import (  # noqa: E402
    AMBIGUOUS_RUN4_EXPOSURE,
    DISPOSITION_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    VERIFIED_NEW,
    VERIFIED_OLD,
    build_source_exposure_participant_record,
    participant_manifest_identity_payload,
    source_exposure_artifact_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reviewed Kawaii current-cohort source-exposure snapshot.")
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--new-kawaii-root",
        type=Path,
        default=None,
        help="Legacy fallback root used only when a required file is absent from workspace/kawaii/session1/outputs.",
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "config")
    args = parser.parse_args()

    repo = args.repository_root.resolve()
    external = args.new_kawaii_root.resolve() if args.new_kawaii_root is not None else None
    archive = args.archive_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session1_outputs = repo / "workspace" / "kawaii" / "session1" / "outputs"

    participants: list[dict] = []
    rows: list[pd.DataFrame] = []

    def session2_log(subject: str) -> Path:
        return repo / "workspace" / "kawaii" / "session2" / "source" / "behavioral_logs" / f"{subject}_session2" / f"{subject}_session2_trial_log_analysis.csv"

    def source(path: Path, role: str, logical: str) -> dict:
        return identify_source_file(path, role, logical_path=logical).to_primitive()

    def authoritative(relative: str) -> tuple[Path, str]:
        workspace_path = session1_outputs / relative
        if workspace_path.is_file():
            return workspace_path, f"workspace/kawaii/session1/outputs/{relative}"
        if external is not None:
            fallback = external / relative
            if fallback.is_file():
                return fallback, f"external/new_kawaii/{relative}"
        raise FileNotFoundError(f"Required Session 1 source is absent from the authoritative workspace: {relative}")

    def participant_record(
        subject: str,
        participant_sources: list[dict],
        expected_counts: dict[str, int],
        *,
        protocol: dict | None = None,
        notes: list[str] | None = None,
    ) -> dict:
        protocol_value = protocol or {
            "recognition_delay_days": 1,
            "protocol_deviation": False,
            "protocol_disposition": "STANDARD",
        }
        notes_value = notes or []
        return build_source_exposure_participant_record(
            subject,
            participant_sources,
            expected_counts,
            protocol=protocol_value,
            notes=notes_value,
        )

    def pool_rows(subject: str, pool_path: Path, source_identity: str) -> pd.DataFrame:
        session2 = _read_session2(session2_log(subject), subject)
        pool = pd.read_csv(pool_path)
        _require_unique(pool, ["image_filename"], f"{subject} recognition pool")
        _require_values(pool, "subject_id", {subject}, f"{subject} recognition pool")
        _require_values(pool, "session1", {1}, f"{subject} recognition pool")
        pool_status = pool[["image_filename", "old_new_status"]].rename(columns={"old_new_status": "source_old_new"})
        merged = session2.merge(pool_status, on="image_filename", validate="one_to_one", how="left")
        if merged["source_old_new"].isna().any():
            raise ValueError(f"{subject}: Session 2 item is absent from the reviewed Session 1 pool.")
        return _disposition_rows(merged, source_identity=source_identity)

    def complete_exposure_participant(
        subject: str,
        expected_counts: dict[str, int],
        *,
        notes: list[str] | None = None,
    ) -> tuple[dict, pd.DataFrame]:
        """Build one standard complete-session disposition from execution evidence.

        The recognition pool supplies the candidate universe; the non-empty
        Session 1 image-feedback rows and independently written seen-image file
        must agree exactly with the pool's old set. Empty feedback images are
        accepted only for explicit too-slow trials, where no image was shown.
        """

        log_relative = f"task_logs/{subject}_session1/{subject}_session1_matlab_trial_log_analysis.csv"
        seen_relative = f"task_logs/{subject}_session1/{subject}_session1_matlab_seen_images.csv"
        pool_relative = f"session2_inputs/{subject}_session1_recognition_pool.csv"
        log_path, log_logical = authoritative(log_relative)
        seen_path, seen_logical = authoritative(seen_relative)
        pool_path, pool_logical = authoritative(pool_relative)
        session2_path = session2_log(subject)
        participant_sources = [
            source(log_path, "session1_execution_log", log_logical),
            source(seen_path, "session1_seen_images", seen_logical),
            source(pool_path, "session1_recognition_pool", pool_logical),
            source(
                session2_path,
                "session2_behavioral_log",
                f"workspace/kawaii/session2/source/behavioral_logs/{subject}_session2/{subject}_session2_trial_log_analysis.csv",
            ),
        ]
        record = participant_record(subject, participant_sources, expected_counts, notes=notes)
        disposition = pool_rows(subject, pool_path, record["source_identity"])

        execution = pd.read_csv(log_path)
        image_feedback = execution.loc[execution["condition"].eq("image_feedback")].copy()
        feedback_images = image_feedback["feedback_image"].fillna("").astype(str).str.strip()
        empty_feedback = feedback_images.eq("")
        if not image_feedback.loc[empty_feedback, "feedback_type"].astype(str).eq("too_slow").all():
            raise ValueError(f"{subject}: empty Session 1 image-feedback evidence is not confined to too-slow trials.")
        exposed = image_feedback.loc[~empty_feedback, "feedback_image"].astype(str)
        if exposed.duplicated(keep=False).any():
            raise ValueError(f"{subject}: duplicated non-empty Session 1 exposure evidence.")

        seen = pd.read_csv(seen_path)
        _require_unique(seen, ["image_filename"], f"{subject} seen-image source")
        exposed_set = set(exposed)
        seen_set = set(seen["image_filename"].astype(str))
        old_set = set(disposition.loc[disposition["source_exposure_status"].eq(VERIFIED_OLD), "image_filename"])
        _require_equal_sets(exposed_set, seen_set, f"{subject} execution/seen exposure")
        _require_equal_sets(exposed_set, old_set, f"{subject} exposure/pool old set")
        _require_expected_counts(disposition, expected_counts, subject)
        return record, disposition

    # Standard complete-session evidence: execution and seen-image sources
    # independently establish the exact old set represented by the pool.
    sub005_record, sub005_rows = complete_exposure_participant(
        "sub005",
        _expected(200, 200, 0, 64, 136, 31, 169),
    )
    participants.append(sub005_record)
    rows.append(sub005_rows)

    # sub009: direct run-2 exposure plus independent complete pool.
    sub009_log, sub009_logical = authoritative("task_logs/sub009_session1/sub009_session1_matlab_trial_log_analysis.csv")
    sub009_pool, sub009_pool_logical = authoritative("session2_inputs/sub009_session1_recognition_pool.csv")
    sub009_s2 = session2_log("sub009")
    sub009_sources = [
        source(sub009_log, "session1_execution_log", sub009_logical),
        source(sub009_pool, "session1_recognition_pool", sub009_pool_logical),
        source(sub009_s2, "session2_behavioral_log", "workspace/kawaii/session2/source/behavioral_logs/sub009_session2/sub009_session2_trial_log_analysis.csv"),
    ]
    sub009_record = participant_record(
        "sub009",
        sub009_sources,
        _expected(102, 102, 0, 47, 55, 24, 78),
        notes=["Session 1 run 1 was lost; the current recognition set uses the surviving verified run-2 pool."],
    )
    sub009_rows = pool_rows("sub009", sub009_pool, sub009_record["source_identity"])
    exposed009 = set(
        pd.read_csv(sub009_log).loc[lambda frame: frame["run"].eq(2) & frame["condition"].eq("image_feedback"), "feedback_image"].astype(str)
    )
    _require_equal_sets(exposed009, set(sub009_rows.loc[sub009_rows["source_exposure_status"].eq(VERIFIED_OLD), "image_filename"]), "sub009 run-2 exposure")
    participants.append(sub009_record)
    rows.append(sub009_rows)

    # sub010: preserve both source families and explicitly verify the one-row restart overlap.
    sub010_first, sub010_first_logical = authoritative("task_logs/sub010_session1/sub010_session1_matlab_trial_log_analysis.csv")
    sub010_restart, sub010_restart_logical = authoritative("task_logs/sub010_2_session1/sub010_2_session1_matlab_trial_log_analysis.csv")
    sub010_pool, sub010_pool_logical = authoritative("session2_inputs/sub010_2_session1_recognition_pool.csv")
    sub010_s2 = session2_log("sub010")
    sub010_sources = [
        source(sub010_first, "session1_execution_log_pre_restart", sub010_first_logical),
        source(sub010_restart, "session1_execution_log_clean_restart", sub010_restart_logical),
        source(sub010_pool, "session1_recognition_pool", sub010_pool_logical),
        source(sub010_s2, "session2_behavioral_log", "workspace/kawaii/session2/source/behavioral_logs/sub010_session2/sub010_session2_trial_log_analysis.csv"),
    ]
    sub010_record = participant_record(
        "sub010",
        sub010_sources,
        _expected(200, 200, 0, 140, 60, 51, 149),
        notes=["Documented restart overlap: run 3, trial 201, item 7204.jpg; clean restart family is authoritative for runs 3-4."],
    )
    first010 = pd.read_csv(sub010_first)
    restart010 = pd.read_csv(sub010_restart)
    overlap_first = first010.loc[first010["trial"].eq(201), "feedback_image"].astype(str).tolist()
    overlap_restart = restart010.loc[restart010["trial"].eq(201), "feedback_image"].astype(str).tolist()
    if overlap_first != ["7204.jpg"] or overlap_restart != ["7204.jpg"]:
        raise ValueError("sub010 documented restart overlap no longer matches run 3/trial 201/item 7204.jpg.")
    sub010_rows = pool_rows("sub010", sub010_pool, sub010_record["source_identity"])
    exposed010 = set(restart010.loc[restart010["condition"].eq("image_feedback"), "feedback_image"].astype(str))
    _require_equal_sets(exposed010, set(sub010_rows.loc[sub010_rows["source_exposure_status"].eq(VERIFIED_OLD), "image_filename"]), "sub010 clean restart exposure")
    participants.append(sub010_record)
    rows.append(sub010_rows)

    # sub011: the dated pre-Session-2 pool is the complete item-level source.
    sub011_log, sub011_logical = authoritative("task_logs/sub011_session1/sub011_session1_matlab_trial_log_analysis.csv")
    sub011_pool, sub011_pool_logical = authoritative("session2_inputs/sub011_session1_recognition_pool.csv")
    sub011_s2 = session2_log("sub011")
    sub011_sources = [
        source(sub011_log, "session1_execution_log", sub011_logical),
        source(sub011_pool, "session1_recognition_pool", sub011_pool_logical),
        source(sub011_s2, "session2_behavioral_log", "workspace/kawaii/session2/source/behavioral_logs/sub011_session2/sub011_session2_trial_log_analysis.csv"),
    ]
    sub011_record = participant_record(
        "sub011",
        sub011_sources,
        _expected(200, 200, 0, 77, 123, 39, 161),
        protocol={
            "recognition_delay_days": 2,
            "protocol_deviation": True,
            "protocol_disposition": "INCLUDED_FLAGGED",
        },
        notes=["Recovered authoritative Session 1 execution log verifies the complete gap-free 200-item exposure set; the dated recognition pool supplies the complete old/new candidate mapping."],
    )
    sub011_rows = pool_rows("sub011", sub011_pool, sub011_record["source_identity"])
    exposed011 = set(
        pd.read_csv(sub011_log).loc[lambda frame: frame["condition"].eq("image_feedback"), "feedback_image"].astype(str)
    )
    _require_equal_sets(exposed011, set(sub011_rows.loc[sub011_rows["source_exposure_status"].eq(VERIFIED_OLD), "image_filename"]), "sub011 execution exposure")
    participants.append(sub011_record)
    rows.append(sub011_rows)

    # sub012: actual run-3 presentation plus the authoritative run-3/run-4 candidate schedule.
    sub012_schedule = archive / "outputs" / "schedules" / "sub012_session1_schedule.csv"
    sub012_run1 = repo / "workspace" / "kawaii" / "session1" / "outputs" / "task_logs" / "sub012_session1" / "sub012_session1_matlab_trial_log_analysis.csv"
    sub012_run2 = repo / "workspace" / "kawaii" / "session1" / "outputs" / "task_logs" / "sub012_2_session1" / "sub012_2_session1_matlab_trial_log_analysis.csv"
    sub012_run3 = repo / "workspace" / "kawaii" / "session1" / "outputs" / "task_logs" / "sub012_3_session1" / "sub012_3_session1_matlab_trial_log_analysis.csv"
    sub012_seen3 = repo / "workspace" / "kawaii" / "session1" / "outputs" / "task_logs" / "sub012_3_session1" / "sub012_3_session1_matlab_seen_images.csv"
    sub012_s2 = session2_log("sub012")
    sub012_sources = [
        source(sub012_schedule, "session1_authoritative_schedule", "archive/outputs/schedules/sub012_session1_schedule.csv"),
        source(sub012_run1, "session1_execution_log_run1", "workspace/kawaii/session1/outputs/task_logs/sub012_session1/sub012_session1_matlab_trial_log_analysis.csv"),
        source(sub012_run2, "session1_execution_log_run2", "workspace/kawaii/session1/outputs/task_logs/sub012_2_session1/sub012_2_session1_matlab_trial_log_analysis.csv"),
        source(sub012_run3, "session1_execution_log_run3", "workspace/kawaii/session1/outputs/task_logs/sub012_3_session1/sub012_3_session1_matlab_trial_log_analysis.csv"),
        source(sub012_seen3, "session1_seen_images_run3", "workspace/kawaii/session1/outputs/task_logs/sub012_3_session1/sub012_3_session1_matlab_seen_images.csv"),
        source(sub012_s2, "session2_behavioral_log", "workspace/kawaii/session2/source/behavioral_logs/sub012_session2/sub012_session2_trial_log_analysis.csv"),
    ]
    sub012_record = participant_record(
        "sub012",
        sub012_sources,
        _expected(98, 98, 204, 76, 22, 39, 59),
        notes=["Run 4 has no execution artifact; its 204 candidate items are authorized ineligible only for old/new-dependent analyses."],
    )
    schedule012 = pd.read_csv(sub012_schedule)
    candidates3 = _candidate_items(schedule012.loc[schedule012["run"].eq(3)])
    candidates4 = _candidate_items(schedule012.loc[schedule012["run"].eq(4)])
    exposed3 = set(pd.read_csv(sub012_run3)["feedback_image"].astype(str))
    seen3 = set(pd.read_csv(sub012_seen3)["image_filename"].astype(str))
    if len(seen3) != 98 or exposed3 != seen3:
        raise ValueError("sub012 run-3 trial and seen-image exposure sets no longer agree at 98 items.")
    session012 = _read_session2(sub012_s2, "sub012")
    all_items = set(session012["image_filename"].astype(str))
    _require_equal_sets(all_items, candidates3 | candidates4, "sub012 Session 2 candidate universe")
    if candidates3 & candidates4 or len(candidates3) != 196 or len(candidates4) != 204:
        raise ValueError("sub012 run-3/run-4 candidate partition is no longer 196/204 disjoint items.")
    status_by_item = {
        **{item: VERIFIED_OLD for item in exposed3},
        **{item: VERIFIED_NEW for item in candidates3 - exposed3},
        **{item: AMBIGUOUS_RUN4_EXPOSURE for item in candidates4},
    }
    session012["source_exposure_status"] = session012["image_filename"].map(status_by_item)
    sub012_rows = _disposition_rows(session012, source_identity=sub012_record["source_identity"], status_column="source_exposure_status")
    participants.append(sub012_record)
    rows.append(sub012_rows)

    sub013_record, sub013_rows = complete_exposure_participant(
        "sub013",
        _expected(198, 202, 0, 53, 145, 16, 186),
        notes=[
            "Session 1 contained two explicit too-slow image-feedback trials (run 2, trials 143 and 178) where no feedback image was presented; the execution and seen-image sources therefore establish 198 old and 202 new items without ambiguity."
        ],
    )
    participants.append(sub013_record)
    rows.append(sub013_rows)

    sub014_record, sub014_rows = complete_exposure_participant(
        "sub014",
        _expected(200, 200, 0, 111, 89, 24, 176),
    )
    participants.append(sub014_record)
    rows.append(sub014_rows)

    manifest_identity_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "disposition_schema_version": DISPOSITION_SCHEMA_VERSION,
        "participants": [participant_manifest_identity_payload(item) for item in participants],
    }
    artifact_id = source_exposure_artifact_identity(participants)
    disposition = pd.concat(rows, ignore_index=True).sort_values(["subject", "trial"], kind="stable").reset_index(drop=True)
    disposition.insert(0, "schema_version", DISPOSITION_SCHEMA_VERSION)
    disposition["source_exposure_artifact_id"] = artifact_id

    table_path = output_dir / "kawaii_session2_source_exposure_disposition_v1.csv"
    disposition.to_csv(table_path, index=False, lineterminator="\n")
    artifact = identify_source_file(
        table_path,
        "source_exposure_disposition_table",
        logical_path=table_path.name,
    ).to_primitive()
    manifest = {
        **manifest_identity_payload,
        "artifact_id": artifact_id,
        "artifact": artifact,
        "participants": participants,
    }
    manifest_path = output_dir / "kawaii_session2_source_exposure_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "artifact_id": artifact_id,
        "manifest": str(manifest_path),
        "table": str(table_path),
        "rows": len(disposition),
        "counts": {
            f"{subject}:{status}": int(count)
            for (subject, status), count in disposition.groupby(["subject", "source_exposure_status"]).size().items()
        },
    }, indent=2, default=str))
    return 0


def _read_session2(path: Path, subject: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["subject_id", "session", "trial", "image_filename", "old_new_status", "recognition_response"]
    if any(column not in frame for column in required):
        raise ValueError(f"{subject}: Session 2 log lacks required columns.")
    _require_values(frame, "subject_id", {subject}, f"{subject} Session 2")
    _require_values(frame, "session", {2}, f"{subject} Session 2")
    _require_unique(frame, ["trial"], f"{subject} Session 2")
    _require_unique(frame, ["image_filename"], f"{subject} Session 2")
    return frame[required].copy()


def _disposition_rows(frame: pd.DataFrame, *, source_identity: str, status_column: str | None = None) -> pd.DataFrame:
    output = frame.copy()
    if status_column is None:
        output["source_exposure_status"] = output["source_old_new"].astype(str).str.lower().map(
            {"old": VERIFIED_OLD, "new": VERIFIED_NEW}
        )
    else:
        output["source_exposure_status"] = output[status_column]
    output["original_old_new_status"] = output["old_new_status"].astype(str).str.strip().str.lower()
    output["source_verified_old_new_status"] = output["source_exposure_status"].map(
        {VERIFIED_OLD: "old", VERIFIED_NEW: "new"}
    )
    verified = output["source_exposure_status"].isin([VERIFIED_OLD, VERIFIED_NEW])
    mismatch = verified & output["source_verified_old_new_status"].ne(output["original_old_new_status"])
    if mismatch.any():
        raise ValueError(f"Source-derived old/new mismatch at trial(s) {output.loc[mismatch, 'trial'].tolist()}.")
    output["recognition_eligible"] = verified
    output["continuous_cuteness_old_new_eligible"] = verified
    output["exposure_exclusion_reason"] = ""
    output.loc[output["source_exposure_status"].eq(AMBIGUOUS_RUN4_EXPOSURE), "exposure_exclusion_reason"] = "ambiguous_run4_exposure"
    output["source_exposure_identity"] = source_identity
    output["subject"] = output["subject_id"]
    output["session"] = "session2"
    return output[
        [
            "subject",
            "session",
            "trial",
            "image_filename",
            "original_old_new_status",
            "source_exposure_status",
            "source_verified_old_new_status",
            "recognition_eligible",
            "continuous_cuteness_old_new_eligible",
            "exposure_exclusion_reason",
            "source_exposure_identity",
            "recognition_response",
        ]
    ]


def _expected(old: int, new: int, ambiguous: int, hit: int, miss: int, false_alarm: int, correct_rejection: int) -> dict[str, int]:
    return {
        VERIFIED_OLD: old,
        VERIFIED_NEW: new,
        AMBIGUOUS_RUN4_EXPOSURE: ambiguous,
        "recognition_eligible": old + new,
        "continuous_cuteness_old_new_eligible": old + new,
        "Hit": hit,
        "Miss": miss,
        "False Alarm": false_alarm,
        "Correct Rejection": correct_rejection,
    }


def _require_expected_counts(frame: pd.DataFrame, expected: dict[str, int], subject: str) -> None:
    verified_old = frame["source_exposure_status"].eq(VERIFIED_OLD)
    verified_new = frame["source_exposure_status"].eq(VERIFIED_NEW)
    ambiguous = frame["source_exposure_status"].eq(AMBIGUOUS_RUN4_EXPOSURE)
    response = frame["recognition_response"].astype(str).str.lower()
    observed = {
        VERIFIED_OLD: int(verified_old.sum()),
        VERIFIED_NEW: int(verified_new.sum()),
        AMBIGUOUS_RUN4_EXPOSURE: int(ambiguous.sum()),
        "recognition_eligible": int(frame["recognition_eligible"].sum()),
        "continuous_cuteness_old_new_eligible": int(frame["continuous_cuteness_old_new_eligible"].sum()),
        "Hit": int((verified_old & response.eq("yes")).sum()),
        "Miss": int((verified_old & response.eq("no")).sum()),
        "False Alarm": int((verified_new & response.eq("yes")).sum()),
        "Correct Rejection": int((verified_new & response.eq("no")).sum()),
    }
    if observed != expected:
        raise ValueError(f"{subject}: source-exposure count mismatch: observed={observed!r}; expected={expected!r}.")


def _candidate_items(frame: pd.DataFrame) -> set[str]:
    return set(frame["feedback_image_if_left"].astype(str)) | set(frame["feedback_image_if_right"].astype(str))


def _require_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns, keep=False).any():
        raise ValueError(f"{label} contains duplicate key {columns!r}.")


def _require_values(frame: pd.DataFrame, column: str, expected: set, label: str) -> None:
    observed = set(frame[column].dropna().tolist())
    if observed != expected:
        raise ValueError(f"{label} {column} scope mismatch: {observed!r} != {expected!r}.")


def _require_equal_sets(left: set[str], right: set[str], label: str) -> None:
    if left != right:
        raise ValueError(f"{label} mismatch: left_only={sorted(left-right)[:10]!r}, right_only={sorted(right-left)[:10]!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
