from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from .bad_channel_disposition import (
    disposition_store_path,
    read_bad_channel_interpolation_disposition,
)
from .ica import gate3b_output_paths
from .ica_identity import ICAFitIdentityError, ica_fit_identity_path, read_ica_fit_artifact_identity
from .manual_review import (
    manual_review_paths,
    read_bad_channel_review_decision,
    read_ica_review_decision,
    read_review_notes,
)
from .manual_review_persistence import (
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    locked_store_set,
    write_yaml_document,
)
from .paths import subject_part_dir, subject_part_stage_dir
from .provenance import PIPELINE_VERSION, build_configuration_provenance, git_commit


def manual_review_snapshot_path(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> Path:
    """Return the per-part manual-review snapshot path for the current outputs."""

    return subject_part_stage_dir(config, subject, recording_part, "provenance", root=root, create=True) / "manual_review_snapshot.yaml"


def build_manual_review_snapshot(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the manual-review decisions actually used for this run."""

    config_provenance = build_configuration_provenance(config_path, root=root)
    ica_review = read_ica_review_decision(config, subject, recording_part, root=root)
    bad_channel_review = read_bad_channel_review_decision(config, subject, recording_part, root=root)
    bad_channel_disposition = read_bad_channel_interpolation_disposition(
        config, subject, recording_part, root=root
    )
    gate3b_outputs = gate3b_output_paths(config, subject, recording_part, root=root)
    suggestions_path = gate3b_outputs["eog_suggestions"]
    suggested_components: list[int] = []
    if suggestions_path.exists():
        suggestions = pd.read_csv(suggestions_path)
        if not suggestions.empty and "suggested_ocular_component" in suggestions:
            suggested_components = sorted(set(
                pd.to_numeric(
                    suggestions.loc[suggestions["suggested_ocular_component"].astype(bool), "component"], errors="coerce"
                ).dropna().astype(int).tolist()
            ))
    current_ica_solution_id: str | None = None
    current_ica_component_count: int | None = None
    ica_decision_state = ica_review.decision_state
    fit_identity_path = ica_fit_identity_path(gate3b_outputs["ica_solution"])
    if fit_identity_path.exists():
        try:
            fit_artifact = read_ica_fit_artifact_identity(fit_identity_path)
            current_ica_solution_id = fit_artifact.solution_fingerprint
            current_ica_component_count = int(
                fit_artifact.fit_context.algorithm_parameters["n_components_effective"]
            )
            if ica_review.complete and (
                ica_review.ica_solution_id != current_ica_solution_id
                or ica_review.fitted_component_count != current_ica_component_count
            ):
                ica_decision_state = "STALE_OR_MISMATCHED"
        except (ICAFitIdentityError, KeyError, TypeError, ValueError):
            if ica_review.complete:
                ica_decision_state = "STALE_OR_MISMATCHED"
    if (
        ica_review.complete
        and ica_review.bad_channel_decision_id != bad_channel_review.semantic_decision_id
    ):
        ica_decision_state = "STALE_OR_MISMATCHED"
    return {
        "subject": subject,
        "recording_part": recording_part,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": git_commit(root) or "unavailable",
        "source_config_hash": config_provenance["source_config_hash"],
        "preprocessing_configuration_hash": config_provenance["preprocessing_configuration_hash"],
        "configuration_authority": config_provenance["configuration_authority"],
        "legacy_config_fields_authority": config_provenance["legacy_config_fields_authority"],
        "bad_channel_review_status": bad_channel_review.review_status,
        "bad_channel_session": bad_channel_review.session,
        "bad_channel_decision_schema_version": bad_channel_review.decision_schema_version,
        "bad_channel_decision_state": bad_channel_review.decision_state,
        "bad_channel_review_complete": bad_channel_review.complete,
        "bad_channel_decision_id": bad_channel_review.decision_id,
        "bad_channel_semantic_decision_id": bad_channel_review.semantic_decision_id,
        "bad_channel_audit_record_id": bad_channel_review.audit_record_id,
        "bad_channel_source_recording_id": bad_channel_review.source_recording_id,
        "bad_channel_review_context_id": bad_channel_review.review_context_id,
        "bad_channel_reviewed_channels": list(bad_channel_review.reviewed_channels),
        "bad_channel_reviewer": bad_channel_review.reviewer,
        "bad_channel_reviewed_at_utc": bad_channel_review.reviewed_at_utc,
        "bad_channel_legacy_ambiguous": bad_channel_review.legacy_ambiguous,
        "confirmed_bad_channels": list(bad_channel_review.confirmed_bad_channels) if bad_channel_review.complete else [],
        "bad_channel_interpolation_disposition_status": bad_channel_disposition.approval_status,
        "bad_channel_interpolation_disposition": bad_channel_disposition.disposition,
        "bad_channel_interpolation_disposition_channels": list(bad_channel_disposition.channels),
        "bad_channel_interpolation_disposition_id": bad_channel_disposition.decision_id,
        "bad_channel_interpolation_disposition_bad_channel_decision_id": bad_channel_disposition.bad_channel_decision_id,
        "bad_channel_interpolation_disposition_review_context_id": bad_channel_disposition.bad_channel_review_context_id,
        "bad_channel_interpolation_disposition_scientific_fingerprint": bad_channel_disposition.scientific_fingerprint,
        "bad_channel_interpolation_disposition_reviewer": bad_channel_disposition.reviewer,
        "bad_channel_interpolation_disposition_reviewed_at_utc": bad_channel_disposition.reviewed_at_utc,
        "ica_review_status": ica_review.review_status,
        "ica_decision_state": ica_decision_state,
        "ica_review_complete": ica_review.complete,
        "ica_decision_schema_version": ica_review.decision_schema_version,
        "ica_decision_id": ica_review.decision_id,
        "ica_audit_record_id": ica_review.audit_record_id,
        "ica_session": ica_review.session,
        "ica_solution_id": ica_review.ica_solution_id,
        "current_ica_solution_id": current_ica_solution_id,
        "ica_fitted_component_count": ica_review.fitted_component_count,
        "current_ica_fitted_component_count": current_ica_component_count,
        "ica_bad_channel_decision_id": ica_review.bad_channel_decision_id,
        "ica_reviewer": ica_review.reviewer,
        "ica_reviewed_at_utc": ica_review.reviewed_at_utc,
        "ica_legacy_ambiguous": ica_review.legacy_ambiguous,
        "automatic_eog_suggested_components": suggested_components,
        "confirmed_ica_components": list(ica_review.confirmed_ica_components) if ica_review.complete else [],
        "notes": read_review_notes(config, subject, recording_part, root=root),
    }


def write_manual_review_snapshot(
    config: Mapping[str, Any],
    subject: str,
    recording_part: str,
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Path:
    """Write the review snapshot associated with the current per-part outputs."""

    path = manual_review_snapshot_path(config, subject, recording_part, root=root)
    decision_paths = manual_review_paths(config, root=root)
    with locked_store_set(
        [
            decision_paths["bad_channels"],
            decision_paths["ica_components"],
            disposition_store_path(config, root=root),
        ],
        timeout_s=lock_timeout_s,
    ):
        payload = build_manual_review_snapshot(
            config, subject, recording_part, root=root, config_path=config_path
        )
        return write_yaml_document(path, payload, lock_timeout_s=lock_timeout_s)


def expected_preprocessing_outputs(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> list[dict[str, Any]]:
    """Return final preprocessing outputs expected for a complete part."""

    part_dir = subject_part_dir(config, subject, recording_part, root=root, create=True)
    alignment = part_dir / "alignment"
    raw_qc = part_dir / "raw_qc"
    preprocessing = part_dir / "preprocessing"
    ica = part_dir / "ica"
    epochs = part_dir / "epochs"
    provenance = part_dir / "provenance"
    outputs = [
        _expected("alignment", alignment / "trial_alignment.csv"),
        _expected("raw_qc", raw_qc / "raw_qc_summary.csv"),
        _expected("raw_qc", raw_qc / "channel_qc.csv"),
        _expected("raw_qc", raw_qc / "report.md"),
        _expected("preprocessing", preprocessing / "filtered_raw.fif"),
        _expected("ica", gate3b_output_paths(config, subject, recording_part, root=root)["ica_solution"]),
        _expected("ica", ica / "ica_review_template.yaml"),
        _expected("ica", ica / "report.md"),
        _expected("ica", ica / "component_summary.csv"),
        _expected("epochs", epochs / "all_image_epochs-epo.fif"),
        _expected("epochs", epochs / "block_processing_summary.csv"),
        _expected("epochs", epochs / "epoch_metadata.csv"),
        _expected("epochs", epochs / "rejection_table.csv"),
        _expected("epochs", epochs / "report.md"),
        _expected("provenance", provenance, filename="*.provenance.json", notes="At least one provenance JSON file exists."),
        _expected("provenance", provenance / "manual_review_snapshot.yaml"),
    ]
    outputs.extend(_expected_block_epoch_outputs(epochs / "block_processing_summary.csv", epochs))
    return outputs


def build_validation_manifest(expected_outputs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build one validation-manifest row per expected output."""

    rows = []
    for item in expected_outputs:
        path = Path(item["path"])
        filename = str(item["filename"])
        exists = _expected_exists(path, filename)
        rows.append(
            {
                "stage": item["stage"],
                "filename": filename,
                "exists": bool(exists),
                "status": "ok" if exists else "missing",
                "notes": item.get("notes", ""),
            }
        )
    return pd.DataFrame(rows, columns=["stage", "filename", "exists", "status", "notes"])


def write_validation_manifest(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> Path:
    """Write validation_manifest.csv and return its path."""

    part_dir = subject_part_dir(config, subject, recording_part, root=root, create=True)
    path = part_dir / "validation_manifest.csv"
    manifest = build_validation_manifest(expected_preprocessing_outputs(config, subject, recording_part, root=root))
    manifest.to_csv(path, index=False)
    return path


def missing_expected_outputs(config: Mapping[str, Any], subject: str, recording_part: str, *, root: Path | None = None) -> list[str]:
    """Return expected output filenames that are currently missing."""

    manifest = build_validation_manifest(expected_preprocessing_outputs(config, subject, recording_part, root=root))
    return manifest.loc[~manifest["exists"].astype(bool), "filename"].astype(str).tolist()


def _expected(stage: str, path: Path, *, filename: str | None = None, notes: str = "") -> dict[str, Any]:
    return {"stage": stage, "path": path, "filename": filename or path.name, "notes": notes}


def _expected_block_epoch_outputs(summary_path: Path, epochs_dir: Path) -> list[dict[str, Any]]:
    """Return expected per-block epoch FIFs for blocks actually present."""

    if not summary_path.exists():
        return []
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return []
    if summary.empty or "block_index" not in summary:
        return []
    expected = []
    for block_index in sorted(set(pd.to_numeric(summary["block_index"], errors="coerce").dropna().astype(int).tolist())):
        expected.append(_expected("epochs", epochs_dir / f"block{block_index:02d}-epo.fif", notes="Per-block epoch FIF for an available retained experimental block."))
    return expected


def _expected_exists(path: Path, filename: str) -> bool:
    if filename == "*.provenance.json":
        return path.exists() and any(path.glob("*.provenance.json"))
    return path.exists()
