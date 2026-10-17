from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .config import CONFIG_NAME
from .resolved_snapshot import (
    ResolvedRunSnapshotError,
    read_resolved_run_snapshot,
    split_authoritative_config_provenance,
)

PIPELINE_VERSION = "session2_eeg_v1.0"


def config_hash(config_path: str | Path | None) -> str | None:
    """Return a SHA256 hash for a config file, or ``None`` when unavailable."""

    if config_path is None:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(root: str | Path | None = None) -> str | None:
    """Return the current git commit if available, without requiring git."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def runtime_versions() -> dict[str, str]:
    """Return runtime versions for reproducibility without requiring optional imports."""

    versions = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        import mne

        versions["mne_version"] = str(mne.__version__)
    except Exception:
        versions["mne_version"] = "unavailable"
    try:
        import numpy as np

        versions["numpy_version"] = str(np.__version__)
    except Exception:
        versions["numpy_version"] = "unavailable"
    return versions


def build_provenance_record(
    *,
    pipeline_stage: str,
    subject: str,
    output_file: str | Path,
    root: str | Path | None = None,
    recording_part: str | None = None,
    config_path: str | Path | None = None,
    input_files: Sequence[str | Path] | None = None,
    parameters_summary: Mapping[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> dict[str, Any]:
    """Build a JSON-serialisable provenance record for a pipeline output."""

    project_root = Path(root).resolve() if root else None
    config_provenance = build_configuration_provenance(config_path, root=project_root)
    resolved_config = config_provenance["source_config_path"]
    cfg_hash = config_provenance["source_config_hash"]
    configuration_authority = config_provenance["configuration_authority"]
    versions = runtime_versions()
    manual_review_snapshot = _manual_review_snapshot_for_output(Path(output_file))
    return {
        "pipeline_stage": pipeline_stage,
        "pipeline_version": pipeline_version,
        "subject": subject,
        "recording_part": recording_part,
        "config_path": _string_or_none(resolved_config),
        "config_hash": cfg_hash,
        "source_config_hash": cfg_hash,
        "preprocessing_configuration_hash": config_provenance["preprocessing_configuration_hash"],
        "configuration_authority": configuration_authority,
        "legacy_config_fields_authority": config_provenance["legacy_config_fields_authority"],
        "git_commit": git_commit(project_root) or "unavailable",
        "mne_version": versions["mne_version"],
        "numpy_version": versions["numpy_version"],
        "python_version": versions["python_version"],
        "manual_review_snapshot_path": _string_or_none(manual_review_snapshot) if manual_review_snapshot and manual_review_snapshot.exists() else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_files": [_string_or_none(path) for path in (input_files or [])],
        "output_file": _string_or_none(output_file),
        "parameters_summary": dict(parameters_summary or {}),
        "warnings": list(warnings or []),
    }


def build_configuration_provenance(
    config_path: str | Path | None,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve source-only legacy fields and an optional snapshot authority."""

    project_root = Path(root).resolve() if root else None
    source_config_path, snapshot_reference = split_authoritative_config_provenance(config_path)
    resolved_config = _resolve_config_path(source_config_path, project_root)
    cfg_hash = config_hash(resolved_config)
    configuration_authority = None
    if snapshot_reference is not None:
        snapshot = read_resolved_run_snapshot(snapshot_reference.path)
        if snapshot.snapshot_content_sha256 != snapshot_reference.snapshot_content_sha256:
            raise ResolvedRunSnapshotError("Resolved snapshot reference content digest is stale or inconsistent.")
        if snapshot.source_config.get("sha256") != cfg_hash:
            raise ResolvedRunSnapshotError(
                "Selected source config content does not match the authoritative resolved snapshot."
            )
        selected_path = snapshot.run_request.get("selected_config_path")
        if resolved_config is None or not isinstance(selected_path, str) or Path(selected_path).resolve() != resolved_config.resolve():
            raise ResolvedRunSnapshotError(
                "Selected source config path does not match the authoritative resolved snapshot."
            )
        configuration_authority = {
            "status": "authoritative_resolved_run_snapshot",
            "snapshot_path": str(snapshot.path),
            "snapshot_schema_version": str(snapshot.payload["schema_version"]),
            "snapshot_state": snapshot.snapshot_state,
            "snapshot_content_sha256": snapshot.snapshot_content_sha256,
            "source_config_sha256": str(snapshot.source_config["sha256"]),
            "scientific_fingerprint": snapshot.scientific_fingerprint,
            "resolved_config_fingerprint": snapshot.resolved_config_fingerprint,
        }
    return {
        "source_config_path": resolved_config,
        "source_config_hash": cfg_hash,
        "preprocessing_configuration_hash": (
            configuration_authority["resolved_config_fingerprint"]
            if configuration_authority is not None
            else cfg_hash
        ),
        "configuration_authority": configuration_authority,
        "legacy_config_fields_authority": (
            "config_path/config_hash identify source YAML only; configuration_authority is authoritative"
            if configuration_authority is not None
            else "legacy_source_yaml_only_no_resolved_snapshot"
        ),
    }


def write_provenance_sidecar(
    output_file: str | Path,
    *,
    pipeline_stage: str,
    subject: str,
    root: str | Path | None = None,
    recording_part: str | None = None,
    config_path: str | Path | None = None,
    input_files: Sequence[str | Path] | None = None,
    parameters_summary: Mapping[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> Path:
    """Write provenance JSON and return its path.

    New subject-centred EEG outputs store provenance together under the
    recording part's ``provenance/`` folder. Legacy/non-subject-centred outputs
    keep the older adjacent ``<output>.provenance.json`` sidecar convention.
    """

    output_path = Path(output_file)
    sidecar = _provenance_path_for_output(output_path, pipeline_stage)
    record = build_provenance_record(
        pipeline_stage=pipeline_stage,
        pipeline_version=pipeline_version,
        subject=subject,
        recording_part=recording_part,
        config_path=config_path,
        root=root,
        input_files=input_files,
        output_file=output_path,
        parameters_summary=parameters_summary,
        warnings=warnings,
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=False), encoding="utf-8")
    return sidecar


def write_group_provenance_sidecar(
    output_file: str | Path,
    *,
    gate: str,
    branch: str,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
    input_files: Sequence[str | Path] | None = None,
    included_subjects: Sequence[str] | None = None,
    excluded_subjects: Sequence[str] | None = None,
    exclusion_reasons: Mapping[str, Any] | None = None,
    parameters_summary: Mapping[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> Path:
    """Write a provenance sidecar for group-level Session 2 EEG outputs."""

    parameters = {
        "gate": gate,
        "branch": branch,
        "included_subjects": list(included_subjects or []),
        "excluded_subjects": list(excluded_subjects or []),
        "exclusion_reasons": dict(exclusion_reasons or {}),
    }
    parameters.update(dict(parameters_summary or {}))
    return write_provenance_sidecar(
        output_file,
        pipeline_stage=f"{gate}_{branch}_group",
        subject="group",
        recording_part=branch,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary=parameters,
        warnings=warnings,
        pipeline_version=pipeline_version,
    )


def _provenance_path_for_output(output_path: Path, pipeline_stage: str) -> Path:
    parts = output_path.parts
    if "subjects" not in parts:
        return output_path.with_suffix(output_path.suffix + ".provenance.json")
    idx = parts.index("subjects")
    if len(parts) <= idx + 3:
        return output_path.with_suffix(output_path.suffix + ".provenance.json")
    part_dir = Path(*parts[: idx + 3])
    rel = output_path.relative_to(part_dir)
    stem = "__".join(rel.parts).replace(output_path.suffix, "")
    clean_stage = str(pipeline_stage).replace("/", "_").replace("\\", "_")
    return part_dir / "provenance" / f"{clean_stage}__{stem}.provenance.json"


def provenance_sidecar_path(output_path: str | Path, pipeline_stage: str) -> Path:
    """Return the deterministic tracked provenance path for an output."""

    return _provenance_path_for_output(Path(output_path), pipeline_stage)


def _manual_review_snapshot_for_output(output_path: Path) -> Path | None:
    parts = output_path.parts
    if "subjects" not in parts:
        return None
    idx = parts.index("subjects")
    if len(parts) <= idx + 3:
        return None
    part_dir = Path(*parts[: idx + 3])
    return part_dir / "provenance" / "manual_review_snapshot.yaml"


def _resolve_config_path(config_path: str | Path | None, root: Path | None) -> Path | None:
    if config_path is None and root is not None:
        return root / "config" / CONFIG_NAME
    if config_path is None:
        return None
    path = Path(config_path)
    if not path.is_absolute() and root is not None:
        return root / path
    return path


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
