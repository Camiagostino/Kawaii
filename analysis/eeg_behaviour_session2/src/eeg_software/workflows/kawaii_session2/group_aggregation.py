from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cuteness_mixed import center_cuteness_trials
from .paths import resolve_session2_eeg_paths, subject_dir
from .provenance import write_group_provenance_sidecar
from .publication import (
    PublicationScope,
    PublicationValidationError,
    validate_current_publication,
    validate_participant_publication_chain,
    validate_publication_scientific_context,
)
from .resolved_snapshot import ResolvedRunSnapshotReference, bind_authoritative_config_provenance
from .stage_identity import (
    StageIdentityError,
    configuration_mapping_fingerprint,
    participant_erp_identity_path,
    read_validated_participant_erp_artifact,
)
from .study_definition import erp_branch_settings
from .subject_selection import parse_subject_args, subject_sort_key


@dataclass(frozen=True)
class Gate5AResult:
    output_files: dict[str, Path]
    warnings: tuple[str, ...]
    participant_summary: pd.DataFrame
    participant_outcomes: tuple["Gate5AParticipantOutcome", ...] = ()
    publication_identity_references: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Gate5AParticipantOutcome:
    subject: str
    branch: str
    status: str
    reason: str


def resolve_gate5a_subjects(config: Mapping[str, Any]) -> list[str]:
    """Resolve Gate 5A subjects from the analysis config block."""

    analysis = dict(config.get("analysis", {}) or {})
    include = analysis.get("include_subjects")
    exclude = analysis.get("exclude_subjects", []) or []
    if include is None:
        raise ValueError("Missing analysis.include_subjects in Session 2 EEG config.")
    subjects = parse_subject_args(include, config=config)
    excluded = set(parse_subject_args(exclude, config=config))
    return [subject for subject in subjects if subject not in excluded]


def resolve_gate5a_summary_subjects(config: Mapping[str, Any]) -> list[str]:
    """Resolve subjects that should appear in Gate 5A participant summaries."""

    analysis = dict(config.get("analysis", {}) or {})
    include = analysis.get("include_subjects")
    exclude = analysis.get("exclude_subjects", []) or []
    if include is None:
        raise ValueError("Missing analysis.include_subjects in Session 2 EEG config.")
    subjects = set(parse_subject_args(include, config=config))
    subjects.update(parse_subject_args(exclude, config=config))
    return sorted(subjects, key=subject_sort_key)


def gate5a_config_exclusion_reasons(config: Mapping[str, Any]) -> dict[str, str]:
    analysis = dict(config.get("analysis", {}) or {})
    reasons = dict(analysis.get("exclusion_reasons", {}) or {})
    return {str(subject): str(reason) for subject, reason in reasons.items()}


def gate5a_group_output_paths(config: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Path]:
    """Return Gate 5A group aggregation output paths."""

    output_dir = resolve_session2_eeg_paths(config, root=root, create=True).output_dir / "group"
    recognition = output_dir / "recognition"
    cuteness = output_dir / "cuteness"
    recognition.mkdir(parents=True, exist_ok=True)
    cuteness.mkdir(parents=True, exist_ok=True)
    return {
        "recognition_dir": recognition,
        "cuteness_dir": cuteness,
        "recognition_measurements_all": recognition / "recognition_measurements_all.csv",
        "recognition_measurements_pooled": recognition / "recognition_measurements_pooled.csv",
        "recognition_measurements_all_wide": recognition / "recognition_measurements_all_wide.csv",
        "recognition_wide_schema_qc": recognition / "wide_schema_qc.csv",
        "recognition_participant_summary": recognition / "participant_summary.csv",
        "cuteness_measurements_all": cuteness / "cuteness_measurements_all.csv",
        "cuteness_measurements_pooled": cuteness / "cuteness_measurements_pooled.csv",
        "cuteness_measurements_all_wide": cuteness / "cuteness_measurements_all_wide.csv",
        "cuteness_wide_schema_qc": cuteness / "wide_schema_qc.csv",
        "cuteness_participant_summary": cuteness / "participant_summary.csv",
        "cuteness_trial_level_all": cuteness / "cuteness_trial_level_all.csv",
    }


def run_gate5a_group_aggregation(
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
    publication_context_reference: str | None = None,
    publication_scientific_fingerprint: str | None = None,
    immutable_upstream_root: str | Path | None = None,
) -> Gate5AResult:
    """Aggregate Gate 4 participant-level ERP outputs for later group analyses."""

    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    subjects = resolve_gate5a_subjects(config)
    summary_subjects = resolve_gate5a_summary_subjects(config)
    excluded_subjects = set(parse_subject_args(dict(config.get("analysis", {}) or {}).get("exclude_subjects", []) or [], config=config))
    exclusion_reasons = gate5a_config_exclusion_reasons(config)
    paths = gate5a_group_output_paths(config, root=root)
    warnings: list[str] = []
    publication_dependencies: dict[str, str] = {}
    publication_inputs: dict[str, list[Path]] = {"recognition": [], "cuteness": []}

    if publication_context_reference is not None and publication_scientific_fingerprint is None:
        raise PublicationValidationError(
            "Gate 5A publication consumption requires the current scientific fingerprint."
        )
    recognition_long, recognition_wide, recognition_wide_qc, recognition_rows = _collect_branch(
        config,
        subjects,
        "recognition",
        root=root,
        warnings=warnings,
        publication_context_reference=publication_context_reference,
        publication_scientific_fingerprint=publication_scientific_fingerprint,
        publication_dependencies=publication_dependencies,
        publication_input_paths=publication_inputs["recognition"],
        immutable_upstream_root=immutable_upstream_root,
    )
    cuteness_long, cuteness_wide, cuteness_wide_qc, cuteness_rows = _collect_branch(
        config,
        subjects,
        "cuteness",
        root=root,
        warnings=warnings,
        publication_context_reference=publication_context_reference,
        publication_scientific_fingerprint=publication_scientific_fingerprint,
        publication_dependencies=publication_dependencies,
        publication_input_paths=publication_inputs["cuteness"],
        immutable_upstream_root=immutable_upstream_root,
    )
    cuteness_trials = _collect_cuteness_trials(
        config,
        subjects,
        root=root,
        warnings=warnings,
        publication_context_reference=publication_context_reference,
        publication_scientific_fingerprint=publication_scientific_fingerprint,
        publication_dependencies=publication_dependencies,
        publication_input_paths=publication_inputs["cuteness"],
        immutable_upstream_root=immutable_upstream_root,
    )
    recognition_pooled = pool_participant_measurements(recognition_long)
    cuteness_pooled = pool_participant_measurements(cuteness_long)
    _apply_config_exclusions(config, excluded_subjects, exclusion_reasons, recognition_rows, cuteness_rows, root=root)

    recognition_summary = _participant_summary(summary_subjects, recognition_rows, cuteness_rows, branch="recognition")
    cuteness_summary = _participant_summary(summary_subjects, recognition_rows, cuteness_rows, branch="cuteness")

    output_files = {
        "recognition_measurements_all": _write_table(recognition_long, paths["recognition_measurements_all"]),
        "recognition_measurements_pooled": _write_table(recognition_pooled, paths["recognition_measurements_pooled"]),
        "recognition_measurements_all_wide": _write_table(recognition_wide, paths["recognition_measurements_all_wide"]),
        "recognition_wide_schema_qc": _write_table(recognition_wide_qc, paths["recognition_wide_schema_qc"]),
        "recognition_participant_summary": _write_table(recognition_summary, paths["recognition_participant_summary"]),
        "cuteness_measurements_all": _write_table(cuteness_long, paths["cuteness_measurements_all"]),
        "cuteness_measurements_pooled": _write_table(cuteness_pooled, paths["cuteness_measurements_pooled"]),
        "cuteness_measurements_all_wide": _write_table(cuteness_wide, paths["cuteness_measurements_all_wide"]),
        "cuteness_wide_schema_qc": _write_table(cuteness_wide_qc, paths["cuteness_wide_schema_qc"]),
        "cuteness_participant_summary": _write_table(cuteness_summary, paths["cuteness_participant_summary"]),
        "cuteness_trial_level_all": _write_table(cuteness_trials, paths["cuteness_trial_level_all"]),
    }
    _write_gate5a_provenance(config, "recognition", output_files, recognition_summary, subjects, warnings, root=root, config_path=config_path, published_input_files=publication_inputs["recognition"] if publication_context_reference is not None else None)
    _write_gate5a_provenance(config, "cuteness", output_files, cuteness_summary, subjects, warnings, root=root, config_path=config_path, published_input_files=publication_inputs["cuteness"] if publication_context_reference is not None else None)
    outcomes = tuple(
        Gate5AParticipantOutcome(
            subject=subject,
            branch=branch,
            status=str(rows.get(subject, {}).get("application_status", "missing_required_input")),
            reason=str(rows.get(subject, {}).get("reason", "not processed")),
        )
        for branch, rows in (("recognition", recognition_rows), ("cuteness", cuteness_rows))
        for subject in summary_subjects
    )
    return Gate5AResult(
        output_files=output_files,
        warnings=tuple(warnings),
        participant_summary=recognition_summary,
        participant_outcomes=outcomes,
        publication_identity_references=publication_dependencies,
    )


def pool_participant_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    """Pool part-level ERP means using retained-trial counts, preserving part provenance."""

    if measurements.empty:
        return measurements.copy()
    validate_part_level_measurements(measurements)
    keys = ["subject", "branch", "condition", "roi", "time_window"]
    rows: list[dict[str, Any]] = []
    for _, group in measurements.groupby(keys, dropna=False, sort=True):
        row = group.iloc[0].to_dict()
        amplitudes = pd.to_numeric(group.get("mean_amplitude_uv"), errors="coerce")
        counts = pd.to_numeric(group["n_epochs"], errors="coerce") if "n_epochs" in group.columns else pd.Series(float("nan"), index=group.index)
        usable = amplitudes.notna() & counts.notna() & counts.gt(0)
        if bool(usable.all()):
            total = float(counts[usable].sum())
            row["mean_amplitude_uv"] = float((amplitudes[usable] * counts[usable]).sum() / total)
            row["n_epochs"] = int(total)
            row["aggregation_status"] = "POOLED_BY_RETAINED_TRIAL_COUNT"
        elif len(group) == 1:
            row["aggregation_status"] = "SINGLE_PART_NO_POOLING_REQUIRED"
        else:
            row["mean_amplitude_uv"] = pd.NA
            row["n_epochs"] = 0
            row["aggregation_status"] = "NOT_POOLED_MISSING_RETAINED_COUNTS"
        parts = sorted(group.get("recording_part", pd.Series(dtype=str)).dropna().astype(str).unique())
        row["recording_part"] = ";".join(parts)
        row["recording_parts"] = ";".join(parts)
        row["n_recording_parts"] = len(parts)
        row["aggregation_method"] = "retained_trial_count_weighted_equivalent_to_trial_pooling"
        rows.append(row)
    pooled = pd.DataFrame(rows)
    if "condition" in pooled.columns and "hit_minus_correct_rejection" in set(pooled["condition"].astype(str)):
        axes = [column for column in ("subject", "branch", "roi", "time_window") if column in pooled.columns]
        for _, group in pooled.groupby(axes, dropna=False):
            conditions = group.set_index("condition")
            if {"hit", "correct_rejection", "hit_minus_correct_rejection"}.issubset(conditions.index):
                target = group.index[group["condition"].astype(str).eq("hit_minus_correct_rejection")][0]
                hit = pd.to_numeric(pd.Series([conditions.loc["hit", "mean_amplitude_uv"]]), errors="coerce").iloc[0]
                cr = pd.to_numeric(pd.Series([conditions.loc["correct_rejection", "mean_amplitude_uv"]]), errors="coerce").iloc[0]
                if pd.notna(hit) and pd.notna(cr):
                    pooled.loc[target, "mean_amplitude_uv"] = float(hit - cr)
                    pooled.loc[target, "aggregation_status"] = "DESCRIPTIVE_DIFFERENCE_FROM_POOLED_CONDITIONS"
    validate_pooled_measurements(pooled)
    return pooled


PART_LEVEL_MEASUREMENT_KEY = (
    "subject",
    "recording_part",
    "branch",
    "condition",
    "roi",
    "time_window",
)
POOLED_MEASUREMENT_KEY = (
    "subject",
    "branch",
    "condition",
    "roi",
    "time_window",
)


def validate_part_level_measurements(measurements: pd.DataFrame) -> None:
    """Reject ambiguous or non-finite Gate 4 measurement units before pooling."""

    _validate_measurement_table(
        measurements,
        key=PART_LEVEL_MEASUREMENT_KEY,
        level="part-level",
    )


def validate_pooled_measurements(measurements: pd.DataFrame) -> None:
    """Reject ambiguous or non-finite Gate 5 participant analysis cells."""

    _validate_measurement_table(
        measurements,
        key=POOLED_MEASUREMENT_KEY,
        level="pooled participant-level",
    )


def _validate_measurement_table(
    measurements: pd.DataFrame,
    *,
    key: Sequence[str],
    level: str,
) -> None:
    if measurements.empty:
        return
    required = [*key, "mean_amplitude_uv", "n_epochs"]
    missing = [column for column in required if column not in measurements.columns]
    if missing:
        raise ValueError(f"{level} measurements lack required column(s): {', '.join(missing)}")
    if measurements[list(key)].isna().any(axis=None):
        bad = measurements.loc[measurements[list(key)].isna().any(axis=1), list(key)]
        raise ValueError(
            f"{level} measurement key contains missing values for key {tuple(key)!r}: "
            f"{bad.head(5).to_dict(orient='records')!r}"
        )
    duplicate = measurements.duplicated(list(key), keep=False)
    if bool(duplicate.any()):
        bad = measurements.loc[duplicate, list(key)].drop_duplicates()
        raise ValueError(
            f"Duplicate {level} measurement key {tuple(key)!r}: "
            f"{bad.head(5).to_dict(orient='records')!r}"
        )
    amplitudes = pd.to_numeric(measurements["mean_amplitude_uv"], errors="coerce")
    nonfinite = ~np.isfinite(amplitudes.to_numpy(dtype=float, na_value=np.nan))
    if bool(nonfinite.any()):
        bad = measurements.loc[nonfinite, [*key, "mean_amplitude_uv"]]
        raise ValueError(
            f"Non-finite {level} mean_amplitude_uv at key {tuple(key)!r}: "
            f"{bad.head(5).to_dict(orient='records')!r}"
        )
    counts = pd.to_numeric(measurements["n_epochs"], errors="coerce")
    count_values = counts.to_numpy(dtype=float, na_value=np.nan)
    invalid_counts = ~np.isfinite(count_values) | (count_values <= 0)
    if bool(invalid_counts.any()):
        bad = measurements.loc[invalid_counts, [*key, "n_epochs"]]
        raise ValueError(
            f"Invalid {level} retained-trial count at key {tuple(key)!r}: "
            f"{bad.head(5).to_dict(orient='records')!r}"
        )


def _collect_cuteness_trials(
    config: Mapping[str, Any], subjects: Sequence[str], *, root: Path | None, warnings: list[str],
    publication_context_reference: str | None = None,
    publication_scientific_fingerprint: str | None = None,
    publication_dependencies: dict[str, str] | None = None,
    publication_input_paths: list[Path] | None = None,
    immutable_upstream_root: str | Path | None = None,
) -> pd.DataFrame:
    """Concatenate trial rows without averaging and recenter across all parts per participant."""

    tables: list[pd.DataFrame] = []
    for subject in subjects:
        for part_dir in _subject_part_dirs(config, subject, root=root):
            path = part_dir / "erp" / "cuteness" / "cuteness_trial_level.csv"
            if publication_context_reference is not None:
                try:
                    publication, _ = _validated_gate4_publication(
                        part_dir,
                        subject,
                        part_dir.name,
                        "cuteness",
                        str(publication_scientific_fingerprint),
                        immutable_upstream_canonical_directory=(
                            None
                            if immutable_upstream_root is None
                            else Path(immutable_upstream_root) / "subjects" / subject / part_dir.name
                        ),
                    )
                    published_path = publication.artifact_path("cuteness_trial_level")
                    if published_path is None:
                        raise PublicationValidationError(
                            "Current Gate 4B publication lacks cuteness_trial_level."
                        )
                    path = published_path
                    if publication_input_paths is not None:
                        publication_input_paths.append(Path(published_path))
                    if publication_dependencies is not None:
                        publication_dependencies[_publication_dependency_key(publication.scope)] = publication.generation_id
                except PublicationValidationError as error:
                    warnings.append(f"cuteness {subject} {part_dir.name}: missing trial-level model input: current publication invalid: {error}")
                    continue
            if not path.exists():
                warnings.append(f"cuteness {subject} {part_dir.name}: missing trial-level model input: cuteness_trial_level")
                continue
            table = pd.read_csv(path)
            if "participant" not in table.columns or "part_or_run" not in table.columns:
                raise StageIdentityError(
                    f"cuteness {subject} {part_dir.name}: trial-level model input lacks participant/part identifiers"
                )
            _require_exact_scope(
                table,
                participant_column="participant",
                recording_part_column="part_or_run",
                participant=subject,
                recording_part=part_dir.name,
                label="cuteness trial-level model input",
            )
            tables.append(table)
    combined = _concat_or_empty(tables)
    return center_cuteness_trials(combined, scope="participant_all_parts") if not combined.empty else combined


def _collect_branch(
    config: Mapping[str, Any],
    subjects: Sequence[str],
    branch: str,
    *,
    root: Path | None,
    warnings: list[str],
    publication_context_reference: str | None = None,
    publication_scientific_fingerprint: str | None = None,
    publication_dependencies: dict[str, str] | None = None,
    publication_input_paths: list[Path] | None = None,
    immutable_upstream_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    long_tables: list[pd.DataFrame] = []
    wide_tables: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, Any]] = {}
    long_columns: list[str] | None = None

    for subject in subjects:
        subject_rows = []
        part_dirs = _subject_part_dirs(config, subject, root=root)
        if not part_dirs:
            warnings.append(f"{branch} {subject}: no recording parts with Gate 4 outputs.")
        for part_dir in part_dirs:
            paths = _branch_input_paths(part_dir, branch)
            publication = None
            gate3c_publication = None
            if publication_context_reference is not None:
                try:
                    publication, gate3c_publication = _validated_gate4_publication(
                        part_dir,
                        subject,
                        part_dir.name,
                        branch,
                        str(publication_scientific_fingerprint),
                        immutable_upstream_canonical_directory=(
                            None
                            if immutable_upstream_root is None
                            else Path(immutable_upstream_root) / "subjects" / subject / part_dir.name
                        ),
                    )
                    published_paths = {
                        "long": publication.artifact_path(f"{branch}_measurements"),
                        "wide": publication.artifact_path(f"{branch}_measurements_wide"),
                        "qc": publication.artifact_path(f"{branch}_qc_summary"),
                        "condition_counts": publication.artifact_path("condition_counts"),
                    }
                    if any(path is None for path in published_paths.values()):
                        raise PublicationValidationError(
                            f"Current Gate 4 {branch} publication lacks required aggregation artifacts."
                        )
                    paths = {label: Path(path) for label, path in published_paths.items()}
                    if publication_input_paths is not None:
                        publication_input_paths.extend(paths.values())
                        identity_path = publication.artifact_path("participant_erp_identity")
                        if identity_path is not None:
                            publication_input_paths.append(Path(identity_path))
                    if publication_dependencies is not None:
                        publication_dependencies[_publication_dependency_key(publication.scope)] = publication.generation_id
                except PublicationValidationError as error:
                    summary = summaries.setdefault(subject, _empty_subject_branch_summary(subject, branch))
                    warning = f"{branch} {subject} {part_dir.name}: missing Gate 4 output(s): current publication invalid: {error}"
                    warnings.append(warning)
                    summary["reason"] = warning
                    continue
            summary = summaries.setdefault(subject, _empty_subject_branch_summary(subject, branch))
            summary["preprocessing_complete"] = summary["preprocessing_complete"] or publication is not None or (part_dir / "epochs" / "all_image_epochs-epo.fif").exists()
            part_epoch_count = (
                _published_epoch_count(gate3c_publication)
                if publication is not None and gate3c_publication is not None
                else _read_epoch_count(part_dir)
            )

            present = [label for label, path in paths.items() if path.exists()]
            missing = [label for label, path in paths.items() if not path.exists()]
            if missing:
                if present:
                    raise StageIdentityError(
                        f"{branch} {subject} {part_dir.name}: partial Gate 4 input set; "
                        f"present={present!r}, missing={missing!r}."
                    )
                warnings.append(f"{branch} {subject} {part_dir.name}: missing Gate 4 output(s): {', '.join(missing)}")
                summary["reason"] = f"missing Gate 4 output(s): {', '.join(missing)}"
                summary["number_of_epochs"] += part_epoch_count
                continue

            read_validated_participant_erp_artifact(
                publication.artifact_path("participant_erp_identity") if publication is not None else participant_erp_identity_path(part_dir / "erp" / branch),
                expected_participant=subject,
                expected_recording_part=part_dir.name,
                expected_branch=branch,
                expected_configuration_fingerprint=configuration_mapping_fingerprint(config),
                expected_branch_settings=erp_branch_settings(config, branch),
                current_output_paths={
                    "condition_counts": paths["condition_counts"],
                    f"{branch}_measurements": paths["long"],
                    f"{branch}_measurements_wide": paths["wide"],
                    f"{branch}_qc_summary": paths["qc"],
                    **(
                        {"cuteness_trial_level": publication.artifact_path("cuteness_trial_level") if publication is not None else part_dir / "erp" / branch / "cuteness_trial_level.csv"}
                        if branch == "cuteness"
                        else {}
                    ),
                },
            )

            long_df = pd.read_csv(paths["long"])
            wide_df = pd.read_csv(paths["wide"])
            qc_df = pd.read_csv(paths["qc"])
            counts_df = pd.read_csv(paths["condition_counts"])
            for table, label in (
                (long_df, "long measurements"),
                (wide_df, "wide measurements"),
                (qc_df, "QC summary"),
            ):
                _require_exact_scope(
                    table,
                    participant_column="subject",
                    recording_part_column="recording_part",
                    participant=subject,
                    recording_part=part_dir.name,
                    label=f"{branch} {label}",
                    branch=branch,
                )
            _validate_table_columns(long_df, long_columns, branch, "long", subject, warnings)
            _validate_wide_identifier_columns(wide_df, branch, subject, part_dir.name, warnings)
            long_columns = long_columns or list(long_df.columns)
            _validate_required_measurement_axes(long_df, branch, subject, warnings)

            long_tables.append(long_df)
            wide_tables.append(wide_df)
            subject_rows.append((long_df, wide_df, qc_df, counts_df))
            accumulated_epochs = int(summary.get("number_of_epochs", 0) or 0)
            summary.update(_included_subject_branch_summary(subject, branch, long_df, qc_df, counts_df))
            summary["number_of_epochs"] = accumulated_epochs + part_epoch_count

        if not subject_rows and subject not in summaries:
            summaries[subject] = _empty_subject_branch_summary(subject, branch, reason="no recording parts with Gate 4 outputs")

    long_all = _concat_or_empty(long_tables)
    wide_all, wide_qc = _align_wide_tables(wide_tables, branch=branch)
    _warn_duplicate_subjects(wide_all, branch, warnings)
    return long_all, wide_all, wide_qc, summaries


def _apply_config_exclusions(
    config: Mapping[str, Any],
    excluded_subjects: set[str],
    exclusion_reasons: Mapping[str, str],
    recognition_rows: dict[str, dict[str, Any]],
    cuteness_rows: dict[str, dict[str, Any]],
    *,
    root: Path | None,
) -> None:
    for subject in excluded_subjects:
        reason = exclusion_reasons.get(subject, "excluded by analysis.exclude_subjects")
        recognition_rows[subject] = _config_excluded_subject_branch_summary(config, subject, "recognition", reason, root=root)
        cuteness_rows[subject] = _config_excluded_subject_branch_summary(config, subject, "cuteness", reason, root=root)


def _subject_part_dirs(config: Mapping[str, Any], subject: str, *, root: Path | None) -> list[Path]:
    base = subject_dir(config, subject, root=root, create=False)
    if not base.exists():
        return []
    return sorted([path for path in base.glob("part*") if path.is_dir()], key=lambda path: path.name)


def _branch_input_paths(part_dir: Path, branch: str) -> dict[str, Path]:
    base = part_dir / "erp" / branch
    return {
        "long": base / f"{branch}_measurements.csv",
        "wide": base / f"{branch}_measurements_wide.csv",
        "qc": base / f"{branch}_qc_summary.csv",
        "condition_counts": base / "condition_counts.csv",
    }


def _validated_gate4_publication(
    part_dir: Path,
    subject: str,
    recording_part: str,
    branch: str,
    scientific_fingerprint: str,
    *,
    expected_generation_id: str | None = None,
    session: str = "session2",
    immutable_upstream_canonical_directory: str | Path | None = None,
):
    stage = "gate4a" if branch == "recognition" else "gate4b"
    publication = validate_current_publication(
        part_dir,
        PublicationScope(
            stage=stage,
            session=session,
            participant=subject,
            recording_part=recording_part,
            branch=branch,
        ),
        expected_generation_id=expected_generation_id,
    )
    ancestors = validate_participant_publication_chain(
        part_dir,
        publication,
        immutable_upstream_canonical_directory=immutable_upstream_canonical_directory,
    )
    gate3c_publications = [item for item in ancestors if item.scope.stage == "gate3c"]
    if len(gate3c_publications) != 1:
        raise PublicationValidationError(
            "Current Gate 4 publication does not bind exactly one validated Gate 3C generation."
        )
    gate3c = gate3c_publications[0]
    validate_publication_scientific_context(gate3c, scientific_fingerprint)
    return publication, gate3c


def validate_gate4_publication_for_group(
    config: Mapping[str, Any],
    part_dir: Path,
    subject: str,
    recording_part: str,
    branch: str,
    scientific_fingerprint: str,
    *,
    expected_generation_id: str | None = None,
    session: str = "session2",
    immutable_upstream_canonical_directory: str | Path | None = None,
):
    """Validate a Gate 4 dependency by lineage and scientific/config authority.

    A participant-stage request context is intentionally not compared with a
    later group-stage request context.  The immutable generation binding,
    participant publication chain, Gate 3C scientific authority, and exact
    participant ERP configuration/artifact identity are all validated instead.
    """

    publication, gate3c = _validated_gate4_publication(
        part_dir,
        subject,
        recording_part,
        branch,
        scientific_fingerprint,
        expected_generation_id=expected_generation_id,
        session=session,
        immutable_upstream_canonical_directory=immutable_upstream_canonical_directory,
    )
    required_labels = {
        "condition_counts": "condition_counts",
        f"{branch}_measurements": f"{branch}_measurements",
        f"{branch}_measurements_wide": f"{branch}_measurements_wide",
        f"{branch}_qc_summary": f"{branch}_qc_summary",
    }
    if branch == "cuteness":
        required_labels["cuteness_trial_level"] = "cuteness_trial_level"
    output_paths = {
        output_label: publication.artifact_path(publication_label)
        for output_label, publication_label in required_labels.items()
    }
    identity_path = publication.artifact_path("participant_erp_identity")
    missing = [label for label, path in output_paths.items() if path is None]
    if identity_path is None:
        missing.append("participant_erp_identity")
    if missing:
        raise PublicationValidationError(
            "Current Gate 4 publication lacks required group-consumption artifacts: "
            + ", ".join(sorted(missing))
            + "."
        )
    read_validated_participant_erp_artifact(
        identity_path,
        expected_participant=subject,
        expected_recording_part=recording_part,
        expected_branch=branch,
        expected_configuration_fingerprint=configuration_mapping_fingerprint(config),
        expected_branch_settings=erp_branch_settings(config, branch),
        current_output_paths=output_paths,
    )
    return publication, gate3c


def _publication_dependency_key(scope: PublicationScope) -> str:
    return f"publication:{scope.stage}:{scope.branch}:{scope.participant}:{scope.recording_part}"


def _empty_subject_branch_summary(subject: str, branch: str, reason: str = "not processed") -> dict[str, Any]:
    return {
        "subject": subject,
        f"{branch}_included": False,
        "reason": reason,
        "preprocessing_complete": False,
        "erp_complete": False,
        "number_of_epochs": 0,
        "number_of_conditions": 0,
        "processing_version": "",
        "application_status": "missing_required_input",
    }


def _config_excluded_subject_branch_summary(config: Mapping[str, Any], subject: str, branch: str, reason: str, *, root: Path | None) -> dict[str, Any]:
    summary = _empty_subject_branch_summary(subject, branch, reason=reason)
    summary["application_status"] = "configured_exclusion"
    part_dirs = _subject_part_dirs(config, subject, root=root)
    summary["preprocessing_complete"] = any((part_dir / "epochs" / "all_image_epochs-epo.fif").exists() for part_dir in part_dirs)
    summary["number_of_epochs"] = sum(_read_epoch_count(part_dir) for part_dir in part_dirs)
    return summary


def _included_subject_branch_summary(subject: str, branch: str, long_df: pd.DataFrame, qc_df: pd.DataFrame, counts_df: pd.DataFrame) -> dict[str, Any]:
    reason = "included"
    included = True
    if not qc_df.empty:
        status = str(qc_df.get("participant_reporting_status", pd.Series([""])).iloc[0])
        qc_reason = str(qc_df.get("participant_reporting_reason", pd.Series([""])).iloc[0])
        if status.lower() == "qc-only":
            included = False
            reason = qc_reason or "QC-only participant"
    return {
        "subject": subject,
        f"{branch}_included": included,
        "reason": reason,
        "preprocessing_complete": True,
        "erp_complete": included and not long_df.empty,
        "number_of_epochs": int(pd.to_numeric(counts_df.get("retained_epochs", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if not counts_df.empty else 0,
        "number_of_conditions": int(long_df["condition"].nunique()) if "condition" in long_df.columns else 0,
        "processing_version": ";".join(sorted(set(long_df.get("processing_version", pd.Series(dtype=str)).dropna().astype(str)))) if not long_df.empty else "",
        "application_status": "included" if included else "scientific_exclusion",
    }


def _participant_summary(subjects: Sequence[str], recognition_rows: Mapping[str, Mapping[str, Any]], cuteness_rows: Mapping[str, Mapping[str, Any]], *, branch: str) -> pd.DataFrame:
    rows = []
    for subject in sorted(subjects, key=subject_sort_key):
        recog = dict(recognition_rows.get(subject, _empty_subject_branch_summary(subject, "recognition")))
        cute = dict(cuteness_rows.get(subject, _empty_subject_branch_summary(subject, "cuteness")))
        active = recog if branch == "recognition" else cute
        rows.append(
            {
                "subject": subject,
                "recognition_included": bool(recog.get("recognition_included", False)),
                "cuteness_included": bool(cute.get("cuteness_included", False)),
                "recognition_reason": recog.get("reason", ""),
                "cuteness_reason": cute.get("reason", ""),
                "preprocessing_complete": bool(recog.get("preprocessing_complete", False) or cute.get("preprocessing_complete", False)),
                "erp_complete": bool(active.get("erp_complete", False)),
                "number_of_epochs": int(active.get("number_of_epochs", 0) or 0),
                "number_of_conditions": int(active.get("number_of_conditions", 0) or 0),
                "processing_version": active.get("processing_version", ""),
            }
        )
    return pd.DataFrame(rows)


def _validate_table_columns(df: pd.DataFrame, expected: list[str] | None, branch: str, table_kind: str, subject: str, warnings: list[str]) -> None:
    if expected is not None and list(df.columns) != expected:
        warnings.append(f"{branch} {subject}: inconsistent {table_kind} measurement columns.")


def _validate_wide_identifier_columns(df: pd.DataFrame, branch: str, subject: str, recording_part: str, warnings: list[str]) -> None:
    missing = [column for column in ("subject", "recording_part") if column not in df.columns]
    if missing:
        warnings.append(f"{branch} {subject} {recording_part}: missing required wide identifier column(s): {', '.join(missing)}.")


def _require_exact_scope(
    table: pd.DataFrame,
    *,
    participant_column: str,
    recording_part_column: str,
    participant: str,
    recording_part: str,
    label: str,
    branch: str | None = None,
) -> None:
    required = [participant_column, recording_part_column]
    if branch is not None:
        required.append("branch")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise StageIdentityError(
            f"{label} lacks required scope column(s): {', '.join(missing)}."
        )
    if table.empty:
        return
    participants = set(table[participant_column].dropna().astype(str))
    parts = set(table[recording_part_column].dropna().astype(str))
    branches = set(table["branch"].dropna().astype(str)) if branch is not None else set()
    if participants != {participant} or parts != {recording_part}:
        raise StageIdentityError(f"{label} contains foreign participant/recording-part scope.")
    if branch is not None and branches != {branch}:
        raise StageIdentityError(f"{label} contains a foreign branch scope.")


def _align_wide_tables(tables: Sequence[pd.DataFrame], *, branch: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align sparse participant-wide measurement rows using the union of columns."""

    if not tables:
        return pd.DataFrame(), pd.DataFrame(columns=["branch", "subject", "recording_part", "missing_columns", "n_missing_columns"])

    identifier_columns = [column for column in ("subject", "recording_part", "branch") if any(column in table.columns for table in tables)]
    all_columns = set().union(*(set(table.columns) for table in tables))
    measurement_columns = sorted(column for column in all_columns if column not in set(identifier_columns))
    ordered_columns = identifier_columns + measurement_columns

    aligned_tables = [table.reindex(columns=ordered_columns) for table in tables]
    qc_rows = []
    for table in tables:
        missing_columns = [column for column in measurement_columns if column not in table.columns]
        subject = _first_value(table, "subject")
        recording_part = _first_value(table, "recording_part")
        qc_rows.append(
            {
                "branch": branch,
                "subject": subject,
                "recording_part": recording_part,
                "missing_columns": ";".join(missing_columns),
                "n_missing_columns": len(missing_columns),
            }
        )
    return pd.concat(aligned_tables, ignore_index=True), pd.DataFrame(qc_rows)


def _first_value(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns or df.empty:
        return ""
    value = df[column].iloc[0]
    return "" if pd.isna(value) else str(value)


def _validate_required_measurement_axes(df: pd.DataFrame, branch: str, subject: str, warnings: list[str]) -> None:
    for column in ("roi", "time_window"):
        if column not in df.columns:
            warnings.append(f"{branch} {subject}: missing measurement axis column {column}.")


def _warn_duplicate_subjects(wide_df: pd.DataFrame, branch: str, warnings: list[str]) -> None:
    if wide_df.empty or "subject" not in wide_df.columns:
        return
    duplicates = wide_df["subject"][wide_df["subject"].duplicated()].dropna().astype(str).unique().tolist()
    if duplicates:
        warnings.append(f"{branch}: duplicate participant rows in wide aggregation: {', '.join(sorted(duplicates, key=subject_sort_key))}.")


def _read_epoch_count(part_dir: Path) -> int:
    return _read_epoch_count_path(part_dir / "epochs" / "epoch_metadata.csv")


def _published_epoch_count(gate3c) -> int:
    metadata = gate3c.artifact_path("epoch_metadata_csv")
    if metadata is None:
        raise PublicationValidationError(
            "Current Gate 3C publication lacks required epoch_metadata_csv."
        )
    return _read_epoch_count_path(Path(metadata))


def _read_epoch_count_path(metadata: Path) -> int:
    if not metadata.exists():
        return 0
    try:
        return max(0, len(pd.read_csv(metadata)))
    except Exception:
        return 0


def _concat_or_empty(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _write_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _write_gate5a_provenance(
    config: Mapping[str, Any],
    branch: str,
    output_files: Mapping[str, Path],
    participant_summary: pd.DataFrame,
    subjects: Sequence[str],
    warnings: Sequence[str],
    *,
    root: Path | None,
    config_path: str | Path | None = None,
    published_input_files: Sequence[Path] | None = None,
) -> None:
    included, excluded, reasons = _summary_subject_sets(participant_summary, branch)
    inputs = (
        list(dict.fromkeys(published_input_files))
        if published_input_files is not None
        else _gate5a_existing_inputs(config, subjects, branch, root=root)
    )
    for label, output in output_files.items():
        if not label.startswith(branch):
            continue
        write_group_provenance_sidecar(
            output,
            gate="gate5a",
            branch=branch,
            root=root,
            input_files=inputs,
            included_subjects=included,
            excluded_subjects=excluded,
            exclusion_reasons=reasons,
            parameters_summary={"analysis_selection": dict(config.get("analysis", {}) or {})},
            warnings=warnings,
            **({"config_path": config_path} if config_path is not None else {}),
        )


def _gate5a_existing_inputs(config: Mapping[str, Any], subjects: Sequence[str], branch: str, *, root: Path | None) -> list[Path]:
    paths: list[Path] = []
    for subject in subjects:
        for part_dir in _subject_part_dirs(config, subject, root=root):
            paths.extend(path for path in _branch_input_paths(part_dir, branch).values() if path.exists())
            trial_path = part_dir / "erp" / "cuteness" / "cuteness_trial_level.csv"
            if branch == "cuteness" and trial_path.exists():
                paths.append(trial_path)
    return paths


def _summary_subject_sets(participant_summary: pd.DataFrame, branch: str) -> tuple[list[str], list[str], dict[str, str]]:
    include_col = f"{branch}_included"
    reason_col = f"{branch}_reason"
    if participant_summary.empty or "subject" not in participant_summary.columns or include_col not in participant_summary.columns:
        return [], [], {}
    included_mask = _as_bool_series(participant_summary[include_col])
    included = sorted(participant_summary.loc[included_mask, "subject"].dropna().astype(str).unique().tolist(), key=subject_sort_key)
    excluded = sorted(participant_summary.loc[~included_mask, "subject"].dropna().astype(str).unique().tolist(), key=subject_sort_key)
    reasons = {}
    if reason_col in participant_summary.columns:
        for _, row in participant_summary.loc[~included_mask].iterrows():
            subject = str(row.get("subject", ""))
            if subject:
                reasons[subject] = str(row.get(reason_col, ""))
    return included, excluded, reasons


def _as_bool_series(series: pd.Series) -> pd.Series:
    return series.map(_as_bool).astype(bool)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "included"}
