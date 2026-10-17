"""Read-only metadata preflight for an isolated R11 Gate 3C replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from eeg_software.project import read_yaml_config

from .publication import PUBLICATION_MANIFEST_SCHEMA, PublicationScope, publication_scope_root
from .manual_review import (
    _bad_channel_decision_identity,
    _bad_channel_semantic_decision_identity,
    _ica_semantic_decision_identity,
)


APPROVED_R11_CORRECTION_REVISION = "5843702e86fad4e223a7bf49d17fe551c74cb199"
EXECUTION_NOT_AUTHORISED = "Participant execution is not authorised by this validator."


class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"
    MISSING = "MISSING"
    INCONSISTENT = "INCONSISTENT"
    NOT_CHECKED = "NOT_CHECKED"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: EvidenceStatus
    detail: str
    participant: str | None = None
    recording_part: str | None = None

    def to_primitive(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "participant": self.participant,
            "recording_part": self.recording_part,
        }


@dataclass(frozen=True)
class R11ReplayPreflightResult:
    upstream_root: Path
    corrected_output_root: Path
    participant_parts: tuple[tuple[str, str], ...]
    checks: tuple[PreflightCheck, ...]

    @property
    def go(self) -> bool:
        return not any(
            check.status in {EvidenceStatus.MISSING, EvidenceStatus.INCONSISTENT}
            for check in self.checks
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "technical_replay_readiness": "GO" if self.go else "NO-GO",
            "upstream_root": self.upstream_root.as_posix(),
            "corrected_output_root": self.corrected_output_root.as_posix(),
            "participant_recording_parts": [
                {"participant": participant, "recording_part": recording_part}
                for participant, recording_part in self.participant_parts
            ],
            "checks": [check.to_primitive() for check in self.checks],
            "disclaimer": EXECUTION_NOT_AUTHORISED,
        }

    def render_text(self) -> str:
        lines = [f"TECHNICAL REPLAY READINESS: {'GO' if self.go else 'NO-GO'}"]
        lines.extend(
            f"{check.status.value} {check.name}"
            + (
                f" [{check.participant}/{check.recording_part}]"
                if check.participant and check.recording_part
                else (f" [{check.participant}]" if check.participant else "")
            )
            + f": {check.detail}"
            for check in self.checks
        )
        lines.append(EXECUTION_NOT_AUTHORISED)
        return "\n".join(lines)


def validate_r11_replay_preflight(
    *,
    upstream_root: str | Path,
    corrected_output_root: str | Path,
    participant_parts: Sequence[tuple[str, str]],
    manual_review_root: str | Path,
    immutable_source_roots: Mapping[str, str | Path],
    correction_revision: str = APPROVED_R11_CORRECTION_REVISION,
) -> R11ReplayPreflightResult:
    """Validate replay metadata without importing MNE or reading FIF samples.

    Large artifacts are checked against their existing manifest sizes and
    hashes are reported as recorded evidence; they are intentionally not
    rehashed here.
    """

    upstream_supplied = Path(upstream_root)
    corrected_supplied = Path(corrected_output_root)
    upstream = upstream_supplied.resolve(strict=False)
    corrected = corrected_supplied.resolve(strict=False)
    manual_supplied = Path(manual_review_root)
    manual = manual_supplied.resolve(strict=False)
    selected = tuple((str(participant), str(recording_part)) for participant, recording_part in participant_parts)
    checks: list[PreflightCheck] = []

    if not selected or len(set(selected)) != len(selected) or any(not participant or not recording_part for participant, recording_part in selected):
        checks.append(PreflightCheck("exact_recording_part_scope", EvidenceStatus.INCONSISTENT, "Selection must contain unique, non-empty participant/recording-part pairs."))
    elif any(recording_part != "part1" for _, recording_part in selected):
        checks.append(PreflightCheck("exact_recording_part_scope", EvidenceStatus.INCONSISTENT, "This bounded R11 replay is authorised only for explicitly selected part1 scopes."))
    else:
        checks.append(PreflightCheck("exact_recording_part_scope", EvidenceStatus.VERIFIED, f"Explicit selection contains {len(selected)} participant/part1 scope(s)."))

    if upstream_supplied.is_symlink() or corrected_supplied.is_symlink():
        checks.append(PreflightCheck("root_alias_isolation", EvidenceStatus.INCONSISTENT, "A supplied root is a symbolic link."))
    elif upstream == corrected or upstream in corrected.parents or corrected in upstream.parents:
        checks.append(PreflightCheck("root_alias_isolation", EvidenceStatus.INCONSISTENT, "Roots are identical or nested after canonical resolution."))
    else:
        checks.append(PreflightCheck("root_alias_isolation", EvidenceStatus.VERIFIED, "Canonical roots are distinct and non-nested."))

    if _paths_overlap(corrected, manual):
        checks.append(PreflightCheck("manual_review_root_isolation", EvidenceStatus.INCONSISTENT, "Corrected output and manual-review authority roots are identical or nested after canonical resolution."))
    else:
        checks.append(PreflightCheck("manual_review_root_isolation", EvidenceStatus.VERIFIED, "Corrected output is canonically isolated from the manual-review authority root."))

    source_failures: list[str] = []
    for label, supplied in sorted(immutable_source_roots.items()):
        supplied_path = Path(supplied)
        source = supplied_path.resolve(strict=False)
        if not source.is_dir():
            source_failures.append(f"missing {label}: {source}")
        elif _paths_overlap(corrected, source):
            source_failures.append(f"unsafe overlap {label}: {source}")
    checks.append(
        PreflightCheck(
            "immutable_source_root_isolation",
            EvidenceStatus.INCONSISTENT if any(item.startswith("unsafe overlap") for item in source_failures) else (EvidenceStatus.MISSING if source_failures else EvidenceStatus.VERIFIED),
            "; ".join(source_failures) if source_failures else "Corrected output is canonically isolated from all configured immutable raw/log/schedule source-authority roots.",
        )
    )

    if upstream.is_dir():
        checks.append(PreflightCheck("upstream_root", EvidenceStatus.VERIFIED, "Immutable upstream root exists and is readable."))
    else:
        checks.append(PreflightCheck("upstream_root", EvidenceStatus.MISSING, f"Missing upstream directory: {upstream}"))

    if corrected.exists() and (not corrected.is_dir() or any(corrected.iterdir())):
        checks.append(PreflightCheck("corrected_namespace", EvidenceStatus.INCONSISTENT, "Corrected output root is not an unused empty directory."))
    else:
        checks.append(PreflightCheck("corrected_namespace", EvidenceStatus.VERIFIED, "Corrected namespace is absent or empty; no directory was created."))

    decisions = _read_manual_decisions(manual, checks)
    historical_revisions: set[str] = set()
    for participant, recording_part in selected:
        _validate_participant_metadata(
            upstream,
            participant,
            recording_part,
            decisions,
            checks,
            historical_revisions,
        )

    if not historical_revisions:
        checks.append(PreflightCheck("revision_separation", EvidenceStatus.MISSING, "No historical producing revision was recoverable from selected provenance."))
    elif correction_revision in historical_revisions:
        checks.append(PreflightCheck("revision_separation", EvidenceStatus.INCONSISTENT, "Correction revision is not distinguishable from all historical producing revisions."))
    else:
        checks.append(PreflightCheck("revision_separation", EvidenceStatus.VERIFIED, f"Correction {correction_revision} differs from historical revision(s): {', '.join(sorted(historical_revisions))}."))

    return R11ReplayPreflightResult(upstream, corrected, selected, tuple(checks))


def _read_manual_decisions(manual: Path, checks: list[PreflightCheck]) -> dict[str, Mapping[str, Any]]:
    paths = {
        "ica": manual / "ica_components.yaml",
        "bad": manual / "bad_channels.yaml",
    }
    if not all(path.is_file() for path in paths.values()):
        checks.append(PreflightCheck("manual_review_authority", EvidenceStatus.MISSING, f"Required manual-review stores are absent under {manual}."))
        return {}
    try:
        value = {key: read_yaml_config(path) for key, path in paths.items()}
    except Exception as error:
        checks.append(PreflightCheck("manual_review_authority", EvidenceStatus.INCONSISTENT, f"Manual-review store cannot be parsed: {error}"))
        return {}
    checks.append(PreflightCheck("manual_review_authority", EvidenceStatus.VERIFIED, f"ICA and bad-channel decision stores resolved from {manual}."))
    return value


def _validate_participant_metadata(
    upstream: Path,
    participant: str,
    recording_part: str,
    decisions: Mapping[str, Mapping[str, Any]],
    checks: list[PreflightCheck],
    historical_revisions: set[str],
) -> None:
    canonical = upstream / "subjects" / participant / recording_part
    ica_decision = _nested(decisions.get("ica", {}), participant, recording_part)
    bad_decision = _nested(decisions.get("bad", {}), participant, recording_part)
    if not _complete_decision(ica_decision) or not _complete_decision(bad_decision):
        checks.append(PreflightCheck("manual_decisions", EvidenceStatus.MISSING, "Complete ICA and bad-channel decisions were not both found.", participant))
        return
    if (
        bad_decision.get("decision_id") != _bad_channel_decision_identity(bad_decision)
        or ica_decision.get("decision_id") != _ica_semantic_decision_identity(ica_decision)
    ):
        checks.append(PreflightCheck("manual_decisions", EvidenceStatus.INCONSISTENT, "A stored manual-decision fingerprint does not match its metadata.", participant))
        return
    if ica_decision.get("bad_channel_decision_id") != _bad_channel_semantic_decision_identity(bad_decision):
        checks.append(PreflightCheck("manual_decisions", EvidenceStatus.INCONSISTENT, "ICA decision is not bound to the current bad-channel decision.", participant))
        return
    checks.append(PreflightCheck("manual_decisions", EvidenceStatus.VERIFIED, "Complete solution-bound ICA and bad-channel decisions agree.", participant))

    gate3b_generation = _generation_from_notes(str(ica_decision.get("notes", "")), "Gate 3B")
    gate3b = _read_generation(canonical, PublicationScope("gate3b", participant=participant, recording_part=recording_part), gate3b_generation, checks, participant)
    if gate3b is None:
        return
    artifacts_b = _validate_manifest_artifacts(
        gate3b,
        {"ica_solution", "ica_fit_identity"},
        checks,
        participant,
        optional={"eog_suggestions"},
    )
    suggestion_path = artifacts_b.get("eog_suggestions")
    suggestion_row = next(
        (
            row
            for row in gate3b.get("artifacts", [])
            if isinstance(row, Mapping) and row.get("label") == "eog_suggestions"
        ),
        None,
    )
    if suggestion_row is None:
        checks.append(PreflightCheck("historical_ica_suggestion_provenance", EvidenceStatus.NOT_CHECKED, "NOT_AVAILABLE: the reviewed Gate 3B publication has no EOG-suggestion artifact.", participant, recording_part))
    elif suggestion_path is None:
        checks.append(PreflightCheck("historical_ica_suggestion_provenance", EvidenceStatus.INCONSISTENT, "NOT_REVALIDATED: the Gate 3B publication binds an EOG-suggestion artifact, but its file or manifest metadata is unavailable or inconsistent.", participant, recording_part))
    else:
        expected_sha256 = str(suggestion_row.get("sha256", ""))
        try:
            observed_sha256 = hashlib.sha256(suggestion_path.read_bytes()).hexdigest()
        except OSError as error:
            checks.append(PreflightCheck("historical_ica_suggestion_provenance", EvidenceStatus.INCONSISTENT, f"NOT_REVALIDATED: could not hash bound EOG-suggestion artifact: {error}", participant, recording_part))
        else:
            if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256 or "") or observed_sha256 != expected_sha256.lower():
                checks.append(PreflightCheck("historical_ica_suggestion_provenance", EvidenceStatus.INCONSISTENT, "NOT_REVALIDATED: EOG-suggestion CSV SHA-256 does not match its immutable publication manifest.", participant, recording_part))
            else:
                checks.append(PreflightCheck("historical_ica_suggestion_provenance", EvidenceStatus.VERIFIED, f"Authoritative historical EOG suggestions resolve from immutable publication artifact {suggestion_path.name}; the small CSV SHA-256 was recomputed and matched.", participant, recording_part))
    identity = _read_small_json(artifacts_b.get("ica_fit_identity"))
    solution_id = _find_string(identity, "solution_fingerprint")
    if solution_id != ica_decision.get("ica_solution_id"):
        checks.append(PreflightCheck("reviewed_ica_identity", EvidenceStatus.INCONSISTENT, "ICA fit identity does not match the reviewed solution ID.", participant))
    else:
        checks.append(PreflightCheck("reviewed_ica_identity", EvidenceStatus.VERIFIED, "Reviewed ICA solution and saved fit identity agree.", participant))

    gate3a_generation = _referenced_generation(gate3b, "gate3a") or _generation_from_notes(str(bad_decision.get("notes", "")), "Gate 3A")
    gate3a = _read_generation(canonical, PublicationScope("gate3a", participant=participant, recording_part=recording_part), gate3a_generation, checks, participant)
    if gate3a is not None:
        artifacts_a = _validate_manifest_artifacts(gate3a, {"filtered_raw_fif", "bad_channel_decisions"}, checks, participant)
        if artifacts_a:
            checks.append(PreflightCheck("reviewed_gate3a_inputs", EvidenceStatus.VERIFIED, "Exact immutable Gate 3A generation and required artifacts are present.", participant))
        for label, path in artifacts_a.items():
            if label.endswith("__provenance") or path.suffix.lower() == ".json":
                historical_revisions.update(_git_revisions(_read_small_json(path)))

    alignment = canonical / "alignment"
    required_alignment = (
        alignment / "trial_alignment.identity.json",
        alignment / "trial_alignment.csv",
        alignment / "task_segments.csv",
        alignment / "image_onset_timing_debug.csv",
    )
    if all(path.is_file() and path.stat().st_size > 0 for path in required_alignment):
        checks.append(PreflightCheck("r11_coordinate_evidence", EvidenceStatus.VERIFIED, "Alignment identity, trial mapping, segment geometry, and marker evidence are present.", participant))
        alignment_identity = _read_small_json(required_alignment[0])
        historical_revisions.update(_git_revisions(alignment_identity))
        required_identity_keys = {
            "brainvision",
            "resolved_config_fingerprint",
            "resolved_snapshot",
            "scientific_fingerprint",
        }
        observed_keys = _all_keys(alignment_identity)
        if required_identity_keys <= observed_keys:
            checks.append(PreflightCheck("source_configuration_provenance", EvidenceStatus.VERIFIED, "Source, resolved snapshot/configuration, and scientific fingerprints are present in alignment identity.", participant))
        else:
            checks.append(PreflightCheck("source_configuration_provenance", EvidenceStatus.MISSING, "Alignment identity lacks: " + ", ".join(sorted(required_identity_keys - observed_keys)), participant))
    else:
        missing = [path.name for path in required_alignment if not path.is_file() or path.stat().st_size <= 0]
        checks.append(PreflightCheck("r11_coordinate_evidence", EvidenceStatus.MISSING, "Missing: " + ", ".join(missing), participant))


def _read_generation(canonical: Path, scope: PublicationScope, generation_id: str | None, checks: list[PreflightCheck], participant: str) -> Mapping[str, Any] | None:
    if not generation_id:
        checks.append(PreflightCheck(f"{scope.stage}_publication", EvidenceStatus.MISSING, "Decision lacks an unambiguous reviewed generation marker.", participant))
        return None
    path = publication_scope_root(canonical, scope) / "generations" / generation_id / "generation.json"
    value = _read_small_json(path)
    if not value:
        checks.append(PreflightCheck(f"{scope.stage}_publication", EvidenceStatus.MISSING, f"Generation manifest is absent: {path}", participant))
        return None
    if value.get("schema") != PUBLICATION_MANIFEST_SCHEMA or value.get("generation_id") != generation_id or value.get("scope") != scope.as_dict():
        checks.append(PreflightCheck(f"{scope.stage}_publication", EvidenceStatus.INCONSISTENT, "Generation manifest identity/scope is inconsistent.", participant))
        return None
    generation_payload = {
        "scope": value.get("scope"),
        "scope_identity": value.get("scope_identity"),
        "context_reference": value.get("context_reference"),
        "identity_references": value.get("identity_references"),
        "artifacts": value.get("artifacts"),
    }
    computed = hashlib.sha256(
        json.dumps(generation_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if computed != generation_id:
        checks.append(PreflightCheck(f"{scope.stage}_publication", EvidenceStatus.INCONSISTENT, "Generation ID does not match the metadata payload.", participant))
        return None
    checks.append(PreflightCheck(f"{scope.stage}_publication", EvidenceStatus.VERIFIED, f"Immutable generation {generation_id} is present.", participant))
    result = dict(value)
    result["_directory"] = str(path.parent)
    return result


def _validate_manifest_artifacts(
    manifest: Mapping[str, Any],
    required: set[str],
    checks: list[PreflightCheck],
    participant: str,
    *,
    optional: set[str] | None = None,
) -> dict[str, Path]:
    directory = Path(str(manifest["_directory"]))
    rows = {str(row.get("label")): row for row in manifest.get("artifacts", []) if isinstance(row, Mapping)}
    paths: dict[str, Path] = {}
    failures: list[str] = []
    for label in required | set(optional or ()) | {name for name in rows if name.endswith("__provenance")}:
        row = rows.get(label)
        if row is None:
            if label in required:
                failures.append(f"missing {label}")
            continue
        path = (directory / str(row.get("relative_path", ""))).resolve(strict=False)
        if path.parent != directory.resolve():
            failures.append(f"path escape {label}")
            continue
        try:
            size = path.stat().st_size
        except OSError:
            failures.append(f"missing file {label}")
            continue
        if size != int(row.get("size_bytes", -1)) or not row.get("sha256"):
            failures.append(f"metadata mismatch {label}")
            continue
        paths[label] = path
    status = EvidenceStatus.INCONSISTENT if failures else EvidenceStatus.INFERRED
    detail = "; ".join(failures) if failures else "Recorded hashes are present and artifact byte sizes agree; large FIF hashes were not recomputed."
    checks.append(PreflightCheck("manifest_artifact_metadata", status, detail, participant))
    return paths


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def _nested(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _complete_decision(value: Mapping[str, Any]) -> bool:
    return value.get("review_status") == "complete" and bool(value.get("decision_id"))


def _generation_from_notes(notes: str, stage: str) -> str | None:
    matches = set(re.findall(rf"Reviewed canonical {re.escape(stage)} generation ([0-9a-f]{{64}})\b", notes))
    return next(iter(matches)) if len(matches) == 1 else None


def _referenced_generation(manifest: Mapping[str, Any], stage: str) -> str | None:
    matches = {
        str(value)
        for key, value in dict(manifest.get("identity_references", {}) or {}).items()
        if f":{stage}:" in str(key)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _read_small_json(path: Path | None) -> Mapping[str, Any]:
    if path is None or not path.is_file() or path.stat().st_size > 10_000_000:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _find_string(value: Any, key: str) -> str | None:
    if isinstance(value, Mapping):
        if isinstance(value.get(key), str):
            return str(value[key])
        for child in value.values():
            result = _find_string(child, key)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_string(child, key)
            if result is not None:
                return result
    return None


def _git_revisions(value: Any) -> set[str]:
    revisions: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in {"git_commit", "git_revision", "repository_commit"} and isinstance(child, str) and re.fullmatch(r"[0-9a-f]{40}", child):
                revisions.add(child)
            revisions.update(_git_revisions(child))
    elif isinstance(value, list):
        for child in value:
            revisions.update(_git_revisions(child))
    return revisions


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        keys.update(str(key) for key in value)
        for child in value.values():
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys
