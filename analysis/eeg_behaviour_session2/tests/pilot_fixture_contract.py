from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


MANIFEST_PATH = Path(__file__).resolve().parent / "fixtures" / "pilot_integration_manifest.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PilotFixtureValidationError(RuntimeError):
    """Raised when the explicitly requested real-pilot corpus is invalid."""


@dataclass(frozen=True)
class VerifiedPilotFixtures:
    root: Path
    corpus_id: str
    files: tuple[Path, ...]


def load_pilot_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise PilotFixtureValidationError(
            f"Unsupported pilot fixture manifest schema in {path}: {manifest.get('schema_version')!r}."
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PilotFixtureValidationError(f"Pilot fixture manifest {path} has no file entries.")
    return manifest


def validate_pilot_fixtures(
    root: Path,
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> VerifiedPilotFixtures:
    """Validate the complete external corpus before any real-pilot test runs."""

    root = root.expanduser().resolve()
    manifest = load_pilot_manifest(manifest_path)
    errors: list[str] = []
    verified: list[Path] = []

    for entry in manifest["files"]:
        relative = Path(str(entry.get("path", "")))
        expected_size = entry.get("size")
        expected_hash = str(entry.get("sha256", "")).lower()
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"invalid manifest path: {relative}")
            continue
        if not isinstance(expected_size, int) or expected_size < 0:
            errors.append(f"invalid size declaration for {relative}: {expected_size!r}")
            continue
        if not SHA256_RE.fullmatch(expected_hash):
            errors.append(f"invalid SHA-256 declaration for {relative}")
            continue

        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"fixture escapes selected root: {relative}")
            continue
        if not candidate.is_file():
            errors.append(f"missing required file: {relative}")
            continue
        actual_size = candidate.stat().st_size
        if actual_size != expected_size:
            errors.append(
                f"size mismatch for {relative}: expected {expected_size}, found {actual_size}"
            )
            continue
        actual_hash = _sha256(candidate)
        if actual_hash != expected_hash:
            errors.append(
                f"SHA-256 mismatch for {relative}: expected {expected_hash}, found {actual_hash}"
            )
            continue
        verified.append(candidate)

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise PilotFixtureValidationError(
            "Pilot integration profile was explicitly requested, but its external "
            f"fixture contract failed for root {root}:\n{details}\n"
            "Supply the exact declared corpus with --pilot-fixture-root or "
            "KAWAII_PILOT_FIXTURE_ROOT; pilot tests were not run."
        )

    return VerifiedPilotFixtures(
        root=root,
        corpus_id=str(manifest.get("corpus_id", "")),
        files=tuple(verified),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
