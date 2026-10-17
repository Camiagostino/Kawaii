"""Run-scoped ownership for disposable Session 2 EEG resources.

The configured temporary root can contain historical or foreign files.  This
module never scans or recursively removes that root: an owner removes only the
exact paths registered in its own, identity-bound directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


TEMPORARY_OWNERSHIP_SCHEMA = "kawaii_temporary_resource_ownership_v1"
_OWNERSHIP_FILE = "ownership.json"


class TemporaryResourceOwnershipError(RuntimeError):
    """Raised when temporary-resource ownership cannot be proven."""


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise TemporaryResourceOwnershipError("Temporary-resource identity components must be non-empty.")
    return cleaned[:96]


@dataclass
class TemporaryResourceOwner:
    """Own exact disposable files for one execution attempt and scope."""

    root: Path
    stage: str
    session: str
    participant: str
    recording_part: str
    run_id: str
    attempt_id: str = field(default_factory=lambda: uuid4().hex)
    _owned: set[Path] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        stage: str,
        session: str = "session2",
        participant: str,
        recording_part: str,
        run_id: str | None = None,
    ) -> "TemporaryResourceOwner":
        owner = cls(
            Path(root),
            _safe_component(stage),
            _safe_component(session),
            _safe_component(participant),
            _safe_component(recording_part),
            _safe_component(run_id or uuid4().hex),
        )
        owner.directory.mkdir(parents=True, exist_ok=False)
        owner._write_ownership()
        return owner

    @property
    def directory(self) -> Path:
        scope = f"{self.session}--{self.participant}--{self.recording_part}"
        return self.root / ".kawaii-run-owned" / self.stage / scope / f"{self.run_id}--{self.attempt_id}"

    @property
    def ownership_path(self) -> Path:
        return self.directory / _OWNERSHIP_FILE

    def allocate(self, name: str) -> Path:
        candidate_name = Path(name)
        if candidate_name.name != name or name in {"", ".", ".."}:
            raise TemporaryResourceOwnershipError(f"Temporary resource name must be one file name: {name!r}")
        path = self.directory / name
        self._assert_contained(path)
        self._owned.add(path)
        self._write_ownership()
        return path

    def cleanup(self, warnings: list[str] | None = None) -> None:
        """Remove registered resources only when the persisted owner matches."""

        warnings = warnings if warnings is not None else []
        try:
            persisted = self._read_verified_ownership()
        except TemporaryResourceOwnershipError as error:
            warnings.append(str(error))
            return
        persisted_names = persisted.get("owned_files", [])
        if any(Path(str(name)).name != str(name) or str(name) in {"", ".", ".."} for name in persisted_names):
            warnings.append(
                f"Temporary ownership registry contains an unsafe resource name; nothing was removed: {self.directory}"
            )
            return
        persisted_files = {self.directory / str(name) for name in persisted_names}
        for path in sorted(self._owned | persisted_files, key=lambda item: str(item)):
            try:
                self._assert_contained(path)
                if path.is_symlink():
                    path.unlink()
                elif path.exists():
                    if not path.is_file():
                        warnings.append(f"Owned temporary path is not a regular file and was retained: {path}")
                        continue
                    path.unlink()
            except (OSError, TemporaryResourceOwnershipError) as error:
                warnings.append(f"Could not remove owned temporary resource {path}: {error}")
        try:
            if self.ownership_path.exists():
                self.ownership_path.unlink()
            self.directory.rmdir()  # fails closed when foreign/unregistered content exists
        except OSError as error:
            if self.directory.exists():
                warnings.append(f"Owned temporary directory retained because it is not empty/removable: {self.directory}: {error}")

    def _metadata(self) -> dict[str, Any]:
        return {
            "schema": TEMPORARY_OWNERSHIP_SCHEMA,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "session": self.session,
            "participant": self.participant,
            "recording_part": self.recording_part,
            "directory": str(self.directory.resolve()),
            "owned_files": sorted(path.name for path in self._owned),
        }

    def _write_ownership(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / f".{_OWNERSHIP_FILE}.{uuid4().hex}.tmp"
        encoded = (json.dumps(self._metadata(), indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, self.ownership_path)
        except Exception:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

    def _read_verified_ownership(self) -> dict[str, Any]:
        try:
            value = json.loads(self.ownership_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TemporaryResourceOwnershipError(
                f"Temporary ownership could not be verified; nothing was removed: {self.directory}: {error}"
            ) from error
        expected = self._metadata()
        for key in ("schema", "run_id", "attempt_id", "stage", "session", "participant", "recording_part", "directory"):
            if value.get(key) != expected[key]:
                raise TemporaryResourceOwnershipError(
                    f"Temporary ownership mismatch for {self.directory}; nothing was removed ({key})."
                )
        if not isinstance(value.get("owned_files"), list):
            raise TemporaryResourceOwnershipError(
                f"Temporary ownership registry is invalid; nothing was removed: {self.directory}"
            )
        return value

    def _assert_contained(self, path: Path) -> None:
        directory = self.directory.resolve()
        candidate = path.resolve(strict=False)
        if candidate.parent != directory:
            raise TemporaryResourceOwnershipError(
                f"Temporary resource escapes its owned directory: {path}"
            )
