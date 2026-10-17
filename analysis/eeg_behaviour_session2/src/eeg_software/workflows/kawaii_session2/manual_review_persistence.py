from __future__ import annotations

from contextlib import contextmanager, ExitStack
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, TypeVar
from uuid import uuid4

import yaml


DEFAULT_LOCK_TIMEOUT_SECONDS = 5.0
DEFAULT_LOCK_POLL_SECONDS = 0.05


class ManualReviewPersistenceError(RuntimeError):
    """Base error for durable manual-review persistence."""


class ManualReviewLockTimeoutError(ManualReviewPersistenceError):
    """Raised when a store lock cannot be acquired within the bounded policy."""


class ManualReviewConflictError(ManualReviewPersistenceError):
    """Raised when a scoped compare-and-swap expectation is stale."""


T = TypeVar("T")


def update_yaml_store(
    path: str | Path,
    *,
    mutate: Callable[[dict[str, Any]], T],
    validate_store: Callable[[Mapping[str, Any], Path], None],
    validate_committed: Callable[[Mapping[str, Any], T, Path], None],
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> T:
    """Lock, reread, mutate, atomically replace, and validate one YAML store."""

    target = _validated_target(path)
    with _exclusive_store_lock(target, timeout_s=lock_timeout_s):
        canonical_exists = target.exists()
        current = _read_yaml_mapping(target)
        validate_store(current, target)
        prior = copy.deepcopy(current)
        result = mutate(current)
        validate_store(current, target)
        if canonical_exists and current == prior:
            validate_committed(current, result, target)
            return result
        serialized = yaml.safe_dump(dict(current), sort_keys=False, allow_unicode=True)
        serialized_value = yaml.safe_load(serialized) or {}
        if not isinstance(serialized_value, dict):
            raise ManualReviewPersistenceError(
                f"Serialized manual-review store is not a mapping: {target}"
            )
        validate_store(serialized_value, target)
        validate_committed(serialized_value, result, target)
        _atomic_replace_text(target, serialized)
        committed = _read_yaml_mapping(target)
        validate_store(committed, target)
        validate_committed(committed, result, target)
        return result


def update_text_store(
    path: str | Path,
    *,
    mutate: Callable[[str], str],
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Path:
    """Serialize a read-modify-write text update and publish it atomically."""

    target = _validated_target(path)
    with _exclusive_store_lock(target, timeout_s=lock_timeout_s):
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        intended = mutate(current)
        if not isinstance(intended, str):
            raise ManualReviewPersistenceError(f"Text-store mutation did not return text: {target}")
        if intended == current and target.exists():
            return target
        _atomic_replace_text(target, intended)
        if target.read_text(encoding="utf-8") != intended:
            raise ManualReviewPersistenceError(
                f"Committed manual-review text failed round-trip validation: {target}"
            )
        return target


def write_yaml_document(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Path:
    """Atomically publish one complete non-authoritative YAML document."""

    target = _validated_target(path)
    intended = dict(payload)
    serialized = yaml.safe_dump(intended, sort_keys=False, allow_unicode=True)
    parsed = yaml.safe_load(serialized) or {}
    if parsed != intended:
        raise ManualReviewPersistenceError(
            f"Manual-review YAML could not be serialized without semantic change: {target}"
        )
    with _exclusive_store_lock(target, timeout_s=lock_timeout_s):
        _atomic_replace_text(target, serialized)
        try:
            committed = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ManualReviewPersistenceError(
                f"Could not validate committed manual-review YAML {target}: {error}"
            ) from error
        if committed != intended:
            raise ManualReviewPersistenceError(
                f"Committed manual-review YAML failed round-trip validation: {target}"
            )
        return target


def store_lock_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(f".{target.name}.lock")


@contextmanager
def locked_store_set(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    timeout_s: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Hold several store locks in deterministic order within one total timeout."""

    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s < 0:
        raise ValueError("Manual-review lock timeout must be a non-negative number of seconds.")
    targets = sorted(
        {_validated_target(path) for path in paths},
        key=lambda value: str(value.absolute()).casefold(),
    )
    deadline = time.monotonic() + float(timeout_s)
    with ExitStack() as stack:
        for target in targets:
            remaining = max(0.0, deadline - time.monotonic())
            stack.enter_context(_exclusive_store_lock(target, timeout_s=remaining))
        yield


@contextmanager
def _exclusive_store_lock(path: Path, *, timeout_s: float) -> Iterator[None]:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s < 0:
        raise ValueError("Manual-review lock timeout must be a non-negative number of seconds.")
    lock_path = store_lock_path(path)
    token = uuid4().hex
    metadata = {
        "schema": "kawaii_manual_review_lock_v1",
        "token": token,
        "target": path.name,
        "host": socket.gethostname(),
        "process_id": os.getpid(),
        "thread_id": threading.get_ident(),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
    deadline = time.monotonic() + float(timeout_s)
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                diagnostic = _lock_diagnostic(lock_path)
                raise ManualReviewLockTimeoutError(
                    f"Timed out after {float(timeout_s):.3f}s acquiring manual-review store lock "
                    f"{lock_path}; existing lock={diagnostic}. The lock was not broken automatically."
                )
            time.sleep(min(DEFAULT_LOCK_POLL_SECONDS, remaining))
        except OSError as error:
            raise ManualReviewPersistenceError(
                f"Could not acquire manual-review store lock {lock_path}: {error}"
            ) from error
    try:
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        except OSError as error:
            try:
                lock_path.unlink()
            except OSError:
                pass
            raise ManualReviewPersistenceError(
                f"Could not initialize manual-review store lock {lock_path}: {error}"
            ) from error
        finally:
            os.close(descriptor)
            descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        _release_owned_lock(lock_path, token)


def _release_owned_lock(lock_path: Path, token: str) -> None:
    if not lock_path.exists():
        return
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if value.get("token") != token:
        return
    try:
        lock_path.unlink()
    except OSError as error:
        raise ManualReviewPersistenceError(
            f"Could not release owned manual-review store lock {lock_path}: {error}"
        ) from error


def _lock_diagnostic(lock_path: Path) -> Any:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(lock_path), "metadata": "unreadable"}


def _validated_target(path: str | Path) -> Path:
    target = Path(path)
    if not target.name or target.name in {".", ".."}:
        raise ManualReviewPersistenceError(f"Invalid manual-review store target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_file():
        raise ManualReviewPersistenceError(
            f"Manual-review store target exists but is not a regular file: {target}"
        )
    return target


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ManualReviewPersistenceError(
            f"Could not read valid manual-review YAML store {path}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise ManualReviewPersistenceError(f"Manual-review store must contain a mapping: {path}")
    return loaded


def _atomic_replace_text(path: Path, content: str) -> None:
    temporary: Path | None = None
    prior_mode: int | None = None
    if path.exists():
        prior_mode = stat.S_IMODE(path.stat().st_mode)
    try:
        temporary = _write_temporary_text(path, content)
        if prior_mode is not None:
            os.chmod(temporary, prior_mode)
        _replace_canonical(temporary, path)
        temporary = None
        _sync_directory(path.parent)
    except ManualReviewPersistenceError:
        raise
    except OSError as error:
        raise ManualReviewPersistenceError(
            f"Could not atomically persist manual-review file {path}: {error}"
        ) from error
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _write_temporary_text(path: Path, content: str) -> Path:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            _write_text_content(handle, content)
            handle.flush()
            os.fsync(handle.fileno())
            return temporary
    except OSError as error:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        raise ManualReviewPersistenceError(
            f"Could not write temporary manual-review file for {path}: {error}"
        ) from error


def _write_text_content(handle, content: str) -> None:
    handle.write(content)


def _replace_canonical(temporary: Path, path: Path) -> None:
    os.replace(temporary, path)


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        # Windows provides atomic same-volume replacement; Python does not expose
        # a portable directory handle that can be fsynced.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError as error:
        raise ManualReviewPersistenceError(
            f"Manual-review file was replaced but its directory could not be synced {directory}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
