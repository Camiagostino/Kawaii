from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from eeg_software.project import ensure_directory

from .identity import (
    SourceIdentityError,
    brainvision_working_lineage_path,
    build_brainvision_working_lineage,
    identify_brainvision_dataset,
    read_brainvision_working_lineage,
    validate_brainvision_working_lineage,
    write_brainvision_working_lineage,
)
from .paths import resolve_session2_eeg_paths


class BrainVisionLoadingError(RuntimeError):
    """Raised when BrainVision recordings cannot be discovered or validated."""


class BrainVisionValidationError(BrainVisionLoadingError):
    """Raised when a discovered BrainVision recording fails validation."""


@dataclass(frozen=True)
class FilenameValidationStatus:
    """Validation state for internal BrainVision filename references."""

    vhdr_datafile_matches: bool
    vhdr_markerfile_matches: bool
    vmrk_datafile_matches: bool
    working_copy_created: bool = False
    messages: tuple[str, ...] = ()
    working_copy_action: str = "not_applicable"
    lineage_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.vhdr_datafile_matches and self.vhdr_markerfile_matches and self.vmrk_datafile_matches


@dataclass(frozen=True)
class ParticipantRecordings:
    """Validated recording parts for one participant.

    A participant may have one or more parts. Parts stay separate through
    loading, alignment, and preprocessing; valid epochs may be combined only
    after part-level preprocessing/QC has succeeded.
    """

    participant: str
    recording_parts: tuple["RecordingPart", ...]


@dataclass(frozen=True)
class RecordingPart:
    """Validated BrainVision recording part ready for later EEG loading.

    ``recording_part`` is the participant-level part label such as ``part1`` or
    ``part2``. ``file_base`` is the BrainVision file stem, such as
    ``kawaii0004_2``. Both are preserved so interrupted/restarted recordings
    remain traceable without treating resumed files as separate participants.

    Paths may point to source files when internal filename references are already
    consistent. If repairable filename mismatches are found, paths point to
    derivative-side working copies. Source files are never modified.
    """

    subject: str
    recording_part: str
    file_base: str
    vhdr_path: Path
    vmrk_path: Path
    eeg_path: Path
    sampling_rate_hz: float
    channel_names: tuple[str, ...]
    filename_validation: FilenameValidationStatus
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BrainVisionHeader:
    """Parsed subset of a BrainVision header or marker file."""

    path: Path
    common_infos: dict[str, str]
    channel_names: tuple[str, ...] = ()
    sampling_rate_hz: float | None = None
    raw_text: str = ""


@dataclass(frozen=True)
class _WorkingCopyResult:
    vhdr_path: Path
    vmrk_path: Path
    eeg_path: Path
    lineage_path: Path
    action: str
    detail: str


SUBJECT_NUMBER_RE = re.compile(r"(\d+)$")
CHANNEL_RE = re.compile(r"^Ch(?P<number>\d+)$", re.IGNORECASE)


def discover_participant_recordings(subject: str, config: Mapping[str, Any], *, root: Path | None = None) -> ParticipantRecordings:
    """Discover and validate all BrainVision parts for one participant."""

    return ParticipantRecordings(participant=subject, recording_parts=tuple(discover_recording_parts(subject, config, root=root)))


def discover_recording_parts(subject: str, config: Mapping[str, Any], *, root: Path | None = None) -> list[RecordingPart]:
    """Discover and validate all BrainVision recording parts for one participant.

    Configured participants may name parts explicitly as ``part1``, ``part2``,
    etc. Unconfigured future participants are discovered from BrainVision file
    stems and assigned stable part labels in sorted order.
    """

    paths = resolve_session2_eeg_paths(config, root=root, create=True)
    part_specs = _configured_part_specs(subject, config)
    if not part_specs:
        part_specs = _discover_part_specs(subject, paths.raw_eeg_dir, config)
    if not part_specs:
        raise BrainVisionLoadingError(f"No BrainVision .vhdr files found for {subject} in {paths.raw_eeg_dir}.")

    parts = []
    for part_label, file_base in part_specs:
        parts.append(validate_recording_part(subject, part_label, file_base, config, raw_eeg_dir=paths.raw_eeg_dir, working_root=paths.brainvision_working_dir))
    return parts


def validate_recording_part(
    subject: str,
    recording_part: str,
    file_base: str,
    config: Mapping[str, Any],
    *,
    raw_eeg_dir: Path,
    working_root: Path,
) -> RecordingPart:
    """Validate one BrainVision triplet and return its loadable paths."""

    source_vhdr = raw_eeg_dir / f"{file_base}.vhdr"
    source_vmrk = raw_eeg_dir / f"{file_base}.vmrk"
    source_eeg = raw_eeg_dir / f"{file_base}.eeg"
    _require_triplet(source_vhdr, source_vmrk, source_eeg)

    vhdr = read_brainvision_header(source_vhdr)
    vmrk = read_brainvision_marker_header(source_vmrk)
    sampling_rate_hz = _require_sampling_rate(vhdr, config)
    channel_names = _require_channel_names(vhdr, config)
    filename_status = _validate_filename_references(vhdr, vmrk, source_vmrk.name, source_eeg.name)

    warnings = list(filename_status.messages)
    if filename_status.ok:
        return RecordingPart(
            subject=subject,
            recording_part=recording_part,
            file_base=file_base,
            vhdr_path=source_vhdr,
            vmrk_path=source_vmrk,
            eeg_path=source_eeg,
            sampling_rate_hz=sampling_rate_hz,
            channel_names=channel_names,
            filename_validation=filename_status,
            warnings=tuple(warnings),
        )

    working = _create_corrected_working_copies(
        subject=subject,
        recording_part=recording_part,
        file_base=file_base,
        source_vhdr=source_vhdr,
        source_vmrk=source_vmrk,
        source_eeg=source_eeg,
        working_root=working_root,
    )
    corrected_status = FilenameValidationStatus(
        vhdr_datafile_matches=filename_status.vhdr_datafile_matches,
        vhdr_markerfile_matches=filename_status.vhdr_markerfile_matches,
        vmrk_datafile_matches=filename_status.vmrk_datafile_matches,
        working_copy_created=True,
        messages=filename_status.messages,
        working_copy_action=working.action,
        lineage_path=working.lineage_path,
    )
    warnings.append(working.detail)

    return RecordingPart(
        subject=subject,
        recording_part=recording_part,
        file_base=file_base,
        vhdr_path=working.vhdr_path,
        vmrk_path=working.vmrk_path,
        eeg_path=working.eeg_path,
        sampling_rate_hz=sampling_rate_hz,
        channel_names=channel_names,
        filename_validation=corrected_status,
        warnings=tuple(warnings),
    )


def read_brainvision_header(path: Path) -> BrainVisionHeader:
    """Read a BrainVision ``.vhdr`` file without loading EEG samples."""

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sections = _parse_ini_like_sections(text)
    common = sections.get("Common Infos", {})
    channel_infos = sections.get("Channel Infos", {})
    channel_names = _parse_channel_names(channel_infos)
    sampling_rate_hz = _sampling_rate_from_common(common)
    return BrainVisionHeader(path=path, common_infos=common, channel_names=channel_names, sampling_rate_hz=sampling_rate_hz, raw_text=text)


def read_brainvision_marker_header(path: Path) -> BrainVisionHeader:
    """Read the header/common-info subset of a BrainVision ``.vmrk`` file."""

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sections = _parse_ini_like_sections(text)
    return BrainVisionHeader(path=path, common_infos=sections.get("Common Infos", {}), raw_text=text)


def concise_recording_summary(parts: Sequence[RecordingPart]) -> str:
    """Return a compact human-readable summary for entry scripts."""

    lines = []
    for part in parts:
        copy_note = "working-copy" if part.filename_validation.working_copy_created else "source"
        lines.append(
            f"{part.subject} {part.recording_part} ({part.file_base}): {part.sampling_rate_hz:g} Hz, "
            f"{len(part.channel_names)} channels, {copy_note}, vhdr={part.vhdr_path}"
        )
        for warning in part.warnings:
            lines.append(f"  WARNING: {warning}")
    return "\n".join(lines)


def _configured_part_specs(subject: str, config: Mapping[str, Any]) -> list[tuple[str, str]]:
    recordings = config.get("recordings", {})
    if not isinstance(recordings, Mapping):
        return []
    participant_recordings = recordings.get("participant_recordings", {}) or {}
    subject_config = participant_recordings.get(subject, {}) if isinstance(participant_recordings, Mapping) else {}
    if not isinstance(subject_config, Mapping):
        return []
    specs = []
    for part_label, part_config in sorted(subject_config.items(), key=lambda item: _part_sort_key(str(item[0]))):
        if isinstance(part_config, Mapping):
            file_base = part_config.get("brainvision_file_base")
        else:
            file_base = part_config
        if file_base:
            specs.append((str(part_label), str(file_base)))
    return specs


def _discover_part_specs(subject: str, raw_eeg_dir: Path, config: Mapping[str, Any]) -> list[tuple[str, str]]:
    number_match = SUBJECT_NUMBER_RE.search(subject)
    if not number_match:
        raise BrainVisionLoadingError(f"Could not infer recording number from subject ID {subject!r}.")
    template = str(config.get("recordings", {}).get("filename_template", "kawaii{subject_number:04d}"))
    base = template.format(subject=subject, subject_number=int(number_match.group(1)))
    vhdr_files = sorted(raw_eeg_dir.glob(f"{base}*.vhdr"), key=lambda path: _recording_sort_key(path.stem))
    return [(f"part{index}", path.stem) for index, path in enumerate(vhdr_files, start=1)]


def _part_sort_key(part_label: str) -> tuple[int, str]:
    match = re.fullmatch(r"part(\d+)", part_label)
    return (int(match.group(1)), part_label) if match else (9999, part_label)


def _recording_sort_key(stem: str) -> tuple[str, int, str]:
    suffix_match = re.search(r"_(\d+)$", stem)
    suffix = int(suffix_match.group(1)) if suffix_match else 0
    base = re.sub(r"_\d+$", "", stem)
    return base, suffix, stem


def _require_triplet(vhdr: Path, vmrk: Path, eeg: Path) -> None:
    missing = [path for path in (vhdr, vmrk, eeg) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise BrainVisionLoadingError(f"Missing BrainVision triplet file(s): {missing_text}")


def _parse_ini_like_sections(text: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        if current is not None and "=" in line:
            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()
    return sections


def _parse_channel_names(channel_infos: Mapping[str, str]) -> tuple[str, ...]:
    indexed: list[tuple[int, str]] = []
    for key, value in channel_infos.items():
        match = CHANNEL_RE.fullmatch(key)
        if not match:
            continue
        channel_name = value.split(",", 1)[0].replace("\\1", ",").strip()
        indexed.append((int(match.group("number")), channel_name))
    return tuple(name for _, name in sorted(indexed))


def _sampling_rate_from_common(common_infos: Mapping[str, str]) -> float | None:
    raw_interval = common_infos.get("SamplingInterval")
    if raw_interval is None:
        return None
    try:
        interval_us = float(raw_interval)
    except ValueError as error:
        raise BrainVisionValidationError(f"Invalid SamplingInterval value {raw_interval!r}.") from error
    if interval_us <= 0:
        raise BrainVisionValidationError(f"SamplingInterval must be positive, got {raw_interval!r}.")
    return 1_000_000.0 / interval_us


def _require_sampling_rate(header: BrainVisionHeader, config: Mapping[str, Any]) -> float:
    if header.sampling_rate_hz is None:
        raise BrainVisionValidationError(f"Missing SamplingInterval in {header.path}.")
    recordings = config.get("recordings", {})
    expected = float(recordings.get("expected_sampling_rate_hz", 500.0)) if isinstance(recordings, Mapping) else 500.0
    tolerance = float(recordings.get("sampling_rate_tolerance_hz", 0.001)) if isinstance(recordings, Mapping) else 0.001
    if abs(header.sampling_rate_hz - expected) > tolerance:
        raise BrainVisionValidationError(f"Invalid sampling rate for {header.path}: expected {expected:g} Hz, got {header.sampling_rate_hz:g} Hz.")
    return header.sampling_rate_hz


def _require_channel_names(header: BrainVisionHeader, config: Mapping[str, Any]) -> tuple[str, ...]:
    channel_names = header.channel_names
    if not channel_names:
        raise BrainVisionValidationError(f"No channel names found in {header.path}.")

    channel_config = config.get("channels", {})
    eeg_config = channel_config.get("eeg", {}) if isinstance(channel_config, Mapping) else {}
    if not isinstance(eeg_config, Mapping):
        eeg_config = {}
    expected_count = int(eeg_config.get("expected_count", len(channel_names)))
    if len(channel_names) != expected_count:
        raise BrainVisionValidationError(f"Invalid channel count for {header.path}: expected {expected_count}, got {len(channel_names)}.")

    expected_names = _expected_channel_name_set(config)
    if expected_names:
        actual = {_canonical_channel_name(channel, config) for channel in channel_names}
        missing = sorted(expected_names - actual)
        extra = sorted(actual - expected_names)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extra:
                details.append(f"extra: {', '.join(extra)}")
            raise BrainVisionValidationError(f"Invalid channel names for {header.path}: {'; '.join(details)}.")
    return channel_names


def _expected_channel_name_set(config: Mapping[str, Any]) -> set[str]:
    channel_config = config.get("channels", {})
    eeg_config = channel_config.get("eeg", {}) if isinstance(channel_config, Mapping) else {}
    if not isinstance(eeg_config, Mapping):
        return set()
    names: set[str] = set()
    for key in ("expected_names", "eog", "unused", "misc"):
        values = eeg_config.get(key, []) or []
        names.update(_canonical_channel_name(str(value), config) for value in values)
    return names


def _canonical_channel_name(channel: str, config: Mapping[str, Any]) -> str:
    channel_config = config.get("channels", {})
    if not isinstance(channel_config, Mapping):
        return str(channel)
    aliases = dict(channel_config.get("channel_name_aliases", {}) or {})
    aliases.update(dict(channel_config.get("channel_name_overrides", {}) or {}))
    return str(aliases.get(str(channel), channel))


def _validate_filename_references(vhdr: BrainVisionHeader, vmrk: BrainVisionHeader, expected_vmrk_name: str, expected_eeg_name: str) -> FilenameValidationStatus:
    vhdr_datafile = _basename(vhdr.common_infos.get("DataFile", ""))
    vhdr_markerfile = _basename(vhdr.common_infos.get("MarkerFile", ""))
    vmrk_datafile = _basename(vmrk.common_infos.get("DataFile", ""))

    vhdr_data_ok = vhdr_datafile == expected_eeg_name
    vhdr_marker_ok = vhdr_markerfile == expected_vmrk_name
    vmrk_data_ok = vmrk_datafile == expected_eeg_name

    messages = []
    if not vhdr_data_ok:
        messages.append(f"{vhdr.path.name} DataFile references {vhdr_datafile!r}, expected {expected_eeg_name!r}.")
    if not vhdr_marker_ok:
        messages.append(f"{vhdr.path.name} MarkerFile references {vhdr_markerfile!r}, expected {expected_vmrk_name!r}.")
    if not vmrk_data_ok:
        messages.append(f"{vmrk.path.name} DataFile references {vmrk_datafile!r}, expected {expected_eeg_name!r}.")

    return FilenameValidationStatus(
        vhdr_datafile_matches=vhdr_data_ok,
        vhdr_markerfile_matches=vhdr_marker_ok,
        vmrk_datafile_matches=vmrk_data_ok,
        messages=tuple(messages),
    )


def _basename(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def _create_corrected_working_copies(
    *,
    subject: str,
    recording_part: str,
    file_base: str,
    source_vhdr: Path,
    source_vmrk: Path,
    source_eeg: Path,
    working_root: Path,
) -> _WorkingCopyResult:
    working_dir = ensure_directory(working_root / subject / recording_part)
    working_vhdr = working_dir / source_vhdr.name
    working_vmrk = working_dir / source_vmrk.name
    working_eeg = working_dir / source_eeg.name
    lineage_path = brainvision_working_lineage_path(working_dir)

    try:
        source_identity = identify_brainvision_dataset(
            source_vhdr,
            source_vmrk,
            source_eeg,
            logical_root=source_vhdr.parent,
            allow_filename_reference_correction=True,
        )
    except SourceIdentityError as error:
        raise BrainVisionValidationError(
            f"Cannot identify original BrainVision dataset for {subject} {recording_part}: {error}"
        ) from error

    working_paths = (working_vhdr, working_vmrk, working_eeg)
    existing_members = tuple(path.exists() for path in working_paths)
    regeneration_reason: str | None = None
    if all(existing_members):
        if not lineage_path.exists():
            regeneration_reason = "legacy working copy has no authoritative lineage"
        else:
            try:
                working_identity = identify_brainvision_dataset(
                    working_vhdr,
                    working_vmrk,
                    working_eeg,
                    logical_root=working_dir,
                )
                lineage = read_brainvision_working_lineage(lineage_path)
                validate_brainvision_working_lineage(
                    lineage,
                    participant=subject,
                    session="session2",
                    recording_part=recording_part,
                    source_dataset=source_identity,
                    working_dataset=working_identity,
                )
            except SourceIdentityError as error:
                regeneration_reason = f"working lineage is stale or invalid: {error}"
            else:
                return _WorkingCopyResult(
                    working_vhdr,
                    working_vmrk,
                    working_eeg,
                    lineage_path,
                    "reused_validated",
                    f"Reused lineage-validated corrected BrainVision working copies in {working_dir}.",
                )
    elif any(existing_members) or lineage_path.exists():
        regeneration_reason = "working copy set is incomplete"

    vhdr_text = source_vhdr.read_text(encoding="utf-8-sig", errors="replace")
    vmrk_text = source_vmrk.read_text(encoding="utf-8-sig", errors="replace")
    _atomic_replace_text(
        working_vhdr,
        _replace_common_info_value(
            vhdr_text,
            {"DataFile": source_eeg.name, "MarkerFile": source_vmrk.name},
        ),
    )
    _atomic_replace_text(
        working_vmrk,
        _replace_common_info_value(vmrk_text, {"DataFile": source_eeg.name}),
    )
    _replace_eeg_link_or_copy(source_eeg, working_eeg)
    try:
        working_identity = identify_brainvision_dataset(
            working_vhdr,
            working_vmrk,
            working_eeg,
            logical_root=working_dir,
        )
        lineage = build_brainvision_working_lineage(
            participant=subject,
            session="session2",
            recording_part=recording_part,
            source_dataset=source_identity,
            working_dataset=working_identity,
        )
        write_brainvision_working_lineage(lineage, lineage_path)
    except SourceIdentityError as error:
        raise BrainVisionValidationError(
            f"Corrected BrainVision working lineage failed for {subject} {recording_part}: {error}"
        ) from error

    if regeneration_reason is None:
        action = "created"
        detail = f"Created corrected BrainVision working copies with authoritative lineage in {working_dir}."
    else:
        action = "regenerated"
        detail = (
            f"Regenerated corrected BrainVision working copies in {working_dir}; "
            f"{regeneration_reason}."
        )
    return _WorkingCopyResult(
        working_vhdr,
        working_vmrk,
        working_eeg,
        lineage_path,
        action,
        detail,
    )


def _replace_common_info_value(text: str, replacements: Mapping[str, str]) -> str:
    lines = text.splitlines()
    in_common = False
    found = {key: False for key in replacements}
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_common = stripped == "[Common Infos]"
        if in_common and "=" in line and not stripped.startswith(";"):
            key, _ = line.split("=", 1)
            clean_key = key.strip()
            if clean_key in replacements:
                out.append(f"{clean_key}={replacements[clean_key]}")
                found[clean_key] = True
                continue
        out.append(line)

    for key, was_found in found.items():
        if not was_found:
            raise BrainVisionValidationError(f"Cannot correct BrainVision header because [Common Infos] lacks {key}.")
    return "\n".join(out) + "\n"


def _replace_eeg_link_or_copy(source_eeg: Path, working_eeg: Path) -> None:
    temporary = _temporary_sibling_path(working_eeg)
    try:
        try:
            os.link(source_eeg, temporary)
        except OSError:
            shutil.copy2(source_eeg, temporary)
        os.replace(temporary, working_eeg)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_replace_text(path: Path, content: str) -> None:
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _temporary_sibling_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary
