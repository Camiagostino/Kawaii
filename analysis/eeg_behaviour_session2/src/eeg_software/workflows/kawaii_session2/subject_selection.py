from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .config import participant_defaults

SUBJECT_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)(?P<number>\d+)$")
RANGE_RE = re.compile(r"^(?P<start_prefix>[A-Za-z]+)?(?P<start>\d+)\s*-\s*(?P<end_prefix>[A-Za-z]+)?(?P<end>\d+)$")


def _validation_config(config: Mapping[str, Any] | None) -> tuple[set[str], int, dict[str, int]]:
    if not config:
        return {"sub", "pilot"}, 3, {"sub": 3, "pilot": 2}
    participants = config.get("participants", {})
    validation = participants.get("validation", {}) if isinstance(participants, Mapping) else {}
    if not isinstance(validation, Mapping):
        validation = {}
    allowed = validation.get("allowed_prefixes", ["sub", "pilot"])
    width = int(validation.get("numeric_width", 3))
    prefix_widths_raw = validation.get("prefix_numeric_widths", {}) or {}
    prefix_widths = {str(prefix): int(prefix_width) for prefix, prefix_width in dict(prefix_widths_raw).items()}
    return {str(prefix) for prefix in allowed}, width, prefix_widths


def normalize_subject_id(subject_id: str, *, config: Mapping[str, Any] | None = None) -> str:
    """Validate and normalize one subject ID.

    Examples
    --------
    ``sub5`` becomes ``sub005`` when the config uses width 3.
    """

    allowed_prefixes, numeric_width, prefix_widths = _validation_config(config)
    text = str(subject_id).strip()
    match = SUBJECT_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid subject ID: {subject_id!r}. Expected IDs like sub005 or pilot03.")

    prefix = match.group("prefix")
    number_text = match.group("number")
    if prefix not in allowed_prefixes:
        raise ValueError(f"Invalid subject prefix in {subject_id!r}. Allowed prefixes: {', '.join(sorted(allowed_prefixes))}.")

    width = max(prefix_widths.get(prefix, numeric_width), len(number_text))
    return f"{prefix}{int(number_text):0{width}d}"


def _expand_range(token: str, *, config: Mapping[str, Any] | None = None) -> list[str]:
    allowed_prefixes, numeric_width, prefix_widths = _validation_config(config)
    match = RANGE_RE.fullmatch(token)
    if not match:
        return [normalize_subject_id(token, config=config)]

    start_prefix = match.group("start_prefix")
    end_prefix = match.group("end_prefix")
    if not start_prefix and not end_prefix:
        start_prefix = "sub"
    prefix = start_prefix or end_prefix
    if start_prefix and end_prefix and start_prefix != end_prefix:
        raise ValueError(f"Cannot expand mixed-prefix subject range: {token!r}.")
    if prefix not in allowed_prefixes:
        raise ValueError(f"Invalid subject prefix in range {token!r}. Allowed prefixes: {', '.join(sorted(allowed_prefixes))}.")

    start_text = match.group("start")
    end_text = match.group("end")
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"Subject ranges must be ascending: {token!r}.")

    width = max(prefix_widths.get(prefix, numeric_width), len(start_text), len(end_text))
    return [f"{prefix}{number:0{width}d}" for number in range(start, end + 1)]


def _split_subject_tokens(subject_args: Sequence[str] | None) -> list[str]:
    if not subject_args:
        return []
    tokens: list[str] = []
    for value in subject_args:
        for piece in str(value).replace(",", " ").split():
            if piece:
                tokens.append(piece)
    return tokens


def parse_subject_args(subject_args: Sequence[str] | None, *, config: Mapping[str, Any] | None = None) -> list[str]:
    """Expand, deduplicate, validate, and naturally sort CLI subject arguments.

    Supports single IDs, multiple IDs, comma-separated values, and ranges such as
    ``sub005-sub020``. Bare numeric ranges such as ``5-8`` default to ``sub``.
    """

    subjects: list[str] = []
    tokens = _split_subject_tokens(subject_args)
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1] == "-":
            token = f"{tokens[i]}-{tokens[i + 2]}"
            i += 3
        else:
            token = tokens[i]
            i += 1
        subjects.extend(_expand_range(token, config=config))
    return sorted(set(subjects), key=subject_sort_key)


def subject_sort_key(subject_id: str) -> tuple[str, int, str]:
    """Natural sort key that keeps prefixes grouped and numbers ordered."""

    match = SUBJECT_RE.fullmatch(subject_id)
    if not match:
        return subject_id, -1, subject_id
    return match.group("prefix"), int(match.group("number")), subject_id


def resolve_subjects(
    subject_args: Sequence[str] | None,
    config: Mapping[str, Any],
    *,
    allow_excluded: bool = False,
) -> list[str]:
    """Resolve analysis subjects from CLI args or YAML defaults.

    If no CLI subjects are supplied, YAML ``participants.defaults.included`` is
    used and YAML exclusions are applied. Explicit CLI subjects are allowed even
    when excluded so pilots can be used for alignment/QC testing; callers should
    warn when that happens.
    """

    defaults = participant_defaults(config)
    source_subjects: Iterable[str]
    explicit_subjects = bool(subject_args)
    if explicit_subjects:
        source_subjects = subject_args
    else:
        source_subjects = defaults["included"]

    subjects = parse_subject_args(list(source_subjects), config=config)
    excluded = set(parse_subject_args(defaults["excluded"], config=config))
    if not explicit_subjects and not allow_excluded:
        subjects = [subject for subject in subjects if subject not in excluded]
    if not subjects:
        raise ValueError("No subjects remain after parsing defaults/CLI arguments and applying exclusions.")
    return subjects


def explicit_excluded_subjects(subject_args: Sequence[str] | None, config: Mapping[str, Any]) -> list[str]:
    """Return explicitly requested subjects that are listed as default exclusions."""

    if not subject_args:
        return []
    defaults = participant_defaults(config)
    subjects = set(parse_subject_args(subject_args, config=config))
    excluded = set(parse_subject_args(defaults["excluded"], config=config))
    return sorted(subjects & excluded, key=subject_sort_key)
