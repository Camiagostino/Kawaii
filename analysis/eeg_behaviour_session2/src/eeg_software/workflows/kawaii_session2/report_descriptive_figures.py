"""Render report-only descriptive ERP figures from a published R11 group generation.

The functions in this module consume existing group-average evokeds only. They
do not load epochs, refit models, or modify any canonical analysis output.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_session2_eeg_config
from .figure_style import FigureStyle, load_figure_style, style_axes, style_legend
from .group_figures import write_group_roi_waveform_figure
from .study_definition import configured_rois, erp_branch_settings


ROI_CHANNELS = ("Cz", "CP1", "CP2", "Pz")
RECOGNITION_ROI_NAMES = ("frontal", "frontocentral", "centroparietal", "parietal")
CUTENESS_CONDITIONS = ("low", "medium_low", "medium_high", "high")
CUTENESS_TOPOGRAPHY_CONDITIONS = ("low", "high")
CUTENESS_TOPOGRAPHY_WINDOW = ("500_800_ms", 0.500, 0.800, "500–800 ms")
TOPOGRAPHY_WINDOWS = (
    ("300_500_ms", 0.300, 0.500, "300–500 ms"),
    ("500_800_ms", 0.500, 0.800, "500–800 ms"),
)
RECOGNITION_COMPARISONS = (
    ("hit_minus_miss", "hit", "miss", "Hit", "Miss", "Hit − Miss"),
    (
        "false_alarm_minus_correct_rejection",
        "false_alarm",
        "correct_rejection",
        "False alarm",
        "Correct rejection",
        "False alarm − Correct rejection",
    ),
    (
        "hit_minus_false_alarm",
        "hit",
        "false_alarm",
        "Hit",
        "False alarm",
        "Hit − False alarm",
    ),
)
REQUIRED_PUBLICATION_ARTIFACTS = (
    "recognition_grand_average_fif",
    "recognition_grand_average_fif__provenance",
    "cuteness_grand_average_fif",
    "cuteness_grand_average_fif__provenance",
)
OUTPUT_SCHEMA = "kawaii_r11_report_descriptive_figures_v3"


@dataclass(frozen=True)
class PublishedGroupSources:
    source_root: Path
    current_manifest: Path
    generation_id: str
    context_reference: str
    artifacts: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]
    participant_ids: tuple[str, ...]
    config_hash: str


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact_path(current_path: Path, generation_id: str, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or not relative or ".." in relative_path.parts:
        raise ValueError(f"Unsafe publication artifact path: {relative!r}")
    generation_root = (current_path.parent / "generations" / generation_id).resolve()
    artifact_path = (generation_root / relative_path).resolve()
    if not artifact_path.is_relative_to(generation_root):
        raise ValueError(f"Publication artifact escapes its generation: {relative!r}")
    return artifact_path


def discover_published_group_sources(source_root: Path) -> PublishedGroupSources:
    """Find and hash-check the current published Gate 5B R11 group evokeds."""

    source_root = Path(source_root).resolve()
    lowered_parts = {part.lower() for part in source_root.parts}
    if "r11_v1" in lowered_parts or "r11_qc_v2" in lowered_parts:
        raise ValueError("Refusing a stale or quarantined R11 source namespace.")
    publication_root = source_root / "group" / ".kawaii-publication"
    candidates: list[tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]] = []
    for current_path in sorted(publication_root.glob("*/current.json")):
        payload = json.loads(current_path.read_text(encoding="utf-8"))
        artifacts = {
            str(item.get("label")): item
            for item in payload.get("artifacts", [])
            if isinstance(item, Mapping)
        }
        if set(REQUIRED_PUBLICATION_ARTIFACTS).issubset(artifacts):
            candidates.append((current_path, payload, artifacts))
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one current group publication containing both recognition "
            f"and cuteness evokeds; found {len(candidates)}."
        )

    current_path, payload, entries = candidates[0]
    if payload.get("schema") != "kawaii_publication_manifest_v1":
        raise ValueError("Unsupported group publication manifest schema.")
    if payload.get("state") != "PUBLISHED" or bool(payload.get("validation_only")):
        raise ValueError("The selected group generation is not a production publication.")
    generation_id = str(payload.get("generation_id", ""))
    if len(generation_id) != 64 or any(char not in "0123456789abcdef" for char in generation_id):
        raise ValueError("The group publication has an invalid generation identity.")

    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for label in REQUIRED_PUBLICATION_ARTIFACTS:
        entry = entries[label]
        path = _safe_artifact_path(current_path, generation_id, str(entry.get("relative_path", "")))
        expected_hash = str(entry.get("sha256", "")).lower()
        if not path.is_file():
            raise FileNotFoundError(f"Published group artifact is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Published group artifact hash mismatch: {label}")
        paths[label] = path
        hashes[label] = actual_hash

    provenance: dict[str, dict[str, Any]] = {}
    for branch in ("recognition", "cuteness"):
        sidecar = json.loads(paths[f"{branch}_grand_average_fif__provenance"].read_text(encoding="utf-8"))
        if sidecar.get("pipeline_stage") != f"gate5b_{branch}_group":
            raise ValueError(f"{branch}: source is not a Gate 5B group evoked publication.")
        participants = tuple(sidecar.get("parameters_summary", {}).get("included_subjects", []))
        if len(participants) != 8 or len(set(participants)) != 8:
            raise ValueError(f"{branch}: expected eight unique included participants.")
        provenance[branch] = sidecar
    recognition_participants = tuple(provenance["recognition"]["parameters_summary"]["included_subjects"])
    cuteness_participants = tuple(provenance["cuteness"]["parameters_summary"]["included_subjects"])
    if recognition_participants != cuteness_participants:
        raise ValueError("Recognition and cuteness group evokeds use different participant cohorts.")
    config_hash = str(provenance["recognition"].get("config_hash", ""))
    if not config_hash or config_hash != str(provenance["cuteness"].get("config_hash", "")):
        raise ValueError("Recognition and cuteness group evokeds have mismatched configuration identities.")

    return PublishedGroupSources(
        source_root=source_root,
        current_manifest=current_path,
        generation_id=generation_id,
        context_reference=str(payload.get("context_reference", "")),
        artifacts=paths,
        artifact_hashes=hashes,
        participant_ids=recognition_participants,
        config_hash=config_hash,
    )


def _condition_key(comment: str) -> str:
    normalized = str(comment).strip().lower().replace("−", "-")
    normalized = normalized.replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    aliases = {
        "false_alarm": "false_alarm",
        "correct_rejection": "correct_rejection",
        "medium_low": "medium_low",
        "medium_high": "medium_high",
        "hit_minus_correct_rejection": "hit_minus_correct_rejection",
    }
    return aliases.get(normalized, normalized)


def read_evoked_mapping(path: Path) -> dict[str, Any]:
    import mne

    mapping: dict[str, Any] = {}
    for evoked in mne.read_evokeds(path, condition=None, verbose=False):
        key = _condition_key(getattr(evoked, "comment", ""))
        if not key or key in mapping:
            raise ValueError(f"Ambiguous evoked comment in {path}: {getattr(evoked, 'comment', '')!r}")
        mapping[key] = evoked
    return mapping


def difference_evoked(first: Any, second: Any, *, label: str) -> Any:
    """Return first minus second after strict channel/time compatibility checks."""

    if tuple(first.ch_names) != tuple(second.ch_names):
        raise ValueError(f"{label}: condition channel inventories differ.")
    first_times = np.asarray(first.times, dtype=float)
    second_times = np.asarray(second.times, dtype=float)
    if first_times.shape != second_times.shape or not np.allclose(first_times, second_times, rtol=0, atol=1e-12):
        raise ValueError(f"{label}: condition time axes differ.")
    result = first.copy()
    result.data = np.asarray(first.data, dtype=float) - np.asarray(second.data, dtype=float)
    result.comment = label
    return result


def roi_trace_uv(evoked: Any, channels: Sequence[str] = ROI_CHANNELS) -> tuple[np.ndarray, np.ndarray]:
    missing = [channel for channel in channels if channel not in evoked.ch_names]
    if missing:
        raise ValueError(f"Evoked {getattr(evoked, 'comment', '')!r} lacks ROI channels: {missing}")
    picks = [evoked.ch_names.index(channel) for channel in channels]
    return np.asarray(evoked.times, dtype=float) * 1000.0, np.asarray(evoked.data[picks], dtype=float).mean(axis=0) * 1e6


def _format_waveform_axis(ax: Any, style: FigureStyle) -> None:
    ax.axvline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=style.axes["reference_alpha"])
    ax.axhline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=style.axes["reference_alpha"])
    for _, start, end, _ in TOPOGRAPHY_WINDOWS:
        ax.axvspan(start * 1000.0, end * 1000.0, color=style.color("primary_window"), alpha=float(style.primary_window_shading["alpha"]), zorder=0)
    ax.set_ylabel("µV")
    style_axes(ax, style)


def write_recognition_roi_waveform(
    first: Any,
    second: Any,
    difference: Any,
    path: Path,
    *,
    keys: tuple[str, str],
    labels: tuple[str, str, str],
    participant_count: int,
    style: FigureStyle,
) -> Path:
    import matplotlib.pyplot as plt

    first_times, first_values = roi_trace_uv(first)
    second_times, second_values = roi_trace_uv(second)
    difference_times, difference_values = roi_trace_uv(difference)
    fig, (conditions_ax, difference_ax) = plt.subplots(
        2,
        1,
        figsize=(8.6, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
        facecolor=style.figure_facecolor,
    )
    for condition, label, times, values in (
        (keys[0], labels[0], first_times, first_values),
        (keys[1], labels[1], second_times, second_values),
    ):
        conditions_ax.plot(
            times,
            values,
            label=label,
            color=style.condition_color("recognition", condition),
            linestyle=style.condition_line_style("recognition", condition),
            linewidth=style.condition_line_width("recognition", condition, roi=True),
        )
    conditions_ax.set_title("Condition waveforms", loc="left", fontsize=style.fonts["panel_title_size"])
    style_legend(conditions_ax, style)
    difference_ax.plot(
        difference_times,
        difference_values,
        label=labels[2],
        color=style.color("difference_waveform"),
        linewidth=style.line_widths["difference_waveform"],
    )
    difference_ax.set_title("Comparison wave", loc="left", fontsize=style.fonts["panel_title_size"])
    style_legend(difference_ax, style)
    for ax in (conditions_ax, difference_ax):
        _format_waveform_axis(ax, style)
    conditions_ax.tick_params(axis="x", labelbottom=False)
    difference_ax.set_xlabel("Time (ms)")
    fig.suptitle(
        f"Recognition: {labels[2]}\nCentroparietal ROI (Cz, CP1, CP2, Pz; N = {participant_count})",
        y=0.985,
        fontsize=style.fonts["title_size"],
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def write_recognition_four_roi_waveform(
    evokeds: Mapping[str, Any],
    path: Path,
    *,
    settings: Mapping[str, Any],
    style: FigureStyle,
) -> Path:
    """Write the established Figure 2 four-ROI layout for two conditions."""

    if len(evokeds) != 2:
        raise ValueError("A recognition four-ROI comparison must contain exactly two conditions.")
    rois = configured_rois(settings)
    if tuple(rois) != RECOGNITION_ROI_NAMES:
        raise ValueError(
            "Recognition configuration must define exactly the established four ROIs "
            f"in order {RECOGNITION_ROI_NAMES}; found {tuple(rois)}."
        )
    for condition, evoked in evokeds.items():
        for roi, channels in rois.items():
            try:
                roi_trace_uv(evoked, channels)
            except ValueError as error:
                raise ValueError(f"{condition}/{roi}: {error}") from error
    return write_group_roi_waveform_figure(
        "recognition",
        settings,
        evokeds,
        path,
        figure_style=style,
    )


def _window_mean_uv(evoked: Any, start_s: float, end_s: float) -> np.ndarray:
    times = np.asarray(evoked.times, dtype=float)
    mask = (times >= start_s) & (times <= end_s)
    if not mask.any():
        raise ValueError(f"Evoked has no samples in {start_s:.3f}–{end_s:.3f} s.")
    return np.asarray(evoked.data[:, mask], dtype=float).mean(axis=1) * 1e6


def _topography_safe_evoked(evoked: Any) -> Any:
    safe = evoked.copy()
    excluded = [channel for channel in ("T1", "T2") if channel in safe.ch_names]
    if excluded:
        safe.drop_channels(excluded)
    return safe


def write_comparison_topographies(
    difference: Any,
    path: Path,
    *,
    title: str,
    participant_count: int,
    style: FigureStyle,
) -> Path:
    import matplotlib.pyplot as plt
    import mne

    safe = _topography_safe_evoked(difference)
    window_values = [_window_mean_uv(safe, start, end) for _, start, end, _ in TOPOGRAPHY_WINDOWS]
    limit = max(float(np.nanmax(np.abs(values))) for values in window_values)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError(f"{title}: cannot determine a non-zero symmetric topography scale.")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1), facecolor=style.figure_facecolor)
    image = None
    for ax, values, (_, _, _, window_label) in zip(axes, window_values, TOPOGRAPHY_WINDOWS, strict=True):
        image, _ = mne.viz.plot_topomap(
            values,
            safe.info,
            axes=ax,
            show=False,
            cmap=style.topography_colormap(difference=True),
            vlim=(-limit, limit),
            contours=int(style.topography["contour_count"]),
            sensors=True,
            outlines="head",
        )
        ax.set_title(window_label, fontsize=style.fonts["panel_title_size"])
    colorbar_axis = fig.add_axes((0.91, 0.19, 0.022, 0.55))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("µV", fontsize=style.topography["colorbar_unit_size"])
    fig.suptitle(f"{title}\nDifference-wave scalp distribution (N = {participant_count})", y=0.98, fontsize=style.fonts["title_size"])
    fig.subplots_adjust(left=0.04, right=0.86, bottom=0.08, top=0.78, wspace=0.20)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def _cuteness_low_high_topography_data(
    evokeds: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
    missing = [condition for condition in CUTENESS_TOPOGRAPHY_CONDITIONS if condition not in evokeds]
    if missing:
        raise ValueError(f"Cuteness grand averages lack topography conditions: {missing}")
    _, start_s, end_s, _ = CUTENESS_TOPOGRAPHY_WINDOW
    safe_evokeds = {
        condition: _topography_safe_evoked(evokeds[condition])
        for condition in CUTENESS_TOPOGRAPHY_CONDITIONS
    }
    channel_inventories = {tuple(evoked.ch_names) for evoked in safe_evokeds.values()}
    if len(channel_inventories) != 1:
        raise ValueError("Low and High cuteness topographies have different channel inventories.")
    window_values = {
        condition: _window_mean_uv(evoked, start_s, end_s)
        for condition, evoked in safe_evokeds.items()
    }
    limit = max(float(np.nanmax(np.abs(values))) for values in window_values.values())
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Cuteness Low/High topographies require a non-zero symmetric scale.")
    return safe_evokeds, window_values, limit


def write_cuteness_low_high_topographies(
    evokeds: Mapping[str, Any],
    path: Path,
    *,
    participant_count: int,
    style: FigureStyle,
) -> Path:
    """Write descriptive Low/High 500–800-ms maps with one signed scale."""

    import matplotlib.pyplot as plt
    import mne

    safe_evokeds, window_values, limit = _cuteness_low_high_topography_data(evokeds)
    window_label = CUTENESS_TOPOGRAPHY_WINDOW[3]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1), facecolor=style.figure_facecolor)
    image = None
    for ax, condition in zip(axes, CUTENESS_TOPOGRAPHY_CONDITIONS, strict=True):
        image, _ = mne.viz.plot_topomap(
            window_values[condition],
            safe_evokeds[condition].info,
            axes=ax,
            show=False,
            cmap=style.topography_colormap(),
            vlim=(-limit, limit),
            contours=int(style.topography["contour_count"]),
            sensors=True,
            outlines="head",
        )
        ax.set_title(style.condition_label(condition), fontsize=style.fonts["panel_title_size"])
    colorbar_axis = fig.add_axes((0.91, 0.19, 0.022, 0.55))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("µV", fontsize=style.topography["colorbar_unit_size"])
    fig.suptitle(
        f"Cuteness ratings · {window_label}\nDescriptive scalp distributions (N = {participant_count})",
        y=0.98,
        fontsize=style.fonts["title_size"],
    )
    fig.subplots_adjust(left=0.04, right=0.86, bottom=0.08, top=0.78, wspace=0.20)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def write_cuteness_descriptive_waveform(
    evokeds: Mapping[str, Any],
    path: Path,
    *,
    participant_count: int,
    style: FigureStyle,
) -> Path:
    import matplotlib.pyplot as plt

    required = CUTENESS_CONDITIONS
    missing = [condition for condition in required if condition not in evokeds]
    if missing:
        raise ValueError(f"Cuteness grand averages lack conditions: {missing}")
    fig, ax = plt.subplots(figsize=(8.2, 4.5), facecolor=style.figure_facecolor)
    for condition in required:
        times, values = roi_trace_uv(evokeds[condition])
        ax.plot(
            times,
            values,
            label=style.condition_label(condition),
            color=style.condition_color("cuteness", condition),
            linestyle=style.condition_line_style("cuteness", condition),
            linewidth=style.condition_line_width("cuteness", condition, roi=True),
        )
    _format_waveform_axis(ax, style)
    ax.set_xlabel("Time (ms)")
    style_legend(ax, style)
    fig.suptitle(
        "Descriptive grand-average ERPs by cuteness rating\n"
        f"Centroparietal ROI (Cz, CP1, CP2, Pz; N = {participant_count})",
        y=0.97,
        fontsize=style.fonts["title_size"],
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def render_r11_report_descriptive_figures(
    source_root: Path,
    output_dir: Path,
    *,
    project_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render the requested figures into a non-canonical output directory."""

    source = discover_published_group_sources(source_root)
    output_dir = Path(output_dir).resolve()
    if output_dir == source.source_root or output_dir.is_relative_to(source.source_root):
        raise ValueError("Report figures must be written outside the immutable R11 source tree.")
    project_root = Path(project_root).resolve()
    config_path = project_root / "config" / "session2_eeg.yaml"
    if _sha256(config_path) != source.config_hash:
        raise ValueError(
            "The local Kawaii analysis configuration does not match the configuration "
            "identity recorded by the published Gate 5B sources."
        )
    config = load_session2_eeg_config(root=project_root, config_path=config_path)
    recognition_settings = erp_branch_settings(config, "recognition")
    recognition_rois = configured_rois(recognition_settings)
    if tuple(recognition_rois) != RECOGNITION_ROI_NAMES:
        raise ValueError(
            "Recognition configuration must define exactly the established four ROIs "
            f"in order {RECOGNITION_ROI_NAMES}; found {tuple(recognition_rois)}."
        )
    style_path = project_root / "config" / "kawaii_session2_figure_style.yaml"
    style = load_figure_style(root=project_root, config_path=style_path)
    recognition = read_evoked_mapping(source.artifacts["recognition_grand_average_fif"])
    cuteness = read_evoked_mapping(source.artifacts["cuteness_grand_average_fif"])
    required_recognition = {item for comparison in RECOGNITION_COMPARISONS for item in comparison[1:3]}
    missing_recognition = sorted(required_recognition.difference(recognition))
    if missing_recognition:
        raise ValueError(f"Recognition grand averages lack conditions: {missing_recognition}")
    missing_cuteness = sorted(set(CUTENESS_CONDITIONS).difference(cuteness))
    if missing_cuteness:
        raise ValueError(f"Cuteness grand averages lack conditions: {missing_cuteness}")
    differences = {
        name: difference_evoked(recognition[first_key], recognition[second_key], label=comparison_label)
        for name, first_key, second_key, _, _, comparison_label in RECOGNITION_COMPARISONS
    }
    # Complete data/ROI/window validation before any overwrite or output creation.
    for evoked in [
        *(recognition[key] for key in sorted(required_recognition)),
        *(differences.values()),
        *(cuteness[key] for key in CUTENESS_CONDITIONS),
    ]:
        roi_trace_uv(evoked)
    for key in sorted(required_recognition):
        for channels in recognition_rois.values():
            roi_trace_uv(recognition[key], channels)
    for difference in differences.values():
        for _, start, end, _ in TOPOGRAPHY_WINDOWS:
            _window_mean_uv(difference, start, end)
    _, _, cuteness_topography_limit_uv = _cuteness_low_high_topography_data(cuteness)

    filenames = [
        *(f"{name}_roi_waveform.png" for name, *_ in RECOGNITION_COMPARISONS),
        *(f"{name}_four_roi_waveforms.{suffix}" for name, *_ in RECOGNITION_COMPARISONS for suffix in ("png", "svg")),
        *(f"{name}_topographies.png" for name, *_ in RECOGNITION_COMPARISONS),
        "cuteness_centroparietal_descriptive_waveforms.png",
        "cuteness_centroparietal_descriptive_waveforms.svg",
        "cuteness_low_high_topographies_500_800ms.png",
        "cuteness_low_high_topographies_500_800ms.svg",
        "generation_manifest.json",
    ]
    existing = [output_dir / filename for filename in filenames if (output_dir / filename).exists()]
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists; use --overwrite or a new path: {output_dir}")
    if overwrite:
        for path in existing:
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    comparisons_manifest: dict[str, Any] = {}
    for name, first_key, second_key, first_label, second_label, comparison_label in RECOGNITION_COMPARISONS:
        difference = differences[name]
        waveform_path = write_recognition_roi_waveform(
            recognition[first_key],
            recognition[second_key],
            difference,
            output_dir / f"{name}_roi_waveform.png",
            keys=(first_key, second_key),
            labels=(first_label, second_label, comparison_label),
            participant_count=len(source.participant_ids),
            style=style,
        )
        comparison_evokeds = {
            first_key: recognition[first_key],
            second_key: recognition[second_key],
        }
        four_roi_paths = tuple(
            write_recognition_four_roi_waveform(
                comparison_evokeds,
                output_dir / f"{name}_four_roi_waveforms.{suffix}",
                settings=recognition_settings,
                style=style,
            )
            for suffix in ("png", "svg")
        )
        topography_path = write_comparison_topographies(
            difference,
            output_dir / f"{name}_topographies.png",
            title=comparison_label,
            participant_count=len(source.participant_ids),
            style=style,
        )
        output_paths.extend((waveform_path, *four_roi_paths, topography_path))
        comparisons_manifest[name] = {
            "minuend": first_key,
            "subtrahend": second_key,
            "roi_waveform": waveform_path.name,
            "four_roi_waveforms": {
                path.suffix.removeprefix("."): path.name for path in four_roi_paths
            },
            "topographies": topography_path.name,
        }

    cuteness_paths = tuple(
        write_cuteness_descriptive_waveform(
            cuteness,
            output_dir / f"cuteness_centroparietal_descriptive_waveforms.{suffix}",
            participant_count=len(source.participant_ids),
            style=style,
        )
        for suffix in ("png", "svg")
    )
    output_paths.extend(cuteness_paths)
    cuteness_topography_paths = tuple(
        write_cuteness_low_high_topographies(
            cuteness,
            output_dir / f"cuteness_low_high_topographies_500_800ms.{suffix}",
            participant_count=len(source.participant_ids),
            style=style,
        )
        for suffix in ("png", "svg")
    )
    output_paths.extend(cuteness_topography_paths)
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "root": str(source.source_root),
            "current_manifest": str(source.current_manifest),
            "generation_id": source.generation_id,
            "context_reference": source.context_reference,
            "config_hash": source.config_hash,
            "analysis_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "participants": list(source.participant_ids),
            "n_participants": len(source.participant_ids),
            "artifacts": {
                label: {"path": str(source.artifacts[label]), "sha256": source.artifact_hashes[label]}
                for label in REQUIRED_PUBLICATION_ARTIFACTS
            },
            "figure_style": {"path": str(style_path), "sha256": _sha256(style_path)},
        },
        "parameters": {
            "roi_channels": list(ROI_CHANNELS),
            "recognition_rois": recognition_rois,
            "time_windows": [
                {"name": name, "start_ms": int(start * 1000), "end_ms": int(end * 1000)}
                for name, start, end, _ in TOPOGRAPHY_WINDOWS
            ],
            "topography_scale": "symmetric about zero within each comparison figure",
            "cuteness_conditions": list(CUTENESS_CONDITIONS),
            "cuteness_topography": {
                "conditions": list(CUTENESS_TOPOGRAPHY_CONDITIONS),
                "window": {
                    "name": CUTENESS_TOPOGRAPHY_WINDOW[0],
                    "start_ms": int(CUTENESS_TOPOGRAPHY_WINDOW[1] * 1000),
                    "end_ms": int(CUTENESS_TOPOGRAPHY_WINDOW[2] * 1000),
                },
                "colormap": str(style.topography["colormap"]),
                "scale": "shared_symmetric_about_zero",
                "vlim_uv": [-cuteness_topography_limit_uv, cuteness_topography_limit_uv],
                "orientation": "nose_up",
                "outputs": {
                    path.suffix.removeprefix("."): path.name for path in cuteness_topography_paths
                },
            },
            "amplitude_orientation": "positive_up",
        },
        "comparisons": comparisons_manifest,
        "outputs": [
            {"path": path.name, "sha256": _sha256(path), "status": "newly_generated"}
            for path in output_paths
        ],
        "canonical_analysis_outputs_modified": False,
    }
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"figures": tuple(output_paths), "manifest": manifest_path, "source": source}
