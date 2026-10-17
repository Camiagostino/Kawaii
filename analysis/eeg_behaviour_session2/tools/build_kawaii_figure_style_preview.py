from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_software.project import get_project_root, read_yaml_config
from eeg_software.workflows.kawaii_session2.figure_style import (
    FigureStyle,
    load_figure_style,
    style_axes,
    style_legend,
    style_topography_figure,
)
from eeg_software.workflows.kawaii_session2.group_figures import (
    write_difference_waveform_figure,
    write_group_roi_waveform_figure,
    write_group_topographies,
)
from eeg_software.workflows.kawaii_session2.study_definition import erp_branch_settings


CHANNELS = [
    "Fz", "F1", "F2", "FCz", "FC1", "FC2", "Cz", "C1", "C2",
    "CPz", "CP1", "CP2", "Pz", "P1", "P2", "POz", "O1", "Oz", "O2",
]
PRIMARY_RECOGNITION_HIT_PREVIEW_COLOR = "#4C6F8C"


def _recognition_primary_preview_style(style: FigureStyle) -> FigureStyle:
    """Lighten only the Hit line in the collaborator-facing primary ROI preview."""

    colors = deepcopy(dict(style.colors))
    recognition_colors = dict(colors["recognition_conditions"])
    recognition_colors["hit"] = PRIMARY_RECOGNITION_HIT_PREVIEW_COLOR
    colors["recognition_conditions"] = recognition_colors
    return replace(style, colors=colors)


def _evoked(times: np.ndarray, data_uv: np.ndarray, comment: str) -> mne.EvokedArray:
    info = mne.create_info(CHANNELS, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    return mne.EvokedArray(data_uv * 1e-6, info, tmin=float(times[0]), comment=comment)


def _synthetic_evokeds() -> tuple[dict[str, mne.EvokedArray], dict[str, mne.EvokedArray]]:
    times = np.arange(-0.2, 1.004, 0.004)
    frontal = np.linspace(1.1, 0.55, len(CHANNELS))[:, None]
    posterior = np.linspace(0.45, 1.25, len(CHANNELS))[:, None]
    n2 = -1.4 * np.exp(-0.5 * ((times - 0.22) / 0.055) ** 2)
    early = np.exp(-0.5 * ((times - 0.40) / 0.09) ** 2)
    late = np.exp(-0.5 * ((times - 0.64) / 0.12) ** 2)
    drift = 0.18 * np.sin(2 * np.pi * 2.2 * times)

    def recognition(scale_early: float, scale_late: float, comment: str) -> mne.EvokedArray:
        values = frontal * (n2 + scale_early * early) + posterior * scale_late * late + drift
        return _evoked(times, values, comment)

    recognition_evokeds = {
        "hit": recognition(3.3, 4.4, "Hit"),
        "miss": recognition(2.3, 2.6, "Miss"),
        "false_alarm": recognition(2.0, 2.2, "False alarm"),
        "correct_rejection": recognition(2.5, 3.0, "Correct rejection"),
    }
    difference = mne.combine_evoked(
        [recognition_evokeds["hit"], recognition_evokeds["correct_rejection"]],
        weights=[1.0, -1.0],
    )
    difference.comment = "Hit − Correct rejection"
    recognition_evokeds["hit_minus_correct_rejection"] = difference

    cuteness_evokeds: dict[str, mne.EvokedArray] = {}
    for condition, scale, comment in (
        ("low", 1.8, "Low"),
        ("medium_low", 2.5, "Medium-low"),
        ("medium_high", 3.2, "Medium-high"),
        ("high", 4.0, "High"),
    ):
        values = frontal * (n2 + 0.65 * scale * early) + posterior * scale * late + drift
        cuteness_evokeds[condition] = _evoked(times, values, comment)
    return recognition_evokeds, cuteness_evokeds


def _write_gallery(paths: list[tuple[str, Path]], destination: Path, *, title: str) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
    for ax, (panel_title, path) in zip(axes.flat, paths):
        ax.imshow(plt.imread(path))
        ax.set_title(panel_title, fontsize=12)
        ax.axis("off")
    for ax in axes.flat[len(paths) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    return destination


def _write_bipolar_diagnostic(
    destination: Path,
    style: FigureStyle,
) -> Path:
    info = mne.create_info(CHANNELS, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    positions = info.get_montage().get_positions()["ch_pos"]
    values_uv = np.asarray(
        [5.0 * positions[channel][0] / max(abs(positions[name][0]) for name in CHANNELS) for channel in CHANNELS]
    )
    evoked = mne.EvokedArray(values_uv[:, None] * 1e-6, info, tmin=0.6, comment="Bipolar diagnostic")
    limit = float(np.max(np.abs(values_uv)))
    fig = evoked.plot_topomap(
        times=[0.6],
        show=False,
        time_unit="s",
        cmap=style.topography_colormap(difference=True),
        vlim=(-limit, limit),
        contours=int(style.topography["contour_count"]),
    )
    fig.axes[0].set_title("")
    fig.set_size_inches(*style.figure_size("topography"), forward=True)
    fig.suptitle("Preview diagnostic · known negative/positive lobes", y=0.98, fontsize=style.fonts["compact_legend_size"])
    style_topography_figure(fig, style)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return destination


def _read_current_publication(root: Path, stage: str, branch: str | None = None) -> tuple[Path, dict[str, Any]]:
    publication_root = root / "workspace" / "kawaii" / "session2" / "analysis" / "group" / ".kawaii-publication"
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in publication_root.glob("*/current.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        scope = dict(record.get("scope", {}) or {})
        if scope.get("stage") != stage:
            continue
        if branch is not None and scope.get("branch") != branch:
            continue
        matches.append((path, record))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one current {stage}/{branch or 'all-branches'} publication; found {len(matches)}.")
    return matches[0]


def _verified_artifact(current_path: Path, publication: Mapping[str, Any], label: str) -> Path:
    matches = [artifact for artifact in publication.get("artifacts", []) if artifact.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"Current publication has {len(matches)} artifacts labelled {label!r}.")
    artifact = matches[0]
    path = current_path.parent / "generations" / str(publication["generation_id"]) / str(artifact["relative_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Published artifact is missing: {path}")
    if path.stat().st_size != int(artifact["size_bytes"]):
        raise ValueError(f"Published artifact size mismatch: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != str(artifact["sha256"]):
        raise ValueError(f"Published artifact SHA-256 mismatch: {path}")
    return path


def _evoked_mapping(path: Path, branch: str) -> dict[str, mne.Evoked]:
    aliases = {
        "hit": "hit",
        "miss": "miss",
        "false alarm": "false_alarm",
        "correct rejection": "correct_rejection",
        "hit - correct rejection": "hit_minus_correct_rejection",
        "hit minus correct rejection": "hit_minus_correct_rejection",
        "low": "low",
        "medium-low": "medium_low",
        "medium low": "medium_low",
        "medium-high": "medium_high",
        "medium high": "medium_high",
        "high": "high",
    }
    result: dict[str, mne.Evoked] = {}
    for evoked in mne.read_evokeds(path, verbose=False):
        normalized = str(evoked.comment).strip().lower().replace("−", "-")
        key = aliases.get(normalized)
        if key is None:
            raise ValueError(f"Unrecognized {branch} evoked comment {evoked.comment!r} in {path}.")
        result[key] = evoked
    return result


def _write_recognition_primary_results(table: pd.DataFrame, destination: Path, style: FigureStyle) -> Path:
    primary = table.loc[table["analysis_status"].astype(str).eq("PRIMARY")].copy()
    if len(primary) != 4:
        raise ValueError(f"Recognition primary preview requires exactly four frozen PRIMARY rows; found {len(primary)}.")
    primary["window_order"] = primary["time_window"].map({"300_500_ms": 0, "500_800_ms": 1})
    primary = primary.sort_values(["window_order", "roi"]).reset_index(drop=True)
    labels = [f"{str(row.roi).replace('_', ' ').title()} · {str(row.time_window).replace('_', '–').replace('–ms', ' ms')}" for row in primary.itertuples()]
    estimates = pd.to_numeric(primary["effect_estimate"], errors="raise").to_numpy(float)
    low = pd.to_numeric(primary["ci_low"], errors="raise").to_numpy(float)
    high = pd.to_numeric(primary["ci_high"], errors="raise").to_numpy(float)
    y = np.arange(len(primary))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 3.15), facecolor=style.figure_facecolor)
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - low, high - estimates]),
        fmt="o",
        color=style.condition_color("recognition", "hit"),
        ecolor=style.color("difference_waveform"),
        elinewidth=1.5,
        capsize=3,
        markersize=5,
    )
    ax.axvline(0.0, color=style.color("reference"), linewidth=1.0, alpha=0.75)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Hit − Correct rejection mean amplitude (µV), 95% CI")
    ax.set_title("Recognition primary effects", fontsize=style.fonts["title_size"])
    adjusted = pd.to_numeric(primary["adjusted_p"], errors="coerce")
    for ypos, upper, pvalue in zip(y, high, adjusted):
        ax.annotate(f"Holm p={pvalue:.2f}", (upper, ypos), xytext=(6, 0), textcoords="offset points", va="center", fontsize=style.fonts["compact_legend_size"], color="#4E5962")
    fig.text(
        0.265,
        0.035,
        "N = 8; 95% CI; Holm-adjusted p values",
        fontsize=style.fonts["compact_legend_size"],
        color="#5F6972",
    )
    style_axes(ax, style)
    fig.subplots_adjust(left=0.265, right=0.91, bottom=0.27, top=0.82)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return destination


def _write_cuteness_primary_coefficient(
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame,
    destination: Path,
    style: FigureStyle,
) -> Path:
    primary = coefficients.loc[coefficients["coefficient_role"].astype(str).eq("PRIMARY_COEFFICIENT")]
    if len(primary) != 1:
        raise ValueError(f"Cuteness coefficient preview requires one PRIMARY_COEFFICIENT row; found {len(primary)}.")
    row = primary.iloc[0]
    estimate = float(row["estimate"])
    low = float(row["confidence_interval_low"])
    high = float(row["confidence_interval_high"])
    if diagnostics.empty:
        raise ValueError("Cuteness coefficient preview requires frozen model diagnostics.")
    diagnostic = diagnostics.iloc[0]
    status = str(diagnostic["model_status"])
    if status != "FIT_WARNING":
        raise ValueError(f"Expected frozen cuteness model status FIT_WARNING; found {status!r}.")
    n_observations = int(diagnostic["n_observations"])
    n_participants = int(diagnostic["n_participants"])
    fig, ax = plt.subplots(figsize=(8.0, 2.65), facecolor=style.figure_facecolor)
    ax.errorbar(
        [estimate],
        [0],
        xerr=[[estimate - low], [high - estimate]],
        fmt="o",
        color=style.condition_color("cuteness", "high"),
        ecolor=style.condition_color("cuteness", "medium_low"),
        elinewidth=2.0,
        capsize=4,
        markersize=7,
    )
    ax.axvline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=style.axes["reference_alpha"])
    ax.set_yticks([0], ["Centred cuteness rating"])
    ax.set_xlabel("ERP change per rating unit (µV), 95% CI")
    ax.set_title("Primary continuous cuteness coefficient", fontsize=style.fonts["title_size"])
    fig.text(
        0.22,
        0.035,
        "Model converged; near-boundary random-effect variance. Interpret cautiously.",
        fontsize=style.fonts["compact_legend_size"],
        color="#6F5652",
    )
    ax.text(
        0.99,
        0.90,
        (
            f"estimate = {estimate:.3f} µV/rating unit\n"
            f"95% CI = [{low:.4f}, {high:.4f}]\n"
            f"p = {float(row['p_value']):.3f}".replace("p = 0.", "p = .")
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=style.fonts["compact_legend_size"],
        color="#4E5962",
    )
    ax.text(
        0.01,
        0.90,
        f"N = {n_observations:,} trials; {n_participants} participants",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.fonts["compact_legend_size"],
        color="#5F6972",
    )
    style_axes(ax, style)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.34, top=0.78)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return destination


def _write_cuteness_primary_roi_descriptive(
    evokeds: Mapping[str, mne.Evoked],
    destination: Path,
    style: FigureStyle,
) -> Path:
    """Plot frozen categorical ERPs descriptively in the continuous-model ROI."""

    conditions = ("low", "medium_low", "medium_high", "high")
    roi_channels = ("Cz", "CP1", "CP2", "Pz")
    missing_conditions = [condition for condition in conditions if condition not in evokeds]
    if missing_conditions:
        raise ValueError(f"Frozen cuteness evokeds lack descriptive conditions: {missing_conditions}")

    traces: list[tuple[str, np.ndarray, np.ndarray]] = []
    for condition in conditions:
        evoked = evokeds[condition]
        missing_channels = [channel for channel in roi_channels if channel not in evoked.ch_names]
        if missing_channels:
            raise ValueError(f"Frozen {condition} evoked lacks primary ROI channels: {missing_channels}")
        indices = [evoked.ch_names.index(channel) for channel in roi_channels]
        times_ms = np.asarray(evoked.times, dtype=float) * 1000.0
        trace_uv = np.asarray(evoked.data[indices], dtype=float).mean(axis=0) * 1e6
        traces.append((condition, times_ms, trace_uv))

    fig, ax = plt.subplots(figsize=(8.0, 4.0), facecolor=style.figure_facecolor)
    for condition, times_ms, trace_uv in traces:
        ax.plot(
            times_ms,
            trace_uv,
            label=style.condition_label(condition),
            color=style.condition_color("cuteness", condition),
            linestyle=style.condition_line_style("cuteness", condition),
            linewidth=style.condition_line_width("cuteness", condition, roi=True),
        )
    ax.axvspan(500.0, 800.0, color=style.color("primary_window"), alpha=float(style.primary_window_shading["alpha"]), zorder=0)
    ax.axvline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=style.axes["reference_alpha"])
    ax.axhline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=style.axes["reference_alpha"])
    all_values = np.concatenate([trace for _, _, trace in traces])
    lower, upper = float(np.min(all_values)), float(np.max(all_values))
    padding = max((upper - lower) * 0.08, 0.25)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("µV")
    fig.suptitle("Descriptive categorical ERPs", y=0.97, fontsize=style.fonts["title_size"])
    ax.set_title(
        "Frozen bins · primary continuous-model ROI: Cz, CP1, CP2, Pz\n"
        "500–800 ms shaded · descriptive only; no categorical inference",
        fontsize=style.fonts["compact_legend_size"],
        color="#5F6972",
        pad=8,
    )
    style_axes(ax, style)
    style_legend(ax, style, size=style.fonts["compact_legend_size"])
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.76)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return destination


def build_preview(root: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_yaml_config(root / "config" / "session2_eeg.yaml")
    style = load_figure_style(root=root)
    recognition_settings = erp_branch_settings(config, "recognition")
    cuteness_settings = erp_branch_settings(config, "cuteness")
    recognition_primary_style = _recognition_primary_preview_style(style)
    recognition, cuteness = _synthetic_evokeds()

    recognition_roi = write_group_roi_waveform_figure(
        "recognition",
        recognition_settings,
        recognition,
        output_dir / "recognition_roi_waveforms.png",
        figure_style=style,
    )
    recognition_primary_roi = write_group_roi_waveform_figure(
        "recognition",
        recognition_settings,
        {condition: recognition[condition] for condition in ("hit", "correct_rejection")},
        output_dir / "recognition_primary_hit_vs_correct_rejection_roi_waveforms.png",
        figure_style=recognition_primary_style,
    )
    recognition_difference = write_difference_waveform_figure(
        recognition["hit_minus_correct_rejection"],
        output_dir / "recognition_hit_minus_correct_rejection.png",
        figure_style=style,
        settings=recognition_settings,
    )
    recognition_topographies = write_group_topographies(
        "recognition",
        recognition_settings,
        recognition,
        output_dir / "recognition_topographies",
        [],
        figure_style=style,
    )
    cuteness_roi = write_group_roi_waveform_figure(
        "cuteness",
        cuteness_settings,
        cuteness,
        output_dir / "cuteness_roi_waveforms.png",
        figure_style=style,
    )
    cuteness_topographies = write_group_topographies(
        "cuteness",
        cuteness_settings,
        cuteness,
        output_dir / "cuteness_topographies",
        [],
        figure_style=style,
    )
    bipolar = _write_bipolar_diagnostic(output_dir / "bipolar_colormap_diagnostic.png", style)

    selected = [
        ("Recognition primary ROI waveforms", recognition_primary_roi),
        ("Hit − Correct rejection", recognition_difference),
        ("Recognition topography", recognition_topographies["topography_hit_minus_correct_rejection_500_800_ms"]),
        ("Cuteness ROI waveforms", cuteness_roi),
        ("Cuteness topography", cuteness_topographies["topography_high_500_800_ms"]),
        ("Diverging-palette diagnostic", bipolar),
    ]
    gallery = _write_gallery(
        selected,
        output_dir / "kawaii_phase_b_synthetic_preview_gallery.png",
        title="Kawaii Session 2 · presentation preview (synthetic data)",
    )

    recognition_current, recognition_publication = _read_current_publication(root, "gate5b", "recognition")
    cuteness_current, cuteness_publication = _read_current_publication(root, "gate5b", "cuteness")
    statistics_current, statistics_publication = _read_current_publication(root, "gate5c")
    recognition_fif = _verified_artifact(recognition_current, recognition_publication, "grand_average_fif")
    cuteness_fif = _verified_artifact(cuteness_current, cuteness_publication, "grand_average_fif")
    recognition_stats_path = _verified_artifact(statistics_current, statistics_publication, "recognition_statistics")
    cuteness_coefficients_path = _verified_artifact(statistics_current, statistics_publication, "cuteness_mixed_model_coefficients")
    cuteness_diagnostics_path = _verified_artifact(statistics_current, statistics_publication, "cuteness_mixed_model_diagnostics")

    real_recognition = _evoked_mapping(recognition_fif, "recognition")
    real_cuteness = _evoked_mapping(cuteness_fif, "cuteness")
    required_recognition = {"hit", "correct_rejection", "hit_minus_correct_rejection"}
    required_cuteness = {"low", "medium_low", "medium_high", "high"}
    if not required_recognition.issubset(real_recognition):
        raise ValueError(f"Frozen recognition evokeds lack: {sorted(required_recognition.difference(real_recognition))}")
    if not required_cuteness.issubset(real_cuteness):
        raise ValueError(f"Frozen cuteness evokeds lack: {sorted(required_cuteness.difference(real_cuteness))}")

    real_dir = output_dir / "frozen_real_data"
    real_recognition_primary = write_group_roi_waveform_figure(
        "recognition",
        recognition_settings,
        {condition: real_recognition[condition] for condition in ("hit", "correct_rejection")},
        real_dir / "recognition_primary_hit_vs_correct_rejection_roi_waveforms.png",
        figure_style=recognition_primary_style,
    )
    real_recognition_difference = write_difference_waveform_figure(
        real_recognition["hit_minus_correct_rejection"],
        real_dir / "recognition_hit_minus_correct_rejection.png",
        figure_style=style,
        settings=recognition_settings,
    )
    real_recognition_topographies = write_group_topographies(
        "recognition",
        recognition_settings,
        {"hit_minus_correct_rejection": real_recognition["hit_minus_correct_rejection"]},
        real_dir / "recognition_topographies",
        [],
        figure_style=style,
    )
    real_cuteness_roi = write_group_roi_waveform_figure(
        "cuteness",
        cuteness_settings,
        real_cuteness,
        real_dir / "cuteness_roi_waveforms.png",
        figure_style=style,
    )
    real_cuteness_primary_descriptive = _write_cuteness_primary_roi_descriptive(
        real_cuteness,
        real_dir / "cuteness_primary_roi_descriptive_waveform.png",
        style,
    )
    real_cuteness_topographies = write_group_topographies(
        "cuteness",
        cuteness_settings,
        real_cuteness,
        real_dir / "cuteness_topographies",
        [],
        figure_style=style,
    )
    real_selected = [
        ("Recognition primary ROI waveforms", real_recognition_primary),
        ("Hit − Correct rejection", real_recognition_difference),
        ("Recognition difference topography", real_recognition_topographies["topography_hit_minus_correct_rejection_500_800_ms"]),
        ("Cuteness ROI waveforms", real_cuteness_roi),
        ("Cuteness topography", real_cuteness_topographies["topography_high_500_800_ms"]),
    ]
    real_gallery = _write_gallery(
        real_selected,
        real_dir / "kawaii_phase_b_frozen_real_data_gallery.png",
        title="Kawaii Session 2 · frozen-result presentation preview",
    )

    summaries_dir = output_dir / "report_summaries"
    recognition_statistics = pd.read_csv(recognition_stats_path)
    cuteness_coefficients = pd.read_csv(cuteness_coefficients_path)
    cuteness_diagnostics = pd.read_csv(cuteness_diagnostics_path)
    recognition_summary = _write_recognition_primary_results(
        recognition_statistics,
        summaries_dir / "recognition_primary_results_forest.png",
        style,
    )
    cuteness_summary = _write_cuteness_primary_coefficient(
        cuteness_coefficients,
        cuteness_diagnostics,
        summaries_dir / "cuteness_primary_coefficient.png",
        style,
    )
    proposed_report_gallery = _write_gallery(
        [
            ("Recognition primary ROI waveform", real_recognition_primary),
            ("Recognition primary effects", recognition_summary),
            ("Cuteness descriptive primary ROI", real_cuteness_primary_descriptive),
            ("Cuteness continuous coefficient", cuteness_summary),
            (
                "Recognition difference topography (optional)",
                real_recognition_topographies["topography_hit_minus_correct_rejection_500_800_ms"],
            ),
        ],
        summaries_dir / "proposed_final_report_gallery.png",
        title="Kawaii Session 2 · proposed final-report figure set",
    )
    diagnostic = cuteness_diagnostics.iloc[0]
    source_record = {
        "preview_only": True,
        "canonical_outputs_modified": False,
        "publications": {
            "gate5b_recognition": {"generation_id": recognition_publication["generation_id"], "artifact": str(recognition_fif)},
            "gate5b_cuteness": {"generation_id": cuteness_publication["generation_id"], "artifact": str(cuteness_fif)},
            "gate5c": {
                "generation_id": statistics_publication["generation_id"],
                "artifacts": [str(recognition_stats_path), str(cuteness_coefficients_path), str(cuteness_diagnostics_path)],
                "cuteness_model_status": str(diagnostic["model_status"]),
                "cuteness_fit_warnings": str(diagnostic["fit_warnings"]),
            },
        },
    }
    source_path = output_dir / "preview_sources.json"
    source_path.write_text(json.dumps(source_record, indent=2) + "\n", encoding="utf-8")
    return [
        gallery,
        recognition_roi,
        recognition_primary_roi,
        recognition_difference,
        recognition_topographies["topography_hit_minus_correct_rejection_500_800_ms"],
        cuteness_roi,
        cuteness_topographies["topography_high_500_800_ms"],
        bipolar,
        real_gallery,
        *(path for _, path in real_selected),
        real_cuteness_primary_descriptive,
        recognition_summary,
        cuteness_summary,
        proposed_report_gallery,
        source_path,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build non-canonical synthetic and frozen-data Kawaii figure-style previews.")
    parser.add_argument("--root", type=Path, default=get_project_root())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "workspace" / "presentation_preview" / "kawaii_phase_b_revision_2026-10-09").resolve()
    canonical = (root / "workspace" / "kawaii" / "session2" / "analysis").resolve()
    if output_dir == canonical or canonical in output_dir.parents:
        raise ValueError("Preview output must remain outside canonical workspace/kawaii/session2/analysis outputs.")
    for path in build_preview(root, output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
