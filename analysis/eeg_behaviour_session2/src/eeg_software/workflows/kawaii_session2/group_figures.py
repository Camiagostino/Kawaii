from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bad_channel_disposition import retained_bad_channel_marker
from .figure_style import (
    FigureStyle,
    default_figure_style,
    load_figure_style,
    shade_primary_windows,
    style_axes,
    style_legend,
    style_topography_figure,
)
from .group_aggregation import (
    pool_participant_measurements,
    validate_gate4_publication_for_group,
    validate_pooled_measurements,
)
from .paths import resolve_session2_eeg_paths, subject_part_dir
from .provenance import write_group_provenance_sidecar
from .publication import PublicationValidationError
from .resolved_snapshot import ResolvedRunSnapshotReference, bind_authoritative_config_provenance
from .study_definition import (
    configured_conditions,
    configured_rois,
    configured_time_windows,
    erp_branch_settings,
    plotting_safe_evokeds,
)


@dataclass(frozen=True)
class Gate5BResult:
    output_files: dict[str, Path]
    warnings: tuple[str, ...]
    info_messages: tuple[str, ...] = ()


def gate5b_output_paths(config: Mapping[str, Any], branch: str, *, root: Path | None = None) -> dict[str, Path]:
    """Return Gate 5B descriptive group-output paths for one branch."""

    base = resolve_session2_eeg_paths(config, root=root, create=True).output_dir / "group" / branch
    figures = base / "figures"
    waveforms = figures / "waveforms"
    topographies = figures / "topographies"
    for path in (base, figures, waveforms, topographies):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "figures": figures,
        "waveforms": waveforms,
        "topographies": topographies,
        "grand_average_fif": base / "grand_average_erps-ave.fif",
        "descriptive_summary": base / "descriptive_summary.csv",
        "group_qc_summary": base / "group_qc_summary.csv",
        "report": base / "report.md",
        "waveform_figure": waveforms / "grand_average_waveforms.png",
        "roi_waveform_figure": waveforms / "roi_grand_average_waveforms.png",
        "difference_waveform_figure": waveforms / "difference_waveform.png",
        "trial_count_figure": figures / "trial_count_summary.png",
        "inclusion_figure": figures / "participant_inclusion_summary.png",
    }


def run_gate5b_group_figures(
    config: Mapping[str, Any],
    *,
    root: Path | None = None,
    config_path: str | Path | None = None,
    resolved_snapshot: ResolvedRunSnapshotReference | None = None,
    published_input_paths_by_branch: Mapping[str, Mapping[str, Path]] | None = None,
    publication_context_reference: str | None = None,
    publication_scientific_fingerprint: str | None = None,
    expected_participant_publications: Mapping[str, str] | None = None,
    immutable_upstream_root: str | Path | None = None,
) -> Gate5BResult:
    """Generate Gate 5B descriptive group ERP outputs only."""

    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    output_files: dict[str, Path] = {}
    warnings: list[str] = []
    info_messages: list[str] = []
    for branch in ("recognition", "cuteness"):
        branch_outputs = run_gate5b_branch(config, branch, root=root, config_path=config_path, warnings=warnings, info_messages=info_messages, published_input_paths=(published_input_paths_by_branch or {}).get(branch), publication_context_reference=publication_context_reference, publication_scientific_fingerprint=publication_scientific_fingerprint, expected_participant_publications=expected_participant_publications, immutable_upstream_root=immutable_upstream_root)
        output_files.update({f"{branch}_{label}": path for label, path in branch_outputs.items()})
    return Gate5BResult(output_files=output_files, warnings=tuple(warnings), info_messages=tuple(info_messages))


def run_gate5b_branch(config: Mapping[str, Any], branch: str, *, root: Path | None = None, config_path: str | Path | None = None, resolved_snapshot: ResolvedRunSnapshotReference | None = None, warnings: list[str] | None = None, info_messages: list[str] | None = None, published_input_paths: Mapping[str, Path] | None = None, publication_context_reference: str | None = None, publication_scientific_fingerprint: str | None = None, expected_participant_publications: Mapping[str, str] | None = None, immutable_upstream_root: str | Path | None = None) -> dict[str, Path]:
    """Generate descriptive group outputs for one ERP branch."""

    config_path = bind_authoritative_config_provenance(config_path, resolved_snapshot)
    warnings = warnings if warnings is not None else []
    info_messages = info_messages if info_messages is not None else []
    if publication_context_reference is not None and publication_scientific_fingerprint is None:
        raise PublicationValidationError(
            "Gate 5B publication consumption requires the current scientific fingerprint."
        )
    settings = erp_branch_settings(config, branch)
    figure_style = load_figure_style(root=root)
    paths = gate5b_output_paths(config, branch, root=root)
    inputs = dict(published_input_paths or gate5a_input_paths(config, branch, root=root))
    missing_inputs = [label for label, path in inputs.items() if not path.exists()]
    if missing_inputs:
        warnings.append(f"{branch}: missing Gate 5A input(s): {', '.join(missing_inputs)}")

    measurements = _read_csv(inputs["measurements"])
    if measurements.empty:
        measurements = pool_participant_measurements(_read_csv(inputs["measurements_parts"]))
    validate_pooled_measurements(measurements)
    wide = _read_csv(inputs["wide"])
    participant_summary = _read_csv(inputs["participant_summary"])
    expected_exclusions = expected_evoked_exclusions(participant_summary, branch)
    for exclusion in expected_exclusions:
        info_messages.append(f"{branch} {exclusion['subject']}: excluded from grand averages ({exclusion['reason']}); missing participant evoked FIFs are expected.")
    included_rows = included_participant_rows(participant_summary, wide, branch)
    included_measurements = filter_included_measurements(measurements, included_rows)

    participant_publication_inputs: list[Path] | None = [] if publication_context_reference is not None else None
    grand_averages = load_grand_average_evokeds(config, branch, settings, included_rows, root=root, warnings=warnings, publication_context_reference=publication_context_reference, publication_scientific_fingerprint=publication_scientific_fingerprint, expected_participant_publications=expected_participant_publications, publication_input_paths=participant_publication_inputs, immutable_upstream_root=immutable_upstream_root)
    output_files: dict[str, Path] = {}
    if grand_averages:
        output_files["grand_average_fif"] = save_grand_average_evokeds(grand_averages, paths["grand_average_fif"])
    else:
        warnings.append(f"{branch}: no grand-average evokeds generated.")

    descriptive = build_descriptive_summary(included_measurements)
    output_files["descriptive_summary"] = _write_table(descriptive, paths["descriptive_summary"])
    qc = build_group_qc_summary(branch, participant_summary, included_measurements, settings, grand_averages, expected_evoked_exclusions=expected_exclusions)
    output_files["group_qc_summary"] = _write_table(qc, paths["group_qc_summary"])

    output_files.update(write_group_figures(branch, settings, grand_averages, included_measurements, participant_summary, paths, warnings, figure_style=figure_style))
    output_files["report"] = write_group_report(branch, paths["report"], output_files, warnings, qc, info_messages=info_messages)
    _write_gate5b_provenance(config, branch, output_files, inputs, participant_summary, included_rows, warnings, root=root, config_path=config_path, published_participant_inputs=participant_publication_inputs)
    return output_files


def gate5a_input_paths(config: Mapping[str, Any], branch: str, *, root: Path | None = None) -> dict[str, Path]:
    base = resolve_session2_eeg_paths(config, root=root, create=False).output_dir / "group" / branch
    return {
        "measurements": base / f"{branch}_measurements_pooled.csv",
        "measurements_parts": base / f"{branch}_measurements_all.csv",
        "wide": base / f"{branch}_measurements_all_wide.csv",
        "participant_summary": base / "participant_summary.csv",
    }


def included_participant_rows(participant_summary: pd.DataFrame, wide: pd.DataFrame, branch: str) -> pd.DataFrame:
    if participant_summary.empty or wide.empty:
        return pd.DataFrame(columns=["subject", "recording_part"])
    include_col = f"{branch}_included"
    if include_col not in participant_summary.columns:
        return pd.DataFrame(columns=["subject", "recording_part"])
    included_subjects = set(participant_summary.loc[_as_bool_series(participant_summary[include_col]), "subject"].astype(str))
    rows = wide[wide["subject"].astype(str).isin(included_subjects)].copy() if "subject" in wide.columns else pd.DataFrame()
    keep = [column for column in ("subject", "recording_part") if column in rows.columns]
    return rows[keep].drop_duplicates().reset_index(drop=True) if keep else pd.DataFrame(columns=["subject", "recording_part"])


def filter_included_measurements(measurements: pd.DataFrame, included_rows: pd.DataFrame) -> pd.DataFrame:
    if measurements.empty or included_rows.empty:
        return pd.DataFrame(columns=measurements.columns)
    if "aggregation_method" in measurements.columns:
        subjects = set(included_rows["subject"].dropna().astype(str))
        return measurements[measurements["subject"].astype(str).isin(subjects)].copy()
    keys = included_rows[["subject", "recording_part"]].drop_duplicates()
    return measurements.merge(keys, on=["subject", "recording_part"], how="inner")


def expected_evoked_exclusions(participant_summary: pd.DataFrame, branch: str) -> list[dict[str, str]]:
    """Return participants whose missing evoked FIFs are expected from Gate 5A inclusion."""

    include_col = f"{branch}_included"
    reason_col = f"{branch}_reason"
    if participant_summary.empty or include_col not in participant_summary.columns:
        return []
    excluded = participant_summary.loc[~_as_bool_series(participant_summary[include_col])].copy()
    rows = []
    for _, item in excluded.iterrows():
        rows.append(
            {
                "subject": str(item.get("subject", "")),
                "reason": str(item.get(reason_col, "excluded by Gate 5A participant_summary.csv")),
            }
        )
    return rows


def load_grand_average_evokeds(config: Mapping[str, Any], branch: str, settings: Mapping[str, Any], included_rows: pd.DataFrame, *, root: Path | None, warnings: list[str], publication_context_reference: str | None = None, publication_scientific_fingerprint: str | None = None, expected_participant_publications: Mapping[str, str] | None = None, publication_input_paths: list[Path] | None = None, immutable_upstream_root: str | Path | None = None) -> dict[str, Any]:
    evokeds_by_condition: dict[str, dict[str, list[Any]]] = {condition: {} for condition in group_condition_order(settings, branch)}
    label_map = condition_label_map(settings, branch)
    for _, row in included_rows.iterrows():
        subject = str(row["subject"])
        recording_part = str(row["recording_part"])
        part_dir = subject_part_dir(config, subject, recording_part, root=root, create=False)
        paths = participant_evoked_paths(part_dir, branch)
        if publication_context_reference is not None:
            stage = "gate4a" if branch == "recognition" else "gate4b"
            key = f"publication:{stage}:{branch}:{subject}:{recording_part}"
            expected_generation = (expected_participant_publications or {}).get(key)
            if expected_generation is None:
                raise PublicationValidationError(
                    f"Gate 5A publication does not bind {branch} {subject} {recording_part}."
                )
            publication, _ = validate_gate4_publication_for_group(
                config,
                part_dir,
                subject,
                recording_part,
                branch,
                str(publication_scientific_fingerprint),
                expected_generation_id=expected_generation,
                immutable_upstream_canonical_directory=(
                    Path(immutable_upstream_root)
                    / "subjects"
                    / subject
                    / recording_part
                    if immutable_upstream_root is not None
                    else None
                ),
            )
            evoked_labels = ["participant_erps_fif"]
            if branch == "recognition":
                evoked_labels.append("difference_wave_fif")
            published_evokeds = {
                label: publication.artifact_path(label) for label in evoked_labels
            }
            missing_evokeds = [
                label for label, path in published_evokeds.items() if path is None
            ]
            if missing_evokeds:
                raise PublicationValidationError(
                    "Current Gate 4 publication lacks required Gate 5B evoked artifacts: "
                    + ", ".join(missing_evokeds)
                    + "."
                )
            paths = [Path(path) for path in published_evokeds.values()]
            if publication_input_paths is not None:
                publication_input_paths.extend(Path(path) for path in paths)
        for path in paths:
            if not path.exists():
                warnings.append(f"{branch} {subject} {recording_part}: missing participant evoked FIF {path.name}.")
                continue
            for evoked in read_evokeds(path):
                condition = label_map.get(str(getattr(evoked, "comment", "")).lower())
                if condition in evokeds_by_condition:
                    evokeds_by_condition[condition].setdefault(subject, []).append(evoked)

    grand_averages: dict[str, Any] = {}
    for condition, subject_evokeds in evokeds_by_condition.items():
        if not subject_evokeds:
            warnings.append(f"{branch}: no included participant evokeds for condition {condition}.")
            continue
        participant_evokeds = [pool_evoked_parts(parts) for parts in subject_evokeds.values()]
        grand = grand_average(participant_evokeds)
        try:
            grand.comment = condition_display_label(condition, settings, branch)
        except Exception:
            pass
        grand_averages[condition] = grand
    return grand_averages


def pool_evoked_parts(evokeds: Sequence[Any]) -> Any:
    """Combine one participant's acquisition parts by retained trial count (`nave`)."""

    part_markers = [
        retained_bad_channel_marker(evoked.info, require_bad=False)
        if hasattr(evoked, "info")
        else ()
        for evoked in evokeds
    ]
    if part_markers and any(marker != part_markers[0] for marker in part_markers[1:]):
        raise ValueError(
            "Cannot pool participant parts with inconsistent retained-bad-channel dispositions."
        )
    evokeds = _drop_retained_bad_channels(evokeds)
    if len(evokeds) == 1:
        return evokeds[0].copy()
    raw_weights = [getattr(evoked, "nave", None) for evoked in evokeds]
    if any(weight is None or float(weight) <= 0 for weight in raw_weights):
        raise ValueError("Cannot pool multiple evoked parts without positive retained-trial counts (`nave`).")
    weights = pd.Series([float(weight) for weight in raw_weights])
    total = float(weights.sum())
    pooled = evokeds[0].copy()
    pooled.data = sum(evoked.data * weight for evoked, weight in zip(evokeds, weights)) / total
    try:
        pooled.nave = int(total)
    except Exception:
        pass
    return pooled


def participant_evoked_paths(part_dir: Path, branch: str) -> list[Path]:
    base = part_dir / "erp" / branch
    paths = [base / "participant_erps-ave.fif"]
    if branch == "recognition":
        paths.append(base / "difference_wave-ave.fif")
    return paths


def read_evokeds(path: Path) -> list[Any]:
    import mne

    return list(mne.read_evokeds(path, condition=None, verbose=False))


def grand_average(evokeds: Sequence[Any]) -> Any:
    import mne

    safe_evokeds = _drop_retained_bad_channels(evokeds)
    return mne.grand_average(safe_evokeds, interpolate_bads=False, drop_bads=False)


def _drop_retained_bad_channels(evokeds: Sequence[Any]) -> list[Any]:
    """Drop only explicitly dispositioned unusable channels at the group boundary."""

    if not evokeds:
        raise ValueError("At least one evoked is required.")
    marked_by_evoked = [
        retained_bad_channel_marker(evoked.info, require_bad=False)
        if hasattr(evoked, "info")
        else ()
        for evoked in evokeds
    ]
    retained = sorted({channel for marked in marked_by_evoked for channel in marked})
    safe = []
    for evoked, marked in zip(evokeds, marked_by_evoked, strict=True):
        invalid_state = [
            channel
            for channel in marked
            if channel in getattr(evoked, "ch_names", ())
            and channel not in evoked.info.get("bads", [])
        ]
        if invalid_state:
            raise ValueError(
                "Retained-uninterpolated channel is present but no longer marked bad at the "
                f"group boundary: {invalid_state}."
            )
        copied = evoked.copy()
        present = [channel for channel in retained if channel in copied.ch_names]
        if present:
            copied.drop_channels(present)
        safe.append(copied)
    inventories = [tuple(evoked.ch_names) for evoked in safe]
    if any(inventory != inventories[0] for inventory in inventories[1:]):
        raise ValueError(
            "Group evoked channel inventories differ after explicit retained-bad-channel exclusion."
        )
    if not inventories[0]:
        raise ValueError("Group aggregation has no channels after retained-bad-channel exclusion.")
    return safe


def save_grand_average_evokeds(evokeds: Mapping[str, Any], path: Path) -> Path:
    import mne

    path.parent.mkdir(parents=True, exist_ok=True)
    mne.write_evokeds(path, list(evokeds.values()), overwrite=True)
    return path


def group_condition_order(settings: Mapping[str, Any], branch: str) -> list[str]:
    if branch == "recognition":
        diff = dict(settings.get("difference_wave", {}) or {})
        return list(configured_conditions(settings).keys()) + [str(diff.get("name", "hit_minus_correct_rejection"))]
    return list(configured_conditions(settings).keys())


def condition_label_map(settings: Mapping[str, Any], branch: str) -> dict[str, str]:
    mapping = {condition.lower(): condition for condition in group_condition_order(settings, branch)}
    for condition, spec in configured_conditions(settings).items():
        mapping[str(spec.get("label", condition)).lower()] = condition
    if branch == "recognition":
        diff = dict(settings.get("difference_wave", {}) or {})
        name = str(diff.get("name", "hit_minus_correct_rejection"))
        mapping[str(diff.get("label", "Hit - Correct Rejection")).lower()] = name
    return mapping


def condition_display_label(condition: str, settings: Mapping[str, Any], branch: str) -> str:
    if branch == "recognition":
        diff = dict(settings.get("difference_wave", {}) or {})
        if condition == str(diff.get("name", "")):
            return str(diff.get("label", condition))
    return str(configured_conditions(settings).get(condition, {}).get("label", condition))


def build_descriptive_summary(measurements: pd.DataFrame) -> pd.DataFrame:
    columns = ["condition", "roi", "time_window", "mean", "standard_deviation", "median", "minimum", "maximum", "n_participants", "n_epochs"]
    if measurements.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in measurements.groupby(["condition", "roi", "time_window"], dropna=False, sort=True):
        values = pd.to_numeric(group["mean_amplitude_uv"], errors="coerce").dropna()
        rows.append(
            {
                "condition": keys[0],
                "roi": keys[1],
                "time_window": keys[2],
                "mean": float(values.mean()) if len(values) else pd.NA,
                "standard_deviation": float(values.std()) if len(values) > 1 else 0.0 if len(values) else pd.NA,
                "median": float(values.median()) if len(values) else pd.NA,
                "minimum": float(values.min()) if len(values) else pd.NA,
                "maximum": float(values.max()) if len(values) else pd.NA,
                "n_participants": int(group["subject"].nunique()) if "subject" in group.columns else int(len(group)),
                "n_epochs": int(pd.to_numeric(group.get("n_epochs", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_group_qc_summary(branch: str, participant_summary: pd.DataFrame, measurements: pd.DataFrame, settings: Mapping[str, Any], grand_averages: Mapping[str, Any], *, expected_evoked_exclusions: Sequence[Mapping[str, str]] | None = None) -> pd.DataFrame:
    include_col = f"{branch}_included"
    reason_col = f"{branch}_reason"
    included = participant_summary[_as_bool_series(participant_summary[include_col])] if include_col in participant_summary.columns else pd.DataFrame()
    excluded = participant_summary[~_as_bool_series(participant_summary[include_col])] if include_col in participant_summary.columns else participant_summary
    observed_conditions = sorted(set(measurements["condition"].dropna().astype(str))) if "condition" in measurements.columns else []
    expected_conditions = group_condition_order(settings, branch)
    missing_conditions = [condition for condition in expected_conditions if condition not in observed_conditions and condition not in grand_averages]
    return pd.DataFrame(
        [
            {
                "branch": branch,
                "participants_included": int(len(included)),
                "participants_excluded": int(len(excluded)),
                "excluded_reasons": excluded[["subject", reason_col]].to_dict(orient="records") if reason_col in excluded.columns else [],
                "rois_analysed": sorted(set(measurements["roi"].dropna().astype(str))) if "roi" in measurements.columns else [],
                "time_windows_analysed": sorted(set(measurements["time_window"].dropna().astype(str))) if "time_window" in measurements.columns else [],
                "conditions_observed": observed_conditions,
                "conditions_missing": missing_conditions,
                "grand_average_conditions": sorted(grand_averages.keys()),
                "expected_missing_evoked_exclusions": list(expected_evoked_exclusions or []),
            }
        ]
    )


def write_group_figures(branch: str, settings: Mapping[str, Any], grand_averages: Mapping[str, Any], measurements: pd.DataFrame, participant_summary: pd.DataFrame, paths: Mapping[str, Path], warnings: list[str], *, figure_style: FigureStyle | None = None) -> dict[str, Path]:
    style = figure_style or default_figure_style()
    outputs: dict[str, Path] = {}
    if grand_averages:
        waveform = write_group_waveform_figure(branch, grand_averages, paths["waveform_figure"], figure_style=style, settings=settings)
        outputs["waveform_figure"] = waveform
        roi_waveform = write_group_roi_waveform_figure(branch, settings, grand_averages, paths["roi_waveform_figure"], figure_style=style)
        outputs["roi_waveform_figure"] = roi_waveform
        if branch == "recognition" and "hit_minus_correct_rejection" in grand_averages:
            outputs["difference_waveform_figure"] = write_difference_waveform_figure(grand_averages["hit_minus_correct_rejection"], paths["difference_waveform_figure"], figure_style=style, settings=settings)
        outputs.update(write_group_topographies(branch, settings, grand_averages, paths["topographies"], warnings, figure_style=style))
    if not measurements.empty:
        outputs["trial_count_figure"] = write_trial_count_figure(branch, measurements, paths["trial_count_figure"], figure_style=style)
    if not participant_summary.empty:
        outputs["inclusion_figure"] = write_inclusion_figure(branch, participant_summary, paths["inclusion_figure"], figure_style=style)
    return outputs


def write_group_waveform_figure(branch: str, evokeds: Mapping[str, Any], path: Path, *, figure_style: FigureStyle | None = None, settings: Mapping[str, Any] | None = None) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    style = figure_style or default_figure_style()
    fig, ax = plt.subplots(figsize=style.figure_size("waveform"), facecolor=style.figure_facecolor)
    for condition, evoked in evokeds.items():
        ax.plot(
            evoked.times * 1000.0,
            evoked.data.mean(axis=0) * 1_000_000.0,
            linewidth=style.condition_line_width(branch, condition),
            linestyle=style.condition_line_style(branch, condition),
            label=style.condition_label(condition),
            color=style.condition_color(branch, condition),
        )
    _format_time_axis(ax, f"{style.branch_label(branch)} grand-average waveforms", style)
    style_legend(ax, style)
    fig.tight_layout()
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def write_group_roi_waveform_figure(branch: str, settings: Mapping[str, Any], evokeds: Mapping[str, Any], path: Path, *, figure_style: FigureStyle | None = None) -> Path:
    import matplotlib.pyplot as plt

    safe_evokeds = plotting_safe_evokeds(evokeds, settings)
    display_evokeds = {
        condition: evoked
        for condition, evoked in safe_evokeds.items()
        if not (branch == "recognition" and condition == "hit_minus_correct_rejection")
    }
    rois = {
        roi: channels
        for roi, channels in configured_rois(settings).items()
        if any(
            any(channel in getattr(evoked, "ch_names", []) for channel in channels)
            for evoked in display_evokeds.values()
        )
    }
    if not rois:
        raise ValueError(f"{branch} ROI figure has no ROI with a plottable channel.")
    path.parent.mkdir(parents=True, exist_ok=True)
    style = figure_style or default_figure_style()
    fig, axes = plt.subplots(len(rois), 1, figsize=style.figure_size("group_roi_waveform", count=len(rois)), sharex=True, facecolor=style.figure_facecolor)
    if len(rois) == 1:
        axes = [axes]
    plotted_values: list[np.ndarray] = []
    for ax, (roi, channels) in zip(axes, rois.items()):
        for condition, evoked in display_evokeds.items():
            available = [channel for channel in channels if channel in getattr(evoked, "ch_names", [])]
            if not available:
                continue
            picks = [evoked.ch_names.index(channel) for channel in available]
            values = evoked.data[picks].mean(axis=0) * 1_000_000.0
            plotted_values.append(np.asarray(values, dtype=float))
            ax.plot(
                evoked.times * 1000.0,
                values,
                linewidth=style.condition_line_width(branch, condition, roi=True),
                linestyle=style.condition_line_style(branch, condition),
                label=style.condition_label(condition),
                color=style.condition_color(branch, condition),
            )
        _format_time_axis(ax, str(roi).replace("_", " ").title(), style)
        ax.title.set_fontsize(style.fonts["panel_title_size"])
        shade_primary_windows(ax, settings, style, roi=str(roi))
        ax.set_ylabel("µV")
        if not bool(style.legend["shared_roi"]):
            style_legend(ax, style, size=float(style.fonts["compact_legend_size"]))
    if plotted_values:
        minimum = min(0.0, min(float(np.nanmin(values)) for values in plotted_values))
        maximum = max(0.0, max(float(np.nanmax(values)) for values in plotted_values))
        span = maximum - minimum
        padding = 0.08 * span if span > 0 else 1.0
        for ax in axes:
            ax.set_ylim(minimum - padding, maximum + padding)
    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time (ms)")
    title = "Recognition ERPs by ROI" if branch == "recognition" else "Cuteness ERPs by ROI"
    fig.suptitle(title, y=0.992, fontsize=style.fonts["title_size"])
    if bool(style.legend["shared_roi"]):
        handles_by_label: dict[str, Any] = {}
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            handles_by_label.update(zip(labels, handles))
        labels = list(handles_by_label)
        handles = [handles_by_label[label] for label in labels]
        fig.legend(
            handles,
            labels,
            loc=str(style.legend["shared_location"]),
            bbox_to_anchor=tuple(style.legend["shared_bbox_to_anchor"]),
            ncol=min(int(style.legend["shared_columns"]), max(1, len(labels))),
            frameon=bool(style.legend["frameon"]),
            fontsize=style.fonts["legend_size"],
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def write_difference_waveform_figure(evoked: Any, path: Path, *, figure_style: FigureStyle | None = None, settings: Mapping[str, Any] | None = None) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    style = figure_style or default_figure_style()
    fig, ax = plt.subplots(figsize=style.figure_size("difference_waveform"), facecolor=style.figure_facecolor)
    ax.plot(evoked.times * 1000.0, evoked.data.mean(axis=0) * 1_000_000.0, color=style.color("difference_waveform"), linewidth=style.line_widths["difference_waveform"])
    _format_time_axis(ax, f"Recognition: {style.condition_label('hit_minus_correct_rejection')}", style)
    ax.title.set_fontsize(style.fonts["difference_title_size"])
    fig.tight_layout()
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def write_group_topographies(branch: str, settings: Mapping[str, Any], evokeds: Mapping[str, Any], topography_dir: Path, warnings: list[str], *, figure_style: FigureStyle | None = None) -> dict[str, Path]:
    style = figure_style or default_figure_style()
    outputs: dict[str, Path] = {}
    topography_dir.mkdir(parents=True, exist_ok=True)
    windows = configured_time_windows(settings, key="topography_time_windows")
    conditions = topography_conditions(settings, branch)
    safe_evokeds = plotting_safe_evokeds(evokeds, settings)
    limits = _topography_limits(conditions, windows, safe_evokeds, style)
    for condition in conditions:
        if condition not in safe_evokeds:
            warnings.append(f"{branch}: topography skipped for unavailable condition {condition}.")
            continue
        evoked = safe_evokeds[condition]
        for window_name, window in windows.items():
            midpoint = (float(window["start_ms"]) + float(window["end_ms"])) / 2000.0
            duration = max(0.001, (float(window["end_ms"]) - float(window["start_ms"])) / 1000.0)
            try:
                is_difference = "minus" in condition or "difference" in condition
                kwargs = {
                    "cmap": style.topography_colormap(difference=is_difference),
                    "contours": int(style.topography["contour_count"]),
                }
                if condition in limits:
                    kwargs["vlim"] = limits[condition]
                fig = evoked.plot_topomap(times=[midpoint], average=duration, show=False, time_unit="s", **kwargs)
            except TypeError:
                try:
                    fig = evoked.plot_topomap(times=[midpoint], average=duration, show=False, time_unit="s")
                except TypeError:
                    fig = evoked.plot_topomap(times=[midpoint], show=False, time_unit="s")
            if fig.axes:
                window_label = str(window.get("label", str(window_name).replace("_", " "))).replace("-", "–", 1)
                fig.axes[0].set_title("")
                fig.set_size_inches(*style.figure_size("topography"), forward=True)
                fig.suptitle(
                    f"{style.condition_label(condition)} · {window_label}",
                    y=0.98,
                    fontsize=style.fonts["compact_legend_size"],
                )
            style_topography_figure(fig, style)
            path = topography_dir / f"{condition}_{window_name}.png"
            fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass
            outputs[f"topography_{condition}_{window_name}"] = path
    return outputs


def topography_conditions(settings: Mapping[str, Any], branch: str) -> list[str]:
    if branch == "recognition":
        diff = str(dict(settings.get("difference_wave", {}) or {}).get("name", "hit_minus_correct_rejection"))
        return ["hit", "correct_rejection", diff]
    configured = [str(item) for item in settings.get("topography_conditions", []) or []]
    return configured or list(configured_conditions(settings).keys())


def _topography_limits(
    conditions: Sequence[str],
    windows: Mapping[str, Mapping[str, Any]],
    evokeds: Mapping[str, Any],
    style: FigureStyle,
) -> dict[str, tuple[float, float]]:
    """Return presentation-only signed-voltage limits without changing data."""

    if not bool(style.topography["symmetric_voltage_limits"]):
        return {}
    maxima = {
        condition: _topography_abs_max(evokeds[condition], windows)
        for condition in conditions
        if condition in evokeds
    }
    difference_conditions = [condition for condition in maxima if "minus" in condition or "difference" in condition]
    ordinary_conditions = [condition for condition in maxima if condition not in difference_conditions]
    limits: dict[str, tuple[float, float]] = {}
    if ordinary_conditions:
        common = max(maxima[condition] for condition in ordinary_conditions)
        for condition in ordinary_conditions:
            limit = common if bool(style.topography["common_condition_scale"]) else maxima[condition]
            if limit > 0:
                limits[condition] = (-limit, limit)
    for condition in difference_conditions:
        limit = maxima[condition]
        if limit > 0 and bool(style.topography["center_difference_at_zero"]):
            limits[condition] = (-limit, limit)
    return limits


def _topography_abs_max(evoked: Any, windows: Mapping[str, Mapping[str, Any]]) -> float:
    maximum = 0.0
    times = np.asarray(evoked.times, dtype=float)
    data = np.asarray(evoked.data, dtype=float)
    for window in windows.values():
        start = float(window["start_ms"]) / 1000.0
        end = float(window["end_ms"]) / 1000.0
        mask = (times >= start) & (times <= end)
        if mask.any():
            maximum = max(maximum, float(np.nanmax(np.abs(data[:, mask].mean(axis=1)))) * 1_000_000.0)
    return maximum


def write_trial_count_figure(branch: str, measurements: pd.DataFrame, path: Path, *, figure_style: FigureStyle | None = None) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    counts = measurements.groupby("condition", dropna=False)["n_epochs"].sum().reset_index() if "n_epochs" in measurements.columns else measurements.groupby("condition", dropna=False).size().reset_index(name="n_epochs")
    style = figure_style or default_figure_style()
    fig, ax = plt.subplots(figsize=style.figure_size("trial_count"), facecolor=style.figure_facecolor)
    labels = [style.condition_label(condition) for condition in counts["condition"].astype(str)]
    ax.bar(labels, pd.to_numeric(counts["n_epochs"], errors="coerce").fillna(0), color=style.bar_color())
    ax.set_title(f"{style.branch_label(branch)}: retained trials")
    ax.set_ylabel("Epochs")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    style_axes(ax, style)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def write_inclusion_figure(branch: str, participant_summary: pd.DataFrame, path: Path, *, figure_style: FigureStyle | None = None) -> Path:
    import matplotlib.pyplot as plt

    include_col = f"{branch}_included"
    included = int(_as_bool_series(participant_summary[include_col]).sum()) if include_col in participant_summary.columns else 0
    excluded = max(0, len(participant_summary) - included)
    path.parent.mkdir(parents=True, exist_ok=True)
    style = figure_style or default_figure_style()
    fig, ax = plt.subplots(figsize=style.figure_size("inclusion"), facecolor=style.figure_facecolor)
    ax.bar(["Included", "Excluded"], [included, excluded], color=[style.bar_color("included"), style.bar_color("excluded")])
    ax.set_title(f"{style.branch_label(branch)}: participant inclusion")
    ax.set_ylabel("Participants")
    fig.tight_layout()
    style_axes(ax, style)
    fig.savefig(path, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)
    return path


def _format_time_axis(ax, title: str, style: FigureStyle | None = None) -> None:
    style = style or default_figure_style()
    alpha = float(style.axes["reference_alpha"])
    ax.axhline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=alpha)
    ax.axvline(0.0, color=style.color("reference"), linewidth=style.line_widths["reference"], alpha=alpha)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)
    style_axes(ax, style)


def write_group_report(branch: str, path: Path, output_files: Mapping[str, Path], warnings: Sequence[str], qc: pd.DataFrame, *, info_messages: Sequence[str] = ()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Gate 5B {branch.title()} Descriptive Group ERP",
        "",
        "## Scope",
        "- Descriptive group-level ERP outputs only.",
        "- No inferential statistics, p-values, confidence intervals, effect sizes, correlations, regressions, or hypothesis tests were run.",
        "",
        "## QC",
    ]
    if not qc.empty:
        row = qc.iloc[0].to_dict()
        lines.extend(f"- {key}: {value}" for key, value in row.items())
    lines.extend(["", "## Outputs"])
    lines.extend(f"- {label}: `{path_value}`" for label, path_value in output_files.items())
    lines.extend(["", "## Info / QC Exclusions"])
    branch_infos = [info for info in info_messages if info.startswith(branch)]
    lines.extend(f"- {info}" for info in branch_infos) if branch_infos else lines.append("- none")
    lines.extend(["", "## Warnings"])
    branch_warnings = [warning for warning in warnings if warning.startswith(branch)]
    lines.extend(f"- {warning}" for warning in branch_warnings) if branch_warnings else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _as_bool_series(series: pd.Series) -> pd.Series:
    """Parse booleans robustly from CSV values without treating 'False' as truthy."""

    return series.map(_as_bool).astype(bool)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "included"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _write_gate5b_provenance(
    config: Mapping[str, Any],
    branch: str,
    output_files: Mapping[str, Path],
    input_files: Mapping[str, Path],
    participant_summary: pd.DataFrame,
    included_rows: pd.DataFrame,
    warnings: Sequence[str],
    *,
    root: Path | None,
    config_path: str | Path | None = None,
    published_participant_inputs: Sequence[Path] | None = None,
) -> None:
    included, excluded, reasons = _summary_subject_sets(participant_summary, branch)
    inputs = [path for path in input_files.values() if path.exists()]
    if published_participant_inputs is not None:
        inputs.extend(published_participant_inputs)
    else:
        for _, row in included_rows.iterrows():
            part_dir = subject_part_dir(config, str(row["subject"]), str(row["recording_part"]), root=root, create=False)
            inputs.extend(path for path in participant_evoked_paths(part_dir, branch) if path.exists())
    inputs = list(dict.fromkeys(inputs))
    for output in output_files.values():
        write_group_provenance_sidecar(
            output,
            gate="gate5b",
            branch=branch,
            root=root,
            input_files=inputs,
            included_subjects=included,
            excluded_subjects=excluded,
            exclusion_reasons=reasons,
            parameters_summary={"erp_branch_settings": dict(erp_branch_settings(config, branch))},
            warnings=warnings,
            **({"config_path": config_path} if config_path is not None else {}),
        )


def _summary_subject_sets(participant_summary: pd.DataFrame, branch: str) -> tuple[list[str], list[str], dict[str, str]]:
    include_col = f"{branch}_included"
    reason_col = f"{branch}_reason"
    if participant_summary.empty or "subject" not in participant_summary.columns or include_col not in participant_summary.columns:
        return [], [], {}
    included_mask = _as_bool_series(participant_summary[include_col])
    included = sorted(participant_summary.loc[included_mask, "subject"].dropna().astype(str).unique().tolist())
    excluded = sorted(participant_summary.loc[~included_mask, "subject"].dropna().astype(str).unique().tolist())
    reasons = {}
    if reason_col in participant_summary.columns:
        for _, row in participant_summary.loc[~included_mask].iterrows():
            subject = str(row.get("subject", ""))
            if subject:
                reasons[subject] = str(row.get(reason_col, ""))
    return included, excluded, reasons
