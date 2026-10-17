from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .provenance import write_provenance_sidecar


def figure_filename(subject: str, recording_part: str, figure_name: str, suffix: str = ".png") -> str:
    """Return a consistent Gate 3A figure filename."""

    clean = str(figure_name).strip().replace(" ", "_").lower()
    return f"{subject}_{recording_part}_{clean}{suffix}"


def save_figure(
    fig,
    path: str | Path,
    *,
    close: bool = True,
    dpi: int = 160,
    facecolor: str | None = None,
) -> Path:
    """Save a matplotlib figure with consistent defaults."""

    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", **({"facecolor": facecolor} if facecolor else {}))
    if close:
        plt.close(fig)
    return output


def save_figure_with_provenance(
    fig,
    path: str | Path,
    *,
    subject: str,
    recording_part: str,
    root: str | Path | None,
    config_path: str | Path | None,
    input_files: Sequence[str | Path],
    parameters_summary: Mapping[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
) -> Path:
    """Save a figure and attach a provenance sidecar."""

    output = save_figure(fig, path)
    write_provenance_sidecar(
        output,
        pipeline_stage="gate3a_qc_figure",
        subject=subject,
        recording_part=recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary=parameters_summary or {},
        warnings=warnings or [],
    )
    return output


def save_figure_with_stage_provenance(
    fig,
    path: str | Path,
    *,
    pipeline_stage: str,
    subject: str,
    recording_part: str,
    root: str | Path | None,
    config_path: str | Path | None,
    input_files: Sequence[str | Path],
    parameters_summary: Mapping[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
    dpi: int = 160,
    facecolor: str | None = None,
) -> Path:
    """Save a figure and attach a provenance sidecar for a named stage."""

    output = save_figure(fig, path, dpi=dpi, facecolor=facecolor)
    write_provenance_sidecar(
        output,
        pipeline_stage=pipeline_stage,
        subject=subject,
        recording_part=recording_part,
        root=root,
        config_path=config_path,
        input_files=input_files,
        parameters_summary=parameters_summary or {},
        warnings=warnings or [],
    )
    return output


def apply_common_formatting(ax, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    """Apply lightweight common formatting to reusable EEG QC axes."""

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return ax


def plot_psd(raw, *, title: str = "PSD", fmin: float = 1.0, fmax: float = 80.0, start: float = 0.0, duration: float | None = 120.0):
    """Create an interpretable EEG median/mean PSD plot."""

    import matplotlib.pyplot as plt
    import numpy as np

    tmin = float(start)
    tmax = None if duration is None else max(tmin, tmin + float(duration))
    if tmax is not None:
        tmax = min(tmax, max(0.0, raw.times[-1]))
    psd = raw.compute_psd(picks="eeg", fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, verbose=False)
    freqs = psd.freqs
    data = psd.get_data()
    data_db = 10.0 * np.log10(np.maximum(data, np.finfo(float).tiny))
    median = np.nanmedian(data_db, axis=0)
    mean = np.nanmean(data_db, axis=0)
    lo, hi = np.nanpercentile(data_db, [10, 90], axis=0)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(freqs, lo, hi, color="#9ecae1", alpha=0.35, label="10-90% EEG channels")
    ax.plot(freqs, median, color="#1f77b4", linewidth=1.8, label="median")
    ax.plot(freqs, mean, color="#f28e2b", linewidth=1.2, label="mean")
    finite = data_db[np.isfinite(data_db)]
    if finite.size:
        ymin, ymax = np.nanpercentile(finite, [2, 98])
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
            ax.set_ylim(ymin, ymax)
    ax.legend(loc="best")
    apply_common_formatting(ax, title=title, xlabel="Frequency (Hz)", ylabel="Power spectral density (dB)")
    return fig


def plot_raw_overview(
    raw,
    *,
    title: str = "Raw EEG overview",
    duration: float = 20.0,
    start: float = 0.0,
    eeg_scaling_uv: float = 100.0,
):
    """Create a fixed-scale, EEG-only stacked overview for reproducible review.

    MNE's ``scalings='auto'`` estimates its scale from a separate data subset,
    which can make a requested short view unusable when that subset contains a
    large transient.  A fixed display scale also makes participant figures
    directly comparable.  This affects the browser rendering only: the Raw
    object and scientific data remain unchanged.
    """

    if float(eeg_scaling_uv) <= 0:
        raise ValueError("eeg_scaling_uv must be positive.")
    import matplotlib.pyplot as plt
    import numpy as np

    view = raw.copy().crop(
        tmin=float(start),
        tmax=min(float(start) + float(duration), raw.times[-1]),
        include_tmax=True,
    ).pick("eeg")
    data_uv = view.get_data() * 1e6
    data_uv -= np.nanmean(data_uv, axis=1, keepdims=True)
    time_s = float(start) + np.arange(data_uv.shape[1]) / float(view.info["sfreq"])
    panel_count = 2 if len(view.ch_names) > 30 else 1
    channel_indices = [indices for indices in np.array_split(np.arange(len(view.ch_names)), panel_count) if len(indices)]
    fig, axes = plt.subplots(1, len(channel_indices), figsize=(15 if len(channel_indices) > 1 else 10, max(7.0, 0.28 * max(map(len, channel_indices)))))
    axes = np.atleast_1d(axes)
    for ax, indices in zip(axes, channel_indices):
        offsets = np.arange(len(indices), dtype=float)
        for offset, channel_index in zip(offsets, indices):
            # One row is a 200-uV peak-to-peak display band. Values are not
            # clipped: large artifacts visibly cross neighbouring rows.
            ax.plot(time_s, offset - data_uv[channel_index] / (2.0 * float(eeg_scaling_uv)), color="black", linewidth=0.45)
        ax.set_yticks(offsets)
        ax.set_yticklabels([view.ch_names[index] for index in indices])
        ax.set_ylim(len(indices) - 0.25, -0.75)
        ax.set_xlim(time_s[0], time_s[-1])
        apply_common_formatting(ax, xlabel="Time (s)", ylabel="EEG channel")
    fig.suptitle(f"{title} - EEG only, fixed +/-{float(eeg_scaling_uv):g} uV half-row display")
    fig.text(0.5, 0.01, "Display mean removed per channel; no filtering or amplitude clipping; large excursions cross row boundaries.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def plot_channel_variance(channel_qc: pd.DataFrame, *, title: str = "Channel variance"):
    """Plot a histogram of channel standard deviations."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    df = channel_qc.copy()
    if "included_in_eeg_qc" in df:
        df = df[df["included_in_eeg_qc"].astype(bool)]
    values = pd.to_numeric(df.get("std_uv", pd.Series(dtype=float)), errors="coerce").dropna()
    ax.hist(values, bins=30, color="#4C78A8", edgecolor="white")
    apply_common_formatting(ax, title=title, xlabel="EEG QC channel std (uV)", ylabel="Count")
    return fig


def plot_line_noise(line_noise: pd.DataFrame, *, title: str = "Line-noise ratio"):
    """Plot line-noise ratio by channel."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not line_noise.empty:
        df = line_noise.copy()
        df["line_noise_ratio"] = pd.to_numeric(df["line_noise_ratio"], errors="coerce")
        ax.bar(df["channel"].astype(str), df["line_noise_ratio"], color="#F58518")
        ax.tick_params(axis="x", rotation=90)
    ax.text(0.01, 0.96, "Proposal-level QC only", transform=ax.transAxes, va="top", fontsize=9)
    apply_common_formatting(ax, title=title, xlabel="Channel", ylabel="Line/broadband power ratio")
    return fig


def plot_bad_channel_summary(
    channel_qc: pd.DataFrame,
    *,
    bad_channel_decisions: pd.DataFrame | None = None,
    title: str = "EEG channel standard deviation with proposal status",
):
    """Plot channel SD while clearly separating proposals and confirmations."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12, 5.6))
    if not channel_qc.empty:
        df = channel_qc.copy()
        if "included_in_eeg_qc" in df:
            df = df[df["included_in_eeg_qc"].astype(bool)]
        if bad_channel_decisions is not None and not bad_channel_decisions.empty:
            status = bad_channel_decisions[["channel", "manual_confirmed_bad_channel"]].copy()
            df = df.merge(status, on="channel", how="left")
        if "manual_confirmed_bad_channel" not in df:
            df["manual_confirmed_bad_channel"] = False
        df["manual_confirmed_bad_channel"] = df["manual_confirmed_bad_channel"].fillna(False).astype(bool)
        df["proposed_bad_channel"] = df["proposed_bad_channel"].astype(bool)
        if "automatic_review_priority" not in df:
            df["automatic_review_priority"] = np.where(df["proposed_bad_channel"], "REVIEW_SUGGESTED", "NO_AUTOMATIC_FLAG")
        colours = df["automatic_review_priority"].map(
            {"REVIEW_SUGGESTED": "#E45756", "METRIC_FLAG_ONLY": "#F2CF5B", "NO_AUTOMATIC_FLAG": "#54A24B"}
        ).fillna("#54A24B")
        bars = ax.bar(df["channel"].astype(str), pd.to_numeric(df["std_uv"], errors="coerce"), color=colours)
        for bar, confirmed in zip(bars, df["manual_confirmed_bad_channel"]):
            if confirmed:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
                bar.set_linewidth(1.2)
        extreme_threshold = pd.to_numeric(df.get("extreme_std_uv_threshold"), errors="coerce").dropna()
        if len(extreme_threshold) and float(extreme_threshold.iloc[0]) <= max(1.0, float(pd.to_numeric(df["std_uv"], errors="coerce").max()) * 1.15):
            ax.axhline(float(extreme_threshold.iloc[0]), color="#7A5195", linestyle="--", linewidth=1.2, label="Extreme-SD proposal threshold")
        reason_codes = {
            "flat": "F",
            "extreme_amplitude": "A",
            "clipping": "C",
            "poor_correlation": "R",
            "robust_relative_deviation": "D",
        }
        for bar, reasons in zip(bars, df.get("automatic_metric_flag_reasons", df.get("automatic_proposal_reasons", pd.Series("", index=df.index)))):
            codes = "/".join(reason_codes.get(reason, reason[:1].upper()) for reason in str(reasons).split(";") if reason)
            if codes:
                ax.annotate(codes, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6)
        ax.tick_params(axis="x", rotation=90)
        handles = [
            Patch(facecolor="#54A24B", label="Not automatically proposed"),
            Patch(facecolor="#F2CF5B", label="Metric flag only"),
            Patch(facecolor="#E45756", label="Automatically proposed (not confirmed)"),
            Patch(facecolor="white", edgecolor="black", hatch="///", label="Manually confirmed overlay"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.text(
            0.01,
            0.99,
            "Bar height = SD only. Codes: F flat, A extreme amplitude, C clipping, R poor correlation, D robust deviation.",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    apply_common_formatting(ax, title=title, xlabel="EEG QC channel", ylabel="Standard deviation (uV)")
    fig.tight_layout()
    return fig


def plot_correlation_qc_summary(channel_qc: pd.DataFrame, *, title: str = "Poor-correlation metric evidence"):
    """Show per-window channel correlation scores as metric-only evidence."""

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 5.8))
    if not channel_qc.empty:
        df = channel_qc.copy()
        if "included_in_eeg_qc" in df:
            df = df[df["included_in_eeg_qc"].astype(bool)]
        channels = df["channel"].astype(str).tolist()
        minimum = pd.to_numeric(df["minimum_window_p98_abs_correlation"], errors="coerce").to_numpy()
        median = pd.to_numeric(df["median_window_p98_abs_correlation"], errors="coerce").to_numpy()
        threshold_values = pd.to_numeric(df["poor_correlation_threshold"], errors="coerce").dropna()
        threshold = float(threshold_values.iloc[0]) if len(threshold_values) else 0.4
        x = np.arange(len(channels))
        proposed = df["poor_correlation_proposal"].astype(bool).to_numpy()
        for index in np.flatnonzero(proposed):
            ax.axvspan(index - 0.45, index + 0.45, color="#F2CF5B", alpha=0.12, linewidth=0)
        ax.plot(x, median, marker="o", markersize=3, linewidth=1.0, color="#4C78A8", label="Median across 1 s windows")
        ax.plot(x, minimum, marker=".", markersize=3, linewidth=0.8, color="#9ECAE1", label="Minimum 1 s window")
        failed = np.isfinite(minimum) & (minimum < threshold)
        ax.scatter(x[failed], minimum[failed], marker="x", s=24, linewidth=1.1, color="#B22222", zorder=4)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Metric threshold ({threshold:g})")
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=90)
        finite_values = np.concatenate([minimum[np.isfinite(minimum)], median[np.isfinite(median)]])
        ax.set_ylim(0.0, max(1.0, float(np.max(finite_values)) * 1.05) if finite_values.size else 1.0)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.text(0.01, 0.99, "Red x = failed window; pale yellow band = poor-correlation metric flag (not a review-shortlist decision)", transform=ax.transAxes, va="top", fontsize=8)
    apply_common_formatting(ax, title=title, xlabel="EEG QC channel", ylabel="98th-percentile absolute peer correlation (1 s, QC-only 1-50 Hz)")
    fig.tight_layout()
    return fig


def plot_montage(raw, *, title: str = "Montage"):
    """Create a montage/sensor plot."""

    fig = raw.plot_sensors(show_names=True, show=False)
    fig.suptitle(title)
    return fig


def gate3a_figure_paths(base_dir: str | Path, subject: str, recording_part: str) -> dict[str, Path]:
    """Return standard Gate 3A QC figure output paths."""

    directory = Path(base_dir)
    return {
        "psd_figure": directory / "psd.png",
        "raw_overview_figure": directory / "raw_overview.png",
        "channel_variance_figure": directory / "channel_variance.png",
        "line_noise_figure": directory / "line_noise.png",
        "montage_figure": directory / "montage.png",
        "bad_channel_figure": directory / "bad_channels.png",
        "correlation_qc_figure": directory / "poor_correlation_qc.png",
    }


def gate3b_ica_figure_paths(base_dir: str | Path, subject: str, recording_part: str) -> dict[str, Path]:
    """Return standard Gate 3B ICA diagnostic figure output paths."""

    directory = Path(base_dir)
    return {
        "component_maps_figure": directory / "component_maps.png",
        "component_sources_figure": directory / "component_time_courses.png",
        "eog_scores_figure": directory / "eog_scores.png",
        "pca_explained_variance_figure": directory / "pca_explained_variance.png",
    }


def gate3b_ica_component_review_paths(
    base_dir: str | Path,
    subject: str,
    recording_part: str,
    n_components: int,
    *,
    page_size: int = 10,
) -> dict[str, list[Path]]:
    """Return paginated component-map/source paths covering every fitted IC."""

    if n_components < 0 or page_size <= 0:
        raise ValueError("n_components must be non-negative and page_size must be positive.")
    n_pages = max(1, (n_components + page_size - 1) // page_size)
    directory = Path(base_dir)
    return {
        "component_maps_pages": [directory / f"component_maps_page_{page:02d}.png" for page in range(1, n_pages + 1)],
        "component_sources_pages": [directory / f"component_sources_page_{page:02d}.png" for page in range(1, n_pages + 1)],
        "component_spectra_pages": [directory / f"component_spectra_page_{page:02d}.png" for page in range(1, n_pages + 1)],
    }


def plot_ica_component_topographies(ica, raw, *, picks: Sequence[int], title: str):
    """Plot ICA maps using only sensors with finite, unique geometry."""

    import matplotlib.pyplot as plt
    import mne
    import numpy as np

    maps = np.asarray(ica.get_components(), dtype=float)
    raw_indices = {name: index for index, name in enumerate(raw.ch_names)}
    usable_rows, positions, seen = [], [], set()
    for row, name in enumerate(ica.ch_names):
        if name not in raw_indices:
            continue
        position = np.asarray(raw.info["chs"][raw_indices[name]]["loc"][:2], dtype=float)
        key = tuple(np.round(position, 12))
        if not np.isfinite(position).all() or np.allclose(position, 0.0) or key in seen:
            continue
        seen.add(key); usable_rows.append(row); positions.append(position)
    if len(usable_rows) < 3:
        raise ValueError("Fewer than three ICA sensors have finite, unique topographic coordinates.")
    n_columns = min(5, max(1, len(picks)))
    n_rows = int(np.ceil(len(picks) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.5 * n_columns, 2.7 * n_rows), squeeze=False)
    for axis, component in zip(axes.flat, picks):
        values = maps[usable_rows, int(component)]
        limit = float(np.nanmax(np.abs(values))) or 1.0
        mne.viz.plot_topomap(values, np.asarray(positions), axes=axis, show=False, contours=0, vlim=(-limit, limit))
        axis.set_title(f"ICA{int(component):03d}")
    for axis in axes.flat[len(picks):]:
        axis.set_visible(False)
    fig.suptitle(title + " (unresolved/duplicate sensor positions omitted)")
    fig.tight_layout()
    return fig


def plot_ica_component_spectra(ica, raw, *, picks: Sequence[int], title: str, fmax: float = 60.0):
    """Plot deterministic Welch spectra for selected ICA source time series."""

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import welch

    sources = np.asarray(ica.get_sources(raw).get_data(picks=list(picks)), dtype=float)
    sfreq = float(raw.info["sfreq"])
    nperseg = min(sources.shape[1], max(256, int(round(4.0 * sfreq))))
    frequencies, power = welch(sources, fs=sfreq, nperseg=nperseg, axis=1)
    keep = frequencies <= min(float(fmax), sfreq / 2.0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row, component in enumerate(picks):
        db = 10.0 * np.log10(np.maximum(power[row, keep], np.finfo(float).tiny))
        ax.plot(frequencies[keep], db, linewidth=1.0, label=f"ICA{int(component):03d}")
    ax.legend(ncol=2, fontsize=8)
    apply_common_formatting(ax, title=title, xlabel="Frequency (Hz)", ylabel="Source PSD (dB/Hz)")
    fig.tight_layout()
    return fig


def plot_ica_eog_scores(eog_suggestions: pd.DataFrame, *, title: str = "ICA EOG correlation scores"):
    """Plot ocular-correlation scores by ICA component."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not eog_suggestions.empty:
        df = eog_suggestions.copy()
        df["score_abs"] = pd.to_numeric(df.get("score_abs", pd.Series(dtype=float)), errors="coerce")
        labels = df["component"].astype(str) + " " + df["eog_channel"].astype(str)
        colours = df["suggested_ocular_component"].astype(bool).map({True: "#E45756", False: "#4C78A8"})
        ax.bar(labels, df["score_abs"], color=colours)
        ax.tick_params(axis="x", rotation=90)
    apply_common_formatting(ax, title=title, xlabel="Component / EOG channel", ylabel="Absolute correlation score")
    return fig


def plot_pca_explained_variance(pca_components: pd.DataFrame, *, title: str = "ICA PCA explained variance"):
    """Plot PCA explained variance ratios available from the fitted ICA object."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not pca_components.empty:
        df = pca_components.copy()
        df["explained_variance_ratio"] = pd.to_numeric(df.get("explained_variance_ratio", pd.Series(dtype=float)), errors="coerce")
        ax.bar(df["pca_component"].astype(str), df["explained_variance_ratio"], color="#54A24B")
    apply_common_formatting(ax, title=title, xlabel="PCA component", ylabel="Explained variance ratio")
    return fig


def gate3c_epoch_figure_paths(base_dir: str | Path, subject: str, recording_part: str) -> dict[str, Path]:
    """Return standard Gate 3C epoch QC figure output paths."""

    directory = Path(base_dir)
    return {
        "epoch_count_figure": directory / "epoch_counts.png",
        "epoch_rejection_figure": directory / "epoch_rejections.png",
        "erp_preview_figure": directory / "erp_preview_old_new.png",
        "ica_before_after_figure": directory / "ica_before_after.png",
        "interpolation_summary_figure": directory / "bad_channel_interpolation.png",
    }


def plot_epoch_counts(condition_counts: pd.DataFrame, *, title: str = "Epoch counts"):
    """Plot retained/rejected epoch counts by condition."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not condition_counts.empty:
        df = condition_counts.copy()
        labels = df["condition"].astype(str)
        retained = pd.to_numeric(df.get("retained_epochs", pd.Series(dtype=float)), errors="coerce").fillna(0)
        rejected = pd.to_numeric(df.get("rejected_epochs", pd.Series(dtype=float)), errors="coerce").fillna(0)
        ax.bar(labels, retained, label="retained", color="#54A24B")
        ax.bar(labels, rejected, bottom=retained, label="rejected", color="#E45756")
        ax.legend()
    apply_common_formatting(ax, title=title, xlabel="Condition", ylabel="Epochs")
    return fig


def plot_epoch_rejection_summary(rejection_table: pd.DataFrame, *, title: str = "Epoch rejection reasons"):
    """Plot rejected epoch counts by rejection reason."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not rejection_table.empty and "rejection_reason" in rejection_table:
        reasons = rejection_table.loc[rejection_table["rejected"].astype(bool), "rejection_reason"].fillna("unknown")
        counts = reasons.value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#E45756")
        ax.tick_params(axis="x", rotation=45)
    apply_common_formatting(ax, title=title, xlabel="Reason", ylabel="Rejected epochs")
    return fig


def plot_interpolation_summary(interpolation_table: pd.DataFrame, *, title: str = "Bad-channel interpolation"):
    """Plot manually confirmed bad channels selected for interpolation."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.8))
    if not interpolation_table.empty:
        df = interpolation_table.copy()
        values = df["interpolated"].astype(bool).map({True: 1, False: 0})
        ax.bar(df["channel"].astype(str), values, color=values.map({1: "#F58518", 0: "#4C78A8"}))
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylim(0, 1.2)
    apply_common_formatting(ax, title=title, xlabel="Channel", ylabel="Interpolated")
    return fig


def plot_erp_preview(epochs, *, title: str = "ERP preview old/new"):
    """Create a descriptive old/new ERP preview from image-onset epochs."""

    import mne

    evokeds = {}
    if getattr(epochs, "metadata", None) is not None and "condition" in epochs.metadata:
        for condition in ("old", "new"):
            indices = epochs.metadata.index[epochs.metadata["condition"].astype(str).eq(condition)].tolist()
            if indices:
                evokeds[condition] = epochs[indices].average()
    if not evokeds:
        evokeds["all"] = epochs.average()
    figures = mne.viz.plot_compare_evokeds(evokeds, picks="eeg", show=False, combine="mean")
    return figures[0] if isinstance(figures, list) else figures


def plot_ica_before_after(raw_before, raw_after, *, title: str = "Before/after ICA", start: float = 0.0, duration: float | None = 20.0):
    """Plot a compact before/after ICA PSD comparison."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    tmin = float(start)
    tmax = None if duration is None else tmin + float(duration)
    for raw, label, colour in ((raw_before, "before", "#4C78A8"), (raw_after, "after", "#E45756")):
        raw_tmax = float(raw.times[-1]) if len(raw.times) else tmin
        psd_tmax = None if tmax is None else min(tmax, raw_tmax)
        psd = raw.compute_psd(picks="eeg", tmin=tmin, tmax=psd_tmax, verbose=False)
        freqs = psd.freqs
        data = psd.get_data().mean(axis=0)
        ax.plot(freqs, data, label=label, color=colour)
    ax.legend()
    apply_common_formatting(ax, title=title, xlabel="Frequency (Hz)", ylabel="Mean PSD")
    return fig
