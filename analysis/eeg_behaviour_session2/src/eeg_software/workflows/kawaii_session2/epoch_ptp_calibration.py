"""Descriptive residual epoch PTP calibration for Phase 2A-4a.

This module is intentionally separate from the production preprocessing path.
It reads frozen Gate 3C epochs and writes decision-support artifacts only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPECTED_TMIN_S = -0.2
EXPECTED_TMAX_S = 1.2
EXPECTED_BASELINE_S = (-0.2, 0.0)
INITIAL_CALIBRATION_PARTICIPANTS = ("sub005", "sub006", "sub007")


def max_eeg_ptp_uv(data_v: np.ndarray, channel_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Return maximum channel PTP per epoch in µV and the producing channel."""

    values = np.asarray(data_v, dtype=float)
    if values.ndim != 3:
        raise ValueError("Epoch data must have shape (epochs, channels, samples).")
    if values.shape[1] != len(channel_names):
        raise ValueError("Channel-name count does not match epoch data.")
    if not len(channel_names):
        raise ValueError("At least one EEG channel is required.")
    if not np.isfinite(values).all():
        raise ValueError("Epoch EEG data must be finite before PTP calculation.")
    channel_ptp_uv = np.ptp(values, axis=2) * 1e6
    maximum_indices = np.argmax(channel_ptp_uv, axis=1)
    maxima = channel_ptp_uv[np.arange(values.shape[0]), maximum_indices]
    return maxima, [str(channel_names[index]) for index in maximum_indices]


def distribution_summary(values_uv: Sequence[float]) -> dict[str, float | int]:
    """Return the prespecified descriptive PTP distribution summary."""

    values = np.asarray(values_uv, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("PTP values must be a non-empty finite one-dimensional array.")
    median = float(np.median(values))
    return {
        "n_epochs": int(values.size),
        "minimum_uv": float(np.min(values)),
        "median_uv": median,
        "mad_uv": float(np.median(np.abs(values - median))),
        "p90_uv": float(np.quantile(values, 0.90)),
        "p95_uv": float(np.quantile(values, 0.95)),
        "p97_5_uv": float(np.quantile(values, 0.975)),
        "p99_uv": float(np.quantile(values, 0.99)),
        "maximum_uv": float(np.max(values)),
    }


def candidate_thresholds(values_uv: Sequence[float]) -> pd.DataFrame:
    """Return descriptive candidates without selecting a final threshold."""

    values = np.asarray(values_uv, dtype=float)
    summary = distribution_summary(values)
    robust_sigma = 1.4826 * float(summary["mad_uv"])
    candidates = [
        ("pooled_p97_5", float(summary["p97_5_uv"]), "Pooled 97.5th percentile"),
        ("pooled_p99", float(summary["p99_uv"]), "Pooled 99th percentile"),
        (
            "pooled_median_plus_6_robust_sigma",
            float(summary["median_uv"]) + 6.0 * robust_sigma,
            "Pooled median + 6 × (1.4826 × MAD)",
        ),
        (
            "reference_330_uv",
            330.0,
            "Reference candidate from the initial calibration discussion; not selected",
        ),
    ]
    return pd.DataFrame(candidates, columns=["candidate", "threshold_uv", "derivation"])


def threshold_impacts(metrics: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize pooled and participant impacts for each candidate."""

    pooled_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        threshold = float(candidate["threshold_uv"])
        rejected = metrics["ptp_max_uv"].astype(float) > threshold
        pooled_rows.append(
            {
                **candidate,
                "total_epochs": int(len(metrics)),
                "epochs_above_threshold": int(rejected.sum()),
                "percent_above_threshold": float(100.0 * rejected.mean()),
                "epochs_retained": int((~rejected).sum()),
                "percent_retained": float(100.0 * (~rejected).mean()),
            }
        )
        for participant, group in metrics.assign(_rejected=rejected).groupby("participant", sort=True):
            n_rejected = int(group["_rejected"].sum())
            participant_rows.append(
                {
                    "candidate": candidate["candidate"],
                    "threshold_uv": threshold,
                    "participant": participant,
                    "total_epochs": int(len(group)),
                    "epochs_above_threshold": n_rejected,
                    "percent_above_threshold": float(100.0 * n_rejected / len(group)),
                    "epochs_retained": int(len(group) - n_rejected),
                    "percent_retained": float(100.0 * (len(group) - n_rejected) / len(group)),
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(participant_rows)


def participant_inventory(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Inventory current Session 2 data without inspecting downstream results."""

    participant_cfg = dict(config.get("participants", {}) or {})
    defaults = dict(participant_cfg.get("defaults", {}) or {})
    analysis = dict(config.get("analysis", {}) or {})
    default_excluded = {str(value) for value in defaults.get("excluded", []) or []}
    analysis_excluded = {str(value) for value in analysis.get("exclude_subjects", []) or []}
    exclusion_reasons = {str(key): str(value) for key, value in dict(analysis.get("exclusion_reasons", {}) or {}).items()}

    participant_ids: set[str] = set()
    log_root = root / "workspace" / "kawaii" / "session2" / "source" / "behavioral_logs"
    log_dirs_by_participant: dict[str, list[Path]] = {}
    if log_root.exists():
        for path in log_root.iterdir():
            match = re.fullmatch(r"((?:sub\d{3})|(?:pilot\d{2}))(?:_[^\\]*)?_session2", path.name)
            if path.is_dir() and match:
                participant_ids.add(match.group(1))
                log_dirs_by_participant.setdefault(match.group(1), []).append(path)

    raw_root = root / "workspace" / "kawaii" / "session2" / "source" / "eeg"
    pilot_recordings = dict(dict(config.get("recordings", {}) or {}).get("participant_recordings", {}) or {})
    raw_to_pilot: dict[str, str] = {}
    for pilot, parts in pilot_recordings.items():
        for part in dict(parts or {}).values():
            raw_to_pilot[str(dict(part or {}).get("brainvision_file_base", ""))] = str(pilot)
    for pilot in default_excluded:
        match = re.fullmatch(r"pilot(\d{2})", pilot)
        if match:
            raw_to_pilot.setdefault(f"kawaii{int(match.group(1)):04d}", pilot)
    if raw_root.exists():
        for path in raw_root.glob("kawaii*.vhdr"):
            base = path.stem
            if base in raw_to_pilot:
                participant_ids.add(raw_to_pilot[base])
                continue
            match = re.fullmatch(r"kawaii(\d{4})", base)
            if match:
                participant_ids.add(f"sub{int(match.group(1)):03d}")

    subject_root = root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects"
    if subject_root.exists():
        participant_ids.update(path.name for path in subject_root.iterdir() if path.is_dir() and re.fullmatch(r"(?:sub\d{3})|(?:pilot\d{2})", path.name))

    rows: list[dict[str, Any]] = []
    for participant in sorted(participant_ids):
        is_pilot = participant.startswith("pilot")
        raw_number = int(re.search(r"\d+", participant).group())
        raw_base = f"kawaii{raw_number:04d}"
        raw_present = (raw_root / f"{raw_base}.vhdr").exists()
        epoch_path = subject_root / participant / "part1" / "epochs" / "all_image_epochs-epo.fif"
        metadata_path = subject_root / participant / "part1" / "epochs" / "epoch_metadata.csv"
        participant_log_dirs = log_dirs_by_participant.get(participant, [])
        event_data_present = any(list(path.glob("*_trial_log_analysis.csv")) and list(path.glob("*_trigger_log.csv")) for path in participant_log_dirs)
        behavioural_status = "YES" if event_data_present else "NO"
        if metadata_path.exists():
            metadata_probe = pd.read_csv(metadata_path, nrows=1)
            if "behavioural_log_available" in metadata_probe and not bool(metadata_probe.iloc[0]["behavioural_log_available"]):
                behavioural_status = "PARTIAL — schedule/event fallback present; original behavioural log absent"

        configured_exclusion = ""
        if participant in default_excluded:
            configured_exclusion = "participant defaults exclude pilot"
        if participant in analysis_excluded:
            configured_exclusion = exclusion_reasons.get(participant, "analysis.exclude_subjects")

        included = bool(not is_pilot and not configured_exclusion and epoch_path.exists() and metadata_path.exists())
        if included:
            reason = "Eligible non-pilot with validated canonical Gate 3C epochs and trial/event metadata."
        elif is_pilot:
            reason = "Pilot; not part of the scientific calibration cohort."
        elif configured_exclusion:
            reason = f"Explicitly excluded: {configured_exclusion}"
        elif not raw_present:
            reason = "No raw Session 2 EEG available and no validated canonical Gate 3C epochs."
        elif not epoch_path.exists() or not metadata_path.exists():
            reason = "Raw EEG/logs available, but validated canonical Gate 3C epoch output is not yet available."
        else:
            reason = "Not eligible under current calibration rules."
        rows.append(
            {
                "participant": participant,
                "raw_eeg_data_present": "YES" if raw_present else "NO",
                "required_preprocessing_output_present": "YES" if epoch_path.exists() and metadata_path.exists() else "NO",
                "behaviour_event_data_present": behavioural_status,
                "participant_type": "pilot" if is_pilot else "non-pilot",
                "configured_exclusion": configured_exclusion,
                "included_in_calibration": "YES" if included else "NO",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def collect_epoch_metrics(root: Path, config: Mapping[str, Any], participants: Sequence[str]) -> pd.DataFrame:
    """Read frozen canonical epochs and return trial-identifying PTP metrics."""

    import mne

    rows: list[pd.DataFrame] = []
    for participant in participants:
        epoch_path = root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / participant / "part1" / "epochs" / "all_image_epochs-epo.fif"
        if not epoch_path.exists():
            raise FileNotFoundError(f"Missing canonical Gate 3C epochs: {epoch_path}")
        epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
        baseline = None if epochs.baseline is None else tuple(float(value) for value in epochs.baseline)
        if not np.isclose(float(epochs.tmin), EXPECTED_TMIN_S) or not np.isclose(float(epochs.tmax), EXPECTED_TMAX_S):
            raise ValueError(f"Unexpected epoch interval for {participant}: {(epochs.tmin, epochs.tmax)}")
        if baseline is None or not np.allclose(baseline, EXPECTED_BASELINE_S):
            raise ValueError(f"Unexpected baseline for {participant}: {baseline}")
        eeg_indices = mne.pick_types(epochs.info, eeg=True, eog=False, meg=False, exclude=[])
        eeg_names = [epochs.ch_names[index] for index in eeg_indices]
        values_uv, maximum_channels = max_eeg_ptp_uv(epochs.get_data(picks=eeg_indices), eeg_names)
        if epochs.metadata is None or len(epochs.metadata) != len(epochs):
            raise ValueError(f"Missing or misaligned epoch metadata for {participant}.")
        metadata = epochs.metadata.reset_index(drop=True)
        identity = pd.DataFrame(
            {
                "participant": participant,
                "calibration_cohort": "initial" if participant in INITIAL_CALIBRATION_PARTICIPANTS else "newly_added",
                "recording_part": metadata.get("recording_part", pd.Series(["part1"] * len(metadata))).astype(str),
                "trial": metadata.get("trial", pd.Series(range(1, len(metadata) + 1))),
                "event_sample": metadata.get("event_sample", pd.Series(epochs.events[:, 0])),
                "event_code": metadata.get("event_code", pd.Series(epochs.events[:, 2])),
                "source_epoch_index": np.arange(len(metadata), dtype=int),
                "ptp_max_uv": values_uv,
                "max_ptp_channel": maximum_channels,
                "n_eeg_channels": len(eeg_names),
                "epoch_tmin_s": float(epochs.tmin),
                "epoch_tmax_s": float(epochs.tmax),
                "baseline_tmin_s": float(baseline[0]),
                "baseline_tmax_s": float(baseline[1]),
            }
        )
        rows.append(identity)
    return pd.concat(rows, ignore_index=True)


def _write_figures(metrics: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = metrics["ptp_max_uv"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50, color="#4472C4", edgecolor="white")
    ax.set(xlabel="Maximum EEG channel PTP per epoch (µV)", ylabel="Epoch count", title="Pooled residual epoch PTP distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "pooled_ptp_histogram.png", dpi=160)
    plt.close(fig)

    tail_start = float(np.quantile(values, 0.90))
    tail = values[values >= tail_start]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tail, bins=40, color="#ED7D31", edgecolor="white")
    ax.set(xlabel="Maximum EEG channel PTP per epoch (µV)", ylabel="Epoch count", title="Upper 10% of pooled residual epoch PTP")
    fig.tight_layout()
    fig.savefig(output_dir / "pooled_ptp_extreme_tail.png", dpi=160)
    plt.close(fig)

    participants = sorted(metrics["participant"].unique())
    participant_values = [metrics.loc[metrics["participant"].eq(participant), "ptp_max_uv"].to_numpy(float) for participant in participants]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(participant_values, tick_labels=participants, showfliers=True)
    ax.set(ylabel="Maximum EEG channel PTP per epoch (µV)", title="Participant-wise residual epoch PTP")
    fig.tight_layout()
    fig.savefig(output_dir / "participant_ptp_distributions.png", dpi=160)
    plt.close(fig)

    ranked = np.sort(values)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(ranked) + 1), ranked, color="#70AD47")
    ax.set(xlabel="Descending rank", ylabel="Maximum EEG channel PTP per epoch (µV)", title="Ranked pooled residual epoch PTP")
    fig.tight_layout()
    fig.savefig(output_dir / "ranked_epoch_ptp.png", dpi=160)
    plt.close(fig)


def run_calibration(root: Path, output_dir: Path) -> None:
    """Run read-only calibration over configured frozen Gate 3C epoch files."""

    from eeg_software.workflows.kawaii_session2.config import load_session2_eeg_config

    config = load_session2_eeg_config(root=root)
    inventory = participant_inventory(root, config)
    participants = inventory.loc[inventory["included_in_calibration"].eq("YES"), "participant"].astype(str).tolist()
    if not participants:
        raise ValueError("No currently eligible participants have validated canonical Gate 3C epochs.")
    metrics = collect_epoch_metrics(root, config, participants)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "participant_inclusion.csv", index=False)
    metrics.to_csv(output_dir / "epoch_ptp_metrics.csv", index=False)

    summaries = []
    for participant, group in metrics.groupby("participant", sort=True):
        summaries.append({"scope": "participant", "participant": participant, **distribution_summary(group["ptp_max_uv"])})
    summaries.append({"scope": "pooled", "participant": "ALL", **distribution_summary(metrics["ptp_max_uv"])})
    pd.DataFrame(summaries).to_csv(output_dir / "distribution_summary.csv", index=False)

    cohort_summaries = [
        {"calibration": "original", "participants": ";".join(INITIAL_CALIBRATION_PARTICIPANTS), **distribution_summary(metrics.loc[metrics["participant"].isin(INITIAL_CALIBRATION_PARTICIPANTS), "ptp_max_uv"])},
        {"calibration": "extended", "participants": ";".join(participants), **distribution_summary(metrics["ptp_max_uv"])},
    ]
    pd.DataFrame(cohort_summaries).to_csv(output_dir / "original_vs_extended_summary.csv", index=False)

    candidates = candidate_thresholds(metrics["ptp_max_uv"])
    pooled_impacts, participant_impacts = threshold_impacts(metrics, candidates)
    pooled_impacts.to_csv(output_dir / "candidate_threshold_impacts.csv", index=False)
    participant_impacts.to_csv(output_dir / "candidate_participant_impacts.csv", index=False)
    metrics.sort_values("ptp_max_uv", ascending=False).assign(pooled_descending_rank=lambda frame: range(1, len(frame) + 1)).to_csv(
        output_dir / "ranked_epoch_ptp.csv", index=False
    )
    _write_figures(metrics, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase2/phase2a4_ptp_calibration"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    run_calibration(root, output_dir)


if __name__ == "__main__":
    main()
