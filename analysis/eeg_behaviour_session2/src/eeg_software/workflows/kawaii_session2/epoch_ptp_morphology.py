"""Generate a condition-blind residual epoch PTP morphology review package."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


LOWER_CANDIDATE_UV = 286.86500419323795
MIDDLE_CANDIDATE_UV = 330.2767354136942
UPPER_CANDIDATE_UV = 505.89503606950973
FORBIDDEN_REVIEW_FIELDS = {
    "condition",
    "old_new_status",
    "recognition_response",
    "recognition_correct",
    "kawaii_rating",
    "arousal_rating",
}


def ptp_stratum(value_uv: float) -> str:
    """Return the prespecified candidate-relevant PTP stratum."""

    value = float(value_uv)
    if value > UPPER_CANDIDATE_UV:
        return "A_above_505_895"
    if value > MIDDLE_CANDIDATE_UV:
        return "B_330_277_to_505_895"
    if value > LOWER_CANDIDATE_UV:
        return "C_286_865_to_330_277"
    return "D_at_or_below_286_865_reference"


def _balanced_reference_sample(group: pd.DataFrame, n_per_participant: int) -> pd.DataFrame:
    """Select high-edge, median-near, and low-edge reference epochs."""

    ordered = group.sort_values(["ptp_max_uv", "source_epoch_index"], kind="mergesort")
    if len(ordered) <= n_per_participant:
        return ordered
    n_edge = n_per_participant // 3
    median = float(ordered["ptp_max_uv"].median())
    high = ordered.tail(n_edge)
    low = ordered.head(n_edge)
    used = set(high.index) | set(low.index)
    remaining = ordered.loc[~ordered.index.isin(used)].copy()
    middle = remaining.assign(_distance=(remaining["ptp_max_uv"] - median).abs()).sort_values(
        ["_distance", "source_epoch_index"], kind="mergesort"
    ).head(n_per_participant - len(used)).drop(columns="_distance")
    return pd.concat([low, middle, high]).drop_duplicates().sort_values(["ptp_max_uv", "source_epoch_index"], kind="mergesort")


def select_review_epochs(metrics: pd.DataFrame, reference_per_participant: int = 6) -> pd.DataFrame:
    """Select all upper-stratum epochs plus a balanced deterministic reference."""

    required = {"participant", "source_epoch_index", "ptp_max_uv", "max_ptp_channel"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Calibration metrics missing required columns: {missing}")
    selected = metrics.copy()
    selected["ptp_stratum"] = selected["ptp_max_uv"].map(ptp_stratum)
    upper = selected[~selected["ptp_stratum"].eq("D_at_or_below_286_865_reference")].copy()
    reference_pool = selected[selected["ptp_stratum"].eq("D_at_or_below_286_865_reference")].copy()
    references = [
        _balanced_reference_sample(group, reference_per_participant)
        for _, group in reference_pool.groupby("participant", sort=True)
    ]
    review = pd.concat([upper, *references], ignore_index=True)
    stratum_order = {
        "A_above_505_895": 0,
        "B_330_277_to_505_895": 1,
        "C_286_865_to_330_277": 2,
        "D_at_or_below_286_865_reference": 3,
    }
    review["_stratum_order"] = review["ptp_stratum"].map(stratum_order)
    review = review.sort_values(
        ["_stratum_order", "participant", "ptp_max_uv", "source_epoch_index"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    review.insert(0, "review_id", [f"PTP-{index:03d}" for index in range(1, len(review) + 1)])
    return review.drop(columns="_stratum_order")


def blinded_review_table(selected: pd.DataFrame) -> pd.DataFrame:
    """Return the condition-blind human review worksheet."""

    table = selected.rename(columns={"source_epoch_index": "epoch_index"})[
        ["review_id", "participant", "epoch_index", "ptp_max_uv", "max_ptp_channel", "ptp_stratum"]
    ].copy()
    table["morphology_assessment"] = ""
    table["notes"] = ""
    if FORBIDDEN_REVIEW_FIELDS.intersection(table.columns):
        raise AssertionError("A forbidden outcome field entered the morphology review table.")
    return table


def _review_figures(root: Path, selected: pd.DataFrame, output_dir: Path, panels_per_page: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_cache = {}
    for participant in sorted(selected["participant"].unique()):
        path = root / "workspace" / "kawaii" / "session2" / "analysis" / "subjects" / participant / "part1" / "epochs" / "all_image_epochs-epo.fif"
        epoch_cache[participant] = mne.read_epochs(path, preload=True, verbose=False)

    for stratum, group in selected.groupby("ptp_stratum", sort=False):
        group = group.reset_index(drop=True)
        n_pages = int(math.ceil(len(group) / panels_per_page))
        for page_index in range(n_pages):
            page = group.iloc[page_index * panels_per_page : (page_index + 1) * panels_per_page]
            fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
            for ax, (_, row) in zip(axes.flat, page.iterrows()):
                epochs = epoch_cache[str(row["participant"])]
                epoch_index = int(row["source_epoch_index"])
                eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False, meg=False, exclude=[])
                eeg_names = [epochs.ch_names[index] for index in eeg_picks]
                data_uv = epochs.get_data(picks=eeg_picks)[epoch_index] * 1e6
                times = epochs.times
                ax.plot(times, data_uv.T, color="#999999", alpha=0.16, linewidth=0.55)
                maximum_index = eeg_names.index(str(row["max_ptp_channel"]))
                ax.plot(times, data_uv[maximum_index], color="#C00000", linewidth=1.5, label=str(row["max_ptp_channel"]))
                for representative, color in (("Fz", "#4472C4"), ("Cz", "#70AD47"), ("Pz", "#7030A0")):
                    if representative in eeg_names and representative != row["max_ptp_channel"]:
                        ax.plot(times, data_uv[eeg_names.index(representative)], color=color, linewidth=0.8, alpha=0.85)
                ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--")
                ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.5)
                ax.set_title(
                    f"{row['review_id']} | {row['participant']} | epoch {epoch_index}\n"
                    f"PTP {float(row['ptp_max_uv']):.1f} µV | max {row['max_ptp_channel']}",
                    fontsize=9,
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (µV)")
                ax.legend(loc="upper right", fontsize=7, frameon=False)
            for ax in axes.flat[len(page) :]:
                ax.axis("off")
            fig.suptitle(f"Condition-blind morphology review — {stratum} — page {page_index + 1}/{n_pages}", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            safe_stratum = stratum.lower()
            fig.savefig(output_dir / f"{safe_stratum}_page_{page_index + 1:02d}.png", dpi=150)
            plt.close(fig)


def run_review(root: Path, output_dir: Path) -> None:
    metrics_path = root / "docs" / "phase2" / "phase2a4_ptp_calibration" / "epoch_ptp_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    selected = select_review_epochs(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    worksheet = blinded_review_table(selected)
    worksheet.to_csv(output_dir / "morphology_review.csv", index=False)
    summary = selected.groupby(["ptp_stratum", "participant"], sort=False).size().rename("n_reviewed").reset_index()
    summary.to_csv(output_dir / "review_sample_summary.csv", index=False)
    _review_figures(root, selected, output_dir / "figures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase2/phase2a4_ptp_morphology"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    run_review(root, output_dir)


if __name__ == "__main__":
    main()
