"""Original-cohort NORM-02 descriptive reproduction.

This intentionally fixed workflow preserves original-cohort source, eligibility,
and result-fingerprint guards. It is not the reusable alternative-dataset path.
No EEG/preprocessing writes, model fitting or inferential tests are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_software.workflows.kawaii_session2.norm02_descriptive import (
    aggregate_descriptive, Norm02ValidationError,
)

SUB012_AMBIGUITY = 190
EXPECTED_NORMATIVE = {"lower_aaps": 1482, "higher_aaps": 1486}
EXPECTED_SUBJECTIVE = {"low": 492, "medium_low": 622, "medium_high": 731, "high": 1123}
FAMILY_COLORS = {
    "lower_aaps": "#54877C", "higher_aaps": "#213E58",
    "low": "#728BA3", "medium_low": "#6C9888",
    "medium_high": "#A65E68", "high": "#263F62",
}
FAMILY_NAMES = {
    "lower_aaps": "Lower AAPS score (≤5.0)",
    "higher_aaps": "Higher AAPS score (>6.5)",
    "low": "Subjective 1–2", "medium_low": "Subjective 3–4",
    "medium_high": "Subjective 5–6", "high": "Subjective 7–9",
}


def validate_exploratory_counts(result, *, full: bool) -> None:
    summaries = result.summary
    if len(summaries) != 6 or summaries.duplicated(["family", "category"]).any():
        raise Norm02ValidationError("Exactly six distinct descriptive category summaries required")
    for family in ("normative", "subjective"):
        rows = summaries.loc[summaries.family == family]
        if len(rows) != (2 if family == "normative" else 4):
            raise Norm02ValidationError("Missing category in descriptive summary")
        if full:
            expected = EXPECTED_NORMATIVE if family == "normative" else EXPECTED_SUBJECTIVE
            actual = dict(zip(rows.category, rows.retained_trials.astype(int)))
            if actual != expected:
                raise Norm02ValidationError(f"Corrected R11 category counts changed: {family}: {actual}")
            want_n = {key: 8 for key in expected}
            if family == "subjective":
                want_n["low"] = 7
            actual_n = dict(zip(rows.category, rows.n_participants.astype(int)))
            if actual_n != want_n:
                raise Norm02ValidationError(f"Contributor N differs from QC expectation: {actual_n}")
    if full:
        cell = result.counts.loc[
            (result.counts.family == "subjective") &
            (result.counts.category == "low") &
            (result.counts.participant == "sub010")
        ]
        if len(cell) != 1 or int(cell.iloc[0].retained_trials) != 8 or bool(cell.iloc[0].eligible_for_group):
            raise Norm02ValidationError("sub010 eight-trial subjective Low exclusion was not preserved")
    if result.waveforms.empty or result.waveforms.groupby(["family", "category"]).size().nunique() != 1:
        raise Norm02ValidationError("Incomplete waveform time grid")
    if not np.isfinite(result.waveforms.mean_amplitude_uv.to_numpy(dtype=float)).all():
        raise Norm02ValidationError("Non-finite descriptive waveforms")


def verify_sub012_metadata(part) -> None:
    md = part.metadata
    if "source_exposure_status" not in md or "M_Kawaii" not in md:
        raise Norm02ValidationError("sub012 missing authoritative exposure status or norms")
    ambiguous = md.source_exposure_status.astype("string").eq("AMBIGUOUS_RUN4_EXPOSURE").fillna(False)
    low = pd.to_numeric(md.M_Kawaii, errors="raise") <= 5.0
    high = pd.to_numeric(md.M_Kawaii, errors="raise") > 6.5
    if (len(md), int(ambiguous.sum()), int(low.sum()), int(high.sum()),
        int((low & ambiguous).sum()), int((high & ambiguous).sum())) != (371,190,183,188,95,95):
        raise Norm02ValidationError("sub012 published exposure/group split changed")


def draw_figure(result, destination: Path, *, full: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    fig.subplots_adjust(left=.075, right=.985, top=.75, bottom=.23, wspace=.08)
    family_titles = (("normative", "AAPS normative cuteness (2 score groups)"),
                     ("subjective", "Individual subjective cuteness (4 rating bins)"))
    summary = result.summary.set_index(["family", "category"])
    for ax, (family, title) in zip(axes, family_titles):
        ax.axvspan(500, 800, color="#E7E9EB", alpha=.6, label="500–800 ms ROI window")
        ax.axhline(0, color="#A4ADB7", lw=0.8)
        ax.axvline(0, color="#9AA6B1", lw=0.8, ls="--")
        rows = result.waveforms.loc[result.waveforms.family == family]
        order = EXPECTED_NORMATIVE if family == "normative" else EXPECTED_SUBJECTIVE
        for category in order:
            curve = rows.loc[rows.category == category].sort_values("time_s")
            if curve.empty:
                raise Norm02ValidationError(f"No waveform for {family}/{category}")
            n = int(summary.loc[(family, category), "n_participants"])
            ax.plot(curve.time_s.to_numpy(dtype=float) * 1000,
                    curve.mean_amplitude_uv.to_numpy(dtype=float),
                    lw=1.8, color=FAMILY_COLORS[category],
                    label=f"{FAMILY_NAMES[category]} · N={n}")
        ax.set(title=title, xlabel="Time from image onset (ms)")
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Mean ROI amplitude (µV)")
    qualifier = "Eight participants · 2,968 retained epochs" if full else "Seven-participant sensitivity · sub012 excluded"
    fig.suptitle("Kawaii Session 2 — descriptive cuteness ERPs\n" + qualifier,
                 y=.97, fontsize=13, fontweight="semibold")
    fig.text(.5,.065,"EXPLORATORY / NOT SCIENTIFICALLY APPROVED · Cz, CP1, CP2, Pz · equal participant weight · no inferential tests",
             ha="center",fontsize=8,color="#7B4146")
    fig.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_source_checkout_head(repository: Path, supplied_head: str) -> None:
    """Do not write a caller-invented Git revision into source provenance."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Norm02ValidationError("Cannot verify source repository Git HEAD") from exc
    observed = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", observed) is None or observed != supplied_head:
        raise Norm02ValidationError("Supplied source HEAD differs from the actual Git checkout")


def write_outputs(full_result, sensitivity_result, bindings, output: Path, *, cohort_head: str) -> Path:
    """Write only after all validation; stage privately then atomically finalise."""
    output = output.resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise FileExistsError("Refusing to overwrite existing NORM-02 exploratory output")
    if not output.parent.is_dir():
        raise FileNotFoundError("Chosen Downloads output parent does not exist")
    with tempfile.TemporaryDirectory(prefix="norm02_stage_", dir=output.parent) as temporary:
        stage = Path(temporary)
        files = {}
        for tag, result in (("eight_all_retained", full_result), ("seven_without_sub012", sensitivity_result)):
            for field in ("waveforms", "counts", "contributions", "summary"):
                name = f"{tag}_{field}.csv"
                getattr(result, field).to_csv(stage / name, index=False)
                files[name] = _sha(stage / name)
            fig = f"{tag}_descriptive_erp.png"
            draw_figure(result, stage / fig, full=(tag=="eight_all_retained"))
            files[fig] = _sha(stage / fig)
        manifest = {
            "schema": "kawaii_norm02_exploratory_unapproved_v1",
            "scientific_status": "EXPLORATORY_NOT_APPROVED_FOR_COLLABORATOR_REPORT",
            "source_repository_head": cohort_head,
            "participant_source_bindings_sha256": dict(sorted(bindings.items())),
            "all_retained_sub012_exposure_ambiguous": 190,
            "normative_classification": "AAPS-score-defined (not confirmed original feedback pools)",
            "subjective_bins": ["1–2", "3–4", "5–6", "7–9"],
            "minimum_trials_per_participant_category": 10,
            "sub010_low_cell": "8 retained; excluded from subjective Low group average",
            "secondary_sensitivity": "seven participants, entirely excludes sub012",
            "sampling": "existing Gate 3C epochs, no new preprocessing/ICA/models",
            "group_weighting": "equal participant weights after participant trial means",
            "statistical_tests": "none",
            "normative_waveforms_member": "eight_all_retained_waveforms.csv",
            "normative_summary_member": "eight_all_retained_summary.csv",
            "file_sha256": files,
        }
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        output.mkdir()  # Reserve unique destination before finalising.
        try:
            for file in stage.iterdir():
                os.replace(file, output / file.name)
            tmpzip = output.parent / (archive.name + ".partial")
            try:
                with zipfile.ZipFile(tmpzip,"w",compression=zipfile.ZIP_DEFLATED) as z:
                    for file in sorted(output.iterdir()):
                        z.write(file, arcname=file.name)
                os.replace(tmpzip,archive)
            finally:
                if tmpzip.exists(): tmpzip.unlink()
        except Exception:
            # Never silently claim that a partial results directory is a published output.
            raise
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Original-cohort NORM02 descriptive reproduction")
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-head",required=True)
    parser.add_argument("--aaps-reference",type=Path,default=None,
                        help="External authorised SHA-pinned AAPS image reference CSV")
    parser.add_argument("--acknowledge-sub012-exploratory",action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_sub012_exploratory:
        parser.error("Explicit --acknowledge-sub012-exploratory required: scientific inclusion not yet approved")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_head):
        parser.error("Source Git HEAD must be exact 40-character lowercase SHA-1")
    repo = args.repo.resolve(strict=True)
    verify_source_checkout_head(repo, args.source_head)
    out = args.output.resolve()
    if out == repo or repo in out.parents:
        parser.error("Refusing to write any outputs within the accepted EEG repository")
    if out.exists() or out.with_suffix(".zip").exists():
        parser.error("Refusing overwrite of output directory/ZIP")
    from eeg_software.workflows.kawaii_session2.config import load_session2_eeg_config
    from eeg_software.workflows.kawaii_session2.norm02_adapter import (
        EXPECTED_SUBJECT_COUNTS, load_aaps_reference, load_accepted_roi_part, validate_full_cohort,
    )
    print("NORM02_EXPLORATORY_UNAPPROVED: START",flush=True)
    reference_path = (args.aaps_reference if args.aaps_reference is not None else
                      repo / "config/kawaii_norm02_aaps_reference_20261101.csv")
    reference = load_aaps_reference(reference_path)
    config = load_session2_eeg_config(root=repo)
    parts = {}
    for subject in EXPECTED_SUBJECT_COUNTS:
        print(f"SOURCE_LOADING: {subject}",flush=True)
        parts[subject] = load_accepted_roi_part(
            repo, subject, config=config,
            allow_sub012_exposure_ambiguous=(subject=="sub012"),
        )
        print(f"SOURCE_LOADED: {subject} n={len(parts[subject].metadata)}",flush=True)
    validate_full_cohort(parts)
    verify_sub012_metadata(parts["sub012"])
    full = aggregate_descriptive(list(parts.values()),aaps_reference=reference)
    validate_exploratory_counts(full, full=True)
    seven = aggregate_descriptive([p for k,p in parts.items() if k!="sub012"],aaps_reference=reference)
    validate_exploratory_counts(seven, full=False)
    archive = write_outputs(full,seven,
        {k:v.source_binding_sha256 for k,v in parts.items()},out,cohort_head=args.source_head)
    print("NORM02_EXPLORATORY_QC: PASS",flush=True)
    print("OUTPUT_ZIP:",archive,flush=True)
    print("SCIENTIFIC_STATUS: EXPLORATORY_NOT_APPROVED",flush=True)

if __name__ == "__main__":
    main()
