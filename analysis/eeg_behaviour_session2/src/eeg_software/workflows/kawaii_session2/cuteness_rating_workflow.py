"""Single entry point for two independently authorised cuteness ERP sources.

Participant: render the accepted, already published Gate 5B group evokeds.
Normative: reuse an authorised NORM-02 archive or calculate descriptive
waveforms from *already corrected*, publication-authenticated Gate 3C epochs.
Neither route reruns preprocessing, ICA, rejected-epoch decisions or models.

The two rating families are NOT interchangeable scientific estimands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

CHOICES = ("participant", "normative", "both")
STATUS = "EXPLORATORY_NOT_APPROVED_FOR_COLLABORATOR_REPORT"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _checked_paths(root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Analysis root must be a directory")
    output = output_dir.resolve()
    if _inside(output, root):
        raise ValueError("Output must be outside the analysis repository")
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing analysis output")
    if not output.parent.is_dir():
        raise FileNotFoundError("Output parent must already exist")
    return root, output


def _checkout_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    )
    head = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("Cannot establish exact Git checkout identity")
    return head


def _call_script(root: Path, name: str, args: list[str]) -> None:
    script = root / "tools" / name
    if not script.is_file():
        raise FileNotFoundError(f"Required analysis script missing: {script}")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.run([sys.executable, str(script), *args], cwd=root, env=env, check=True)


def execute(
    *, root: Path, output_dir: Path, rating_source: str,
    source_root: Path | None = None,
    normative_archive: Path | None = None,
    aaps_reference: Path | None = None,
    acknowledge_sub012_exploratory: bool = False,
) -> Path:
    """Stage outputs privately; publish only on success; never write study sources."""
    if rating_source not in CHOICES:
        raise ValueError(f"rating_source must be one of {CHOICES}")
    root, output = _checked_paths(Path(root), Path(output_dir))
    participant = rating_source in ("participant", "both")
    normative = rating_source in ("normative", "both")
    if normative and not acknowledge_sub012_exploratory:
        raise ValueError(
            "Normative cohort includes 190 exposure-ambiguous sub012 epochs; "
            "explicit --acknowledge-sub012-exploratory required"
        )
    if not normative and (normative_archive is not None or aaps_reference is not None or acknowledge_sub012_exploratory):
        raise ValueError("Normative-only options are invalid with participant rating source")
    if aaps_reference is not None and normative_archive is not None:
        raise ValueError("--aaps-reference is only used when calculating a new normative archive")
    if source_root is not None and not participant:
        raise ValueError("--source-root requires participant or both rating sources")
    if participant:
        source = (Path(source_root) if source_root is not None else
                  root / "workspace/kawaii/session2/corrections/r11_qc_v3")
        source = source.resolve(strict=True)
        if not source.is_dir():
            raise ValueError("Published participant source must be a directory")
    else:
        source = None
    supplied_archive = Path(normative_archive).resolve(strict=True) if normative_archive is not None else None
    if supplied_archive is not None and not supplied_archive.is_file():
        raise ValueError("Normative archive must be a file")
    if aaps_reference is not None and not Path(aaps_reference).resolve(strict=True).is_file():
        raise ValueError("AAPS reference must be an existing file")
    # These are code requirements; validate before any analysis or output writes.
    required_scripts = ["run_kawaii_cuteness_report_figures.py"]
    if normative and supplied_archive is None:
        required_scripts.append("run_kawaii_norm02_exploratory.py")
    for name in required_scripts:
        if not (root / "tools" / name).is_file():
            raise FileNotFoundError(f"Required implementation missing: tools/{name}")

    # The staging folder and final destination share a parent/filesystem.
    with tempfile.TemporaryDirectory(prefix="kawaii_cuteness_stage_", dir=output.parent) as stage_name:
        stage = Path(stage_name)
        norm_zip = supplied_archive
        head = None
        if normative and norm_zip is None:
            head = _checkout_head(root)
            norm_path = stage / "normative_summary"
            cmd = ["--repo", str(root), "--output", str(norm_path), "--source-head", head,
                   "--acknowledge-sub012-exploratory"]
            if aaps_reference is not None:
                cmd += ["--aaps-reference", str(Path(aaps_reference).resolve(strict=True))]
            _call_script(root, "run_kawaii_norm02_exploratory.py", cmd)
            norm_zip = norm_path.with_suffix(".zip")
            if not norm_zip.is_file():
                raise RuntimeError("Normative analysis finished without its expected ZIP")
        fig_args = ["--rating-source", rating_source,
                    "--project-root", str(root), "--output-dir", str(stage / "figures")]
        if source is not None:
            fig_args += ["--source-root", str(source)]
        if norm_zip is not None:
            fig_args += ["--normative-archive", str(norm_zip)]
        _call_script(root, "run_kawaii_cuteness_report_figures.py", fig_args)
        fig_manifest = stage / "figures" / "MANIFEST.json"
        if not fig_manifest.is_file():
            raise RuntimeError("Figure renderer finished without a provenance manifest")
        # The figure command authenticates published participant FIFs or NORM-02
        # summary members. This top-level manifest records orchestration only;
        # it cannot elevate an exploratory analysis to scientific approval.
        manifest = {
            "schema": "kawaii_cuteness_unified_descriptive_v1",
            "rating_source": rating_source,
            "scientific_status": STATUS,
            "participant_source": str(source) if source is not None else None,
            "normative_mode": (
                "existing_archive" if supplied_archive is not None else
                "authenticated_corrected_epochs" if normative else None
            ),
            "normative_archive_sha256": _sha256(norm_zip) if norm_zip is not None else None,
            "normative_checkout_head": head,
            "figure_manifest_sha256": _sha256(fig_manifest),
            "no_preprocessing_ica_or_model_fitting": True,
            "separate_rating_authorities": True,
            "normative_exposure_ambiguity_acknowledged": bool(normative),
            "note": "Descriptive category waveforms only, not hypothesis testing or scientific approval",
        }
        (stage / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError("Output appeared during analysis; refusing overwrite")
        os.rename(stage, output)
    return output


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rating-source", choices=CHOICES, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="Analysis checkout root (not an EEG input path)")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Current corrected Gate 5B group publication root for participant/both")
    parser.add_argument("--normative-archive", type=Path, default=None,
                        help="Existing authorised NORM-02 ZIP (skips new normative aggregation)")
    parser.add_argument("--aaps-reference", type=Path, default=None,
                        help="Privately supplied SHA-pinned AAPS crosswalk for new normative aggregation")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh output directory outside the checkout")
    parser.add_argument("--acknowledge-sub012-exploratory", action="store_true",
                        help="Explicitly retain exploratory sub012 exposure-ambiguous epochs")


def run_namespace(args: argparse.Namespace) -> int:
    output = execute(
        root=args.root, output_dir=args.output_dir, rating_source=args.rating_source,
        source_root=args.source_root, normative_archive=args.normative_archive,
        aaps_reference=args.aaps_reference,
        acknowledge_sub012_exploratory=args.acknowledge_sub012_exploratory,
    )
    print("KAWAII_CUTENESS_DESCRIPTIVE: PASS")
    print("OUTPUT:", output)
    print("SCIENTIFIC_STATUS:", STATUS)
    print("ORIGINAL_EEG_PREPROCESSING_OR_MODEL_FITTING: NONE")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run_namespace(parser.parse_args(argv))
