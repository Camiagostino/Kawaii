"""NORM-INF-01/02: original Kawaii cohort source-verified inference runner.

Read-only source loading; never modifies EEG publications. Outputs outside repo.
No statistical result without the authenticated primary seven-person input.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.normative_inference import (
    WINDOW_S, InferenceStop, category_contrasts, continuous_mixed_model,
    holm_two, validate_trials,
)
from eeg_software.workflows.kawaii_session2.norm02_adapter import (
    EXPECTED_SUBJECT_COUNTS, load_aaps_reference, load_accepted_roi_part,
)

PRIMARY = ("sub005", "sub009", "sub010", "sub011", "sub013", "sub014", "sub015")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _head(repo: Path) -> str:
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       text=True, capture_output=True, check=True)
    head = p.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise InferenceStop("Unverified source checkout")
    return head


def _boolean(series: pd.Series) -> np.ndarray:
    # Do not allow truthiness conversion of strings such as 'False'.
    values = series.to_numpy(dtype=object)
    if not all(isinstance(x,(bool,np.bool_)) for x in values):
        raise InferenceStop("Source eligibility must be genuine Boolean values")
    return np.asarray(values,dtype=bool)


def part_to_rows(part, *, reference: dict[str,float], allow_ambiguous: bool) -> tuple[pd.DataFrame,dict]:
    """Only source-validator-produced AcceptedPartTraces enter this stage."""
    md = part.metadata.reset_index(drop=True)
    essential = (
        "image_filename", "M_Kawaii", "trial", "event_sample",
        "continuous_cuteness_old_new_eligible", "source_verified_old_new_status",
        "source_exposure_status",
    )
    missing = [x for x in essential if x not in md.columns]
    if missing:
        raise InferenceStop(f"Source-verified metadata lacks: {missing}")
    times = np.asarray(part.times_s,dtype=float)
    if times.ndim != 1 or len(times)<3 or np.any(np.diff(times)<=0) or not np.isfinite(times).all():
        raise InferenceStop("Invalid accepted epoch times")
    mask = (times >= WINDOW_S[0]-1e-9) & (times <= WINDOW_S[1]+1e-9)
    if int(mask.sum()) < 2 or abs(times[mask][0]-WINDOW_S[0])>.002 or abs(times[mask][-1]-WINDOW_S[1])>.002:
        raise InferenceStop("Accepted epoch time grid lacks exact ROI window")
    traces = np.asarray(part.traces_uv,dtype=float)
    if traces.shape != (len(md),len(times)) or not np.isfinite(traces).all():
        raise InferenceStop("Authenticated ROI traces/metadata shape mismatch")
    eligible = _boolean(md["continuous_cuteness_old_new_eligible"])
    rejected = md.loc[~eligible]
    if len(rejected):
        if not allow_ambiguous or part.participant != "sub012" or len(rejected)!=190 or not rejected["source_exposure_status"].astype(str).eq("AMBIGUOUS_RUN4_EXPOSURE").all():
            raise InferenceStop("Source-ineligible trials found outside declared sub012 ambiguity")
    accepted_md = md.loc[eligible].reset_index(drop=True)
    scores=[]
    for image, score in zip(accepted_md.image_filename,accepted_md.M_Kawaii):
        key=str(image).strip()
        if key not in reference or not np.isfinite(float(score)) or abs(float(score)-reference[key])>1e-8:
            raise InferenceStop("Mismatch against authenticated 400-image AAPS reference")
        scores.append(reference[key])
    oldnew = accepted_md["source_verified_old_new_status"].astype(str).str.strip().str.lower()
    if not oldnew.isin(("old","new")).all():
        raise InferenceStop("Invalid source-verified old/new label among accepted trials")
    exposure = accepted_md["source_exposure_status"].astype(str).str.strip()
    if not np.array_equal(exposure.to_numpy(),
                          oldnew.map({"old":"VERIFIED_OLD", "new":"VERIFIED_NEW"}).to_numpy()):
        raise InferenceStop("Verified old/new label conflicts with source-exposure status")
    rows=pd.DataFrame({
        "participant":part.participant,
        "recording_part":part.recording_part,
        "trial":accepted_md.trial.to_numpy(),
        "event_sample":accepted_md.event_sample.to_numpy(),
        "image_filename":accepted_md.image_filename.astype(str).to_numpy(),
        "M_Kawaii":scores,
        "erp_mean_uv":traces[eligible,:][:,mask].mean(axis=1),
        "verified_old_new":oldnew.to_numpy(),
        "source_eligible":True,
        "source_binding_sha256":part.source_binding_sha256,
    })
    return rows,{
        "participant":part.participant,
        "source_binding_sha256":part.source_binding_sha256,
        "retained_epochs":len(md),
        "excluded_source_ineligible":int((~eligible).sum()),
        "source_eligible_trials":int(eligible.sum()),
    }


def collect_authenticated_trials(repo:Path,reference_path:Path,*,sub012_sensitivity:bool):
    from eeg_software.workflows.kawaii_session2.config import load_session2_eeg_config
    config=load_session2_eeg_config(root=repo)
    reference=load_aaps_reference(reference_path)
    participants=list(PRIMARY)+(["sub012"] if sub012_sensitivity else [])
    chunks=[]; bindings=[]
    for subject in participants:
        print(f"SOURCE_LOADING: {subject}",flush=True)
        part=load_accepted_roi_part(repo,subject,config=config,
                                    allow_sub012_exposure_ambiguous=(subject=="sub012"))
        if len(part.metadata)!=EXPECTED_SUBJECT_COUNTS[subject]:
            raise InferenceStop(f"Retention changed for {subject}")
        chunk,record=part_to_rows(part,reference=reference,allow_ambiguous=(subject=="sub012"))
        chunks.append(chunk);bindings.append(record)
        print(f"SOURCE_VERIFIED: {subject} retained={record['retained_epochs']} eligible={record['source_eligible_trials']}",flush=True)
    data=pd.concat(chunks,ignore_index=True)
    if data.duplicated(["participant","recording_part","trial"]).any():
        raise InferenceStop("Trial identity collision across files")
    if sum(x['retained_epochs'] for x in bindings)!= (2968 if sub012_sensitivity else 2597):
        raise InferenceStop("Unexpected accepted cohort EEG-retained denominator")
    if sum(x['source_eligible_trials'] for x in bindings)!= (2778 if sub012_sensitivity else 2597):
        raise InferenceStop("Unexpected verified inference-eligible denominator")
    primary=data.loc[data.participant.isin(PRIMARY)].copy().reset_index(drop=True)
    validate_trials(primary)
    # Preflight the minimum eligible trial count without calculating test statistics.
    count_table=primary.assign(category=np.where(primary.M_Kawaii<=5.0,'lower_aaps','higher_aaps')).groupby(
        ['participant','category']).size().unstack(fill_value=0)
    if set(count_table.columns)!={'lower_aaps','higher_aaps'} or (count_table<10).any().any():
        raise InferenceStop('Verified primary category count below ten for a participant')
    if primary.participant.nunique()!=7:
        raise InferenceStop("Seven primary participant authorities not all present")
    if sub012_sensitivity:
        validate_trials(data,include_sub012=True)
    return primary,data,bindings


def run(args)->int:
    repo=args.repo.resolve(strict=True)
    reference_path=args.aaps_reference.resolve(strict=True)
    if not repo.is_dir() or not (repo/"src"/"eeg_software").is_dir():
        raise InferenceStop("--repo must be exact internal EEG-software checkout")
    if not reference_path.is_file():
        raise InferenceStop("Pinned AAPS reference unavailable")
    source_head=_head(repo)
    primary,all_eligible,bindings=collect_authenticated_trials(
        repo,reference_path,sub012_sensitivity=args.include_verified_sub012_sensitivity
    )
    # No inferential computation or output during preflight.
    # No real-data analysis, no output, until preflight and owner GO.
    inventory={
        "code_head":source_head,
        "aaps_reference_sha256":_hash(reference_path),
        "roi":["Cz","CP1","CP2","Pz"], "window_s":list(WINDOW_S),
        "source_bindings":bindings,
        "n_primary_rows":len(primary),
        "n_primary_items":int(primary.image_filename.nunique()),
        "n_primary_participants":int(primary.participant.nunique()),
        "include_verified_sub012_sensitivity":bool(args.include_verified_sub012_sensitivity),
        "cohort_exclusion":"sub012 excluded from primary, all 190 exposure-ambiguous trials excluded from any sensitivity",
        "inference_scope":"post-descriptive exploratory, two co-primary endpoints Holm-adjusted",
        "scientific_status":"NOT_YET_REVIEWED",
    }
    if args.preflight:
        # No inferential metrics printed during preflight.
        print("PREPLAN_EVIDENCE_REUSE_CHECK: PASS")
        print(json.dumps(inventory,indent=2,sort_keys=True))
        return 0
    if not args.run_analysis:
        raise InferenceStop("Specify --preflight or --run-analysis explicitly")
    output=args.output.resolve()
    if output==repo or repo in output.parents or output.exists() or not output.parent.is_dir():
        raise InferenceStop("New output directory required outside repository")
    # Compute primary tests only after all source checks pass.
    contrasts,category=category_contrasts(primary)
    continuous=continuous_mixed_model(primary)
    multiplicity=holm_two(category["p_unadjusted"],continuous["p_unadjusted"])
    # Sensitivity: category contrast only; not another primary family test.
    contrast_sensitivity=None;category_sensitivity=None
    if args.include_verified_sub012_sensitivity:
        contrast_sensitivity,category_sensitivity=category_contrasts(all_eligible)
    inventory["analysis_completed_utc"]=datetime.now(timezone.utc).isoformat()
    inventory["primary_source_rows_sha256"]=sha256(primary.to_csv(index=False).encode()).hexdigest()
    if args.include_verified_sub012_sensitivity:
        inventory["all_verified_rows_sha256"]=sha256(all_eligible.to_csv(index=False).encode()).hexdigest()
    with tempfile.TemporaryDirectory(dir=output.parent,prefix="kawaii_norminf_stage_") as temporary:
        folder=Path(temporary)
        contrasts.to_csv(folder/"NORM_INF_01_primary_participant_contrasts.csv",index=False)
        (folder/"NORM_INF_PRIMARY_RESULTS.json").write_text(json.dumps({
            "categorical":category,"continuous":continuous,"two_test_holm":multiplicity,
            "interpretation_status":"EXPLORATORY_UNREVIEWED", "no_causal_or_arousal_independent_claim":True,
        },indent=2,sort_keys=True)+"\n",encoding="utf-8")
        if contrast_sensitivity is not None:
            contrast_sensitivity.to_csv(folder/"NORM_INF_01_eight_source_verified_sensitivity_contrasts.csv",index=False)
            (folder/"NORM_INF_01_eight_sensitivity.json").write_text(
                json.dumps(category_sensitivity,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        inventory["output_sha256"]={f.name:_hash(f) for f in folder.iterdir() if f.is_file()}
        (folder/"SOURCE_AND_METHOD_MANIFEST.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        folder.rename(output)
    print("NORM_INF_ANALYSIS: COMPLETE_UNREVIEWED")
    print("OUTPUT:",output)
    print("No code, EEG, cohort reports or GitHub files were modified")
    return 0


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--aaps-reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,help="Fresh directory outside repo (required for --run-analysis)")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight",action="store_true")
    mode.add_argument("--run-analysis",action="store_true")
    parser.add_argument("--include-verified-sub012-sensitivity",action="store_true")
    args=parser.parse_args(argv)
    if args.run_analysis and args.output is None:
        parser.error("--run-analysis requires --output")
    try:
        return run(args)
    except (InferenceStop,ValueError,FileNotFoundError) as exc:
        print("NORM_INF_STOP:",exc,file=sys.stderr)
        return 3

if __name__=="__main__":
    raise SystemExit(main())
