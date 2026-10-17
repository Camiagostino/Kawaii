"""Render presentation-only Kawaii report figures from derived, versioned outputs.

This module never imports EEG epochs or scientific modelling functions. An
independent scientific/data validation is mandatory; the input manifest alone
is a chain-of-custody assertion, NOT evidence that the R11 replay was valid.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure_style import load_figure_style, style_axes, FigureStyle

INPUT_KEYS = (
    "recognition_participants", "secondary_erp", "exploratory_erp",
    "primary_coefficients", "primary_diagnostics", "arousal_coefficient",
    "participant_slopes", "style_config",
)
SOURCE_SCHEMA = "kawaii_report_figure_source_manifest_v1"
OUTPUT_SCHEMA = "kawaii_report_figure_generation_v1"
CELLS = (("frontal", "300_500_ms"), ("frontocentral", "300_500_ms"),
         ("centroparietal", "500_800_ms"), ("parietal", "500_800_ms"))
PRIMARY_MODEL_TERM = "cuteness_centered"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def _finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: nonfinite/nonnumeric value")


def _bool_flag(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name}: expected explicit Boolean, found {value!r}")


def _source_manifest(path: Path, *, allow_historical: bool) -> tuple[dict[str, Any], dict[str, Path]]:
    source = json.loads(path.read_text(encoding="utf8"))
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("Unsupported report source manifest schema")
    status = source.get("scientific_data_status")
    if status == "HISTORICAL_REFERENCE_NOT_CORRECTED" and allow_historical:
        pass
    elif status != "R11_CORRECTED_VALIDATED":
        raise ValueError("Source manifest is not approved corrected data; historical fixtures require explicit flag")
    if not str(source.get("scientific_generation_id", "")).strip():
        raise ValueError("Missing scientific generation identity")
    if status == "R11_CORRECTED_VALIDATED" and not str(source.get("r11_validation_report_id", "")).strip():
        raise ValueError("Missing independently approved R11 validation report identity")
    entries = source.get("inputs")
    if not isinstance(entries, dict) or set(entries) != set(INPUT_KEYS):
        raise ValueError("Source manifest must declare exactly the eight report inputs")
    base = path.parent.resolve()
    paths: dict[str, Path] = {}
    for key in INPUT_KEYS:
        entry = entries[key]
        if not isinstance(entry, dict):
            raise ValueError(f"{key}: invalid manifest entry")
        name = Path(str(entry.get("path", "")))
        if name.is_absolute() or not str(name) or ".." in name.parts:
            raise ValueError(f"{key}: source paths must be relative and remain within the source root")
        resolved = (base / name).resolve()
        if not resolved.is_relative_to(base) or not resolved.is_file():
            raise ValueError(f"{key}: input outside source root or missing")
        declared = str(entry.get("sha256", "")).lower()
        if len(declared) != 64 or any(c not in "0123456789abcdef" for c in declared):
            raise ValueError(f"{key}: invalid declared sha256")
        if _sha(resolved) != declared:
            raise ValueError(f"{key}: source SHA256 does not match the manifest")
        paths[key] = resolved
    return source, paths


def validate_recognition(frame: pd.DataFrame) -> pd.DataFrame:
    name="recognition"
    _require_columns(frame,{"subject","dprime","hit_rate","false_alarm_rate"},name)
    if len(frame)!=8 or frame.subject.isna().any() or frame.subject.duplicated().any():
        raise ValueError("Recognition must have eight unique participants")
    _finite(frame,["dprime","hit_rate","false_alarm_rate"],name)
    if not frame[["hit_rate","false_alarm_rate"]].apply(pd.to_numeric).ge(0).all().all() or not frame[["hit_rate","false_alarm_rate"]].apply(pd.to_numeric).le(1).all().all():
        raise ValueError("Invalid recognition rates")
    return frame.sort_values("subject").reset_index(drop=True)


def _holm(raw: pd.Series) -> np.ndarray:
    vals = raw.to_numpy(dtype=float)
    if not np.isfinite(vals).all() or np.any(vals < 0) or np.any(vals > 1):
        raise ValueError("Invalid raw p values")
    order = np.argsort(vals, kind="mergesort")
    corrected = np.empty(len(vals), dtype=float)
    corrected[order] = np.minimum(1, np.maximum.accumulate((len(vals)-np.arange(len(vals))) * vals[order]))
    return corrected


def validate_erp(frame: pd.DataFrame, *, exploratory: bool) -> pd.DataFrame:
    name="exploratory ERP" if exploratory else "secondary ERP"
    _require_columns(frame,{"comparison","roi","time_window","effect_estimate","ci_low","ci_high","adjusted_p","raw_p","correction_family_size","correction_method","hypothesis_family","analysis_status","confidence_level"},name)
    expected_comps={"hit_vs_false_alarm"} if exploratory else {"hit_vs_miss","false_alarm_vs_correct_rejection"}
    expected_size=4 if exploratory else 8
    expected_family="recognition_true_vs_false_exploratory" if exploratory else "recognition_secondary_memory_outcome"
    expected_status="EXPLORATORY" if exploratory else "SECONDARY"
    if len(frame)!=expected_size or frame[["comparison","roi","time_window"]].duplicated().any():
        raise ValueError(f"{name}: wrong number of unique comparisons")
    actual={(str(r.comparison),str(r.roi),str(r.time_window)) for r in frame.itertuples()}
    expected={(c,r,t) for c in expected_comps for r,t in CELLS}
    if actual!=expected:
        raise ValueError(f"{name}: missing or unexpected ROI/window/comparison keys")
    for col, expected_value in (("correction_family_size",expected_size),("correction_method","holm"),("hypothesis_family",expected_family),("analysis_status",expected_status)):
        if not frame[col].eq(expected_value).all():
            raise ValueError(f"{name}: invalid {col}")
    if not pd.to_numeric(frame["confidence_level"],errors="coerce").eq(.95).all():
        raise ValueError(f"{name}: confidence interval must be 95%, unadjusted")
    _finite(frame,["effect_estimate","ci_low","ci_high","adjusted_p","raw_p"],name)
    if not ((frame.ci_low<=frame.effect_estimate)&(frame.effect_estimate<=frame.ci_high)).all():
        raise ValueError(f"{name}: effect not contained in interval")
    if not np.allclose(frame["adjusted_p"].to_numpy(float),_holm(frame["raw_p"]),rtol=0,atol=2e-7):
        raise ValueError(f"{name}: Holm-adjusted p values do not match declared separate family")
    return frame.copy()


def validate_coefficients(primary: pd.DataFrame, diagnostics: pd.DataFrame, arousal: pd.DataFrame) -> tuple[pd.Series,pd.Series,str]:
    _require_columns(primary,{"model_id","term","analysis_status","estimate","confidence_interval_low","confidence_interval_high","p_value"},"primary model")
    _require_columns(diagnostics,{"model_id","n_observations","n_participants","converged","fit_warnings","model_status"},"primary diagnostics")
    _require_columns(arousal,{"model_id","term","n_observations","n_participants","refit_performed","converged","estimate_uv_per_centered_rating_unit","ci95_low","ci95_high","p_value","fit_warnings"},"arousal")
    p=primary.loc[primary.term.eq(PRIMARY_MODEL_TERM)&primary.analysis_status.eq("PRIMARY")]
    if len(p)!=1 or len(arousal)!=1 or len(diagnostics)!=1:
        raise ValueError("Expected one primary cuteness coefficient, one arousal coefficient and one diagnostic row")
    pr=p.iloc[0];ar=arousal.iloc[0];dg=diagnostics.iloc[0]
    mid=str(pr.model_id)
    if not mid or any(str(row.model_id)!=mid for row in (dg,ar)):
        raise ValueError("Primary, arousal and diagnostic source model_id mismatch")
    if str(ar.term)!="arousal_centered":
        raise ValueError("Arousal term identity mismatch")
    if int(dg.n_participants)!=8 or int(ar.n_participants)!=8 or int(dg.n_observations)!=int(ar.n_observations):
        raise ValueError("The two model coefficients are not from the same participant/trial set")
    if not _bool_flag(dg.converged,'primary converged') or not _bool_flag(ar.converged,'arousal converged') or _bool_flag(ar.refit_performed,'arousal refit_performed'):
        raise ValueError("Model failed convergence or arousal coefficient was refitted")
    if str(dg.model_status) not in {"FIT_WARNING", "FIT_OK"}:
        raise ValueError("Primary diagnostics do not identify an accepted fit")
    values=[(pr.estimate,pr.confidence_interval_low,pr.confidence_interval_high,pr.p_value),(ar.estimate_uv_per_centered_rating_unit,ar.ci95_low,ar.ci95_high,ar.p_value)]
    for estimate,lo,hi,pv in values:
        if not all(np.isfinite(float(v)) for v in (estimate,lo,hi,pv)) or not float(lo)<=float(estimate)<=float(hi) or not 0<=float(pv)<=1:
            raise ValueError("Invalid coefficient estimate/CI/p")
    warning="; ".join(w for w in (str(dg.fit_warnings),str(ar.fit_warnings)) if w and w.lower() not in {"nan","none"})
    return pr,ar,warning


def validate_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    name="participant slopes"
    _require_columns(frame,{"participant","eligible_n","slope","ci_low","ci_high","direction"},name)
    if len(frame)!=8 or frame.participant.isna().any() or frame.participant.duplicated().any():
        raise ValueError("Expected eight unique participant slopes")
    _finite(frame,["eligible_n","slope","ci_low","ci_high"],name)
    if not (frame.eligible_n>0).all() or not ((frame.ci_low<=frame.slope)&(frame.slope<=frame.ci_high)).all():
        raise ValueError("Invalid slope eligible N or confidence interval")
    expected = frame.slope.map(lambda x:"positive" if x>0 else "negative" if x<0 else "zero")
    if not expected.eq(frame.direction).all():
        raise ValueError("Participant-slope direction mismatch")
    return frame.sort_values("participant").reset_index(drop=True)


def _save(fig, path: Path, style: FigureStyle) -> dict[str, Path]:
    target={ext:path.with_suffix('.'+ext) for ext in ('png','svg')}
    for out in target.values():
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite figure: {out}")
    fig.savefig(target['png'],dpi=style.dpi,facecolor=style.figure_facecolor,bbox_inches='tight')
    fig.savefig(target['svg'],facecolor=style.figure_facecolor,bbox_inches='tight')
    plt.close(fig)
    return target


def _f01(frame: pd.DataFrame, base: Path, style: FigureStyle) -> dict[str,Path]:
    frame=validate_recognition(frame)
    x=np.arange(len(frame));colors=style.colors['recognition_conditions'];f=style.fonts
    fig,(left,right)=plt.subplots(1,2,figsize=(11.,4.25),gridspec_kw={'width_ratios':[.92,1.25]},facecolor=style.figure_facecolor)
    left.bar(x,frame.dprime,width=.65,color=colors['hit'],alpha=.88)
    left.axhline(0,color=style.colors['reference'],lw=.75)
    left.axhline(float(frame.dprime.mean()),color=colors['hit_minus_correct_rejection'],ls='--',lw=1)
    left.set_xticks(x,[str(s).replace('sub','') for s in frame.subject]);left.set_ylabel("Sensitivity (d′)")
    left.set_title('A  Recognition sensitivity',loc='left',fontsize=float(f['panel_title_size']));style_axes(left,style)
    left.text(.03,.97,f"Mean d′ = {frame.dprime.mean():.2f}",transform=left.transAxes,va='top',fontsize=f['tick_label_size'])
    width=.36
    right.bar(x-width/2,frame.hit_rate,width,color=colors['hit'],label='Hit rate')
    right.bar(x+width/2,frame.false_alarm_rate,width,color=colors['false_alarm'],label='False-alarm rate')
    right.set_ylim(0,1.03);right.set_xticks(x,[str(s).replace('sub','') for s in frame.subject]);right.set_ylabel('Proportion of trials')
    right.set_title('B  Recognition response rates',loc='left',fontsize=f['panel_title_size']);style_axes(right,style)
    right.legend(loc='upper center',ncol=2,frameon=False,fontsize=f['legend_size'])
    fig.supxlabel('Participant',fontsize=f['axis_label_size'],y=.025);fig.subplots_adjust(left=.075,right=.985,top=.86,bottom=.15,wspace=.32)
    return _save(fig,base/'F01_participant_recognition_performance',style)


def _f02(sec: pd.DataFrame, exp: pd.DataFrame, base:Path, style:FigureStyle) -> dict[str,Path]:
    sec=validate_erp(sec,exploratory=False);exp=validate_erp(exp,exploratory=True)
    colors=style.colors['recognition_conditions'];fonts=style.fonts
    ymin=min(float(d.ci_low.min()) for d in (sec,exp));ymax=max(float(d.ci_high.max()) for d in (sec,exp));span=max(abs(ymin),abs(ymax),.01)*1.13
    fig,(a,b)=plt.subplots(2,1,figsize=(10.,7.8),gridspec_kw={'height_ratios':[1.8,1]},facecolor=style.figure_facecolor)
    labels={"frontal":"Frontal","frontocentral":"Frontocentral","centroparietal":"Centroparietal","parietal":"Parietal"}
    for ax,df,primary,title in ((a,sec,True,'A  Secondary comparisons · Holm family of 8 tests'),(b,exp,False,'B  Exploratory comparison · Holm family of 4 tests')):
        rowlabels=[]
        for roi,tm in CELLS:
            lab=f"{labels[roi]} · {'300–500 ms' if '300' in tm else '500–800 ms'}"
            rowlabels.extend([lab+'  |  H − M','  |  FA − CR'] if primary else [lab])
        for row in df.itertuples(index=False):
            cell=CELLS.index((row.roi,row.time_window))
            which=0 if row.comparison=='hit_vs_miss' else 1
            index=(2*cell+which) if primary else cell
            y=len(rowlabels)-1-index
            color=(colors['hit'] if which==0 else colors['false_alarm']) if primary else colors['miss']
            ax.errorbar(float(row.effect_estimate),y,xerr=[[float(row.effect_estimate-row.ci_low)],[float(row.ci_high-row.effect_estimate)]],fmt='o',ms=5,capsize=3,color=color,lw=1.6)
            ax.text(1.02,y,f'{float(row.adjusted_p):.3f}',transform=ax.get_yaxis_transform(),va='center',fontsize=fonts['tick_label_size'])
        ax.axvline(0,color=style.colors['reference'],lw=.85,ls='--');ax.set_xlim(-span,span)
        ax.set_yticks(list(range(len(rowlabels))),list(reversed(rowlabels)));ax.set_ylim(-.6,len(rowlabels)-.35)
        ax.set_title(title,loc='left',fontsize=fonts['panel_title_size']);style_axes(ax,style)
        ax.text(1.015,1.055,'Holm p',transform=ax.transAxes,ha='left',fontsize=fonts['tick_label_size'])
    a.tick_params(axis='x',labelbottom=False);b.set_xlabel('Mean ERP amplitude difference (µV); unadjusted 95% CIs')
    fig.subplots_adjust(left=.29,right=.88,top=.94,bottom=.105,hspace=.36)
    return _save(fig,base/'F02_secondary_recognition_erp_forest',style)


def _f03(primary:pd.DataFrame, diagnostics:pd.DataFrame, arousal:pd.DataFrame, base:Path,style:FigureStyle)->dict[str,Path]:
    pr,ar,warnings=validate_coefficients(primary,diagnostics,arousal)
    fig,ax=plt.subplots(figsize=(8.2,3.1),facecolor=style.figure_facecolor)
    colors=style.colors['recognition_conditions']
    values=[(float(pr.estimate),float(pr.confidence_interval_low),float(pr.confidence_interval_high),float(pr.p_value)),
            (float(ar.estimate_uv_per_centered_rating_unit),float(ar.ci95_low),float(ar.ci95_high),float(ar.p_value))]
    for y,(x,lo,hi,p),color in zip([1,0],values,[colors['hit'],colors['false_alarm']]):
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',markersize=7,color=color,capsize=4,lw=2)
        ax.text(1.045,y,f'{p:.3f}',transform=ax.get_yaxis_transform(),va='center',fontsize=style.fonts['tick_label_size'])
    ax.axvline(0,color=style.colors['reference'],ls='--',lw=.8)
    ax.set_yticks([1,0],['Cuteness','Arousal (covariate)']);ax.set_ylim(-.55,1.55)
    ax.set_xlabel('ERP amplitude change per centred rating point (µV); unadjusted 95% CIs')
    ax.set_title('Cuteness and arousal coefficients from the same mixed model',loc='left',fontsize=style.fonts['panel_title_size'])
    style_axes(ax,style);ax.text(1.015,1.02,'Unadjusted p',transform=ax.transAxes,fontsize=style.fonts['tick_label_size'])
    fig.subplots_adjust(left=.25,right=.85,top=.79,bottom=.26)
    return _save(fig,base/'F03_cuteness_arousal_model_coefficients',style)


def _s02(frame:pd.DataFrame,base:Path,style:FigureStyle)->dict[str,Path]:
    data=validate_slopes(frame)
    fig,ax=plt.subplots(figsize=(7.8,3.55),facecolor=style.figure_facecolor)
    yp=np.arange(len(data))[::-1]
    for y,row in zip(yp,data.itertuples(index=False)):
        col=style.colors['recognition_conditions']['hit' if float(row.slope)>=0 else 'correct_rejection']
        ax.errorbar(float(row.slope),y,xerr=[[float(row.slope-row.ci_low)],[float(row.ci_high-row.slope)]],fmt='o',markersize=5.4,capsize=2.8,elinewidth=1.45,color=col)
    ax.axvline(0,color=style.colors['reference'],ls='--',lw=.8)
    ax.set_yticks(yp,[str(p).replace('sub','') for p in data.participant]);ax.set_ylim(-.55,len(data)-.38)
    lo=float(data.ci_low.min());hi=float(data.ci_high.max());ax.set_xlim(lo-.055,hi+.09)
    ax.set_xlabel('Descriptive within-participant cuteness slope (µV/rating point); unadjusted 95% CIs')
    ax.set_ylabel('Participant');ax.set_title('Participant-level cuteness–ERP associations',loc='left',fontsize=style.fonts['panel_title_size'])
    style_axes(ax,style);fig.subplots_adjust(left=.14,right=.982,top=.88,bottom=.195)
    # Deliberately S02: avoids collision with existing S01 recognition ROI-waveform figure.
    return _save(fig,base/'S02_participant_cuteness_slopes',style)


def render_report_figures(manifest_path:Path,output_dir:Path,*,allow_historical:bool=False)->dict[str,Any]:
    """Validate all derived inputs, then render four plots into an absent output root.

    Only the *separately approved* corrected-data manifest may be used for
    collaborator release. HISTORICAL_REFERENCE_NOT_CORRECTED is test-only.
    """
    manifest_path=Path(manifest_path).resolve();output_dir=Path(output_dir).resolve()
    source,paths=_source_manifest(manifest_path,allow_historical=allow_historical)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to populate existing report namespace: {output_dir}")
    if output_dir==manifest_path.parent or output_dir.is_relative_to(manifest_path.parent):
        raise ValueError("Report output must not be within the immutable input tree")
    # Fail BEFORE writing any figures.
    a=validate_recognition(pd.read_csv(paths['recognition_participants']))
    b=validate_erp(pd.read_csv(paths['secondary_erp']),exploratory=False)
    c=validate_erp(pd.read_csv(paths['exploratory_erp']),exploratory=True)
    d=pd.read_csv(paths['primary_coefficients']);e=pd.read_csv(paths['primary_diagnostics']);f=pd.read_csv(paths['arousal_coefficient'])
    _,_,warnings=validate_coefficients(d,e,f)
    g=validate_slopes(pd.read_csv(paths['participant_slopes']))
    style=load_figure_style(root=manifest_path.parent,config_path=paths['style_config'])
    output_dir.mkdir(parents=True,exist_ok=False)
    made:dict[str,Path]={}
    try:
        for render_args in ((_f01,(a,output_dir,style)),(_f02,(b,c,output_dir,style)),(_f03,(d,e,f,output_dir,style)),(_s02,(g,output_dir,style))):
            result=render_args[0](*render_args[1]);made.update({p.name:p for p in result.values()})
        output_manifest={
          'schema_version':OUTPUT_SCHEMA,
          'scientific_data_status':source['scientific_data_status'],
          'scientific_generation_id':source['scientific_generation_id'],
          'r11_validation_report_id':source.get('r11_validation_report_id'),
          'source_manifest_sha256':_sha(manifest_path),
          'source_input_sha256':{key:_sha(path) for key,path in paths.items()},
          'plotting_only':True,'scientific_model_refit':False,
          'model_warning':warnings,
          'figures':{key:_sha(path) for key,path in sorted(made.items())},
          'collaborator_release_approval':'NOT_GRANTED_BY_THIS_RUN',
          'note':'S02 renumbers the handover S01 participant-slope plot to avoid an existing waveform S01.'
        }
        (output_dir/'report_figure_manifest.json').write_text(json.dumps(output_manifest,indent=2,sort_keys=True)+'\n',encoding='utf8')
        return output_manifest
    except Exception:
        # Never delete output data opportunistically: inspect partial scratch separately.
        raise
