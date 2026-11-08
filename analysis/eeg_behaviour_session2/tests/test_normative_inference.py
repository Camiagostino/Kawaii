"""Synthetic, data-free NORM-INF-01/02 contracts."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

CORE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0,str(CORE))
from eeg_software.workflows.kawaii_session2.normative_inference import (
    InferenceStop, category_contrasts, continuous_mixed_model, holm_two, validate_trials,
)


def fixture_trials(n_subjects=7,n_items=30,seed=21):
    rng=np.random.default_rng(seed)
    ids=[f"participant_{i:02d}" for i in range(n_subjects)]
    scores=np.r_[np.linspace(2,4,n_items//2),np.linspace(7,8,n_items-n_items//2)]
    rows=[]
    item_deviation=rng.normal(0,.12,n_items)
    for p,person in enumerate(ids):
        intercept=rng.normal(0,.3)
        for i,score in enumerate(scores):
            rows.append({
                "participant":person,"recording_part":"part1","trial":i+1,
                "event_sample":(i+1)*1000,"image_filename":f"img_{i:03d}.png",
                "M_Kawaii":float(score),
                "erp_mean_uv":float(0.4*(score-5)+intercept+item_deviation[i]+rng.normal(0,.3)),
                "verified_old_new":"old" if (i+p)%2 else "new",
                "source_eligible":True,"source_binding_sha256":f"{p+1:064x}",
            })
    return pd.DataFrame(rows)


def test_categorical_unit_is_participant_not_trial():
    data=fixture_trials()
    contrasts,result=category_contrasts(data)
    assert len(contrasts)==7
    assert result['n_participants']==7 and result['n_trials']==210
    assert result['n_items']==30 and result['min_cell']==10
    assert result['positive_participants']==7
    assert result['ci_95_uv'][0]<result['contrast_mean_uv']<result['ci_95_uv'][1]
    assert 0 <= result['signflip_p_sensitivity']<=1
    assert result['df']==6


def test_eligibility_fail_closed_and_no_silent_row_drop():
    data=fixture_trials()
    data.loc[0,'source_eligible']=False
    with pytest.raises(InferenceStop,match='non-eligible'):
        validate_trials(data)
    data=fixture_trials()
    data.loc[0,'verified_old_new']='unknown'
    with pytest.raises(InferenceStop,match='Unverified'):
        validate_trials(data)
    data=fixture_trials()
    data.loc[0,'image_filename']=data.loc[1,'image_filename']
    with pytest.raises(InferenceStop,match='same stimulus'):
        validate_trials(data)


def test_original_sub012_rejected_by_primary():
    data=fixture_trials()
    data.loc[:2,'participant']='sub012'
    with pytest.raises(InferenceStop,match='sub012'):
        validate_trials(data)


def test_minimum_cell_enforced():
    df=fixture_trials()
    df=df.drop(df[(df.participant=='participant_00') & (df.M_Kawaii<=5)].index[:8])
    with pytest.raises(InferenceStop,match='10-trial'):
        category_contrasts(df)


def test_holm_math():
    assert holm_two(.03,.01)['NORM-INF-01']==pytest.approx(.03)
    assert holm_two(.03,.01)['NORM-INF-02']==pytest.approx(.02)
    assert holm_two(.01,.9)['NORM-INF-02']==pytest.approx(.9)
    with pytest.raises(InferenceStop): holm_two(float('nan'),.1)


def test_continuous_crossed_model_synthetic():
    table=fixture_trials(n_subjects=7,n_items=30)
    outcome=continuous_mixed_model(table)
    assert outcome['converged'] is True
    assert outcome['n_trials']==210
    assert outcome['coefficient_uv_per_point']>0
    assert outcome['ci_95_uv_per_point'][0]<outcome['coefficient_uv_per_point']<outcome['ci_95_uv_per_point'][1]
    assert set(outcome['variance_components'])=={'image','participant'}
    assert outcome['formula']=='erp_mean_uv ~ aaps_c + C(verified_old_new)'


def test_source_loader_filters_exactly_190_ambiguous_and_no_label_imputation(monkeypatch):
    """Test the thin real-data adapter with accepted in-memory synthetic traces."""
    import types, importlib.util
    import sys
    module_name='eeg_software.workflows.kawaii_session2.norm02_adapter'
    dummy=types.ModuleType(module_name)
    dummy.EXPECTED_SUBJECT_COUNTS={'sub012':371}
    dummy.load_aaps_reference=lambda p:None
    dummy.load_accepted_roi_part=lambda *args,**kwargs:None
    monkeypatch.setitem(sys.modules,module_name,dummy)
    p=Path(__file__).resolve().parents[1]/'tools/run_kawaii_normative_inference.py'
    spec=importlib.util.spec_from_file_location('normative_runner_test',p)
    module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    n=371
    image_names=[f'img_{j%20:03d}.png' for j in range(n)]
    md=pd.DataFrame({
        'image_filename':image_names,'M_Kawaii':[3.0 if j%20<10 else 7.0 for j in range(n)],
        'trial':np.arange(1,n+1),'event_sample':np.arange(1,n+1)*1000,
        'continuous_cuteness_old_new_eligible':[True]*181+[False]*190,
        'source_verified_old_new_status':['old']*181+[pd.NA]*190,
        'source_exposure_status':['VERIFIED_OLD']*181+['AMBIGUOUS_RUN4_EXPOSURE']*190,
    })
    times=np.linspace(-.2,1.2,701)
    part=types.SimpleNamespace(participant='sub012',recording_part='part1',metadata=md,
                              times_s=times,traces_uv=np.ones((n,len(times))),
                              source_binding_sha256='a'*64)
    reference={name:(3.0 if j<10 else 7.0) for j,name in enumerate(sorted(set(image_names)))}
    # Use each filename's score from the source metadata, independent of sort order.
    reference=md.drop_duplicates('image_filename').set_index('image_filename')['M_Kawaii'].to_dict()
    rows,record=module.part_to_rows(part,reference=reference,allow_ambiguous=True)
    assert len(rows)==181
    assert record['excluded_source_ineligible']==190
    assert rows.verified_old_new.eq('old').all()
    with pytest.raises(InferenceStop,match='outside declared'):
        module.part_to_rows(part,reference=reference,allow_ambiguous=False)
    md.loc[200,'source_exposure_status']='VERIFIED_NEW'
    with pytest.raises(InferenceStop,match='outside declared'):
        module.part_to_rows(part,reference=reference,allow_ambiguous=True)


def test_source_eligibility_string_false_rejected(monkeypatch):
    # Regression: bool('False') is True, so coercion is forbidden.
    import types, importlib.util,sys
    name='eeg_software.workflows.kawaii_session2.norm02_adapter'
    dummy=types.ModuleType(name)
    dummy.EXPECTED_SUBJECT_COUNTS={}
    dummy.load_aaps_reference=lambda p:None
    dummy.load_accepted_roi_part=lambda *args,**kwargs:None
    monkeypatch.setitem(sys.modules,name,dummy)
    p=Path(__file__).resolve().parents[1]/'tools/run_kawaii_normative_inference.py'
    spec=importlib.util.spec_from_file_location('normative_runner_test_string',p)
    module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    with pytest.raises(InferenceStop,match='genuine Boolean'):
        module._boolean(pd.Series(['False','True']))
