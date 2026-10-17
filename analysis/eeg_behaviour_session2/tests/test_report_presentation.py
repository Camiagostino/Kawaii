"""Synthetic presentation-only tests; no participant data, EEG, or model fitting."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.report_presentation import (
    render_report_figures, validate_erp, validate_coefficients,validate_slopes,
    SOURCE_SCHEMA,
)

CELLS=[('frontal','300_500_ms'),('frontocentral','300_500_ms'),('centroparietal','500_800_ms'),('parietal','500_800_ms')]


def _families(exploratory: bool) -> pd.DataFrame:
    comps=['hit_vs_false_alarm'] if exploratory else ['hit_vs_miss','false_alarm_vs_correct_rejection']
    n=4 if exploratory else 8
    return pd.DataFrame([{'comparison':c,'roi':r,'time_window':t,
        'effect_estimate':.1,'ci_low':-.2,'ci_high':.4,
        'raw_p':.9,'adjusted_p':1.0,
        'correction_family_size':n,'correction_method':'holm',
        'hypothesis_family':'recognition_true_vs_false_exploratory' if exploratory else 'recognition_secondary_memory_outcome',
        'analysis_status':'EXPLORATORY' if exploratory else 'SECONDARY',
        'confidence_level':.95} for c in comps for r,t in CELLS])


def _tables():
    subjects=[f'sub{n:03}' for n in (5,9,10,11,12,13,14,15)]
    rec=pd.DataFrame({'subject':subjects,'dprime':np.linspace(.1,.8,8),'hit_rate':[.6]*8,'false_alarm_rate':[.2]*8})
    primary=pd.DataFrame([{'model_id':'M1','term':'cuteness_centered','analysis_status':'PRIMARY','estimate':.073,'confidence_interval_low':.0003,'confidence_interval_high':.146,'p_value':.049}])
    diag=pd.DataFrame([{'model_id':'M1','n_observations':2752,'n_participants':8,'converged':True,'fit_warnings':'near-boundary variance','model_status':'FIT_WARNING'}])
    ar=pd.DataFrame([{'model_id':'M1','term':'arousal_centered','n_observations':2752,'n_participants':8,'converged':True,'refit_performed':False,'estimate_uv_per_centered_rating_unit':.017,'ci95_low':-.092,'ci95_high':.128,'p_value':.759,'fit_warnings':'near-boundary variance'}])
    slopes=pd.DataFrame({'participant':subjects,'eligible_n':[210]*8,'slope':np.linspace(-.1,.3,8),'ci_low':np.linspace(-.1,.3,8)-.2,'ci_high':np.linspace(-.1,.3,8)+.2,'direction':['negative']*2+['positive']*6})
    return {'recognition_participants':rec,'secondary_erp':_families(False),'exploratory_erp':_families(True),'primary_coefficients':primary,'primary_diagnostics':diag,'arousal_coefficient':ar,'participant_slopes':slopes}


@pytest.fixture
def source(tmp_path):
    src=tmp_path/'inputs';src.mkdir()
    entries={}
    for key,table in _tables().items():
        path=src/f'{key}.csv';table.to_csv(path,index=False)
        entries[key]={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    style=Path(__file__).resolve().parents[1]/'config'/'kawaii_session2_figure_style.yaml'
    shutil.copyfile(style,src/style.name)
    entries['style_config']={'path':style.name,'sha256':hashlib.sha256((src/style.name).read_bytes()).hexdigest()}
    manifest={'schema_version':SOURCE_SCHEMA,'scientific_data_status':'HISTORICAL_REFERENCE_NOT_CORRECTED','scientific_generation_id':'synthetic-not-a-scientific-run','inputs':entries}
    path=src/'sources.json';path.write_text(json.dumps(manifest))
    return path


def _adjust_manifest(path,change):
    m=json.loads(path.read_text());change(m);path.write_text(json.dumps(m))


def test_renders_four_png_and_svg_to_separate_dir(source, tmp_path):
    output=tmp_path/'new_report'
    meta=render_report_figures(source,output,allow_historical=True)
    assert len(meta['figures'])==8
    assert len(list(output.glob('*.png')))==4 and len(list(output.glob('*.svg')))==4
    assert (output/'S02_participant_cuteness_slopes.png').exists()
    assert meta['scientific_data_status']=='HISTORICAL_REFERENCE_NOT_CORRECTED'
    assert meta['scientific_model_refit'] is False
    assert 'near-boundary' in meta['model_warning']
    assert len(meta['source_input_sha256'])==8
    assert (output/'report_figure_manifest.json').exists()


def test_refuses_historical_fixture_as_corrected(source, tmp_path):
    with pytest.raises(ValueError,match='not approved corrected data'):
        render_report_figures(source,tmp_path/'figures')
    assert not (tmp_path/'figures').exists()


def test_rejects_tampered_hash_without_creating_outputs(source,tmp_path):
    (source.parent/'participant_slopes.csv').write_text('bad\n')
    with pytest.raises(ValueError,match='SHA256'):
        render_report_figures(source,tmp_path/'figures',allow_historical=True)
    assert not (tmp_path/'figures').exists()


def test_fails_closed_if_output_root_exists(source,tmp_path):
    path=tmp_path/'figs';path.mkdir()
    with pytest.raises(FileExistsError):render_report_figures(source,path,allow_historical=True)


def test_holm_families_must_be_separate_and_complete():
    df=_families(False);validate_erp(df,exploratory=False)
    df.at[0,'hypothesis_family']='recognition_true_vs_false_exploratory'
    with pytest.raises(ValueError,match='hypothesis_family'):validate_erp(df,exploratory=False)
    df=_families(True);df.loc[0,'adjusted_p']=.123
    with pytest.raises(ValueError,match='Holm-adjusted'):validate_erp(df,exploratory=True)
    df=_families(False);df.loc[0,'roi']='frontocentral'
    with pytest.raises(ValueError,match='keys|unique'):validate_erp(df,exploratory=False)


def test_rejects_arousal_from_different_model_or_refit():
    tab=_tables();p=tab['primary_coefficients'];d=tab['primary_diagnostics'];a=tab['arousal_coefficient']
    validate_coefficients(p,d,a)
    a.loc[0,'model_id']='M2'
    with pytest.raises(ValueError,match='model_id'):validate_coefficients(p,d,a)
    a.loc[0,'model_id']='M1';a.loc[0,'refit_performed']=True
    with pytest.raises(ValueError,match='refitted'):validate_coefficients(p,d,a)


def test_rejects_bad_slope_sign():
    df=_tables()['participant_slopes'];validate_slopes(df)
    df.loc[0,'direction']='positive'
    with pytest.raises(ValueError,match='direction'):validate_slopes(df)


def test_rejects_missing_validation_authority_for_corrected_run(source,tmp_path):
    _adjust_manifest(source,lambda m:m.update(scientific_data_status='R11_CORRECTED_VALIDATED'))
    with pytest.raises(ValueError,match='validation report'):render_report_figures(source,tmp_path/'figs')


def test_rejects_manifest_path_escape(source,tmp_path):
    def change(m):m['inputs']['style_config']['path']='../secret.yaml'
    _adjust_manifest(source,change)
    with pytest.raises(ValueError,match='relative'):render_report_figures(source,tmp_path/'figs',allow_historical=True)
