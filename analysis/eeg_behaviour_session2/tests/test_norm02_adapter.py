"""Bounded source-contract tests; no EEG recordings or participant files."""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
import sys
import types

import numpy as np
import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2 import norm02_adapter as adapter
from eeg_software.workflows.kawaii_session2.norm02_descriptive import AcceptedPartTraces, Norm02ValidationError
from eeg_software.workflows.kawaii_session2.publication import PublicationScope


def _publications(participant="sub005", stale=False, missing=False):
    gate3scope = PublicationScope(stage="gate3c", participant=participant, recording_part="part1")
    gate4scope = PublicationScope(stage="gate4b", branch="cuteness", participant=participant, recording_part="part1")
    arts = [SimpleNamespace(label="epochs_fif", path=Path("/safe/epochs.fif"), sha256="a"*64),
            SimpleNamespace(label="epoch_metadata_csv", path=Path("/safe/meta.csv"), sha256="b"*64),
            SimpleNamespace(label="interpolation_summary", path=Path("/safe/interp.csv"), sha256="d"*64)]
    if missing: arts.pop()
    a = SimpleNamespace(scope=gate3scope, generation_id="3"*64, context_reference="sha256:"+"c"*64, artifacts=arts)
    b = SimpleNamespace(scope=gate4scope, generation_id="4"*64,
         identity_references={f"immutable-publication:gate3c:{gate3scope.identity}": "5"*64 if stale else a.generation_id}, artifacts=[])
    return a, b


def test_resolves_exact_published_pair_without_alternate_source(tmp_path, monkeypatch):
    p3,p4 = _publications()
    calls=[]
    from eeg_software.workflows.kawaii_session2 import publication
    def fake_validator(path, scope):
        calls.append((Path(path), scope.stage))
        return p3 if scope.stage=="gate3c" else p4
    monkeypatch.setattr(publication,"validate_current_publication",fake_validator)
    one=adapter.resolve_published_sources(tmp_path,"sub005")
    assert one.epochs_fif==Path('/safe/epochs.fif')
    assert one.epoch_metadata_csv==Path('/safe/meta.csv')
    assert one.interpolation_summary_csv==Path('/safe/interp.csv')
    assert one.gate3c_generation=="3"*64 and one.gate4b_generation=="4"*64
    assert len(one.source_binding_sha256)==64
    assert len(calls)==2 and calls[0][0]==calls[1][0]
    assert "r11_qc_v3" in str(calls[0][0])


@pytest.mark.parametrize('stale,missing', [(True,False),(False,True)])
def test_refuses_stale_upstream_or_missing_artifact(tmp_path,monkeypatch,stale,missing):
    a,b=_publications(stale=stale,missing=missing)
    from eeg_software.workflows.kawaii_session2 import publication
    monkeypatch.setattr(publication,'validate_current_publication',lambda path,scope:a if scope.stage=='gate3c' else b)
    with pytest.raises(Norm02ValidationError): adapter.resolve_published_sources(tmp_path,'sub005')


def test_pub_validator_failure_propagates(tmp_path,monkeypatch):
    from eeg_software.workflows.kawaii_session2 import publication
    def fail(path,scope): raise publication.PublicationValidationError('invalid source bytes')
    monkeypatch.setattr(publication,'validate_current_publication',fail)
    with pytest.raises(publication.PublicationValidationError, match='source bytes'):
        adapter.resolve_published_sources(tmp_path,'sub005')


def test_no_original_or_stale_part_scope(tmp_path):
    with pytest.raises(Norm02ValidationError): adapter.corrected_part_root(tmp_path,'sub006')
    with pytest.raises(Norm02ValidationError): adapter.corrected_part_root(tmp_path,'sub005','part2')


def test_sub012_requires_conscious_exposure_decision(tmp_path):
    with pytest.raises(Norm02ValidationError,match='explicit disposition'):
        adapter.load_accepted_roi_part(tmp_path,'sub012',config={})


def _synthetic_reference_file(directory: Path) -> Path:
    """Create a deterministic 400-row test-only CSV; not the private AAPS crosswalk."""
    values = [4.0 + (i % 10) / 100 for i in range(200)] + [
        7.0 + (i % 10) / 100 for i in range(200)
    ]
    frame = pd.DataFrame({
        "image_filename": [f"synthetic_{i:03d}.jpg" for i in range(400)],
        "M_Kawaii": values,
    })
    path = directory / "synthetic_aaps_reference.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


@pytest.fixture
def synthetic_pinned_reference(tmp_path, monkeypatch):
    """Pin synthetic bytes only inside each test; production identity never changes."""
    path = _synthetic_reference_file(tmp_path)
    synthetic_digest = sha256(path.read_bytes()).hexdigest()
    assert synthetic_digest != adapter.PINNED_AAPS_REFERENCE_SHA256
    monkeypatch.setattr(adapter, "PINNED_AAPS_REFERENCE_SHA256", synthetic_digest)
    return path


def test_pinned_reference_accepts_only_400_200_200(synthetic_pinned_reference):
    r = adapter.load_aaps_reference(synthetic_pinned_reference)
    assert len(r) == 400
    assert sum(v <= 5 for v in r.values()) == 200
    assert sum(v > 6.5 for v in r.values()) == 200


def test_reference_rejects_duplicate_and_missing_and_wrong_schema(tmp_path, synthetic_pinned_reference):
    frame = pd.read_csv(synthetic_pinned_reference)
    f = tmp_path / "x.csv"
    frame.loc[1, "image_filename"] = frame.loc[0, "image_filename"]
    frame.to_csv(f, index=False)
    with pytest.raises(Norm02ValidationError, match="400 distinct"):
        adapter.load_aaps_reference(f)
    frame = frame.iloc[:-1]
    frame.to_csv(f, index=False)
    with pytest.raises(Norm02ValidationError, match="400 distinct"):
        adapter.load_aaps_reference(f)
    frame["wrong"] = 42
    frame.to_csv(f, index=False)
    with pytest.raises(Norm02ValidationError, match="exactly"):
        adapter.load_aaps_reference(f)


def test_reference_tampering_fails_even_when_group_counts_still_match(tmp_path, synthetic_pinned_reference):
    frame = pd.read_csv(synthetic_pinned_reference)
    frame.loc[0, "M_Kawaii"] = float(frame.loc[0, "M_Kawaii"]) + 0.01
    f = tmp_path / "changed.csv"
    frame.to_csv(f, index=False)
    with pytest.raises(Norm02ValidationError, match="pinned source"):
        adapter.load_aaps_reference(f)


def test_private_production_pin_rejects_synthetic_fixture_without_override(tmp_path):
    """Without the test-only pin override, an arbitrary 400-row CSV fails closed."""
    path = _synthetic_reference_file(tmp_path)
    with pytest.raises(Norm02ValidationError, match="pinned source"):
        adapter.load_aaps_reference(path)


def test_full_cohort_only_exact_eight_subjects():
    with pytest.raises(Norm02ValidationError,match='eight-participant'):
        adapter.validate_full_cohort({})
    items={}
    for s,n in adapter.EXPECTED_SUBJECT_COUNTS.items():
        items[s]=AcceptedPartTraces(s,'part1',np.arange(4),np.zeros((n,4)),pd.DataFrame(index=range(n)),'0'*64)
    adapter.validate_full_cohort(items)
    x=items['sub009']
    items['sub009']=AcceptedPartTraces(x.participant,x.recording_part,x.times_s,x.traces_uv,x.metadata.iloc[:-1],x.source_binding_sha256)
    with pytest.raises(Norm02ValidationError,match='retention'): adapter.validate_full_cohort(items)


def test_loader_reuses_authority_validators_and_keeps_rows(tmp_path,monkeypatch):
    # Controlled mock of the existing Gate 4 APIs: checks call routing and no row filtering.
    n=adapter.EXPECTED_SUBJECT_COUNTS['sub005']
    md=pd.DataFrame({'event_sample':range(1,n+1),'trial':range(1,n+1),
       'image_filename':['1.jpg']*n,'M_Kawaii':[3.0]*n,'kawaii_rating':[4]*n})
    class FakeEpochs:
        times=np.linspace(-.2,1.2,8)
        baseline=(-.2,0.0)
        ch_names=['Cz','CP1','CP2','Pz']
        info={'bads':[]}
        def __len__(self): return n
        def get_data(self,picks):
            assert picks==self.ch_names
            return np.ones((n,4,8))*2e-6
    seen=[]
    fakeerp=types.ModuleType('eeg_software.workflows.kawaii_session2.erp')
    fakeerp.load_gate3c_epochs_and_metadata=lambda paths:(seen.append('load') or FakeEpochs(),md.copy())
    fakeerp._validate_and_rebind_gate3c_epoch_metadata=lambda e,m,**kw:(seen.append('binding') or e,m,'identity')
    fakeerp.prepare_cuteness_inputs=lambda e,m,**kw:(seen.append('exposure') or e,SimpleNamespace(metadata=m,required=True))
    fakeerp.gate4b_cuteness_settings=lambda config:{'minimum_trials':{'low':10}}
    fakeerp.validate_epochs_metadata=lambda e,m,settings,**kw:seen.append('validation')
    fakeerp.apply_erp_baseline=lambda e,settings:seen.append('baseline_check')
    fakebad=types.ModuleType('eeg_software.workflows.kawaii_session2.bad_channel_disposition')
    fakebad.validate_downstream_retained_bad_channel_safety=lambda info,settings:seen.append('bad_channels')
    monkeypatch.setitem(sys.modules,fakeerp.__name__,fakeerp)
    monkeypatch.setitem(sys.modules,fakebad.__name__,fakebad)
    dummy=adapter.Norm02BoundSources('sub005','part1',Path('/safe/fif'),Path('/safe/csv'),'3'*64,'4'*64,'a'*64,'b'*64,'c'*64)
    monkeypatch.setattr(adapter,'resolve_published_sources',lambda r,s:dummy)
    x=adapter.load_accepted_roi_part(tmp_path,'sub005',config={})
    assert x.traces_uv.shape==(n,8) and np.allclose(x.traces_uv,2)
    assert list(x.metadata.columns)==list(md.columns)
    assert seen==['load','binding','exposure','bad_channels','validation','baseline_check']


# NORM-02: published interpolation disposition, never a mutable MNE bad-flag workaround.
def _interpolation_summary(path, *, channel="CP2", **overrides):
    from hashlib import sha256
    records=[]
    for name in ("Cz", "CP1", "CP2", "Pz"):
        flagged = name == channel
        record = dict(
            channel=name,
            manual_confirmed_bad_channel=flagged,
            interpolated=flagged,
            retained_bad_uninterpolated=False,
            interpolation_method="mne_interpolate_bads" if flagged else "",
            interpolation_geometry_safety_status="safe_unresolved_non_targets_excluded",
            bad_channel_decision_state="REVIEWED_WITH_BAD_CHANNELS",
            bad_channel_decision_id="sha256:" + "a"*64,
        )
        if flagged:
            record.update(overrides)
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)
    return sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("changed", [
    {"interpolated": False},
    {"retained_bad_uninterpolated": True},
    {"manual_confirmed_bad_channel": False},
    {"interpolation_method": "unknown"},
    {"interpolation_geometry_safety_status": "unsafe"},
    {"bad_channel_decision_state": "UNREVIEWED"},
    {"bad_channel_decision_id": "missing"},
])
def test_roi_interpolation_provenance_rejects_unsafe_disposition(tmp_path, changed):
    path=tmp_path/"summary.csv"
    digest=_interpolation_summary(path, **changed)
    source=SimpleNamespace(interpolation_summary_csv=path, interpolation_summary_sha256=digest)
    epochs=SimpleNamespace(info={"bads":["CP2"]})
    with pytest.raises(Norm02ValidationError, match="safe published interpolation disposition"):
        adapter.authenticated_interpolated_roi_channels(epochs, source)


def test_roi_interpolation_provenance_passes_only_authenticated_record(tmp_path):
    path=tmp_path/"summary.csv"
    digest=_interpolation_summary(path)
    source=SimpleNamespace(interpolation_summary_csv=path, interpolation_summary_sha256=digest)
    epochs=SimpleNamespace(info={"bads":["CP2"]})
    assert adapter.authenticated_interpolated_roi_channels(epochs, source)==("CP2",)
    # Once the source file changes, previously authenticated publication identity is stale.
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(Norm02ValidationError, match="hash mismatch"):
        adapter.authenticated_interpolated_roi_channels(epochs, source)


def test_roi_interpolation_missing_or_duplicate_records_fail(tmp_path):
    path=tmp_path/"summary.csv"
    digest=_interpolation_summary(path)
    source=SimpleNamespace(interpolation_summary_csv=path, interpolation_summary_sha256=digest)
    epochs=SimpleNamespace(info={"bads":["CP2"]})
    assert adapter.authenticated_interpolated_roi_channels(SimpleNamespace(info={"bads": []}), source)==()
    assert adapter.authenticated_interpolated_roi_channels(epochs,source)==("CP2",)
    frame=pd.read_csv(path)
    frame=frame.loc[frame.channel!="Cz"]
    frame.to_csv(path,index=False)
    source=SimpleNamespace(interpolation_summary_csv=path,interpolation_summary_sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest())
    with pytest.raises(Norm02ValidationError,match="missing/duplicate"):
        adapter.authenticated_interpolated_roi_channels(epochs,source)


def test_roi_interpolation_provenance_required(tmp_path):
    src=SimpleNamespace(interpolation_summary_csv=None,interpolation_summary_sha256="")
    with pytest.raises(Norm02ValidationError,match="Missing authenticated"):
        adapter.authenticated_interpolated_roi_channels(SimpleNamespace(info={"bads":["CP2"]}),src)
