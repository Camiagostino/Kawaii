"""Synthetic-only regressions for isolated NORM-02 descriptive aggregation."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from eeg_software.workflows.kawaii_session2.norm02_descriptive import (
    AcceptedPartTraces, Norm02ValidationError, accepted_roi_traces_uv,
    aggregate_descriptive, normative_category, subjective_category,
)

DIGEST = hashlib.sha256(b"synthetic source binding").hexdigest()
TIMES = np.linspace(-.2, 1.2, 8)
REFERENCE = {"1.jpg": 3.0, "2.jpg": 7.0}


def part(subject, part_name, images, ratings, amplitudes):
    n = len(images)
    return AcceptedPartTraces(subject, part_name, TIMES, np.tile(np.asarray(amplitudes, dtype=float)[:, None], (1, len(TIMES))),
        pd.DataFrame({"image_filename": images,
                      "M_Kawaii": [REFERENCE[im] for im in images],
                      "kawaii_rating": ratings,
                      "trial": np.arange(n) + 1,
                      "event_sample": (np.arange(n) + 1) * 1000}), DIGEST)


def test_two_score_groups_and_boundary_rejection():
    assert normative_category(5) == "lower_aaps"
    assert normative_category(6.50001) == "higher_aaps"
    for value in (5.01, 6.5, 0.5, np.nan, None):
        with pytest.raises(Norm02ValidationError): normative_category(value)


def test_subjective_four_bins():
    assert [subjective_category(i) for i in range(1,10)] == [
        "low","low","medium_low","medium_low","medium_high","medium_high","high","high","high"]
    for value in (0, 10, 2.2, "", None):
        with pytest.raises(Norm02ValidationError): subjective_category(value)


def test_generic_core_accepts_alternative_participant_and_image_ids():
    reference = {"stimulus-A.png": 3.0}
    n = 10
    item = AcceptedPartTraces(
        "participant_alpha", "part1", TIMES, np.zeros((n, len(TIMES))),
        pd.DataFrame({
            "image_filename": ["stimulus-A.png"] * n,
            "M_Kawaii": [3.0] * n,
            "kawaii_rating": [2] * n,
            "trial": np.arange(n) + 1,
            "event_sample": (np.arange(n) + 1) * 1000,
        }),
        DIGEST,
    )
    result = aggregate_descriptive([item], aaps_reference=reference)
    assert set(result.counts.participant) == {"participant_alpha"}
    assert int(result.summary.query('category == "lower_aaps"').iloc[0].retained_trials) == n


def test_equal_participant_weight_not_trial_weighted():
    # Participant A has twice as many trials, but B carries equal group weight.
    a = part("sub005", "part1", ["1.jpg"]*20, [1]*20, [0]*20)
    b = part("sub009", "part1", ["1.jpg"]*10, [2]*10, [12]*10)
    result = aggregate_descriptive([a,b], aaps_reference=REFERENCE)
    norm = result.waveforms.query('family == "normative" and category == "lower_aaps"')
    assert np.allclose(norm.mean_amplitude_uv, 6)
    summ = result.summary.query('family == "normative" and category == "lower_aaps"').iloc[0]
    assert int(summ.n_participants) == 2 and int(summ.retained_trials) == 30


def test_parts_weighted_by_nave_then_participant_equal():
    # sub005: one trial at 0, nine at 10 -> participant = 9; sub009: ten at 5 -> 5; mean 7
    result = aggregate_descriptive([
        part("sub005","part1",["2.jpg"],[9],[0]),
        part("sub005","part2",["2.jpg"]*9,[9]*9,[10]*9),
        part("sub009","part1",["2.jpg"]*10,[8]*10,[5]*10),
    ], aaps_reference=REFERENCE)
    hi=result.waveforms.query('family == "normative" and category == "higher_aaps"')
    assert np.allclose(hi.mean_amplitude_uv,7)
    assert int(result.summary.query('family == "normative" and category == "higher_aaps"').iloc[0].n_participants)==2


def test_sparse_participant_excluded_only_from_one_category():
    p1=part("sub010","part1",["1.jpg"]*8+["2.jpg"]*12,[1]*8+[9]*12,[10]*8+[2]*12)
    p2=part("sub011","part1",["1.jpg"]*10+["2.jpg"]*10,[1]*10+[9]*10,[4]*10+[8]*10)
    result=aggregate_descriptive([p1,p2],aaps_reference=REFERENCE)
    lo=result.summary.query('family == "subjective" and category == "low"').iloc[0]
    hi=result.summary.query('family == "subjective" and category == "high"').iloc[0]
    assert (int(lo.n_participants),int(lo.retained_trials),int(lo.excluded_trials))==(1,18,8)
    assert int(hi.n_participants)==2
    assert bool(result.counts.query('participant == "sub010" and category == "low"').iloc[0].eligible_for_group)==False


def test_all_retained_not_mixed_model_eligible_subset():
    p=part("sub012","part1",["1.jpg"]*10+["2.jpg"]*10,[3]*20,[1]*10+[7]*10)
    p.metadata["model_eligible_reduced"] = False
    result=aggregate_descriptive([p],aaps_reference=REFERENCE)
    assert result.counts.query('family == "normative"').retained_trials.sum()==20


def test_malformed_source_and_alignment_fail_closed():
    p=part("sub005","part1",["1.jpg"]*10,[1]*10,[0]*10)
    with pytest.raises(Norm02ValidationError,match="mismatch"):
        aggregate_descriptive([p],aaps_reference={"1.jpg":4.0})
    p.metadata.loc[1,"trial"] = 1
    with pytest.raises(Norm02ValidationError,match="duplicate"):
        aggregate_descriptive([p],aaps_reference=REFERENCE)
    p.metadata.loc[1,"trial"]=2
    with pytest.raises(Norm02ValidationError,match="Duplicate"):
        aggregate_descriptive([p,p],aaps_reference=REFERENCE)


def test_reject_inconsistent_time_grid():
    a=part("sub005","part1",["1.jpg"]*10,[1]*10,[0]*10)
    b=part("sub009","part1",["2.jpg"]*10,[9]*10,[1]*10)
    b=AcceptedPartTraces(b.participant,b.recording_part,TIMES+.001,b.traces_uv,b.metadata,b.source_binding_sha256)
    with pytest.raises(Norm02ValidationError,match="time grids"):
        aggregate_descriptive([a,b],aaps_reference=REFERENCE)


def test_mne_roi_conversion_no_mutation():
    mne=pytest.importorskip("mne")
    info=mne.create_info(["Cz","CP1","CP2","Pz"],sfreq=5,ch_types="eeg")
    signal=np.zeros((2,4,8))
    signal[:,:,2:]=2e-6
    epochs=mne.EpochsArray(signal,info,tmin=-.2,baseline=(-.2,0),verbose=False)
    old=epochs.get_data().copy()
    assert np.allclose(accepted_roi_traces_uv(epochs)[:,:2],0)
    assert np.allclose(accepted_roi_traces_uv(epochs)[:,2:],2)
    assert np.array_equal(epochs.get_data(),old)
    epochs.info["bads"]=["Cz"]
    with pytest.raises(Norm02ValidationError,match="bad channel"):
        accepted_roi_traces_uv(epochs)


def test_nan_amplitude_and_bad_digest_rejected():
    p=part("sub005","part1",["1.jpg"]*10,[1]*10,[0]*10)
    p.traces_uv[0,0]=np.nan
    with pytest.raises(Norm02ValidationError,match="Non-finite"):
        aggregate_descriptive([p],aaps_reference=REFERENCE)
    p=part("sub005","part1",["1.jpg"]*10,[1]*10,[0]*10)
    q=AcceptedPartTraces(p.participant,p.recording_part,p.times_s,p.traces_uv,p.metadata,"")
    with pytest.raises(Norm02ValidationError,match="binding digest"):
        aggregate_descriptive([q],aaps_reference=REFERENCE)


def test_roi_rejects_missing_baseline():
    mne=pytest.importorskip("mne")
    info=mne.create_info(["Cz","CP1","CP2","Pz"],sfreq=5,ch_types="eeg")
    epochs=mne.EpochsArray(np.ones((2,4,8))*1e-6,info,tmin=-.2,verbose=False)
    with pytest.raises(Norm02ValidationError,match="baseline"):
        accepted_roi_traces_uv(epochs)


def test_mne_interpolated_still_flagged_needs_authenticated_exemption():
    mne=pytest.importorskip("mne")
    info=mne.create_info(["Cz", "CP1", "CP2", "Pz"], sfreq=5, ch_types="eeg")
    epochs=mne.EpochsArray(np.ones((2,4,8))*1e-6, info, tmin=-.2, baseline=(-.2,0), verbose=False)
    epochs.info["bads"]=["CP2"]
    with pytest.raises(Norm02ValidationError,match="bad channel"):
        accepted_roi_traces_uv(epochs)
    original=epochs.get_data().copy()
    output=accepted_roi_traces_uv(epochs,authenticated_interpolated_roi=("CP2",))
    assert output.shape==(2,8)
    assert epochs.info["bads"]==["CP2"]
    assert np.array_equal(epochs.get_data(),original)
    with pytest.raises(Norm02ValidationError,match="do not match"):
        accepted_roi_traces_uv(epochs,authenticated_interpolated_roi=("Cz",))
