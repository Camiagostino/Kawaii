"""Synthetic-only NORM-02 exploratory output and decision guards."""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import zipfile

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from tools.run_kawaii_norm02_exploratory import (
    validate_exploratory_counts, verify_sub012_metadata, write_outputs, verify_source_checkout_head,
)
from eeg_software.workflows.kawaii_session2.norm02_descriptive import Norm02ValidationError


def fake_result(n8=True):
    norm = {"lower_aaps": 1482,"higher_aaps":1486} if n8 else {"lower_aaps":1299,"higher_aaps":1298}
    sub = {"low":492,"medium_low":622,"medium_high":731,"high":1123} if n8 else {"low":390,"medium_low":551,"medium_high":640,"high":1016}
    rows=[]
    counts=[]
    waves=[]
    for family, d in (("normative",norm),("subjective",sub)):
        for category,n in d.items():
            participants=7 if n8 and family=="subjective" and category=="low" else (8 if n8 else 7)
            rows.append(dict(family=family,category=category,n_participants=participants,
                             retained_trials=n,contributed_trials=n-8 if n8 and family=="subjective" and category=="low" else n,
                             excluded_trials=8 if n8 and family=="subjective" and category=="low" else 0))
            for time in (-0.2,0.0,0.5,0.8,1.2):
                waves.append(dict(family=family,category=category,time_s=time,mean_amplitude_uv=float(time),n_participants=participants))
    counts.append(dict(family="subjective",category="low",participant="sub010",retained_trials=8,eligible_for_group=False))
    return SimpleNamespace(summary=pd.DataFrame(rows),counts=pd.DataFrame(counts),
                           waveforms=pd.DataFrame(waves),contributions=pd.DataFrame([dict(x=1)]))


def test_full_expected_counts_and_sparse_contribution():
    validate_exploratory_counts(fake_result(), full=True)


def test_wrong_counts_stop_before_export():
    result=fake_result()
    result.summary.loc[result.summary.category=="lower_aaps","retained_trials"]+=1
    with pytest.raises(Norm02ValidationError,match="category counts changed"):
        validate_exploratory_counts(result, full=True)


def test_ineligible_sub010_low_required():
    result=fake_result()
    result.counts.loc[0,"eligible_for_group"]=True
    with pytest.raises(Norm02ValidationError,match="exclusion was not preserved"):
        validate_exploratory_counts(result, full=True)


def test_sub012_ambiguous_split_and_mismatch():
    values=[3.5]*95+[7.5]*95+[3.5]*88+[7.5]*93
    states=["AMBIGUOUS_RUN4_EXPOSURE"]*190+["KNOWN"]*181
    p=SimpleNamespace(metadata=pd.DataFrame({"source_exposure_status":states,"M_Kawaii":values}))
    verify_sub012_metadata(p)
    p.metadata.loc[0,"source_exposure_status"]="KNOWN"
    with pytest.raises(Norm02ValidationError,match="split changed"):
        verify_sub012_metadata(p)


def test_figure_csv_manifest_and_archive_synthetic(tmp_path):
    output=tmp_path/"KAWAII_NORM02_EXPLORATORY_QC_TEST"
    archive=write_outputs(fake_result(),fake_result(False),{"sub005":"a"*64},output,cohort_head="f"*40)
    assert archive.exists()
    assert len(list(output.iterdir()))==11  # 2 x (four csv + one figure), plus manifest
    manifest=json.loads((output/"MANIFEST.json").read_text())
    assert manifest["scientific_status"]=="EXPLORATORY_NOT_APPROVED_FOR_COLLABORATOR_REPORT"
    assert len(manifest["file_sha256"])==10
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert "MANIFEST.json" in z.namelist()
    with pytest.raises(FileExistsError):
        write_outputs(fake_result(),fake_result(False),{},output,cohort_head="f"*40)


def test_missing_waveform_rejected():
    result=fake_result()
    result.waveforms=result.waveforms.loc[result.waveforms.category!="low"]
    # No waveforms for Low but summary has it, must not silently export
    # Future strict output validation is performed in draw_figure.
    from tools.run_kawaii_norm02_exploratory import draw_figure
    with pytest.raises(Norm02ValidationError,match="No waveform"):
        draw_figure(result,Path("unused.png"),full=True)


def test_erp_window_shading_uses_milliseconds(tmp_path, monkeypatch):
    """Axes use milliseconds: the 500–800 ms highlight must not be 0.5–0.8."""
    from matplotlib.axes import Axes
    from tools.run_kawaii_norm02_exploratory import draw_figure

    called = []
    original_axvspan = Axes.axvspan

    def record_window(self, xmin, xmax, *args, **kwargs):
        called.append((xmin, xmax))
        return original_axvspan(self, xmin, xmax, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvspan", record_window)
    output = tmp_path / "ms_axis_window.png"
    draw_figure(fake_result(), output, full=True)
    assert output.is_file()
    assert called == [(500, 800), (500, 800)]


def test_figure_footer_clears_xlabels(tmp_path, monkeypatch):
    """Rendered status footer must not overlap axis captions in both cohorts."""
    from matplotlib.figure import Figure
    from tools.run_kawaii_norm02_exploratory import draw_figure

    captured = []
    actual_savefig = Figure.savefig

    def verify_layout(self, *args, **kwargs):
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        footers = [t for t in self.texts if
                   t.get_text().startswith("EXPLORATORY / NOT SCIENTIFICALLY APPROVED")]
        assert len(footers) == 1
        foot_box = footers[0].get_window_extent(renderer)
        assert 0 < foot_box.y0 < foot_box.y1 < self.bbox.height
        assert len(self.axes) == 2
        for ax in self.axes:
            label_box = ax.xaxis.label.get_window_extent(renderer)
            assert foot_box.y1 < label_box.y0 - 3, "Footer overlaps x-axis label"
        captured.append(True)
        return actual_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", verify_layout)
    for full in (True, False):
        path = tmp_path / ("eight.png" if full else "seven.png")
        draw_figure(fake_result(full), path, full=full)
        assert path.exists()
    assert captured == [True, True]


def test_source_checkout_head_requires_exact_current_revision(tmp_path, monkeypatch):
    def fake_git(command, **kwargs):
        assert command == ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]
        assert kwargs["check"] is True
        return SimpleNamespace(stdout="a" * 40 + "\n")

    monkeypatch.setattr(subprocess, "run", fake_git)
    verify_source_checkout_head(tmp_path, "a" * 40)
    with pytest.raises(Norm02ValidationError, match="differs from the actual Git checkout"):
        verify_source_checkout_head(tmp_path, "b" * 40)


def test_source_checkout_head_fails_closed_if_git_is_unavailable(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(subprocess, "run", no_git)
    with pytest.raises(Norm02ValidationError, match="Cannot verify source repository Git HEAD"):
        verify_source_checkout_head(tmp_path, "a" * 40)
