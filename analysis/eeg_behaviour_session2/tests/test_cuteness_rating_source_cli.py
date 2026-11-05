"""Focused, fixture-independent tests for the report-only rating-source selector."""
from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/run_kawaii_cuteness_report_figures.py'
spec = importlib.util.spec_from_file_location('kawaii_cuteness_report_cli', SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _archive(tmp_path):
    path = tmp_path / 'normative.zip'
    times = np.linspace(-.2, 1.2, 701)
    rows = []
    for cat, level in [('lower_aaps', .1), ('higher_aaps', .7)]:
        rows.extend(dict(family='normative', category=cat, time_s=float(t),
                         mean_amplitude_uv=float(level + np.sin(t)), n_participants=8)
                    for t in times)
    waves = pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
    summary = pd.DataFrame([
        dict(family='normative', category=cat, n_participants=8,
             retained_trials=n, contributed_trials=n, excluded_trials=0, minimum_trials=10)
        for cat, n in [('lower_aaps', 1482), ('higher_aaps', 1486)]
    ]).to_csv(index=False).encode('utf-8')
    manifest = {
        'schema': 'kawaii_norm02_exploratory_unapproved_v1',
        'scientific_status': module.SCIENTIFIC_STATUS,
        'source_repository_head': 'a'*40,
        'all_retained_sub012_exposure_ambiguous': 190,
        'participant_source_bindings_sha256': {
            sub: f'{i:064x}' for i, sub in enumerate(module.SOURCE_IDS, start=1)
        },
        'file_sha256': {module.FILE_WAVEFORM: sha256(waves).hexdigest(),
                        module.FILE_SUMMARY: sha256(summary).hexdigest()},
    }
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(module.FILE_WAVEFORM, waves)
        z.writestr(module.FILE_SUMMARY, summary)
        z.writestr('MANIFEST.json', json.dumps(manifest))
    return path


def test_explicit_rating_source_selection():
    for value in ('participant', 'normative', 'both'):
        assert module.parse_args(['--rating-source', value, '--output-dir', '/tmp/selected']).rating_source == value
    with pytest.raises(SystemExit):
        module.parse_args(['--rating-source', 'unrecognised', '--output-dir', '/tmp/no'])


def test_archive_returns_exact_two_score_defined_curves(tmp_path):
    archive = _archive(tmp_path)
    curves, provenance = module.read_normative_archive(archive)
    assert len(curves) == 1402
    assert set(curves.category) == set(module.NORMATIVE_CATEGORIES)
    assert provenance['category_counts'] == {'lower_aaps': 1482, 'higher_aaps': 1486}
    assert len(provenance['participant_source_bindings_sha256']) == 8


def test_archive_rejects_hash_tampering(tmp_path):
    archive = _archive(tmp_path)
    with zipfile.ZipFile(archive) as z:
        files = {name: z.read(name) for name in z.namelist()}
    files[module.FILE_WAVEFORM] = b"tampering"
    archive.unlink()
    with zipfile.ZipFile(archive, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
    with pytest.raises(ValueError, match="hash mismatch"):
        module.read_normative_archive(archive)


def test_archive_rejects_wrong_category_count(tmp_path):
    archive = _archive(tmp_path)
    with zipfile.ZipFile(archive) as z:
        data = {n: z.read(n) for n in z.namelist()}
    summary = pd.read_csv(__import__('io').BytesIO(data[module.FILE_SUMMARY]))
    summary.loc[0, 'retained_trials'] = 1483
    data[module.FILE_SUMMARY] = summary.to_csv(index=False).encode()
    manifest = json.loads(data['MANIFEST.json'])
    manifest['file_sha256'][module.FILE_SUMMARY] = sha256(data[module.FILE_SUMMARY]).hexdigest()
    data['MANIFEST.json'] = json.dumps(manifest).encode()
    archive.unlink()
    with zipfile.ZipFile(archive, 'w') as z:
        for n, b in data.items(): z.writestr(n, b)
    with pytest.raises(ValueError, match='counts'):
        module.read_normative_archive(archive)


def test_missing_selected_source_fails_before_output(tmp_path):
    p = tmp_path / 'figs'
    with pytest.raises(ValueError, match='source-root'):
        module.run(rating_source='participant', output_dir=p, project_root=tmp_path)
    with pytest.raises(ValueError, match='normative-archive'):
        module.run(rating_source='both', output_dir=p, project_root=tmp_path, source_root=tmp_path)
    assert not p.exists()


def test_output_inside_repo_or_existing_is_rejected(tmp_path):
    archive = _archive(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        module.run(rating_source='normative', output_dir=tmp_path/'inside',
                   project_root=tmp_path, normative_archive=archive)
    assert not (tmp_path/'inside').exists()
    p = tmp_path.parent / 'existing'; p.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match='fresh'):
        module._check_paths(tmp_path, p, None, archive)


def test_normative_mode_creates_only_pinned_figure_and_manifest(tmp_path, monkeypatch):
    """End-to-end render without any EEG source module or files."""
    from types import ModuleType, SimpleNamespace
    archive = _archive(tmp_path)
    repo = tmp_path / 'repo'
    repo.mkdir()
    destination = tmp_path / 'figure_output'
    for name in ('eeg_software', 'eeg_software.workflows',
                 'eeg_software.workflows.kawaii_session2'):
        pkg = ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    fake_style = ModuleType('eeg_software.workflows.kawaii_session2.figure_style')
    fake_style.load_figure_style = lambda **kwargs: SimpleNamespace(
        figure_facecolor='white', dpi=75, fonts={'title_size': 12},
        color=lambda label: {'primary_window': '#DDDDDD', 'reference': '#777777'}[label],
        condition_color=lambda _, label: '#777777' if label == 'low' else '#222222')
    fake_style.style_axes = lambda ax, style: None
    fake_style.style_legend = lambda ax, style: ax.legend()
    monkeypatch.setitem(sys.modules, fake_style.__name__, fake_style)
    result = module.run(rating_source='normative', output_dir=destination,
                        project_root=repo, normative_archive=archive)
    assert result == destination
    assert sorted(p.name for p in destination.iterdir()) == [
        'Figure_5b_Normative_AAPS_Cuteness.png', 'MANIFEST.json']
    manifest = json.loads((destination/'MANIFEST.json').read_text())
    assert manifest['rating_source'] == 'normative'
    assert 'participant' not in manifest['provenance']
    filename = 'Figure_5b_Normative_AAPS_Cuteness.png'
    assert (destination/filename).stat().st_size > 1000
    assert sha256((destination/filename).read_bytes()).hexdigest() == manifest['file_sha256'][filename]


@pytest.mark.parametrize('rating_source', ['participant', 'both'])
def test_participant_source_dispatch_and_manifest(tmp_path, monkeypatch, rating_source):
    """The accepted participant writer is used, not NORM-02 participant re-averaging."""
    from types import ModuleType, SimpleNamespace
    archive = _archive(tmp_path)
    repo = tmp_path/'repo'; repo.mkdir()
    source_root = tmp_path/'published'; source_root.mkdir()
    pointer = source_root/'current.json'; pointer.write_text('{"status":"source"}')
    source = SimpleNamespace(generation_id='a'*64, current_manifest=pointer,
                             artifact_hashes={'cuteness_grand_average_fif': 'b'*64},
                             participant_ids=module.SOURCE_IDS)
    monkeypatch.setattr(module, 'load_participant_source',
                        lambda source_root, project_root: (source, object()))
    for name in ('eeg_software', 'eeg_software.workflows',
                 'eeg_software.workflows.kawaii_session2'):
        pkg = ModuleType(name); pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    style = ModuleType('eeg_software.workflows.kawaii_session2.figure_style')
    style.load_figure_style = lambda **kwargs: SimpleNamespace(
        figure_facecolor='white', dpi=60, fonts={'title_size': 10},
        color=lambda label: {'primary_window': '#DDDDDD', 'reference': '#777777'}[label],
        condition_color=lambda _, label: '#777777' if label == 'low' else '#222222')
    style.style_axes = lambda ax, style: None
    style.style_legend = lambda ax, style: ax.legend()
    monkeypatch.setitem(sys.modules, style.__name__, style)
    participant = ModuleType('eeg_software.workflows.kawaii_session2.report_descriptive_figures')
    participant.write_cuteness_descriptive_waveform = (
        lambda evokeds, path, *, participant_count, style:
        path.write_bytes(b'dummy accepted participant report render'))
    monkeypatch.setitem(sys.modules, participant.__name__, participant)
    output = tmp_path/'figures'
    module.run(rating_source=rating_source, output_dir=output, project_root=repo,
               source_root=source_root,
               normative_archive=(archive if rating_source == 'both' else None))
    names = {p.name for p in output.iterdir()}
    assert 'Figure_5a_Participant_Rated_Cuteness.png' in names
    assert ('Figure_5b_Normative_AAPS_Cuteness.png' in names) == (rating_source == 'both')
    manifest = json.loads((output/'MANIFEST.json').read_text())
    assert manifest['rating_source'] == rating_source
    assert manifest['provenance']['participant']['source_generation_id'] == 'a'*64
    assert ('normative' in manifest['provenance']) == (rating_source == 'both')
