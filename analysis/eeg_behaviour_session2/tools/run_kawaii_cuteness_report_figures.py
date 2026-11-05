"""Report-only Kawaii cuteness ERP figures with explicit rating-source selection.

No accepted EEG epochs are loaded or modified. Participant-rated waveforms come
from the authenticated R11 Gate 5B group publication, while normative waveforms
come from the separately reviewed NORM-02 Run 01 summary archive. This is NOT a
selector for mixed-model inference and does not make the two estimands identical.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

NORMATIVE_CATEGORIES = ('lower_aaps', 'higher_aaps')
REQUIRED_NORMATIVE_COUNTS = {'lower_aaps': 1482, 'higher_aaps': 1486}
SOURCE_IDS = ('sub005', 'sub009', 'sub010', 'sub011', 'sub012', 'sub013', 'sub014', 'sub015')
FILE_WAVEFORM = 'eight_all_retained_waveforms.csv'
FILE_SUMMARY = 'eight_all_retained_summary.csv'
SCIENTIFIC_STATUS = 'EXPLORATORY_NOT_APPROVED_FOR_COLLABORATOR_REPORT'


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha(Path(path).read_bytes())


def read_normative_archive(path: Path):
    """Verify prior reviewed archive and extract only the two normative curves."""
    path = Path(path).resolve(strict=True)
    with zipfile.ZipFile(path, 'r') as z:
        if z.testzip() is not None:
            raise ValueError('NORM-02 ZIP CRC failure')
        names = z.namelist()
        if len(names) != len(set(names)) or any('/' in n or '\\' in n for n in names):
            raise ValueError('Archive contains duplicate or nested member paths')
        if not {'MANIFEST.json', FILE_WAVEFORM, FILE_SUMMARY}.issubset(names):
            raise ValueError('Missing normative source members')
        manifest = json.loads(z.read('MANIFEST.json'))
        if manifest.get('schema') != 'kawaii_norm02_exploratory_unapproved_v1':
            raise ValueError('Unexpected normative source schema')
        if manifest.get('scientific_status') != SCIENTIFIC_STATUS:
            raise ValueError('Unexpected normative scientific status')
        if manifest.get('all_retained_sub012_exposure_ambiguous') != 190:
            raise ValueError('sub012 inclusion caveat changed')
        bindings = manifest.get('participant_source_bindings_sha256', {})
        if set(bindings) != set(SOURCE_IDS) or any(
            re.fullmatch(r'[a-f0-9]{64}', str(v)) is None for v in bindings.values()
        ) or len(set(bindings.values())) != 8:
            raise ValueError('Normative participant source binding inventory changed')
        expected_hashes = manifest.get('file_sha256', {})
        for member in (FILE_WAVEFORM, FILE_SUMMARY):
            if _sha(z.read(member)) != expected_hashes.get(member):
                raise ValueError(f'Normative source hash mismatch: {member}')
        waveforms = pd.read_csv(BytesIO(z.read(FILE_WAVEFORM)))
        summary = pd.read_csv(BytesIO(z.read(FILE_SUMMARY)))

    for columns, table in (
        ({'family', 'category', 'time_s', 'mean_amplitude_uv', 'n_participants'}, waveforms),
        ({'family', 'category', 'retained_trials', 'n_participants', 'minimum_trials'}, summary),
    ):
        if not columns.issubset(table.columns):
            raise ValueError('Normative source CSV schema changed')
    normative = waveforms.loc[waveforms['family'] == 'normative'].copy()
    ns = summary.loc[summary['family'] == 'normative'].copy()
    if len(ns) != 2 or set(ns['category']) != set(NORMATIVE_CATEGORIES):
        raise ValueError('Expected exactly two normative category summaries')
    if dict(zip(ns['category'], ns['retained_trials'].astype(int))) != REQUIRED_NORMATIVE_COUNTS:
        raise ValueError('Corrected normative group counts changed')
    if not (ns['n_participants'].astype(int) == 8).all() or not (ns['minimum_trials'].astype(int) == 10).all():
        raise ValueError('Normative contributor number or minimum count changed')
    if len(normative) != 1402 or normative.duplicated(['category', 'time_s']).any():
        raise ValueError('Normative waveform cardinality/duplicates changed')
    grid = np.linspace(-0.2, 1.2, 701)
    for category in NORMATIVE_CATEGORIES:
        rows = normative.loc[normative['category'] == category].sort_values('time_s')
        if len(rows) != 701 or not (rows['n_participants'].astype(int) == 8).all():
            raise ValueError('Normative participant/trace length changed')
        times = rows['time_s'].to_numpy(dtype=float)
        amplitudes = rows['mean_amplitude_uv'].to_numpy(dtype=float)
        if not np.isfinite(amplitudes).all() or not np.allclose(times, grid, rtol=0, atol=1e-10):
            raise ValueError('Normative waveform time grid or amplitude invalid')
    return normative, {
        'input_zip_sha256': _file_hash(path),
        'input_source_head': manifest['source_repository_head'],
        'participant_source_bindings_sha256': dict(sorted(bindings.items())),
        'source_members_sha256': {key: expected_hashes[key] for key in (FILE_WAVEFORM, FILE_SUMMARY)},
        'sub012_exposure_ambiguous_epochs': 190,
        'category_counts': REQUIRED_NORMATIVE_COUNTS,
        'category_rule': 'AAPS score-defined <=5.0 versus >6.5; NOT verified original feedback pools',
        'cohort': 'all eight corrected R11 V3 participants',
    }


def render_normative_figure(waveforms, destination: Path, *, style) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eeg_software.workflows.kawaii_session2.figure_style import style_axes, style_legend

    fig, ax = plt.subplots(figsize=(8.2, 4.5), facecolor=style.figure_facecolor)
    ax.axvspan(500, 800, color=style.color('primary_window'), alpha=.18, zorder=0)
    ax.axvline(0, color=style.color('reference'), lw=.8, alpha=.65)
    ax.axhline(0, color=style.color('reference'), lw=.8, alpha=.65)
    for category, label, color in (
        ('lower_aaps', 'Lower normative cuteness (≤5)', style.condition_color('cuteness', 'low')),
        ('higher_aaps', 'Higher normative cuteness (>6.5)', style.condition_color('cuteness', 'high')),
    ):
        rows = waveforms.loc[waveforms.category == category].sort_values('time_s')
        ax.plot(rows.time_s.to_numpy(dtype=float) * 1000,
                rows.mean_amplitude_uv.to_numpy(dtype=float),
                label=label, color=color, linewidth=1.8)
    ax.set(xlabel='Time (ms)', ylabel='µV')
    style_axes(ax, style)
    style_legend(ax, style)
    fig.suptitle('Descriptive grand-average ERPs by normative cuteness\n'
                 'Centroparietal ROI (Cz, CP1, CP2, Pz; N = 8)',
                 fontsize=style.fonts['title_size'], y=.97)
    fig.text(.5, .04, 'AAPS score-defined groups · exploratory, not scientifically approved',
             ha='center', fontsize=7.7, color='#6B5359')
    fig.tight_layout(rect=(0, .08, 1, .87))
    fig.savefig(destination, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)


def load_participant_source(source_root: Path, project_root: Path):
    """Reuse strict published-source verification from the accepted R11 report renderer."""
    from eeg_software.workflows.kawaii_session2.report_descriptive_figures import (
        discover_published_group_sources, read_evoked_mapping, CUTENESS_CONDITIONS, roi_trace_uv,
    )
    source = discover_published_group_sources(source_root)
    config = Path(project_root) / 'config/session2_eeg.yaml'
    if _file_hash(config) != source.config_hash:
        raise ValueError('Config hash differs from published Gate 5B source')
    cuteness = read_evoked_mapping(source.artifacts['cuteness_grand_average_fif'])
    if not set(CUTENESS_CONDITIONS).issubset(cuteness):
        raise ValueError('Published subjective categories missing')
    for key in CUTENESS_CONDITIONS:
        times, values = roi_trace_uv(cuteness[key])
        if not np.isfinite(values).all() or not np.isfinite(times).all():
            raise ValueError(f'Non-finite subjective curve: {key}')
    return source, cuteness


def _check_paths(project_root: Path, output_dir: Path, source_root: Path | None, normative_archive: Path | None):
    project_root = project_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    if (output_dir == project_root or project_root in output_dir.parents or
            output_dir.exists() or not output_dir.parent.is_dir()):
        raise ValueError('Output must be a fresh folder outside the internal repository')
    if source_root is not None:
        source_root = Path(source_root).resolve()
        if output_dir == source_root or source_root in output_dir.parents:
            raise ValueError('Output must be outside published group source')
    if normative_archive is not None and output_dir == normative_archive.resolve():
        raise ValueError('Output conflicts with normative source')
    return output_dir


def run(*, rating_source: str, output_dir: Path, project_root: Path,
        source_root: Path | None = None, normative_archive: Path | None = None):
    if rating_source not in ('participant', 'normative', 'both'):
        raise ValueError('rating_source must be participant, normative or both')
    needs_participant = rating_source in ('participant', 'both')
    needs_normative = rating_source in ('normative', 'both')
    if needs_participant and source_root is None:
        raise ValueError('--source-root is required for participant selection')
    if needs_normative and normative_archive is None:
        raise ValueError('--normative-archive is required for normative selection')
    output_dir = _check_paths(Path(project_root), Path(output_dir),
                              source_root if needs_participant else None,
                              normative_archive if needs_normative else None)
    # Complete all source validation before writing any output.
    waveforms, norm_info = read_normative_archive(normative_archive) if needs_normative else (None, None)
    source, evokeds = load_participant_source(source_root, project_root) if needs_participant else (None, None)
    from eeg_software.workflows.kawaii_session2.figure_style import load_figure_style
    style = load_figure_style(root=project_root)
    files = {}
    provenance = {}
    with tempfile.TemporaryDirectory(prefix='kawaii_rating_source_', dir=output_dir.parent) as temp:
        work = Path(temp)
        if needs_participant:
            from eeg_software.workflows.kawaii_session2.report_descriptive_figures import (
                write_cuteness_descriptive_waveform,
            )
            name = 'Figure_5a_Participant_Rated_Cuteness.png'
            write_cuteness_descriptive_waveform(
                evokeds, work/name, participant_count=len(source.participant_ids), style=style,
            )
            files[name] = _file_hash(work/name)
            provenance['participant'] = {
                'source_generation_id': source.generation_id,
                'source_pointer_sha256': _file_hash(source.current_manifest),
                'published_fif_sha256': source.artifact_hashes['cuteness_grand_average_fif'],
                'participant_ids': list(source.participant_ids),
                'categories': ['1–2','3–4','5–6','7–9'],
                'source_kind': 'authenticated Gate 5B R11 V3 subjective grand averages',
            }
        if needs_normative:
            name = 'Figure_5b_Normative_AAPS_Cuteness.png'
            render_normative_figure(waveforms, work/name, style=style)
            files[name] = _file_hash(work/name)
            provenance['normative'] = norm_info
        manifest = {
            'schema': 'kawaii_report_cuteness_rating_source_v1',
            'rating_source': rating_source,
            'scientific_status': SCIENTIFIC_STATUS,
            'note': 'Descriptive group-average figures only; no categorical tests or mixed-model inference',
            'separate_source_authorities': True,
            'provenance': provenance,
            'file_sha256': files,
        }
        (work/'MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        if output_dir.exists():
            raise FileExistsError('Output appeared while rendering; refusing to overwrite')
        os.rename(work, output_dir)  # Only completed, source-verified output is made visible.
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rating-source', choices=('participant', 'normative', 'both'), required=True)
    parser.add_argument('--source-root', type=Path,
                        help='Current authenticated corrected R11 V3 group publication root (participant/both)')
    parser.add_argument('--normative-archive', type=Path,
                        help='Verified NORM-02 Run 01 ZIP containing eight-participant score-defined waveforms (normative/both)')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh destination outside repository')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run(rating_source=args.rating_source, output_dir=args.output_dir,
                 project_root=args.project_root, source_root=args.source_root,
                 normative_archive=args.normative_archive)
    print('RATING_SOURCE_FIGURES: PASS')
    print('OUTPUT:', result)
    print('SCIENTIFIC_STATUS:', SCIENTIFIC_STATUS)
    print('EEG_PROCESSING_OR_MODEL_FITTING: NONE')


if __name__ == '__main__':
    main()
