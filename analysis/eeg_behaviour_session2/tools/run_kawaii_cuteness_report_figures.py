"""Report-only Kawaii cuteness ERP figures with explicit rating-source selection.

No accepted EEG epochs are loaded or modified. Participant-rated waveforms come
from an authenticated group publication, while normative waveforms come from a
separately authorised descriptive summary archive. This is NOT a
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
FILE_WAVEFORM = 'eight_all_retained_waveforms.csv'
FILE_SUMMARY = 'eight_all_retained_summary.csv'
SCIENTIFIC_STATUS = 'EXPLORATORY_NOT_APPROVED_FOR_COLLABORATOR_REPORT'


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha(Path(path).read_bytes())


def read_normative_archive(path: Path):
    """Verify an authorised archive and extract only the two normative curves."""
    path = Path(path).resolve(strict=True)
    with zipfile.ZipFile(path, 'r') as z:
        if z.testzip() is not None:
            raise ValueError('NORM-02 ZIP CRC failure')
        names = z.namelist()
        if len(names) != len(set(names)) or any('/' in n or '\\' in n for n in names):
            raise ValueError('Archive contains duplicate or nested member paths')
        if 'MANIFEST.json' not in names:
            raise ValueError('Missing normative source manifest')
        manifest = json.loads(z.read('MANIFEST.json'))
        if manifest.get('schema') != 'kawaii_norm02_exploratory_unapproved_v1':
            raise ValueError('Unexpected normative source schema')
        if manifest.get('scientific_status') != SCIENTIFIC_STATUS:
            raise ValueError('Unexpected normative scientific status')
        waveform_member = manifest.get('normative_waveforms_member', FILE_WAVEFORM)
        summary_member = manifest.get('normative_summary_member', FILE_SUMMARY)
        if any(
            not isinstance(member, str) or not member or '/' in member or '\\' in member
            for member in (waveform_member, summary_member)
        ) or not {waveform_member, summary_member}.issubset(names):
            raise ValueError('Missing or invalid normative source members')
        bindings = manifest.get('participant_source_bindings_sha256', {})
        if not isinstance(bindings, dict) or not bindings or any(
            not isinstance(key, str) or not key.strip() or
            re.fullmatch(r'[a-f0-9]{64}', str(value)) is None
            for key, value in bindings.items()
        ) or len(set(bindings.values())) != len(bindings):
            raise ValueError('Invalid normative participant source binding inventory')
        expected_hashes = manifest.get('file_sha256', {})
        for member in (waveform_member, summary_member):
            if _sha(z.read(member)) != expected_hashes.get(member):
                raise ValueError(f'Normative source hash mismatch: {member}')
        waveforms = pd.read_csv(BytesIO(z.read(waveform_member)))
        summary = pd.read_csv(BytesIO(z.read(summary_member)))

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
    retained = pd.to_numeric(ns['retained_trials'], errors='coerce')
    contributors = pd.to_numeric(ns['n_participants'], errors='coerce')
    minimum = pd.to_numeric(ns['minimum_trials'], errors='coerce')
    count_columns = (retained, contributors, minimum)
    if any(values.isna().any() or not np.isfinite(values).all() for values in count_columns):
        raise ValueError('Normative summary counts must be finite numeric values')
    if any(not np.equal(values, np.floor(values)).all() for values in count_columns):
        raise ValueError('Normative summary counts must be integers')
    if (retained < 0).any() or (contributors < 1).any() or (minimum < 1).any():
        raise ValueError('Normative summary counts are invalid')
    if (contributors > len(bindings)).any():
        raise ValueError('Normative contributor count exceeds authenticated participant bindings')
    if (retained < contributors * minimum).any():
        raise ValueError('Normative retained-trial count is inconsistent with contributor minimums')
    optional_counts = {}
    for column in ('contributed_trials', 'excluded_trials'):
        if column not in ns:
            continue
        values = pd.to_numeric(ns[column], errors='coerce')
        if (
            values.isna().any()
            or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
            or (values < 0).any()
            or (values > retained).any()
        ):
            raise ValueError(f'Normative {column} counts are invalid')
        optional_counts[column] = values
    if 'contributed_trials' in optional_counts and (
        optional_counts['contributed_trials'] < contributors * minimum
    ).any():
        raise ValueError('Normative contributed-trial count is inconsistent with contributor minimums')
    if set(optional_counts) == {'contributed_trials', 'excluded_trials'} and not np.equal(
        optional_counts['contributed_trials'] + optional_counts['excluded_trials'], retained
    ).all():
        raise ValueError('Normative contributed/excluded counts do not sum to retained trials')
    if normative.empty or normative.duplicated(['category', 'time_s']).any():
        raise ValueError('Normative waveform rows are empty or duplicated')
    grid = None
    for category in NORMATIVE_CATEGORIES:
        rows = normative.loc[normative['category'] == category].sort_values('time_s')
        times = rows['time_s'].to_numpy(dtype=float)
        amplitudes = rows['mean_amplitude_uv'].to_numpy(dtype=float)
        expected_n = int(ns.loc[ns['category'] == category, 'n_participants'].iloc[0])
        observed_n = pd.to_numeric(rows['n_participants'], errors='coerce')
        if len(rows) < 3 or observed_n.isna().any() or not (observed_n.astype(int) == expected_n).all():
            raise ValueError('Normative contributor inventory or trace length is invalid')
        if not np.isfinite(times).all() or not np.isfinite(amplitudes).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Normative waveform time grid or amplitude invalid')
        if grid is None:
            grid = times
        elif not np.allclose(times, grid, rtol=0, atol=1e-10):
            raise ValueError('Normative category time grids differ')
    category_counts = dict(zip(ns['category'], retained.astype(int)))
    return normative, {
        'input_zip_sha256': _file_hash(path),
        'input_source_head': manifest.get('source_repository_head'),
        'participant_source_bindings_sha256': dict(sorted(bindings.items())),
        'source_members_sha256': {
            key: expected_hashes[key] for key in (waveform_member, summary_member)
        },
        'category_counts': category_counts,
        'category_rule': 'AAPS score-defined <=5.0 versus >6.5; NOT verified original feedback pools',
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
        participant_count = int(rows.n_participants.iloc[0])
        ax.plot(rows.time_s.to_numpy(dtype=float) * 1000,
                rows.mean_amplitude_uv.to_numpy(dtype=float),
                label=f'{label}; N = {participant_count}', color=color, linewidth=1.8)
    ax.set(xlabel='Time (ms)', ylabel='µV')
    style_axes(ax, style)
    style_legend(ax, style)
    fig.suptitle('Descriptive grand-average ERPs by normative cuteness\n'
                 'Centroparietal ROI (Cz, CP1, CP2, Pz)',
                 fontsize=style.fonts['title_size'], y=.97)
    fig.text(.5, .04, 'AAPS score-defined categories',
             ha='center', fontsize=7.7, color='#6B5359')
    fig.tight_layout(rect=(0, .08, 1, .87))
    fig.savefig(destination, dpi=style.dpi, facecolor=style.figure_facecolor)
    plt.close(fig)


def load_participant_source(source_root: Path, project_root: Path):
    """Reuse strict published-source verification from the existing report renderer."""
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
        raise ValueError('Output must be a fresh folder outside the analysis repository')
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
                'source_kind': 'authenticated participant-rated group grand averages',
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
                        help='Authenticated participant-rated group publication root (participant/both)')
    parser.add_argument('--normative-archive', type=Path,
                        help='Authorised descriptive normative ZIP with score-defined waveforms (normative/both)')
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
