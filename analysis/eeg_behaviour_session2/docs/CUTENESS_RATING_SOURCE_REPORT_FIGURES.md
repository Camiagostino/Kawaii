# Report-only cuteness ERP figures: participant and normative ratings

**Status:** Software interface tested with synthetic fixtures and accepted internal figure sources. Normative study figures and numerical results remain **exploratory / not scientifically approved for public reporting**. These commands do **not** preprocess EEG, recalculate evoked responses, fit mixed models or establish scientific source authority.

## What the selector does

`tools/run_kawaii_cuteness_report_figures.py` renders existing descriptive centroparietal ROI waveforms (Cz, CP1, CP2, Pz), with time in milliseconds and the predefined 500–800 ms window highlighted:

| Rating source | Input authority | Output figure | Category definition |
|---|---|---|---|
| `participant` | Accepted corrected R11 V3 Gate 5B group publication, including its current pointer, authentic FIF and matching local run configuration | `Figure_5a_Participant_Rated_Cuteness.png` | Participant's own post-hoc cuteness ratings in four groups: 1–2, 3–4, 5–6, 7–9 |
| `normative` | Separately supplied, **authorised** NORM-02 summary archive derived from the AAPS source mapping | `Figure_5b_Normative_AAPS_Cuteness.png` | AAPS-image-score-defined categories: lower <= 5.0 and higher > 6.5; not verified as the original experimental feedback pools |
| `both` | Both independent inputs, with separate provenance | Both PNGs | Separate descriptive estimands; **not** a direct statistical test |

This selector targets the specifically documented eight-participant corrected cohort and is **not** a general-purpose normative ERP calculator. Its normative branch checks archived waveform grids, category totals, participant-source bindings, file hashes and study caveats. It does **not** independently establish that an arbitrarily supplied archive is the laboratory-approved one. Operators must verify the archive's **external SHA-256** against the separately approved reference and obtain permission to use the archive. Do not commit source ZIPs to this public repository.

## Usage

From the `analysis/eeg_behaviour_session2` directory in the supported Python 3.11 environment:

```bash
# Normative-only: no Gate 5B FIF required, but an approved summary ZIP is required.
python tools/run_kawaii_cuteness_report_figures.py \
  --rating-source normative \
  --normative-archive /path/to/authorised/NORM02_summary.zip \
  --output-dir /path/outside/this/repository/kawaii_normative_figures

# Participant-only: requires the current corrected R11 V3 publication and a
# local, Git-ignored config/session2_eeg.yaml matching that source's hash.
python tools/run_kawaii_cuteness_report_figures.py \
  --rating-source participant \
  --source-root /path/to/approved/r11_qc_v3 \
  --output-dir /path/outside/this/repository/kawaii_participant_figures

# Both sources, with a different fresh output directory.
python tools/run_kawaii_cuteness_report_figures.py \
  --rating-source both \
  --source-root /path/to/approved/r11_qc_v3 \
  --normative-archive /path/to/authorised/NORM02_summary.zip \
  --output-dir /path/outside/this/repository/kawaii_both_figures
```

The output directory must be a **new, non-existing directory outside the analysis checkout**; the parent must already exist. Source verification precedes output creation. The completed output includes the requested PNG(s) and `MANIFEST.json` containing file SHA-256 hashes and separate input provenance. Never treat the example placeholders as real study paths.

## Interpretation, approval and privacy boundaries

- The participant-rated and AAPS-defined categories are **different constructs**. The normative split is based on score thresholds; equivalence to the experimental positive/negative feedback pools has not been verified.
- The underlying exploratory source included a subset of exposure-ambiguous but EEG-retained epochs. Their eligibility requires scientific disposition before approved reporting. Exact study-specific counts are encoded in the implementation contract and must be cleared for public release by the study lead.
- The participant-level descriptive contrasts are heterogeneous. **There is no normative category significance test in this figure command.** Do not interpret a positive or negative descriptive grand average as a consistent or statistically significant participant effect.
- A participant with eight Low subjective-category trials does not contribute to that particular four-category mean under the existing ten-trial minimum. This does not imply their exclusion from other analyses.
- The corrected selected mixed model is a **different analysis**: it adjusts for old/new but not arousal. The attempted full arousal-adjusted model did not converge. None of these statistical results is recalculated or justified by this figure selector.
- The output manifest can include pseudonymous subject IDs, archive bindings and publication hashes. Treat these and the output PNGs as **study-derived materials**, not automatically suitable for public GitHub or public figures. Keep the AAPS workbook, source-level CSVs, EEG/FIF files, manual QC decisions, original exposure records and approved source archives outside Git.
- Scientific-owner review, participant/privacy authority and any AAPS licensing/redistribution checks remain separate from code testing and GitHub merging.

## Tests

Fixture-independent tests require no real EEG or source archive:

```bash
PYTHONPATH=src python -m pytest -q tests/test_cuteness_rating_source_cli.py tests/test_collaborator_release_contract.py
```

On Windows, the Git checkout must preserve `config/kawaii_session2_figure_style.yaml` as LF because the existing configuration hashes the **exact approved bytes**. The scoped `.gitattributes` rule for that file fixes the Windows checkout mismatch without changing any scientific figure-style settings or the approved hash.

**Reproduction limit:** this public software release alone cannot recreate the archived normative summary from raw EEG or establish laboratory sign-off of the original cohort; those source-authority steps are controlled separately.
