# Kawaii Session 2 — one command for participant and normative cuteness ERPs

**Scope:** descriptive centroparietal ERP curves from **already accepted corrected EEG sources**. This is a cohort-specific analysis interface, not a new EEG preprocessing pipeline, a general AAPS analysis, or a statistical test. Real study sources, reviewed EEG/ICA decisions, and the AAPS reference are **not included in this public code repository**.

## User-facing command

From the `analysis/eeg_behaviour_session2` checkout using the supported Python 3.11 environment, run `kawaii-session2-eeg cuteness --help`. A direct alternative is `python tools/run_kawaii_cuteness_analysis.py --help`.

```bash
# PARTICIPANT: render from an accepted corrected Gate 5B publication
kawaii-session2-eeg cuteness --root . --rating-source participant \
  --source-root /approved/r11_qc_v3 \
  --output-dir /private/new/cuteness_participant

# NORMATIVE: reuse an existing, externally authorised NORM-02 ZIP
kawaii-session2-eeg cuteness --root . --rating-source normative \
  --normative-archive /private/approved/NORM02_summary.zip \
  --acknowledge-sub012-exploratory \
  --output-dir /private/new/cuteness_normative

# BOTH: use accepted participant Gate 5B plus compute a fresh normative
# *descriptive* aggregate from already corrected, authenticated Gate 3C epochs
kawaii-session2-eeg cuteness --root . --rating-source both \
  --source-root /approved/r11_qc_v3 \
  --aaps-reference /private/approved/kawaii_norm02_reference.csv \
  --acknowledge-sub012-exploratory \
  --output-dir /private/new/cuteness_both
```

To use `both` **without** recalculating normative waveforms, pass both `--source-root` and `--normative-archive` and the acknowledgement flag. For new normative descriptive aggregation, omit `--normative-archive`. The default AAPS reference path is `config/kawaii_norm02_aaps_reference_20261101.csv` within the local checkout; using `--aaps-reference` allows a private external location **without disabling the existing fixed SHA-256 check**. The corrected cohort's published epoch data and reviewed configuration must still be made available in the expected original-cohort `workspace/` source layout.

## What the selector means

| Source | Authority | What is (re)computed? | Rating categories |
|---|---|---|---|
| `participant` | Current corrected R11 V3 Gate 5B cuteness grand-average publication | **Figure only** from accepted evokeds; no EEG rereading | Four individual ratings: 1–2, 3–4, 5–6, 7–9 |
| `normative` with ZIP | Independently approved NORM-02 score-defined archive | **Figure only**; archive's internal SHA-256/provenance checked | Lower AAPS score ≤5 and higher >6.5 |
| `normative` without ZIP | Accepted corrected Gate 3C epochs+metadata bound to Gate 4B plus verified AAPS crosswalk | Existing **descriptive** participant-mean and group aggregation, then figure; no preprocessing or inferential fitting | Lower AAPS score ≤5 and higher >6.5 |
| `both` | Both **separate** authorities | Appropriate combination of the above, never a new statistical comparison | Both distinct category families |

**Important:** Selecting two families does not imply their categories or source estimands are identical. The originally assigned experimental feedback pools have **not** been independently shown to equal the AAPS-score-defined groups.

## Inputs, outputs and safeguards

- Always use a new, non-existing `--output-dir` outside the analysis checkout. The parent directory must already exist. Completed outputs are staged before appearing at their final location.
- The participant branch checks published Gate 5B source manifests and original configuration identity. The normative calculation branch uses authenticated current Gate 3C/Gate 4B original-cohort publications and the pinned AAPS score crosswalk.
- Output includes `figures/` (PNG and `MANIFEST.json`), a top-level orchestration manifest and, **only for a new normative aggregation**, `normative_summary/` plus its `.zip` and source-hash manifest. These outputs may contain pseudonymous participant/source references and should remain in approved private storage.
- No original EEG filtering, ICA, bad-channel/epoch rejections, R11 replay, mixed-model fit or statistical significance test is run. Existing participant and normative computations are reused, not rewritten.
- This command does **not** establish that a supplied ZIP/source is scientifically approved merely because it passes integrity checks. Confirm external source identities and laboratory approval separately.

## Scientific interpretation and privacy

AAPS score groups are exploratory, not validated original feedback groups; all-eight normative results include 190 exposure-ambiguous EEG-retained epochs from sub012, hence **explicit `--acknowledge-sub012-exploratory`** is compulsory for normative/both modes and does not confer approval. A low subjective bin with only eight retained trials from sub010 does not contribute under the established ten-trial participant/category minimum. Individual normative higher-minus-lower contrasts vary in sign; do not present the descriptive grand-average direction as robust, statistically significant or arousal-independent.

Do not distribute source FIFs, participant-level behaviour, original AAPS workbook, SHA-pinned private reference CSV, source-exposure dispositions, manual QC/ICA decisions or derived ZIPs through public GitHub without explicit authorisation. The public suitability of embedded pseudonymous participant IDs/counts also requires scientific-owner clearance.

## Verification and release status

Run `python -m pytest -q` from the public analysis checkout in the supported Python 3.11 environment. The fixture-independent suite covers the unified command, figure-source selector, normative reference adapter, aggregation and runner using synthetic test inputs; it does **not** need the private AAPS crosswalk, participant recordings or the original lab workspace. The real-source `both` route has separately been checked against previously authenticated participant and NORM-02 sources, producing figures byte-identical to earlier reviewed outputs. Fresh normative aggregation through the unified command has **not** been independently exercised here with real corrected EEG inputs.

**Scientific/release status:** the normative results remain exploratory and are **not approved for public results reporting**. Running tests or validating a source archive does not establish laboratory approval of the underlying EEG inputs, participant eligibility or normative source. The public release of embedded participant identifiers, cohort counts and AAPS-related metadata remains subject to study-lead/privacy/licensing authorisation.
