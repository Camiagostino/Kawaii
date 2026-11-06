# Descriptive cuteness rating sources

`kawaii-session2-eeg cuteness` renders descriptive centroparietal ERP curves from already authenticated sources. It does not preprocess EEG, change ICA/QC decisions, reject epochs, or fit inferential models.

Participant and normative ratings are separate authorities:

| `--rating-source` | Required input | Categories |
|---|---|---|
| `participant` | `--source-root` pointing to an authenticated participant-rated group publication | Individual ratings 1–2, 3–4, 5–6 and 7–9 |
| `normative` | `--normative-archive` containing source-bound descriptive waveforms | AAPS score-defined lower ≤5.0 and higher >6.5 |
| `both` | Both inputs above | Both category families, kept separate |

Selecting `both` does not make the category families equivalent and does not perform a statistical comparison. Continuous normative inferential modelling is not implemented.

From the analysis directory:

```bash
kawaii-session2-eeg cuteness --rating-source participant \
  --source-root /approved/group-publication \
  --output-dir /private/new/cuteness-participant

kawaii-session2-eeg cuteness --rating-source normative \
  --normative-archive /approved/normative-summary.zip \
  --output-dir /private/new/cuteness-normative

kawaii-session2-eeg cuteness --rating-source both \
  --source-root /approved/group-publication \
  --normative-archive /approved/normative-summary.zip \
  --output-dir /private/new/cuteness-both
```

The output directory must be new and outside the checkout. Participant publications retain their publication/config authentication. Normative archives retain member hashes, source-binding hashes, category provenance and internally consistent participant/count/time-grid checks; external approval of an archive remains the operator's responsibility.

The fixed original-cohort computation remains available only through the explicit `--reproduce-original-cohort` mode and its required eligibility safeguard. That mode intentionally preserves the original participant inventory, retention fingerprints, private reference hash and publication identity. It is not the interface for an alternative dataset. For new datasets, use the configured participant/group workflows and supply a separately authorised descriptive normative archive when selecting normative figures.
