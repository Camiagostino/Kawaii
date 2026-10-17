# Reproduce the existing Kawaii Session 2 analyses

This guide is for authorised members of the Kawaii study group. The repository contains the **corrected main EEG and behavioural analysis code**, not the original participant recordings or approved human-review records. Consult [Original-cohort reference inputs](ORIGINAL_COHORT_REFERENCE.md) before claiming a numerical reproduction.

**Status (16 October 2026):** original-cohort input categories are present in the internal research repository and some file identities are recorded. The complete accepted Gate 3–5 generation selection, behavioural-result provenance and output baselines have **not** yet been established for this GitHub checkout. The 375 passing synthetic collaborator tests are software checks, not an end-to-end reproduction.

## Obtain approved study material

Obtain from the study lead/lab's authorised storage, **not Git history**:

- BrainVision `.vhdr`, `.vmrk` and `.eeg` EEG recordings; Session 2 schedules and behavioural logs.
- Session 1 task logs, seen-image/exposure evidence and recognition-pool files needed for old/new classification.
- An approved run configuration (`config/session2_eeg.yaml`, kept Git-ignored) with original cohort inclusion/exclusion and manual overrides.
- The matching source-exposure manifest and disposition CSV. Confirm the manifest artifact ID is the value pinned in the run configuration, and that the disposition table matches the manifest's recorded artifact identity.
- Reviewed bad-channel and ICA-component decisions from the approved manual-review directory, plus relevant epoch/segment exclusions.
- Selected **corrected** Gate 3, participant ERP (Gate 4) and group (Gate 5) generation/current manifests, and reference figures/tables. Exclude the quarantined R11 QC V2 branch and superseded/pre-correction results.

If any source identity or review gate fails, **stop**. Do not invent exposure authority, disable validation or mix output generations to make processing continue.

## Set up

Use Python 3.11 and install from the repository root:

```bash
python -m pip install -e .
kawaii-session2-eeg --help
kawaii-session2-eeg participant --help
kawaii-session2-eeg group --help
```

Use the lab-approved config, never the synthetic `config/session2_eeg.example.yaml` for reproducing original participants. Make sure configured paths resolve to approved study storage. **Use a fresh reproduction output namespace** and keep accepted publications unchanged. A fresh end-to-end run should start only after the original accepted publication identities have been signed off.

## Main EEG workflow

These are individual validated **stage entry points**, not a one-click, unattended pipeline. Check prerequisites, review decisions and publication identities at each transition:

```bash
# After data, source identity and review prerequisites are approved:
kawaii-session2-eeg participant gate3 --root . --config config/session2_eeg.yaml --subjects <approved_ids>

# After accepted Gate 3C epochs/metadata and review gates exist:
kawaii-session2-eeg participant recognition --root . --config config/session2_eeg.yaml --subjects <approved_ids>
kawaii-session2-eeg participant cuteness --root . --config config/session2_eeg.yaml --subjects <approved_ids>

# Only once Gate 4 publications and group membership are validated:
kawaii-session2-eeg group prepare --root . --config config/session2_eeg.yaml
kawaii-session2-eeg group figures --root . --config config/session2_eeg.yaml
kawaii-session2-eeg group statistics --root . --config config/session2_eeg.yaml
```

`gate3` includes manual bad-channel and ICA review gates. Stop and perform required human review when requested. The corrected epoch-coordinate handling is part of the **ordinary main pipeline**. The separate `r11_replay_preflight.py` is an advanced safeguard for optional isolated correction replay, **not** another analysis pipeline.

## Behavioural recognition, ratings and secondary ERP analyses

Recognition performance/response bias and rating–memory analyses use the **original, source-verified behavioural data**, not simply the EEG-retained epochs. Copy the repository's secondary YAML to a Git-ignored local file (for example, `workspace/kawaii/session2/local_config/secondary.yaml`). Replace synthetic subjects/placeholders with the approved cohort, verified source logs and accepted output tables. Keep the scientific contrasts and multiplicity-correction settings unchanged unless the group authorises a methodological revision.

```bash
kawaii-session2-eeg behaviour --root . --config config/session2_eeg.yaml \
  --secondary-config workspace/kawaii/session2/local_config/secondary.yaml --check-inputs

# Only when the input preflight and source-eligibility review pass:
kawaii-session2-eeg behaviour --root . --config config/session2_eeg.yaml \
  --secondary-config workspace/kawaii/session2/local_config/secondary.yaml

# For secondary ERP comparisons, verify the selected accepted Gate 5 tables first:
kawaii-session2-eeg secondary --root . --config config/session2_eeg.yaml \
  --secondary-config workspace/kawaii/session2/local_config/secondary.yaml \
  --stages true-vs-false-erp --check-inputs
```

`--check-inputs` checks selected inputs, **not** scientific equivalence of publication generations. Use `kawaii-session2-eeg secondary --help` for other stages.

## Reproduction acceptance

Before calling a run a reproduction, record the repository commit, exact config and source hashes, cohort/eligibility counts, review decisions, all Gate 3–5 generation IDs, model warnings and output hashes. Compare the resulting statistical tables and figures against **study-lead-approved references**; document any expected nondeterminism or discrepancies. See [Original-cohort reference inputs](ORIGINAL_COHORT_REFERENCE.md) for the current identity checklist and open approvals.
