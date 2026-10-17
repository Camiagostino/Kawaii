# Analyse additional Kawaii participants

The Kawaii Session 2 pipeline supports adding participants to an **established, source-verified study**. The code has not been demonstrated to initialise a fully empty source-exposure authority. Do not create invented manifests to bypass that constraint.

## Before participant intake

1. Work in a **separate output namespace** and back up the established cohort configuration, exposure manifest and disposition table. Never overwrite the existing cohort's accepted publications.
2. Copy `config/session2_eeg.example.yaml` to a **Git-ignored local run config**, then use the approved scientific settings, participant naming convention, recording mappings and review/output paths. Do not use the synthetic example subjects for real data.
3. Obtain the approved study's existing source-exposure manifest and matching disposition CSV; verify the config's expected artifact ID before appending anything.
4. Supply original EEG `.vhdr/.vmrk/.eeg` files, Session 1 execution/seen-image and recognition pool evidence, Session 2 logs and schedules. The current standard intake expects **400 Session 1 and Session 2 trials**, and **8 Session 2 blocks** under the reviewed task protocol.
5. Ask the study lead to review restarts, overlapping acquisitions, altered trial counts, abnormal delays or missing source evidence as protocol exceptions. Do not relax validations to accommodate unexpected inputs.

## Intake and scientific review

From the study checkout:

```bash
# Source-readiness inspection; does not persist config/exposure changes
kawaii-session2-eeg participant intake --root . --config config/session2_eeg.yaml \
  --subjects <new_subject_id> --dry-run

# Only after human review of the dry-run summary:
kawaii-session2-eeg participant intake --root . --config config/session2_eeg.yaml \
  --subjects <new_subject_id>
```

The **non-dry-run intake writes** to the selected core configuration and source-exposure authority records. Afterward, review the new artifact ID, appended participant entry and disposition rows; verify that existing participant identities/decisions remained unchanged. Obtain approval for the expanded cohort definition.

## Run the normal analysis stages

Use the **same corrected EEG pipeline** as the original cohort: `participant gate3` (including bad-channel and ICA review), then `participant recognition`/`participant cuteness`, then `group prepare`, `group figures`, and `group statistics` after cohort approval. Run `behaviour` with a secondary YAML using the expanded participant set and the correct verified behavioural sources; preflight ERP tables separately before `secondary` analyses.

Do not treat validation-only flags (`--skip-ica*`), `--clean-subject-output`, or optional R11 corrected-replay flags as standard workflow shortcuts. Review output generations and compare QC/statistics to the original cohort without overwriting it.

## Limits and scope

- Source-exposure eligibility requires the Session 1 evidence as well as Session 2 data. This repository is primarily **Session 2 EEG and recognition/rating behaviour**, not a complete general-purpose Session 1 behavioural-analysis package.
- The historical, eight-person R11 descriptive-figure runner is not a generic new-participant workflow. Standard group figure generation is.
- Passing the fixture-independent unit tests does not replace a scientifically reviewed first real-data run for a new participant.
