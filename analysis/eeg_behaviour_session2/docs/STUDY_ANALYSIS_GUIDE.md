# Kawaii Session 2 — analysis operating guide

This guide describes the study's analysis commands and input requirements. Run commands from the analysis directory (containing `pyproject.toml`), using **Python 3.11**, after `python -m pip install -e .`.

## Workflow map

| Analysis | Code | Command |
|---|---|---|
| Participant intake | `participant_intake.py` | `kawaii-session2-eeg participant intake --subjects SUBJECT_ID` |
| Gate 3 EEG processing | `participant_workflow.py` and processing modules | `kawaii-session2-eeg participant gate3 --subjects SUBJECT_ID` |
| Recognition and cuteness Gate 4 ERPs | Participant workflow and `erp.py` | `kawaii-session2-eeg participant recognition --subjects SUBJECT_ID` or `kawaii-session2-eeg participant cuteness --subjects SUBJECT_ID` |
| Gate 5 aggregation, figures and statistics | `group_workflow.py` | `kawaii-session2-eeg group prepare`, `group figures`, `group statistics` |
| d′, response bias and recognition outcomes | `secondary_analyses/recognition_behavior.py` | `kawaii-session2-eeg behaviour` |
| Rating–memory associations | `secondary_analyses/rating_memory.py` | `kawaii-session2-eeg behaviour` |
| Exploratory recognition ERP comparisons | `secondary_analyses/recognition_erp.py` | `kawaii-session2-eeg secondary --stages recognition-outcome-erp true-vs-false-erp` |
| Existing arousal coefficient extraction | `secondary_analyses/arousal_erp.py` | `kawaii-session2-eeg secondary --stages arousal-erp` |
| Retrospective response-timing QC | Timing-QC workflow | `kawaii-session2-eeg timing-qc` |

The `behaviour` shortcut runs **recognition-behavior** and **rating-memory** only. The `secondary` command defaults to those same stages; ERP stages must be requested explicitly. Use `kawaii-session2-eeg --help`, `participant --help` or `group --help` to inspect options.

## Input configuration

The repository includes two sets of YAML files:

| Purpose | EEG | Behavioural/secondary |
|---|---|---|
| Original cohort | `session2_eeg.original_cohort.yaml` | `kawaii_session2_secondary_analyses.original_cohort.yaml` |
| New cohort/template | `session2_eeg.example.yaml` | `kawaii_session2_secondary_analyses.example.yaml` |

Choose the appropriate files from `config/` and copy them to **Git-ignored working configurations**:

```text
config/session2_eeg.yaml
config/kawaii_session2_secondary_analyses.yaml
```

For example, on Linux/WSL2, to prepare **new** working copies for the original cohort (do not overwrite an existing reviewed working copy):

```bash
cp config/session2_eeg.original_cohort.yaml config/session2_eeg.yaml
cp config/kawaii_session2_secondary_analyses.original_cohort.yaml config/kawaii_session2_secondary_analyses.yaml
```

For new participants or another dataset, copy the corresponding `.example.yaml` files instead. Set approved participant IDs, data and output paths, source-exposure records and manual-review locations. Keep the scientific conditions, correction families, adjustment rules and figure-style identity unchanged unless the group approves a methodological revision. The source-exposure manifest and QC/ICA review records must be obtained from authorised lab storage; they are not included in the code repository.

**Important:** the original secondary YAML points to `workspace/kawaii/session2/analysis/group/`, which may contain **historical outputs rather than the accepted corrected publications**. Confirm input lineage before using those tables; choose a new output directory for reruns.

## Running behavioural and secondary analyses

**Behavioural analysis:** reads source-exposure evidence and original behavioural logs, verifies file size and SHA-256, applies reviewed recognition eligibility/disposition, and checks outcome counts. It does not use EEG-rejected epochs to calculate behavioural performance.

```bash
# Read-only input check
kawaii-session2-eeg behaviour --check-inputs

# Run recognition and rating–memory analyses
kawaii-session2-eeg behaviour
```

**Secondary ERP analysis:** reads existing group ERP tables; it does **not** rerun EEG preprocessing or establish accepted publication lineage merely by opening a CSV.

```bash
# Check an exploratory ERP comparison's inputs
kawaii-session2-eeg secondary --stages true-vs-false-erp --check-inputs

# Run it after verifying the intended inputs
kawaii-session2-eeg secondary --stages true-vs-false-erp
```

If you use a different local secondary YAML, specify it with `--secondary-config PATH_TO_FILE`. Without `--check-inputs`, the secondary workflows write CSV, PNG and manifest results into the configured `output_dir`.

The **arousal-coefficient stage** extracts a coefficient from an existing model; it does not refit one. If a corrected model excludes arousal, do not attribute an older coefficient to the corrected fit.

## Scripts and maintenance

`tools/README.md` describes supporting reporting and verification tools. `scripts/README.md` documents a compatibility directory; historical Phase 2b/R11 recovery scripts are not normal analysis entry points.

See [Reproducing the existing study](REPRODUCE_EXISTING_STUDY.md), [Adding participants](ADD_NEW_PARTICIPANTS.md) and [Collaborator testing](COLLABORATOR_TESTING.md) for further details.
