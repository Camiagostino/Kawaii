# Kawaii study — Session 2 EEG and behavioural analyses

This directory contains **Session 2 EEG preprocessing, recognition/cuteness ERP analyses, behavioural recognition and rating analyses**, study configurations, tests and experimental-task scripts.

## Quick start

Use **Python 3.11**, preferably on Linux/WSL2. From `analysis/eeg_behaviour_session2/` (the directory containing `pyproject.toml`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`.

Before running analyses, prepare local configurations from the files in `config/`:

| Local configuration (Git-ignored) | Purpose |
|---|---|
| `config/session2_eeg.yaml` | EEG parameters, participants, source paths and QC settings |
| `config/kawaii_session2_secondary_analyses.yaml` | Behavioural/secondary analyses, participants and input/output paths |

Use the corresponding `.example.yaml` templates for another dataset, or the `.original_cohort.yaml` configurations when reproducing the original study. Verify all file paths and source identities; the original secondary configuration may point to historical output directories rather than the intended input generation. Keep working configurations and study data outside version control.

## Running the analyses

The `kawaii-session2-eeg` command runs the EEG and behavioural workflows. Run commands from the analysis directory after activating the environment. Use `kawaii-session2-eeg --help` and its subcommand help for additional options.

### EEG analysis

| Stage | What it does | Command |
|---|---|---|
| Participant EEG processing | Preprocessing, ICA, QC and epoching | `kawaii-session2-eeg participant gate3 --subjects SUBJECT_ID` |
| Participant ERPs | Recognition and cuteness analyses | `kawaii-session2-eeg participant recognition --subjects SUBJECT_ID` and `kawaii-session2-eeg participant cuteness --subjects SUBJECT_ID` |
| Group analysis | Aggregation, figures and statistics | Commands below |

Replace `SUBJECT_ID` with the intended identifier and configure its source files. EEG processing may require manual bad-channel and ICA review before later stages can run. After the participant analyses:

```bash
kawaii-session2-eeg group prepare
kawaii-session2-eeg group figures
kawaii-session2-eeg group statistics
```

To add participants to an existing study, prepare their recording, Session 1 exposure/recognition evidence, Session 2 logs, schedule and reviewed source records. Inspect the intake before writing changes:

```bash
kawaii-session2-eeg participant intake --subjects SUBJECT_ID --dry-run
```

Review the dry-run output and approved study configuration before invoking intake without `--dry-run`. The intake expects the corresponding experiment's source-exposure authority; it does not construct an unverified authority from an empty dataset.

### Behavioural and secondary analysis

The behavioural workflow calculates recognition performance (including **d′ and response bias**) and associations between ratings and recognition. After preparing `config/kawaii_session2_secondary_analyses.yaml`:

```bash
kawaii-session2-eeg behaviour --check-inputs
kawaii-session2-eeg behaviour
```

The behavioural workflow uses source-verified behavioural records and approved eligibility, **not EEG artefact rejection**.

For optional secondary ERP comparisons, inspect the input tables and use the corresponding stages:

```bash
kawaii-session2-eeg secondary --stages true-vs-false-erp --check-inputs
kawaii-session2-eeg secondary --stages true-vs-false-erp
```

Other available stages include `recognition-outcome-erp` and `arousal-erp`; use `kawaii-session2-eeg secondary --help` for their options. The arousal stage reads an existing coefficient rather than fitting a new model.

### Descriptive cuteness ERP curves

Use the same command to render ERP curves categorised by **participant ratings**, **normative AAPS ratings**, or **both**. These are separate rating sources, and this command does not rerun EEG preprocessing or fit a continuous normative model.

| Rating source | Required input | Categories |
|---|---|---|
| `participant` | Authenticated group publication, supplied with `--source-root` | Participant cuteness ratings: 1–2, 3–4, 5–6 and 7–9 |
| `normative` | Authenticated descriptive archive, supplied with `--normative-archive` | AAPS cuteness scores: lower ≤5.0 and higher >6.5 |
| `both` | Both inputs | Separate figures for each rating source |

```bash
kawaii-session2-eeg cuteness --rating-source participant \
  --source-root /path/to/group-publication \
  --output-dir /path/to/new/participant-curves

kawaii-session2-eeg cuteness --rating-source normative \
  --normative-archive /path/to/normative-summary.zip \
  --output-dir /path/to/new/normative-curves

kawaii-session2-eeg cuteness --rating-source both \
  --source-root /path/to/group-publication \
  --normative-archive /path/to/normative-summary.zip \
  --output-dir /path/to/new/both-curves
```

Select a **new output directory outside the checkout**. The source archive must contain its expected provenance and matching waveform/participant metadata; archive integrity checks do not supply the underlying data or grant access rights. For reproducing the original study's normative aggregation rather than using an existing archive, inspect `kawaii-session2-eeg cuteness --help` and use the explicit `--reproduce-original-cohort` mode with its required original sources and eligibility checks. That mode is not intended for alternative cohorts.

The `--root` flag selects another analysis directory; `--config` and `--secondary-config` select alternative YAML configurations.

## Study code layout

```text
config/                 Scientific settings and configuration templates
src/eeg_software/
  workflows/kawaii_session2/
    participant_workflow.py   Participant EEG and ERP stages
    group_workflow.py         Group-level EEG analyses
    secondary_analyses/       Behavioural and additional ERP analyses
    application_cli.py        Command-line interface
tools/                  Additional utilities
scripts/                Compatibility directory
tests/                  Synthetic tests
matlab/                 Experimental-task and schedule scripts
```

EEG processing is implemented across modules including `preprocessing.py`, `ica.py`, `epoching.py` and `erp.py`.

## Data and reproducibility

Study data are **not bundled**. Obtain approved EEG recordings, behavioural and exposure records, local configuration, and reviewed QC/ICA decisions from the study's authorised storage. Keep inputs and generated results outside Git. When reproducing an existing analysis, verify the selected source and output generations before running the corresponding stage; do not overwrite accepted outputs.

For the Session 2 acquisition task and schedule-generation instructions, see [`matlab/README_SESSION2_recognition_task.md`](matlab/README_SESSION2_recognition_task.md).

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The portable test suite uses synthetic fixtures and does not require participant recordings or the private AAPS reference.
