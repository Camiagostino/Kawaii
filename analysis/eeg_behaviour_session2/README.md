# Kawaii study — Session 2 EEG and behavioural analyses

This repository contains **Session 2 EEG preprocessing, recognition/cuteness ERP analyses, behavioural recognition and rating analyses**, study configurations, tests and supporting experimental-task scripts.

## Quick start

Use **Python 3.11**, preferably on Linux/WSL2. From the analysis directory (containing `pyproject.toml`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, if Python 3.11 is installed, create the environment with `py -3.11 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`. Dependencies are installed from `pyproject.toml`.

**Before running analyses:** prepare the study configuration and make the approved data and QC/ICA decisions available. For the original cohort, copy the `.original_cohort.yaml` configurations to the local working filenames below; for new datasets, start from the corresponding `.example.yaml` files.

| Local working configuration (Git-ignored) | Controls |
|---|---|
| `config/session2_eeg.yaml` | EEG parameters, participants, source paths and QC settings |
| `config/kawaii_session2_secondary_analyses.yaml` | Behavioural/secondary analyses, participant selection and input/output paths |

The original-cohort configurations are included in `config/`, but **their source paths and selected outputs must still be verified**. In particular, the original secondary configuration points to historical group-output paths, not automatically to the accepted corrected results. See the [Study analysis guide](docs/STUDY_ANALYSIS_GUIDE.md).

## Running the analyses

The `kawaii-session2-eeg` command runs the EEG and behavioural workflows. Run these commands from the analysis directory after activating the Python environment.

### EEG analysis

| Stage | What it does | Command |
|---|---|---|
| 1. Participant EEG processing | Preprocessing, ICA, QC and epoching | `kawaii-session2-eeg participant gate3 --subjects SUBJECT_ID` |
| 2. Participant ERPs | Recognition and cuteness analyses | `kawaii-session2-eeg participant recognition --subjects SUBJECT_ID` and `kawaii-session2-eeg participant cuteness --subjects SUBJECT_ID` |
| 3. Group results | Aggregate results, generate figures and run statistics | Run the three commands below |

Replace `SUBJECT_ID` with an approved participant identifier. Processing may require manual bad-channel and ICA review before continuing.

After processing participants, run the group stages in order:

```bash
kawaii-session2-eeg group prepare
kawaii-session2-eeg group figures
kawaii-session2-eeg group statistics
```

### Behavioural analysis

The behavioural workflow calculates recognition performance (including **d′ and response bias**) and examines associations between ratings and recognition.

After preparing `config/kawaii_session2_secondary_analyses.yaml` with the intended inputs:

```bash
# Check inputs without generating results
kawaii-session2-eeg behaviour --check-inputs

# Run the analysis
kawaii-session2-eeg behaviour
```

Results are saved to the configured output directory. Behavioural recognition uses source-verified behavioural records and approved eligibility, **not EEG artefact rejection**.

### Additional analyses and options

Exploratory ERP comparisons, response-timing QC, participant intake and advanced options are described in the [Study analysis guide](docs/STUDY_ANALYSIS_GUIDE.md). Use `kawaii-session2-eeg --help` to see available commands.

By default, `--config` uses `config/session2_eeg.yaml` and `--secondary-config` uses `config/kawaii_session2_secondary_analyses.yaml`. Supply either flag to select a different file; `--root` selects another analysis directory.

## Study code layout

```text
config/                 Scientific parameters and configuration templates
src/eeg_software/
  workflows/kawaii_session2/
    participant_workflow.py   Participant EEG and ERP stages
    group_workflow.py         Group-level EEG analyses
    secondary_analyses/       Behavioural and additional ERP analyses
    application_cli.py        Command-line interface
tools/                  Additional reporting and verification utilities
scripts/                Compatibility directory; not a normal analysis entry point
tests/                  Synthetic tests and analysis contracts
matlab/                 Experimental-task and schedule scripts
docs/                   Analysis instructions and supporting documentation
```

The actual EEG processing is implemented across modules such as `preprocessing.py`, `ica.py`, `epoching.py` and `erp.py`, not solely in `participant_workflow.py`.

## Data, configuration and reproducibility

- **Study data are not bundled.** Use approved storage for EEG recordings, behavioural logs, source-exposure records and manual QC/ICA decisions.
- Scientific settings (filtering, referencing, ICA, epoching, artefact rejection and ERP analyses) are defined in the EEG configuration. Keep validated settings unchanged unless the group approves a methodological revision.
- The **original-cohort configuration** uses local `workspace/` paths extensively. The **example configuration** illustrates `data/`, `outputs/` and `user_state/` directories. These local directories are excluded from Git; their locations are configurable.
- Check secondary ERP input tables against the intended accepted analysis generation. `--check-inputs` validates selected files but does **not** independently establish publication lineage. Avoid overwriting accepted results.

See [Reproduce the existing study](docs/REPRODUCE_EXISTING_STUDY.md) or [Analyse additional participants](docs/ADD_NEW_PARTICIPANTS.md) for detailed requirements.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The shared tests use synthetic fixtures and do not require original participant data.
