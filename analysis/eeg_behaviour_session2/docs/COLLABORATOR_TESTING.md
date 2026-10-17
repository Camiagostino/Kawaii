# Testing the collaborator Kawaii Session 2 analysis repository

## Portable tests

From the checkout root, in the supported **Python 3.11** environment:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The curated release keeps **38** `test_*.py` files (including the collaborator example-config and release-contract tests). Tests exercise signal/QC, analysis logic, provenance/identity and the Kawaii EEG/behavioural CLI using synthetic examples. They must run **without** `config/session2_eeg.yaml`, participant recordings or pilot fixtures. The config example is tested using a temporary filesystem fixture.

The full historical development test suite remains in the original local research repository. It includes study-cohort fingerprint and results regressions, Phase 2b validation scripts, R11 recovery/publishing checks and tests dependent on the internal configuration and authorised data. Their absence from the collaborator release is **not** evidence that those scientific obligations were waived or passed.

Tests check software behaviour, not participant-level study validity. To reproduce actual study results, obtain approved EEG/behavioural data, source-exposure manifests, reviewed QC/ICA decisions and run-specific configuration separately. Do not commit them to GitHub.

## Configurations

- `config/session2_eeg.example.yaml`: synthetic collaborator template; the operator copies to ignored `config/session2_eeg.yaml` for real runs.
- `config/kawaii_session2_secondary_analyses.example.yaml`: synthetic secondary template, with `USER_SUPPLIED` paths and fixed inference parameters. Copy it to an ignored local analysis file for actual datasets.
- `config/kawaii_session2_figure_style.yaml`: approved figure-style authority pinned by SHA-256 in the secondary template.

Do not mistake a passing test profile for confirmation of authorisation to use or share raw/participant-level EEG or behavioural data.
