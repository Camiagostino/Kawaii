# Kawaii study tools — intended use

The supported operator interface is `kawaii-session2-eeg`. For secondary analyses, use `kawaii-session2-eeg secondary` or the existing Python module CLI directly. Most scripts in this directory are not day-to-day analysis commands.

| Script | Purpose | Intended operator |
|---|---|---|
| `run_kawaii_r11_report_descriptive_figures.py` | Regenerate provenance-checked supplementary report figures from selected existing analysis outputs | Report preparation after confirming the source generation |
| `run_kawaii_report_presentation.py` | Generate presentation/report views from versioned analysis evidence | Report preparation, with input validation |
| `build_kawaii_figure_style_preview.py` | Preview approved plotting style | Internal visual QA |
| `build_kawaii_source_exposure_snapshot.py` | Prepare source-exposure validation snapshot | Controlled scientific data preparation |
| `verify_kawaii_source_exposure_snapshot.py` | Verify source-exposure snapshot | Controlled scientific verification |

Do not invoke internal recovery tools just to run ordinary Session 2 analyses. They can have specialised input contracts or production-write modes. This inventory is for developer navigation and **is not automatically a list of scripts for the collaborator release**.

## Unified report-only rating-source selector

`run_kawaii_cuteness_report_figures.py` renders existing descriptive ERP figures using `--rating-source participant|normative|both` and separate authenticated inputs. This is a **report-only**, cohort-specific command, not a new preprocessing/ERP aggregation pipeline or an inferential analysis. See [`docs/CUTENESS_RATING_SOURCE_REPORT_FIGURES.md`](../docs/CUTENESS_RATING_SOURCE_REPORT_FIGURES.md). Normative outputs are exploratory and require scientific approval before external use.

## Unified descriptive analysis command

`run_kawaii_cuteness_analysis.py` is a thin alternative entry point for `kawaii-session2-eeg cuteness --rating-source participant|normative|both`. It wraps the existing authenticated participant publication and NORM-02 descriptive computation/archive reuse. `run_kawaii_norm02_exploratory.py` is an **advanced**, deliberately cohort-specific normative computation helper, not a general AAPS classifier or EEG preprocessing command. See [`docs/CUTENESS_UNIFIED_ANALYSIS.md`](../docs/CUTENESS_UNIFIED_ANALYSIS.md).
