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
