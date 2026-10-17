"""Secondary and exploratory analyses for Kawaii Session 2.

These modules consume validated Session 2 behavioural and EEG outputs and reuse
existing Kawaii eligibility, statistics, and figure-style helpers. They do not
rerun preprocessing, ICA, epoching, or participant ERP generation.
"""

from .config import load_secondary_analysis_config
from .arousal_erp import extract_existing_arousal_erp, plot_existing_arousal_erp_coefficient
from .recognition_behavior import (
    compute_signal_detection_metrics,
    dprime_inference,
    load_source_verified_behavioural_trials,
    signal_detection_group_summary,
)
from .rating_memory import (
    compute_rating_memory_family,
    plot_rating_memory_forest,
    summarise_participant_rating_means,
)
from .recognition_erp import (
    compute_erp_family,
    load_recognition_measurements,
    validate_secondary_against_gate5,
)
from .workflow import SecondaryAnalysisRunResult, run_secondary_analyses

__all__ = [
    "SecondaryAnalysisRunResult",
    "extract_existing_arousal_erp",
    "compute_erp_family",
    "compute_rating_memory_family",
    "compute_signal_detection_metrics",
    "dprime_inference",
    "load_recognition_measurements",
    "plot_existing_arousal_erp_coefficient",
    "plot_rating_memory_forest",
    "load_secondary_analysis_config",
    "load_source_verified_behavioural_trials",
    "run_secondary_analyses",
    "signal_detection_group_summary",
    "summarise_participant_rating_means",
    "validate_secondary_against_gate5",
]
