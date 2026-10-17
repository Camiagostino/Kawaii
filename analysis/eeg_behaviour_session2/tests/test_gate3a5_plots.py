from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_software.workflows.kawaii_session2.plots import (  # noqa: E402
    figure_filename,
    gate3a_figure_paths,
    plot_bad_channel_summary,
    plot_correlation_qc_summary,
    plot_psd,
    plot_raw_overview,
    save_figure_with_provenance,
)
from eeg_software.workflows.kawaii_session2.preprocessing import raw_fif_output_paths  # noqa: E402


class TestGate3A5Plots(unittest.TestCase):
    def test_figure_filename_and_paths_are_explicit(self) -> None:
        self.assertEqual(figure_filename("sub005", "part1", "line noise"), "sub005_part1_line_noise.png")
        paths = gate3a_figure_paths("qc", "sub005", "part1")
        self.assertEqual(paths["psd_figure"], Path("qc") / "psd.png")
        self.assertEqual(paths["correlation_qc_figure"], Path("qc") / "poor_correlation_qc.png")

    def test_raw_fif_output_names_are_explicit(self) -> None:
        paths = raw_fif_output_paths("derivatives", "sub005", "part1")
        self.assertEqual(paths["raw_fif"], Path("derivatives") / "sub005_part1_raw_original.fif")
        self.assertEqual(paths["filtered_raw_fif"], Path("derivatives") / "sub005_part1_raw_filtered.fif")

    def test_figure_save_writes_provenance_sidecar(self) -> None:
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "session2_eeg.yaml"
            config.write_text("preprocessing: {}\n", encoding="utf-8")
            input_file = root / "input.csv"
            input_file.write_text("x\n1\n", encoding="utf-8")
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            output = root / "figure.png"

            save_figure_with_provenance(
                fig,
                output,
                subject="sub005",
                recording_part="part1",
                root=root,
                config_path=config,
                input_files=[input_file],
                parameters_summary={"figure_type": "synthetic"},
                warnings=[],
            )

            sidecar = output.with_suffix(".png.provenance.json")
            self.assertTrue(output.exists())
            self.assertTrue(sidecar.exists())
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["pipeline_stage"], "gate3a_qc_figure")
            self.assertEqual(payload["input_files"], [str(input_file)])
            self.assertEqual(payload["output_file"], str(output))

    def test_raw_overview_uses_configurable_short_window_arguments(self) -> None:
        import numpy as np

        class Raw:
            ch_names = ["Cz", "Fz"]
            times = [0.0, 20.0]
            info = {"sfreq": 100.0}

            def __init__(self):
                self.kwargs = None

            def copy(self):
                return self

            def pick(self, picks):
                self.pick_arg = picks
                return self

            def crop(self, **kwargs):
                self.crop_kwargs = kwargs
                return self

            def get_data(self):
                return np.vstack([np.linspace(-100e-6, 100e-6, 1201), np.zeros(1201)])

        raw = Raw()
        fig = plot_raw_overview(raw, duration=12.0, start=3.0)

        self.assertIsNotNone(fig)
        self.assertEqual(raw.crop_kwargs["tmin"], 3.0)
        self.assertEqual(raw.pick_arg, "eeg")
        self.assertEqual(len(fig.axes), 1)
        self.assertIn("fixed +/-100 uV", fig._suptitle.get_text())
        self.assertEqual([label.get_text() for label in fig.axes[0].get_yticklabels()], ["Cz", "Fz"])

    def test_bad_channel_plot_labels_std_and_separates_proposal_from_confirmation(self) -> None:
        import pandas as pd

        qc = pd.DataFrame(
            {
                "channel": ["Cz", "Fz"],
                "included_in_eeg_qc": [True, True],
                "std_uv": [10.0, 20.0],
                "extreme_std_uv_threshold": [500.0, 500.0],
                "proposed_bad_channel": [True, False],
                "automatic_proposal_reasons": ["poor_correlation", ""],
            }
        )
        decisions = pd.DataFrame({"channel": ["Cz", "Fz"], "manual_confirmed_bad_channel": [False, True]})

        fig = plot_bad_channel_summary(qc, bad_channel_decisions=decisions)

        ax = fig.axes[0]
        self.assertEqual(ax.get_ylabel(), "Standard deviation (uV)")
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertIn("Automatically proposed (not confirmed)", labels)
        self.assertIn("Manually confirmed overlay", labels)
        self.assertEqual(ax.patches[1].get_hatch(), "///")

    def test_correlation_qc_plot_shows_windows_threshold_and_failed_points(self) -> None:
        import pandas as pd

        qc = pd.DataFrame(
            {
                "channel": ["Cz", "Fz"],
                "included_in_eeg_qc": [True, True],
                "minimum_window_p98_abs_correlation": [0.2, 0.7],
                "median_window_p98_abs_correlation": [0.5, 0.8],
                "poor_correlation_threshold": [0.4, 0.4],
                "poor_correlation_proposal": [True, False],
            }
        )

        fig = plot_correlation_qc_summary(qc)

        ax = fig.axes[0]
        self.assertEqual(ax.get_ylabel(), "98th-percentile absolute peer correlation (1 s, QC-only 1-50 Hz)")
        labels = [line.get_label() for line in ax.lines]
        self.assertIn("Median across 1 s windows", labels)
        self.assertIn("Minimum 1 s window", labels)
        self.assertIn("Metric threshold (0.4)", labels)

    def test_psd_plot_uses_db_scaling_and_robust_axis_limits(self) -> None:
        import numpy as np

        class PSD:
            freqs = np.array([1.0, 2.0, 3.0])

            def get_data(self):
                return np.array([[1e-12, 1e-10, 1e-8], [1e-11, 1e-9, 1e-7]])

        class Raw:
            times = np.array([0.0, 10.0])

            def compute_psd(self, **kwargs):
                self.kwargs = kwargs
                return PSD()

        raw = Raw()
        fig = plot_psd(raw, fmin=1.0, fmax=3.0)
        ax = fig.axes[0]

        self.assertEqual(raw.kwargs["picks"], "eeg")
        self.assertEqual(ax.get_ylabel(), "Power spectral density (dB)")
        self.assertLess(ax.get_ylim()[0], ax.get_ylim()[1])


if __name__ == "__main__":
    unittest.main()
