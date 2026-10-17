from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest

from eeg_software.project import get_project_root
from eeg_software.workflows.kawaii_session2.figure_style import (
    FIGURE_STYLE_SCHEMA_VERSION,
    FigureStyleConfigError,
    figure_style_from_mapping,
    load_figure_style,
)
from eeg_software.workflows.kawaii_session2.group_figures import write_group_waveform_figure


class _Evoked:
    times = np.asarray([0.0, 0.1, 0.2])
    data = np.asarray([[0.0, 1e-6, 0.0]])


def test_repository_figure_style_loads() -> None:
    style = load_figure_style(root=get_project_root())

    assert style.dpi == 320
    assert style.condition_color("recognition", "hit") == "#173B63"
    assert style.condition_color("cuteness", "high") == "#163A5F"
    assert style.condition_line_style("recognition", "miss") == "dashed"
    assert style.condition_line_width("recognition", "miss", roi=True) == 1.65
    assert style.condition_label("hit_minus_correct_rejection") == "Hit − Correct rejection"
    assert style.figure_size("waveform") == (8.2, 4.6)
    assert style.primary_window_shading["enabled"] is True
    assert style.topography["common_condition_scale"] is True
    assert style.topography["colormap_colors"][2] == "#F2F1ED"


def test_missing_optional_style_keys_use_safe_defaults(tmp_path: Path) -> None:
    path = tmp_path / "style.yaml"
    path.write_text(
        f"schema_version: {FIGURE_STYLE_SCHEMA_VERSION}\ndpi: 144\n",
        encoding="utf-8",
    )

    style = load_figure_style(config_path=path)

    assert style.dpi == 144
    assert style.line_widths["waveform"] == 1.5
    assert style.figure_size("statistics") == (8.0, 4.5)
    assert style.legend["location"] == "best"


@pytest.mark.parametrize(
    "contents, expected",
    [
        (f"schema_version: {FIGURE_STYLE_SCHEMA_VERSION}\nrois: [Fz, Cz]\n", "Unknown figure style key"),
        (f"schema_version: {FIGURE_STYLE_SCHEMA_VERSION}\ndpi: not-a-number\n", "dpi must be a positive integer"),
        ("schema_version: [unterminated\n", "not valid YAML"),
    ],
)
def test_malformed_or_scientific_style_schema_fails_clearly(
    tmp_path: Path, contents: str, expected: str
) -> None:
    path = tmp_path / "style.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(FigureStyleConfigError, match=expected):
        load_figure_style(config_path=path)


def test_group_waveform_plot_consumes_figure_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    style = figure_style_from_mapping(
        {
            "schema_version": FIGURE_STYLE_SCHEMA_VERSION,
            "dpi": 111,
            "figure_facecolor": "#FAFAFA",
            "figure_sizes": {"waveform": [4.0, 3.0]},
            "colors": {"recognition_conditions": {"hit": "#123456"}},
        }
    )
    captured: dict[str, object] = {}

    def _capture_savefig(figure, path, **kwargs):
        captured["size"] = tuple(figure.get_size_inches())
        captured["dpi"] = kwargs.get("dpi")
        captured["facecolor"] = kwargs.get("facecolor")
        captured["line_color"] = figure.axes[0].lines[0].get_color()

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _capture_savefig)

    output = write_group_waveform_figure(
        "recognition",
        {"hit": _Evoked()},
        tmp_path / "waveform.png",
        figure_style=style,
    )

    assert output == tmp_path / "waveform.png"
    assert captured == {
        "size": (4.0, 3.0),
        "dpi": 111,
        "facecolor": "#FAFAFA",
        "line_color": "#123456",
    }
