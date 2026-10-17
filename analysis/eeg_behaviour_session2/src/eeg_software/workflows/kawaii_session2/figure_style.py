from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from eeg_software.project import get_project_root


FIGURE_STYLE_SCHEMA_VERSION = "kawaii_session2_figure_style_v1"
DEFAULT_FIGURE_STYLE_PATH = Path("config") / "kawaii_session2_figure_style.yaml"


class FigureStyleConfigError(ValueError):
    """Raised when presentation-only figure styling is invalid."""


DEFAULT_FIGURE_STYLE: dict[str, Any] = {
    "schema_version": FIGURE_STYLE_SCHEMA_VERSION,
    "dpi": 200,
    "figure_facecolor": "#FFFFFF",
    "fonts": {
        "title_size": 12,
        "panel_title_size": 10,
        "difference_title_size": 11,
        "axis_label_size": 10,
        "tick_label_size": 10,
        "legend_size": 8,
        "compact_legend_size": 7,
    },
    "legend": {
        "location": "best",
        "frameon": True,
        "columns": 1,
        "shared_roi": False,
        "shared_location": "upper center",
        "shared_columns": 4,
        "shared_bbox_to_anchor": [0.5, 0.96],
    },
    "axes": {
        "grid": False,
        "grid_axis": "y",
        "grid_alpha": 0.12,
        "hide_top_spine": True,
        "hide_right_spine": True,
        "spine_color": "#B8C0C8",
        "spine_width": 0.8,
        "reference_alpha": 0.65,
    },
    "labels": {
        "branches": {"recognition": "Recognition", "cuteness": "Cuteness"},
        "conditions": {
            "hit": "Hit",
            "miss": "Miss",
            "correct_rejection": "Correct rejection",
            "false_alarm": "False alarm",
            "hit_minus_correct_rejection": "Hit − Correct rejection",
            "low": "Low",
            "medium_low": "Medium-low",
            "medium_high": "Medium-high",
            "high": "High",
        },
    },
    "line_widths": {
        "waveform": 1.5,
        "roi_waveform": 1.2,
        "secondary_waveform": 1.2,
        "secondary_roi_waveform": 1.1,
        "difference_waveform": 1.8,
        "reference": 0.8,
        "participant_roi_reference": 0.7,
    },
    "line_styles": {
        "recognition_conditions": {
            "hit": "solid",
            "correct_rejection": "solid",
            "miss": "solid",
            "false_alarm": "solid",
            "hit_minus_correct_rejection": "solid",
        },
        "cuteness_bins": {
            "low": "solid",
            "medium_low": "solid",
            "medium_high": "solid",
            "high": "solid",
        },
    },
    "colors": {
        "recognition_conditions": {
            "hit": "#1F77B4",
            "miss": "#FF7F0E",
            "false_alarm": "#2CA02C",
            "correct_rejection": "#D62728",
            "hit_minus_correct_rejection": "#9467BD",
        },
        "cuteness_bins": {
            "low": "#1F77B4",
            "medium_low": "#FF7F0E",
            "medium_high": "#2CA02C",
            "high": "#D62728",
        },
        "bars": {
            "default": "#4C78A8",
            "included": "#4C78A8",
            "excluded": "#E45756",
            "threshold_met": "#4C78A8",
            "threshold_failed": "#E45756",
            "effect_size": "#59A14F",
        },
        "difference_waveform": "#333333",
        "reference": "#777777",
        "primary_window": "#DCE6F1",
    },
    "topography": {
        "colormap": "RdBu_r",
        "difference_colormap": "RdBu_r",
        "colormap_colors": ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"],
        "symmetric_voltage_limits": False,
        "center_difference_at_zero": False,
        "common_condition_scale": False,
        "contour_count": 6,
        "contour_color": "#555555",
        "contour_line_width": 0.5,
        "contour_alpha": 0.65,
        "sensor_color": "#444444",
        "sensor_size": 2.0,
        "sensor_alpha": 0.75,
        "colorbar_unit_size": 8,
    },
    "figure_sizes": {
        "waveform": [8.0, 4.5],
        "difference_waveform": [8.0, 4.5],
        "trial_count": [8.0, 4.5],
        "participant_trial_count": [9.0, 4.5],
        "inclusion": [5.5, 4.0],
        "statistics": [8.0, 4.5],
        "topography": [4.8, 3.8],
        "group_roi_waveform": {"width": 8.0, "minimum_height": 3.0, "height_per_panel": 2.3},
        "participant_roi_waveform": {"width": 8.0, "minimum_height": 3.0, "height_per_panel": 2.4},
        "effect_size": {"width": 10.0, "minimum_height": 4.5, "height_per_row": 0.25},
    },
    "primary_window_shading": {"enabled": False, "alpha": 0.18},
}


@dataclass(frozen=True)
class FigureStyle:
    dpi: int
    figure_facecolor: str
    fonts: Mapping[str, float]
    legend: Mapping[str, Any]
    axes: Mapping[str, Any]
    labels: Mapping[str, Any]
    line_widths: Mapping[str, float]
    line_styles: Mapping[str, Any]
    colors: Mapping[str, Any]
    topography: Mapping[str, Any]
    figure_sizes: Mapping[str, Any]
    primary_window_shading: Mapping[str, Any]

    def condition_color(self, branch: str, condition: str) -> str | None:
        palette = "recognition_conditions" if str(branch) == "recognition" else "cuteness_bins"
        value = dict(self.colors.get(palette, {}) or {}).get(str(condition))
        return str(value) if value else None

    def condition_label(self, condition: str) -> str:
        value = dict(self.labels.get("conditions", {}) or {}).get(str(condition))
        return str(value) if value else str(condition).replace("_", " ").capitalize()

    def branch_label(self, branch: str) -> str:
        value = dict(self.labels.get("branches", {}) or {}).get(str(branch))
        return str(value) if value else str(branch).replace("_", " ").title()

    def condition_line_style(self, branch: str, condition: str) -> str:
        palette = "recognition_conditions" if str(branch) == "recognition" else "cuteness_bins"
        return str(dict(self.line_styles.get(palette, {}) or {}).get(str(condition), "solid"))

    def condition_line_width(self, branch: str, condition: str, *, roi: bool = False) -> float:
        primary = str(branch) != "recognition" or str(condition) in {"hit", "correct_rejection"}
        if roi:
            key = "roi_waveform" if primary else "secondary_roi_waveform"
        else:
            key = "waveform" if primary else "secondary_waveform"
        return float(self.line_widths[key])

    def topography_colormap(self, *, difference: bool = False) -> Any:
        from matplotlib.colors import LinearSegmentedColormap

        name = str(self.topography["difference_colormap" if difference else "colormap"])
        return LinearSegmentedColormap.from_list(name, list(self.topography["colormap_colors"]))

    def color(self, key: str, default: str | None = None) -> str | None:
        value = self.colors.get(key, default)
        return str(value) if value is not None else None

    def bar_color(self, key: str = "default") -> str:
        return str(dict(self.colors.get("bars", {}) or {}).get(key, "#4C78A8"))

    def figure_size(self, key: str, *, count: int | None = None) -> tuple[float, float]:
        value = self.figure_sizes[key]
        if isinstance(value, Mapping):
            if count is None:
                raise FigureStyleConfigError(f"Figure size '{key}' requires a panel/row count.")
            width = float(value["width"])
            height = max(float(value["minimum_height"]), float(value["height_per_panel" if "height_per_panel" in value else "height_per_row"]) * count)
            return width, height
        return float(value[0]), float(value[1])


def default_figure_style() -> FigureStyle:
    """Return the validated built-in presentation defaults."""

    return figure_style_from_mapping(DEFAULT_FIGURE_STYLE)


def load_figure_style(
    *, root: str | Path | None = None, config_path: str | Path | None = None
) -> FigureStyle:
    """Load and validate the independent Kawaii Session 2 figure style file."""

    project_root = Path(root).resolve() if root is not None else get_project_root().resolve()
    path = Path(config_path) if config_path is not None else project_root / DEFAULT_FIGURE_STYLE_PATH
    if not path.is_absolute():
        path = project_root / path
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if config_path is None:
            return default_figure_style()
        raise FigureStyleConfigError(f"Figure style configuration does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise FigureStyleConfigError(f"Figure style configuration is not valid YAML: {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise FigureStyleConfigError(f"Figure style configuration must be a mapping: {path}")
    try:
        return figure_style_from_mapping(loaded)
    except FigureStyleConfigError as exc:
        raise FigureStyleConfigError(f"Invalid figure style configuration {path}: {exc}") from exc


def figure_style_from_mapping(value: Mapping[str, Any]) -> FigureStyle:
    """Validate style data, filling omitted optional presentation keys."""

    if not isinstance(value, Mapping):
        raise FigureStyleConfigError("Figure style must be a mapping.")
    _reject_unknown_keys(value, DEFAULT_FIGURE_STYLE, "figure style")
    schema = value.get("schema_version")
    if schema != FIGURE_STYLE_SCHEMA_VERSION:
        raise FigureStyleConfigError(
            f"schema_version must be '{FIGURE_STYLE_SCHEMA_VERSION}', got {schema!r}."
        )
    merged = _deep_merge(DEFAULT_FIGURE_STYLE, value)
    _validate_style(merged)
    return FigureStyle(
        dpi=int(merged["dpi"]),
        figure_facecolor=str(merged["figure_facecolor"]),
        fonts=dict(merged["fonts"]),
        legend=dict(merged["legend"]),
        axes=dict(merged["axes"]),
        labels=deepcopy(merged["labels"]),
        line_widths=dict(merged["line_widths"]),
        line_styles=deepcopy(merged["line_styles"]),
        colors=deepcopy(merged["colors"]),
        topography=dict(merged["topography"]),
        figure_sizes=deepcopy(merged["figure_sizes"]),
        primary_window_shading=dict(merged["primary_window_shading"]),
    )


def style_axes(ax: Any, style: FigureStyle) -> None:
    """Apply typography and face color without changing plotted content."""

    ax.figure.set_facecolor(style.figure_facecolor)
    ax.set_facecolor(style.figure_facecolor)
    ax.title.set_fontsize(style.fonts["title_size"])
    ax.xaxis.label.set_fontsize(style.fonts["axis_label_size"])
    ax.yaxis.label.set_fontsize(style.fonts["axis_label_size"])
    ax.tick_params(axis="both", labelsize=style.fonts["tick_label_size"])
    axes = style.axes
    if bool(axes["grid"]):
        ax.grid(True, axis=str(axes["grid_axis"]), alpha=float(axes["grid_alpha"]))
    else:
        ax.grid(False)
    if bool(axes["hide_top_spine"]):
        ax.spines["top"].set_visible(False)
    if bool(axes["hide_right_spine"]):
        ax.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color(str(axes["spine_color"]))
        ax.spines[name].set_linewidth(float(axes["spine_width"]))


def style_legend(ax: Any, style: FigureStyle, *, size: float | None = None) -> Any:
    """Create a consistently styled legend for axes with labelled artists."""

    return ax.legend(
        loc=str(style.legend["location"]),
        frameon=bool(style.legend["frameon"]),
        ncol=int(style.legend["columns"]),
        fontsize=float(size if size is not None else style.fonts["legend_size"]),
    )


def style_topography_figure(fig: Any, style: FigureStyle) -> None:
    """Apply non-scientific contour, sensor, unit-label, and margin styling."""

    from matplotlib.collections import LineCollection, PathCollection

    fig.set_facecolor(style.figure_facecolor)
    if not fig.axes:
        return
    map_axis = fig.axes[0]
    for collection in map_axis.collections:
        if isinstance(collection, LineCollection):
            collection.set_color(str(style.topography["contour_color"]))
            collection.set_linewidth(float(style.topography["contour_line_width"]))
            collection.set_alpha(float(style.topography["contour_alpha"]))
        elif isinstance(collection, PathCollection):
            collection.set_edgecolor(str(style.topography["sensor_color"]))
            collection.set_alpha(float(style.topography["sensor_alpha"]))
            collection.set_sizes([float(style.topography["sensor_size"]) ** 2])
    for line in map_axis.lines:
        if line.get_marker() not in {None, "None", ""}:
            line.set_color(str(style.topography["sensor_color"]))
            line.set_markersize(float(style.topography["sensor_size"]))
            line.set_alpha(float(style.topography["sensor_alpha"]))
    if len(fig.axes) > 1:
        colorbar_axis = fig.axes[-1]
        colorbar_axis.title.set_fontsize(float(style.topography["colorbar_unit_size"]))
        colorbar_axis.tick_params(labelsize=float(style.fonts["compact_legend_size"]))
    # MNE may install a constrained-layout engine; disable it before applying
    # the presentation margins to avoid an incompatible-layout warning.
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.08, top=0.86)


def shade_primary_windows(
    ax: Any,
    settings: Mapping[str, Any],
    style: FigureStyle,
    *,
    roi: str | None = None,
) -> None:
    """Shade scientifically configured primary windows when presentation enables it."""

    shading = style.primary_window_shading
    if not bool(shading["enabled"]):
        return
    windows: set[tuple[float, float]] = set()
    hierarchy = dict(settings.get("scientific_hierarchy", {}) or {})
    configured = dict(settings.get("measurement_time_windows", {}) or {})
    for cell in hierarchy.get("primary_cells", []) or []:
        if not isinstance(cell, Mapping):
            continue
        if roi is not None and str(cell.get("roi", "")) != str(roi):
            continue
        window = configured.get(str(cell.get("time_window", "")))
        if isinstance(window, Mapping) and "start_ms" in window and "end_ms" in window:
            windows.add((float(window["start_ms"]), float(window["end_ms"])))
    primary_model = dict(settings.get("primary_model", {}) or {})
    model_window = primary_model.get("window")
    model_roi = dict(primary_model.get("roi", {}) or {}).get("name")
    if (
        isinstance(model_window, Mapping)
        and "start_ms" in model_window
        and "end_ms" in model_window
        and (roi is None or model_roi is None or str(model_roi) == str(roi))
    ):
        windows.add((float(model_window["start_ms"]), float(model_window["end_ms"])))
    for start_ms, end_ms in sorted(windows):
        ax.axvspan(
            start_ms,
            end_ms,
            color=style.color("primary_window", "#DCE6F1"),
            alpha=float(shading["alpha"]),
            linewidth=0,
            zorder=0,
        )


def _deep_merge(defaults: Mapping[str, Any], supplied: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(defaults))
    for key, value in supplied.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _reject_unknown_keys(value: Mapping[str, Any], schema: Mapping[str, Any], path: str) -> None:
    unknown = sorted(set(value).difference(schema))
    if unknown:
        raise FigureStyleConfigError(f"Unknown {path} key(s): {', '.join(unknown)}.")
    for key, nested in value.items():
        if isinstance(nested, Mapping) and isinstance(schema.get(key), Mapping):
            _reject_unknown_keys(nested, schema[key], f"{path}.{key}")


def _validate_style(style: Mapping[str, Any]) -> None:
    _positive_number(style["dpi"], "dpi", integer=True)
    _non_empty_string(style["figure_facecolor"], "figure_facecolor")
    for key, value in style["fonts"].items():
        _positive_number(value, f"fonts.{key}")
    for key, value in style["line_widths"].items():
        _positive_number(value, f"line_widths.{key}")
    valid_line_styles = {"solid", "dashed", "dashdot", "dotted", "-", "--", "-.", ":"}
    for group, values in style["line_styles"].items():
        for key, value in values.items():
            if value not in valid_line_styles:
                raise FigureStyleConfigError(f"line_styles.{group}.{key} is not a supported line style.")
    _non_empty_string(style["legend"]["location"], "legend.location")
    if not isinstance(style["legend"]["frameon"], bool):
        raise FigureStyleConfigError("legend.frameon must be a boolean.")
    _positive_number(style["legend"]["columns"], "legend.columns", integer=True)
    if not isinstance(style["legend"]["shared_roi"], bool):
        raise FigureStyleConfigError("legend.shared_roi must be a boolean.")
    _non_empty_string(style["legend"]["shared_location"], "legend.shared_location")
    _positive_number(style["legend"]["shared_columns"], "legend.shared_columns", integer=True)
    anchor = style["legend"]["shared_bbox_to_anchor"]
    if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
        raise FigureStyleConfigError("legend.shared_bbox_to_anchor must contain [x, y].")
    for index, value in enumerate(anchor):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FigureStyleConfigError(f"legend.shared_bbox_to_anchor[{index}] must be numeric.")
    axes = style["axes"]
    for key in ("grid", "hide_top_spine", "hide_right_spine"):
        if not isinstance(axes[key], bool):
            raise FigureStyleConfigError(f"axes.{key} must be a boolean.")
    if axes["grid_axis"] not in {"x", "y", "both"}:
        raise FigureStyleConfigError("axes.grid_axis must be one of: x, y, both.")
    alpha = axes["grid_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
        raise FigureStyleConfigError("axes.grid_alpha must be between 0 and 1.")
    _non_empty_string(axes["spine_color"], "axes.spine_color")
    _positive_number(axes["spine_width"], "axes.spine_width")
    reference_alpha = axes["reference_alpha"]
    if isinstance(reference_alpha, bool) or not isinstance(reference_alpha, (int, float)) or not 0.0 <= float(reference_alpha) <= 1.0:
        raise FigureStyleConfigError("axes.reference_alpha must be between 0 and 1.")
    for group, labels in style["labels"].items():
        for key, label in labels.items():
            _non_empty_string(label, f"labels.{group}.{key}")
    for group, value in style["colors"].items():
        if isinstance(value, Mapping):
            for key, color in value.items():
                _non_empty_string(color, f"colors.{group}.{key}")
        else:
            _non_empty_string(value, f"colors.{group}")
    for key in ("colormap", "difference_colormap"):
        _non_empty_string(style["topography"][key], f"topography.{key}")
    colors = style["topography"]["colormap_colors"]
    if not isinstance(colors, (list, tuple)) or len(colors) < 3:
        raise FigureStyleConfigError("topography.colormap_colors must contain at least three colors.")
    for index, color in enumerate(colors):
        _non_empty_string(color, f"topography.colormap_colors[{index}]")
    for key in ("symmetric_voltage_limits", "center_difference_at_zero", "common_condition_scale"):
        if not isinstance(style["topography"][key], bool):
            raise FigureStyleConfigError(f"topography.{key} must be a boolean.")
    _positive_number(style["topography"]["contour_count"], "topography.contour_count", integer=True)
    _non_empty_string(style["topography"]["contour_color"], "topography.contour_color")
    _positive_number(style["topography"]["contour_line_width"], "topography.contour_line_width")
    _non_empty_string(style["topography"]["sensor_color"], "topography.sensor_color")
    _positive_number(style["topography"]["sensor_size"], "topography.sensor_size")
    _positive_number(style["topography"]["colorbar_unit_size"], "topography.colorbar_unit_size")
    for key in ("contour_alpha", "sensor_alpha"):
        value = style["topography"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise FigureStyleConfigError(f"topography.{key} must be between 0 and 1.")
    for key, value in style["figure_sizes"].items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                _positive_number(nested_value, f"figure_sizes.{key}.{nested_key}")
        elif not isinstance(value, (list, tuple)) or len(value) != 2:
            raise FigureStyleConfigError(f"figure_sizes.{key} must contain [width, height].")
        else:
            _positive_number(value[0], f"figure_sizes.{key}[0]")
            _positive_number(value[1], f"figure_sizes.{key}[1]")
    shading = style["primary_window_shading"]
    if not isinstance(shading["enabled"], bool):
        raise FigureStyleConfigError("primary_window_shading.enabled must be a boolean.")
    alpha = shading["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
        raise FigureStyleConfigError("primary_window_shading.alpha must be between 0 and 1.")


def _positive_number(value: Any, path: str, *, integer: bool = False) -> None:
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or float(value) <= 0:
        qualifier = "positive integer" if integer else "positive number"
        raise FigureStyleConfigError(f"{path} must be a {qualifier}.")


def _non_empty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FigureStyleConfigError(f"{path} must be a non-empty string.")
