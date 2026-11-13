"""Rendering of the step 5 statistical report as a single image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from stylometry.distance import METRIC_LABELS, FloatArray, StrArray
from stylometry.distance_statistics import author_distance_groups, statistic_matrix

# Palette: reference categorical slots 1 (blue) and 2 (orange), light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

MAX_WORDS_SHOWN = 30


@dataclass(frozen=True)
class ReportData:
    """Everything the figure needs.

    Attributes:
        distances: Full text-by-text distance matrix for the primary metric.
        authors: Author of each row of ``distances``.
        statistics: Output of :func:`stylometry.distance_statistics.group_statistics`.
        word_weights: Table with ``word`` and ``f_ratio`` columns.
        metric: Primary metric name.
        baseline_metric: Baseline metric name.
        accuracy: Leave-one-out accuracy per author for the primary metric.
        baseline_accuracy: Same for the baseline metric.
        overall_accuracy: Overall leave-one-out accuracy, primary metric.
        baseline_overall_accuracy: Overall leave-one-out accuracy, baseline metric.
    """

    distances: FloatArray
    authors: StrArray
    statistics: pd.DataFrame
    word_weights: pd.DataFrame
    metric: str
    baseline_metric: str
    accuracy: dict[str, float]
    baseline_accuracy: dict[str, float]
    overall_accuracy: float
    baseline_overall_accuracy: float


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG relative luminance of an sRGB color with channels in [0, 1]."""
    red, green, blue = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _prefers_white_text(background: tuple[float, float, float]) -> bool:
    """Whether white text has more contrast than the ink color on ``background``."""
    lum = _relative_luminance(background)
    ink = _relative_luminance((0x0B / 255, 0x0B / 255, 0x0B / 255))
    return (1.05) / (lum + 0.05) > (lum + 0.05) / (ink + 0.05)


def _style_axes(ax: Axes, *, grid_axis: Literal["x", "y"] = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)


def _heatmap(fig: Figure, ax: Axes, data: ReportData, names: list[str]) -> None:
    means = statistic_matrix(data.statistics, names, "mean")
    stds = statistic_matrix(data.statistics, names, "std")
    image = ax.imshow(means, cmap=SEQUENTIAL, vmin=0.0, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    norm_max = float(np.nanmax(means)) or 1.0
    font = max(6.0, 10.5 - 0.6 * len(names))
    for i in range(len(names)):
        for j in range(len(names)):
            mean, std = means[i, j], stds[i, j]
            dark_cell = _prefers_white_text(SEQUENTIAL(mean / norm_max)[:3])
            spread = "n/a" if np.isnan(std) else f"{std:.2f}"
            ax.text(
                j,
                i,
                f"{mean:.2f}\n±{spread}",
                ha="center",
                va="center",
                fontsize=font,
                color="#ffffff" if dark_cell else INK,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    colorbar.set_label("mean distance", color=INK_SECONDARY, fontsize=9)
    _title(ax, "Mean distance ± std (diagonal: within author)")


def _distributions(ax: Axes, data: ReportData, names: list[str]) -> None:
    groups = author_distance_groups(data.distances, data.authors)
    positions = np.arange(len(names))
    offset, width = 0.2, 0.32
    for series, color, shift in ((0, BLUE, -offset), (1, ORANGE, offset)):
        box = ax.boxplot(
            [groups[name][series] for name in names],
            positions=positions + shift,
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": color,
                "markeredgecolor": SURFACE,
                "alpha": 0.8,
            },
            medianprops={"color": INK, "linewidth": 1.4},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
            boxprops={"edgecolor": color, "linewidth": 1.2},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
    ax.set_xticks(positions, names, rotation=35, ha="right")
    ax.set_ylabel("distance", color=INK_SECONDARY, fontsize=9)
    _style_axes(ax)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.6, edgecolor=BLUE, label="within author"),
            Patch(facecolor=ORANGE, alpha=0.6, edgecolor=ORANGE, label="to all other authors"),
        ],
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    _title(ax, "Distance distributions per author")


def _word_signal(ax: Axes, data: ReportData) -> None:
    table = data.word_weights.sort_values("f_ratio", ascending=False)
    shown = table.head(MAX_WORDS_SHOWN).iloc[::-1]
    y = np.arange(len(shown))
    f = shown["f_ratio"].to_numpy(dtype=float)
    ax.hlines(y, 1.0, f, color=BLUE, linewidth=1.6, alpha=0.7)
    ax.plot(f, y, "o", color=BLUE, markersize=5, markeredgecolor=SURFACE)
    ax.axvline(1.0, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, shown["word"].tolist())
    ax.set_xlabel(
        "F-ratio = between-author / within-author variance (log scale)",
        color=INK_SECONDARY,
        fontsize=9,
    )
    _style_axes(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    ax.text(
        1.0, len(shown) - 0.2, "  F = 1: no signal", color=INK_SECONDARY, fontsize=8, va="bottom"
    )
    suffix = "" if len(table) <= MAX_WORDS_SHOWN else f" (top {MAX_WORDS_SHOWN} of {len(table)})"
    _title(ax, f"Discriminative power of each shared stable word{suffix}")


def _accuracy(ax: Axes, data: ReportData, names: list[str]) -> None:
    positions = np.arange(len(names))
    width = 0.38
    primary = METRIC_LABELS[data.metric]
    baseline = METRIC_LABELS[data.baseline_metric]
    if data.metric == data.baseline_metric:
        baseline = f"{baseline} (baseline)"
    ax.bar(
        positions - width / 2,
        [data.accuracy[n] for n in names],
        width,
        color=BLUE,
        label=f"{primary} (overall {data.overall_accuracy:.0%})",
    )
    ax.bar(
        positions + width / 2,
        [data.baseline_accuracy[n] for n in names],
        width,
        color=ORANGE,
        label=f"{baseline} (overall {data.baseline_overall_accuracy:.0%})",
    )
    ax.axhline(1.0 / len(names), color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.text(
        len(names) - 0.5,
        1.0 / len(names) + 0.015,
        "chance",
        color=INK_SECONDARY,
        fontsize=8,
        ha="right",
    )
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(positions, names, rotation=35, ha="right")
    ax.set_ylabel("fraction attributed correctly", color=INK_SECONDARY, fontsize=9)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left", ncol=1)
    _title(ax, "Leave-one-out author attribution")


def render_report(data: ReportData, path: Path) -> None:
    """Draw the four-panel report and save it as a PNG.

    Panels: mean distance and standard deviation for every author pair; distance
    distributions per author; per-word discriminative power; leave-one-out attribution
    accuracy for the primary metric versus the baseline.
    """
    names = sorted(set(data.authors.tolist()))
    n_texts = len(data.authors)
    n_words = len(data.word_weights)
    fig = Figure(figsize=(15, 11.5), facecolor=SURFACE)
    grid = fig.add_gridspec(
        2, 2, hspace=0.50, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.12
    )
    _heatmap(fig, fig.add_subplot(grid[0, 0]), data, names)
    _distributions(fig.add_subplot(grid[0, 1]), data, names)
    _word_signal(fig.add_subplot(grid[1, 0]), data)
    _accuracy(fig.add_subplot(grid[1, 1]), data, names)
    fig.suptitle(
        f"Stylometric report — {n_texts} texts, {len(names)} authors, {n_words} shared "
        f"stable words, {METRIC_LABELS[data.metric]} distance",
        x=0.07,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
