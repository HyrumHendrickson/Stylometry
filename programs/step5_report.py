"""Step 5: draw the report image.

The image has three panels:
  1. Author matrix: mean distance (and standard deviation) between the texts of each pair of
     authors. The diagonal is the distance between texts by the same author.
  2. Every shared word with its F-ratio, i.e. its power to tell authors apart.
  3. Text matrix: the distance between every pair of texts.

Reads temp/distance_matrix.csv and temp/word_power.csv. Writes results/stylometry_report.png.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from programs.common import (
    DISTANCE_MATRIX,
    REPORT_IMAGE,
    ROOT,
    WORD_POWER,
    read_table,
    root_from_command_line,
)

COLORS = "Blues"  # light = similar texts, dark = different texts
BAR_COLOR = "#2a78d6"


def author_matrix(distances: np.ndarray, authors: np.ndarray, names: list[str]):
    """Mean and standard deviation of the distances for every pair of authors."""
    size = len(names)
    mean, std = np.full((size, size), np.nan), np.full((size, size), np.nan)
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            block = distances[np.ix_(authors == first, authors == second)]
            # Within one author, use each pair of texts once and skip a text paired with itself.
            values = block[np.triu_indices(len(block), k=1)] if i == j else block.ravel()
            mean[i, j] = values.mean()
            if len(values) > 1:
                std[i, j] = values.std(ddof=1)
    return mean, std


def draw_author_matrix(fig: Figure, ax, mean: np.ndarray, std: np.ndarray, names: list[str]):
    image = ax.imshow(mean, cmap=COLORS, vmin=0)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            label = f"{mean[i, j]:.2f}"
            if not np.isnan(std[i, j]):
                label += f"\n±{std[i, j]:.2f}"
            dark = mean[i, j] > 0.6 * np.nanmax(mean)
            ax.text(j, i, label, ha="center", va="center", color="white" if dark else "black")
    fig.colorbar(image, ax=ax, label="mean distance")
    ax.set_title("Authors: mean distance ± std (diagonal = same author)", loc="left")


def draw_word_power(ax, power: pd.DataFrame):
    power = power.iloc[::-1]  # most powerful word at the top
    y = np.arange(len(power))
    f = power["f_ratio"].to_numpy()
    ax.barh(y, f - 1, left=1, color=BAR_COLOR)  # bars start at F = 1, "no power"
    for position, value in zip(y, f):
        ax.text(max(value, 1), position, f" {value:.1f}", va="center", fontsize=7)
    ax.axvline(1, color="gray", linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y, power["word"], fontsize=8)
    ax.set_ylim(-0.7, len(power) - 0.3)
    ax.set_xlabel("F-ratio = variance between authors / variance within authors (log scale)")
    ax.set_title(f"Words: power to tell authors apart ({len(power)} words)", loc="left")


def draw_text_matrix(fig: Figure, ax, distances: np.ndarray, text_ids: list[str], authors):
    image = ax.imshow(distances, cmap=COLORS, vmin=0)
    font_size = min(8, 160 / len(text_ids))
    ax.set_xticks(range(len(text_ids)), text_ids, rotation=90, fontsize=font_size)
    ax.set_yticks(range(len(text_ids)), text_ids, fontsize=font_size)
    for edge in np.flatnonzero(authors[1:] != authors[:-1]) + 0.5:  # lines between authors
        ax.axhline(edge, color="black", linewidth=1)
        ax.axvline(edge, color="black", linewidth=1)
    fig.colorbar(image, ax=ax, label="distance", shrink=0.8)
    ax.set_title("Texts: distance between every pair of texts", loc="left")


def run(root: Path = ROOT) -> None:
    matrix = read_table(root / DISTANCE_MATRIX, index_col=0)
    power = read_table(root / WORD_POWER).sort_values("f_ratio", ascending=False)

    distances = matrix.to_numpy(dtype=float)
    text_ids = matrix.index.astype(str).tolist()
    authors = np.array([text_id.split("/")[0] for text_id in text_ids])  # text_id = author/file
    names = list(dict.fromkeys(authors))  # authors in order of appearance
    mean, std = author_matrix(distances, authors, names)

    fig = Figure(figsize=(22, max(12, 0.45 * len(power))), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])
    draw_author_matrix(fig, fig.add_subplot(grid[0, 0]), mean, std, names)
    draw_word_power(fig.add_subplot(grid[1, 0]), power)
    draw_text_matrix(fig, fig.add_subplot(grid[:, 1]), distances, text_ids, authors)
    fig.suptitle(f"{len(text_ids)} texts, {len(names)} authors, {len(power)} shared stable words")

    image = root / REPORT_IMAGE
    image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image, dpi=110)
    print(f"Saved {image}")


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
