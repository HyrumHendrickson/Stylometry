"""Settings and small helpers shared by the step programs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent

# Settings.
EXTENSION = ".txt"  # only files with this extension are read from input/
ENCODING = "utf-8-sig"  # utf-8, ignoring a leading byte-order mark
MIN_SHARE_OF_TEXTS = 1  # a word is used only if it appears in at least this share of the texts
NUMBER_OF_GROUPS = None  # step 5: how many authors to propose; None = choose automatically
# (the command-line option --groups N overrides this)

# Files passed between the steps, relative to the project root.
WORD_FREQUENCIES = "temp/word_frequencies.csv"
STABLE_WORDS = "temp/stable_words.csv"
DISTANCE_MATRIX = "temp/distance_matrix.csv"
REPORT_IMAGE = "results/stylometry_report.png"
WORDS_USED = "results/words_used.txt"
GROUPS_IMAGE = "results/proposed_groups.png"


def fail(message: str) -> None:
    """Stop with a readable error message instead of a traceback."""
    raise SystemExit(f"error: {message}")


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV written by an earlier step.

    NA conversion is off because "null", "nan" and "na" are valid words.
    """
    if not path.is_file():
        fail(f"{path.name} not found. Run the earlier steps first.")
    return pd.read_csv(path, keep_default_na=False, na_filter=False, **kwargs)


def write_table(table: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, **kwargs)


def parse_command_line(description: str, groups: bool = False) -> argparse.Namespace:
    """Read the command line. --root is always accepted; --groups only where it is used."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=ROOT, help="project folder (default: here)")
    if groups:
        parser.add_argument(
            "--groups",
            type=int,
            help="number of groups (proposed authors) to sort the texts into; "
            "default: NUMBER_OF_GROUPS in programs/common.py",
        )
    args = parser.parse_args()
    args.root = args.root.resolve()
    return args


def root_from_command_line(description: str) -> Path:
    """The project folder from the command line (the option every step accepts)."""
    return parse_command_line(description).root


def draw_matrix(distances, labels, title, path):
    """Save a picture of a distance matrix. Light = similar texts, dark = different texts."""
    count = len(labels)
    size = max(8, 0.2 * count) + 4  # inches; about 0.2 inch per text
    fig = Figure(figsize=(size, size), layout="constrained")
    ax = fig.add_subplot()
    smallest = distances[~np.eye(count, dtype=bool)].min()  # ignore the zero diagonal
    image = ax.imshow(distances, cmap="Blues", vmin=smallest)
    ax.set_xticks(range(count), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(count), labels, fontsize=8)
    fig.colorbar(image, ax=ax, label="distance", shrink=0.8)
    ax.set_title(title, loc="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    print(f"Saved {path}")
