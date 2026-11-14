"""Settings and small helpers shared by the step programs."""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Settings.
EXTENSION = ".txt"  # only files with this extension are read from each author folder
ENCODING = "utf-8-sig"  # utf-8, ignoring a leading byte-order mark
TARGET_SHARED_WORDS = 30  # step 3: minimum number of words stable for every author

# Files passed between the steps, relative to the project root.
WORD_FREQUENCIES = "temp/word_frequencies.csv"
STABLE_WORDS = "temp/stable_words_by_author.csv"
SHARED_WORDS = "temp/shared_stable_words.csv"
WORD_POWER = "temp/word_power.csv"
DISTANCE_MATRIX = "temp/distance_matrix.csv"
REPORT_IMAGE = "results/stylometry_report.png"


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


def root_from_command_line(description: str) -> Path:
    """Read the optional --root argument (the project folder to analyze)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=ROOT, help="project folder (default: here)")
    return parser.parse_args().root.resolve()
