"""Step 3: the distance between every pair of texts.

Each text becomes a vector of the square roots of its frequencies for the chosen words (the
square root evens out the variance of common and rare words). Each word is then standardized
across all texts (subtract its mean, divide by its standard deviation) so every word counts
on the same scale. The distance between two texts is one minus the cosine of the angle between
their vectors (0 = same direction, 1 = unrelated, 2 = opposite), so it compares the pattern of
a text's high and low words and not how extreme they are.

Reads temp/word_frequencies.csv and temp/stable_words.csv and writes temp/distance_matrix.csv
(text by text, in file-name order).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from programs.common import (
    DISTANCE_MATRIX,
    ROOT,
    STABLE_WORDS,
    WORD_FREQUENCIES,
    read_table,
    root_from_command_line,
    write_table,
)


def distance_matrix(frequencies: pd.DataFrame, words: list[str]) -> pd.DataFrame:
    text_ids = sorted(frequencies["text_id"].unique())
    wide = (
        frequencies[frequencies["word"].isin(words)]
        .pivot(index="text_id", columns="word", values="frequency")
        .reindex(index=text_ids, columns=words)
        .fillna(0.0)  # a word missing from a text has frequency 0
    )
    x = np.sqrt(wide.to_numpy())
    spread = x.std(axis=0, ddof=1)
    z = (x - x.mean(axis=0)) / np.where(spread > 0, spread, 1)

    lengths = np.linalg.norm(z, axis=1, keepdims=True)
    unit = z / np.where(lengths > 0, lengths, 1)
    distances = np.maximum(1 - unit @ unit.T, 0)  # cosine distance
    np.fill_diagonal(distances, 0)
    return pd.DataFrame(distances, index=text_ids, columns=text_ids).rename_axis("text_id")


def run(root: Path = ROOT) -> None:
    frequencies = read_table(root / WORD_FREQUENCIES)
    words = read_table(root / STABLE_WORDS)["word"].tolist()
    matrix = distance_matrix(frequencies, words)
    write_table(matrix, root / DISTANCE_MATRIX)
    print(f"{len(matrix)} x {len(matrix)} distance matrix over {len(words)} words")


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
