"""Step 4: how well each shared word tells authors apart, and the distance between texts.

Each text becomes a vector of the square roots of its shared-word frequencies (the square
root evens out the variance of common and rare words).

Power of a word = F-ratio from a one-way ANOVA: variance between author means divided by
variance within authors. F near 1 means the word does not separate authors. Each word is
weighted by max(F - 1, 0), scaled so the weights sum to 1.

Distance between two texts u and v:  sqrt( sum_j weight_j * (u_j - v_j)^2 / within_j )

Reads temp/word_frequencies.csv and temp/shared_stable_words.csv. Writes
temp/word_power.csv (word, f_ratio, weight) and temp/distance_matrix.csv (text by text).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from programs.common import (
    DISTANCE_MATRIX,
    ROOT,
    SHARED_WORDS,
    WORD_FREQUENCIES,
    WORD_POWER,
    read_table,
    root_from_command_line,
    write_table,
)


def word_power(x: np.ndarray, authors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-way ANOVA per word. x has one row per text and one column per word.

    Returns (f_ratio, weight, within_variance), each with one value per word.
    """
    names = np.unique(authors)
    n_texts, n_authors = len(x), len(names)
    grand_mean = x.mean(axis=0)
    between = np.zeros(x.shape[1])
    within = np.zeros(x.shape[1])
    for name in names:
        rows = x[authors == name]
        mean = rows.mean(axis=0)
        between += len(rows) * (mean - grand_mean) ** 2
        within += ((rows - mean) ** 2).sum(axis=0)

    between_variance = between / (n_authors - 1)
    within_variance = np.maximum(within / (n_texts - n_authors), 1e-12)  # avoid dividing by 0
    f_ratio = between_variance / within_variance

    excess = np.maximum(f_ratio - 1, 0)
    weight = excess / excess.sum() if excess.any() else np.full(len(excess), 1 / len(excess))
    return f_ratio, weight, within_variance


def run(root: Path = ROOT) -> None:
    frequencies = read_table(root / WORD_FREQUENCIES)
    words = read_table(root / SHARED_WORDS)["word"].tolist()

    # Texts ordered by author, then name, so each author's texts sit together.
    texts = frequencies[["author", "text_id"]].drop_duplicates().sort_values(["author", "text_id"])
    wide = (
        frequencies[frequencies["word"].isin(words)]
        .pivot(index="text_id", columns="word", values="frequency")
        .reindex(index=texts["text_id"], columns=words)
        .fillna(0.0)  # a word missing from a text has frequency 0
    )
    x = np.sqrt(wide.to_numpy())

    f_ratio, weight, within_variance = word_power(x, texts["author"].to_numpy())
    power = pd.DataFrame({"word": words, "f_ratio": f_ratio, "weight": weight})
    write_table(power.sort_values("f_ratio", ascending=False), root / WORD_POWER, index=False)

    # Weighted Euclidean distance, computed as an ordinary distance after rescaling each word.
    scaled = x * np.sqrt(weight / within_variance)
    squared_norms = (scaled**2).sum(axis=1)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2 * scaled @ scaled.T
    distances = np.sqrt(np.maximum(squared, 0))
    np.fill_diagonal(distances, 0)
    matrix = pd.DataFrame(distances, index=texts["text_id"], columns=texts["text_id"])
    write_table(matrix, root / DISTANCE_MATRIX)

    best = power.sort_values("f_ratio", ascending=False)["word"].head(5)
    print(f"{len(matrix)} x {len(matrix)} distance matrix. Best words: {', '.join(best)}")


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
