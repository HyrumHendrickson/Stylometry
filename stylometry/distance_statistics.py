"""Step 5 statistics: spread of within-author and between-author distances."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stylometry.distance import FloatArray, StrArray

STATISTICS_COLUMNS = ["group", "author_a", "author_b", "n_distances", "mean", "std"]


def _summarize(values: FloatArray) -> tuple[int, float, float]:
    n = int(values.size)
    mean = float(values.mean()) if n else float("nan")
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    return n, mean, std


def group_statistics(distances: FloatArray, authors: StrArray) -> pd.DataFrame:
    """Summarize the distance distribution of every author and every author pair.

    * ``within`` rows (``author_a == author_b``): the distances between all unordered pairs
      of distinct texts by that author (``n(n-1)/2`` values).
    * ``between`` rows: the distances from every text of ``author_a`` to every text of
      ``author_b`` (``n_a * n_b`` values), listed once per unordered author pair.

    The reported ``std`` is the sample standard deviation (``ddof=1``); it is ``NaN`` for a
    group with fewer than two distances.
    """
    authors = np.asarray(authors)
    names = sorted(set(authors.tolist()))
    index = {name: np.flatnonzero(authors == name) for name in names}
    rows: list[dict[str, object]] = []
    for i, first in enumerate(names):
        for second in names[i:]:
            block = distances[np.ix_(index[first], index[second])]
            if first == second:
                values = block[np.triu_indices(block.shape[0], k=1)]
                group = "within"
            else:
                values = block.ravel()
                group = "between"
            n, mean, std = _summarize(values)
            rows.append(
                {
                    "group": group,
                    "author_a": first,
                    "author_b": second,
                    "n_distances": n,
                    "mean": mean,
                    "std": std,
                }
            )
    return pd.DataFrame(rows, columns=STATISTICS_COLUMNS)


def statistic_matrix(stats: pd.DataFrame, names: list[str], column: str) -> FloatArray:
    """Arrange one statistic as a symmetric authors-by-authors matrix.

    The diagonal holds the within-author value.
    """
    position = {name: i for i, name in enumerate(names)}
    matrix = np.full((len(names), len(names)), np.nan)
    for row in stats.itertuples(index=False):
        i, j = position[row.author_a], position[row.author_b]  # type: ignore[index]
        value = float(getattr(row, column))
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def author_distance_groups(
    distances: FloatArray, authors: StrArray
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """For each author, the within-author distances and the distances to all other authors."""
    authors = np.asarray(authors)
    groups: dict[str, tuple[FloatArray, FloatArray]] = {}
    for name in sorted(set(authors.tolist())):
        own = np.flatnonzero(authors == name)
        other = np.flatnonzero(authors != name)
        within = distances[np.ix_(own, own)][np.triu_indices(len(own), k=1)]
        between = distances[np.ix_(own, other)].ravel()
        groups[name] = (within, between)
    return groups


def accuracy_by_author(true: StrArray, predicted: StrArray) -> tuple[dict[str, float], float]:
    """Per-author and overall fraction of correctly attributed texts."""
    true = np.asarray(true)
    predicted = np.asarray(predicted)
    per_author = {
        name: float(np.mean(predicted[true == name] == name)) for name in sorted(set(true.tolist()))
    }
    return per_author, float(np.mean(true == predicted))
