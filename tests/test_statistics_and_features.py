"""Tests for step 5 statistics, the feature table and CSV helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stylometry.distance_statistics import (
    accuracy_by_author,
    author_distance_groups,
    group_statistics,
    statistic_matrix,
)
from stylometry.errors import StylometryError
from stylometry.features import FeatureMatrix
from stylometry.tables import read_table, write_table


def _line_distances() -> tuple[np.ndarray, np.ndarray]:
    positions = np.array([0.0, 1.0, 3.0, 10.0, 12.0])
    authors = np.array(["A", "A", "A", "B", "B"])
    return np.abs(positions[:, None] - positions[None, :]), authors


def test_group_statistics_counts_and_values() -> None:
    distances, authors = _line_distances()
    stats = group_statistics(distances, authors).set_index(["author_a", "author_b"])

    within_a = stats.loc[("A", "A")]
    assert within_a["group"] == "within"
    assert within_a["n_distances"] == 3  # n(n-1)/2 for 3 texts
    assert within_a["mean"] == pytest.approx(2.0)  # distances 1, 3, 2
    assert within_a["std"] == pytest.approx(1.0)

    within_b = stats.loc[("B", "B")]
    assert within_b["n_distances"] == 1
    assert np.isnan(within_b["std"])

    between = stats.loc[("A", "B")]
    assert between["group"] == "between"
    assert between["n_distances"] == 6  # 3 x 2
    values = np.array([10, 12, 9, 11, 7, 9], dtype=float)
    assert between["mean"] == pytest.approx(values.mean())
    assert between["std"] == pytest.approx(values.std(ddof=1))


def test_statistic_matrix_is_symmetric_with_within_on_diagonal() -> None:
    distances, authors = _line_distances()
    stats = group_statistics(distances, authors)
    means = statistic_matrix(stats, ["A", "B"], "mean")
    assert means[0, 0] == pytest.approx(2.0)
    assert means[0, 1] == means[1, 0]


def test_author_distance_groups() -> None:
    distances, authors = _line_distances()
    within, between = author_distance_groups(distances, authors)["B"]
    assert within.tolist() == [2.0]
    assert sorted(between.tolist()) == [7.0, 9.0, 9.0, 10.0, 11.0, 12.0]


def test_accuracy_by_author() -> None:
    true = np.array(["A", "A", "B", "B"])
    predicted = np.array(["A", "B", "B", "B"])
    per_author, overall = accuracy_by_author(true, predicted)
    assert per_author == {"A": 0.5, "B": 1.0}
    assert overall == pytest.approx(0.75)


def test_feature_matrix_round_trip_with_colliding_word_names() -> None:
    frequencies = pd.DataFrame(
        {
            "text_id": ["A/1", "A/1", "B/1", "B/1"],
            "author": ["A", "A", "B", "B"],
            "word": ["author", "text_id", "author", "text_id"],
            "frequency": [0.1, 0.2, 0.3, 0.4],
        }
    )
    totals = pd.DataFrame(
        {"text_id": ["B/1", "A/1"], "author": ["B", "A"], "total_tokens": [10, 10]}
    )
    features = FeatureMatrix.build(frequencies, totals, ["text_id", "author"])
    assert features.text_ids == ["A/1", "B/1"]
    assert features.values.tolist() == [[0.2, 0.1], [0.4, 0.3]]
    restored = FeatureMatrix.from_frame(features.to_frame())
    assert restored.words == ["text_id", "author"]
    assert restored.values.tolist() == features.values.tolist()
    assert restored.authors.tolist() == ["A", "B"]


def test_feature_matrix_requires_word_columns() -> None:
    with pytest.raises(StylometryError):
        FeatureMatrix.from_frame(pd.DataFrame({"text_id": ["a"], "author": ["A"]}))


def test_tables_keep_words_that_look_like_na(tmp_path: Path) -> None:
    frame = pd.DataFrame({"word": ["null", "nan", "na", "none"], "count": [1, 2, 3, 4]})
    path = tmp_path / "sub" / "words.csv"
    write_table(frame, path)
    restored = read_table(path, "step1")
    assert restored["word"].tolist() == ["null", "nan", "na", "none"]


def test_read_table_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StylometryError, match="Run step9 first"):
        read_table(tmp_path / "absent.csv", "step9")
