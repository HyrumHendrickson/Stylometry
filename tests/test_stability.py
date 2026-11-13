"""Tests for steps 2 and 3."""

from __future__ import annotations

import pandas as pd
import pytest

from stylometry.errors import StylometryError
from stylometry.stability import (
    rank_stable_words,
    rank_stable_words_by_author,
    select_shared_words,
)


def test_rank_orders_by_cv_and_drops_absent_words() -> None:
    wide = pd.DataFrame(
        {
            "steady": [0.10, 0.10, 0.10],
            "mild": [0.05, 0.06, 0.07],
            "erratic": [0.01, 0.10, 0.02],
            "missing": [0.0, 0.10, 0.10],
        },
        index=["t1", "t2", "t3"],
    )
    ranking = rank_stable_words(wide)
    assert ranking["word"].tolist() == ["steady", "mild", "erratic"]
    assert ranking["rank"].tolist() == [1, 2, 3]
    assert ranking.loc[0, "cv"] == pytest.approx(0.0)
    expected_cv = wide["mild"].std(ddof=1) / wide["mild"].mean()
    assert ranking.loc[1, "cv"] == pytest.approx(expected_cv)


def test_rank_breaks_ties_by_higher_mean_then_alphabetically() -> None:
    # Binary-exact values give identical CVs for x and y (y = x / 2).
    wide = pd.DataFrame(
        {"x": [0.25, 0.75], "y": [0.125, 0.375], "w": [0.125, 0.375]}, index=["t1", "t2"]
    )
    assert rank_stable_words(wide)["word"].tolist() == ["x", "w", "y"]


def test_rank_requires_two_texts() -> None:
    with pytest.raises(StylometryError):
        rank_stable_words(pd.DataFrame({"a": [0.1]}, index=["t1"]))


def _long(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["text_id", "author", "word", "frequency"])


def test_rank_by_author_is_independent_per_author() -> None:
    table = _long(
        [
            ("A/1", "A", "the", 0.10),
            ("A/2", "A", "the", 0.10),
            ("A/1", "A", "dragon", 0.02),  # only in one of A's texts
            ("A/1", "A", "of", 0.05),
            ("A/2", "A", "of", 0.07),
            ("B/1", "B", "the", 0.08),
            ("B/2", "B", "the", 0.12),
            ("B/1", "B", "of", 0.05),
            ("B/2", "B", "of", 0.051),
        ]
    )
    ranking = rank_stable_words_by_author(table)
    by_author = {a: g["word"].tolist() for a, g in ranking.groupby("author")}
    assert by_author == {"A": ["the", "of"], "B": ["of", "the"]}
    assert (ranking["rank"] >= 1).all()


def _rankings() -> pd.DataFrame:
    order_a = ["w1", "w2", "w3", "w4", "only_a"]
    order_b = ["w2", "w1", "w4", "w3"]
    rows = [("A", r, w) for r, w in enumerate(order_a, start=1)]
    rows += [("B", r, w) for r, w in enumerate(order_b, start=1)]
    return pd.DataFrame(rows, columns=["author", "rank", "word"])


def test_select_shared_words_picks_smallest_k() -> None:
    selection = select_shared_words(_rankings(), target=2)
    assert selection.k == 2
    assert selection.words["word"].tolist() == ["w1", "w2"]
    assert selection.words["entry_rank"].tolist() == [2, 2]


def test_select_shared_words_includes_everything_that_enters_at_k() -> None:
    selection = select_shared_words(_rankings(), target=3)
    assert selection.k == 4
    assert selection.words["word"].tolist() == ["w1", "w2", "w3", "w4"]


def test_select_shared_words_excludes_words_missing_for_an_author() -> None:
    selection = select_shared_words(_rankings(), target=4)
    assert "only_a" not in selection.words["word"].tolist()


def test_select_shared_words_fails_when_target_unreachable() -> None:
    with pytest.raises(StylometryError, match="Only 4"):
        select_shared_words(_rankings(), target=5)
