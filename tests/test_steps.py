"""Unit tests for the small pieces of each step."""

import numpy as np
import pandas as pd
import pytest

from programs.step1_word_frequencies import tokenize
from programs.step2_stable_words import rank_words
from programs.step3_shared_words import select_shared_words
from programs.step4_distance_matrix import word_power
from programs.step5_report import author_matrix


def test_tokenize_lowercases_and_normalizes_apostrophes():
    assert tokenize("Don’t STOP, it's 3 o'clock!") == ["don't", "stop", "it's", "o'clock"]


def test_rank_words_orders_by_consistency_and_drops_missing_words():
    rows = pd.DataFrame(
        {
            "text_id": ["t1", "t1", "t1", "t2", "t2", "t2"],
            "word": ["steady", "erratic", "rare", "steady", "erratic", "other"],
            "frequency": [0.10, 0.02, 0.01, 0.10, 0.20, 0.01],
        }
    )
    ranking = rank_words(rows)
    assert ranking["word"].tolist() == ["steady", "erratic"]  # "rare" and "other" miss a text
    assert ranking["rank"].tolist() == [1, 2]


def test_select_shared_words_uses_worst_rank_and_keeps_ties():
    rankings = pd.DataFrame(
        {
            "author": ["a"] * 3 + ["b"] * 3,
            "word": ["x", "y", "z", "y", "x", "z"],
            "rank": [1, 2, 3, 1, 2, 3],
        }
    )
    shared = select_shared_words(rankings, target=1)
    assert shared["word"].tolist() == ["x", "y"]  # both enter at rank 2, so both are kept


def test_select_shared_words_fails_when_too_few():
    rankings = pd.DataFrame({"author": ["a", "b"], "word": ["x", "y"], "rank": [1, 1]})
    with pytest.raises(SystemExit):
        select_shared_words(rankings, target=1)


def test_word_power_finds_the_word_that_separates_authors():
    rng = np.random.default_rng(0)
    authors = np.repeat(["a", "b"], 10)
    separating = np.r_[rng.normal(0, 0.1, 10), rng.normal(3, 0.1, 10)]
    noise = rng.normal(0, 1, 20)
    f_ratio, weight, _ = word_power(np.column_stack([separating, noise]), authors)
    assert f_ratio[0] > 100
    assert f_ratio[1] < 5
    assert weight.sum() == pytest.approx(1)
    assert weight[0] > 0.9


def test_author_matrix_ignores_self_distances():
    distances = np.array([[0, 1, 5, 5], [1, 0, 5, 5], [5, 5, 0, 2], [5, 5, 2, 0]], dtype=float)
    mean, _ = author_matrix(distances, np.array(["a", "a", "b", "b"]), ["a", "b"])
    assert mean.tolist() == [[1, 5], [5, 2]]
