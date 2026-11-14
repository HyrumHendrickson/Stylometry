"""Unit tests for the small pieces of each step."""

import numpy as np
import pandas as pd
import pytest

from programs.step1_word_frequencies import tokenize
from programs.step2_stable_words import stable_words
from programs.step3_distance_matrix import distance_matrix
from programs.step5_groups import propose_groups, silhouette


def test_tokenize_lowercases_and_normalizes_apostrophes():
    assert tokenize("Don’t STOP, it's 3 o'clock!") == ["don't", "stop", "it's", "o'clock"]


def test_stable_words_keeps_words_found_in_enough_texts():
    texts = ["t1", "t2", "t3", "t4", "t5"]
    rows = [(t, "the", 0.1) for t in texts] + [("t1", "of", 0.1), ("t2", "of", 0.1)]
    rows += [("t1", "name", 0.1)]
    frequencies = pd.DataFrame(rows, columns=["text_id", "word", "frequency"])
    # "the" is in 5/5 texts, "of" in 2/5 (40%, kept at the 25% default), "name" in 1/5 (20%).
    assert stable_words(frequencies)["word"].tolist() == ["the", "of"]


def test_stable_words_fails_when_no_word_qualifies():
    rows = [(f"t{i}", word, 0.1) for i, word in enumerate("abcde")]  # each word in 1 of 5 texts
    frequencies = pd.DataFrame(rows, columns=["text_id", "word", "frequency"])
    with pytest.raises(SystemExit):
        stable_words(frequencies)


def test_distance_matrix_is_cosine_distance_in_range():
    rng = np.random.default_rng(0)
    rows = [(t, w, rng.uniform(0.01, 0.1)) for t in ("b", "a", "c") for w in ("x", "y", "z")]
    frequencies = pd.DataFrame(rows, columns=["text_id", "word", "frequency"])
    matrix = distance_matrix(frequencies, ["x", "y", "z"])
    assert matrix.index.tolist() == ["a", "b", "c"]
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
    assert (matrix.to_numpy().diagonal() == 0).all()
    assert matrix.to_numpy().min() >= 0 and matrix.to_numpy().max() <= 2


def blocks(*sizes):
    """Distances for blocks of texts: 1 inside a block, growing with the gap between blocks."""
    block = np.repeat(np.arange(len(sizes)), sizes)
    distances = 1.0 + 4.0 * np.abs(block[:, None] - block[None, :])
    distances[block[:, None] == block[None, :]] = 1.0
    np.fill_diagonal(distances, 0)
    return distances


def two_blocks():
    return blocks(3, 3)


def test_propose_groups_finds_the_blocks():
    groups = propose_groups(two_blocks())
    assert groups[:3].tolist() == [groups[0]] * 3
    assert groups[3:].tolist() == [groups[3]] * 3
    assert sorted(set(groups)) == [1, 2]


def test_propose_groups_obeys_a_requested_number():
    groups = propose_groups(blocks(2, 2, 2), number=3)
    assert len(set(groups)) == 3
    groups = propose_groups(blocks(2, 2, 2), number=2)
    assert len(set(groups)) == 2


def test_silhouette_is_high_for_separated_groups_and_low_for_mixed_ones():
    distances = two_blocks()
    assert silhouette(distances, np.array([1, 1, 1, 2, 2, 2])) > 0.5
    assert silhouette(distances, np.array([1, 2, 1, 2, 1, 2])) < 0
