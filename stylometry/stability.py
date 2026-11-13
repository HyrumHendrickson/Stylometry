"""Steps 2 and 3: stable words per author and the words stable for every author."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from stylometry.errors import StylometryError

LOGGER = logging.getLogger(__name__)

RANKING_COLUMNS = ["author", "rank", "word", "mean_frequency", "std_frequency", "cv"]


def rank_stable_words(wide: pd.DataFrame) -> pd.DataFrame:
    """Rank one author's words from most to least consistent.

    A word is *eligible* only if it occurs in every one of the author's texts; a word that
    vanishes from some text is not used at a consistent rate. Eligible words are ranked by
    the coefficient of variation ``cv = s / mean`` of their relative frequency across the
    author's texts (``s`` is the sample standard deviation), ascending. Ties are broken by
    higher mean frequency, then alphabetically, so the ranking is deterministic.

    Args:
        wide: Relative frequencies, one row per text and one column per word. Absent words
            must be filled with zero.

    Returns:
        Columns ``rank`` (1 = most stable), ``word``, ``mean_frequency``,
        ``std_frequency`` and ``cv``.
    """
    if len(wide) < 2:
        raise StylometryError("At least two texts are required to rank word stability.")
    eligible = wide.loc[:, (wide > 0).all(axis=0)]
    summary = pd.DataFrame(
        {
            "word": eligible.columns,
            "mean_frequency": eligible.mean(axis=0).to_numpy(),
            "std_frequency": eligible.std(axis=0, ddof=1).to_numpy(),
        }
    )
    summary["cv"] = summary["std_frequency"] / summary["mean_frequency"]
    summary = summary.sort_values(
        ["cv", "mean_frequency", "word"], ascending=[True, False, True], kind="stable"
    ).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def rank_stable_words_by_author(frequencies: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`rank_stable_words` independently to each author.

    Args:
        frequencies: The long table produced by step 1 (``text_id, author, word,
            frequency`` columns).

    Returns:
        One concatenated table with columns :data:`RANKING_COLUMNS`.
    """
    parts: list[pd.DataFrame] = []
    for author, author_rows in frequencies.groupby("author", sort=True):
        n_texts = author_rows["text_id"].nunique()
        texts_per_word = author_rows.groupby("word")["text_id"].nunique()
        eligible_words = texts_per_word.index[texts_per_word == n_texts]
        wide = (
            author_rows[author_rows["word"].isin(eligible_words)]
            .pivot(index="text_id", columns="word", values="frequency")
            .fillna(0.0)
        )
        ranking = rank_stable_words(wide)
        ranking.insert(0, "author", author)
        LOGGER.info("%s: %d texts, %d words present in every text", author, n_texts, len(ranking))
        parts.append(ranking)
    return pd.concat(parts, ignore_index=True)[RANKING_COLUMNS]


@dataclass(frozen=True)
class SharedWordSelection:
    """Result of the cross-author comparison.

    Attributes:
        words: One row per shared word with its ``entry_rank`` (the smallest list length K
            at which the word appears in every author's top-K) and each author's own rank.
        k: The list length K that was selected.
        target: The requested minimum number of shared words.
    """

    words: pd.DataFrame
    k: int
    target: int


def select_shared_words(rankings: pd.DataFrame, target: int) -> SharedWordSelection:
    """Find the words that are stable for every author.

    For a list length ``K`` let ``S_a(K)`` be author ``a``'s ``K`` most stable words. The
    shared set is the intersection of all ``S_a(K)``. A word with per-author ranks
    ``r_a`` belongs to that intersection exactly when ``K >= max_a r_a``, so each word has
    an ``entry_rank = max_a r_a`` and the intersection grows monotonically with ``K``.
    This function returns the smallest ``K`` whose intersection holds at least ``target``
    words, together with that intersection.

    Raises:
        StylometryError: If fewer than ``target`` words are eligible for every author.
    """
    ranks = rankings.pivot(index="word", columns="author", values="rank").dropna()
    if len(ranks) < target:
        raise StylometryError(
            f"Only {len(ranks)} word(s) occur in every text of every author, fewer than the "
            f"requested {target}. Lower stability.target_shared_words in config/config.toml "
            "or add longer/more texts."
        )
    ranks = ranks.astype(int)
    ranks.insert(0, "entry_rank", ranks.max(axis=1))
    ranks = ranks.sort_values(["entry_rank", "word"], kind="stable")
    k = int(ranks["entry_rank"].iloc[target - 1])
    shared = ranks[ranks["entry_rank"] <= k].reset_index()
    return SharedWordSelection(words=shared, k=k, target=target)
