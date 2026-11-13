"""Step 1: relative word frequencies for every text."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

import pandas as pd

from stylometry.corpus import TextRecord, read_text
from stylometry.errors import StylometryError
from stylometry.tokenizer import tokenize

LOGGER = logging.getLogger(__name__)

COUNT_COLUMNS = ["text_id", "author", "word", "count", "frequency"]
TOTAL_COLUMNS = ["text_id", "author", "total_tokens"]


def build_frequency_tables(
    records: Sequence[TextRecord], encoding: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count every word in every text.

    The relative frequency of word ``w`` in a text is ``count(w) / total_tokens``. Texts
    are processed independently of each other.

    Returns:
        A pair ``(counts, totals)``. ``counts`` is a sparse long table with columns
        ``text_id, author, word, count, frequency`` (absent words are implicitly zero).
        ``totals`` has one row per text with its token count.

    Raises:
        StylometryError: If a text contains no word tokens.
    """
    text_ids: list[str] = []
    authors: list[str] = []
    words: list[str] = []
    counts: list[int] = []
    frequencies: list[float] = []
    totals: list[tuple[str, str, int]] = []

    for record in records:
        tokens = tokenize(read_text(record.path, encoding))
        if not tokens:
            raise StylometryError(f"{record.path} contains no word tokens.")
        counter = Counter(tokens)
        total = len(tokens)
        totals.append((record.text_id, record.author, total))
        for word, count in counter.items():
            text_ids.append(record.text_id)
            authors.append(record.author)
            words.append(word)
            counts.append(count)
            frequencies.append(count / total)
        LOGGER.debug("%s: %d tokens, %d distinct words", record.text_id, total, len(counter))

    count_table = pd.DataFrame(
        {
            "text_id": text_ids,
            "author": authors,
            "word": words,
            "count": counts,
            "frequency": frequencies,
        },
        columns=COUNT_COLUMNS,
    )
    total_table = pd.DataFrame(totals, columns=TOTAL_COLUMNS)
    return count_table, total_table
