"""Step 4 data structure: one frequency vector per text over the shared stable words."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stylometry.distance import FloatArray, StrArray
from stylometry.errors import StylometryError

#: Prefix of word columns in the saved CSV, so a word like "author" cannot collide with
#: the ``text_id`` and ``author`` metadata columns.
WORD_PREFIX = "freq:"


@dataclass(frozen=True)
class FeatureMatrix:
    """Frequency vectors for all texts.

    Attributes:
        text_ids: Text identifiers, ordered by author then filename.
        authors: Author of each row.
        words: Shared stable words (the columns).
        values: Relative frequencies, shape ``(len(text_ids), len(words))``.
    """

    text_ids: list[str]
    authors: StrArray
    words: list[str]
    values: FloatArray

    @classmethod
    def build(
        cls, frequencies: pd.DataFrame, totals: pd.DataFrame, shared_words: list[str]
    ) -> FeatureMatrix:
        """Assemble vectors from the step 1 table, restricted to ``shared_words``.

        Texts are ordered by ``(author, text_id)``; a word absent from a text has frequency
        zero.
        """
        ordered = totals.sort_values(["author", "text_id"], kind="stable")
        text_ids = ordered["text_id"].tolist()
        authors = ordered["author"].to_numpy(dtype=str)
        selected = frequencies[frequencies["word"].isin(shared_words)]
        wide = (
            selected.pivot(index="text_id", columns="word", values="frequency")
            .reindex(index=text_ids, columns=shared_words)
            .fillna(0.0)
        )
        return cls(text_ids, authors, list(shared_words), wide.to_numpy(dtype=np.float64))

    def to_frame(self) -> pd.DataFrame:
        """Tabular form with ``text_id``, ``author`` and one ``freq:<word>`` column per word."""
        frame = pd.DataFrame(self.values, columns=[WORD_PREFIX + w for w in self.words])
        frame.insert(0, "author", self.authors)
        frame.insert(0, "text_id", self.text_ids)
        return frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> FeatureMatrix:
        """Inverse of :meth:`to_frame`."""
        word_columns = [c for c in frame.columns if str(c).startswith(WORD_PREFIX)]
        if not word_columns:
            raise StylometryError("Feature table has no word columns.")
        return cls(
            text_ids=frame["text_id"].astype(str).tolist(),
            authors=frame["author"].to_numpy(dtype=str),
            words=[str(c)[len(WORD_PREFIX) :] for c in word_columns],
            values=frame[word_columns].to_numpy(dtype=np.float64),
        )
