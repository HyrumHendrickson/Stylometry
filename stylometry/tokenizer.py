"""Word tokenization."""

from __future__ import annotations

import re

# A token is a run of Unicode letters, optionally joined by single apostrophes
# ("don't", "o'clock"). Digits, underscores, hyphens and other punctuation separate tokens.
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens.

    Typographic apostrophes are normalized to ``'``, so straight and curly
    apostrophes give the same token.
    """
    return _TOKEN.findall(text.translate(_APOSTROPHES).lower())
