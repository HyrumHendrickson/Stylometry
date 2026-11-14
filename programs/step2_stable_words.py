"""Step 2: choose the words to compare the texts on.

A word is used only if it appears in at least MIN_SHARE_OF_TEXTS of the texts (see
programs/common.py). Words that appear in only a few texts are topical (character names,
places) and would make texts look different because of their subject, not their style.

Reads temp/word_frequencies.csv and writes temp/stable_words.csv (word, share), where share
is the fraction of texts that contain the word.
"""

from pathlib import Path

import pandas as pd

from programs.common import (
    MIN_SHARE_OF_TEXTS,
    ROOT,
    STABLE_WORDS,
    WORD_FREQUENCIES,
    fail,
    read_table,
    root_from_command_line,
    write_table,
)


def stable_words(frequencies: pd.DataFrame) -> pd.DataFrame:
    """Words found in enough texts, most widespread first."""
    share = frequencies.groupby("word")["text_id"].nunique() / frequencies["text_id"].nunique()
    share = share[share >= MIN_SHARE_OF_TEXTS].rename("share").reset_index()
    if share.empty:
        fail(f"no word appears in {MIN_SHARE_OF_TEXTS:.0%} of the texts. Lower MIN_SHARE_OF_TEXTS.")
    return share.sort_values(["share", "word"], ascending=[False, True])


def run(root: Path = ROOT) -> None:
    words = stable_words(read_table(root / WORD_FREQUENCIES))
    write_table(words, root / STABLE_WORDS, index=False)
    print(f"{len(words)} words appear in at least {MIN_SHARE_OF_TEXTS:.0%} of the texts")


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
