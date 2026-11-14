"""Step 3: find the words that are stable for every author.

Take each author's K most stable words and intersect the lists. A word is in the
intersection once K reaches its worst rank among the authors (its "entry rank"), so the
shared set only grows as K grows. This picks the smallest K that gives at least
TARGET_SHARED_WORDS words. Topical words (character names, places) are not stable for every
author, so this step also filters them out.

Reads temp/stable_words_by_author.csv and writes temp/shared_stable_words.csv.
"""

from pathlib import Path

import pandas as pd

from programs.common import (
    ROOT,
    SHARED_WORDS,
    STABLE_WORDS,
    TARGET_SHARED_WORDS,
    fail,
    read_table,
    root_from_command_line,
    write_table,
)


def select_shared_words(rankings: pd.DataFrame, target: int) -> pd.DataFrame:
    """Return columns word, entry_rank for the shared words, best first."""
    ranks = rankings.pivot(index="word", columns="author", values="rank").dropna()
    if len(ranks) < target:
        fail(
            f"only {len(ranks)} words appear in every text of every author, fewer than the "
            f"{target} requested. Lower TARGET_SHARED_WORDS in programs/common.py or use "
            "longer or more texts."
        )
    entry = ranks.max(axis=1).astype(int).rename("entry_rank").reset_index()
    entry = entry.sort_values(["entry_rank", "word"])
    k = entry["entry_rank"].iloc[target - 1]
    return entry[entry["entry_rank"] <= k]


def run(root: Path = ROOT) -> None:
    rankings = read_table(root / STABLE_WORDS)
    shared = select_shared_words(rankings, TARGET_SHARED_WORDS)
    write_table(shared, root / SHARED_WORDS, index=False)
    print(f"{len(shared)} shared stable words: {', '.join(shared['word'])}")


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
