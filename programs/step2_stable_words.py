"""Step 2: for each author, rank the words used at the most consistent rate.

A word qualifies for an author only if it appears in every one of that author's texts.
Qualifying words are ranked by the coefficient of variation (standard deviation / mean) of
their frequency across the author's texts, lowest first (rank 1 = most consistent).
Ties go to the more frequent word, then alphabetical order.

Reads temp/word_frequencies.csv and writes temp/stable_words_by_author.csv.
"""

from pathlib import Path

import pandas as pd

from programs.common import (
    ROOT,
    STABLE_WORDS,
    WORD_FREQUENCIES,
    read_table,
    root_from_command_line,
    write_table,
)


def rank_words(author_rows: pd.DataFrame) -> pd.DataFrame:
    """Rank the words of one author. Input: that author's rows from step 1."""
    # One row per text, one column per word; a word missing from any text is dropped.
    wide = author_rows.pivot(index="text_id", columns="word", values="frequency").dropna(axis=1)
    ranking = pd.DataFrame({"mean": wide.mean(), "std": wide.std()})
    ranking["cv"] = ranking["std"] / ranking["mean"]
    ranking = ranking.rename_axis("word").reset_index()
    ranking = ranking.sort_values(["cv", "mean", "word"], ascending=[True, False, True])
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking[["rank", "word", "cv"]]


def run(root: Path = ROOT) -> None:
    frequencies = read_table(root / WORD_FREQUENCIES)
    rankings = pd.concat(
        [rank_words(rows).assign(author=author) for author, rows in frequencies.groupby("author")]
    )
    write_table(rankings[["author", "rank", "word", "cv"]], root / STABLE_WORDS, index=False)
    for author, rows in rankings.groupby("author"):
        print(f"{author}: {len(rows)} words appear in every text")


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
