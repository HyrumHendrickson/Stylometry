"""Step 2: find, for each author, the words used at the most consistent rate.

Reads ``temp/word_frequencies.csv``. For each author independently, keeps the words that
occur in every one of that author's texts and ranks them by the coefficient of variation
of their relative frequency (lowest first). Writes ``temp/stable_words_by_author.csv``.
"""

from __future__ import annotations

from collections.abc import Sequence

from stylometry.cli import execute, parse_args
from stylometry.config import Config, ProjectPaths
from stylometry.stability import rank_stable_words_by_author
from stylometry.tables import read_table, write_table


def run(paths: ProjectPaths, config: Config) -> None:
    """Execute step 2."""
    del config  # No tunable parameters; the list length is chosen in step 3.
    frequencies = read_table(paths.word_frequencies, "programs.step1_word_frequencies")
    rankings = rank_stable_words_by_author(frequencies)
    write_table(rankings, paths.stable_words)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    return execute(parse_args(__doc__, argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
