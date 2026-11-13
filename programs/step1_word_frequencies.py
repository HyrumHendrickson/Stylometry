"""Step 1: compute the relative frequency of every word in every text.

Reads ``input/<author>/*.txt`` (never modified) and writes ``temp/word_frequencies.csv``
(sparse long table: text_id, author, word, count, frequency) and ``temp/text_totals.csv``.
Each text is processed independently.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from stylometry.cli import execute, parse_args
from stylometry.config import Config, ProjectPaths
from stylometry.corpus import discover_texts
from stylometry.frequencies import build_frequency_tables
from stylometry.tables import write_table

LOGGER = logging.getLogger("step1")


def run(paths: ProjectPaths, config: Config) -> None:
    """Execute step 1."""
    records = discover_texts(
        paths.input_dir, config.corpus.file_extension, config.corpus.min_texts_per_author
    )
    counts, totals = build_frequency_tables(records, config.corpus.encoding)
    write_table(counts, paths.word_frequencies)
    write_table(totals, paths.text_totals)
    LOGGER.info(
        "Counted %d texts by %d authors (%d distinct words). Wrote %s",
        len(totals),
        totals["author"].nunique(),
        counts["word"].nunique(),
        paths.word_frequencies.name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    return execute(parse_args(__doc__, argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
