"""Step 3: find the stable words that every author has in common.

Reads ``temp/stable_words_by_author.csv``. Chooses the smallest list length K such that the
intersection of every author's K most stable words has at least
``stability.target_shared_words`` words, and writes that intersection to
``temp/shared_stable_words.csv`` (plus ``shared_stable_words_meta.json``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from stylometry.cli import execute, parse_args
from stylometry.config import Config, ProjectPaths
from stylometry.stability import select_shared_words
from stylometry.tables import read_table, write_table

LOGGER = logging.getLogger("step3")


def run(paths: ProjectPaths, config: Config) -> None:
    """Execute step 3."""
    rankings = read_table(paths.stable_words, "programs.step2_stable_words")
    selection = select_shared_words(rankings, config.stability.target_shared_words)
    write_table(selection.words, paths.shared_words)
    meta = {
        "k": selection.k,
        "target_shared_words": selection.target,
        "n_shared_words": len(selection.words),
        "authors": sorted(rankings["author"].unique().tolist()),
    }
    paths.shared_words_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "K = %d gives %d shared stable words: %s",
        selection.k,
        len(selection.words),
        ", ".join(selection.words["word"].tolist()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    return execute(parse_args(__doc__, argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
