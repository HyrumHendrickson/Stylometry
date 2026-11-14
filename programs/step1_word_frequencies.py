"""Step 1: the relative frequency of every word in every text.

Reads input/<author>/*.txt (never modified) and writes temp/word_frequencies.csv with one
row per word per text: text_id, author, word, frequency (= count / words in that text).
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from programs.common import (
    ENCODING,
    EXTENSION,
    ROOT,
    WORD_FREQUENCIES,
    fail,
    root_from_command_line,
    write_table,
)

# A word is a run of letters, optionally joined by single apostrophes ("don't").
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
CURLY_APOSTROPHES = str.maketrans("’‘ʼ", "'''")


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.translate(CURLY_APOSTROPHES).lower())


def find_texts(input_dir: Path) -> dict[str, list[Path]]:
    """Map each author folder name to its text files."""
    if not input_dir.is_dir():
        fail(f"input folder not found: {input_dir}")
    texts = {}
    for folder in sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name[0] != "."):
        files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() == EXTENSION]
        if files:
            texts[folder.name] = files
    if len(texts) < 2:
        fail(f"need at least two author folders containing {EXTENSION} files in {input_dir}")
    for author, files in texts.items():
        if len(files) < 2:
            fail(f"author {author!r} has one text; at least two are needed")
    return texts


def run(root: Path = ROOT) -> None:
    rows = []
    for author, files in find_texts(root / "input").items():
        for file in files:
            try:
                words = tokenize(file.read_text(encoding=ENCODING))
            except UnicodeDecodeError:
                fail(f"cannot read {file} as {ENCODING}; change ENCODING in programs/common.py")
            if not words:
                fail(f"{file} contains no words")
            for word, count in Counter(words).items():
                rows.append((f"{author}/{file.name}", author, word, count / len(words)))

    frequencies = pd.DataFrame(rows, columns=["text_id", "author", "word", "frequency"])
    write_table(frequencies, root / WORD_FREQUENCIES, index=False)
    print(
        f"{frequencies['text_id'].nunique()} texts by {frequencies['author'].nunique()} authors, "
        f"{frequencies['word'].nunique()} distinct words"
    )


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
