"""Step 1: the relative frequency of every word in every text.

Reads the text files directly inside input/ (never modified; subfolders are ignored) and
writes temp/word_frequencies.csv with one row per word per text: text_id (the file name),
word, frequency (= count / words in that text).
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


def find_texts(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        fail(f"input folder not found: {input_dir}")
    files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() == EXTENSION]
    if len(files) < 3:
        fail(f"need at least three {EXTENSION} files directly inside {input_dir}")
    return files


def run(root: Path = ROOT) -> None:
    rows = []
    for file in find_texts(root / "input"):
        try:
            words = tokenize(file.read_text(encoding=ENCODING))
        except UnicodeDecodeError:
            fail(f"cannot read {file} as {ENCODING}; change ENCODING in programs/common.py")
        if not words:
            fail(f"{file} contains no words")
        for word, count in Counter(words).items():
            rows.append((file.name, word, count / len(words)))

    frequencies = pd.DataFrame(rows, columns=["text_id", "word", "frequency"])
    write_table(frequencies, root / WORD_FREQUENCIES, index=False)
    print(
        f"{frequencies['text_id'].nunique()} texts, {frequencies['word'].nunique()} distinct words"
    )


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
