"""Step 4: write the report.

Image: one matrix: the distance between every pair of texts, in file-name order. Light cells are
similar texts, dark cells are different texts. Name files so that texts you expect to be alike
sit together (for example twain_huckleberry_finn_1.txt) and they show up as light blocks.

Word list: results/words_used.txt has every word the distances are based on, one per line,
starting with the words found in the most texts.

Reads temp/distance_matrix.csv and temp/stable_words.csv. Writes results/stylometry_report.png
and results/words_used.txt.
"""

from pathlib import Path

from programs.common import (
    DISTANCE_MATRIX,
    REPORT_IMAGE,
    ROOT,
    STABLE_WORDS,
    WORDS_USED,
    draw_matrix,
    read_table,
    root_from_command_line,
)


def run(root: Path = ROOT) -> None:
    matrix = read_table(root / DISTANCE_MATRIX, index_col=0)
    labels = [str(name).removesuffix(".txt") for name in matrix.index]
    title = f"Distance between every pair of {len(labels)} texts"
    draw_matrix(matrix.to_numpy(dtype=float), labels, title, root / REPORT_IMAGE)

    words = read_table(root / STABLE_WORDS)["word"]  # already ordered, most widespread first
    (root / WORDS_USED).write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"Saved {len(words)} words to {root / WORDS_USED}")


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
