"""Optional: score the result by treating file names as hidden author labels. Not a step.

The pipeline never sees authors. If you named files <author>_<work>_<number>.txt (for example
twain_tom_sawyer_2.txt), this program takes the text before the first underscore as the
author and the name without the trailing _<number> as the work, then reports:

  * nearest neighbour: how often a text's closest text from a different work has the same
    author (100% is perfect);
  * AUC: the chance that a random same-author pair from different works is closer than a
    random pair from different authors (1.0 is perfect, 0.5 is chance);
  * how the proposed groups of step 5 split by author.

Use it to compare settings in programs/common.py. Reads temp/distance_matrix.csv.
Run:  python -m programs.check_names [--root PATH]
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from programs.common import (
    DISTANCE_MATRIX,
    NUMBER_OF_GROUPS,
    ROOT,
    fail,
    read_table,
    root_from_command_line,
)
from programs.step5_groups import propose_groups


def run(root: Path = ROOT) -> None:
    matrix = read_table(root / DISTANCE_MATRIX, index_col=0)
    distances = matrix.to_numpy(dtype=float)
    names = matrix.index.astype(str)
    author = np.array([name.split("_")[0] for name in names])
    work = np.array([re.sub(r"_\d+\.\w+$", "", name) for name in names])
    same_author = author[:, None] == author[None, :]
    same_work = work[:, None] == work[None, :]

    other_works = np.where(same_work, np.inf, distances)
    nearest_correct = (author[other_works.argmin(axis=1)] == author).mean()

    close = distances[same_author & ~same_work]
    far = distances[~same_author]
    if close.size == 0:
        fail("need two or more works per author, named <author>_<work>_<number>.txt")
    ranks = pd.Series(np.concatenate([close, far])).rank().to_numpy()
    auc = 1 - (ranks[: len(close)].sum() - len(close) * (len(close) + 1) / 2) / (
        len(close) * len(far)
    )

    print(f"nearest neighbour (other works): {nearest_correct:.1%}")
    print(f"AUC: {auc:.3f}")
    groups = propose_groups(distances, NUMBER_OF_GROUPS)
    print("\nproposed groups by author (rows = groups):")
    print(pd.crosstab(pd.Series(groups, name="group"), pd.Series(author, name="author")))


if __name__ == "__main__":
    run(root_from_command_line(__doc__))
