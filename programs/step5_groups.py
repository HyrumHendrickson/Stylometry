"""Step 5: sort the texts into groups, each group being one proposed author.

Texts are clustered by average linkage: every text starts alone and the two closest groups
are merged repeatedly, where the distance between groups is the mean distance between their
texts. This gives a tree; cutting it into k groups gives k proposed authors.

If NUMBER_OF_GROUPS is None (programs/common.py), k is the number from 2 up to 30 with the
best silhouette score: how much closer a text is to its own group than to the nearest other
group. Set NUMBER_OF_GROUPS, or run with --groups N, if you know how many authors there
are.

Reads temp/distance_matrix.csv and writes results/proposed_groups.png: one column per group
listing its texts (G1, G2, ...), in file-name order.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from programs.common import (
    DISTANCE_MATRIX,
    GROUPS_IMAGE,
    NUMBER_OF_GROUPS,
    ROOT,
    fail,
    parse_command_line,
    read_table,
)

MAX_GROUPS = 30  # the most groups tried when choosing k automatically


def silhouette(distances: np.ndarray, groups: np.ndarray) -> float:
    """Average silhouette: near 1 = tight, well-separated groups; 0 or less = overlapping."""
    scores = []
    for i, group in enumerate(groups):
        own = groups == group
        own[i] = False
        if not own.any():  # a group of one text
            scores.append(0.0)
            continue
        inside = distances[i, own].mean()
        nearest_other = min(distances[i, groups == g].mean() for g in set(groups) - {group})
        scores.append((nearest_other - inside) / max(inside, nearest_other))
    return float(np.mean(scores))


def propose_groups(distances: np.ndarray, number: int | None = None) -> np.ndarray:
    """Return each text's group number (1, 2, ...). Similar groups get neighbouring numbers."""
    if number is not None and not 2 <= number <= len(distances):
        fail(f"the number of groups must be from 2 to {len(distances)} (the number of texts)")
    tree = linkage(squareform(distances, checks=False), method="average")
    if number is None:
        upper = min(MAX_GROUPS, len(distances) - 1)
        number = max(
            range(2, upper + 1),
            key=lambda k: silhouette(distances, fcluster(tree, k, "maxclust")),
        )
    cut = fcluster(tree, number, "maxclust")
    numbers = {old: new for new, old in enumerate(dict.fromkeys(cut[leaves_list(tree)]), start=1)}
    return np.array([numbers[g] for g in cut])


def draw_groups(members: list[list[str]], path: Path) -> None:
    """Save a picture with one column per group, listing the group's texts."""
    longest = max(len(name) for names in members for name in names)
    width = len(members) * (0.075 * longest + 0.6)  # inches; the font is 8 point monospace
    height = 0.14 * (max(len(names) for names in members) + 3) + 0.8
    fig = Figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    total = sum(len(names) for names in members)
    ax.text(
        0.01,
        1 - 0.3 / height,
        f"{total} texts sorted into {len(members)} proposed authors",
        fontsize=11,
        fontweight="bold",
        va="center",
    )
    for number, names in enumerate(members):
        heading = f"G{number + 1}  ({len(names)} texts)"
        ax.text(
            (number + 0.02) / len(members),
            1 - 0.8 / height,
            heading + "\n\n" + "\n".join(names),
            family="monospace",
            fontsize=8,
            va="top",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    print(f"Saved {path}")


def run(root: Path = ROOT, groups: int | None = None) -> None:
    """groups: how many groups to make; None uses NUMBER_OF_GROUPS."""
    matrix = read_table(root / DISTANCE_MATRIX, index_col=0)
    number = NUMBER_OF_GROUPS if groups is None else groups
    groups = propose_groups(matrix.to_numpy(dtype=float), number)
    names = np.array([str(name).removesuffix(".txt") for name in matrix.index])
    members = [names[groups == g].tolist() for g in range(1, groups.max() + 1)]
    draw_groups(members, root / GROUPS_IMAGE)
    print(", ".join(f"G{g + 1}: {len(texts)} texts" for g, texts in enumerate(members)))


if __name__ == "__main__":
    args = parse_command_line(__doc__, groups=True)
    run(args.root, args.groups)
