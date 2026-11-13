"""Step 4: vectorize every text and compute all pairwise distances.

Each text becomes a vector of its relative frequencies over the shared stable words. The
configured metric (default: Fisher-weighted) is fitted on all texts and used to compute the
full text-by-text distance matrix. Writes ``temp/feature_vectors.csv``,
``temp/word_weights.csv`` (per-word F-ratio and weight) and ``temp/distance_matrix.csv``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from stylometry.cli import execute, parse_args
from stylometry.config import Config, ProjectPaths
from stylometry.distance import discrimination_stats, make_model, variance_stabilize
from stylometry.features import FeatureMatrix
from stylometry.tables import read_table, write_table

LOGGER = logging.getLogger("step4")


def run(paths: ProjectPaths, config: Config) -> None:
    """Execute step 4."""
    frequencies = read_table(paths.word_frequencies, "programs.step1_word_frequencies")
    totals = read_table(paths.text_totals, "programs.step1_word_frequencies")
    shared = read_table(paths.shared_words, "programs.step3_shared_words")

    features = FeatureMatrix.build(frequencies, totals, shared["word"].tolist())
    write_table(features.to_frame(), paths.feature_vectors)

    floor = config.distance.variance_floor
    stats = discrimination_stats(variance_stabilize(features.values), features.authors, floor)
    weights = pd.DataFrame(
        {
            "word": features.words,
            "within_variance": stats.within_variance,
            "between_variance": stats.between_variance,
            "f_ratio": stats.f_ratio,
            "weight": stats.weights,
        }
    ).sort_values("f_ratio", ascending=False)
    write_table(weights, paths.word_weights)

    model = make_model(config.distance.metric, floor).fit(features.values, features.authors)
    matrix = pd.DataFrame(
        model.matrix(features.values), index=features.text_ids, columns=features.text_ids
    )
    matrix.index.name = "text_id"
    write_table(matrix, paths.distance_matrix, index=True)
    LOGGER.info(
        "%s distances for %d texts over %d words. Most discriminative: %s",
        config.distance.metric,
        len(features.text_ids),
        len(features.words),
        ", ".join(weights["word"].head(5)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    return execute(parse_args(__doc__, argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
