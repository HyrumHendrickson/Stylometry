"""Step 5: statistical report.

For every author, computes the standard deviation (and mean) of the distances between that
author's own texts, and for every pair of authors the standard deviation of the distances
between their texts. Adds a leave-one-out attribution test (primary metric versus baseline)
and saves everything as one image in ``results/``, plus ``distance_statistics.csv``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from stylometry.cli import execute, parse_args
from stylometry.config import Config, ProjectPaths
from stylometry.distance import METRIC_LABELS, leave_one_out_predictions
from stylometry.distance_statistics import accuracy_by_author, group_statistics
from stylometry.errors import StylometryError
from stylometry.features import FeatureMatrix
from stylometry.plotting import ReportData, render_report
from stylometry.tables import read_table, write_table

LOGGER = logging.getLogger("step5")


def _format_summary(
    stats: pd.DataFrame, accuracy: dict[str, float], overall: float, metric: str
) -> str:
    table = stats.copy()
    table["pair"] = [
        a if g == "within" else f"{a} vs {b}"
        for g, a, b in zip(table["group"], table["author_a"], table["author_b"], strict=True)
    ]
    body = table[["group", "pair", "n_distances", "mean", "std"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    )
    accuracy_lines = "\n".join(f"  {name}: {value:.0%}" for name, value in accuracy.items())
    return (
        f"\nDistance statistics ({METRIC_LABELS[metric]})\n{body}\n\n"
        f"Leave-one-out attribution accuracy: {overall:.0%}\n{accuracy_lines}\n"
    )


def run(paths: ProjectPaths, config: Config) -> None:
    """Execute step 5."""
    features = FeatureMatrix.from_frame(
        read_table(paths.feature_vectors, "programs.step4_distance_matrix")
    )
    distances = read_table(paths.distance_matrix, "programs.step4_distance_matrix", index_col=0)
    if list(distances.index.astype(str)) != features.text_ids:
        raise StylometryError("distance_matrix.csv and feature_vectors.csv disagree; rerun step 4.")
    word_weights = read_table(paths.word_weights, "programs.step4_distance_matrix")

    matrix = distances.to_numpy(dtype=float)
    stats = group_statistics(matrix, features.authors)
    write_table(stats, paths.report_statistics)

    floor = config.distance.variance_floor
    metric, baseline = config.distance.metric, config.distance.baseline_metric
    predicted = leave_one_out_predictions(features.values, features.authors, metric, floor)
    predicted_baseline = leave_one_out_predictions(
        features.values, features.authors, baseline, floor
    )
    accuracy, overall = accuracy_by_author(features.authors, predicted)
    baseline_accuracy, baseline_overall = accuracy_by_author(features.authors, predicted_baseline)

    render_report(
        ReportData(
            distances=matrix,
            authors=features.authors,
            statistics=stats,
            word_weights=word_weights,
            metric=metric,
            baseline_metric=baseline,
            accuracy=accuracy,
            baseline_accuracy=baseline_accuracy,
            overall_accuracy=overall,
            baseline_overall_accuracy=baseline_overall,
        ),
        paths.report_image,
    )
    print(_format_summary(stats, accuracy, overall, metric))
    LOGGER.info(
        "Baseline (%s) leave-one-out accuracy: %.0f%%",
        METRIC_LABELS[baseline],
        100 * baseline_overall,
    )
    LOGGER.info("Saved report image to %s", paths.report_image)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    return execute(parse_args(__doc__, argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
