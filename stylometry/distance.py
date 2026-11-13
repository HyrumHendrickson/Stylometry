"""Distance models between word-frequency vectors (step 4 and the leave-one-out test).

All models consume *relative frequencies* (rows = texts, columns = words) and expose the
same interface: ``fit(x, labels)`` learns any corpus-dependent parameters from labelled
training rows, and ``pairwise(a, b)`` returns the distance between every row of ``a`` and
every row of ``b``.

The default model is :class:`FisherDistance`. Its motivation is that the words selected as
stable for every author include many whose rates barely differ between authors. Treating
every word equally (as Burrows's Delta does after z-scoring) lets those uninformative words
add noise of the same size as the few informative ones. The Fisher-weighted distance
estimates, from the labelled texts, how much each word separates the authors relative to
how much it varies within an author, and weights words accordingly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stylometry.errors import StylometryError

LOGGER = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
StrArray = NDArray[np.str_]

METRICS = ("fisher", "delta", "euclidean", "cosine")

METRIC_LABELS = {
    "fisher": "Fisher-weighted",
    "delta": "Burrows's Delta",
    "euclidean": "Euclidean (sqrt-frequency)",
    "cosine": "Cosine",
}


@dataclass(frozen=True)
class DiscriminationStats:
    """One-way ANOVA quantities per word, computed on variance-stabilized frequencies.

    Attributes:
        within_variance: Pooled within-author variance ``MSW_j`` (floored).
        between_variance: Between-author mean square ``MSB_j``.
        f_ratio: ``MSB_j / MSW_j``. Values near 1 mean the word's rate does not differ
            between authors beyond within-author noise.
        weights: Non-negative weights summing to 1, proportional to ``max(F_j - 1, 0)``.
            If no word has ``F_j > 1`` the weights are uniform.
    """

    within_variance: FloatArray
    between_variance: FloatArray
    f_ratio: FloatArray
    weights: FloatArray


def variance_stabilize(frequencies: FloatArray) -> FloatArray:
    """Square-root transform of relative frequencies.

    Word counts are approximately Poisson, so the variance of a relative frequency grows
    with its mean. The square root makes the variance roughly independent of the mean, which
    puts frequent and infrequent words on a comparable scale.
    """
    return np.sqrt(np.asarray(frequencies, dtype=np.float64))


def discrimination_stats(
    x: FloatArray, labels: StrArray, variance_floor: float = 1e-12
) -> DiscriminationStats:
    """Compute per-word ANOVA statistics and Fisher weights.

    With ``N`` rows, ``A`` classes, class means ``m_a``, class sizes ``n_a`` and grand mean
    ``m``::

        MSB_j = sum_a n_a (m_aj - m_j)^2 / (A - 1)
        MSW_j = sum_a sum_{i in a} (x_ij - m_aj)^2 / (N - A)
        F_j   = MSB_j / MSW_j
        w_j   = max(F_j - 1, 0) / sum_k max(F_k - 1, 0)

    Args:
        x: Feature matrix (already variance-stabilized), shape ``(N, m)``.
        labels: Author label per row.
        variance_floor: Lower bound for ``MSW_j``.

    Raises:
        StylometryError: If there are fewer than two classes or no spare degrees of freedom.
    """
    labels = np.asarray(labels)
    classes, inverse = np.unique(labels, return_inverse=True)
    n_rows, n_features = x.shape
    n_classes = len(classes)
    if n_classes < 2:
        raise StylometryError("At least two authors are required.")
    if n_rows <= n_classes:
        raise StylometryError("Need more texts than authors to estimate within-author variance.")

    grand_mean = x.mean(axis=0)
    ss_between = np.zeros(n_features)
    ss_within = np.zeros(n_features)
    for index in range(n_classes):
        block = x[inverse == index]
        class_mean = block.mean(axis=0)
        ss_between += block.shape[0] * (class_mean - grand_mean) ** 2
        ss_within += ((block - class_mean) ** 2).sum(axis=0)

    ms_between = ss_between / (n_classes - 1)
    ms_within = np.maximum(ss_within / (n_rows - n_classes), variance_floor)
    f_ratio = ms_between / ms_within

    raw = np.maximum(f_ratio - 1.0, 0.0)
    total = raw.sum()
    if total > 0:
        weights = raw / total
    else:
        LOGGER.warning("No word has F > 1; falling back to uniform weights.")
        weights = np.full(n_features, 1.0 / n_features)
    return DiscriminationStats(ms_within, ms_between, f_ratio, weights)


def _row_distances(a: FloatArray, b: FloatArray, norm: str) -> FloatArray:
    """Distance between every row of ``a`` and every row of ``b`` (row-by-row, low memory)."""
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    for i, row in enumerate(a):
        diff = b - row
        if norm == "l2":
            out[i] = np.sqrt((diff**2).sum(axis=1))
        else:
            out[i] = np.abs(diff).mean(axis=1)
    return out


class DistanceModel(ABC):
    """Common interface of all distance models."""

    name: str

    def fit(self, x: FloatArray, labels: StrArray) -> DistanceModel:
        """Learn corpus-dependent parameters from labelled training rows. Returns ``self``."""
        return self

    @abstractmethod
    def pairwise(self, a: FloatArray, b: FloatArray) -> FloatArray:
        """Distances between each row of ``a`` and each row of ``b``."""

    def matrix(self, x: FloatArray) -> FloatArray:
        """Symmetric self-distance matrix with an exactly zero diagonal."""
        d = self.pairwise(x, x)
        d = (d + d.T) / 2.0
        np.fill_diagonal(d, 0.0)
        return d


class FisherDistance(DistanceModel):
    """Fisher-weighted standardized Euclidean distance on square-root frequencies.

    ``d(u, v)^2 = sum_j w_j (z_uj - z_vj)^2 / MSW_j`` with ``z = sqrt(frequency)``, where
    ``MSW_j`` is the pooled within-author variance and ``w_j`` the Fisher weight from
    :func:`discrimination_stats`. This is a diagonal form of linear discriminant analysis.
    Because the weights sum to one, ``d^2`` is a weighted average of squared within-author
    standard deviations separating the two texts.
    """

    name = "fisher"

    def __init__(self, variance_floor: float = 1e-12) -> None:
        self._floor = variance_floor
        self._scale: FloatArray | None = None
        self._stats: DiscriminationStats | None = None

    @property
    def stats(self) -> DiscriminationStats:
        """Statistics learned by :meth:`fit`."""
        if self._stats is None:
            raise StylometryError("FisherDistance has not been fitted.")
        return self._stats

    def fit(self, x: FloatArray, labels: StrArray) -> DistanceModel:
        self._stats = discrimination_stats(variance_stabilize(x), labels, self._floor)
        self._scale = np.sqrt(self._stats.weights / self._stats.within_variance)
        return self

    def pairwise(self, a: FloatArray, b: FloatArray) -> FloatArray:
        if self._scale is None:
            raise StylometryError("FisherDistance has not been fitted.")
        return _row_distances(
            variance_stabilize(a) * self._scale, variance_stabilize(b) * self._scale, "l2"
        )


class BurrowsDelta(DistanceModel):
    """Burrows's Delta: mean absolute difference of corpus-standardized frequencies."""

    name = "delta"

    def __init__(self) -> None:
        self._mean: FloatArray | None = None
        self._std: FloatArray | None = None

    def fit(self, x: FloatArray, labels: StrArray) -> DistanceModel:
        if x.shape[0] < 2:
            raise StylometryError("Burrows's Delta needs at least two texts to fit.")
        self._mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=1)
        self._std = np.where(std > 0, std, 1.0)
        return self

    def pairwise(self, a: FloatArray, b: FloatArray) -> FloatArray:
        if self._mean is None or self._std is None:
            raise StylometryError("BurrowsDelta has not been fitted.")
        return _row_distances((a - self._mean) / self._std, (b - self._mean) / self._std, "l1")


class SqrtEuclidean(DistanceModel):
    """Unweighted Euclidean distance on square-root frequencies."""

    name = "euclidean"

    def pairwise(self, a: FloatArray, b: FloatArray) -> FloatArray:
        return _row_distances(variance_stabilize(a), variance_stabilize(b), "l2")


class CosineDistance(DistanceModel):
    """Cosine distance ``1 - cos(u, v)`` on raw relative frequencies."""

    name = "cosine"

    @staticmethod
    def _unit(x: FloatArray) -> FloatArray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 0, norms, 1.0)

    def pairwise(self, a: FloatArray, b: FloatArray) -> FloatArray:
        similarity = self._unit(a) @ self._unit(b).T
        return np.clip(1.0 - similarity, 0.0, 2.0)


def make_model(name: str, variance_floor: float = 1e-12) -> DistanceModel:
    """Create a distance model by name (one of :data:`METRICS`)."""
    if name == "fisher":
        return FisherDistance(variance_floor)
    if name == "delta":
        return BurrowsDelta()
    if name == "euclidean":
        return SqrtEuclidean()
    if name == "cosine":
        return CosineDistance()
    raise StylometryError(f"Unknown metric {name!r}; choose from {', '.join(METRICS)}.")


def leave_one_out_predictions(
    x: FloatArray, labels: StrArray, metric: str, variance_floor: float = 1e-12
) -> StrArray:
    """Nearest-neighbour author prediction for every text, with no label leakage.

    For each text the model (including all weights and standardization constants) is refit
    on the other texts only, and the held-out text is assigned the author of its nearest
    remaining text.
    """
    labels = np.asarray(labels)
    n_rows = x.shape[0]
    predictions = np.empty(n_rows, dtype=labels.dtype)
    for i in range(n_rows):
        keep = np.arange(n_rows) != i
        model = make_model(metric, variance_floor).fit(x[keep], labels[keep])
        distances = model.pairwise(x[i : i + 1], x[keep])[0]
        predictions[i] = labels[keep][int(np.argmin(distances))]
    return predictions
