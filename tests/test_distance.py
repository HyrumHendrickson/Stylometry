"""Tests for the distance models and Fisher weighting."""

from __future__ import annotations

import numpy as np
import pytest

from stylometry.distance import (
    METRICS,
    BurrowsDelta,
    CosineDistance,
    FisherDistance,
    discrimination_stats,
    leave_one_out_predictions,
    make_model,
    variance_stabilize,
)
from stylometry.errors import StylometryError


def test_discrimination_stats_matches_hand_computation() -> None:
    x = np.array([[0.0], [2.0], [10.0], [12.0]])
    labels = np.array(["a", "a", "b", "b"])
    stats = discrimination_stats(x, labels)
    # Class means 1 and 11, grand mean 6: MSB = 100, MSW = 4 / (4 - 2) = 2.
    assert stats.between_variance[0] == pytest.approx(100.0)
    assert stats.within_variance[0] == pytest.approx(2.0)
    assert stats.f_ratio[0] == pytest.approx(50.0)
    assert stats.weights[0] == pytest.approx(1.0)


def test_discrimination_stats_uniform_when_no_signal() -> None:
    x = np.array([[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0]])
    labels = np.array(["a", "a", "b", "b"])
    stats = discrimination_stats(x, labels)
    assert stats.f_ratio == pytest.approx([0.0, 0.0])
    assert stats.weights == pytest.approx([0.5, 0.5])


def test_discrimination_stats_validates_shape() -> None:
    with pytest.raises(StylometryError):
        discrimination_stats(np.ones((3, 2)), np.array(["a", "a", "a"]))
    with pytest.raises(StylometryError):
        discrimination_stats(np.ones((2, 2)), np.array(["a", "b"]))


def _two_signal_dataset(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """3 authors x 6 texts, 30 words; only words 0 and 1 differ between authors."""
    rng = np.random.default_rng(seed)
    n_words = 30
    base = np.linspace(0.02, 0.005, n_words)
    rows, labels = [], []
    for author, shift in zip("abc", (0.0, 1.0, -1.0), strict=True):
        mean = base.copy()
        mean[:2] *= 1.0 + 0.5 * shift
        for _ in range(6):
            rows.append(mean * np.exp(0.12 * rng.normal(size=n_words)))
            labels.append(author)
    return np.array(rows), np.array(labels)


def test_fisher_weights_concentrate_on_informative_words() -> None:
    x, labels = _two_signal_dataset()
    stats = discrimination_stats(variance_stabilize(x), labels)
    assert stats.weights[:2].sum() > 0.8
    assert stats.weights.sum() == pytest.approx(1.0)
    assert (stats.weights >= 0).all()


def test_fisher_beats_delta_when_few_words_carry_the_signal() -> None:
    x, labels = _two_signal_dataset()
    fisher = leave_one_out_predictions(x, labels, "fisher")
    delta = leave_one_out_predictions(x, labels, "delta")
    fisher_accuracy = float(np.mean(fisher == labels))
    delta_accuracy = float(np.mean(delta == labels))
    assert fisher_accuracy >= 0.9
    assert fisher_accuracy >= delta_accuracy


@pytest.mark.parametrize("metric", METRICS)
def test_models_produce_valid_distance_matrices(metric: str) -> None:
    x, labels = _two_signal_dataset()
    d = make_model(metric).fit(x, labels).matrix(x)
    assert d.shape == (len(x), len(x))
    assert np.isfinite(d).all()
    assert (d >= 0).all()
    assert np.array_equal(d, d.T)
    assert np.all(np.diag(d) == 0.0)


def test_fisher_single_feature_formula() -> None:
    x = np.array([[0.04], [0.09], [0.25], [0.36]])
    labels = np.array(["a", "a", "b", "b"])
    model = FisherDistance().fit(x, labels)
    z = np.sqrt(x[:, 0])
    msw = (((z[:2] - z[:2].mean()) ** 2).sum() + ((z[2:] - z[2:].mean()) ** 2).sum()) / 2
    expected = abs(z[0] - z[3]) / np.sqrt(msw)
    assert model.pairwise(x[:1], x[3:])[0, 0] == pytest.approx(expected)
    assert model.stats.weights[0] == pytest.approx(1.0)


def test_fisher_requires_fit() -> None:
    with pytest.raises(StylometryError):
        FisherDistance().pairwise(np.ones((1, 2)), np.ones((1, 2)))


def test_burrows_delta_matches_definition() -> None:
    x = np.array([[0.1, 0.5], [0.2, 0.3], [0.4, 0.4]])
    model = BurrowsDelta().fit(x, np.array(["a", "a", "b"]))
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    expected = np.abs(z[0] - z[2]).mean()
    assert model.pairwise(x[:1], x[2:])[0, 0] == pytest.approx(expected)


def test_cosine_ignores_scale() -> None:
    a = np.array([[0.1, 0.2, 0.3]])
    b = a * 3.0
    assert CosineDistance().pairwise(a, b)[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_unknown_metric() -> None:
    with pytest.raises(StylometryError):
        make_model("manhattan")


def test_leave_one_out_ignores_the_held_out_texts_own_label() -> None:
    """Relabelling a text must not change how that text is classified (no label leakage)."""
    x, labels = _two_signal_dataset()
    baseline = leave_one_out_predictions(x, labels, "fisher")
    relabelled = labels.copy()
    relabelled[0] = "c"
    changed = leave_one_out_predictions(x, relabelled, "fisher")
    assert changed[0] == baseline[0]
