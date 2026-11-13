"""End-to-end tests of ``run_full_test.py`` on a synthetic corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import run_full_test
from stylometry.config import ProjectPaths


def _digest(folder: Path) -> dict[str, str]:
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_discover_programs_are_in_numeric_order() -> None:
    assert run_full_test.discover_programs() == [
        "step1_word_frequencies",
        "step2_stable_words",
        "step3_shared_words",
        "step4_distance_matrix",
        "step5_report",
    ]


def test_full_pipeline(synthetic_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = _digest(synthetic_project / "input")
    assert run_full_test.main(["--root", str(synthetic_project)]) == 0
    capsys.readouterr()

    assert _digest(synthetic_project / "input") == before, "input/ must never be modified"

    paths = ProjectPaths(synthetic_project)
    for artifact in (
        paths.word_frequencies,
        paths.text_totals,
        paths.stable_words,
        paths.shared_words,
        paths.shared_words_meta,
        paths.feature_vectors,
        paths.word_weights,
        paths.distance_matrix,
        paths.report_statistics,
        paths.report_image,
    ):
        assert artifact.is_file() and artifact.stat().st_size > 0, artifact.name

    meta = json.loads(paths.shared_words_meta.read_text(encoding="utf-8"))
    assert meta["n_shared_words"] >= 30
    shared = pd.read_csv(paths.shared_words, keep_default_na=False)
    assert not shared["word"].str.startswith("zq").any(), "topical words must not be shared"

    matrix = pd.read_csv(paths.distance_matrix, index_col=0)
    assert matrix.shape == (20, 20)
    assert matrix.to_numpy().diagonal().tolist() == [0.0] * 20

    stats = pd.read_csv(paths.report_statistics)
    assert (stats["group"] == "within").sum() == 4
    assert (stats["group"] == "between").sum() == 6

    weights = pd.read_csv(paths.word_weights, keep_default_na=False)
    informative = {"to", "it", "i"}  # COMMON_WORDS[3, 11, 19] carry author-specific rates
    assert informative & set(weights["word"].head(8))


def test_pipeline_is_repeatable(
    synthetic_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_full_test.main(["--root", str(synthetic_project)]) == 0
    first = ProjectPaths(synthetic_project).distance_matrix.read_bytes()
    assert run_full_test.main(["--root", str(synthetic_project)]) == 0
    capsys.readouterr()
    assert ProjectPaths(synthetic_project).distance_matrix.read_bytes() == first


def test_pipeline_stops_at_first_failure(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_full_test.main(["--root", str(project_root)]) == 1
    captured = capsys.readouterr()
    assert "FAILED at step1_word_frequencies" in captured.err
    assert not ProjectPaths(project_root).stable_words.exists()
