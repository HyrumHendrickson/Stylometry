"""End-to-end test on a synthetic corpus."""

import hashlib

import pandas as pd
import pytest

import run_full_test


def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*.txt")}


def test_full_pipeline(project):
    before = digest(project / "input")
    run_full_test.main(project)
    assert digest(project / "input") == before, "input/ must never be modified"

    assert (project / "results" / "stylometry_report.png").stat().st_size > 0

    shared = pd.read_csv(project / "temp" / "shared_stable_words.csv", keep_default_na=False)
    assert len(shared) >= 30
    assert not shared["word"].str.startswith("zq").any(), "topical words must not be shared"

    matrix = pd.read_csv(project / "temp" / "distance_matrix.csv", index_col=0)
    assert matrix.shape == (20, 20)
    assert (matrix.to_numpy().diagonal() == 0).all()

    # The synthetic corpus gives "to", "it" and "i" author-specific rates.
    power = pd.read_csv(project / "temp" / "word_power.csv", keep_default_na=False)
    assert {"to", "it", "i"} <= set(power["word"].head(5))
    assert len(power) == len(shared)


def test_texts_by_one_author_are_closer_than_texts_by_different_authors(project):
    run_full_test.main(project)
    matrix = pd.read_csv(project / "temp" / "distance_matrix.csv", index_col=0)
    authors = matrix.index.str.split("/").str[0].to_numpy()
    same = authors[:, None] == authors[None, :]
    off_diagonal = ~(matrix.to_numpy() == 0)
    assert matrix.to_numpy()[same & off_diagonal].mean() < matrix.to_numpy()[~same].mean()


def test_pipeline_stops_with_a_clear_message_when_input_is_empty(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(SystemExit, match="step 1 failed"):
        run_full_test.main(tmp_path)
