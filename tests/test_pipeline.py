"""End-to-end test on a synthetic corpus."""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_full_test
from programs import check_names
from programs.step5_groups import propose_groups


def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*.txt")}


def test_full_pipeline(project):
    before = digest(project / "input")
    run_full_test.main(project)
    assert digest(project / "input") == before, "input/ must never be modified"
    assert (project / "results" / "stylometry_report.png").stat().st_size > 0

    words = pd.read_csv(project / "temp" / "stable_words.csv", keep_default_na=False)
    assert len(words) >= 40
    assert not words["word"].str.startswith("zq").any(), "topical words must not be used"

    listed = (project / "results" / "words_used.txt").read_text(encoding="utf-8").splitlines()
    assert listed == words["word"].tolist(), "words_used.txt lists every word used, in order"

    matrix = pd.read_csv(project / "temp" / "distance_matrix.csv", index_col=0)
    assert matrix.shape == (25, 25)
    assert (matrix.to_numpy().diagonal() == 0).all()
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
    assert (project / "results" / "proposed_groups.png").stat().st_size > 0


def test_texts_by_one_author_are_closer_than_texts_by_different_authors(project):
    run_full_test.main(project)
    matrix = pd.read_csv(project / "temp" / "distance_matrix.csv", index_col=0)
    authors = matrix.index.str.split("_").str[0].to_numpy()  # authora_work0_1.txt -> authora
    same = authors[:, None] == authors[None, :]
    distances = matrix.to_numpy()
    assert distances[same & (distances > 0)].mean() < distances[~same].mean()


def test_proposed_groups_match_the_authors(project):
    run_full_test.main(project)
    matrix = pd.read_csv(project / "temp" / "distance_matrix.csv", index_col=0)
    authors = matrix.index.str.split("_").str[0].to_numpy()
    groups = propose_groups(matrix.to_numpy())  # number of groups chosen automatically
    assert len(set(groups)) == 5
    for group in set(groups):
        assert len(set(authors[groups == group])) == 1, "each group holds a single author"


def test_check_names_runs_on_the_pipeline_output(project, capsys):
    run_full_test.main(project)
    capsys.readouterr()
    check_names.run(project)
    output = capsys.readouterr().out
    assert "AUC" in output and "proposed groups by author" in output


def test_groups_option_sets_the_number_of_groups(project, capsys):
    run_full_test.main(project, groups=3)
    summary = capsys.readouterr().out.splitlines()[-1]
    assert summary.startswith("G1:") and "G3:" in summary and "G4:" not in summary


def test_groups_flag_on_the_command_line(project):
    root = Path(run_full_test.__file__).parent
    command = [sys.executable, "run_full_test.py", "--root", str(project), "--groups", "2"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "G2:" in result.stdout and "G3:" not in result.stdout


def test_groups_flag_rejects_an_impossible_number(project):
    with pytest.raises(SystemExit, match="step 5 failed.*from 2 to 25"):
        run_full_test.main(project, groups=26)
    with pytest.raises(SystemExit, match="step 5 failed"):
        run_full_test.main(project, groups=1)


def test_pipeline_stops_with_a_clear_message_when_input_is_empty(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(SystemExit, match="step 1 failed"):
        run_full_test.main(tmp_path)
