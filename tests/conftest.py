"""Shared fixtures."""

import pytest

from tests.synthetic import write_corpus


@pytest.fixture
def project(tmp_path):
    """A project folder whose input/ holds 25 synthetic texts by 5 authors, all in one folder."""
    write_corpus(tmp_path / "input", n_authors=5, signal=0.5, informative=tuple(range(40)))
    return tmp_path
