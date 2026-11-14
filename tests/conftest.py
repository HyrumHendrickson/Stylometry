"""Shared fixtures."""

import pytest

from tests.synthetic import write_corpus


@pytest.fixture
def project(tmp_path):
    """A project folder whose input/ holds a synthetic 4-author corpus."""
    write_corpus(tmp_path / "input")
    return tmp_path
