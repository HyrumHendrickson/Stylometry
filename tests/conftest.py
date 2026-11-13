"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.synthetic import write_corpus

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """An empty project tree with the repository's default configuration."""
    root = tmp_path / "project"
    (root / "input").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "config", root / "config")
    return root


@pytest.fixture
def synthetic_project(project_root: Path) -> Path:
    """A project tree whose ``input/`` holds a 4-author synthetic corpus."""
    write_corpus(project_root / "input")
    return project_root
