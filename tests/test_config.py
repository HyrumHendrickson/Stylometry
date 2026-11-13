"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from stylometry.config import PROJECT_ROOT, Config, load_config
from stylometry.errors import StylometryError


def test_shipped_config_matches_dataclass_defaults() -> None:
    assert load_config(PROJECT_ROOT / "config" / "config.toml") == Config()


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_partial_config_uses_defaults(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, "[stability]\ntarget_shared_words = 12\n"))
    assert config.stability.target_shared_words == 12
    assert config.distance.metric == "fisher"


@pytest.mark.parametrize(
    "text",
    [
        "[stability]\ntarget_shared_wrods = 3\n",
        "[unknown]\nx = 1\n",
        '[distance]\nmetric = "manhattan"\n',
        "[corpus]\nmin_texts_per_author = 1\n",
        "[stability]\ntarget_shared_words = 0\n",
        "not toml [",
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(StylometryError):
        load_config(_write(tmp_path, text))


def test_missing_config(tmp_path: Path) -> None:
    with pytest.raises(StylometryError, match="not found"):
        load_config(tmp_path / "nope.toml")
