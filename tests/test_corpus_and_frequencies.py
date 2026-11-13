"""Tests for tokenization, corpus discovery and step 1 frequency tables."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from stylometry.corpus import TextRecord, discover_texts
from stylometry.errors import StylometryError
from stylometry.frequencies import build_frequency_tables
from stylometry.tokenizer import tokenize


def test_tokenize_basic() -> None:
    assert tokenize("The Dog's bone, don\u2019t 42 well-known_x") == [
        "the",
        "dog's",
        "bone",
        "don't",
        "well",
        "known",
        "x",
    ]


def test_tokenize_unicode_and_empty() -> None:
    assert tokenize("Café naïve") == ["café", "naïve"]
    assert tokenize("123 ... --") == []


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_frequency_tables(tmp_path: Path) -> None:
    first = _write(tmp_path / "ann" / "one.txt", "a a b c")
    second = _write(tmp_path / "bob" / "two.txt", "b b b d")
    records = [TextRecord("ann", "ann/one.txt", first), TextRecord("bob", "bob/two.txt", second)]
    counts, totals = build_frequency_tables(records, "utf-8")

    assert totals.set_index("text_id")["total_tokens"].to_dict() == {
        "ann/one.txt": 4,
        "bob/two.txt": 4,
    }
    lookup = counts.set_index(["text_id", "word"])["frequency"]
    assert lookup[("ann/one.txt", "a")] == pytest.approx(0.5)
    assert lookup[("bob/two.txt", "b")] == pytest.approx(0.75)
    sums = counts.groupby("text_id")["frequency"].sum()
    assert sums.to_numpy() == pytest.approx([1.0, 1.0])


def test_text_without_words_is_rejected(tmp_path: Path) -> None:
    empty = _write(tmp_path / "ann" / "blank.txt", "1234 !!!")
    with pytest.raises(StylometryError, match="no word tokens"):
        build_frequency_tables([TextRecord("ann", "ann/blank.txt", empty)], "utf-8")


def test_discover_texts_ignores_noise(tmp_path: Path) -> None:
    for author in ("ann", "bob"):
        _write(tmp_path / author / "1.txt", "x")
        _write(tmp_path / author / "2.TXT", "x")
        _write(tmp_path / author / "notes.md", "x")
    _write(tmp_path / ".hidden" / "1.txt", "x")
    _write(tmp_path / "loose.txt", "x")
    records = discover_texts(tmp_path, ".txt", min_texts_per_author=2)
    assert [r.text_id for r in records] == ["ann/1.txt", "ann/2.TXT", "bob/1.txt", "bob/2.TXT"]


def test_discover_texts_requires_two_authors(tmp_path: Path) -> None:
    _write(tmp_path / "ann" / "1.txt", "x")
    _write(tmp_path / "ann" / "2.txt", "x")
    with pytest.raises(StylometryError, match="two authors"):
        discover_texts(tmp_path, ".txt", 2)


def test_discover_texts_requires_two_texts_per_author(tmp_path: Path) -> None:
    for name in ("1.txt", "2.txt"):
        _write(tmp_path / "ann" / name, "x")
    _write(tmp_path / "bob" / "1.txt", "x")
    with pytest.raises(StylometryError, match="bob"):
        discover_texts(tmp_path, ".txt", 2)


def test_discover_texts_warns_below_minimum(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    for author in ("ann", "bob"):
        for name in ("1.txt", "2.txt"):
            _write(tmp_path / author / name, "x")
    with caplog.at_level(logging.WARNING):
        discover_texts(tmp_path, ".txt", min_texts_per_author=3)
    assert "unreliable" in caplog.text


def test_missing_input_folder(tmp_path: Path) -> None:
    with pytest.raises(StylometryError, match="not found"):
        discover_texts(tmp_path / "nope", ".txt", 2)
