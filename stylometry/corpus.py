"""Discovery and loading of the user-curated ``input/`` corpus."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from stylometry.errors import StylometryError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextRecord:
    """One writing sample.

    Attributes:
        author: Name of the author folder.
        text_id: Unique identifier, ``"<author>/<filename>"``.
        path: Location of the file on disk.
    """

    author: str
    text_id: str
    path: Path


def discover_texts(input_dir: Path, extension: str, min_texts_per_author: int) -> list[TextRecord]:
    """List every text in ``input_dir/<author>/*<extension>``.

    The corpus is never modified. Hidden folders and loose files directly inside
    ``input_dir`` are ignored. Authors without any matching file are skipped with a
    warning.

    Raises:
        StylometryError: If the folder is missing, fewer than two authors have texts, or an
            author has fewer than two texts (within-author variability is undefined).
    """
    if not input_dir.is_dir():
        raise StylometryError(f"Input folder not found: {input_dir}")

    suffix = extension.lower()
    records: list[TextRecord] = []
    counts: dict[str, int] = {}
    for author_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if author_dir.name.startswith("."):
            continue
        files = sorted(
            f for f in author_dir.iterdir() if f.is_file() and f.name.lower().endswith(suffix)
        )
        if not files:
            LOGGER.warning(
                "Author folder %r contains no %s files; skipping.", author_dir.name, extension
            )
            continue
        counts[author_dir.name] = len(files)
        records.extend(TextRecord(author_dir.name, f"{author_dir.name}/{f.name}", f) for f in files)

    if len(counts) < 2:
        raise StylometryError(
            f"At least two authors with texts are required; found {len(counts)} in {input_dir}."
        )
    for author, n_texts in counts.items():
        if n_texts < 2:
            raise StylometryError(
                f"Author {author!r} has {n_texts} text; at least 2 are required to measure "
                "within-author consistency."
            )
        if n_texts < min_texts_per_author:
            LOGGER.warning(
                "Author %r has only %d texts (< %d); statistics will be unreliable.",
                author,
                n_texts,
                min_texts_per_author,
            )
    return records


def read_text(path: Path, encoding: str) -> str:
    """Read one text file, reporting decoding problems with the offending path."""
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise StylometryError(
            f"Cannot decode {path} as {encoding}. Convert the file or change corpus.encoding "
            "in config/config.toml."
        ) from exc
