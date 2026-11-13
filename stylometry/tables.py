"""Reading and writing the CSV tables that connect the pipeline steps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stylometry.errors import StylometryError


def read_table(path: Path, producer: str, **kwargs: object) -> pd.DataFrame:
    """Read a CSV written by an earlier step.

    Automatic NA conversion is disabled: words such as "null", "nan" and "na" are valid
    tokens and must not be turned into missing values.

    Args:
        path: File to read.
        producer: Name of the program that creates the file, used in the error message.
        **kwargs: Extra keyword arguments forwarded to ``pandas.read_csv``.
    """
    if not path.is_file():
        raise StylometryError(f"Missing {path.name}. Run {producer} first.")
    return pd.read_csv(path, keep_default_na=False, na_filter=False, **kwargs)  # type: ignore[call-overload,no-any-return]


def write_table(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write a DataFrame to CSV, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
