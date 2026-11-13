"""Exception types for the stylometry pipeline."""

from __future__ import annotations


class StylometryError(Exception):
    """A user-facing, expected failure (bad input, missing intermediate file, ...).

    The command-line programs report these as a single error line instead of a traceback.
    """
