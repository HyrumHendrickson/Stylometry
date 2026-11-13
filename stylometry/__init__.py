"""Stability-based stylometry.

The package implements a five-step pipeline: per-text word frequencies, per-author
stable-word selection, cross-author shared stable words, pairwise text distances, and
a statistical report. The command-line programs in ``programs/`` are thin wrappers
around the functions defined here.
"""

__version__ = "0.1.0"
