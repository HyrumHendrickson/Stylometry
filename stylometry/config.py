"""Project paths and configuration loading."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from stylometry.distance import METRICS
from stylometry.errors import StylometryError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

T = TypeVar("T")


@dataclass(frozen=True)
class ProjectPaths:
    """Locations of every directory and intermediate artifact in a project tree."""

    root: Path = PROJECT_ROOT

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def temp_dir(self) -> Path:
        return self.root / "temp"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def config_file(self) -> Path:
        return self.root / "config" / "config.toml"

    # Step 1 outputs.
    @property
    def word_frequencies(self) -> Path:
        return self.temp_dir / "word_frequencies.csv"

    @property
    def text_totals(self) -> Path:
        return self.temp_dir / "text_totals.csv"

    # Step 2 output.
    @property
    def stable_words(self) -> Path:
        return self.temp_dir / "stable_words_by_author.csv"

    # Step 3 outputs.
    @property
    def shared_words(self) -> Path:
        return self.temp_dir / "shared_stable_words.csv"

    @property
    def shared_words_meta(self) -> Path:
        return self.temp_dir / "shared_stable_words_meta.json"

    # Step 4 outputs.
    @property
    def feature_vectors(self) -> Path:
        return self.temp_dir / "feature_vectors.csv"

    @property
    def word_weights(self) -> Path:
        return self.temp_dir / "word_weights.csv"

    @property
    def distance_matrix(self) -> Path:
        return self.temp_dir / "distance_matrix.csv"

    # Step 5 outputs.
    @property
    def report_image(self) -> Path:
        return self.results_dir / "stylometry_report.png"

    @property
    def report_statistics(self) -> Path:
        return self.results_dir / "distance_statistics.csv"

    def ensure_output_dirs(self) -> None:
        """Create ``temp/`` and ``results/`` if they do not exist."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CorpusConfig:
    """How the ``input/`` folder is read."""

    file_extension: str = ".txt"
    encoding: str = "utf-8-sig"
    min_texts_per_author: int = 3


@dataclass(frozen=True)
class StabilityConfig:
    """How stable words are selected."""

    target_shared_words: int = 30


@dataclass(frozen=True)
class DistanceConfig:
    """Which distance metrics are used."""

    metric: str = "fisher"
    baseline_metric: str = "delta"
    variance_floor: float = 1e-12


@dataclass(frozen=True)
class Config:
    """Complete pipeline configuration."""

    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)

    def __post_init__(self) -> None:
        if self.corpus.min_texts_per_author < 2:
            raise StylometryError("corpus.min_texts_per_author must be at least 2.")
        if self.stability.target_shared_words < 1:
            raise StylometryError("stability.target_shared_words must be at least 1.")
        for key in ("metric", "baseline_metric"):
            value = getattr(self.distance, key)
            if value not in METRICS:
                raise StylometryError(
                    f"distance.{key} = {value!r} is not one of {', '.join(METRICS)}."
                )
        if self.distance.variance_floor <= 0:
            raise StylometryError("distance.variance_floor must be positive.")


def _build_section(cls: type[T], values: Mapping[str, Any], name: str) -> T:
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise StylometryError(f"Unknown key(s) in [{name}]: {', '.join(unknown)}.")
    return cls(**values)


def load_config(path: Path) -> Config:
    """Load and validate a TOML configuration file.

    Missing sections and keys fall back to their defaults; unknown keys are rejected so
    that typos do not pass silently.
    """
    if not path.is_file():
        raise StylometryError(f"Configuration file not found: {path}")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise StylometryError(f"Invalid TOML in {path}: {exc}") from exc

    sections = {"corpus": CorpusConfig, "stability": StabilityConfig, "distance": DistanceConfig}
    unknown_sections = sorted(set(data) - set(sections))
    if unknown_sections:
        raise StylometryError(f"Unknown section(s) in {path}: {', '.join(unknown_sections)}.")
    built: dict[str, Any] = {
        name: _build_section(cls, data.get(name, {}), name) for name, cls in sections.items()
    }
    return Config(**built)
