"""Shared command-line plumbing for the step programs."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from stylometry.config import PROJECT_ROOT, Config, ProjectPaths, load_config
from stylometry.errors import StylometryError

LOGGER = logging.getLogger(__name__)

StepFunction = Callable[[ProjectPaths, Config], None]


def parse_args(description: str | None, argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the options shared by every step program."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="project root containing input/, temp/, results/ and config/ (default: this repo)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="configuration file (default: <root>/config/config.toml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


def execute(args: argparse.Namespace, step: StepFunction) -> int:
    """Run one step, converting expected failures into an error message and exit code 1."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    paths = ProjectPaths(args.root.resolve())
    try:
        config = load_config(args.config or paths.config_file)
        paths.ensure_output_dirs()
        step(paths, config)
    except StylometryError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0
