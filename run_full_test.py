#!/usr/bin/env python3
"""Run every program in ``programs/`` in order and present the results.

Programs are discovered by the pattern ``programs/step<N>_*.py`` and run in numeric order,
each in its own interpreter process. The first failure stops the run.

Usage::

    python run_full_test.py [--root PATH] [--config PATH] [-v]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROGRAMS_DIR = REPO_ROOT / "programs"
_STEP_PATTERN = re.compile(r"^step(\d+)_\w+$")


def discover_programs(programs_dir: Path = PROGRAMS_DIR) -> list[str]:
    """Module names of all step programs, sorted by step number."""
    found: list[tuple[int, str]] = []
    for path in programs_dir.glob("step*.py"):
        match = _STEP_PATTERN.match(path.stem)
        if match:
            found.append((int(match.group(1)), path.stem))
    return [name for _, name in sorted(found)]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full pipeline. Returns the exit code of the first failing step, or 0."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="project root to analyze")
    parser.add_argument("--config", type=Path, default=None, help="configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    options = ["--root", str(args.root.resolve())]
    if args.config is not None:
        options += ["--config", str(args.config.resolve())]
    if args.verbose:
        options.append("--verbose")

    programs = discover_programs()
    if not programs:
        print(f"No step programs found in {PROGRAMS_DIR}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    for number, name in enumerate(programs, start=1):
        print(f"\n=== [{number}/{len(programs)}] {name} ===", flush=True)
        step_started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", f"programs.{name}", *options], cwd=REPO_ROOT, check=False
        )
        if result.returncode != 0:
            print(f"\nFAILED at {name} (exit code {result.returncode}).", file=sys.stderr)
            return result.returncode
        print(f"({time.perf_counter() - step_started:.1f}s)", flush=True)

    image = args.root.resolve() / "results" / "stylometry_report.png"
    print(f"\nAll {len(programs)} steps finished in {time.perf_counter() - started:.1f}s.")
    print(f"Report image: {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
