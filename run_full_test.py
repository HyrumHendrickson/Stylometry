"""Run the five steps in order.

Usage: python run_full_test.py [--root PATH] [--groups N]

--groups N sorts the texts into N proposed authors (default: NUMBER_OF_GROUPS in
programs/common.py, which chooses automatically when it is None).
"""

import sys
from pathlib import Path

from programs import (
    step1_word_frequencies,
    step2_stable_words,
    step3_distance_matrix,
    step4_report,
    step5_groups,
)
from programs.common import parse_command_line

STEPS = [
    step1_word_frequencies,
    step2_stable_words,
    step3_distance_matrix,
    step4_report,
    step5_groups,
]


def main(root: Path, groups: int | None = None) -> None:
    for number, step in enumerate(STEPS, start=1):
        print(f"\n=== step {number}/{len(STEPS)}: {step.__name__.split('.')[-1]} ===")
        options = {"groups": groups} if step is step5_groups else {}
        try:
            step.run(root, **options)
        except SystemExit as error:
            sys.exit(f"step {number} failed: {error}")


if __name__ == "__main__":
    args = parse_command_line(__doc__, groups=True)
    main(args.root, args.groups)
