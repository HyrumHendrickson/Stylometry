"""Run the five steps in order. Usage: python run_full_test.py [--root PATH]"""

import sys
from pathlib import Path

from programs import (
    step1_word_frequencies,
    step2_stable_words,
    step3_shared_words,
    step4_distance_matrix,
    step5_report,
)
from programs.common import root_from_command_line

STEPS = [
    step1_word_frequencies,
    step2_stable_words,
    step3_shared_words,
    step4_distance_matrix,
    step5_report,
]


def main(root: Path) -> None:
    for number, step in enumerate(STEPS, start=1):
        print(f"\n=== step {number}/{len(STEPS)}: {step.__name__.split('.')[-1]} ===")
        try:
            step.run(root)
        except SystemExit as error:
            sys.exit(f"step {number} failed: {error}")


if __name__ == "__main__":
    main(root_from_command_line(__doc__))
