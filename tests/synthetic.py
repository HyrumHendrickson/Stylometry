"""Synthetic corpus generator used by the tests.

Every author draws words from the same 40 common words, but a few of them ("informative"
words) have author-specific rates. Authors also have private topical words and share a long
tail of filler words. Each text gets extra multiplicative jitter so that within-author
variability exceeds pure Poisson noise, like real prose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COMMON_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "he",
    "for", "it", "with", "as", "his", "on", "be", "at", "by", "i",
    "this", "had", "not", "are", "but", "from", "or", "have", "an", "they",
    "which", "one", "you", "were", "her", "all", "she", "there", "would", "their",
]  # fmt: skip


def letters(index: int, prefix: str = "") -> str:
    """Encode an integer as a digit-free string so the tokenizer keeps it whole."""
    chars: list[str] = []
    n = index
    while True:
        chars.append(chr(ord("a") + n % 26))
        n //= 26
        if n == 0:
            break
    return prefix + "".join(chars)


def write_corpus(
    input_dir: Path,
    *,
    n_authors: int = 4,
    n_texts: int = 5,
    n_tokens: int = 5000,
    signal: float = 0.6,
    jitter: float = 0.15,
    seed: int = 0,
    informative: tuple[int, ...] = (3, 11, 19),
) -> dict[str, list[Path]]:
    """Write ``input_dir/<author>/text_<k>.txt`` files and return the paths by author."""
    rng = np.random.default_rng(seed)
    n_common, n_topic, n_filler = len(COMMON_WORDS), 15, 150
    base = 1.0 / (np.arange(n_common) + 2.0) ** 0.8
    base /= base.sum()
    filler = 1.0 / (np.arange(n_filler) + 1.0)
    filler /= filler.sum()
    filler_words = [letters(i, "fil") for i in range(n_filler)]

    written: dict[str, list[Path]] = {}
    for a in range(n_authors):
        name = f"author_{letters(a)}"
        rates = base.copy()
        for index in informative:
            rates[index] *= np.exp(signal * rng.normal())
        rates /= rates.sum()
        topic_words = [letters(i, f"zq{letters(a)}q") for i in range(n_topic)]
        topic = np.full(n_topic, 1.0 / n_topic)
        words = COMMON_WORDS + topic_words + filler_words
        probs = np.concatenate([0.7 * rates, 0.2 * topic, 0.1 * filler])

        folder = input_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        written[name] = []
        for k in range(n_texts):
            noisy = probs * np.exp(jitter * rng.normal(size=probs.size))
            noisy /= noisy.sum()
            counts = rng.multinomial(n_tokens, noisy)
            tokens = np.repeat(np.array(words), counts)
            rng.shuffle(tokens)
            path = folder / f"text_{k}.txt"
            path.write_text(" ".join(tokens.tolist()) + "\n", encoding="utf-8")
            written[name].append(path)
    return written
