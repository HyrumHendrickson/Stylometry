# Stylometry

Authorship analysis from the words each author uses at a *consistent* rate. Every word is
counted, the words each author uses steadily are found, and the words steady for **all**
authors are kept. The texts are then compared on those words, and the report shows which
words tell authors apart and how far apart authors and individual texts are.

## Quick start

```bash
pip install -r requirements.txt

# Put your corpus in input/, one folder per author, plain-text files inside:
#   input/Jane Austen/chapter_01.txt
#   input/Mark Twain/sample_a.txt

python run_full_test.py          # runs all five steps
```

The report is `results/stylometry_report.png`. Any step can also be run alone, from the
repository root, for example `python -m programs.step3_shared_words`. All commands accept
`--root PATH` to analyze a different project folder.

Every author needs at least 2 texts; 3 or more is recommended. Texts should be of similar
length (split long works into equal chunks first) and free of boilerplate. `input/` is never
modified.

## Steps

| Step | Program | Reads | Writes |
|---|---|---|---|
| 1 | `step1_word_frequencies` | `input/` | `temp/word_frequencies.csv` |
| 2 | `step2_stable_words` | step 1 | `temp/stable_words_by_author.csv` |
| 3 | `step3_shared_words` | step 2 | `temp/shared_stable_words.csv` |
| 4 | `step4_distance_matrix` | steps 1, 3 | `temp/word_power.csv`, `temp/distance_matrix.csv` |
| 5 | `step5_report` | step 4 | `results/stylometry_report.png` |

Settings (file extension, encoding, minimum number of shared words) are the constants at the
top of `programs/common.py`.

## Method

Text $i$ has $T_i$ words, of which word $w$ occurs $c_{iw}$ times.

1. **Frequencies.** $f_{iw} = c_{iw} / T_i$.
2. **Stable words per author.** A word qualifies for an author only if it appears in every
   one of that author's texts. Qualifying words are ranked by the coefficient of variation
   $s/\bar f$ of their frequency across the author's texts (lowest = rank 1).
3. **Shared words.** A word is in the intersection of every author's top-$K$ lists once
   $K$ reaches its worst rank among the authors. The smallest $K$ that gives at least 30
   words is used. Topical words (names, places) are not stable for every author, so they drop out.
4. **Word power and distance.** With $z_{iw} = \sqrt{f_{iw}}$, each shared word gets a
   one-way ANOVA $F$-ratio: variance between author means over variance within authors.
   $F \approx 1$ means the word does not separate authors. Words are weighted by
   $w_j = \max(F_j - 1, 0) / \sum_k \max(F_k - 1, 0)$ and the distance between texts $u, v$ is

   $$d(u,v) = \sqrt{\sum_j w_j \frac{(z_{uj} - z_{vj})^2}{\mathrm{MSW}_j}}$$

   where $\mathrm{MSW}_j$ is the within-author variance of word $j$.
5. **Report.** Three panels:
   * **Authors:** mean distance ± standard deviation between the texts of each pair of
     authors; the diagonal is the distance between texts by the same author.
   * **Words:** every shared word with its $F$-ratio, best first.
   * **Texts:** the distance between every pair of texts (lines separate authors). The same
     numbers are in `temp/distance_matrix.csv`.

## Caveats

* Word selection and the weights use the author labels, so within-author distances are
  smaller than they would be for unseen texts.
* With few texts per author the variance estimates, and so the $F$-ratios, are noisy.
* The shared set must be stable for *every* author, so adding authors shrinks it. If there
  are too few words, step 3 stops and says so.
* The tokenizer treats runs of Unicode letters as words, so it does not suit languages
  written without spaces.

## Development

```bash
pip install pytest ruff
pytest
ruff check .
```

Tests use synthetic corpora, so no copyrighted text is stored in the repository.

## License

MIT. See [LICENSE](LICENSE).
