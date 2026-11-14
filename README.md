# Stylometry

Compare writing samples by the words they use. Put any number of text files in `input/`; the
report shows how far apart every pair of texts is. Texts by the same author come out close
together, so they appear as light blocks along the diagonal when the files are named so that
related texts sit next to each other.

No author labels are used anywhere.

## Quick start

```bash
pip install -r requirements.txt

# Put plain-text files directly in input/ (subfolders are ignored), for example:
#   input/twain_huckleberry_finn_1.txt
#   input/twain_huckleberry_finn_2.txt
#   input/poe_tales_1.txt

python run_full_test.py          # runs all five steps
python run_full_test.py --groups 5   # same, sorting the texts into 5 groups
```

The report is `results/stylometry_report.png`. Any step can also be run alone, from the
repository root, for example `python -m programs.step3_distance_matrix`. All commands accept
`--root PATH` to analyze a different project folder. `run_full_test.py` and
`programs.step5_groups` also accept `--groups N`, the number of proposed authors to sort the
texts into (from 2 to the number of texts).

Texts should be of similar length (split long works into equal chunks of 5,000 to 10,000
words first), free of boilerplate, and there should be at least three of them; many more is
better. The matrix lists texts in file-name order, so start each name with the author or work
you want grouped together. `input/` is never modified.

## Steps

| Step | Program | Reads | Writes |
|---|---|---|---|
| 1 | `step1_word_frequencies` | `input/` | `temp/word_frequencies.csv` |
| 2 | `step2_stable_words` | step 1 | `temp/stable_words.csv` |
| 3 | `step3_distance_matrix` | steps 1, 2 | `temp/distance_matrix.csv` |
| 4 | `step4_report` | steps 2, 3 | `results/stylometry_report.png`, `results/words_used.txt` |
| 5 | `step5_groups` | step 3 | `results/proposed_groups.png` |

Settings (file extension, encoding, `MIN_SHARE_OF_TEXTS`, `NUMBER_OF_GROUPS`) are the
constants at the top of `programs/common.py`.

## Method

Text $i$ has $T_i$ words, of which word $w$ occurs $c_{iw}$ times.

1. **Frequencies.** $f_{iw} = c_{iw} / T_i$.
2. **Words used.** A word is used if it appears in at least `MIN_SHARE_OF_TEXTS` of the texts
   (default 0.25). Words found in only a few texts are topical (character names, places) and
   would separate texts by subject rather than style. Lowering the value admits more words.
3. **Distance.** With $z_{iw} = \sqrt{f_{iw}}$ (the square root evens out the variance of
   common and rare words), each word is standardized across all texts:
   $\tilde z_{iw} = (z_{iw} - \bar z_w)/s_w$. The distance between texts $u$ and $v$ is the
   cosine distance $1 - \cos(\tilde z_u, \tilde z_v)$, which compares the pattern of a text's
   high and low words and not how extreme they are.
4. **Report.** One matrix of the distance between every pair of texts, in file-name order.
   Light cells are similar, dark cells are different. The numbers are in
   `temp/distance_matrix.csv`. `results/words_used.txt` lists every word the distances are
   based on, one per line, the words found in the most texts first (`temp/stable_words.csv`
   also gives the share of texts for each).
5. **Proposed authors.** The texts are clustered by average linkage: the two closest groups
   are merged repeatedly, where the distance between groups is the mean distance between their
   texts. The number of groups is `--groups N` if given, otherwise `NUMBER_OF_GROUPS`, otherwise
   (if that is `None`) the number from 2 to 30 with the best silhouette score (how much closer
   a text is to its own group than to the nearest other group).
   `results/proposed_groups.png` lists the texts of each group, one column per group
   (G1, G2, ...).

## Checking the result

The method never sees authors, but if you name files `<author>_<work>_<number>.txt` you can
score it afterwards:

```bash
python -m programs.check_names
```

It reads the author from the file name and prints how often a text's nearest text from a
different work has the same author, the chance that a same-author pair is closer than a
different-author pair (AUC), and how the proposed groups split by author. Use it to compare
settings; it is not part of the pipeline.

## Caveats

* The method has no author labels, so it cannot say which words separate authors, and the
  groups are proposals: they have numbers, not names, and the automatic number of groups can
  be wrong. Set `NUMBER_OF_GROUPS` if you know it.
* A word used by only one author passes `MIN_SHARE_OF_TEXTS` if that author wrote at least
  that share of the texts. With four equally sized authors and the default 0.25, character
  names can slip in; raise the value if the authors are few.
* Every word counts equally, so the method assumes authors differ across many words, as real
  prose does. If the difference sits in only a few words among hundreds, it is diluted and
  the groups come out wrong.
* Chunks of the same book are closer to each other than to other books by the same author,
  so expect blocks within blocks.
* Distances depend on the whole collection (words are standardized across all texts), so
  adding or removing texts changes every number.
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
