# Stylometry

Authorship attribution from the words each author uses at a *consistent* rate.

Instead of starting from a hand-picked list of function words, this project counts every
word, finds the words each author uses steadily from one text to the next, keeps the words
that are steady for **all** authors, and measures how far apart every pair of texts is in
that shared feature space. A report shows how tightly each author's texts cluster compared
with how far they sit from every other author's.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Add your corpus: one folder per author, plain-text files inside.
#    input/
#    ├── Jane Austen/
#    │   ├── chapter_01.txt
#    │   └── chapter_02.txt
#    └── Mark Twain/
#        ├── sample_a.txt
#        └── sample_b.txt

# 2. Run everything.
python run_full_test.py
```

The report is written to `results/stylometry_report.png`. Each step can also be run on its
own, for example `python -m programs.step3_shared_words` (always from the repository root).

### Preparing the corpus

* Every author needs **at least 2** texts; **3 or more** (ideally many more) are recommended
  and fewer than `corpus.min_texts_per_author` triggers a warning.
* Texts should be comparable in length. Long works should be split into similarly sized
  chunks (for example, 5,000 words each) *before* being placed in `input/`; the pipeline
  never splits or edits text.
* Remove boilerplate (licence headers, tables of contents) yourself. The pipeline reads
  files exactly as given and never modifies `input/`.
* Files are read as UTF-8 by default (`corpus.encoding` in `config/config.toml`).

## Repository layout

| Path | Purpose |
|---|---|
| `input/` | Your corpus, `input/<author>/*.txt`. Never written to. |
| `temp/` | Intermediate data passed between steps. Never deleted, overwritten on rerun. |
| `results/` | Final report image and statistics table. |
| `config/config.toml` | Tunable parameters (all defaults shown). |
| `programs/` | One program per step (`step1_...` to `step5_...`). |
| `stylometry/` | Library with the actual implementation; the programs are thin wrappers. |
| `run_full_test.py` | Runs every program in `programs/` in order and shows the results. |
| `tests/` | `pytest` suite using synthetic corpora. |

## The five steps

| Step | Program | Reads | Writes (in `temp/` unless noted) |
|---|---|---|---|
| 1 | `step1_word_frequencies` | `input/` | `word_frequencies.csv`, `text_totals.csv` |
| 2 | `step2_stable_words` | step 1 | `stable_words_by_author.csv` |
| 3 | `step3_shared_words` | step 2 | `shared_stable_words.csv`, `shared_stable_words_meta.json` |
| 4 | `step4_distance_matrix` | steps 1, 3 | `feature_vectors.csv`, `word_weights.csv`, `distance_matrix.csv` |
| 5 | `step5_report` | step 4 | `results/stylometry_report.png`, `results/distance_statistics.csv` |

If a step fails, `run_full_test.py` stops and reports which one.

## Methodology

Notation: there are $A$ authors and $N$ texts. Text $i$ has $T_i$ word tokens, of which word
$w$ occurs $c_{iw}$ times.

### Step 1: relative frequencies

Each text is tokenized independently (lowercase letter sequences, internal apostrophes
kept) and

$$f_{iw} = \frac{c_{iw}}{T_i}.$$

### Step 2: stable words per author

For author $a$ with texts $i \in S_a$, a word is *eligible* only if $f_{iw} > 0$ for every
$i \in S_a$. Eligible words are ranked by the coefficient of variation

$$\mathrm{CV}_a(w) = \frac{s_a(w)}{\bar f_a(w)},$$

where $\bar f_a(w)$ and $s_a(w)$ are the mean and sample standard deviation of $f_{iw}$ over
$S_a$. Rank 1 is the lowest CV (ties: higher mean, then alphabetical). Because sampling noise
shrinks as counts grow, the most stable words are generally also frequent ones, as
expected.

### Step 3: words stable for every author

Let $r_a(w)$ be the rank of $w$ for author $a$. The top-$K$ list of author $a$ is
$L_a(K) = \{ w : r_a(w) \le K \}$, and the shared set is $\bigcap_a L_a(K)$. A word belongs to
the intersection exactly when $K \ge \max_a r_a(w)$, so every word has an *entry rank*
$\max_a r_a(w)$ and the shared set grows monotonically with $K$. The program chooses the
smallest $K$ whose shared set has at least `stability.target_shared_words` (default 30)
words. Words that are not eligible for some author (for example, character names and
other topical vocabulary) can never be shared, so this intersection acts as a topic filter in
place of a hand-made function-word list.

### Step 4: distance between texts

The shared words form $m$ features. Let $z_{iw} = \sqrt{f_{iw}}$. Relative frequencies are
approximately Poisson-distributed, so their variance grows with their mean, and the square
root makes the variance roughly independent of the mean.

Many words that are stable for *every* author are also used at similar rates by every author,
and such words carry no authorship signal. A metric that weights all words equally (as
Burrows's Delta does after z-scoring) lets those words add noise on the same scale as the
few words that do differ. The default metric therefore estimates, for each word $j$, how
much it separates authors relative to how much it varies within an author, using one-way
ANOVA mean squares:

$$\mathrm{MSB}_j = \frac{\sum_a n_a(\bar z_{aj} - \bar z_j)^2}{A-1}, \qquad
\mathrm{MSW}_j = \frac{\sum_a \sum_{i \in S_a} (z_{ij} - \bar z_{aj})^2}{N-A}, \qquad
F_j = \frac{\mathrm{MSB}_j}{\mathrm{MSW}_j}.$$

A word with $F_j \approx 1$ differs between authors no more than chance would suggest. The
weights and the **Fisher-weighted distance** are

$$w_j = \frac{\max(F_j - 1,\,0)}{\sum_k \max(F_k - 1,\,0)}, \qquad
d(u,v) = \sqrt{\sum_{j} w_j\,\frac{(z_{uj}-z_{vj})^2}{\mathrm{MSW}_j}}.$$

This is a diagonal form of linear discriminant analysis: words that separate authors dominate
the distance, uninformative words contribute almost nothing, and erratic words are
down-weighted by their within-author variance. If no word has $F_j > 1$ the weights fall
back to uniform. Other metrics can be selected in `config/config.toml`: Burrows's Delta
(`delta`), plain Euclidean distance on $\sqrt{f}$ (`euclidean`), and cosine distance
(`cosine`).

### Step 5: statistics and report

* **Within author.** For each author, the distances between all $\binom{n_a}{2}$ unordered
  pairs of their texts: mean and sample standard deviation.
* **Between authors.** For each pair of authors $(a, b)$, the $n_a n_b$ distances from texts
  of $a$ to texts of $b$: mean and sample standard deviation.
* **Leave-one-out attribution.** Each text is held out; weights and all standardization
  constants are refit on the remaining texts; the held-out text is assigned the author of its
  nearest neighbour. Accuracy is reported for the primary metric and the baseline
  (`distance.baseline_metric`, default Burrows's Delta).

The image contains four panels: mean distance with standard deviation for every author pair
(diagonal: within-author), per-author distance distributions, the $F$-ratio of each shared
word, and leave-one-out accuracy. `results/distance_statistics.csv` has the same numbers in
tabular form.

## Interpreting results and caveats

* **Selection uses labels.** Step 3's word selection and the Fisher weights use author
  labels. The leave-one-out accuracy refits the weights without the held-out text, but the
  word selection in steps 2 and 3 still saw it, so accuracy is mildly optimistic. For an
  unbiased estimate, keep some texts out of `input/` and classify them separately.
* **The within/between standard deviations are descriptive.** The distance matrix uses
  weights fitted on all texts, so within-author distances are somewhat smaller than they
  would be for unseen texts.
* **Few texts per author** make the variance estimates, and therefore the $F$-ratios and
  weights, noisy. Prefer many comparable-length texts.
* **Strict intersection.** Because shared words must be stable for *every* author, adding
  authors shrinks the pool. If the target cannot be met, step 3 stops with an explanatory
  message; lower `stability.target_shared_words` or supply longer or more texts.
* **Language.** No language-specific resources are used; the tokenizer handles any script
  whose words are Unicode letter sequences separated by non-letters (it is not suitable for
  languages written without word separators).

## Configuration

`config/config.toml` documents every option:

| Key | Default | Meaning |
|---|---|---|
| `corpus.file_extension` | `".txt"` | File extension read from each author folder |
| `corpus.encoding` | `"utf-8-sig"` | Text encoding |
| `corpus.min_texts_per_author` | `3` | Warn below this; error below 2 |
| `stability.target_shared_words` | `30` | Minimum size of the shared stable-word set |
| `distance.metric` | `"fisher"` | `fisher`, `delta`, `euclidean` or `cosine` |
| `distance.baseline_metric` | `"delta"` | Metric the report compares against |
| `distance.variance_floor` | `1e-12` | Numerical floor for within-author variance |

Both `run_full_test.py` and the individual programs accept `--root PATH` (use another
project directory), `--config PATH` and `-v/--verbose`.

## Development

```bash
pip install -e ".[dev]"
pytest              # unit tests and an end-to-end run on a synthetic corpus
ruff check . && ruff format --check .
mypy
```

The test suite generates synthetic corpora in a temporary directory, so no copyrighted
text is stored in the repository.

## License

MIT. See [LICENSE](LICENSE).
