# Method

## 1. Sanitisation and bounds

`pipeline/sanitize.py` runs on every narrative before it is featurised or displayed.
Script blocks are removed with their contents, remaining tags are stripped, control
characters are dropped, whitespace is collapsed, text is bounded at 5,000 characters with
a visible `[truncated]` marker, and the display form is HTML-escaped. Empty, whitespace,
`None` and NaN narratives all become the empty string with `has_narrative=False`. The
featuriser uses the unescaped bounded text.

## 2. Duplicate detection

Exact duplicates share the SHA-256 of the normalised text (lower case, punctuation
removed, whitespace collapsed). Near duplicates share 5-word shingles with Jaccard
similarity at or above 0.8; candidate pairs come from a shingle posting list, so only
texts that share at least one shingle are compared. Both kinds are merged into groups by
union-find. Empty texts are never grouped.

## 3. Chronology-aware split

Rows are ordered by `date_received`; the first 60 percent are train, the next 20 percent
validation, the rest test. Then each duplicate group is assigned as a whole to the split
of its earliest member. That direction is deliberate: a row can move to an earlier split,
never a later one, so the evaluation splits never contain a narrative that was also seen
in training. The number of moved rows is reported in `metrics.json`.

## 4. Leakage guard

`pipeline/leakage.py` validates the requested structured features against the target and
returns a frozen `FeatureSpec`. It refuses the target itself, the child category that
encodes it (`sub_product` for `product`, `sub_issue` for `issue`), and the post-event
fields `company_response`, `timely` and `consumer_disputed`, which only exist after the
complaint was routed and answered. Anything outside `state`, `submitted_via`, `company` is
also refused, so a new column cannot slip in without a code change.

## 5. Features and model

TF-IDF (word 1-2 grams, at most 5,000 terms, `min_df=2`, sublinear tf) on the narrative
plus one-hot on the structured fields, fitted on train only, horizontally stacked.
Logistic regression with `class_weight="balanced"`, `C=1.0`, lbfgs, seed 42. Persistence
is numpy `.npz` for arrays and JSON for vocabulary, categories and classes. A test checks
that the round-tripped model produces the same probabilities.

## 6. Abstention and routing

Confidence is the maximum class probability. The coverage/accuracy curve is computed on a
grid of thresholds from 0.00 to 0.95. The threshold is chosen on the validation split as
the smallest value whose covered accuracy meets the operating target (0.98 in
`configs/default.yaml`) with at least 20 percent coverage; it is then applied unchanged to
test. Items below the threshold are routed to `human_review`, the rest to `auto`.

The 0.98 target is a policy setting. It was chosen so that the demo routes a visible
number of items; a deployment would set it from its own error tolerance.

## 7. Reviewer override log

`pipeline/review.py` is append-only JSONL. Each `override` or `confirm` writes one event
with an id, UTC timestamp, complaint id, reviewer id, from/to labels and a sanitised
reason. Overrides must change the label; unknown complaint ids are rejected when a known
set is supplied. The demo uses a simulated reviewer that applies the fixture labels to the
routed items, which is stated in the log summary.

## 8. Trends

Monthly counts by product and by product x issue. A theme is flagged when its mean over
the last 3 months is at least 2.0 times its mean over the preceding 12 months and the
recent total is at least 15. The report carries the note that counts are allegations, not
findings, and not a company ranking.

## 9. Topic landscape

TruncatedSVD with two components on the train TF-IDF matrix, at most 500 points. The
output and the figure are labelled "schematic navigation only; axes are SVD directions
with no unit". On the fixture the two components explain about 4 and 6 percent of the
variance (computed by `make demo`, seed 42), which is why it is a navigation aid and not a
metric.

## 10. Summariser

`TemplateSummariser` quotes the first two sentences of the sanitised narrative and cites
their character offsets. `SummariserBackend` is the adapter protocol. No provider is wired
and the smoke, demo and test paths need no API key.

## Settings

All settings are in `configs/default.yaml`. None of them is taken from a paper; this is an
original project.
