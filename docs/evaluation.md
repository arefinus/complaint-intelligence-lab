# Evaluation

## Protocol

1. Validate the frame; sanitise narratives.
2. Group exact and near-duplicate narratives.
3. Split chronologically 60/20/20; groups follow their earliest member.
4. Fit features and model on train only.
5. On validation: compute the coverage/accuracy curve and choose the threshold.
6. On test, once: macro-F1, per-class recall, confusion matrix with explicit class order,
   coverage and accuracy at the selected threshold and along the grid, disagreement
   counts by true class, and the routing decision per row.
7. Simulated review of routed items (fixture labels) into the audit log.

## Results on the fixture

Computed by `make demo` on synthetic fixture v1, seed 42, not a benchmark result.
Evidence label: `demo`. Source file: `examples/output/metrics.json`.

Split counts: train 2,726, valid 653, test 621 (365 rows moved by the duplicate rule).

Test, all 621 rows: accuracy 0.882, macro-F1 0.865, 73 disagreements.

Per-class recall (all test rows):

| Class | Support | Recall |
|---|---|---|
| Checking or savings account | 66 | 0.894 |
| Credit card | 69 | 0.841 |
| Credit reporting | 189 | 0.894 |
| Debt collection | 88 | 0.909 |
| Money transfer | 62 | 0.855 |
| Mortgage | 59 | 0.814 |
| Student loan | 40 | 0.925 |
| Vehicle loan or lease | 48 | 0.917 |

Abstention: threshold 0.30 selected on validation (target 0.98 covered accuracy, minimum
coverage 0.2). On test: coverage 0.890, 553 rows auto, 68 routed to human review;
accuracy on covered rows 0.982, macro-F1 on covered rows 0.978.

Simulated review of the 68 routed rows: 63 overrides, 5 confirmations, 68 audit events.

Trend rule: one theme flagged, `Money transfer / Fraud or scam`, recent mean 15.3 per
month against a baseline of 3.9 per month, ratio 3.9, 46 complaints in the last three
months. This is the increase the fixture generator injects, so the rule found what it
was built to find; that is a check of the code, not evidence about any real market.

Topic landscape: two SVD components explaining about 4 and 6 percent of variance.
Schematic only.

## What these numbers mean

The fixture narratives are built from a small template vocabulary, so the text signal is
far cleaner than real complaint text. The figures above demonstrate that the protocol
runs and that its parts interact as intended (duplicates stay on one side, the threshold
is chosen before test is touched, routed items are the ones the model gets wrong). They
say nothing about performance on CFPB data.

## Local test run (2026-09-27)

```
$ python -m pytest -q
70 passed in 46.16s
```

Test groups: fixture and CFPB client (10), duplicates and split (12), leakage, features
and model (13), abstention, metrics, trends, landscape, portfolio (12), sanitisation,
review log, summariser (11), CLI and end to end (12). Failure cases named in the spec are
covered: duplicate narratives across splits, script-tag sanitisation, long-text bounds,
empty narratives, class imbalance, refusal of leaking features, override audit events,
monotone coverage, chronology of the split, and missing-input exits.
