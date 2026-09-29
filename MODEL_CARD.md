# Model card: TF-IDF + logistic regression complaint categoriser (demo)

## Summary

A linear baseline that predicts a complaint's `product` from its narrative text and three
intake fields (`state`, `submitted_via`, `company`). It exists to demonstrate a defensible
evaluation and routing protocol, not to be deployed. Evidence label for every number here:
`demo`.

## Intended use

- Show a customer-experience analyst which complaint themes are growing and which
  individual items the model is unsure about and should be read by a person.
- Serve as a reference implementation of leakage guards, duplicate-aware chronological
  splits, abstention and an auditable override log.

## Out of scope

- Any decision about a consumer or a company. Complaints are allegations.
- Ranking companies by complaint volume.
- Running on real CFPB data without re-checking the fixture assumptions (narrative share,
  class set, drift over years).

## Training data

Authored synthetic fixture v1 (4,000 rows, seed 42, see DATA_CARD.md and
`examples/fixtures/v1/FIXTURE.md`). No CFPB rows were used. The narratives are template
sentences, so the task is easier than real text; treat every metric as a property of the
fixture.

## Features and the leakage guard

- TF-IDF, word 1-2 grams, at most 5,000 terms, `min_df=2`, sublinear tf, fitted on train.
- One-hot on `state`, `submitted_via`, `company`, unknown categories ignored.
- Refused by `pipeline/leakage.py`: `company_response`, `timely`, `consumer_disputed`
  (post-event outcomes), `sub_product` when predicting `product`, `sub_issue` when
  predicting `issue`, and the target itself. Tests cover each refusal.

## Evaluation protocol

Chronological split with duplicate groups kept on one side. The abstention threshold is
chosen on the validation split for an operating target of 0.98 accuracy on covered items
(a policy setting in `configs/default.yaml`, chosen so the demo exercises the review
route) and then applied unchanged to test.

## Results on the fixture

Computed by `make demo` on synthetic fixture v1, seed 42, not a benchmark result.
Test split, 621 rows:

| Quantity | Value |
|---|---|
| Macro-F1, all rows | 0.865 |
| Accuracy, all rows | 0.882 |
| Selected threshold (on valid) | 0.30 |
| Coverage on test | 0.890 (553 auto, 68 routed to review) |
| Accuracy on covered rows | 0.982 |
| Macro-F1 on covered rows | 0.978 |
| Lowest per-class recall (all rows) | Mortgage, 0.814 |
| Highest per-class recall (all rows) | Student loan, 0.925 |

Full per-class recall, the confusion matrix with class order, the coverage/accuracy curve
and disagreement counts are in `examples/output/metrics.json`.

## Limitations

- The fixture's templates make the text signal much cleaner than real narratives; the
  coverage/accuracy curve will look different on real data.
- The model has no notion of time beyond the split; drift is not modelled.
- Structured features include `company`, which is legitimate at intake but means the model
  partly learns which fictional company sells which product. On real data this should be
  ablated before use.
- Abstention is based on the maximum softmax probability, which is a convenient but
  imperfect confidence signal.

## Artifacts

`examples/output/model/`: `featuriser.json` (vocabulary, categories), `featuriser.npz`
(idf), `classifier.json` (classes, config), `classifier.npz` (coefficients). No pickle.
