# Data card

## Two sources, one schema

| | Authored synthetic fixture (shipped) | CFPB Consumer Complaint Database (not shipped) |
|---|---|---|
| Provider | This repository, `tools/make_fixture.py` | Consumer Financial Protection Bureau (US) |
| Terms | MIT, same as the code | Public data. Read the data-use notes first: https://www.consumerfinance.gov/complaint/data-use/ |
| Where | `examples/fixtures/v1/complaints_synthetic.csv` | API: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/ (docs: https://cfpb.github.io/api/ccdb/) |
| Rows | 4,000, 24 months (2024-07 to 2026-06), seed 42 | Millions; fetched on demand with `make fetch` |
| Used by tests and CI | Yes | Never |

The fixture uses the public schema's column names so that the same pipeline runs on both.
`FIXTURE.md` in the fixture directory states its provenance.

## What the CFPB data is and is not

- A complaint is an **allegation as submitted by a consumer**. It is not an adjudicated
  finding. The provider itself states that it does not verify all the facts alleged.
- **Complaint volume is not a company quality ranking.** Volume follows company size,
  product mix, customer base and channel access. This repository never produces a
  company league table, and its trend flags are prompts to look, not conclusions.
- Narratives are published only with consumer consent and after scrubbing, but they are
  free text about personal circumstances. The client excludes them from caches unless
  `--include-narrative` is passed, and the CLI prints a privacy note when it is.

## Record unit and target

- Record unit: one complaint.
- Target for the baseline: `product` (8 classes in the fixture). `issue` is supported as an
  alternative target through the config.

## Columns (public schema names)

| Column | Type | Available at intake | Used as feature | Note |
|---|---|---|---|---|
| `complaint_id` | string | yes | no | synthetic `SYN-CMP-000001` style in the fixture |
| `date_received` | date | yes | split only | chronological split key |
| `product` | category | target | target | |
| `sub_product` | category | yes | **refused** when target is `product` | encodes the target |
| `issue` | category | optional target | no | |
| `sub_issue` | category | yes | **refused** when target is `issue` | encodes the target |
| `company` | category | yes | yes | fictional demo names in the fixture |
| `state` | category | yes | yes | |
| `submitted_via` | category | yes | yes | |
| `company_response` | category | **no** | **refused** | post-event outcome |
| `timely` | Yes/No | **no** | **refused** | post-event outcome |
| `consumer_disputed` | Yes/No/N/A | **no** | **refused** | post-event outcome |
| `complaint_what_happened` | free text | yes (when consented) | yes | narrative; sanitised and bounded |

## Splits

Chronological: the earliest 60 percent of rows train, the next 20 percent validate, the
latest 20 percent test. Exact and near-duplicate narratives are grouped first and a whole
group follows its earliest member, so no evaluation row has a duplicate in training.
Counts on the fixture (computed by `make demo`, seed 42): train 2,726, valid 653, test 621;
365 rows moved by the duplicate rule.

## Missingness

The fixture has about 5 percent empty narratives by design. Empty text yields an all-zero
TF-IDF row and the structured features carry the prediction. The real data has a much
higher share of missing narratives because consent is opt-in; the pipeline handles that the
same way.

## Class balance

Skewed by design (Credit reporting is the largest class). The baseline uses
`class_weight="balanced"` and reports per-class recall so that minority classes are
visible.
