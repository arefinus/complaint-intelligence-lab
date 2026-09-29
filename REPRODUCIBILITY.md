# Reproducibility

## Exact commands

```
pip install -e . && pip install pytest        # make setup
python tools/make_fixture.py --rows 4000 --seed 42
python -m complaint_intelligence_lab smoke     # make smoke  (fixture subset, under 30 s)
python -m complaint_intelligence_lab demo      # make demo   (writes examples/output/ and docs/figures/)
python -m pytest -q                            # make test
python -m complaint_intelligence_lab evaluate --predictions examples/output/predictions_test.csv --threshold 0.3
```

## Seeds and determinism

- Fixture: numpy `default_rng(42)`; the CSV is byte-stable across runs on the same
  numpy/pandas versions (tested).
- Model: `LogisticRegression(random_state=42)` with the lbfgs solver; TF-IDF and one-hot
  are deterministic. TruncatedSVD uses `random_state=42`.
- Every run writes `run_manifest.yaml` with the configuration hash, the split manifest
  hash and the fixture SHA-256.

## What is reproduced and what is not

- Reproduced here: everything in `examples/output/` from the shipped fixture.
- Not reproduced: any result on real CFPB data. No CFPB rows are downloaded, stored or
  evaluated in this repository. `make fetch` exists for a user who has read the provider's
  data-use notes and wants to run the same pipeline locally; its output is not part of any
  committed artifact.
- Evidence label for all committed numbers: `demo`.

## Environment

Developed on Python 3.10 with numpy 1.26, pandas 2.3, scipy 1.15, scikit-learn 1.7.
CI runs the same commands on 3.10, 3.11 and 3.12. Minor numeric differences across
scikit-learn versions are possible in the third decimal of the metrics; the split
membership and the fixture do not change.
