# Contributing

Thank you for considering a contribution.

## Ground rules

- No real complaint data in the repository. Tests and demos run on the authored synthetic
  fixture in `examples/fixtures/v1/`. If you need a different shape of data, extend
  `tools/make_fixture.py` and bump the fixture version.
- No network access in tests, smoke or demo. The CFPB client takes an injectable transport
  so it can be tested with a fake.
- Every number that appears in documentation must have been produced by a script in this
  repository, and the text must say so.
- Keep modules small and typed. Fail fast at boundaries with an actionable message.
- Do not add `eval`, `exec` or pickle-based persistence. Model artifacts are numpy `.npz`
  plus JSON.

## Workflow

```
pip install -e . && pip install pytest
python tools/make_fixture.py
python -m pytest -q
python -m complaint_intelligence_lab smoke
```

Open a pull request against `main` with a short description of the behaviour change and
the test that covers it. The CI matrix runs on Python 3.10, 3.11 and 3.12.
