PYTHON ?= python
PKG = complaint_intelligence_lab

.PHONY: setup smoke demo test evaluate train fetch figures fixture

setup:
	pip install -e . && pip install pytest

smoke:
	$(PYTHON) -m $(PKG) smoke

demo:
	$(PYTHON) -m $(PKG) demo

test:
	$(PYTHON) -m pytest -q

evaluate:
	$(PYTHON) -m $(PKG) evaluate --predictions examples/output/predictions_test.csv --threshold 0.3

train:
	$(PYTHON) -m $(PKG) train

figures:
	$(PYTHON) -m $(PKG) figures

fixture:
	$(PYTHON) tools/make_fixture.py --rows 4000 --seed 42

# Public CFPB API. Prints the data-use notes URL first; never run in CI or tests.
# Requires `pip install requests`. Pass --yes after reading the notes.
fetch:
	$(PYTHON) -m $(PKG) fetch --date-min $${DATE_MIN:-2025-01-01} --date-max $${DATE_MAX:-2025-01-31} --size $${SIZE:-1000} $(FETCH_ARGS)
