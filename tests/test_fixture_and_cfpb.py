from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from complaint_intelligence_lab.cfpb.client import DATA_USE_URL, CfpbClient, FetchRequest
from complaint_intelligence_lab.cfpb.fields import CFPB_COLUMNS, NARRATIVE
from complaint_intelligence_lab.cfpb.schema import SchemaError, validate_frame
from complaint_intelligence_lab.cfpb.synthetic import COMPANIES, MONTHS, generate_fixture


def test_fixture_has_cfpb_schema_columns(fixture_path: Path) -> None:
    frame = pd.read_csv(fixture_path, dtype=str, keep_default_na=False)
    assert list(frame.columns) == CFPB_COLUMNS + [NARRATIVE]
    assert len(frame) == 4000


def test_fixture_spans_24_months_and_uses_synthetic_ids(fixture_path: Path) -> None:
    frame = pd.read_csv(fixture_path, dtype=str, keep_default_na=False)
    months = pd.to_datetime(frame["date_received"]).dt.to_period("M").nunique()
    assert months == MONTHS
    assert frame["complaint_id"].str.match(r"^SYN-CMP-\d{6}$").all()
    assert set(frame["company"]).issubset(set(COMPANIES))
    assert all("Demo" in c for c in COMPANIES)


def test_fixture_generation_is_deterministic() -> None:
    a = generate_fixture(rows=120, seed=42)
    b = generate_fixture(rows=120, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_fixture_narratives_carry_no_digits_or_at_signs() -> None:
    frame = generate_fixture(rows=300, seed=3)
    text = " ".join(frame[NARRATIVE])
    assert not any(ch.isdigit() for ch in text)
    assert "@" not in text


def test_fixture_contains_empty_and_duplicate_narratives() -> None:
    frame = generate_fixture(rows=500, seed=5)
    assert (frame[NARRATIVE] == "").sum() > 0
    non_empty = frame.loc[frame[NARRATIVE] != "", NARRATIVE]
    assert non_empty.duplicated().sum() > 0


def test_validate_frame_reports_every_missing_column() -> None:
    frame = pd.DataFrame({"complaint_id": ["a"], "date_received": ["2025-01-01"]})
    with pytest.raises(SchemaError) as excinfo:
        validate_frame(frame)
    assert "product" in str(excinfo.value) and "issue" in str(excinfo.value)


def test_validate_frame_rejects_bad_dates_and_duplicate_ids() -> None:
    base = {"product": ["p", "p"], "issue": ["i", "i"]}
    with pytest.raises(SchemaError, match="unparseable"):
        validate_frame(pd.DataFrame({"complaint_id": ["a", "b"], "date_received": ["2025-01-01", "not a date"], **base}))
    with pytest.raises(SchemaError, match="unique"):
        validate_frame(pd.DataFrame({"complaint_id": ["a", "a"], "date_received": ["2025-01-01", "2025-01-02"], **base}))


def _payload(n: int = 3) -> dict:
    hits = []
    for i in range(n):
        hits.append(
            {
                "_source": {
                    **{c: f"v{i}" for c in CFPB_COLUMNS},
                    "date_received": "2025-01-01",
                    NARRATIVE: "free text that must not be cached by default",
                }
            }
        )
    return {"hits": {"hits": hits}}


def test_client_drops_narrative_by_default(tmp_path: Path) -> None:
    calls: list[dict] = []

    def transport(url: str, params: dict) -> dict:
        calls.append(params)
        return _payload()

    client = CfpbClient(tmp_path, transport=transport)
    frame = client.fetch(FetchRequest("2025-01-01", "2025-01-31"))
    assert NARRATIVE not in frame.columns
    cached = json.loads(client.cache_path(FetchRequest("2025-01-01", "2025-01-31")).read_text())
    assert cached["narrative_included"] is False
    assert "free text" not in json.dumps(cached)
    assert "downloaded_at" in cached and cached["params"]["date_received_min"] == "2025-01-01"
    assert cached["data_use_url"] == DATA_USE_URL


def test_client_keeps_narrative_only_on_opt_in(tmp_path: Path) -> None:
    client = CfpbClient(tmp_path, transport=lambda u, p: _payload())
    frame = client.fetch(FetchRequest("2025-01-01", "2025-01-31", include_narrative=True))
    assert NARRATIVE in frame.columns
    assert frame.attrs["narrative_included"] is True


def test_client_serves_from_cache_without_calling_transport(tmp_path: Path) -> None:
    counter = {"n": 0}

    def transport(url: str, params: dict) -> dict:
        counter["n"] += 1
        return _payload()

    client = CfpbClient(tmp_path, transport=transport)
    req = FetchRequest("2025-02-01", "2025-02-28")
    client.fetch(req)
    client.fetch(req)
    assert counter["n"] == 1
    client.fetch(req, force=True)
    assert counter["n"] == 2
