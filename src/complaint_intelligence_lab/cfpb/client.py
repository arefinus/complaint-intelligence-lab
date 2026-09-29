"""Client for the CFPB Consumer Complaint Database public API with a dated cache.

The public API returns whole complaint records. This client projects the response down
to the structured fields and drops the free-text narrative before anything is written to
disk, unless ``include_narrative=True`` is set explicitly. Narratives are consumer-written
text and may contain personal details that the provider scrubs but does not guarantee to
remove; keep them out of caches unless the analysis needs them.

Network access happens only through ``fetch``; tests inject a fake transport. Nothing in
``make smoke``, ``make demo`` or the test suite calls the network.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .fields import CFPB_COLUMNS, NARRATIVE

API_BASE = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
API_DOCS_URL = "https://cfpb.github.io/api/ccdb/"
DATA_USE_URL = "https://www.consumerfinance.gov/complaint/data-use/"
DATABASE_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/"

PRIVACY_NOTE = (
    "Narratives are consumer-written free text. The provider publishes them only with "
    "consumer consent and after scrubbing, but they can still describe personal "
    "circumstances. This client excludes them by default; pass include_narrative=True "
    "only when the analysis needs the text, and do not redistribute cached narratives."
)

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class FetchRequest:
    date_received_min: str
    date_received_max: str
    size: int = 1000
    product: str | None = None
    include_narrative: bool = False

    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "date_received_min": self.date_received_min,
            "date_received_max": self.date_received_max,
            "size": self.size,
            "no_aggs": "true",
            "format": "json",
        }
        if self.product:
            params["product"] = self.product
        if self.include_narrative:
            params["has_narrative"] = "true"
        return params

    def cache_key(self) -> str:
        blob = json.dumps(self.params(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class CfpbClient:
    cache_dir: Path
    transport: Transport | None = None
    fields: tuple[str, ...] = field(default_factory=lambda: tuple(CFPB_COLUMNS))

    def cache_path(self, request: FetchRequest) -> Path:
        return self.cache_dir / f"cfpb_{request.cache_key()}.json"

    def fetch(self, request: FetchRequest, force: bool = False) -> pd.DataFrame:
        """Return a frame for ``request``, serving from cache when present."""
        path = self.cache_path(request)
        if path.exists() and not force:
            return self._frame_from_cache(json.loads(path.read_text(encoding="utf-8")))
        transport = self.transport or _requests_transport
        payload = transport(API_BASE, request.params())
        rows = _project(payload, self.fields, request.include_narrative)
        record = {
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "endpoint": API_BASE,
            "params": request.params(),
            "fields": list(self.fields) + ([NARRATIVE] if request.include_narrative else []),
            "narrative_included": request.include_narrative,
            "data_use_url": DATA_USE_URL,
            "row_count": len(rows),
            "rows": rows,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        return self._frame_from_cache(record)

    @staticmethod
    def _frame_from_cache(record: dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame(record["rows"], columns=record["fields"])
        frame.attrs["downloaded_at"] = record["downloaded_at"]
        frame.attrs["narrative_included"] = record["narrative_included"]
        return frame


def _project(payload: dict[str, Any], fields: tuple[str, ...], include_narrative: bool) -> list[dict[str, Any]]:
    hits = payload.get("hits", {}).get("hits", [])
    keep = list(fields) + ([NARRATIVE] if include_narrative else [])
    out: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source", hit)
        out.append({k: source.get(k) for k in keep})
    return out


def _requests_transport(url: str, params: dict[str, Any]) -> dict[str, Any]:
    import requests  # imported lazily so tests never need it

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()
