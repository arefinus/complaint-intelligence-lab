"""Portfolio JSON export consumed by the owner's portfolio site."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

PORTFOLIO_SCHEMA_VERSION = 1
MODE = "synthetic-demo"


def build_portfolio(themes: list[dict[str, Any]], routing: list[dict[str, Any]], metrics: dict[str, Any], review: dict[str, Any], max_routing: int = 50) -> dict[str, Any]:
    return {
        "schemaVersion": PORTFOLIO_SCHEMA_VERSION,
        "mode": MODE,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "evidence": "demo",
        "note": "computed on authored synthetic fixture v1, seed 42; not a benchmark result; complaints are allegations, not findings",
        "themes": [
            {
                "product": t["product"],
                "issue": t["issue"],
                "recentMeanPerMonth": t["recent_mean_per_month"],
                "baselineMeanPerMonth": t["baseline_mean_per_month"],
                "ratio": t["ratio"],
                "flag": t["flag"],
            }
            for t in themes
        ],
        "routing": [
            {
                "complaintId": r["complaint_id"],
                "predictedLabel": r["predicted_label"],
                "confidence": r["confidence"],
                "route": r["route"],
                "hasNarrative": r["has_narrative"],
            }
            for r in routing[:max_routing]
        ],
        "metrics": {
            "macroF1": metrics["all_rows"]["macro_f1"],
            "coverage": metrics["abstention"]["coverage"],
            "threshold": metrics["abstention"]["threshold"],
            "nTest": metrics["all_rows"]["n"],
        },
        "review": review,
    }
