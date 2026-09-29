"""Monthly trend aggregation by product and issue with a simple attention flag.

A theme is flagged when its mean monthly volume over the most recent window is at least
``ratio_threshold`` times its mean over the preceding baseline window, and the recent
window holds at least ``min_count`` complaints. Complaint volume reflects who complains
and how; it is not a ranking of company quality and the flag is a prompt to look, not a
finding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrendConfig:
    recent_months: int = 3
    baseline_months: int = 12
    ratio_threshold: float = 2.0
    min_count: int = 15


def monthly_counts(frame: pd.DataFrame, by: tuple[str, ...] = ("product", "issue")) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date_received"])
    month = dates.dt.to_period("M").astype(str)
    grouped = frame.assign(month=month).groupby(["month", *by], observed=True).size().rename("count").reset_index()
    return grouped.sort_values(["month", *by]).reset_index(drop=True)


def theme_flags(counts: pd.DataFrame, config: TrendConfig = TrendConfig(), by: tuple[str, ...] = ("product", "issue")) -> list[dict[str, Any]]:
    months = sorted(counts["month"].unique())
    if len(months) < config.recent_months + 1:
        return []
    recent = months[-config.recent_months :]
    baseline = months[-(config.recent_months + config.baseline_months) : -config.recent_months]
    if not baseline:
        return []
    out: list[dict[str, Any]] = []
    for key, sub in counts.groupby(list(by), observed=True):
        key_t = key if isinstance(key, tuple) else (key,)
        series = sub.set_index("month")["count"]
        recent_mean = float(series.reindex(recent, fill_value=0).mean())
        base_mean = float(series.reindex(baseline, fill_value=0).mean())
        recent_total = int(series.reindex(recent, fill_value=0).sum())
        ratio = (recent_mean / base_mean) if base_mean > 0 else (float("inf") if recent_mean > 0 else 0.0)
        flagged = recent_total >= config.min_count and ratio >= config.ratio_threshold
        out.append(
            {
                **{k: str(v) for k, v in zip(by, key_t)},
                "recent_months": recent,
                "recent_mean_per_month": round(recent_mean, 3),
                "baseline_mean_per_month": round(base_mean, 3),
                "recent_total": recent_total,
                "ratio": (None if ratio == float("inf") else round(ratio, 3)),
                "flag": bool(flagged),
            }
        )
    out.sort(key=lambda d: (not d["flag"], -(d["ratio"] if d["ratio"] is not None else 1e9)))
    return out


def trend_report(frame: pd.DataFrame, config: TrendConfig = TrendConfig()) -> dict[str, Any]:
    by_product = monthly_counts(frame, by=("product",))
    by_theme = monthly_counts(frame, by=("product", "issue"))
    return {
        "note": "Complaint counts are allegations as submitted, not adjudicated findings, and not a company quality ranking.",
        "config": config.__dict__,
        "monthly_by_product": by_product.to_dict(orient="records"),
        "monthly_by_theme": by_theme.to_dict(orient="records"),
        "themes": theme_flags(by_theme, config),
    }
