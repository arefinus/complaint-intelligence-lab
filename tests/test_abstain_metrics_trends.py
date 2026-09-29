from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from complaint_intelligence_lab.pipeline.abstain import DEFAULT_GRID, choose_threshold, coverage_curve, route
from complaint_intelligence_lab.pipeline.landscape import LABEL, topic_landscape
from complaint_intelligence_lab.pipeline.metrics import classification_report
from complaint_intelligence_lab.pipeline.trends import TrendConfig, monthly_counts, theme_flags, trend_report
from complaint_intelligence_lab.portfolio import build_portfolio


def test_coverage_is_monotone_non_increasing_in_threshold() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.2, 1.0, size=500)
    correct = rng.random(500) < conf
    curve = coverage_curve(conf, correct)
    coverages = [p.coverage for p in curve]
    assert all(a >= b for a, b in zip(coverages, coverages[1:]))
    assert curve[0].threshold == 0.0 and curve[0].coverage == 1.0


def test_curve_reports_none_accuracy_when_nothing_covered() -> None:
    curve = coverage_curve(np.array([0.1, 0.2]), np.array([True, False]), grid=(0.0, 0.5))
    assert curve[1].n_covered == 0 and curve[1].accuracy is None


def test_choose_threshold_picks_smallest_meeting_target() -> None:
    curve = coverage_curve(np.array([0.3, 0.6, 0.9, 0.95]), np.array([False, True, True, True]), grid=(0.0, 0.5, 0.9))
    assert choose_threshold(curve, target_accuracy=0.99, min_coverage=0.2) == 0.5
    assert choose_threshold(curve, target_accuracy=0.5, min_coverage=0.2) == 0.0


def test_choose_threshold_falls_back_when_target_unreachable() -> None:
    curve = coverage_curve(np.array([0.3, 0.6, 0.9]), np.array([False, False, False]), grid=(0.0, 0.5))
    assert choose_threshold(curve, target_accuracy=0.9) in {0.0, 0.5}
    with pytest.raises(ValueError):
        choose_threshold([], 0.9)


def test_route_splits_on_threshold() -> None:
    routes = route(np.array([0.2, 0.7, 0.7]), 0.7)
    assert list(routes) == ["human_review", "auto", "auto"]


def test_confusion_matrix_keeps_requested_class_order() -> None:
    y_true = np.array(["b", "a", "b", "c"])
    y_pred = np.array(["b", "b", "b", "c"])
    rep = classification_report(y_true, y_pred, ("a", "b", "c"))
    assert rep["confusion_matrix"]["class_order"] == ["a", "b", "c"]
    assert rep["confusion_matrix"]["rows_true_cols_pred"] == [[0, 1, 0], [0, 2, 0], [0, 0, 1]]
    assert rep["per_class_recall"] == {"a": 0.0, "b": 1.0, "c": 1.0}
    assert rep["disagreements"] == {"total": 1, "by_true_class": {"a": 1, "b": 0, "c": 0}}


def test_macro_f1_matches_hand_computation() -> None:
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    rep = classification_report(y_true, y_pred, ("a", "b"))
    # a: precision 1, recall .5 -> f1 .6667 ; b: precision .6667, recall 1 -> f1 .8 ; macro .7333
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_monthly_counts_sum_to_rows(small_frame: pd.DataFrame) -> None:
    counts = monthly_counts(small_frame)
    assert counts["count"].sum() == len(small_frame)
    assert set(counts.columns) == {"month", "product", "issue", "count"}


def test_theme_flag_fires_on_injected_bump_and_respects_min_count() -> None:
    months = [f"2025-{m:02d}" for m in range(1, 13)] + ["2026-01", "2026-02", "2026-03"]
    rows = []
    for m in months:
        rows.append({"month": m, "product": "P", "issue": "quiet", "count": 5})
        rows.append({"month": m, "product": "P", "issue": "loud", "count": 20 if m.startswith("2026") else 5})
        rows.append({"month": m, "product": "P", "issue": "tiny", "count": 3 if m.startswith("2026") else 1})
    counts = pd.DataFrame(rows)
    flags = {f["issue"]: f for f in theme_flags(counts, TrendConfig(recent_months=3, baseline_months=12, ratio_threshold=2.0, min_count=15))}
    assert flags["loud"]["flag"] is True and flags["loud"]["ratio"] == 4.0
    assert flags["quiet"]["flag"] is False
    assert flags["tiny"]["flag"] is False  # ratio 3 but recent_total 9 < min_count


def test_trend_report_carries_allegation_note(small_frame: pd.DataFrame) -> None:
    report = trend_report(small_frame)
    assert "allegations" in report["note"] and "not a company quality ranking" in report["note"]
    assert isinstance(report["themes"], list)


def test_topic_landscape_is_two_dimensional_and_labelled_schematic() -> None:
    from scipy import sparse

    rng = np.random.default_rng(1)
    X = sparse.csr_matrix(rng.random((50, 20)))
    labels = np.array(["a", "b"] * 25)
    out = topic_landscape(X, labels, max_points=20)
    assert out["label"] == LABEL and "no unit" in LABEL
    assert out["n_points"] == 20 and all(set(p) == {"x", "y", "label"} for p in out["points"])
    with pytest.raises(ValueError):
        topic_landscape(sparse.csr_matrix(rng.random((2, 20))), labels[:2])


def test_portfolio_schema() -> None:
    metrics = {"all_rows": {"macro_f1": 0.5, "n": 10}, "abstention": {"coverage": 0.9, "threshold": 0.4}}
    themes = [{"product": "P", "issue": "I", "recent_mean_per_month": 1.0, "baseline_mean_per_month": 0.5, "ratio": 2.0, "flag": True}]
    routing = [{"complaint_id": "SYN-CMP-000001", "predicted_label": "P", "confidence": 0.3, "route": "human_review", "has_narrative": True}]
    out = build_portfolio(themes, routing, metrics, {"events": 0})
    assert out["schemaVersion"] == 1 and out["mode"] == "synthetic-demo"
    assert out["themes"][0]["flag"] is True and out["routing"][0]["route"] == "human_review"
    assert set(DEFAULT_GRID) >= {0.0, 0.5, 0.95}
