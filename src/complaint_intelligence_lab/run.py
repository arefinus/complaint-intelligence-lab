"""End-to-end orchestration used by the CLI: prepare, train, evaluate, route, report."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cfpb.fields import NARRATIVE
from .cfpb.schema import validate_frame
from .config import LabConfig
from .manifest import now_iso, sha256_file, write_manifest
from .pipeline.abstain import DEFAULT_GRID, choose_threshold, coverage_curve, route
from .pipeline.dedup import find_duplicates
from .pipeline.landscape import topic_landscape
from .pipeline.leakage import check_features
from .pipeline.metrics import classification_report
from .pipeline.model import ComplaintClassifier
from .pipeline.review import ReviewLog
from .pipeline.sanitize import sanitize_text
from .pipeline.split import assert_no_group_crosses, chronological_split
from .pipeline.summarise import TemplateSummariser
from .pipeline.trends import trend_report
from .portfolio import build_portfolio

PREDICTION_COLUMNS = ("complaint_id", "date_received", "true_label", "predicted_label", "confidence", "route", "split")


@dataclass(frozen=True)
class Prepared:
    frame: pd.DataFrame
    split: np.ndarray
    group_id: np.ndarray
    split_hash: str
    dedup_summary: dict[str, int]
    moved_for_dedup: int


def load_fixture(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"fixture not found at {path}. Generate it with `python tools/make_fixture.py` "
            "(authored synthetic data; no CFPB rows are shipped)."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if limit is not None:
        frame = frame.head(limit)
    return validate_frame(frame, require_narrative=True)


def prepare(frame: pd.DataFrame, config: LabConfig) -> Prepared:
    dedup = find_duplicates(
        frame[NARRATIVE].tolist(), k=config.dedup.shingle_k, threshold=config.dedup.jaccard_threshold
    )
    split = chronological_split(
        frame["date_received"], dedup.group_id, config.split.train_frac, config.split.valid_frac
    )
    assert_no_group_crosses(split.split, dedup.group_id)
    return Prepared(
        frame=frame.reset_index(drop=True),
        split=split.split,
        group_id=dedup.group_id,
        split_hash=split.manifest_hash(frame["complaint_id"]),
        dedup_summary={
            "exact_pairs": dedup.exact_pairs,
            "near_pairs": dedup.near_pairs,
            "groups_with_duplicates": dedup.n_groups_with_duplicates,
            "rows_in_duplicate_groups": int(dedup.is_duplicate().sum()),
        },
        moved_for_dedup=split.moved_for_dedup,
    )


def train_model(prepared: Prepared, config: LabConfig) -> ComplaintClassifier:
    spec = check_features(config.target, list(config.structured_features))
    if not config.use_narrative:
        spec = spec.__class__(target=spec.target, structured=spec.structured, use_narrative=False)
    train = prepared.frame[prepared.split == "train"]
    return ComplaintClassifier(spec, config.tfidf, config.model).fit(train)


def predict_split(model: ComplaintClassifier, prepared: Prepared, name: str) -> pd.DataFrame:
    sub = prepared.frame[prepared.split == name]
    pred = model.predict(sub)
    return pd.DataFrame(
        {
            "complaint_id": sub["complaint_id"].to_numpy(),
            "date_received": sub["date_received"].dt.strftime("%Y-%m-%d").to_numpy(),
            "true_label": sub[model.spec.target].astype(str).to_numpy(),
            "predicted_label": pred.labels,
            "confidence": np.round(pred.confidence, 6),
            "split": name,
        }
    )


def evaluate_predictions(preds: pd.DataFrame, classes: tuple[str, ...], threshold: float) -> dict[str, Any]:
    correct = (preds["true_label"] == preds["predicted_label"]).to_numpy()
    curve = coverage_curve(preds["confidence"].to_numpy(), correct, DEFAULT_GRID)
    covered = preds["confidence"].to_numpy() >= threshold
    report = classification_report(preds["true_label"].to_numpy(), preds["predicted_label"].to_numpy(), classes)
    covered_report = (
        classification_report(preds["true_label"].to_numpy()[covered], preds["predicted_label"].to_numpy()[covered], classes)
        if covered.any()
        else None
    )
    return {
        "evidence": "demo",
        "note": "computed by `make demo` on synthetic fixture v1, seed 42, not a benchmark result",
        "all_rows": report,
        "abstention": {
            "threshold": float(threshold),
            "coverage": float(covered.mean()),
            "n_routed_to_review": int((~covered).sum()),
            "covered_rows": covered_report,
            "curve": [p.to_dict() for p in curve],
        },
    }


def run_pipeline(config: LabConfig, out_dir: Path, mode: str, limit: int | None = None) -> dict[str, Any]:
    """Run everything on the fixture and write outputs. Returns a summary dict."""
    started = now_iso()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_fixture(config.fixture_path(), limit=limit)
    prepared = prepare(frame, config)
    model = train_model(prepared, config)
    model_dir = out_dir / "model"
    model.save(model_dir)

    valid = predict_split(model, prepared, "valid")
    test = predict_split(model, prepared, "test")
    v_correct = (valid["true_label"] == valid["predicted_label"]).to_numpy()
    v_curve = coverage_curve(valid["confidence"].to_numpy(), v_correct, DEFAULT_GRID)
    threshold = choose_threshold(v_curve, config.abstention.target_accuracy, config.abstention.min_coverage)
    for part in (valid, test):
        part["route"] = route(part["confidence"].to_numpy(), threshold)

    preds_path = out_dir / "predictions_test.csv"
    test[list(PREDICTION_COLUMNS)].to_csv(preds_path, index=False, lineterminator="\n")
    valid[list(PREDICTION_COLUMNS)].to_csv(out_dir / "predictions_valid.csv", index=False, lineterminator="\n")

    metrics = evaluate_predictions(test, model.classes, threshold)
    metrics["validation_curve"] = [p.to_dict() for p in v_curve]
    metrics["threshold_selected_on"] = "valid"
    metrics["dedup"] = prepared.dedup_summary
    metrics["split"] = {
        "counts": {s: int((prepared.split == s).sum()) for s in ("train", "valid", "test")},
        "moved_for_dedup": prepared.moved_for_dedup,
        "rule": "chronological; duplicate groups follow their earliest member",
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    trends = trend_report(prepared.frame, config.trends)
    (out_dir / "trends.json").write_text(json.dumps(trends, indent=1), encoding="utf-8")

    train_rows = prepared.frame[prepared.split == "train"]
    landscape = topic_landscape(
        model.transform(train_rows), train_rows[config.target].to_numpy(), config.seed, config.landscape_max_points
    )
    (out_dir / "topic_landscape.json").write_text(json.dumps(landscape, indent=1), encoding="utf-8")

    routing = routing_table(test, prepared.frame, config.target)
    (out_dir / "routing.json").write_text(json.dumps(routing, indent=1), encoding="utf-8")

    review = simulated_review(test, out_dir / "review_log.jsonl")

    portfolio = build_portfolio(trends["themes"], routing, metrics, review)
    (out_dir / "portfolio.json").write_text(json.dumps(portfolio, indent=1), encoding="utf-8")

    manifest_path = write_manifest(
        out_dir,
        mode=mode,
        status="completed",
        started_at=started,
        seed=config.seed,
        configuration_hash=config.hash(),
        split_manifest_hash=prepared.split_hash,
        sample_counts=metrics["split"]["counts"],
        metrics_file=str(metrics_path.relative_to(out_dir)),
        predictions_file=str(preds_path.relative_to(out_dir)),
        extra={"fixture_sha256": sha256_file(config.fixture_path()), "threshold": threshold},
    )
    return {
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "rows": int(len(frame)),
        "threshold": threshold,
        "macro_f1_test": metrics["all_rows"]["macro_f1"],
        "coverage_test": metrics["abstention"]["coverage"],
        "flagged_themes": [t for t in trends["themes"] if t["flag"]],
        "review": review,
    }


def routing_table(test: pd.DataFrame, frame: pd.DataFrame, target: str, limit: int = 200) -> list[dict[str, Any]]:
    summariser = TemplateSummariser()
    merged = test.merge(frame[["complaint_id", NARRATIVE, "issue", "product"]], on="complaint_id", how="left")
    review_first = merged.sort_values(["route", "confidence"], ascending=[False, True]).head(limit)
    rows: list[dict[str, Any]] = []
    for rec in review_first.itertuples(index=False):
        clean = sanitize_text(getattr(rec, NARRATIVE), 280)
        rows.append(
            {
                "complaint_id": rec.complaint_id,
                "date_received": rec.date_received,
                "predicted_label": rec.predicted_label,
                "confidence": float(rec.confidence),
                "route": rec.route,
                "has_narrative": clean.has_narrative,
                "display_text": clean.text,
                "summary": summariser.summarise(getattr(rec, NARRATIVE), rec.product, rec.issue).to_dict(),
            }
        )
    return rows


def simulated_review(test: pd.DataFrame, log_path: Path) -> dict[str, Any]:
    """Demo only: a simulated reviewer confirms or overrides routed items using fixture labels."""
    if log_path.exists():
        log_path.unlink()
    log = ReviewLog(log_path, known_ids=set(test["complaint_id"]))
    routed = test[test["route"] == "human_review"]
    for rec in routed.itertuples(index=False):
        if rec.predicted_label == rec.true_label:
            log.confirm(rec.complaint_id, rec.predicted_label, "SIM-REVIEWER-01", "simulated review on fixture labels")
        else:
            log.override(rec.complaint_id, rec.predicted_label, rec.true_label, "SIM-REVIEWER-01", "simulated review on fixture labels")
    return {"note": "simulated reviewer using fixture labels; demo only", **log.summary()}
