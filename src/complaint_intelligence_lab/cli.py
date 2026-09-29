"""Command line interface: smoke, demo, fetch, train, evaluate, route, figures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .cfpb.client import API_BASE, API_DOCS_URL, DATA_USE_URL, PRIVACY_NOTE, CfpbClient, FetchRequest
from .config import PACKAGE_ROOT, LabConfig, load_config
from .figures import render_all
from .run import (
    PREDICTION_COLUMNS,
    evaluate_predictions,
    load_fixture,
    predict_split,
    prepare,
    run_pipeline,
    train_model,
)
from .pipeline.abstain import route as route_confidence
from .pipeline.model import ComplaintClassifier

SMOKE_LIMIT = 1200
REQUIRED_OUTPUTS = ("metrics.json", "trends.json", "topic_landscape.json", "routing.json", "portfolio.json", "run_manifest.yaml", "predictions_test.csv")


def _config(args: argparse.Namespace) -> LabConfig:
    return load_config(Path(args.config) if args.config else None)


def cmd_smoke(args: argparse.Namespace) -> int:
    config = _config(args)
    out = Path(args.out) if args.out else config.output_path() / "smoke"
    summary = run_pipeline(config, out, mode="smoke", limit=SMOKE_LIMIT)
    missing = [f for f in REQUIRED_OUTPUTS if not (out / f).exists()]
    if missing:
        print(f"smoke FAILED: missing outputs {missing}", file=sys.stderr)
        return 1
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for key in ("all_rows", "abstention", "dedup", "split"):
        if key not in metrics:
            print(f"smoke FAILED: metrics.json lacks {key!r}", file=sys.stderr)
            return 1
    portfolio = json.loads((out / "portfolio.json").read_text(encoding="utf-8"))
    if portfolio.get("mode") != "synthetic-demo" or "themes" not in portfolio or "routing" not in portfolio:
        print("smoke FAILED: portfolio.json schema mismatch", file=sys.stderr)
        return 1
    print(f"smoke ok: {summary['rows']} rows, outputs in {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    config = _config(args)
    out = Path(args.out) if args.out else config.output_path()
    summary = run_pipeline(config, out, mode="demo")
    figures = render_all(out, PACKAGE_ROOT / "docs" / "figures")
    print(json.dumps({k: v for k, v in summary.items() if k != "flagged_themes"}, indent=1))
    print(f"flagged themes: {[(t['product'], t['issue']) for t in summary['flagged_themes']]}")
    print(f"figures: {[str(p) for p in figures]}")
    print("all numbers above were computed by this run on synthetic fixture v1 (seed 42); not a benchmark result")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = _config(args)
    out = Path(args.out) if args.out else config.output_path()
    frame = load_fixture(config.fixture_path())
    prepared = prepare(frame, config)
    model = train_model(prepared, config)
    model.save(out / "model")
    valid = predict_split(model, prepared, "valid")
    valid["route"] = "n/a"
    valid[list(PREDICTION_COLUMNS)].to_csv(out / "predictions_valid.csv", index=False, lineterminator="\n")
    print(f"trained on {int((prepared.split == 'train').sum())} rows; model saved to {out / 'model'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    path = Path(args.predictions)
    if not path.exists():
        print(f"evaluate: predictions file not found: {path}. Run `demo` or `train` first.", file=sys.stderr)
        return 2
    preds = pd.read_csv(path, dtype={"complaint_id": str, "true_label": str, "predicted_label": str})
    needed = {"true_label", "predicted_label", "confidence"}
    if not needed.issubset(preds.columns):
        print(f"evaluate: predictions file lacks columns {sorted(needed - set(preds.columns))}", file=sys.stderr)
        return 2
    classes = tuple(sorted(set(preds["true_label"]) | set(preds["predicted_label"])))
    report = evaluate_predictions(preds, classes, args.threshold)
    print(json.dumps(report["all_rows"], indent=1))
    print(f"abstention at {args.threshold}: coverage {report['abstention']['coverage']:.3f}")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    config = _config(args)
    model_dir = Path(args.model_dir) if args.model_dir else config.output_path() / "model"
    try:
        model = ComplaintClassifier.load(model_dir)
    except FileNotFoundError as exc:
        print(f"route: {exc}", file=sys.stderr)
        return 2
    path = Path(args.input)
    if not path.exists():
        print(f"route: input not found: {path}", file=sys.stderr)
        return 2
    frame = load_fixture(path)
    pred = model.predict(frame)
    routes = route_confidence(pred.confidence, args.threshold)
    out = pd.DataFrame({"complaint_id": frame["complaint_id"], "predicted_label": pred.labels, "confidence": pred.confidence.round(6), "route": routes})
    print(out.to_csv(index=False, lineterminator="\n"))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    config = _config(args)
    out = Path(args.out) if args.out else config.output_path()
    try:
        written = render_all(out, PACKAGE_ROOT / "docs" / "figures")
    except FileNotFoundError as exc:
        print(f"figures: {exc}", file=sys.stderr)
        return 2
    print("\n".join(str(p) for p in written))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    print("CFPB Consumer Complaint Database: read the data-use notes before downloading:")
    print(f"  {DATA_USE_URL}")
    print(f"  API documentation: {API_DOCS_URL}")
    print(f"  Endpoint: {API_BASE}")
    print("Complaints are allegations as submitted, not adjudicated findings.")
    if args.include_narrative:
        print("PRIVACY NOTE: " + PRIVACY_NOTE)
    if not args.yes:
        print("Re-run with --yes to acknowledge the terms and start the download.", file=sys.stderr)
        return 2
    client = CfpbClient(Path(args.cache_dir))
    request = FetchRequest(args.date_min, args.date_max, size=args.size, product=args.product, include_narrative=args.include_narrative)
    frame = client.fetch(request)
    print(f"fetched {len(frame)} rows (downloaded_at={frame.attrs['downloaded_at']}) -> {client.cache_path(request)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="complaint_intelligence_lab", description="Complaint categorisation and review routing on the CFPB schema.")
    parser.add_argument("--config", default=None, help="path to a YAML config (default configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke", help="fast offline check on a fixture subset"); p.add_argument("--out", default=None); p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("demo", help="full demonstration on the fixture, writes examples/output/"); p.add_argument("--out", default=None); p.set_defaults(func=cmd_demo)
    p = sub.add_parser("train", help="train the baseline on the fixture train split"); p.add_argument("--out", default=None); p.set_defaults(func=cmd_train)
    p = sub.add_parser("evaluate", help="evaluate a predictions CSV")
    p.add_argument("--predictions", default=str(PACKAGE_ROOT / "examples" / "output" / "predictions_test.csv"))
    p.add_argument("--threshold", type=float, default=0.5); p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("route", help="score a CFPB-schema CSV with a saved model and print routes")
    p.add_argument("--input", required=True); p.add_argument("--model-dir", default=None); p.add_argument("--threshold", type=float, default=0.5); p.set_defaults(func=cmd_route)
    p = sub.add_parser("figures", help="render SVG figures from demo outputs"); p.add_argument("--out", default=None); p.set_defaults(func=cmd_figures)
    p = sub.add_parser("fetch", help="download from the public CFPB API (never in CI or tests)")
    p.add_argument("--date-min", required=True); p.add_argument("--date-max", required=True)
    p.add_argument("--size", type=int, default=1000); p.add_argument("--product", default=None)
    p.add_argument("--cache-dir", default="data/cfpb_cache"); p.add_argument("--include-narrative", action="store_true")
    p.add_argument("--yes", action="store_true", help="acknowledge the data-use notes"); p.set_defaults(func=cmd_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
