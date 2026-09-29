from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from complaint_intelligence_lab.cfpb.fields import NARRATIVE
from complaint_intelligence_lab.cfpb.synthetic import generate_fixture, write_fixture
from complaint_intelligence_lab.cli import main
from complaint_intelligence_lab.config import LabConfig, load_config
from complaint_intelligence_lab.figures import render_all
from complaint_intelligence_lab.run import run_pipeline


def _small_config(tmp_path: Path) -> LabConfig:
    frame = generate_fixture(rows=700, seed=9)
    paths = write_fixture(frame, tmp_path / "fx", seed=9)
    return LabConfig(fixture=str(paths["csv"]), output_dir=str(tmp_path / "out"))


def test_run_pipeline_writes_every_output_and_manifest(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    out = tmp_path / "out"
    summary = run_pipeline(config, out, mode="smoke")
    for name in ("metrics.json", "trends.json", "topic_landscape.json", "routing.json", "portfolio.json", "predictions_test.csv", "run_manifest.yaml"):
        assert (out / name).exists(), name
    manifest = yaml.safe_load((out / "run_manifest.yaml").read_text())
    for key in ("schema_version", "project_id", "run_id", "status", "mode", "evidence_status", "git_commit", "data", "configuration_hash", "seed", "environment", "metrics_file", "predictions_file", "started_at", "finished_at"):
        assert key in manifest, key
    assert manifest["evidence_status"] == "demo" and manifest["git_commit"] is None
    assert manifest["data"]["split_manifest_hash"] and manifest["seed"] == 42
    assert summary["rows"] == 700


def test_metrics_json_has_required_blocks_and_threshold_from_validation(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    run_pipeline(config, tmp_path / "out", mode="smoke")
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["threshold_selected_on"] == "valid"
    assert set(metrics["all_rows"]) >= {"macro_f1", "per_class_recall", "confusion_matrix", "disagreements"}
    assert metrics["abstention"]["curve"][0]["threshold"] == 0.0
    assert metrics["note"].startswith("computed by")


def test_routing_display_text_is_sanitised(tmp_path: Path) -> None:
    frame = generate_fixture(rows=700, seed=11)
    idx = frame.index[-70:-65]  # frame is date-sorted; these land in the test split
    frame.loc[idx, NARRATIVE] = '<script>alert(1)</script> the servicer added an escrow shortage'
    paths = write_fixture(frame, tmp_path / "fx", seed=11)
    config = LabConfig(fixture=str(paths["csv"]), output_dir=str(tmp_path / "out"))
    run_pipeline(config, tmp_path / "out", mode="smoke")
    routing = json.dumps(json.load((tmp_path / "out" / "routing.json").open()))
    assert "<script" not in routing.lower()


def test_cli_smoke_returns_zero(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    frame = generate_fixture(rows=700, seed=13)
    paths = write_fixture(frame, tmp_path / "fx", seed=13)
    config_path.write_text(yaml.safe_dump({"fixture": str(paths["csv"]), "output_dir": str(tmp_path / "out")}))
    assert main(["--config", str(config_path), "smoke", "--out", str(tmp_path / "smoke")]) == 0
    assert (tmp_path / "smoke" / "portfolio.json").exists()


def test_cli_evaluate_missing_predictions_exits_2(tmp_path: Path) -> None:
    assert main(["evaluate", "--predictions", str(tmp_path / "missing.csv")]) == 2


def test_cli_evaluate_on_written_predictions(tmp_path: Path, capsys) -> None:
    config = _small_config(tmp_path)
    run_pipeline(config, tmp_path / "out", mode="smoke")
    assert main(["evaluate", "--predictions", str(tmp_path / "out" / "predictions_test.csv"), "--threshold", "0.5"]) == 0
    printed = capsys.readouterr().out
    assert "macro_f1" in printed


def test_cli_route_without_model_exits_2(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({"output_dir": str(tmp_path / "nothing")}))
    assert main(["--config", str(config_path), "route", "--input", str(tmp_path / "x.csv")]) == 2


def test_cli_figures_without_outputs_exits_2(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({"output_dir": str(tmp_path / "nothing")}))
    assert main(["--config", str(config_path), "figures"]) == 2


def test_fetch_requires_acknowledgement_and_prints_terms_url(tmp_path: Path, capsys) -> None:
    code = main(["fetch", "--date-min", "2025-01-01", "--date-max", "2025-01-31", "--cache-dir", str(tmp_path)])
    assert code == 2
    out = capsys.readouterr()
    assert "consumerfinance.gov/complaint/data-use" in out.out
    assert "allegations" in out.out
    assert not list(tmp_path.iterdir())  # nothing was downloaded


def test_render_all_writes_three_svgs(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    run_pipeline(config, tmp_path / "out", mode="smoke")
    written = render_all(tmp_path / "out", tmp_path / "figs")
    assert len(written) == 3 and all(p.read_text().startswith("<svg") for p in written)


def test_default_config_loads_and_hashes() -> None:
    config = load_config()
    assert config.seed == 42 and config.target == "product"
    assert len(config.hash()) == 64


def test_review_log_written_when_items_are_routed(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    summary = run_pipeline(config, tmp_path / "out", mode="smoke")
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    routed = metrics["abstention"]["n_routed_to_review"]
    if routed:
        assert (tmp_path / "out" / "review_log.jsonl").exists()
        assert summary["review"]["events"] == routed
    else:
        assert summary["review"]["events"] == 0
    preds = pd.read_csv(tmp_path / "out" / "predictions_test.csv")
    assert set(preds["route"]).issubset({"auto", "human_review"})
