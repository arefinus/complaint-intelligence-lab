"""Hand-written SVG figures from demo outputs. No plotting library required."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

W, H = 720, 420
M = {"l": 70, "r": 30, "t": 40, "b": 60}
PALETTE = ["#1f5fbf", "#d1495b", "#3a8f5c", "#8f5fbf", "#d98c1f", "#2a9d9d", "#6b6b6b", "#b3306e"]


def _svg(body: str, title: str, subtitle: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="12">\n'
        f'<rect width="{W}" height="{H}" fill="white"/>\n'
        f'<text x="{M["l"]}" y="22" font-size="15" font-weight="bold">{title}</text>\n'
        f'<text x="{M["l"]}" y="36" fill="#555">{subtitle}</text>\n'
        + body
        + "</svg>\n"
    )


def _scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def coverage_accuracy_svg(metrics: dict[str, Any]) -> str:
    curve = [p for p in metrics["abstention"]["curve"] if p["accuracy_on_covered"] is not None]
    thr = metrics["abstention"]["threshold"]
    x0, x1, y0, y1 = M["l"], W - M["r"], H - M["b"], M["t"]
    body = f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>\n'
    for k in range(6):
        v = k / 5
        y = _scale(v, 0, 1, y0, y1)
        x = _scale(v, 0, 1, x0, x1)
        body += f'<text x="{x0 - 8}" y="{y + 4}" text-anchor="end" fill="#555">{v:.1f}</text>'
        body += f'<text x="{x}" y="{y0 + 16}" text-anchor="middle" fill="#555">{v:.1f}</text>\n'
    body += f'<text x="{(x0 + x1) / 2}" y="{H - 18}" text-anchor="middle">confidence threshold</text>'
    body += f'<text transform="translate(18,{(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">coverage / accuracy on covered</text>\n'
    for key, colour, label in (("coverage", PALETTE[0], "coverage"), ("accuracy_on_covered", PALETTE[1], "accuracy on covered")):
        pts = " ".join(f"{_scale(p['threshold'], 0, 1, x0, x1):.1f},{_scale(p[key], 0, 1, y0, y1):.1f}" for p in curve)
        body += f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>\n'
    xt = _scale(thr, 0, 1, x0, x1)
    body += f'<line x1="{xt}" y1="{y0}" x2="{xt}" y2="{y1}" stroke="#999" stroke-dasharray="4 3"/>'
    body += f'<text x="{xt + 4}" y="{y1 + 12}" fill="#555">selected on valid: {thr:.2f}</text>\n'
    body += f'<rect x="{x1 - 190}" y="{y1}" width="12" height="12" fill="{PALETTE[0]}"/><text x="{x1 - 172}" y="{y1 + 10}">coverage</text>'
    body += f'<rect x="{x1 - 190}" y="{y1 + 18}" width="12" height="12" fill="{PALETTE[1]}"/><text x="{x1 - 172}" y="{y1 + 28}">accuracy on covered</text>\n'
    return _svg(body, "Coverage and accuracy versus abstention threshold (test split)",
                "demo on synthetic fixture v1, seed 42; not a benchmark result")


def trends_svg(trends: dict[str, Any]) -> str:
    rows = trends["monthly_by_product"]
    months = sorted({r["month"] for r in rows})
    products = sorted({r["product"] for r in rows})
    table = {(r["month"], r["product"]): r["count"] for r in rows}
    ymax = max(table.values()) if table else 1
    x0, x1, y0, y1 = M["l"], W - M["r"] - 170, H - M["b"], M["t"]
    body = f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>\n'
    for k in range(5):
        v = ymax * k / 4
        y = _scale(v, 0, ymax, y0, y1)
        body += f'<text x="{x0 - 8}" y="{y + 4}" text-anchor="end" fill="#555">{v:.0f}</text>\n'
    for i, m in enumerate(months):
        if i % 3 == 0:
            x = _scale(i, 0, max(len(months) - 1, 1), x0, x1)
            body += f'<text x="{x}" y="{y0 + 16}" text-anchor="middle" fill="#555" font-size="10">{m}</text>\n'
    for j, p in enumerate(products):
        colour = PALETTE[j % len(PALETTE)]
        pts = " ".join(
            f"{_scale(i, 0, max(len(months) - 1, 1), x0, x1):.1f},{_scale(table.get((m, p), 0), 0, ymax, y0, y1):.1f}"
            for i, m in enumerate(months)
        )
        body += f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.8"/>\n'
        ly = y1 + 14 * j
        body += f'<rect x="{x1 + 12}" y="{ly}" width="10" height="10" fill="{colour}"/><text x="{x1 + 26}" y="{ly + 9}" font-size="10">{p}</text>\n'
    body += f'<text x="{(x0 + x1) / 2}" y="{H - 18}" text-anchor="middle">month received</text>'
    body += f'<text transform="translate(18,{(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">complaints per month</text>\n'
    return _svg(body, "Monthly complaint volume by product (synthetic fixture)",
                "allegations as submitted, not findings; not a company quality ranking")


def landscape_svg(landscape: dict[str, Any]) -> str:
    pts = landscape["points"]
    labels = sorted({p["label"] for p in pts})
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    x0, x1, y0, y1 = M["l"], W - M["r"] - 170, H - M["b"], M["t"]
    body = f'<rect x="{x0}" y="{y1}" width="{x1 - x0}" height="{y0 - y1}" fill="none" stroke="#ccc"/>\n'
    for p in pts:
        colour = PALETTE[labels.index(p["label"]) % len(PALETTE)]
        cx = _scale(p["x"], min(xs), max(xs), x0 + 6, x1 - 6)
        cy = _scale(p["y"], min(ys), max(ys), y0 - 6, y1 + 6)
        body += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{colour}" fill-opacity="0.7"/>\n'
    for j, lab in enumerate(labels):
        ly = y1 + 14 * j
        body += f'<rect x="{x1 + 12}" y="{ly}" width="10" height="10" fill="{PALETTE[j % len(PALETTE)]}"/><text x="{x1 + 26}" y="{ly + 9}" font-size="10">{lab}</text>\n'
    body += f'<text x="{(x0 + x1) / 2}" y="{H - 18}" text-anchor="middle">SVD component 1 (no unit)</text>'
    body += f'<text transform="translate(18,{(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">SVD component 2 (no unit)</text>\n'
    return _svg(body, "Topic landscape: TruncatedSVD of TF-IDF (train rows)", landscape["label"])


def render_all(output_dir: Path, figures_dir: Path) -> list[Path]:
    required = ["metrics.json", "trends.json", "topic_landscape.json"]
    missing = [f for f in required if not (output_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing demo outputs {missing} in {output_dir}; run `python -m complaint_intelligence_lab demo` first"
        )
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    trends = json.loads((output_dir / "trends.json").read_text(encoding="utf-8"))
    landscape = json.loads((output_dir / "topic_landscape.json").read_text(encoding="utf-8"))
    written = []
    for name, svg in (
        ("coverage_accuracy.svg", coverage_accuracy_svg(metrics)),
        ("monthly_trends.svg", trends_svg(trends)),
        ("topic_landscape.svg", landscape_svg(landscape)),
    ):
        path = figures_dir / name
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    return written
