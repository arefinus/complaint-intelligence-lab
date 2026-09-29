"""Render docs/figures/*.svg from examples/output/ (no plotting library needed).

Usage: python tools/render_figures.py [--out examples/output] [--figures docs/figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complaint_intelligence_lab.figures import render_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "examples" / "output")
    parser.add_argument("--figures", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args(argv)
    try:
        written = render_all(args.out, args.figures)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
