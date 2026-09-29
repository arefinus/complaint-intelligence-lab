"""Generate the authored synthetic CFPB-schema fixture (v1).

Authored synthetic demonstration data. Fictional identifiers. Not derived from any
person, company, account or dataset. Narratives are assembled from short generic
phrase templates and contain no personal data.

Usage:
    python tools/make_fixture.py [--rows 4000] [--seed 42] [--out examples/fixtures/v1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complaint_intelligence_lab.cfpb.synthetic import (  # noqa: E402
    FIXTURE_VERSION,
    generate_fixture,
    write_fixture,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "examples" / "fixtures" / "v1")
    args = parser.parse_args(argv)
    frame = generate_fixture(rows=args.rows, seed=args.seed)
    paths = write_fixture(frame, args.out, seed=args.seed)
    print(f"fixture {FIXTURE_VERSION}: {len(frame)} rows -> {paths['csv']}")
    print(f"FIXTURE.md -> {paths['card']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
