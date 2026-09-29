from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from complaint_intelligence_lab.cfpb.schema import validate_frame  # noqa: E402
from complaint_intelligence_lab.cfpb.synthetic import generate_fixture  # noqa: E402
from complaint_intelligence_lab.config import LabConfig  # noqa: E402

SMALL_ROWS = 700


@pytest.fixture(scope="session")
def small_frame() -> pd.DataFrame:
    """A small synthetic frame generated in-process (seed 7 so it differs from the fixture)."""
    return validate_frame(generate_fixture(rows=SMALL_ROWS, seed=7), require_narrative=True)


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    path = ROOT / "examples" / "fixtures" / "v1" / "complaints_synthetic.csv"
    if not path.exists():
        pytest.skip("fixture not generated; run tools/make_fixture.py")
    return path


@pytest.fixture()
def config(tmp_path: Path) -> LabConfig:
    return LabConfig(output_dir=str(tmp_path / "out"))
