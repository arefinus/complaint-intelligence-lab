"""Schema validation for CFPB-shaped frames."""
from __future__ import annotations

import pandas as pd

from .fields import CFPB_COLUMNS, NARRATIVE, REQUIRED_MINIMUM


class SchemaError(ValueError):
    """Raised when a frame does not match the CFPB complaint schema."""


def validate_frame(frame: pd.DataFrame, require_narrative: bool = False) -> pd.DataFrame:
    """Validate column presence and basic types; return a normalised copy.

    Raises ``SchemaError`` naming every missing column. ``date_received`` is parsed to
    datetime; unparseable dates are an error because the chronological split depends on
    them.
    """
    missing = [c for c in REQUIRED_MINIMUM if c not in frame.columns]
    if require_narrative and NARRATIVE not in frame.columns:
        missing.append(NARRATIVE)
    if missing:
        raise SchemaError(
            "frame is missing required CFPB columns: "
            + ", ".join(missing)
            + ". Expected the public schema names: "
            + ", ".join(CFPB_COLUMNS + [NARRATIVE])
        )
    out = frame.copy()
    parsed = pd.to_datetime(out["date_received"], errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        raise SchemaError(f"{bad} rows have an unparseable date_received value")
    out["date_received"] = parsed
    if NARRATIVE not in out.columns:
        out[NARRATIVE] = ""
    out[NARRATIVE] = out[NARRATIVE].fillna("").astype(str)
    for col in CFPB_COLUMNS:
        if col in out.columns and col != "date_received":
            out[col] = out[col].fillna("").astype(str)
    if out["complaint_id"].duplicated().any():
        raise SchemaError("complaint_id values must be unique")
    return out
