"""CFPB schema, API client and synthetic fixture generator."""
from __future__ import annotations

from .client import API_BASE, DATA_USE_URL, PRIVACY_NOTE, CfpbClient, FetchRequest
from .fields import (
    CFPB_COLUMNS,
    CHILD_OF_TARGET,
    INTAKE_STRUCTURED_FIELDS,
    NARRATIVE,
    POST_EVENT_FIELDS,
)
from .schema import SchemaError, validate_frame

__all__ = [
    "API_BASE",
    "DATA_USE_URL",
    "PRIVACY_NOTE",
    "CfpbClient",
    "FetchRequest",
    "CFPB_COLUMNS",
    "CHILD_OF_TARGET",
    "INTAKE_STRUCTURED_FIELDS",
    "NARRATIVE",
    "POST_EVENT_FIELDS",
    "SchemaError",
    "validate_frame",
]
