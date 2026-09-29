"""CFPB Consumer Complaint Database column names used by this repository."""
from __future__ import annotations

# Structured fields (public schema names).
CFPB_COLUMNS: list[str] = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "state",
    "submitted_via",
    "company_response",
    "timely",
    "consumer_disputed",
]

# Free-text field. Excluded from API caches unless explicitly opted in.
NARRATIVE = "complaint_what_happened"

# Fields that are only known after a complaint has been routed and answered. They
# encode outcome information and must never be used as features.
POST_EVENT_FIELDS: frozenset[str] = frozenset({"company_response", "timely", "consumer_disputed"})

# Hierarchical children that encode their parent target almost deterministically.
CHILD_OF_TARGET: dict[str, str] = {"product": "sub_product", "issue": "sub_issue"}

# Fields that may be used as structured features when predicting product or issue.
INTAKE_STRUCTURED_FIELDS: tuple[str, ...] = ("state", "submitted_via", "company")

REQUIRED_MINIMUM: tuple[str, ...] = ("complaint_id", "date_received", "product", "issue")
