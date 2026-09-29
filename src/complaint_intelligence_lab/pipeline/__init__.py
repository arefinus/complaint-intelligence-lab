"""Deduplication, splitting, leakage guard, features, model, abstention and reporting."""
from __future__ import annotations

from .abstain import choose_threshold, coverage_curve, route
from .dedup import DedupResult, find_duplicates
from .features import Featuriser, TfidfConfig
from .leakage import FeatureSpec, LeakageError, check_features
from .metrics import classification_report
from .model import ComplaintClassifier, ModelConfig
from .review import AuditEvent, ReviewLog
from .sanitize import CleanText, sanitize_text
from .split import SplitResult, assert_no_group_crosses, chronological_split
from .summarise import TemplateSummariser
from .trends import TrendConfig, trend_report

__all__ = [
    "AuditEvent",
    "CleanText",
    "ComplaintClassifier",
    "DedupResult",
    "FeatureSpec",
    "Featuriser",
    "LeakageError",
    "ModelConfig",
    "ReviewLog",
    "SplitResult",
    "TemplateSummariser",
    "TfidfConfig",
    "TrendConfig",
    "assert_no_group_crosses",
    "check_features",
    "choose_threshold",
    "chronological_split",
    "classification_report",
    "coverage_curve",
    "find_duplicates",
    "route",
    "sanitize_text",
    "trend_report",
]
