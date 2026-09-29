"""Label-leakage guard for feature selection.

Refuses any feature that (a) is the target, (b) is a hierarchical child that encodes the
target, or (c) is only known after the complaint has been routed and answered.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..cfpb.fields import CHILD_OF_TARGET, INTAKE_STRUCTURED_FIELDS, NARRATIVE, POST_EVENT_FIELDS


class LeakageError(ValueError):
    """Raised when a requested feature would leak the target or a post-event outcome."""


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    structured: tuple[str, ...]
    use_narrative: bool = True

    def all_columns(self) -> tuple[str, ...]:
        cols = list(self.structured)
        if self.use_narrative:
            cols.append(NARRATIVE)
        return tuple(cols)


def check_features(target: str, structured: list[str] | tuple[str, ...]) -> FeatureSpec:
    """Validate a feature list against the target and return a frozen spec."""
    if target not in ("product", "issue"):
        raise LeakageError(f"unsupported target {target!r}; expected 'product' or 'issue'")
    problems: list[str] = []
    for col in structured:
        if col == target:
            problems.append(f"{col!r} is the target itself")
        elif col in POST_EVENT_FIELDS:
            problems.append(f"{col!r} is a post-event outcome field, only known after routing")
        elif CHILD_OF_TARGET.get(target) == col:
            problems.append(f"{col!r} is a child category that encodes the target {target!r}")
        elif col == NARRATIVE:
            problems.append("pass the narrative through use_narrative, not as a structured field")
        elif col not in INTAKE_STRUCTURED_FIELDS:
            problems.append(f"{col!r} is not an allowed intake field {INTAKE_STRUCTURED_FIELDS}")
    if problems:
        raise LeakageError("refused feature set: " + "; ".join(problems))
    return FeatureSpec(target=target, structured=tuple(structured))
