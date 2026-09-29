"""Reviewer override log. Every override appends one audit event; nothing is edited."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .sanitize import sanitize_text

MAX_REASON_CHARS = 500


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    complaint_id: str
    reviewer_id: str
    from_label: str
    to_label: str
    reason: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class ReviewLog:
    """Append-only JSONL log of reviewer decisions."""

    def __init__(self, path: Path, known_ids: set[str] | None = None) -> None:
        self.path = path
        self.known_ids = known_ids
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, event: AuditEvent) -> AuditEvent:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.to_json() + "\n")
        return event

    def override(self, complaint_id: str, predicted: str, corrected: str, reviewer_id: str, reason: str) -> AuditEvent:
        if self.known_ids is not None and complaint_id not in self.known_ids:
            raise KeyError(f"unknown complaint_id {complaint_id!r}")
        if not reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        if predicted == corrected:
            raise ValueError("an override must change the label; use confirm() to record agreement")
        return self._append(self._event("override", complaint_id, reviewer_id, predicted, corrected, reason))

    def confirm(self, complaint_id: str, predicted: str, reviewer_id: str, reason: str = "") -> AuditEvent:
        if self.known_ids is not None and complaint_id not in self.known_ids:
            raise KeyError(f"unknown complaint_id {complaint_id!r}")
        return self._append(self._event("confirm", complaint_id, reviewer_id, predicted, predicted, reason))

    @staticmethod
    def _event(kind: str, complaint_id: str, reviewer_id: str, from_label: str, to_label: str, reason: str) -> AuditEvent:
        return AuditEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=kind,
            complaint_id=str(complaint_id),
            reviewer_id=sanitize_text(reviewer_id, 64).text,
            from_label=sanitize_text(from_label, 128).text,
            to_label=sanitize_text(to_label, 128).text,
            reason=sanitize_text(reason, MAX_REASON_CHARS).text,
        )

    def events(self) -> Iterator[AuditEvent]:
        if not self.path.exists():
            return iter(())
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return (AuditEvent(**json.loads(line)) for line in lines if line.strip())

    def summary(self) -> dict[str, int]:
        events = list(self.events())
        overrides = [e for e in events if e.event_type == "override"]
        return {
            "events": len(events),
            "overrides": len(overrides),
            "confirmations": sum(1 for e in events if e.event_type == "confirm"),
            "distinct_complaints": len({e.complaint_id for e in events}),
            "distinct_reviewers": len({e.reviewer_id for e in events}),
        }
