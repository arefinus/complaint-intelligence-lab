from __future__ import annotations

import json
from pathlib import Path

import pytest

from complaint_intelligence_lab.pipeline.review import AuditEvent, ReviewLog
from complaint_intelligence_lab.pipeline.sanitize import MAX_NARRATIVE_CHARS, TRUNCATION_MARKER, model_text, sanitize_text
from complaint_intelligence_lab.pipeline.summarise import TemplateSummariser

SCRIPT_PAYLOAD = 'Hello <script>alert("x")</script><img src=x onerror="alert(1)"> world & <b>bold</b>'


def test_script_tag_payload_is_neutralised() -> None:
    clean = sanitize_text(SCRIPT_PAYLOAD)
    lowered = clean.text.lower()
    assert "<script" not in lowered and "<img" not in lowered and "onerror" not in lowered
    assert "alert" not in lowered
    assert "<" not in clean.text and ">" not in clean.text
    assert "&amp;" in clean.text
    assert clean.text.startswith("Hello") and clean.text.endswith("bold")


def test_control_characters_are_removed() -> None:
    clean = sanitize_text("line\x00one\x07 two\ttabs\r\nkept")
    assert "\x00" not in clean.text and "\x07" not in clean.text
    assert clean.text == "lineone two tabs kept"


def test_long_text_is_bounded_and_flagged() -> None:
    clean = sanitize_text("word " * 5000)
    assert clean.truncated is True
    assert len(clean.text) <= MAX_NARRATIVE_CHARS
    assert clean.text.endswith(TRUNCATION_MARKER.strip())
    assert clean.original_length == 25000
    short = sanitize_text("short", max_chars=100)
    assert short.truncated is False
    with pytest.raises(ValueError):
        sanitize_text("x", max_chars=0)


def test_empty_narratives_are_handled() -> None:
    for value in ("", "   ", None, float("nan")):
        clean = sanitize_text(value)
        assert clean.text == "" and clean.has_narrative is False and clean.original_length == 0
        assert model_text(value) == ""


def test_sanitise_is_idempotent_on_plain_text() -> None:
    once = sanitize_text("plain text, nothing to escape.").text
    assert sanitize_text(once).text == once


def test_override_writes_one_audit_event_with_required_fields(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "log.jsonl", known_ids={"SYN-CMP-000001"})
    event = log.override("SYN-CMP-000001", "Mortgage", "Credit card", "reviewer-a", "text mentions a card")
    assert isinstance(event, AuditEvent)
    assert event.event_type == "override" and event.from_label == "Mortgage" and event.to_label == "Credit card"
    assert event.event_id.startswith("EVT-") and event.timestamp
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["complaint_id"] == "SYN-CMP-000001"


def test_each_override_appends_and_summary_counts(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "log.jsonl")
    log.override("a", "x", "y", "r1", "")
    log.override("b", "x", "z", "r2", "")
    log.confirm("c", "x", "r1")
    assert log.summary() == {"events": 3, "overrides": 2, "confirmations": 1, "distinct_complaints": 3, "distinct_reviewers": 2}
    reloaded = ReviewLog(tmp_path / "log.jsonl")
    assert [e.event_type for e in reloaded.events()] == ["override", "override", "confirm"]


def test_override_validation(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "log.jsonl", known_ids={"a"})
    with pytest.raises(KeyError):
        log.override("unknown", "x", "y", "r", "")
    with pytest.raises(ValueError, match="change the label"):
        log.override("a", "x", "x", "r", "")
    with pytest.raises(ValueError, match="reviewer_id"):
        log.override("a", "x", "y", "  ", "")


def test_override_reason_is_sanitised(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "log.jsonl")
    event = log.override("a", "x", "y", "r", SCRIPT_PAYLOAD)
    assert "<script" not in event.reason.lower() and "<" not in event.reason


def test_summariser_cites_spans_that_match_the_input() -> None:
    narrative = "A charge on my statement is for a purchase I did not make. I contacted the company. Please review."
    summary = TemplateSummariser().summarise(narrative, "Credit card", "Fees or interest")
    assert summary.backend == "template-v1"
    assert len(summary.citations) == 2
    for c in summary.citations:
        assert narrative[c.start : c.end] == c.text
        assert f"[{c.start}:{c.end}]" in summary.summary


def test_summariser_is_deterministic_and_handles_empty() -> None:
    s = TemplateSummariser()
    a = s.summarise("Some text here.", "P", "I")
    b = s.summarise("Some text here.", "P", "I")
    assert a == b
    empty = s.summarise("", "P", "I")
    assert empty.citations == () and "No narrative" in empty.summary


def test_summariser_escapes_markup_in_output() -> None:
    out = TemplateSummariser().summarise(SCRIPT_PAYLOAD, "<b>P</b>", "I")
    assert "<" not in out.summary
