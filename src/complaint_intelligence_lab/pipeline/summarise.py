"""Deterministic template summariser with cited input spans.

There is no language-model provider here. ``SummariserBackend`` is the adapter interface;
``TemplateSummariser`` is the only implementation and it only ever quotes the input.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Protocol

from .sanitize import sanitize_text

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


@dataclass(frozen=True)
class Citation:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Summary:
    summary: str
    citations: tuple[Citation, ...]
    backend: str

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "citations": [c.__dict__ for c in self.citations],
            "backend": self.backend,
        }


class SummariserBackend(Protocol):
    name: str

    def summarise(self, narrative: str, product: str, issue: str, max_sentences: int = 2) -> Summary: ...


class TemplateSummariser:
    """Quote the first sentences of the sanitised narrative and cite their offsets."""

    name = "template-v1"

    def summarise(self, narrative: str, product: str, issue: str, max_sentences: int = 2) -> Summary:
        clean = sanitize_text(narrative)
        product_s = sanitize_text(product, 128).text or "unspecified product"
        issue_s = sanitize_text(issue, 256).text or "unspecified issue"
        if not clean.has_narrative:
            return Summary(
                summary=f"Complaint filed under {product_s} / {issue_s}. No narrative was provided.",
                citations=(),
                backend=self.name,
            )
        text = html.unescape(clean.text)
        citations: list[Citation] = []
        for match in _SENTENCE_RE.finditer(text):
            fragment = match.group(0).strip()
            if not fragment:
                continue
            start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
            end = start + len(fragment)
            citations.append(Citation(start=start, end=end, text=html.escape(fragment)))
            if len(citations) >= max_sentences:
                break
        quoted = " ".join(f"\"{c.text}\" [{c.start}:{c.end}]" for c in citations)
        summary = f"Complaint filed under {product_s} / {issue_s}. Consumer states: {quoted}"
        return Summary(summary=summary, citations=tuple(citations), backend=self.name)


def default_backend() -> SummariserBackend:
    return TemplateSummariser()
