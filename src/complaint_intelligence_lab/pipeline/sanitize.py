"""Text sanitisation, length bounds and empty-narrative handling.

Any text that reaches a report, a routing list or a figure passes through
``sanitize_text`` first. Tags are removed, remaining angle brackets and ampersands are
escaped, control characters are dropped and the length is bounded.
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass

MAX_NARRATIVE_CHARS = 5000
TRUNCATION_MARKER = " [truncated]"

_TAG_RE = re.compile(r"<[^>]*>")
_SCRIPT_BLOCK_RE = re.compile(r"<\s*script\b.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CleanText:
    text: str
    has_narrative: bool
    truncated: bool
    original_length: int


def is_empty_narrative(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def sanitize_text(value: object, max_chars: int = MAX_NARRATIVE_CHARS) -> CleanText:
    """Return display-safe text with bounded length.

    Script blocks are removed with their contents, all other tags are stripped, and the
    result is HTML-escaped so that any residual ``<``, ``>`` or ``&`` is inert. The
    function is idempotent on its own output up to re-escaping of entities, which is why
    callers unescape before re-sanitising if they need to (they normally do not).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if is_empty_narrative(value):
        return CleanText(text="", has_narrative=False, truncated=False, original_length=0)
    raw = str(value)
    original_length = len(raw)
    text = html.unescape(raw)
    text = _SCRIPT_BLOCK_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _CONTROL_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    truncated = False
    if len(text) > max_chars:
        text = text[: max_chars - len(TRUNCATION_MARKER)].rstrip() + TRUNCATION_MARKER
        truncated = True
    text = html.escape(text, quote=True)
    return CleanText(
        text=text,
        has_narrative=bool(text),
        truncated=truncated,
        original_length=original_length,
    )


def model_text(value: object, max_chars: int = MAX_NARRATIVE_CHARS) -> str:
    """Plain (unescaped) bounded text for feature extraction.

    Escaping is a display concern; the vectoriser works on the unescaped words. Tags,
    scripts and control characters are still removed.
    """
    clean = sanitize_text(value, max_chars=max_chars)
    return html.unescape(clean.text)
