"""Built-in output auditors.

Registered via import side-effect from ``audits/__init__.py``. Each is a pure
function of a ``ResponseRecord``. None of these are perfect oracles — gibberish
detection in particular is heuristic — so most findings are ``warn`` and meant
as a "look here" signal, not proof of a bug. The high-confidence ones
(``empty_content``, JSON parse failure) are ``error``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import List

from . import (
    SEVERITY_ERROR,
    SEVERITY_WARN,
    Finding,
    ResponseRecord,
    register,
)

# A response billed at least this many completion tokens but carrying no
# visible text is almost certainly the "special tokens / empty deltas" bug.
_EMPTY_CONTENT_MIN_TOKENS = 5
# Only flag a visible/reported token gap when the server reported a lot and the
# client saw drastically fewer — keeps false positives down for servers that
# pack several tokens into one streaming delta.
_LOW_VISIBLE_MIN_REPORTED = 64
_LOW_VISIBLE_RATIO = 0.1

_WS_RUN_RE = re.compile(r"\s{80,}")  # 80+ consecutive whitespace chars
_NONWS_RE = re.compile(r"\S")

_EXPECTED_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "eos"}


@register("json_schema")
def json_schema(record: ResponseRecord) -> List[Finding]:
    """Validate the response parses as JSON and conforms to its schema.

    Only runs when the request carried a ``validate_schema`` (grammar
    workloads). Subsumes the harness's original inline schema check.
    """
    if record.validate_schema is None:
        return []
    if record.content_truncated:
        # We deliberately dropped part of the content; a parse failure here
        # would be our fault, not the server's. Can't judge.
        return []
    try:
        parsed = json.loads(record.content)
    except json.JSONDecodeError as e:
        return [
            Finding(
                check="json_schema",
                severity=SEVERITY_ERROR,
                detail=f"invalid_json: {e}"[:200],
            )
        ]
    try:
        import jsonschema  # type: ignore
    except ImportError:
        # JSON parsed; we just can't check conformance without the dep.
        return []
    try:
        jsonschema.validate(instance=parsed, schema=record.validate_schema)
    except jsonschema.ValidationError as e:
        return [
            Finding(
                check="json_schema",
                severity=SEVERITY_ERROR,
                detail=f"schema_violation path={list(e.path)} {e.message}"[:200],
            )
        ]
    return []


@register("length_consistency")
def length_consistency(record: ResponseRecord) -> List[Finding]:
    """Cross-check server-reported completion tokens against visible output."""
    findings: List[Finding] = []
    reported = record.reported_completion_tokens
    if reported >= _EMPTY_CONTENT_MIN_TOKENS and not record.content.strip():
        findings.append(
            Finding(
                check="length_consistency",
                severity=SEVERITY_ERROR,
                detail=(
                    f"empty_content: server billed {reported} completion tokens "
                    "but the response carried no visible text"
                ),
                value=float(reported),
            )
        )
    elif (
        record.stream
        and reported >= _LOW_VISIBLE_MIN_REPORTED
        and record.observed_visible_tokens < _LOW_VISIBLE_RATIO * reported
        and record.content.strip()
    ):
        findings.append(
            Finding(
                check="length_consistency",
                severity=SEVERITY_WARN,
                detail=(
                    f"low_visible_ratio: reported={reported} but only "
                    f"{record.observed_visible_tokens} visible content chunks"
                ),
                value=float(record.observed_visible_tokens),
            )
        )
    return findings


@register("degeneration")
def degeneration(record: ResponseRecord) -> List[Finding]:
    """Heuristic gibberish / decode-degeneration detector (warn-level)."""
    content = record.content
    if not content:
        return []
    findings: List[Finding] = []

    # Runaway whitespace — the classic xgrammar greedy-whitespace attractor.
    ws_run = _WS_RUN_RE.search(content)
    nonws = len(_NONWS_RE.findall(content))
    ws_frac = 1.0 - (nonws / len(content)) if content else 0.0
    if ws_run is not None or (len(content) > 200 and ws_frac > 0.6):
        findings.append(
            Finding(
                check="degeneration",
                severity=SEVERITY_WARN,
                detail=f"whitespace_loop: ws_fraction={ws_frac:.2f}",
                value=ws_frac,
            )
        )

    # Token-level repetition / low lexical diversity.
    words = content.split()
    if len(words) >= 30:
        unique_ratio = len(set(words)) / len(words)
        trigrams = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
        top_trigram, top_count = trigrams.most_common(1)[0]
        if unique_ratio < 0.15 or top_count > 10:
            findings.append(
                Finding(
                    check="degeneration",
                    severity=SEVERITY_WARN,
                    detail=(
                        f"repetition: unique_ratio={unique_ratio:.2f} "
                        f"top_trigram x{top_count}"
                    ),
                    value=unique_ratio,
                )
            )
    return findings


@register("finish_reason")
def finish_reason(record: ResponseRecord) -> List[Finding]:
    """Flag finish_reason values outside the known-good set."""
    fr = record.finish_reason
    if fr is not None and fr not in _EXPECTED_FINISH_REASONS:
        return [
            Finding(
                check="finish_reason",
                severity=SEVERITY_WARN,
                detail=f"unexpected_finish_reason={fr!r}",
            )
        ]
    return []
