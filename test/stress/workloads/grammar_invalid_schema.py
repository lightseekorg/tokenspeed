"""Malformed schemas that xgrammar should reject during compile.

The expected behavior: the request finishes with an abort / error
response that surfaces the compile failure — not a server crash, not
a hang, not a successful "pretend it's valid" generation. A crash or
hang here is a hole in the GrammarManager error path (grammar future
failure propagation, cache-of-failure semantics).
"""

from __future__ import annotations

from typing import AsyncIterator

from ..client import ChatRequest
from . import register

# Each entry is a schema that SHOULD fail to compile. Variety so the
# backend's failure paths are exercised end-to-end, not just one code
# path. None of these should validate — the harness reports responses
# as "completed" (server returned something) but the request_error
# event is what we really care about.
_BAD_SCHEMAS = [
    # Unknown type.
    {"type": "not_a_json_type"},
    # Recursive ref with no base case.
    {"type": "object", "properties": {"x": {"$ref": "#"}}, "required": ["x"]},
    # Pattern that xgrammar's grammar compiler chokes on (nested quantifier
    # over a character class with anchors).
    {"type": "string", "pattern": "^((.+)+)+$"},
    # Contradictory constraints.
    {"type": "object", "properties": {}, "required": ["missing_field"]},
    # items requires type but gives invalid value.
    {"type": "array", "items": "not_a_schema"},
]


@register("grammar_invalid_schema")
async def grammar_invalid_schema(
    max_tokens: int = 64,
    stream: bool = False,
) -> AsyncIterator[ChatRequest]:
    idx = 0
    while True:
        schema = _BAD_SCHEMAS[idx % len(_BAD_SCHEMAS)]
        idx += 1
        yield ChatRequest(
            messages=[{"role": "user", "content": "Output a JSON object."}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=stream,
            extra={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bad_schema",
                        "schema": schema,
                        "strict": True,
                    },
                },
            },
            # Don't set validate_schema — we expect the server to
            # reject these at compile, not to generate valid output.
            workload=f"grammar_invalid_schema/variant{idx % len(_BAD_SCHEMAS)}",
        )
