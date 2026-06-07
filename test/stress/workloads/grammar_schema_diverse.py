"""Many DISTINCT grammar keys hammering the compile cache.

Each request uses a schema whose shape varies by rid — forcing the
grammar backend to compile a fresh grammar for most requests (rather
than reusing a cache hit). This exercises:
  - the GrammarManager queue under sustained compile load
  - per-key compile timeouts, retry / escalation
  - the backend cache under eviction pressure
  - admission sync across attn_tp ranks (all ranks must agree on which
    keys are ready)

A correctness regression here typically shows up as requests stuck
forever in the queue, or invalid-JSON responses when the bitmask is
built from the wrong grammar instance.
"""

from __future__ import annotations

import random
from typing import AsyncIterator

from ..client import ChatRequest
from . import register


def _make_schema(variant: int) -> dict:
    """Emit a schema that differs structurally by variant — distinct
    grammar keys (string != structural equivalence) so the backend
    cannot coalesce on textual identity.
    """
    n_items = 2 + (variant % 5)
    required = [f"field_{i}" for i in range(n_items)]
    props = {
        f"field_{i}": (
            {"type": "integer", "minimum": 0, "maximum": 1000 + variant}
            if i % 2 == 0
            else {"type": "string"}
        )
        for i in range(n_items)
    }
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


@register("grammar_schema_diverse")
async def grammar_schema_diverse(
    max_tokens: int = 128,
    seed: int = 0,
    num_variants: int = 64,
) -> AsyncIterator[ChatRequest]:
    rng = random.Random(seed)
    variant = 0
    while True:
        v = variant % num_variants
        variant += 1
        schema = _make_schema(v)
        req_fields = ", ".join(schema["required"])
        prompt = (
            f"Output a JSON object with the following required keys: "
            f"{req_fields}. Respond with ONLY the JSON object."
        )
        yield ChatRequest(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
            extra={
                "seed": rng.randint(0, 2**31 - 1),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"diverse_v{v}",
                        "schema": schema,
                        "strict": True,
                    },
                },
            },
            validate_schema=schema,
            workload=f"grammar_schema_diverse/v{v}",
        )
