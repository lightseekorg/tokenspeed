"""Grammar-constrained requests for the ``burst`` arrival kind.

Pair with ``--arrival burst --burst-size N --burst-gap S``: the runner
dispatches ``N`` requests, blocks until every one completes (in-flight
→ 0), sleeps ``S`` seconds, then dispatches the next batch. This
exercises the scheduler's 0 → N → 0 transitions — admission from
empty and drain-to-empty without any in-flight requests keeping the
pipeline warm. Stale-state bugs in the grammar pipeline (e.g.
``prev_batch`` dangling when the last live request terminates) are the
kind of thing this surfaces.

The workload itself just emits grammar-constrained requests; the
drain-to-zero behavior is implemented in ``runner.run`` because only
the runner can observe in-flight count.
"""

from __future__ import annotations

import random
from typing import AsyncIterator

from ..client import ChatRequest
from . import register
from .grammar_schema import _PASSING, _PROMPTS, _SCHEMAS


@register("grammar_burst_drain")
async def grammar_burst_drain(
    max_tokens: int = 96,
    seed: int = 0,
) -> AsyncIterator[ChatRequest]:
    rng = random.Random(seed)
    names = list(_PASSING)
    idx = 0
    while True:
        name = names[idx % len(names)]
        idx += 1
        schema = _SCHEMAS[name]
        yield ChatRequest(
            messages=[{"role": "user", "content": _PROMPTS[name]}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
            extra={
                "seed": rng.randint(0, 2**31 - 1),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                        "strict": True,
                    },
                },
            },
            validate_schema=schema,
            workload=f"grammar_burst_drain/{name}",
        )
