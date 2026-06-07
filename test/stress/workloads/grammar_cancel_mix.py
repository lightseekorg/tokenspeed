"""Grammar-constrained requests crossed with cancellations.

Mixes the `grammar_schema` workload's body with `cancel_mix`'s timing:
some requests stream and are cancelled at queue / prefill / decode,
some complete normally. Exercises the abort paths that must both cancel
a still-compiling grammar future AND release any request resources
cleanly — including the known FSM race on in-flight streaming cancels.
"""

from __future__ import annotations

import random
from typing import AsyncIterator

from ..client import ChatRequest
from . import register
from .grammar_schema import _PASSING, _PROMPTS, _SCHEMAS


@register("grammar_cancel_mix")
async def grammar_cancel_mix(
    cancel_fraction: float = 0.5,
    max_tokens: int = 128,
    seed: int = 0,
) -> AsyncIterator[ChatRequest]:
    rng = random.Random(seed)
    stages = ["queue", "prefill", "decode"]
    names = list(_PASSING)
    idx = 0
    while True:
        name = names[idx % len(names)]
        idx += 1
        schema = _SCHEMAS[name]
        stream = True  # cancellation is only meaningful mid-stream
        cancel_stage = None
        if rng.random() < cancel_fraction:
            cancel_stage = rng.choice(stages)
        yield ChatRequest(
            messages=[{"role": "user", "content": _PROMPTS[name]}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=stream,
            cancel_at_stage=cancel_stage,
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
            workload=f"grammar_cancel_mix/{name}/{cancel_stage or 'nocancel'}",
        )
