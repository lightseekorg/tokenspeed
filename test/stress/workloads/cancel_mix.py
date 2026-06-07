"""Stream + non-stream mix with mid-lifecycle cancellation.

Each yielded request is randomly assigned:
  - a format (streaming vs non-streaming)
  - a cancel plan (none / at queue / at prefill / at decode)

The goal is to hammer request-slot + KV-page release paths. A leak in any
stage shows up as growing memory or wedged slots over a long run.
"""

from __future__ import annotations

import random
from typing import AsyncIterator

from ..client import ChatRequest
from . import register

_QUESTIONS = [
    "Give me three tips for debugging a CUDA illegal memory access.",
    "Explain paged attention in one paragraph.",
    "What's the difference between prefill and decode phases in an LLM server?",
    "Write a haiku about memory fragmentation.",
    "List five reasons inter-token latency might spike under load.",
    "Describe the role of a radix tree in KV-cache sharing.",
]


@register("cancel_mix")
async def cancel_mix(
    cancel_fraction: float = 0.5,
    stream_fraction: float = 0.5,
    max_tokens: int = 256,
    seed: int = 0,
) -> AsyncIterator[ChatRequest]:
    rng = random.Random(seed)
    stages = ["queue", "prefill", "decode"]
    while True:
        q = rng.choice(_QUESTIONS)
        stream = rng.random() < stream_fraction
        cancel_stage = None
        if rng.random() < cancel_fraction:
            cancel_stage = rng.choice(stages)
            # Cancelling a non-streaming request at "decode" is ill-defined
            # (we never observe intermediate tokens). Fall back to a timed
            # cancel that lands mid-server-side-decode.
            if not stream and cancel_stage == "decode":
                cancel_stage = None
                yield ChatRequest(
                    messages=[{"role": "user", "content": q}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stream=False,
                    cancel_after_s=0.3,
                    workload="cancel_mix/nostream_timed",
                )
                continue
        yield ChatRequest(
            messages=[{"role": "user", "content": q}],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=stream,
            cancel_at_stage=cancel_stage,
            workload=f"cancel_mix/{'stream' if stream else 'nostream'}"
            f"/{cancel_stage or 'nocancel'}",
        )
