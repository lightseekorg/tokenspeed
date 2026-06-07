"""Shared-prefix workload.

A small pool of long "system" prompts; each request reuses one of them and
appends a short unique user turn. Stresses radix cache + hicache: the first
request per prefix pays the full prefill, subsequent ones should hit cache.
"""

from __future__ import annotations

import random
from typing import AsyncIterator

from ..client import ChatRequest
from . import register

_LOREM = (
    "You are a careful, senior engineer. When answering, be concise, cite "
    "specific file paths and line numbers when relevant, and avoid speculation. "
)


def _make_prefix(seed: int, approx_chars: int) -> str:
    rng = random.Random(seed)
    # Deterministic filler so the same seed always yields the same prefix bytes.
    words = []
    while sum(len(w) + 1 for w in words) < approx_chars:
        words.append(
            rng.choice(
                [
                    "system",
                    "cache",
                    "prefill",
                    "decode",
                    "kernel",
                    "scheduler",
                    "attention",
                    "tensor",
                    "throughput",
                    "latency",
                    "paged",
                    "radix",
                    "speculative",
                    "draft",
                    "verify",
                    "retraction",
                    "fragmentation",
                ]
            )
        )
    return _LOREM + " ".join(words)


@register("shared_prefix")
async def shared_prefix(
    num_prefixes: int = 4,
    prefix_chars: int = 4000,
    max_tokens: int = 128,
    stream: bool = True,
    seed: int = 0,
) -> AsyncIterator[ChatRequest]:
    """Yields an infinite stream of requests that share one of `num_prefixes` prompts."""
    rng = random.Random(seed)
    prefixes = [_make_prefix(seed + i, prefix_chars) for i in range(num_prefixes)]
    counter = 0
    while True:
        idx = rng.randrange(num_prefixes)
        user = f"[{counter}] In one sentence, what is the capital of France?"
        yield ChatRequest(
            messages=[
                {"role": "system", "content": prefixes[idx]},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=stream,
            workload=f"shared_prefix#{idx}",
        )
        counter += 1
