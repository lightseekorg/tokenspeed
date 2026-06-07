"""Retract-stress workload.

Designed to force frequent KV retract→host_writeback→host_loadback cycles,
which is the bug surface where spec-dec + KVStore corruption is suspected.

Each request asks for a long generation; high concurrency means total
KV demand exceeds the device pool, forcing retracts. When retracted
requests resume, they issue host loadbacks — that's the path that races
with drafter forward in the spec-dec hot path.

A fraction of requests share a small pool of long system prompts so
the radix cache also sees host hits on prefix prefill (a second loadback
trigger).
"""

from __future__ import annotations

import random
from typing import AsyncIterator

from ..client import ChatRequest
from . import register

_FILLER_WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
    "scheduler",
    "kernel",
    "attention",
    "tensor",
    "throughput",
    "latency",
    "draft",
    "verify",
    "retract",
]

_CHARS_PER_TOKEN = 4

_SHARED_PREFIXES = [
    "You are a long-context assistant tasked with summarizing technical content.",
    "You are an editor reviewing a manuscript and producing detailed feedback.",
    "You are a teacher creating an extensive lesson plan with examples.",
    "You are a researcher producing a comprehensive literature review.",
]


def _filler(n_chars: int, rng: random.Random) -> str:
    out = []
    total = 0
    while total < n_chars:
        w = rng.choice(_FILLER_WORDS)
        out.append(w)
        total += len(w) + 1
    return " ".join(out)


@register("retract_stress")
async def retract_stress(
    cached_fraction: float = 0.3,
    prompt_min_tokens: int = 8000,
    prompt_max_tokens: int = 32000,
    gen_min_tokens: int = 8000,
    gen_max_tokens: int = 32000,
    seed: int = 0,
    cancel_fraction: float = 0.0,
) -> AsyncIterator[ChatRequest]:
    rng = random.Random(seed)
    counter = 0
    cancel_stages = ["queue", "prefill", "decode"]
    while True:
        counter += 1
        prompt_tokens = rng.randint(prompt_min_tokens, prompt_max_tokens)
        max_tokens = rng.randint(gen_min_tokens, gen_max_tokens)
        is_cached = rng.random() < cached_fraction
        cancel_stage = (
            rng.choice(cancel_stages) if rng.random() < cancel_fraction else None
        )

        if is_cached:
            system_prompt = _SHARED_PREFIXES[counter % len(_SHARED_PREFIXES)]
            filler_chars = max(
                prompt_tokens * _CHARS_PER_TOKEN - len(system_prompt), 64
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prompt + " " + _filler(filler_chars, rng),
                },
                {
                    "role": "user",
                    "content": f"[req {counter}] Write a long, coherent extension; take your time.",
                },
            ]
            tag = "cached"
        else:
            filler_chars = prompt_tokens * _CHARS_PER_TOKEN
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"[unique-{rng.randint(0, 2**31 - 1)}] "
                        "The text below is a block of words. Write a long, "
                        "coherent narrative incorporating at least thirty of them. "
                        "Take your time and produce a thorough essay.\n\n"
                        + _filler(filler_chars, rng)
                    ),
                },
            ]
            tag = "fresh"

        yield ChatRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
            workload=f"retract_stress/{tag}",
            cancel_at_stage=cancel_stage,
            extra={"seed": rng.randint(0, 2**31 - 1)},
        )
