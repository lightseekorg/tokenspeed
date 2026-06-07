"""Long-context / long-generation workload.

NOTE: Retraction is not yet supported by the engine. This workload is
deliberately tuned to stay *below* the retraction threshold -- it exists to
stress long prompts and long generations, not eviction. Keep prompts +
generations per in-flight request well under the KV budget.
Tune prompt_chars_max / max_tokens_max / concurrency accordingly when
invoking.
"""

from __future__ import annotations

import random
from typing import AsyncIterator

from ..client import ChatRequest
from . import register


def _filler(n_chars: int, rng: random.Random) -> str:
    words = [
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
    ]
    out: list[str] = []
    total = 0
    while total < n_chars:
        w = rng.choice(words)
        out.append(w)
        total += len(w) + 1
    return " ".join(out)


@register("long_context")
async def long_context(
    prompt_chars_min: int = 2000,
    prompt_chars_max: int = 8000,
    max_tokens_min: int = 128,
    max_tokens_max: int = 512,
    stream: bool = True,
    seed: int = 0,
) -> AsyncIterator[ChatRequest]:
    rng = random.Random(seed)
    while True:
        prompt_chars = rng.randint(prompt_chars_min, prompt_chars_max)
        max_tokens = rng.randint(max_tokens_min, max_tokens_max)
        filler = _filler(prompt_chars, rng)
        yield ChatRequest(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Below is a collection of random words. Please write a "
                        "long coherent story that incorporates at least twenty "
                        "of them. Take your time.\n\n" + filler
                    ),
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=stream,
            workload="long_context",
        )
