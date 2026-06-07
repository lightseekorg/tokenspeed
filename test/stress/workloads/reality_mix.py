"""Realistic mixed-traffic workload for long-running soak tests.

Blends every axis the production server has to handle at once:

* **Grammar vs non-grammar** (60/40 default). Grammar requests rotate
  through the validated-passing schemas; non-grammar requests are plain
  text.
* **Prompt length**: short (100-2k tok), medium (2k-16k), long (16k-64k),
  huge (~100k), drawn by configurable weights.
* **Generation length**: short (100-400), medium (400-2k), long (2k-10k),
  also weighted.
* **Cached vs fresh prompts**: a pool of shared "system" prefixes exercises
  the radix/hi-cache on repeat traffic; the unique-filler variants bypass
  the cache and force fresh prefill.
* **Cancels** (configurable fraction) at random stages, to exercise the
  grammar-queued abort path + pending_aborts sweep under steady load.

Pair this with ``--arrival sawtooth`` so concurrency dynamically
ramps low↔high, hitting both idle-server and fully-loaded regimes
within the same run. Intended for 1h+ soak runs — the mix is
deliberately varied enough that any one pathological pattern
(compile starvation, matcher-state drift, cache invalidation race)
should surface within the run.
"""

from __future__ import annotations

import random
from typing import AsyncIterator, Dict, List

from ..client import ChatRequest
from . import register
from .grammar_schema import _PASSING, _PROMPTS, _SCHEMAS

# Token budget buckets. Chosen so the "huge" tier is rare enough to avoid
# KV exhaustion under high concurrency, but still hits the chunked-prefill
# path repeatedly across a 2h run.
_PROMPT_BUCKETS = [
    # (name, min_tokens, max_tokens, weight)
    ("short", 100, 2_000, 50),
    ("medium", 2_000, 16_000, 30),
    ("long", 16_000, 64_000, 15),
    ("huge", 90_000, 100_000, 5),
]

_GEN_BUCKETS = [
    # (name, min_tokens, max_tokens, weight)
    ("short", 100, 400, 45),
    ("medium", 400, 2_000, 30),
    ("long", 2_000, 6_000, 20),
    ("very_long", 6_000, 10_000, 5),
]

# Approx 4 chars per token (conservative lower bound across the gpt-oss
# tokenizer). Overshooting the char count a bit is fine — the server
# tokenizes the actual string.
_CHARS_PER_TOKEN = 4

# Shared-system-prompt pool. Every cached-variant request picks one of
# these, so the radix cache gets repeated hits across the run.
_SHARED_SYSTEM_PROMPTS = [
    "You are a careful, senior software engineer. Always cite file:line when possible.",
    "You are a meticulous code reviewer. Be concise. Flag only real defects.",
    "You are a helpful technical writer. Explain concepts crisply with concrete examples.",
    "You are a database architect. Prefer transactional correctness over convenience.",
]

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
    "paged",
    "radix",
    "scheduler",
    "kernel",
    "attention",
    "tensor",
    "throughput",
    "latency",
    "draft",
    "verify",
]


def _filler(n_chars: int, rng: random.Random) -> str:
    """Deterministic-ish word soup of approximately ``n_chars`` characters."""
    out: List[str] = []
    total = 0
    while total < n_chars:
        w = rng.choice(_FILLER_WORDS)
        out.append(w)
        total += len(w) + 1
    return " ".join(out)


def _weighted_choice(buckets, rng: random.Random):
    """Pick one (name, lo, hi) from ``buckets`` by weight (last field)."""
    total = sum(b[-1] for b in buckets)
    r = rng.uniform(0, total)
    acc = 0.0
    for b in buckets:
        acc += b[-1]
        if r <= acc:
            return b
    return buckets[-1]


@register("reality_mix")
async def reality_mix(
    grammar_fraction: float = 0.6,
    cancel_fraction: float = 0.15,
    cached_fraction: float = 0.5,
    seed: int = 0,
    max_tokens_cap: int = 0,
    very_long_weight: int = 5,
) -> AsyncIterator[ChatRequest]:
    rng = random.Random(seed)
    grammar_names = list(_PASSING)
    cancel_stages = ["queue", "prefill", "decode"]
    req_counter = 0

    # Optionally widen the "very_long" generation bucket so soak runs can
    # exercise the long-decode + KV-pressure regime that surfaces spec-dec
    # corruption bugs. Setting max_tokens_cap=65536 + very_long_weight=20
    # makes ~20% of requests generate up to 64k tokens.
    gen_buckets = list(_GEN_BUCKETS)
    if max_tokens_cap > 0:
        # Replace the very_long bucket with a stretched one.
        gen_buckets[-1] = ("very_long", 6_000, max_tokens_cap, very_long_weight)

    while True:
        req_counter += 1

        # --- pick buckets ---
        prompt_bucket = _weighted_choice(_PROMPT_BUCKETS, rng)
        gen_bucket = _weighted_choice(gen_buckets, rng)
        prompt_tokens = rng.randint(prompt_bucket[1], prompt_bucket[2])
        max_tokens = rng.randint(gen_bucket[1], gen_bucket[2])

        is_grammar = rng.random() < grammar_fraction
        is_cached = rng.random() < cached_fraction
        cancel_stage = (
            rng.choice(cancel_stages) if rng.random() < cancel_fraction else None
        )

        # --- build messages ---
        if is_cached:
            system_prompt = _SHARED_SYSTEM_PROMPTS[
                req_counter % len(_SHARED_SYSTEM_PROMPTS)
            ]
            # Cached path: long fixed system prompt + a short unique user turn
            # so the server's radix cache hits on the system portion.
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
                    "content": f"[req {req_counter}] Summarize the above briefly.",
                },
            ]
            cache_tag = "cached"
        else:
            # Fresh path: unique filler per request, cache miss every time.
            filler_chars = prompt_tokens * _CHARS_PER_TOKEN
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"[unique-{rng.randint(0, 2**31 - 1)}] "
                        "The following is a block of random words. "
                        "Write a long coherent story using at least twenty "
                        "of them; take your time.\n\n" + _filler(filler_chars, rng)
                    ),
                }
            ]
            cache_tag = "fresh"

        extra: Dict = {"seed": rng.randint(0, 2**31 - 1)}
        validate_schema = None
        grammar_tag = "plain"

        if is_grammar:
            name = grammar_names[req_counter % len(grammar_names)]
            schema = _SCHEMAS[name]
            # Tack the schema's guidance prompt onto the user message so
            # the model has a reasonable chance of producing valid JSON.
            messages.append({"role": "user", "content": _PROMPTS[name]})
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
            validate_schema = schema
            grammar_tag = f"grammar/{name}"
            # Grammar outputs must be bounded — 10k-token JSON isn't
            # useful and blows past any sensible schema response size.
            max_tokens = min(max_tokens, 800)

        yield ChatRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7 if not is_grammar else 0.0,
            stream=True,
            cancel_at_stage=cancel_stage,
            extra=extra,
            validate_schema=validate_schema,
            workload=(
                f"reality/{grammar_tag}/{cache_tag}/"
                f"prompt-{prompt_bucket[0]}/gen-{gen_bucket[0]}/"
                f"{cancel_stage or 'nocancel'}"
            ),
        )
