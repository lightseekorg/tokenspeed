"""Grammar-constrained workload that rotates through JSON schemas.

Each yielded request sets ``response_format: {"type": "json_schema", ...}``
AND a matching ``validate_schema`` so the client can verify the server's
output parses and conforms. The goal is to stress the xgrammar +
capturable-grammar pipeline under continuous batching: whenever batch
composition changes (requests finishing, new ones joining, ramping
concurrency) the matcher state for each live request must remain
correct.

Some schemas intentionally *trigger* greedy-whitespace-loop degeneration
on gpt-oss (pre-existing xgrammar behaviour); these are excluded from
the default rotation via ``only_passing=True`` so the validation signal
is actually useful.
"""

from __future__ import annotations

import random
from typing import AsyncIterator, Dict

from ..client import ChatRequest
from . import register

# Keep these in lockstep with the prompts so the model has a reasonable
# chance of producing a schema-valid response under greedy sampling.
_SCHEMAS: Dict[str, Dict] = {
    "person_basic": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 200},
            "email": {"type": "string"},
        },
        "required": ["name", "age", "email"],
        "additionalProperties": False,
    },
    "enum_color": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "color": {"type": "string", "enum": ["red", "green", "blue", "yellow"]},
            "available": {"type": "boolean"},
        },
        "required": ["id", "color", "available"],
        "additionalProperties": False,
    },
    "deep_nested": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "profile": {
                        "type": "object",
                        "properties": {
                            "bio": {"type": "string"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 4,
                            },
                        },
                        "required": ["bio", "tags"],
                        "additionalProperties": False,
                    },
                },
                "required": ["id", "profile"],
                "additionalProperties": False,
            },
        },
        "required": ["user"],
        "additionalProperties": False,
    },
    # Schemas below tend to trigger whitespace-loop degeneration on
    # gpt-oss-120b under greedy sampling. Included here for reference/
    # regression probing; NOT in the default rotation.
    "nested_address": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["street", "city", "country", "zip"],
                "additionalProperties": False,
            },
        },
        "required": ["name", "address"],
        "additionalProperties": False,
    },
}

_PROMPTS: Dict[str, str] = {
    # Prompts explicitly instruct the model to output JSON matching the
    # schema. Without this, gpt-oss tends to produce prose and then get
    # stuck in xgrammar's "whitespace-allowed-between-tokens" attractor
    # under greedy sampling.
    "person_basic": (
        "Output a JSON object describing a fictional person. Keys: "
        "name (string), age (integer 18-80), email (string). "
        "Respond with ONLY the JSON object, no prose."
    ),
    "enum_color": (
        "Output a JSON object describing a product. Keys: id (integer), "
        "color (one of red|green|blue|yellow), available (boolean). "
        "Respond with ONLY the JSON object."
    ),
    "deep_nested": (
        "Output a JSON object describing a user. Keys: user.id (integer), "
        "user.profile.bio (string), user.profile.tags (array of 2-4 strings). "
        "Respond with ONLY the JSON object."
    ),
    "nested_address": (
        "Output a JSON object describing a person. Keys: name (string), "
        "address.{street, city, country, zip} (all strings). "
        "Respond with ONLY the JSON object."
    ),
}

# Schemas the model can satisfy reliably on gpt-oss-120b under greedy
# sampling. Used when ``only_passing=True``.
_PASSING = ("person_basic", "enum_color", "deep_nested")


# Padding text used to inflate prompts past the chunked-prefill threshold
# (server default is 16384). Repeated enough times, each request is
# guaranteed to span multiple EXTEND chunks, exercising the path where
# intermediate chunks must NOT advance the grammar matcher by their
# (garbage) sampled tokens.
_PAD_PARA = (
    "In the year 1923, a peculiar cartographer set out to map the "
    "uncharted wetlands of the northern delta, carrying only a brass "
    "compass, a hand-cranked calculator, and a well-worn leather journal "
    "whose pages were filled with sketches of migratory birds and brief "
    "notes on their seasonal feeding grounds. "
)


def _long_prompt_padding(target_tokens: int) -> str:
    # ~50 tokens per _PAD_PARA paragraph for gpt-oss; pad to roughly
    # target_tokens so we cross the chunked-prefill threshold. Exact
    # count doesn't matter.
    approx_tokens_per_para = 50
    reps = max(1, target_tokens // approx_tokens_per_para)
    return _PAD_PARA * reps


@register("grammar_schema")
async def grammar_schema(
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 0,
    stream: bool = True,
    only_passing: bool = True,
    long_prompt_tokens: int = 0,
) -> AsyncIterator[ChatRequest]:
    """Rotate through schemas, yielding one ChatRequest per step.

    ``only_passing`` restricts rotation to schemas that generate valid
    output under greedy sampling on the reference model (gpt-oss-120b).
    Set ``False`` to include the known-degenerate schemas as well — useful
    when A/B'ing baselines, but expect failures.

    ``long_prompt_tokens > 0`` pads each prompt with ~N additional tokens
    of filler narrative, forcing the server to run chunked prefill. Use
    this to exercise the per-chunk grammar-advance skip path.
    """
    rng = random.Random(seed)
    names = list(_PASSING if only_passing else _SCHEMAS.keys())
    padding = _long_prompt_padding(long_prompt_tokens) if long_prompt_tokens > 0 else ""
    idx = 0
    while True:
        name = names[idx % len(names)]
        idx += 1
        schema = _SCHEMAS[name]
        base_prompt = _PROMPTS[name]
        prompt = f"{padding}\n\n{base_prompt}" if padding else base_prompt
        # Tiny jitter in seed keeps temperature=0 determinism per request
        # but varies across requests so the scheduler sees distinct IDs.
        req_seed = rng.randint(0, 2**31 - 1)
        yield ChatRequest(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            extra={
                "seed": req_seed,
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
            workload=f"grammar_schema/{name}",
        )
