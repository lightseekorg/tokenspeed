# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pure, GPU-free helpers for the compute_log_probs API (RL-plan Milestone 2).

The engine scores ``prompt + completion`` sequences by reusing the normal
generation path: a forward-only ``generate`` call with ``return_logprob=True``
and ``logprob_start_len=len(prompt)`` makes ``meta_info['input_token_logprobs']``
carry exactly the per-completion-token logprobs. These helpers build that call
and parse its result; ``Engine.compute_log_probs`` wires them to ``self.generate``.
"""

from __future__ import annotations

from typing import Any, Callable

DEFAULT_TEMPERATURE = 1.0
# Set to 1 if the GPU spike shows max_new_tokens=0 is unsupported; the single
# generated token lands in output_token_logprobs, never input_token_logprobs.
SCORE_MAX_NEW_TOKENS = 0


class InvalidSequenceError(ValueError):
    """Raised when a sequence cannot be scored (empty prompt or completion)."""


def validate_sequence(
    prompt_token_ids: list[int], completion_token_ids: list[int]
) -> None:
    if not prompt_token_ids:
        raise InvalidSequenceError(
            "prompt_token_ids must be non-empty: the first completion token needs "
            "prior context to be scored."
        )
    if not completion_token_ids:
        raise InvalidSequenceError(
            "completion_token_ids must be non-empty: nothing to score."
        )


def build_score_kwargs(
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    """Build the kwargs for an internal forward-only ``Engine.generate`` call."""
    validate_sequence(prompt_token_ids, completion_token_ids)
    # Note: compute_log_probs_core separately gates on temperature != 1.0 for v1;
    # the two checks serve different audiences (standalone helper vs. v1 core path),
    # so the divergence is intentional, not accidental.
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return {
        "input_ids": list(prompt_token_ids) + list(completion_token_ids),
        "sampling_params": {
            "max_new_tokens": SCORE_MAX_NEW_TOKENS,
            "temperature": temperature,
        },
        "return_logprob": True,
        # The logprob of completion token c_j is read from the logits at the
        # *preceding* position, so scoring starts one token before the
        # completion: logprob_start_len = len(prompt) - 1. The engine returns
        # one entry per position from there to the end — the M completion
        # logprobs followed by one trailing sampled-position entry (target token
        # -1) that extract_completion_logprobs drops. (Verified on B200.)
        "logprob_start_len": len(prompt_token_ids) - 1,
    }


def extract_completion_logprobs(
    meta_info: dict[str, Any], num_completion: int
) -> tuple[list[float], list[int]]:
    """Split ``meta_info['input_token_logprobs']`` into (log_probs, tokens).

    Each entry is a ``(logprob, token_id, text_or_None)`` tuple. The engine
    returns the M completion logprobs (aligned to ``logprob_start_len =
    len(prompt) - 1``) followed by one trailing sampled-position entry, so we
    keep the first ``num_completion``. Fewer than that means the logprob window
    was wrong (or input logprobs were not produced), so we fail loudly rather
    than return a silently-misaligned array.
    """
    entries = meta_info.get("input_token_logprobs")
    if not entries or len(entries) < num_completion:
        got = 0 if entries is None else len(entries)
        raise ValueError(
            f"expected at least {num_completion} completion logprobs, got {got}; "
            "check logprob_start_len alignment / input-logprob support."
        )
    entries = entries[:num_completion]
    log_probs = [float(e[0]) for e in entries]
    tokens = [int(e[1]) for e in entries]
    return log_probs, tokens


def compute_log_probs_core(
    sequences: list[dict[str, list[int]]],
    generate_fn: Callable[..., dict[str, Any]],
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, list[list[float]]]:
    """Score each sequence by calling ``generate_fn`` and parsing the result.

    ``generate_fn`` must have the signature of ``Engine.generate`` and return a
    single result dict (non-streaming) carrying ``meta_info``. v1 supports only
    ``temperature == 1.0`` (raw log_softmax), matching the engine's default
    ``temp_scaled_logprobs=False`` path; other values raise ``NotImplementedError``.
    """
    if temperature != DEFAULT_TEMPERATURE:
        raise NotImplementedError(
            "compute_log_probs v1 supports temperature=1.0 (raw log_softmax) only; "
            f"got {temperature}. Sampling-temperature scaling is a follow-up."
        )

    log_probs_out: list[list[float]] = []
    tokens_out: list[list[int]] = []
    for seq in sequences:
        prompt_ids = seq["prompt_token_ids"]
        completion_ids = seq["completion_token_ids"]
        kwargs = build_score_kwargs(prompt_ids, completion_ids, temperature)
        result = generate_fn(**kwargs)
        log_probs, tokens = extract_completion_logprobs(
            result["meta_info"], len(completion_ids)
        )
        log_probs_out.append(log_probs)
        tokens_out.append(tokens)
    return {"log_probs": log_probs_out, "tokens": tokens_out}
