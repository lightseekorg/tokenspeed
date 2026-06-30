"""Regression: max_new_tokens must always resolve to a finite cap bounded by
the remaining context window — even when the caller omits it.

Root cause context (dashllm1.log, DP4+EP4 Qwen3.5-397B, MTP, CUDA graph ON):
both ``Req.check_finished`` and ``RequestState.check_finished`` decide the
length stop via ``len(output_ids) >= sampling_params.max_new_tokens``. When a
request arrives with ``max_new_tokens`` unset, that field stayed ``None`` all
the way down (io_struct fills ``sampling_params={}`` but never defaults
max_new_tokens), so the length check could not fire. The request then ran on
toward the per-request KV page-table capacity (``ceil(context_len /
page_size)``) — the exact boundary the trtllm/MTP page-allocation path is
sensitive to (engine-crash on overflow).

``InputProcessor.tokenize_one_request`` is the single construction site for
``TokenizedGenerateReqInput`` (hence the single source of truth feeding both
``check_finished`` paths). The fix resolves ``max_new_tokens`` there for BOTH
cases:
  * omitted (None)             -> context_len - input_token_num  (silent)
  * present but over the limit -> truncated to the same cap       (+warn)

These tests drive ``tokenize_one_request`` with a lightweight stub engine
(``input_ids`` provided so no tokenizer is needed) and assert the resolved
``max_new_tokens`` on the returned tokenized request.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.engine.input_processor import InputProcessor
from tokenspeed.runtime.engine.io_struct import GenerateReqInput

CONTEXT_LEN = 100


class _StubTokenizer:
    # SamplingParams.normalize() only calls tokenizer.encode() for stop strings;
    # with no stop strings it is never invoked, but provide it for safety.
    def encode(self, text, add_special_tokens=False):  # noqa: D401
        return [0] * len(text)


def _make_processor() -> InputProcessor:
    engine = SimpleNamespace(
        context_len=CONTEXT_LEN,
        is_generation=True,
        tokenizer=_StubTokenizer(),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        server_args=SimpleNamespace(
            reasoning_parser=None,
            enable_prefix_caching=False,
        ),
        model_config=SimpleNamespace(
            vocab_size=32000,
            is_multimodal_active=False,
        ),
    )
    return InputProcessor(engine)


def _tokenize(obj: GenerateReqInput):
    obj.normalize_batch_and_arguments()
    proc = _make_processor()
    return asyncio.run(proc.tokenize_one_request(obj))


def test_omitted_max_new_tokens_is_capped_to_context_room():
    # 10-token prompt, no max_new_tokens -> cap = context_len - 10 = 90.
    obj = GenerateReqInput(input_ids=list(range(10)), sampling_params={})
    out = _tokenize(obj)
    assert out.sampling_params.max_new_tokens == CONTEXT_LEN - 10


def test_explicit_over_limit_max_new_tokens_is_truncated():
    # prompt 10 + requested 200 would exceed context_len -> truncate to 90.
    obj = GenerateReqInput(
        input_ids=list(range(10)), sampling_params={"max_new_tokens": 200}
    )
    out = _tokenize(obj)
    assert out.sampling_params.max_new_tokens == CONTEXT_LEN - 10


def test_explicit_within_limit_max_new_tokens_is_preserved():
    # prompt 10 + requested 30 fits -> left untouched.
    obj = GenerateReqInput(
        input_ids=list(range(10)), sampling_params={"max_new_tokens": 30}
    )
    out = _tokenize(obj)
    assert out.sampling_params.max_new_tokens == 30


def test_resolved_cap_is_never_none():
    """The invariant both check_finished paths depend on: never None."""
    for sp in ({}, {"max_new_tokens": None}):
        obj = GenerateReqInput(input_ids=list(range(5)), sampling_params=dict(sp))
        out = _tokenize(obj)
        assert out.sampling_params.max_new_tokens is not None
        assert out.sampling_params.max_new_tokens == CONTEXT_LEN - 5


def test_batch_each_item_gets_own_dict():
    """normalize_batch_and_arguments must give each batch item its own dict."""
    obj = GenerateReqInput(
        input_ids=[list(range(90)), list(range(10)), list(range(5))],
        sampling_params={},
    )
    obj.normalize_batch_and_arguments()

    # Each item must have a distinct dict (no aliasing).
    assert obj.sampling_params[0] is not obj.sampling_params[1]
    assert obj.sampling_params[1] is not obj.sampling_params[2]

    proc = _make_processor()
    results = []
    for i in range(obj.batch_size):
        sub = obj[i]
        results.append(asyncio.run(proc.tokenize_one_request(sub)))

    assert results[0].sampling_params.max_new_tokens == CONTEXT_LEN - 90
    assert results[1].sampling_params.max_new_tokens == CONTEXT_LEN - 10
    assert results[2].sampling_params.max_new_tokens == CONTEXT_LEN - 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
