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

"""CPU unit tests for the Score API and the decision adapter layer.

Covers the score plumbing that needs no GPU: SamplingParams validation
and wire roundtrip, the gather/softmax helpers, SamplingBatchInfo
slicing, the decision adapter compile/extract logic, the adapter
registry, and the Engine.score/async_decision join logic (with a stub
frontend). The scheduler-side readout is covered by
test_score_output_processor.py; end-to-end model tests live in
test_score_e2e_gpu.py.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys

import msgspec
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.decision import (
    STYLE_FUSED_CHOICE,
    STYLE_POINTWISE_YESNO,
    DecisionRequest,
    GenericDecisionAdapter,
    get_decision_adapter,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.sampling.score_utils import (
    build_score_label_ids,
    finalize_score_row,
    gather_score_logprobs,
)

# ---------------------------------------------------------------------------
# SamplingParams validation
# ---------------------------------------------------------------------------


def _verify(params: SamplingParams) -> None:
    params.verify(vocab_size=100)


def test_score_params_valid():
    _verify(
        SamplingParams(
            max_new_tokens=0, score_label_token_ids=[5, 9], score_apply_softmax=True
        )
    )


def test_score_params_require_zero_max_new_tokens():
    with pytest.raises(ValueError, match="requires max_new_tokens=0"):
        _verify(SamplingParams(max_new_tokens=8, score_label_token_ids=[5, 9]))


def test_score_params_reject_empty_labels():
    with pytest.raises(ValueError, match="non-empty"):
        _verify(SamplingParams(max_new_tokens=0, score_label_token_ids=[]))


def test_score_params_reject_out_of_vocab_label():
    with pytest.raises(ValueError, match=r"\[0, 99\]"):
        _verify(SamplingParams(max_new_tokens=0, score_label_token_ids=[100]))


@pytest.mark.parametrize("apply_softmax", (True, False))
def test_score_apply_softmax_requires_labels(apply_softmax):
    with pytest.raises(ValueError, match="score_apply_softmax"):
        _verify(SamplingParams(score_apply_softmax=apply_softmax))


@pytest.mark.parametrize("token_id", (True, 1.5, "1"))
def test_score_params_reject_non_integer_labels(token_id):
    with pytest.raises(ValueError, match="integer token IDs"):
        _verify(
            SamplingParams(
                max_new_tokens=0,
                score_label_token_ids=[token_id],
                score_apply_softmax=False,
            )
        )


def test_score_params_reject_duplicate_labels():
    with pytest.raises(ValueError, match="unique token IDs"):
        _verify(
            SamplingParams(
                max_new_tokens=0,
                score_label_token_ids=[1, 1],
                score_apply_softmax=False,
            )
        )


def test_score_params_require_explicit_normalization():
    with pytest.raises(ValueError, match="explicitly True or False"):
        _verify(SamplingParams(max_new_tokens=0, score_label_token_ids=[1]))


def test_score_gather_uses_emitted_prefill_prefix():
    # One completed prefill, one skipped incomplete prefill, then decode logits.
    logits = torch.tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    labels = torch.tensor([[0, 2], [1, 2]])
    out = gather_score_logprobs(logits, labels, 1, logprob_order="torch")
    assert out.shape == (1, 2)
    assert torch.allclose(
        out, torch.log_softmax(logits[:1], dim=-1).gather(-1, labels[:1])
    )


def test_score_gather_skips_all_incomplete_prefills():
    # The only logits belong to decode; no prefill row was emitted.
    assert (
        gather_score_logprobs(
            torch.ones(1, 3), torch.tensor([[0, 2], [1, 2]]), 0, logprob_order="torch"
        )
        is None
    )


def test_score_gather_forwards_resolved_logprob_order(monkeypatch):
    from tokenspeed.runtime.sampling import score_utils

    logits = torch.ones(3, 4)
    labels = torch.tensor([[0, 2], [1, 3]])
    calls = []

    def gather(rows, targets, *, logprob_order):
        calls.append((rows, targets, logprob_order))
        return torch.full_like(targets, -0.5, dtype=torch.float32)

    monkeypatch.setattr(score_utils, "gather_token_logprobs", gather)
    result = gather_score_logprobs(logits, labels, 1, logprob_order="megatron")
    assert result.tolist() == [[-0.5, -0.5]]
    rows, targets, order = calls[0]
    assert torch.equal(rows, logits[:1])
    assert torch.equal(targets, labels[:1])
    assert order == "megatron"


def test_registry_direct_family_registration_wins(monkeypatch):
    from tokenspeed.runtime.decision import registry

    factory = lambda family: ("custom", family)
    monkeypatch.setitem(registry._ADAPTERS, "qwen3", factory)
    assert get_decision_adapter("qwen3") == ("custom", "qwen3")


def test_score_params_default_to_no_scoring():
    params = SamplingParams()
    assert params.score_label_token_ids is None
    assert params.score_apply_softmax is None
    _verify(params)


def test_score_params_msgpack_roundtrip():
    params = SamplingParams(
        max_new_tokens=0, score_label_token_ids=[3, 7, 11], score_apply_softmax=True
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(params), type=SamplingParams
    )
    assert decoded.score_label_token_ids == [3, 7, 11]
    assert decoded.score_apply_softmax is True


def test_score_params_decode_older_payload_defaults():
    # Peers running an older build send shorter positional arrays; the new
    # tail fields must fall back to their defaults.
    legacy = SamplingParams(max_new_tokens=4)
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(legacy), type=SamplingParams
    )
    assert decoded.score_label_token_ids is None
    assert decoded.score_apply_softmax is None


# ---------------------------------------------------------------------------
# score_utils helpers
# ---------------------------------------------------------------------------


def test_build_score_label_ids_none_without_scores():
    params = [SamplingParams(), SamplingParams()]
    assert build_score_label_ids(params, 2, "cpu") is None


def test_build_score_label_ids_pads_to_max_labels():
    params = [
        SamplingParams(max_new_tokens=0, score_label_token_ids=[5, 9]),
        SamplingParams(),
        SamplingParams(max_new_tokens=0, score_label_token_ids=[1, 2, 3]),
    ]
    label_ids = build_score_label_ids(params, 3, "cpu")
    assert label_ids.shape == (3, 3)
    assert label_ids.tolist() == [[5, 9, 0], [0, 0, 0], [1, 2, 3]]


def test_build_score_label_ids_covers_extend_rows_only():
    params = [
        SamplingParams(max_new_tokens=0, score_label_token_ids=[5]),
        SamplingParams(max_new_tokens=0, score_label_token_ids=[6]),
    ]
    # Only the first row is an extend row in this batch.
    label_ids = build_score_label_ids(params, 1, "cpu")
    assert label_ids.tolist() == [[5]]


def test_gather_score_logprobs_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(3, 50)
    label_ids = torch.tensor([[0, 1], [2, 3], [4, 5]])
    gathered = gather_score_logprobs(logits, label_ids, 3, logprob_order="torch")
    expected = torch.log_softmax(logits.float(), dim=-1).gather(-1, label_ids)
    assert torch.equal(gathered, expected)


def test_finalize_score_row_raw_passthrough():
    row = [-0.3, -2.1, -1.0]
    assert finalize_score_row(row, apply_softmax=False) is row


def test_finalize_score_row_softmax_normalizes_across_labels():
    row = [-0.3, -2.1, -1.0]
    out = finalize_score_row(row, apply_softmax=True)
    assert math.isclose(sum(out), 1.0, rel_tol=1e-9)
    exps = [math.exp(v) for v in row]
    expected = [v / sum(exps) for v in exps]
    assert out == pytest.approx(expected, rel=1e-12)
    # Numerical-stability smoke: huge negative logits must not underflow.
    stable = finalize_score_row([-10000.0, -10001.0], apply_softmax=True)
    assert math.isclose(sum(stable), 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# SamplingBatchInfo slicing
# ---------------------------------------------------------------------------


def test_sampling_batch_info_slices_score_label_ids():
    label_ids = torch.tensor([[5, 9], [0, 0], [1, 2]])
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    info = SamplingBatchInfo(score_label_ids=label_ids)
    assert info[:2].score_label_ids.tolist() == [[5, 9], [0, 0]]
    assert info[2:].score_label_ids.tolist() == [[1, 2]]
    assert SamplingBatchInfo()[:2].score_label_ids is None


# ---------------------------------------------------------------------------
# Decision request + generic adapter
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Maps each known label string to a fixed single token id."""

    VOCAB = {"Yes": 100, "No": 200, "A": 301, "B": 302, "C": 303, "Multi": 41}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        if text.endswith("\n"):
            return [17]
        for label, token_id in self.VOCAB.items():
            if text.endswith(f"\n{label}"):
                return [17, token_id]
        if text == "Multi Token":
            return [41, 42]
        if text not in self.VOCAB:
            raise KeyError(text)
        return [self.VOCAB[text]]


def _request(**overrides) -> DecisionRequest:
    fields = {
        "query": "Which team owns this ticket?",
        "candidates": ["billing", "tech", "sales"],
        "style": STYLE_POINTWISE_YESNO,
        "adapter": "generic",
        "apply_softmax": True,
    }
    fields.update(overrides)
    return DecisionRequest(**fields)


def test_decision_request_validation():
    with pytest.raises(ValueError, match="query"):
        _request(query="")
    with pytest.raises(ValueError, match="candidates"):
        _request(candidates=[])
    with pytest.raises(ValueError, match="style"):
        _request(style="setwise")


def test_generic_adapter_pointwise_compile():
    call = GenericDecisionAdapter().compile(_request(), _StubTokenizer())
    # Pointwise: one item per candidate, others hidden; Yes/No columns.
    assert len(call.items) == 3
    assert "billing" in call.items[0] and "tech" not in call.items[0]
    assert call.label_token_ids == [100, 200]
    assert call.apply_softmax is True
    assert (call.query + call.items[0]).startswith(
        "Which team owns this ticket?\n\nProposed answer:"
    )


def test_generic_adapter_fused_compile():
    call = GenericDecisionAdapter().compile(
        _request(style=STYLE_FUSED_CHOICE), _StubTokenizer()
    )
    assert len(call.items) == 1
    assert "A) billing" in call.items[0] and "C) sales" in call.items[0]
    assert call.label_token_ids == [301, 302, 303]
    assert (call.query + call.items[0]).startswith(
        "Which team owns this ticket?\n\nA) billing"
    )


def test_generic_adapter_rejects_multi_token_label():
    adapter = GenericDecisionAdapter()
    with pytest.raises(ValueError, match="exactly one token"):
        adapter.compile(_request(), _MultiTokenTokenizer())


class _MultiTokenTokenizer(_StubTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text.endswith("\nYes"):
            return [17, 100, 101]
        return super().encode(text, add_special_tokens)


def test_generic_adapter_uses_contextual_label_ids():
    class ContextTokenizer(_StubTokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens)
            return [ids[0], ids[1] + 1000] if len(ids) == 2 else ids

    call = GenericDecisionAdapter().compile(_request(), ContextTokenizer())
    assert call.label_token_ids == [1100, 1200]


def test_generic_adapter_rejects_boundary_token_merge():
    class MergeTokenizer(_StubTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.endswith("\nYes"):
                return [99]
            return super().encode(text, add_special_tokens)

    with pytest.raises(ValueError, match="without changing the prompt"):
        GenericDecisionAdapter().compile(_request(), MergeTokenizer())


def test_generic_adapter_rejects_context_dependent_candidate_label_ids():
    class CandidateTokenizer(_StubTokenizer):
        def encode(self, text, add_special_tokens=False):
            if "tech" in text and text.endswith("\nYes"):
                return [17, 101]
            return super().encode(text, add_special_tokens)

    with pytest.raises(ValueError, match="differ between candidate"):
        GenericDecisionAdapter().compile(_request(), CandidateTokenizer())


def test_generic_adapter_extract_pointwise():
    adapter = GenericDecisionAdapter()
    req = _request()
    result = adapter.extract(req, [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert result.answer == "tech"
    assert result.answer_index == 1
    assert result.probabilities == [0.1, 0.8, 0.3]
    result_raw = adapter.extract(
        _request(apply_softmax=False), [[-2.0, -0.1], [-0.2, -3.0], [-1.0, -1.0]]
    )
    assert result_raw.answer_index == 1
    assert result_raw.probabilities is None


def test_generic_adapter_extract_fused():
    adapter = GenericDecisionAdapter()
    result = adapter.extract(_request(style=STYLE_FUSED_CHOICE), [[0.2, 0.7, 0.1]])
    assert result.answer == "tech"
    assert result.probabilities == [0.2, 0.7, 0.1]


def test_generic_adapter_extract_row_count_mismatch():
    adapter = GenericDecisionAdapter()
    with pytest.raises(ValueError, match="score rows"):
        adapter.extract(_request(), [[0.5, 0.5]])
    with pytest.raises(ValueError, match="exactly 1"):
        adapter.extract(_request(style=STYLE_FUSED_CHOICE), [[0.5], [0.5]])


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


def test_registry_resolves_generic_and_family_alias():
    assert get_decision_adapter("generic").family == "generic"
    # Family aliases resolve to a generic scaffold named after the family.
    assert get_decision_adapter("kimi_k3").family == "kimi_k3"


def test_registry_rejects_unknown_adapter():
    with pytest.raises(ValueError, match="Unknown decision adapter"):
        get_decision_adapter("no_such_family")


# ---------------------------------------------------------------------------
# Engine score/decision API (stub frontend)
# ---------------------------------------------------------------------------


class _StubTokenizerManager:
    def __init__(self, outputs, tokenizer=None):
        self._outputs = outputs
        self.tokenizer = tokenizer
        self.last_obj = None

    async def generate_request(self, obj):
        self.last_obj = obj
        yield self._outputs


def _engine_with(outputs, tokenizer=None):
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine.__new__(Engine)
    engine.tokenizer_manager = _StubTokenizerManager(outputs, tokenizer)
    return engine


def test_engine_async_score_joins_rows_in_items_order():
    outputs = [
        {"scores": [0.1, 0.9], "meta_info": {}},
        {"scores": [0.8, 0.2], "meta_info": {}},
    ]
    engine = _engine_with(outputs)
    out = asyncio.run(
        engine.async_score(
            query="Q?",
            items=["i0", "i1"],
            label_token_ids=[100, 200],
            apply_softmax=True,
        )
    )
    assert out == {"scores": [[0.1, 0.9], [0.8, 0.2]]}
    obj = engine.tokenizer_manager.last_obj
    assert obj.text == ["Q?i0", "Q?i1"]
    for params in obj.sampling_params:
        assert params["max_new_tokens"] == 0
        assert params["score_label_token_ids"] == [100, 200]
        assert params["score_apply_softmax"] is True


def test_engine_async_score_missing_row_raises():
    engine = _engine_with([{"meta_info": {"finish_reason": {"type": "length"}}}])
    with pytest.raises(RuntimeError, match="without a score readout"):
        asyncio.run(
            engine.async_score(
                query="Q?",
                items=["i0"],
                label_token_ids=[100, 200],
                apply_softmax=False,
            )
        )


@pytest.mark.parametrize("row", ([0.75, 0.25], None))
def test_engine_async_score_rejects_aborted_item(row):
    engine = _engine_with(
        [{"scores": row, "meta_info": {"finish_reason": {"type": "abort"}}}]
    )
    with pytest.raises(RuntimeError, match="score item 0 aborted"):
        asyncio.run(
            engine.async_score(
                query="Q?", items=["i0"], label_token_ids=[100, 200], apply_softmax=True
            )
        )


def test_engine_async_decision_rejects_aborted_scores():
    engine = _engine_with(
        [{"scores": [0.75, 0.25], "meta_info": {"finish_reason": {"type": "abort"}}}],
        tokenizer=_StubTokenizer(),
    )
    with pytest.raises(RuntimeError, match="aborted"):
        asyncio.run(engine.async_decision(_request()))


def test_engine_async_score_validates_arguments():
    engine = _engine_with([])
    with pytest.raises(ValueError, match="items"):
        asyncio.run(
            engine.async_score(
                query="Q?", items=[], label_token_ids=[1], apply_softmax=True
            )
        )
    with pytest.raises(ValueError, match="label_token_ids"):
        asyncio.run(
            engine.async_score(
                query="Q?", items=["i0"], label_token_ids=[], apply_softmax=True
            )
        )


def test_engine_async_decision_end_to_end():
    outputs = [
        {"scores": [0.1, 0.9], "meta_info": {}},
        {"scores": [0.8, 0.2], "meta_info": {}},
        {"scores": [0.3, 0.7], "meta_info": {}},
    ]
    engine = _engine_with(outputs, tokenizer=_StubTokenizer())
    result = asyncio.run(engine.async_decision(_request()))
    assert result.answer == "tech"
    assert result.answer_index == 1
    assert result.probabilities == [0.1, 0.8, 0.3]
    call_obj = engine.tokenizer_manager.last_obj
    assert call_obj.sampling_params[0]["score_label_token_ids"] == [100, 200]


def test_engine_async_decision_requires_tokenizer():
    engine = _engine_with([], tokenizer=None)
    with pytest.raises(ValueError, match="skip_tokenizer_init"):
        asyncio.run(engine.async_decision(_request()))


# ---------------------------------------------------------------------------
# Wire output structs
# ---------------------------------------------------------------------------


def _batch_out(score_vals):
    from tokenspeed.runtime.engine.io_struct import BatchTokenIDOut

    return BatchTokenIDOut(
        rids=["a", "b"],
        finished_reasons=[None, None],
        decoded_texts=["", ""],
        decode_ids=[[1], [2]],
        read_offsets=[0, 0],
        output_ids=[[1], [2]],
        output_multi_ids=[[], []],
        skip_special_tokens=[True, True],
        spaces_between_special_tokens=[True, True],
        no_stop_trim=[False, False],
        prompt_tokens=[3, 3],
        completion_tokens=[1, 1],
        cached_tokens=[0, 0],
        spec_verify_ct=[0, 0],
        input_token_logprobs_val=[],
        input_token_logprobs_idx=[],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[],
        input_top_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        input_token_ids_logprobs_val=[],
        input_token_ids_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        output_hidden_states=[],
        batch_accept_draft_tokens=[],
        output_extra_infos=[{}, {}],
        generated_time=0.0,
        output_score_vals=score_vals,
    )


def test_batch_token_id_out_slim_carries_scores():
    from tokenspeed.runtime.engine.io_struct import BatchTokenIDOutSlim

    slim = BatchTokenIDOutSlim.from_full(_batch_out([[0.8, 0.2], []]))
    assert slim.output_score_vals == [[0.8, 0.2], []]
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(slim), type=BatchTokenIDOutSlim
    )
    assert decoded.output_score_vals == [[0.8, 0.2], []]


def test_batch_token_id_out_slim_defaults_score_column():
    from tokenspeed.runtime.engine.io_struct import BatchTokenIDOutSlim

    # A full batch without score metadata yields a non-ragged empty column.
    slim = BatchTokenIDOutSlim.from_full(_batch_out(None))
    assert slim.output_score_vals == [[], []]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
