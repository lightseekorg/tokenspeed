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

"""Prompt (input) logprobs, SGLang dialect: ``return_logprob`` with
``logprob_start_len``.

Covers the frontend's ingress resolution of ``logprob_start_len`` and its
refusals, the executor's expansion of a forward's row plan and the logits
it gathers, and the frontend codec. The control plane's plan, accumulation
and emission are ``ts-control``'s (its ingress, output and serving tests).
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

import pytest  # noqa: E402
import torch  # noqa: E402

from tokenspeed.runtime.engine.input_processor import InputProcessor  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    BatchTokenIDOut,
    GenerateReqInput,
    MsgpackDecoder,
    MsgpackEncoder,
    TokenizedGenerateReqInput,
    ipc_message_union,
)
from tokenspeed.runtime.engine.logprobs import (  # noqa: E402
    LogprobsProcessor,
    resolve_logprob_start_len,
)
from tokenspeed.runtime.execution.types import (  # noqa: E402
    ForwardLogprobs,
    InputLogprobPlan,
)

# --------------------------------------------------------------------------
# Ingress: logprob_start_len resolution and validation
# --------------------------------------------------------------------------


def test_resolve_logprob_start_len_follows_sglang():
    assert resolve_logprob_start_len(-1, 5) == 4
    assert resolve_logprob_start_len(None, 5) == 4
    assert resolve_logprob_start_len(0, 5) == 0
    assert resolve_logprob_start_len(4, 5) == 4
    assert resolve_logprob_start_len(-1, 0) == 0
    with pytest.raises(ValueError, match="smaller than the prompt length"):
        resolve_logprob_start_len(5, 5)
    with pytest.raises(ValueError, match="-1 or >= 0"):
        resolve_logprob_start_len(-3, 5)


class _StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


def _input_processor(
    *,
    enable_output_logprobs: bool = True,
    supports_prompt_logprobs: bool = True,
    **server_args,
) -> InputProcessor:
    engine = SimpleNamespace(
        context_len=100,
        max_req_input_len=99,
        is_generation=True,
        tokenizer=_StubTokenizer(),
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        server_args=SimpleNamespace(
            reasoning_parser=None,
            enable_prefix_caching=True,
            enable_output_logprobs=enable_output_logprobs,
            language_model_only=False,
            **{
                "logprob_order": "torch",
                "mapping": None,
                "disaggregation_mode": "null",
                **server_args,
            },
        ),
        model_config=SimpleNamespace(
            vocab_size=32000, is_multimodal=False, is_multimodal_active=False
        ),
        supports_prompt_logprobs=supports_prompt_logprobs,
    )
    return InputProcessor(engine)


def _tokenize(
    *,
    input_ids=None,
    processor: InputProcessor | None = None,
    sampling_params: dict | None = None,
    **kwargs,
) -> TokenizedGenerateReqInput:
    obj = GenerateReqInput(
        input_ids=list(range(10)) if input_ids is None else input_ids,
        sampling_params=dict(sampling_params or {}),
        **kwargs,
    )
    obj.normalize_batch_and_arguments()
    processor = processor if processor is not None else _input_processor()
    return asyncio.run(processor.tokenize_one_request(obj))


def test_input_processor_accepts_and_resolves_logprob_start_len():
    assert _tokenize(return_logprob=True).logprob_start_len == 9
    assert _tokenize(return_logprob=True, logprob_start_len=-1).logprob_start_len == 9
    assert _tokenize(return_logprob=True, logprob_start_len=0).logprob_start_len == 0
    assert _tokenize(return_logprob=True, logprob_start_len=7).logprob_start_len == 7
    # A vLLM-dialect request (no return_logprob) never asks for prompt logprobs.
    assert _tokenize(logprob_start_len=0).logprob_start_len == 9


def test_input_processor_rejects_out_of_range_start_and_unsupported_knobs():
    with pytest.raises(ValueError, match="smaller than the prompt length"):
        _tokenize(return_logprob=True, logprob_start_len=10)
    with pytest.raises(ValueError, match="top_logprobs_num"):
        _tokenize(return_logprob=True, top_logprobs_num=3)
    with pytest.raises(ValueError, match="token_ids_logprob"):
        _tokenize(return_logprob=True, token_ids_logprob=[1, 2])


def test_input_processor_refuses_prompt_logprobs_the_engine_cannot_compute():
    """A narrowing model or a pipeline split is a 400 at the ingress; the
    default start (no prompt rows) stays accepted, and so does a vLLM-dialect
    request, whatever its logprob_start_len says."""
    cannot = _input_processor(supports_prompt_logprobs=False)
    with pytest.raises(ValueError, match="not supported by this engine"):
        _tokenize(return_logprob=True, logprob_start_len=0, processor=cannot)
    with pytest.raises(ValueError, match="not supported by this engine"):
        _tokenize(return_logprob=True, logprob_start_len=8, processor=cannot)
    assert (
        _tokenize(
            return_logprob=True, logprob_start_len=-1, processor=cannot
        ).logprob_start_len
        == 9
    )
    assert (
        _tokenize(
            return_logprob=True, logprob_start_len=9, processor=cannot
        ).logprob_start_len
        == 9
    )
    assert _tokenize(logprob_start_len=0, processor=cannot).logprob_start_len == 9
    # Until the scheduler reported the capability, the engine cannot promise it.
    unknown = _input_processor(supports_prompt_logprobs=None)
    with pytest.raises(ValueError, match="not supported by this engine"):
        _tokenize(return_logprob=True, logprob_start_len=0, processor=unknown)


def test_input_processor_refuses_prompt_logprobs_for_out_of_vocab_ids():
    """The targets are read from the device input ids, so every prompt token
    must own a logits column; a client id past the vocab is a 400, not a
    device fault."""
    with pytest.raises(ValueError, match="inside the vocabulary"):
        _tokenize(return_logprob=True, logprob_start_len=0, input_ids=[1, 2, 32000])
    with pytest.raises(ValueError, match="inside the vocabulary"):
        _tokenize(return_logprob=True, logprob_start_len=0, input_ids=[1, -1, 3])
    # Without prompt rows the ids are not the logprob path's concern.
    assert (
        _tokenize(return_logprob=True, input_ids=[1, 2, 32000]).logprob_start_len == 2
    )


def test_input_processor_takes_top_k_for_a_request_sampled_in_its_prefill():
    """Top-k alternatives exist for the prompt rows and the token a prefill
    samples, so a request generating at most one token gets them; a longer
    generation would reach a captured decode step, which computes none."""
    for max_new_tokens in (0, 1):
        tokenized = _tokenize(
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=5,
            sampling_params={"max_new_tokens": max_new_tokens},
        )
        assert tokenized.top_logprobs_num == 5
        assert tokenized.logprob_start_len == 0
    # The default start (no prompt rows) takes it too: the sampled token's.
    assert (
        _tokenize(
            return_logprob=True,
            top_logprobs_num=20,
            sampling_params={"max_new_tokens": 1},
        ).top_logprobs_num
        == 20
    )
    # No logprobs asked for: nothing to gate.
    assert _tokenize(top_logprobs_num=3).top_logprobs_num == 0


@pytest.mark.parametrize(
    ("server_args", "kwargs", "message"),
    [
        ({}, {"sampling_params": {"max_new_tokens": 2}}, "max_new_tokens <= 1"),
        ({}, {"sampling_params": {}}, "max_new_tokens <= 1"),
        ({}, {"top_logprobs_num": 21}, "between 1 and 20"),
        ({"logprob_order": "megatron"}, {}, "logprob-order torch"),
        ({"mapping": SimpleNamespace(has_pp=True)}, {}, "pipeline parallelism"),
        ({"disaggregation_mode": "prefill"}, {}, "disaggregated"),
    ],
)
def test_input_processor_refuses_top_k_it_cannot_compute(server_args, kwargs, message):
    request = {
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 5,
        "sampling_params": {"max_new_tokens": 1},
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        _tokenize(processor=_input_processor(**server_args), **request)


# --------------------------------------------------------------------------
# Forward thread: the plan's device staging
# --------------------------------------------------------------------------


def _staging_executor(monkeypatch, *, shifted_ids: list[int], vocab_size: int = 32):
    from tokenspeed.runtime.execution import model_executor as executor_module
    from tokenspeed.runtime.execution.nan_guard import NanGuard

    monkeypatch.setattr(executor_module, "is_pin_memory_available", lambda: False)
    executor = object.__new__(executor_module.ModelExecutor)
    executor.device = "cpu"
    executor.runtime_states = SimpleNamespace(vocab_size=vocab_size)
    executor.config = SimpleNamespace(input_logprob_chunk_tokens=3)
    executor.input_buffers = SimpleNamespace(
        shifted_prefill_ids_buf=torch.tensor(shifted_ids, dtype=torch.int32)
    )
    executor.nan_guard = NanGuard(max_bs=4, device="cpu")
    executor.nan_guard.reset(4)
    return executor


def test_executor_expands_the_plan_into_rows_targets_and_slots(monkeypatch):
    """The per-slot triples become one arange per slot; each row's target is
    the next prompt token read from the scheduler's shifted input ids, which
    cover the chunk boundary (row 2 of slot 0 predicts a token of the next
    chunk)."""
    # Two extend slots: rows 0..2 (slot 0) and 3..5 (slot 1) of a 6-row forward.
    shifted = [11, 12, 13, 21, 22, 23]
    executor = _staging_executor(monkeypatch, shifted_ids=shifted)
    assert executor._input_logprob_rows(None, 2, 6, None) is None

    plan = InputLogprobPlan(row_starts=(0, 4), counts=(3, 2), position_starts=(5, 0))
    staged = executor._input_logprob_rows(plan, 2, 6, None)
    assert staged.rows.tolist() == [0, 1, 2, 4, 5]
    assert staged.targets.tolist() == [11, 12, 13, 22, 23]
    assert staged.slots.tolist() == [0, 0, 0, 1, 1]
    assert (
        staged.rows.dtype == staged.targets.dtype == staged.slots.dtype == torch.int64
    )
    assert staged.num_input_rows == 6
    assert staged.chunk_tokens == 3
    assert staged.rows_per_rank is None and staged.num_result_rows == 5
    assert staged.top_k == 0
    assert executor.nan_guard.flags.tolist() == [0, 0, 0, 0]
    plan = InputLogprobPlan(
        row_starts=(0, 4), counts=(3, 2), position_starts=(5, 0), top_k=4
    )
    assert executor._input_logprob_rows(plan, 2, 6, None).top_k == 4

    # A slot without rows contributes nothing and shifts no other slot.
    plan = InputLogprobPlan(
        row_starts=(0, 0, 3), counts=(2, 0, 1), position_starts=(0, 0, 7)
    )
    staged = executor._input_logprob_rows(plan, 3, 6, None)
    assert staged.rows.tolist() == [0, 1, 3]
    assert staged.slots.tolist() == [0, 0, 2]

    with pytest.raises(RuntimeError, match="past the forward"):
        executor._input_logprob_rows(
            InputLogprobPlan(row_starts=(4,), counts=(3,), position_starts=(0,)),
            1,
            6,
            None,
        )


def test_executor_flags_an_out_of_vocab_target_instead_of_scoring_it(monkeypatch):
    """The ingress keeps such ids out; one that slips through terminates its
    request through the NaN guard, exactly like a NaN sample, and is clamped
    only so the gather cannot fault."""
    executor = _staging_executor(monkeypatch, shifted_ids=[1, 9, 2, 3], vocab_size=8)
    plan = InputLogprobPlan(row_starts=(0, 2), counts=(2, 2), position_starts=(0, 0))
    staged = executor._input_logprob_rows(plan, 2, 4, None)
    assert staged.targets.tolist() == [1, 7, 2, 3]
    assert executor.nan_guard.flags.tolist() == [1, 0, 0, 0]


def test_executor_stages_a_shards_rows_of_the_plan(monkeypatch):
    """Under a query shard each rank keeps as its rows the plan's rows inside
    its shard, re-based to it, with the per-rank split of the activation
    gather; the targets and slots stay the whole plan's (every rank scores
    every row once the activations are gathered) and the target audit runs
    over the whole plan on every rank, so the NaN flags agree across the
    group. A rank without any planned row stages none and still carries the
    split."""
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan

    # Two requests of 4 and 6 rows over four shards of [3, 3, 2, 2] rows; the
    # plan wants rows 1..3 of the first and rows 4..7 of the second (slot 1's
    # first two rows 4, 5 are on rank 1, rows 6, 7 on rank 2; rank 3 none).
    shifted = list(range(100, 110))
    shifted[6] = 999  # out of vocab: flags slot 1 on every rank
    plans = {}
    for rank in range(4):
        executor = _staging_executor(monkeypatch, shifted_ids=shifted, vocab_size=200)
        shard = QueryShardPlan.from_forward(
            total_tokens=10, input_lengths=[4, 6], size=4, rank=rank
        )
        plan = InputLogprobPlan(
            row_starts=(1, 4), counts=(3, 4), position_starts=(1, 0)
        )
        staged = executor._input_logprob_rows(plan, 2, 10, shard)
        plans[rank] = staged
        assert staged.rows_per_rank == (2, 3, 2, 0)
        assert staged.num_input_rows == shard.local_rows
        assert staged.num_result_rows == 7
        assert staged.slots.tolist() == [0, 0, 0, 1, 1, 1, 1]
        # The flagged row's target is clamped; every rank holds every target.
        assert staged.targets.tolist() == [101, 102, 103, 104, 105, 199, 107]
        assert executor.nan_guard.flags.tolist() == [0, 1, 0, 0]
    assert plans[0].rows.tolist() == [1, 2]
    assert plans[1].rows.tolist() == [0, 1, 2]  # rows 3, 4, 5 re-based to shard 1
    assert plans[2].rows.tolist() == [0, 1]  # rows 6, 7 re-based to shard 2
    assert plans[3].rows.tolist() == []


def test_nan_guard_flags_the_slot_of_a_non_finite_prompt_logprob():
    from tokenspeed.runtime.execution.nan_guard import NanGuard

    guard = NanGuard(max_bs=3, device="cpu")
    guard.reset(3)
    logprobs = torch.tensor([-1.0, float("nan"), -2.0, float("-inf")])
    guard.audit_input_logprobs(logprobs, torch.tensor([0, 0, 1, 2]), 3)
    assert guard.flags.tolist() == [1, 0, 1]
    assert torch.isfinite(logprobs).all()
    assert logprobs[0].item() == -1.0 and logprobs[2].item() == -2.0


# --------------------------------------------------------------------------
# Wire: the per-request lists with a leading None survive the IPC codec
# --------------------------------------------------------------------------


def test_batch_token_id_out_codec_carries_prompt_logprobs():
    out = BatchTokenIDOut(
        rids=["r1"],
        finished_reasons=[None],
        decoded_texts=[""],
        decode_ids=[[10]],
        read_offsets=[0],
        output_ids=[[10]],
        output_multi_ids=[[]],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
        no_stop_trim=[False],
        prompt_tokens=[3],
        completion_tokens=[1],
        cached_tokens=[0],
        spec_verify_ct=[0],
        input_token_logprobs_val=[[None, -0.5, -1.25]],
        input_token_logprobs_idx=[[7, 8, 9]],
        output_token_logprobs_val=[[-0.75]],
        output_token_logprobs_idx=[[10]],
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
        output_extra_infos=[{}],
        generated_time=0.0,
    )
    frames = MsgpackEncoder().encode(out)
    decoded = MsgpackDecoder(ipc_message_union()).decode(frames)
    assert isinstance(decoded, BatchTokenIDOut)
    assert decoded.input_token_logprobs_val == [[None, -0.5, -1.25]]
    assert decoded.input_token_logprobs_idx == [[7, 8, 9]]
    assert decoded.output_token_logprobs_val == [[-0.75]]


def test_batch_token_id_out_codec_carries_top_logprobs():
    out = BatchTokenIDOut(
        rids=["r1"],
        finished_reasons=[None],
        decoded_texts=[""],
        decode_ids=[[10]],
        read_offsets=[0],
        output_ids=[[10]],
        output_multi_ids=[[]],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
        no_stop_trim=[False],
        prompt_tokens=[3],
        completion_tokens=[1],
        cached_tokens=[0],
        spec_verify_ct=[0],
        input_token_logprobs_val=[[None, -0.5, -1.25]],
        input_token_logprobs_idx=[[7, 8, 9]],
        output_token_logprobs_val=[[-0.75]],
        output_token_logprobs_idx=[[10]],
        input_top_logprobs_val=[[None, [-0.5, -1.5], [-0.25, -1.25]]],
        input_top_logprobs_idx=[[None, [8, 3], [4, 9]]],
        output_top_logprobs_val=[[[-0.75, -0.9]]],
        output_top_logprobs_idx=[[[10, 11]]],
        input_token_ids_logprobs_val=[],
        input_token_ids_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        output_hidden_states=[],
        batch_accept_draft_tokens=[],
        output_extra_infos=[{}],
        generated_time=0.0,
    )
    frames = MsgpackEncoder().encode(out)
    decoded = MsgpackDecoder(ipc_message_union()).decode(frames)
    assert decoded.input_top_logprobs_val == [[None, [-0.5, -1.5], [-0.25, -1.25]]]
    assert decoded.input_top_logprobs_idx == [[None, [8, 3], [4, 9]]]
    assert decoded.output_top_logprobs_val == [[[-0.75, -0.9]]]
    assert decoded.output_top_logprobs_idx == [[[10, 11]]]

    # The frontend renders them as (logprob, id, text) tuples, None first.

    info: dict = {}
    LogprobsProcessor(SimpleNamespace(tokenizer=None)).convert_logprob_style(
        info, "sglang", 2, None, False, decoded, 0
    )
    assert info["input_top_logprobs"] == [
        None,
        [(-0.5, 8, None), (-1.5, 3, None)],
        [(-0.25, 4, None), (-1.25, 9, None)],
    ]
    assert info["output_top_logprobs"] == [[(-0.75, 10, None), (-0.9, 11, None)]]


def test_sampled_top_logprobs_are_the_prefill_rows_log_softmax_top_k():
    from tokenspeed.runtime.execution.model_executor import _set_sampled_top_logprobs
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

    torch.manual_seed(0)
    logits = (torch.randn(3, 50) * 4).to(torch.bfloat16)
    out = LogitsProcessorOutput(next_token_logits=logits)
    _set_sampled_top_logprobs(out, logits, 0)
    assert out.sampled_top_logprobs_val is None
    assert ForwardLogprobs.of(out) is None
    _set_sampled_top_logprobs(out, logits, 5)
    expected = torch.log_softmax(logits.float(), dim=-1).topk(5, dim=-1)
    assert torch.equal(out.sampled_top_logprobs_val, expected.values)
    assert torch.equal(out.sampled_top_logprobs_idx, expected.indices)
    bundle = ForwardLogprobs.of(out).to_host()
    assert bundle.input_token_logprobs is None
    assert torch.equal(bundle.output_top_logprobs_idx, expected.indices)


@pytest.mark.parametrize("chunk_tokens", [1, 2, 64])
def test_prompt_top_k_rides_the_chunk_loop_of_the_prompt_logprobs(chunk_tokens):
    """With a plan's top_k every chunk makes two calls: gather_token_logprobs
    for the targets' values (bit for bit the logprob without top-k), then
    top_token_logprobs for each row's alternatives; top_k 0 leaves the output
    as it was, and a top_k past the vocabulary returns every token."""
    from tokenspeed.runtime.execution.context import InputLogprobRows
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )

    torch.manual_seed(0)
    vocab = 12
    hidden = torch.randint(-8, 9, (5, 4)).float() / 8
    weight = torch.randint(-8, 9, (vocab, 4)).float() / 8
    rows, targets = [0, 1, 3], [5, 2, 3]
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=vocab),
        dp_lm_head_tp=False,
    )

    def run(top_k):
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            query_shard=None,
            gather_ids=torch.tensor([2, 4]),
            input_logprob_rows=InputLogprobRows(
                rows=torch.tensor(rows),
                targets=torch.tensor(targets),
                slots=torch.zeros(len(rows), dtype=torch.int64),
                num_input_rows=5,
                chunk_tokens=chunk_tokens,
                rows_per_rank=None,
                top_k=top_k,
            ),
        )
        return processor(
            input_ids=None,
            hidden_states=hidden,
            lm_head=SimpleNamespace(weight=weight),
            logits_metadata=metadata,
        )

    plain, with_top = run(0), run(3)
    assert plain.input_top_logprobs_val is None
    assert torch.equal(with_top.input_token_logprobs, plain.input_token_logprobs)
    logprobs = torch.log_softmax(hidden[rows] @ weight.T, dim=-1)
    top = logprobs.topk(3, dim=-1)
    assert torch.equal(with_top.input_top_logprobs_val, top.values)
    assert torch.equal(with_top.input_top_logprobs_idx, top.indices)
    assert with_top.input_top_logprobs_idx.dtype == torch.int64
    every = run(vocab + 8)
    whole = logprobs.topk(vocab, dim=-1)
    assert torch.equal(every.input_top_logprobs_val, whole.values)
    assert torch.equal(every.input_top_logprobs_idx, whole.indices)


def test_prompt_rows_and_sampled_rows_take_the_same_top_k_of_the_same_logits():
    """docs/design/numerics.md: one arithmetic for prompt and output. Scored
    as prompt rows (in uneven chunks) or sampled, the same bf16 logits rows
    get the same alternatives bit for bit, values and ids, exact ties in
    the top-k included."""
    from tokenspeed.runtime.execution.context import InputLogprobRows
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.model_executor import _set_sampled_top_logprobs
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )

    torch.manual_seed(0)
    vocab, k, num_rows = 64, 8, 5
    # The identity head makes each row's logits its activations, exactly.
    # Three tokens tie for every row's largest logit, and the coarse grid
    # ties more of the rest.
    hidden = (torch.randint(-16, 17, (num_rows, vocab)) / 4).to(torch.bfloat16)
    hidden[:, [3, 17, 40]] = 5.0
    processor = LogitsProcessor(
        config=SimpleNamespace(model_type="test", vocab_size=vocab),
        dp_lm_head_tp=False,
    )
    out = processor(
        input_ids=None,
        hidden_states=hidden,
        lm_head=SimpleNamespace(weight=torch.eye(vocab, dtype=torch.bfloat16)),
        logits_metadata=LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            query_shard=None,
            gather_ids=torch.arange(num_rows),
            input_logprob_rows=InputLogprobRows(
                rows=torch.arange(num_rows),
                targets=torch.zeros(num_rows, dtype=torch.int64),
                slots=torch.zeros(num_rows, dtype=torch.int64),
                num_input_rows=num_rows,
                chunk_tokens=2,
                rows_per_rank=None,
                top_k=k,
            ),
        ),
    )
    _set_sampled_top_logprobs(out, out.next_token_logits, k)

    assert torch.equal(out.input_top_logprobs_val, out.sampled_top_logprobs_val)
    assert torch.equal(out.input_top_logprobs_idx, out.sampled_top_logprobs_idx)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
