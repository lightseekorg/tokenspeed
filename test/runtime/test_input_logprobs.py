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

Covers the control-plane half of the feature on CPU: the ingress resolution
of ``logprob_start_len``, the per-forward row plan built beside the other
gathers, the per-request accumulation across prefill chunks with the SGLang
``[None] + rows`` finalization, the one-shot emission, the decode role's empty
lists, the admission-probe cap handed to the C++ scheduler, and the wire codec.
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

from tokenspeed.runtime.engine import (  # noqa: E402
    generation_output_processor as output_module,
)
from tokenspeed.runtime.engine.generation_output_processor import (  # noqa: E402
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.engine.input_processor import InputProcessor  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    BatchTokenIDOut,
    GenerateReqInput,
    MsgpackDecoder,
    MsgpackEncoder,
    TokenizedGenerateReqInput,
    ipc_message_union,
)
from tokenspeed.runtime.engine.logprobs import resolve_logprob_start_len  # noqa: E402
from tokenspeed.runtime.engine.request_handler import RequestHandler  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    UNBOUNDED_CACHED_PREFIX_TOKENS,
    input_logprob_plan_for_forward,
    make_spec,
)
from tokenspeed.runtime.execution.types import InputLogprobPlan  # noqa: E402
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

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
    *, enable_output_logprobs: bool = True, supports_prompt_logprobs: bool = True
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
            disaggregation_mode="null",
            language_model_only=False,
        ),
        model_config=SimpleNamespace(
            vocab_size=32000, is_multimodal=False, is_multimodal_active=False
        ),
        supports_prompt_logprobs=supports_prompt_logprobs,
    )
    return InputProcessor(engine)


def _tokenize(
    *, input_ids=None, processor: InputProcessor | None = None, **kwargs
) -> TokenizedGenerateReqInput:
    obj = GenerateReqInput(
        input_ids=list(range(10)) if input_ids is None else input_ids,
        sampling_params={},
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


# --------------------------------------------------------------------------
# RequestState
# --------------------------------------------------------------------------


def _state(
    input_ids: list[int],
    *,
    start: int = -1,
    return_logprob: bool = True,
    computes_prompt_logprobs: bool = True,
    unpadded: list[int] | None = None,
) -> RequestState:
    return RequestState(
        prompt_input_ids=input_ids,
        sampling_params=SamplingParams(max_new_tokens=8, stop=[], ignore_eos=True),
        stream=False,
        tokenizer=SimpleNamespace(eos_token_id=None, additional_stop_token_ids=None),
        return_logprob=return_logprob,
        logprob_start_len=start,
        computes_prompt_logprobs=computes_prompt_logprobs,
        prompt_input_ids_unpadded=unpadded,
    )


def test_request_state_resolves_the_start_and_sizes_the_accumulator():
    default = _state([1, 2, 3, 4])
    assert default.logprob_start_len == 3
    assert not default.wants_input_logprobs
    assert default.input_token_logprobs is None

    wanting = _state([1, 2, 3, 4], start=1)
    assert wanting.wants_input_logprobs
    # Positions 1 and 2 need logits; position 3 predicts the first output.
    assert wanting.input_token_logprobs == [None, None]

    assert not _state(
        [1, 2, 3, 4], start=1, return_logprob=False
    ).returns_input_logprobs
    decode_role = _state([1, 2, 3, 4], start=1, computes_prompt_logprobs=False)
    assert not decode_role.wants_input_logprobs
    assert decode_role.input_token_logprobs is None


# --------------------------------------------------------------------------
# The per-forward plan
# --------------------------------------------------------------------------


class _Op:
    def __init__(
        self,
        request_ids,
        input_lengths,
        extend_prefix_lens,
        num_extends,
        prefill_lengths=None,
    ):
        self.request_ids = request_ids
        self.input_lengths = input_lengths
        self.extend_prefix_lens = extend_prefix_lens
        self.extend_replay_lens = [0] * len(extend_prefix_lens)
        # The C++ op's per-slot total prefill size; defaults to the chunk end.
        self.prefill_lengths = (
            prefill_lengths
            if prefill_lengths is not None
            else [
                p + n for p, n in zip(extend_prefix_lens, input_lengths[:num_extends])
            ]
        )
        self._num_extends = num_extends

    def num_extends(self):
        return self._num_extends


def _plan(op, states) -> InputLogprobPlan | None:
    return input_logprob_plan_for_forward(op, states)


def _triples(plan: InputLogprobPlan) -> tuple[tuple[int, int, int], ...]:
    return tuple(zip(plan.row_starts, plan.counts, plan.position_starts))


def test_plan_single_chunk_from_position_zero():
    ids = [10, 11, 12, 13, 14]
    plan = _plan(_Op(["a"], [5], [0], 1), {"a": _state(ids, start=0)})
    # Rows 0..3 predict ids[1..4]; the last row predicts the first output.
    assert _triples(plan) == ((0, 4, 0),)
    assert plan.num_rows == 4


def test_plan_follows_three_chunks_of_one_prompt():
    ids = list(range(100, 110))
    state = _state(ids, start=0)
    chunks = [_plan(_Op(["a"], [4], [p], 1), {"a": state}) for p in (0, 4, 8)]
    assert _triples(chunks[0]) == ((0, 4, 0),)
    assert _triples(chunks[1]) == ((0, 4, 4),)
    # The final chunk feeds positions 8 and 9; only 8 predicts a prompt token.
    assert _triples(chunks[2]) == ((0, 1, 8),)


def test_plan_with_a_prefix_hit_at_the_start_and_a_start_inside_a_chunk():
    ids = list(range(100, 110))
    # Hit exactly at the start: the whole chunk is wanted.
    plan = _plan(_Op(["a"], [6], [4], 1), {"a": _state(ids, start=4)})
    assert _triples(plan) == ((0, 5, 4),)
    # Start inside the second chunk: rows before it are skipped.
    state = _state(ids, start=6)
    assert _plan(_Op(["a"], [4], [0], 1), {"a": state}) is None
    plan = _plan(_Op(["a"], [6], [4], 1), {"a": state})
    assert _triples(plan) == ((2, 3, 6),)


def test_plan_clips_a_rebased_window_and_skips_finalized_requests():
    ids = list(range(100, 110))
    state = _state(ids, start=5)
    # After retraction the prefill window carries 3 generated tokens past the
    # prompt; only prompt positions are gathered.
    op = _Op(["a"], [13], [0], 1)
    plan = _plan(op, {"a": state})
    assert _triples(plan) == ((5, 4, 5),)
    state.input_token_logprobs_val = [None]
    assert _plan(op, {"a": state}) is None


def test_plan_ignores_default_starts_decode_rows_and_mixes_requests():
    ids = list(range(100, 110))
    states = {
        "default": _state(ids),  # last token only: no GPU work
        "off": _state(ids, start=0, return_logprob=False),
        "want": _state(ids, start=2),
        "decode": _state(ids, start=0),
    }
    assert _plan(_Op(["default", "off"], [10, 10], [0, 0], 2), states) is None
    # Two extend rows then a decode row: the second extend's rows are offset
    # by the first extend's length; the decode row is never considered.
    plan = _plan(_Op(["default", "want", "decode"], [10, 5, 1], [0, 0], 2), states)
    assert _triples(plan) == ((0, 0, 0), (12, 3, 2))
    assert plan.num_rows == 3


def test_plan_validates_its_shape():
    with pytest.raises(ValueError, match="equal lengths"):
        InputLogprobPlan((0, 1), (5,), (0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        InputLogprobPlan((0,), (-1,), (0,))
    with pytest.raises(ValueError, match="at least one row"):
        InputLogprobPlan((0, 0), (0, 0), (0, 0))


# --------------------------------------------------------------------------
# Commit path: accumulation, finalization, one-shot emission
# --------------------------------------------------------------------------


class _Sender:
    def __init__(self):
        self.items = []

    def send_pyobj(self, obj):
        self.items.append(obj)


class _Result:
    def __init__(
        self, output_tokens, plan=None, input_token_logprobs=None, nan_flags=None
    ):
        self.output_tokens = torch.tensor(output_tokens, dtype=torch.int32)
        self.output_lengths = torch.ones(len(output_tokens), dtype=torch.int32)
        self.output_logprobs = (
            torch.full((len(output_tokens),), -0.5) if output_tokens else None
        )
        self.output_nan_flags = (
            torch.tensor(nan_flags, dtype=torch.int32)
            if nan_flags is not None
            else None
        )
        self.grammar_completion = None
        self.next_input_ids = None
        self.input_logprob_plan = plan
        self.input_token_logprobs = (
            torch.tensor(input_token_logprobs, dtype=torch.float32)
            if input_token_logprobs is not None
            else None
        )


def _processor(sender=None) -> OutputProcesser:
    return OutputProcesser(
        sender or _Sender(),
        attn_tp_rank=0,
        metrics=SimpleNamespace(enabled=False, record_nan_abort=lambda: None),
    )


def _run_chunk(
    processor, state, rid, ids, prefix, length, logprobs, *, prefill=False, nan=False
):
    """Commit one prefill chunk of ``rid`` with its planned prompt logprobs."""
    op = _Op([rid], [length], [prefix], 1, prefill_lengths=[len(ids)])
    plan = input_logprob_plan_for_forward(op, {rid: state})
    final = prefix + length >= len(ids)
    if plan is not None:
        assert len(logprobs) == plan.num_rows
    result = _Result(
        [77] if final else [0],
        plan,
        logprobs if plan else None,
        nan_flags=[int(nan)],
    )
    return processor.post_process_forward_op(op, result, is_prefill_instance=prefill)


def test_chunks_accumulate_and_finalize_into_the_sglang_lists():
    ids = list(range(100, 110))
    sender = _Sender()
    processor = _processor(sender)
    state = _state(ids, start=2)
    processor.rid_to_state["a"] = state

    _run_chunk(processor, state, "a", ids, 0, 4, [-1.0, -2.0])  # positions 2, 3
    assert state.input_token_logprobs == [-1.0, -2.0, None, None, None, None, None]
    assert state.input_token_logprobs_val is None
    assert sender.items == []
    _run_chunk(processor, state, "a", ids, 4, 4, [-3.0, -4.0, -5.0, -6.0])
    _run_chunk(
        processor, state, "a", ids, 8, 2, [-7.0]
    )  # position 8; 9 is the output's

    # len(ids) - start entries: (None, ids[2]) then one logprob per prompt token.
    assert state.input_token_logprobs_val == [
        None,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
    ]
    assert state.input_token_logprobs_idx == ids[2:]
    assert state.input_token_logprobs is None
    assert state.output_ids == [77]
    assert state.output_token_logprobs_val == [-0.5]

    # Shipped once with the first frame, then [] on later frames.
    assert len(sender.items) == 1
    out = sender.items[0]
    assert out.input_token_logprobs_val == [state.input_token_logprobs_val]
    assert out.input_token_logprobs_idx == [ids[2:]]
    assert out.output_token_logprobs_val == [[-0.5]]
    state.output_ids.append(78)
    state.stream = True
    processor.stream_output(["a"], [state])
    assert sender.items[1].input_token_logprobs_val == [[]]
    assert sender.items[1].input_token_logprobs_idx == [[]]


def test_a_re_prefill_after_retraction_overwrites_the_same_positions():
    ids = list(range(100, 110))
    processor = _processor()
    state = _state(ids, start=0)
    processor.rid_to_state["a"] = state
    _run_chunk(processor, state, "a", ids, 0, 6, [-1.0] * 6)
    # Retracted and readmitted from scratch: the first chunk comes back.
    _run_chunk(processor, state, "a", ids, 0, 6, [-1.0] * 6)
    assert state.input_token_logprobs == [-1.0] * 6 + [None] * 3
    _run_chunk(processor, state, "a", ids, 6, 4, [-2.0] * 3)
    assert state.input_token_logprobs_val == [None] + [-1.0] * 6 + [-2.0] * 3
    assert len(state.input_token_logprobs_val) == len(ids)


def test_a_nan_flag_on_an_intermediate_chunk_terminates_the_request_at_completion():
    """The NaN guard flags the chunk whose prompt logprobs (or last-row logits)
    were corrupt; the chunk owes no token, so the flag is kept on the request
    and the abort is issued when the prompt completes, like a flag on the
    final chunk. The sanitized values ship with the aborted request, never a
    NaN."""
    from tokenspeed.runtime.engine.request_types import ABORT_CODE

    ids = list(range(100, 108))
    sender = _Sender()
    processor = _processor(sender)
    state = _state(ids, start=0)
    processor.rid_to_state["a"] = state
    # The guard already sanitized the corrupt row (here to -4.0) and set the flag.
    _run_chunk(processor, state, "a", ids, 0, 4, [-1.0, -4.0, -1.0, -1.0], nan=True)
    assert state.numerical_error_detected
    assert not state.finished
    _run_chunk(processor, state, "a", ids, 4, 4, [-2.0, -2.0, -2.0])
    assert state.finished
    assert state.finished_reason.err_type == ABORT_CODE.NumericalError
    [out] = sender.items
    assert out.input_token_logprobs_val == [
        [None, -1.0, -4.0, -1.0, -1.0, -2.0, -2.0, -2.0]
    ]


def test_finalize_refuses_a_gap_in_the_prompt_logprobs():
    state = _state([1, 2, 3, 4], start=0)
    state.record_input_token_logprobs(0, [-1.0, -2.0])
    with pytest.raises(RuntimeError, match="incomplete"):
        state.finalize_input_token_logprobs()


def test_default_start_yields_only_the_none_entry_without_gpu_work():
    ids = [5, 6, 7]
    sender = _Sender()
    processor = _processor(sender)
    state = _state(ids)
    processor.rid_to_state["a"] = state
    op = _Op(["a"], [3], [0], 1)
    assert input_logprob_plan_for_forward(op, {"a": state}) is None
    processor.post_process_forward_op(op, _Result([77]), is_prefill_instance=False)
    assert state.input_token_logprobs_val == [None]
    assert state.input_token_logprobs_idx == [7]
    assert sender.items[0].input_token_logprobs_val == [[None]]
    assert sender.items[0].input_token_logprobs_idx == [[7]]


def test_requests_without_logprobs_and_the_decode_role_ship_empty_lists():
    ids = [5, 6, 7]
    for state in (
        _state(ids, start=0, return_logprob=False),
        _state(ids, start=0, computes_prompt_logprobs=False),
    ):
        sender = _Sender()
        processor = _processor(sender)
        processor.rid_to_state["a"] = state
        op = _Op(["a"], [3], [0], 1)
        assert input_logprob_plan_for_forward(op, {"a": state}) is None
        processor.post_process_forward_op(op, _Result([77]), is_prefill_instance=False)
        assert state.input_token_logprobs_val is None
        assert sender.items[0].input_token_logprobs_val == [[]]
        assert sender.items[0].input_token_logprobs_idx == [[]]


def test_prefill_node_ships_prompt_logprobs_in_its_finished_frame():
    ids = list(range(100, 106))
    sender = _Sender()
    processor = _processor(sender)
    state = _state(ids, start=3)
    processor.rid_to_state["a"] = state
    _run_chunk(processor, state, "a", ids, 0, 6, [-1.0, -2.0], prefill=True)
    # Nothing streams until the KV transfer succeeded ...
    assert sender.items == []
    assert state.input_token_logprobs_val == [None, -1.0, -2.0]
    processor.finish_prefill_request("a")
    # ... then the finished frame carries the bootstrap token and both lists.
    [out] = sender.items
    assert out.output_ids == [[77]]
    assert out.output_token_logprobs_val == [[-0.5]]
    assert out.input_token_logprobs_val == [[None, -1.0, -2.0]]
    assert out.input_token_logprobs_idx == [[103, 104, 105]]


def test_decode_node_prepends_the_bootstrap_logprob_to_output_logprobs(
    caplog, monkeypatch
):
    # The colorful logger does not propagate; let caplog see its records.
    monkeypatch.setattr(output_module.logger, "propagate", True)
    processor = _processor()
    state = _state([1, 2, 3], start=0, computes_prompt_logprobs=False)
    state.computed_length = 3
    processor.rid_to_state["d"] = state
    with caplog.at_level("WARNING"):
        processor.on_remote_prefill_done("d", 101, 2, -0.25)
    assert state.output_ids == [101]
    assert state.output_token_logprobs_val == [-0.25]
    assert state.output_token_logprobs_idx == [101]
    assert "no bootstrap logprob" not in caplog.text

    # A prefill peer that sends no logprob (older version, or logprobs off
    # there): the token still lands, and the one-entry-short list is logged
    # since the client cannot tell from the data.
    other = _state([1, 2, 3], start=0, computes_prompt_logprobs=False)
    processor.rid_to_state["e"] = other
    with caplog.at_level("WARNING"):
        processor.on_remote_prefill_done("e", 102, 2, None)
    assert other.output_ids == [102]
    assert other.output_token_logprobs_val == []
    assert (
        "rid=e returns logprobs but the prefill peer sent no bootstrap logprob"
        in caplog.text
    )


def test_result_rows_must_match_the_plan():
    ids = list(range(100, 106))
    processor = _processor()
    state = _state(ids, start=0)
    processor.rid_to_state["a"] = state
    op = _Op(["a"], [6], [0], 1)
    plan = input_logprob_plan_for_forward(op, {"a": state})
    with pytest.raises(RuntimeError, match="rows for a plan"):
        processor.post_process_forward_op(
            op, _Result([77], plan, [-1.0]), is_prefill_instance=False
        )


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
    assert executor.nan_guard.flags.tolist() == [0, 0, 0, 0]

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
# Admission: the prefix-probe cap handed to the C++ scheduler
# --------------------------------------------------------------------------


def _handler(disaggregation_mode: str) -> RequestHandler:
    handler = object.__new__(RequestHandler)
    handler.server_args = SimpleNamespace(
        disaggregation_mode=disaggregation_mode,
        disaggregation_bootstrap_port=0,
    )
    handler.tokenizer = None
    handler.hf_eos_token_id = None
    handler.max_req_len = 4096
    # The layouts that refuse prompt logprobs or cap the generation budget
    # (LM-head TP / head TP under attention DP) are off in these cases.
    handler.supports_input_logprobs = True
    handler.max_new_tokens_budget = None
    return handler


def _recv_req(**overrides) -> TokenizedGenerateReqInput:
    fields = dict(
        rid="r",
        input_ids=list(range(10)),
        sampling_params=SamplingParams(max_new_tokens=4),
        return_logprob=True,
        logprob_start_len=3,
    )
    fields.update(overrides)
    return TokenizedGenerateReqInput(**fields)


@pytest.mark.parametrize(
    ("mode", "overrides", "expected"),
    [
        ("null", {}, 3),
        ("prefill", {}, 3),
        ("decode", {}, UNBOUNDED_CACHED_PREFIX_TOKENS),
        ("null", {"logprob_start_len": -1}, UNBOUNDED_CACHED_PREFIX_TOKENS),
        ("null", {"logprob_start_len": 0}, 0),
        (
            "null",
            {"return_logprob": False, "logprob_start_len": 0},
            UNBOUNDED_CACHED_PREFIX_TOKENS,
        ),
    ],
)
def test_handle_generate_request_caps_the_prefix_probe_at_the_start(
    mode, overrides, expected
):
    spec, state, _ = _handler(mode).handle_generate_request(_recv_req(**overrides))
    assert spec.max_cached_prefix_tokens == expected
    assert spec.max_new_tokens == 4
    assert state.wants_input_logprobs == (expected != UNBOUNDED_CACHED_PREFIX_TOKENS)


def test_make_spec_passes_the_cap_explicitly():
    spec = make_spec("r", [1, 2, 3], max_cached_prefix_tokens=2)
    assert spec.max_cached_prefix_tokens == 2
    assert (
        make_spec(
            "r", [1, 2, 3], max_cached_prefix_tokens=UNBOUNDED_CACHED_PREFIX_TOKENS
        ).max_cached_prefix_tokens
        == 2**31 - 1
    )


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
