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

from __future__ import annotations

import torch

from tokenspeed.runtime.engine.generation_output_processor import (
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class _Sender:
    def __init__(self):
        self.items = []

    def send_pyobj(self, obj):
        self.items.append(obj)


class _Tokenizer:
    eos_token_id = None
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(str(i) for i in ids)


class _Metrics:
    enabled = False

    def __init__(self):
        self.nan_aborts = 0

    def record_nan_abort(self):
        self.nan_aborts += 1


class _ForwardOp:
    request_ids = ["prefill", "decode"]
    request_pool_indices = [0, 1]
    input_lengths = [4, 1]
    extend_prefix_lens = [0]

    def num_extends(self):
        return 1


class _ExecutionResult:
    output_tokens = torch.tensor([11, 22], dtype=torch.int32)
    output_lengths = torch.tensor([1, 1], dtype=torch.int32)
    output_logprobs = None
    output_nan_flags = None
    grammar_completion = None

    def sync(self):
        return None


def _state(input_ids: list[int], *, computed_length: int = 0) -> RequestState:
    state = RequestState(
        prompt_input_ids=input_ids,
        sampling_params=SamplingParams(max_new_tokens=8, stop=[], ignore_eos=True),
        stream=False,
        tokenizer=_Tokenizer(),
    )
    state.computed_length = computed_length
    return state


def test_mixed_forward_updates_reserve_for_decode_slots_only():
    sender = _Sender()
    processor = OutputProcesser(
        sender,
        global_rank=0,
        metrics=_Metrics(),
    )
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    processor.rid_to_state["decode"] = _state([5, 6, 7], computed_length=3)

    events = processor.post_process_forward_op(_ForwardOp(), _ExecutionResult())

    reserve_events = [
        event for event in events if type(event).__name__ == "UpdateReserveNumTokens"
    ]
    assert len(reserve_events) == 1
    assert reserve_events[0].request_id == "decode"
    assert reserve_events[0].reserve_num_tokens_in_next_schedule_event == 1


def test_mark_abort_notify_client_flag():
    """Pause-initiated aborts must flag the request to stream a terminating
    finish to the (passive) client; client-initiated aborts must not."""
    sender = _Sender()
    processor = OutputProcesser(sender, global_rank=0, metrics=_Metrics())

    pause_state = _state([1, 2, 3])
    processor.rid_to_state["pause"] = pause_state
    processor.mark_abort("pause", notify_client=True)
    assert pause_state.to_abort
    assert pause_state.abort_notify_client
    assert pause_state.finished  # finished_reason materialized

    client_state = _state([1, 2, 3])
    processor.rid_to_state["client"] = client_state
    processor.mark_abort("client")  # default: client tore down its own state
    assert client_state.to_abort
    assert not client_state.abort_notify_client


def test_nan_flag_finishes_request_with_numerical_error():
    """A request flagged by the NaN guard is finished with
    ABORT_CODE.NumericalError while the rest of the batch continues."""
    from tokenspeed.runtime.engine.request_types import ABORT_CODE, FINISH_ABORT

    sender = _Sender()
    metrics = _Metrics()
    processor = OutputProcesser(sender, global_rank=0, metrics=metrics)
    prefill_state = _state([1, 2, 3, 4])
    decode_state = _state([5, 6, 7], computed_length=3)
    processor.rid_to_state["prefill"] = prefill_state
    processor.rid_to_state["decode"] = decode_state

    result = _ExecutionResult()
    # Flag only the decode slot.
    result.output_nan_flags = torch.tensor([0, 1], dtype=torch.int32)

    events = processor.post_process_forward_op(_ForwardOp(), result)

    # Flagged request: aborted with NumericalError, removed from tracking.
    # The scheduler gets an Abort (NOT Finish) event — AbortEvent skips the
    # radix-tree insert and host-KV writeback, so corrupted KV is not reused.
    assert isinstance(decode_state.finished_reason, FINISH_ABORT)
    assert decode_state.finished_reason.err_type == ABORT_CODE.NumericalError
    assert "decode" not in processor.rid_to_state
    abort_events = [e for e in events if type(e).__name__ == "Abort"]
    assert [e.request_id for e in abort_events] == ["decode"]
    assert not [e for e in events if type(e).__name__ == "Finish"]
    assert metrics.nan_aborts == 1

    # Unflagged request keeps running untouched.
    assert not prefill_state.finished
    assert "prefill" in processor.rid_to_state
    assert prefill_state.output_ids == [11]

    # The abort finish reason is streamed to the client.
    assert len(sender.items) == 1
    out = sender.items[0]
    idx = out.rids.index("decode")
    assert out.finished_reasons[idx]["type"] == "abort"
    assert out.finished_reasons[idx]["err_type"] == ABORT_CODE.NumericalError.value


def test_nan_flag_keeps_single_sanitized_token():
    """A NaN-flagged spec-decode slot keeps exactly one (sanitized) token so
    extend-result accounting matches a normal mid-step finish."""
    sender = _Sender()
    metrics = _Metrics()
    processor = OutputProcesser(
        sender,
        global_rank=0,
        spec_algorithm="eagle",
        spec_num_tokens=4,
        metrics=metrics,
    )
    decode_state = _state([5, 6, 7], computed_length=3)
    processor.rid_to_state["decode"] = decode_state

    class _SpecForwardOp:
        request_ids = ["decode"]
        request_pool_indices = [0]
        input_lengths = [1]
        extend_prefix_lens = []

        def num_extends(self):
            return 0

    result = _ExecutionResult()
    result.output_tokens = torch.tensor([11, 22, 33, 44], dtype=torch.int32)
    result.output_lengths = torch.tensor([3], dtype=torch.int32)
    result.output_nan_flags = torch.tensor([1], dtype=torch.int32)

    events = processor.post_process_forward_op(_SpecForwardOp(), result)

    assert decode_state.finished
    # Only the first of the 3 accepted tokens is kept.
    assert decode_state.output_ids == [11]
    extend_events = [e for e in events if type(e).__name__ == "ExtendResult"]
    assert len(extend_events) == 1
    assert list(extend_events[0].tokens) == [11]
    assert metrics.nan_aborts == 1


def test_nan_flag_skips_first_token_pd_handoff():
    """NaN-terminated requests must not hand their bootstrap token to the PD
    transfer layer — their KV is suspect."""
    sender = _Sender()
    processor = OutputProcesser(sender, global_rank=0, metrics=_Metrics())
    processor.rid_to_state["prefill"] = _state([1, 2, 3, 4])
    processor.rid_to_state["decode"] = _state([5, 6, 7], computed_length=3)

    result = _ExecutionResult()
    result.next_input_ids = None
    result.output_nan_flags = torch.tensor([1, 0], dtype=torch.int32)

    handoffs = []
    processor.post_process_forward_op(
        _ForwardOp(),
        result,
        on_first_token=lambda rid, *a: handoffs.append(rid),
    )

    # Flagged prefill slot is skipped; the healthy decode slot still hands off.
    assert handoffs == ["decode"]
