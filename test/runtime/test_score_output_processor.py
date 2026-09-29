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

"""Scheduler-side Score API readout tests.

Drives ``OutputProcesser.post_process_forward_op`` with fakes (same
pattern as test_generation_output_processor.py) to pin the score
contract: the label readout is captured exactly once, at the final
prefill chunk, finalized per the request's ``score_apply_softmax`` flag,
and packed into the ``output_score_vals`` column of ``BatchTokenIDOut``.
"""

from __future__ import annotations

import math

import pytest
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

    def record_nan_abort(self):
        pass


class _ForwardOp:
    # One extend request ("score"), prompt fully inside this chunk.
    request_ids = ["score"]
    request_pool_indices = [0]
    input_lengths = [4]
    prefill_lengths = [4]
    extend_prefix_lens = [0]
    extend_replay_lens = [0]

    def num_extends(self):
        return 1


class _MidChunkForwardOp(_ForwardOp):
    # Same request, but this chunk ends mid-prompt: no answer boundary yet.
    prefill_lengths = [8]


class _ExecutionResult:
    output_tokens = torch.tensor([11], dtype=torch.int32)
    output_lengths = torch.tensor([1], dtype=torch.int32)
    output_logprobs = None
    output_nan_flags = None
    grammar_completion = None
    next_input_ids = None
    score_logprobs = torch.tensor([[-0.5, -2.0]])


def _score_state(apply_softmax: bool) -> RequestState:
    return RequestState(
        prompt_input_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(
            max_new_tokens=0,
            score_label_token_ids=[5, 9],
            score_apply_softmax=apply_softmax,
        ),
        stream=False,
        tokenizer=_Tokenizer(),
    )


def _processor_with(state: RequestState):
    sender = _Sender()
    processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    processor.rid_to_state["score"] = state
    return processor, sender


def test_score_readout_raw_logprobs():
    processor, sender = _processor_with(_score_state(apply_softmax=False))
    events = processor.post_process_forward_op(
        _ForwardOp(), _ExecutionResult(), is_prefill_instance=False
    )

    # The request finished (max_new_tokens=0) and its row was emitted raw.
    assert len(sender.items) == 1
    out = sender.items[0]
    assert out.output_score_vals == [[-0.5, -2.0]]
    assert out.finished_reasons[0] == {"type": "length", "length": 0}
    event_names = [type(event).__name__ for event in events]
    assert "Finish" in event_names
    # The sampled token still rode the FSM events, but the score column is
    # the contract output.
    assert "ExtendResult" in event_names


def test_score_readout_apply_softmax():
    processor, sender = _processor_with(_score_state(apply_softmax=True))
    processor.post_process_forward_op(
        _ForwardOp(), _ExecutionResult(), is_prefill_instance=False
    )

    (row,) = sender.items[0].output_score_vals
    # [-0.5, -2.0] -> softmax across the label set.
    expected_yes = 1.0 / (1.0 + math.exp(-1.5))
    assert row[0] == pytest.approx(expected_yes, rel=1e-6)
    assert sum(row) == pytest.approx(1.0, rel=1e-9)
    # Raw logprob ranking is preserved: label 0 stays the argmax.
    assert row[0] > row[1]


def test_score_readout_skips_mid_chunk():
    processor, sender = _processor_with(_score_state(apply_softmax=True))
    events = processor.post_process_forward_op(
        _MidChunkForwardOp(), _ExecutionResult(), is_prefill_instance=False
    )

    # Mid-chunk: no answer boundary, no finish, no streamed output, and the
    # request stays registered for the final chunk.
    assert sender.items == []
    assert processor.rid_to_state["score"].score_vals is None
    assert [type(event).__name__ for event in events] == ["ExtendResult"]


def test_score_request_without_readout_finishes_without_scores():
    # The fully-cached-prefill edge produces no logits row; the request must
    # still finish cleanly (the frontend surfaces the missing scores).
    processor, sender = _processor_with(_score_state(apply_softmax=True))
    result = _ExecutionResult()
    result.score_logprobs = None
    processor.post_process_forward_op(_ForwardOp(), result, is_prefill_instance=False)

    assert len(sender.items) == 1
    assert sender.items[0].output_score_vals == [[]]
    assert sender.items[0].finished_reasons[0] == {"type": "length", "length": 0}
