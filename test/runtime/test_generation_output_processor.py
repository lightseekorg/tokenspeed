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
