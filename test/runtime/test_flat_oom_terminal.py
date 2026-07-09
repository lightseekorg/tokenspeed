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

"""Unit tests for EventLoop._handle_flat_oom_terminals.

The C++ flat scheduler terminalizes a never-fits request internally
(AbortEvent; the reaper reclaims its resources) and reports the id on
``ExecutionPlan.flat_oom_request_ids``. The Python event loop must stream a
terminating abort finish to the client — otherwise the client hangs forever
— and must NOT send a ForwardEvent.Abort back (the request is already gone
scheduler-side).
"""

from __future__ import annotations

from tokenspeed.runtime.engine.event_loop import EventLoop
from tokenspeed.runtime.engine.generation_output_processor import (
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.engine.request_types import FINISH_ABORT
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


class _Plan:
    """ExecutionPlan stub carrying only the flat OOM binding field."""

    def __init__(self, flat_oom_request_ids):
        self.flat_oom_request_ids = flat_oom_request_ids


class _RadixPlan:
    """ExecutionPlan stub WITHOUT the field (older/radix binding shape)."""


def _state(input_ids: list[int]) -> RequestState:
    return RequestState(
        prompt_input_ids=input_ids,
        sampling_params=SamplingParams(max_new_tokens=8, stop=[], ignore_eos=True),
        stream=False,
        tokenizer=_Tokenizer(),
    )


def _make_loop(sender: _Sender) -> EventLoop:
    # The handler only touches self.output_processor; skip full engine init.
    loop = EventLoop.__new__(EventLoop)
    loop.output_processor = OutputProcesser(sender, attn_tp_rank=0, metrics=_Metrics())
    return loop


def test_flat_oom_terminal_streams_abort_finish_to_client():
    sender = _Sender()
    loop = _make_loop(sender)
    state = _state([1, 2, 3, 4])
    loop.output_processor.rid_to_state["oom"] = state

    loop._handle_flat_oom_terminals(_Plan(["oom"]))

    # Request finished with an error abort carrying the OOM message.
    assert isinstance(state.finished_reason, FINISH_ABORT)
    assert state.finished_reason.is_error
    assert "cannot fit" in state.finished_reason.message

    # The finish was streamed to the client and the state dropped from
    # tracking (publish_finished_at_admission pops it).
    assert "oom" not in loop.output_processor.rid_to_state
    assert len(sender.items) == 1
    out = sender.items[0]
    idx = out.rids.index("oom")
    reason = out.finished_reasons[idx]
    assert reason["type"] == "abort"
    assert "cannot fit" in reason["message"]


def test_flat_oom_terminal_leaves_other_requests_running():
    sender = _Sender()
    loop = _make_loop(sender)
    oom_state = _state([1, 2, 3, 4])
    live_state = _state([5, 6, 7])
    loop.output_processor.rid_to_state["oom"] = oom_state
    loop.output_processor.rid_to_state["live"] = live_state

    loop._handle_flat_oom_terminals(_Plan(["oom"]))

    assert not live_state.finished
    assert "live" in loop.output_processor.rid_to_state


def test_flat_oom_terminal_missing_rid_is_skipped():
    sender = _Sender()
    loop = _make_loop(sender)

    # rid already gone (e.g. client abort raced ahead): no crash, no stream.
    loop._handle_flat_oom_terminals(_Plan(["ghost"]))

    assert sender.items == []


def test_flat_oom_terminal_already_finished_state_is_skipped():
    sender = _Sender()
    loop = _make_loop(sender)
    state = _state([1, 2, 3])
    state.set_finish_with_abort("earlier abort")
    loop.output_processor.rid_to_state["done"] = state

    loop._handle_flat_oom_terminals(_Plan(["done"]))

    # Not re-finished, not streamed again, message unchanged.
    assert state.finished_reason.message == "earlier abort"
    assert sender.items == []
    assert "done" in loop.output_processor.rid_to_state


def test_plan_without_flat_oom_field_is_noop():
    sender = _Sender()
    loop = _make_loop(sender)
    state = _state([1, 2, 3])
    loop.output_processor.rid_to_state["r"] = state

    loop._handle_flat_oom_terminals(_RadixPlan())
    loop._handle_flat_oom_terminals(_Plan([]))

    assert not state.finished
    assert sender.items == []
