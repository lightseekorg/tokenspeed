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

from types import SimpleNamespace

import torch
from tokenspeed_scheduler import RequestSpec, Scheduler, SchedulerConfig

import tokenspeed.runtime.engine.event_loop as event_loop_module
from tokenspeed.runtime.engine.event_loop import (
    EventLoop,
    _overlap_result_will_hit_length_limit,
)
from tokenspeed.runtime.engine.generation_output_processor import (
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class _Tokenizer:
    eos_token_id = None
    additional_stop_token_ids = None


class _Metrics:
    enabled = False


class _Sender:
    def __init__(self, trace):
        self.trace = trace

    def send_pyobj(self, _):
        self.trace.append("stream")


class _PrefillOp:
    request_ids = ["r"]
    request_pool_indices = [0]
    extend_prefix_lens = [0]
    prefill_lengths = [3]

    def __init__(self, input_length):
        self.input_lengths = [input_length]

    @staticmethod
    def num_extends():
        return 1


class _DecodeOp:
    request_ids = ["r"]
    request_pool_indices = [0]
    input_lengths = [1]
    extend_prefix_lens = []

    @staticmethod
    def num_extends():
        return 0


class _Result:
    output_tokens = torch.tensor([123], dtype=torch.int32)
    output_lengths = torch.tensor([1], dtype=torch.int32)
    output_logprobs = None
    output_nan_flags = None
    grammar_completion = None
    next_input_ids = None

    def __init__(self, trace):
        self.trace = trace

    def sync(self):
        self.trace.append("sync")


def _state(max_new_tokens=1, prompt_length=3):
    return RequestState(
        prompt_input_ids=list(range(prompt_length)),
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens,
            stop=[],
            ignore_eos=True,
        ),
        stream=False,
        tokenizer=_Tokenizer(),
    )


def test_intermediate_prefill_does_not_flush_overlap():
    state = _state()

    assert not _overlap_result_will_hit_length_limit(
        _PrefillOp(input_length=2), {"r": state}
    )
    assert _overlap_result_will_hit_length_limit(
        _PrefillOp(input_length=3), {"r": state}
    )


def test_only_last_decode_step_flushes_overlap():
    state = _state(max_new_tokens=2)
    assert not _overlap_result_will_hit_length_limit(_DecodeOp(), {"r": state})

    state.output_ids.append(101)
    assert _overlap_result_will_hit_length_limit(_DecodeOp(), {"r": state})


def test_length_terminal_result_commits_token_before_finish_and_release(monkeypatch):
    trace = []
    state = _state()
    processor = OutputProcesser(
        _Sender(trace),
        attn_tp_rank=0,
        metrics=_Metrics(),
    )
    processor.rid_to_state["r"] = state

    loop = EventLoop.__new__(EventLoop)
    loop.output_processor = processor
    loop.request_handler = SimpleNamespace(
        forward_ct=0,
        _profile_batch_predicate=lambda _: None,
    )
    loop.kv_transfer = None
    loop.scheduler = object()
    loop._publish_scheduler_kv_events = lambda: trace.append("publish")

    captured_events = []

    def record_advance(scheduler, events):
        assert scheduler is loop.scheduler
        captured_events.extend(events)
        trace.append("advance")

    monkeypatch.setattr(event_loop_module, "advance_forward", record_advance)

    committed = loop._commit_length_terminal_overlap_result(
        _PrefillOp(input_length=3), _Result(trace)
    )

    assert committed
    assert state.output_ids == [123]
    assert "r" not in processor.rid_to_state
    assert [type(event).__name__ for event in captured_events] == [
        "ExtendResult",
        "Finish",
    ]
    assert captured_events[0].tokens == [123]
    # Result synchronization happens before ExtendResult/Finish reach the
    # scheduler; Finish therefore cannot release KV ahead of the final token.
    assert trace == ["sync", "stream", "advance", "publish"]


def test_exact_fit_scheduler_finishes_without_retract_warning(capfd):
    config = SchedulerConfig()
    config.block_size = 4
    config.max_scheduled_tokens = 4
    config.max_batch_size = 1
    # Page zero is reserved, leaving exactly two usable pages: seven prompt
    # tokens plus one generated token fill them exactly.
    config.num_device_pages = 3
    config.num_host_pages = 0
    config.decode_input_tokens = 1
    config.overlap_schedule_depth = 1
    scheduler = Scheduler(config)
    spec = RequestSpec()
    spec.request_id = "r"
    spec.tokens = list(range(7))
    scheduler.submit_requests([spec])

    first_plan = scheduler.next_execution_plan()
    final_plan = scheduler.next_execution_plan()
    assert first_plan.forward[0].input_lengths == [4]
    final_prefill = final_plan.forward[0]
    assert final_prefill.input_lengths == [3]

    trace = []
    state = _state(prompt_length=7)
    processor = OutputProcesser(
        _Sender(trace),
        attn_tp_rank=0,
        metrics=_Metrics(),
    )
    processor.rid_to_state["r"] = state
    loop = EventLoop.__new__(EventLoop)
    loop.output_processor = processor
    loop.request_handler = SimpleNamespace(
        forward_ct=0,
        _profile_batch_predicate=lambda _: None,
    )
    loop.kv_transfer = None
    loop.scheduler = scheduler
    loop._publish_scheduler_kv_events = lambda: trace.append("publish")

    terminal_plan, committed = loop._next_overlap_execution_plan(
        final_prefill, _Result(trace)
    )
    assert committed
    assert not any(op.request_ids for op in terminal_plan.forward)
    assert scheduler.get_request_token_size("r") == -1
    assert state.output_ids == [123]
    assert state.finished_reason.to_json()["type"] == "length"
    captured = capfd.readouterr()
    assert "Retract failed" not in captured.out + captured.err


def test_nonterminal_result_remains_deferred(monkeypatch):
    trace = []
    loop = EventLoop.__new__(EventLoop)
    loop.output_processor = SimpleNamespace(rid_to_state={"r": _state(2)})

    monkeypatch.setattr(
        event_loop_module,
        "advance_forward",
        lambda *_: trace.append("advance"),
    )

    committed = loop._commit_length_terminal_overlap_result(
        _PrefillOp(input_length=3), _Result(trace)
    )

    assert not committed
    assert trace == []
