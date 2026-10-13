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

"""tokenspeed:request_queue_time_seconds — the per-request waiting-queue histogram."""

from __future__ import annotations

import time
from types import SimpleNamespace

from prometheus_client import CollectorRegistry

from tokenspeed.runtime.engine.event_loop import EventLoop
from tokenspeed.runtime.engine.generation_output_processor import RequestState
from tokenspeed.runtime.metrics.collector import EngineMetrics
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

LABELS = {"model_name": "m"}
HISTOGRAM = "tokenspeed:request_queue_time_seconds"


def _state() -> RequestState:
    return RequestState(
        prompt_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=8, stop=[], ignore_eos=True),
        stream=False,
        tokenizer=None,
        computes_prompt_logprobs=True,
    )


def _make_loop(*, enabled: bool = True):
    registry = CollectorRegistry()
    metrics = EngineMetrics(LABELS, enabled=enabled, registry=registry)
    submitted = []
    loop = SimpleNamespace(
        metrics=metrics,
        output_processor=SimpleNamespace(rid_to_state={}),
        _l3_hooks=SimpleNamespace(submit_requests=submitted.extend),
    )
    return loop, registry, submitted


def _sample(registry, suffix: str, le: str | None = None):
    labels = dict(LABELS)
    if le is not None:
        labels["le"] = le
    return registry.get_sample_value(f"{HISTOGRAM}_{suffix}", labels)


def test_submit_stamps_enqueue_time_on_the_registered_state():
    loop, _, submitted = _make_loop()
    state = _state()
    loop.output_processor.rid_to_state["r"] = state

    before = time.time()
    specs = [SimpleNamespace(request_id="r"), SimpleNamespace(request_id="ghost")]
    EventLoop._submit_scheduler_requests(loop, specs)

    # One stamp per batch, after the batch was submitted to the waiting queue;
    # an unregistered rid is skipped rather than an error.
    assert before <= state.scheduler_enqueue_time <= time.time()
    assert submitted == specs


def test_first_dispatch_observes_the_queue_wait_in_its_bucket():
    loop, registry, _ = _make_loop()
    state = _state()
    loop.output_processor.rid_to_state["r"] = state
    state.scheduler_enqueue_time = time.time() - 4.0

    forward_op = SimpleNamespace(request_ids=["r"])
    EventLoop._record_request_queue_time(loop, forward_op)

    assert state.queue_time_recorded
    assert _sample(registry, "count") == 1.0
    assert 3.5 < _sample(registry, "sum") < 4.5
    # 4s lands between the 2.5 and 5.0 bucket bounds of the grid.
    assert _sample(registry, "bucket", le="2.5") == 0.0
    assert _sample(registry, "bucket", le="5.0") == 1.0


def test_a_request_is_observed_once_across_redispatches():
    loop, registry, _ = _make_loop()
    state = _state()
    loop.output_processor.rid_to_state["r"] = state
    state.scheduler_enqueue_time = time.time() - 1.0

    forward_op = SimpleNamespace(request_ids=["r"])
    EventLoop._record_request_queue_time(loop, forward_op)
    # Later decode rounds, or a retraction re-dispatch, carry the same rid but
    # must not produce a second observation.
    EventLoop._record_request_queue_time(loop, forward_op)

    assert _sample(registry, "count") == 1.0


def test_unstamped_and_unregistered_requests_are_skipped():
    loop, registry, _ = _make_loop()
    loop.output_processor.rid_to_state["r"] = _state()

    forward_op = SimpleNamespace(request_ids=["r", "ghost"])
    EventLoop._record_request_queue_time(loop, forward_op)

    assert _sample(registry, "count") is None


def test_disabled_metrics_neither_stamp_nor_observe():
    loop, _, submitted = _make_loop(enabled=False)
    state = _state()
    loop.output_processor.rid_to_state["r"] = state

    EventLoop._submit_scheduler_requests(loop, [SimpleNamespace(request_id="r")])
    assert state.scheduler_enqueue_time == 0.0
    assert submitted

    forward_op = SimpleNamespace(request_ids=["r"])
    EventLoop._record_request_queue_time(loop, forward_op)
    assert not state.queue_time_recorded
