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

"""Request-side metric labels, token histograms, and inter-token latency."""

import math
import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest
from prometheus_client import CollectorRegistry

from tokenspeed.runtime.engine.request_types import (
    ABORT_CODE,
    FINISH_ABORT,
)
from tokenspeed.runtime.metrics.collector import (
    _ABORT_REASON_LABELS,
    EngineMetrics,
    RequestFinishStats,
    RequestMetrics,
    build_1_2_5_buckets,
    finish_reason_label,
)

LABELS = {"model_name": "m"}
MAX_MODEL_LEN = 100
ITL = "tokenspeed:inter_token_latency_seconds"
SCHEDULER_ITL = "tokenspeed:scheduler_inter_token_latency_seconds"
TTFT = "tokenspeed:time_to_first_token_seconds"
E2E = "tokenspeed:e2e_request_latency_seconds"
REQUEST_PROMPT_TOKENS = "tokenspeed:request_prompt_tokens"
REQUEST_GENERATION_TOKENS = "tokenspeed:request_generation_tokens"
NUM_REQUESTS = "tokenspeed:num_requests_total"
REQUEST_SUCCESS = "tokenspeed:request_success_total"
NUM_PREEMPTIONS = "tokenspeed:num_preemptions_total"


def _make_metrics():
    registry = CollectorRegistry()
    metrics = RequestMetrics(
        LABELS,
        enabled=True,
        max_model_len=MAX_MODEL_LEN,
        reporters=["prometheus"],
        registry=registry,
    )
    return metrics, registry


def _cumulative_buckets(registry, name):
    """(le, cumulative count) pairs exposed for `name`, sorted by le."""
    buckets = []
    for metric in registry.collect():
        if metric.name != name:
            continue
        for sample in metric.samples:
            if sample.name.endswith("_bucket"):
                le = sample.labels["le"]
                buckets.append((math.inf if le == "+Inf" else float(le), sample.value))
    return sorted(buckets)


DECODE_TPOT = "tokenspeed:decode_time_per_output_token_seconds"


def test_batched_inter_token_latency_matches_per_token_observe():
    registry = CollectorRegistry()
    metrics = RequestMetrics(
        LABELS,
        enabled=True,
        max_model_len=MAX_MODEL_LEN,
        reporters=["prometheus"],
        registry=registry,
    )
    metrics.observe_inter_token_latency(0.030, 3)  # 0.010/token
    metrics.observe_inter_token_latency(0.400, 4)  # 0.100/token
    metrics.observe_scheduler_inter_token_latency(0.020, 2)

    def sample(name, **labels):
        return registry.get_sample_value(name, {**LABELS, **labels})

    assert sample(f"{ITL}_bucket", le="0.01") == 3.0
    assert sample(f"{ITL}_bucket", le="0.1") == 7.0
    assert sample(f"{ITL}_count") == 7.0
    assert sample(f"{ITL}_sum") == pytest.approx(0.43)
    assert sample(f"{SCHEDULER_ITL}_count") == 2.0
    assert sample(f"{SCHEDULER_ITL}_sum") == pytest.approx(0.02)


def test_build_1_2_5_buckets_grid():
    # 1-2-5 mantissas across decades, capped at max_model_len.
    assert build_1_2_5_buckets(100) == [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    assert build_1_2_5_buckets(99) == [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    assert build_1_2_5_buckets(1) == [1.0]


def test_finish_reason_label_maps_the_bounded_vocabulary():
    assert set(_ABORT_REASON_LABELS) == set(ABORT_CODE)
    assert finish_reason_label(None) == "unknown"
    assert finish_reason_label({"type": "stop", "matched": 5}) == "stop"
    assert finish_reason_label({"type": "length", "length": 16}) == "length"
    assert (
        finish_reason_label({"type": "abort", "message": "x", "err_type": 521})
        == "abort:transfer_failed"
    )
    assert (
        finish_reason_label({"type": "abort", "message": "x", "err_type": 522})
        == "abort:unknown_error"
    )
    assert (
        finish_reason_label({"type": "abort", "message": "x", "err_type": 523})
        == "abort:numerical_error"
    )
    assert (
        finish_reason_label({"type": "abort", "message": "x", "err_type": 524})
        == "abort:capacity_abort"
    )
    # An abort with a missing or unrecognized err_type stays plain "abort" —
    # forward-compatible with a newer peer's abort codes, still bounded.
    assert finish_reason_label({"type": "abort", "message": "x"}) == "abort"
    assert finish_reason_label({"type": "abort", "err_type": 999}) == "abort"
    assert finish_reason_label({"type": "some_future_reason"}) == "unknown"
    # The object form (BaseFinishReason before to_json()) maps the same way.
    assert finish_reason_label({"type": "stop", "matched": 5}) == "stop"
    assert finish_reason_label({"type": "length", "length": 16}) == "length"
    assert finish_reason_label(FINISH_ABORT()) == "abort:unknown_error"
    assert (
        finish_reason_label(FINISH_ABORT(err_type=ABORT_CODE.NumericalError))
        == "abort:numerical_error"
    )
    assert (
        finish_reason_label(FINISH_ABORT(err_type=ABORT_CODE.CapacityAbort))
        == "abort:capacity_abort"
    )


def test_request_counters_carry_finish_reason_and_stream_labels():
    metrics, registry = _make_metrics()
    metrics.record_request_finish(
        RequestFinishStats(
            prompt_tokens=8,
            generation_tokens=4,
            e2e_latency=2.0,
            finish_reason="stop",
            stream=True,
        )
    )
    metrics.record_request_finish(
        RequestFinishStats(
            prompt_tokens=4,
            generation_tokens=1,
            e2e_latency=0.5,
            finished_ok=False,
            finish_reason="abort:capacity_abort",
            stream=False,
        )
    )

    stop_labels = {**LABELS, "finish_reason": "stop", "stream": "true"}
    abort_labels = {
        **LABELS,
        "finish_reason": "abort:capacity_abort",
        "stream": "false",
    }
    assert registry.get_sample_value(NUM_REQUESTS, stop_labels) == 1.0
    assert registry.get_sample_value(NUM_REQUESTS, abort_labels) == 1.0
    # Success counts only the non-abort finish.
    assert registry.get_sample_value(REQUEST_SUCCESS, stop_labels) == 1.0
    assert registry.get_sample_value(REQUEST_SUCCESS, abort_labels) is None
    # No series without the new labels remains.
    assert registry.get_sample_value(NUM_REQUESTS, LABELS) is None


def test_ttft_and_e2e_carry_stream_label():
    metrics, registry = _make_metrics()
    metrics.observe_time_to_first_token(0.4, stream=True)
    metrics.record_request_finish(
        RequestFinishStats(
            prompt_tokens=8,
            generation_tokens=4,
            e2e_latency=2.0,
            finish_reason="stop",
            stream=True,
        )
    )

    stream_labels = {**LABELS, "stream": "true"}
    assert registry.get_sample_value(f"{TTFT}_count", stream_labels) == 1.0
    assert registry.get_sample_value(f"{E2E}_count", stream_labels) == 1.0
    # No series without the stream label remains.
    assert registry.get_sample_value(f"{TTFT}_count", LABELS) is None
    assert registry.get_sample_value(f"{E2E}_count", LABELS) is None


def test_request_token_histograms_bucket_grid():
    metrics, registry = _make_metrics()
    metrics.record_request_finish(
        RequestFinishStats(
            prompt_tokens=8,
            generation_tokens=3,
            e2e_latency=1.0,
            finish_reason="stop",
            stream=False,
        )
    )

    child_labels = {**LABELS, "stream": "false"}
    prompt_buckets = dict(_cumulative_buckets(registry, REQUEST_PROMPT_TOKENS))
    generation_buckets = dict(_cumulative_buckets(registry, REQUEST_GENERATION_TOKENS))
    # Grid: the 1-2-5 series capped at max_model_len (100 here).
    assert sorted(prompt_buckets) == [
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        math.inf,
    ]
    assert (
        registry.get_sample_value(f"{REQUEST_PROMPT_TOKENS}_count", child_labels) == 1.0
    )
    assert (
        registry.get_sample_value(f"{REQUEST_PROMPT_TOKENS}_sum", child_labels) == 8.0
    )
    # prompt_tokens=8 lands in the le=10 bucket; generation_tokens=3 in le=5.
    assert prompt_buckets[5.0] == 0.0
    assert prompt_buckets[10.0] == 1.0
    assert generation_buckets[2.0] == 0.0
    assert generation_buckets[5.0] == 1.0
    assert (
        registry.get_sample_value(f"{REQUEST_GENERATION_TOKENS}_sum", child_labels)
        == 3.0
    )


def test_engine_preemptions_counter():
    registry = CollectorRegistry()
    metrics = EngineMetrics(
        LABELS, enabled=True, reporters=["prometheus"], registry=registry
    )

    metrics.record_preemptions(0)
    assert registry.get_sample_value(NUM_PREEMPTIONS, LABELS) is None
    metrics.record_preemptions(2)
    metrics.record_preemptions(1)
    assert registry.get_sample_value(NUM_PREEMPTIONS, LABELS) == 3.0


def test_decode_time_per_output_token_recording():
    registry = CollectorRegistry()
    metrics = RequestMetrics(
        LABELS,
        enabled=True,
        max_model_len=MAX_MODEL_LEN,
        reporters=["prometheus"],
        registry=registry,
    )
    metrics.observe_decode_time_per_output_token(0.010)
    metrics.observe_decode_time_per_output_token(0.030)

    def sample(name, **labels):
        return registry.get_sample_value(name, {**LABELS, **labels})

    assert sample(f"{DECODE_TPOT}_bucket", le="0.01") == 1.0
    assert sample(f"{DECODE_TPOT}_bucket", le="0.03") == 2.0
    assert sample(f"{DECODE_TPOT}_count") == 2.0
    assert sample(f"{DECODE_TPOT}_sum") == pytest.approx(0.04)


def test_observe_inter_token_latency_rejects_nonpositive_token_counts():
    registry = CollectorRegistry()
    metrics = RequestMetrics(
        LABELS,
        enabled=True,
        max_model_len=MAX_MODEL_LEN,
        reporters=["prometheus"],
        registry=registry,
    )
    for observe in (
        metrics.observe_inter_token_latency,
        metrics.observe_scheduler_inter_token_latency,
    ):
        observe(0.030, 0)
        observe(0.030, -2)

    def sample(name, **labels):
        return registry.get_sample_value(name, {**LABELS, **labels})

    # No labeled series may come into existence from rejected observations.
    assert sample(f"{ITL}_count") is None
    assert sample(f"{SCHEDULER_ITL}_count") is None

    metrics.observe_inter_token_latency(0.030, 3)
    metrics.observe_scheduler_inter_token_latency(0.030, 3)
    assert sample(f"{ITL}_count") == 3.0
    assert sample(f"{ITL}_sum") == pytest.approx(0.03)
    assert sample(f"{SCHEDULER_ITL}_count") == 3.0
