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

"""Request-side inter-token latency histograms."""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest
from prometheus_client import CollectorRegistry

from tokenspeed.runtime.metrics.collector import RequestMetrics

LABELS = {"model_name": "m"}
ITL = "tokenspeed:inter_token_latency_seconds"
SCHEDULER_ITL = "tokenspeed:scheduler_inter_token_latency_seconds"
DECODE_TPOT = "tokenspeed:decode_time_per_output_token_seconds"


def test_batched_inter_token_latency_matches_per_token_observe():
    registry = CollectorRegistry()
    metrics = RequestMetrics(LABELS, enabled=True, registry=registry)
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


def test_decode_time_per_output_token_recording():
    registry = CollectorRegistry()
    metrics = RequestMetrics(LABELS, enabled=True, registry=registry)
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
    metrics = RequestMetrics(LABELS, enabled=True, registry=registry)
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
