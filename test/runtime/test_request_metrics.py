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
