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

"""Record the latency of selected functions."""

from tokenspeed.runtime.metrics.utils import exponential_buckets

enable_metrics = False


def enable_func_timer():
    """Enable Prometheus-backed function latency metrics."""
    # We need to import prometheus_client after setting the env variable
    # `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import Histogram

    global enable_metrics, FUNC_LATENCY
    enable_metrics = True

    FUNC_LATENCY = Histogram(
        "tokenspeed:func_latency_seconds",
        "Function latency in seconds",
        # captures latency in range [50ms - ~50s]
        buckets=exponential_buckets(start=0.05, width=1.5, length=18),
        labelnames=["name"],
    )


FUNC_LATENCY = None
