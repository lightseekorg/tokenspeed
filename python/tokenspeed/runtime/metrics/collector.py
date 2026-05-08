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

from dataclasses import dataclass
from enum import Enum

from tokenspeed.runtime.utils.env import envs

TOKENSPEED_TEST_REQUEST_TIME_STATS = envs.TOKENSPEED_TEST_REQUEST_TIME_STATS.get()


@dataclass
class TimeStats:
    """
    Store the timestamps for each stage of a request.

    Unified: wait_queue -> forward -> completion
    Prefill: bootstrap_queue -> wait_queue -> forward -> transfer_queue -> completion
    Decode: prealloc_queue -> transfer_queue -> wait_queue -> forward -> completion
    """

    lb_entry_time: float = 0.0
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    completion_time: float = 0.0
    prefill_bootstrap_queue_entry_time: float = 0.0
    prefill_transfer_queue_entry_time: float = 0.0
    decode_prealloc_queue_entry_time: float = 0.0
    decode_transfer_queue_entry_time: float = 0.0

    class RequestType(Enum):
        UNIFIED = "unified"
        PREFILL = "prefill"
        DECODE = "decode"
        INVALID = "invalid"

    def get_queueing_time(self) -> float:
        return self.forward_entry_time - self.wait_queue_entry_time

    def __str__(self) -> str:
        # if unified
        _type = self.get_type()

        if _type == self.RequestType.UNIFIED:
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if TOKENSPEED_TEST_REQUEST_TIME_STATS:
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time}"
        if _type == self.RequestType.PREFILL:
            bootstrap_duration = (
                self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time
            )

            queue_duration = self.forward_entry_time - self.wait_queue_entry_time

            forward_duration = self.completion_time - self.forward_entry_time

            if TOKENSPEED_TEST_REQUEST_TIME_STATS:
                assert (
                    bootstrap_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"
            return f"bootstrap_duration={self.format_duration(bootstrap_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.prefill_bootstrap_queue_entry_time}"
        # if decode
        if _type == self.RequestType.DECODE:
            prealloc_duration = (
                self.decode_transfer_queue_entry_time
                - self.decode_prealloc_queue_entry_time
            )

            transfer_duration = (
                self.wait_queue_entry_time - self.decode_transfer_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if TOKENSPEED_TEST_REQUEST_TIME_STATS:
                assert (
                    prealloc_duration >= 0
                    and transfer_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"prealloc_duration={self.format_duration(prealloc_duration)}, transfer_duration={self.format_duration(transfer_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.decode_prealloc_queue_entry_time}"
        return "Invalid Time Stats"

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"

    def get_type(self) -> RequestType:
        """Determine the type of request based on timestamp values."""
        if (
            self.prefill_bootstrap_queue_entry_time == 0.0
            and self.prefill_transfer_queue_entry_time == 0.0
            and self.decode_prealloc_queue_entry_time == 0.0
            and self.decode_transfer_queue_entry_time == 0.0
        ):
            return self.RequestType.UNIFIED
        elif (
            self.prefill_bootstrap_queue_entry_time > 0.0
            and self.prefill_transfer_queue_entry_time > 0.0
        ):
            return self.RequestType.PREFILL
        elif (
            self.decode_prealloc_queue_entry_time > 0.0
            and self.decode_transfer_queue_entry_time > 0.0
            and self.wait_queue_entry_time > 0.0
        ):
            return self.RequestType.DECODE
        else:
            return self.RequestType.INVALID


@dataclass
class RequestFinishStats:
    prompt_tokens: int
    generation_tokens: int
    e2e_latency: float
    cached_prompt_tokens: int = 0
    finished_ok: bool = True


class EngineMetrics:
    def __init__(
        self,
        labels: dict[str, str],
        *,
        enabled: bool,
        registry=None,
    ) -> None:
        self.enabled = enabled
        self.labels = labels

        if self.enabled:
            self._init_prometheus(labels, registry=registry)

    def _init_prometheus(self, labels: dict[str, str], *, registry=None) -> None:
        from prometheus_client import Counter, Gauge, Histogram

        labelnames = list(labels.keys())
        kw = {"registry": registry} if registry is not None else {}
        self.num_requests_running = Gauge(
            name="tokenspeed:num_requests_running",
            documentation="Requests with scheduler-side generation state (decode path).",
            labelnames=labelnames,
            multiprocess_mode="livemax",
            **kw,
        )
        self.num_requests_waiting = Gauge(
            name="tokenspeed:num_requests_waiting",
            documentation="Requests waiting in the C++ scheduler queue.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
            **kw,
        )
        self.kv_cache_usage_ratio = Gauge(
            name="tokenspeed:kv_cache_usage_ratio",
            documentation="Fraction of device KV pages in use (0-1).",
            labelnames=labelnames,
            multiprocess_mode="livemax",
            **kw,
        )
        self.iteration_tokens_total = Histogram(
            name="tokenspeed:iteration_tokens_total",
            documentation="Tokens scheduled in one scheduler forward step.",
            labelnames=labelnames,
            buckets=[
                0.0,
                1.0,
                2.0,
                4.0,
                8.0,
                16.0,
                32.0,
                64.0,
                128.0,
                256.0,
                512.0,
                1024.0,
                2048.0,
                4096.0,
                8192.0,
            ],
            **kw,
        )
        self.spec_decode_num_accepted_tokens = Counter(
            name="tokenspeed:spec_decode_num_accepted_tokens_total",
            documentation=(
                "Accepted speculative draft tokens (excludes the bonus token sampled "
                "after verify)."
            ),
            labelnames=labelnames,
            **kw,
        )
        self.spec_decode_num_draft_tokens = Counter(
            name="tokenspeed:spec_decode_num_draft_tokens_total",
            documentation="Draft tokens proposed across verify steps.",
            labelnames=labelnames,
            **kw,
        )
        self.spec_decode_num_drafts = Counter(
            name="tokenspeed:spec_decode_num_drafts_total",
            documentation="Number of speculative verify rounds (per request-slot).",
            labelnames=labelnames,
            **kw,
        )

    def set_scheduler_snapshot(
        self, *, running: int, waiting: int, kv_cache_usage_ratio: float
    ) -> None:
        if not self.enabled:
            return
        self.num_requests_running.labels(**self.labels).set(running)
        self.num_requests_waiting.labels(**self.labels).set(waiting)
        self.kv_cache_usage_ratio.labels(**self.labels).set(kv_cache_usage_ratio)

    def observe_iteration_tokens(self, num_tokens: float) -> None:
        if self.enabled and num_tokens >= 0:
            self.iteration_tokens_total.labels(**self.labels).observe(num_tokens)

    def record_scheduler_iteration(
        self,
        *,
        running: int,
        waiting: int,
        num_active_pages: int,
        num_total_pages: int,
        num_iteration_tokens: int,
    ) -> None:
        if not self.enabled:
            return
        ratio = num_active_pages / num_total_pages if num_total_pages else 0.0
        self.set_scheduler_snapshot(
            running=running,
            waiting=waiting,
            kv_cache_usage_ratio=ratio,
        )
        if num_iteration_tokens > 0:
            self.observe_iteration_tokens(float(num_iteration_tokens))

    def record_spec_decode_step(
        self,
        *,
        num_decode_slots: int,
        accepted_draft_tokens: int,
        draft_width: int,
    ) -> None:
        if not self.enabled or num_decode_slots <= 0:
            return
        self.spec_decode_num_drafts.labels(**self.labels).inc(num_decode_slots)
        self.spec_decode_num_draft_tokens.labels(**self.labels).inc(
            num_decode_slots * draft_width
        )
        self.spec_decode_num_accepted_tokens.labels(**self.labels).inc(
            max(0, accepted_draft_tokens)
        )


class RequestMetrics:
    def __init__(
        self,
        labels: dict[str, str],
        *,
        enabled: bool,
        registry=None,
    ) -> None:
        self.enabled = enabled
        self.labels = labels

        if self.enabled:
            self._init_prometheus(labels, registry=registry)

    def _init_prometheus(self, labels: dict[str, str], *, registry=None) -> None:
        # We need to import prometheus_client after setting the env variable PROMETHEUS_MULTIPROC_DIR
        from prometheus_client import Counter, Histogram

        labelnames = list(labels.keys())
        kw = {"registry": registry} if registry is not None else {}

        self.prompt_tokens_total = Counter(
            name="tokenspeed:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
            **kw,
        )

        self.generation_tokens_total = Counter(
            name="tokenspeed:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
            **kw,
        )

        self.num_requests_total = Counter(
            name="tokenspeed:num_requests_total",
            documentation="Number of requests processed.",
            labelnames=labelnames,
            **kw,
        )

        self.request_success_total = Counter(
            name="tokenspeed:request_success_total",
            documentation="Requests that finished without an abort-style finish.",
            labelnames=labelnames,
            **kw,
        )

        self.prefix_cache_hits_total = Counter(
            name="tokenspeed:prefix_cache_hits_total",
            documentation=(
                "Prompt tokens served from prefix cache. Hit ratio = "
                "prefix_cache_hits_total / prompt_tokens_total."
            ),
            labelnames=labelnames,
            **kw,
        )

        self.histogram_time_to_first_token = Histogram(
            name="tokenspeed:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.1,
                0.3,
                0.5,
                0.7,
                0.9,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                120,
                160,
            ],
            **kw,
        )

        self.histogram_time_per_output_token = Histogram(
            name="tokenspeed:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.002,
                0.005,
                0.010,
                0.020,
                0.030,
                0.040,
                0.050,
                0.060,
                0.070,
                0.080,
                0.090,
                0.100,
                0.150,
                0.200,
                0.300,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
            ],
            **kw,
        )

        self.histogram_inter_token_latency_seconds = Histogram(
            name="tokenspeed:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labelnames,
            buckets=[
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.050,
                0.075,
                0.100,
                0.150,
                0.200,
                0.300,
                0.400,
                0.500,
                0.750,
                1.000,
                2.000,
            ],
            **kw,
        )

        self.histogram_e2e_request_latency = Histogram(
            name="tokenspeed:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labelnames,
            buckets=[
                0.1,
                0.2,
                0.4,
                0.8,
                1,
                2,
                5,
                10,
                20,
                40,
                60,
                80,
                100,
                150,
                200,
                250,
                300,
                350,
                500,
                1000,
            ],
            **kw,
        )

    def record_request_finish(self, stats: RequestFinishStats) -> None:
        if not self.enabled:
            return
        self.prompt_tokens_total.labels(**self.labels).inc(stats.prompt_tokens)
        self.generation_tokens_total.labels(**self.labels).inc(stats.generation_tokens)
        self.num_requests_total.labels(**self.labels).inc(1)
        self.prefix_cache_hits_total.labels(**self.labels).inc(
            stats.cached_prompt_tokens
        )
        if stats.finished_ok:
            self.request_success_total.labels(**self.labels).inc(1)
        self.histogram_e2e_request_latency.labels(**self.labels).observe(
            stats.e2e_latency
        )
        if stats.generation_tokens >= 1:
            self.histogram_time_per_output_token.labels(**self.labels).observe(
                stats.e2e_latency / stats.generation_tokens
            )

    def observe_time_to_first_token(self, value: float) -> None:
        if not self.enabled:
            return
        self.histogram_time_to_first_token.labels(**self.labels).observe(value)

    def observe_inter_token_latency(self, interval: float, num_new_tokens: int) -> None:
        if not self.enabled:
            return
        adjusted_interval = interval / num_new_tokens
        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = self.histogram_inter_token_latency_seconds.labels(**self.labels)
        his._sum.inc(interval)

        for i, bound in enumerate(his._upper_bounds):
            if adjusted_interval <= bound:
                his._buckets[i].inc(num_new_tokens)
                break


class KVTransferMetrics:

    def __init__(self, labels: dict[str, str], metrics_reporters: list[str]) -> None:
        pass

    def record_kv_transfer_timeout(self) -> None:
        return

    def record_kv_transfer_failure(self) -> None:
        return

    def observe_kv_transfer_latency(self, transfer_time_seconds: float) -> None:
        return
