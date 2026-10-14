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
from typing import TYPE_CHECKING

from tokenspeed_kernel.compile_monitor import CompileStats

from tokenspeed.runtime.engine.request_types import ABORT_CODE, BaseFinishReason
from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    # Imported lazily at runtime; see _init_prometheus for the ordering
    # requirement with PROMETHEUS_MULTIPROC_DIR.
    from prometheus_client import Histogram

TOKENSPEED_TEST_REQUEST_TIME_STATS = envs.TOKENSPEED_TEST_REQUEST_TIME_STATS.get()


def _raise_for_negative_durations(**durations: float) -> None:
    negative_durations = [
        f"{name}={duration} < 0" for name, duration in durations.items() if duration < 0
    ]
    if negative_durations:
        raise ValueError(" or ".join(negative_durations))


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
                _raise_for_negative_durations(
                    queue_duration=queue_duration,
                    forward_duration=forward_duration,
                )

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time}"
        if _type == self.RequestType.PREFILL:
            bootstrap_duration = (
                self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time
            )

            queue_duration = self.forward_entry_time - self.wait_queue_entry_time

            forward_duration = self.completion_time - self.forward_entry_time

            if TOKENSPEED_TEST_REQUEST_TIME_STATS:
                _raise_for_negative_durations(
                    bootstrap_duration=bootstrap_duration,
                    queue_duration=queue_duration,
                    forward_duration=forward_duration,
                )
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
                _raise_for_negative_durations(
                    prealloc_duration=prealloc_duration,
                    transfer_duration=transfer_duration,
                    queue_duration=queue_duration,
                    forward_duration=forward_duration,
                )

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


# Stable snake_case strings for the bounded ABORT_CODE enum, used in the
# finish_reason label (never the unbounded abort message).
_ABORT_REASON_LABELS = {
    ABORT_CODE.TransferFailed: "transfer_failed",
    ABORT_CODE.UnknownError: "unknown_error",
    ABORT_CODE.NumericalError: "numerical_error",
    ABORT_CODE.CapacityAbort: "capacity_abort",
}


def finish_reason_label(finished_reason: dict | BaseFinishReason | None) -> str:
    """Map a finish reason to its bounded ``finish_reason`` label value.

    Accepts the wire form (``BaseFinishReason.to_json()`` dicts) and the
    object form alike. The label carries the bounded taxonomy only: the
    type (``stop`` / ``length`` / ``abort``) and, for aborts, the bounded
    ``ABORT_CODE`` subclass as ``abort:<code>``. An abort with an
    unrecognized or missing ``err_type`` stays plain ``abort``; an
    unrecognized form maps to ``unknown``. Unbounded payloads (abort
    message, matched stop string/token) never become labels.
    """
    if finished_reason is None:
        return "unknown"
    reason = (
        finished_reason
        if isinstance(finished_reason, dict)
        else finished_reason.to_json()
    )
    reason_type = reason.get("type")
    if reason_type in ("stop", "length"):
        return reason_type
    if reason_type != "abort":
        return "unknown"
    err_type = reason.get("err_type")
    try:
        code = err_type if isinstance(err_type, ABORT_CODE) else ABORT_CODE(err_type)
    except (ValueError, TypeError):
        return "abort"
    # A future ABORT_CODE member without a label entry stays plain "abort",
    # like an unrecognized err_type; the mapping test guards against drift.
    label = _ABORT_REASON_LABELS.get(code)
    return f"abort:{label}" if label is not None else "abort"


def stream_label(stream: bool) -> str:
    """The ``stream`` label value for a request's streaming flag."""
    return "true" if stream else "false"


def build_1_2_5_buckets(max_value: int) -> list[float]:
    """A 1-2-5 exponential bucket grid: 1, 2, 5 scaled by increasing powers
    of ten, capped at ``max_value`` (e.g. 100 -> 1, 2, 5, 10, 20, 50, 100)."""
    exponent = 0
    buckets: list[float] = []
    while True:
        for m in (1, 2, 5):
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(float(value))
            else:
                return buckets
        exponent += 1


@dataclass
class RequestFinishStats:
    prompt_tokens: int
    generation_tokens: int
    e2e_latency: float
    cached_prompt_tokens: int = 0
    finished_ok: bool = True
    # Bounded finish-reason label value (``finish_reason_label``).
    finish_reason: str = "unknown"
    # The request's streaming flag, exported as the "stream" label.
    stream: bool = False


_TIME_PER_OUTPUT_TOKEN_BUCKETS = [
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
]

_INTER_TOKEN_LATENCY_BUCKETS = [
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
]


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
        # Serving-time compile totals already exported; counters only move
        # forward, so each round adds the difference.
        self._exported_serving_compiles = 0
        self._exported_serving_compile_seconds = 0.0

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
        # Wire name follows vLLM's `vllm:kv_cache_usage_perc` for s/vllm:/tokenspeed:/g
        # parity even though the value is a 0-1 ratio, not a percentage.
        self.kv_cache_usage_ratio = Gauge(
            name="tokenspeed:kv_cache_usage_perc",
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
            name="tokenspeed:spec_decode_num_accepted_tokens",
            documentation=(
                "Accepted speculative draft tokens (excludes the bonus token sampled "
                "after verify)."
            ),
            labelnames=labelnames,
            **kw,
        )
        self.spec_decode_num_draft_tokens = Counter(
            name="tokenspeed:spec_decode_num_draft_tokens",
            documentation="Draft tokens proposed across verify steps.",
            labelnames=labelnames,
            **kw,
        )
        self.spec_decode_num_drafts = Counter(
            name="tokenspeed:spec_decode_num_drafts",
            documentation="Number of speculative verify rounds (per request-slot).",
            labelnames=labelnames,
            **kw,
        )
        self.jit_serving_compiles = Counter(
            name="tokenspeed:jit_serving_compiles",
            documentation=(
                "Triton kernel compilations after startup; each one stalls the "
                "forward thread."
            ),
            labelnames=labelnames,
            **kw,
        )
        self.jit_serving_compile_seconds = Counter(
            name="tokenspeed:jit_serving_compile_seconds",
            documentation="Wall time spent in Triton kernel compilations after startup.",
            labelnames=labelnames,
            **kw,
        )
        self.num_nan_aborted_requests = Counter(
            name="tokenspeed:num_nan_aborted_requests",
            documentation=(
                "Requests terminated by the NaN guard (NaN in logits or an "
                "out-of-vocab sampled token id)."
            ),
            labelnames=labelnames,
            **kw,
        )
        self.num_capacity_aborted_requests = Counter(
            name="tokenspeed:num_capacity_aborted_requests",
            documentation=(
                "Requests the scheduler aborted because a capacity retraction "
                "could not image its victim (retraction snapshot pool or image "
                "rows exhausted); clients may resubmit them."
            ),
            labelnames=labelnames,
            **kw,
        )
        # Exported as tokenspeed:num_preemptions_total.
        self.num_preemptions = Counter(
            name="tokenspeed:num_preemptions",
            documentation=(
                "Requests preempted by a capacity retraction: the scheduler "
                "suspended a resident request (its pages imaged to the Host "
                "snapshot pool; it later resumes by restore, nothing is "
                "recomputed) to free Device pages for a blocked admission. "
                "One increment per retraction victim per round, counted from "
                "the plan's snapshot-store ops; forced retractions from "
                "--debug-force-retraction-interval count too. A retraction "
                "whose victim cannot be imaged aborts it instead — see "
                "tokenspeed:num_capacity_aborted_requests."
            ),
            labelnames=labelnames,
            **kw,
        )
        # Request-latency bucket grid: 0.3 s to 7680 s.
        self.request_queue_time_seconds = Histogram(
            name="tokenspeed:request_queue_time_seconds",
            documentation=(
                "Wall time from submission to the scheduler's waiting queue "
                "to first dispatch into a forward batch, observed once per "
                "request. Grammar-compile, pause-hold, and EPD embedding "
                "waits precede submission and are excluded, as are "
                "tokenizer-manager and IPC wait."
            ),
            labelnames=labelnames,
            buckets=[
                0.3,
                0.5,
                0.8,
                1.0,
                1.5,
                2.0,
                2.5,
                5.0,
                10.0,
                15.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                120.0,
                240.0,
                480.0,
                960.0,
                1920.0,
                7680.0,
            ],
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

    def record_jit_compiles(self, stats: CompileStats | None) -> None:
        """Export serving-time compilations; ``None`` when not monitored."""
        if not self.enabled or stats is None:
            return
        compiles = stats.serving_compiles - self._exported_serving_compiles
        if compiles <= 0:
            return
        self.jit_serving_compiles.labels(**self.labels).inc(compiles)
        self.jit_serving_compile_seconds.labels(**self.labels).inc(
            stats.serving_seconds - self._exported_serving_compile_seconds
        )
        self._exported_serving_compiles = stats.serving_compiles
        self._exported_serving_compile_seconds = stats.serving_seconds

    def record_nan_abort(self) -> None:
        if not self.enabled:
            return
        self.num_nan_aborted_requests.labels(**self.labels).inc()

    def record_capacity_abort(self) -> None:
        if not self.enabled:
            return
        self.num_capacity_aborted_requests.labels(**self.labels).inc()

    def record_preemptions(self, count: int) -> None:
        """Increment by the round's retraction count (plan snapshot ops)."""
        if not self.enabled or count <= 0:
            return
        self.num_preemptions.labels(**self.labels).inc(count)

    def observe_request_queue_time(self, queue_time: float) -> None:
        if not self.enabled:
            return
        self.request_queue_time_seconds.labels(**self.labels).observe(queue_time)


class RequestMetrics:
    def __init__(
        self,
        labels: dict[str, str],
        *,
        enabled: bool,
        max_model_len: int,
        registry=None,
    ) -> None:
        self.enabled = enabled
        self.labels = labels

        if self.enabled:
            self._init_prometheus(
                labels, max_model_len=max_model_len, registry=registry
            )

    def _init_prometheus(
        self, labels: dict[str, str], *, max_model_len: int, registry=None
    ) -> None:
        # We need to import prometheus_client after setting the env variable PROMETHEUS_MULTIPROC_DIR
        from prometheus_client import Counter, Histogram

        labelnames = list(labels.keys())
        # Per-request enrichment: counters additionally carry finish_reason
        # and stream; histograms carry stream only.
        finish_labelnames = [*labelnames, "finish_reason", "stream"]
        stream_labelnames = [*labelnames, "stream"]
        kw = {"registry": registry} if registry is not None else {}

        self.prompt_tokens_total = Counter(
            name="tokenspeed:prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
            **kw,
        )

        self.generation_tokens_total = Counter(
            name="tokenspeed:generation_tokens",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
            **kw,
        )

        # tokenspeed-only Counter that tracks every finished request
        # regardless of finish reason.
        self.num_requests_total = Counter(
            name="tokenspeed:num_requests",
            documentation="Number of requests processed.",
            labelnames=finish_labelnames,
            **kw,
        )

        self.request_success_total = Counter(
            name="tokenspeed:request_success",
            documentation="Requests that finished without an abort-style finish.",
            labelnames=finish_labelnames,
            **kw,
        )

        self.prefix_cache_hits_total = Counter(
            name="tokenspeed:prefix_cache_hits",
            documentation=(
                "Prompt tokens served from prefix cache. Hit ratio = "
                "prefix_cache_hits_total / prompt_tokens_total. This is the "
                "cached leg of the cached/uncached split: cached tokens = "
                "tokenspeed:prefix_cache_hits_total, uncached tokens = "
                "tokenspeed:prompt_tokens_total - "
                "tokenspeed:prefix_cache_hits_total."
            ),
            labelnames=labelnames,
            **kw,
        )

        self.histogram_time_to_first_token = Histogram(
            name="tokenspeed:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=stream_labelnames,
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
            name="tokenspeed:request_time_per_output_token_seconds",
            documentation=(
                "Histogram of end-to-end request latency divided by output "
                "tokens in seconds (includes queueing and TTFT)."
            ),
            labelnames=labelnames,
            buckets=_TIME_PER_OUTPUT_TOKEN_BUCKETS,
            **kw,
        )

        self.histogram_decode_time_per_output_token = Histogram(
            name="tokenspeed:decode_time_per_output_token_seconds",
            documentation=(
                "Histogram of decode time per output token after the first "
                "output in seconds (excludes queueing and TTFT)."
            ),
            labelnames=labelnames,
            buckets=_TIME_PER_OUTPUT_TOKEN_BUCKETS,
            **kw,
        )

        self.histogram_inter_token_latency_seconds = Histogram(
            name="tokenspeed:inter_token_latency_seconds",
            documentation="Histogram of frontend wall-clock inter-token latency in seconds.",
            labelnames=labelnames,
            buckets=_INTER_TOKEN_LATENCY_BUCKETS,
            **kw,
        )

        self.histogram_scheduler_inter_token_latency_seconds = Histogram(
            name="tokenspeed:scheduler_inter_token_latency_seconds",
            documentation="Histogram of scheduler-side inter-token latency in seconds.",
            labelnames=labelnames,
            buckets=_INTER_TOKEN_LATENCY_BUCKETS,
            **kw,
        )

        self.histogram_e2e_request_latency = Histogram(
            name="tokenspeed:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=stream_labelnames,
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

        # Per-request token distributions at request finish, on a 1-2-5
        # bucket grid capped at the model's max length.
        request_token_buckets = build_1_2_5_buckets(max_model_len)
        self.histogram_request_prompt_tokens = Histogram(
            name="tokenspeed:request_prompt_tokens",
            documentation=(
                "Histogram of per-request prompt token counts at request "
                "finish (1-2-5 bucket grid capped at the model's max length)."
            ),
            labelnames=stream_labelnames,
            buckets=request_token_buckets,
            **kw,
        )
        self.histogram_request_generation_tokens = Histogram(
            name="tokenspeed:request_generation_tokens",
            documentation=(
                "Histogram of per-request generation token counts at request "
                "finish (1-2-5 bucket grid capped at the model's max length)."
            ),
            labelnames=stream_labelnames,
            buckets=request_token_buckets,
            **kw,
        )

    def record_request_finish(self, stats: RequestFinishStats) -> None:
        if not self.enabled:
            return
        self.prompt_tokens_total.labels(**self.labels).inc(stats.prompt_tokens)
        self.generation_tokens_total.labels(**self.labels).inc(stats.generation_tokens)
        finish_labels = {
            **self.labels,
            "finish_reason": stats.finish_reason,
            "stream": stream_label(stats.stream),
        }
        self.num_requests_total.labels(**finish_labels).inc(1)
        self.prefix_cache_hits_total.labels(**self.labels).inc(
            stats.cached_prompt_tokens
        )
        if stats.finished_ok:
            self.request_success_total.labels(**finish_labels).inc(1)
        stream_labels = {**self.labels, "stream": stream_label(stats.stream)}
        self.histogram_e2e_request_latency.labels(**stream_labels).observe(
            stats.e2e_latency
        )
        self.histogram_request_prompt_tokens.labels(**stream_labels).observe(
            stats.prompt_tokens
        )
        self.histogram_request_generation_tokens.labels(**stream_labels).observe(
            stats.generation_tokens
        )
        if stats.generation_tokens >= 1:
            self.histogram_time_per_output_token.labels(**self.labels).observe(
                stats.e2e_latency / stats.generation_tokens
            )

    def observe_time_to_first_token(self, value: float, *, stream: bool) -> None:
        if not self.enabled:
            return
        self.histogram_time_to_first_token.labels(
            **self.labels, stream=stream_label(stream)
        ).observe(value)

    def observe_decode_time_per_output_token(self, value: float) -> None:
        if not self.enabled:
            return
        self.histogram_decode_time_per_output_token.labels(**self.labels).observe(value)

    def observe_inter_token_latency(self, interval: float, num_new_tokens: int) -> None:
        if not self.enabled or num_new_tokens <= 0:
            return
        self._observe_batched(
            self.histogram_inter_token_latency_seconds, interval, num_new_tokens
        )

    def observe_scheduler_inter_token_latency(
        self, interval: float, num_new_tokens: int
    ) -> None:
        if not self.enabled or num_new_tokens <= 0:
            return
        self._observe_batched(
            self.histogram_scheduler_inter_token_latency_seconds,
            interval,
            num_new_tokens,
        )

    def _observe_batched(
        self, histogram: "Histogram", interval: float, num_new_tokens: int
    ) -> None:
        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = histogram.labels(**self.labels)
        his._sum.inc(interval)
        adjusted_interval = interval / num_new_tokens
        for i, bound in enumerate(his._upper_bounds):
            if adjusted_interval <= bound:
                his._buckets[i].inc(num_new_tokens)
                break


class KVTransferMetrics:
    """PD KV transfer metrics, recorded by the transfer managers.

    Lives on every PD engine (prefill and decode roles each record their own
    view); series from the node's transfer processes aggregate in Prometheus
    multiprocess mode.
    """

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
        from prometheus_client import Counter, Histogram

        labelnames = list(labels.keys())
        kw = {"registry": registry} if registry is not None else {}
        self.kv_transfer_latency_seconds = Histogram(
            name="tokenspeed:kv_transfer_latency_seconds",
            documentation="Wall time of one KV cache transfer send to a peer.",
            labelnames=labelnames,
            buckets=[
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                30.0,
                60.0,
            ],
            **kw,
        )
        # `stage` is the bounded TransferPoll state the request was stuck in:
        # "bootstrapping" (prefill never received the cache manifest) or
        # "waiting_for_input" (decode never received the transfer-done signal).
        self.kv_transfer_timeouts = Counter(
            name="tokenspeed:kv_transfer_timeouts",
            documentation="KV transfers that timed out, by transfer stage.",
            labelnames=[*labelnames, "stage"],
            **kw,
        )
        self.kv_transfer_failures = Counter(
            name="tokenspeed:kv_transfer_failures",
            documentation=(
                "KV transfers aborted on failure: send failures on the prefill "
                "node, and rooms failed on prefill-node loss on the decode node. "
                "Timeouts are counted separately."
            ),
            labelnames=labelnames,
            **kw,
        )

    def record_kv_transfer_timeout(self, stage: str) -> None:
        if not self.enabled:
            return
        self.kv_transfer_timeouts.labels(**self.labels, stage=stage).inc()

    def record_kv_transfer_failure(self) -> None:
        if not self.enabled:
            return
        self.kv_transfer_failures.labels(**self.labels).inc()

    def observe_kv_transfer_latency(self, transfer_time_seconds: float) -> None:
        if not self.enabled:
            return
        self.kv_transfer_latency_seconds.labels(**self.labels).observe(
            transfer_time_seconds
        )
