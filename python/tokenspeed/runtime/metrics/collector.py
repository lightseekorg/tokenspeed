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

from typing import TYPE_CHECKING

from tokenspeed.runtime.metrics.events import MetricEventSink
from tokenspeed.runtime.metrics.reporter import MetricFactory

if TYPE_CHECKING:
    from tokenspeed_kernel.compile_monitor import CompileStats


class EngineMetrics:
    """The scheduler loop's metrics: their names, labels and buckets.

    Every engine rank records them in Rust (``ts-metrics``), which keeps
    these definitions; its fixtures are generated from this class
    (``tokenspeed-engine/crates/ts-metrics/tests/fixtures/generate.py``), so
    a change here is a change of the Rust metrics too.
    """

    def __init__(
        self,
        labels: dict[str, str],
        *,
        enabled: bool,
        reporters: list[str],
        event_sink: MetricEventSink | None = None,
        registry=None,
    ) -> None:
        self.enabled = enabled
        self.labels = labels
        # Serving-time compile totals already exported; counters only move
        # forward, so each round adds the difference.
        self._exported_serving_compiles = 0
        self._exported_serving_compile_seconds = 0.0

        factory = MetricFactory(
            enabled=enabled,
            reporters=reporters,
            event_sink=event_sink,
            registry=registry,
        )
        labelnames = list(labels.keys())
        self.num_requests_running = factory.gauge(
            name="tokenspeed:num_requests_running",
            documentation="Requests with scheduler-side generation state (decode path).",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_requests_waiting = factory.gauge(
            name="tokenspeed:num_requests_waiting",
            documentation="Requests waiting in the C++ scheduler queue.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        # Wire name follows vLLM's `vllm:kv_cache_usage_perc` for s/vllm:/tokenspeed:/g
        # parity even though the value is a 0-1 ratio, not a percentage.
        self.kv_cache_usage_ratio = factory.gauge(
            name="tokenspeed:kv_cache_usage_perc",
            documentation="Fraction of device KV pages in use (0-1).",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.iteration_tokens_total = factory.histogram(
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
        )
        self.spec_decode_num_accepted_tokens = factory.counter(
            name="tokenspeed:spec_decode_num_accepted_tokens",
            documentation=(
                "Accepted speculative draft tokens (excludes the bonus token sampled "
                "after verify)."
            ),
            labelnames=labelnames,
        )
        self.spec_decode_num_draft_tokens = factory.counter(
            name="tokenspeed:spec_decode_num_draft_tokens",
            documentation="Draft tokens proposed across verify steps.",
            labelnames=labelnames,
        )
        self.spec_decode_num_drafts = factory.counter(
            name="tokenspeed:spec_decode_num_drafts",
            documentation="Number of speculative verify rounds (per request-slot).",
            labelnames=labelnames,
        )
        self.jit_serving_compiles = factory.counter(
            name="tokenspeed:jit_serving_compiles",
            documentation=(
                "Triton kernel compilations after startup; each one stalls the "
                "forward thread."
            ),
            labelnames=labelnames,
        )
        self.jit_serving_compile_seconds = factory.counter(
            name="tokenspeed:jit_serving_compile_seconds",
            documentation="Wall time spent in Triton kernel compilations after startup.",
            labelnames=labelnames,
        )
        self.num_nan_aborted_requests = factory.counter(
            name="tokenspeed:num_nan_aborted_requests",
            documentation=(
                "Requests terminated by the NaN guard (NaN in logits or an "
                "out-of-vocab sampled token id)."
            ),
            labelnames=labelnames,
        )
        self.num_capacity_aborted_requests = factory.counter(
            name="tokenspeed:num_capacity_aborted_requests",
            documentation=(
                "Requests the scheduler aborted because a capacity retraction "
                "could not image its victim (retraction snapshot pool or image "
                "rows exhausted); clients may resubmit them."
            ),
            labelnames=labelnames,
        )
        # Exported as tokenspeed:num_preemptions_total.
        self.num_preemptions = factory.counter(
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
        )
        # Request-latency bucket grid: 0.3 s to 7680 s.
        self.request_queue_time_seconds = factory.histogram(
            name="tokenspeed:request_queue_time_seconds",
            documentation=(
                "Wall time from submission to the scheduler's waiting queue "
                "to first dispatch into a forward batch, observed once per "
                "request. Grammar-compile, pause-hold, and EPD embedding "
                "waits precede submission and are excluded, as are "
                "frontend and IPC wait."
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

    def record_jit_compiles(self, stats: "CompileStats | None") -> None:
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


class KVTransferMetrics:
    """PD KV transfer metrics: their names, labels and buckets.

    Every PD engine rank records them in Rust (``ts-metrics``; prefill and
    decode roles each record their own view), which keeps these
    definitions, as for ``EngineMetrics``.
    """

    def __init__(
        self,
        labels: dict[str, str],
        *,
        enabled: bool,
        reporters: list[str],
        event_sink: MetricEventSink | None = None,
        registry=None,
    ) -> None:
        self.enabled = enabled
        self.labels = labels

        factory = MetricFactory(
            enabled=enabled,
            reporters=reporters,
            event_sink=event_sink,
            registry=registry,
        )
        labelnames = list(labels.keys())
        self.kv_transfer_latency_seconds = factory.histogram(
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
        )
        # `stage` is the bounded TransferPoll state the request was stuck in:
        # "bootstrapping" (prefill never received the cache manifest) or
        # "waiting_for_input" (decode never received the transfer-done signal).
        self.kv_transfer_timeouts = factory.counter(
            name="tokenspeed:kv_transfer_timeouts",
            documentation="KV transfers that timed out, by transfer stage.",
            labelnames=[*labelnames, "stage"],
        )
        self.kv_transfer_failures = factory.counter(
            name="tokenspeed:kv_transfer_failures",
            documentation=(
                "KV transfers aborted on failure: send failures on the prefill "
                "node, and rooms failed on prefill-node loss on the decode node. "
                "Timeouts are counted separately."
            ),
            labelnames=labelnames,
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
