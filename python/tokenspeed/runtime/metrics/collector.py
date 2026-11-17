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

import math
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
            documentation="Requests in the scheduler's running batch: prefill, prefill-done, and decode rows.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_requests_inflight = factory.gauge(
            name="tokenspeed:num_requests_inflight",
            documentation="Requests admitted and not finished, including PD transfer and waiting states.",
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
                "Triton and CuTe DSL kernel compilations after startup; each "
                "one stalls the forward thread."
            ),
            labelnames=labelnames,
        )
        self.jit_serving_compile_seconds = factory.counter(
            name="tokenspeed:jit_serving_compile_seconds",
            documentation=(
                "Wall time spent in Triton and CuTe DSL kernel compilations "
                "after startup."
            ),
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
                "could not image its victim (retraction snapshot pool, image "
                "rows or Host L2 exhausted), or because a lone request made no "
                "progress for 4096 consecutive scheduler plans; clients may "
                "resubmit them. Forced retractions from "
                "--debug-force-retraction-interval aborted the same way count "
                "too."
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
        self.kv_token_usage_ratio = factory.gauge(
            name="tokenspeed:kv_token_usage_perc",
            documentation=(
                "Fraction of usable KV pages resident (active plus reclaimable "
                "prefix-cache entries), 0-1."
            ),
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_used_tokens = factory.gauge(
            name="tokenspeed:num_used_tokens",
            documentation="KV tokens held by resident pages (active plus reclaimable).",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_free_tokens = factory.gauge(
            name="tokenspeed:num_free_tokens",
            documentation="KV tokens not held by active requests (empty plus evictable).",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_prefill_prealloc_queue_reqs = factory.gauge(
            name="tokenspeed:num_prefill_prealloc_queue_reqs",
            documentation="Prefill requests waiting for their decode peer to connect.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_prefill_inflight_queue_reqs = factory.gauge(
            name="tokenspeed:num_prefill_inflight_queue_reqs",
            documentation="Prefill requests whose KV a transfer still pins.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_decode_prealloc_queue_reqs = factory.gauge(
            name="tokenspeed:num_decode_prealloc_queue_reqs",
            documentation="Decode requests waiting for their prefill peer to connect.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_decode_kv_wait_reqs = factory.gauge(
            name="tokenspeed:num_decode_kv_wait_reqs",
            documentation="Decode requests waiting for the prefill peer's KV: admitted with destination pages held, with no schedulable work.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_decode_transfer_queue_reqs = factory.gauge(
            name="tokenspeed:num_decode_transfer_queue_reqs",
            documentation="Decode requests whose KV a transfer still pins.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.num_retracted_queue_reqs = factory.gauge(
            name="tokenspeed:num_retracted_queue_reqs",
            documentation="Requests suspended with a retraction image.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.generation_tokens = factory.counter(
            name="tokenspeed:generation_tokens",
            documentation="Generated tokens committed by decode forwards.",
            labelnames=labelnames,
        )
        self.prompt_tokens = factory.counter(
            name="tokenspeed:prompt_tokens",
            documentation="Prompt tokens of requests that ended, including aborted requests.",
            labelnames=labelnames,
        )
        self.cache_hit_tokens = factory.counter(
            name="tokenspeed:cache_hit_tokens",
            documentation=(
                "Prompt tokens of finished requests served from the prefix cache."
            ),
            labelnames=labelnames,
        )
        self.time_to_first_token_seconds = factory.histogram(
            name="tokenspeed:time_to_first_token_seconds",
            documentation=(
                "Wall time from request creation to its first output token, on this role."
            ),
            labelnames=labelnames,
            buckets=[
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
                120.0,
                300.0,
            ],
        )
        self.request_time_seconds = factory.histogram(
            name="tokenspeed:request_time_seconds",
            documentation=(
                "Wall time from request creation to its end on this role (prefill: "
                "KV transfer done; decode: generation done)."
            ),
            labelnames=labelnames,
            buckets=[
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
                120.0,
                300.0,
            ],
        )
        self.decode_time_seconds = factory.histogram(
            name="tokenspeed:decode_time_seconds",
            documentation="Wall time from the first token to the request's end.",
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
        self.time_per_output_token_seconds = factory.histogram(
            name="tokenspeed:time_per_output_token_seconds",
            documentation="Decode time divided by generated tokens after the first one.",
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
            ],
        )
        self.completion_tokens = factory.counter(
            name="tokenspeed:completion_tokens",
            documentation=(
                "Output tokens of requests that ended, including a prefill role's "
                "bootstrap token."
            ),
            labelnames=labelnames,
        )
        self.grammar_queue_reqs = factory.gauge(
            name="tokenspeed:grammar_queue_reqs",
            documentation="Admissions waiting for their grammar to compile.",
            labelnames=labelnames,
            multiprocess_mode="livemax",
        )
        self.prefix_prompt_tokens = factory.counter(
            name="tokenspeed:prefix_prompt_tokens",
            documentation="Prompt tokens of admitted requests.",
            labelnames=labelnames,
        )
        self.prefix_device_hit_tokens = factory.counter(
            name="tokenspeed:prefix_device_hit_tokens",
            documentation=(
                "Admitted prompt tokens served from the Device prefix cache (L1)."
            ),
            labelnames=labelnames,
        )
        self.prefix_host_hit_tokens = factory.counter(
            name="tokenspeed:prefix_host_hit_tokens",
            documentation=(
                "Admitted prompt tokens loaded from the Host cache (L2, including "
                "L3-supplied)."
            ),
            labelnames=labelnames,
        )
        self.prefix_l3_hit_tokens = factory.counter(
            name="tokenspeed:prefix_l3_hit_tokens",
            documentation="Token spans of successful storage keys (L3), counted per cache group.",
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
        self,
        *,
        running: int,
        inflight: int,
        waiting: int,
        kv_cache_usage_ratio: float,
    ) -> None:
        if not self.enabled:
            return
        self.num_requests_running.labels(**self.labels).set(running)
        self.num_requests_inflight.labels(**self.labels).set(inflight)
        self.num_requests_waiting.labels(**self.labels).set(waiting)
        self.kv_cache_usage_ratio.labels(**self.labels).set(kv_cache_usage_ratio)

    def observe_iteration_tokens(self, num_tokens: float) -> None:
        if self.enabled and num_tokens >= 0:
            self.iteration_tokens_total.labels(**self.labels).observe(num_tokens)

    def record_scheduler_iteration(
        self,
        *,
        running: int,
        inflight: int,
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
            inflight=inflight,
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

    def record_cache_occupancy(
        self,
        *,
        used_pages: int,
        active_pages: int,
        total_pages: int,
        token_capacity: int,
    ) -> None:
        if not self.enabled or total_pages <= 0 or token_capacity < 0:
            return
        used = min(max(used_pages, 0), total_pages)
        active = min(max(active_pages, 0), total_pages)
        self.kv_token_usage_ratio.labels(**self.labels).set(used / total_pages)
        self.num_used_tokens.labels(**self.labels).set(
            used * token_capacity // total_pages
        )
        self.num_free_tokens.labels(**self.labels).set(
            (total_pages - active) * token_capacity // total_pages
        )

    def record_prefill_queues(
        self, *, prealloc: int, inflight: int, retracted: int
    ) -> None:
        if not self.enabled:
            return
        self.num_prefill_prealloc_queue_reqs.labels(**self.labels).set(max(prealloc, 0))
        self.num_prefill_inflight_queue_reqs.labels(**self.labels).set(max(inflight, 0))
        self.num_retracted_queue_reqs.labels(**self.labels).set(max(retracted, 0))

    def record_decode_queues(
        self, *, prealloc: int, kv_wait: int, transfer: int, retracted: int
    ) -> None:
        if not self.enabled:
            return
        self.num_decode_prealloc_queue_reqs.labels(**self.labels).set(max(prealloc, 0))
        self.num_decode_kv_wait_reqs.labels(**self.labels).set(max(kv_wait, 0))
        self.num_decode_transfer_queue_reqs.labels(**self.labels).set(max(transfer, 0))
        self.num_retracted_queue_reqs.labels(**self.labels).set(max(retracted, 0))

    def record_generation_tokens(self, count: int) -> None:
        if not self.enabled or count <= 0:
            return
        self.generation_tokens.labels(**self.labels).inc(count)

    def record_request_tokens(self, prompt_tokens: int, cache_hit_tokens: int) -> None:
        if not self.enabled:
            return
        if prompt_tokens > 0:
            self.prompt_tokens.labels(**self.labels).inc(prompt_tokens)
        if cache_hit_tokens > 0:
            self.cache_hit_tokens.labels(**self.labels).inc(cache_hit_tokens)

    def observe_time_to_first_token(self, seconds: float) -> None:
        if self.enabled and math.isfinite(seconds) and seconds >= 0.0:
            self.time_to_first_token_seconds.labels(**self.labels).observe(seconds)

    def observe_request_time(self, seconds: float) -> None:
        if self.enabled and math.isfinite(seconds) and seconds >= 0.0:
            self.request_time_seconds.labels(**self.labels).observe(seconds)

    def observe_decode_time(self, seconds: float) -> None:
        if self.enabled and math.isfinite(seconds) and seconds >= 0.0:
            self.decode_time_seconds.labels(**self.labels).observe(seconds)

    def observe_time_per_output_token(self, seconds: float) -> None:
        if self.enabled and math.isfinite(seconds) and seconds >= 0.0:
            self.time_per_output_token_seconds.labels(**self.labels).observe(seconds)

    def record_completion_tokens(self, count: int) -> None:
        if self.enabled and count > 0:
            self.completion_tokens.labels(**self.labels).inc(count)

    def record_grammar_queue(self, waiting: int) -> None:
        if self.enabled:
            self.grammar_queue_reqs.labels(**self.labels).set(max(waiting, 0))

    def record_prefix_hits(
        self,
        prompt_tokens: int,
        device_hit_tokens: int,
        host_hit_tokens: int,
        l3_hit_tokens: int,
    ) -> None:
        """Counters gained since the last report (the Rust host diffs them)."""
        if not self.enabled:
            return
        if prompt_tokens > 0:
            self.prefix_prompt_tokens.labels(**self.labels).inc(prompt_tokens)
        if device_hit_tokens > 0:
            self.prefix_device_hit_tokens.labels(**self.labels).inc(device_hit_tokens)
        if host_hit_tokens > 0:
            self.prefix_host_hit_tokens.labels(**self.labels).inc(host_hit_tokens)
        if l3_hit_tokens > 0:
            self.prefix_l3_hit_tokens.labels(**self.labels).inc(l3_hit_tokens)


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
        self.kv_transfer_size_bytes = factory.counter(
            name="tokenspeed:kv_transfer_size_bytes",
            documentation="Bytes one KV cache transfer send moved to a peer.",
            labelnames=labelnames,
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

    def record_kv_transfer_size(self, bytes_moved: int) -> None:
        if self.enabled and bytes_moved > 0:
            self.kv_transfer_size_bytes.labels(**self.labels).inc(bytes_moved)

    def observe_kv_transfer_latency(self, transfer_time_seconds: float) -> None:
        if not self.enabled:
            return
        self.kv_transfer_latency_seconds.labels(**self.labels).observe(
            transfer_time_seconds
        )
