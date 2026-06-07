"""Arrival processes + lifecycle orchestration.

A workload is an async generator that yields ChatRequest objects. The runner
decides *when* each yielded request fires (constant rate, Poisson arrivals,
or burst pattern) and caps in-flight concurrency.

Includes a simple circuit breaker: if requests fail back-to-back faster than
they succeed, the runner trips open and stops dispatching new work for a
cool-off window. Without this, a dead server turns "constant" arrival into a
tight loop that emits millions of ClientConnectorErrors per minute.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp

from .audits import AuditConfig
from .client import ChatRequest, send_chat
from .events import RUN_FINISHED, RUN_STARTED, JsonlSink


@dataclass
class ArrivalSpec:
    """How requests are paced."""

    # One of:
    #   "constant" - as-fast-as-possible up to max_concurrency
    #   "poisson"  - mean rate req/s
    #   "bursty"   - burst_size requests every burst_period_s
    #   "sawtooth" - in-flight cap ramps linearly between min_concurrency
    #                and max_concurrency with triangle_period_s full
    #                period (half up, half down). Useful for stressing
    #                continuous-batching transitions.
    #   "burst"    - dispatch burst_size requests, wait for all in-flight
    #                to finish (drain to zero), sleep burst_gap_s, repeat.
    #                Forces scheduler through 0 -> N -> 0 -> N cycles.
    kind: str = "constant"
    rate: float = 10.0  # for poisson
    burst_size: int = 20  # for bursty + burst
    burst_period_s: float = 5.0  # for bursty
    burst_gap_s: float = 1.0  # for burst (post-drain sleep)
    # For sawtooth: cap floor and period.
    min_concurrency: int = 1
    triangle_period_s: float = 20.0
    max_concurrency: int = 64
    duration_s: Optional[float] = None
    max_requests: Optional[int] = None


@dataclass
class BreakerSpec:
    """Circuit breaker. Trips open on a burst of consecutive errors.

    Defaults: trip after 32 consecutive errors (any kind), re-close after 5s
    of no new dispatch. Disable with `enabled=False`.
    """

    enabled: bool = True
    trip_after_consecutive_errors: int = 32
    open_duration_s: float = 5.0


class _Breaker:
    def __init__(self, spec: BreakerSpec, sink: JsonlSink):
        self.spec = spec
        self.sink = sink
        self._consecutive_errors = 0
        self._open_until: float = 0.0
        self._tripped_count = 0

    def record(self, ok: bool) -> None:
        if not self.spec.enabled:
            return
        if ok:
            self._consecutive_errors = 0
            return
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.spec.trip_after_consecutive_errors:
            self._open_until = time.time() + self.spec.open_duration_s
            self._tripped_count += 1
            self._consecutive_errors = 0
            self.sink.emit(
                "breaker_open",
                reopen_in_s=self.spec.open_duration_s,
                trip_count=self._tripped_count,
            )

    async def wait_if_open(self) -> None:
        if not self.spec.enabled:
            return
        remaining = self._open_until - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)


def _sawtooth_cap(spec: ArrivalSpec, elapsed: float) -> int:
    """Linear triangle: min -> max -> min over triangle_period_s."""
    period = max(1e-3, spec.triangle_period_s)
    t = (elapsed % period) / period  # 0..1
    # Tent function: 0 -> 1 at t=0.5 -> 0 at t=1
    phase = 2 * t if t < 0.5 else 2 * (1 - t)
    cap = spec.min_concurrency + phase * (spec.max_concurrency - spec.min_concurrency)
    return max(spec.min_concurrency, int(round(cap)))


async def _arrival_gate(spec: ArrivalSpec, deadline: float) -> AsyncIterator[None]:
    """Yields one token per request that should fire, respecting pacing."""
    if spec.kind == "constant":
        # Semaphore is the real throttle, but we still yield to the loop so
        # inflight completions can run and so the breaker can interpose.
        while time.time() < deadline:
            yield None
            await asyncio.sleep(0)
        return
    if spec.kind == "poisson":
        # Exponential inter-arrival with mean 1/rate.
        while time.time() < deadline:
            yield None
            delay = random.expovariate(spec.rate) if spec.rate > 0 else 0.0
            await asyncio.sleep(delay)
        return
    if spec.kind == "bursty":
        while time.time() < deadline:
            for _ in range(spec.burst_size):
                yield None
            await asyncio.sleep(spec.burst_period_s)
        return
    if spec.kind == "sawtooth":
        # Dispatch tokens aggressively; the in-flight gate in `run`
        # reads the time-varying cap and throttles actual dispatch.
        while time.time() < deadline:
            yield None
            await asyncio.sleep(0)
        return
    raise ValueError(f"unknown arrival kind: {spec.kind}")


async def run(
    base_url: str,
    model: str,
    workload: AsyncIterator[ChatRequest],
    arrival: ArrivalSpec,
    sink: JsonlSink,
    timeout_s: float = 600.0,
    breaker: Optional[BreakerSpec] = None,
    audit_cfg: Optional[AuditConfig] = None,
) -> None:
    """Drive a single workload against the server until the arrival spec is done."""
    sink.emit(
        RUN_STARTED,
        base_url=base_url,
        model=model,
        arrival=arrival.__dict__,
    )
    start = time.time()
    deadline = start + (arrival.duration_s or 10**9)
    sent = 0
    inflight: set[asyncio.Task] = set()

    # Time-varying in-flight cap. For "sawtooth", cap ramps between
    # ``min_concurrency`` and ``max_concurrency``; everything else uses
    # a constant cap equal to ``max_concurrency``.
    def current_cap() -> int:
        if arrival.kind == "sawtooth":
            return _sawtooth_cap(arrival, time.time() - start)
        return arrival.max_concurrency

    async def wait_for_slot() -> None:
        # Simple polling gate (10ms resolution). Adequate for stress
        # tests where actual latency is O(100ms-seconds) per request.
        while len(inflight) >= current_cap():
            await asyncio.sleep(0.01)

    async def wait_for_drain() -> None:
        while inflight:
            await asyncio.sleep(0.01)

    brk = _Breaker(breaker or BreakerSpec(), sink)

    connector = aiohttp.TCPConnector(
        limit=arrival.max_concurrency * 2, force_close=False
    )
    async with aiohttp.ClientSession(connector=connector) as session:

        async def _one(req: ChatRequest) -> None:
            outcome = await send_chat(
                session,
                base_url,
                model,
                req,
                sink,
                timeout_s=timeout_s,
                audit_cfg=audit_cfg,
            )
            # Cancellations are an expected test-behaviour, not a server
            # fault: they don't count against the breaker.
            if outcome == "error":
                brk.record(ok=False)
            elif outcome == "completed":
                brk.record(ok=True)

        async def dispatch_one() -> bool:
            """Pull one request from the workload and fire it. Returns
            False if the workload is exhausted or the max_requests cap
            is hit."""
            nonlocal sent
            if arrival.max_requests is not None and sent >= arrival.max_requests:
                return False
            await brk.wait_if_open()
            try:
                req = await workload.__anext__()
            except StopAsyncIteration:
                return False
            t = asyncio.create_task(_one(req))
            inflight.add(t)
            t.add_done_callback(inflight.discard)
            sent += 1
            return True

        if arrival.kind == "burst":
            # Dispatch N, drain to 0, sleep, repeat. Explicit drain is
            # the whole point of this mode — it isolates each burst from
            # the previous so the scheduler transitions through empty.
            while time.time() < deadline:
                exhausted = False
                for _ in range(arrival.burst_size):
                    if not await dispatch_one():
                        # Workload ran out or --max-requests hit; drain
                        # the in-flight tail then exit.
                        exhausted = True
                        break
                    # Concurrency cap still applies within a burst.
                    await wait_for_slot()
                await wait_for_drain()
                if exhausted or time.time() >= deadline:
                    break
                await asyncio.sleep(arrival.burst_gap_s)
        else:
            async for _ in _arrival_gate(arrival, deadline):
                await wait_for_slot()
                if not await dispatch_one():
                    break

        # Final drain.
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

    sink.emit(RUN_FINISHED, sent=sent, duration_s=time.time() - start)
