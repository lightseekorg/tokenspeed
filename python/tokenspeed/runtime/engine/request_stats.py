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

"""Per-request performance stats for --enable-log-request-stats.

Two cohesive pieces:
- ``RequestStatsTracker``: a mutable, host-side accumulator that the engine fires
  lifecycle events into (scheduled/prefill-done/first-token/finish, preemption).
  ``NOOP_STATS`` is a shared null-object instance carried by requests until
  ``register()`` attaches a real tracker, so call sites need no per-request guard.
- ``RequestStats``: the immutable, rounded summary derived from a tracker +
  RequestState via ``from_state``, logged as a Python-object repr.

Everything here is host-side; building these introduces no GPU sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed.runtime.engine.request_types import FINISH_ABORT

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.generation_output_processor import RequestState


def _ms(end: float, start: float) -> float:
    # 0 for unset (0.0) timestamps, not a garbage epoch-sized number
    return round((end - start) * 1e3, 2) if end > 0.0 and start > 0.0 else 0.0


class RequestStatsTracker:
    """Host-side per-request timing/preemption accumulator (--enable-log-request-stats).

    Attached to a RequestState only when logging is on; the engine fires
    lifecycle events into it. No GPU sync.
    """

    def __init__(self) -> None:
        self.scheduled_time = 0.0
        self.prefill_done_time = 0.0
        self.first_token_time = 0.0
        self.finish_time = 0.0
        self.preempt_count = 0
        self.preempt_time = 0.0
        self._preempted_last_step = False

    def mark_scheduled(self, now: float) -> None:
        if self.scheduled_time == 0.0:
            self.scheduled_time = now

    def mark_prefill_done(self, now: float) -> None:
        if self.prefill_done_time == 0.0:
            self.prefill_done_time = now

    def mark_first_token(self, now: float) -> None:
        if self.first_token_time == 0.0:
            self.first_token_time = now

    def mark_finish(self, now: float) -> None:
        self.finish_time = now

    def record_decode_step(self, step_dt: float, prefilling_others: bool) -> None:
        # count each prefill-of-others interruption (rising edge) and its time
        if prefilling_others:
            if not self._preempted_last_step:
                self.preempt_count += 1
            self.preempt_time += step_dt
            self._preempted_last_step = True
        else:
            self._preempted_last_step = False


class _NoOpStatsTracker(RequestStatsTracker):
    """Null-object tracker used when --enable-log-request-stats is off, so the engine
    can fire lifecycle events unconditionally without a per-request guard. The
    only stats check lives in _log_request_stats (which skips this singleton).

    Frozen on purpose: it is a SHARED singleton, so any attribute write raises.
    If a future tracker mutator is added without a no-op override here, the call
    falls through to the base method and fails loudly instead of silently
    corrupting shared state across all flag-off requests.
    """

    def __init__(self) -> None:
        pass  # carries no per-instance state; every method below is a no-op

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("NOOP_STATS is a read-only no-op singleton")

    def mark_scheduled(self, now: float) -> None:
        pass

    def mark_prefill_done(self, now: float) -> None:
        pass

    def mark_first_token(self, now: float) -> None:
        pass

    def mark_finish(self, now: float) -> None:
        pass

    def record_decode_step(self, step_dt: float, prefilling_others: bool) -> None:
        pass


# Shared singleton: requests carry this until register() attaches a real tracker.
NOOP_STATS = _NoOpStatsTracker()


@dataclass
class RequestStats:
    """Host-side per-request perf summary (--enable-log-request-stats), logged as repr.

    Durations are ms, rates are 0-1, ``*_ts`` are epoch seconds; ``acc_*`` are
    None when spec decode is off. No GPU sync.
    """

    status: str
    reason: str

    prompt_tokens: int
    cache_tokens: int
    output_tokens: int
    cache_hit_rate: float

    queue_ms: float
    prefill_ms: float
    ttft_ms: float
    total_ms: float
    preempt_ms: float
    preempt_count: int
    decode_tps: float

    acc_len: float | None
    acc_rate: float | None

    recv_ts: float
    commit_ts: float
    finish_ts: float

    @classmethod
    def from_state(
        cls, rs: RequestState, spec_algorithm, spec_num_tokens
    ) -> RequestStats:
        t = rs.stats
        prompt = rs.input_length
        output = rs.output_length

        decode_window = (
            t.finish_time - t.first_token_time
            if t.finish_time > 0.0 and t.first_token_time > 0.0
            else 0.0
        )
        decode_tps = (
            round((output - 1) / decode_window, 1)
            if decode_window > 0.0 and output > 1
            else 0.0
        )

        # spec acceptance; None when spec decode is off
        if spec_algorithm is not None and rs.spec_verify_ct > 0:
            acc_len = rs.accept_draft_tokens or 0.0
            acc_rate = (
                round(max(0.0, acc_len - 1.0) / spec_num_tokens, 4)
                if spec_num_tokens
                else 0.0
            )
            acc_len = round(acc_len, 2)
        else:
            acc_len = acc_rate = None

        return cls(
            status=(
                "aborted"
                if isinstance(rs.finished_reason, FINISH_ABORT)
                else "finished"
            ),
            reason=(
                rs.finished_reason.to_json().get("type", "unknown")
                if rs.finished_reason is not None
                else "unknown"
            ),
            prompt_tokens=prompt,
            cache_tokens=rs.cached_tokens,
            output_tokens=output,
            cache_hit_rate=round(rs.cached_tokens / prompt, 4) if prompt > 0 else 0.0,
            queue_ms=_ms(t.scheduled_time, rs.created_time),
            prefill_ms=_ms(t.prefill_done_time, t.scheduled_time),
            ttft_ms=_ms(t.first_token_time, rs.created_time),
            total_ms=_ms(t.finish_time, rs.created_time),
            preempt_ms=round(t.preempt_time * 1e3, 2),
            preempt_count=t.preempt_count,
            decode_tps=decode_tps,
            acc_len=acc_len,
            acc_rate=acc_rate,
            recv_ts=round(rs.created_time, 3),
            commit_ts=round(t.scheduled_time, 3),
            finish_ts=round(t.finish_time, 3),
        )
