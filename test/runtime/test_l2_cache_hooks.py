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

"""CPU-only, single-rank tests for L2CacheHooks (attn_tp_size == 1, so no
collectives run). The cross-rank payload agreement itself is covered by the
pop_common_cache_event_payloads tests; these drive submit/poll bookkeeping
with a fake device handle.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed_scheduler import Cache  # noqa: E402

from tokenspeed.runtime.engine import cache_hooks as cache_hooks_module  # noqa: E402
from tokenspeed.runtime.engine.cache_hooks import L2CacheHooks  # noqa: E402


class _FakeWriteBackOp:
    def __init__(self, op_ids) -> None:
        self.op_ids = op_ids


class _Device:
    """The DeviceHandle surface the hooks use. Submission rides
    ``DeviceHandle.execute`` with the rest of the round's plan-derived device
    work; this side only counts and polls."""

    def __init__(self) -> None:
        self.results: list = []

    def poll_cache_results(self) -> list:
        results, self.results = self.results, []
        return results


def _hooks(device, speculative_algorithm=None) -> L2CacheHooks:
    return L2CacheHooks(
        device,
        speculative_algorithm=speculative_algorithm,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        global_rank=0,
    )


def _writeback_done_event(op_id: int):
    event = Cache.WriteBackDoneEvent()
    event.op_id = op_id
    return event


@pytest.fixture()
def fake_cache_ops(monkeypatch: pytest.MonkeyPatch):
    # The C++ op bindings (Cache.WriteBackOp) expose no Python constructor, so
    # substitute the type the isinstance check dispatches on.
    monkeypatch.setattr(
        cache_hooks_module,
        "Cache",
        SimpleNamespace(WriteBackOp=_FakeWriteBackOp, LoadBackOp=()),
    )


def test_disabled_kvstore_is_a_no_op() -> None:
    hooks = _hooks(None)
    hooks.count_plan_ops(SimpleNamespace(cache=[SimpleNamespace()]))
    assert hooks.poll_ready_events() == []


def test_submit_counts_in_flight_and_rejects_unknown_ops(fake_cache_ops) -> None:
    device = _Device()
    hooks = _hooks(device)
    plan = SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[1, 2])])

    hooks.count_plan_ops(plan)

    assert hooks._num_inflight == 2

    with pytest.raises(TypeError, match="unsupported cache op kind"):
        hooks.count_plan_ops(SimpleNamespace(cache=[object()]))


def test_poll_returns_completed_events_and_settles_inflight(fake_cache_ops) -> None:
    device = _Device()
    hooks = _hooks(device)
    hooks.count_plan_ops(SimpleNamespace(cache=[_FakeWriteBackOp(op_ids=[7])]))

    # Nothing completed yet: in flight, but no ready payloads.
    assert hooks.poll_ready_events() == []

    device.results = [_writeback_done_event(7)]
    events = hooks.poll_ready_events()

    assert [type(e).__name__ for e in events] == ["WriteBackDoneEvent"]
    assert events[0].op_id == 7
    assert hooks._num_inflight == 0
    # Settled: the idle short-circuit now skips polling work entirely.
    assert hooks.poll_ready_events() == []
    assert hooks._pending_payloads == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
