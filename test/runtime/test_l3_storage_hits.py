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

"""CPU-only tests for EventLoop L3 admit-path registration.

These bind ``_submit_scheduler_requests`` / ``_register_l3_storage_hits`` /
``_revalidate_queued_l3_hits`` onto a fake loop so the merge-time wiring
(pause flush, EPD drain, the normal admit path, and pre-plan revalidation
of queued hits) can be checked without a model or GPU.
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

from tokenspeed.runtime.engine.event_loop import EventLoop  # noqa: E402


class _Scheduler:
    def __init__(self) -> None:
        self.submitted: list[list] = []
        self.registered = None
        self.unregistered = None
        self.clear_result = True
        self.hash_calls: list[list[int]] = []
        self.waiting_hashes: list[str] = []

    def submit_requests(self, specs) -> None:
        self.submitted.append(list(specs))

    def waiting_prefix_hashes(self):
        return list(self.waiting_hashes)

    def prefix_hashes_for_tokens(self, tokens):
        self.hash_calls.append(list(tokens))
        return [f"h{len(tokens)}"]

    def expand_prefix_keys(self, hashes):
        return [0] * len(hashes), list(hashes), [0] * len(hashes)

    def register_storage_keys(self, groups, hashes, offsets) -> None:
        self.registered = (list(groups), list(hashes), list(offsets))

    def unregister_storage_keys(self, groups, hashes, offsets) -> None:
        self.unregistered = (list(groups), list(hashes), list(offsets))

    def clear_cache(self) -> bool:
        return self.clear_result

    def can_clear_cache(self) -> bool:
        return self.clear_result


class _Device:
    def __init__(self, exists_flags: list[bool] | None) -> None:
        self.exists_flags = exists_flags
        self.pages = None
        self.rotations = 0
        self.prefetch_ok = True
        self.prefetch_pages = False
        self.prefetch_calls = 0
        self.invalidations = 0

    def query_l3_storage(self, pages):
        self.pages = list(pages)
        return None if self.exists_flags is None else list(self.exists_flags)

    def rotate_l3_namespace(self) -> None:
        self.rotations += 1

    def plan_has_l3_prefetch(self, plan) -> bool:
        del plan
        return self.prefetch_pages

    def prefetch_l3_load_backs(self, plan) -> bool:
        del plan
        self.prefetch_calls += 1
        return self.prefetch_ok

    def invalidate_l3_prefetch(self) -> None:
        self.invalidations += 1

    def l3_prefetch_storage_keys(self, plan):
        del plan
        return [0], ["h4"], [0]


class _Loop:
    """Only the EventLoop methods under test plus the state they read."""

    _submit_scheduler_requests = EventLoop._submit_scheduler_requests
    _register_l3_storage_hits = EventLoop._register_l3_storage_hits
    _revalidate_queued_l3_hits = EventLoop._revalidate_queued_l3_hits
    _sync_l3_storage_keys = EventLoop._sync_l3_storage_keys
    _converge_l3_exists = EventLoop._converge_l3_exists
    _clear_cache = EventLoop._clear_cache
    _can_clear_cache = EventLoop._can_clear_cache
    _recover_if_l3_prefetch_failed = EventLoop._recover_if_l3_prefetch_failed

    def __init__(self, exists_flags=None) -> None:
        self._device = _Device(exists_flags)
        self.scheduler = _Scheduler()
        self.attn_tp_size = 1
        self.attn_tp_cpu_group = None
        self.attn_cp_size = 1
        self.attn_cp_cpu_group = None
        self.pp_size = 1
        self.pp_cpu_group = None
        self._enable_l3_storage = exists_flags is not None
        self.request_handler = SimpleNamespace(
            converge_replica_decision=lambda local_ok: local_ok
        )


def _spec(rid: str, tokens: list[int]):
    return SimpleNamespace(request_id=rid, tokens=tokens)


def test_submit_without_l3_still_admits() -> None:
    loop = _Loop(exists_flags=None)
    spec = _spec("r0", [1, 2, 3, 4])

    loop._submit_scheduler_requests([spec])

    assert loop.scheduler.submitted == [[spec]]
    assert loop.scheduler.registered is None
    assert loop.scheduler.hash_calls == []
    assert loop._device.pages is None


def test_submit_registers_only_keys_l3_reports_present() -> None:
    loop = _Loop(exists_flags=[True])
    spec = _spec("r0", [1, 2, 3, 4])

    loop._submit_scheduler_requests([spec])

    assert loop.scheduler.submitted == [[spec]]
    assert loop.scheduler.hash_calls == [[1, 2, 3, 4]]
    assert loop._device.pages == [(0, 0, "h4", 0)]
    assert loop.scheduler.registered == ([0], ["h4"], [0])


def test_submit_skips_register_when_l3_misses() -> None:
    loop = _Loop(exists_flags=[False])
    spec = _spec("r0", [1, 2, 3, 4])

    loop._submit_scheduler_requests([spec])

    assert loop.scheduler.submitted == [[spec]]
    assert loop.scheduler.registered is None
    assert loop.scheduler.unregistered == ([0], ["h4"], [0])


def test_successful_clear_rotates_l3_namespace() -> None:
    loop = _Loop(exists_flags=[True])
    assert loop._clear_cache()
    assert loop._device.rotations == 1

    loop.scheduler.clear_result = False
    assert not loop._clear_cache()
    assert loop._device.rotations == 1


def test_can_clear_does_not_rotate_l3_namespace() -> None:
    loop = _Loop(exists_flags=[True])
    assert loop._can_clear_cache()
    assert loop._device.rotations == 0
    loop.scheduler.clear_result = False
    assert not loop._can_clear_cache()
    assert loop._device.rotations == 0


def test_replica_min_reduces_tp_then_cp_then_pp(monkeypatch) -> None:
    groups_seen = []

    def fake_all_reduce(flags, op=None, group=None):
        groups_seen.append(group)
        flags.fill_(0)

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.event_loop.dist.all_reduce",
        fake_all_reduce,
    )

    loop = _Loop(exists_flags=[True])
    loop.attn_tp_size = 2
    loop.attn_tp_cpu_group = "tp"
    loop.attn_cp_size = 2
    loop.attn_cp_cpu_group = "cp"
    loop.pp_size = 2
    loop.pp_cpu_group = "pp"

    loop._submit_scheduler_requests([_spec("r0", [1, 2, 3, 4])])

    assert groups_seen == ["tp", "cp", "pp"]
    assert loop.scheduler.registered is None
    assert loop.scheduler.unregistered == ([0], ["h4"], [0])


def test_enable_cp_min_uses_cp_group_when_tp_is_one(monkeypatch) -> None:
    groups_seen = []

    def fake_all_reduce(flags, op=None, group=None):
        groups_seen.append(group)

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.event_loop.dist.all_reduce",
        fake_all_reduce,
    )

    loop = _Loop(exists_flags=[True])
    loop.attn_tp_size = 1
    loop.attn_tp_cpu_group = "tp"
    loop.attn_cp_size = 4
    loop.attn_cp_cpu_group = "cp"

    loop._submit_scheduler_requests([_spec("r0", [1, 2, 3, 4])])

    assert groups_seen == ["cp"]
    assert loop.scheduler.registered == ([0], ["h4"], [0])


def test_pp_min_runs_when_attn_tp_is_one(monkeypatch) -> None:
    groups_seen = []

    def fake_all_reduce(flags, op=None, group=None):
        groups_seen.append(group)

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.event_loop.dist.all_reduce",
        fake_all_reduce,
    )

    loop = _Loop(exists_flags=[True])
    loop.pp_size = 2
    loop.pp_cpu_group = "pp"

    loop._submit_scheduler_requests([_spec("r0", [1, 2, 3, 4])])

    assert groups_seen == ["pp"]
    assert loop.scheduler.registered == ([0], ["h4"], [0])


def test_revalidate_unregisters_stale_queued_l3_hits() -> None:
    """A queued hit that later misses must drop the scheduler key before admit."""

    loop = _Loop(exists_flags=[True])
    spec = _spec("r0", [1, 2, 3, 4])
    loop._submit_scheduler_requests([spec])
    assert loop.scheduler.registered == ([0], ["h4"], [0])
    assert loop.scheduler.unregistered is None

    loop._device.exists_flags = [False]
    loop.scheduler.waiting_hashes = ["h4"]
    loop._revalidate_queued_l3_hits()

    assert loop.scheduler.unregistered == ([0], ["h4"], [0])
    assert loop.scheduler.hash_calls == [[1, 2, 3, 4]]


def test_revalidate_skipped_without_l3() -> None:
    loop = _Loop(exists_flags=None)
    loop.scheduler.waiting_hashes = ["h4"]
    loop._revalidate_queued_l3_hits()
    assert loop._device.pages is None
    assert loop.scheduler.registered is None
    assert loop.scheduler.unregistered is None


def test_vanished_l3_prefetch_unregisters_and_retracts(monkeypatch) -> None:
    retracts: list[str] = []

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.event_loop.make_retract_event",
        lambda rid: retracts.append(rid) or f"retract:{rid}",
    )

    loop = _Loop(exists_flags=[True])
    loop._device.prefetch_pages = True
    loop._device.prefetch_ok = False
    forward_op = SimpleNamespace(request_ids=["r0", "r1"])

    events = loop._recover_if_l3_prefetch_failed(SimpleNamespace(), forward_op)

    assert loop._device.prefetch_calls == 1
    assert loop._device.invalidations == 1
    assert loop.scheduler.unregistered == ([0], ["h4"], [0])
    assert retracts == ["r0", "r1"]
    assert events == ["retract:r0", "retract:r1"]


def test_l3_prefetch_success_does_not_retract() -> None:
    loop = _Loop(exists_flags=[True])
    loop._device.prefetch_pages = True
    loop._device.prefetch_ok = True
    forward_op = SimpleNamespace(request_ids=["r0"])

    events = loop._recover_if_l3_prefetch_failed(SimpleNamespace(), forward_op)

    assert events == []
    assert loop._device.invalidations == 0
    assert loop.scheduler.unregistered is None
    assert loop._device.prefetch_calls == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
