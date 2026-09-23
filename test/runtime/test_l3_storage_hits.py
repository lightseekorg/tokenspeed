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

"""L3 hooks tests with a real component and fake scheduler/device boundaries.

The hooks need neither EventLoop construction nor a model/GPU. Exercise
admission, replica convergence and plan recovery directly; loop-level tests
separately cover dispatch suppression and centralized scheduler feedback.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# CPU-only tests scheduled with the other runtime hooks tests.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.cache.l3.backend import L3UnreadKeySet  # noqa: E402
from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks  # noqa: E402


class _Scheduler:
    def __init__(self) -> None:
        self.submitted: list[list] = []
        self.registered = None
        self.unregistered = None
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


class _Device:
    def __init__(self, exists_flags: list[bool] | None) -> None:
        self.exists_flags = exists_flags
        self.pages = None
        self.rotations = 0
        self.prefetch_ok = True
        self.prefetch_pages = False
        self.prefetch_calls = 0
        self.invalidations = 0
        self._l3_unread = L3UnreadKeySet(capacity=8)

    def query_l3_storage(self, pages):
        self.pages = list(pages)
        return None if self.exists_flags is None else list(self.exists_flags)

    def delete_l3_namespace(self) -> bool:
        self.rotations += 1
        self._l3_unread.clear()
        return True

    def mark_l3_keys_unread(self, groups, hashes, offsets) -> None:
        self._l3_unread.mark(groups=groups, hashes=hashes, offsets=offsets)

    def l3_key_is_unread(self, group_id, content_hash, page_offset) -> bool:
        return self._l3_unread.contains(
            group_id=int(group_id),
            content_hash=str(content_hash),
            page_offset=int(page_offset),
        )

    def forget_l3_unread_keys(self, groups, hashes, offsets) -> None:
        self._l3_unread.forget(groups=groups, hashes=hashes, offsets=offsets)

    def plan_has_l3_prefetch(self, plan) -> bool:
        del plan
        return self.prefetch_pages

    def prefetch_l3_load_backs(self, plan) -> list[bool]:
        self.prefetch_calls += 1
        groups, _hashes, _offsets = self.l3_prefetch_storage_keys(plan)
        flags = self.prefetch_ok
        if isinstance(flags, bool):
            return [bool(flags)] * len(groups)
        return [bool(flag) for flag in flags]

    def invalidate_l3_prefetch(self) -> None:
        self.invalidations += 1

    def l3_prefetch_storage_keys(self, plan):
        del plan
        return [0], ["h4"], [0]


class _Harness:
    """Construct the real hooks with explicit scheduler/device dependencies."""

    def __init__(self, exists_flags) -> None:
        self.device = _Device(exists_flags)
        self.scheduler = _Scheduler()
        self.hooks = L3CacheHooks(
            self.scheduler,
            self.device if exists_flags is not None else None,
            attn_tp_size=1,
            attn_tp_cpu_group=None,
            attn_cp_size=1,
            attn_cp_cpu_group=None,
            pp_size=1,
            pp_cpu_group=None,
        )


def test_hooks_require_explicit_configuration() -> None:
    for param in inspect.signature(L3CacheHooks.__init__).parameters.values():
        assert param.default is inspect.Parameter.empty


def test_l3_exists_reduce_doubles_require_op_and_group() -> None:
    tree = ast.parse(Path(__file__).read_text())
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "fake_all_reduce":
            continue
        found = True
        defaults = dict(
            zip((arg.arg for arg in node.args.kwonlyargs), node.args.kw_defaults)
        )
        assert node.args.defaults == []
        assert defaults["op"] is None
        assert defaults["group"] is None
    assert found


def _spec(rid: str, tokens: list[int]):
    return SimpleNamespace(request_id=rid, tokens=tokens)


def _recover_plan(*, remote_prefill):
    return SimpleNamespace(remote_prefill=remote_prefill)


def test_submit_without_l3_still_admits() -> None:
    ctx = _Harness(exists_flags=None)
    spec = _spec("r0", [1, 2, 3, 4])

    ctx.hooks.submit_requests([spec])

    assert ctx.scheduler.submitted == [[spec]]
    assert ctx.scheduler.registered is None
    assert ctx.scheduler.hash_calls == []
    assert ctx.device.pages is None


def test_submit_registers_only_keys_l3_reports_present() -> None:
    ctx = _Harness(exists_flags=[True])
    spec = _spec("r0", [1, 2, 3, 4])

    ctx.hooks.submit_requests([spec])

    assert ctx.scheduler.submitted == [[spec]]
    assert ctx.scheduler.hash_calls == [[1, 2, 3, 4]]
    assert ctx.device.pages == [(0, 0, "h4", 0)]
    assert ctx.scheduler.registered == ([0], ["h4"], [0])


def test_submit_skips_register_when_l3_misses() -> None:
    ctx = _Harness(exists_flags=[False])
    spec = _spec("r0", [1, 2, 3, 4])

    ctx.hooks.submit_requests([spec])

    assert ctx.scheduler.submitted == [[spec]]
    assert ctx.scheduler.registered is None
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])


def test_replica_min_reduces_tp_then_cp_then_pp(monkeypatch) -> None:
    groups_seen = []

    def fake_all_reduce(flags, *, op, group):
        groups_seen.append(group)
        flags.fill_(0)

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.dist.all_reduce",
        fake_all_reduce,
    )

    ctx = _Harness(exists_flags=[True])
    ctx.hooks = L3CacheHooks(
        ctx.scheduler,
        ctx.device,
        attn_tp_size=2,
        attn_tp_cpu_group="tp",
        attn_cp_size=2,
        attn_cp_cpu_group="cp",
        pp_size=2,
        pp_cpu_group="pp",
    )

    ctx.hooks.submit_requests([_spec("r0", [1, 2, 3, 4])])

    assert groups_seen == ["tp", "cp", "pp"]
    assert ctx.scheduler.registered is None
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])


def test_enable_cp_min_uses_cp_group_when_tp_is_one(monkeypatch) -> None:
    groups_seen = []

    def fake_all_reduce(flags, *, op, group):
        groups_seen.append(group)

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.dist.all_reduce",
        fake_all_reduce,
    )

    ctx = _Harness(exists_flags=[True])
    ctx.hooks = L3CacheHooks(
        ctx.scheduler,
        ctx.device,
        attn_tp_size=1,
        attn_tp_cpu_group="tp",
        attn_cp_size=4,
        attn_cp_cpu_group="cp",
        pp_size=1,
        pp_cpu_group=None,
    )

    ctx.hooks.submit_requests([_spec("r0", [1, 2, 3, 4])])

    assert groups_seen == ["cp"]
    assert ctx.scheduler.registered == ([0], ["h4"], [0])


def test_pp_min_runs_when_attn_tp_is_one(monkeypatch) -> None:
    groups_seen = []

    def fake_all_reduce(flags, *, op, group):
        groups_seen.append(group)

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.dist.all_reduce",
        fake_all_reduce,
    )

    ctx = _Harness(exists_flags=[True])
    ctx.hooks = L3CacheHooks(
        ctx.scheduler,
        ctx.device,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        attn_cp_size=1,
        attn_cp_cpu_group=None,
        pp_size=2,
        pp_cpu_group="pp",
    )

    ctx.hooks.submit_requests([_spec("r0", [1, 2, 3, 4])])

    assert groups_seen == ["pp"]
    assert ctx.scheduler.registered == ([0], ["h4"], [0])


def test_revalidate_unregisters_stale_queued_l3_hits() -> None:
    """A queued hit that later misses must drop the scheduler key before admit."""

    ctx = _Harness(exists_flags=[True])
    spec = _spec("r0", [1, 2, 3, 4])
    ctx.hooks.submit_requests([spec])
    assert ctx.scheduler.registered == ([0], ["h4"], [0])
    assert ctx.scheduler.unregistered is None

    ctx.device.exists_flags = [False]
    ctx.scheduler.waiting_hashes = ["h4"]
    ctx.hooks.revalidate_queued_hits()

    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])
    assert ctx.scheduler.hash_calls == [[1, 2, 3, 4]]


def test_revalidate_skipped_without_l3() -> None:
    ctx = _Harness(exists_flags=None)
    ctx.scheduler.waiting_hashes = ["h4"]
    ctx.hooks.revalidate_queued_hits()
    assert ctx.device.pages is None
    assert ctx.scheduler.registered is None
    assert ctx.scheduler.unregistered is None


def test_revalidate_skips_device_when_waiting_hashes_empty() -> None:
    ctx = _Harness(exists_flags=[True])
    ctx.scheduler.waiting_hashes = []
    ctx.hooks.revalidate_queued_hits()
    assert ctx.device.pages is None
    assert ctx.scheduler.registered is None
    assert ctx.scheduler.unregistered is None


def test_vanished_l3_prefetch_unregisters_and_retracts(monkeypatch) -> None:
    retracts: list[str] = []

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: retracts.append(rid) or f"retract:{rid}",
    )

    ctx = _Harness(exists_flags=[True])
    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = False
    forward_op = SimpleNamespace(request_ids=["r0", "r1"])

    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), forward_op
    )

    assert ctx.device.prefetch_calls == 1
    assert ctx.device.invalidations == 1
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])
    assert retracts == ["r0", "r1"]
    assert safe_forward is None
    assert events == ["retract:r0", "retract:r1"]


def test_vanished_l3_prefetch_retracts_remote_prefill(monkeypatch) -> None:
    """D-role admit has no local forward; vanished L3 must still retract."""

    retracts: list[str] = []

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: retracts.append(rid) or f"retract:{rid}",
    )

    ctx = _Harness(exists_flags=[True])
    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = False
    remote_prefill = SimpleNamespace(request_ids=["r0", "r1"])

    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=remote_prefill), None
    )

    assert ctx.device.prefetch_calls == 1
    assert ctx.device.invalidations == 1
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])
    assert retracts == ["r0", "r1"]
    assert safe_forward is None
    assert events == ["retract:r0", "retract:r1"]


def test_vanished_l3_prefetch_retracts_forward_and_remote_prefill_once(
    monkeypatch,
) -> None:
    """A D-role round may carry both a local decode and a remote admission."""

    retracts: list[str] = []

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: retracts.append(rid) or f"retract:{rid}",
    )

    ctx = _Harness(exists_flags=[True])
    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = False
    forward_op = SimpleNamespace(request_ids=["r1", "r2"])
    remote_prefill = SimpleNamespace(request_ids=["r0", "r1"])

    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=remote_prefill), forward_op
    )

    assert retracts == ["r1", "r2", "r0"]
    assert safe_forward is None
    assert events == ["retract:r1", "retract:r2", "retract:r0"]


def test_failed_l3_prefetch_is_not_reregistered_while_exists_stays_true(
    monkeypatch,
) -> None:
    """A get-failure must not be re-admitted from a later batch_exists hit."""

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: f"retract:{rid}",
    )
    ctx = _Harness(exists_flags=[True])
    spec = _spec("r0", [1, 2, 3, 4])
    ctx.hooks.submit_requests([spec])
    assert ctx.scheduler.registered == ([0], ["h4"], [0])

    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = False
    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), SimpleNamespace(request_ids=["r0"])
    )
    assert safe_forward is None
    assert events
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])
    ctx.scheduler.registered = None
    ctx.scheduler.unregistered = None
    ctx.scheduler.waiting_hashes = ["h4"]
    ctx.hooks.revalidate_queued_hits()

    assert ctx.scheduler.registered is None
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])


def test_unread_miss_is_min_reduced_with_exists() -> None:
    """Local unread must enter the replica MIN, not filter after it."""

    ctx = _Harness(exists_flags=[True])
    ctx.device.mark_l3_keys_unread([0], ["h4"], [0])
    probed: list[list[bool]] = []
    bound = ctx.hooks._converge_l3_exists

    def wrapped(exists):
        probed.append(list(exists))
        return bound(exists)

    ctx.hooks._converge_l3_exists = wrapped
    ctx.hooks.submit_requests([_spec("r0", [1, 2, 3, 4])])
    assert probed == [[False]]
    assert ctx.scheduler.registered is None
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])


def test_namespace_delete_forgets_unread_l3_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: f"retract:{rid}",
    )
    ctx = _Harness(exists_flags=[True])
    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = False
    ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), SimpleNamespace(request_ids=["r0"])
    )
    assert ctx.device.delete_l3_namespace()
    ctx.scheduler.registered = None
    ctx.scheduler.unregistered = None
    ctx.scheduler.waiting_hashes = ["h4"]
    ctx.hooks.revalidate_queued_hits()
    assert ctx.scheduler.registered == ([0], ["h4"], [0])


def test_prefetch_rpc_error_converges_then_retracts(monkeypatch) -> None:
    """A local batch_get_into exception must still enter the replica MIN."""

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: f"retract:{rid}",
    )
    ctx = _Harness(exists_flags=[True])
    ctx.device.prefetch_pages = True
    probed: list[list[bool]] = []
    bound = ctx.hooks._converge_l3_exists

    def boom(plan) -> list[bool]:
        del plan
        ctx.device.prefetch_calls += 1
        raise RuntimeError("batch_get_into failed")

    def wrapped(exists):
        probed.append(list(exists))
        return bound(exists)

    ctx.device.prefetch_l3_load_backs = boom
    ctx.hooks._converge_l3_exists = wrapped
    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), SimpleNamespace(request_ids=["r0"])
    )
    assert probed == [[False]]
    assert safe_forward is None
    assert events == ["retract:r0"]
    assert ctx.device.invalidations == 1
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])


def test_exists_rpc_error_converges_as_misses() -> None:
    """A local batch_exists exception must still enter the replica MIN."""

    ctx = _Harness(exists_flags=[True])
    probed: list[list[bool]] = []
    bound = ctx.hooks._converge_l3_exists

    def wrapped(exists):
        probed.append(list(exists))
        return bound(exists)

    ctx.hooks._converge_l3_exists = wrapped

    def boom(pages):
        ctx.device.pages = list(pages)
        raise RuntimeError("batch_is_exist failed")

    ctx.device.query_l3_storage = boom
    ctx.hooks.submit_requests([_spec("r0", [1, 2, 3, 4])])
    assert probed == [[False]]
    assert ctx.scheduler.registered is None
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])


def test_exists_length_mismatch_converges_as_misses() -> None:
    ctx = _Harness(exists_flags=[True, True])
    probed: list[list[bool]] = []
    bound = ctx.hooks._converge_l3_exists

    def wrapped(exists):
        probed.append(list(exists))
        return bound(exists)

    ctx.hooks._converge_l3_exists = wrapped
    ctx.hooks.submit_requests([_spec("r0", [1, 2, 3, 4])])
    assert probed == [[False]]
    assert ctx.scheduler.registered is None
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])


def test_prefetch_length_mismatch_converges_as_misses(monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: f"retract:{rid}",
    )
    ctx = _Harness(exists_flags=[True])
    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = [True, True]
    probed: list[list[bool]] = []
    bound = ctx.hooks._converge_l3_exists

    def wrapped(exists):
        probed.append(list(exists))
        return bound(exists)

    ctx.hooks._converge_l3_exists = wrapped
    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), SimpleNamespace(request_ids=["r0"])
    )
    assert probed == [[False]]
    assert safe_forward is None
    assert events == ["retract:r0"]
    assert ctx.scheduler.unregistered == ([0], ["h4"], [0])


def test_l3_prefetch_success_does_not_retract() -> None:
    ctx = _Harness(exists_flags=[True])
    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = True
    forward_op = SimpleNamespace(request_ids=["r0"])

    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), forward_op
    )

    assert safe_forward is forward_op
    assert events == []
    assert ctx.device.invalidations == 0
    assert ctx.scheduler.unregistered is None
    assert ctx.device.prefetch_calls == 1


def test_mixed_l3_prefetch_blacklists_only_failed_pages(monkeypatch) -> None:
    """A vanished tail must not unread pages whose replica get succeeded."""

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: f"retract:{rid}",
    )
    ctx = _Harness(exists_flags=[True, True])
    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = [True, False]

    def keys(plan):
        del plan
        return [0, 0], ["h4", "h5"], [0, 0]

    ctx.device.l3_prefetch_storage_keys = keys
    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), SimpleNamespace(request_ids=["r0"])
    )
    assert safe_forward is None
    assert events == ["retract:r0"]
    assert ctx.scheduler.unregistered == ([0], ["h5"], [0])
    assert ctx.device.l3_key_is_unread(0, "h4", 0) is False
    assert ctx.device.l3_key_is_unread(0, "h5", 0) is True

    ctx.scheduler.registered = None
    ctx.scheduler.unregistered = None
    ctx.scheduler.waiting_hashes = ["h4", "h5"]
    ctx.hooks.revalidate_queued_hits()
    assert ctx.scheduler.registered == ([0], ["h4"], [0])
    assert ctx.scheduler.unregistered == ([0], ["h5"], [0])


def test_successful_republish_clears_unread_l3_key(monkeypatch) -> None:
    """A Host backup that creates a missing object may restore L3 reuse."""

    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: f"retract:{rid}",
    )
    ctx = _Harness(exists_flags=[True])
    ctx.device.prefetch_pages = True
    ctx.device.prefetch_ok = False
    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), SimpleNamespace(request_ids=["r0"])
    )
    assert safe_forward is None
    assert events
    assert ctx.device.l3_key_is_unread(0, "h4", 0) is True
    ctx.device.forget_l3_unread_keys([0], ["h4"], [0])
    ctx.scheduler.registered = None
    ctx.scheduler.unregistered = None
    ctx.scheduler.waiting_hashes = ["h4"]
    ctx.hooks.revalidate_queued_hits()
    assert ctx.scheduler.registered == ([0], ["h4"], [0])
    assert ctx.scheduler.unregistered is None


@pytest.mark.parametrize("exists_flags", [None, [True]])
def test_prepare_without_l3_pages_preserves_forward(exists_flags) -> None:
    ctx = _Harness(exists_flags=exists_flags)
    forward = SimpleNamespace(request_ids=["r0"])
    safe_forward, events = ctx.hooks.prepare_forward(
        _recover_plan(remote_prefill=None), forward
    )
    assert safe_forward is forward
    assert events == []
    assert ctx.device.prefetch_calls == 0
    assert ctx.device.invalidations == 0


def test_empty_submit_does_not_probe() -> None:
    ctx = _Harness(exists_flags=[True])
    ctx.hooks.submit_requests([])
    assert ctx.scheduler.submitted == [[]]
    assert ctx.scheduler.hash_calls == []
    assert ctx.device.pages is None


@pytest.fixture
def loop_methods():
    # Load the actual scheduling methods without importing model/GPU backends.
    # No rewriting: the collaborator boundaries below are the same ones the
    # running loop uses. This keeps the complete round test runnable on CPU.
    path = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/engine/event_loop.py"
    )
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "EventLoop"
    )
    names = {"event_loop", "_drain_in_flight", "_get_forward_op"}
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "deque": deque,
        "maybe_control_plane_guard": nullcontext,
        "PlannedForward": SimpleNamespace,
        "ngram_inputs_for_forward": lambda *args: None,
        "advance_scheduler": lambda scheduler, events: scheduler.advance(events),
    }
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize("depth", [0, 1, 4])
@pytest.mark.parametrize("has_dp", [False, True])
@pytest.mark.parametrize("remote_only", [False, True])
def test_l3_recovery_preserves_round_order(
    loop_methods, monkeypatch, depth, has_dp, remote_only
) -> None:
    """Cache ops still execute, and prior commits precede the single tail retract."""
    trace = []
    first_forward = SimpleNamespace(request_ids=["running"], input_lengths=[1])
    failed_forward = SimpleNamespace(
        request_ids=["running", "new"], input_lengths=[1, 4]
    )
    first_plan = SimpleNamespace(forward=[first_forward], remote_prefill=None)
    failed_plan = SimpleNamespace(
        forward=[] if remote_only else [failed_forward],
        remote_prefill=SimpleNamespace(request_ids=["remote"]),
    )
    plans = iter([first_plan, failed_plan])

    def next_plan():
        plan = next(plans)
        trace.append(("plan", plan))
        return plan

    scheduler = SimpleNamespace(
        waiting_prefix_hashes=lambda: ["h4"],
        expand_prefix_keys=lambda hashes: ([0], hashes, [0]),
        register_storage_keys=Mock(),
        unregister_storage_keys=Mock(),
        next_execution_plan=next_plan,
        advance=lambda events: trace.append(("advance", list(events))),
    )

    def exists(pages):
        trace.append(("probe", pages))
        return [True]

    def execute(plan, planned, *, submit_remote_prefill):
        trace.append(("execute", plan, planned, submit_remote_prefill))
        return object() if planned is not None else None

    device = SimpleNamespace(
        query_l3_storage=exists,
        l3_key_is_unread=lambda **kwargs: False,
        plan_has_l3_prefetch=lambda plan: plan is failed_plan,
        l3_prefetch_storage_keys=lambda plan: ([0], ["h4"], [0]),
        prefetch_l3_load_backs=lambda plan: [False],
        invalidate_l3_prefetch=Mock(),
        mark_l3_keys_unread=Mock(),
        execute=execute,
        run_idle_forward=lambda metadata: trace.append(("idle",)),
    )
    hooks = L3CacheHooks(
        scheduler,
        device,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        attn_cp_size=1,
        attn_cp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.engine.l3_cache_hooks.make_retract_event",
        lambda rid: f"retract:{rid}",
    )

    def commit(forward, result):
        assert forward is first_forward
        trace.append(("commit",))
        return ["committed:running"]

    loop = SimpleNamespace(
        in_flight_depth=depth,
        _shutdown_complete=Mock(side_effect=[False, False, True]),
        _process_new_requests=Mock(),
        _get_forward_op=lambda plan: loop_methods["_get_forward_op"](loop, plan),
        _drain_in_flight=lambda pending: loop_methods["_drain_in_flight"](
            loop, pending
        ),
        _epd_hooks=SimpleNamespace(
            drain_ready_embeddings=Mock(), assert_embeddings_received=Mock()
        ),
        _cache_hooks=SimpleNamespace(
            poll_ready_events=Mock(side_effect=[["cache0"], ["cache1"]]),
            count_plan_ops=Mock(),
        ),
        _l3_hooks=hooks,
        _pause=SimpleNamespace(forward_blocked=False, maybe_finish_drain=Mock()),
        scheduler=scheduler,
        _device=device,
        _get_scheduler_stats=Mock(return_value={}),
        load_reporter=SimpleNamespace(observe=Mock()),
        _num_running=Mock(return_value=1),
        _record_scheduler_iteration_metrics=Mock(),
        has_dp=has_dp,
        _dp_sync_and_check=lambda forward: SimpleNamespace(
            need_idle_forward=forward is None
        ),
        _gather_sampling_params=Mock(return_value=[]),
        _gather_grammar_state=Mock(return_value=None),
        output_processor=SimpleNamespace(rid_to_state={}),
        _ngram_context_len=0,
        _dispatch_depends_on_pending_commit=Mock(return_value=False),
        _mark_stats_scheduled=Mock(),
        _batch_logger=SimpleNamespace(log_dispatch=Mock()),
        model_config=SimpleNamespace(is_multimodal_active=False),
        _pd_hooks=SimpleNamespace(poll_transfer_events=Mock(return_value=[])),
        _commit_forward_results=commit,
        _publish_scheduler_kv_events=Mock(),
    )
    loop_methods["event_loop"](loop)

    executions = [entry for entry in trace if entry[0] == "execute"]
    assert len(executions) == 2
    assert executions[0][2].forward_op is first_forward
    assert executions[0][3] is True
    assert executions[1] == ("execute", failed_plan, None, False)
    assert sum(entry[0] == "idle" for entry in trace) == int(has_dp)
    retracts = ([] if remote_only else ["retract:running", "retract:new"]) + [
        "retract:remote"
    ]
    advances = [entry[1] for entry in trace if entry[0] == "advance"]
    if depth == 0:
        assert advances == [["cache0"], ["committed:running"], ["cache1"], retracts]
    else:
        assert advances == [["cache0"], ["cache1"], ["committed:running", *retracts]]
    assert trace.index(("commit",)) < trace.index(("advance", advances[-1]))
    for plan in (first_plan, failed_plan):
        plan_index = trace.index(("plan", plan))
        assert trace[plan_index - 1][0] == "probe"
        assert trace[plan_index - 2][0] == "advance"
    device.invalidate_l3_prefetch.assert_called_once_with()
    scheduler.unregister_storage_keys.assert_called_once_with([0], ["h4"], [0])
    loop._cache_hooks.count_plan_ops.assert_any_call(failed_plan)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
