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

The hooks need neither EventLoop construction nor a model/GPU. Exercise the
submit-time probe, the replica convergence of exists and of in-flight
prefetches, and the loop's ordering of the convergence against the cache poll;
loop-level tests separately cover centralized scheduler feedback.
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

from tokenspeed.runtime.engine import l3_cache_hooks as hooks_module  # noqa: E402
from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks  # noqa: E402


class _Scheduler:
    def __init__(self) -> None:
        self.submitted: list[list] = []
        self.registered = None
        self.hash_calls: list[list[int]] = []

    def submit_requests(self, specs) -> None:
        self.submitted.append(list(specs))

    def prefix_hashes_for_tokens(self, tokens):
        self.hash_calls.append(list(tokens))
        return [f"h{len(tokens)}"]

    def expand_prefix_keys(self, hashes):
        return [0] * len(hashes), list(hashes), [0] * len(hashes)

    def register_storage_keys(self, groups, hashes, offsets) -> None:
        self.registered = (list(groups), list(hashes), list(offsets))


class _Device:
    """The DeviceHandle surface the hooks use: the exists probe and the
    prefetch lane's progress/completion."""

    def __init__(self, exists_flags: list[bool] | None) -> None:
        self.exists_flags = exists_flags
        self.pages = None
        self.progress: dict[int, tuple[bool, int]] = {}
        self.completed: list[tuple[int, int]] = []

    def query_l3_storage(self, pages):
        self.pages = list(pages)
        return None if self.exists_flags is None else list(self.exists_flags)

    def l3_prefetch_progress(self):
        return dict(self.progress)

    def complete_l3_prefetch(self, op_id, landed_pages) -> None:
        self.completed.append((int(op_id), int(landed_pages)))
        self.progress.pop(int(op_id))


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


def test_replica_min_reduces_tp_then_pp(monkeypatch) -> None:
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
        pp_size=2,
        pp_cpu_group="pp",
    )

    ctx.hooks.submit_requests([_spec("r0", [1, 2, 3, 4])])

    assert groups_seen == ["tp", "pp"]
    assert ctx.scheduler.registered is None


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
        pp_size=2,
        pp_cpu_group="pp",
    )

    ctx.hooks.submit_requests([_spec("r0", [1, 2, 3, 4])])

    assert groups_seen == ["pp"]
    assert ctx.scheduler.registered == ([0], ["h4"], [0])


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


def test_empty_submit_does_not_probe() -> None:
    ctx = _Harness(exists_flags=[True])
    ctx.hooks.submit_requests([])
    assert ctx.scheduler.submitted == [[]]
    assert ctx.scheduler.hash_calls == []
    assert ctx.device.pages is None


# ----------------------------------------------------------------------
# Per-round convergence of in-flight prefetches
# ----------------------------------------------------------------------


def test_converge_completes_only_ops_every_rank_finished(monkeypatch) -> None:
    """The replica MIN of (done, landed): a rank still fetching holds the op
    (its landed count is the sentinel), a finished one is completed with the
    common prefix, and the collective runs once per round in TP-then-PP order
    over the mirrored op-id set."""
    reductions = []

    def fake_all_reduce(flags, *, op, group):
        reductions.append((group, flags.tolist()))
        # The peer: op 3 done with 2 pages, op 5 still fetching, op 8 done with 7.
        peer = {3: (1, 2), 5: (0, hooks_module._UNLANDED), 8: (1, 7)}
        for index, op_id in enumerate((3, 5, 8)):
            flags[2 * index] = min(int(flags[2 * index]), peer[op_id][0])
            flags[2 * index + 1] = min(int(flags[2 * index + 1]), peer[op_id][1])

    monkeypatch.setattr(hooks_module.dist, "all_reduce", fake_all_reduce)
    ctx = _Harness(exists_flags=[True])
    ctx.hooks = L3CacheHooks(
        ctx.scheduler,
        ctx.device,
        attn_tp_size=2,
        attn_tp_cpu_group="tp",
        pp_size=2,
        pp_cpu_group="pp",
    )
    # This rank: op 3 landed 4 (the peer only 2), op 5 done with 1 (the peer
    # is not done), op 8 landed 7 (agreed).
    ctx.device.progress = {8: (True, 7), 3: (True, 4), 5: (True, 1)}

    ctx.hooks.converge_prefetches()

    assert [group for group, _ in reductions] == ["tp", "pp"]
    assert reductions[0][1] == [1, 4, 1, 1, 1, 7]  # op-id order: 3, 5, 8
    assert ctx.device.completed == [(3, 2), (8, 7)]
    assert ctx.device.progress == {5: (True, 1)}


def test_converge_is_a_no_op_with_nothing_in_flight_or_without_l3(monkeypatch) -> None:
    def fake_all_reduce(flags, *, op, group):
        raise AssertionError("no collective without an in-flight prefetch")

    monkeypatch.setattr(hooks_module.dist, "all_reduce", fake_all_reduce)
    ctx = _Harness(exists_flags=[True])
    ctx.hooks = L3CacheHooks(
        ctx.scheduler,
        ctx.device,
        attn_tp_size=2,
        attn_tp_cpu_group="tp",
        pp_size=1,
        pp_cpu_group=None,
    )
    ctx.hooks.converge_prefetches()
    assert ctx.device.completed == []
    _Harness(exists_flags=None).hooks.converge_prefetches()


def test_single_rank_converges_locally() -> None:
    ctx = _Harness(exists_flags=[True])
    ctx.device.progress = {2: (False, 0), 4: (True, 3)}
    ctx.hooks.converge_prefetches()
    assert ctx.device.completed == [(4, 3)]
    assert ctx.device.progress == {2: (False, 0)}


def test_the_hooks_no_longer_skip_forwards_or_reprobe_queued_hits() -> None:
    # Only the L3-to-Host leg can miss, and it runs before admission: the
    # hooks have no forward to skip, nothing to retract and no queue to
    # re-probe. The surface is the submit-time probe and the convergence.
    public = sorted(name for name in vars(L3CacheHooks) if not name.startswith("_"))
    assert public == ["converge_prefetches", "submit_requests"]


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
        "input_logprob_plan_for_forward": lambda *args: None,
        "advance_scheduler": lambda scheduler, events: scheduler.advance(events),
    }
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize("depth", [0, 1, 4])
@pytest.mark.parametrize("has_dp", [False, True])
def test_prefetch_convergence_precedes_the_rounds_cache_poll(
    loop_methods, monkeypatch, depth, has_dp
) -> None:
    """Each round converges the in-flight prefetches before the cache poll,
    so an op that landed is acknowledged in that round and reaches the
    scheduler at the head advance; the forward is never touched and the
    remote prefill always goes out."""
    trace = []
    first_forward = SimpleNamespace(request_ids=["running"], input_lengths=[1])
    first_plan = SimpleNamespace(
        forward=[first_forward], remote_prefill=None, aborts=[]
    )
    second_plan = SimpleNamespace(
        forward=[], remote_prefill=SimpleNamespace(request_ids=["remote"]), aborts=[]
    )
    plans = iter([first_plan, second_plan])

    def next_plan():
        plan = next(plans)
        trace.append(("plan", plan))
        return plan

    scheduler = SimpleNamespace(
        next_execution_plan=next_plan,
        advance=lambda events: trace.append(("advance", list(events))),
    )

    def execute(plan, planned):
        trace.append(("execute", plan, planned))
        return object() if planned is not None else None

    progress = {7: (False, 0)}

    def l3_prefetch_progress():
        trace.append(("progress", dict(progress)))
        return dict(progress)

    def complete_l3_prefetch(op_id, landed):
        trace.append(("complete", op_id, landed))
        progress.pop(op_id)

    device = SimpleNamespace(
        query_l3_storage=lambda pages: [True],
        l3_prefetch_progress=l3_prefetch_progress,
        complete_l3_prefetch=complete_l3_prefetch,
        execute=execute,
        run_idle_forward=lambda metadata: trace.append(("idle",)),
    )
    hooks = L3CacheHooks(
        scheduler,
        device,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
    )

    def poll_ready_events():
        trace.append(("poll",))
        # The lane lands op 7 between the rounds; the second round's
        # convergence completes it and this poll yields its ACK.
        if ("complete", 7, 3) in trace:
            return ["prefetch_done:7"]
        progress[7] = (True, 3)
        return ["cache0"]

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
        _eplb_hooks=SimpleNamespace(note_round=Mock()),
        _cache_hooks=SimpleNamespace(
            poll_ready_events=poll_ready_events, count_plan_ops=Mock()
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
        output_processor=SimpleNamespace(
            rid_to_state={},
            finish_scheduler_aborted_requests=lambda aborts: trace.append(
                ("aborts", list(aborts))
            ),
        ),
        _ngram_context_len=0,
        _request_history_rows=None,
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
    # The plan's remote prefill is submitted unconditionally: no withhold.
    assert executions[1] == ("execute", second_plan, None)
    assert sum(entry[0] == "idle" for entry in trace) == int(has_dp)
    # Round 1: the op is still fetching; round 2: converged and acknowledged
    # before the poll, so the ACK reaches the scheduler at the head.
    kinds = [entry[0] for entry in trace]
    first_progress = kinds.index("progress")
    assert kinds[first_progress + 1] == "poll"
    assert ("complete", 7, 3) in trace
    assert kinds.index("complete") < kinds.index("poll", kinds.index("complete"))
    advances = [entry[1] for entry in trace if entry[0] == "advance"]
    if depth == 0:
        assert advances == [["cache0"], ["committed:running"], ["prefetch_done:7"]]
    else:
        assert advances == [["cache0"], ["prefetch_done:7"], ["committed:running"]]
    for plan in (first_plan, second_plan):
        plan_index = trace.index(("plan", plan))
        assert trace[plan_index + 1] == ("aborts", [])
    loop._cache_hooks.count_plan_ops.assert_any_call(second_plan)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
