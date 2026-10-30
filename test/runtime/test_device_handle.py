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

"""The cross-plane contract, at the places that can break it.

Two halves, both stated in the capture contract of ``execution/device.py``.
What a launched forward reads is frozen; and the device side is not
reachable from the control plane at all, because the data plane holds a
``DeviceHandle`` rather than the executor behind it.

The first half is easy to lose in specific spots, so each gets a test: the
PD completion path (device writes issued straight from an event handler)
and the multimodal gather (a live request struct captured by reference).

The second half is exercised the way the forward thread experiences it: the
tests build a real ``DeviceHandle`` over a stub executor and drive its
operations, asserting the work that reaches the device side, in order, and
the facts that come back.
"""

from __future__ import annotations

import functools
import json
import threading
from contextlib import nullcontext
from dataclasses import replace
from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from tokenspeed_scheduler import (
    AttnKind,
    CacheGroupConfig,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)
from transformers import PretrainedConfig

from tokenspeed.runtime.cache.transfer.ops import (
    HostTier,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine import scheduler_utils as scheduler_utils_module
from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan
from tokenspeed.runtime.execution.device import DeviceHandle, EncoderModelFacts
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import PlannedForward
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    multimodal_context_for_forward,
)
from tokenspeed.runtime.pd.host_peer import build_host_peer  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-2gpu")


def _execute(handle, plan, planned):
    return handle.execute(plan, planned, cache_ops=cache_ops_from_plan(plan).device)


@pytest.mark.parametrize(
    "algorithm,has_draft,prefill_only,reserve,warmup_fails,expected_rewarm",
    [
        ("DFLASH", True, False, True, False, True),
        ("DFLASH", True, False, True, True, True),
        ("DFLASH", True, True, True, False, False),
        ("EAGLE3", True, False, True, False, False),
        (None, False, False, True, False, False),
        ("DFLASH", True, False, False, False, False),
    ],
)
def test_dflash_rebound_stream_is_eagerly_warmed_before_final_capture(
    monkeypatch,
    algorithm,
    has_draft,
    prefill_only,
    reserve,
    warmup_fails,
    expected_rewarm,
):
    """Drive real startup/rebind; substitute unavailable weights/cache/device work."""
    from tokenspeed.runtime.engine import scheduler_utils
    from tokenspeed.runtime.execution import cudagraph_memory, device, factory
    from tokenspeed.runtime.layers.attention import registry

    events = []

    class FinalCapture(Exception):
        pass

    class WarmupFailure(Exception):
        pass

    class Executor:
        attn_backend = draft_attn_backend = None

        def __init__(self):
            self.config = SimpleNamespace(prefill_only=prefill_only)
            self.forward_step = self
            self.generation = 0

        def warmup_decode_path(self, *, batch_sizes, graph_phase):
            events.append(("warmup", self.generation, batch_sizes, graph_phase))
            if self.generation:
                assert not torch.is_grad_enabled()
            if self.generation and warmup_fails:
                raise WarmupFailure("rebound initialization failed")

        def autotune(self):
            if has_draft and not prefill_only:
                self.warmup_decode_path(batch_sizes=(1,), graph_phase=False)

        def release_graphs(self):
            events.append(("release", self.generation))

        def set_cache_pool(self, target, draft):
            self.generation += 1
            events.append(("rebind", self.generation, target, draft))

        def capture_graphs(self, *, entries, observer):
            events.append(("capture", self.generation, entries))
            if entries is None:
                raise FinalCapture

    model = SimpleNamespace(
        model=SimpleNamespace(),
        prepare_multimodal_runtime=lambda: None,
        prepare_communication_runtime=lambda count: None,
    )
    pool_generation = [0]

    def build(*args, **kwargs):
        pool_generation[0] += 1
        return SimpleNamespace(
            attn_backend=None,
            draft_attn_backend=None,
            token_to_kv_pool=f"target{pool_generation[0]}",
            draft_token_to_kv_pool=f"draft{pool_generation[0]}",
            profiled_cache_bytes=4096,
        )

    def probe(executor, args, gpu_id, *, startup_resident_bytes):
        executor.capture_graphs(entries=5, observer=None)
        return 4096

    monkeypatch.setattr(
        factory,
        "create_model_runner",
        lambda *args: (model, model if has_draft else None),
    )
    monkeypatch.setattr(factory, "create_model_executor", lambda **kwargs: Executor())
    monkeypatch.setattr(
        factory,
        "ModelExecutorConfig",
        SimpleNamespace(from_server_args=lambda **kwargs: None),
    )
    monkeypatch.setattr(registry, "create_attn_components", build)
    monkeypatch.setattr(cudagraph_memory, "probe_cudagraph_memory", probe)
    monkeypatch.setattr(device, "probe_arena_floor", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        device,
        "current_platform",
        lambda: SimpleNamespace(
            is_nvidia=False, is_cdna4=False, restore_stack_limit=nullcontext
        ),
    )
    monkeypatch.setattr(
        scheduler_utils,
        "scheduler_cache_geometry_from_pool",
        lambda pool: SimpleNamespace(prefix_granularity=1),
    )
    monkeypatch.setattr(scheduler_utils, "pool_to_cache_groups", lambda pool: [])
    args = SimpleNamespace(
        disaggregation_mode="null",
        chunked_prefill_size=8192,
        attention_backend=None,
        drafter_attention_backend=None,
        disable_cudagraph_memory_reserve=not reserve,
        enforce_eager=False,
        enable_memory_saver=False,
        enable_external_encoder=False,
        device="cuda",
        speculative_algorithm=algorithm,
    )
    failure = WarmupFailure if warmup_fails else FinalCapture
    with pytest.raises(failure):
        device.build_device_side(
            server_args=args,
            model_config=SimpleNamespace(context_len=4096, is_multimodal_active=False),
            draft_model_config=None,
            gpu_id=0,
            global_rank=0,
            attn_tp_rank=0,
            min_per_gpu_mem=0,
            overlap_schedule_depth=0,
            decode_input_tokens=16,
            max_batch_size=16,
            kv_transfer_builder=functools.partial(build_host_peer, outbox=None),
        )
    rebound_warmup = ("warmup", 1, (1,), False)
    assert (rebound_warmup in events) == expected_rewarm
    if reserve:
        probe_index = events.index(("capture", 0, 5))
        rebind_index = events.index(("rebind", 1, "target2", "draft2"))
        assert probe_index < rebind_index
        if has_draft and not prefill_only:
            assert events.index(("warmup", 0, (1,), False)) < probe_index
        if expected_rewarm:
            assert rebind_index < events.index(rebound_warmup)
        if warmup_fails:
            assert ("capture", 1, None) not in events
        else:
            assert events[-1] == ("capture", 1, None)
    else:
        assert events[-1] == ("capture", 0, None)


def _peer(trace=None):
    """A P-role transfer peer the handle will recognize by type."""

    class _Peer:
        mode = "prefill"

        def execute(self, op) -> None:
            if trace is not None:
                trace.append(("send", op))

    return _Peer()


def _handle(trace, **kwargs):
    return DeviceHandle(
        SimpleNamespace(
            device="cpu",
            execute_forward_op=lambda *a, **k: trace.append("forward"),
            complete_pending_drafting=lambda: None,
            execution_stream="execution-stream",
            default_stream="default-stream",
            write_remote_spec_candidate_ids=lambda idx, ids: trace.append(
                ("candidates", idx, list(ids))
            ),
            mark_remote_cache_ready=lambda slot: trace.append(("ready", slot)),
        ),
        **kwargs,
    )


def _plan(
    *,
    pages_to_zero=(),
    pages_to_copy=None,
    cache=(),
    remote_decode=None,
    remote_prefill=None,
):
    """A round's ExecutionPlan as ``execute`` reads it — every field the real
    plan exposes, so a fake cannot hide a renamed one. ``pages_to_zero`` is
    the real plan's shape: every cache group listed, empty or not, with the
    given pages under ``"kv"``. ``pages_to_copy`` is the
    ``pages_to_copy_arrays()`` view: ``[n, 2]`` arrays per group id."""
    copies = {
        group_id: np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        for group_id, pairs in (pages_to_copy or {}).items()
    }
    return SimpleNamespace(
        pages_to_zero={"kv": list(pages_to_zero), "state": []},
        pages_to_copy_arrays=lambda: copies,
        cache=list(cache),
        remote_decode=remote_decode,
        remote_prefill=remote_prefill,
    )


class _WriteBackOp:
    """The wire face of ``Cache.WriteBackOp`` (no Python constructor), passed
    through to the executor as one batch."""

    def __init__(self, label) -> None:
        self.label = label
        self.op_ids = [1]

    def __repr__(self) -> str:
        return f"WriteBackOp({self.label!r})"


class _LoadBackOp(_WriteBackOp):
    def __repr__(self) -> str:
        return f"LoadBackOp({self.label!r})"


class _SnapshotBatch:
    """The wire face of ``Cache.SnapshotOp``: ``rows`` is
    ``[(op_id, request_pool_index, snapshot_slot)]``, one per victim, each with
    one tail row of group 0."""

    def __init__(self, rows) -> None:
        self.op_ids = [op_id for op_id, _, _ in rows]
        self.request_ids = [f"r{op_id}" for op_id, _, _ in rows]
        self.request_pool_indices = [slot for _, slot, _ in rows]
        self.snapshot_slots = [blob for _, _, blob in rows]
        self.group_ids = [[0] for _ in rows]
        self.src_pages = [[2] for _ in rows]
        self.dst_pages = [[3] for _ in rows]


class _RestoreBatch:
    """The wire face of ``Cache.RestoreOp``; not a ``_SnapshotBatch`` subclass,
    as the adapter dispatches on the type."""

    def __init__(self, rows) -> None:
        store = _SnapshotBatch(rows)
        self.op_ids = store.op_ids
        self.request_ids = store.request_ids
        self.request_pool_indices = store.request_pool_indices
        self.snapshot_slots = store.snapshot_slots
        self.group_ids = store.group_ids
        self.src_pages = store.src_pages
        self.dst_pages = store.dst_pages
        self.content_hashes = [[""] for _ in rows]
        self.page_offsets = [[0] for _ in rows]
        self.source_tiers = [[int(HostTier.SNAPSHOT_POOL)] for _ in rows]


@pytest.fixture()
def wire_ops(monkeypatch):
    """Dispatch the plan adapter on the fakes above instead of the bindings."""
    monkeypatch.setattr(
        scheduler_utils_module,
        "Cache",
        SimpleNamespace(
            WriteBackOp=_WriteBackOp,
            LoadBackOp=_LoadBackOp,
            SnapshotOp=_SnapshotBatch,
            RestoreOp=_RestoreBatch,
            StorageQueryOp=scheduler_utils_module.Cache.StorageQueryOp,
            StorageGetOp=scheduler_utils_module.Cache.StorageGetOp,
            StoragePutOp=scheduler_utils_module.Cache.StoragePutOp,
            StorageCancelOp=scheduler_utils_module.Cache.StorageCancelOp,
        ),
    )


def _planned(*, num_extends, label=None):
    """A round whose op carries only what routing reads."""
    return PlannedForward(
        forward_op=SimpleNamespace(
            label=label,
            request_ids=["a"],
            request_pool_indices=[0],
            num_extends=lambda: num_extends,
        ),
        sampling_params_list=[],
        dp_metadata=None,
        grammar_inputs=None,
        multimodal_context=None,
        ngram_inputs=None,
        request_history_seeds=None,
        input_logprob_plan=None,
        sampled_top_k=0,
        stop_criteria=(),
    )


def _profile_batch(mode):
    """Real immutable scheduler batches, including a resumed prefill chunk."""
    config = SchedulerConfig()
    config.prefix_granularity = 16
    config.max_scheduled_tokens = 8
    config.max_batch_size = 4
    config.num_device_pages = 64
    config.num_snapshot_pages = 1
    config.enable_mixed_prefill_decode = True
    config.cache_groups = [
        CacheGroupConfig(
            group_id="full_attention",
            block_granularity=16,
            total_pages=64,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(config)
    request = RequestSpec()
    request.request_id = 'old-"雪'
    request.tokens = list(range(12 if mode == ForwardMode.EXTEND else 3))
    scheduler.submit_requests([request])
    scheduler.next_execution_plan()
    batch = scheduler.next_execution_plan().forward[0]
    if mode != ForwardMode.MIXED:
        return batch
    event = ForwardEvent.ExtendResult()
    event.request_id = request.request_id
    event.tokens = [99]
    scheduler.advance(ExecutionEvent().add_event(event))
    request = RequestSpec()
    request.request_id = "new-\\request"
    request.tokens = list(range(20))
    scheduler.submit_requests([request])
    return scheduler.next_execution_plan().forward[0]


@pytest.mark.parametrize(
    "mode", [ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.MIXED]
)
def test_forward_profile_range_preserves_real_batch_metadata_and_result(mode):
    batch = _profile_batch(mode)
    result = object()

    def cpu_forward(op, sampling_params, **kwargs):
        assert op is batch
        assert kwargs["capture_next_input_ids"]
        with torch.profiler.record_function("cpu_model_work"):
            torch.ones(2, device="cpu").add_(1)
        return result

    handle = DeviceHandle(SimpleNamespace(device="cpu", execute_forward_op=cpu_forward))
    planned = replace(_planned(num_extends=batch.num_extends()), forward_op=batch)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        assert handle._launch_forward(planned, capture_next_input_ids=True) is result
    ranges = [
        event
        for event in profiler.events()
        if event.name.startswith("tokenspeed.forward.v1:")
    ]
    assert len(ranges) == 1
    metadata = json.loads(ranges[0].name.removeprefix("tokenspeed.forward.v1:"))
    assert metadata == {
        "mode": mode.name,
        "num_extends": batch.num_extends(),
        "request_ids": batch.request_ids,
        "input_lengths": batch.input_lengths,
        "prefill_lengths": batch.prefill_lengths,
        "extend_prefix_lens": batch.extend_prefix_lens,
    }
    if mode == ForwardMode.EXTEND:
        assert metadata["extend_prefix_lens"] == [8]
        assert metadata["input_lengths"] == [4]
        assert metadata["prefill_lengths"] == [12]
    elif mode == ForwardMode.DECODE:
        assert metadata["extend_prefix_lens"] == []
        assert metadata["input_lengths"] == [1]
    else:
        assert metadata["request_ids"] == ["new-\\request", 'old-"雪']
        assert metadata["input_lengths"] == [7, 1]
    inner = next(event for event in profiler.events() if event.name == "cpu_model_work")
    assert inner.thread == ranges[0].thread
    assert inner.time_range.start >= ranges[0].time_range.start
    assert inner.time_range.end <= ranges[0].time_range.end


def test_profile_range_closes_and_propagates_original_forward_exception():
    batch = _profile_batch(ForwardMode.EXTEND)
    failure = RuntimeError("original forward failure")

    def cpu_forward(*args, **kwargs):
        raise failure

    handle = DeviceHandle(SimpleNamespace(device="cpu", execute_forward_op=cpu_forward))
    planned = replace(_planned(num_extends=1), forward_op=batch)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        with pytest.raises(RuntimeError) as raised:
            handle._launch_forward(planned)
        assert raised.value is failure
        with torch.profiler.record_function("after_failed_forward"):
            pass
    phase = next(
        event
        for event in profiler.events()
        if event.name.startswith("tokenspeed.forward.v1:")
    )
    after = next(
        event for event in profiler.events() if event.name == "after_failed_forward"
    )
    assert phase.time_range.end <= after.time_range.start


@pytest.mark.parametrize("missing_state", [False, True])
def test_unprofiled_forward_never_reads_profile_metadata(monkeypatch, missing_state):
    # This ordinary test op intentionally has no profile metadata fields.
    # Missing internal profiler support must not break nonprofiled serving.
    if missing_state:
        monkeypatch.delattr(torch.autograd.profiler, "_is_profiler_enabled")
    else:
        assert not torch.autograd.profiler._is_profiler_enabled
    trace = []
    handle = _handle(trace)
    handle._launch_forward(_planned(num_extends=0))
    assert trace == ["forward"]


# ----------------------------------------------------------------------
# PD completion: device writes belong to the data plane, in order.
# ----------------------------------------------------------------------


def test_remote_prefill_landing_writes_both_on_the_device():
    trace: list = []
    _handle(trace).run_remote_prefill_landing((7, [11, 12]), 7)
    # The candidates precede the readiness arm: hydration reads the row the
    # candidates were just written into.
    assert trace == [("candidates", 7, [11, 12]), ("ready", 7)]


def test_a_landing_without_device_work_touches_nothing():
    trace: list = []
    _handle(trace).run_remote_prefill_landing(None, None)
    assert trace == []


def test_an_aborted_request_lands_its_candidates_without_arming():
    # The control plane passes no slot for a request aborted while its KV
    # moved: the candidates still land, the slot is not armed.
    trace: list = []
    _handle(trace).run_remote_prefill_landing((7, [11, 12]), None)
    assert trace == [("candidates", 7, [11, 12])]


# ----------------------------------------------------------------------
# L2 cache plans: write-backs launch AHEAD of the zeroing (a stream-ordered
# one must read the reused pages' old bytes, and fences the caller's stream
# before the zeroing is enqueued), load-backs behind it.
# ----------------------------------------------------------------------


def test_one_plan_orders_write_backs_zeroing_then_load_backs(wire_ops):
    """The plan's order is the correctness order for same-round page reuse:
    the write-backs are submitted ahead of the new owner's zeroing (a
    stream-ordered snapshot copy lands its fence on the default stream the
    zeroing runs on), and the load-backs target zeroed pages. Every stream is
    named: the copies are told which stream wrote the pages and which one
    the fence lands on; the loads are told which stream zeroed their
    destinations. The executor sees the plan's cache ops once, adapted, and
    the same tuple on both submissions."""
    trace: list = []
    store, load = _WriteBackOp("store"), _LoadBackOp("load")
    plan = _plan(pages_to_zero=[3, 4], cache=[store, load])
    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            submit_write_backs=lambda ops, *, prerequisite_stream, fence_stream: (
                trace.append(("write_backs", ops, prerequisite_stream, fence_stream))
            ),
            submit_load_backs=lambda ops, *, prerequisite_stream: trace.append(
                ("load_backs", ops, prerequisite_stream)
            ),
            poll_results=lambda: ["done"],
        ),
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append(
        ("zero", {group_id: tuple(ids) for group_id, ids in pages.items()})
    )

    _execute(handle, plan, None)

    assert trace == [
        ("write_backs", (store, load), "execution-stream", "default-stream"),
        ("zero", {"kv": (3, 4)}),
        ("load_backs", (store, load), "default-stream"),
    ]
    # Polling only reads the acknowledgments: it submits nothing.
    assert handle.poll_cache_results() == ["done"]
    assert len(trace) == 3


def test_page_copies_follow_the_stores_and_precede_the_zeroing(wire_ops):
    """A prefix hit's private copy (or an owner's move off its published
    block) goes after the stores' fence, before the zeroing -- its source may
    be re-granted and zeroed by this very plan -- and only when some group
    has pairs. Groups without pairs never reach the executor."""
    trace: list = []
    store = _WriteBackOp("store")
    plan = _plan(
        pages_to_zero=[3],
        pages_to_copy={"kv": [(3, 8), (5, 9)], "state": []},
        cache=[store],
    )
    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            submit_write_backs=lambda ops, *, prerequisite_stream, fence_stream: (
                trace.append(("write_backs", ops))
            ),
            submit_load_backs=lambda ops, *, prerequisite_stream: trace.append(
                ("load_backs", ops)
            ),
        ),
    )
    handle._executor.copy_cache_pages = lambda pairs: trace.append(
        ("copy", {gid: arr.tolist() for gid, arr in pairs.items()})
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append(
        ("zero", {group_id: tuple(ids) for group_id, ids in pages.items()})
    )

    _execute(handle, plan, None)

    assert trace == [
        ("write_backs", (store,)),
        ("copy", {"kv": [[3, 8], [5, 9]]}),
        ("zero", {"kv": (3,)}),
        ("load_backs", (store,)),
    ]

    # A plan with no pairs runs no copy at all.
    trace.clear()
    _execute(handle, _plan(pages_to_zero=[2], pages_to_copy={"kv": []}), None)
    assert trace == [("zero", {"kv": (2,)}), ("load_backs", ())]


def test_a_failed_page_copy_fails_the_plan(wire_ops):
    """The copy has no completion event, so its failure raises from the
    plan's ``execute`` itself, before the zeroing and the forward behind it
    run."""
    trace: list = []
    handle = _handle(trace)

    def failing_copy(pairs):
        raise IndexError("local block ID 99 outside [0, 8)")

    handle._executor.copy_cache_pages = failing_copy
    handle._executor.zero_cache_pages = lambda pages: trace.append("zero")
    with pytest.raises(IndexError, match="99"):
        _execute(
            handle,
            _plan(pages_to_zero=[3], pages_to_copy={"kv": [(1, 99)]}),
            _planned(num_extends=0),
        )
    assert trace == []


def test_snapshot_stores_fence_before_zeroing_and_restores_follow_it(wire_ops):
    """A retraction's image rides ``execute`` with no operation of its own:
    the snapshot store goes to ``submit_write_backs`` (whose stream-ordered
    lane fences the default stream the zeroing runs on) ahead of the
    zeroing, the restore to ``submit_load_backs`` behind it. The binding's
    batches arrive at the executor as one per-request op each -- the unit it
    exports/imports a slot for -- carrying the victim's slot on the store and
    the new slot on the restore."""
    trace: list = []
    plan = _plan(
        pages_to_zero=[9],
        cache=[
            _WriteBackOp("l2-leg"),
            _SnapshotBatch([(7, 3, 0), (8, 5, 1)]),
            _RestoreBatch([(9, 4, 2)]),
        ],
    )

    def submit_write_backs(ops, *, prerequisite_stream, fence_stream):
        trace.append(("write_backs", ops, fence_stream))

    def submit_load_backs(ops, *, prerequisite_stream):
        trace.append(("load_backs", ops, prerequisite_stream))

    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            submit_write_backs=submit_write_backs,
            submit_load_backs=submit_load_backs,
        ),
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append(
        ("zero", {group_id: tuple(ids) for group_id, ids in pages.items()})
    )

    _execute(handle, plan, None)

    kinds = [entry[0] for entry in trace]
    assert kinds == ["write_backs", "zero", "load_backs"]
    # Stores precede the zeroing, and the fence they record lands on the
    # stream the zeroing runs on.
    stores = trace[0][1]
    assert trace[0][2] == "default-stream"
    assert [type(op).__name__ for op in stores] == [
        "_WriteBackOp",
        "SnapshotOp",
        "SnapshotOp",
        "RestoreOp",
    ]
    victims = [op for op in stores if isinstance(op, SnapshotOp)]
    assert [(op.op_id, op.request_pool_index, op.snapshot_slot) for op in victims] == [
        (7, 3, 0),
        (8, 5, 1),
    ]
    # The restore is behind the zeroing and names the resumed request's new
    # slot; it orders after the default stream, where the zeroing ran.
    restores = [op for op in trace[-1][1] if isinstance(op, RestoreOp)]
    assert [(op.op_id, op.request_pool_index, op.snapshot_slot) for op in restores] == [
        (9, 4, 2)
    ]
    assert trace[-1][2] == "default-stream"
    assert kinds.index("zero") < kinds.index("load_backs")


def test_a_victims_slot_regranted_in_the_same_plan_is_read_behind_the_fence(
    wire_ops,
):
    """The store's ``request_pool_index`` is the victim's slot, which the
    scheduler frees in the same plan build; its next owner writes it in a
    forward. The handle enqueues every forward behind the store submission,
    whose fence the forward's prologue inherits, so the export reads the
    victim's bytes even when this very plan's batch reuses the slot."""
    trace: list = []
    plan = _plan(pages_to_zero=[2], cache=[_SnapshotBatch([(1, 6, 0)])])
    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            submit_write_backs=lambda ops, *, prerequisite_stream, fence_stream: (
                trace.append(("store", [op.request_pool_index for op in ops]))
            ),
            submit_load_backs=lambda ops, *, prerequisite_stream: (
                trace.append("load_backs")
            ),
        ),
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append("zero")
    handle._executor.execute_forward_op = lambda op, *a, **k: trace.append(
        ("forward", list(op.request_pool_indices))
    )
    planned = _planned(num_extends=1)
    planned.forward_op.request_pool_indices = [6]  # the slot just freed

    _execute(handle, plan, planned)

    assert trace == [("store", [6]), "zero", "load_backs", ("forward", [6])]


def test_page_zeroing_without_l2_runs_only_zeroing():
    trace: list = []
    handle = _handle(trace)
    handle._executor.zero_cache_pages = lambda pages: trace.append(
        ("zero", {group_id: tuple(ids) for group_id, ids in pages.items()})
    )

    _execute(handle, _plan(pages_to_zero=[3, 4]), None)

    assert trace == [("zero", {"kv": (3, 4)})]


def test_a_plan_with_no_device_work_only_points_the_load_waits_at_no_load():
    """A plan with no device work enqueues nothing, and it settles no pending
    draft round; it still reaches the Host cache's load-back submission,
    which points the layerwise load waits at the plan's (absent) load."""
    trace: list = []
    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            submit_load_backs=lambda ops, *, prerequisite_stream: trace.append(
                ("load_backs", ops, prerequisite_stream)
            ),
        ),
    )
    handle._executor.complete_pending_drafting = lambda: trace.append("settle")

    _execute(handle, _plan(), None)

    assert trace == [("load_backs", (), "default-stream")]


def test_a_failed_cache_submission_fails_the_plan(wire_ops):
    """A submission that raised produces no completion acks; it raises from
    the plan's ``execute``, so its ops never sit counted in flight."""
    trace: list = []

    def exploding(ops, *, prerequisite_stream, fence_stream):
        raise ValueError("bad cache op")

    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            submit_write_backs=exploding,
            submit_load_backs=lambda ops, *, prerequisite_stream: trace.append(
                "load_backs"
            ),
        ),
    )

    with pytest.raises(ValueError, match="bad cache op"):
        _execute(handle, _plan(cache=[_WriteBackOp("op")]), None)
    assert trace == []


def test_shutdown_cache_closes_the_host_cache():
    trace: list = []
    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            shutdown=lambda: trace.append("l2_shutdown"),
        ),
    )

    handle.shutdown_cache()
    # Without a Host cache there is nothing to close.
    _handle(trace).shutdown_cache()

    assert trace == ["l2_shutdown"]


def test_cache_polling_without_kvstore_refuses_loudly():
    with pytest.raises(RuntimeError, match="enable-kvstore"):
        _handle([]).poll_cache_results()


# ----------------------------------------------------------------------
# The transfer peer: a ForwardBatch the model does not run.
# ----------------------------------------------------------------------


def test_an_unrecognized_transfer_peer_is_refused():
    """The role is read off the peer at construction, so a peer of the wrong
    type is a startup error, not a surprise at the first transfer."""
    with pytest.raises(TypeError, match="transfer peer"):
        _handle([], kv_transfer=SimpleNamespace())


def test_the_remote_decode_keeps_the_forward_order():
    """The send walks KV-pool device memory the forwards wrote: it follows
    them in the forward thread's order, not by the loop's scheduling
    rules."""
    trace: list = []
    handle = _handle(trace, kv_transfer=_peer(trace))

    chunk = _planned(num_extends=1, label="CHUNK")
    remote_decode = SimpleNamespace(request_ids=["done"])

    _execute(handle, _plan(), chunk)
    _execute(handle, _plan(remote_decode=remote_decode), None)

    # The send follows the forwards whose KV it reads (the scheduler emits a
    # remote decode only after its final chunk's result landed), handed over
    # without waiting, like the forwards themselves.
    assert trace == ["forward", ("send", remote_decode)]


# ----------------------------------------------------------------------
# EPD receive: the encoder facts a multimodal prefill node reads.
# ----------------------------------------------------------------------


class _DiscreteEncoderModel:
    """A prefill model can declare discrete output without loading a vision tower."""

    def __init__(self, output_width: int) -> None:
        self.config = PretrainedConfig(hidden_size=64)
        self.output_width = output_width

    def get_encoder_output_spec(self) -> tuple[int, torch.dtype]:
        return self.output_width, torch.int64


class _VisionEncoderModel:
    def __init__(self, vision_tower: torch.nn.Module) -> None:
        self.config = PretrainedConfig(hidden_size=64)
        self.vision_tower = vision_tower
        self.num_deepstack_embeddings = 3


def test_encoder_facts_preserve_discrete_output_spec_without_a_vision_tower():
    facts = EncoderModelFacts.from_model(_DiscreteEncoderModel(8), device="cpu")

    assert facts == EncoderModelFacts(
        device="cpu", hidden=8, num_deepstack=0, dtype=torch.int64
    )


@pytest.mark.parametrize("output_width", [0, -1])
def test_encoder_facts_reject_nonpositive_output_width(output_width):
    with pytest.raises(ValueError, match="Encoder output width must be positive"):
        EncoderModelFacts.from_model(_DiscreteEncoderModel(output_width), device="cpu")


def test_encoder_facts_fall_back_to_vision_dtype():
    class VisionTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dtype = torch.bfloat16

    facts = EncoderModelFacts.from_model(
        _VisionEncoderModel(VisionTower()), device="cpu"
    )

    assert facts == EncoderModelFacts(
        device="cpu", hidden=64, num_deepstack=3, dtype=torch.bfloat16
    )


def test_encoder_facts_fall_back_to_vision_parameter_dtype():
    vision = torch.nn.Linear(2, 4, bias=False, device="cpu", dtype=torch.float32)
    facts = EncoderModelFacts.from_model(_VisionEncoderModel(vision), device="cpu")

    assert facts == EncoderModelFacts(
        device="cpu", hidden=64, num_deepstack=3, dtype=torch.float32
    )


# ----------------------------------------------------------------------
# Multimodal gather: the forward gets a snapshot, not the live struct.
# ----------------------------------------------------------------------


def _mm_inputs(positions=None, delta=None):
    return MultimodalInputs(
        mm_items=[MultimodalDataItem(modality=Modality.IMAGE, hash=1, pad_value=1)],
        mrope_positions=positions,
        mrope_position_delta=delta,
    )


def _forward_op(num_extends=1):
    return SimpleNamespace(
        request_ids=["r0"],
        num_extends=lambda: num_extends,
        extend_prefix_lens=[0],
        input_lengths=[4],
    )


def test_gathered_context_does_not_see_later_control_plane_edits():
    positions = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    mm = _mm_inputs(positions=positions)
    state = SimpleNamespace(
        multimodal_inputs=mm,
    )

    ctx = multimodal_context_for_forward(
        _forward_op(), {"r0": state}, max_encoder_tokens=1234
    )

    # The next round's gather extends the live struct's table; a forward
    # already dispatched with the previous context must not observe it.
    mm.mrope_positions = torch.zeros(3, 8, dtype=torch.int64)
    assert ctx.mm_inputs[0] is not mm
    assert torch.equal(ctx.mm_inputs[0].mrope_positions, positions)
    assert ctx.max_encoder_tokens == 1234


def test_gather_resolves_the_decode_delta_on_the_live_struct():
    mm = _mm_inputs(delta=torch.tensor([[5]], dtype=torch.int64))
    state = SimpleNamespace(
        multimodal_inputs=mm,
    )

    ctx = multimodal_context_for_forward(
        _forward_op(num_extends=0), {"r0": state}, max_encoder_tokens=8192
    )

    # Resolved on the control plane, so the forward only reads — and resolved
    # on the LIVE struct, so the next round's snapshot inherits it instead of
    # paying the item() again.
    assert mm.mrope_position_delta_scalar == 5
    assert ctx.mm_inputs[0].mrope_position_delta_scalar == 5


# ----------------------------------------------------------------------
# The forward thread's device; the dispatch guard: device work on the
# control thread raises.
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs a second GPU to tell devices apart"
)
def test_the_forward_thread_takes_the_device_the_handle_was_built_on():
    """Torch starts every new thread on device 0; the handle resolves an
    index-less device on the building thread and gives it to the forward
    thread."""
    seen = []

    def forward_thread(handle):
        seen.append(torch.cuda.current_device())
        handle.bind_forward_thread()
        seen.append(torch.cuda.current_device())

    previous = torch.cuda.current_device()
    torch.cuda.set_device(1)
    try:
        handle = DeviceHandle(SimpleNamespace(device="cuda"))
    finally:
        torch.cuda.set_device(previous)
    thread = threading.Thread(target=forward_thread, args=(handle,))
    thread.start()
    thread.join()
    assert seen == [0, 1]


def test_the_guard_rejects_cuda_factories_before_they_run():
    """Caught at dispatch, so this holds even on a CUDA-less machine."""
    from tokenspeed.runtime.execution.device import _NoDeviceWork

    with _NoDeviceWork():
        cpu = torch.zeros(3)
        cpu += 1  # CPU work passes
        with pytest.raises(RuntimeError, match="Principle 1"):
            torch.empty(2, device="cuda")


def test_the_guard_scans_tensor_lists():
    from tokenspeed.runtime.execution.device import _NoDeviceWork

    with _NoDeviceWork():
        # aten::cat takes a List[Tensor]; the scan must descend into it.
        out = torch.cat([torch.ones(2), torch.ones(2)])
        assert out.numel() == 4


def test_the_guard_is_on_by_default(monkeypatch):
    import contextlib

    from tokenspeed.runtime.execution.device import (
        _NoDeviceWork,
        maybe_control_plane_guard,
    )

    monkeypatch.delenv("TOKENSPEED_GUARD_CONTROL_PLANE", raising=False)
    assert isinstance(maybe_control_plane_guard(), _NoDeviceWork)
    monkeypatch.setenv("TOKENSPEED_GUARD_CONTROL_PLANE", "0")
    assert isinstance(maybe_control_plane_guard(), contextlib.nullcontext)


def test_metadata_only_ops_are_recognized_by_schema():
    """Views alias without writing; kernels do not. The rule must separate
    them without an op-by-op allowlist."""
    from tokenspeed.runtime.execution.device import _only_aliases_inputs

    assert _only_aliases_inputs(torch.ops.aten.view.default)
    assert _only_aliases_inputs(torch.ops.aten.slice.Tensor)
    # Data-producing and in-place ops both stay banned.
    assert not _only_aliases_inputs(torch.ops.aten.add.Tensor)
    assert not _only_aliases_inputs(torch.ops.aten.clone.default)
    assert not _only_aliases_inputs(torch.ops.aten.copy_.default)


def test_expert_load_reset_and_dump_ride_the_execution_stream(tmp_path):
    """The load counters are zeroed and read back on the execution stream the
    routing kernels bump them on, and the dump is this rank's record alone:
    no collective, so a profile stop that reaches attention-DP workers
    independently cannot deadlock."""
    import contextlib

    from tokenspeed.runtime.moe import expert_location

    placement = expert_location.ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([[0, 1, 2, 0], [2, 1, 0, 1]]),
        3,
        ep_size=2,
        ep_rank=1,
        ep_rank_nodes=(0, 0),
    )
    placement.enable_load_recording(Mapping(rank=1, world_size=2, moe_ep_size=2))
    placement.physical_load.copy_(torch.tensor([[3, 1, 2, 5], [0, 4, 0, 4]]))
    streams: list = []

    class _DeviceModule:
        @staticmethod
        def stream(stream):
            streams.append(stream)
            return contextlib.nullcontext()

    handle = DeviceHandle(
        SimpleNamespace(
            device="cpu",
            device_module=_DeviceModule(),
            execution_stream="execution-stream",
            complete_pending_drafting=lambda: None,
        )
    )
    expert_location.set_global_expert_location_metadata(placement)
    try:
        with mock.patch.object(
            torch.distributed, "all_reduce", side_effect=AssertionError("collective")
        ):
            record = handle.dump_expert_load(str(tmp_path / "load.pt"))
        handle.reset_expert_load()
    finally:
        expert_location.set_global_expert_location_metadata(None)
    assert streams == ["execution-stream", "execution-stream"]
    saved = torch.load(tmp_path / "load.pt", weights_only=True)
    assert torch.equal(saved["physical_count"], record["physical_count"])
    assert record["physical_count"].tolist() == [[3, 1, 2, 5], [0, 4, 0, 4]]
    # Logical 0 owns physical 0 and 3 in layer 0, physical 2 in layer 1.
    assert record["logical_count"].tolist() == [[8, 1, 2], [0, 8, 0]]
    assert record["ep_rank"] == 1 and record["ep_size"] == 2
    assert not placement.physical_load.any()
    with pytest.raises(RuntimeError, match="not being recorded"):
        handle.dump_expert_load(str(tmp_path / "none.pt"))


def test_expert_load_counters_start_serving_at_zero_after_the_startup_forwards():
    """Autotune, warm-up and capture route tokens through the MoE layers; the
    builder zeroes the counters after capture (device drained, reset on the
    execution stream) and before the handle exists, so the first window a
    rebalance or profile sees is traffic only."""
    import contextlib

    from tokenspeed.runtime.execution.device import start_expert_load_window
    from tokenspeed.runtime.moe import expert_location

    placement = expert_location.ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([[0, 1, 2, 0]]), 3, ep_size=2, ep_rank=0, ep_rank_nodes=(0, 0)
    )
    placement.enable_load_recording(Mapping(rank=0, world_size=2, moe_ep_size=2))
    placement.physical_load.copy_(torch.tensor([[3, 1, 2, 5]]))  # startup routes
    trace: list = []

    class _DeviceModule:
        @staticmethod
        def synchronize():
            trace.append("synchronize")

        @staticmethod
        def stream(stream):
            trace.append(("stream", stream))
            return contextlib.nullcontext()

    executor = SimpleNamespace(
        device_module=_DeviceModule(), execution_stream="execution-stream"
    )
    expert_location.set_global_expert_location_metadata(placement)
    try:
        start_expert_load_window(executor)
    finally:
        expert_location.set_global_expert_location_metadata(None)
    assert trace == ["synchronize", ("stream", "execution-stream")]
    assert not placement.physical_load.any()
    # Without load recording there is nothing to reset and nothing is touched.
    start_expert_load_window(SimpleNamespace())


def test_expert_load_snapshot_reads_then_zeroes_on_the_execution_stream():
    """The online rebalance's window boundary: one operation that copies the
    counters to the host and zeroes them on the execution stream, with no
    collective (the gate reduces the result afterwards)."""
    import contextlib

    from tokenspeed.runtime.moe import expert_location

    placement = expert_location.ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([[0, 1, 2, 0], [2, 1, 0, 1]]),
        3,
        ep_size=2,
        ep_rank=1,
        ep_rank_nodes=(0, 0),
    )
    placement.enable_load_recording(Mapping(rank=1, world_size=2, moe_ep_size=2))
    placement.physical_load.copy_(torch.tensor([[3, 1, 2, 5], [0, 4, 0, 4]]))
    streams: list = []

    class _DeviceModule:
        @staticmethod
        def stream(stream):
            streams.append(stream)
            return contextlib.nullcontext()

    handle = DeviceHandle(
        SimpleNamespace(
            device="cpu",
            device_module=_DeviceModule(),
            execution_stream="execution-stream",
            complete_pending_drafting=lambda: None,
        )
    )
    expert_location.set_global_expert_location_metadata(placement)
    try:
        with mock.patch.object(
            torch.distributed, "all_reduce", side_effect=AssertionError("collective")
        ):
            snapshot = handle.snapshot_expert_load()
    finally:
        expert_location.set_global_expert_location_metadata(None)
    assert streams == ["execution-stream"]
    assert snapshot.physical_count.tolist() == [[3, 1, 2, 5], [0, 4, 0, 4]]
    assert snapshot.logical_count.tolist() == [[8, 1, 2], [0, 8, 0]]
    assert torch.equal(
        snapshot.physical_to_logical_map, placement.physical_to_logical_map_cpu
    )
    assert not placement.physical_load.any()


def test_expert_placement_chunk_applies_then_synchronizes_the_execution_stream():
    """A chunk is one blocking operation: the updater moves the weights and
    switches the tables inside the execution stream, and the stream is
    synchronized before the call returns so a P2P failure surfaces in the
    op's reply. Without --enable-eplb there is no updater to run."""
    import contextlib

    from tokenspeed.runtime.moe.expert_rebalance import SlotMoves

    trace: list = []

    class _Stream:
        def synchronize(self):
            trace.append("synchronize")

    class _DeviceModule:
        @staticmethod
        def stream(stream):
            trace.append(("stream", stream))
            return contextlib.nullcontext()

    class _Updater:
        def apply(self, layer_ids, new_rows, moves_by_layer):
            trace.append(("apply", tuple(layer_ids), new_rows.tolist(), moves_by_layer))

    stream = _Stream()
    moves = {0: SlotMoves(recv=(), send=(), local_copy=((1, 0),), free_rider=())}
    rows = torch.tensor([[0, 0, 1, 2]])
    handle = DeviceHandle(
        SimpleNamespace(
            device="cpu",
            device_module=_DeviceModule(),
            execution_stream=stream,
            model_runner=SimpleNamespace(expert_location_updater=_Updater()),
            complete_pending_drafting=lambda: None,
        )
    )
    handle.apply_expert_placement((0,), rows, moves)
    assert trace == [
        ("stream", stream),
        ("apply", (0,), [[0, 0, 1, 2]], moves),
        "synchronize",
    ]
    handle = DeviceHandle(
        SimpleNamespace(
            device="cpu",
            model_runner=SimpleNamespace(expert_location_updater=None),
            complete_pending_drafting=lambda: None,
        )
    )
    with pytest.raises(RuntimeError, match="--enable-eplb"):
        handle.apply_expert_placement((0,), rows, moves)


def test_device_rejects_storage_commands_before_submitting_any_work():
    from test.l3_test_utils import storage_ops, storage_scheduler

    plan = storage_scheduler().next_execution_plan()
    trace = []
    handle = _handle(trace)
    with pytest.raises(TypeError, match="unsupported Device cache op"):
        handle.execute(plan, None, cache_ops=storage_ops(plan))
    assert trace == []


def test_device_consumes_the_explicit_snapshot_without_reading_plan_cache():
    class Plan:
        def __init__(self):
            self.pages_to_zero = {}
            self.remote_decode = None
            self.remote_prefill = None
            self.reads = 0

        def pages_to_copy_arrays(self):
            return {}

        @property
        def cache(self):
            self.reads += 1
            if self.reads > 1:
                raise AssertionError("plan.cache was already captured")
            return ()

    plan = Plan()
    prepared = cache_ops_from_plan(plan)
    trace = []
    handle = _handle(trace)
    handle.execute(plan, None, cache_ops=prepared.device)
    assert plan.reads == 1 and trace == []


def test_every_task_but_a_forward_runs_behind_a_pending_draft_round():
    """A drafter may leave its round pending after a forward; the next forward
    finishes it itself, and every other forward-thread task finishes it
    first."""
    trace: list = []
    handle = _handle(trace)
    handle._executor.complete_pending_drafting = lambda: trace.append("settle")
    handle._executor.zero_cache_pages = lambda pages: trace.append("zero")
    handle._executor.execute_idle_forward = lambda dp: trace.append("idle")
    handle._executor.device_module = SimpleNamespace(
        synchronize=lambda: trace.append("synchronize")
    )

    _execute(handle, _plan(), _planned(num_extends=0))
    assert trace == ["forward"]
    trace.clear()
    _execute(handle, _plan(pages_to_zero=[3]), _planned(num_extends=0))
    assert trace == ["settle", "zero", "forward"]
    trace.clear()
    handle.run_idle_forward(None)
    handle.synchronize()
    handle.run_remote_prefill_landing(None, 4)
    assert trace == ["settle", "idle", "settle", "synchronize", "settle", ("ready", 4)]


# ----------------------------------------------------------------------
# Memory-saver release/wake: what surrounds the unmap, in order.
# ----------------------------------------------------------------------


class _SleepArena:
    def __init__(self, trace) -> None:
        self._trace = trace

    def synchronize(self) -> None:
        self._trace.append("arena.synchronize")


class _SleepPool:
    def __init__(self, trace) -> None:
        self._trace = trace
        self.arena = _SleepArena(trace)

    def clear_kv_buffers(self) -> None:
        self._trace.append("clear_kv_buffers")


class _SleepDeviceModule:
    def __init__(self, trace) -> None:
        self._trace = trace

    def synchronize(self) -> None:
        self._trace.append("device.synchronize")


class _SleepExecutor:
    """Only what the release/wake operations read off the executor."""

    def __init__(self, trace) -> None:
        self.device = "cpu"
        self.device_module = _SleepDeviceModule(trace)
        self.token_to_kv_pool = _SleepPool(trace)

    def complete_pending_drafting(self) -> None:
        # No drafter.
        pass


class _RegisteringDecodePeer:
    """A decode-role peer: what a sleep release asks of it."""

    mode = "decode"

    def __init__(self, trace) -> None:
        self._trace = trace

    def execute(self, op) -> None:
        self._trace.append(("receive", op))

    def release_cache_registration(self) -> None:
        self._trace.append("peer.deregister")

    def restore_cache_registration(self) -> None:
        self._trace.append("peer.register")


def test_kv_release_quiesces_copies_and_deregisters_the_peer():
    trace = []
    handle = DeviceHandle(
        _SleepExecutor(trace), kv_transfer=_RegisteringDecodePeer(trace)
    )

    handle.prepare_kv_release()
    # The offload copy streams, then the device, then the registration that
    # pins the pages about to be unmapped.
    assert trace == [
        "arena.synchronize",
        "device.synchronize",
        "peer.deregister",
    ]
    trace.clear()

    handle.finish_kv_resume()
    # Zero the re-mapped garbage before a peer can write into it again.
    assert trace == ["clear_kv_buffers", "peer.register"]


def test_kv_release_without_a_peer_only_quiesces_and_repairs():
    trace = []
    handle = DeviceHandle(_SleepExecutor(trace))

    handle.prepare_kv_release()
    handle.finish_kv_resume()
    handle.synchronize()

    assert trace == [
        "arena.synchronize",
        "device.synchronize",
        "clear_kv_buffers",
        "device.synchronize",
    ]


def test_nccl_suspend_and_resume_reach_the_communicators(monkeypatch):
    from tokenspeed.runtime.execution import device as device_module

    trace = []
    monkeypatch.setattr(
        device_module, "suspend_nccl_communicators", lambda: trace.append("suspend")
    )
    monkeypatch.setattr(
        device_module, "resume_nccl_communicators", lambda: trace.append("resume")
    )
    handle = DeviceHandle(_SleepExecutor(trace))

    handle.suspend_nccl()
    handle.resume_nccl()

    assert trace == ["suspend", "resume"]
