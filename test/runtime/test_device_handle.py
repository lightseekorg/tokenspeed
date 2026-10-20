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

Two halves, both stated in ``execution/forward_thread.py``. What the control
plane hands over travels in a closure and is frozen once submitted; and the
device side is not reachable from the control plane at all, because the loop
holds a ``DeviceHandle`` rather than the executor behind it.

The first half is easy to lose in three specific spots, so each gets a test:
the PD completion path (device writes issued straight from an event handler),
the multimodal gather (a live request struct captured by reference), and the
SHM release (a resource freed while a queued forward may still read it).

The second half is exercised the way the loop experiences it: the tests build
a real ``DeviceHandle`` over a stub executor and drive its operations,
asserting the work that crosses to the device side and the facts that come
back.
"""

from __future__ import annotations

import json
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from tokenspeed_scheduler import (
    PD,
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
    PrefetchOp,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine import scheduler_utils as scheduler_utils_module
from tokenspeed.runtime.execution.device import DeviceHandle, EncoderModelFacts
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import PlannedForward
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    multimodal_context_for_forward,
)
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
from tokenspeed.runtime.pd.transfer_hooks import PdTransferHooks


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
        lambda: SimpleNamespace(is_nvidia=False, restore_stack_limit=nullcontext),
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


class _ForwardThread:
    """Records what crossed to the data plane, and in what order.

    Mirrors the real contract: ``submit`` returns a future resolved with the
    callable's result or exception (never raising at the call site), ``run``
    blocks and re-raises.
    """

    def __init__(self, trace) -> None:
        self._trace = trace

    def submit(self, fn):
        self._trace.append("submit")
        future: Future = Future()
        try:
            future.set_result(fn())
        except BaseException as exc:  # noqa: BLE001 — mirrored to the future
            future.set_exception(exc)
        return future

    def run(self, fn):
        self._trace.append("run")
        return fn()


def _peer(trace=None):
    """A P-role transfer peer the handle will recognize by type."""

    class _Peer(DisaggPrefillExecutor):
        def __init__(self) -> None:
            pass

        def execute(self, op) -> None:
            if trace is not None:
                trace.append(("send", op))

        def prepare_prefill(self, op) -> None:
            if trace is not None:
                trace.append(("arm", op))

    return _Peer()


class _DecodeExecutor(DisaggDecodeExecutor):
    """A decode-role kv_transfer the hooks will recognize by type.

    Subclasses rather than fakes because the hooks dispatch on
    ``isinstance``; the real ``__init__`` (sockets, KV manager) is bypassed
    on purpose — only the three pop/generate methods are exercised.
    """

    def __init__(self, events, slot=None, candidates=None) -> None:
        self._events = events
        self._slot = slot
        self._candidates = candidates

    def generate_events(self):
        return self._events

    def pop_remote_cached_tokens(self, request_id):
        return 0

    def pop_remote_bootstrap_logprob(self, request_id):
        return None

    def pop_remote_cache_slot(self, req_id):
        return self._slot

    def pop_remote_spec_candidate_ids(self, req_id):
        return self._candidates


def _handle(trace, **kwargs):
    return DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread(trace),
            execute_forward_op=lambda *a, **k: trace.append("forward"),
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


class _PrefetchBatch:
    """The wire face of ``Cache.PrefetchOp``: ``rows`` is
    ``[(op_id, pages)]``, one op per waiting request, one row per page."""

    def __init__(self, rows) -> None:
        self.op_ids = [op_id for op_id, _ in rows]
        self.request_ids = [f"r{op_id}" for op_id, _ in rows]
        self.first_pages = [0 for _ in rows]
        self.num_pages = [pages for _, pages in rows]
        self.group_ids = [[0] * pages for _, pages in rows]
        self.host_pages = [[1 + i for i in range(pages)] for _, pages in rows]
        self.content_hashes = [[f"h{i}" for i in range(pages)] for _, pages in rows]
        self.page_offsets = [list(range(pages)) for _, pages in rows]
        self.page_indices = [list(range(pages)) for _, pages in rows]


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
            PrefetchOp=_PrefetchBatch,
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
    trace = []
    result = SimpleNamespace(sync=lambda: trace.append("sync"))

    def cpu_forward(op, sampling_params, **kwargs):
        assert op is batch
        assert kwargs["capture_next_input_ids"]
        with torch.profiler.record_function("cpu_model_work"):
            torch.ones(2, device="cpu").add_(1)
        return result

    handle = DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread(trace),
            execute_forward_op=cpu_forward,
        )
    )
    planned = replace(_planned(num_extends=batch.num_extends()), forward_op=batch)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        assert (
            handle._submit_forward(planned, capture_next_input_ids=True).result()
            is result
        )
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
    assert trace == ["submit", "sync"]


def test_profile_range_closes_and_propagates_original_forward_exception():
    batch = _profile_batch(ForwardMode.EXTEND)
    failure = RuntimeError("original forward failure")

    def cpu_forward(*args, **kwargs):
        raise failure

    handle = DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread([]),
            execute_forward_op=cpu_forward,
        )
    )
    planned = replace(_planned(num_extends=1), forward_op=batch)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        pending = handle._submit_forward(planned)
        with pytest.raises(RuntimeError) as raised:
            pending.result()
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
    handle._submit_forward(_planned(num_extends=0))
    assert trace == ["submit", "forward"]


def _loop(trace, kv_transfer, state):
    output_processor = SimpleNamespace(
        rid_to_state={"r0": state} if state is not None else {},
        on_remote_prefill_done=lambda rid, tok, cached_tokens, logprob: trace.append(
            ("bootstrap", tok)
        ),
        finish_remote_prefill_only_request=lambda rid: [],
    )
    return SimpleNamespace(
        kv_transfer=kv_transfer,
        output_processor=output_processor,
    )


def _decoding_state():
    return SimpleNamespace(to_abort=False, finished=False)


# ----------------------------------------------------------------------
# PD completion: device writes belong to the data plane, in one submission.
# ----------------------------------------------------------------------


def test_remote_prefill_completion_lands_both_writes_on_the_device():
    trace: list = []
    event = PD.RemotePrefillDoneEvent("r0", 42)
    kv_transfer = _DecodeExecutor([event], slot=7, candidates=(7, [11, 12]))
    state = _decoding_state()

    hooks = PdTransferHooks(_loop(trace, kv_transfer, state), _handle(trace))
    hooks.poll_transfer_events()

    # One crossing, not two, and the candidates precede the readiness arm:
    # hydration reads the row the candidates were just written into.
    assert trace == [
        ("bootstrap", 42),
        "run",
        ("candidates", 7, [11, 12]),
        ("ready", 7),
    ]


def test_completion_without_device_work_submits_nothing():
    trace: list = []
    event = PD.RemotePrefillDoneEvent("r0", 42)
    kv_transfer = _DecodeExecutor([event], slot=None, candidates=None)
    state = _decoding_state()

    hooks = PdTransferHooks(_loop(trace, kv_transfer, state), _handle(trace))
    hooks.poll_transfer_events()

    assert trace == [("bootstrap", 42)]


def test_an_aborted_request_still_lands_its_candidates_but_is_not_armed():
    trace: list = []
    event = PD.RemotePrefillDoneEvent("r0", 42)
    kv_transfer = _DecodeExecutor([event], slot=7, candidates=(7, [11, 12]))
    aborted = SimpleNamespace(to_abort=True, finished=False)

    hooks = PdTransferHooks(_loop(trace, kv_transfer, aborted), _handle(trace))
    hooks.poll_transfer_events()

    assert trace == ["run", ("candidates", 7, [11, 12])]


# ----------------------------------------------------------------------
# L2 cache plans: write-backs launch AHEAD of the zeroing (a stream-ordered
# one must read the reused pages' old bytes, and fences the caller's stream
# before the zeroing is enqueued), load-backs behind it.
# ----------------------------------------------------------------------


def test_one_plan_orders_write_backs_zeroing_then_load_backs(wire_ops):
    """The FIFO carries the correctness order for same-round page reuse: the
    write-backs are submitted ahead of the new owner's zeroing (a
    stream-ordered snapshot copy lands its fence on the default stream the
    zeroing runs on), and the load-backs target zeroed pages. Every stream is
    named: the copies are told which stream wrote the pages and which one
    the fence lands on; the loads are told which stream zeroed their
    destinations. The executor sees the plan's cache ops once, adapted, and
    the same list on both submissions."""
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
            submit_prefetches=lambda ops: None,
            poll_results=lambda: ["done"],
        ),
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append(
        ("zero", {group_id: tuple(ids) for group_id, ids in pages.items()})
    )

    handle.execute(plan, None)

    assert trace == [
        "submit",
        ("write_backs", [store, load], "execution-stream", "default-stream"),
        "submit",
        ("zero", {"kv": (3, 4)}),
        "submit",
        ("load_backs", [store, load], "default-stream"),
    ]
    # Polling never touches the FIFO — the round head must not wait on it.
    assert handle.poll_cache_results() == ["done"]
    assert trace[-1] != "submit"


def test_page_copies_follow_the_stores_and_precede_the_zeroing(wire_ops):
    """A prefix hit's private copy (or an owner's move off its published
    block) rides the same FIFO: after the stores' fence, before the zeroing
    -- its source may be re-granted and zeroed by this very plan -- and only
    when some group has pairs. Groups without pairs never reach the
    executor."""
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
            submit_prefetches=lambda ops: None,
        ),
    )
    handle._executor.copy_cache_pages = lambda pairs: trace.append(
        ("copy", {gid: arr.tolist() for gid, arr in pairs.items()})
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append(
        ("zero", {group_id: tuple(ids) for group_id, ids in pages.items()})
    )

    handle.execute(plan, None)

    assert trace == [
        "submit",
        ("write_backs", [store]),
        "submit",
        ("copy", {"kv": [[3, 8], [5, 9]]}),
        "submit",
        ("zero", {"kv": (3,)}),
        "submit",
        ("load_backs", [store]),
    ]

    # A plan with no pairs submits no copy at all.
    trace.clear()
    handle.execute(_plan(pages_to_zero=[2], pages_to_copy={"kv": []}), None)
    assert trace == ["submit", ("zero", {"kv": (2,)})]


def test_a_failed_page_copy_surfaces_at_the_next_round(wire_ops):
    """The copy has no completion event, so its exception would otherwise
    vanish on the data plane; ``execute`` settles the previous round's copy
    submissions before enqueueing anything."""
    trace: list = []
    handle = _handle(trace)

    def failing_copy(pairs):
        raise IndexError("local block ID 99 outside [0, 8)")

    handle._executor.copy_cache_pages = failing_copy
    handle.execute(_plan(pages_to_copy={"kv": [(1, 99)]}), None)
    with pytest.raises(RuntimeError, match="page copy submission failed"):
        handle.execute(_plan(), None)


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
            submit_prefetches=lambda ops: None,
        ),
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append(
        ("zero", {group_id: tuple(ids) for group_id, ids in pages.items()})
    )

    handle.execute(plan, None)

    kinds = [entry[0] if isinstance(entry, tuple) else entry for entry in trace]
    assert kinds == ["submit", "write_backs", "submit", "zero", "submit", "load_backs"]
    # Stores precede the zeroing on the FIFO, and the fence they record lands
    # on the stream the zeroing runs on.
    stores = trace[1][1]
    assert trace[1][2] == "default-stream"
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
            submit_prefetches=lambda ops: None,
        ),
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append("zero")
    handle._executor.execute_forward_op = lambda op, *a, **k: trace.append(
        ("forward", list(op.request_pool_indices))
    )
    planned = _planned(num_extends=1)
    planned.forward_op.request_pool_indices = [6]  # the slot just freed

    handle.execute(plan, planned)

    order = [entry for entry in trace if entry != "submit"]
    assert order == [("store", [6]), "zero", "load_backs", ("forward", [6])]


def test_page_zeroing_without_l2_submits_only_zeroing():
    trace: list = []
    handle = _handle(trace)
    handle._executor.zero_cache_pages = lambda pages: trace.append(
        ("zero", {group_id: tuple(ids) for group_id, ids in pages.items()})
    )

    handle.execute(_plan(pages_to_zero=[3, 4]), None)

    assert trace == ["submit", ("zero", {"kv": (3, 4)})]


def test_a_plan_with_no_device_work_submits_nothing():
    trace: list = []
    handle = _handle(trace, host_cache_executor=SimpleNamespace())

    handle.execute(_plan(), None)

    assert trace == []


def test_a_failed_cache_submission_surfaces_at_the_next_poll(wire_ops):
    """A submission that raised produces no completion acks; swallowing it
    would leave its ops counted in flight forever."""
    trace: list = []

    def exploding(ops, *, prerequisite_stream, fence_stream):
        raise ValueError("bad cache op")

    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            submit_write_backs=exploding,
            submit_load_backs=lambda ops, *, prerequisite_stream: None,
            submit_prefetches=lambda ops: None,
            poll_results=lambda: [],
        ),
    )

    # Submission itself never raises (fire-and-forget)...
    handle.execute(_plan(cache=[_WriteBackOp("op")]), None)
    # ...the failure re-raises at the round head, data-plane cause chained.
    with pytest.raises(RuntimeError, match="cache-plan submission failed") as info:
        handle.poll_cache_results()
    assert isinstance(info.value.__cause__, ValueError)


def test_prefetches_start_on_the_control_plane_between_zeroing_and_loads(wire_ops):
    """An L3 prefetch fills Host pages on the executor's own lane: no stream,
    no Device page, so it is started right on the control plane -- after the
    stores and the zeroing were queued, before the loads -- and never rides
    the FIFO. Each op reaches the executor once, per request."""
    trace: list = []
    plan = _plan(
        pages_to_zero=[2],
        cache=[
            _WriteBackOp("store"),
            _PrefetchBatch([(4, 3), (5, 1)]),
            _LoadBackOp("load"),
        ],
    )
    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            submit_write_backs=lambda ops, *, prerequisite_stream, fence_stream: (
                trace.append("write_backs")
            ),
            submit_prefetches=lambda ops: trace.append(
                (
                    "prefetches",
                    [
                        (op.op_id, len(op.rows))
                        for op in ops
                        if isinstance(op, PrefetchOp)
                    ],
                )
            ),
            submit_load_backs=lambda ops, *, prerequisite_stream: trace.append(
                "load_backs"
            ),
        ),
    )
    handle._executor.zero_cache_pages = lambda pages: trace.append("zero")

    handle.execute(plan, None)

    assert trace == [
        "submit",
        "write_backs",
        "submit",
        "zero",
        ("prefetches", [(4, 3), (5, 1)]),
        "submit",
        "load_backs",
    ]


def test_shutdown_cache_joins_submissions_then_closes_on_the_forward_thread():
    from concurrent.futures import Future

    trace: list = []
    handle = _handle(
        trace,
        host_cache_executor=SimpleNamespace(
            shutdown=lambda: trace.append("l2_shutdown"),
        ),
    )
    for queue in (
        handle._l2_submissions,
        handle._transfer_submissions,
        handle._copy_submissions,
    ):
        pending = Future()
        pending.set_result(None)
        queue.append(pending)

    handle.shutdown_cache()

    assert list(handle._l2_submissions) == []
    assert list(handle._transfer_submissions) == []
    assert list(handle._copy_submissions) == []
    assert trace == ["run", "l2_shutdown"]


def test_shutdown_cache_surfaces_a_failed_final_round_copy_without_a_host_cache():
    """A copy or transfer queued by the last round has no next round to settle
    it; shutdown joins it and re-raises even with no Host cache to close."""
    from concurrent.futures import Future

    handle = _handle([])
    failed = Future()
    failed.set_exception(IndexError("local block ID 99 outside [0, 8)"))
    handle._copy_submissions.append(failed)
    settled = Future()
    settled.set_result(None)
    handle._transfer_submissions.append(settled)

    with pytest.raises(IndexError, match="99"):
        handle.shutdown_cache()
    assert list(handle._copy_submissions) == []
    assert list(handle._transfer_submissions) == []


def test_cache_polling_without_kvstore_refuses_loudly():
    with pytest.raises(RuntimeError, match="enable-kvstore"):
        _handle([]).poll_cache_results()


# ----------------------------------------------------------------------
# The transfer peer: a ForwardBatch the model does not run.
# ----------------------------------------------------------------------


def test_an_unrecognized_transfer_peer_is_refused():
    """The role is read off the peer at construction, so a peer of the wrong
    type is a startup error, not a surprise at the first transfer."""
    with pytest.raises(TypeError, match="Disagg"):
        _handle([], kv_transfer=SimpleNamespace())


def test_the_remote_decode_and_the_arming_ride_the_fifo():
    """The send walks KV-pool device memory the forwards wrote, and the
    arming must precede the forward it arms. Both orderings come from the
    FIFO, not from the loop's scheduling rules."""
    trace: list = []
    handle = _handle(trace, kv_transfer=_peer(trace))

    chunk = _planned(num_extends=1, label="CHUNK")
    remote_decode = SimpleNamespace(request_ids=["done"])

    handle.execute(_plan(), chunk)
    handle.execute(_plan(remote_decode=remote_decode), None)

    # Arming is enqueued before the forward it arms; the send follows the
    # forwards whose KV it reads (the scheduler emits a remote decode only
    # after its final chunk's result landed), submitted asynchronously like
    # the forwards themselves.
    assert trace[:3] == ["submit", ("arm", chunk.forward_op), "submit"]
    assert trace[-2:] == ["submit", ("send", remote_decode)]


# ----------------------------------------------------------------------
# EPD admission: encoder facts resolve past the gate, never before.
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


def test_text_only_pd_nodes_never_read_the_encoder_facts():
    """The facts callable must not fire unless the node is an EPD prefill.

    Reading the vision tower's dtype raises on a text-only model, and every
    text-only PD node passes through this factory — so the facts are handed
    over as a bound method and resolved only past the manager gate. Passing
    the VALUE here once crashed every text-only PD deployment at startup.
    """
    from tokenspeed.runtime.epd.prefill_admission import make_epd_prefill_admission

    def facts():  # pragma: no cover — reaching this is the failure
        raise AssertionError("encoder facts read on a non-EPD node")

    admission = make_epd_prefill_admission(
        SimpleNamespace(disaggregation_mode="decode"),
        0,
        model_config=SimpleNamespace(is_multimodal_active=False),
        encoder_model_facts=facts,
        mapping=None,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pg_manager=None,
    )

    assert admission is None


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
# SHM release: freed behind any forward that captured the features.
# ----------------------------------------------------------------------


def test_shm_release_is_queued_behind_the_forwards_that_may_read_it():
    from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser

    trace: list = []
    released: list = []
    processor = OutputProcesser(
        send_to_tokenizer=lambda *a, **k: None,
        metrics=SimpleNamespace(),
        defer_to_device=_handle(trace).run_multimodal_work,
    )
    state = SimpleNamespace(
        has_pending_multimodal_features=lambda: True,
        release_pending_multimodal_features=lambda: released.append("released"),
    )

    processor._release_multimodal_features(state)

    assert trace == ["submit"]
    assert released == ["released"]


def test_a_request_with_no_pending_features_releases_inline():
    from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser

    trace: list = []
    released: list = []
    processor = OutputProcesser(
        send_to_tokenizer=lambda *a, **k: None,
        metrics=SimpleNamespace(),
        defer_to_device=_handle(trace).run_multimodal_work,
    )
    state = SimpleNamespace(
        has_pending_multimodal_features=lambda: False,
        release_pending_multimodal_features=lambda: released.append("released"),
    )

    processor._release_multimodal_features(state)

    assert trace == []
    assert released == ["released"]


# ----------------------------------------------------------------------
# The opt-in dispatch guard: device work on the control thread raises.
# ----------------------------------------------------------------------


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


def test_epd_receive_allocation_crosses_through_the_runner(monkeypatch):
    """The job's device steps run wherever the runner says — the engine
    passes ``DeviceHandle.run_multimodal_work``, so they land on the forward
    thread; ``None`` (the blocking test wrapper) runs them inline."""
    from tokenspeed.runtime.epd import prefill_admission

    # Force the legacy path: with no pool, the receive buffer must be
    # allocated through the runner.
    monkeypatch.setattr(prefill_admission, "_get_pool", lambda engine, device: None)

    ran: list = []

    def runner(work):
        ran.append(work)
        return work()

    item = SimpleNamespace(
        encode_handshake={
            "bootstrap_host": "h",
            "bootstrap_port": 1,
            "bootstrap_room": 2,
        },
        encoded=None,
        offsets=[(0, 3)],  # 4 encoded tokens
    )

    class _Receiver:
        def __init__(self, manager, addr, room) -> None:
            pass

        def poll(self):  # pragma: no cover — not driven here
            return "Bootstrapped"

    job = prefill_admission.EmbeddingReceiveJob(
        [item],
        SimpleNamespace(engine=SimpleNamespace(register=lambda *a: None)),
        hidden=8,
        num_deepstack=0,
        dtype=torch.float32,
        device="cpu",
        receiver_factory=_Receiver,
        run_device_work=runner,
    )

    assert len(ran) == 1  # exactly the recv-buffer allocation
    assert job._items[0].recv_main.shape == (4, 8)


def test_prefill_usage_hook_records_committed_totals_and_skips_retired_requests():
    from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor

    recorded = []
    logprobs = []
    transfer = object.__new__(DisaggPrefillExecutor)
    transfer.senders = {"hit": SimpleNamespace(bootstrap_room=9)}
    transfer.kv_manager = SimpleNamespace(
        record_cached_tokens=lambda room, count: recorded.append((room, count)),
        record_bootstrap_logprob=lambda room, lp: logprobs.append((room, lp)),
    )
    state = SimpleNamespace(cached_tokens=1280, output_token_logprobs_val=[])
    loop = SimpleNamespace(
        kv_transfer=transfer,
        output_processor=SimpleNamespace(rid_to_state={"hit": state}),
    )
    hooks = PdTransferHooks(loop, None)
    hooks.record_prefill_usage(["hit", "retired"])
    # An intermediate chunk has no bootstrap token yet; the final chunk's
    # commit appends it with its logprob, which then rides the status message.
    assert logprobs == []
    state.cached_tokens = 1536
    state.output_token_logprobs_val = [-0.75]
    hooks.record_prefill_usage(["hit"])
    assert recorded == [(9, 1280), (9, 1536)]
    assert logprobs == [(9, -0.75)]
    loop.kv_transfer = None
    hooks.record_prefill_usage(["hit"])
    assert recorded == [(9, 1280), (9, 1536)]


def test_expert_load_reset_and_dump_ride_the_data_plane_on_the_execution_stream(
    tmp_path,
):
    """The load counters are zeroed and read back through the forward thread,
    on the execution stream the routing kernels bump them on, and the dump is
    this rank's record alone: no collective, so a profile stop that reaches
    attention-DP workers independently cannot deadlock."""
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
    trace: list = []
    streams: list = []

    class _DeviceModule:
        @staticmethod
        def stream(stream):
            streams.append(stream)
            return contextlib.nullcontext()

    handle = DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread(trace),
            device_module=_DeviceModule(),
            execution_stream="execution-stream",
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
    assert trace == ["run", "run"]
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
    """The online rebalance's window boundary: one data-plane crossing that
    copies the counters to the host and zeroes them on the execution stream,
    with no collective (the gate reduces the result afterwards)."""
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
    trace: list = []
    streams: list = []

    class _DeviceModule:
        @staticmethod
        def stream(stream):
            streams.append(stream)
            return contextlib.nullcontext()

    handle = DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread(trace),
            device_module=_DeviceModule(),
            execution_stream="execution-stream",
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
    assert trace == ["run"] and streams == ["execution-stream"]
    assert snapshot.physical_count.tolist() == [[3, 1, 2, 5], [0, 4, 0, 4]]
    assert snapshot.logical_count.tolist() == [[8, 1, 2], [0, 8, 0]]
    assert torch.equal(
        snapshot.physical_to_logical_map, placement.physical_to_logical_map_cpu
    )
    assert not placement.physical_load.any()


def test_expert_placement_chunk_applies_then_synchronizes_the_execution_stream():
    """A chunk is one blocking data-plane crossing: the updater moves the
    weights and switches the tables inside the execution stream, and the
    stream is synchronized before the call returns so a P2P failure surfaces
    in the op's reply. Without --enable-eplb there is no updater to run."""
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
            forward_thread=_ForwardThread(trace),
            device_module=_DeviceModule(),
            execution_stream=stream,
            model_runner=SimpleNamespace(expert_location_updater=_Updater()),
        )
    )
    handle.apply_expert_placement((0,), rows, moves)
    assert trace == [
        "run",
        ("stream", stream),
        ("apply", (0,), [[0, 0, 1, 2]], moves),
        "synchronize",
    ]
    handle = DeviceHandle(
        SimpleNamespace(
            forward_thread=_ForwardThread([]),
            model_runner=SimpleNamespace(expert_location_updater=None),
        )
    )
    with pytest.raises(RuntimeError, match="--enable-eplb"):
        handle.apply_expert_placement((0,), rows, moves)


# ----------------------------------------------------------------------
# Memory-saver release/wake: what surrounds the unmap runs on the data plane.
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
        self.forward_thread = _ForwardThread(trace)
        self.device_module = _SleepDeviceModule(trace)
        self.token_to_kv_pool = _SleepPool(trace)


class _RegisteringDecodePeer(DisaggDecodeExecutor):
    """A decode-role peer; the real ``__init__`` (sockets, engine) is bypassed."""

    def __init__(self, trace) -> None:
        self._trace = trace

    def release_cache_registration(self) -> None:
        self._trace.append("peer.deregister")

    def restore_cache_registration(self) -> None:
        self._trace.append("peer.register")


def test_kv_release_quiesces_copies_and_deregisters_the_peer_on_the_forward_thread():
    trace = []
    handle = DeviceHandle(
        _SleepExecutor(trace), kv_transfer=_RegisteringDecodePeer(trace)
    )

    handle.prepare_kv_release()
    # One blocking submission: the offload copy streams, then the device, then
    # the registration that pins the pages about to be unmapped.
    assert trace == [
        "run",
        "arena.synchronize",
        "device.synchronize",
        "peer.deregister",
    ]
    trace.clear()

    handle.finish_kv_resume()
    # Zero the re-mapped garbage before a peer can write into it again.
    assert trace == ["run", "clear_kv_buffers", "peer.register"]


def test_kv_release_without_a_peer_only_quiesces_and_repairs():
    trace = []
    handle = DeviceHandle(_SleepExecutor(trace))

    handle.prepare_kv_release()
    handle.finish_kv_resume()
    handle.synchronize()

    assert trace == [
        "run",
        "arena.synchronize",
        "device.synchronize",
        "run",
        "clear_kv_buffers",
        "run",
        "device.synchronize",
    ]


def test_nccl_suspend_and_resume_ride_the_forward_thread(monkeypatch):
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

    assert trace == ["run", "suspend", "run", "resume"]
