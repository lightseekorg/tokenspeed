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

"""Execution slots reach the attention backends through the forward inputs.

The scheduler exports each forward row's execution slot -- the decode permit
on a PD decode node, the request row on every other role -- and whether the
row is the request's first forward on it (docs/design/scheduler.md,
"Execution slots"). Real scheduler plans drive the real ``InputBuffers`` and
the executor's real forward entry points (``DeviceHandle.execute`` into
``ModelExecutor.execute_forward_op``, ``execute_idle_forward``,
``finish_memory_resume``); only the forward runner below the executor is a
stand-in, which captures and replays CUDA graphs of a backend reading the slots
it was bound. The slots ride the request rows' H2D, padding rows read
``max_bs + 1``, a dummy forward reads padding only, and a fresh row's
workspace is cleared on the execution stream after the plan's state folds and
before the forward. The slot buffers keep their addresses and contents across
the memory saver's release and resume.

Profiled checks and the memory-saver cycles run in a new interpreter (see
``_in_fresh_process``), so collecting this module starts no profiler session
and does no CUDA work of its own.
"""

from __future__ import annotations

import os
import subprocess
import sys
from itertools import compress
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=90, suite="runtime-1gpu")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

import tokenspeed_engine.scheduler as ts  # noqa: E402

from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan  # noqa: E402
from tokenspeed.runtime.execution import model_executor  # noqa: E402
from tokenspeed.runtime.execution.device import DeviceHandle  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.execution.nan_guard import NanGuard  # noqa: E402
from tokenspeed.runtime.execution.request_slots import RequestSlotLayout  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.execution.types import (  # noqa: E402
    DpForwardMetadata,
    PlannedForward,
    StopCriteria,
)
from tokenspeed.runtime.layers.attention.backends.base import (  # noqa: E402
    CachePoolBinding,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import (  # noqa: E402
    TorchMemorySaverAdapter,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _in_fresh_process(check: str, *args) -> None:
    """Run this module's ``check(*args)`` in a new interpreter.

    The profiler records CUDA activity only in a process whose first profiler
    session collected it: after another module's CPU-only session, every
    later session of the process records no CUDA event. A CUDA session also
    initializes CUDA. A new interpreter gives each profiled check a clean
    profiler and leaves the test process's CUDA state alone; the memory
    saver's preload hooks likewise need a process that installs them before
    CUDA initializes.
    """
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from test.runtime.execution.test_execution_slots import "
            f"{check}; {check}{args!r}",
        ],
        cwd=_REPO_ROOT,
        check=True,
        timeout=600,
    )


Role = ts.SchedulerConfig.Role
VOCAB = 128
# Records a slot holds before any owner of this test ran: a previous owner's.
STALE = 7


def _config(role, max_bs: int, residents: int) -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.role = role
    cfg.prefix_granularity = 2
    cfg.max_batch_size = max_bs
    cfg.max_resident_requests = residents
    cfg.max_scheduled_tokens = 64
    cfg.num_device_pages = 257
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 257
    cfg.max_retracted_requests = 4
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.overlap_schedule_depth = 1
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=2,
            total_pages=257,
            attn_kind=ts.AttnKind.Full,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    cfg.host_group_packing = [1]
    return cfg


def _deferred_decode_config() -> ts.SchedulerConfig:
    """D role, one decode permit, two residents, a History group and a State
    group whose decode state is deferred, and a forced retraction at the
    fifth plan."""
    cfg = ts.SchedulerConfig()
    cfg.role = Role.D
    cfg.prefix_granularity = 2
    cfg.num_device_pages = 33
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 33
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 1
    cfg.max_resident_requests = 2
    cfg.max_retracted_requests = 2
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    cfg.debug_force_retraction_interval = 5
    cfg.debug_force_retraction_max_per_request = 1
    history = ts.CacheGroupConfig(
        group_id="history",
        block_granularity=2,
        total_pages=33,
        attn_kind=ts.AttnKind.Full,
        transfer_policy=ts.CacheTransferPolicy.FullSuffix,
    )
    state = ts.CacheGroupConfig(
        group_id="state",
        block_granularity=0,
        total_pages=33,
        attn_kind=ts.AttnKind.State,
        transfer_policy=ts.CacheTransferPolicy.LatestSnapshot,
    )
    state.deferred_decode_state = True
    cfg.cache_groups = [history, state]
    cfg.host_group_packing = [1, 1]
    return cfg


def _spec(request_id: str, tokens: list[int]) -> ts.RequestSpec:
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(tokens)
    spec.max_new_tokens = 16
    return spec


def _send(scheduler, event) -> None:
    scheduler.advance(ts.ExecutionEvent().add_event(event))


def _advance(scheduler, request_id: str, tokens: list[int]) -> None:
    event = ts.ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = tokens
    _send(scheduler, event)


def _finish(scheduler, request_id: str) -> None:
    event = ts.ForwardEvent.Finish()
    event.request_id = request_id
    _send(scheduler, event)


def _submit_pd(scheduler, *request_ids: str) -> None:
    """Bootstrapped D-role requests, each admitted by its own plan's landing."""
    scheduler.submit_requests(
        [_spec(rid, [100 * i + 1, 100 * i + 2]) for i, rid in enumerate(request_ids)]
    )
    for rid in request_ids:
        _send(scheduler, ts.PD.BootstrappedEvent(rid))
    for rid in request_ids:
        landing = scheduler.next_execution_plan().remote_prefill
        assert list(landing.request_ids) == [rid]


def _land(scheduler, *request_ids: str) -> None:
    for rid in request_ids:
        _send(scheduler, ts.PD.RemotePrefillDoneEvent(rid, 42))


def _forward(plan):
    batches = [batch for batch in plan.forward if batch.request_ids]
    return batches[0] if batches else None


class _SlotRecords(CachePoolBinding):
    """A leaf keeping per-execution-slot records, the way KDA's retained
    decode records will: a count per slot, folded by a retraction, cleared for
    a fresh owner and advanced by every forward on the slot. It logs each call
    with its stream; its device work runs on the stream it is handed."""

    def __init__(self, trace: list, max_bs: int):
        self._init_pool_binding()
        self.trace = trace
        self.execution_slots: torch.Tensor | None = None
        self.counts = torch.full(
            (RequestSlotLayout(max_bs).capacity,),
            STALE,
            dtype=torch.int32,
            device="cuda",
        )
        self.folded: list[torch.Tensor] = []
        self.forward_read: list[torch.Tensor] = []

    def bind_execution_slots(self, execution_slots):
        self.execution_slots = execution_slots

    def fold_deferred_state(self, state_folds, *, stream):
        rows = {group: folds.tolist() for group, folds in state_folds.items()}
        self.trace.append(("fold", rows, stream))
        with torch.cuda.stream(stream):
            for folds in state_folds.values():
                slots = folds[:, 2].to(device="cuda", dtype=torch.int64)
                self.folded.append(self.counts[slots])

    def reset_execution_workspace(self, execution_slots, *, stream):
        with torch.cuda.stream(stream):
            self.trace.append(("reset", execution_slots.clone(), stream))
            self.counts.index_fill_(0, execution_slots, 0)

    def forward(self, bs: int, stream) -> None:
        """What a decode commit does: read the rows' counts, then add one."""
        self.trace.append(("forward", bs, stream))
        with torch.cuda.stream(stream):
            slots = self.execution_slots[:bs]
            self.forward_read.append(self.counts[slots])
            self.counts.index_add_(
                0, slots, torch.ones(bs, dtype=torch.int32, device="cuda")
            )

    def read(self, bs: int) -> torch.Tensor:
        """The execution slots a forward over ``bs`` rows reads, padding
        included: what a graph recorded at that bucket sees at replay."""
        return self.execution_slots[:bs].clone()


class _Composite(CachePoolBinding):
    """A parent node: the executor talks to it, the leaf keeps the records."""

    skips_incomplete_prefill_outputs = False

    def __init__(self, leaf: _SlotRecords):
        self._init_pool_binding()
        self.leaf = leaf

    def child_backends(self):
        return (self.leaf,)


def _executor(
    max_bs: int, residents: int, saver: TorchMemorySaverAdapter | None = None
) -> tuple[ModelExecutor, _SlotRecords]:
    """The executor's input staging over real buffers, allocated in the
    memory saver's named regions as the constructor allocates them, its
    backends bound by the constructor's own binding step."""
    if saver is None:
        saver = TorchMemorySaverAdapter.create(enable=False, device="cuda")
    trace: list = []
    with saver.named_region("attention_state"):
        leaf = _SlotRecords(trace, max_bs)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cuda"
    executor.device_module = torch.cuda
    executor.default_stream = torch.cuda.current_stream()
    executor.execution_stream = torch.cuda.Stream()
    executor.memory_saver_adapter = saver
    executor.attn_backend = _Composite(leaf)
    pool_padding = RequestSlotLayout(residents).padding
    with saver.named_region("executor_inputs"):
        executor.input_buffers = InputBuffers(
            max_bs=max_bs,
            max_num_tokens=64,
            state_write_padding_pool_index=pool_padding,
            device="cuda",
        )
    with saver.named_region("runtime_states"):
        executor.runtime_states = RuntimeStates(
            req_pool_size=pool_padding,
            input_vocab_size=VOCAB,
            vocab_size=VOCAB,
            output_length=1,
            device="cuda",
        )
        executor.request_ends = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
    executor._bind_forward_rows()
    return executor, leaf


class _NoTables:
    """The batch's packed cache tables: the stand-in forwards read none."""

    @staticmethod
    def tables(*, active_forward_op):
        return {}

    tables_cpu = tables


class _ForwardRunner:
    """The forward runner below the executor (``ForwardStepRunner``'s place).

    With graph buckets it captures the leaf's slot read per bucket as
    ``ForwardStepRunner._capture_one`` does -- dummy warmup forwards on the
    capture stream, then the recorded one -- and replays the bucket a batch
    (or a DP-idle rank's empty batch) pads to. Without, it runs the leaf's
    forward eagerly. Each live row samples one token."""

    draft_overlap = None
    disable = False
    last_graph_bs = 0

    def __init__(self, leaf: _SlotRecords, buckets: tuple[int, ...]):
        self.leaf = leaf
        self.buckets = sorted(buckets)
        self.stream = torch.cuda.Stream()
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        # (bucket, slots) each capture warmup's dummy forward read.
        self.dummy_reads: list[tuple[int, list[int]]] = []
        # (bucket, slots) each replay read.
        self.replays: list[tuple[int, torch.Tensor]] = []

    def capture(self, *, entries, observer):
        assert entries is None
        for bs in self.buckets:
            with torch.cuda.stream(self.stream):
                for _ in range(2):
                    torch.cuda.synchronize()
                    self.dummy_reads.append((bs, self.leaf.read(bs).tolist()))
                torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                read = self.leaf.read(bs)
            self.graphs[bs] = (graph, read)

    def can_run(self, bs, ctx) -> bool:
        return bool(self.buckets)

    def padded_bs(self, bs, ctx) -> int:
        return next(bucket for bucket in self.buckets if bucket >= bs)

    def __call__(self, *, bs, ctx, **kwargs):
        if self.buckets:
            bucket = self.padded_bs(bs, ctx)
            graph, read = self.graphs[bucket]
            graph.replay()
            self.replays.append((bucket, read.clone()))
        else:
            self.leaf.forward(bs, torch.cuda.current_stream())
        tokens = torch.ones(bs, dtype=torch.int32, device="cuda")
        return tokens, torch.ones_like(tokens), None, None, None

    def last_replay(self) -> tuple[int, list[int]]:
        torch.cuda.synchronize()
        bucket, read = self.replays[-1]
        return bucket, read.tolist()


def _serving(monkeypatch, max_bs: int, residents: int, buckets=(), saver=None):
    """An executor whose real forward entry points run down to a stand-in
    forward runner, and the handle the control plane drives it through."""
    executor, leaf = _executor(max_bs, residents, saver)
    runner = _ForwardRunner(leaf, buckets)
    monkeypatch.setattr(
        model_executor,
        "build_managed_cache_metadata",
        lambda forward_op, **kwargs: _NoTables(),
    )
    # The captures' workspace freeze is process-wide; these graphs need none.
    monkeypatch.setattr(
        model_executor,
        "workspace_pool",
        lambda device: SimpleNamespace(freeze=lambda: None),
    )
    executor.forward_step = runner
    executor.prefill_graph = SimpleNamespace(disable=True)
    executor.config = SimpleNamespace(
        overlap_schedule_depth=0,
        output_length=1,
        query_shard_size=1,
        spec_algo=None,
        global_rank=0,
        max_req_pool_size=residents,
        enable_cudagraph_gc=True,
    )
    executor.log_step = 0
    executor.nan_guard = NanGuard.create(False, max_bs, "cuda")
    executor._validate_page_ids = False
    executor._cache_runtime_contract = SimpleNamespace(group_specs=())
    executor.drafter = None
    executor.tree_spec = None
    executor.grammar_runtime = None
    executor._active_multimodal_context = None
    executor._active_positions_override = None
    executor.token_to_kv_pool = SimpleNamespace()
    executor.model_runner = SimpleNamespace(
        mapping=SimpleNamespace(requires_forward_coordination=False)
    )
    executor.sampling_backend = SimpleNamespace(
        config=SimpleNamespace(synthetic_acceptance_length=None),
        prepare_step=lambda **kwargs: None,
        get_packed_output_d2h=lambda tokens, lengths: None,
    )
    executor.dp_sampling_runtime_config = SimpleNamespace(
        enabled=False, min_bs=None, topology=None
    )
    executor.complete_pending_drafting = lambda: None
    executor.zero_cache_pages = lambda pages: None
    executor.copy_cache_pages = lambda copies: None
    handle = DeviceHandle(
        executor,
        host_cache_executor=SimpleNamespace(
            submit_write_backs=lambda ops, *, prerequisite_stream, fence_stream: None,
            submit_load_backs=lambda ops, *, prerequisite_stream: None,
        ),
    )
    return executor, leaf, runner, handle


def _execute(handle, plan, admission: dict[str, int]):
    """The control plane's round: the plan's device work and forward."""
    op = _forward(plan)
    planned = None
    if op is not None:
        planned = PlannedForward(
            forward_op=op,
            sampling_params_list=[],
            dp_metadata=None,
            grammar_inputs=None,
            multimodal_context=None,
            ngram_inputs=None,
            request_history_seeds=None,
            input_logprob_plan=None,
            sampled_top_k=0,
            stop_criteria=tuple(StopCriteria((), 1 << 20) for _ in op.request_ids),
            admission_ids=tuple(admission[rid] for rid in op.request_ids),
        )
    handle.execute(
        plan,
        planned,
        cache_ops=cache_ops_from_plan(
            plan, request_admission=admission.__getitem__
        ).device,
    )
    return op


def _idle_round() -> DpForwardMetadata:
    """This rank idles while another decodes two rows."""
    return DpForwardMetadata(
        global_num_tokens=[0, 2],
        global_batch_size=[0, 2],
        global_num_decodes=[0, 2],
        global_forward_mode=[ForwardMode.IDLE, ForwardMode.DECODE],
        all_decode_or_idle=True,
        all_extend=False,
        need_idle_forward=True,
    )


def _fill(executor, op) -> None:
    """The forward prologue's staging on the execution stream."""
    executor.execution_stream.wait_stream(executor.default_stream)
    with torch.cuda.stream(executor.execution_stream):
        executor._fill_batch_inputs(op, sum(op.input_lengths))
    torch.cuda.synchronize()


def _resets(leaf) -> list[list[int]]:
    torch.cuda.synchronize()
    return [entry[1].tolist() for entry in leaf.trace if entry[0] == "reset"]


def _decode_node_with_extra_slots():
    """Three decode permits, six residents: the last two admitted requests
    hold request rows 5 and 6, beyond any permit."""
    scheduler = ts.Scheduler(_config(Role.D, 3, 6))
    _submit_pd(scheduler, "a", "b", "c", "d", "e", "f")
    _land(scheduler, "e", "f")
    return scheduler


def test_decode_rows_carry_their_permits_not_their_request_rows():
    scheduler = _decode_node_with_extra_slots()
    executor, leaf = _executor(max_bs=3, residents=6)
    ib = executor.input_buffers
    op = _forward(scheduler.next_execution_plan())
    rows, slots = list(op.request_pool_indices), list(op.execution_slots)
    assert sorted(rows) == [5, 6]
    assert sorted(slots) == [1, 2]
    assert list(op.execution_slot_fresh) == [True, True]

    _fill(executor, op)

    assert ib.execution_slots_buf.tolist() == slots + [4]
    assert ib.req_pool_indices_buf[:2].tolist() == rows
    assert ib.state_write_req_pool_indices_buf[:2].tolist() == rows
    # The executor bound the backend to the very buffer the fill wrote, which
    # eager and graph forwards both read.
    assert leaf.execution_slots.data_ptr() == ib.execution_slots_buf.data_ptr()
    assert _resets(leaf) == [slots]


def test_a_mixed_batch_keeps_decode_slots_after_the_extend_rows():
    cfg = _config(Role.Fused, 4, 4)
    cfg.enable_mixed_prefill_decode = True
    scheduler = ts.Scheduler(cfg)
    executor, leaf = _executor(max_bs=4, residents=4)
    ib = executor.input_buffers
    scheduler.submit_requests([_spec("decoding", list(range(1, 9)))])
    _fill(executor, _forward(scheduler.next_execution_plan()))
    _advance(scheduler, "decoding", [42])
    _fill(executor, _forward(scheduler.next_execution_plan()))
    _advance(scheduler, "decoding", [43])
    scheduler.submit_requests([_spec("new", list(range(11, 19)))])
    op = _forward(scheduler.next_execution_plan())
    assert list(op.request_ids) == ["new", "decoding"]
    assert op.num_extends() == 1
    new_row, decoding_row = list(op.request_pool_indices)

    _fill(executor, op)

    # A fused node's slot is the request row, extend rows first.
    assert ib.execution_slots_buf.tolist() == [new_row, decoding_row, 5, 5]
    assert ib.req_pool_indices_buf[:2].tolist() == [new_row, decoding_row]
    # Only the admission's first chunk starts a slot.
    assert _resets(leaf) == [[decoding_row], [new_row]]


def test_padding_rows_take_the_execution_padding_slot():
    """max_bs + 1, never the request-pool padding row: with extra slots the
    request pool's padding row is a different, larger row."""
    scheduler = ts.Scheduler(_config(Role.D, 3, 6))
    _submit_pd(scheduler, "a", "b", "c", "d", "e", "f")
    executor, _ = _executor(max_bs=3, residents=6)
    ib = executor.input_buffers
    padding = RequestSlotLayout(3).padding
    assert ib.execution_slot_padding == padding == 4
    assert ib.state_write_padding_pool_index == RequestSlotLayout(6).padding == 7
    # Capture and DP-idle replays read the padding rows before any batch.
    assert ib.execution_slots_buf.tolist() == [padding] * 3

    _land(scheduler, "d", "e", "f")
    full = _forward(scheduler.next_execution_plan())
    _fill(executor, full)
    assert sorted(ib.execution_slots_buf.tolist()) == [1, 2, 3]
    finished, kept = list(full.request_ids[:2]), full.request_ids[2]
    for rid in full.request_ids:
        _advance(scheduler, rid, [43])
    for rid in finished:
        _finish(scheduler, rid)

    # A smaller batch: the rows a larger one wrote become padding again, in
    # the very upload that writes the batch's rows (a graph replay at the
    # larger bucket reads them).
    op = _forward(scheduler.next_execution_plan())
    assert list(op.request_ids) == [kept]
    _fill(executor, op)
    assert ib.execution_slots_buf.tolist() == list(op.execution_slots) + [padding] * 2
    assert ib.req_pool_indices_buf[1:].tolist() == [0, 0]
    assert ib.state_write_req_pool_indices_buf[1:].tolist() == [7, 7]

    # A dummy batch smaller than the buffer pads every row all the same.
    ib.fill_dummy_decode_buffers(batch_size=1, total_tokens=1)
    assert ib.execution_slots_buf.tolist() == [padding] * 3
    assert ib.req_pool_indices_buf[:1].tolist() == [0]
    assert ib.fresh_execution_slots().numel() == 0


def test_a_regranted_slot_is_folded_then_cleared_before_its_new_owners_forward(
    monkeypatch,
):
    """A retraction releases the victim's permit and the same plan hands it to
    another request: the handle folds the victim's records from it, then the
    executor's forward clears it for its new owner, then the forward reads it,
    all in that order on the execution stream."""
    scheduler = ts.Scheduler(_deferred_decode_config())
    executor, leaf, _, handle = _serving(monkeypatch, max_bs=1, residents=2)
    stream = executor.execution_stream
    admission = {"next": 1, "victim": 2}

    # "next" is admitted first, so the victim's request row is not its permit.
    _submit_pd(scheduler, "next", "victim")  # plans 1, 2
    _land(scheduler, "victim")
    first = _execute(handle, scheduler.next_execution_plan(), admission)  # 3
    assert list(first.execution_slots) == [1]
    assert list(first.request_pool_indices) == [2]
    assert list(first.execution_slot_fresh) == [True]
    _advance(scheduler, "victim", [43])
    _execute(handle, scheduler.next_execution_plan(), admission)  # 4
    _advance(scheduler, "victim", [44])
    _land(scheduler, "next")

    del leaf.trace[:]
    retract = scheduler.next_execution_plan()  # 5: the forced retraction
    assert scheduler.retracted_size() == 1
    op = _execute(handle, retract, admission)
    assert list(op.request_ids) == ["next"]
    assert list(op.execution_slots) == [1]
    assert list(op.execution_slot_fresh) == [True]
    torch.cuda.synchronize()

    kinds = [entry[0] for entry in leaf.trace]
    assert kinds == ["fold", "reset", "forward"]
    assert all(entry[-1] == stream for entry in leaf.trace)
    [(group, folds)] = leaf.trace[0][1].items()
    assert group == "state"
    [(request_row, _, execution_slot)] = folds
    assert (request_row, execution_slot) == (2, 1)
    assert leaf.trace[1][1].tolist() == [1]
    # Plan 3 cleared the previous owner's records and both decodes added one:
    # the fold read the victim's two, the new owner starts from none.
    assert [read.tolist() for read in leaf.folded] == [[2]]
    assert [read.tolist() for read in leaf.forward_read] == [[0], [1], [0]]


def test_graph_replays_and_recaptures_read_the_scheduled_slots_or_padding(
    monkeypatch,
):
    """Graphs recorded at buckets 1 and 3 read the bound execution slots: a
    full round its permits, a smaller round padded to the larger bucket its
    permits then padding, a DP-idle replay at the smaller bucket and every
    dummy forward of a capture -- at startup or a memory resume's recapture
    after serving -- padding only, with no fresh slot left over."""
    scheduler = ts.Scheduler(_config(Role.D, 3, 6))
    _submit_pd(scheduler, "a", "b", "c", "d", "e", "f")
    executor, leaf, runner, handle = _serving(
        monkeypatch, max_bs=3, residents=6, buckets=(1, 3)
    )
    ib = executor.input_buffers
    bound = ib.execution_slots_buf.data_ptr()
    padding = ib.execution_slot_padding
    admission = {rid: i + 1 for i, rid in enumerate("abcdef")}
    dummy_padding = [(1, [padding])] * 2 + [(3, [padding] * 3)] * 2

    executor.capture_graphs(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    assert runner.dummy_reads == dummy_padding

    # (a) A full round.
    _land(scheduler, "d", "e", "f")
    full = _execute(handle, scheduler.next_execution_plan(), admission)
    assert sorted(full.execution_slots) == [1, 2, 3]
    assert runner.last_replay() == (3, list(full.execution_slots))

    # (b) A smaller round, one of its rows a fresh owner's, pads to bucket 3.
    for rid in full.request_ids:
        _advance(scheduler, rid, [43])
    kept = full.request_ids[2]
    for rid in full.request_ids[:2]:
        _finish(scheduler, rid)
    _land(scheduler, "a")
    smaller = _execute(handle, scheduler.next_execution_plan(), admission)
    assert sorted(smaller.request_ids) == sorted([kept, "a"])
    slots = list(smaller.execution_slots)
    assert runner.last_replay() == (3, slots + [padding])
    fresh = [slots[list(smaller.request_ids).index("a")]]
    assert ib.fresh_execution_slots().tolist() == fresh

    # (c) A DP-idle replay at the smaller bucket: every slot padding, the
    # served round's fresh slot forgotten.
    executor.execute_idle_forward(_idle_round())
    assert runner.last_replay() == (1, [padding])
    assert ib.execution_slots_buf.tolist() == [padding] * 3
    assert ib.req_pool_indices_buf[:1].tolist() == [0]
    assert ib.fresh_execution_slots().numel() == 0

    # Serving resumes: the scheduler's slots again.
    for rid in smaller.request_ids:
        _advance(scheduler, rid, [44])
    steady = _execute(handle, scheduler.next_execution_plan(), admission)
    assert sorted(steady.execution_slots) == sorted(slots)
    assert not any(steady.execution_slot_fresh)
    assert runner.last_replay() == (3, list(steady.execution_slots) + [padding])

    # (d) A recapture after serving -- the executor's forget step, then its
    # captures, as a memory resume and a forward reload run them -- while a
    # round's upload is still in flight on the execution stream behind other
    # work, a fresh slot staged: the reset lands after the upload and forgets
    # the fresh slot, so the dummy forwards read padding, not served permits.
    del runner.dummy_reads[:]
    # Warm the step's allocations on the current stream, as serving has: a
    # cold allocator's first cudaMalloc would itself wait for the device.
    executor.forget_served_inputs()
    torch.cuda.synchronize()
    with torch.cuda.stream(executor.execution_stream):
        torch.cuda._sleep(1_000_000_000)
        executor._fill_batch_inputs(smaller, sum(smaller.input_lengths))
    assert ib.fresh_execution_slot_count == 1
    executor.forget_served_inputs()
    # Not vacuous: the upload had not landed when the reset was issued.
    assert not executor.execution_stream.query()
    assert ib.fresh_execution_slots().numel() == 0
    executor.capture_graphs(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    assert runner.dummy_reads == dummy_padding
    assert ib.execution_slots_buf.tolist() == [padding] * 3

    # The handle's memory resume recaptures the same way, and the new graphs
    # replay the scheduler's slots from the same bound buffer.
    del runner.dummy_reads[:]
    handle._released_dp_projections = ()
    handle.finish_memory_resume()
    assert runner.dummy_reads == dummy_padding
    for rid in steady.request_ids:
        _advance(scheduler, rid, [45])
    resumed = _execute(handle, scheduler.next_execution_plan(), admission)
    assert sorted(resumed.execution_slots) == sorted(slots)
    assert runner.last_replay() == (3, list(resumed.execution_slots) + [padding])
    assert ib.execution_slots_buf.data_ptr() == bound
    assert leaf.execution_slots.data_ptr() == bound


class _DeviceWork(NamedTuple):
    kernels: int
    host_to_device: int
    # Kernels, copies and memsets alike.
    operations: int


def _device_work(fn) -> _DeviceWork:
    """The device work ``fn`` issues on the current stream, once warm. Call
    from ``_in_fresh_process`` only."""
    stream = torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.default_stream())
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = host_to_device = operations = 0
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        operations += 1
        if event.name.startswith("Memcpy HtoD"):
            host_to_device += 1
        elif not event.name.startswith(("Memcpy", "Memset")):
            kernels += 1
    return _DeviceWork(kernels, host_to_device, operations)


def _check_slots_ride_the_request_rows_upload(fresh: bool) -> None:
    scheduler = _decode_node_with_extra_slots()
    executor, leaf = _executor(max_bs=3, residents=6)
    ib = executor.input_buffers
    first = _forward(scheduler.next_execution_plan())
    for rid in first.request_ids:
        _advance(scheduler, rid, [43])
    op = first if fresh else _forward(scheduler.next_execution_plan())
    assert all(op.execution_slot_fresh) is fresh
    assert any(op.execution_slot_fresh) is fresh

    staged = []
    bulk = InputBuffers._bulk_pinned

    def counted(self, *specs):
        staged.append(specs)
        return bulk(self, *specs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(InputBuffers, "_bulk_pinned", counted)
        with torch.cuda.stream(executor.execution_stream):
            work = _device_work(
                lambda: executor._fill_batch_inputs(op, sum(op.input_lengths))
            )

    num_fresh = 2 if fresh else 0
    expected = [((num_fresh + 3 + 2, torch.int64), (2, torch.int32))] * 2
    assert staged == expected, staged
    assert work.kernels > 0, work
    scrub_registers = 0 if ib._pad_tape is None else 1
    assert work.host_to_device == 3 + scrub_registers, work
    slots = ib.execution_slots_buf.tolist()
    assert slots == list(op.execution_slots) + [4], slots
    resets = _resets(leaf)
    assert resets == ([list(op.execution_slots)] * 2 if fresh else []), resets


@pytest.mark.parametrize("fresh", [True, False])
def test_slots_ride_the_request_rows_upload(fresh):
    """One pinned staging allocation and one H2D carry the fresh slots, every
    row's execution slot with its padding, and the request rows: the fill
    issues the copies it did before execution slots (the rows, the
    state-write rows, the lengths, the padding scrub's registers)."""
    _in_fresh_process("_check_slots_ride_the_request_rows_upload", fresh)


def _check_a_batch_without_fresh_rows_resets_nothing() -> None:
    scheduler = _decode_node_with_extra_slots()
    executor, leaf = _executor(max_bs=3, residents=6)
    ib = executor.input_buffers
    first = _forward(scheduler.next_execution_plan())
    _fill(executor, first)
    for rid in first.request_ids:
        _advance(scheduler, rid, [43])
    op = _forward(scheduler.next_execution_plan())
    assert not any(op.execution_slot_fresh)
    assert list(op.execution_slots) == list(first.execution_slots)
    del leaf.trace[:]

    total = sum(op.input_lengths)
    with torch.cuda.stream(executor.execution_stream):
        staged = _device_work(lambda: executor._fill_batch_inputs(op, total))
        fill = _device_work(
            lambda: ib.fill_input_buffers(op, executor.runtime_states, total)
        )

    assert leaf.trace == [], leaf.trace
    assert ib.fresh_execution_slots().numel() == 0
    assert fill.kernels > 0, fill
    assert staged == fill, (staged, fill)


def test_a_batch_without_fresh_rows_resets_nothing():
    """Steady decode rounds call no reset and launch nothing beyond the fill."""
    _in_fresh_process("_check_a_batch_without_fresh_rows_resets_nothing")


def _check_a_dummy_fill_pads_every_slot(bucket: int) -> None:
    scheduler = _decode_node_with_extra_slots()
    executor, _ = _executor(max_bs=3, residents=6)
    ib = executor.input_buffers
    padding = ib.execution_slot_padding
    served = _forward(scheduler.next_execution_plan())
    assert all(served.execution_slot_fresh)

    for rows in (bucket, 0):
        _fill(executor, served)
        assert ib.fresh_execution_slots().numel() == 2
        ib.fill_dummy_decode_buffers(batch_size=rows, total_tokens=rows)
        torch.cuda.synchronize()
        slots = ib.execution_slots_buf.tolist()
        assert slots == [padding] * 3, (rows, slots)
        assert ib.req_pool_indices_buf[:rows].tolist() == [0] * rows
        assert ib.fresh_execution_slots().numel() == 0

    def request_rows_zeroed(self, batch_size):
        if batch_size > 0:
            self.req_pool_indices_buf[:batch_size].fill_(0)

    with torch.cuda.stream(executor.execution_stream):
        dummy = _device_work(
            lambda: ib.fill_dummy_decode_buffers(batch_size=bucket, total_tokens=bucket)
        )
        idle = _device_work(
            lambda: ib.fill_dummy_decode_buffers(batch_size=0, total_tokens=0)
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(InputBuffers, "_reset_row_slots", request_rows_zeroed)
            before = _device_work(
                lambda: ib.fill_dummy_decode_buffers(
                    batch_size=bucket, total_tokens=bucket
                )
            )

    assert before.operations > 0, before
    assert dummy.operations == before.operations, (dummy, before)
    assert idle.operations == 0, idle


@pytest.mark.parametrize("bucket", [1, 3])
def test_a_dummy_fill_pads_every_slot_in_the_request_row_launch(bucket):
    """A capture's or DP-idle replay's dummy fill after a served round, at a
    bucket of any size: every execution slot padding and no fresh slot left,
    for the device work of zeroing the request rows alone (what the fill did
    before execution slots); with no row and nothing served, no work."""
    _in_fresh_process("_check_a_dummy_fill_pads_every_slot", bucket)


def _ordinary_bytes() -> int:
    """Bytes held by ordinary Torch CUDA allocator segments on this device."""
    return sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == torch.cuda.current_device()
        and tuple(segment["segment_pool_id"]) == (0, 0)
    )


def _check_slots_survive_release_and_resume() -> None:
    """Three rounds of serving, each released and resumed through the
    handle's weights lifecycle, then served on the graphs recorded before the
    release and recaptured as a memory resume does."""
    saver = TorchMemorySaverAdapter.create(enable=True, device="cuda")
    scheduler = ts.Scheduler(_config(Role.D, 3, 6))
    _submit_pd(scheduler, "a", "b", "c", "d", "e", "f")
    admission = {rid: i + 1 for i, rid in enumerate("abcdef")}
    with pytest.MonkeyPatch.context() as monkeypatch:
        executor, leaf, runner, handle = _serving(
            monkeypatch, max_bs=3, residents=6, buckets=(1, 3), saver=saver
        )
        ib = executor.input_buffers
        padding = ib.execution_slot_padding
        template = [padding] * 3 + [0] * 3
        dummy_padding = [(1, [padding])] * 2 + [(3, [padding] * 3)] * 2
        # The row-slot vectors the graphs and the backends read, and the
        # dummy forwards' template: persistent, in the executor_inputs region.
        persistent = (
            ib.fresh_execution_slots_buf,
            ib.execution_slots_buf,
            ib.req_pool_indices_buf,
            ib._dummy_row_slots,
        )
        pointers = [tensor.data_ptr() for tensor in persistent]
        persistent_bytes = sum(
            storage.nbytes()
            for storage in {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage()
                for tensor in persistent
            }.values()
        )
        assert leaf.execution_slots.data_ptr() == pointers[1]

        executor.capture_graphs(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
        assert runner.dummy_reads == dummy_padding, runner.dummy_reads
        _land(scheduler, "d", "e", "f")
        live = list(
            _execute(handle, scheduler.next_execution_plan(), admission).request_ids
        )
        for cycle, new in enumerate("abc"):
            # A served round padded to bucket 3, a fresh owner on the permit
            # of a request that finished: its slots stay in the buffers.
            for rid in live:
                _advance(scheduler, rid, [43])
            for rid in live[:-1]:
                _finish(scheduler, rid)
            _land(scheduler, new)
            served = _execute(handle, scheduler.next_execution_plan(), admission)
            live = list(served.request_ids)
            assert len(live) == 2 and new in live, live
            slots = list(served.execution_slots)
            rows = list(served.request_pool_indices)
            fresh = [slots[live.index(new)]]
            assert list(compress(slots, served.execution_slot_fresh)) == fresh
            replay = runner.last_replay()
            assert replay == (3, slots + [padding]), (cycle, replay)
            assert ib.fresh_execution_slots().tolist() == fresh
            assert ib.req_pool_indices_buf[:2].tolist() == rows
            contents = [tensor.tolist() for tensor in persistent]
            assert contents[1:] == [slots + [padding], rows + [0], template], contents

            # Release: no ordinary allocator segment stays occupied.
            del runner.replays[:], leaf.trace[:], leaf.folded[:], leaf.forward_read[:]
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free_before = torch.cuda.mem_get_info()[0]
            handle.release_weights(saver)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            freed = torch.cuda.mem_get_info()[0] - free_before
            assert freed >= persistent_bytes, (cycle, freed, persistent_bytes)
            ordinary_bytes = _ordinary_bytes()
            assert ordinary_bytes == 0, (cycle, ordinary_bytes)
            handle.resume_weights(saver)

            # Resume: the same addresses, the same slots and template.
            resumed = [tensor.data_ptr() for tensor in persistent]
            assert resumed == pointers, (cycle, resumed, pointers)
            restored = [tensor.tolist() for tensor in persistent]
            assert restored == contents, (cycle, restored, contents)
            assert ib.fresh_execution_slots().tolist() == fresh

            # The graphs recorded before the release read the next round's
            # packed upload from the restored buffer.
            for rid in live:
                _advance(scheduler, rid, [44])
            steady = _execute(handle, scheduler.next_execution_plan(), admission)
            assert sorted(steady.execution_slots) == sorted(slots)
            assert not any(steady.execution_slot_fresh)
            replay = runner.last_replay()
            assert replay == (3, list(steady.execution_slots) + [padding]), replay
            assert ib.fresh_execution_slots().numel() == 0
            live = list(steady.request_ids)

            # A DP-idle replay at bucket 1 resets from the restored template.
            executor.execute_idle_forward(_idle_round())
            replay = runner.last_replay()
            assert replay == (1, [padding]), (cycle, replay)
            assert ib.execution_slots_buf.tolist() == [padding] * 3
            assert ib.req_pool_indices_buf[:1].tolist() == [0]
            assert ib.fresh_execution_slots().numel() == 0

            # The memory resume's recapture: padding in every dummy forward;
            # the next cycle's round replays the new graphs.
            del runner.dummy_reads[:]
            handle._released_dp_projections = ()
            handle.finish_memory_resume()
            assert runner.dummy_reads == dummy_padding, (cycle, runner.dummy_reads)
            assert ib.execution_slots_buf.tolist() == [padding] * 3
            assert ib._dummy_row_slots.tolist() == template
            assert [tensor.data_ptr() for tensor in persistent] == pointers
            assert leaf.execution_slots.data_ptr() == pointers[1]


@pytest.mark.skipif(
    torch.version.hip is not None,
    reason="requires native CUDA VMM memory-saver hooks",
)
def test_slots_survive_release_and_resume():
    """The packed row slots and the dummy template live in the executor's
    CPU-backed named region: a weights release unmaps them with no ordinary
    allocator segment left occupied, and every resume restores them at the
    same addresses, so graphs recorded before the release, the next round's
    upload, a DP-idle reset and a recapture all read what they did."""
    memory_saver = pytest.importorskip("torch_memory_saver")
    # Preload hooks must be installed before CUDA initializes in the child.
    with memory_saver.configure_subprocess():
        _in_fresh_process("_check_slots_survive_release_and_resume")
