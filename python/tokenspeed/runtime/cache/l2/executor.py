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

"""Descriptor-driven executor for compact Host cache transfers, two tiers.

One executor, two pinned Host buffers over the same field geometry:

* the **L2 prefix tier** (``--enable-kvstore``): hash-complete prefix pages the
  scheduler publishes (``--kvstore-write-policy write_through``) or evicts
  (``write_back``), stored by ``WriteBackOp``, loaded by ``LoadBackOp`` with
  layerwise consumer fences, optionally backed by L3;
* the **retraction snapshot pool** (on by default where the engine retracts;
  sized by ``tokenspeed.runtime.cache.l2.sizing``): the request-private tail
  of a retracted request's image -- its unaligned tail
  pages, every block of a group L2 never holds, and its slot-state blob --
  stored by ``SnapshotOp`` and read back, together with the request's pinned
  L2 entries, by one ``RestoreOp`` whose rows name their source tier.

L3 is an independent Host-only executor. The copy ACK publishes L2 and
releases its Device source without waiting for a remote PUT; C++ then owns
backup admission and source references.

A retraction image is split between the two by publishability (the
scheduler's ``TakeImage``); the executor only chooses a buffer per row. Both
legs of a store ride the write stream under one fence on the forward
thread's default stream, because the victim's Device pages are re-granted in
the same plan; a restore rides the load stream after the plan's zeroing and
is acknowledged asynchronously, nothing in the round reading its pages. Every
row of every op passes one ownership translation on both ends (``ownership.py``),
so under KVP each rank copies exactly the blocks it owns.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import NamedTuple

import torch
from tokenspeed_kernel.ops.kvcache.host_transfer import (
    HostTransferWorkspace,
    transfer_cache_blocks,
    wait_layer_ready,
)
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.host_storage import (
    HostCacheStorage,
    compute_host_group_packing,
    compute_host_lcm_block_bytes,
    validate_host_group_packing,
)
from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
from tokenspeed.runtime.cache.l2.sizing import (
    RetractionPoolRequest,
    gigabytes_to_lcm_blocks,
    resolve_retraction_pool,
)
from tokenspeed.runtime.cache.transfer.lanes import (
    CompletionQueue,
    HostTransferLane,
    build_transfer_geometry,
    check_host_memory,
    host_memory_share,
    largest_pinned_fit,
    load_stream_priority,
    new_cache_stream,
    pinned_host_bytes,
)
from tokenspeed.runtime.cache.transfer.layout import combine_cache_transfer_layouts
from tokenspeed.runtime.cache.transfer.ops import (
    HostTier,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation
from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.execution.slot_state import SlotStateExporter, SlotStateLayout
from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()

_HOST_MEM_HEADROOM_BYTES = 10 * (1024**3)


def _ordered_unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


class _AckKind(Enum):
    """Which scheduler ACK an in-flight copy's op ids become on completion."""

    WRITE_BACK = "WriteBackDone"
    LOAD_BACK = "LoadBackDone"
    SNAPSHOT = "SnapshotDone"
    RESTORE = "RestoreDone"


class _Ack(NamedTuple):
    """The scheduler ACK kind and IDs of an in-flight Host copy."""

    kind: _AckKind
    op_ids: list[int]


def _num_host_lcm_blocks(
    *,
    host_lcm_block_bytes: int,
    device_lcm_blocks: int,
    host_ratio: float,
    host_size_gb: float,
) -> int:
    if host_size_gb > 0:
        count = gigabytes_to_lcm_blocks(
            host_size_gb, host_lcm_block_bytes=host_lcm_block_bytes
        )
    else:
        count = int(device_lcm_blocks * host_ratio)
    if count <= 0:
        raise ValueError("Host L2 resolved to zero LCM blocks")
    return count


def _min_over_ranks(value: int, cpu_group) -> int:
    """``value``'s minimum over the gloo ``cpu_group``; ``value`` without one."""
    if cpu_group is None:
        return value
    tensor = torch.tensor(value, dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
    )
    return int(tensor.item())


def allocate_blob_arena(rows: int, nbytes: int) -> torch.Tensor:
    """The pinned ``[rows, nbytes]`` uint8 slot-state arena."""
    return torch.empty((rows, nbytes), dtype=torch.uint8, pin_memory=nbytes > 0)


class HostCacheExecutor:
    """Execute group-aware D2H/H2D operations against the two compact Host tiers."""

    def __init__(
        self,
        device_pool,
        *,
        draft_pool=None,
        l2_tier: bool,
        host_ratio: float,
        host_size_gb: float,
        snapshot_pool: RetractionPoolRequest,
        slot_state_exporters: Sequence[SlotStateExporter] | None,
        io_backend: str,
        attn_tp_rank: int,
        kvp_rank: int,
        nprocs_per_node: int,
        world_cpu_group,
    ):
        """
        Args:
            device_pool: The target cache pool; its arena publishes the
                transfer layout, the runtime contract and the group order.
            draft_pool: The draft pool sharing the arena, or None.
            l2_tier: Whether to allocate the L2 prefix tier
                (``--enable-kvstore``). Without it ``WriteBackOp`` /
                ``LoadBackOp`` are refused and a retraction image lives
                wholly in the snapshot pool.
            host_ratio: L2 LCM blocks as a multiple of the Device's, when
                ``host_size_gb`` is 0; capped to what fits in every rank's
                share of its node's Host memory.
            host_size_gb: L2 size in decimal gigabytes; overrides the ratio
                when positive and is never capped: one that does not fit is
                refused.
            snapshot_pool: The retraction snapshot pool's knobs as
                ``ServerArgs`` resolved them, sized here against this
                rank's layout (``resolve_retraction_pool``); a disabled
                request builds no pool (``SnapshotOp`` / ``RestoreOp`` are
                then refused).
            slot_state_exporters: The owners of per-slot state outside the
                cache groups, in blob order (``ModelExecutor.
                slot_state_exporters``); laid out once here into the arena
                row. May be None only without a pool.
            io_backend: ``"direct"`` (DMA ranges) or ``"kernel"`` (mapped-Host
                Triton copies).
            attn_tp_rank: Attention-TP rank; rank 0 logs.
            kvp_rank: This rank in the KVP subgroup; selects the blocks it
                owns of a sharded group (the identity for replicated groups).
            nprocs_per_node: The ranks on this node. They all pin their Host
                tiers here at once, so each checks against an equal share
                of the Host memory the node has available.
            world_cpu_group: The gloo group of every rank, over which the Host
                tier sizes are agreed and compared; None with a single rank.
        """
        if io_backend not in ("direct", "kernel"):
            raise ValueError(f"unsupported KVStore IO backend {io_backend!r}")
        if not l2_tier and snapshot_pool.disabled:
            raise ValueError(
                "a Host cache executor needs the L2 tier, a snapshot pool or both"
            )
        if not snapshot_pool.disabled and slot_state_exporters is None:
            raise ValueError("a snapshot pool needs the slot-state exporters")
        self.attn_tp_rank = attn_tp_rank
        self.transfer_backend = "dma" if io_backend == "direct" else "auto"
        target_layout = device_pool.cache_transfer_layout()
        draft_layout = (
            draft_pool.cache_transfer_layout() if draft_pool is not None else None
        )
        contract = device_pool.arena.runtime_contract
        scheduler_group_ids = tuple(spec.group_id for spec in contract.group_specs)
        self.layout = combine_cache_transfer_layouts(
            target_layout,
            draft_layout,
            group_ids=scheduler_group_ids or None,
        )
        host_lcm_block_bytes = compute_host_lcm_block_bytes(self.layout)
        shard_by_id = {
            spec.group_id: int(spec.shard_count) for spec in contract.group_specs
        }
        self.host_group_packing = tuple(
            packing * shard_by_id[group.group_id]
            for group, packing in zip(
                self.layout.groups,
                compute_host_group_packing(self.layout),
                strict=True,
            )
        )

        # --- L2 prefix tier -------------------------------------------------
        host_lcm_blocks = 0
        if l2_tier:
            host_lcm_blocks = _num_host_lcm_blocks(
                host_lcm_block_bytes=host_lcm_block_bytes,
                device_lcm_blocks=self.layout.num_lcm_blocks,
                host_ratio=host_ratio,
                host_size_gb=host_size_gb,
            )
        # --- retraction snapshot pool -----------------------------------------
        self.snapshot_sizing = resolve_retraction_pool(
            snapshot_pool,
            l2_tier=l2_tier,
            device_lcm_blocks=self.layout.num_lcm_blocks,
            host_lcm_block_bytes=host_lcm_block_bytes,
        )
        snapshot_lcm_blocks = self.snapshot_sizing.lcm_blocks
        self.blob_bytes = 0
        # The blob layout is measured once here; every store and restore
        # slices the arena row at its recorded offsets.
        self._slot_state: SlotStateLayout | None = None
        if snapshot_lcm_blocks:
            self._slot_state = SlotStateLayout(slot_state_exporters)
            self.blob_bytes = self._slot_state.nbytes
        self.max_retracted_requests = self.snapshot_sizing.max_retracted_requests
        snapshot_bytes = snapshot_lcm_blocks * host_lcm_block_bytes
        arena_bytes = (self.max_retracted_requests + 1) * self.blob_bytes
        share_bytes = host_memory_share(
            nprocs_per_node=nprocs_per_node, headroom_bytes=_HOST_MEM_HEADROOM_BYTES
        )
        # Size each buffer using the effective pinned allocator configuration.
        pool_pinned_bytes = pinned_host_bytes(snapshot_bytes) + pinned_host_bytes(
            arena_bytes
        )
        # The L2 blocks every rank's share holds beside the pool. Agreeing on
        # it is also a barrier: no rank pins before every rank read its share.
        fitting_l2_blocks = _min_over_ranks(
            largest_pinned_fit(
                share_bytes - pool_pinned_bytes, block_bytes=host_lcm_block_bytes
            )
            // host_lcm_block_bytes,
            world_cpu_group,
        )
        if l2_tier and host_size_gb <= 0 and 0 < fitting_l2_blocks < host_lcm_blocks:
            # Only the ratio's size shrinks, to one size on every rank: the
            # mirrored schedulers share one Host page count. An explicit
            # --kvstore-size that does not fit is refused below instead.
            logger.warning(
                "Capped the compact Host L2 from "
                f"{host_lcm_blocks * host_lcm_block_bytes / 1e9:.2f} GB to "
                f"{fitting_l2_blocks * host_lcm_block_bytes / 1e9:.2f} GB per "
                "rank, the most every rank's share of its node holds: "
                f"--kvstore-ratio {host_ratio} asks for more. The "
                f"{nprocs_per_node} ranks of a node pin Host memory together, "
                "under the pinned allocator settings; this rank's share "
                f"is {share_bytes / 1e9:.2f} GB (the node's available Host "
                "memory over its ranks, less "
                f"{_HOST_MEM_HEADROOM_BYTES / 1e9:.2f} GB headroom each)"
            )
            host_lcm_blocks = fitting_l2_blocks
        # The ranks' mirrored schedulers share one page count per Host tier,
        # so every rank must resolve the same counts. A size in gigabytes
        # resolves to different counts where the ranks' Host LCM blocks differ
        # in bytes, as they can on pipeline stages, which store different
        # layers. The comparison is a collective, so it runs before
        # check_host_memory, whose rank-local refusal would otherwise leave
        # the other ranks waiting in it; every rank raises together.
        for name, count in (
            ("Host L2 LCM blocks", host_lcm_blocks),
            ("snapshot pool LCM blocks", snapshot_lcm_blocks),
            ("snapshot pool slot-state rows", self.max_retracted_requests),
        ):
            smallest = _min_over_ranks(count, world_cpu_group)
            largest = -_min_over_ranks(-count, world_cpu_group)
            if smallest != largest:
                raise ValueError(
                    f"the ranks size the Host tiers differently: {smallest} to "
                    f"{largest} {name} (this rank's Host LCM block is "
                    f"{host_lcm_block_bytes} bytes). The mirrored schedulers "
                    "need the same page counts on every rank: size Host L2 with "
                    "--kvstore-ratio, which every rank resolves to one count, "
                    "rather than --kvstore-size, and the snapshot pool with "
                    "--retraction-snapshot-ratio rather than "
                    "--retraction-snapshot-host-gb"
                )
        validate_host_group_packing(
            self.host_group_packing,
            cpu_group=world_cpu_group,
        )
        l2_bytes = host_lcm_blocks * host_lcm_block_bytes
        # Both tiers count against the one check, at the sizes they pin.
        l2_pinned_bytes = pinned_host_bytes(l2_bytes)
        check_host_memory(
            l2_pinned_bytes + pool_pinned_bytes,
            share_bytes=share_bytes,
            purpose="the compact Host cache",
        )
        logger.info(
            f"Host pinned memory budget: L2 payload={l2_bytes / 1e9:.2f} GB, "
            f"L2 reservation={l2_pinned_bytes / 1e9:.2f} GB, "
            f"snapshot reservation={pool_pinned_bytes / 1e9:.2f} GB, "
            f"rank share={share_bytes / 1e9:.2f} GB"
        )

        self.host_storage: HostCacheStorage | None = None
        self.block_owners: BlockOwnerTranslation | None = None
        self._transfer_geometry = None
        if l2_tier:
            self.host_storage = HostCacheStorage(
                self.layout, num_host_lcm_blocks=host_lcm_blocks
            )
            # Scheduler block ids on both ends of every L2 copy to this rank's
            # local ids.
            self.block_owners = BlockOwnerTranslation.for_host_pool(
                self.layout,
                contract,
                num_host_lcm_blocks=host_lcm_blocks,
                host_group_packing=self.host_storage.group_packing,
                rank=kvp_rank,
            )
        # The scheduler wire includes logical null LCMBlock 0 in its counts;
        # 0 L2 pages means no L2 tier, 1 snapshot page means no pool (nothing
        # can be imaged; a capacity-blocked round aborts a resident instead).
        self.num_host_pages = host_lcm_blocks + 1 if l2_tier else 0
        self.num_snapshot_pages = snapshot_lcm_blocks + 1
        self.snapshot_storage: HostCacheStorage | None = None
        self.snapshot_block_owners: BlockOwnerTranslation | None = None
        self._snapshot_geometry = None
        self.blob_arena: torch.Tensor | None = None
        if snapshot_lcm_blocks:
            self.snapshot_storage = HostCacheStorage(
                self.layout, num_host_lcm_blocks=snapshot_lcm_blocks
            )
            self.snapshot_block_owners = BlockOwnerTranslation.for_host_pool(
                self.layout,
                contract,
                num_host_lcm_blocks=snapshot_lcm_blocks,
                host_group_packing=self.snapshot_storage.group_packing,
                rank=kvp_rank,
            )
            # One row per retracted request, indexed by the op's snapshot_slot.
            # The scheduler numbers slots from 1, like request-pool indices.
            self.blob_arena = allocate_blob_arena(
                self.max_retracted_requests + 1, self.blob_bytes
            )
        if l2_tier:
            logger.info(
                f"Allocated {l2_bytes / 1000000000.0:.2f} GB compact Host L2 ("
                f"{host_lcm_blocks!s} LCM blocks, {host_lcm_block_bytes!s} bytes/block)",
            )
        if attn_tp_rank == 0:
            logger.info(
                self.snapshot_sizing.describe(
                    host_lcm_block_bytes=host_lcm_block_bytes,
                    blob_bytes=self.blob_bytes,
                    l2_tier=l2_tier,
                )
            )

        # Layerwise load fences exist for L2 prefix loads only: a restore's
        # pages are read by nothing in the round, and without an L2 tier no
        # load-back can occur, so the pools keep no tracker.
        self._load_trackers = []
        if l2_tier:
            pool_layouts = [(device_pool, target_layout)]
            if draft_pool is not None and self.layout is not target_layout:
                pool_layouts.append((draft_pool, draft_layout))
            for pool, layout in pool_layouts:
                tracker = LayerwiseLoadTracker(len(layout.consumers))
                pool.register_layerwise_load_tracker(tracker)
                self._load_trackers.append((tracker, len(layout.consumers)))
        # Every copy runs on its own stream, ordered after the prerequisite
        # stream the caller names per submission: for a store the one the
        # forwards wrote the source pages on, for a load the one that zeroed
        # the destination pages. What differs per store is who waits on the
        # copy: a stream-ordered op (a retraction image's L2 leg, every
        # snapshot store, and under the Host write-back policy the store of
        # what the plan's admissions and restores evicted: sources this very
        # plan may re-grant) fences the fence stream the caller names on its
        # completion, so the plan's copies, zeroing, load-backs, restores and
        # forwards stay behind it; a pinned op (an ordinary publication under
        # write-through, whose sources the scheduler holds until the ACK)
        # fences nothing and never holds up the round. An L2
        # load's consumers are fenced per layer by the tracker events; a
        # restore's by its ACK.
        self.write_stream = new_cache_stream(None)
        self.load_stream = new_cache_stream(load_stream_priority())
        # Both streams consume these immutable tables; the kernel backend
        # publishes them to the Device once. One geometry per Host buffer.
        if l2_tier:
            self._transfer_geometry = build_transfer_geometry(
                self.layout, self.host_storage, io_backend=io_backend
            )
        if snapshot_lcm_blocks:
            self._snapshot_geometry = build_transfer_geometry(
                self.layout, self.snapshot_storage, io_backend=io_backend
            )
        # One staging lane per submission kind a round may make, so none
        # waits on another's pinned metadata tables: the stream-ordered and
        # pinned L2 write-backs, the snapshot store, and the restore's two
        # tiers.
        self._ordered_write_lane = HostTransferLane()
        self._pinned_write_lane = HostTransferLane()
        self._snapshot_write_lane = HostTransferLane()
        self._restore_l2_lane = HostTransferLane()
        self._restore_pool_lane = HostTransferLane()
        # A tracker waits for an event set's previous final-layer event before
        # reusing its index. Aligning workspaces to those indices keeps each
        # load's pinned and Device block-ID tables immutable until all
        # consumers of that table have completed.
        self._load_workspaces: tuple[HostTransferWorkspace, ...] = ()
        if self._load_trackers:
            load_workspace_count = len(self._load_trackers[0][0].event_sets)
            if any(
                len(tracker.event_sets) != load_workspace_count
                for tracker, _ in self._load_trackers
            ):
                raise RuntimeError("target and draft Host-load event sets diverged")
            self._load_workspaces = tuple(
                HostTransferWorkspace() for _ in range(load_workspace_count)
            )

        # Submission runs on the forward thread and polling on the control
        # plane (event queries only). Every copy of every kind -- L2
        # write-back and load-back, snapshot store and restore -- is one
        # entry of the one completion queue, released as the ACK its _Ack
        # names once its event completes; the queue is the cross-thread
        # handoff. Loads with nothing owned to copy have no event and are
        # released from ``_ready_load_acks``; the lock protects that handoff.
        self._completions = CompletionQueue()
        self._ack_lock = threading.Lock()
        self._load_poisoned = False
        self._ready_load_acks: list[int] = []

        # Resolve each buffer binding on its eventual copy stream at startup,
        # so the first serving transfer only uploads its per-operation rows.
        for storage, write_workspaces, load_workspaces in (
            (
                self.host_storage,
                (
                    self._ordered_write_lane.workspace,
                    self._pinned_write_lane.workspace,
                ),
                (self._restore_l2_lane.workspace, *self._load_workspaces),
            ),
            (
                self.snapshot_storage,
                (self._snapshot_write_lane.workspace,),
                (self._restore_pool_lane.workspace,),
            ),
        ):
            if storage is None:
                continue
            for stream, workspaces in (
                (self.write_stream, write_workspaces),
                (self.load_stream, load_workspaces),
            ):
                with device_module.stream(stream):
                    for workspace in workspaces:
                        workspace.prepare_backend(
                            self.layout.buffers,
                            storage.host_buffer,
                            backend=self.transfer_backend,
                        )
        self.write_stream.synchronize()
        self.load_stream.synchronize()

    def submit_write_backs(
        self, cache_ops: Sequence, *, prerequisite_stream, fence_stream
    ) -> None:
        """Enqueue the plan's D2H copies on the write stream.

        Must run BEFORE the plan's Device-to-Device copies and page zeroing.
        Every copy is ordered behind ``prerequisite_stream`` -- here the
        stream the forwards wrote the source pages on -- so it reads their
        final bytes. The scheduler marks each L2 op ``source_pinned``: a
        pinned op's sources stay cached and unevictable until the ACK, so its
        copy rides the write stream and nobody waits on it; an unpinned op's
        sources may already be granted to another request in this very plan,
        so it goes first and ``fence_stream`` waits on its completion -- the
        plan's copies, zeroing, load-backs, restores and forwards are ordered
        behind that wait. A retraction image's snapshot leg (``SnapshotOp``:
        tail pages into the pool, then the victim's slot state into the
        arena) follows the stream-ordered L2 rows on the same stream, so the
        one fence covers both legs.

        Args:
            cache_ops: The round's cache ops as ``scheduler_utils.
                cache_ops_from_plan`` adapts them; the ``Cache.WriteBackOp``
                batches and the per-request ``SnapshotOp`` entries are read
                here, the rest belongs to ``submit_load_backs``.
            prerequisite_stream: The stream whose completed work every copy
                must observe -- the model executor's execution stream, where
                the forwards wrote the source pages and the slot state.
            fence_stream: The stream a stream-ordered op's completion fences
                -- the one the plan's Device-to-Device copies and page zeroing
                run on next.
        """
        ordered_op_ids: list[int] = []
        ordered_transfers: list[tuple[int, int, int]] = []
        pinned_op_ids: list[int] = []
        pinned_transfers: list[tuple[int, int, int]] = []
        snapshot_ops: list[SnapshotOp] = []
        snapshot_transfers: list[tuple[int, int, int]] = []
        # The whole plan is validated and translated before the first launch:
        # a bad op must not leave the ordered rows in flight without the
        # fence that follows them.
        for operation in cache_ops:
            if isinstance(operation, Cache.WriteBackOp):
                self._append_write_backs(
                    operation,
                    ordered_op_ids=ordered_op_ids,
                    ordered_transfers=ordered_transfers,
                    pinned_op_ids=pinned_op_ids,
                    pinned_transfers=pinned_transfers,
                )
            elif isinstance(operation, SnapshotOp):
                snapshot_ops.append(operation)
                self._append_snapshot_store(operation, transfers=snapshot_transfers)
        self._check_distinct(snapshot_ops)
        fence = None
        try:
            fence = self._start_writing(
                ordered_op_ids,
                ordered_transfers,
                lane=self._ordered_write_lane,
                prerequisite_stream=prerequisite_stream,
            )
            if snapshot_ops:
                # Recorded after the ordered L2 rows on the same stream, so
                # waiting on it waits on both legs.
                fence = self._start_snapshot_store(
                    snapshot_ops,
                    snapshot_transfers,
                    prerequisite_stream=prerequisite_stream,
                )
        except BaseException:
            # Whatever of the two legs launched is in flight on the write
            # stream without its event (an exporter raised mid-export, a lane
            # failed); fence the stream's tail anyway, or the plan's zeroing
            # could run over copy sources still being read.
            tail = device_module.Event()
            tail.record(self.write_stream)
            fence_stream.wait_event(tail)
            raise
        if fence is not None:
            fence_stream.wait_event(fence)
        self._start_writing(
            pinned_op_ids,
            pinned_transfers,
            lane=self._pinned_write_lane,
            prerequisite_stream=prerequisite_stream,
        )

    def submit_load_backs(self, cache_ops: Sequence, *, prerequisite_stream) -> None:
        """Launch the plan's H2D loads; runs after the plan's page zeroing.

        The device handle calls it on every plan, cache ops or none: it also
        points the layerwise load trackers at this plan's load, or at none,
        which is what the forward's layers and a prefill graph replay wait on.

        Every load-back row reads a Host L2 entry that is already there (an
        L3 prefetch, when one ran, filled and published it before the
        request was admitted), so a load cannot miss. A ``RestoreOp`` rides
        the same load stream with its rows split by source tier (the
        request's pinned L2 entries, the pool's tail pages), then its
        slot-state import; it arms no layerwise tracker, because the request
        is not schedulable until the ACK, and its destinations are
        overwritten whole rather than zeroed first.

        Args:
            cache_ops: The round's cache ops as ``scheduler_utils.
                cache_ops_from_plan`` adapts them; the ``Cache.LoadBackOp``
                batches and the per-request ``RestoreOp`` entries are read
                here.
            prerequisite_stream: The stream whose completed work every copy
                must observe -- the one the plan's page zeroing ran on (behind
                the store fence), so the loads land on zeroed destination
                pages and nothing reads a restored page before its image.
        """
        op_ids: list[int] = []
        transfers: list[tuple[int, int, int]] = []
        restore_ops: list[RestoreOp] = []
        restore_rows: dict[HostTier, list[tuple[int, int, int]]] = {
            HostTier.L2: [],
            HostTier.SNAPSHOT_POOL: [],
        }
        for operation in cache_ops:
            if isinstance(operation, Cache.LoadBackOp):
                self._append_transfers(
                    operation.op_ids,
                    operation.group_ids,
                    operation.src_pages,
                    operation.dst_pages,
                    collected_op_ids=op_ids,
                    transfers=transfers,
                    source_is_device=False,
                    tier=HostTier.L2,
                )
            elif isinstance(operation, RestoreOp):
                restore_ops.append(operation)
                self._append_restore(operation, rows_by_tier=restore_rows)
        self._check_distinct(restore_ops)
        # Loads first: the forward reads them layer by layer behind the
        # layerwise fences, while a restore is read by nothing this round, so
        # it rides the same stream behind them instead of delaying them.
        load_index = self._start_loading(
            op_ids, transfers, prerequisite_stream=prerequisite_stream
        )
        for tracker, _ in self._load_trackers:
            tracker.set_consumers(load_index if load_index is not None else -1)
        if restore_ops:
            self._start_restore(
                restore_ops, restore_rows, prerequisite_stream=prerequisite_stream
            )

    def _append_write_backs(
        self,
        operation,
        *,
        ordered_op_ids: list[int],
        ordered_transfers: list[tuple[int, int, int]],
        pinned_op_ids: list[int],
        pinned_transfers: list[tuple[int, int, int]],
    ) -> None:
        source_pinned = operation.source_pinned
        if len(source_pinned) != len(operation.op_ids):
            raise ValueError("ragged cache operation batch")
        for index, pinned in enumerate(source_pinned):
            op_ids, transfers = (
                (pinned_op_ids, pinned_transfers)
                if pinned
                else (ordered_op_ids, ordered_transfers)
            )
            self._append_transfers(
                operation.op_ids[index : index + 1],
                operation.group_ids[index : index + 1],
                operation.src_pages[index : index + 1],
                operation.dst_pages[index : index + 1],
                collected_op_ids=op_ids,
                transfers=transfers,
                source_is_device=True,
                tier=HostTier.L2,
            )

    def _owners(self, tier: HostTier) -> BlockOwnerTranslation:
        """The ownership translation of one Host tier; raises without the tier."""
        if tier is HostTier.L2:
            if self.block_owners is None:
                raise RuntimeError(
                    "cache op names Host L2 blocks but this engine has no L2 tier "
                    "(--disable-kvstore)"
                )
            return self.block_owners
        if self.snapshot_block_owners is None:
            raise RuntimeError(
                "cache op names snapshot-pool blocks but this engine has no "
                "retraction snapshot pool (--retraction-snapshot-ratio 0)"
            )
        return self.snapshot_block_owners

    def _owned_rows(
        self,
        op_id: int,
        groups: Sequence[int],
        sources: Sequence[int],
        destinations: Sequence[int],
        *,
        source_is_device: bool,
        tier: HostTier,
    ) -> list[tuple[int, tuple[int, int, int]]]:
        """One op's wire rows through the ownership translation of ``tier``.

        The one place rows become ``(group_index, device_block, host_block)``:
        every row passes the translation on both ends, so a sharded group's
        row is kept only by the rank that owns it (the scheduler pairs blocks
        of equal residue) and a replicated group's row translates to itself.

        Returns:
            ``(position, local_row)`` for the rows this rank keeps, with
            ``position`` indexing the op's wire rows.
        """
        if not (len(groups) == len(sources) == len(destinations)):
            raise ValueError(f"ragged cache operation {op_id}")
        rows = []
        for group, source, destination in zip(groups, sources, destinations):
            device_block_id, host_block_id = (
                (source, destination) if source_is_device else (destination, source)
            )
            rows.append((int(group), int(device_block_id), int(host_block_id)))
        return self._owners(tier).owned_positions(rows)

    def _append_transfers(
        self,
        operation_ids: Sequence[int],
        group_ids: Sequence[Sequence[int]],
        src_blocks: Sequence[Sequence[int]],
        dst_blocks: Sequence[Sequence[int]],
        *,
        collected_op_ids: list[int],
        transfers: list[tuple[int, int, int]],
        source_is_device: bool,
        tier: HostTier,
    ) -> None:
        """Collect one batch's ops and this rank's local block triples.

        Every row passes :meth:`_owned_rows`. An op whose every row belongs
        to other ranks is still collected: this rank acknowledges it from an
        empty copy.

        """
        if not (
            len(operation_ids) == len(group_ids) == len(src_blocks) == len(dst_blocks)
        ):
            raise ValueError("ragged cache operation batch")
        for op_id, groups, sources, destinations in zip(
            operation_ids, group_ids, src_blocks, dst_blocks
        ):
            # An op with no rows at all is a malformed plan: the scheduler
            # would hold a ticket for a copy nobody performs.
            if not groups:
                raise ValueError(f"cache operation {op_id} carries no transfers")
            kept = self._owned_rows(
                op_id,
                groups,
                sources,
                destinations,
                source_is_device=source_is_device,
                tier=tier,
            )
            collected_op_ids.append(int(op_id))
            transfers.extend(row for _, row in kept)

    def _check_snapshot_op(self, op: SnapshotOp | RestoreOp) -> None:
        if self.snapshot_storage is None:
            raise RuntimeError(
                f"{type(op).__name__} {op.op_id} needs the retraction snapshot pool, "
                "which --retraction-snapshot-ratio 0 removed"
            )
        if not 1 <= int(op.snapshot_slot) <= self.max_retracted_requests:
            raise IndexError(
                f"snapshot slot {op.snapshot_slot} outside "
                f"[1, {self.max_retracted_requests}] for op {op.op_id}"
            )

    def _append_snapshot_store(
        self, op: SnapshotOp, *, transfers: list[tuple[int, int, int]]
    ) -> None:
        """A store's tail rows. Empty transfers are legal: every page of the
        victim went to L2 and the image is its slot state alone."""
        self._check_snapshot_op(op)
        if not op.transfers:
            return
        ignored: list[int] = []
        self._append_transfers(
            [op.op_id],
            [[t.group_id for t in op.transfers]],
            [[t.source_page for t in op.transfers]],
            [[t.destination_page for t in op.transfers]],
            collected_op_ids=ignored,
            transfers=transfers,
            source_is_device=True,
            tier=HostTier.SNAPSHOT_POOL,
        )

    def _append_restore(
        self, op: RestoreOp, *, rows_by_tier: dict[HostTier, list[tuple[int, int, int]]]
    ) -> None:
        """A restore's rows, split by the Host tier each one reads. Empty
        transfers are legal, mirroring the store: the image is its slot state
        alone."""
        self._check_snapshot_op(op)
        if len(op.source_tier) != len(op.transfers):
            raise ValueError(f"ragged cache operation {op.op_id}")
        for tier in (HostTier.L2, HostTier.SNAPSHOT_POOL):
            rows = [
                transfer
                for transfer, source_tier in zip(op.transfers, op.source_tier)
                if HostTier(source_tier) is tier
            ]
            if not rows:
                continue
            ignored: list[int] = []
            self._append_transfers(
                [op.op_id],
                [[t.group_id for t in rows]],
                [[t.source_page for t in rows]],
                [[t.destination_page for t in rows]],
                collected_op_ids=ignored,
                transfers=rows_by_tier[tier],
                source_is_device=False,
                tier=tier,
            )

    @staticmethod
    def _check_distinct(ops: Sequence[SnapshotOp | RestoreOp]) -> None:
        """One plan names each snapshot op id and each arena row at most once.

        Runs with the other plan checks, before any copy is launched.
        """
        op_ids = [int(op.op_id) for op in ops]
        if len(set(op_ids)) != len(op_ids):
            raise ValueError(f"duplicate snapshot op id in one plan: {op_ids}")
        slots = [int(op.snapshot_slot) for op in ops]
        if len(set(slots)) != len(slots):
            raise ValueError(f"duplicate snapshot slot in one plan: {slots}")

    def _start_snapshot_store(
        self,
        ops: Sequence[SnapshotOp],
        transfers: Sequence[tuple[int, int, int]],
        *,
        prerequisite_stream,
    ):
        """Image the ops' tail rows and slot state on the write stream.

        Returns the completion event the caller fences the default stream on.
        The slot-state exporters' tensors live on the execution stream the
        pages were written on, so the one wait covers the rows and the
        exports; the victim's req-pool slot is reused only behind the fence,
        so the export reads its bytes. The ops were validated at collection.
        """
        if self.attn_tp_rank == 0:
            logger.info(
                f"[snapshot] store started: operations={len(ops):d} blocks="
                f"{len(transfers):d}",
            )
        self.write_stream.wait_stream(prerequisite_stream)
        if transfers:
            self._snapshot_write_lane.start_d2h(
                transfers,
                device_buffers=self.layout.buffers,
                host_buffer=self.snapshot_storage.host_buffer,
                geometry=self._snapshot_geometry,
                stream=self.write_stream,
                prerequisite_stream=None,
                backend=self.transfer_backend,
            )
        # Reading the victim's slot here is safe although the scheduler freed
        # it in this plan: the victim is quiescent (chooseVictim takes no
        # request with a forward in flight) and the FIFO submits every forward
        # of this plan behind the fence recorded below.
        for op in ops:
            self._slot_state.export(
                int(op.request_pool_index),
                self.blob_arena[int(op.snapshot_slot)],
                self.write_stream,
                request_id=str(op.request_id),
            )
        finish = device_module.Event()
        finish.record(self.write_stream)
        self._completions.push(
            finish,
            _Ack(
                kind=_AckKind.SNAPSHOT,
                op_ids=[int(op.op_id) for op in ops],
            ),
        )
        return finish

    def _start_restore(
        self,
        ops: Sequence[RestoreOp],
        rows_by_tier: dict[HostTier, Sequence[tuple[int, int, int]]],
        *,
        prerequisite_stream,
    ) -> None:
        """Copy both tiers' rows and the slot state into the restored requests.

        One event after the L2-tier rows, the pool-tier rows and the imports,
        so the scheduler sees one ``RestoreDone`` per op; the layerwise
        tracker is not armed. A row overwrites its whole destination block,
        so the scheduler lists no restore destination for zeroing; the copies
        still order behind the stream the caller names, which carries the
        plan's store fence and its zeroing of other pages. The ops were
        validated at collection; an op with no rows restores its slot state
        alone.
        """
        if get_is_capture_mode():
            raise RuntimeError("a snapshot restore must run outside graph capture")
        l2_rows = rows_by_tier[HostTier.L2]
        pool_rows = rows_by_tier[HostTier.SNAPSHOT_POOL]
        if self.attn_tp_rank == 0:
            logger.info(
                f"[snapshot] restore started: operations={len(ops):d} "
                f"l2_blocks={len(l2_rows):d} pool_blocks={len(pool_rows):d}",
            )
        # Behind the plan's zeroing and store fence on the stream the caller named.
        self.load_stream.wait_stream(prerequisite_stream)
        if l2_rows:
            self._restore_l2_lane.start_h2d(
                l2_rows,
                device_buffers=self.layout.buffers,
                host_buffer=self.host_storage.host_buffer,
                geometry=self._transfer_geometry,
                stream=self.load_stream,
                prerequisite_stream=None,
                backend=self.transfer_backend,
            )
        if pool_rows:
            self._restore_pool_lane.start_h2d(
                pool_rows,
                device_buffers=self.layout.buffers,
                host_buffer=self.snapshot_storage.host_buffer,
                geometry=self._snapshot_geometry,
                stream=self.load_stream,
                prerequisite_stream=None,
                backend=self.transfer_backend,
            )
        for op in ops:
            self._slot_state.import_(
                int(op.request_pool_index),
                self.blob_arena[int(op.snapshot_slot)],
                self.load_stream,
                request_id=str(op.request_id),
            )
        finish = device_module.Event()
        finish.record(self.load_stream)
        self._completions.push(
            finish,
            _Ack(
                kind=_AckKind.RESTORE,
                op_ids=[int(op.op_id) for op in ops],
            ),
        )

    def _start_writing(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        *,
        lane: HostTransferLane,
        prerequisite_stream,
    ):
        """Launch one D2H batch on the write stream; return its completion event.

        Returns None when the lane has no ops this round.
        """
        if not op_ids:
            return None
        op_ids = _ordered_unique(op_ids)
        if self.attn_tp_rank == 0:
            logger.info(
                f"[L2] writeback started: operations={len(op_ids):d} blocks="
                f"{len(transfers):d} pinned={lane is self._pinned_write_lane!s}",
            )
        finish = lane.start_d2h(
            transfers,
            device_buffers=self.layout.buffers,
            host_buffer=self.host_storage.host_buffer,
            geometry=self._transfer_geometry,
            stream=self.write_stream,
            prerequisite_stream=prerequisite_stream,
            backend=self.transfer_backend,
        )
        self._completions.push(
            finish,
            _Ack(
                kind=_AckKind.WRITE_BACK,
                op_ids=op_ids,
            ),
        )
        return finish

    def _start_loading(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        *,
        prerequisite_stream,
    ) -> int | None:
        if self._load_poisoned:
            raise RuntimeError(
                "L2 cache executor is poisoned after failed Host-load retirement"
            )
        if not op_ids:
            return None
        if get_is_capture_mode():
            raise RuntimeError("Host cache load must run outside CUDA Graph capture")
        op_ids = _ordered_unique(op_ids)
        if not transfers:
            # Every row belongs to other KVP ranks: nothing to copy, no layer
            # fence to arm (the trackers see no load this round), and the op
            # is acknowledged from an empty copy -- the hooks' replica
            # intersection completes it once the owners have copied theirs.
            with self._ack_lock:
                self._ready_load_acks.extend(op_ids)
            return None
        if self.attn_tp_rank == 0:
            logger.info(
                f"[L2] load started: operations={len(op_ids):d} blocks="
                f"{len(transfers):d}",
            )

        # The device handle zeroes the plan's freshly allocated Device blocks
        # (``pages_to_zero``) on the prerequisite stream before submitting
        # the load. Recording the start event there
        # makes the H2D copy wait for that zeroing; per-layer ready flags
        # (Triton) or events (DMA) then keep model consumers from reading
        # partially restored cache state.
        load_index = None
        finish = None
        flags = None
        active_trackers = []
        try:
            for tracker, consumer_count in self._load_trackers:
                current_load_index = tracker.begin_load()
                load_events = tracker.event_sets[current_load_index]
                # Register the generation immediately after begin_load so an
                # exception in tracker convergence or start-event setup still
                # retires every target/draft event set that advanced.
                active_trackers.append((load_events, consumer_count))
                if load_index is None:
                    load_index = current_load_index
                elif current_load_index != load_index:
                    raise RuntimeError("target and draft Host-load trackers diverged")
                load_events.start_event.record(prerequisite_stream)
                load_events.start_event.wait(self.load_stream)
            if load_index is None:
                raise RuntimeError("cache transfer layout has no layer consumers")

            device = self.layout.buffers[0].device
            workspace = self._load_workspaces[load_index]
            num_blocks, _ = workspace.load_block_transfers(
                transfers, geometry=self._transfer_geometry
            )
            layer_slices = self._transfer_geometry.layer_slices
            # Resolve the transport before choosing the consumer wait protocol.
            with device_module.stream(self.load_stream):
                mode = workspace.prepare_backend(
                    self.layout.buffers,
                    self.host_storage.host_buffer,
                    backend=self.transfer_backend,
                )
                if mode.uses_device_tables:
                    workspace.commit_block_transfers(
                        num_blocks,
                        device,
                        non_blocking=True,
                    )
                if mode.layer_ready:
                    flags = workspace.prepare_layer_ready(len(layer_slices), device)
                    for load_events, _ in active_trackers:
                        load_events.layer_ready_init_event.record(self.load_stream)
            if mode.layer_ready:
                flag_offset = 0
                for load_events, consumer_count in active_trackers:
                    load_events.layer_ready_flags = flags[
                        flag_offset : flag_offset + consumer_count
                    ]
                    load_events.wait_layer_ready = wait_layer_ready
                    flag_offset += consumer_count
                transfer_cache_blocks(
                    "h2d",
                    self.layout.buffers,
                    self.host_storage.host_buffer,
                    self._transfer_geometry,
                    workspace,
                    self.load_stream,
                    num_blocks=num_blocks,
                    geometry_offset=0,
                    num_geometry_rows=self._transfer_geometry.num_field_rows,
                    backend=self.transfer_backend,
                    layer_ready_flags=flags,
                    grid_cap=None,
                )
                finish = device_module.Event()
                finish.record(self.load_stream)
                for load_events, consumer_count in active_trackers:
                    load_events.set_completion(finish)
            else:
                for load_events, _ in active_trackers:
                    load_events.layer_ready_flags = None
                    load_events.wait_layer_ready = None
                flat_layer_index = 0
                for load_events, consumer_count in active_trackers:
                    for layer_index in range(consumer_count):
                        geometry_offset, num_geometry_rows = layer_slices[
                            flat_layer_index
                        ]
                        transfer_cache_blocks(
                            "h2d",
                            self.layout.buffers,
                            self.host_storage.host_buffer,
                            self._transfer_geometry,
                            workspace,
                            self.load_stream,
                            num_blocks=num_blocks,
                            geometry_offset=geometry_offset,
                            num_geometry_rows=num_geometry_rows,
                            backend=self.transfer_backend,
                            grid_cap=None,
                            layer_ready_flags=None,
                        )
                        finish = device_module.Event()
                        finish.record(self.load_stream)
                        load_events.layer_done_events[layer_index] = finish
                        flat_layer_index += 1
            if finish is None:
                raise RuntimeError("cache transfer layout has no layer consumers")
            self._completions.push(
                finish,
                _Ack(kind=_AckKind.LOAD_BACK, op_ids=op_ids),
            )
            return load_index
        except BaseException as original_error:
            self._retire_failed_load(active_trackers, flags, original_error)
            raise

    def _retire_failed_load(self, active_trackers, flags, original_error) -> None:
        """Retire submitted GPU readers without publishing a success ACK."""
        if not active_trackers:
            return
        try:
            if flags is not None:
                with device_module.stream(self.load_stream):
                    flags.fill_(1)
            retirement = device_module.Event()
            retirement.record(self.load_stream)
            for load_events, _ in active_trackers:
                load_events.set_completion(retirement)
            return
        except BaseException as retirement_error:
            # If event publication fails, only stream completion permits reuse.
            try:
                self.load_stream.synchronize()
            except BaseException as sync_error:
                self._load_poisoned = True
                add_note = getattr(original_error, "add_note", None)
                if add_note is not None:
                    try:
                        add_note(
                            "Host-load retirement failed; executor poisoned: "
                            f"retirement error={retirement_error!r}; "
                            f"synchronize error={sync_error!r}"
                        )
                    except BaseException:
                        pass

    def poll_results(self) -> list:
        """The completed ops' ACKs, both tiers; event queries only, never blocks."""
        results: list = []
        with self._ack_lock:
            results.extend(self._load_done(op_id) for op_id in self._ready_load_acks)
            self._ready_load_acks.clear()
        for ack in self._completions.pop_ready():
            if ack.kind is _AckKind.WRITE_BACK:
                results.extend(self._write_done(op_id) for op_id in ack.op_ids)
            elif ack.kind is _AckKind.LOAD_BACK:
                results.extend(self._load_done(op_id) for op_id in ack.op_ids)
            elif ack.kind is _AckKind.SNAPSHOT:
                results.extend(self._snapshot_done(op_id) for op_id in ack.op_ids)
            else:
                results.extend(self._restore_done(op_id) for op_id in ack.op_ids)
        return results

    @staticmethod
    def _write_done(op_id: int):
        event = Cache.WriteBackDoneEvent()
        event.op_id = op_id
        return event

    @staticmethod
    def _load_done(op_id: int):
        return Cache.LoadBackDoneEvent(op_id)

    @staticmethod
    def _snapshot_done(op_id: int):
        event = Cache.SnapshotDoneEvent()
        event.op_id = op_id
        return event

    @staticmethod
    def _restore_done(op_id: int):
        event = Cache.RestoreDoneEvent()
        event.op_id = op_id
        return event

    def shutdown(self) -> None:
        # The fences and start events live on streams the callers named per
        # submission; the whole device covers them and the transfer streams.
        device_module.synchronize()
        self._completions.drop_all()
        with self._ack_lock:
            self._ready_load_acks.clear()
