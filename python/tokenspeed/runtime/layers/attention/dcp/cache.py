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

"""Explicit collective reconstruction of bounded MLA prefill history."""

from __future__ import annotations

import bisect
import logging
import struct
import threading
import zlib
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import split_slots_by_owner

from tokenspeed.runtime.distributed.comm_backend import (
    CommBackend,
    CommWork,
    get_global_backend,
)
from tokenspeed.runtime.distributed.comm_ops import (
    all_gather_single,
    all_gather_single_async,
)
from tokenspeed.runtime.distributed.mapping import KVP_HISTORY_GROUP_NAME
from tokenspeed.runtime.layers.attention.chunk import get_max_chunk_capacity
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    owned_prefix_chunk_rows,
)

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._cute_dsl.paged_row_all_gather import (
        PagedRowAllGatherState,
    )

    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryGatherWorkspace:
    """The buffers a request group's gathered history lands in.

    Allocated once per engine (the recipe reserves the bytes from the cache
    budget, ``OrdinaryRecipe.workspace_bytes``) by the target's DSA leaf and
    shared with the draft's: the two never gather at the same time, since the
    draft's extend step follows the target's forward on the same stream.

    Under a KVP prefill over gathered histories (``dsa_kvp_history_prefill``)
    ``kv`` and ``index_k`` view the same leading bytes: a layer's top-k
    consumes its gathered index-K rows before the layer gathers its latent
    rows, on the same stream, so the buffer holds the wider of the two.

    Attributes:
        rows: Row capacity, a whole number of kernel pages.
        kv: ``[rows, kv_cache_dim]`` latent rows in the KV cache dtype.
        index_k: ``[rows, row_bytes]`` uint8 index-K rows packed in the
            plane's format, ``row_bytes`` being
            ``configs.dsa.index_k_row_bytes(index_head_dim, index_k_format)``
            (FP8 bytes then fp32 scales, or the bf16 key's bytes;
            ``kv_cache.dsa.split_index_k_rows`` views them apart).
        index_k_format: The index-K plane format the rows are packed in
            (``configs.dsa.INDEX_K_FORMATS``), recorded as the allocating
            leaf named it; the leaf that adopts the workspace checks it
            against its own.
    """

    rows: int
    kv: torch.Tensor
    index_k: torch.Tensor
    index_k_format: str

    def __post_init__(self) -> None:
        if (
            self.kv.dim() != 2
            or self.index_k.dim() != 2
            or self.kv.shape[0] != self.rows
            or self.index_k.shape[0] != self.rows
            or self.index_k.dtype != torch.uint8
        ):
            raise ValueError(
                f"history workspace of {self.rows} rows does not match kv "
                f"{tuple(self.kv.shape)} / index_k {tuple(self.index_k.shape)} "
                f"{self.index_k.dtype}"
            )

    @property
    def nbytes(self) -> int:
        if self.kv.data_ptr() == self.index_k.data_ptr():
            return max(self.kv.nbytes, self.index_k.nbytes)
        return self.kv.nbytes + self.index_k.nbytes


@dataclass(frozen=True)
class HistoryGatherPlan:
    """How one request group's history rows split over the page owners.

    Built once per forward and group from the virtual slots of every history
    row (prefix plus this chunk, request-major, position order) and the
    host-side owned-row counts; reused by every layer's KV and index-K
    gathers. Every rank sends ``send_rows`` rows -- the largest owner's
    count: its own rows in position order, then a tail nobody reads -- so the
    all-gather output is ``[degree * send_rows]`` rows, rank-major, and
    ``gathered_index`` reads it back in position order (the layout of
    :class:`PrefixHistoryWindow`).

    Attributes:
        virtual_slots: ``[rows]`` int64 virtual cache slots in position order.
        send_slots: ``[send_rows]`` physical slots this rank sends, in the
            virtual slots' integer dtype: its own rows in position order,
            then slot 0 (the null page) for the tail nobody reads.
        gathered_index: ``[rows]`` int32 row of the all-gather output that
            holds each history row, position order.
        owned_rows_per_rank: Host counts of rows each owner holds.
        group: The owner group the gather runs over (a singleton without
            page sharding).
        rank: This rank's position in ``group``.
    """

    virtual_slots: torch.Tensor
    send_slots: torch.Tensor
    gathered_index: torch.Tensor
    owned_rows_per_rank: tuple[int, ...]
    group: tuple[int, ...]
    rank: int

    @property
    def rows(self) -> int:
        return sum(self.owned_rows_per_rank)

    @property
    def send_rows(self) -> int:
        """Rows every rank contributes to the all-gather."""
        return max(self.owned_rows_per_rank)


def plan_history_gather(
    virtual_slots: torch.Tensor,
    *,
    placement: CachePlacement | None,
    owned_rows_per_rank: Sequence[int],
) -> HistoryGatherPlan:
    """Plan the owner-split gather of a request group's history rows.

    Args:
        virtual_slots: ``[rows]`` int64 virtual slots of every history row,
            position order, identical on every rank. Each row must have a valid
            owner; callers validate the host page table with
            ``owned_history_rows`` before constructing the slots.
        placement: The group's page ownership, or ``None`` when this rank
            holds every row (the gather is then a local slot gather).
        owned_rows_per_rank: Host counts of rows each owner contributes
            (``owned_history_rows`` summed over the group's requests); one
            entry without a placement. Counts must match the slots on every rank.

    Returns:
        The plan; no device synchronization.
    """
    counts = tuple(int(count) for count in owned_rows_per_rank)
    if placement is None:
        if len(counts) != 1:
            raise ValueError("an unsharded history gather has one owner")
        group, rank = (0,), 0
    else:
        group, rank = placement.group, placement.rank
        if len(counts) != len(group):
            raise ValueError(
                f"owned_rows_per_rank names {len(counts)} owners for a group of "
                f"{len(group)}"
            )
    rows = virtual_slots.numel()
    if sum(counts) != rows:
        raise ValueError(
            f"owned rows sum to {sum(counts)} but the history has {rows} rows"
        )
    if placement is None:
        # This rank holds every row: it sends them all, in position order.
        send_slots = virtual_slots
        gathered_index = torch.arange(
            rows, dtype=torch.int32, device=virtual_slots.device
        )
    else:
        # Split the rows by owner; the tail past this rank's own
        # rows stays slot 0, a row of the null page nobody reads.
        send_slots = virtual_slots.new_zeros(max(counts))
        gathered_index = torch.empty(
            rows, dtype=torch.int32, device=virtual_slots.device
        )
        split_slots_by_owner(
            virtual_slots.contiguous(),
            block_granularity=placement.block_granularity,
            degree=len(group),
            rank=rank,
            send_rows=max(counts),
            gathered_index=gathered_index,
            fetch=send_slots,
        )
    return HistoryGatherPlan(
        virtual_slots=virtual_slots,
        send_slots=send_slots,
        gathered_index=gathered_index,
        owned_rows_per_rank=counts,
        group=group,
        rank=rank,
    )


def gather_history_rows(
    plan: HistoryGatherPlan,
    send_rows: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Assemble a request group's history rows in position order.

    Every rank sends ``plan.send_rows`` rows (``plan.send_slots`` of its own
    cache plane, as 2-D rows: its own rows, then a tail nobody reads); one
    all-gather of equal shares puts them rank-major, and one row gather
    (``plan.gathered_index``) writes them into ``out`` in position order.
    Pure data movement: the rows travel as their bytes, so a row lands as
    the owner's bytes whatever its dtype. Without page sharding the rows are
    already local and only the row gather runs.

    Args:
        plan: The group's gather plan.
        send_rows: ``[plan.send_rows, width]`` rows this rank sends; any
            dtype.
        out: ``[>= rows, width]`` destination of ``send_rows``' dtype; the
            leading ``rows`` are written.

    Returns:
        ``out[:rows]``.
    """
    rows = plan.rows
    if send_rows.dim() != 2 or send_rows.shape[0] != plan.send_rows:
        raise ValueError(
            f"rank {plan.rank} sends {plan.send_rows} rows, got "
            f"{tuple(send_rows.shape)}"
        )
    if (
        out.dim() != 2
        or out.shape[0] < rows
        or out.shape[1] != send_rows.shape[1]
        or out.dtype != send_rows.dtype
    ):
        raise ValueError(
            f"history destination {tuple(out.shape)} {out.dtype} does not hold "
            f"{rows} rows of {send_rows.shape[1]} {send_rows.dtype}"
        )
    # Bytes: the collective and the row gather then serve every dtype (fp8
    # included) and keep every bit.
    payload = send_rows.contiguous().view(torch.uint8)
    if len(plan.group) == 1:
        gathered = payload
    else:
        gathered = payload.new_empty(
            (len(plan.group) * plan.send_rows, payload.shape[1])
        )
        all_gather_single(gathered, payload, plan.group)
    target = out[:rows]
    torch.index_select(gathered, 0, plan.gathered_index, out=target.view(torch.uint8))
    return target


def all_gather_history_rows(
    plan: HistoryGatherPlan,
    send_rows: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """A request group's history rows in the all-gather's rank-major layout.

    :func:`gather_history_rows` without the read-back: row ``i`` of the
    group's history is row ``plan.gathered_index[i]`` of the result, so a
    consumer that reads rows by index maps its indices instead of moving the
    rows. The rows travel as their bytes, as there.

    Args:
        plan: The group's gather plan.
        send_rows: ``[plan.send_rows, width]`` rows this rank sends; any
            dtype.
        out: ``[>= len(plan.group) * plan.send_rows, width]`` destination of
            ``send_rows``' dtype.

    Returns:
        ``out[:len(plan.group) * plan.send_rows]``.
    """
    gathered_rows = len(plan.group) * plan.send_rows
    if send_rows.dim() != 2 or send_rows.shape[0] != plan.send_rows:
        raise ValueError(
            f"rank {plan.rank} sends {plan.send_rows} rows, got "
            f"{tuple(send_rows.shape)}"
        )
    if (
        out.dim() != 2
        or out.shape[0] < gathered_rows
        or out.shape[1] != send_rows.shape[1]
        or out.dtype != send_rows.dtype
    ):
        raise ValueError(
            f"history destination {tuple(out.shape)} {out.dtype} does not hold "
            f"{gathered_rows} rows of {send_rows.shape[1]} {send_rows.dtype}"
        )
    target = out[:gathered_rows]
    payload = send_rows.contiguous().view(torch.uint8)
    if len(plan.group) == 1:
        target.view(torch.uint8).copy_(payload)
    else:
        all_gather_single(target.view(torch.uint8), payload, plan.group)
    return target


# Consecutive prefix-chunk loops one NCCL history all-gather covers. A gather
# costs a fixed host launch whatever it moves, so one per layer serves a
# typical multi-turn history; the window still bounds the transient buffer, at
# about this many loops' rows of cache-dtype bytes.
PREFIX_HISTORY_WINDOW_LOOPS = 4


@dataclass(frozen=True)
class PrefixHistoryWindow:
    """One all-gather of consecutive prefix-chunk loops' rows, every layer.

    Every rank sends ``send_rows`` rows -- the largest owner's count: its own
    rows first, a tail nobody reads -- so the all-gather output is
    ``[degree * send_rows]`` rows, rank-major, with rank ``k``'s rows from row
    ``k * send_rows``. ``gathered_index`` reads it back in the loops' order.

    Attributes:
        group: The page owners the gather runs over.
        rank: This rank's position in ``group``.
        owned_rows_per_rank: Host counts of the window rows each owner holds.
        send_rows: Rows every rank contributes to the all-gather.
        fetch_slots: ``[send_rows]`` physical slots this rank sends, in the
            virtual slots' integer dtype: its own rows in order, then slot 0
            (the null page) for the tail nobody reads.
        gathered_index: ``[rows]`` int32 row of the all-gather output that
            holds each row of the window, loop-major.
        loop_bounds: ``[first loop's start, ..., last loop's end]`` row
            offsets of the covered loops within ``gathered_index``.
    """

    group: tuple[int, ...]
    rank: int
    owned_rows_per_rank: tuple[int, ...]
    send_rows: int
    fetch_slots: torch.Tensor
    gathered_index: torch.Tensor
    loop_bounds: tuple[int, ...]


def plan_prefix_history_gather(
    chunk_kv_indices_list: Sequence[torch.Tensor],
    *,
    page_table_cpu: torch.Tensor,
    chunk_bounds_cpu: torch.Tensor,
    page_size: int,
    placement: CachePlacement,
    max_loops: int | None,
    max_send_rows: int | None,
) -> tuple[PrefixHistoryWindow, ...]:
    """Plan one forward's page-owner gathers of its prefix-chunk loops.

    A window takes consecutive loops: the next loop joins while the window
    keeps at most ``max_loops`` loops and its busiest owner at most
    ``max_send_rows`` rows, so a loop past ``max_send_rows`` alone makes a
    window of its own. The owners' counts decide, not an even split, so
    uneven ownership packs fewer loops. Every MLA layer reuses the plan. The
    owned-row counts come from the host page table, so nothing waits on the
    device; one kernel per window (``split_slots_by_owner``) gives every row
    its all-gather output row and this rank's rows their physical slots.

    Args:
        chunk_kv_indices_list: Each loop's ``[rows]`` virtual slots, position
            order (``chunk.get_chunks_paged``), identical on every rank.
        page_table_cpu: ``[requests, columns]`` host kernel-page table of the
            extend requests.
        chunk_bounds_cpu: ``[loops + 1, requests]`` host prefix positions.
        page_size: Tokens per kernel page.
        placement: The MLA history group's page ownership.
        max_loops: The most loops a window takes, or None for no bound (the
            NCCL all-gather's windows: ``PREFIX_HISTORY_WINDOW_LOOPS``).
        max_send_rows: The most rows a window's busiest owner holds, or None
            for no bound (the fused all-gather's: its receive slot's rows).

    Returns:
        The windows, in loop order.
    """
    if chunk_bounds_cpu.shape[0] != len(chunk_kv_indices_list) + 1:
        raise ValueError("prefix boundaries must delimit every history loop")
    if (max_loops is not None and max_loops < 1) or (
        max_send_rows is not None and max_send_rows < 1
    ):
        raise ValueError(
            f"window bounds must be positive, got max_loops {max_loops} and "
            f"max_send_rows {max_send_rows}"
        )
    counts = owned_prefix_chunk_rows(
        page_table_cpu,
        chunk_bounds_cpu,
        page_size=page_size,
        placement=placement,
    ).tolist()
    degree = len(placement.group)
    windows = []
    first = 0
    while first < len(counts):
        # The same windows on every rank, which holds the same host table.
        end, owned = first + 1, counts[first]
        while end < len(counts) and (max_loops is None or end - first < max_loops):
            joined = [mine + more for mine, more in zip(owned, counts[end])]
            if max_send_rows is not None and max(joined) > max_send_rows:
                break
            owned, end = joined, end + 1
        loops = chunk_kv_indices_list[first:end]
        first = end
        slots = torch.cat(loops) if len(loops) > 1 else loops[0]
        owned = tuple(owned)
        if sum(owned) != slots.numel():
            raise ValueError(
                f"owned rows sum to {sum(owned)} but the window has "
                f"{slots.numel()} rows"
            )
        send_rows = max(owned)
        # Split the rows by owner. Every rank sends send_rows
        # rows: its own first, then local slot 0 -- a row of the null page --
        # for the tail no gathered_index entry reads.
        fetch = slots.new_zeros(send_rows)
        gathered_index = torch.empty(
            slots.shape, dtype=torch.int32, device=slots.device
        )
        split_slots_by_owner(
            slots,
            block_granularity=placement.block_granularity,
            degree=degree,
            rank=placement.rank,
            send_rows=send_rows,
            gathered_index=gathered_index,
            fetch=fetch,
        )
        bounds = [0]
        for loop in loops:
            bounds.append(bounds[-1] + loop.numel())
        windows.append(
            PrefixHistoryWindow(
                group=placement.group,
                rank=placement.rank,
                owned_rows_per_rank=owned,
                send_rows=send_rows,
                fetch_slots=fetch,
                gathered_index=gathered_index,
                loop_bounds=tuple(bounds),
            )
        )
    return tuple(windows)


# Backends and plans whose kernels never wait across ranks inside a kernel
# that needs the whole GPU. A fused history window in flight on the same GPU
# could deadlock with one that does (MegaMoE, DeepEP's low-latency all-to-all,
# the fused DP projections; the kernel package's communication README), and
# such a kernel can run from a CUDA graph, where no call site could wait for
# the window first. So any other configuration keeps the NCCL all-gather. Add
# a new MoE solution or all-to-all here only after checking that.
_FUSED_HISTORY_SAFE_MOE_BACKENDS = frozenset(
    {
        "auto",
        "triton",
        "flashinfer_trtllm",
        "flashinfer_cutlass",
        "flashinfer_cutedsl",
        "marlin",
    }
)
_FUSED_HISTORY_SAFE_MOE_SOLUTIONS = frozenset(
    {
        "triton",
        "flashinfer_trtllm",
        "flashinfer_cutlass",
        "flashinfer_cutedsl",
        "marlin",
        "torch",
    }
)


class FusedHistoryTransport:
    """A KVP history group's fused all-gather, shared by the target's and the
    draft's prefix-history reads.

    Every rank plans the same windows and hands this transport the same ones,
    in the same order, so the state numbers them alike. The transport keeps
    the kernel's slot contract (the kernel package's communication README):
    one reader at a time, at most two windows not yet released, and one
    consumer stream for its life, on which each launch's ready point follows
    the reads before it; a window's slot is released once its reads are
    enqueued there. A failed launch or drain leaves the transport poisoned --
    the ranks' epochs, or the slots' order, are no longer known -- and every
    later launch raises, whether or not the window would be fused.

    Args:
        state: The group's ``create_paged_row_all_gather_state``.
    """

    def __init__(self, state: PagedRowAllGatherState):
        # Not through select_kernel: a collective with state, whose platform
        # the startup vote decided (no state off SM100 and SM103).
        from tokenspeed_kernel.ops.communication.cute_dsl import (
            cute_dsl_paged_row_all_gather,
        )

        self.state = state
        self._gather = cute_dsl_paged_row_all_gather
        # The epochs whose slot may still be read, oldest first, with the
        # fetch slots their kernel reads and the event that ends it (None for
        # a launch that failed): held until the consumer stream waits for the
        # kernel, and for good once the transport is poisoned.
        self._open: deque[tuple[int, torch.Tensor, torch.cuda.Event | None]] = deque()
        self._reader: object | None = None
        self._stream: torch.cuda.Stream | None = None
        # The slots' rows per slot, cache dtype and row shape.
        self._rows: dict[tuple[int, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
        self.poisoned = False

    def accepts(self, row_bytes: int, slots_dtype: torch.dtype) -> bool:
        """Whether windows of ``row_bytes``-byte rows, planned from virtual
        slots of ``slots_dtype``, fit the compiled kernel: its row width and
        its int32 fetch slots. Equal on every rank, whose pools are built
        alike."""
        return row_bytes == self.state.row_bytes and slots_dtype == torch.int32

    def launch(
        self,
        cache: torch.Tensor,
        window: PrefixHistoryWindow,
        header: int,
        *,
        reader: object,
    ) -> tuple[torch.Tensor, torch.cuda.Event, int] | None:
        """Start gathering ``window``'s rows of ``cache`` into the next
        epoch's slot for ``reader``, after the current stream's work so far.

        Args:
            cache: One layer's ``[slots, *row]`` latent plane.
            window: The planned window to gather.
            header: The window's identity, equal on every rank, below 2**63.
            reader: The window's only consumer.

        Returns:
            None, on every rank alike, for a window the transport does not
            take: past the slot, rows of another layout than compiled, or
            fetch slots not of the kernel's int32. Otherwise the slot as
            ``[degree * cap_rows, *row]`` rows of the cache dtype (rank
            ``p``'s rows from row ``p * send_rows``), the event its reader
            waits for, and the epoch.
        """
        if self.poisoned:
            raise RuntimeError(
                "the fused history all-gather refuses launches after a failed "
                "launch or drain"
            )
        if torch.cuda.is_current_stream_capturing():
            # Prefill graphs keep attention in eager segments and capture
            # without a prefix, so no captured forward reads prefix history.
            raise RuntimeError(
                "a fused history window launched under graph capture: no "
                "captured forward reads prefix history"
            )
        state = self.state
        # Equal on every rank: every rank builds its pool's layout (dtype, row
        # shape, strides) the same way, send_rows is the window's, and the
        # planner's code fixes the slots' dtype.
        if (
            not cache.is_contiguous()
            or not self.accepts(
                cache.shape[1:].numel() * cache.element_size(),
                window.fetch_slots.dtype,
            )
            or window.send_rows * state.row_bytes > state.cap_bytes
        ):
            return None
        # Refused before the launch changes anything: a launch that would
        # reuse a slot still being read.
        if self._open and self._reader is not reader:
            raise RuntimeError(
                "two readers interleave fused history windows: epoch "
                f"{state.epoch + 1} would reuse the slot of epoch {state.epoch - 1} "
                "while another reader may read it"
            )
        if len(self._open) == 2:
            raise RuntimeError(
                f"a third fused history window in flight (epochs "
                f"{[e for e, _, _ in self._open]} not released): epoch "
                f"{state.epoch + 1} would overwrite a slot still being read"
            )
        self._check_stream("launch")
        try:
            # The launch orders the kernel after the consumer's work so far:
            # the cache rows, a Host load's wait, the reads of epoch - 2 from
            # this slot. It raises on this rank's own properties (the plane's
            # alignment, its device): falling back here would leave the other
            # ranks waiting in the kernel.
            epoch, done = self._gather(
                state,
                cache,
                window.fetch_slots,
                window.owned_rows_per_rank[window.rank],
                window.send_rows,
                header,
            )
        except BaseException:
            # The kernel may be in flight: keep its fetch slots for good.
            self.poisoned = True
            self._reader = reader
            self._open.append((state.epoch, window.fetch_slots, None))
            raise
        self._reader = reader
        self._open.append((epoch, window.fetch_slots, done))
        key = (epoch & 1, cache.dtype, tuple(cache.shape[1:]))
        rows = self._rows.get(key)
        if rows is None:
            rows = state.receive_view(epoch).view(cache.dtype).view(-1, *key[2])
            self._rows[key] = rows
        return rows, done, epoch

    def taken(self, epoch: int, reader: object) -> None:
        """``reader`` takes ``epoch``, which it launched, on the consumer
        stream and not under graph capture: a graph would replay reads of a
        slot that later epochs overwrite."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"fused history epoch {epoch} taken under graph capture: a "
                "capture begins with no fused window in flight"
            )
        if self._reader is not reader or all(
            open_epoch != epoch for open_epoch, _, _ in self._open
        ):
            raise RuntimeError(
                f"fused history epoch {epoch} taken but not in flight for this "
                f"reader (in flight: {[e for e, _, _ in self._open]})"
            )
        self._check_stream("take")

    def release(self, epoch: int, reader: object) -> None:
        """The reads of ``epoch`` are enqueued on the consumer stream: its
        slot is free for ``epoch + 2``."""
        self._check_stream("release")
        if not self._open or self._open[0][0] != epoch or self._reader is not reader:
            raise RuntimeError(
                f"fused history epoch {epoch} released out of order (in flight: "
                f"{[e for e, _, _ in self._open]})"
            )
        self._open.popleft()

    def drain(self, reader: object) -> None:
        """Order the consumer stream, whatever stream is current, after every
        window ``reader`` has in flight, and drop them. A poisoned transport
        keeps their fetch slots: a kernel whose launch failed, or whose end it
        could not wait for, may still read them."""
        if self._reader is not reader or not self._open:
            return
        try:
            for _, _, done in self._open:
                if done is not None:
                    self._stream.wait_event(done)
        except BaseException:
            self.poisoned = True
            raise
        if not self.poisoned:
            self._open.clear()

    def holds(self, reader: object) -> bool:
        """Whether ``reader`` has a window launched and not yet released."""
        return bool(self._open) and self._reader is reader

    def _check_stream(self, what: str) -> None:
        stream = torch.cuda.current_stream(self.state.device)
        if self._stream is not None and stream != self._stream:
            raise RuntimeError(
                f"fused history {what} on stream {stream}, not {self._stream}: "
                "the transport's launches and reads stay on one stream, whose "
                "ready event orders a slot's reads before its reuse"
            )
        self._stream = stream


# Every KVP history group's fused transport, by group: built by
# prepare_prefix_history_transport, taken by the group's readers.
_FUSED_HISTORY_TRANSPORTS: dict[tuple[int, ...], FusedHistoryTransport] = {}


def prepare_prefix_history_transport(
    server_args: ServerArgs,
    model_config: ModelConfig,
    models: Sequence[torch.nn.Module],
    device: torch.device,
) -> None:
    """Collectively build the fused all-gather of this rank's KVP history
    group, or keep its NCCL all-gather; every rank of the group takes the same
    branch.

    Every rank calls this once at the same program point: after the target
    and draft models and their communication buffers are built, before graph
    capture and before the KV cache is sized, which then leaves the
    transport's workspace out. Once the group's transport is built, another
    call does nothing; after a refusal, it votes again. The group keeps
    the NCCL all-gather when its model reads no MLA prefix history through
    ``PrefixHistoryGather`` (DeepSeek V4.1 gathers its own) or reads
    ``per_token_head`` rows, on a decode role, without chunked prefill, with
    a receive slot of 2 GiB or more (the factory's bound on the receive
    workspace), or when a configured kernel could deadlock with a fused
    window (``_FUSED_HISTORY_SAFE_MOE_BACKENDS``).

    A rank's part of a receive slot holds the MLA history capacity
    (``chunk.get_max_chunk_capacity``) in rows: why, in
    ``docs/design/cache-concepts.md``; the sizing and when the group keeps
    NCCL, in ``docs/serving/parallelism.md``.

    Args:
        server_args: The launch's resolved server arguments.
        model_config: The target model's config.
        models: The built target model and, with speculation, the draft's.
        device: This rank's device.
    """
    mapping = server_args.mapping
    if not mapping.attn.has_kvp:
        return
    group = tuple(mapping.attn.kvp_group)
    if group in _FUSED_HISTORY_TRANSPORTS:
        return
    # Not through the registry: a collective state factory, which votes on the
    # platform itself (every rank gets None off SM100 and SM103).
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        PAGED_ROW_ALL_GATHER_SLOT_BYTES_LIMIT,
        create_paged_row_all_gather_state,
    )

    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.execution.distributed_initializer import (
        DistributedConfig,
    )
    from tokenspeed.runtime.layers.attention.configs.base import resolve_dtype
    from tokenspeed.runtime.layers.linear import (
        DPColumnParallelLinear,
        DPRowParallelLinear,
    )
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.layers.moe.utils import All2AllBackend

    moe_backends = {server_args.moe_backend}
    if server_args.speculative_algorithm is not None:
        moe_backends.add(server_args.draft_moe_backend or server_args.moe_backend)
    reason = None
    row_bytes = 0
    cap_rows = 0
    # V4.1 gathers its own KVP history, outside PrefixHistoryGather.
    if (
        model_config.attention_arch != AttentionArch.MLA
        or server_args.attention_backend == "deepseek_v41"
    ):
        reason = "the model reads no MLA prefix history"
    elif server_args.kv_cache_quant_method != "none":
        reason = f"KV cache quantization {server_args.kv_cache_quant_method!r}"
    elif server_args.disaggregation_mode == "decode":
        # A decode role never computes prompt rows: no prefix-chunk loops.
        reason = "a decode role reads no prefix history"
    elif not moe_backends <= _FUSED_HISTORY_SAFE_MOE_BACKENDS:
        reason = f"MoE backends {sorted(moe_backends)}"
    elif not All2AllBackend(server_args.all2all_backend).is_none():
        reason = f"all-to-all backend {server_args.all2all_backend!r}"
    elif get_max_chunk_capacity() <= 0:
        # The chunk planner refuses such a capacity: no prefix-chunk loops.
        reason = "chunked prefill is off"
    else:
        cap_rows = get_max_chunk_capacity()
        row_bytes = (
            model_config.kv_lora_rank + model_config.qk_rope_head_dim
        ) * resolve_dtype(server_args.kv_cache_dtype).itemsize
        # The factory refuses such a slot too; checked here to name it.
        slot_bytes = len(group) * cap_rows * row_bytes
        if slot_bytes >= PAGED_ROW_ALL_GATHER_SLOT_BYTES_LIMIT:
            reason = (
                f"a receive slot of {slot_bytes / 2**20:.0f} MiB, not under the "
                f"fused all-gather's {PAGED_ROW_ALL_GATHER_SLOT_BYTES_LIMIT >> 30} "
                "GiB"
            )
    for module in (module for model in models for module in model.modules()):
        if reason is not None:
            break
        if isinstance(module, MoELayer) and (
            module.plan["solution"] not in _FUSED_HISTORY_SAFE_MOE_SOLUTIONS
            or module.plan["a2a_backend"] == "deepep"
        ):
            reason = (
                f"MoE plan {module.plan['solution']!r} on all-to-all "
                f"{module.plan['a2a_backend']!r}"
            )
        elif (
            isinstance(module, (DPColumnParallelLinear, DPRowParallelLinear))
            and module.projection_workspace is not None
            and (
                module.projection_workspace.fused_column is not None
                or module.projection_workspace.fused_row is not None
            )
        ):
            reason = "fused DP projections"
    timeout_s = (
        server_args.distributed_timeout_seconds
        if server_args.distributed_timeout_seconds is not None
        else DistributedConfig.distributed_timeout_seconds
    )
    # In-kernel waits end before the process group's watchdog, so a stuck
    # peer gets the kernel's diagnostic first.
    timeout_ns = int((timeout_s - 30 if timeout_s >= 90 else timeout_s / 2) * 1e9)
    state = create_paged_row_all_gather_state(
        pg_manager.get_process_group("gloo", group),
        device,
        row_bytes=row_bytes,
        cap_bytes=cap_rows * row_bytes,
        timeout_ns=timeout_ns,
        modes=("multicast", "peer"),
        eligible=reason is None,
    )
    log = mapping.rank == group[0]
    if state is None:
        if log:
            logger.info(
                f"KVP history group {group} keeps its NCCL all-gather: "
                f"{reason or 'the fused all-gather is unavailable on some rank'}"
            )
        return
    _FUSED_HISTORY_TRANSPORTS[group] = FusedHistoryTransport(state)
    if log:
        logger.info(
            f"KVP history group {group} gathers through the fused all-gather: "
            f"{state.mode}, {state.num_ctas} CTAs, "
            f"{state.workspace.numel() / 2**20:.1f} MiB workspace, "
            f"{state.device_bytes / 2**20:.1f} MiB device memory, "
            f"{state.cap_bytes // state.row_bytes} rows of {state.row_bytes} B per "
            "rank and slot"
        )


@dataclass
class _LaunchedGather:
    """A window gather in flight: an NCCL all-gather's work (``epoch`` None),
    or the fused transport's event that ends ``epoch``."""

    rows: torch.Tensor
    work: CommWork | torch.cuda.Event
    epoch: int | None


class PrefixHistoryGather:
    """A page-sharded MLA leaf's prefix-chunk history reads.

    :meth:`prepare` records each forward's prefix-chunk loops; :meth:`read`
    yields one layer's rows of every loop. The rows come from their page
    owners, in windows of consecutive loops (:meth:`windows`) that the
    forward's first read plans: planning is host launch work, so deferring it
    from before the forward's first kernel to that layer lets it run while
    the device works through the layers queued ahead.

    Each window is all-gathered asynchronously on ``comm``, a backend on the
    named KVP history groups (``KVP_HISTORY_GROUP_NAME``) whose communicators
    are apart from the attention TP collectives over the same ranks, one
    window ahead: a read takes its window and starts the next one it will
    need -- this layer's next, else the next sharded layer's first -- so that
    transfer runs under the compute in between (for a hybrid model, the
    linear-attention layers between two MLA layers) instead of queueing with
    those collectives. The gathers are collectives: every rank reads every
    layer's history in the same order. Every layer of the group is read once
    per forward, in ascending order: a read starts the next layer's first
    window, which the fused transport holds until that layer reads it.

    With a fused ``transport`` (``prepare_prefix_history_transport``), a
    window goes through it instead: one kernel on the transport's stream
    stores every rank's rows, read straight out of its cache, into every
    rank's receive slot, so no pack copy and no NCCL channel runs. A window
    the transport does not take (``FusedHistoryTransport.launch``) keeps the
    all-gather, on every rank alike. Since the slots' reuse follows the reads,
    a read checks at every resume that it continues where it started (the
    same forward, thread and stream, and no capture begun with a fused window
    in flight), before it enqueues anything.
    """

    def __init__(
        self,
        placement: CachePlacement,
        *,
        page_size: int,
        comm: CommBackend,
        transport: FusedHistoryTransport | None,
    ):
        self._placement = placement
        self._page_size = page_size
        self._comm = comm
        self._transport = transport
        # The current forward's loops and planning inputs, and its windows
        # once the first read planned them.
        self._chunk_kv_indices_list: Sequence[torch.Tensor] = ()
        self._page_table_cpu: torch.Tensor | None = None
        self._chunk_bounds_cpu: torch.Tensor | None = None
        self._windows: tuple[PrefixHistoryWindow, ...] | None = None
        self._launched: dict[tuple[int, int], _LaunchedGather] = {}
        # Ascending ids of the layers sharing this history, learned from the
        # pool at the first read; fixed while the leaf stays bound.
        self._sharded_layer_ids: list[int] | None = None
        # Prepares so far: a read of an older forward neither resumes nor
        # cleans up after a newer prepare.
        self._generation = 0
        # The fused windows' checksums, by window index, for this forward.
        self._checksums: dict[int, int] = {}

    @classmethod
    def on_history_groups(
        cls, placement: CachePlacement, *, page_size: int
    ) -> PrefixHistoryGather:
        """Reads that gather on the named KVP history groups, which the
        distributed initializer creates whenever KVP is on, or through the
        group's fused transport when startup built one."""
        comm = get_global_backend().for_named_groups(KVP_HISTORY_GROUP_NAME)
        return cls(
            placement,
            page_size=page_size,
            comm=comm,
            transport=_FUSED_HISTORY_TRANSPORTS.get(placement.group),
        )

    def prepare(
        self,
        chunk_kv_indices_list: Sequence[torch.Tensor],
        *,
        page_table_cpu: torch.Tensor | None,
        chunk_bounds_cpu: torch.Tensor,
    ) -> None:
        """Record a forward's prefix-chunk loops; plans nothing yet.

        Waits for the gathers an earlier forward left in flight first, so it
        runs outside any graph capture while there are some: the server
        captures only before it serves, when none can be.

        Args:
            chunk_kv_indices_list: Each loop's ``[rows]`` virtual slots,
                position order (``chunk.get_chunks_paged``), identical on
                every rank.
            page_table_cpu: ``[requests, columns]`` host kernel-page table of
                the extend requests; required when there are loops.
            chunk_bounds_cpu: ``[loops + 1, requests]`` host prefix positions.
        """
        self._generation += 1
        self._checksums.clear()
        # A forward that stopped before draining its reads leaves gathers in
        # flight; they are waited on before they are dropped.
        self._abandon()
        if chunk_kv_indices_list and page_table_cpu is None:
            raise RuntimeError(
                "a page-sharded MLA prefix needs the host page table to split "
                "its history gather by page owner"
            )
        self._chunk_kv_indices_list = chunk_kv_indices_list
        self._page_table_cpu = page_table_cpu
        self._chunk_bounds_cpu = chunk_bounds_cpu
        # A forward without prefix loops has nothing to gather.
        self._windows = None if chunk_kv_indices_list else ()

    def windows(self, pool: MLATokenToKVPool) -> tuple[PrefixHistoryWindow, ...]:
        """The current forward's windows of ``pool``'s rows, in loop order,
        planned at the first call (the forward's first read makes it).

        Windows the fused transport takes join loops while their busiest
        owner fits its receive slot, which any one loop does; NCCL windows
        keep ``PREFIX_HISTORY_WINDOW_LOOPS`` loops, which bounds their
        all-gather's transient buffer. A reader whose rows the transport does
        not take (another row width, or virtual slots of another integer
        type) plans NCCL windows, judged from the pool's row layout: reading a
        plane would wait for its layer's Host load.
        """
        if self._windows is None:
            transport = self._transport
            fused = transport is not None and transport.accepts(
                pool.kv_cache_dim * pool.dtype.itemsize,
                self._chunk_kv_indices_list[0].dtype,
            )
            self._windows = plan_prefix_history_gather(
                self._chunk_kv_indices_list,
                page_table_cpu=self._page_table_cpu,
                chunk_bounds_cpu=self._chunk_bounds_cpu,
                page_size=self._page_size,
                placement=self._placement,
                max_loops=None if fused else PREFIX_HISTORY_WINDOW_LOOPS,
                max_send_rows=(
                    transport.state.cap_bytes // transport.state.row_bytes
                    if fused
                    else None
                ),
            )
        return self._windows

    def read(
        self,
        pool: MLATokenToKVPool,
        layer: PagedAttention,
        chunk_kv_indices_list: Sequence[torch.Tensor],
        *,
        dst_dtype: torch.dtype,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield every prefix-chunk loop's latent and RoPE keys of ``layer``.

        Every loop's rows are read out of its gathered window through the
        pool's own latent read: bitwise the unsharded read, since the bytes
        are the owners'.

        A read stopped early (its generator closed or collected, or an
        exception) waits for the windows it left in flight and drops them --
        on the thread and stream it read on, outside a graph capture, and
        only before this reader's next prepare, which drains them otherwise --
        so the next reader, the draft after the target, finds the transport
        free; a failure there poisons the transport. A read resumed
        after the next prepare, on another thread or stream, or under a
        capture begun with a fused window in flight raises before it enqueues
        anything.

        Yields:
            ``[rows, 1, kv_lora_rank]`` and ``[rows, 1, qk_rope_head_dim]``
            of ``dst_dtype``, one pair per loop, in loop order.
        """
        if chunk_kv_indices_list is not self._chunk_kv_indices_list:
            raise ValueError("the prefix-chunk loops are not the prepared forward's")
        windows = self.windows(pool)
        if not windows:
            return
        generation = self._generation
        thread = threading.get_ident()
        device = windows[0].fetch_slots.device
        stream = torch.cuda.current_stream(device)
        try:
            for index, window in enumerate(windows):
                rows, epoch = self._take(pool, layer.layer_id, index)
                if index + 1 < len(windows):
                    self._launch(pool, layer.layer_id, index + 1)
                else:
                    following = self._next_sharded_layer(pool, layer.layer_id)
                    if following is not None:
                        self._launch(pool, following, 0)
                bounds = window.loop_bounds
                for loop, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
                    keys = pool.read_latent_rows(
                        rows, window.gathered_index[start:end], dst_dtype
                    )
                    if epoch is not None and loop == len(bounds) - 2:
                        # The read copied the window's last rows out: once it
                        # is enqueued, the slot is free for epoch + 2.
                        self._transport.release(epoch, self)
                    yield keys
                    # The raw stream handle, not torch.cuda.current_stream():
                    # this runs per loop on the host-bound prefill path.
                    if (
                        self._generation != generation
                        or threading.get_ident() != thread
                        or torch._C._cuda_getCurrentRawStream(device.index)
                        != stream.cuda_stream
                        or (
                            self._transport is not None
                            and self._transport.holds(self)
                            and torch.cuda.is_current_stream_capturing()
                        )
                    ):
                        raise RuntimeError(
                            "a prefix history read resumed after the next "
                            "forward's prepare, on another thread or stream, or "
                            "under a graph capture begun with a fused window in "
                            "flight"
                        )
        except BaseException as stop:
            # Inside a graph capture -- of the current stream, or of the read's
            # stream, which the cleanup waits on -- the waits would join the
            # capture or break it: this reader's next prepare, outside it,
            # drains instead.
            capturing = torch.cuda.is_current_stream_capturing()
            with torch.cuda.stream(stream):
                capturing = capturing or torch.cuda.is_current_stream_capturing()
            if (
                self._generation != generation
                or threading.get_ident() != thread
                or capturing
            ):
                raise
            try:
                with torch.cuda.stream(stream):
                    self._abandon()
            except Exception as failure:
                if isinstance(stop, GeneratorExit):
                    # A close: the cleanup's failure is the error to report.
                    raise failure from None
                stop.add_note(
                    f"the prefix history read's cleanup failed too: {failure!r}"
                )
            raise

    def _abandon(self) -> None:
        """Order the reads' stream after every gather in flight, then drop
        them: an NCCL window's work is waited on the current stream, the fused
        transport's windows on its consumer stream. Every step is tried; the
        first failure is raised after the rest, with the transport poisoned."""
        failure = None
        for launched in self._launched.values():
            if launched.epoch is None:
                try:
                    launched.work.wait()
                except Exception as error:
                    failure = failure or error
        self._launched.clear()
        if self._transport is not None:
            try:
                self._transport.drain(self)
            except Exception as error:
                failure = failure or error
            if failure is not None:
                self._transport.poisoned = True
        if failure is not None:
            raise failure

    def _launch(self, pool: MLATokenToKVPool, layer_id: int, index: int) -> None:
        """Start all-gathering window ``index`` of ``layer_id`` from the page
        owners: through the fused transport when it takes the window, else
        every rank copies the rows it owns out of its cache as raw cache-dtype
        bytes and one all-gather collects every owner's."""
        if (layer_id, index) in self._launched:
            return
        window = self.windows(pool)[index]
        cache = pool.get_key_buffer(layer_id)
        if self._transport is not None:
            checksum = self._checksums.get(index)
            if checksum is None:
                owned = window.owned_rows_per_rank
                checksum = zlib.crc32(
                    struct.pack(
                        f"<{3 + len(owned)}q",
                        self._generation,
                        index,
                        window.send_rows,
                        *owned,
                    )
                )
                self._checksums[index] = checksum
            # The window's identity, equal on every rank, which plans the same
            # forward and reads the layers in the same order; the kernel
            # checks it across ranks.
            header = (layer_id << 32) | checksum
            fused = self._transport.launch(cache, window, header, reader=self)
            if fused is not None:
                rows, done, epoch = fused
                self._launched[(layer_id, index)] = _LaunchedGather(
                    rows=rows, work=done, epoch=epoch
                )
                return
        # Byte rows of any row stride; the send rows land contiguous.
        send = cache.view(torch.uint8).index_select(0, window.fetch_slots)
        gathered = send.new_empty(
            (len(window.group) * window.send_rows, *send.shape[1:])
        )
        work = all_gather_single_async(gathered, send, window.group, backend=self._comm)
        self._launched[(layer_id, index)] = _LaunchedGather(
            rows=gathered.view(cache.dtype), work=work, epoch=None
        )

    def _take(
        self, pool: MLATokenToKVPool, layer_id: int, index: int
    ) -> tuple[torch.Tensor, int | None]:
        """Window ``index`` of ``layer_id``, gathered, for the current stream:
        ``[degree * send_rows, 1, kv_cache_dim]`` (or a fused slot's
        ``[degree * cap_rows, 1, kv_cache_dim]``) rank-major rows in the cache
        dtype, which ``window.gathered_index`` addresses, and its fused epoch
        (None on the all-gather)."""
        self._launch(pool, layer_id, index)
        launched = self._launched[(layer_id, index)]
        if launched.epoch is not None:
            self._transport.taken(launched.epoch, self)
        launched.work.wait()
        # Dropped only once taken: a failed take leaves it to the cleanup.
        del self._launched[(layer_id, index)]
        return launched.rows, launched.epoch

    def _next_sharded_layer(self, pool: MLATokenToKVPool, layer_id: int) -> int | None:
        """The layer after ``layer_id`` whose history rides the same group."""
        if self._sharded_layer_ids is None:
            groups = pool.history_group_by_layer()
            self._sharded_layer_ids = sorted(
                other for other, group in groups.items() if group == groups[layer_id]
            )
        index = bisect.bisect_right(self._sharded_layer_ids, layer_id)
        if index == len(self._sharded_layer_ids):
            return None
        return self._sharded_layer_ids[index]
