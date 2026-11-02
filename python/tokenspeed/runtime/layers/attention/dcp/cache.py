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
    all_gather_single_async,
    token_all_gather_rows,
)
from tokenspeed.runtime.distributed.mapping import KVP_HISTORY_GROUP_NAME
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    cyclic_slot_owner,
    owned_prefix_chunk_rows,
    resolve_cache_slots,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass(frozen=True)
class HistoryGatherWorkspace:
    """The buffers a request group's gathered history lands in.

    Allocated once per engine (the recipe reserves the bytes from the cache
    budget, ``OrdinaryRecipe.workspace_bytes``) by the target's DSA leaf and
    shared with the draft's: the two never gather at the same time, since the
    draft's extend step follows the target's forward on the same stream.

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
        return self.kv.nbytes + self.index_k.nbytes


@dataclass(frozen=True)
class HistoryGatherPlan:
    """How one request group's history rows split over the page owners.

    Built once per forward and group from the virtual slots of every history
    row (prefix plus this chunk, request-major, position order) and the
    host-side owned-row counts; reused by every layer's KV and index-K
    gathers. Rank ``k``'s rows are the ``owned_rows_per_rank[k]`` rows whose
    owner is ``k``, in position order, so the all-gathered buffer is
    rank-major and ``order`` maps it back to position order.

    Attributes:
        virtual_slots: ``[rows]`` int64 virtual cache slots in position order.
        local_slots: ``[rows]`` int64 this rank's physical slots (0 where
            another rank owns the row).
        order: ``[rows]`` int64 permutation: position of the i-th row of the
            rank-major gathered buffer.
        owned_rows_per_rank: Host counts of rows each owner contributes.
        group: The owner group the gather runs over (a singleton without
            page sharding).
        rank: This rank's position in ``group``.
    """

    virtual_slots: torch.Tensor
    local_slots: torch.Tensor
    order: torch.Tensor
    owned_rows_per_rank: tuple[int, ...]
    group: tuple[int, ...]
    rank: int

    @property
    def rows(self) -> int:
        return sum(self.owned_rows_per_rank)

    @property
    def local_fetch_slots(self) -> torch.Tensor:
        """Physical slots of the rows this rank contributes, position order."""
        first = sum(self.owned_rows_per_rank[: self.rank])
        mine = self.order[first : first + self.owned_rows_per_rank[self.rank]]
        return self.local_slots.index_select(0, mine)


def plan_history_gather(
    virtual_slots: torch.Tensor,
    *,
    placement: CachePlacement | None,
    owned_rows_per_rank: Sequence[int],
) -> HistoryGatherPlan:
    """Plan the owner-split gather of a request group's history rows.

    Args:
        virtual_slots: ``[rows]`` int64 virtual slots of every history row,
            position order, identical on every rank.
        placement: The group's page ownership, or ``None`` when this rank
            holds every row (the gather is then a local slot gather).
        owned_rows_per_rank: Host counts of rows each owner contributes
            (``owned_history_rows`` summed over the group's requests); one
            entry without a placement.

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
    if sum(counts) != virtual_slots.numel():
        raise ValueError(
            f"owned rows sum to {sum(counts)} but the history has "
            f"{virtual_slots.numel()} rows"
        )
    local_slots, _owned = resolve_cache_slots(virtual_slots, placement)
    owner = cyclic_slot_owner(virtual_slots, placement)
    order = torch.argsort(owner, stable=True)
    return HistoryGatherPlan(
        virtual_slots=virtual_slots,
        local_slots=local_slots,
        order=order,
        owned_rows_per_rank=counts,
        group=group,
        rank=rank,
    )


# Row dtypes the index-copy kernels implement; others scatter as bytes.
_SCATTER_DTYPES = frozenset(
    {torch.bfloat16, torch.float16, torch.float32, torch.uint8, torch.int32}
)


def gather_history_rows(
    plan: HistoryGatherPlan,
    local_rows: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Assemble a request group's history rows in position order.

    Every rank contributes the rows it owns (``plan.local_fetch_slots`` of its
    own cache plane, as 2-D rows); one all-gather with per-rank counts puts
    them rank-major, and the plan's order scatters them into position order.
    Pure data movement: a row's bytes are the owner's bytes wherever the
    gather lands them, whatever their dtype (:func:`token_all_gather_rows`
    moves them as bf16 pairs of their bytes). Without page sharding the rows
    are already local and only the scatter runs.

    Args:
        plan: The group's gather plan.
        local_rows: ``[owned_rows_per_rank[rank], width]`` rows this rank
            contributes, position order; any dtype with an even row byte
            width.
        out: ``[>= rows, width]`` destination of ``local_rows``' dtype; the
            leading ``rows`` are written.

    Returns:
        ``out[:rows]``.
    """
    rows = plan.rows
    if (
        local_rows.dim() != 2
        or local_rows.shape[0] != plan.owned_rows_per_rank[plan.rank]
    ):
        raise ValueError(
            f"rank {plan.rank} contributes {plan.owned_rows_per_rank[plan.rank]} "
            f"rows, got {tuple(local_rows.shape)}"
        )
    if (
        out.dim() != 2
        or out.shape[0] < rows
        or out.shape[1] != local_rows.shape[1]
        or out.dtype != local_rows.dtype
    ):
        raise ValueError(
            f"history destination {tuple(out.shape)} {out.dtype} does not hold "
            f"{rows} rows of {local_rows.shape[1]} {local_rows.dtype}"
        )
    if len(plan.group) == 1:
        gathered = local_rows
    else:
        gathered = token_all_gather_rows(
            local_rows, plan.group, list(plan.owned_rows_per_rank)
        )
    target = out[:rows]
    if local_rows.dtype in _SCATTER_DTYPES:
        target.index_copy_(0, plan.order, gathered)
    else:
        # fp8 rows: the index-copy kernels lack the dtype; scatter the bytes.
        target.view(torch.uint8).index_copy_(
            0, plan.order, gathered.contiguous().view(torch.uint8)
        )
    return target


# Consecutive prefix-chunk loops one history all-gather covers. A gather costs
# a fixed host launch whatever it moves, so one per layer serves a typical
# multi-turn history; the window still bounds the transient buffer, at about
# this many loops' rows of cache-dtype bytes.
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
    prefix_lens_cpu: torch.Tensor,
    chunk_len: int,
    page_size: int,
    placement: CachePlacement,
) -> tuple[PrefixHistoryWindow, ...]:
    """Plan one forward's page-owner gathers of its prefix-chunk loops.

    Consecutive loops share a window (``PREFIX_HISTORY_WINDOW_LOOPS``); every
    MLA layer reuses the plan. The owned-row counts come from the host page
    table, so nothing waits on the device; ``split_slots_by_owner`` gives
    every row its all-gather output row and this rank's rows their physical slots.

    Args:
        chunk_kv_indices_list: Each loop's ``[rows]`` virtual slots, position
            order (``chunk.get_chunks_paged``), identical on every rank.
        page_table_cpu: ``[requests, columns]`` host kernel-page table of the
            extend requests.
        prefix_lens_cpu: ``[requests]`` host cached-prefix lengths.
        chunk_len: Prefix rows every request contributes to one loop.
        page_size: Tokens per kernel page.
        placement: The MLA history group's page ownership.

    Returns:
        The windows, in loop order.
    """
    counts = owned_prefix_chunk_rows(
        page_table_cpu,
        prefix_lens_cpu,
        chunk_len=chunk_len,
        num_chunks=len(chunk_kv_indices_list),
        page_size=page_size,
        placement=placement,
    )
    degree = len(placement.group)
    windows = []
    for first in range(0, len(chunk_kv_indices_list), PREFIX_HISTORY_WINDOW_LOOPS):
        loops = chunk_kv_indices_list[first : first + PREFIX_HISTORY_WINDOW_LOOPS]
        slots = torch.cat(loops) if len(loops) > 1 else loops[0]
        owned = tuple(counts[first : first + len(loops)].sum(dim=0).tolist())
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


@dataclass
class _LaunchedGather:
    """A window gather in flight."""

    rows: torch.Tensor
    work: CommWork


class PrefixHistoryGather:
    """A page-sharded MLA leaf's prefix-chunk history reads.

    :meth:`prepare` records each forward's prefix-chunk loops; :meth:`read`
    yields one layer's rows of every loop. The rows come from their page
    owners, in windows of consecutive loops (``PREFIX_HISTORY_WINDOW_LOOPS``)
    that the forward's first read plans: planning is host launch work, so
    deferring it from before the forward's first kernel to that layer lets
    it run while the device works through the layers queued ahead.

    Each window is all-gathered asynchronously on ``comm``, a backend on the
    named KVP history groups (``KVP_HISTORY_GROUP_NAME``) whose communicators
    are apart from the attention TP collectives over the same ranks, one
    window ahead: a read takes its window and starts the next one it will
    need -- this layer's next, else the next sharded layer's first -- so that
    transfer runs under the compute in between (for a hybrid model, the
    linear-attention layers between two MLA layers) instead of queueing with
    those collectives. The gathers are collectives: every rank reads every
    layer's history in the same order.
    """

    def __init__(self, placement: CachePlacement, *, page_size: int, comm: CommBackend):
        self._placement = placement
        self._page_size = page_size
        self._comm = comm
        # The current forward's loops and planning inputs, and its windows
        # once the first read planned them.
        self._chunk_kv_indices_list: Sequence[torch.Tensor] = ()
        self._page_table_cpu: torch.Tensor | None = None
        self._prefix_lens_cpu: torch.Tensor | None = None
        self._chunk_len = 0
        self._windows: tuple[PrefixHistoryWindow, ...] | None = None
        self._launched: dict[tuple[int, int], _LaunchedGather] = {}
        # Ascending ids of the layers sharing this history, learned from the
        # pool at the first read; fixed while the leaf stays bound.
        self._sharded_layer_ids: list[int] | None = None

    @classmethod
    def on_history_groups(
        cls, placement: CachePlacement, *, page_size: int
    ) -> PrefixHistoryGather:
        """Reads that gather on the named KVP history groups, which the
        distributed initializer creates whenever KVP is on."""
        comm = get_global_backend().for_named_groups(KVP_HISTORY_GROUP_NAME)
        return cls(placement, page_size=page_size, comm=comm)

    def prepare(
        self,
        chunk_kv_indices_list: Sequence[torch.Tensor],
        *,
        page_table_cpu: torch.Tensor | None,
        prefix_lens_cpu: torch.Tensor,
        chunk_len: int,
    ) -> None:
        """Record a forward's prefix-chunk loops; plans nothing yet.

        Args:
            chunk_kv_indices_list: Each loop's ``[rows]`` virtual slots,
                position order (``chunk.get_chunks_paged``), identical on
                every rank.
            page_table_cpu: ``[requests, columns]`` host kernel-page table of
                the extend requests; required when there are loops.
            prefix_lens_cpu: ``[requests]`` host cached-prefix lengths.
            chunk_len: Prefix rows every request contributes to one loop.
        """
        # A forward that stopped before draining its reads leaves gathers in
        # flight; the current stream waits on them before they are dropped.
        for launched in self._launched.values():
            launched.work.wait()
        self._launched.clear()
        if chunk_kv_indices_list and page_table_cpu is None:
            raise RuntimeError(
                "a page-sharded MLA prefix needs the host page table to split "
                "its history gather by page owner"
            )
        self._chunk_kv_indices_list = chunk_kv_indices_list
        self._page_table_cpu = page_table_cpu
        self._prefix_lens_cpu = prefix_lens_cpu
        self._chunk_len = chunk_len
        # A forward without prefix loops has nothing to gather.
        self._windows = None if chunk_kv_indices_list else ()

    @property
    def windows(self) -> tuple[PrefixHistoryWindow, ...]:
        """The current forward's windows, in loop order (planned at first
        access)."""
        if self._windows is None:
            self._windows = plan_prefix_history_gather(
                self._chunk_kv_indices_list,
                page_table_cpu=self._page_table_cpu,
                prefix_lens_cpu=self._prefix_lens_cpu,
                chunk_len=self._chunk_len,
                page_size=self._page_size,
                placement=self._placement,
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

        Yields:
            ``[rows, 1, kv_lora_rank]`` and ``[rows, 1, qk_rope_head_dim]``
            of ``dst_dtype``, one pair per loop, in loop order.
        """
        if chunk_kv_indices_list is not self._chunk_kv_indices_list:
            raise ValueError("the prefix-chunk loops are not the prepared forward's")
        windows = self.windows
        for index, window in enumerate(windows):
            rows = self._take(pool, layer.layer_id, index)
            if index + 1 < len(windows):
                self._launch(pool, layer.layer_id, index + 1)
            else:
                following = self._next_sharded_layer(pool, layer.layer_id)
                if following is not None:
                    self._launch(pool, following, 0)
            bounds = window.loop_bounds
            for start, end in zip(bounds[:-1], bounds[1:]):
                yield pool.read_latent_rows(
                    rows, window.gathered_index[start:end], dst_dtype
                )

    def _launch(self, pool: MLATokenToKVPool, layer_id: int, index: int) -> None:
        """Start all-gathering window ``index`` of ``layer_id`` from the page
        owners: every rank copies the rows it owns out of its cache as raw
        cache-dtype bytes and one all-gather collects every owner's."""
        if (layer_id, index) in self._launched:
            return
        window = self.windows[index]
        cache = pool.get_key_buffer(layer_id)
        # Byte rows of any row stride; the send rows land contiguous.
        send = cache.view(torch.uint8).index_select(0, window.fetch_slots)
        gathered = send.new_empty(
            (len(window.group) * window.send_rows, *send.shape[1:])
        )
        work = all_gather_single_async(gathered, send, window.group, backend=self._comm)
        self._launched[(layer_id, index)] = _LaunchedGather(
            rows=gathered.view(cache.dtype), work=work
        )

    def _take(self, pool: MLATokenToKVPool, layer_id: int, index: int) -> torch.Tensor:
        """Window ``index`` of ``layer_id``, gathered, for the current stream:
        ``[degree * send_rows, 1, kv_cache_dim]`` rank-major rows in the cache
        dtype, which ``window.gathered_index`` addresses."""
        self._launch(pool, layer_id, index)
        launched = self._launched.pop((layer_id, index))
        launched.work.wait()
        return launched.rows

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
