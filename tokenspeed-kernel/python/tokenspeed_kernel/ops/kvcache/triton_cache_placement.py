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

"""Place virtual cache blocks, token slots and kernel pages on local ranks.

Under KV parallelism (KVP) a sharded cache group deals its virtual blocks
page-cyclically to the ranks of the KVP group; one cyclic block-placement rule
serves both position-preserving slot mapping and compact page tables with
local token counts. These operations transform addresses only; KV payload
reads/writes belong to their cache-layout kernels.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def virtual_block_to_local(block, DEGREE: tl.constexpr, RANK: tl.constexpr):
    """Return a safe physical page and its explicit owner mask."""
    positive = tl.maximum(block - 1, 0)
    owned = (block > 0) & (positive % DEGREE == RANK)
    local = positive // DEGREE + 1
    return tl.where(owned, local, 0), owned


# The slot count follows the batch; unspecialized so every count shares one binary.
@triton.jit(do_not_specialize=["count"])
def _translate_virtual_slots(
    source,
    destination,
    owner_mask,
    count,
    ROWS: tl.constexpr,
    VIRTUAL_COUNT: tl.constexpr,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < count
    raw = tl.load(source + offsets, mask=active, other=0).to(tl.int64)
    safe = tl.maximum(raw, 0)
    local, owner = virtual_block_to_local(safe // ROWS, DEGREE, RANK)
    owner = owner & (raw >= ROWS) & (raw < VIRTUAL_COUNT * ROWS)
    slot = tl.where(owner, local * ROWS + safe % ROWS, 0)
    tl.store(destination + offsets, slot, mask=active)
    tl.store(owner_mask + offsets, owner, mask=active)


def virtual_slots_to_local(
    slots: torch.Tensor,
    *,
    rows_per_page: int,
    virtual_block_count: int,
    degree: int,
    rank: int,
    out: torch.Tensor | None = None,
    owner_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate scheduler slots, with page 0 and nonowners explicitly invalid.

    Args:
        slots: Integer virtual slots; negative entries are also invalid.
        rows_per_page: Unchanged physical rows per owned cache block. Use 1
            to translate a block table instead of row slots.
        virtual_block_count: Scheduler capacity including null block 0.
        degree: Number of owners (the KVP group width), or 1 for a
            replicated group.
        rank: Owner index in [0, degree) (the KVP rank); replicated groups
            use 0.
        out: Optional contiguous output tensor with the same shape/dtype.
        owner_mask: Optional contiguous boolean output of the same shape.

    Returns:
        Safe local slots and a boolean mask. Every false-mask slot is 0;
        callers must use the mask to suppress all payload and scale stores.
    """
    if rows_per_page <= 0 or virtual_block_count <= 1 or degree <= 0:
        raise ValueError(
            "virtual slot geometry must be positive and include usable pages"
        )
    if not 0 <= rank < degree:
        raise ValueError("virtual slot owner rank is out of range")
    if slots.dtype not in (torch.int32, torch.int64):
        raise TypeError("virtual slots must be int32 or int64")
    if out is None:
        out = torch.empty_like(slots, memory_format=torch.contiguous_format)
    if owner_mask is None:
        owner_mask = torch.empty(slots.shape, dtype=torch.bool, device=slots.device)
    for value, dtype in ((out, slots.dtype), (owner_mask, torch.bool)):
        if (
            value.shape != slots.shape
            or value.dtype != dtype
            or value.device != slots.device
            or not value.is_contiguous()
        ):
            raise ValueError("virtual slot outputs must match shape, device and dtype")
    if slots.numel() == 0:
        return out, owner_mask
    if slots.is_cuda or slots.device.type == "npu":
        source = slots.contiguous()
        _translate_virtual_slots[(triton.cdiv(slots.numel(), 256),)](
            source,
            out,
            owner_mask,
            slots.numel(),
            ROWS=rows_per_page,
            VIRTUAL_COUNT=virtual_block_count,
            DEGREE=degree,
            RANK=rank,
            BLOCK=256,
        )
    else:
        safe = slots.to(torch.int64).clamp_min(0)
        block = safe // rows_per_page
        positive = (block - 1).clamp_min(0)
        owned = (
            (slots >= rows_per_page)
            & (slots < virtual_block_count * rows_per_page)
            & (positive % degree == rank)
        )
        local = (positive // degree + 1) * rows_per_page + safe % rows_per_page
        out.copy_(torch.where(owned, local, 0))
        owner_mask.copy_(owned)
    return out, owner_mask


@triton.jit(do_not_specialize=["COUNT"])
def _count_slots_by_owner(
    Slots,
    Counts,
    # The row count follows the forward's history; runtime so every history
    # shares one binary.
    COUNT,
    ROWS: tl.constexpr,
    DEGREE: tl.constexpr,
    OWNERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Counts[b, o]: rows of block b that owner o holds.
    block = tl.program_id(0)
    offsets = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < COUNT
    slot = tl.load(Slots + offsets, mask=active, other=ROWS).to(tl.int64)
    owner = ((slot // ROWS - 1) % DEGREE).to(tl.int32)
    owners = tl.arange(0, OWNERS)
    hit = (owner[:, None] == owners[None, :]) & active[:, None]
    tl.store(
        Counts + block.to(tl.int64) * OWNERS + owners, tl.sum(hit.to(tl.int32), axis=0)
    )


@triton.jit(do_not_specialize=["SSTRIDE", "OSTRIDE", "WIDTH"])
def _compact_owned_slots(
    Slots,
    Lens,
    Out,
    Counts,
    # Batch-dependent width and strides stay runtime: one binary for every batch shape.
    SSTRIDE,
    OSTRIDE,
    WIDTH,
    ROWS: tl.constexpr,
    VIRTUAL_COUNT: tl.constexpr,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per row carries the owned count, so owned slots keep selection order.
    row = tl.program_id(0).to(tl.int64)
    if Lens is None:
        length = WIDTH
    else:
        length = tl.minimum(tl.maximum(tl.load(Lens + row), 0), WIDTH)
    count = 0
    for start in range(0, WIDTH, BLOCK):
        col = start + tl.arange(0, BLOCK)
        live = col < length
        raw = tl.load(Slots + row * SSTRIDE + col, mask=live, other=-1).to(tl.int64)
        safe = tl.maximum(raw, 0)
        local, owner = virtual_block_to_local(safe // ROWS, DEGREE, RANK)
        valid = live & owner & (raw >= ROWS) & (raw < VIRTUAL_COUNT * ROWS)
        dest = count + tl.cumsum(valid.to(tl.int32), axis=0) - 1
        tl.store(
            Out + row * OSTRIDE + dest,
            (local * ROWS + safe % ROWS).to(tl.int32),
            mask=valid,
        )
        count += tl.sum(valid.to(tl.int32), axis=0)
    # Padding is disjoint from the owned stores, so the loops need no ordering.
    for start in range(0, WIDTH, BLOCK):
        col = start + tl.arange(0, BLOCK)
        tl.store(
            Out + row * OSTRIDE + col,
            tl.full((BLOCK,), -1, tl.int32),
            mask=(col >= count) & (col < WIDTH),
        )
    tl.store(Counts + row, count)


def compact_owned_slots(
    slots: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    rows_per_page: int,
    virtual_block_count: int,
    degree: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one owner's selected slots, as local slots, at each row's front.

    A row's live entries are its first ``lengths[row]`` columns. Of those,
    the ones ``virtual_slots_to_local`` marks as owned by ``rank`` move to
    the front in their original order, translated to local slots; the rest
    of the row is ``-1``. A sparse core that reads only ``[0, count)`` then
    visits this rank's share of the selection instead of the whole width.

    Args:
        slots: Int32 or int64 virtual slots [rows, width]; negative entries
            are invalid.
        lengths: Int32 or int64 live entries per row [rows], clamped to
            [0, width], or None when every column is live.
        rows_per_page: Unchanged physical rows per owned cache block.
        virtual_block_count: Scheduler capacity including null block 0.
        degree: Number of owners (the KVP group width).
        rank: Owner index in [0, degree).

    Returns:
        ``(local_slots, counts)``: int32 [rows, width] local slots, owned
        entries first and ``-1`` after, and int32 [rows] owned counts. Both
        are new tensors; ``slots`` is not modified.
    """
    if rows_per_page <= 0 or virtual_block_count <= 1 or degree <= 0:
        raise ValueError(
            "virtual slot geometry must be positive and include usable pages"
        )
    if not 0 <= rank < degree:
        raise ValueError("virtual slot owner rank is out of range")
    if slots.dtype not in (torch.int32, torch.int64):
        raise TypeError("virtual slots must be int32 or int64")
    if slots.dim() != 2:
        raise ValueError("owned slot compaction takes [rows, width] slots")
    if lengths is not None and (
        lengths.shape != slots.shape[:1]
        or lengths.dtype not in (torch.int32, torch.int64)
        or lengths.device != slots.device
    ):
        raise ValueError(
            "owned slot lengths must be int32 or int64 [rows] on the slots' device"
        )
    rows, width = slots.shape
    out = torch.empty((rows, width), dtype=torch.int32, device=slots.device)
    counts = torch.empty((rows,), dtype=torch.int32, device=slots.device)
    if slots.numel() == 0:
        return out, counts.zero_()
    if slots.is_cuda:
        source = slots if slots.stride(1) == 1 else slots.contiguous()
        _compact_owned_slots[(rows,)](
            source,
            None if lengths is None else lengths.contiguous(),
            out,
            counts,
            source.stride(0),
            out.stride(0),
            width,
            ROWS=rows_per_page,
            VIRTUAL_COUNT=virtual_block_count,
            DEGREE=degree,
            RANK=rank,
            BLOCK=256,
        )
    else:
        local, owned = virtual_slots_to_local(
            slots,
            rows_per_page=rows_per_page,
            virtual_block_count=virtual_block_count,
            degree=degree,
            rank=rank,
        )
        cols = torch.arange(width, device=slots.device)
        if lengths is not None:
            owned &= cols < lengths.to(torch.int64).clamp(0, width).unsqueeze(1)
        # Stable partition onto unique destinations: each output is written once.
        prefix = owned.cumsum(1)
        counts.copy_(prefix[:, -1])
        destinations = torch.where(owned, prefix - 1, prefix[:, -1:] + cols - prefix)
        out.scatter_(1, destinations, torch.where(owned, local, -1).to(torch.int32))
    return out, counts


@triton.jit(do_not_specialize=["COUNT", "SEND_ROWS"])
def _split_slots_by_owner(
    Slots,
    Carry,
    GatheredIndex,
    Fetch,
    # The row count follows the forward's history and the send width its
    # largest owner; runtime so every history shares one binary.
    COUNT,
    SEND_ROWS,
    ROWS: tl.constexpr,
    DEGREE: tl.constexpr,
    OWNERS: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each program places one block of rows; Carry[b, o] is owner o's row
    # count in the blocks before b.
    block = tl.program_id(0)
    owners = tl.arange(0, OWNERS)
    # A single tile has no preceding rows. Keep the grid check at runtime
    # so crossing the one-tile boundary does not compile a new binary.
    if tl.num_programs(0) == 1:
        carry = tl.zeros((OWNERS,), dtype=tl.int32)
    else:
        carry = tl.load(Carry + block.to(tl.int64) * OWNERS + owners)
    offsets = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < COUNT
    slot = tl.load(Slots + offsets, mask=active, other=ROWS).to(tl.int64)
    previous = slot // ROWS - 1
    owner = (previous % DEGREE).to(tl.int32)
    hit = (owner[:, None] == owners[None, :]) & active[:, None]
    running = tl.cumsum(hit.to(tl.int32), axis=0) + carry[None, :]
    within = tl.sum(tl.where(hit, running, 0), axis=1) - 1
    tl.store(
        GatheredIndex + offsets, owner.to(tl.int64) * SEND_ROWS + within, mask=active
    )
    local = (previous // DEGREE + 1) * ROWS + slot % ROWS
    tl.store(
        Fetch + within,
        local.to(Fetch.dtype.element_ty),
        mask=active & (owner == RANK),
    )


def split_slots_by_owner(
    slots: torch.Tensor,
    *,
    block_granularity: int,
    degree: int,
    rank: int,
    send_rows: int,
    gathered_index: torch.Tensor,
    fetch: torch.Tensor,
) -> None:
    """Lay rows out for an all-gather over their cyclic page owners.

    Row ``i`` belongs to owner ``(slots[i] // block_granularity - 1) % degree``,
    the rule of ``virtual_slots_to_local``, and every owner keeps its rows in
    order. With every owner sending ``send_rows`` rows, the rank-major
    all-gather output holds row ``i`` at ``owner * send_rows`` plus the count
    of that owner's rows before ``i``.

    Args:
        slots: Int32 or int64 virtual slots [rows], contiguous. Every row must
            have an owner: ``slots >= block_granularity``.
        block_granularity: Tokens per scheduler ownership block.
        degree: Number of owners.
        rank: The owner whose rows ``fetch`` receives.
        send_rows: Rows every owner contributes; at least any owner's count.
            The gathered span, ``degree * send_rows``, must fit int32 indexing.
        gathered_index: Int32 output [rows]: each row's all-gather output row.
        fetch: Output [>= rank's row count] in the slots' dtype: the physical
            slots of ``rank``'s rows, in order. Entries past that count keep
            their contents.

    Returns:
        None; both outputs are written in place.
    """
    if block_granularity <= 0 or degree <= 0 or not 0 <= rank < degree:
        raise ValueError("invalid owner split geometry")
    # GatheredIndex and the per-owner prefix counts use signed int32.
    if send_rows < 0 or degree * send_rows > 2**31:
        raise ValueError("owner split gather rows must fit int32 indices")
    if (
        slots.dim() != 1
        or not slots.is_contiguous()
        or gathered_index.shape != slots.shape
        or gathered_index.dtype != torch.int32
        or fetch.dim() != 1
        or fetch.dtype != slots.dtype
    ):
        raise ValueError("owner split buffers disagree with the slots")
    if slots.numel() == 0:
        return
    if slots.is_cuda:
        # Count each block's rows per owner, scan the counts across blocks,
        # then place every block in parallel from its owners' carried counts.
        owners = triton.next_power_of_2(degree)
        blocks = triton.cdiv(slots.numel(), 1024)
        # The one-tile kernel does not read Carry; reuse an int32 output
        # pointer to keep the same signature without allocating scratch.
        carry = gathered_index
        if blocks > 1:
            counts = torch.empty(
                (blocks, owners), dtype=torch.int32, device=slots.device
            )
            _count_slots_by_owner[(blocks,)](
                slots,
                counts,
                slots.numel(),
                ROWS=block_granularity,
                DEGREE=degree,
                OWNERS=owners,
                BLOCK=1024,
            )
            carry = torch.cumsum(counts, dim=0, dtype=torch.int32).sub_(counts)
        _split_slots_by_owner[(blocks,)](
            slots,
            carry,
            gathered_index,
            fetch,
            slots.numel(),
            send_rows,
            ROWS=block_granularity,
            DEGREE=degree,
            OWNERS=owners,
            RANK=rank,
            BLOCK=1024,
        )
    else:
        previous = slots.to(torch.int64) // block_granularity - 1
        owner = previous % degree
        seen = torch.cumsum(
            torch.arange(degree).unsqueeze(1) == owner.unsqueeze(0), dim=1
        )
        within = seen.gather(0, owner.unsqueeze(0)).squeeze(0) - 1
        gathered_index.copy_(owner * send_rows + within)
        mine = owner == rank
        local = (previous // degree + 1) * block_granularity
        local += slots.to(torch.int64) % block_granularity
        fetch[within[mine]] = local[mine].to(fetch.dtype)


@triton.jit
def _compact_owned_pages(
    Table,
    Lens,
    Out,
    LocalLens,
    PagePrefix,
    # Table geometry follows the batch; runtime so every batch shape shares
    # one binary. BLOCK buckets COLS for the column range.
    TSTRIDE,
    OSTRIDE,
    COLS,
    PAGE: tl.constexpr,
    SUBPAGES: tl.constexpr,
    VIRTUAL_COUNT: tl.constexpr,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    page = tl.load(Table + row * TSTRIDE + col, col < COLS, 0)
    length = tl.load(Lens + row)
    block = page // SUBPAGES
    local_block, owner = virtual_block_to_local(block, DEGREE, RANK)
    owner &= block < VIRTUAL_COUNT
    valid = (col < COLS) & owner
    if PagePrefix is None:
        valid &= col * PAGE < length
    dest = tl.cumsum(valid.to(tl.int32)) - 1
    if PagePrefix is not None:
        tl.store(PagePrefix + row * (COLS + 1), 0)
        tl.store(PagePrefix + row * (COLS + 1) + col + 1, dest + 1, col < COLS)
    local = local_block * SUBPAGES + page % SUBPAGES
    tl.store(Out + row * OSTRIDE + dest, local, valid)
    rows = tl.minimum(PAGE, tl.maximum(length - col * PAGE, 0))
    count = tl.sum(tl.where(valid, rows, 0))
    tl.store(LocalLens + row, count)


def _compact_dcp_pages_ascend(
    table,
    lengths,
    *,
    page_size,
    subpages,
    virtual_block_count,
    degree,
    rank,
    out,
    local_lengths,
    page_prefix,
):
    # The Ascend Triton launcher for this kernel initializes a lock with
    # synchronous rtMemcpy. That launcher cannot execute during graph capture.
    # An inclusive owner scan gives stable destinations without sorting.
    # Invalid pages receive the remaining, also unique, destinations so scatter
    # writes each output element once and clears every unused table entry.
    cols = torch.arange(table.shape[1], device=table.device, dtype=torch.int64)
    blocks = table // subpages
    valid = (blocks > 0) & (blocks < virtual_block_count)
    valid &= (blocks - 1) % degree == rank
    if page_prefix is None:
        valid &= cols[None, :] * page_size < lengths[:, None]
    prefix = valid.cumsum(1, dtype=torch.int32)
    if page_prefix is not None:
        page_prefix[:, 0].zero_()
        page_prefix[:, 1:].copy_(prefix)
    destinations = torch.where(
        valid, prefix - 1, prefix[:, -1:] + cols[None, :] - prefix
    )
    local_pages = ((blocks - 1) // degree + 1) * subpages + table % subpages
    out.scatter_(1, destinations.to(torch.int64), torch.where(valid, local_pages, 0))
    visible = (lengths[:, None] - cols[None, :] * page_size).clamp(0, page_size)
    local_lengths.copy_(torch.where(valid, visible, 0).sum(1))


def compact_kvp_pages(
    table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    page_size: int,
    block_granularity: int,
    virtual_block_count: int,
    degree: int,
    rank: int,
    out: torch.Tensor,
    local_lengths: torch.Tensor,
    page_prefix: torch.Tensor | None = None,
) -> None:
    """Pack owned kernel pages in token order into persistent output buffers.

    ``table`` holds virtual kernel pages, not scheduler blocks. ``lengths``
    are global causal endpoints. Outputs are local physical pages and the
    number of valid local tokens, including a possibly partial last page.
    Without page_prefix, zero-length shards retain a zero page table. With
    page_prefix, local_lengths still excludes reserve pages from attention.

    Args:
        table: Int32 virtual kernel pages [batch, max_pages].
        lengths: Global valid token counts [batch].
        page_size: Tokens per kernel page.
        block_granularity: Tokens per scheduler ownership block.
        virtual_block_count: Exclusive upper bound of scheduler block IDs.
        degree: Number of page owners (the KVP group width).
        rank: Owner index within the KVP group.
        out: Preallocated physical page table, with the same shape as table.
        local_lengths: Preallocated local token counts, shaped like lengths.
        page_prefix: Optional int32 output [batch, max_pages + 1], contiguous.
            When supplied, retain all allocated owned pages, including draft
            reserve pages beyond lengths, and record exclusive owned-page counts.
            Omitting it packs only pages intersecting the current lengths.

    Returns:
        None; both output buffers are refreshed in place.
    """
    if (
        page_size <= 0
        or block_granularity <= 0
        or block_granularity % page_size
        or virtual_block_count <= 1
        or not 0 <= rank < degree
    ):
        raise ValueError("invalid KVP page geometry")
    if (
        table.ndim != 2
        or out.shape != table.shape
        or local_lengths.shape != lengths.shape
    ):
        raise ValueError("KVP page buffers disagree")
    if table.device.type == "npu":
        _compact_dcp_pages_ascend(
            table,
            lengths,
            page_size=page_size,
            subpages=block_granularity // page_size,
            virtual_block_count=virtual_block_count,
            degree=degree,
            rank=rank,
            out=out,
            local_lengths=local_lengths,
            page_prefix=page_prefix,
        )
        return
    out.zero_()
    if table.is_cuda:
        _compact_owned_pages[(table.shape[0],)](
            table,
            lengths,
            out,
            local_lengths,
            page_prefix,
            table.stride(0),
            out.stride(0),
            table.shape[1],
            page_size,
            block_granularity // page_size,
            virtual_block_count,
            degree,
            rank,
            triton.next_power_of_2(table.shape[1]),
            # The prefix output increases live scan state. More threads limit
            # register spills on long rows, while short rows favor four warps.
            num_warps=8 if page_prefix is not None and table.shape[1] > 2048 else 4,
        )
    else:
        subpages = block_granularity // page_size
        for row in range(table.shape[0]):
            count = 0
            tokens = 0
            if page_prefix is not None:
                page_prefix[row, 0] = 0
            for col, page in enumerate(table[row].tolist()):
                block = page // subpages
                if page_prefix is None and col * page_size >= int(lengths[row]):
                    break
                if 0 < block < virtual_block_count and (block - 1) % degree == rank:
                    out[row, count] = (
                        (block - 1) // degree + 1
                    ) * subpages + page % subpages
                    count += 1
                    tokens += min(
                        page_size, max(int(lengths[row]) - col * page_size, 0)
                    )
                if page_prefix is not None:
                    page_prefix[row, col + 1] = count
            local_lengths[row] = tokens


@triton.jit(do_not_specialize=["PSTRIDE", "VSTRIDE", "OSTRIDE", "COLS", "QUERIES"])
def _local_visible_lengths(
    Prefix,
    Visible,
    Out,
    LocalLens,
    PSTRIDE,
    VSTRIDE,
    OSTRIDE,
    COLS,
    QUERIES,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    q = tl.arange(0, BLOCK)
    length = tl.load(Visible + row * VSTRIDE + q, q < QUERIES, 0)
    page = length // PAGE
    before = tl.load(Prefix + row * PSTRIDE + page, q < QUERIES, 0)
    # At the exact table endpoint only the sentinel prefix is needed.
    after = tl.load(
        Prefix + row * PSTRIDE + page + 1,
        (q < QUERIES) & (page < COLS),
        0,
    )
    local = before * PAGE + tl.where(after > before, length % PAGE, 0)
    tl.store(Out + row * OSTRIDE + q, local, q < QUERIES)
    if LocalLens is not None:
        tl.store(tl.broadcast_to(LocalLens + row, (BLOCK,)), local, q == QUERIES - 1)


def kvp_local_visible_lengths(
    page_prefix: torch.Tensor,
    visible_lengths: torch.Tensor,
    *,
    page_size: int,
    out: torch.Tensor,
    local_lengths: torch.Tensor | None = None,
) -> None:
    """Translate global query endpoints through an owned-page prefix table.

    Args:
        page_prefix: Exclusive owned-page counts [batch, max_pages + 1] from
            compact_kvp_pages. Includes allocated reserve pages.
        visible_lengths: Int32 global exclusive endpoints [batch, queries],
            each in [0, max_pages * page_size]. The last dimension is contiguous.
            Causal verify supplies one endpoint per query; noncausal draft may
            repeat the same endpoint. Reanchoring may decrease endpoints.
        page_size: Tokens per kernel page.
        out: Persistent int32 output with the same shape as visible_lengths
            and a contiguous last dimension. Receives local token counts.
        local_lengths: Optional contiguous [batch] output for the final query's
            local endpoint. When supplied, query endpoints must be nondecreasing.

    Returns:
        None; out is refreshed without scanning or rebuilding page tables.
    """
    if visible_lengths.is_cuda:
        _local_visible_lengths[(visible_lengths.shape[0],)](
            page_prefix,
            visible_lengths,
            out,
            local_lengths,
            page_prefix.stride(0),
            visible_lengths.stride(0),
            out.stride(0),
            page_prefix.shape[1] - 1,
            visible_lengths.shape[1],
            page_size,
            triton.next_power_of_2(visible_lengths.shape[1]),
        )
    else:
        page = visible_lengths.long() // page_size
        before = page_prefix.gather(1, page)
        after = page_prefix.gather(1, (page + 1).clamp_max(page_prefix.shape[1] - 1))
        out.copy_(before * page_size + (after > before) * (visible_lengths % page_size))
        if local_lengths is not None:
            local_lengths.copy_(out[:, -1])


# A mixed round's decode rows start past its extends: inputs at any alignment.
@triton.jit(
    do_not_specialize=["PSTRIDE", "COLS", "OFFSET"],
    do_not_specialize_on_alignment=["Prefix", "SeqLens"],
)
def _decode_visible_lengths(
    Prefix,
    SeqLens,
    LocalOut,
    CacheOut,
    PSTRIDE,
    COLS,
    OFFSET,
    QUERIES: tl.constexpr,
    GROUPS: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    q = tl.arange(0, BLOCK)
    live = q < QUERIES
    length = tl.maximum(tl.load(SeqLens + row).to(tl.int32) + OFFSET + q, 0)
    page = length // PAGE
    before = tl.load(Prefix + row * PSTRIDE + page, live, 0)
    after = tl.load(Prefix + row * PSTRIDE + page + 1, live & (page < COLS), 0)
    local = before * PAGE + tl.where(after > before, length % PAGE, 0)
    tl.store(LocalOut + row * QUERIES + q, local, live)
    cache = tl.maximum(local, 1)
    for group in tl.static_range(GROUPS):
        tl.store(CacheOut + (row * QUERIES + q) * GROUPS + group, cache, live)


def kvp_decode_visible_lengths(
    page_prefix: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    offset: int,
    queries: int,
    groups: int,
    page_size: int,
    local_lengths: torch.Tensor,
    cache_lengths: torch.Tensor,
) -> None:
    """:func:`kvp_local_visible_lengths` of a decode's query endpoints, and
    the kernel cache lengths derived from them, in one launch.

    Query ``q < queries`` of request ``r`` ends at ``max(seq_lens[r] + offset
    + q, 0)``; its local token count goes to ``local_lengths[r * queries + q]``
    and, at least one, to ``cache_lengths`` at ``(r * queries + q) * groups +
    g`` for every ``g < groups``: the endpoints, translation, clamp and
    repeat a decode over ``groups`` folded head groups would run as separate
    launches, value for value.

    Args:
        page_prefix: Exclusive owned-page counts [batch, max_pages + 1] from
            compact_kvp_pages, rows contiguous.
        seq_lens: Global sequence lengths [batch], contiguous.
        offset: First query's endpoint relative to ``seq_lens``.
        queries: Query rows per request.
        groups: Copies of each cache length.
        page_size: Tokens per kernel page.
        local_lengths: Contiguous int32 [batch * queries] output.
        cache_lengths: Contiguous int32 [batch * queries * groups] output.
    """
    batch = seq_lens.shape[0]
    if (
        seq_lens.dim() != 1
        or not seq_lens.is_contiguous()
        or page_prefix.dim() != 2
        or page_prefix.shape[0] != batch
        or page_prefix.stride(1) != 1
        or queries < 1
        or groups < 1
    ):
        raise ValueError("decode visible lengths need [batch] lengths and prefix rows")
    if (
        local_lengths.shape != (batch * queries,)
        or cache_lengths.shape != (batch * queries * groups,)
        or local_lengths.dtype != torch.int32
        or cache_lengths.dtype != torch.int32
        or not local_lengths.is_contiguous()
        or not cache_lengths.is_contiguous()
    ):
        raise ValueError("decode visible lengths need contiguous int32 outputs")
    if seq_lens.is_cuda:
        if batch:
            _decode_visible_lengths[(batch,)](
                page_prefix,
                seq_lens,
                local_lengths,
                cache_lengths,
                page_prefix.stride(0),
                page_prefix.shape[1] - 1,
                offset,
                queries,
                groups,
                page_size,
                triton.next_power_of_2(queries),
            )
        return
    endpoints = (
        seq_lens.to(torch.int32)[:, None]
        + torch.arange(offset, offset + queries, dtype=torch.int32)
    ).clamp_min(0)
    local = torch.empty_like(endpoints)
    kvp_local_visible_lengths(page_prefix, endpoints, page_size=page_size, out=local)
    local_lengths.copy_(local.reshape(-1))
    cache_lengths.copy_(local.reshape(-1).clamp_min(1).repeat_interleave(groups))
