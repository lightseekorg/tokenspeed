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

"""The router's stacked per-group tables: the one block -> page mapping.

``GroupTableStacks`` turns the bridge's raw per-group block tables (scheduler
pages of each group's ``block_granularity``, batch-ordered, ``-1`` ragged
padding, ``0`` null holes) into one ``[G, max_bs, stack_max_num_pages]``
stack of kernel page tables — group ``g`` expanded to its leaf's
``kernel_page_size`` and padded to the leaf's ``max_num_pages``;
``stack_max_num_pages`` is the widest group's — plus the ``[G, max_bs * N]``
stack of decode write slots derived from those tables. Both fills are one
launch for every group (the bridge uploads all groups into one packed
buffer, so the unpack reads it directly); the padding contract is uniform:
request rows past the live batch and page columns past a group's
``max_num_pages`` are 0, the zero-initialized dummy page, safe to
dereference and never a live request's cache.

The stacks are allocated once (``init_cuda_graph_state``) and refilled in
place. Leaves copy their ``[bs, max_num_pages]`` view into their own
graph-recorded buffers, but three consumers read the stack storage directly
inside captured graphs and so pin its address: the decode write-slot views
the router publishes, the full-history table the block drafters get from
``draft_history_view``, and the per-group tables the QSA indexer takes from
``table()``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from tokenspeed.runtime.layers.attention.backends.paged.write_locations import (
    decode_write_locations,
    extend_write_locations,
    window_write_locations,
)
from tokenspeed.runtime.utils.triton import tl, triton


@dataclass(frozen=True)
class GroupTableSpec:
    """Geometry of one routed group's kernel page table.

    Attributes:
        group_id: The cache group id (the key in every delivered dict).
        block_granularity: Tokens per scheduler block (raw table unit).
        kernel_page_size: Tokens per kernel page of the consuming leaf; must
            divide ``block_granularity``.
        max_num_pages: Kernel pages per request the leaf reads
            (``leaf.max_num_pages``).
    """

    group_id: str
    block_granularity: int
    kernel_page_size: int
    max_num_pages: int

    @property
    def ratio(self) -> int:
        if self.kernel_page_size <= 0 or self.block_granularity % self.kernel_page_size:
            raise ValueError(
                f"cache group {self.group_id!r}: block_granularity "
                f"{self.block_granularity} is not a positive multiple of the "
                f"kernel page size {self.kernel_page_size}"
            )
        return self.block_granularity // self.kernel_page_size


# Groups one packed fill serves; a router with more launches once per group.
_PACKED_MAX_GROUPS = 8


@triton.jit
def _unpack_row(
    src_ptr,  # this group's raw table [>= actual_bs, num_blocks] int32
    dst_ptr,  # this group's stack [max_bs, stack_max_num_pages] int32
    b,  # the request row
    live,  # whether the row is a live request
    num_blocks,  # scheduler blocks per request (the raw table's column count)
    src_stride_b,
    ratio,  # kernel pages per scheduler block
    max_num_pages,  # kernel pages per request the leaf reads
    dst_stride_b,
    stack_max_num_pages,  # the stack's column count (widest group)
    BLOCK_COLS: tl.constexpr,
):
    """Expand one request row of a group's raw block table into its stack."""
    col_off = tl.arange(0, BLOCK_COLS)
    for col0 in range(0, stack_max_num_pages, BLOCK_COLS):
        col = col0 + col_off
        block = col // ratio
        # Columns past this group's own max_num_pages are the padding
        # contract's tail: always the null page 0, never expansion residue
        # (the slot kernels clamp at the stack's column count, so they can
        # read them).
        in_table = live & (block < num_blocks) & (col < max_num_pages)
        raw = tl.load(
            src_ptr + b.to(tl.int64) * src_stride_b + block, mask=in_table, other=0
        )
        # Holes and ragged padding (id <= 0) collapse onto the dummy page 0;
        # a real block expands to its ratio consecutive kernel pages.
        page = tl.maximum(raw, 0).to(tl.int64) * ratio + col % ratio
        page = tl.where(raw > 0, page, 0)
        vals = tl.where(in_table, page, 0)
        tl.store(
            dst_ptr + b.to(tl.int64) * dst_stride_b + col,
            vals.to(tl.int32),
            mask=col < stack_max_num_pages,
        )


@triton.jit(
    do_not_specialize=["num_blocks", "src_stride_b", "actual_bs"],
    do_not_specialize_on_alignment=["src_ptr"],
)
def _unpack_group_kernel(
    src_ptr,
    dst_ptr,
    num_blocks,
    src_stride_b,
    ratio,
    max_num_pages,
    dst_stride_b,
    stack_max_num_pages,
    actual_bs,
    BLOCK_COLS: tl.constexpr,
):
    """Expand one group's raw block table into its kernel-page stack, one
    request per program (arguments as :func:`_unpack_row`): the fill of
    tables that do not share the bridge's packed upload."""
    b = tl.program_id(0)  # grid == padded bs
    _unpack_row(
        src_ptr,
        dst_ptr,
        b,
        b < actual_bs,
        num_blocks,
        src_stride_b,
        ratio,
        max_num_pages,
        dst_stride_b,
        stack_max_num_pages,
        BLOCK_COLS,
    )


@triton.jit(
    do_not_specialize=[
        "actual_bs",
        *(f"offset{i}" for i in range(_PACKED_MAX_GROUPS)),
        *(f"blocks{i}" for i in range(_PACKED_MAX_GROUPS)),
    ],
    do_not_specialize_on_alignment=["packed_ptr"],
)
def _unpack_packed_groups_kernel(
    packed_ptr,  # the bridge's one packed upload of every group's raw table
    dst_ptr,  # the stack [G, max_bs, stack_max_num_pages] int32
    offset0,
    offset1,
    offset2,
    offset3,
    offset4,
    offset5,
    offset6,
    offset7,
    blocks0,
    blocks1,
    blocks2,
    blocks3,
    blocks4,
    blocks5,
    blocks6,
    blocks7,
    dst_stride_g,
    dst_stride_b,
    stack_max_num_pages,
    actual_bs,
    RATIOS: tl.constexpr,
    MAX_NUM_PAGES: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Every group's fill in one launch, one (request, group) per program.

    Group ``i`` reads the row-major ``[>= actual_bs, blocks_i]`` table at
    element ``offset_i`` of the packed upload. Offsets and block counts change
    every step, so they are plain scalars kept out of the compile key (Triton
    would specialize them inside a tuple); groups past ``NUM_GROUPS`` are 0.
    """
    b = tl.program_id(0)  # grid == (padded bs, groups)
    g = tl.program_id(1)
    offsets = (offset0, offset1, offset2, offset3, offset4, offset5, offset6, offset7)
    blocks = (blocks0, blocks1, blocks2, blocks3, blocks4, blocks5, blocks6, blocks7)
    for i in tl.static_range(NUM_GROUPS):
        if g == i:
            _unpack_row(
                packed_ptr + offsets[i],
                dst_ptr + dst_stride_g.to(tl.int64) * i,
                b,
                b < actual_bs,
                blocks[i],
                blocks[i],
                RATIOS[i],
                MAX_NUM_PAGES[i],
                dst_stride_b,
                stack_max_num_pages,
                BLOCK_COLS,
            )


def _packed_layout(
    srcs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, list[int]] | None:
    """The shared storage and per-group element offsets of tables that are
    row-major views into one upload (the bridge packs every group into one),
    or None when any is not and each group needs its own launch."""
    if len(srcs) > _PACKED_MAX_GROUPS:
        return None
    storage = srcs[0].untyped_storage().data_ptr()
    for src in srcs:
        if (
            src.untyped_storage().data_ptr() != storage
            or src.shape[1] == 0
            or src.stride(1) != 1
            or src.stride(0) != src.shape[1]
        ):
            return None
    start = min(src.storage_offset() for src in srcs)
    end = max(src.storage_offset() + src.numel() for src in srcs)
    first = min(srcs, key=lambda src: src.storage_offset())
    base = first.as_strided((end - start,), (1,))
    return base, [src.storage_offset() - start for src in srcs]


@triton.jit(
    do_not_specialize=[
        "nb0",
        "nb1",
        "nb2",
        "nb3",
        "st0",
        "st1",
        "st2",
        "st3",
        "actual_bs",
        "tokens_per_req",
    ],
    do_not_specialize_on_alignment=["t0", "t1", "t2", "t3"],
)
def _fill_decode_kernel(
    t0,
    t1,
    t2,
    t3,  # each group's raw table [>= actual_bs, num_blocks] int32
    nb0,
    nb1,
    nb2,
    nb3,  # each group's scheduler blocks per request
    st0,
    st1,
    st2,
    st3,  # each group's raw-table row stride
    dst_ptr,  # the stack [G, max_bs, stack_max_num_pages] int32
    dst_stride_g,
    dst_stride_b,
    stack_max_num_pages,
    actual_bs,
    seq_lens_ptr,  # [>= bs] int32
    locs_ptr,  # decode write slots [G, cap] int32
    locs_stride_g,
    tokens_per_req,
    sl0,
    sl1,
    sl2,
    sl3,  # each leaf target's seq_lens [>= bs] int32
    pt0,
    pt1,
    pt2,
    pt3,  # each leaf target's page table [>= bs, P] int32
    pts0,
    pts1,
    pts2,
    pts3,  # each leaf target page table's row stride
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    R3: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    PS2: tl.constexpr,
    PS3: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    F2: tl.constexpr,
    F3: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """A decode refresh of up to four groups in one launch: program
    ``(b, g)`` expands request ``b`` of group ``g`` into the stack (as
    ``_unpack_group_kernel``), computes its decode write slots (as
    ``_decode_locs_kernel``, from the raw table so no program reads another's
    stores) and, when group ``g`` has a leaf target (``F > 0``), copies the
    row (when ``T``) and ``max(seq_lens[b], F)`` into the leaf's buffers.
    ``R``, ``P``, ``PS``, ``F`` and ``T`` are the groups' static geometry,
    seq_lens floors and table flags."""
    b = tl.program_id(0)
    g = tl.program_id(1)
    src_ptr = tl.where(g == 0, t0, tl.where(g == 1, t1, tl.where(g == 2, t2, t3)))
    num_blocks = tl.where(
        g == 0, nb0, tl.where(g == 1, nb1, tl.where(g == 2, nb2, nb3))
    )
    src_stride_b = tl.where(
        g == 0, st0, tl.where(g == 1, st1, tl.where(g == 2, st2, st3))
    )
    ratio = tl.where(g == 0, R0, tl.where(g == 1, R1, tl.where(g == 2, R2, R3)))
    max_num_pages = tl.where(g == 0, P0, tl.where(g == 1, P1, tl.where(g == 2, P2, P3)))
    page_size = tl.where(g == 0, PS0, tl.where(g == 1, PS1, tl.where(g == 2, PS2, PS3)))
    floor = tl.where(g == 0, F0, tl.where(g == 1, F1, tl.where(g == 2, F2, F3)))
    seq_dst = tl.where(g == 0, sl0, tl.where(g == 1, sl1, tl.where(g == 2, sl2, sl3)))
    table_dst = tl.where(g == 0, pt0, tl.where(g == 1, pt1, tl.where(g == 2, pt2, pt3)))
    table_dst_stride = tl.where(
        g == 0, pts0, tl.where(g == 1, pts1, tl.where(g == 2, pts2, pts3))
    )
    has_target = floor > 0
    has_table = (
        tl.where(g == 0, T0, tl.where(g == 1, T1, tl.where(g == 2, T2, T3))) != 0
    )
    live = b < actual_bs
    src_row = src_ptr + b.to(tl.int64) * src_stride_b
    col_off = tl.arange(0, BLOCK_COLS)
    for col0 in range(0, stack_max_num_pages, BLOCK_COLS):
        col = col0 + col_off
        block = col // ratio
        in_table = live & (block < num_blocks) & (col < max_num_pages)
        raw = tl.load(src_row + block, mask=in_table, other=0)
        page = tl.maximum(raw, 0).to(tl.int64) * ratio + col % ratio
        page = tl.where(raw > 0, page, 0)
        vals = tl.where(in_table, page, 0).to(tl.int32)
        tl.store(
            dst_ptr
            + g.to(tl.int64) * dst_stride_g
            + b.to(tl.int64) * dst_stride_b
            + col,
            vals,
            mask=col < stack_max_num_pages,
        )
        tl.store(
            table_dst + b.to(tl.int64) * table_dst_stride + col,
            vals,
            mask=has_table & (col < max_num_pages),
        )
    seq_len = tl.load(seq_lens_ptr + b)
    tl.store(seq_dst + b, tl.maximum(seq_len, floor), mask=has_target)
    # The decode window: positions seq_len - n .. seq_len - 1.
    step = tl.arange(0, BLOCK_TOKENS)
    in_window = step < tokens_per_req
    pos = tl.maximum(seq_len.to(tl.int64) - tokens_per_req + step, 0)
    page_idx = pos // page_size
    overflow = page_idx >= max_num_pages
    block = page_idx // ratio
    raw = tl.load(
        src_row + block,
        mask=in_window & live & (block < num_blocks) & ~overflow,
        other=0,
    )
    page = tl.where(raw > 0, raw.to(tl.int64) * ratio + page_idx % ratio, 0)
    loc = tl.where(overflow | (page <= 0), 0, page * page_size + pos % page_size)
    tl.store(
        locs_ptr + g * locs_stride_g + b * tokens_per_req + step,
        loc.to(tl.int32),
        mask=in_window,
    )


@dataclass(frozen=True)
class DecodeFillTarget:
    """A leaf's decode buffers that :meth:`GroupTableStacks.fill_decode`
    writes: ``max(seq_lens, seq_lens_floor)``, and the group's table rows
    when ``page_table`` is set."""

    seq_lens: torch.Tensor
    page_table: torch.Tensor | None
    seq_lens_floor: int


class GroupTableStacks:
    """Stacked kernel page tables and decode write slots for a router's groups.

    Attributes:
        tables: ``[G, max_bs, stack_max_num_pages]`` int32 kernel page
            tables, group-major in ``group_ids`` order.
        decode_locs: ``[G, max_bs * max_tokens_per_req]`` int32 decode write
            slots (token-major per request) refreshed by
            :meth:`compute_decode_locations`; the router publishes per-bs
            views of it, so it is address-stable for the graph's lifetime.
        page_sizes: ``[G]`` int32 device tensor of kernel page sizes.
        page_sizes_cpu: Immutable host tuple from the same specs; avoids
            device scalar reads in the Torch write-location fallback.
    """

    def __init__(
        self,
        groups: Sequence[GroupTableSpec],
        *,
        max_bs: int,
        max_tokens_per_req: int,
        max_extend_tokens: int,
        device,
    ) -> None:
        if not groups:
            raise ValueError("GroupTableStacks needs at least one group")
        self.group_ids: tuple[str, ...] = tuple(spec.group_id for spec in groups)
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError(f"duplicate cache group ids: {self.group_ids}")
        self._specs = {spec.group_id: spec for spec in groups}
        self._index = {gid: i for i, gid in enumerate(self.group_ids)}
        self._ratios = tuple(spec.ratio for spec in groups)
        self._max_num_pages = tuple(spec.max_num_pages for spec in groups)
        self.page_sizes_cpu = tuple(int(spec.kernel_page_size) for spec in groups)
        if any(size <= 0 for size in self.page_sizes_cpu):
            raise ValueError("kernel page sizes must be positive host integers")
        self.max_bs = int(max_bs)
        self.max_tokens_per_req = max(int(max_tokens_per_req), 1)
        g = len(groups)
        stack_max_num_pages = max(self._max_num_pages)
        self.tables = torch.zeros(
            (g, self.max_bs, stack_max_num_pages), dtype=torch.int32, device=device
        )
        self.decode_locs = torch.zeros(
            (g, self.max_bs * self.max_tokens_per_req), dtype=torch.int32, device=device
        )
        # Extend spans up to this many tokens live here, tail at the dummy slot 0,
        # so a prefill graph's padded prologue can record and replay the address.
        self.extend_locs = torch.zeros(
            (g, max(int(max_extend_tokens), 0)), dtype=torch.int32, device=device
        )
        self._extend_total = 0
        self._extend_span: torch.Tensor | None = None
        # A MIXED round's forward carries the decode rows after the extend span.
        self._forward_total = 0
        self._decode_tail: torch.Tensor | None = None
        self.page_sizes = torch.tensor(
            self.page_sizes_cpu, dtype=torch.int32, device=device
        )
        if self.tables.is_cuda and g <= _PACKED_MAX_GROUPS:
            # Compile this geometry's packed fill now, not while serving.
            zeros = torch.zeros((g, 1), dtype=torch.int32, device=device)
            self.fill(1, 1, dict(zip(self.group_ids, zeros.split(1))))

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def index(self, group_id: str) -> int:
        return self._index[group_id]

    def group_capacity_tokens(self, group_id: str) -> int:
        """Per-request token capacity of one group's own columns."""
        i = self._index[group_id]
        return self._max_num_pages[i] * self._specs[group_id].kernel_page_size

    def group_kernel_page_size(self, group_id: str) -> int:
        """One group's kernel page size from the host-side spec (the device
        ``page_sizes`` mirror feeds the kernels; reading it back would sync)."""
        return int(self._specs[group_id].kernel_page_size)

    def table(self, group_id: str, bs: int) -> torch.Tensor:
        """``[bs, max_num_pages]`` kernel page table view of one group."""
        i = self._index[group_id]
        return self.tables[i, :bs, : self._max_num_pages[i]]

    def host_table(
        self, group_id: str, block_table_cpu: torch.Tensor, num_requests: int
    ) -> torch.Tensor:
        """Expand one group's host block table into kernel pages: the
        ``[num_requests, blocks * ratio]`` host mirror of :meth:`table`'s
        leading rows (unpadded), the same block-to-page math as the device
        fill, for leaves that count page ownership on the host."""
        i = self._index[group_id]
        ratio = self._ratios[i]
        src = block_table_cpu[:num_requests].to(torch.int64).clamp_min(0)
        if ratio != 1:
            offsets = torch.arange(ratio, dtype=torch.int64)
            src = (src.unsqueeze(-1) * ratio + offsets).reshape(num_requests, -1)
            # Page 0 stays 0 across its ratio slots.
            src = torch.where(src >= ratio, src, torch.zeros_like(src))
        return src.to(torch.int32)

    def decode_locations(
        self, group_id: str, bs: int, tokens_per_req: int
    ) -> torch.Tensor:
        """``[bs * tokens_per_req]`` decode write-slot view of one group."""
        n = max(int(tokens_per_req), 1)
        if n > self.max_tokens_per_req:
            raise RuntimeError(
                f"decode write slots sized for {self.max_tokens_per_req} tokens per "
                f"request, asked for {n}"
            )
        return self.decode_locs[self._index[group_id], : bs * n]

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def fill(
        self, bs: int, actual_bs: int, block_tables: Mapping[str, torch.Tensor]
    ) -> None:
        """Copy this step's raw group tables into the stack, expanded to
        kernel pages and padded.

        Args:
            bs: Requests to prepare (the padded graph batch, or ``actual_bs``).
            actual_bs: Live requests; ``block_tables`` requests past it are
                ignored and the stack's requests ``[actual_bs, bs)`` are
                zeroed.
            block_tables: ``group_id -> [>= actual_bs, num_blocks]`` int32
                raw scheduler tables covering every routed group (the
                router's ``_check_live_delivery`` guards a live batch's dict).
        """
        if bs < actual_bs or actual_bs < 0:
            raise RuntimeError(f"need 0 <= actual_bs <= bs, got {actual_bs=} {bs=}")
        if bs > self.max_bs:
            raise RuntimeError(f"bs={bs} exceeds the stack capacity {self.max_bs}")
        if bs == 0:
            return
        if actual_bs == 0:
            # Idle / warmup: no live requests to read — every request's row
            # is the null page, whatever placeholder (or nothing) was
            # delivered.
            self.tables[:, :bs].zero_()
            return
        srcs = [block_tables[gid] for gid in self.group_ids]
        for gid, src in zip(self.group_ids, srcs):
            if src.ndim != 2 or src.dtype != torch.int32:
                raise RuntimeError(
                    f"cache group {gid!r} table must be a 2-D int32 tensor, got "
                    f"{tuple(src.shape)} {src.dtype}"
                )
            if src.shape[0] < actual_bs:
                raise RuntimeError(
                    f"cache group {gid!r} table has {src.shape[0]} request rows for "
                    f"a live batch of {actual_bs}"
                )
        if self.tables.is_cuda:
            stack_max_num_pages = self.tables.shape[2]
            block_cols = 128 if stack_max_num_pages >= 128 else 64
            packed = _packed_layout(srcs)
            if packed is not None:
                base, offsets = packed
                pad = [0] * (_PACKED_MAX_GROUPS - len(srcs))
                _unpack_packed_groups_kernel[(bs, len(srcs))](
                    base,
                    self.tables,
                    *offsets,
                    *pad,
                    *(src.shape[1] for src in srcs),
                    *pad,
                    self.tables.stride(0),
                    self.tables.stride(1),
                    stack_max_num_pages,
                    actual_bs,
                    RATIOS=self._ratios,
                    MAX_NUM_PAGES=self._max_num_pages,
                    NUM_GROUPS=len(srcs),
                    BLOCK_COLS=block_cols,
                )
                return
            for i, src in enumerate(srcs):
                if src.stride(1) != 1:
                    src = src.contiguous()
                _unpack_group_kernel[(bs,)](
                    src,
                    self.tables[i],
                    src.shape[1],
                    src.stride(0),
                    self._ratios[i],
                    self._max_num_pages[i],
                    self.tables.stride(1),
                    stack_max_num_pages,
                    actual_bs,
                    BLOCK_COLS=block_cols,
                )
            return
        self._fill_torch(bs, actual_bs, srcs)

    def _fill_torch(
        self, bs: int, actual_bs: int, srcs: Sequence[torch.Tensor]
    ) -> None:
        """CPU reference of the unpack kernel (unit tests); ``actual_bs > 0``
        (``fill`` handles the idle case before dispatching here)."""
        for i, src in enumerate(srcs):
            dst = self.tables[i, :bs]
            max_num_pages = self._max_num_pages[i]
            ratio = self._ratios[i]
            # Only the scheduler blocks whose expansion lands inside the
            # leaf's max_num_pages are read.
            num_blocks = min(src.shape[1], -(-max_num_pages // ratio))
            live = src[:actual_bs, :num_blocks].clamp_min(0)
            if ratio != 1:
                offsets = torch.arange(ratio, dtype=live.dtype, device=live.device)
                live = (live.unsqueeze(-1) * ratio + offsets).reshape(actual_bs, -1)
                live = torch.where(
                    live >= ratio, live, torch.zeros_like(live)
                )  # page 0 stays 0 across its ratio slots
            num_pages = min(live.shape[1], max_num_pages)
            dst[:actual_bs, :num_pages].copy_(live[:, :num_pages])
            dst[:actual_bs, num_pages:].zero_()
            dst[actual_bs:bs].zero_()

    def fill_decode(
        self,
        bs: int,
        actual_bs: int,
        block_tables: Mapping[str, torch.Tensor],
        seq_lens: torch.Tensor,
        tokens_per_req: int,
        targets: Mapping[str, DecodeFillTarget | None],
    ) -> bool:
        """:meth:`fill`, then :meth:`compute_decode_locations`, then each
        target's copies, in one launch. Returns False, having done nothing,
        when the launch cannot serve the step (a host stack, more than four
        groups, an idle step, a strided table or target); the caller then
        runs the three steps.

        Args:
            bs, actual_bs, block_tables: As :meth:`fill`.
            seq_lens: ``[>= bs]`` int32 lengths, as
                :meth:`compute_decode_locations`.
            tokens_per_req: The decode window width.
            targets: ``group_id -> target`` for the groups whose leaf takes
                the copies (None for the others).
        """
        n = max(int(tokens_per_req), 1)
        groups = len(self.group_ids)
        if (
            not self.tables.is_cuda
            or groups > 4
            or actual_bs <= 0
            or not 0 < bs <= self.max_bs
            or actual_bs > bs
            or n > self.max_tokens_per_req
            or seq_lens.dtype != torch.int32
        ):
            return False
        srcs = [block_tables[gid] for gid in self.group_ids]
        if any(
            src.ndim != 2
            or src.dtype != torch.int32
            or src.shape[0] < actual_bs
            or src.stride(1) != 1
            for src in srcs
        ):
            return False
        leaf_targets = [targets.get(gid) for gid in self.group_ids]
        for i, target in enumerate(leaf_targets):
            if target is None:
                continue
            if (
                target.seq_lens.dtype != torch.int32
                or target.seq_lens.shape[0] < bs
                or target.seq_lens_floor < 1
            ):
                return False
            table = target.page_table
            if table is not None and (
                table.dtype != torch.int32
                or table.stride(1) != 1
                or table.shape[1] != self._max_num_pages[i]
                or table.shape[0] < bs
            ):
                return False
        pad = 4 - groups
        # Groups without a target point at the (static) stack, never written:
        # a per-step tensor there would key new binaries on its alignment.
        seq_dst = [seq_lens if t is None else t.seq_lens for t in leaf_targets]
        tables = [None if t is None else t.page_table for t in leaf_targets]
        table_dst = [self.tables if t is None else t for t in tables]
        table_dst_strides = [0 if t is None else t.stride(0) for t in tables]
        stack_max_num_pages = self.tables.shape[2]
        _fill_decode_kernel[(bs, groups)](
            *srcs,
            *([srcs[0]] * pad),
            *(src.shape[1] for src in srcs),
            *([0] * pad),
            *(src.stride(0) for src in srcs),
            *([0] * pad),
            self.tables,
            self.tables.stride(0),
            self.tables.stride(1),
            stack_max_num_pages,
            actual_bs,
            seq_lens,
            self.decode_locs,
            self.decode_locs.stride(0),
            n,
            *seq_dst,
            *([seq_lens] * pad),
            *table_dst,
            *([self.tables] * pad),
            *table_dst_strides,
            *([0] * pad),
            *self._ratios,
            *([1] * pad),
            *self._max_num_pages,
            *([0] * pad),
            *(self._specs[gid].kernel_page_size for gid in self.group_ids),
            *([1] * pad),
            *(0 if t is None else t.seq_lens_floor for t in leaf_targets),
            *([0] * pad),
            *(int(t is not None) for t in tables),
            *([0] * pad),
            BLOCK_COLS=128 if stack_max_num_pages >= 128 else 64,
            BLOCK_TOKENS=triton.next_power_of_2(self.max_tokens_per_req),
        )
        return True

    def compute_decode_locations(
        self, bs: int, seq_lens: torch.Tensor, tokens_per_req: int
    ) -> None:
        """Refresh ``decode_locs[:, : bs * tokens_per_req]`` from the current
        stack and the live ``seq_lens`` (one launch for every group)."""
        decode_write_locations(
            self.tables,
            self.page_sizes,
            seq_lens,
            self.decode_locs,
            bs,
            tokens_per_req,
            page_sizes_cpu=self.page_sizes_cpu,
        )

    def compute_window_locations(
        self, bs: int, window_starts: torch.Tensor, tokens_per_req: int
    ) -> None:
        """:meth:`compute_decode_locations` of the windows starting at ``window_starts``."""
        window_write_locations(
            self.tables,
            self.page_sizes,
            window_starts,
            self.decode_locs,
            bs,
            tokens_per_req,
            page_sizes_cpu=self.page_sizes_cpu,
        )

    def extend_locations(
        self,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        total_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Every group's extend write slots over the current stack
        (``group_id -> [total_tokens]``): views of the persistent buffer when the
        span fits it, with the tail zeroed, else fresh tensors."""
        self._extend_total = self._forward_total = total_tokens
        self._decode_tail = None
        if total_tokens <= self.extend_locs.shape[1]:
            out = self.extend_locs[:, :total_tokens]
            self.extend_locs[:, total_tokens:].zero_()
        else:
            out = torch.empty(
                (len(self.group_ids), total_tokens),
                dtype=torch.int32,
                device=self.tables.device,
            )
        locs = extend_write_locations(
            self.tables,
            self.page_sizes,
            extend_prefix_lens,
            extend_seq_lens,
            total_tokens,
            out,
            page_sizes_cpu=self.page_sizes_cpu,
        )
        self._extend_span = locs
        return {gid: locs[i] for i, gid in enumerate(self.group_ids)}

    def append_decode_rows(self, first_token: int, count: int) -> None:
        """Place a MIXED round's decode rows, ``decode_locs[:, first:first + count]``
        after its extend span, retaining the tail for padding past the buffer."""
        total = self._extend_total
        self._forward_total = total + count
        tail = self.decode_locs[:, first_token : first_token + count]
        self._decode_tail = tail
        if self._forward_total <= self.extend_locs.shape[1]:
            self.extend_locs[:, total : self._forward_total].copy_(tail)
            self.extend_locs[:, self._forward_total :].zero_()

    def padded_extend_span(self, group_id: str, rows: int) -> torch.Tensor:
        """The forward's rows (the extend span, then a MIXED round's decode rows)
        widened to ``rows`` with the dummy slot 0: a view of the buffer's zero
        tail where the graph-padded forward records it, else a fresh tensor for
        the eager forward wider than the buffer."""
        total = self._forward_total
        if rows < total:
            raise ValueError(f"a span of {total} tokens cannot pad to {rows} rows")
        index = self._index[group_id]
        if rows <= self.extend_locs.shape[1]:
            return self.extend_locs[index, :rows]
        padded = torch.zeros(rows, dtype=torch.int32, device=self.tables.device)
        padded[: self._extend_total] = self._extend_span[index]
        if self._decode_tail is not None:
            padded[self._extend_total : total] = self._decode_tail[index]
        return padded
