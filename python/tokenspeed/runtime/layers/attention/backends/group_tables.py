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
rows past the live batch and columns past a group's table are 0, the
zero-initialized dummy page, safe to dereference and never a live request's
cache.

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
import triton
import triton.language as tl

from tokenspeed.runtime.layers.attention.backends.write_locations import (
    decode_write_locations,
    extend_write_locations,
)


@dataclass(frozen=True)
class GroupTableSpec:
    """Geometry of one routed group's kernel page table.

    Attributes:
        group_id: The cache group id (the key in every delivered dict).
        block_granularity: Tokens per scheduler block (raw table unit).
        kernel_page_size: Tokens per kernel page of the consuming leaf; must
            divide ``block_granularity``.
        max_num_pages: Kernel pages per row the leaf reads
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


@triton.jit(do_not_specialize=["num_blocks", "src_stride_b", "actual_bs"])
def _unpack_group_kernel(
    src_ptr,  # this group's raw table [>= actual_bs, num_blocks] int32
    dst_ptr,  # this group's stack rows [max_bs, stack_max_num_pages] int32
    num_blocks,  # scheduler blocks per raw row (the source column count)
    src_stride_b,
    ratio,  # kernel pages per scheduler block
    max_num_pages,  # kernel pages per row the leaf reads
    dst_stride_b,
    stack_max_num_pages,  # the stack's column count (widest group)
    actual_bs,
    BLOCK_COLS: tl.constexpr,
):
    """Expand one group's raw block table into its kernel-page stack rows.

    One launch per group with plain scalar arguments: the per-step values
    (block count, source stride, live rows) change every step, and hosting
    them in a device-side metadata tensor would cost a pinned staging
    allocation plus an H2D copy on the latency-critical bs=1 refresh path.
    ``do_not_specialize`` keeps those varying scalars from triggering
    recompiles.
    """
    b = tl.program_id(0)  # grid == padded bs
    live = b < actual_bs
    col_off = tl.arange(0, BLOCK_COLS)
    for col0 in range(0, stack_max_num_pages, BLOCK_COLS):
        col = col0 + col_off
        block = col // ratio
        # Columns past this group's own max_num_pages are the padding
        # contract's tail: always the null page 0, never expansion residue
        # (the slot kernels clamp at the stack's column count, so they can
        # read them).
        in_row = live & (block < num_blocks) & (col < max_num_pages)
        raw = tl.load(
            src_ptr + b.to(tl.int64) * src_stride_b + block, mask=in_row, other=0
        )
        # Holes and ragged padding (id <= 0) collapse onto the dummy page 0;
        # a real block expands to its ratio consecutive kernel pages.
        page = tl.maximum(raw, 0).to(tl.int64) * ratio + col % ratio
        page = tl.where(raw > 0, page, 0)
        vals = tl.where(in_row, page, 0)
        tl.store(
            dst_ptr + b * dst_stride_b + col,
            vals.to(tl.int32),
            mask=col < stack_max_num_pages,
        )


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
    """

    def __init__(
        self,
        groups: Sequence[GroupTableSpec],
        *,
        max_bs: int,
        max_tokens_per_req: int,
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
        self.page_sizes = torch.tensor(
            [spec.kernel_page_size for spec in groups], dtype=torch.int32, device=device
        )

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
            bs: Rows to prepare (the padded graph batch, or ``actual_bs``).
            actual_bs: Live rows; ``block_tables`` rows past it are ignored
                and stack rows ``[actual_bs, bs)`` are zeroed.
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
            # Idle / warmup: no live rows to read — every row is the null
            # page, whatever placeholder (or nothing) was delivered.
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
                    f"cache group {gid!r} table has {src.shape[0]} rows for a live "
                    f"batch of {actual_bs}"
                )
        if self.tables.is_cuda:
            stack_max_num_pages = self.tables.shape[2]
            block_cols = 128 if stack_max_num_pages >= 128 else 64
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

    def compute_decode_locations(
        self, bs: int, seq_lens: torch.Tensor, tokens_per_req: int
    ) -> None:
        """Refresh ``decode_locs[:, : bs * tokens_per_req]`` from the current
        stack and the live ``seq_lens`` (one launch for every group)."""
        decode_write_locations(
            self.tables, self.page_sizes, seq_lens, self.decode_locs, bs, tokens_per_req
        )

    def extend_locations(
        self,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        total_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Every group's extend write slots over the current stack
        (``group_id -> [total_tokens]`` fresh tensors; extend metadata is
        rebuilt per round)."""
        locs = extend_write_locations(
            self.tables,
            self.page_sizes,
            extend_prefix_lens,
            extend_seq_lens,
            total_tokens,
        )
        return {gid: locs[i] for i, gid in enumerate(self.group_ids)}
