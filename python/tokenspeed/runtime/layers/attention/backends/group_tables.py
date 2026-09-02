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
padding, ``0`` null holes) into one ``[G, max_bs, Wmax]`` stack of kernel page
tables — group ``g`` expanded to its leaf's ``kernel_page_size`` and padded
to the leaf's ``max_num_pages`` — plus the ``[G, max_bs * N]`` stack of
decode write slots derived from those tables. Both fills are one launch for
every group (the bridge uploads all groups into one packed buffer, so the
unpack reads it directly); the padding contract is uniform: rows past the
live batch and columns past a group's table are 0, the zero-initialized
dummy page, safe to dereference and never a live request's cache.

The stack is scratch for the router: leaves copy their ``[bs, W_g]`` view
into their own graph-recorded buffers, so nothing here is pointer-frozen by
a captured graph except the decode write-slot views the router publishes.
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
        width: Kernel pages per row the leaf reads (``leaf.max_num_pages``).
    """

    group_id: str
    block_granularity: int
    kernel_page_size: int
    width: int

    @property
    def ratio(self) -> int:
        if self.kernel_page_size <= 0 or self.block_granularity % self.kernel_page_size:
            raise ValueError(
                f"cache group {self.group_id!r}: block_granularity "
                f"{self.block_granularity} is not a positive multiple of the "
                f"kernel page size {self.kernel_page_size}"
            )
        return self.block_granularity // self.kernel_page_size


@triton.jit
def _unpack_kernel(
    src_ptr,  # packed int32 upload holding every group's rows back-to-back
    meta_ptr,  # [G, 4] int32: (element offset, source columns, page ratio, width)
    dst_ptr,  # [G, max_bs, Wmax] int32
    stride_g,
    stride_b,
    wmax,
    actual_bs,
    BLOCK_W: tl.constexpr,
):
    g = tl.program_id(0)
    b = tl.program_id(1)  # grid axis 1 == padded bs
    off = tl.load(meta_ptr + g * 4).to(tl.int64)
    cols = tl.load(meta_ptr + g * 4 + 1).to(tl.int64)
    ratio = tl.load(meta_ptr + g * 4 + 2).to(tl.int64)
    width = tl.load(meta_ptr + g * 4 + 3).to(tl.int64)
    live = b < actual_bs
    w_off = tl.arange(0, BLOCK_W)
    for w0 in range(0, wmax, BLOCK_W):
        w = w0 + w_off
        src_col = w // ratio
        # Columns past this group's own width are the padding contract's
        # tail: always the null page 0, never expansion residue (the slot
        # kernels clamp at the stack's wmax, so they can read them).
        in_row = live & (src_col < cols) & (w < width)
        raw = tl.load(src_ptr + off + b * cols + src_col, mask=in_row, other=0)
        # Holes and ragged padding (id <= 0) collapse onto the dummy page 0;
        # a real page expands to its ratio consecutive kernel pages.
        page = tl.maximum(raw, 0).to(tl.int64) * ratio + w % ratio
        page = tl.where(raw > 0, page, 0)
        vals = tl.where(in_row, page, 0)
        tl.store(
            dst_ptr + g * stride_g + b * stride_b + w, vals.to(tl.int32), mask=w < wmax
        )


class GroupTableStacks:
    """Stacked kernel page tables and decode write slots for a router's groups.

    Attributes:
        tables: ``[G, max_bs, Wmax]`` int32 kernel page tables, group-major in
            ``group_ids`` order.
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
        self._widths = tuple(spec.width for spec in groups)
        self.max_bs = int(max_bs)
        self.max_tokens_per_req = max(int(max_tokens_per_req), 1)
        g = len(groups)
        wmax = max(self._widths)
        self.tables = torch.zeros(
            (g, self.max_bs, wmax), dtype=torch.int32, device=device
        )
        self.decode_locs = torch.zeros(
            (g, self.max_bs * self.max_tokens_per_req), dtype=torch.int32, device=device
        )
        self.page_sizes = torch.tensor(
            [spec.kernel_page_size for spec in groups], dtype=torch.int32, device=device
        )
        self._unpack_meta_device = torch.zeros((g, 4), dtype=torch.int32, device=device)

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def index(self, group_id: str) -> int:
        return self._index[group_id]

    def spec(self, group_id: str) -> GroupTableSpec:
        return self._specs[group_id]

    def group_capacity_tokens(self, group_id: str) -> int:
        """Per-request token capacity of one group's own columns."""
        i = self._index[group_id]
        return self._widths[i] * self._specs[group_id].kernel_page_size

    def table(self, group_id: str, bs: int) -> torch.Tensor:
        """``[bs, W_g]`` kernel page table view of one group."""
        i = self._index[group_id]
        return self.tables[i, :bs, : self._widths[i]]

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
            block_tables: ``group_id -> [>= actual_bs, cols]`` int32 raw
                scheduler tables; every routed group must be present.
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
        missing = [gid for gid in self.group_ids if gid not in block_tables]
        if missing:
            raise RuntimeError(
                f"block_tables is missing routed cache groups {missing} "
                f"(delivered: {sorted(block_tables)})"
            )
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
        if not self._fill_packed(bs, actual_bs, srcs):
            self._fill_per_group(bs, actual_bs, srcs)

    def _fill_packed(
        self, bs: int, actual_bs: int, srcs: Sequence[torch.Tensor]
    ) -> bool:
        """One launch over the bridge's packed upload. Requires every source
        to be a view of one storage with the same row count (the bridge
        contract); anything else takes the per-group path."""
        if not self.tables.is_cuda or actual_bs == 0:
            return False
        base = srcs[0].untyped_storage().data_ptr()
        rows = srcs[0].shape[0]
        for src in srcs:
            if (
                src.untyped_storage().data_ptr() != base
                or src.shape[0] != rows
                or src.shape[1] == 0
                or src.stride(1) != 1
                or src.stride(0) != src.shape[1]
            ):
                return False
        # Fresh pinned host staging each step: a persistent one would race
        # the next step's fill under overlap scheduling.
        meta = torch.empty((len(srcs), 4), dtype=torch.int32, pin_memory=True)
        for i, src in enumerate(srcs):
            meta[i, 0] = src.storage_offset()
            meta[i, 1] = src.shape[1]
            meta[i, 2] = self._ratios[i]
            meta[i, 3] = self._widths[i]
        self._unpack_meta_device.copy_(meta, non_blocking=True)
        src0 = srcs[0]
        packed = torch.as_strided(
            src0, (src0.untyped_storage().nbytes() // 4,), (1,), storage_offset=0
        )
        wmax = self.tables.shape[2]
        _unpack_kernel[(len(srcs), bs)](
            packed,
            self._unpack_meta_device,
            self.tables,
            self.tables.stride(0),
            self.tables.stride(1),
            wmax,
            actual_bs,
            BLOCK_W=128 if wmax >= 128 else 64,
        )
        return True

    def _fill_per_group(
        self, bs: int, actual_bs: int, srcs: Sequence[torch.Tensor]
    ) -> None:
        for i, src in enumerate(srcs):
            dst = self.tables[i, :bs]
            width = self._widths[i]
            ratio = self._ratios[i]
            if actual_bs > 0:
                cols = min(src.shape[1], -(-width // ratio))
                live = src[:actual_bs, :cols].clamp_min(0)
                if ratio != 1:
                    offsets = torch.arange(ratio, dtype=live.dtype, device=live.device)
                    live = (live.unsqueeze(-1) * ratio + offsets).reshape(actual_bs, -1)
                    live = torch.where(
                        live >= ratio, live, torch.zeros_like(live)
                    )  # page 0 stays 0 across its ratio slots
                cols = min(live.shape[1], width)
                dst[:actual_bs, :cols].copy_(live[:, :cols])
                dst[:actual_bs, cols:].zero_()
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
