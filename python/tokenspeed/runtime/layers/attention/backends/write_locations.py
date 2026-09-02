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

"""KV write-location math over stacked per-group kernel page tables.

One invariant, three shapes: ``slot = table[req, pos // P] * P + pos % P``
with ``P`` the group's kernel page size. Decode writes a trailing window of
``n`` positions per request (``n == 1`` plain decode, ``n == N`` target
verify or a block draft), extend writes the ``[prefix, prefix + new)`` span
per request in q/k/v token order. Null and hole pages (id <= 0) and positions
past the table route to slot 0, the zero-initialized dummy page that never
aliases a live request.

Every function takes the stacked ``[G, rows, W]`` table the router fills
(``group_tables.GroupTableStacks``) so all groups' locations come from one
launch. CUDA tensors run the triton kernels; CPU tensors take the torch path
(unit tests), which is the reference the kernels are pinned against.

The token-shaped sibling — arbitrary positions over one group's raw table,
failing closed to the ``-1`` sentinel (V4's SWA / compressor-state /
indexer-state writes) — is ``page_table.group_slot_mapping_from_raw``; both
spell the same invariant.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_rows", "tokens_per_req"])
def _decode_locs_kernel(
    tables_ptr,  # [G, rows, W] int32
    page_sizes_ptr,  # [G] int32
    seq_lens_ptr,  # [bs] int32
    out_ptr,  # [G, cap] int32
    stride_g,
    stride_b,
    width,
    out_stride_g,
    num_rows,  # bs * tokens_per_req
    tokens_per_req,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0)
    start = tl.program_id(1) * BLOCK
    rows = start + tl.arange(0, BLOCK)
    mask = rows < num_rows
    req = rows // tokens_per_req
    t = rows % tokens_per_req
    seq_len = tl.load(seq_lens_ptr + req, mask=mask, other=1).to(tl.int64)
    pos = tl.maximum(seq_len - tokens_per_req + t, 0)
    page_size = tl.load(page_sizes_ptr + g).to(tl.int64)
    page_idx = pos // page_size
    overflow = page_idx >= width
    page_idx = tl.minimum(page_idx, width - 1)
    page = tl.load(
        tables_ptr + g * stride_g + req * stride_b + page_idx, mask=mask, other=0
    ).to(tl.int64)
    loc = page * page_size + pos % page_size
    loc = tl.where(overflow | (page <= 0), 0, loc)
    tl.store(out_ptr + g * out_stride_g + rows, loc.to(tl.int32), mask=mask)


@triton.jit(do_not_specialize=["width"])
def _extend_locs_kernel(
    tables_ptr,  # [G, rows, W] int32
    page_sizes_ptr,  # [G] int32
    prefix_lens_ptr,  # [bs] int32
    extend_lens_ptr,  # [bs] int32
    cu_extend_lens_ptr,  # [bs + 1] int32
    out_ptr,  # [G, total] int32
    stride_g,
    stride_b,
    width,
    out_stride_g,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0)
    req = tl.program_id(1)
    prefix = tl.load(prefix_lens_ptr + req).to(tl.int64)
    num_new = tl.load(extend_lens_ptr + req)
    out_base = tl.load(cu_extend_lens_ptr + req).to(tl.int64)
    page_size = tl.load(page_sizes_ptr + g).to(tl.int64)
    for block_start in range(0, num_new, BLOCK):
        offs = block_start + tl.arange(0, BLOCK)
        mask = offs < num_new
        pos = prefix + offs
        page_idx = pos // page_size
        overflow = page_idx >= width
        page_idx = tl.minimum(page_idx, width - 1)
        page = tl.load(
            tables_ptr + g * stride_g + req * stride_b + page_idx,
            mask=mask,
            other=0,
        ).to(tl.int64)
        loc = page * page_size + pos % page_size
        loc = tl.where(overflow | (page <= 0), 0, loc)
        tl.store(
            out_ptr + g * out_stride_g + out_base + offs, loc.to(tl.int32), mask=mask
        )


def _decode_positions(seq_lens: torch.Tensor, tokens_per_req: int) -> torch.Tensor:
    """``[bs * n]`` token-major write positions of the trailing window."""
    n = tokens_per_req
    lens = seq_lens.to(torch.int64)
    if n == 1:
        return (lens - 1).clamp_min(0)
    steps = torch.arange(n, device=seq_lens.device, dtype=torch.int64)
    return (lens.unsqueeze(1) - n + steps).clamp_min(0).reshape(-1)


def _gather_slots(
    table: torch.Tensor, req: torch.Tensor, pos: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Torch reference of the slot rule for one group's ``[rows, W]`` table."""
    width = table.shape[1]
    page_idx = pos // page_size
    overflow = page_idx >= width
    page = table[req, page_idx.clamp_max(width - 1)].to(torch.int64)
    loc = page * page_size + pos % page_size
    return torch.where(overflow | (page <= 0), torch.zeros_like(loc), loc).to(
        torch.int32
    )


def decode_write_locations(
    tables: torch.Tensor,
    page_sizes: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    bs: int,
    tokens_per_req: int,
) -> None:
    """Fill ``out[:, : bs * tokens_per_req]`` with every group's decode
    write slots.

    Args:
        tables: ``[G, rows >= bs, W]`` int32 stacked kernel page tables.
        page_sizes: ``[G]`` int32 kernel page size per group.
        seq_lens: ``[>= bs]`` int32 lengths including the newest token(s);
            request ``b`` writes positions ``seq_lens[b] - n .. seq_lens[b] - 1``.
        out: ``[G, cap]`` int32 destination, ``cap >= bs * tokens_per_req``;
            filled token-major per request (the verify layout).
        bs: Rows to compute (padded rows read seq_len 1 and land on slot 0).
        tokens_per_req: Trailing window width ``n``.
    """
    n = max(int(tokens_per_req), 1)
    num_rows = bs * n
    if num_rows == 0 or tables.shape[0] == 0:
        return
    if out.shape[1] < num_rows:
        raise RuntimeError(
            f"decode write-location buffer holds {out.shape[1]} slots per group, "
            f"need {num_rows} (bs={bs}, tokens_per_req={n})"
        )
    if tables.is_cuda:
        block = 128
        grid = (tables.shape[0], triton.cdiv(num_rows, block))
        _decode_locs_kernel[grid](
            tables,
            page_sizes,
            seq_lens,
            out,
            tables.stride(0),
            tables.stride(1),
            tables.shape[2],
            out.stride(0),
            num_rows,
            n,
            BLOCK=block,
        )
        return
    pos = _decode_positions(seq_lens[:bs], n)
    req = torch.arange(bs, device=tables.device).repeat_interleave(n)
    for g in range(tables.shape[0]):
        out[g, :num_rows] = _gather_slots(tables[g], req, pos, int(page_sizes[g]))


def extend_write_locations(
    tables: torch.Tensor,
    page_sizes: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """Every group's extend write slots, ``[G, total_tokens]`` in q/k/v
    token order (request-major, ``cu_extend_seq_lens`` layout).

    Args:
        tables: ``[G, rows >= bs, W]`` int32 stacked kernel page tables.
        page_sizes: ``[G]`` int32 kernel page size per group.
        extend_prefix_lens: ``[bs]`` int32 cached prefix per request.
        extend_seq_lens: ``[bs]`` int32 new tokens per request.
        total_tokens: ``sum(extend_seq_lens)`` from the host mirror — no
            device sync here.

    Returns:
        A fresh ``[G, total_tokens]`` int32 tensor (extend metadata is
        rebuilt per round and is never graph-recorded).
    """
    bs = extend_seq_lens.shape[0]
    out = torch.empty(
        (tables.shape[0], total_tokens), dtype=torch.int32, device=tables.device
    )
    if total_tokens == 0 or tables.shape[0] == 0 or bs == 0:
        return out
    cu_extend = torch.zeros(bs + 1, dtype=torch.int32, device=tables.device)
    torch.cumsum(extend_seq_lens, dim=0, out=cu_extend[1:])
    if tables.is_cuda:
        _extend_locs_kernel[(tables.shape[0], bs)](
            tables,
            page_sizes,
            extend_prefix_lens,
            extend_seq_lens,
            cu_extend,
            out,
            tables.stride(0),
            tables.stride(1),
            tables.shape[2],
            out.stride(0),
            BLOCK=128,
        )
        return out
    lens = extend_seq_lens.to(torch.int64)
    req = torch.arange(bs, device=tables.device).repeat_interleave(lens)
    starts = cu_extend[:-1].to(torch.int64).repeat_interleave(lens)
    pos = extend_prefix_lens.to(torch.int64)[req] + (
        torch.arange(total_tokens, device=tables.device) - starts
    )
    for g in range(tables.shape[0]):
        out[g] = _gather_slots(tables[g], req, pos, int(page_sizes[g]))
    return out


def check_write_locations(
    table: torch.Tensor, locs: torch.Tensor, page_size: int, *, what: str
) -> None:
    """TOKENSPEED_CACHE_DEBUG assertion (eager only, device sync): every slot
    lands in a real page of ``table``. Not for padded batches — dummy rows
    resolve to page 0 by contract."""
    pages = (locs.to(torch.int64) // page_size).to(torch.int32)
    assert (pages != 0).all(), f"{what}: cache write location in null page 0"
    real = table[table > 0]
    assert torch.isin(pages, real).all(), f"{what}: cache write pages escape the table"
