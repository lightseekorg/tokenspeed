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

"""Grouped BF16 expert-route GEMMs for gfx950 prefill.

An atomic-free stable aligner packs each local expert's routes contiguously,
in route order, and cuts them into 256-row tiles plus at most one 128-row
remainder tile; rows past a tile's route count are masked. Each workgroup
computes one ``BLOCK_M x 256`` output tile of one expert with the eight-wave,
two-buffer, warp-pipelined MFMA/LDS schedule of the dense prefill GEMM
(``gemm/fp16/largem.py``), whose activation rows are gathered through the
sorted route ids instead of read contiguously.

  Stage 1: the tile's left half reads gate rows ``[n, n + 128)`` and its right
           half the matching up rows ``I + [n, n + 128)``, so SwiGLU pairs
           share a lane; it stores ``inter[route, n:n + 128]``.
  Stage 2: ``out[route, n:n + 256] = inter[route] @ w2[e, n:n + 256].T``,
           unweighted.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = [
    "gluon_bf16_moe_routes_prefill_stage1_gfx950",
    "gluon_bf16_moe_routes_prefill_stage2_gfx950",
    "gluon_bf16_moe_routes_zero_gfx950",
    "launch_gluon_bf16_moe_routes_prefill_gfx950",
]

# Route aligner: per-block expert histograms (count), one prefix sum into the
# 256- and 128-row tile lists (offsets), and a stable scatter of each route to
# its sorted row (scatter). Batch-varying integers are runtime values.
_ALIGN_BLOCK_ROUTES = 256
_RANK_STEP = 64


def _align_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (grid={grid[0]})"}


# Route and route-block counts follow the batch; one binary serves every count.
@gluon.jit(launch_metadata=_align_launch_metadata, do_not_specialize=("num_routes",))
def gluon_bf16_moe_routes_count_gfx950(
    ids_ptr,
    block_counts_ptr,
    num_routes,
    expert_start,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_ROUTES: gl.constexpr,
):
    """Count route block ``block``'s routes per expert into ``block_counts``."""
    L: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    block = gl.program_id(0)
    route = block * BLOCK_ROUTES + gl.arange(0, BLOCK_ROUTES, L)
    valid = route < num_routes
    local_expert = gl.load(ids_ptr + route, mask=valid, other=-1) - expert_start
    owned = valid & (local_expert >= 0) & (local_expert < NUM_EXPERTS)
    counts = gl.histogram(
        gl.where(owned, local_expert, 0), BLOCK_E, mask=owned, layout=L
    )
    gl.store(block_counts_ptr + block * BLOCK_E + gl.arange(0, BLOCK_E, L), counts)


@gluon.jit
def _exclusive_cumsum(values, layout: gl.constexpr):
    # Scan in a plain 1-D layout (the scan lowering rejects slice layouts).
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    flat = gl.convert_layout(values, L1)
    return gl.convert_layout(gl.cumsum(flat, axis=0) - flat, layout)


@gluon.jit(
    launch_metadata=_align_launch_metadata, do_not_specialize=("num_route_blocks",)
)
def gluon_bf16_moe_routes_offsets_gfx950(
    block_counts_ptr,
    prior_block_counts_ptr,
    expert_row0_ptr,
    big_expert_ptr,
    big_row0_ptr,
    big_num_rows_ptr,
    small_expert_ptr,
    small_row0_ptr,
    small_num_rows_ptr,
    num_tiles_ptr,
    num_route_blocks,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BIG_ROWS: gl.constexpr,
):
    """Route-block and expert prefix sums, and both tile lists."""
    SMALL_ROWS: gl.constexpr = BIG_ROWS // 2
    SCAN_BLOCKS: gl.constexpr = 16
    # A thread holds SCAN_BLOCKS counts of one expert, so the route-block scan
    # stays in registers.
    LT: gl.constexpr = gl.BlockedLayout(
        [SCAN_BLOCKS, 1], [1, 64], [1, gl.num_warps()], [0, 1]
    )
    LE: gl.constexpr = gl.SliceLayout(0, LT)
    LR: gl.constexpr = gl.SliceLayout(1, LT)
    expert = gl.arange(0, BLOCK_E, LE)
    is_expert = expert < NUM_EXPERTS
    expert_routes = gl.zeros((BLOCK_E,), gl.int32, LE)
    for block0 in range(0, num_route_blocks, SCAN_BLOCKS):
        block = block0 + gl.arange(0, SCAN_BLOCKS, LR)
        offs = block[:, None] * BLOCK_E + expert[None, :]
        mask = (block < num_route_blocks)[:, None] & is_expert[None, :]
        counts = gl.load(block_counts_ptr + offs, mask=mask, other=0)
        gl.store(
            prior_block_counts_ptr + offs,
            gl.cumsum(counts, axis=0) - counts + expert_routes[None, :],
            mask=mask,
        )
        expert_routes += gl.sum(counts, axis=0)
    expert_row0 = _exclusive_cumsum(expert_routes, LE)
    gl.store(expert_row0_ptr + expert, expert_row0, mask=is_expert)
    full_tiles = expert_routes // BIG_ROWS
    rem_rows = expert_routes - full_tiles * BIG_ROWS
    big_tiles = full_tiles + (rem_rows > SMALL_ROWS).to(gl.int32)
    small_tiles = ((rem_rows > 0) & (rem_rows <= SMALL_ROWS)).to(gl.int32)
    first_big_tile = _exclusive_cumsum(big_tiles, LE)
    first_small_tile = _exclusive_cumsum(small_tiles, LE)
    gl.store(num_tiles_ptr, gl.sum(big_tiles, axis=0))
    gl.store(num_tiles_ptr + 1, gl.sum(small_tiles, axis=0))
    # Each expert labels its own tiles, one per expert per step.
    for k in range(0, gl.max(big_tiles, axis=0)):
        live = is_expert & (k < big_tiles)
        gl.store(big_expert_ptr + first_big_tile + k, expert, mask=live)
        gl.store(
            big_row0_ptr + first_big_tile + k, expert_row0 + k * BIG_ROWS, mask=live
        )
        gl.store(
            big_num_rows_ptr + first_big_tile + k,
            gl.minimum(expert_routes - k * BIG_ROWS, BIG_ROWS),
            mask=live,
        )
    live = is_expert & (small_tiles > 0)
    gl.store(small_expert_ptr + first_small_tile, expert, mask=live)
    gl.store(
        small_row0_ptr + first_small_tile,
        expert_row0 + full_tiles * BIG_ROWS,
        mask=live,
    )
    gl.store(small_num_rows_ptr + first_small_tile, rem_rows, mask=live)


@gluon.jit(launch_metadata=_align_launch_metadata, do_not_specialize=("num_routes",))
def gluon_bf16_moe_routes_scatter_gfx950(
    ids_ptr,
    prior_block_counts_ptr,
    expert_row0_ptr,
    sorted_routes_ptr,
    num_routes,
    expert_start,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_ROUTES: gl.constexpr,
    RANK_STEP: gl.constexpr,
):
    """Store each owned route of ``block`` at its stable sorted row."""
    LS: gl.constexpr = gl.BlockedLayout([1, 1], [64, 1], [gl.num_warps(), 1], [0, 1])
    LI: gl.constexpr = gl.SliceLayout(1, LS)
    LJ: gl.constexpr = gl.SliceLayout(0, LS)
    block = gl.program_id(0)
    route = block * BLOCK_ROUTES + gl.arange(0, BLOCK_ROUTES, LI)
    valid = route < num_routes
    local_expert = gl.load(ids_ptr + route, mask=valid, other=-1) - expert_start
    owned = valid & (local_expert >= 0) & (local_expert < NUM_EXPERTS)
    rank = gl.zeros((BLOCK_ROUTES,), gl.int32, LI)
    for step0 in range(0, BLOCK_ROUTES, RANK_STEP):
        other_route = block * BLOCK_ROUTES + step0 + gl.arange(0, RANK_STEP, LJ)
        other_expert = (
            gl.load(ids_ptr + other_route, mask=other_route < num_routes, other=-1)
            - expert_start
        )
        earlier = (other_expert[None, :] == local_expert[:, None]) & (
            other_route[None, :] < route[:, None]
        )
        rank += gl.sum(earlier.to(gl.int32), axis=1)
    safe_expert = gl.where(owned, local_expert, 0)
    prior = gl.load(
        prior_block_counts_ptr + block * BLOCK_E + safe_expert, mask=owned, other=0
    )
    row0 = gl.load(expert_row0_ptr + safe_expert, mask=owned, other=0)
    gl.store(sorted_routes_ptr + row0 + prior + rank, route, mask=owned)


class _RouteTiles(NamedTuple):
    """Local routes packed per expert and cut into big and small tiles.

    ``sorted_routes`` holds route indices grouped by expert in route order.
    Tile list ``i`` (0 = big, 1 = small) has ``num_tiles[i]`` live tiles; big
    tile ``t`` covers ``big_num_rows[t]`` sorted rows from ``big_row0[t]``,
    all routed to local expert ``big_expert[t]``, and likewise ``small_*``.
    """

    sorted_routes: torch.Tensor
    big_expert: torch.Tensor
    big_row0: torch.Tensor
    big_num_rows: torch.Tensor
    small_expert: torch.Tensor
    small_row0: torch.Tensor
    small_num_rows: torch.Tensor
    num_tiles: torch.Tensor


def _align_routes(
    topk_ids: torch.Tensor,
    num_experts: int,
    expert_start: int,
    big_rows: int,
) -> _RouteTiles:
    """Group the local routes of ``topk_ids`` into full and half tiles.

    Full tiles hold ``big_rows`` rows and half tiles ``big_rows / 2``.

    Args:
        topk_ids: Contiguous int32 ``[tokens, top_k]`` global expert ids.
        num_experts: Local expert count.
        expert_start: Global id of local expert 0.
        big_rows: Rows of a full tile.

    Returns:
        The tile lists; see :class:`_RouteTiles`.
    """
    device = topk_ids.device
    num_routes = topk_ids.numel()
    num_route_blocks = triton.cdiv(num_routes, _ALIGN_BLOCK_ROUTES)
    max_big_tiles = triton.cdiv(num_routes, big_rows) + num_experts
    block_e = triton.next_power_of_2(num_experts)

    def ints(*shape):
        return torch.empty(shape, dtype=torch.int32, device=device)

    block_counts = ints(num_route_blocks, block_e)
    prior_block_counts = ints(num_route_blocks, block_e)
    expert_row0 = ints(block_e)
    tiles = _RouteTiles(
        ints(num_routes),
        ints(max_big_tiles),
        ints(max_big_tiles),
        ints(max_big_tiles),
        ints(num_experts),
        ints(num_experts),
        ints(num_experts),
        ints(2),
    )
    gluon_bf16_moe_routes_count_gfx950[(num_route_blocks,)](
        topk_ids,
        block_counts,
        num_routes,
        expert_start,
        NUM_EXPERTS=num_experts,
        BLOCK_E=block_e,
        BLOCK_ROUTES=_ALIGN_BLOCK_ROUTES,
        num_warps=4,
    )
    gluon_bf16_moe_routes_offsets_gfx950[(1,)](
        block_counts,
        prior_block_counts,
        expert_row0,
        tiles.big_expert,
        tiles.big_row0,
        tiles.big_num_rows,
        tiles.small_expert,
        tiles.small_row0,
        tiles.small_num_rows,
        tiles.num_tiles,
        num_route_blocks,
        NUM_EXPERTS=num_experts,
        BLOCK_E=block_e,
        BIG_ROWS=big_rows,
        num_warps=4,
    )
    gluon_bf16_moe_routes_scatter_gfx950[(num_route_blocks,)](
        topk_ids,
        prior_block_counts,
        expert_row0,
        tiles.sorted_routes,
        num_routes,
        expert_start,
        NUM_EXPERTS=num_experts,
        BLOCK_E=block_e,
        BLOCK_ROUTES=_ALIGN_BLOCK_ROUTES,
        RANK_STEP=_RANK_STEP,
        num_warps=4,
    )
    return tiles


_BLOCK_N = 256
# Route rows of a full tile; an expert's remainder of at most half this many
# rows takes one half-height tile.
_BIG_TILE_ROWS = 256
_NUM_WARPS = 8
_ZERO_BLOCK_ROUTES = 16
NUM_XCDS = gl.constexpr(8)
GROUP_SIZE_M = gl.constexpr(4)


def _prefill_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (tokens={args['num_tokens']})"}


@gluon.constexpr_function
def _load_layout_a(half_m: int):
    # [half_m, 64]: 16 bytes per lane; eight waves stack along M.
    if half_m == 128:
        return gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
            warp_bases=[[1, 0], [2, 0], [4, 0]],
            block_bases=[],
            shape=[half_m, 64],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [8, 0], [16, 0], [32, 0]],
        warp_bases=[[1, 0], [2, 0], [4, 0]],
        block_bases=[],
        shape=[half_m, 64],
    )


@gluon.constexpr_function
def _shared_layout_a(half_m: int):
    # Offsets follow the load layout's lane order (a warp's copy fills one
    # contiguous range), then its register and warp rows; padded every 512.
    if half_m == 128:
        rows = [[16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]]
    else:
        rows = [[8, 0], [16, 0], [32, 0], [1, 0], [2, 0], [4, 0]]
    return gl.PaddedSharedLayout(
        [[512, 16]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]] + rows,
        [],
        [half_m, 64],
    )


@gluon.constexpr_function
def _store_layout(half_m: int):
    # Covers a [half_m, 128] quadrant once with eight waves.
    return gl.BlockedLayout([half_m // 32, 8], [4, 16], [8, 1], [1, 0])


@gluon.jit
def _grouped_tile(
    a_ptr,
    b_ptr,
    a_top_offsets,
    a_bot_offsets,
    right_delta,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Accumulate ``a[rows] @ b[cols].T`` for one ``BLOCK_M x 256`` tile.

    ``a_top_offsets`` / ``a_bot_offsets`` are the gathered element offsets of
    K step 0 of the top and bottom half rows in the A load layout. ``b_ptr``
    points at the tile's first weight row; the right half starts
    ``right_delta`` rows further. Returns the four FP32 quadrants (top-left,
    bottom-left, top-right, bottom-right).
    """
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = 128
    BK: gl.constexpr = 64
    gl.static_assert(K % (2 * BK) == 0 and K >= 4 * BK, "K must pair 64-wide steps")
    g_load_layout_b: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2], [0, 4]],
        block_bases=[],
        shape=[BK, HALF_N],
    )
    shared_layout_a: gl.constexpr = _shared_layout_a(HALF_M)
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
        ],
        [],
        [BK, HALF_N],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 4]
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    smem_a_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, HALF_M, BK], shared_layout_a
    )
    smem_a_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, HALF_M, BK], shared_layout_a
    )
    smem_b_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, BK, HALF_N], shared_layout_b
    )
    smem_b_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, BK, HALF_N], shared_layout_b
    )

    offs_bn = gl.arange(0, HALF_N, gl.SliceLayout(0, g_load_layout_b))
    offs_bk = gl.arange(0, BK, gl.SliceLayout(1, g_load_layout_b))
    b_left_offsets = offs_bk[:, None] + offs_bn[None, :] * K
    b_right_offsets = offs_bk[:, None] + (offs_bn + right_delta)[None, :] * K
    a_top_next = a_top_offsets + BK
    a_bot_next = a_bot_offsets + BK
    b_left_next = b_left_offsets + BK
    b_right_next = b_right_offsets + BK
    a_base = a_ptr
    b_base = b_ptr

    acc_tl = gl.zeros((HALF_M, HALF_N), gl.float32, mfma_layout)
    acc_bl = gl.zeros((HALF_M, HALF_N), gl.float32, mfma_layout)
    acc_tr = gl.zeros((HALF_M, HALF_N), gl.float32, mfma_layout)
    acc_br = gl.zeros((HALF_M, HALF_N), gl.float32, mfma_layout)

    ITERS: gl.constexpr = K // BK

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_b_left.index(0), b_base, b_left_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_a_top.index(0), a_base, a_top_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_a_bot.index(0), a_base, a_bot_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_b_right.index(0), b_base, b_right_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_b_left.index(1), b_base, b_left_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_a_top.index(1), a_base, a_top_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_a_bot.index(1), a_base, a_bot_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_b_right.index(1), b_base, b_right_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    a_base += 2 * BK
    b_base += 2 * BK

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smem_b_left.index(0).load(dot_b)
    a_top = smem_a_top.index(0).load(dot_a)

    for _ in tl.range(0, ITERS - 2, 2):
        gl.amd.cdna4.async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.cdna4.mfma(a_top, b_left, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smem_a_bot.index(0).load(dot_a)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem_b_left.index(0), b_base, b_left_offsets
            )
            gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.cdna4.mfma(a_bot, b_left, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smem_b_right.index(0).load(dot_b)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem_a_top.index(0), a_base, a_top_offsets
            )
            gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.cdna4.mfma(a_top, b_right, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smem_b_left.index(1).load(dot_b)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem_a_bot.index(0), a_base, a_bot_offsets
            )
            gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.cdna4.mfma(a_bot, b_right, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smem_a_top.index(1).load(dot_a)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem_b_right.index(0), b_base, b_right_offsets
            )
            gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.cdna4.mfma(a_top, b_left, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smem_a_bot.index(1).load(dot_a)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem_b_left.index(1), b_base, b_left_next
            )
            gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.cdna4.mfma(a_bot, b_left, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smem_b_right.index(1).load(dot_b)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem_a_top.index(1), a_base, a_top_next
            )
            gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.cdna4.mfma(a_top, b_right, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smem_b_left.index(0).load(dot_b)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem_a_bot.index(1), a_base, a_bot_next
            )
            gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.cdna4.mfma(a_bot, b_right, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smem_a_top.index(0).load(dot_a)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem_b_right.index(1), b_base, b_right_next
            )
            gl.amd.cdna4.async_copy.commit_group()
            a_base += 2 * BK
            b_base += 2 * BK

    acc_tl = gl.amd.cdna4.mfma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx: gl.constexpr = (ITERS - 2) % 2
    a_bot = smem_a_bot.index(l_idx).load(dot_a)
    acc_bl = gl.amd.cdna4.mfma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = smem_b_right.index(l_idx).load(dot_b)
    acc_tr = gl.amd.cdna4.mfma(a_top, b_right, acc_tr)
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx: gl.constexpr = 1 - l_idx
    b_left = smem_b_left.index(g_idx).load(dot_b)
    acc_br = gl.amd.cdna4.mfma(a_bot, b_right, acc_br)
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = smem_a_top.index(g_idx).load(dot_a)
    acc_tl = gl.amd.cdna4.mfma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smem_a_bot.index(g_idx).load(dot_a)
    acc_bl = gl.amd.cdna4.mfma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smem_b_right.index(g_idx).load(dot_b)
    acc_tr = gl.amd.cdna4.mfma(a_top, b_right, acc_tr)
    acc_br = gl.amd.cdna4.mfma(a_bot, b_right, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@gluon.jit
def _tile_ids(
    tile_expert_ptr,
    tile_row0_ptr,
    tile_num_rows_ptr,
    num_tiles_ptr,
    N_TILES: gl.constexpr,
):
    """Return this program's ``(pid_n, expert, row0, num_rows)``.

    ``expert`` is -1 past the live route tiles. The grid covers the worst
    case, but only the first ``num_tiles`` route tiles hold routes. Programs
    are numbered over those live tiles only, so the XCD remap below spreads
    them over all eight XCDs instead of handing the live prefix to the first
    few.
    """
    pid = gl.program_id(0)
    num_m_tiles = gl.load(num_tiles_ptr)
    grid_mn = num_m_tiles * N_TILES
    live = pid < grid_mn
    # Adjacent programs land on different XCDs; give each XCD a contiguous
    # range of tiles, then walk GROUP_SIZE_M route tiles per weight column.
    pids_per_xcd = (grid_mn + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = grid_mn % NUM_XCDS
    tall_xcds = gl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    pid = gl.where(
        xcd < tall_xcds,
        xcd * pids_per_xcd + local_pid,
        tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid,
    )
    group = GROUP_SIZE_M * N_TILES
    first_m = (pid // group) * GROUP_SIZE_M
    group_m = gl.maximum(gl.minimum(num_m_tiles - first_m, GROUP_SIZE_M), 1)
    pid_m = first_m + (pid % group) % group_m
    pid_n = (pid % group) // group_m
    expert = gl.load(tile_expert_ptr + pid_m, mask=live, other=-1)
    row0 = gl.load(tile_row0_ptr + pid_m, mask=live, other=0)
    num_rows = gl.load(tile_num_rows_ptr + pid_m, mask=live, other=0)
    return pid_n, expert, row0, num_rows


@gluon.jit(launch_metadata=_prefill_launch_metadata, do_not_specialize=("num_tokens",))
def gluon_bf16_moe_routes_prefill_stage1_gfx950(
    x_ptr,
    w13_ptr,
    inter_ptr,
    sorted_routes_ptr,
    tile_expert_ptr,
    tile_row0_ptr,
    tile_num_rows_ptr,
    num_tiles_ptr,
    num_tokens,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Gate/up GEMM + SwiGLU of one ``BLOCK_M`` route tile x 128 columns."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    N_TILES: gl.constexpr = INTERMEDIATE // 128
    pid_n, expert, row0, num_rows = _tile_ids(
        tile_expert_ptr, tile_row0_ptr, tile_num_rows_ptr, num_tiles_ptr, N_TILES
    )
    if expert >= 0:
        LA: gl.constexpr = _load_layout_a(HALF_M)
        r = gl.arange(0, HALF_M, gl.SliceLayout(1, LA))
        k = gl.arange(0, 64, gl.SliceLayout(0, LA))
        tile_routes = sorted_routes_ptr + row0
        # Rows past the tile's routes read token 0 and are not stored.
        top_tokens = gl.load(tile_routes + r, mask=r < num_rows, other=0) // TOPK
        bot_tokens = (
            gl.load(tile_routes + HALF_M + r, mask=r + HALF_M < num_rows, other=0)
            // TOPK
        )
        acc_tl, acc_bl, acc_tr, acc_br = _grouped_tile(
            x_ptr,
            w13_ptr
            + expert.to(gl.int64) * (2 * INTERMEDIATE * HIDDEN)
            + (pid_n * 128) * HIDDEN,
            top_tokens[:, None] * HIDDEN + k[None, :],
            bot_tokens[:, None] * HIDDEN + k[None, :],
            INTERMEDIATE,
            HIDDEN,
            BLOCK_M,
        )
        SL: gl.constexpr = _store_layout(HALF_M)
        sr = gl.arange(0, HALF_M, gl.SliceLayout(1, SL))
        sc = pid_n * 128 + gl.arange(0, 128, gl.SliceLayout(0, SL))
        for half in gl.static_range(2):
            if half == 0:
                gate = acc_tl
                up = acc_tr
            else:
                gate = acc_bl
                up = acc_br
            h = gate * (1.0 / (1.0 + gl.exp(-gate))) * up
            h = gl.convert_layout(h.to(inter_ptr.dtype.element_ty), SL)
            live = sr + half * HALF_M < num_rows
            route = gl.load(tile_routes + half * HALF_M + sr, mask=live, other=0)
            gl.amd.cdna4.buffer_store(
                h,
                inter_ptr,
                route[:, None] * INTERMEDIATE + sc[None, :],
                mask=live[:, None] & (sc < INTERMEDIATE)[None, :],
            )


@gluon.jit(launch_metadata=_prefill_launch_metadata, do_not_specialize=("num_tokens",))
def gluon_bf16_moe_routes_prefill_stage2_gfx950(
    inter_ptr,
    w2_ptr,
    out_ptr,
    sorted_routes_ptr,
    tile_expert_ptr,
    tile_row0_ptr,
    tile_num_rows_ptr,
    num_tiles_ptr,
    num_tokens,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Down GEMM of one ``BLOCK_M`` route tile x 256 output columns."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    N_TILES: gl.constexpr = HIDDEN // 256
    pid_n, expert, row0, num_rows = _tile_ids(
        tile_expert_ptr, tile_row0_ptr, tile_num_rows_ptr, num_tiles_ptr, N_TILES
    )
    if expert >= 0:
        LA: gl.constexpr = _load_layout_a(HALF_M)
        r = gl.arange(0, HALF_M, gl.SliceLayout(1, LA))
        k = gl.arange(0, 64, gl.SliceLayout(0, LA))
        tile_routes = sorted_routes_ptr + row0
        top_routes = gl.load(tile_routes + r, mask=r < num_rows, other=0)
        bot_routes = gl.load(
            tile_routes + HALF_M + r, mask=r + HALF_M < num_rows, other=0
        )
        acc_tl, acc_bl, acc_tr, acc_br = _grouped_tile(
            inter_ptr,
            w2_ptr
            + expert.to(gl.int64) * (HIDDEN * INTERMEDIATE)
            + (pid_n * 256) * INTERMEDIATE,
            top_routes[:, None] * INTERMEDIATE + k[None, :],
            bot_routes[:, None] * INTERMEDIATE + k[None, :],
            128,
            INTERMEDIATE,
            BLOCK_M,
        )
        SL: gl.constexpr = _store_layout(HALF_M)
        sr = gl.arange(0, HALF_M, gl.SliceLayout(1, SL))
        sc = gl.arange(0, 128, gl.SliceLayout(0, SL))
        for quad in gl.static_range(4):
            if quad == 0:
                acc = acc_tl
            elif quad == 1:
                acc = acc_bl
            elif quad == 2:
                acc = acc_tr
            else:
                acc = acc_br
            value = gl.convert_layout(acc.to(out_ptr.dtype.element_ty), SL)
            live = sr + (quad % 2) * HALF_M < num_rows
            route = gl.load(tile_routes + (quad % 2) * HALF_M + sr, mask=live, other=0)
            col = pid_n * 256 + (quad // 2) * 128 + sc
            gl.amd.cdna4.buffer_store(
                value,
                out_ptr,
                route[:, None] * HIDDEN + col[None, :],
                mask=live[:, None] & (col < HIDDEN)[None, :],
            )


@gluon.jit(launch_metadata=_prefill_launch_metadata, do_not_specialize=("num_tokens",))
def gluon_bf16_moe_routes_zero_gfx950(
    ids_ptr,
    out_ptr,
    num_tokens,
    expert_start,
    HIDDEN: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_ROUTES: gl.constexpr,
):
    """Zero the output rows of routes whose expert is not local.

    One program checks ``BLOCK_ROUTES`` routes and writes only when one of
    them is absent, so a rank that owns every selected expert reads just the
    ids.
    """
    L: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [gl.num_warps(), 1], [1, 0])
    route = gl.program_id(0) * BLOCK_ROUTES + gl.arange(
        0, BLOCK_ROUTES, gl.SliceLayout(1, L)
    )
    valid = route < num_tokens * TOPK
    local_expert = (
        gl.load(ids_ptr + route, mask=valid, other=expert_start) - expert_start
    )
    absent = valid & ((local_expert < 0) | (local_expert >= NUM_EXPERTS))
    if gl.max(absent.to(gl.int32), axis=0) > 0:
        col = gl.arange(0, 512, gl.SliceLayout(0, L))
        zeros = gl.zeros((BLOCK_ROUTES, 512), out_ptr.dtype.element_ty, L)
        for col_block in gl.static_range(HIDDEN // 512):
            gl.amd.cdna4.buffer_store(
                zeros,
                out_ptr,
                route[:, None] * HIDDEN + col_block * 512 + col[None, :],
                mask=absent[:, None] & (col < 512)[None, :],
            )


def launch_gluon_bf16_moe_routes_prefill_gfx950(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_start: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write unweighted BF16 expert routes into ``output`` with grouped GEMMs.

    Args mirror ``launch_gluon_bf16_moe_routes_decode_gfx950``; rows of routes
    whose expert is not local are zeroed. Each stage runs once over the full
    256-row tiles and once over the 128-row remainder tiles.
    """
    num_tokens, hidden = x.shape
    num_experts, two_i, _ = w13.shape
    intermediate_size = two_i // 2
    top_k = topk_ids.shape[1]
    gluon_bf16_moe_routes_zero_gfx950[
        (triton.cdiv(num_tokens * top_k, _ZERO_BLOCK_ROUTES),)
    ](
        topk_ids,
        output,
        num_tokens,
        expert_start,
        HIDDEN=hidden,
        TOPK=top_k,
        NUM_EXPERTS=num_experts,
        BLOCK_ROUTES=_ZERO_BLOCK_ROUTES,
        num_warps=4,
    )
    tiles = _align_routes(topk_ids, num_experts, expert_start, _BIG_TILE_ROWS)
    tile_lists = (
        (_BIG_TILE_ROWS, tiles.big_expert, tiles.big_row0, tiles.big_num_rows, 0),
        (
            _BIG_TILE_ROWS // 2,
            tiles.small_expert,
            tiles.small_row0,
            tiles.small_num_rows,
            1,
        ),
    )
    inter = torch.empty(
        (num_tokens * top_k, intermediate_size), dtype=x.dtype, device=x.device
    )
    # Accumulators stay in VGPRs, as in the dense prefill GEMM.
    attrs = (("amdgpu-agpr-alloc", "0,0"),)
    for block_m, tile_expert, tile_row0, tile_num_rows, list_index in tile_lists:
        gluon_bf16_moe_routes_prefill_stage1_gfx950[
            (tile_expert.shape[0] * (intermediate_size // 128),)
        ](
            x,
            w13,
            inter,
            tiles.sorted_routes,
            tile_expert,
            tile_row0,
            tile_num_rows,
            tiles.num_tiles[list_index:],
            num_tokens,
            HIDDEN=hidden,
            INTERMEDIATE=intermediate_size,
            TOPK=top_k,
            BLOCK_M=block_m,
            num_warps=_NUM_WARPS,
            llvm_fn_attrs=attrs,
        )
    for block_m, tile_expert, tile_row0, tile_num_rows, list_index in tile_lists:
        gluon_bf16_moe_routes_prefill_stage2_gfx950[
            (tile_expert.shape[0] * (hidden // _BLOCK_N),)
        ](
            inter,
            w2,
            output,
            tiles.sorted_routes,
            tile_expert,
            tile_row0,
            tile_num_rows,
            tiles.num_tiles[list_index:],
            num_tokens,
            HIDDEN=hidden,
            INTERMEDIATE=intermediate_size,
            TOPK=top_k,
            BLOCK_M=block_m,
            num_warps=_NUM_WARPS,
            llvm_fn_attrs=attrs,
        )
    return output
