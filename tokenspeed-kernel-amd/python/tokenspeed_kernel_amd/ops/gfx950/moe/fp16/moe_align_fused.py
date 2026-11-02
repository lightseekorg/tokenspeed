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

"""Fused small-M MoE block-align (decode): single-kernel, sync-free, pure Gluon.

A single workgroup doing O(num_experts) work, in Gluon (which has no LDS int
atomics). One kernel:

  * in-kernel sentinel/zero init of the output + ``gl.barrier`` (folds away the
    separate init-kernel launch),
  * EP localization from global expert IDs plus ``gl.histogram`` -> per-expert
    counts (remote routes are masked; EP = num_experts, no dump bin),
  * **single-block collapse**: top-k contains each expert at most once per
    token, so at M <= block_m, count[e] <= M <= block_m and every hit expert is
    exactly one block. So blocks_pe = (count>0), row_off = block_off*block_m, and
    ``sorted_expert_ids`` is a cheap O(E) scatter (no [NB, E] tile),
  * a [G, G] compare-tile stable rank (no atomics),
  * ``gl.gather`` of the per-expert block offset + scatter each slot to
    ``block_off[e]*block_m + rank``.

Sync-free: outputs sized at the bound of bucketed routes * block_m;
``num_valid`` (real EM) is on-device and the GEMM stages early-out on the padded
tail. Decode-only (EM_MAX + the O(G^2) rank tile grow with M*topk).

An optional ``output_to_zero`` is cleared by extra workgroups of the same
launch, so callers that accumulate expert outputs with atomics need no separate
fill launch.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
    _ZERO_BLOCK,
    _zero_block,
    _zero_target,
)


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


@gluon.jit(do_not_specialize=("G", "sentinel", "zero_numel"))
def _fused_align_kernel(
    ids_ptr,  # [G] int32  flat topk_ids
    wts_ptr,  # [G] fp32   flat topk_weights
    sti_ptr,  # [EM_MAX] int32  out (packed slot<<24|token)
    sw_ptr,  # [EM_MAX] fp32   out (routed weight)
    sei_ptr,  # [NB_MAX] int32  out (expert per block, -1 pad)
    nv_ptr,  # [1] int32       out (EM)
    zero_ptr,  # buffer cleared for the caller (unused when zero_numel is 0)
    G,
    num_experts,
    block_m,
    sentinel,
    nb_max,
    em_max,
    zero_numel,  # 16-byte vectors to clear
    TOPK: gl.constexpr,
    GP: gl.constexpr,  # next_pow2(G) == nb_max
    EP: gl.constexpr,  # next_pow2(num_experts)
    EXPERT_START: gl.constexpr,
    INIT_TILE: gl.constexpr,
    ZERO_BLOCK: gl.constexpr,
):
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # [GP]
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # [EP]
    LR: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # init tile
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])  # 2D rank

    # Workgroups after the first only clear the caller's output.
    pid = gl.program_id(0)
    if pid > 0:
        _zero_block(zero_ptr, zero_numel, pid - 1, ZERO_BLOCK, 4)
        return

    # ---- in-kernel init of the full output (folds away a separate launch) ----
    for r0 in range(0, em_max, INIT_TILE):
        r = r0 + gl.arange(0, INIT_TILE, layout=LR)
        rm = r < em_max
        gl.store(
            sti_ptr + r, gl.full([INIT_TILE], sentinel, gl.int32, layout=LR), mask=rm
        )
        gl.store(sw_ptr + r, gl.full([INIT_TILE], 0.0, gl.float32, layout=LR), mask=rm)
    jb = gl.arange(0, GP, layout=LG)
    gl.store(sei_ptr + jb, gl.full([GP], -1, gl.int32, layout=LG), mask=jb < nb_max)
    gl.barrier()  # order init stores before the (overlapping) scatter below

    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    global_idx = gl.load(ids_ptr + g, mask=gmask, other=EXPERT_START)
    idx = global_idx - EXPERT_START
    route_mask = gmask & (idx >= 0) & (idx < num_experts)
    safe_idx = gl.where(route_mask, idx, 0)
    vals = gl.load(wts_ptr + g, mask=route_mask, other=0.0)
    tok = g // TOPK
    slot = g % TOPK
    packed = ((slot << 24) | tok).to(gl.int32)

    # ---- per-expert counts (masked histogram -> masked lanes excluded) ----
    counts = gl.histogram(safe_idx, EP, mask=route_mask, layout=LE)
    e = gl.arange(0, EP, layout=LE)
    valid_e = e < num_experts
    # single-block collapse: 1 block per hit expert (count<=M<=block_m)
    hit = valid_e & (counts > 0)
    blocks_pe = hit.to(gl.int32)
    block_off = gl.cumsum(blocks_pe, 0) - blocks_pe  # exclusive
    num_blocks = gl.sum(blocks_pe, 0)
    gl.store(nv_ptr, num_blocks * block_m)  # EM

    # ---- sorted_expert_ids: scatter e -> sei[block_off[e]] (O(E)) ----
    gl.store(sei_ptr + block_off, e.to(gl.int32), mask=hit)

    # ---- stable per-expert rank via [G, G] compare tile ----
    idx_row = gl.expand_dims(
        gl.convert_layout(safe_idx, gl.SliceLayout(1, LT)),
        1,
    )
    idx_col = gl.expand_dims(
        gl.convert_layout(safe_idx, gl.SliceLayout(0, LT)),
        0,
    )
    valid_row = gl.expand_dims(
        gl.convert_layout(route_mask, gl.SliceLayout(1, LT)),
        1,
    )
    valid_col = gl.expand_dims(
        gl.convert_layout(route_mask, gl.SliceLayout(0, LT)),
        0,
    )
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = ((idx_row == idx_col) & (g_col < g_row) & valid_row & valid_col).to(
        gl.int32
    )
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)  # [GP]

    # ---- dest = block_off[expert]*block_m + rank, then scatter ----
    dest = gl.gather(block_off, safe_idx, axis=0) * block_m + rank
    gl.store(sti_ptr + dest, packed, mask=route_mask)
    gl.store(sw_ptr + dest, vals, mask=route_mask)


def moe_align_block_size_fused(
    topk_ids: torch.Tensor,  # [M, topk] int
    topk_weights: torch.Tensor,  # [M, topk] float
    num_experts: int,
    block_m: int,
    *,
    expert_start: int = 0,
    output_to_zero: torch.Tensor | None = None,
):
    """Single-kernel sync-free decode block-align (pure Gluon). Same return
    contract as ``moe_align_block_size``.

    ``topk_ids`` may use global expert IDs. ``expert_start`` identifies the
    first expert owned by this rank; routes outside the contiguous local range
    are ignored without a separate localization kernel.

    ``output_to_zero`` is an optional contiguous tensor that the same launch
    clears, for callers that accumulate expert outputs into it with atomics.
    """
    assert topk_ids.shape == topk_weights.shape
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    device = topk_ids.device
    M, topk = topk_ids.shape
    G = M * topk
    sentinel = M
    assert M <= block_m, f"fused small-M align needs M ({M}) <= block_m ({block_m})"

    # Bucket static bounds so changing live token counts reuses compiled code.
    NB_MAX = _next_pow2(max(1, G))
    EM_MAX = NB_MAX * block_m
    GP = _next_pow2(G)
    EP = _next_pow2(num_experts)

    ids = topk_ids.reshape(-1).to(torch.int32).contiguous()
    wts = topk_weights.reshape(-1).to(torch.float32).contiguous()
    sti = torch.empty(EM_MAX, dtype=torch.int32, device=device)
    sw = torch.empty(EM_MAX, dtype=torch.float32, device=device)
    sei = torch.empty(NB_MAX, dtype=torch.int32, device=device)
    nv = torch.empty(1, dtype=torch.int32, device=device)
    zero_target, zero_numel = _zero_target(output_to_zero, sti)

    _fused_align_kernel[(1 + triton.cdiv(zero_numel, _ZERO_BLOCK),)](
        ids,
        wts,
        sti,
        sw,
        sei,
        nv,
        zero_target,
        G,
        num_experts,
        block_m,
        sentinel,
        NB_MAX,
        EM_MAX,
        zero_numel,
        TOPK=topk,
        GP=GP,
        EP=EP,
        EXPERT_START=expert_start,
        INIT_TILE=1024,
        ZERO_BLOCK=_ZERO_BLOCK,
        num_warps=4,
    )
    return sti, sei, sw, nv
