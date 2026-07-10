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

"""Device-side MoE routing + block alignment (pure Gluon).

Drop-in replacement for the pure-torch ``moe_align_block_size`` that produces
the identical contract (``sorted_token_ids`` / ``sorted_expert_ids`` /
``sorted_weights`` / ``num_valid_ids``) but runs entirely on the GPU. Used by
the prefill path (large M), where the fused single-CTA align does not scale
(O(G^2) rank tile). Written in **Gluon** so the whole MoE compiles through one
Gluon backend.

Kernels:
  1. ``_init_kernel``   -- parallel fill of the padding sentinel / zero weights
                           and zero of the per-expert count buffer.
  2. ``_count_kernel``  -- **parallel** per-CTA ``gl.histogram`` of an N-chunk,
                           atomic-added into a global ``counts`` buffer. This
                           replaces the old single-CTA histogram-over-N, which
                           was the align bottleneck (20->122us as M grows).
  3. ``_offsets_kernel``-- single CTA but only O(E) work: prefix-sum counts ->
                           per-expert start rows, write ``sorted_expert_ids`` +
                           ``num_valid`` (EM) / ``num_blocks``.
  4. ``_scatter_kernel``-- parallel: each routed slot atomically claims a rank
                           within its expert (``buffer_atomic_add`` old value)
                           and writes its packed id + weight to that row.

Order within an expert is arbitrary (atomics), unlike the stable reference, but
MoE is order-independent. One host sync reads EM/num_blocks to slice tight
outputs. CDNA3/4 ``buffer_atomic_add`` is float-only, so counts / rank counters
are fp32 (values are small ints, exact in fp32).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _init_kernel(
    sti_ptr, sw_ptr, EM_max, sentinel, BLOCK: gl.constexpr, NW: gl.constexpr
):
    L: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=L)
    mask = offs < EM_max
    gl.store(sti_ptr + offs, gl.full([BLOCK], sentinel, gl.int32, layout=L), mask=mask)
    gl.store(sw_ptr + offs, gl.full([BLOCK], 0.0, gl.float32, layout=L), mask=mask)


@gluon.jit
def _count_kernel(
    exp_ptr,
    N,
    num_experts,
    counts_ptr,
    BLOCK_N: gl.constexpr,
    BLOCK_E: gl.constexpr,
    NW: gl.constexpr,
):
    # Parallel per-CTA histogram of an N-chunk -> atomic-add into global counts.
    LN: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)
    offs = pid * BLOCK_N + gl.arange(0, BLOCK_N, layout=LN)
    nmask = offs < N
    e = gl.load(exp_ptr + offs, mask=nmask, other=num_experts)  # pad -> dump bin
    hist = gl.histogram(e, BLOCK_E, mask=nmask, layout=LE).to(gl.float32)
    e_ids = gl.arange(0, BLOCK_E, layout=LE)
    gl.amd.cdna4.buffer_atomic_add(counts_ptr, e_ids, hist, mask=e_ids < num_experts)


@gluon.jit
def _offsets_kernel(
    counts_ptr,  # [BLOCK_E] float32 (from _count_kernel)
    num_experts,
    block_m,
    nb_max,
    row_off_ptr,  # [BLOCK_E] out: per-expert start row
    fill_ctr_ptr,  # [BLOCK_E] out: zeroed rank counters
    sei_ptr,  # [nb_max]  out: sorted_expert_ids
    num_valid_ptr,  # [1] out: EM
    num_blocks_ptr,  # [1] out: num_blocks
    BLOCK_E: gl.constexpr,
    BLOCK_NB: gl.constexpr,
    NW: gl.constexpr,
):
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW, 1], [1, 0])

    # ---- read parallel-computed counts (single CTA, only O(E) work) ----
    e_ids = gl.arange(0, BLOCK_E, layout=LE)
    valid_e = e_ids < num_experts
    counts = gl.load(counts_ptr + e_ids, mask=valid_e, other=0.0).to(gl.int32)

    blocks_pe = (counts + block_m - 1) // block_m
    padded = blocks_pe * block_m
    row_off = gl.associative_scan(padded, 0, _add) - padded  # exclusive rows
    blocks_incl = gl.associative_scan(blocks_pe, 0, _add)  # inclusive blocks
    EM = gl.sum(padded, 0)
    num_blocks = gl.sum(blocks_pe, 0)

    gl.store(row_off_ptr + e_ids, row_off, mask=valid_e)
    # fill_ctr is float32: CDNA3/4 buffer_atomic_add only supports float. Rank
    # values are small ints, exact in fp32.
    gl.store(
        fill_ctr_ptr + e_ids,
        gl.full([BLOCK_E], 0.0, gl.float32, layout=LE),
        mask=valid_e,
    )  # noqa: E501
    gl.store(num_valid_ptr, EM)
    gl.store(num_blocks_ptr, num_blocks)

    # ---- sorted_expert_ids: expert(b) = #{e : blocks_incl[e] <= b} ----
    for b0 in range(0, nb_max, BLOCK_NB):
        bb = b0 + gl.arange(0, BLOCK_NB, layout=gl.SliceLayout(1, LT))
        bi_row = gl.expand_dims(
            gl.convert_layout(blocks_incl, gl.SliceLayout(0, LT)), 0
        )
        ve_row = gl.expand_dims(
            gl.convert_layout(valid_e.to(gl.int32), gl.SliceLayout(0, LT)), 0
        )
        bb_col = gl.expand_dims(bb, 1)
        cmp = ((bi_row <= bb_col) & (ve_row == 1)).to(gl.int32)
        expert_b = gl.sum(cmp, axis=1)  # [BLOCK_NB] (Slice(1,LT))
        bbL = b0 + gl.arange(0, BLOCK_NB, layout=LE)
        expert_bL = gl.convert_layout(expert_b, LE)
        sei_val = gl.where(
            bbL < num_blocks, expert_bL, gl.full([BLOCK_NB], -1, gl.int32, layout=LE)
        )
        gl.store(sei_ptr + bbL, sei_val, mask=bbL < nb_max)


@gluon.jit
def _scatter_kernel(
    exp_ptr,  # [N] int32 flat topk_ids
    w_ptr,  # [N] float32 flat topk_weights
    N,
    topk,
    num_experts,
    row_off_ptr,  # [BLOCK_E]
    fill_ctr_ptr,  # [BLOCK_E] float32
    sti_ptr,  # [EM_max]
    sw_ptr,  # [EM_max]
    BLOCK: gl.constexpr,
    NW: gl.constexpr,
):
    L: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=L)
    mask = offs < N
    e = gl.load(exp_ptr + offs, mask=mask, other=num_experts)
    valid = mask & (e < num_experts)
    # Each slot atomically claims the next free rank within its expert; the
    # buffer_atomic_add returns the pre-add value (the rank). CDNA3/4 only
    # supports float atomic add, so the counter is fp32 (ranks exact in fp32).
    ones = gl.full([BLOCK], 1.0, gl.float32, layout=L)
    rank_f = gl.amd.cdna4.buffer_atomic_add(fill_ctr_ptr, e, ones, mask=valid)
    rank = rank_f.to(gl.int32)
    ro = gl.load(row_off_ptr + e, mask=valid, other=0)
    dest = ro + rank
    tok = offs // topk
    slot = offs % topk
    packed = ((slot << 24) | tok).to(gl.int32)
    w = gl.load(w_ptr + offs, mask=mask, other=0.0)
    gl.store(sti_ptr + dest, packed, mask=valid)
    gl.store(sw_ptr + dest, w, mask=valid)


def moe_align_block_size_device(
    topk_ids: torch.Tensor,  # [M, topk] int32
    topk_weights: torch.Tensor,  # [M, topk] float32
    num_experts: int,
    block_m: int,
):
    """Fully-device MoE block alignment (pure Gluon). Same return contract as
    the torch ``moe_align_block_size``."""
    assert topk_ids.shape == topk_weights.shape
    device = topk_ids.device
    M, topk = topk_ids.shape
    N = M * topk
    sentinel = M

    exp_flat = topk_ids.reshape(-1).to(torch.int32).contiguous()
    w_flat = topk_weights.reshape(-1).to(torch.float32).contiguous()

    EM_max = N + num_experts * block_m
    nb_max = EM_max // block_m + 1
    BLOCK_E = triton.next_power_of_2(num_experts + 1)  # +1 so the dump bin is in-range

    sti = torch.empty(EM_max, dtype=torch.int32, device=device)
    sw = torch.empty(EM_max, dtype=torch.float32, device=device)
    sei = torch.empty(nb_max, dtype=torch.int32, device=device)
    row_off = torch.empty(BLOCK_E, dtype=torch.int32, device=device)
    fill_ctr = torch.empty(BLOCK_E, dtype=torch.float32, device=device)
    counts = torch.zeros(
        BLOCK_E, dtype=torch.float32, device=device
    )  # count_kernel accum
    meta = torch.empty(2, dtype=torch.int32, device=device)  # [EM, num_blocks]

    INIT_BLOCK = 1024
    _init_kernel[(triton.cdiv(EM_max, INIT_BLOCK),)](
        sti, sw, EM_max, sentinel, BLOCK=INIT_BLOCK, NW=4, num_warps=4
    )
    BLOCK_N = 1024
    _count_kernel[(triton.cdiv(N, BLOCK_N),)](
        exp_flat,
        N,
        num_experts,
        counts,
        BLOCK_N=BLOCK_N,
        BLOCK_E=BLOCK_E,
        NW=4,
        num_warps=4,
    )
    _offsets_kernel[(1,)](
        counts,
        num_experts,
        block_m,
        nb_max,
        row_off,
        fill_ctr,
        sei,
        meta[0:1],
        meta[1:2],
        BLOCK_E=BLOCK_E,
        BLOCK_NB=64,
        NW=4,
        num_warps=4,
    )
    _scatter_kernel[(triton.cdiv(N, 256),)](
        exp_flat,
        w_flat,
        N,
        topk,
        num_experts,
        row_off,
        fill_ctr,
        sti,
        sw,
        BLOCK=256,
        NW=4,
        num_warps=4,
    )

    em, num_blocks = (int(x) for x in meta.tolist())  # single host sync
    num_valid_ids = meta[0:1]
    return sti[:em], sei[:num_blocks], sw[:em], num_valid_ids
