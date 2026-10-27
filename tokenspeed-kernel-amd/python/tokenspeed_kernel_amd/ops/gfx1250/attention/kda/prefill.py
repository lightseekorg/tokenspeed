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

"""Chunk-parallel KDA prefill for GFX1250.

The production ``kda_recurrent`` runs a serial token-by-token scan that is
optimal for decode but leaves the machine idle during long prefills. This module
provides a chunk-parallel prefill built on the gated delta-rule (WY) chunking
scheme (Yang et al., DeltaNet-2). For each chunk the intra-chunk work is a set of
matmuls (parallel across all chunks); only the inter-chunk state carry stays
sequential:

    bg = cumsum(g)                              # per-channel gate
    Akk = tril(diag(beta) . Kd . Ki^T, -1)       # state-independent
    T   = (I + Akk)^{-1}                         # FP32, by band
    u   = T . (beta.V) , W = T . (beta.Kd)
    Aqk = tril(scale.Qd.Ki^T, 0), Kg = Kn.e^(bg_last-bg)
    -- sequential over chunks --
    v_new = u - W . H
    o     = Qd . H + tril(Qd Ki^T, 0) . v_new
    H     = exp(bg_last) . H + (Kn.exp(bg_last-bg))^T . v_new

with Kd = Kn.e^{bg}, Ki = Kn.e^{-bg}, Qd = scale . Qn.e^{bg}.

Variable-length prefill: multiple requests are packed into one flat token buffer
and delimited by ``cu_seqlens`` (a prefix-sum of per-sequence lengths).
``_chunk_indices`` maps each global chunk to
its ``(sequence, local-chunk)`` so chunks never span a sequence boundary, the
chunk-local cumsum resets per sequence, and the sequential scan restarts ``H``
from each sequence's own initial state. A single sequence is just the ``N = 1``
case (``cu_seqlens = [0, T]``).

One program prepares each (chunk, head). It holds the chunk as four 16-token
bands, so pairwise gates are split at band boundaries: each exponential spans
at most one band, or decays between bands. Akk and Aqk come from batched band
WMMAs, I + Akk is inverted in FP32 by band, and W, U, Qg, Kg, Aqk and the
chunk's decay go to memory. The walk then carries the state across chunks.
When too few sequences would idle the GPU, it splits each sequence into time
groups that are summarized, chained and walked in parallel.
"""

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_CHUNK_SIZE = 64
_SUBCHUNK_SIZE = 16
_PREPARE_WARPS = 4
cdna5 = gl.amd.cdna5


@gluon.constexpr_function
def _wmma_layout(warps, axis, rank, instr_k=32):
    """Transposed WMMA layout with ``warps`` warps stacked along ``axis``."""
    bases = []
    for b in range(warps.bit_length() - 1):
        basis = [0] * rank
        basis[axis] = 1 << b
        bases.append(basis)
    return gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=bases,
        reg_bases=[],
        instr_shape=[16, 16, instr_k],
        rank=rank,
    )


@gluon.jit
def _band_index(n: gl.constexpr, axis: gl.constexpr, layout: gl.constexpr):
    """arange(n) along ``axis`` of a rank-3 tensor in ``layout``."""
    if axis == 0:
        index = gl.arange(0, n, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
        index = index[:, None, None]
    elif axis == 1:
        index = gl.arange(0, n, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
        index = index[None, :, None]
    else:
        index = gl.arange(0, n, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
        index = index[None, None, :]
    return index


@gluon.jit
def _mm_f32(
    lhs, rhs, layout: gl.constexpr, a_layout: gl.constexpr, b_layout: gl.constexpr
):
    acc = gl.zeros(lhs.shape, gl.float32, layout)
    return cdna5.wmma(
        gl.convert_layout(lhs, a_layout), gl.convert_layout(rhs, b_layout), acc
    )


@gluon.jit
def _shift_bands(
    diagonal, s: gl.constexpr, NB: gl.constexpr, BC: gl.constexpr, layout: gl.constexpr
):
    """The [NB * BC, BC] blocks in LDS ``diagonal`` moved down ``s`` bands, as [NB, BC, BC].

    Bands below s are zero.
    """
    band = _band_index(NB, 0, layout)
    shifted = gl.zeros([NB, BC, BC], gl.float32, layout)
    for b in gl.static_range(s, NB):
        part = diagonal.slice((b - s) * BC, BC, dim=0).load(gl.SliceLayout(0, layout))
        shifted = gl.where(band == b, part[None, :, :], shifted)
    return shifted


@gluon.jit
def _column_beta(
    beta_base, band, rows_left, H: gl.constexpr, BC: gl.constexpr, layout: gl.constexpr
):
    """sigmoid(beta) of column band ``band``, along the last axis of ``layout``; 0 past the sequence."""
    column = band * BC + gl.arange(
        0, BC, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    inside = column < rows_left
    beta = cdna5.buffer_load(
        beta_base, gl.minimum(column, rows_left - 1) * H, mask=inside, other=0.0
    ).to(gl.float32)
    return gl.where(inside, gl.fdiv(1.0, 1.0 + gl.exp(-beta)), 0.0)[None, None, :]


@gluon.jit
def gluon_kda_paged_prefill_prepare_gfx1250(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    a_log,
    dt_bias,
    qg,
    w,
    u,
    kg,
    aqk,
    decay,
    cu_seqlens,
    chunk_indices,
    SCALE: gl.constexpr,
    H: gl.constexpr,
    D: gl.constexpr,
    BT: gl.constexpr,
    BC: gl.constexpr,
    NW: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
):
    """Everything the walk reads for one (chunk, head), in one program.

    The chunk is held as [NB, BC, D] bands, one per warp, so the band-local
    gate cumsum and every [BC, BC] block product run batched over bands. Akk
    never leaves registers: block (b, J) of every band comes from one batched
    WMMA per column band J, each band's diagonal block is inverted by repeated
    squaring, and the off-diagonal inverse blocks are built a diagonal at a
    time from band-shifted copies read back from LDS.
    """
    NB: gl.constexpr = BT // BC
    LOG2E: gl.constexpr = 1.4426950408889634
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    sequence = gl.load(chunk_indices + chunk * 2).to(gl.int32)
    local_chunk = gl.load(chunk_indices + chunk * 2 + 1).to(gl.int32)
    begin = gl.load(cu_seqlens + sequence).to(gl.int32)
    token0 = local_chunk * BT
    rows_left = gl.load(cu_seqlens + sequence + 1).to(gl.int32) - begin - token0

    mb: gl.constexpr = _wmma_layout(NW, 0, 3)
    a3: gl.constexpr = gl.DotOperandLayout(0, mb, k_width=8)
    b3: gl.constexpr = gl.DotOperandLayout(1, mb, k_width=8)
    mf: gl.constexpr = _wmma_layout(NW, 0, 3, 4)
    af: gl.constexpr = gl.DotOperandLayout(0, mf, k_width=2)
    bf: gl.constexpr = gl.DotOperandLayout(1, mf, k_width=2)
    mma: gl.constexpr = _wmma_layout(NW, 0, 2)
    a_op: gl.constexpr = gl.DotOperandLayout(0, mma, k_width=8)
    b_op: gl.constexpr = gl.DotOperandLayout(1, mma, k_width=8)
    # [NB, D] per-band rows, with the bands in each thread.
    band_layout: gl.constexpr = gl.BlockedLayout([NB, 1], [1, 32], [1, NW], [1, 0])

    # Q, K, the raw gate and V arrive by TDM, which zero-fills rows past the
    # sequence. V stays in LDS as U's right-hand side.
    length = rows_left + token0
    head_base = (begin * H + head) * D
    tile_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[D, 8]], [BT, D], [1, 0]
    )
    g_smem = gl.allocate_shared_memory(raw_g.dtype.element_ty, [BT, D], tile_layout)
    q_smem = gl.allocate_shared_memory(q.dtype.element_ty, [BT, D], tile_layout)
    k_smem = gl.allocate_shared_memory(k.dtype.element_ty, [BT, D], tile_layout)
    value_rhs = gl.allocate_shared_memory(gl.bfloat16, [BT, D], tile_layout)
    cdna5.tdm.async_load(
        cdna5.tdm.make_tensor_descriptor(
            raw_g + head_base, (length, D), (H * D, 1), (BT, D), tile_layout
        ),
        [token0, 0],
        g_smem,
    )
    cdna5.tdm.async_load(
        cdna5.tdm.make_tensor_descriptor(
            q + head_base, (length, D), (H * D, 1), (BT, D), tile_layout
        ),
        [token0, 0],
        q_smem,
    )
    cdna5.tdm.async_load(
        cdna5.tdm.make_tensor_descriptor(
            k + head_base, (length, D), (H * D, 1), (BT, D), tile_layout
        ),
        [token0, 0],
        k_smem,
    )
    cdna5.tdm.async_load(
        cdna5.tdm.make_tensor_descriptor(
            v + head_base, (length, D), (H * D, 1), (BT, D), tile_layout
        ),
        [token0, 0],
        value_rhs,
    )

    # [NB, BC, D] tiles live in the FP32 WMMA result layout, a band per warp:
    # the in-band cumsum is a lower-triangular product, which yields it there.
    # Buffer accesses still form an address on masked-off lanes, so rows past
    # the sequence are clamped into it.
    tile_base = ((begin + token0) * H + head) * D
    row = _band_index(NB, 0, mf) * BC + _band_index(BC, 1, mf)
    tile_offsets = gl.minimum(row, rows_left - 1) * (H * D) + _band_index(D, 2, mf)
    tile_mask = row < rows_left
    bias = gl.load(
        dt_bias
        + head * D
        + gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(1, bf)))
    ).to(gl.float32)[None, None, :]
    rate = gl.exp(gl.load(a_log + head).to(gl.float32))
    beta_row = (
        gl.arange(0, NB, layout=gl.SliceLayout(1, gl.SliceLayout(2, mf)))[:, None] * BC
        + gl.arange(0, BC, layout=gl.SliceLayout(0, gl.SliceLayout(2, mf)))[None, :]
    )
    beta_mask = beta_row < rows_left
    beta = cdna5.buffer_load(
        raw_beta + (begin + token0) * H + head,
        gl.minimum(beta_row, rows_left - 1) * H,
        mask=beta_mask,
        other=0.0,
    ).to(gl.float32)
    beta = gl.where(beta_mask, gl.fdiv(1.0, 1.0 + gl.exp(-beta)), 0.0)
    # The same betas along block columns, which scale the inverse's columns.
    beta_base = raw_beta + (begin + token0) * H + head
    beta0 = _column_beta(beta_base, 0, rows_left, H, BC, mf)
    beta1 = _column_beta(beta_base, 1, rows_left, H, BC, mf)
    beta2 = _column_beta(beta_base, 2, rows_left, H, BC, mf)
    beta3 = _column_beta(beta_base, 3, rows_left, H, BC, mf)

    cdna5.tdm.async_wait(1)
    g_mask = _band_index(NB, 0, bf) * BC + _band_index(BC, 1, bf) < rows_left
    gate = g_smem.reshape([NB, BC, D]).load(bf).to(gl.float32)
    if HAS_LOWER_BOUND:
        slope = -rate * LOG2E
        gate = gl.fdiv(LOWER_BOUND, 1.0 + gl.exp2(gate * slope + slope * bias))
    else:
        x = gate + bias
        gate = -rate * (gl.maximum(x, 0.0) + gl.log(1.0 + gl.exp(-gl.abs(x))))
    gate = gl.where(g_mask, gate, 0.0)

    # G = pivot + in_band, in log2 units: each band's gate relative to the
    # token before it. exp(G) and exp(last - G) are split into a per-band
    # factor and exp(+-in_band), so no FP32 gate tile stays live.
    lower = gl.where(
        (_band_index(BC, 2, af) <= _band_index(BC, 1, af))
        & (_band_index(NB, 0, af) >= 0),
        LOG2E,
        0.0,
    )
    in_band = cdna5.wmma(lower, gate, gl.zeros([NB, BC, D], gl.float32, mf))
    band_total = gl.convert_layout(gl.sum(gate, axis=1), band_layout) * LOG2E
    pivot = gl.cumsum(band_total, 0) - band_total
    last = gl.sum(band_total, axis=0)
    gl.store(
        decay
        + ((begin // BT + sequence + local_chunk) * H + head) * D
        + gl.arange(0, D, layout=gl.SliceLayout(0, band_layout)),
        gl.exp2(last),
    )
    rise = gl.exp2(in_band)
    fall = gl.exp2(-in_band)
    to_pivot = gl.convert_layout(gl.exp2(pivot), gl.SliceLayout(1, mf))[:, None, :]
    to_last = gl.convert_layout(gl.exp2(last[None, :] - pivot), gl.SliceLayout(1, mf))[
        :, None, :
    ]

    # Rows decayed to their band's pivot (lhs) and columns lifted from it; a
    # bridge exp(pivot_b - pivot_J) on row band b's rows meets column band J.
    q_f = q_smem.reshape([NB, BC, D]).load(mf).to(gl.float32)
    q_f = q_f * gl.rsqrt(gl.sum(q_f * q_f, axis=2) + 1e-6)[:, :, None]
    q_rows = q_f * rise * SCALE
    cdna5.buffer_store(
        (q_rows * to_pivot).to(qg.dtype.element_ty),
        qg + tile_base,
        tile_offsets,
        mask=tile_mask,
    )
    q_lhs = gl.convert_layout(q_rows.to(gl.bfloat16), a3)
    k_f = k_smem.reshape([NB, BC, D]).load(mf).to(gl.float32)
    k_f = k_f * gl.rsqrt(gl.sum(k_f * k_f, axis=2) + 1e-6)[:, :, None]
    k_cols = k_f * fall
    cdna5.buffer_store(
        (k_cols * to_last).to(kg.dtype.element_ty),
        kg + tile_base,
        tile_offsets,
        mask=tile_mask,
    )
    # Q and K are in registers now, so their LDS holds the lifted columns and
    # W's right-hand side.
    lifted = k_smem
    lifted.store(gl.reshape(k_cols.to(gl.bfloat16), [BT, D]))
    k_rows = k_f * rise
    key_rhs = q_smem
    key_rhs.store(gl.reshape((k_rows * to_pivot).to(gl.bfloat16), [BT, D]))
    k_lhs = gl.convert_layout(k_rows.to(gl.bfloat16), a3)

    block_row = _band_index(NB, 0, mb) * BC + _band_index(BC, 1, mb)
    block_col = _band_index(BC, 2, mb)
    block_beta = gl.convert_layout(beta, gl.SliceLayout(2, mb))[:, :, None]
    band_mb = _band_index(NB, 0, mb)
    aqk_base = ((begin + token0) * H + head) * BT
    aqk_offsets = gl.minimum(block_row, rows_left - 1) * (H * BT) + block_col
    aqk_mask = block_row < rows_left
    pivot_band = gl.arange(0, NB, layout=gl.SliceLayout(1, band_layout))[:, None]
    n0 = gl.zeros([NB, BC, BC], gl.float32, mb)
    n1 = gl.zeros([NB, BC, BC], gl.float32, mb)
    n2 = gl.zeros([NB, BC, BC], gl.float32, mb)
    n3 = gl.zeros([NB, BC, BC], gl.float32, mb)
    for J in gl.static_range(NB):
        pivot_J = gl.sum(gl.where(pivot_band == J, pivot, 0.0), axis=0)
        # Clamped where b < J, whose blocks are masked anyway. The bridge
        # scales the shared column operand, once for both products.
        bridge = gl.exp2(gl.minimum(pivot - pivot_J[None, :], 0.0))
        bridge = gl.convert_layout(bridge, gl.SliceLayout(2, b3))[:, :, None]
        column_band = (
            lifted.slice(J * BC, BC, dim=0).permute([1, 0]).load(gl.SliceLayout(0, b3))
        )
        rhs = (column_band.to(gl.float32)[None, :, :] * bridge).to(gl.bfloat16)
        akk = cdna5.wmma(k_lhs, rhs, gl.zeros([NB, BC, BC], gl.float32, mb))
        aqk_block = cdna5.wmma(q_lhs, rhs, gl.zeros([NB, BC, BC], gl.float32, mb))
        column = J * BC + block_col
        cdna5.buffer_store(
            gl.where(block_row >= column, aqk_block, 0.0).to(aqk.dtype.element_ty),
            aqk + aqk_base + J * BC,
            aqk_offsets,
            mask=aqk_mask,
        )
        akk = gl.where(block_row > column, -block_beta * akk, 0.0)
        n0 = gl.where(band_mb == J, akk, n0)
        n1 = gl.where(band_mb == J + 1, akk, n1)
        n2 = gl.where(band_mb == J + 2, akk, n2)
        n3 = gl.where(band_mb == J + 3, akk, n3)
        # Overlapping the bands' products would cost occupancy.
        gl.amd.hint.sched_barrier()

    # Diagonal blocks: (I - N)^-1 = (I + N)(I + N^2)(I + N^4)(I + N^8).
    n0 = gl.convert_layout(n0, mf)
    power = _mm_f32(n0, n0, mf, af, bf)
    x0 = n0 + gl.where(_band_index(BC, 1, mf) == _band_index(BC, 2, mf), 1.0, 0.0)
    x0 += _mm_f32(x0, power, mf, af, bf)
    power = _mm_f32(power, power, mf, af, bf)
    x0 += _mm_f32(x0, power, mf, af, bf)
    power = _mm_f32(power, power, mf, af, bf)
    x0 += _mm_f32(x0, power, mf, af, bf)

    # Off-diagonals X_{b, b-d} = X_bb sum_s N_{b, b-s} X_{b-s, b-d}, with
    # X_{b-s, b-d} diagonal d - s shifted down s bands. N_{b, b-s} is zero for
    # b < s, so so is diagonal d below band d.
    n1 = gl.convert_layout(n1, af)
    n2 = gl.convert_layout(n2, af)
    n3 = gl.convert_layout(n3, af)
    diagonal_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    diagonal0 = gl.allocate_shared_memory(gl.float32, [BT, BC], diagonal_layout)
    diagonal1 = gl.allocate_shared_memory(gl.float32, [BT, BC], diagonal_layout)
    diagonal2 = gl.allocate_shared_memory(gl.float32, [BT, BC], diagonal_layout)
    diagonal0.store(gl.reshape(x0, [BT, BC]))
    zero = gl.zeros([NB, BC, BC], gl.float32, mf)
    acc = cdna5.wmma(n1, _shift_bands(diagonal0, 1, NB, BC, bf), zero)
    x1 = _mm_f32(x0, acc, mf, af, bf)
    diagonal1.store(gl.reshape(x1, [BT, BC]))
    acc = cdna5.wmma(n1, _shift_bands(diagonal1, 1, NB, BC, bf), zero)
    acc = cdna5.wmma(n2, _shift_bands(diagonal0, 2, NB, BC, bf), acc)
    x2 = _mm_f32(x0, acc, mf, af, bf)
    diagonal2.store(gl.reshape(x2, [BT, BC]))
    acc = cdna5.wmma(n1, _shift_bands(diagonal2, 1, NB, BC, bf), zero)
    acc = cdna5.wmma(n2, _shift_bands(diagonal1, 2, NB, BC, bf), acc)
    acc = cdna5.wmma(n3, _shift_bands(diagonal0, 3, NB, BC, bf), acc)
    x3 = _mm_f32(x0, acc, mf, af, bf)

    # The inverse by column band, times diag(beta): band b of column band J is
    # diagonal b - J. Then W and U are plain [BT, BT] x [BT, D] products.
    inverse = gl.allocate_shared_memory(
        gl.bfloat16,
        [BT, BT],
        gl.PaddedSharedLayout.with_identity_for([[BT, 8]], [BT, BT], [1, 0]),
    )
    selector = _band_index(NB, 0, mf)
    for J in gl.static_range(NB):
        block = gl.where(selector == J, x0, 0.0)
        block = gl.where(selector == J + 1, x1, block)
        block = gl.where(selector == J + 2, x2, block)
        block = gl.where(selector == J + 3, x3, block)
        if J == 0:
            block = block * beta0
        elif J == 1:
            block = block * beta1
        elif J == 2:
            block = block * beta2
        else:
            block = block * beta3
        inverse.slice(J * BC, BC, dim=1).store(
            gl.reshape(block.to(gl.bfloat16), [BT, BC])
        )
    # A column slice at a time keeps the B operands small. Each lane's results
    # are 8 consecutive columns, so they store as they are.
    cdna5.tdm.async_wait(0)
    inverse_a = inverse.load(a_op)
    BN: gl.constexpr = 32
    flat_rows = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))[:, None]
    flat_offsets = (
        gl.minimum(flat_rows, rows_left - 1) * (H * D)
        + gl.arange(0, BN, layout=gl.SliceLayout(0, mma))[None, :]
    )
    flat_mask = flat_rows < rows_left
    zero_flat = gl.zeros([BT, BN], gl.float32, mma)
    for n in gl.static_range(D // BN):
        w_tile = cdna5.wmma(
            inverse_a, key_rhs.slice(n * BN, BN, dim=1).load(b_op), zero_flat
        )
        cdna5.buffer_store(
            w_tile.to(w.dtype.element_ty),
            w + tile_base + n * BN,
            flat_offsets,
            mask=flat_mask,
        )
        u_tile = cdna5.wmma(
            inverse_a, value_rhs.slice(n * BN, BN, dim=1).load(b_op), zero_flat
        )
        cdna5.buffer_store(
            u_tile.to(u.dtype.element_ty),
            u + tile_base + n * BN,
            flat_offsets,
            mask=flat_mask,
        )


@gluon.jit
def _decay_row(
    begin, sequence, head, H: gl.constexpr, K: gl.constexpr, BT: gl.constexpr
):
    """Offset of a sequence's first chunk row in the per-chunk decay table.

    begin // BT + sequence + local_chunk is distinct for every chunk of every
    sequence and stays below T // BT + 2 * N, so the table needs no chunk
    offsets.
    """
    return ((begin // BT + sequence) * H + head) * K


@gluon.constexpr_function
def _walk_ring_layout(rows, cols):
    """Padded LDS image of a [rows, cols] bf16 ring slot."""
    return gl.PaddedSharedLayout.with_identity_for([[cols, 8]], [rows, cols], [1, 0])


@gluon.jit
def _walk_issue(
    w_desc,
    u_desc,
    kg_desc,
    q_desc,
    a_desc,
    w_ring,
    u_ring,
    kg_ring,
    q_ring,
    a_ring,
    slot,
    token0,
    value0,
    WRITE_OUTPUT: gl.constexpr,
):
    """Start one chunk's TDM copies into ring ``slot``.

    The descriptors end at the sequence (and U's at V), and TDM zero-fills
    past them, so the walk needs no row masks and transfer rows read no U.
    """
    cdna5.tdm.async_load(w_desc, [token0, 0], w_ring.index(slot))
    cdna5.tdm.async_load(kg_desc, [token0, 0], kg_ring.index(slot))
    if WRITE_OUTPUT:
        cdna5.tdm.async_load(q_desc, [token0, 0], q_ring.index(slot))
        cdna5.tdm.async_load(a_desc, [token0, 0], a_ring.index(slot))
    cdna5.tdm.async_load(u_desc, [token0, value0], u_ring.index(slot))


@gluon.jit
def _walk_chunks(
    state,
    w,
    u,
    kg,
    qg,
    aqk,
    decay,
    output,
    begin,
    length,
    sequence,
    head,
    chunk_lo,
    chunk_hi,
    value0,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    WRITE_OUTPUT: gl.constexpr,
):
    """Carry ``BV`` state rows from ``value0`` across chunks [chunk_lo, chunk_hi).

    Per chunk, with S the [BV, K] rows:

    ``v = U^T - S W^T``
    ``o = Qg S^T + Aqk v^T``   (WRITE_OUTPUT)
    ``S = exp(bg_last) S + v Kg``

    Each warp owns 16 whole state rows, so S and v stay in registers, where
    they are register-trivial conversions of the WMMA result. W, Kg, U, Qg
    and Aqk arrive by TDM through a two-slot LDS ring.
    """
    state_layout: gl.constexpr = _wmma_layout(NW, 0, 2)
    a_op: gl.constexpr = gl.DotOperandLayout(0, state_layout, k_width=8)
    b_op: gl.constexpr = gl.DotOperandLayout(1, state_layout, k_width=8)
    # The output, token-major with warps along value columns, so its stores
    # run along V and S^T and v^T are register-trivial B operands.
    out_layout: gl.constexpr = _wmma_layout(NW, 1, 2)
    out_a: gl.constexpr = gl.DotOperandLayout(0, out_layout, k_width=8)
    out_b: gl.constexpr = gl.DotOperandLayout(1, out_layout, k_width=8)
    key_layout: gl.constexpr = _walk_ring_layout(BT, K)
    u_layout: gl.constexpr = _walk_ring_layout(BT, BV)
    key_base = (begin * H + head) * K
    value_base = (begin * H + head) * V
    w_desc = cdna5.tdm.make_tensor_descriptor(
        w + key_base, (length, K), (H * K, 1), (BT, K), key_layout
    )
    kg_desc = cdna5.tdm.make_tensor_descriptor(
        kg + key_base, (length, K), (H * K, 1), (BT, K), key_layout
    )
    u_desc = cdna5.tdm.make_tensor_descriptor(
        u + value_base, (length, V), (H * V, 1), (BT, BV), u_layout
    )
    w_ring = gl.allocate_shared_memory(gl.bfloat16, [2, BT, K], key_layout)
    kg_ring = gl.allocate_shared_memory(gl.bfloat16, [2, BT, K], key_layout)
    u_ring = gl.allocate_shared_memory(gl.bfloat16, [2, BT, BV], u_layout)
    if WRITE_OUTPUT:
        a_layout: gl.constexpr = _walk_ring_layout(BT, BT)
        q_desc = cdna5.tdm.make_tensor_descriptor(
            qg + key_base, (length, K), (H * K, 1), (BT, K), key_layout
        )
        a_desc = cdna5.tdm.make_tensor_descriptor(
            aqk + (begin * H + head) * BT, (length, BT), (H * BT, 1), (BT, BT), a_layout
        )
        q_ring = gl.allocate_shared_memory(gl.bfloat16, [2, BT, K], key_layout)
        a_ring = gl.allocate_shared_memory(gl.bfloat16, [2, BT, BT], a_layout)
    else:
        q_desc = w_desc
        a_desc = w_desc
        q_ring = w_ring
        a_ring = w_ring

    keys = gl.arange(0, K, layout=gl.SliceLayout(0, state_layout))
    out_rows = gl.arange(0, BT, layout=gl.SliceLayout(1, out_layout))
    out_cols = gl.arange(0, BV, layout=gl.SliceLayout(0, out_layout))
    decay_base = _decay_row(begin, sequence, head, H, K, BT)

    if chunk_lo < chunk_hi:
        _walk_issue(
            w_desc,
            u_desc,
            kg_desc,
            q_desc,
            a_desc,
            w_ring,
            u_ring,
            kg_ring,
            q_ring,
            a_ring,
            0,
            chunk_lo * BT,
            value0,
            WRITE_OUTPUT,
        )
    chunk_decay = cdna5.buffer_load(decay + decay_base, chunk_lo * (H * K) + keys)
    for local_chunk in range(chunk_lo, chunk_hi):
        token0 = local_chunk * BT
        slot = (local_chunk - chunk_lo) % 2
        # Wait before issuing the next chunk, so the wait never covers it.
        cdna5.tdm.async_wait(0)
        if local_chunk + 1 < chunk_hi:
            _walk_issue(
                w_desc,
                u_desc,
                kg_desc,
                q_desc,
                a_desc,
                w_ring,
                u_ring,
                kg_ring,
                q_ring,
                a_ring,
                1 - slot,
                token0 + BT,
                value0,
                WRITE_OUTPUT,
            )
        # The row after a sequence's last chunk is still inside the table.
        next_decay = cdna5.buffer_load(
            decay + decay_base, (local_chunk + 1) * (H * K) + keys
        )

        state_bf16 = state.to(gl.bfloat16)
        state_a = gl.convert_layout(state_bf16, a_op)
        w_t = w_ring.index(slot).permute([1, 0]).load(b_op)
        zero = gl.zeros([BV, BT], gl.float32, state_layout)
        u_t = u_ring.index(slot).permute([1, 0]).load(state_layout)
        v = (u_t.to(gl.float32) - cdna5.wmma(state_a, w_t, zero)).to(gl.bfloat16)
        v_a = gl.convert_layout(v, a_op)
        if WRITE_OUTPUT:
            # o = Qg S^T + Aqk v^T; the producer zero-fills Aqk above the diagonal.
            out = cdna5.wmma(
                q_ring.index(slot).load(out_a),
                gl.convert_layout(gl.permute(state_bf16, [1, 0]), out_b),
                gl.zeros([BT, BV], gl.float32, out_layout),
            )
            out = cdna5.wmma(
                a_ring.index(slot).load(out_a),
                gl.convert_layout(gl.permute(v, [1, 0]), out_b),
                out,
            )
            # Buffer stores still form an address on masked-off lanes.
            out_token = gl.minimum(out_rows[:, None], length - token0 - 1) + token0
            cdna5.buffer_store(
                out.to(output.dtype.element_ty),
                output + value_base,
                out_token * (H * V) + value0 + out_cols[None, :],
                mask=out_rows[:, None] < length - token0,
            )
        kg_tile = kg_ring.index(slot).load(b_op)
        state = cdna5.wmma(v_a, kg_tile, state * chunk_decay[None, :])
        chunk_decay = next_decay
    return state


@gluon.jit
def _state_rows(
    value0,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
):
    """Offsets of rows [value0, value0 + BV) of a [V, K] state and their in-range mask."""
    state_layout: gl.constexpr = _wmma_layout(NW, 0, 2)
    rows = value0 + gl.arange(0, BV, layout=gl.SliceLayout(1, state_layout))
    keys = gl.arange(0, K, layout=gl.SliceLayout(0, state_layout))
    offsets = rows[:, None] * K + keys[None, :]
    return offsets, (rows[:, None] < V) & (keys[None, :] < K), rows, keys


@gluon.jit
def _walk_group_chunks(num_chunks, num_groups, group):
    """Chunks [lo, hi) of one time group: near-equal runs, the first ones a chunk longer."""
    base = num_chunks // num_groups
    extra = num_chunks % num_groups
    lo = group * base + gl.minimum(group, extra)
    hi = lo + base + (group < extra).to(gl.int32)
    return lo, hi


@gluon.jit
def gluon_kda_paged_prefill_state_walk_gfx1250(
    w,
    u,
    kg,
    qg,
    aqk,
    decay,
    initial_state,
    output,
    final_state,
    group_entry,
    cu_seqlens,
    num_groups,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    FROM_GROUP_ENTRY: gl.constexpr,
):
    """Walk one (row block, sequence, head, time group) and write its output.

    Unsplit, the single group walks from the initial state. Split, each group
    starts from its ``group_entry`` state, which the summary and chain kernels
    produce. The last group writes the final state.
    """
    value0 = gl.program_id(0) * BV
    sequence_head = gl.program_id(1)
    group = gl.program_id(2)
    sequence = sequence_head // H
    head = sequence_head % H
    begin = gl.load(cu_seqlens + sequence).to(gl.int32)
    length = gl.load(cu_seqlens + sequence + 1).to(gl.int32) - begin
    chunk_lo, chunk_hi = _walk_group_chunks(gl.cdiv(length, BT), num_groups, group)
    offsets, mask, _, _ = _state_rows(value0, K, V, BV, NW)
    plane = sequence_head.to(gl.int64) * (V * K)
    if FROM_GROUP_ENTRY:
        start = (
            group_entry + (group * gl.num_programs(1)).to(gl.int64) * (V * K) + plane
        )
    else:
        start = initial_state + plane
    state = cdna5.buffer_load(start, offsets, mask=mask, other=0.0)
    state = _walk_chunks(
        state,
        w,
        u,
        kg,
        qg,
        aqk,
        decay,
        output,
        begin,
        length,
        sequence,
        head,
        chunk_lo,
        chunk_hi,
        value0,
        H,
        K,
        V,
        BT,
        BV,
        NW,
        True,
    )
    if group == num_groups - 1:
        cdna5.buffer_store(state, final_state + plane, offsets, mask=mask)


@gluon.jit
def gluon_kda_paged_prefill_group_summary_gfx1250(
    w,
    u,
    kg,
    decay,
    initial_state,
    group_entry,
    group_b,
    group_m,
    cu_seqlens,
    num_groups,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
):
    """Summarize one time group of a split walk as ``S_out = S_in M + B``.

    A chunk maps the state affinely, ``S' = S (D - W^T Kg) + U^T Kg``, so a
    run of chunks composes to ``S_in M + B``. Rows [0, V) start at zero and
    give B; transfer rows [V, V + K) start as the identity, read no U, and
    give M. Group 0 knows its entry, the initial state: it copies it to
    ``group_entry[0]`` and walks it, value rows only, into ``group_entry[1]``.
    The last group needs no summary and is not launched.
    """
    value0 = gl.program_id(0) * BV
    sequence_head = gl.program_id(1)
    group = gl.program_id(2)
    if group == 0 and value0 >= V:
        return
    sequence = sequence_head // H
    head = sequence_head % H
    begin = gl.load(cu_seqlens + sequence).to(gl.int32)
    length = gl.load(cu_seqlens + sequence + 1).to(gl.int32) - begin
    chunk_lo, chunk_hi = _walk_group_chunks(gl.cdiv(length, BT), num_groups, group)
    offsets, mask, rows, keys = _state_rows(value0, K, V, BV, NW)
    planes = gl.num_programs(1)
    plane = sequence_head.to(gl.int64) * (V * K)
    group_plane = (group * planes).to(gl.int64) * (V * K) + plane
    if group == 0:
        state = cdna5.buffer_load(initial_state + plane, offsets, mask=mask, other=0.0)
        cdna5.buffer_store(state, group_entry + group_plane, offsets, mask=mask)
    else:
        state = gl.where(rows[:, None] - V == keys[None, :], 1.0, 0.0)
    state = _walk_chunks(
        state,
        w,
        u,
        kg,
        w,
        w,
        decay,
        w,
        begin,
        length,
        sequence,
        head,
        chunk_lo,
        chunk_hi,
        value0,
        H,
        K,
        V,
        BT,
        BV,
        NW,
        False,
    )
    if group == 0:
        dest = group_entry + planes.to(gl.int64) * (V * K) + group_plane
    else:
        dest = group_b + group_plane
    cdna5.buffer_store(state, dest, offsets, mask=mask)
    if group != 0:
        cdna5.buffer_store(
            state.to(group_m.dtype.element_ty),
            group_m + (group * planes + sequence_head).to(gl.int64) * (K * K),
            offsets - V * K,
            mask=(rows[:, None] >= V) & (keys[None, :] < K),
        )


@gluon.jit
def gluon_kda_paged_prefill_chain_groups_gfx1250(
    group_entry,
    group_b,
    group_m,
    num_groups,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
):
    """Chain group summaries into entry states: ``entry[g + 1] = entry[g] M_g + B_g``.

    M and B arrive by TDM through an LDS ring a few groups ahead, since a
    step is short next to a load. M's B-operand reads run down columns,
    which global loads would split into 2-byte accesses.
    """
    NW: gl.constexpr = 4
    RING: gl.constexpr = 4
    layout: gl.constexpr = _wmma_layout(NW, 1, 2)
    a_op: gl.constexpr = gl.DotOperandLayout(0, layout, k_width=8)
    b_op: gl.constexpr = gl.DotOperandLayout(1, layout, k_width=8)
    m_layout: gl.constexpr = _walk_ring_layout(K, K)
    b_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[K, 4]], [BV, K], [1, 0]
    )
    m_ring = gl.allocate_shared_memory(gl.bfloat16, [RING, K, K], m_layout)
    b_ring = gl.allocate_shared_memory(gl.float32, [RING, BV, K], b_layout)
    value0 = gl.program_id(0) * BV
    sequence_head = gl.program_id(1)
    planes = gl.num_programs(1)
    # Prefetches past the last group read rows past the table, which TDM
    # zero-fills.
    m_desc = cdna5.tdm.make_tensor_descriptor(
        group_m, (num_groups * planes * K, K), (K, 1), (K, K), m_layout
    )
    b_desc = cdna5.tdm.make_tensor_descriptor(
        group_b, (num_groups * planes * V, K), (K, 1), (BV, K), b_layout
    )
    rows = value0 + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    keys = gl.arange(0, K, layout=gl.SliceLayout(0, layout))
    state_offsets = rows[:, None] * K + keys[None, :]
    state = cdna5.buffer_load(
        group_entry + (planes + sequence_head).to(gl.int64) * (V * K), state_offsets
    )
    for lead in gl.static_range(RING - 1):
        plane = (lead + 1) * planes + sequence_head
        cdna5.tdm.async_load(m_desc, [plane * K, 0], m_ring.index(lead))
        cdna5.tdm.async_load(b_desc, [plane * V + value0, 0], b_ring.index(lead))
    for group in range(1, num_groups - 1):
        slot = (group - 1) % RING
        # Wait for this group's pair only; later groups stay in flight.
        cdna5.tdm.async_wait(2 * (RING - 2))
        ahead = (group + RING - 1) * planes + sequence_head
        cdna5.tdm.async_load(
            m_desc, [ahead * K, 0], m_ring.index((slot + RING - 1) % RING)
        )
        cdna5.tdm.async_load(
            b_desc, [ahead * V + value0, 0], b_ring.index((slot + RING - 1) % RING)
        )
        state = cdna5.wmma(
            gl.convert_layout(state.to(gl.bfloat16), a_op),
            m_ring.index(slot).load(b_op),
            b_ring.index(slot).load(layout),
        )
        next_plane = ((group + 1) * planes + sequence_head).to(gl.int64) * (V * K)
        cdna5.buffer_store(state, group_entry + next_plane, state_offsets)


def launch_gluon_kda_paged_prefill_gfx1250(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run chunk-parallel KDA prefill on packed GFX1250 inputs.

    Args:
        q: Packed queries with shape ``[1,T,H,K]``.
        k: Packed keys with shape ``[1,T,H,K]``.
        v: Packed values with shape ``[1,T,H,V]``.
        g_raw: Raw per-key-channel gates with shape ``[1,T,H,K]``.
        beta_logits: Raw delta coefficients with shape ``[1,T,H]``.
        A_log: Per-head log decay with shape ``[H]``.
        dt_bias: Per-head, per-key-channel gate bias with shape ``[H,K]``.
        initial_state: Initial V-major state ``[N,H,V,K]`` (value-major,
            matching the gfx1250 decode recurrent-state pool layout).
        cu_seqlens: Packed-sequence prefix sums with shape ``[N+1]``.
        lower_bound: Optional lower bound used by the safe decay gate.
        cu_seqlens_cpu: Host copy of ``cu_seqlens``, from which the chunk
            schedule is planned once per forward. Without it, ``cu_seqlens``
            is read back on every call.

    Returns:
        The packed output ``[1,T,H,V]`` and final state ``[N,H,V,K]``.
    """
    tensors = (q, k, v, g_raw, beta_logits, A_log, dt_bias, initial_state)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx1250 Gluon KDA prefill requires GPU tensors")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("gfx1250 Gluon KDA prefill requires packed [1,T,H,K] inputs")
    if q.shape != k.shape or q.shape != g_raw.shape:
        raise ValueError("q, k, and raw_g must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q through the head dimension")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError("beta_logits must have shape [1,T,H]")
    if initial_state.ndim != 4:
        raise ValueError("initial_state must use the V-major [N,H,V,K] layout")

    q = q[0].contiguous()
    k = k[0].contiguous()
    v = v[0].contiguous()
    g_raw = g_raw[0].contiguous()
    beta_logits = beta_logits[0].contiguous()
    heads, key_dim = q.shape[1:]
    value_dim = v.shape[-1]
    if initial_state.shape[1:] != (heads, value_dim, key_dim):
        raise ValueError("initial_state must have shape [N,H,V,K]")
    if key_dim != 128 or value_dim != 128:
        raise ValueError("gfx1250 Gluon KDA prefill currently specializes K=V=128")
    cu_seqlens = cu_seqlens.to(device=q.device).contiguous()
    if cu_seqlens.numel() - 1 != initial_state.shape[0]:
        raise ValueError("cu_seqlens and initial_state must describe the same batch")
    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    elif cu_seqlens_cpu.shape != cu_seqlens.shape:
        raise ValueError("cu_seqlens_cpu must mirror cu_seqlens")

    chunk_size = _CHUNK_SIZE
    total_tokens = q.shape[0]
    chunk_indices = _chunk_indices(cu_seqlens_cpu, chunk_size, q.device)
    num_chunks = chunk_indices.shape[0]
    num_sequences = cu_seqlens.numel() - 1

    qg = torch.empty_like(q, dtype=torch.bfloat16)
    w = torch.empty_like(k, dtype=torch.bfloat16)
    kg = torch.empty_like(k, dtype=torch.bfloat16)
    u = torch.empty_like(v, dtype=torch.bfloat16)
    aqk = torch.empty(
        total_tokens, heads, chunk_size, device=q.device, dtype=torch.bfloat16
    )
    # Each chunk's exp(bg_last), one row per chunk at the rows _decay_row assigns.
    decay = torch.empty(
        total_tokens // chunk_size + 2 * num_sequences,
        heads,
        key_dim,
        device=q.device,
        dtype=torch.float32,
    )
    gluon_kda_paged_prefill_prepare_gfx1250[(num_chunks, heads)](
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log.contiguous(),
        dt_bias.view(heads, key_dim).contiguous(),
        qg,
        w,
        u,
        kg,
        aqk,
        decay,
        cu_seqlens,
        chunk_indices,
        key_dim**-0.5,
        H=heads,
        D=key_dim,
        BT=chunk_size,
        BC=_SUBCHUNK_SIZE,
        NW=_PREPARE_WARPS,
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=_PREPARE_WARPS,
    )
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    _launch_state_walk(
        w=w,
        u=u,
        kg=kg,
        qg=qg,
        aqk=aqk,
        decay=decay,
        initial_state=initial_state.contiguous(),
        output=output,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        num_sequences=num_sequences,
        heads=heads,
        num_chunks=num_chunks,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=chunk_size,
    )
    return output.unsqueeze(0), final_state


@functools.cache
def _num_cus(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# The latest plan: [cu_seqlens_cpu, device, chunk_size, chunk_indices].
_latest_plan: list = []


def _chunk_indices(
    cu_seqlens_cpu: torch.Tensor, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """``[chunks, 2]`` int32 ``(sequence, local_chunk)`` pairs on ``device``.

    Built on the host from ``cu_seqlens_cpu`` and uploaded without blocking.
    The metadata builder makes the host bounds fresh per batch, never writes
    them in place, and hands the same tensor to every layer of a forward, so
    the latest plan is reused by tensor identity.
    """
    if cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.dim() != 1:
        raise ValueError("cu_seqlens_cpu must be a 1-D host tensor")
    device = torch.device(device)
    if _latest_plan:
        bounds, plan_device, plan_chunk_size, indices = _latest_plan
        if (
            bounds is cu_seqlens_cpu
            and plan_device == device
            and plan_chunk_size == chunk_size
        ):
            return indices
    bounds = cu_seqlens_cpu.to(torch.int64)
    chunks = (bounds[1:] - bounds[:-1] + chunk_size - 1) // chunk_size
    sequence = torch.repeat_interleave(torch.arange(chunks.numel()), chunks)
    first = torch.repeat_interleave(torch.cumsum(chunks, 0) - chunks, chunks)
    local = torch.arange(sequence.numel()) - first
    host = torch.stack([sequence, local], 1).to(torch.int32).pin_memory()
    indices = host.to(device, non_blocking=True)
    _latest_plan[:] = [cu_seqlens_cpu, device, chunk_size, indices]
    return indices


def _scan_groups(
    num_sequences: int, heads: int, num_chunks: int, block_value: int, value_dim: int
) -> int:
    """Time groups per sequence, so few long sequences still fill the GPU."""
    chunks_per_sequence = num_chunks // max(num_sequences, 1)
    planes = num_sequences * heads * (value_dim // block_value)
    cus = _num_cus(torch.cuda.current_device())
    groups = min(cus // planes, chunks_per_sequence // 4, 32)
    # Short sequences whose unsplit walk already fills the GPU skip the split.
    unsplit_fills = num_sequences * heads * (value_dim // 32) >= 3 * cus // 4
    if (
        chunks_per_sequence < 24
        or groups < 4
        or (unsplit_fills and chunks_per_sequence < 48)
    ):
        return 1
    return groups


def _launch_state_walk(
    *,
    w: torch.Tensor,
    u: torch.Tensor,
    kg: torch.Tensor,
    qg: torch.Tensor,
    aqk: torch.Tensor,
    decay: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_sequences: int,
    heads: int,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
    chunk_size: int,
) -> None:
    """Walk the chunk recurrence into ``output`` and ``final_state``.

    When too few sequences would idle the GPU, each sequence is split into
    time groups: every group is summarized at once, the summaries are chained
    into each group's entry state, and every group is then walked at once.
    """
    planes = num_sequences * heads
    cus = _num_cus(torch.cuda.current_device())
    # When the split runs out of chunks before CUs, its walk takes half-width
    # row blocks, which walk a chunk faster.
    chunk_limited = num_chunks // max(num_sequences, 1) // 4 < cus // planes
    split_block = value_dim // 2 if chunk_limited else value_dim
    groups = _scan_groups(num_sequences, heads, num_chunks, split_block, value_dim)
    common = dict(
        H=heads, K=key_dim, V=value_dim, BT=chunk_size, num_stages=1, waves_per_eu=1
    )
    walk_args = (w, u, kg, qg, aqk, decay, initial_state, output, final_state)

    def walk(block: int, warps: int, group_entry: torch.Tensor) -> None:
        gluon_kda_paged_prefill_state_walk_gfx1250[
            (value_dim // block, planes, groups)
        ](
            *walk_args,
            group_entry,
            cu_seqlens,
            groups,
            BV=block,
            NW=warps,
            FROM_GROUP_ENTRY=groups > 1,
            num_warps=warps,
            # The backend defaults to coexec only up to four warps.
            llvm_fn_attrs="amdgpu-sched-strategy=coexec",
            **common,
        )

    if groups == 1:
        # The narrowest row block that still keeps every program on its own
        # CU, with a warp per 16 rows.
        block = value_dim
        for width in (32, 64):
            if planes * (value_dim // width) <= cus:
                block = width
                break
        walk(block, max(block // 16, 2), initial_state)
        return
    group_entry = torch.empty(
        (groups, planes, value_dim, key_dim),
        dtype=torch.float32,
        device=initial_state.device,
    )
    group_b = torch.empty_like(group_entry)
    group_m = torch.empty(
        (groups, planes, key_dim, key_dim),
        dtype=torch.bfloat16,
        device=initial_state.device,
    )
    # A summary walks value rows and transfer rows; it halves its row block
    # while that still fits on the CUs.
    summary_block = value_dim + key_dim
    if 2 * planes * (groups - 1) <= cus:
        summary_block //= 2
    gluon_kda_paged_prefill_group_summary_gfx1250[
        ((value_dim + key_dim) // summary_block, planes, groups - 1)
    ](
        w,
        u,
        kg,
        decay,
        initial_state,
        group_entry,
        group_b,
        group_m,
        cu_seqlens,
        groups,
        BV=summary_block,
        NW=4,
        num_warps=4,
        **common,
    )
    if groups > 2:
        gluon_kda_paged_prefill_chain_groups_gfx1250[(value_dim // 16, planes)](
            group_entry,
            group_b,
            group_m,
            groups,
            K=key_dim,
            V=value_dim,
            BV=16,
            num_warps=4,
        )
    walk(split_block, 8 if split_block == value_dim else 4, group_entry)


__all__ = ["launch_gluon_kda_paged_prefill_gfx1250"]
