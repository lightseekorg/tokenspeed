# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Chunk-parallel Kimi Delta Attention prefill.

The production ``kda_recurrent`` runs a serial token-by-token scan that is
optimal for decode but leaves the machine idle during long prefills. This module
provides a chunk-parallel prefill built on the gated delta-rule (WY) chunking
scheme (Yang et al., DeltaNet-2). For each chunk the intra-chunk work is a set of
matmuls (parallel across all chunks); only the inter-chunk state carry stays
sequential:

    bg = cumsum(g)                              # per-channel gate
    A  = tril(-diag(beta) . Kd . Ki^T, -1)      # state-independent
    T  = (I - A)^{-1}                           # solve_tril
    u  = T . (beta.V) ,   W = T . (beta.Kd)     # parallel over chunks
    -- sequential over chunks --
    v_new = u - W . H
    o     = Qd . H + tril(Qd Ki^T, 0) . v_new
    H     = exp(bg_last) . H + (Kn.exp(bg_last-bg))^T . v_new

with Kd = Kn.e^{bg}, Ki = Kn.e^{-bg}, Qd = scale . Qn.e^{bg}. Chunk-local
exponentials that couple two tokens are formed as (bg_a - bg_b) with the larger
term subtracted, so the exponent stays <= 0.

Variable-length prefill: multiple requests are packed into one flat token buffer
and delimited by ``cu_seqlens`` (a prefix-sum of per-sequence lengths). Following
the repo's GDN chunk kernels, ``prepare_chunk_indices`` maps each global chunk to
its ``(sequence, local-chunk)`` so chunks never span a sequence boundary, the
chunk-local cumsum resets per sequence, and the sequential scan restarts ``H``
from each sequence's own initial state. A single sequence is just the ``N = 1``
case (``cu_seqlens = [0, T]``).

The math is validated against the serial recurrence in the kernel test suite.
The factored WY/output steps form a raw ``e^{-bg}`` term that would
overflow fp32 for long chunks, so the KKt/output kernels sub-chunk
each chunk into ``BC`` rows referenced to the sub-chunk start to bound it.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.linear.cumsum import (
    chunk_local_cumsum_vector,
)
from tokenspeed_kernel.ops.attention.triton.linear.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from tokenspeed_kernel.ops.attention.triton.linear.l2norm import l2norm_fwd
from tokenspeed_kernel.ops.attention.triton.linear.solve_tril import solve_tril

__all__ = ["kda_chunk_prefill"]


@triton.jit
def _kda_prepare_gate_beta_kernel(
    raw_g,
    raw_beta,
    a_log,
    dt_bias,
    gate,
    beta,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_LOWER_BOUND: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    linear = (token_idx * H + head_idx) * D + offsets
    x = tl.load(raw_g + linear, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(
        dt_bias + head_idx * D + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x += bias
    a = tl.load(a_log + head_idx).to(tl.float32)
    if HAS_LOWER_BOUND:
        g = LOWER_BOUND * tl.sigmoid(tl.exp(a) * x)
    else:
        softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        g = -tl.exp(a) * softplus
    tl.store(gate + linear, g, mask=mask)

    raw_b = tl.load(raw_beta + token_idx * H + head_idx).to(tl.float32)
    tl.store(beta + token_idx * H + head_idx, tl.sigmoid(raw_b))


@triton.jit(do_not_specialize=["T"])
def _kkt_vector_fwd_kernel(
    kn,
    bg,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Strictly-lower KKt score ``A[i,j] = beta_i . <Kn_i e^{bg_i}, Kn_j e^{-bg_j}>``.

    Two-level tiled: each ``BC``-row sub-chunk references both operands
    to its first row ``R``, so the decay exponents stay bounded (row ``bg_i-R<=0``;
    col ``R-bg_j`` up to ``BC*|lb|``) while the product is exactly ``e^{bg_i-bg_j}``.
    """
    i_c, i_h = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    o_c = tl.arange(0, BC)
    NSUB: tl.constexpr = BT // BC
    base = (bos * H + i_h) * K

    for ri in range(NSUB):
        row0 = i_t * BT + ri * BC
        # per-channel reference: gate cumsum at this sub-chunk's first row
        p_R = tl.make_block_ptr(
            bg + base, (T, K), (H * K, 1), (row0, 0), (1, K), (1, 0)
        )
        b_R = tl.load(p_R, boundary_check=(0, 1), padding_option="zero")
        p_knr = tl.make_block_ptr(
            kn + base, (T, K), (H * K, 1), (row0, 0), (BC, K), (1, 0)
        )
        p_bgr = tl.make_block_ptr(
            bg + base, (T, K), (H * K, 1), (row0, 0), (BC, K), (1, 0)
        )
        p_betar = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (row0,), (BC,), (0,)
        )
        b_knr = tl.load(p_knr, boundary_check=(0, 1), padding_option="zero")
        b_bgr = tl.load(p_bgr, boundary_check=(0, 1), padding_option="zero")
        b_betar = tl.load(p_betar, boundary_check=(0,), padding_option="zero")
        # row factor: bg_r <= R  ->  exponent <= 0  (bounded)
        b_kd = b_knr * tl.exp(b_bgr - b_R) * b_betar[:, None]

        for cj in range(ri + 1):
            col0 = i_t * BT + cj * BC
            p_knc = tl.make_block_ptr(
                kn + base, (T, K), (H * K, 1), (col0, 0), (BC, K), (1, 0)
            )
            p_bgc = tl.make_block_ptr(
                bg + base, (T, K), (H * K, 1), (col0, 0), (BC, K), (1, 0)
            )
            b_knc = tl.load(p_knc, boundary_check=(0, 1), padding_option="zero")
            b_bgc = tl.load(p_bgc, boundary_check=(0, 1), padding_option="zero")
            # column factor: exponent <= BC*|lb| (diagonal block); <= 0 earlier
            b_ki = b_knc * tl.exp(b_R - b_bgc)
            # bf16 MFMA (fp32 accum): A entries <=1 so lossless, avoids slow fp32
            b_Ablk = tl.dot(b_kd.to(tl.bfloat16), tl.trans(b_ki).to(tl.bfloat16))
            if cj == ri:
                b_Ablk = tl.where(o_c[:, None] > o_c[None, :], b_Ablk, 0.0)
            p_Ablk = tl.make_block_ptr(
                A + (bos * H + i_h) * BT,
                (T, BT),
                (H * BT, 1),
                (row0, cj * BC),
                (BC, BC),
                (1, 0),
            )
            tl.store(p_Ablk, b_Ablk.to(p_Ablk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _wu_vector_fwd_kernel(
    Tinv,
    beta_v,
    beta_kd,
    u,
    w,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_c, i_h = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    p_T = tl.make_block_ptr(
        Tinv + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_T = tl.load(p_T, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_bv = tl.make_block_ptr(
            beta_v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_bv = tl.load(p_bv, boundary_check=(0, 1))
        b_u = tl.dot(b_T.to(tl.bfloat16), b_bv.to(tl.bfloat16))
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_bkd = tl.make_block_ptr(
            beta_kd + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_bkd = tl.load(p_bkd, boundary_check=(0, 1))
        b_w = tl.dot(b_T.to(tl.bfloat16), b_bkd.to(tl.bfloat16))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _gate_apply_kernel(
    kn,
    bg,
    beta,
    v,
    kd_beta,
    beta_v,
    kn_out,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fused per-channel gate application (one pass, avoids torch temporaries).

    Emits kd_beta = Kn.e^{bg}.beta (bounded, bg<=0) and beta_v = beta.V for W/u,
    plus ``kn_out`` (bf16 kn for the scan). The overflow-prone ``e^{-bg}`` factors
    are formed in the KKt/output kernels, not materialized here.
    """
    row = tl.program_id(0)  # token * H + head
    b_beta = tl.load(beta + row).to(tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    base_k = row * K + offs_k
    b_kn = tl.load(kn + base_k, mask=mask_k, other=0.0).to(tl.float32)
    b_bg = tl.load(bg + base_k, mask=mask_k, other=0.0).to(tl.float32)
    tl.store(kd_beta + base_k, b_kn * tl.exp(b_bg) * b_beta, mask=mask_k)
    tl.store(kn_out + base_k, b_kn.to(kn_out.dtype.element_ty), mask=mask_k)

    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    base_v = row * V + offs_v
    b_v = tl.load(v + base_v, mask=mask_v, other=0.0).to(tl.float32)
    tl.store(beta_v + base_v, (b_v * b_beta).to(beta_v.dtype.element_ty), mask=mask_v)


@triton.jit(do_not_specialize=["T"])
def _state_scan_fwd_kernel(
    w,
    u,
    kn,
    bg,
    h0,
    h_ckpt,
    vnew,
    hT,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """Sequential-over-chunks state recurrence, one program per (sequence, head).

    Emits per-chunk state checkpoints ``h_ckpt`` (state at chunk start) and the
    corrected values ``vnew`` so the heavy output matmuls can run fully parallel
    in a separate kernel. Keeps H_block [K, BV] resident and always fp32. State
    restarts from each sequence's own ``h0`` slot, so packed sequences never
    carry state across a boundary.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    o_t = tl.arange(0, BT)

    p_h0 = tl.make_block_ptr(
        h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
    )
    b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_ck = tl.make_block_ptr(
            h_ckpt + ((boh + i_t) * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (K, BV),
            (1, 0),
        )
        tl.store(p_ck, b_h.to(p_ck.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        p_kn = tl.make_block_ptr(
            kn + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        p_bg = tl.make_block_ptr(
            bg + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_kn = tl.load(p_kn, boundary_check=(0, 1))
        b_bg = tl.load(p_bg, boundary_check=(0, 1), padding_option="zero")
        b_u = tl.load(p_u, boundary_check=(0, 1))

        row_valid = (i_t * BT + o_t) < T
        bg_last = tl.min(tl.where(row_valid[:, None], b_bg, 1e30), axis=0)

        # bf16 MFMA with fp32 accumulation (matches chunk_delta_h); the state H
        # is stored/carried in fp32 and only cast to bf16 for the matmuls.
        b_vnew = b_u - tl.dot(b_w.to(tl.bfloat16), b_h.to(tl.bfloat16))
        p_vn = tl.make_block_ptr(
            vnew + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        tl.store(p_vn, b_vnew.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        b_kend = b_kn * tl.exp(bg_last[None, :] - b_bg)
        b_kend = tl.where(row_valid[:, None], b_kend, 0.0)
        b_h = tl.exp(bg_last)[:, None] * b_h + tl.dot(
            tl.trans(b_kend.to(tl.bfloat16)), b_vnew.to(tl.bfloat16)
        )

    p_hT = tl.make_block_ptr(
        hT + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
    )
    tl.store(p_hT, b_h.to(p_hT.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _output_fwd_kernel(
    qn,
    kn,
    bg,
    h_ckpt,
    vnew,
    o,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Fully-parallel per-chunk output: o = Qd.H + tril(Qd Ki^T,0).v_new.

    Intra term two-level sub-chunked like the KKt kernel to bound
    ``e^{-bg}``; inter term uses ``Qd = scale.Qn.e^{bg}`` (bg<=0, bounded).
    """
    i_c, i_h = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    o_c = tl.arange(0, BC)
    NSUB: tl.constexpr = BT // BC
    base = (bos * H + i_h) * K

    p_ck = tl.make_block_ptr(
        h_ckpt + ((boh + i_t) * H + i_h) * K * V,
        (K, V),
        (V, 1),
        (0, 0),
        (K, V),
        (1, 0),
    )
    b_h = tl.load(p_ck, boundary_check=(0, 1)).to(tl.bfloat16)

    for ri in range(NSUB):
        row0 = i_t * BT + ri * BC
        p_R = tl.make_block_ptr(
            bg + base, (T, K), (H * K, 1), (row0, 0), (1, K), (1, 0)
        )
        p_qnr = tl.make_block_ptr(
            qn + base, (T, K), (H * K, 1), (row0, 0), (BC, K), (1, 0)
        )
        p_bgr = tl.make_block_ptr(
            bg + base, (T, K), (H * K, 1), (row0, 0), (BC, K), (1, 0)
        )
        b_R = tl.load(p_R, boundary_check=(0, 1), padding_option="zero")
        b_qnr = tl.load(p_qnr, boundary_check=(0, 1), padding_option="zero")
        b_bgr = tl.load(p_bgr, boundary_check=(0, 1), padding_option="zero")
        # inter term: qd = scale.Qn.e^{bg} (bg<=0, bounded)
        b_qd = (b_qnr * tl.exp(b_bgr) * scale).to(tl.bfloat16)
        b_o = tl.dot(b_qd, b_h)
        # intra term: reference row factor to R (exponent bg_r - R <= 0)
        b_qref = (b_qnr * tl.exp(b_bgr - b_R) * scale).to(tl.bfloat16)

        for cj in range(ri + 1):
            col0 = i_t * BT + cj * BC
            p_knc = tl.make_block_ptr(
                kn + base, (T, K), (H * K, 1), (col0, 0), (BC, K), (1, 0)
            )
            p_bgc = tl.make_block_ptr(
                bg + base, (T, K), (H * K, 1), (col0, 0), (BC, K), (1, 0)
            )
            p_vnc = tl.make_block_ptr(
                vnew + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (col0, 0),
                (BC, V),
                (1, 0),
            )
            b_knc = tl.load(p_knc, boundary_check=(0, 1), padding_option="zero")
            b_bgc = tl.load(p_bgc, boundary_check=(0, 1), padding_option="zero")
            b_vnc = tl.load(p_vnc, boundary_check=(0, 1)).to(tl.bfloat16)
            b_kref = (b_knc * tl.exp(b_R - b_bgc)).to(tl.bfloat16)
            b_Ao = tl.dot(b_qref, tl.trans(b_kref))
            if cj == ri:
                b_Ao = tl.where(o_c[:, None] >= o_c[None, :], b_Ao, 0.0)
            b_o += tl.dot(b_Ao.to(tl.bfloat16), b_vnc)

        p_o = tl.make_block_ptr(
            o + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (row0, 0),
            (BC, V),
            (1, 0),
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def kda_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None = -5.0,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    inplace: bool = False,
    chunk_size: int = 64,
    block_value: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel KDA prefill. Drop-in for ``kda_recurrent`` on prefill.

    Supports packed variable-length prefill: ``q/k/v`` are a single flat token
    buffer ``[total_tokens, heads, dim]`` and ``cu_seqlens`` (length ``N + 1``)
    delimits the ``N`` sequences. ``state_indices`` (length ``N``) selects each
    sequence's initial-state slot in ``state``. A single sequence is the
    ``N = 1`` case; if ``cu_seqlens`` is None it is treated as ``[0, total]``.
    """
    if not q.is_cuda:
        raise ValueError("KDA chunk prefill requires GPU tensors")
    total_tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]

    single_state = state.ndim == 3
    if single_state:
        state = state.unsqueeze(0)
    if state.ndim != 4 or state.shape[1:] != (heads, key_dim, value_dim):
        raise ValueError("state must be [slots, heads, key_dim, value_dim]")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    raw_g = raw_g.contiguous()
    beta = beta.contiguous()

    # Numerical safety: the WY/output steps need ``Kn·exp(-bg)``,
    # which overflows fp32 once ``|bg| = n*|lower_bound|`` passes ~88.7 (-> NaN,
    # garbage output). The KKt/output kernels sub-chunk each chunk into ``BC``
    # rows referenced to the sub-chunk start, bounding the exponent to
    # ``BC*|lower_bound|`` while the state scan still runs at the full ``BT``.
    # ``BC`` is the largest power of two (<= BT) keeping that bound under 80.
    BT = chunk_size
    BC = BT
    if lower_bound is not None and lower_bound < 0.0:
        cap = int(80.0 / abs(lower_bound))
        safe_bc = 1 << (cap.bit_length() - 1) if cap >= 1 else 1
        BC = min(BT, max(1, safe_bc))
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, total_tokens], device=q.device, dtype=torch.int32)
    else:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    num_sequences = cu_seqlens.numel() - 1

    if state_indices is None:
        state_idx = torch.arange(num_sequences, device=q.device, dtype=torch.int64)
    else:
        state_idx = state_indices.to(device=q.device, dtype=torch.int64)
    if state_idx.numel() != num_sequences:
        raise ValueError("state_indices must have one entry per sequence")

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    num_chunks = chunk_indices.shape[0]

    gate = torch.empty_like(raw_g, dtype=torch.float32)
    beta_sig = torch.empty_like(beta, dtype=torch.float32)
    block_dim = triton.next_power_of_2(key_dim)
    _kda_prepare_gate_beta_kernel[(total_tokens, heads)](
        raw_g,
        beta,
        a_log,
        dt_bias,
        gate,
        beta_sig,
        H=heads,
        D=key_dim,
        BLOCK_D=block_dim,
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=min(max(block_dim // 32, 1), 8),
    )

    # l2norm normalizes in fp32 internally; ask for fp32 output directly so we
    # skip a bf16 store + widening pass and keep the full-precision normalized
    # value (feeds the fp32 KKt and the sequential state scan).
    qn = l2norm_fwd(q, output_dtype=torch.float32)
    kn = l2norm_fwd(k, output_dtype=torch.float32)

    # chunk-local cumulative gate, reset at every sequence boundary
    bg = chunk_local_cumsum_vector(
        gate.unsqueeze(0), BT, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )[0].contiguous()

    # fused gate application: kd_beta (bounded), beta_v, kn_scan. The e^{-bg}
    # factors are formed in the KKt/output kernels, not materialized.
    kd_beta = torch.empty(
        total_tokens, heads, key_dim, device=q.device, dtype=torch.float32
    )
    # beta_v is only consumed as a bf16 matmul operand (W/u kernel), so store bf16.
    beta_v = torch.empty(
        total_tokens, heads, value_dim, device=q.device, dtype=torch.bfloat16
    )
    # bf16 passthrough of kn for the memory-bound scan (fp32 kn stays for KKt)
    kn_scan = torch.empty(
        total_tokens, heads, key_dim, device=q.device, dtype=torch.bfloat16
    )
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    _gate_apply_kernel[(total_tokens * heads,)](
        kn,
        bg,
        beta_sig,
        v,
        kd_beta,
        beta_v,
        kn_scan,
        H=heads,
        K=key_dim,
        V=value_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=min(max(max(block_k, block_v) // 32, 1), 8),
    )

    # zeroed: the sub-chunk KKt writes only the lower-triangular blocks, so the
    # strictly-upper blocks must already be 0 for the triangular solve.
    A = torch.zeros(1, total_tokens, heads, BT, device=q.device, dtype=torch.float32)
    _kkt_vector_fwd_kernel[(num_chunks, heads)](
        kn,
        bg,
        beta_sig,
        A,
        cu_seqlens,
        chunk_indices,
        total_tokens,
        H=heads,
        K=key_dim,
        BT=BT,
        BC=BC,
        num_warps=8,
        num_stages=3,
    )
    # A is already fp32 and solve_tril reads it in fp32; pass it straight through
    # (no bf16 downcast pass, and no precision loss into the triangular solve).
    # Tinv is only consumed by W/u as a bf16 operand, so emit it bf16 directly.
    Tinv = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=torch.bfloat16)

    # u and w are consumed only by the scan; u feeds the fp32 residual
    # v_new = u - w@h (bf16 operand, fp32 accumulate, matching the reference
    # chunk_delta_h) and w feeds a bf16 MFMA, so store both bf16 to halve their
    # load traffic in the memory-bound scan.
    u = torch.empty(
        total_tokens, heads, value_dim, device=q.device, dtype=torch.bfloat16
    )
    w = torch.empty(total_tokens, heads, key_dim, device=q.device, dtype=torch.bfloat16)
    _wu_vector_fwd_kernel[(num_chunks, heads)](
        Tinv,
        beta_v,
        kd_beta,
        u,
        w,
        cu_seqlens,
        chunk_indices,
        total_tokens,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=BT,
        BK=min(64, key_dim),
        BV=min(64, value_dim),
        num_warps=8,
        num_stages=3,
    )

    final_state = state if inplace else state.clone()
    h0 = state[state_idx].contiguous()

    # o is accumulated in fp32 registers and written once (never read back), then
    # returned as q.dtype; store it directly in q.dtype (matches reference chunk_o).
    o = torch.empty(total_tokens, heads, value_dim, device=q.device, dtype=q.dtype)
    hT = torch.empty(
        num_sequences, heads, key_dim, value_dim, device=q.device, dtype=torch.float32
    )

    # Kernel A: sequential state recurrence only -> per-chunk state checkpoints
    # and corrected values v_new. One program per (sequence, head); state resets
    # per sequence. Lightweight, so low occupancy is acceptable.
    # h_ckpt and vnew are produced by the scan and consumed only by the output
    # kernel, which casts them to bf16; store bf16 (h_ckpt is the largest tensor,
    # so this also cuts the scan's per-chunk checkpoint-store traffic).
    h_ckpt = torch.empty(
        num_chunks, heads, key_dim, value_dim, device=q.device, dtype=torch.bfloat16
    )
    vnew = torch.empty(
        total_tokens, heads, value_dim, device=q.device, dtype=torch.bfloat16
    )
    NV = triton.cdiv(value_dim, block_value)
    # BV=16, num_warps=4, num_stages=2 won a launch-config sweep at 1k and 8k;
    # the two-kernel split shrank the scan's LDS footprint enough to pipeline
    # loads (num_stages=2), which the old monolithic scan could not afford.
    _state_scan_fwd_kernel[(NV, num_sequences * heads)](
        w,
        u,
        kn_scan,
        bg,
        h0,
        h_ckpt,
        vnew,
        hT,
        cu_seqlens,
        chunk_offsets,
        total_tokens,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=BT,
        BV=block_value,
        num_warps=4,
        num_stages=2,
    )

    # Kernel B: fully-parallel output over all chunks (high occupancy).
    _output_fwd_kernel[(num_chunks, heads)](
        qn,
        kn,
        bg,
        h_ckpt,
        vnew,
        o,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        key_dim**-0.5,
        total_tokens,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=BT,
        BC=BC,
        num_warps=4,
        num_stages=2,
    )
    final_state[state_idx] = hT.to(final_state.dtype)

    output = o.to(q.dtype)
    if single_state:
        final_state = final_state.squeeze(0)
    return output, final_state
