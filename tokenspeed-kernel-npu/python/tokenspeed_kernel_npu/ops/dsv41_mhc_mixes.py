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


"""FP32 HC4 projection with fused RMS and coefficient evaluation."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _hc4_store_coefficients(
    mixes,
    inv_rms,
    Scale,
    Base,
    Pre,
    Post,
    Comb,
    row,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
):
    i = tl.arange(0, 4)
    pre = tl.gather(mixes, i, 0) * inv_rms
    post = tl.gather(mixes, 4 + i, 0) * inv_rms
    pre = tl.sigmoid(pre * tl.load(Scale) + tl.load(Base + i)) + EPS
    post = 2.0 * tl.sigmoid(post * tl.load(Scale + 1) + tl.load(Base + 4 + i))
    # Each writer owns a 64-byte row. Narrow adjacent rows caused corrupted
    # batched output with this Ascend compiler's sub-block stores.
    tl.store(Pre + row * 16 + i, pre)
    tl.store(Post + row * 16 + i, post)
    j = tl.arange(0, 16)
    comb = tl.gather(mixes, 8 + j, 0) * inv_rms
    comb = comb * tl.load(Scale + 2) + tl.load(Base + 8 + j)
    comb = tl.reshape(comb, (4, 4))
    comb = tl.exp(comb - tl.max(comb, 1)[:, None])
    comb = comb / tl.sum(comb, 1)[:, None] + EPS
    comb = comb / (tl.sum(comb, 0)[None, :] + EPS)
    for _ in tl.static_range(1, ITERS):
        comb = comb / (tl.sum(comb, 1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, 0)[None, :] + EPS)
    tl.store(Comb + row * 16 + j, tl.reshape(comb, (16,)))


@triton.jit(do_not_specialize=["N"])
def _hc4_project_partial(
    X,
    W,
    P,
    N,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
):
    base = tl.program_id(0) * R
    group = tl.program_id(1)
    part = tl.program_id(2)
    cols = group * C + tl.arange(0, C)
    k = part * B + tl.arange(0, B)
    w = tl.load(
        W + cols[:, None] * K + k[None, :], (cols[:, None] < 24) & (k[None, :] < K), 0
    )
    for step in tl.static_range(R):
        row = base + step
        x = tl.load(X + row * K + k, (row < N) & (k < K), 0).to(tl.float32)
        val = tl.sum(w * x[None, :], 1)
        tl.store(P + row * PARTS * 32 + part * 32 + cols, val, (row < N) & (cols < 24))
        if group == 0:
            rms = tl.sum(x * x, 0)
            tl.store(P + row * PARTS * 32 + part * 32 + 24, rms, row < N)


@triton.jit(do_not_specialize=["N"])
def _hc4_reduce_coefficients(
    P,
    S,
    Base,
    Pre,
    Post,
    Comb,
    N,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    PB: tl.constexpr,
    EPS: tl.constexpr,
    NEPS: tl.constexpr,
    ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    parts = tl.arange(0, PB)
    cols = tl.arange(0, 32)
    m = tl.sum(
        tl.load(
            P + row * PARTS * 32 + parts[:, None] * 32 + cols[None, :],
            (parts[:, None] < PARTS) & (cols[None, :] < 25),
            0,
        ),
        0,
    )
    rms = tl.sum(tl.where(cols == 24, m, 0), 0)
    inv = tl.rsqrt(rms / K + NEPS)
    _hc4_store_coefficients(m, inv, S, Base, Pre, Post, Comb, row, EPS, ITERS)


@triton.jit
def _hc4_coefficients_rms(
    X,
    M,
    Scale,
    Base,
    Pre,
    Post,
    Comb,
    XS0,
    XS1,
    MS0,
    MS1,
    DIM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    total = tl.full((), 0.0, tl.float32)
    for start in range(0, DIM, BLOCK):
        x = tl.load(X + row * XS0 + (start + d) * XS1, start + d < DIM, other=0).to(
            tl.float32
        )
        total = total + tl.sum(x * x, 0)
    inv_rms = tl.rsqrt(total / DIM + NORM_EPS)
    columns = tl.arange(0, 32)
    mixes = tl.load(M + row * MS0 + columns * MS1, columns < 24, other=0)
    _hc4_store_coefficients(
        mixes, inv_rms, Scale, Base, Pre, Post, Comb, row, EPS, ITERS
    )


@triton.jit
def _hc4_coefficients_rms_batch(
    X,
    M,
    Scale,
    Base,
    Pre,
    Post,
    Comb,
    XS0,
    XS1,
    MS0,
    MS1,
    N,
    DIM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, BLOCK)
    total = tl.full((ROWS,), 0.0, tl.float32)
    for start in range(0, DIM, BLOCK):
        x = tl.load(
            X + rows[:, None] * XS0 + (start + d[None, :]) * XS1,
            (rows[:, None] < N) & (start + d[None, :] < DIM),
            other=0,
        ).to(tl.float32)
        total = total + tl.sum(x * x, 1)
    inv = tl.rsqrt(total / DIM + NORM_EPS)
    i = tl.arange(0, 4)
    pre = (
        tl.load(M + rows[:, None] * MS0 + i[None, :] * MS1, rows[:, None] < N, other=0)
        * inv[:, None]
    )
    post = (
        tl.load(
            M + rows[:, None] * MS0 + (4 + i[None, :]) * MS1, rows[:, None] < N, other=0
        )
        * inv[:, None]
    )
    pre = tl.sigmoid(pre * tl.load(Scale) + tl.load(Base + i)[None, :]) + EPS
    post = 2.0 * tl.sigmoid(post * tl.load(Scale + 1) + tl.load(Base + 4 + i)[None, :])
    tl.store(Pre + rows[:, None] * 16 + i[None, :], pre, rows[:, None] < N)
    tl.store(Post + rows[:, None] * 16 + i[None, :], post, rows[:, None] < N)
    c0 = (
        tl.load(
            M + rows[:, None] * MS0 + (8 + i[None, :]) * MS1, rows[:, None] < N, other=0
        )
        * inv[:, None]
    )
    c0 = c0 * tl.load(Scale + 2) + tl.load(Base + 8 + i)[None, :]
    c0 = tl.exp(c0 - tl.max(c0, 1)[:, None])
    c0 = c0 / tl.sum(c0, 1)[:, None] + EPS
    c1 = (
        tl.load(
            M + rows[:, None] * MS0 + (12 + i[None, :]) * MS1,
            rows[:, None] < N,
            other=0,
        )
        * inv[:, None]
    )
    c1 = c1 * tl.load(Scale + 2) + tl.load(Base + 12 + i)[None, :]
    c1 = tl.exp(c1 - tl.max(c1, 1)[:, None])
    c1 = c1 / tl.sum(c1, 1)[:, None] + EPS
    c2 = (
        tl.load(
            M + rows[:, None] * MS0 + (16 + i[None, :]) * MS1,
            rows[:, None] < N,
            other=0,
        )
        * inv[:, None]
    )
    c2 = c2 * tl.load(Scale + 2) + tl.load(Base + 16 + i)[None, :]
    c2 = tl.exp(c2 - tl.max(c2, 1)[:, None])
    c2 = c2 / tl.sum(c2, 1)[:, None] + EPS
    c3 = (
        tl.load(
            M + rows[:, None] * MS0 + (20 + i[None, :]) * MS1,
            rows[:, None] < N,
            other=0,
        )
        * inv[:, None]
    )
    c3 = c3 * tl.load(Scale + 2) + tl.load(Base + 20 + i)[None, :]
    c3 = tl.exp(c3 - tl.max(c3, 1)[:, None])
    c3 = c3 / tl.sum(c3, 1)[:, None] + EPS
    col = (c0 + c2) + (c1 + c3) + EPS
    c0 = c0 / col
    c1 = c1 / col
    c2 = c2 / col
    c3 = c3 / col
    for _ in tl.static_range(1, ITERS):
        c0 = c0 / (tl.sum(c0, 1)[:, None] + EPS)
        c1 = c1 / (tl.sum(c1, 1)[:, None] + EPS)
        c2 = c2 / (tl.sum(c2, 1)[:, None] + EPS)
        c3 = c3 / (tl.sum(c3, 1)[:, None] + EPS)
        col = (c0 + c2) + (c1 + c3) + EPS
        c0 = c0 / col
        c1 = c1 / col
        c2 = c2 / col
        c3 = c3 / col
    j = tl.arange(0, 16)
    ix = tl.broadcast_to((j % 4)[None, :], (ROWS, 16))
    v0 = tl.gather(c0, ix, 1)
    v1 = tl.gather(c1, ix, 1)
    v2 = tl.gather(c2, ix, 1)
    v3 = tl.gather(c3, ix, 1)
    comb = tl.where(
        j[None, :] < 4,
        v0,
        tl.where(j[None, :] < 8, v1, tl.where(j[None, :] < 12, v2, v3)),
    )
    tl.store(Comb + rows[:, None] * 16 + j[None, :], comb, rows[:, None] < N)


def mhc_mixes(x, weight, scale, base, norm_eps, hc_eps, sinkhorn_iters):
    """Project [T,4,H] and return FP32 pre/post [T,4] and comb [T,4,4].

    The [24,4*H] projection and its reduction retain FP32 arithmetic. ``scale`` and ``base``
    contain the three learned scales and 24 learned coefficient biases.
    Input RMS and all coefficient normalization run in one kernel.
    """
    if x.ndim != 3 or x.shape[1] != 4 or weight.shape != (24, 4 * x.shape[2]):
        raise ValueError("HC4 mixes require [tokens,4,hidden] and [24,4*hidden]")
    n = x.shape[0]
    # Vector projection avoids the FP32 cast/GEMM overhead for narrow M.
    # Wider batches retain the GEMM path: reusing weights across many rows
    # outweighs the extra launches there. Both keep FP32 weights/arithmetic.
    if 0 < n <= 32 and x.is_contiguous() and weight.is_contiguous():
        dim = 4 * x.shape[2]
        tile = min(4096, triton.next_power_of_2(dim))
        parts = triton.cdiv(dim, tile)
        row_tile = 1 if n <= 2 else 2
        partials = torch.empty((n, parts, 32), device=x.device, dtype=torch.float32)
        pre = torch.empty((n, 16), device=x.device, dtype=torch.float32)
        post = torch.empty_like(pre)
        comb = torch.empty_like(pre)
        _hc4_project_partial[(triton.cdiv(n, row_tile), 6, parts)](
            x,
            weight,
            partials,
            n,
            dim,
            parts,
            tile,
            4,
            row_tile,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        _hc4_reduce_coefficients[(n,)](
            partials,
            scale,
            base,
            pre,
            post,
            comb,
            n,
            dim,
            parts,
            triton.next_power_of_2(parts),
            hc_eps,
            norm_eps,
            sinkhorn_iters,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        return pre[:, :4], post[:, :4], comb.reshape(n, 4, 4)
    flat = x.flatten(-2).float()
    n = flat.shape[0]
    if flat.shape[1] == 20480 and 96 <= n <= 192:
        # The verify batch has a very narrow [tokens, 24] output. Reversing
        # the operands gives CANN a faster FP32 tiling for these shapes.
        # Restore contiguous rows for the coefficient kernel; strided reads
        # otherwise erase the projection speedup. Small/prefill batches keep
        # their original tiling.
        mixes = torch.nn.functional.linear(weight.float(), flat).T.contiguous()
    else:
        mixes = torch.nn.functional.linear(flat, weight.float())
    pre = torch.empty((n, 16), device=x.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty_like(pre)
    if 96 <= n <= 192:
        # Four rows per program amortize the small Sinkhorn operations. Keep
        # every reduction two-dimensional: this Ascend compiler mishandles
        # the equivalent [rows,4,4] reduction/transpose formulation.
        _hc4_coefficients_rms_batch[(triton.cdiv(n, 4),)](
            flat,
            mixes,
            scale,
            base,
            pre,
            post,
            comb,
            *flat.stride(),
            *mixes.stride(),
            n,
            flat.shape[1],
            norm_eps,
            hc_eps,
            sinkhorn_iters,
            1024,
            4,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    elif n:
        _hc4_coefficients_rms[(n,)](
            flat,
            mixes,
            scale,
            base,
            pre,
            post,
            comb,
            *flat.stride(),
            *mixes.stride(),
            flat.shape[1],
            norm_eps,
            hc_eps,
            sinkhorn_iters,
            min(4096, triton.next_power_of_2(flat.shape[1])),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return pre[:, :4], post[:, :4], comb.reshape(n, 4, 4)
