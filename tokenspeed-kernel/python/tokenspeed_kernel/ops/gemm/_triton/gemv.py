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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _rowcta_gemv_add3_kernel(
    x_ptr,
    w_ptr,
    a_ptr,
    c_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    """Row dot-product with a fused two-addend epilogue:
    ``out[n] = a[n] + x . w[n] + c[n]`` (the MoE residual accumulate rides
    the up-projection store; a/c row strides support lane column slices)."""
    n = tl.program_id(0)
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + n * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    av = tl.load(a_ptr + n).to(tl.float32)
    cv = tl.load(c_ptr + n).to(tl.float32)
    tl.store(
        out_ptr + n,
        (av + tl.sum(acc) + cv).to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _row_dot(x_ptr, w_ptr, K: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    return tl.sum(acc)


@triton.jit
def _rowcta_gemv_kernel(x_ptr, w_ptr, out_ptr, K: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0)
    value = _row_dot(x_ptr, w_ptr + n * K, K, BK)
    tl.store(out_ptr + n, value.to(out_ptr.dtype.element_ty))


@triton.jit
def _rows_dot_block(
    acc, x_ptr, w_ptr, n, rows, row_mask, kb, K: tl.constexpr, BK: tl.constexpr
):
    offs = kb + tl.arange(0, BK)
    col_mask = offs < K
    wv = tl.load(w_ptr + n * K + offs, mask=col_mask, other=0.0).to(tl.float32)
    # Cap the activation vector at four elements, the FP32 width: a wider BF16
    # load changes the accumulator layout and with it the reduction order.
    xv = tl.load(
        tl.max_contiguous(x_ptr + rows[:, None] * K + offs[None, :], [1, 4]),
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    return acc + xv * wv[None, :]


@triton.jit
def _rowcta_multirow_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """One weight row against every activation row; ``BM`` pads ``M``.

    ``M`` and ``N`` stay runtime values so decode batch sizes that share a
    ``BM`` reuse one compiled kernel, except ``M`` = 16: Triton compiles
    integer arguments divisible by 16 separately. Unrolling the K loop hides
    load latency for ``BM`` up to 8. For ``BM`` = 16, and past 15 blocks of
    512 for ``BM`` of 2 to 8, the unrolled loop can spill registers heavily,
    so the launcher uses the plain loop for ``BM`` = 16 and the registry keeps
    multi-row calls within 15 blocks. Both loops add the same products in the
    same order, so the result does not depend on ``UNROLL``.
    """
    n = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BM)
    row_mask = rows < M
    acc = tl.zeros([BM, BK], tl.float32)
    if UNROLL:
        for kb in tl.static_range(0, K, BK):
            acc = _rows_dot_block(acc, x_ptr, w_ptr, n, rows, row_mask, kb, K, BK)
    else:
        for kb in tl.range(0, K, BK):
            acc = _rows_dot_block(acc, x_ptr, w_ptr, n, rows, row_mask, kb, K, BK)
    values = tl.sum(acc, axis=1)
    tl.store(out_ptr + rows * N + n, values.to(out_ptr.dtype.element_ty), mask=row_mask)


@triton.jit
def _grouped_rowcta_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
    X_GROUP_STRIDE: tl.constexpr,
    W_GROUP_STRIDE: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    OUT_GROUP_STRIDE: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    value = _row_dot(
        x_ptr + group * X_GROUP_STRIDE,
        w_ptr + group * W_GROUP_STRIDE + n * W_ROW_STRIDE,
        K,
        BK,
    )
    tl.store(out_ptr + group * OUT_GROUP_STRIDE + n, value.to(out_ptr.dtype.element_ty))


def triton_rowcta_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for ``M == 1`` decode activations.

    Args:
        x: ``[1, K]`` contiguous bf16 activation row.
        weight: ``[N, K]`` contiguous bf16 weight.
        out: optional ``[1, N]`` destination.

    Returns:
        ``[1, N]`` output in ``x``'s dtype.
    """
    assert x.shape[0] == 1 and x.stride(-1) == 1 and weight.stride(-1) == 1
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    # BK=512 (4 fp32 accumulator regs/thread): standalone parity, and aux-stream kernels co-reside instead of stalling behind the GEMV wave.
    _rowcta_gemv_kernel[(n,)](
        x.view(-1),
        weight,
        out.view(-1),
        K=k,
        BK=512,
        num_warps=4,
    )
    return out


def triton_rowcta_gemm_fp32(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for up to 16 FP32 decode rows, one CTA per weight row.

    On Blackwell, Torch serves one FP32 row with a GEMV kernel and more rows
    with a SIMT GEMM plus a split-K reduction. A narrow FP32 projection such
    as an expert router is bound by its weight read, which this kernel
    streams once while the activation rows stay in cache; past 16 rows every
    CTA re-reading the activations costs more than it saves. Products and
    accumulation stay FP32 without fused multiply-add, independent of
    Torch's matmul precision setting. The registry selects it for at most
    1024 outputs, K a multiple of 4, and K up to 65536 for one row or 7680
    for more rows; other shapes compute correctly but can be slower than
    Torch.

    Args:
        x: ``[M, K]`` contiguous FP32 or BF16 activations, ``1 <= M <= 16``.
        weight: ``[N, K]`` contiguous FP32 weight.
        out: optional contiguous ``[M, N]`` destination.

    Returns:
        ``[M, N]`` FP32 output.
    """
    m, k = x.shape
    n = weight.shape[0]
    assert 1 <= m <= 16 and x.is_contiguous() and weight.is_contiguous()
    if out is None:
        out = torch.empty(m, n, dtype=torch.float32, device=x.device)
    if n == 0 or k == 0:
        return out.zero_()
    if m == 1:
        # One block over the whole row (up to 64K elements): each output is a
        # single reduction tree.
        block_m = 1
        block_k = min(triton.next_power_of_2(k), 65536)
        num_warps = 8 if block_k >= 4096 else 4
    else:
        block_m = triton.next_power_of_2(m)
        block_k = min(triton.next_power_of_2(k), 512)
        num_warps = 8 if block_m >= 16 else 4
    _rowcta_multirow_kernel[(n,)](
        x,
        weight,
        out,
        M=m,
        N=n,
        K=k,
        BM=block_m,
        BK=block_k,
        UNROLL=block_m <= 8,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return out


def rowcta_gemv_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a + x @ weight.T + c`` for ``M == 1`` (fused MoE residual epilogue).

    Args:
        x: ``[1, K]`` bf16 latent row; weight: ``[N, K]``.
        a/c: ``[1, N]`` addends (``c`` may be a wider-lane column slice --
            only unit inner stride is required).

    Returns:
        ``[1, N]`` prefix row.
    """
    assert x.shape[0] == 1 and a.shape == (1, weight.shape[0])
    assert a.stride(1) == 1 and c.stride(1) == 1 and c.shape[1] == weight.shape[0]
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    _rowcta_gemv_add3_kernel[(n,)](
        x.view(-1),
        weight,
        a,
        c,
        out,
        K=k,
        BK=512,
        num_warps=4,
    )
    return out


def grouped_bf16_projection_rowcta(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Single-token grouped projection; the public API validates the layout."""
    groups, rows, dim = weight.shape
    if out is None:
        out = torch.empty((1, groups, rows), dtype=x.dtype, device=x.device)
    _grouped_rowcta_gemv_kernel[(rows, groups)](
        x,
        weight,
        out,
        K=dim,
        BK=4096,
        X_GROUP_STRIDE=x.stride(1),
        W_GROUP_STRIDE=weight.stride(0),
        W_ROW_STRIDE=weight.stride(1),
        OUT_GROUP_STRIDE=out.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
