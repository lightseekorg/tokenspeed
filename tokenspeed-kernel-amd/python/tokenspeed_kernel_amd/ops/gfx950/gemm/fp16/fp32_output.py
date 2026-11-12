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


"""Fixed-reduction BF16 projections with FP32 output on gfx950."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
    _get_partial_scratch,
    gluon_mm_a16w16_medium_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.splitk import check_splitk_counters


def supports_gluon_mm_bf16_fp32_gfx950(m: int, n: int, k: int) -> bool:
    if m < 0 or (n, k) not in {
        (8, 4608),
        (384, 4608),
        (3792, 4608),
        (4608, 1536),
    }:
        return False
    if max(m * k * 2, m * n * 4) > 2**32:
        return False
    # Split-K forms partial-tile element offsets with signed int32 arithmetic.
    tiles = triton.cdiv(m, 16) * triton.cdiv(n, 64)
    split_k = _projection_split_k(n, k)
    return split_k == 1 or tiles * split_k * 16 * 64 <= 2**31


def _check_buffer_span(tensor: torch.Tensor) -> None:
    # Raw-buffer instructions use 32-bit byte offsets from the view's base.
    span = (
        sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        + 1
    )
    if tensor.numel() and span * tensor.element_size() > 2**32:
        raise ValueError("gfx950 BF16 projection requires 32-bit buffer offsets")


# Fixed K partitions per projection shape. They shorten FP32 accumulation without
# a batch-dependent tree, and give decode's one 16-row tile enough CTAs: without
# a split, the router's 6 tiles (N=384) and wo's 72 tiles each walk all of K.
# The split does not depend on M, so a row's result does not depend on its
# batch. Measured cold on MI350X, us per launch at 8 / 128 / 2048 rows, split 1
# -> chosen split: router (384, 4608) 31.5 / 31.2 / 46.9 -> 4.8 / 11.3 / 63.3;
# wo (4608, 1536) 17.9 / 18.0 / 122.6 -> 9.4 / 17.5 / 160.9. Larger wo splits
# gain little more at 8 rows and cost more at 2048. A split must divide K into
# 64-wide blocks.
_PROJECTION_SPLIT_K = {
    (3792, 4608): 12,
    (384, 4608): 18,
    (4608, 1536): 3,
}


def _projection_split_k(n: int, k: int) -> int:
    return _PROJECTION_SPLIT_K.get((n, k), 1)


def gluon_mm_bf16_fp32_num_counters_gfx950(m: int, n: int, k: int) -> int:
    """Return int32 arrival-count entries for A[m,k] @ B[n,k].T.

    No entries are required when the projection does not partition K.
    """
    if _projection_split_k(n, k) == 1:
        return 0
    return triton.cdiv(m, 16) * triton.cdiv(n, 64)


def _projection_metadata(
    grid: tuple[int, ...], kernel: Any, args: dict[str, Any]
) -> dict[str, str | int]:
    m, k = args["A"].shape
    n = args["B"].shape[0]
    return {
        "name": kernel.name,
        "flops32": 2 * m * n * k,
        "bytes": (m * k + n * k) * 2 + m * n * 4,
    }


@gluon.jit(launch_metadata=_projection_metadata)
def gluon_mm_bf16_fp32_narrow_gfx950(
    A,
    B,
    C,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
    row = gl.program_id(0)
    columns = gl.program_id(1) * 4 + gl.arange(0, 4, gl.SliceLayout(1, layout))
    k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
    a = gl.load(A + row * stride_am + k * stride_ak, k < K, other=0).to(gl.float32)
    b = gl.load(
        B + columns[:, None] * stride_bn + k[None, :] * stride_bk,
        (columns[:, None] < N) & (k[None, :] < K),
        other=0,
    ).to(gl.float32)
    value = gl.sum(a[None, :] * b, axis=1)
    gl.store(C + row * stride_cm + columns * stride_cn, value, columns < N)


def launch_gluon_mm_bf16_fp32_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None,
    *,
    counters: torch.Tensor,
) -> torch.Tensor:
    """Compute measured BF16 A[M,K] @ B[N,K].T with FP32 output.

    A and B are colocated GPU matrices with unit inner strides. The optional
    out tensor is FP32 [M,N] and may have row padding. Return out, or a new tensor.
    Each view must fit within the kernel's 32-bit buffer-offset range.
    Counters are zeroed int32 storage of at least the reported counter count;
    every launch resets them to zero for subsequent eager or graph execution.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("BF16 projection requires A[M,K] and B[N,K]")
    if A.dtype != torch.bfloat16 or B.dtype != A.dtype:
        raise ValueError("BF16 projection requires BF16 operands")
    if not A.is_cuda or A.device != B.device:
        raise ValueError("BF16 projection requires colocated GPU operands")
    m, k = A.shape
    n = B.shape[0]
    if not supports_gluon_mm_bf16_fp32_gfx950(m, n, k):
        raise ValueError(f"Unsupported FP32 projection shape {(m, n, k)}")
    if A.stride(1) != 1 or B.stride(1) != 1:
        raise ValueError("gfx950 BF16 projection requires unit inner strides")
    _check_buffer_span(A)
    _check_buffer_span(B)
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=A.device)
    if out.shape != (m, n) or out.dtype != torch.float32 or out.device != A.device:
        raise ValueError("Projection output must be colocated FP32 [M,N]")
    _check_buffer_span(out)
    if not m:
        return out
    if n == 8:
        gluon_mm_bf16_fp32_narrow_gfx950[(m, triton.cdiv(n, 4))](
            A,
            B,
            out,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            out.stride(0),
            out.stride(1),
            N=n,
            K=k,
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        # Direct-to-LDS loads need 32-bit alignment: copy odd-padded or offset views.
        if A.stride(0) % 2 or A.data_ptr() % 4:
            A = A.clone(memory_format=torch.contiguous_format)
        if B.stride(0) % 2 or B.data_ptr() % 4:
            B = B.clone(memory_format=torch.contiguous_format)
        split_k = _projection_split_k(n, k)
        tiles_m, tiles_n = triton.cdiv(m, 16), triton.cdiv(n, 64)
        num_tiles = tiles_m * tiles_n
        if split_k > 1:
            check_splitk_counters(counters, num_tiles, A.device, "gfx950 FP32 GEMM")
            # Bucket capacity, not arithmetic, to bound eager scratch variants.
            capacity = triton.next_power_of_2(tiles_m) * tiles_n
            partial = _get_partial_scratch(A.device, capacity, split_k, 64, 16)
        else:
            partial = counters = out
        gluon_mm_a16w16_medium_gfx950[(num_tiles * split_k,)](
            A,
            B,
            out,
            out,
            out,
            partial,
            counters,
            m,
            n,
            k,
            A.stride(0),
            A.stride(1),
            B.stride(1),
            B.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=64,
            WARPS_M=1,
            WARPS_N=4,
            NUM_BUFFERS=2,
            GROUP_SIZE_M=2,
            ADD3=False,
            SPLIT_K=split_k,
            NUM_XCDS=1,
            PID_OFFSET=0,
            num_warps=4,
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
        )
    return out
