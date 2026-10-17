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

"""Small-M FP32 GEMV against a row-major ``[K, N]`` weight (MoE routers).

One split-K launch streams every weight tile exactly once and reduces the
partials in a fixed order inside the kernel, so the result is deterministic
and no second launch or cuBLAS split-K reduce is needed.
"""

from __future__ import annotations

import os

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = ["fp32_router_gemv_counters", "triton_fp32_router_gemv"]


ROUTER_GEMV_BLOCK_K = 512
ROUTER_GEMV_BLOCK_N = 32
# Cold-L2 B300: the FMA path wins at 1-2 rows (5.3/6.4 us vs 8.6/20.6) and loses by 4.
ROUTER_GEMV_MAX_ROWS = 2
# Cold-L2 B300, 8192 x 256: the TF32 tensor-core path takes 7.2 us at 3-16 rows vs
# cuBLAS's 11.3 us (GEMM, split-K reduce and scale) and ties it past 32.
ROUTER_DOT_MAX_ROWS = 32


def _identity_router_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach()


@triton.jit
def _fp32_router_gemv_kernel(
    x_ptr,
    weight_ptr,
    partial_ptr,
    out_ptr,
    counter_ptr,
    multiplier,
    rows: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # The last program on a column block (by ticket) reduces its partials and re-arms.
    split = tl.program_id(0)
    col_block = tl.program_id(1)
    ks = split * BLOCK_K + tl.arange(0, BLOCK_K)
    ns = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    k_mask = ks < K
    n_mask = ns < N
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Dependents may launch now; their own wait still covers this grid's completion.
        tl.extra.cuda.gdc_launch_dependents()
    weight = tl.load(
        weight_ptr + ks[:, None] * N + ns[None, :],
        mask=k_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    for row in tl.static_range(rows):
        x = tl.load(x_ptr + row * K + ks, mask=k_mask, other=0.0)
        acc = tl.sum(weight * x[:, None], axis=0)
        tl.store(partial_ptr + (split * rows + row) * N + ns, acc, mask=n_mask)
    tl.debug_barrier()
    ticket = tl.atomic_add(counter_ptr + col_block, 1, sem="acq_rel", scope="gpu")
    # The barrier orders partial reads after the acquire, however the atomic lowers.
    tl.debug_barrier()
    if ticket == SPLITS - 1:
        for row in tl.static_range(rows):
            acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for s in tl.static_range(SPLITS):
                acc += tl.load(
                    partial_ptr + (s * rows + row) * N + ns,
                    mask=n_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.store(out_ptr + row * N + ns, acc * multiplier, mask=n_mask)
        tl.atomic_xchg(counter_ptr + col_block, 0, sem="release", scope="gpu")


@triton.jit
def _round_to_tf32(v):
    # cvt.rn rounds to the nearest TF32, ties to even, keeping NaN and Inf, as cuBLAS's TF32 path does.
    bits = tl.inline_asm_elementwise(
        "cvt.rn.tf32.f32 $0, $1;",
        "=r,f",
        [v],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _tf32_router_dot_kernel(
    x_ptr,
    weight_ptr,
    partial_ptr,
    out_ptr,
    counter_ptr,
    multiplier,
    rows,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNKS: tl.constexpr,
    SPLITS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # Same split-K ticket reduction as the FMA kernel, with each split's chunks on tensor cores.
    split = tl.program_id(0)
    col_block = tl.program_id(1)
    ms = tl.arange(0, BLOCK_M)
    ns = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = ms < rows
    n_mask = ns < N
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Dependents may launch now; their own wait still covers this grid's completion.
        tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for chunk in tl.range(0, CHUNKS):
        ks = (split * CHUNKS + chunk) * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        weight = tl.load(
            weight_ptr + ks[:, None] * N + ns[None, :],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        x = tl.load(
            x_ptr + ms[:, None] * K + ks[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(
            _round_to_tf32(x), _round_to_tf32(weight), acc, input_precision="tf32"
        )
    tile = (split * BLOCK_M + ms[:, None]) * N + ns[None, :]
    tl.store(partial_ptr + tile, acc, mask=m_mask[:, None] & n_mask[None, :])
    tl.debug_barrier()
    ticket = tl.atomic_add(counter_ptr + col_block, 1, sem="acq_rel", scope="gpu")
    # The barrier orders partial reads after the acquire, however the atomic lowers.
    tl.debug_barrier()
    if ticket == SPLITS - 1:
        total = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            total += tl.load(
                partial_ptr + (s * BLOCK_M + ms[:, None]) * N + ns[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
        tl.store(
            out_ptr + ms[:, None] * N + ns[None, :],
            total * multiplier,
            mask=m_mask[:, None] & n_mask[None, :],
        )
        tl.atomic_xchg(counter_ptr + col_block, 0, sem="release", scope="gpu")


def _cublas_rounds_to_tf32() -> bool:
    """Whether cuBLAS's FP32 matmul here uses TF32: torch allows it and no driver override forbids it."""
    return (
        current_platform().is_nvidia
        and torch.backends.cuda.matmul.fp32_precision == "tf32"
        and os.environ.get("NVIDIA_TF32_OVERRIDE") != "0"
    )


def _tf32_router_dot(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    counters: torch.Tensor,
    use_pdl: bool,
) -> torch.Tensor:
    rows, hidden = x.shape
    columns = weight.shape[1]
    block_m = max(16, triton.next_power_of_2(rows))
    # Measured on 8192-wide rows: longer K slabs at up to 16 rows, more pipelined chunks above.
    block_k, chunks, num_stages = (256, 2, 3) if rows <= 16 else (128, 4, 4)
    splits = triton.cdiv(hidden, block_k * chunks)
    partial = torch.empty(
        (splits, block_m, columns), dtype=torch.float32, device=x.device
    )
    out = torch.empty((rows, columns), dtype=torch.float32, device=x.device)
    launch = {"launch_pdl": True} if use_pdl else {}
    _tf32_router_dot_kernel[(splits, triton.cdiv(columns, ROUTER_GEMV_BLOCK_N))](
        x,
        weight,
        partial,
        out,
        counters,
        multiplier,
        rows,
        K=hidden,
        N=columns,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=ROUTER_GEMV_BLOCK_N,
        CHUNKS=chunks,
        SPLITS=splits,
        ENABLE_PDL=use_pdl,
        num_warps=4,
        num_stages=num_stages,
        **launch,
    )
    return out


def fp32_router_gemv_counters(num_columns: int, device: torch.device) -> torch.Tensor:
    """Allocate the zeroed completion counters ``fp32_router_gemv`` requires.

    Args:
        num_columns: Weight column count ``N``.
        device: Device the router weight lives on.

    Returns:
        An INT32 zero tensor with one counter per column block. The kernel
        returns it to zero, so one buffer serves every call from one call
        site; calls that may run concurrently need their own buffers.
    """
    if num_columns <= 0:
        raise ValueError(f"num_columns must be positive, got {num_columns}")
    return torch.zeros(
        triton.cdiv(num_columns, ROUTER_GEMV_BLOCK_N), dtype=torch.int32, device=device
    )


@register_kernel(
    "gemm",
    "prepared_fp32_router_gemv",
    name="triton_prepared_fp32_router_gemv",
    solution="triton",
    signatures=format_signatures(("x", "weight"), "dense", {torch.float32}),
    priority=Priority.PORTABLE,
    weight_preprocessor=_identity_router_weight,
)
@register_kernel(
    "gemm",
    "fp32_router_gemv",
    name="triton_fp32_router_gemv",
    solution="triton",
    signatures=format_signatures(("x", "weight"), "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def triton_fp32_router_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    counters: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """Triton kernel of :func:`tokenspeed_kernel.ops.gemm.fp32_router_gemv`."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be two-dimensional")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("fp32_router_gemv operates on FP32 inputs")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    rows, hidden = x.shape
    if weight.shape[0] != hidden:
        raise ValueError("weight rows must match x's final dimension")
    columns = weight.shape[1]
    if (
        counters.shape != (triton.cdiv(columns, ROUTER_GEMV_BLOCK_N),)
        or counters.dtype != torch.int32
        or counters.device != weight.device
    ):
        raise ValueError("counters must come from fp32_router_gemv_counters(columns)")
    if rows == 0:
        return x.new_empty((0, columns))
    use_pdl = enable_pdl and current_platform().is_hopper_plus
    if rows > ROUTER_GEMV_MAX_ROWS:
        # Where cuBLAS would round to TF32 anyway, the tensor-core split-K kernel rounds the same way.
        if rows <= ROUTER_DOT_MAX_ROWS and _cublas_rounds_to_tf32():
            return _tf32_router_dot(x, weight, multiplier, counters, use_pdl)
        return torch.matmul(x, weight) * multiplier
    splits = triton.cdiv(hidden, ROUTER_GEMV_BLOCK_K)
    partial = torch.empty((splits, rows, columns), dtype=torch.float32, device=x.device)
    out = torch.empty((rows, columns), dtype=torch.float32, device=x.device)
    launch = {"launch_pdl": True} if use_pdl else {}
    _fp32_router_gemv_kernel[(splits, triton.cdiv(columns, ROUTER_GEMV_BLOCK_N))](
        x,
        weight,
        partial,
        out,
        counters,
        multiplier,
        rows=rows,
        K=hidden,
        N=columns,
        BLOCK_K=ROUTER_GEMV_BLOCK_K,
        BLOCK_N=ROUTER_GEMV_BLOCK_N,
        SPLITS=splits,
        ENABLE_PDL=use_pdl,
        num_warps=4,
        **launch,
    )
    return out
