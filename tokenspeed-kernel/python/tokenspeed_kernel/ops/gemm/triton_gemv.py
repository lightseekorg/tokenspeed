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

"""Row-per-CTA M=1 bf16 GEMV.

Streams each weight row through one CTA (whole row in a single masked load,
dot against the L2-resident activation, one store). In the L2-cold regime a
decode step actually runs in, this beats every cublasLt tactic on the K3
skinny shapes by 13-14% (measured: 6288x7168 15.8us vs 18.1; 3584x7168
10.2us vs 11.9) while staying ~10% off the pure read+sum ceiling.
Deterministic by construction: one fixed-order reduction per output, no
split-K phase.
"""

from __future__ import annotations

import functools

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["decode_gemv", "rowcta_gemv", "rowcta_merged_front"]


@triton.jit
def _rowcta_gemv_add3_kernel(
    x_ptr,
    w_ptr,
    a_ptr,
    c_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Row dot-product with a fused two-addend epilogue:
    ``out[n] = a[n] + x . w[n] + c[n]`` (the MoE residual accumulate rides
    the up-projection store; a/c row strides support lane column slices)."""
    if ENABLE_PDL:
        # x/a/c are all written by the predecessor projection kernels; the
        # weight loop and the epilogue both consume them, so wait up front
        # before the first dependent read (conservative whole-kernel fence).
        tl.extra.cuda.gdc_wait()
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
    if ENABLE_PDL:
        # Output row stored; release the dependent (MoE/norm) kernel's launch.
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _rowcta_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        # x is the predecessor's activation row (weight is a static parameter,
        # but x is consumed inside the reduction loop): fence before the loop.
        tl.extra.cuda.gdc_wait()
    n = tl.program_id(0)
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + n * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    tl.store(out_ptr + n, tl.sum(acc).to(out_ptr.dtype.element_ty))
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _rowcta_merged_front_kernel(
    x_ptr,
    w_ptr,
    gate_ptr,
    routed_ptr,
    GATE_ROWS: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Merged-front sweep: one CTA per row of a stacked ``[gate | routed]``
    weight. Rows ``< GATE_ROWS`` store their fp32 dot to ``gate_ptr`` (router
    logits); the rest cast to ``routed_ptr``'s dtype (bf16 latent input). One
    launch, one L2-resident read of ``x`` for both projections."""
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    n = tl.program_id(0)
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + n * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    total = tl.sum(acc)
    if n < GATE_ROWS:
        tl.store(gate_ptr + n, total.to(gate_ptr.dtype.element_ty))
    else:
        tl.store(routed_ptr + (n - GATE_ROWS), total.to(routed_ptr.dtype.element_ty))
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def rowcta_merged_front(
    x: torch.Tensor,
    merged_weight: torch.Tensor,
    gate_rows: int,
    gate_out: torch.Tensor | None = None,
    routed_out: torch.Tensor | None = None,
    *,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-read merged front GEMV for ``M == 1`` decode.

    Reads ``x`` once and sweeps a vertically stacked
    ``merged_weight = cat([gate_weight, routed_down_weight])`` in one launch,
    emitting the router logits (fp32) and the latent routed input (bf16)
    separately. This is the K3 "fused front" concept -- gate + latent
    down-projection share their input, so their weights are merged and read
    together (design source: the SGLang K3 fused-front PR, Apache-2.0).

    Args:
        x: ``[1, K]`` contiguous bf16 activation row.
        merged_weight: ``[gate_rows + latent, K]`` contiguous bf16 weight, the
            router rows stacked on top of the latent down-projection rows.
        gate_rows: number of leading rows that are router logits.
        gate_out: optional ``[1, gate_rows]`` fp32 destination.
        routed_out: optional ``[1, latent]`` bf16 destination.
        enable_pdl: launch with programmatic dependent launch and fence the
            activation read (NVIDIA only; ignored elsewhere). Safe only when
            the predecessor writing ``x`` is chained on the same stream/graph.

    Returns:
        ``(gate_out [1, gate_rows] fp32, routed_out [1, latent] bf16)``.
    """
    assert x.shape[0] == 1 and x.stride(-1) == 1 and merged_weight.stride(-1) == 1
    n, k = merged_weight.shape
    latent = n - gate_rows
    assert latent > 0 and gate_rows > 0
    if gate_out is None:
        gate_out = torch.empty(1, gate_rows, dtype=torch.float32, device=x.device)
    if routed_out is None:
        routed_out = torch.empty(1, latent, dtype=x.dtype, device=x.device)
    pdl_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _rowcta_merged_front_kernel[(n,)](
        x.view(-1),
        merged_weight,
        gate_out.view(-1),
        routed_out.view(-1),
        GATE_ROWS=gate_rows,
        K=k,
        BK=512,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **pdl_kwargs,
    )
    return gate_out, routed_out


def rowcta_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """``x @ weight.T`` for ``M == 1`` decode activations.

    Args:
        x: ``[1, K]`` contiguous bf16 activation row.
        weight: ``[N, K]`` contiguous bf16 weight.
        out: optional ``[1, N]`` destination.
        enable_pdl: Launch with programmatic dependent launch and fence the
            activation read with ``gdc_wait`` / ``gdc_launch_dependents``
            (NVIDIA only; ignored elsewhere). Safe only when the predecessor
            kernel that writes ``x`` is chained on the same stream/graph.

    Returns:
        ``[1, N]`` output in ``x``'s dtype.
    """
    assert x.shape[0] == 1 and x.stride(-1) == 1 and weight.stride(-1) == 1
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    pdl_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    # BK=512 (4 fp32 accumulator regs/thread): standalone parity, and aux-stream kernels co-reside instead of stalling behind the GEMV wave.
    _rowcta_gemv_kernel[(n,)](
        x.view(-1),
        weight,
        out.view(-1),
        K=k,
        BK=512,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **pdl_kwargs,
    )
    return out


# ---------------------------------------------------------------------------
# Registry dispatch: the model calls decode_gemv unconditionally; the winner
# per M comes from the kernel registry (rowcta owns M == 1, cublasLt-backed
# torch.mm everything else). A future multi-M kernel (e.g. CuteDSL) slots in
# by registering another spec -- no model change.
# ---------------------------------------------------------------------------


_BF16_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.bfloat16),
        )
    }
)


@register_kernel(
    "gemm",
    "decode_gemv",
    name="rowcta_gemv_triton",
    solution="triton",
    signatures=_BF16_SIG,
    # m=1 streaming only; n/k floors bound the validated envelope (N 2304-7168, K 1536-7168; N=7168 with K<=1536 loses to cublasLt).
    traits={
        "m": frozenset({1}),
        "n_min_128": frozenset({True}),
        "k_min_128": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)
def _rowcta_spec(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rowcta_gemv(x, weight)


@register_kernel(
    "gemm",
    "decode_gemv",
    name="decode_gemv_torch",
    solution="torch",
    signatures=_BF16_SIG,
    traits={},
    priority=Priority.PORTABLE,
)
def _torch_spec(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x @ weight.t()


@functools.lru_cache(maxsize=64)
def _select(m: int, n: int, k: int, on_cuda: bool):
    if not on_cuda:
        return _torch_spec
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import (
        spec_matches_shape_traits,
        spec_matches_traits,
    )

    reg = KernelRegistry.get()
    for spec in reg.get_for_operator("gemm", "decode_gemv"):
        if spec_matches_traits(spec, {"m": m}) and spec_matches_shape_traits(
            spec, {"N": n, "K": k}
        ):
            return reg.get_impl(spec.name)
    return _torch_spec


def decode_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` with registry-selected decode kernels.

    Selection is cached per (M, N, K, device kind); the shape traits keep
    the specialized kernels inside their validated envelope and everything
    else routes to the portable fallback.
    """
    return _select(x.shape[0], weight.shape[0], weight.shape[1], x.is_cuda)(x, weight)


def rowcta_gemv_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """``a + x @ weight.T + c`` for ``M == 1`` (fused MoE residual epilogue).

    Args:
        x: ``[1, K]`` bf16 latent row; weight: ``[N, K]``.
        a/c: ``[1, N]`` addends (``c`` may be a wider-lane column slice --
            only unit inner stride is required).
        enable_pdl: Launch with programmatic dependent launch and fence the
            predecessor reads with ``gdc_wait`` / ``gdc_launch_dependents``
            (NVIDIA only; ignored elsewhere). Safe only when the kernels
            writing ``x``/``a``/``c`` are chained on the same stream/graph.

    Returns:
        ``[1, N]`` prefix row.
    """
    assert x.shape[0] == 1 and a.shape == (1, weight.shape[0])
    assert a.stride(1) == 1 and c.stride(1) == 1 and c.shape[1] == weight.shape[0]
    n, k = weight.shape
    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)
    pdl_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _rowcta_gemv_add3_kernel[(n,)](
        x.view(-1),
        weight,
        a,
        c,
        out,
        K=k,
        BK=512,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **pdl_kwargs,
    )
    return out
