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

"""Triton FP16/BF16 GEMV implementations.

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
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["decode_gemv", "rowcta_gemv"]


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
def _rowcta_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0)
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + n * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc += wv * xv
    tl.store(out_ptr + n, tl.sum(acc).to(out_ptr.dtype.element_ty))


def rowcta_gemv(
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
