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
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "blockscale_dequant_bf16",
    "blockscale_w8a16_gemv",
    "decode_gemv",
    "rowcta_gemv",
]


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


# Registry dispatch: rowcta owns M == 1 while torch handles other shapes.
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
    traits={
        "m": frozenset({1}),
        "n_min_128": frozenset({True}),
        "k_min_128": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)
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


@register_kernel(
    "gemm",
    "decode_gemv",
    name="decode_gemv_torch",
    solution="torch",
    signatures=_BF16_SIG,
    traits={},
    priority=Priority.PORTABLE,
)
def _torch_decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return torch.mm(x, weight.t(), out=out)
    return x @ weight.t()


@functools.lru_cache(maxsize=64)
def _select(m: int, n: int, k: int, on_cuda: bool):
    if not on_cuda:
        return _torch_decode_gemv
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
    return _torch_decode_gemv


def decode_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T`` with registry-selected decode kernels.

    Selection is cached per (M, N, K, device kind); the shape traits keep
    the specialized kernels inside their validated envelope and everything
    else routes to the portable fallback.
    """
    expected = (x.shape[0], weight.shape[0])
    if out is not None:
        if (
            tuple(out.shape) != expected
            or out.dtype != x.dtype
            or out.device != x.device
            or out.stride(-1) != 1
        ):
            raise ValueError(f"out must match x and have shape {expected}")
        if not out.is_contiguous():
            return _torch_decode_gemv(x, weight, out)
    return _select(x.shape[0], weight.shape[0], weight.shape[1], x.is_cuda)(
        x, weight, out
    )


@triton.jit
def _blockscale_w8a16_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    KB,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """Row-per-CTA W8A16 GEMV for small-M decode batches.

    One CTA owns output row ``n`` and dots it against up to ``BM`` activation
    rows. With ``IS_FP8`` the weight row is FP8 with per-[128, BK]-block f32
    dequant scales and is dequantized in registers through bf16 — the exact
    load-time-dequant rounding sequence — before the fp32 accumulation. The
    ``IS_FP8=False`` flavor consumes a bf16 weight through the identical
    accumulation structure, so the two flavors are bitwise-comparable.
    """
    n = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    mask_m = offs_m < M
    acc = tl.zeros([BM], tl.float32)
    s_row = s_ptr + (n // 128) * KB
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        wv_raw = tl.load(w_ptr + n * K + offs)
        if IS_FP8:
            s = tl.load(s_row + kb // 128)
            wv = (wv_raw.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
        else:
            wv = wv_raw.to(tl.float32)
        xv = tl.load(
            x_ptr + offs_m[:, None] * K + offs[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(xv * wv[None, :], axis=1)
    tl.store(
        out_ptr + offs_m * N + n,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m,
    )


def blockscale_w8a16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequant(weight).T`` for small-M (<= 32) decode batches (W8A16).

    Activations stay bf16 (never quantized); FP8 weight rows are dequantized
    in registers per [128, 128] scale block through bf16 — bit-identical to a
    load-time ``(w.float() * scale).to(bf16)`` dequant feeding the same
    accumulation. Weight-bandwidth-bound: the FP8 read halves the dominant
    traffic vs the bf16 GEMV.

    Args:
        x: ``[M <= 32, K]`` contiguous bf16 activations.
        weight: ``[N, K]`` contiguous weight, FP8 (e4m3) with ``weight_scale``
            or bf16 (test/reference flavor, ``weight_scale`` must be None).
        weight_scale: f32 ``[ceil(N/128), K/128]`` block dequant multipliers.
        out: optional ``[M, N]`` contiguous destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    m, k = x.shape
    n, wk = weight.shape
    assert m <= 32, f"blockscale_w8a16_gemv covers decode M <= 32, got {m}"
    assert wk == k and k % 128 == 0, f"K must match and be 128-aligned: {k=} {wk=}"
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and weight.is_contiguous()
    is_fp8 = weight.dtype == torch.float8_e4m3fn
    if is_fp8:
        assert weight_scale is not None and weight_scale.dtype == torch.float32
        assert weight_scale.shape == (
            (n + 127) // 128,
            k // 128,
        ), f"{weight_scale.shape=} does not match {weight.shape=}"
        assert weight_scale.is_contiguous()
    else:
        assert weight.dtype == torch.bfloat16 and weight_scale is None
    if out is None:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    bm = max(1, triton.next_power_of_2(m))
    _blockscale_w8a16_gemv_kernel[(n,)](
        x,
        weight,
        weight_scale if is_fp8 else weight,  # unused pointer in bf16 flavor
        out,
        m,
        n,
        k // 128,
        K=k,
        BM=bm,
        BK=128,
        IS_FP8=is_fp8,
        num_warps=4,
    )
    return out


@triton.jit
def _blockscale_dequant_bf16_kernel(
    w_ptr,
    s_ptr,
    out_ptr,
    KB,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0)
    kb = tl.program_id(1) * BK
    offs = kb + tl.arange(0, BK)
    wv = tl.load(w_ptr + n * K + offs).to(tl.float32)
    s = tl.load(s_ptr + (n // 128) * KB + kb // 128)
    tl.store(out_ptr + n * K + offs, (wv * s).to(tl.bfloat16))


def blockscale_dequant_bf16(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Dequantize a block-scaled FP8 weight into a bf16 scratch buffer.

    Elementwise ``(w.float() * scale).to(bf16)`` — bit-identical to the
    load-time dequant recipe — so large-M callers can keep the FP8 weight
    resident and feed the vendor BLAS from a transient scratch.

    Args:
        weight: ``[N, K]`` contiguous FP8 (e4m3), ``K % 128 == 0``.
        weight_scale: f32 ``[ceil(N/128), K/128]`` dequant multipliers.
        out: ``[N, K]`` contiguous bf16 destination (reused across layers).
    """
    n, k = weight.shape
    assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous()
    assert k % 128 == 0 and weight_scale.shape == ((n + 127) // 128, k // 128)
    assert out.shape == (n, k) and out.dtype == torch.bfloat16
    assert out.is_contiguous() and weight_scale.is_contiguous()
    _blockscale_dequant_bf16_kernel[(n, k // 128)](
        weight,
        weight_scale,
        out,
        k // 128,
        K=k,
        BK=128,
        num_warps=1,
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
