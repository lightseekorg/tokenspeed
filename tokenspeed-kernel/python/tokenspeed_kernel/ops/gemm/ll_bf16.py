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

"""Registry entry for the vendored CuTe low-latency BF16 router GEMM.

Also exposes :func:`ll_bf16_mm`, the dense-linear form of the same two kernels:
BF16 output with an optional fused ``[N]`` bias, the shape vLLM reaches through
``--linear-backend flashinfer_cutedsl``. FlashInfer upstreamed those kernels as
``mm_bf16``'s ``cute-dsl`` backend, so :func:`ll_bf16_mm` calls FlashInfer when
the installed wheel declares it and runs the vendored copy otherwise; older
wheels therefore keep working without pinning a floor.

:func:`ll_bf16_mm` is deliberately *not* registered on ``gemm/decode_gemv``:
swept on B200 (sm100) against cublas, skinny, and tgv at the six ``(N, K)`` the
decode path actually dispatches, cold L2, CUDA-graph timed, it clears the
route's 4% margin at only 4 of the 33 ``(M, N, K)`` points measured and loses by
up to 3.5x at wide N -- 7168x768 at M = 4 costs 11.84us against cublas 3.39. The
kernel is bandwidth-optimal only when N is narrow enough that one CTA per column
still saturates, which the router shape satisfies and these projections do not.
``test/gemm_tuning/tune_route.py`` carries it as a candidate so a future sweep
can earn it entries the way skinny and tgv did.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import get_args

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import MAX_M, ll_bf16_router

_BF16_IN_FP32_OUT = {
    format_signature(
        hidden_states=dense_tensor_format(torch.bfloat16),
        weight=dense_tensor_format(torch.bfloat16),
    )
}


@register_kernel(
    "gemm",
    "router_projection",
    name="cute_dsl_ll_bf16_router",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        # Split-K reduces through DSMEM inside a thread block cluster.
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=_BF16_IN_FP32_OUT,
    priority=Priority.SPECIALIZED,
    traits={"out_dtype": frozenset({"float32"})},
    tags={"nvidia", "cute_dsl", "decode", "moe"},
)
def cute_dsl_ll_bf16_router(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Router logits in FP32 from BF16 operands, for decode-sized M.

    Args:
        hidden_states: ``[M, K]`` contiguous BF16 activation.
        weight: ``[N, K]`` contiguous BF16 router weight.
        out: Optional ``[M, N]`` FP32 destination; allocated when omitted.

    Returns:
        ``[M, N]`` FP32 router logits, ``out`` when it was given.
    """
    return ll_bf16_router(hidden_states, weight, out)


def ll_bf16_router_supported(
    hidden_states: torch.Tensor, weight: torch.Tensor, m: int
) -> bool:
    """Whether the vendored driver can serve this call.

    Args:
        hidden_states: ``[M, K]`` activation; weight: ``[N, K]`` weight.
        m: Token count, which selects the dot-product or split-K backend.

    Returns:
        True when a vendored kernel is compilable and applicable here.
    """
    return ll_bf16_router.supports(hidden_states, weight, m)


# The kernels issue 32-byte vector loads from the base of each operand.
_PTR_ALIGN = 32
# Both backends step K in multiples of 128.
_K_ALIGN = 128

# FlashInfer upstreamed these kernels as this backend, for sm100/sm103 only.
_FLASHINFER_BACKEND = "cute-dsl"
_FLASHINFER_ARCHS = frozenset({ArchVersion(10, 0), ArchVersion(10, 3)})


def _declares_cute_dsl_backend(mm_bf16: Callable[..., object]) -> bool:
    """Whether this ``mm_bf16`` lists :data:`_FLASHINFER_BACKEND`.

    Args:
        mm_bf16: FlashInfer's entry point, whose ``backend`` annotation is the
            ``Literal`` of the backends that build it.

    Returns:
        True on wheels carrying the upstreamed kernels, False on earlier ones,
        which name every other backend but not this one.
    """
    try:
        # eval_str resolves the Literal even if FlashInfer postpones annotations.
        backend = inspect.signature(mm_bf16, eval_str=True).parameters["backend"]
    except (KeyError, NameError, TypeError, ValueError):
        return False
    return _FLASHINFER_BACKEND in get_args(backend.annotation)


@functools.lru_cache(maxsize=1)
def _flashinfer_mm_bf16() -> Callable[..., torch.Tensor] | None:
    """FlashInfer's ``mm_bf16`` when it can run these kernels.

    Returns:
        The callable, or None when the platform is outside
        :data:`_FLASHINFER_ARCHS` or the wheel predates the backend -- either
        way the signal to run the vendored copy instead.
    """
    if current_platform().arch_version not in _FLASHINFER_ARCHS:
        return None
    try:
        from flashinfer import mm_bf16
    except ImportError:
        return None
    return mm_bf16 if _declares_cute_dsl_backend(mm_bf16) else None


def ll_bf16_mm_supported(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    """Whether :func:`ll_bf16_mm` can serve these operands.

    Args:
        x: ``[..., K]`` BF16 activation; its leading dims flatten to ``M``.
        weight: ``[N, K]`` BF16 weight.
        bias: Optional ``[N]`` BF16 bias.

    Returns:
        True when every guard holds, so the caller can fall back otherwise.
    """
    if weight.ndim != 2 or x.ndim < 1:
        return False
    k = weight.shape[1]
    n = weight.shape[0]
    # Checked before dividing by k, and on the originals: a reshape of a
    # non-contiguous x would copy, voiding the pointer-alignment check below.
    if k <= 0 or n <= 0 or x.shape[-1] != k or k % _K_ALIGN:
        return False
    if not x.is_contiguous() or not weight.is_contiguous():
        return False
    m = x.numel() // k
    if not 1 <= m <= MAX_M:
        return False
    if x.data_ptr() % _PTR_ALIGN or weight.data_ptr() % _PTR_ALIGN:
        return False
    if bias is not None and (
        bias.ndim != 1
        or bias.shape[0] != n
        or bias.dtype is not torch.bfloat16
        or not bias.is_contiguous()
        or bias.device != x.device
    ):
        return False
    return ll_bf16_router.supports(x.view(m, k), weight, m)


def ll_bf16_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T (+ bias)`` in BF16, the dense-linear form of the router GEMM.

    Runs FlashInfer's ``cute-dsl`` ``mm_bf16`` backend where the wheel declares
    it, and the vendored copy of the same kernels otherwise. Accumulation stays
    in FP32 and the bias folds into the epilogue either way, so the result rounds
    once. Call :func:`ll_bf16_mm_supported` first; operands outside the guard
    raise.

    Args:
        x: ``[..., K]`` contiguous BF16 activation; leading dims flatten to
            ``M``, which must not exceed :data:`MAX_M`.
        weight: ``[N, K]`` contiguous BF16 weight.
        bias: Optional contiguous ``[N]`` BF16 bias.
        out: Optional ``[..., N]`` BF16 destination; allocated when omitted.

    Returns:
        ``[..., N]`` BF16 tensor carrying ``x``'s leading dims, ``out`` when it
        was given.
    """
    k = weight.shape[1]
    m = x.numel() // k
    n = weight.shape[0]
    # view, not reshape: a reshaped non-contiguous out would take the writes.
    flat_out = None if out is None else out.view(m, n)
    mm_bf16 = _flashinfer_mm_bf16()
    if mm_bf16 is None:
        result = ll_bf16_router(
            x.view(m, k),
            weight,
            flat_out,
            bias=bias,
            out_dtype=torch.bfloat16,
        )
    else:
        # weight.t() is the (K, N) column-major B it wants, at no copy.
        result = mm_bf16(
            x.view(m, k),
            weight.t(),
            bias=bias,
            pdl=pdl_enabled(),
            out=flat_out,
            backend=_FLASHINFER_BACKEND,
        )
    return result.view(*x.shape[:-1], n)


__all__ = [
    "MAX_M",
    "cute_dsl_ll_bf16_router",
    "ll_bf16_mm",
    "ll_bf16_mm_supported",
    "ll_bf16_router_supported",
]
