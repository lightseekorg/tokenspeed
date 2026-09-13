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

"""Ascend (NPU) GEMM kernels registered under the ``npu`` solution.

The kernels mirror the portable reference semantics of ``numerics.reference``
but are gated to the Ascend vendor and carry ``solution="npu"`` so NPU routing
is explicit at the selection layer. Dense GEMMs are served by ``torch.mm`` /
``torch.bmm`` (the torch_npu runtime implements these natively).

FP8 note: on CANN 9.0 + torch_npu 2.10 the runtime cannot materialize FP8
(E4M3/E5M2) tensor values (casts and in-place copies into FP8 output fail with
``aclnnInplaceCopy``), so the FP8-block and FP8-scaled contracts are served as
*dequantized* BF16/FP16 operations: the caller keeps activations/weights in
BF16/FP16 and passes the FP8 scale tensors alongside. This preserves the FP8
scale math (block-scale expansion, scaled products) through fp32 accumulation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    format_signatures,
)

__all__ = [
    "npu_bmm",
    "npu_mm",
    "npu_mm_fp8_blockscale",
    "npu_mm_fp8_scaled",
]

_NPU_CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))

_FP8_TENSOR_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="tensor",
)
_FP8_BLOCK_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_DENSE_GEMM_SIGNATURES = format_signatures(
    ("a", "b"), "dense", {torch.bfloat16, torch.float16}
)
# Dense npu kernels intentionally sit one step above the vendor-agnostic
# ``torch_mm``/``torch_bmm`` references (PORTABLE+3) so that Ascend routing is
# explicit (solution="npu") while remaining inside the portable band.
_GEMM_NPU_PRIORITY = Priority.PORTABLE + 4
_FP8_SCALED_SIGNATURES = format_signatures(
    ("a", "b"), "scaled-fp8", {torch.bfloat16, torch.float16}, scale=_FP8_TENSOR_SCALE
)
_FP8_BLOCK_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {torch.bfloat16, torch.float16}, scale=_FP8_BLOCK_SCALE
)

# Non-finite value retained for half/bfloat16 maxima; E4M3 max is 448.
_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _as_baddbmm_alpha(alpha: torch.Tensor | float | int | None) -> float | int:
    if alpha is None:
        return 1
    if isinstance(alpha, torch.Tensor):
        if alpha.numel() != 1:
            raise ValueError(
                f"npu_bmm alpha expects a scalar tensor, got {alpha.shape}"
            )
        return alpha.item()
    return alpha


@register_kernel(
    "gemm",
    "mm",
    name="npu_mm",
    solution="npu",
    capability=_NPU_CAPABILITY,
    signatures=_DENSE_GEMM_SIGNATURES,
    traits={},
    priority=_GEMM_NPU_PRIORITY,
    tags={"portability", "determinism"},
)
def npu_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense ``A @ B.T`` on Ascend NPU via torch.mm.

    Mirrors the portable ``torch_mm`` reference contract; dimensions follow the
    caller's declared shape (``B`` may be ``[K, N]`` or ``[N, K]`` and the
    result is always ``[M, N]`` with ``K`` taken from ``A``).
    """
    if A_scales is not None or B_scales is not None:
        raise ValueError("npu_mm does not support scale tensors")
    if block_size is not None:
        raise ValueError("block_size is not supported for dense npu mm")

    if out is not None:
        if out_dtype != A.dtype:
            raise ValueError(
                f"npu_mm out= requires out_dtype {A.dtype}, got {out_dtype}"
            )
        output = torch.mm(A, B.T, out=out)
        if alpha is not None:
            output.mul_(alpha.to(dtype=output.dtype))
        if bias is not None:
            output.add_(bias.to(dtype=output.dtype))
        return output

    if alpha is None:
        # F.linear fuses the bias add inside the GEMM epilogue.
        output = F.linear(A, B, bias)
    else:
        output = F.linear(A, B)
        output = output * alpha.to(dtype=output.dtype)
        if bias is not None:
            output = output + bias.to(dtype=output.dtype)
    return output.to(out_dtype)


@register_kernel(
    "gemm",
    "bmm",
    name="npu_bmm",
    solution="npu",
    capability=_NPU_CAPABILITY,
    signatures=_DENSE_GEMM_SIGNATURES,
    traits={},
    priority=_GEMM_NPU_PRIORITY,
    tags={"portability", "determinism"},
)
def npu_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched ``A @ B.transpose(1, 2)`` on Ascend NPU via torch.bmm."""
    if A_scales is not None or B_scales is not None:
        raise ValueError("npu_bmm does not support scale tensors")
    if block_size is not None:
        raise ValueError("block_size is not supported for dense npu bmm")
    if A.ndim != 3:
        raise ValueError(f"npu_bmm expects A=[B, M, K], got {A.shape}")
    if B.ndim != 3:
        raise ValueError(f"npu_bmm expects B=[B, N, K], got {B.shape}")
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"npu_bmm batch mismatch: {A.shape=} {B.shape=}")
    if A.shape[2] != B.shape[2]:
        raise ValueError(f"npu_bmm K mismatch: {A.shape=} {B.shape=}")
    if out is not None and out_dtype != A.dtype:
        raise ValueError(f"npu_bmm out= requires out_dtype {A.dtype}, got {out_dtype}")

    if bias is not None and out_dtype == A.dtype:
        bias = bias.to(dtype=A.dtype)
        if bias.ndim == 1:
            bias_view = bias.view(1, 1, -1)
        elif bias.ndim == 2:
            bias_view = bias.view(bias.shape[0], 1, bias.shape[1])
        else:
            raise ValueError(
                f"npu_bmm bias expects shape [N] or [B, N], got {bias.shape}"
            )
        if out is not None:
            return torch.baddbmm(
                bias_view,
                A,
                B.transpose(1, 2),
                alpha=_as_baddbmm_alpha(alpha),
                out=out,
            )
        return torch.baddbmm(
            bias_view,
            A,
            B.transpose(1, 2),
            alpha=_as_baddbmm_alpha(alpha),
        )

    output = torch.bmm(A, B.transpose(1, 2), out=out)
    if alpha is not None:
        output.mul_(alpha.to(dtype=output.dtype))
    if bias is not None:
        bias = bias.to(dtype=output.dtype)
        if bias.ndim == 1:
            bias_view = bias.view(1, 1, -1)
        elif bias.ndim == 2:
            bias_view = bias.view(bias.shape[0], 1, bias.shape[1])
        else:
            raise ValueError(
                f"npu_bmm bias expects shape [N] or [B, N], got {bias.shape}"
            )
        output.add_(bias_view)
    if output.dtype != out_dtype:
        output = output.to(out_dtype)
    return output


@register_kernel(
    "gemm",
    "mm",
    name="npu_mm_fp8_scaled",
    solution="npu",
    capability=_NPU_CAPABILITY,
    signatures=_FP8_SCALED_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE + 1,
    tags={"portability"},
)
def npu_mm_fp8_scaled(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tensor-scaled FP8 matmul served as a dequantized BF16/FP16 operation.

    Computes ``(A * A_scale) @ (B * B_scale).T`` with fp32 accumulation. Both
    scale tensors must be scalar ``(1,)`` tensors. ``B`` may be ``[K, N]`` or
    ``[N, K]`` (the K dimension is read from ``A``).
    """
    assert block_size is None, "block_size is not supported for fp8 scaled npu mm"
    assert (
        A_scales is not None and B_scales is not None
    ), "A_scales and B_scales are required for fp8 scaled npu mm"
    assert A_scales.shape == (
        1,
    ), f"A_scales must have shape (1,), got {A_scales.shape}"
    assert B_scales.shape == (
        1,
    ), f"B_scales must have shape (1,), got {B_scales.shape}"
    if A.shape[1] != B.shape[1] and A.shape[1] != B.shape[0]:
        raise ValueError(
            "npu_mm_fp8_scaled K mismatch: "
            f"A K={A.shape[1]}, B shape={tuple(B.shape)}"
        )

    a_scale = A_scales.to(device=A.device).float().item()
    b_scale = B_scales.to(device=A.device).float().item()
    output = (A.float() * a_scale) @ (B.float() * b_scale).T

    if alpha is not None:
        output = output * alpha.float()
    output = output.to(out_dtype)
    if out is not None:
        out.copy_(output)
        return out
    return output


def _npu_blockscale_quantize(
    A: torch.Tensor,
    *,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate FP8 block-scale activation quantization without FP8 storage.

    Computes the per-block scale ``max(|tile|) / 448`` and the scaled
    activation blocks, held in the input dtype (thence in fp32 by the caller).
    This replicates the FP8 online-quantization scale math using only dtypes
    the NPU runtime can materialize.
    """
    k_tiles = math.ceil(A.shape[-1] / block_k)
    A_q = torch.empty_like(A)
    A_scales = torch.empty(
        (*A.shape[:-1], k_tiles),
        device=A.device,
        dtype=torch.float32,
    )
    min_scale = torch.finfo(torch.float32).tiny
    for tile_idx in range(k_tiles):
        start = tile_idx * block_k
        end = min(start + block_k, A.shape[-1])
        tile = A[..., start:end].float()
        scale = (tile.abs().amax(dim=-1) / _FP8_E4M3_MAX).clamp_min(min_scale)
        A_scales[..., tile_idx] = scale
        A_q[..., start:end] = (tile / scale.unsqueeze(-1)).to(A.dtype)
    return A_q, A_scales


@register_kernel(
    "gemm",
    "mm",
    name="npu_mm_fp8_blockscale",
    solution="npu",
    capability=_NPU_CAPABILITY,
    signatures=_FP8_BLOCK_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE + 1,
    tags={"portability"},
)
def npu_mm_fp8_blockscale(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-scaled FP8 matmul served as a dequantized BF16/FP16 operation.

    A/B are dequantized BF16/FP16 tensors; ``A_scales`` (``[M, k_tiles]``) and
    ``B_scales`` (``[n_tiles, k_tiles]``) carry the FP8 block scales. When
    ``A_scales`` is omitted the kernel performs the FP8 online-quantization
    scale math internally (see :func:`_npu_blockscale_quantize`).
    """
    assert block_size is not None, "block_size is required for mxfp8 npu mm"
    if A_scales is not None:
        A_scales = A_scales.to(device=A.device)
    else:
        A, A_scales = _npu_blockscale_quantize(A, block_k=block_size[1])
    assert B_scales is not None, "B_scales is required for mxfp8 npu mm"
    B_scales = B_scales.to(device=A.device)
    assert A.ndim == 2 and B.ndim == 2, f"Expected 2D inputs, got {A.ndim=} {B.ndim=}"

    M, K = A.shape
    N, K_b = B.shape
    assert K_b == K, f"Expected B in [N, K] layout, got shape={tuple(B.shape)}"

    block_n, block_k = block_size
    k_tiles = math.ceil(K / block_k)
    n_tiles = math.ceil(N / block_n)
    assert A_scales.shape == (M, k_tiles), (
        f"A_scales shape mismatch: expected {(M, k_tiles)}, "
        f"got {tuple(A_scales.shape)}"
    )
    assert B_scales.shape == (n_tiles, k_tiles), (
        f"B_scales shape mismatch: expected {(n_tiles, k_tiles)}, "
        f"got {tuple(B_scales.shape)}"
    )

    A_scaled = A_scales.float().repeat_interleave(block_k, dim=1)[:, :K]
    B_scaled = (
        B_scales.float()
        .repeat_interleave(block_n, dim=0)
        .repeat_interleave(block_k, dim=1)[:N, :K]
    )
    output = (A.float() * A_scaled) @ (B.float() * B_scaled).T

    if alpha is not None:
        output = output * alpha.float()
    output = output.to(out_dtype)
    if out is not None:
        out.copy_(output)
        return out
    return output
