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

import math

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signatures

fp8_dtype = Platform.get().fp8e4m3fn.dtype
_FP8_BLOCK_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_FP8_TENSOR_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="tensor",
)
_FP8_CHANNEL_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="channel",
)
_MXFP8_UE8M0_1X32_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(1, 32),
)
# No reference is registered for the mixed float-A/ue8m0-B (1,32) signature:
# it exists only so kernels win dispatch before the gemm dispatcher requants
# activations to ue8m0; kernel impls never execute with float (1,32) A scales.
_MXFP8_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {fp8_dtype}, scale=_FP8_BLOCK_SCALE
) | format_signatures(("a", "b"), "mxfp8", {fp8_dtype}, scale=_MXFP8_UE8M0_1X32_SCALE)
_FP8_TENSOR_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "scaled-fp8", {fp8_dtype}, scale=_FP8_TENSOR_SCALE
)
_FP8_SCALED_BMM_FORMAT_SIGNATURES = _FP8_TENSOR_FORMAT_SIGNATURES | format_signatures(
    ("a", "b"), "scaled-fp8", {fp8_dtype}, scale=_FP8_CHANNEL_SCALE
)
_DENSE_GEMM_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "dense", {torch.bfloat16, torch.float16, torch.float32}
)


def _dequant_scales(scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype == torch.uint8:
        # ue8m0: biased power-of-two exponent bytes.
        return torch.exp2(scales.float() - 127.0)
    return scales.float()


def _as_baddbmm_alpha(alpha: torch.Tensor | float | int | None) -> float | int:
    if alpha is None:
        return 1
    if isinstance(alpha, torch.Tensor):
        if alpha.numel() != 1:
            raise ValueError(
                f"torch_bmm alpha expects a scalar tensor, got {alpha.shape}"
            )
        return alpha.item()
    return alpha


def _reference_mxfp8_quantize(
    A: torch.Tensor,
    *,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_tiles = math.ceil(A.shape[-1] / block_k)
    qA = torch.empty(A.shape, device=A.device, dtype=fp8_dtype)
    A_scales = torch.empty(
        (*A.shape[:-1], k_tiles),
        device=A.device,
        dtype=torch.float32,
    )
    fp8_max = float(torch.finfo(fp8_dtype).max)
    min_scale = torch.finfo(torch.float32).tiny
    for tile_idx in range(k_tiles):
        start = tile_idx * block_k
        end = min(start + block_k, A.shape[-1])
        tile = A[..., start:end].float()
        scale = (tile.abs().amax(dim=-1) / fp8_max).clamp_min(min_scale)
        A_scales[..., tile_idx] = scale
        qA[..., start:end] = (tile / scale.unsqueeze(-1)).to(fp8_dtype)
    return qA, A_scales


@register_kernel(
    "gemm",
    "mm",
    name="torch_mm_fp8_blockscale",
    solution="reference",
    signatures=_MXFP8_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE + 2,
    tags={"portability"},
)
def torch_mm_fp8_blockscale(
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
    assert block_size is not None, "block_size is required for mxfp8 reference"
    if A_scales is None:
        A, A_scales = _reference_mxfp8_quantize(A, block_k=block_size[1])
    assert B_scales is not None, "B_scales is required for mxfp8 reference"
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

    A_scaled = _dequant_scales(A_scales).repeat_interleave(block_k, dim=1)[:, :K]
    B_scaled = (
        _dequant_scales(B_scales)
        .repeat_interleave(block_n, dim=0)
        .repeat_interleave(block_k, dim=1)[:N, :K]
    )
    output = (A.float() * A_scaled) @ (B.float() * B_scaled).T

    if alpha is not None:
        output = output * alpha.float()
    output = output.to(out_dtype)
    if out is not None:
        # Reference expressions materialize through fp32 dequantization first,
        # so support pre-allocated output by copying into caller storage.
        out.copy_(output)
        return out
    return output


@register_kernel(
    "gemm",
    "mm",
    name="torch_mm_fp8_scaled_mnk",
    solution="reference",
    signatures=_FP8_TENSOR_FORMAT_SIGNATURES,
    traits={
        "b_layout": frozenset({"NK"}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def torch_mm_fp8_scaled_mnk(
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
    assert block_size is None, "block_size is not supported for fp8 scaled reference"
    assert (
        A_scales is not None and B_scales is not None
    ), "A_scales and B_scales are required for fp8 scaled reference"
    assert A_scales.shape == (1,), "A_scales must have shape (1,)"
    assert B_scales.shape == (1,), "B_scales must have shape (1,)"

    assert (
        A.shape[1] == B.shape[1]
    ), f"Expected A and B to have the same K dimension, got {tuple(A.shape)} and {tuple(B.shape)}"

    A_scales = float(A_scales.item())
    B_scales = float(B_scales.item())
    output = (A.float() * A_scales) @ (B.float() * B_scales).T

    if alpha is not None:
        output = output * alpha.float()
    output = output.to(out_dtype)
    if out is not None:
        # Reference expressions materialize through fp32 dequantization first,
        # so support pre-allocated output by copying into caller storage.
        out.copy_(output)
        return out
    return output


@register_kernel(
    "gemm",
    "mm",
    name="torch_mm_fp8_scaled_nkm",
    solution="reference",
    signatures=_FP8_TENSOR_FORMAT_SIGNATURES,
    traits={
        "b_layout": frozenset({"KN"}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def torch_mm_fp8_scaled_nkm(
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
    assert block_size is None, "block_size is not supported for fp8 scaled reference"
    assert (
        A_scales is not None and B_scales is not None
    ), "A_scales and B_scales are required for fp8 scaled reference"
    assert A_scales.shape == (1,), "A_scales must have shape (1,)"
    assert B_scales.shape == (1,), "B_scales must have shape (1,)"

    assert (
        A.shape[1] == B.shape[0]
    ), f"Expected A and B to have the same K dimension, got {tuple(A.shape)} and {tuple(B.shape)}"

    output = (A.float() * float(A_scales.item())) @ (B.float() * float(B_scales.item()))

    if alpha is not None:
        output = output * alpha.float()
    output = output.to(out_dtype)
    if out is not None:
        # Reference expressions materialize through fp32 dequantization first,
        # so support pre-allocated output by copying into caller storage.
        out.copy_(output)
        return out
    return output


@register_kernel(
    "gemm",
    "mm",
    name="torch_mm",
    solution="reference",
    signatures=_DENSE_GEMM_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE + 3,
    tags={"determinism", "portability"},
)
def torch_mm(
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
    if out is not None:
        if out_dtype != A.dtype:
            raise ValueError(
                f"torch_mm out= requires out_dtype {A.dtype}, got {out_dtype}"
            )
        # F.linear has no portable out= form, so write the GEMM natively and
        # apply the epilogue in place.
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
    name="torch_bmm_fp8_blockscale",
    solution="reference",
    signatures=_MXFP8_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE + 2,
    tags={"portability"},
)
def torch_bmm_fp8_blockscale(
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
    assert block_size is not None, "block_size is required for mxfp8 reference"
    if A_scales is None:
        A, A_scales = _reference_mxfp8_quantize(A, block_k=block_size[1])
    assert B_scales is not None, "B_scales is required for mxfp8 reference"
    assert A.ndim == 3 and B.ndim == 3, f"Expected 3D inputs, got {A.ndim=} {B.ndim=}"

    batch, M, K = A.shape
    B_batch, N, K_b = B.shape
    assert B_batch == batch, f"Expected matching batch dims, got {A.shape=} {B.shape=}"
    assert K_b == K, f"Expected B in [B, N, K] layout, got shape={tuple(B.shape)}"

    block_n, block_k = block_size
    k_tiles = math.ceil(K / block_k)
    n_tiles = math.ceil(N / block_n)
    assert A_scales.shape == (batch, M, k_tiles), (
        f"A_scales shape mismatch: expected {(batch, M, k_tiles)}, "
        f"got {tuple(A_scales.shape)}"
    )
    assert B_scales.shape == (batch, n_tiles, k_tiles), (
        f"B_scales shape mismatch: expected {(batch, n_tiles, k_tiles)}, "
        f"got {tuple(B_scales.shape)}"
    )

    A_scaled = A_scales.float().repeat_interleave(block_k, dim=2)[:, :, :K]
    B_scaled = (
        B_scales.float()
        .repeat_interleave(block_n, dim=1)
        .repeat_interleave(block_k, dim=2)[:, :N, :K]
    )
    output = torch.bmm(A.float() * A_scaled, (B.float() * B_scaled).transpose(1, 2))

    if alpha is not None:
        output = output * alpha.float()
    output = output.to(out_dtype)
    if out is not None:
        # Reference expressions materialize through fp32 dequantization first,
        # so support pre-allocated output by copying into caller storage.
        out.copy_(output)
        return out
    return output


def _bmm_scaled_fp8_scale(
    scale: torch.Tensor | None,
    *,
    batch: int,
    rows: int,
    name: str,
) -> float | torch.Tensor:
    assert scale is not None, f"{name} is required for fp8 scaled reference"
    if scale.shape == (1,):
        return float(scale.item())
    if scale.shape == (batch, rows):
        return scale.float().unsqueeze(-1)
    raise AssertionError(
        f"{name} shape mismatch: expected (1,) or {(batch, rows)}, "
        f"got {tuple(scale.shape)}"
    )


@register_kernel(
    "gemm",
    "bmm",
    name="torch_bmm_fp8_scaled",
    solution="reference",
    signatures=_FP8_SCALED_BMM_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def torch_bmm_fp8_scaled(
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
    assert block_size is None, "block_size is not supported for fp8 scaled reference"
    assert A.ndim == 3 and B.ndim == 3, f"Expected 3D inputs, got {A.ndim=} {B.ndim=}"

    batch, M, K = A.shape
    B_batch, N, K_b = B.shape
    assert B_batch == batch, f"Expected matching batch dims, got {A.shape=} {B.shape=}"
    assert K_b == K, f"Expected B in [B, N, K] layout, got shape={tuple(B.shape)}"

    A_scale = _bmm_scaled_fp8_scale(A_scales, batch=batch, rows=M, name="A_scales")
    B_scale = _bmm_scaled_fp8_scale(B_scales, batch=batch, rows=N, name="B_scales")
    output = torch.bmm(A.float() * A_scale, (B.float() * B_scale).transpose(1, 2))

    if alpha is not None:
        output = output * alpha.float()
    output = output.to(out_dtype)
    if out is not None:
        # Reference expressions materialize through fp32 dequantization first,
        # so support pre-allocated output by copying into caller storage.
        out.copy_(output)
        return out
    return output


@register_kernel(
    "gemm",
    "bmm",
    name="torch_bmm",
    solution="reference",
    signatures=_DENSE_GEMM_FORMAT_SIGNATURES,
    priority=Priority.PORTABLE + 3,
    tags={"determinism", "portability"},
)
def torch_bmm(
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
    if A_scales is not None:
        raise ValueError("A_scales are not supported for dense reference BMM")
    if B_scales is not None:
        raise ValueError("B_scales are not supported for dense reference BMM")
    if block_size is not None:
        raise ValueError("block_size is not supported for dense reference BMM")
    if A.ndim != 3:
        raise ValueError(f"torch_bmm expects A=[B, M, K], got {A.shape}")
    if B.ndim != 3:
        raise ValueError(f"torch_bmm expects B=[B, N, K], got {B.shape}")
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"torch_bmm batch mismatch: {A.shape=} {B.shape=}")
    if A.shape[2] != B.shape[2]:
        raise ValueError(f"torch_bmm K mismatch: {A.shape=} {B.shape=}")
    if out is not None and out_dtype != A.dtype:
        raise ValueError(
            f"torch_bmm out= requires out_dtype {A.dtype}, got {out_dtype}"
        )
    if bias is not None and out_dtype == A.dtype:
        bias = bias.to(dtype=A.dtype)
        # torch.baddbmm is the batched equivalent of the dense F.linear path:
        # it combines bias, alpha, matmul, and out= for native output dtype.
        if bias.ndim == 1:
            bias_view = bias.view(1, 1, -1)
        elif bias.ndim == 2:
            bias_view = bias.view(bias.shape[0], 1, bias.shape[1])
        else:
            raise ValueError(
                f"torch_bmm bias expects shape [N] or [B, N], got {bias.shape}"
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
        # Match mm bias broadcasting for either a shared [N] vector or a
        # per-batch [B, N] bias matrix.
        if bias.ndim == 1:
            bias_view = bias.view(1, 1, -1)
        elif bias.ndim == 2:
            bias_view = bias.view(bias.shape[0], 1, bias.shape[1])
        else:
            raise ValueError(
                f"torch_bmm bias expects shape [N] or [B, N], got {bias.shape}"
            )
        output.add_(bias_view)
    if output.dtype != out_dtype:
        # torch.bmm only writes the input dtype; non-native output dtypes are
        # represented as a reference cast after the native epilogue.
        output = output.to(out_dtype)
    return output
