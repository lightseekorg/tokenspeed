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
from tokenspeed_kernel.numerics.torch.embedding import rotate
from tokenspeed_kernel.numerics.torch.quantization import (
    quantize_fp8_groups,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
)

fp8_dtype = torch.float8_e4m3fn
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
    solution="torch",
    signatures=_MXFP8_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE + 2,
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
    solution="torch",
    signatures=_FP8_TENSOR_FORMAT_SIGNATURES,
    traits={
        "b_layout": frozenset({"NK"}),
    },
    priority=Priority.PORTABLE,
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
    assert A_scales.numel() == 1, "A_scales must be a single per-tensor scale"
    assert B_scales.numel() == 1, "B_scales must be a single per-tensor scale"

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
    solution="torch",
    signatures=_FP8_TENSOR_FORMAT_SIGNATURES,
    traits={
        "b_layout": frozenset({"KN"}),
    },
    priority=Priority.PORTABLE,
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
    assert A_scales.numel() == 1, "A_scales must be a single per-tensor scale"
    assert B_scales.numel() == 1, "B_scales must be a single per-tensor scale"

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
    solution="torch",
    signatures=_DENSE_GEMM_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE + 3,
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
    solution="torch",
    signatures=_MXFP8_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE + 2,
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
    solution="torch",
    signatures=_FP8_SCALED_BMM_FORMAT_SIGNATURES,
    traits={},
    priority=Priority.PORTABLE,
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
    solution="torch",
    signatures=_DENSE_GEMM_FORMAT_SIGNATURES,
    priority=Priority.PORTABLE + 3,
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


@register_kernel(
    "gemm",
    "grouped_bf16_projection",
    name="torch_grouped_bf16_projection",
    solution="torch",
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={},
    priority=Priority.PORTABLE,
)
def torch_grouped_bf16_projection(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Preserve the original batched projection and its BF16 rounding."""
    result = torch.einsum("tgd,grd->tgr", x, weight)
    if out is None:
        return result
    out.copy_(result)
    return out


def fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """FP32 ``fma(a, b, c)`` exactly.

    ``a * b`` of two FP32 values is exact in FP64; TwoSum gives the tail of
    ``p + c``, and the FP32 rounding of the FP64 sum is the exact sum's
    except on an FP32 midpoint, where the tail's sign decides.
    """
    p = a.double() * b.double()
    c64 = c.double()
    s = p + c64
    bb = s - p
    tail = (p - (s - bb)) + (c64 - bb)
    r = s.float()
    up = torch.nextafter(r, torch.full_like(r, float("inf")))
    down = torch.nextafter(r, torch.full_like(r, float("-inf")))
    above = (s == (r.double() + up.double()) * 0.5) & (tail > 0)
    below = (s == (r.double() + down.double()) * 0.5) & (tail < 0)
    return torch.where(above, up, torch.where(below, down, r))


@register_kernel(
    "gemm",
    "mv",
    name="torch_simt_gemm_fp32",
    solution="torch",
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.float32),
            )
        }
    ),
    traits={},
    priority=Priority.REFERENCE,
)
def torch_simt_gemm_fp32(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` in the exact FP32 summation order of
    ``gluon_simt_gemm_fp32`` (see ``ops/gemm/_gluon/gemv.py``).

    Four K quarters each run 128 FMA chains of 128-column steps; a warp lane
    adds its four chains in order, the warp reduces with an xor butterfly,
    and the quarters add in order. Emulated on the CPU, so slow; K must be a
    multiple of 512.
    """
    device = x.device
    x, weight = x.float().cpu(), weight.float().cpu()
    (m, k), n = x.shape, weight.shape[0]
    assert k % 512 == 0, f"K={k} must be a multiple of 512"
    chunk = k // 4
    total = torch.zeros(m, n)
    for s in range(4):
        acc = torch.zeros(m, n, 128)
        for i in range(chunk // 128):
            k0 = s * chunk + 128 * i
            acc = fma32(
                x[:, None, k0 : k0 + 128].expand(m, n, 128),
                weight[None, :, k0 : k0 + 128].expand(m, n, 128),
                acc,
            )
        # Lane 4 t + v: each warp lane t adds its four chains in order, then
        # the warp runs the xor butterfly.
        lanes = acc.view(m, n, 32, 4)
        u = ((lanes[..., 0] + lanes[..., 1]) + lanes[..., 2]) + lanes[..., 3]
        for offset in (16, 8, 4, 2, 1):
            u = u + u[..., torch.arange(32) ^ offset]
        total = total + u[..., 0]
    if out is not None:
        out.copy_(total)
        return out
    return total.to(device)


@register_kernel(
    "gemm",
    "dsv4_grouped_output_projection",
    name="torch_dsv4_grouped_output_projection",
    solution="torch",
    signatures=frozenset(
        format_signature(
            attention=dense_tensor_format(input_dtype),
            weight=dense_tensor_format(fp8_dtype),
        )
        for input_dtype in (torch.float16, torch.bfloat16)
    ),
    traits={},
    priority=Priority.REFERENCE,
)
def torch_dsv4_grouped_output_projection(
    *,
    attention: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    num_groups: int,
    heads_per_group: int,
    output_dim: int,
    nope_dim: int,
    rope_dim: int,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """DeepSeek-V4 grouped output projection of the attention output.

    Inverse-rotates the last ``rope_dim`` features of each head (GPT-J pairs),
    quantizes each group's FP32 row to FP8 with power-of-two scales per
    ``block_size[1]`` columns (no intermediate BF16 rounding), and multiplies
    by the canonical ``[num_groups * output_dim, heads_per_group * head_dim]``
    FP8 weight dequantized with its ``block_size`` FP32 scales.

    Returns:
        ``[tokens, num_groups, output_dim]`` in FP32, unrounded.
    """
    tokens = attention.shape[0]
    values = attention.float().clone()
    cos, sin = cos_sin_cache[positions].float().unsqueeze(1).chunk(2, dim=-1)
    values[..., nope_dim : nope_dim + rope_dim] = rotate(
        values[..., nope_dim : nope_dim + rope_dim], cos, -sin, is_neox=False
    )
    block_n, block_k = block_size
    codes, scales = quantize_fp8_groups(
        values.reshape(tokens, num_groups, -1),
        block_k,
        power_of_two_scale=True,
        min_scale=1e-10 / 448.0,
    )
    dequantized = codes.float() * scales.repeat_interleave(block_k, dim=-1)
    dequantized_weight = weight.float() * weight_scale.repeat_interleave(
        block_n, dim=0
    ).repeat_interleave(block_k, dim=1)
    grouped_weight = dequantized_weight.reshape(num_groups, output_dim, -1)
    return torch.bmm(
        dequantized.transpose(0, 1), grouped_weight.transpose(1, 2)
    ).transpose(0, 1)
