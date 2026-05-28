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
_MXFP8_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {fp8_dtype}, scale=_FP8_BLOCK_SCALE
)
_FP8_TENSOR_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "scaled-fp8", {fp8_dtype}, scale=_FP8_TENSOR_SCALE
)
_DENSE_GEMM_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "dense", {torch.bfloat16, torch.float16, torch.float32}
)


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
) -> torch.Tensor:
    assert block_size is not None, "block_size is required for mxfp8 reference"
    assert (
        A_scales is not None and B_scales is not None
    ), "A_scales and B_scales are required for mxfp8 reference"
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
    return output.to(out_dtype)


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
    return output.to(out_dtype)


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
    return output.to(out_dtype)


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
) -> torch.Tensor:
    if alpha is None:
        # F.linear fuses the bias add inside the GEMM epilogue.
        output = F.linear(A, B, bias)
    else:
        output = F.linear(A, B)
        output = output * alpha.to(dtype=output.dtype)
        if bias is not None:
            output = output + bias.to(dtype=output.dtype)
    return output.to(out_dtype)
