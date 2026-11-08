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

"""Weight layouts the NVFP4 GEMM+SwiGLU kernel reads, built with torch alone."""

from __future__ import annotations

import torch


def swizzle_blockscale_2d(scales):
    """Swizzle 2D FP8 block scales for CUTLASS."""
    M, K = scales.shape

    def round_up(x, m):
        return (x + m - 1) // m * m

    M_padded = round_up(M, 128)
    K_padded = round_up(K, 4)
    padded = torch.zeros((M_padded, K_padded), dtype=scales.dtype, device=scales.device)
    padded[:M, :K] = scales
    rows, cols = padded.shape
    padded = padded.reshape(rows // 128, 4, 32, cols // 4, 4)
    padded = padded.permute((0, 3, 2, 1, 4))
    return padded.contiguous().reshape(M_padded, K_padded)


def interleave_linear_and_gate(
    tensor: torch.Tensor,
    group_size: int = 64,
    dim: int = 0,
) -> torch.Tensor:
    """Interleave ``[linear all][gate all]`` as ``[linear chunk][gate chunk]``.

    This matches the FC1 GEMM+SwiGLU preprocessing layout expected by
    ``cute_dsl_nvfp4_dense_gemm_swiglu_blackwell``.
    """
    if tensor.ndim == 0:
        raise ValueError("expected a tensor with at least one dimension")

    dim = dim % tensor.ndim
    sizes = tensor.size()
    dim_size = sizes[dim]
    if dim_size % (group_size * 2) != 0:
        raise ValueError(
            f"dimension {dim} size {dim_size} must be divisible by "
            f"2 * group_size={2 * group_size}"
        )

    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    return (
        tensor.reshape(
            *prev_sizes,
            2,
            dim_size // (group_size * 2),
            group_size,
            *post_sizes,
        )
        .transpose(dim, dim + 1)
        .reshape(*sizes)
        .contiguous()
    )
