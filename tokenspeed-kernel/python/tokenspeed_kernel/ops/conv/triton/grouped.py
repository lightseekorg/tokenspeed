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

"""DFlash2's grouped dynamic depthwise convolution over flat blocks.

``y[t, j, k] = sum_i (base[i, j, k] + delta[t, i, j]) * x[t - i, j, k]``
for taps ``i`` with ``(t mod block_size) >= i``: rows are packed blocks of
``block_size`` tokens and a tap never reaches across a block boundary. The
tap mask follows from the row index alone, so the kernel derives it from
``program_id`` instead of materializing positions, coefficients or a padded
shift of the input.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["dflash2_grouped_conv"]


@triton.jit
def _grouped_conv_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    y_ptr,
    x_stride_row,
    y_stride_row,
    delta_stride_row,
    delta_stride_tap,
    base_stride_tap,
    num_channels,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < num_channels
    groups = offs // GROUP_SIZE
    position = row % BLOCK_SIZE

    delta_row = delta_ptr + row * delta_stride_row
    acc = (
        tl.load(base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        + tl.load(delta_row + groups, mask=mask, other=0.0).to(tl.float32)
    ) * tl.load(x_ptr + row * x_stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    for tap in tl.static_range(1, TAPS):
        if position >= tap:
            coefficient = tl.load(
                base_ptr + tap * base_stride_tap + offs, mask=mask, other=0.0
            ).to(tl.float32) + tl.load(
                delta_row + tap * delta_stride_tap + groups, mask=mask, other=0.0
            ).to(
                tl.float32
            )
            acc += coefficient * tl.load(
                x_ptr + (row - tap) * x_stride_row + offs, mask=mask, other=0.0
            ).to(tl.float32)
    tl.store(
        y_ptr + row * y_stride_row + offs, acc.to(y_ptr.dtype.element_ty), mask=mask
    )


def dflash2_grouped_conv(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Convolve packed blocks with per-row, per-group dynamic coefficients.

    Args:
        x: Input rows ``[T, C]`` packed as ``T // block_size`` blocks, with
            ``stride(-1) == 1``.
        delta: Per-row coefficient offsets ``[T, taps, num_groups]``; any
            strides, so a slice of the projection output works uncopied.
        base: Static per-channel taps ``[taps, C]`` with ``stride(-1) == 1``.
        block_size: Tokens per block; taps never cross a block boundary.
        group_size: Channels sharing one ``delta`` entry;
            ``C == num_groups * group_size``.

    Returns:
        ``[T, C]`` with the same dtype and layout as ``x``.

    Raises:
        ValueError: The operands are not the shape, layout or device the
            kernel indexes with.
    """
    if x.ndim != 2 or delta.ndim != 3 or base.ndim != 2:
        raise ValueError(
            f"expected x [T, C], delta [T, taps, groups], base [taps, C]; got "
            f"{tuple(x.shape)}, {tuple(delta.shape)}, {tuple(base.shape)}"
        )
    num_rows, num_channels = x.shape
    taps = base.shape[0]
    if delta.shape[0] != num_rows or delta.shape[1] != taps:
        raise ValueError(
            f"delta {tuple(delta.shape)} must be [{num_rows}, {taps}, groups]"
        )
    if base.shape[1] != num_channels:
        raise ValueError(f"base {tuple(base.shape)} must be [{taps}, {num_channels}]")
    if delta.shape[2] * group_size != num_channels:
        raise ValueError(
            f"{delta.shape[2]} groups of {group_size} do not cover {num_channels} "
            "channels"
        )
    if not 1 <= taps <= block_size:
        raise ValueError(f"taps must be in [1, block_size={block_size}], got {taps}")
    if x.stride(1) != 1 or base.stride(1) != 1:
        raise ValueError("x and base rows must be contiguous")
    if not (x.is_cuda and delta.is_cuda and base.is_cuda):
        raise ValueError("dflash2_grouped_conv requires CUDA tensors")

    y = torch.empty_like(x)
    if num_rows == 0:
        return y

    block_c = min(1024, triton.next_power_of_2(num_channels))
    _grouped_conv_kernel[(num_rows, triton.cdiv(num_channels, block_c))](
        x,
        delta,
        base,
        y,
        x.stride(0),
        y.stride(0),
        delta.stride(0),
        delta.stride(1),
        base.stride(0),
        num_channels,
        BLOCK_SIZE=block_size,
        GROUP_SIZE=group_size,
        TAPS=taps,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return y
