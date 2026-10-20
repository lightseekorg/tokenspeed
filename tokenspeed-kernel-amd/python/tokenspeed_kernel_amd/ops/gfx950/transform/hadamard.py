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

"""GFX950 Gluon transforms."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = ["launch_gluon_hadamard_transform_128_gfx950"]


def _hadamard_launch_metadata(grid, kernel, args):
    elements = args["x"].numel()
    return {
        "name": kernel.name,
        "flops32": elements * 8,
        "bytes": elements * args["x"].element_size() * 2,
    }


@gluon.constexpr_function
def _hadamard_128_layout():
    # One wave owns a row. Each lane retains two FP32 values throughout the
    # butterfly, avoiding shared memory and cross-wave synchronization.
    return gl.BlockedLayout([2], [64], [1], [0])


@gluon.jit
def _hadamard_128_butterfly_stage(
    values,
    groups: gl.constexpr,
    stride: gl.constexpr,
):
    values = values.reshape((groups, 2, stride)).permute((0, 2, 1))
    pair_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2],
        [groups, stride, 1],
        [1, 1, 1],
        [2, 1, 0],
    )
    values = gl.convert_layout(values, pair_layout)
    left, right = gl.split(values)
    values = gl.join(left + right, left - right)
    return values.permute((0, 2, 1)).reshape((128,))


@gluon.jit(launch_metadata=_hadamard_launch_metadata)
def gluon_hadamard_transform_128_gfx950(
    x,
    out,
    scale: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, 128, layout=_hadamard_128_layout())
    values = gl.load(x + row * 128 + offsets).to(gl.float32)

    # This order is deliberately different from the conventional increasing
    # stride order. It reproduces the portable kernel's FP32 reduction tree.
    values = _hadamard_128_butterfly_stage(values, 8, 8)
    values = _hadamard_128_butterfly_stage(values, 16, 4)
    values = _hadamard_128_butterfly_stage(values, 32, 2)
    values = _hadamard_128_butterfly_stage(values, 64, 1)
    values = _hadamard_128_butterfly_stage(values, 4, 16)
    values = _hadamard_128_butterfly_stage(values, 2, 32)
    values = _hadamard_128_butterfly_stage(values, 1, 64)

    values = gl.convert_layout(values, _hadamard_128_layout())
    gl.store(out + row * 128 + offsets, values * scale)


def launch_gluon_hadamard_transform_128_gfx950(
    x: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Apply the length-128 transform to a contiguous GFX950 BF16 tensor."""
    shape = x.shape
    x_2d = x.reshape(-1, 128)
    out = torch.empty_like(x_2d)
    if x_2d.shape[0] == 0:
        return out.reshape(shape)
    gluon_hadamard_transform_128_gfx950[(x_2d.shape[0],)](
        x_2d,
        out,
        scale=float(scale),
        num_warps=1,
    )
    return out.reshape(shape)
