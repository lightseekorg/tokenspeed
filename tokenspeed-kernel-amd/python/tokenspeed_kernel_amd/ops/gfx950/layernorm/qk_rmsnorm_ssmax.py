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

"""Beam TP8 query normalization with the positional scale on gfx950."""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon


def _launch_metadata(grid, kernel, args):
    rows = args["q_ptr"].shape[0]
    return {
        "name": kernel.name,
        "bytes": rows * 768 * 4 + 128 * 2 + rows * args["positions_ptr"].element_size(),
        "flops32": rows * 6 * 128 * 6,
    }


@gluon.jit(launch_metadata=_launch_metadata)
def gluon_qk_rmsnorm_ssmax_gfx950(
    q_ptr,
    weight_ptr,
    positions_ptr,
    output_ptr,
    q_row_stride,
    eps,
):
    token = gl.program_id(0).to(gl.int64)
    head = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    offsets = gl.arange(0, 128, layout=layout)
    x = gl.load(q_ptr + token * q_row_stride + head * 128 + offsets).to(gl.float32)
    weight = gl.load(weight_ptr + offsets).to(gl.float32)
    variance = gl.sum(x * x, axis=0) * 0.0078125
    normalized = x * gl.rsqrt(variance + eps)
    # The portable producer stores this value before positional scaling.
    affine = (normalized * weight).to(gl.bfloat16).to(gl.float32)
    position = gl.load(positions_ptr + token).to(gl.float32)
    ratio = (position + 1.0) * 0.00048828125
    scale = gl.extra.libdevice.exp(0.1 * gl.extra.libdevice.log1p(ratio))
    gl.store(output_ptr + token * 768 + head * 128 + offsets, affine * scale)


def launch_gluon_qk_rmsnorm_ssmax_gfx950(q, weight, positions, eps):
    """Write fresh contiguous BF16 queries; the public entry validates metadata."""
    output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    gluon_qk_rmsnorm_ssmax_gfx950[(q.shape[0], 6)](
        q,
        weight,
        positions,
        output,
        q.stride(0),
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
