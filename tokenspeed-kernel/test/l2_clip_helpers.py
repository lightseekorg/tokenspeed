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

"""The standalone Triton L2 clip, the bitwise oracle of the fused head norms.

Serving clips attention heads only inside ``rel_mha_head_norms``; this kernel
runs the same ``_l2_clipped`` arithmetic one (token, head) row per program.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.layernorm._triton.precision import _l2_clipped


@triton.jit
def _l2_clip_kernel(
    X,
    Scale,
    OutputScale,
    Out,
    x_stride_t,
    x_stride_h,
    threshold,
    eps,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_OUTPUT_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    value = tl.load(
        X + token * x_stride_t + head * x_stride_h + cols, cols < WIDTH, 0
    ).to(tl.float32)
    if HAS_SCALE:
        value = value * tl.load(Scale + head * WIDTH + cols, cols < WIDTH, 0).to(
            tl.float32
        )
    output = _l2_clipped(value, threshold, eps).to(Out.dtype.element_ty).to(tl.float32)
    if HAS_OUTPUT_SCALE:
        output = output * tl.load(OutputScale + token).to(tl.float32)
    tl.store(Out + (token * HEADS + head) * WIDTH + cols, output, cols < WIDTH)


def triton_l2_clip(
    x: torch.Tensor,
    threshold: float,
    eps: float,
    scale: torch.Tensor | None,
    output_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Clip possibly strided token/head rows and preserve intermediate rounding."""
    tokens, heads, width = x.shape
    if width > 65536:
        raise ValueError("L2 clipping width exceeds the portable Triton row limit")
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if tokens:
        _l2_clip_kernel[(tokens, heads)](
            x,
            scale,
            output_scale,
            output,
            x.stride(0),
            x.stride(1),
            threshold,
            eps,
            heads,
            width,
            scale is not None,
            output_scale is not None,
            triton.next_power_of_2(width),
            enable_fp_fusion=False,
        )
    return output
