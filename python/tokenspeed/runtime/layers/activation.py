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

"""Fused operators for activation layers."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.utils import (
    get_colorful_logger,
)
from tokenspeed.runtime.utils.pdl import pdl_enabled

_is_amd = current_platform().is_amd

if not _is_amd:
    from tokenspeed_kernel.ops.activation.flashinfer import (
        silu_and_mul,
    )

logger = get_colorful_logger(__name__)


class SiluAndMul(torch.nn.Module):

    def forward(self, x: torch.Tensor, fp8_out: bool = False) -> torch.Tensor:
        if _is_amd:
            d = x.shape[-1] // 2
            return F.silu(x[..., :d]) * x[..., d:]
        else:

            def get_tma_aligned_scale(x):
                aligned_size = (x.shape[-2] + 3) // 4 * 4
                x_s = torch.empty(
                    x.shape[:-2] + (x.shape[-1] // 128, aligned_size),
                    device=x.device,
                    dtype=torch.float32,
                ).permute(-1, -2)[: x.shape[-2], :]
                return x_s

            d = x.shape[-1] // 2
            output_shape = x.shape[:-1] + (d,)
            if fp8_out:
                out = torch.empty(
                    output_shape, dtype=torch.float8_e4m3fn, device=x.device
                )
                scale = get_tma_aligned_scale(out)
                from tokenspeed_kernel.ops.activation.cuda import (
                    silu_and_mul_fuse_block_quant,
                )

                out, scale = silu_and_mul_fuse_block_quant(
                    x, scale, out, enable_pdl=pdl_enabled()
                )
                return out, scale
            else:
                out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
                silu_and_mul(x, out, enable_pdl=pdl_enabled())
                return out


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def compute_swiglu(gelu, linear, scale, alpha, limit):
    gelu = gelu.to(tl.float32) * scale
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32) * scale
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)

    s = gelu / (1 + tl.exp(-alpha * gelu))

    return tl.fma(s, linear, s)  # (s * (linear + 1))


@triton.jit(repr=lambda _: "_swiglu")
def swiglu_fn(input, alpha, limit, exclusive_sum, local_num_experts):
    begin = exclusive_sum[0]
    end = exclusive_sum[local_num_experts]
    input = input[begin:end]

    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    return compute_swiglu(gelu, linear, 1.0, alpha, limit)


@dataclass
class SwigluArg:
    alpha: float
    limit: float
