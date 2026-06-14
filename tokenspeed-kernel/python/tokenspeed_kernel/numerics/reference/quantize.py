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

"""Reference fp8 quantization kernels.

Each reference returns ``qweight.float()`` — the fp8 quantized values cast back
to fp32 for comparison. The scale tensor's layout differs across producers
(SM90 vs SM100, row-major vs column-major) so we don't compare it directly;
the qweight values tell us whether the per-group statistics + rounding
agree.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_DTYPE = Platform.get().fp8e4m3fn.dtype
_FP8_FINFO = torch.finfo(_FP8_DTYPE)
_FP8_MAX = _FP8_FINFO.max  # 448 for e4m3fn


def _quantize_fp8(x_fp32: torch.Tensor, max_abs: torch.Tensor) -> torch.Tensor:
    """scale = max_abs/FP8_MAX (clamped), quantize → fp8, cast back to fp32."""
    scale = (max_abs / _FP8_MAX).clamp(min=1e-10)
    return (x_fp32 / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE).float()


@register_kernel(
    "quantize",
    "fp8_token_group_128",
    name="torch_fp8_token_group_128",
    solution="reference",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={},
    priority=10,
    tags={"determinism", "portability"},
)
def torch_fp8_token_group_128(x: torch.Tensor) -> torch.Tensor:
    """Per-token grouped fp8 quantization with group size 128."""
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    M, K = x.shape
    assert K % 128 == 0, f"K={K} must be divisible by group_size=128"
    x_grouped = x.float().view(M, K // 128, 128)
    max_abs = x_grouped.abs().amax(dim=-1, keepdim=True)
    return _quantize_fp8(x_grouped, max_abs).view(M, K)


@register_kernel(
    "quantize",
    "fp8_token",
    name="torch_fp8_token",
    solution="reference",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={},
    priority=10,
    tags={"determinism", "portability"},
)
def torch_fp8_token(x: torch.Tensor) -> torch.Tensor:
    """Per-token fp8 quantization (one scale per row)."""
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    x_fp32 = x.float()
    return _quantize_fp8(x_fp32, x_fp32.abs().amax(dim=-1, keepdim=True))


@register_kernel(
    "quantize",
    "fp8_tensor",
    name="torch_fp8_tensor",
    solution="reference",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={},
    priority=10,
    tags={"determinism", "portability"},
)
def torch_fp8_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor fp8 quantization (one scalar scale for the whole tensor)."""
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    x_fp32 = x.float()
    return _quantize_fp8(x_fp32, x_fp32.abs().amax())
