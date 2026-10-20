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

"""PyTorch references for ``tokenspeed_kernel.ops.communication``: the
normalization math fused around all-reduces (the collectives themselves are
checked against NCCL)."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "communication",
    "sandwich_norm",
    name="torch_sandwich_norm",
    solution="torch",
    signatures=format_signatures(
        "stream", "dense", {torch.bfloat16, torch.float32, torch.float64}
    ),
    priority=Priority.REFERENCE,
)
def torch_sandwich_norm(
    stream: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    weight: torch.Tensor,
    *,
    post_eps: float,
    eps: float,
    x_scale: float,
    residual_scale: float,
    round_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The residual boundary between two RMSNorms, in FP64.

    ``x = rmsnorm(stream) * post_weight * x_scale``; ``r = residual *
    residual_scale``; ``res = x + r``; ``out = rmsnorm(res) * weight``.
    Epsilons and scales are taken as FP32 values. ``round_bf16`` rounds every
    stage through FP32 to BF16, as the kernels round FP32 values; otherwise
    nothing rounds.

    Returns:
        ``(out, res, x, r)``: BF16 when rounding, else FP64.
    """

    def fp32(value: float) -> float:
        return float(torch.tensor(value, dtype=torch.float32))

    def rounded(values: torch.Tensor) -> torch.Tensor:
        return values.float().to(torch.bfloat16) if round_bf16 else values

    def norm(values: torch.Tensor, gamma: torch.Tensor, e: float) -> torch.Tensor:
        v64 = values.double()
        rstd = torch.rsqrt(v64.square().mean(-1, keepdim=True) + fp32(e))
        return rounded(v64 * rstd * gamma.double())

    x = rounded(norm(stream, post_weight, post_eps).double() * fp32(x_scale))
    r = rounded(residual.double() * fp32(residual_scale))
    res = rounded(x.double() + r.double())
    return norm(res, weight, eps), res, x, r
