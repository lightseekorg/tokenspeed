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

"""PyTorch references for ``tokenspeed_kernel.ops.ple``."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "ple",
    "gate_norm",
    name="torch_ple_gate_norm",
    solution="torch",
    signatures=format_signatures(
        "key", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_ple_gate_norm(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_gamma: torch.Tensor,
    query_gamma: torch.Tensor,
    out_gamma: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer-embedding gate of four streams, then its RMSNorm.

    Key and query streams are RMSNormed (rounded to the input dtype); their
    signed-sqrt scaled dot product gates ``value`` through a sigmoid.

    Args:
        key, query: ``[tokens, 4 * head_dim]``.
        value: ``[tokens, head_dim]``.
        key_gamma, query_gamma, out_gamma: ``[4 * head_dim]`` norm weights.

    Returns:
        ``(gated, normed)``, each ``[tokens, 4 * head_dim]`` in the input dtype.
    """
    dtype = key.dtype

    def norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x = x.float().reshape(-1, 4, head_dim)
        return (
            x
            * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
            * gamma.float().reshape(1, 4, head_dim)
        ).to(dtype)

    k, q = norm(key, key_gamma), norm(query, query_gamma)
    dot = (k.float() * q.float()).sum(-1, keepdim=True) / head_dim**0.5
    gate = torch.sign(dot) * torch.sqrt(torch.clamp(dot.abs(), min=1e-6))
    gated = (torch.sigmoid(gate) * value.float()[:, None, :]).to(dtype).flatten(1)
    return gated, norm(gated, out_gamma).flatten(1)
