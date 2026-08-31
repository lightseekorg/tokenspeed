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

"""PyTorch reference implementations for gated-residual hyperconnections."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_MIX_SIGNATURES = format_signatures(
    ("normalized", "projection_weight", "up_weight"), "dense", _DTYPES
)
_EPILOGUE_SIGNATURES = format_signatures(("gate", "normalized"), "dense", _DTYPES)
_COMBINE_SIGNATURES = format_signatures(
    ("block_output", "residual", "inject_logits"), "dense", _DTYPES
)


@register_kernel(
    "hyperconnection",
    "mix",
    name="torch_hyperconnection_mix",
    solution="torch",
    signatures=_MIX_SIGNATURES,
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_hyperconnection_mix(
    normalized: torch.Tensor,
    projection_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    projection_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference fused-projection gated-residual mix."""
    projected = F.linear(normalized, projection_weight)
    down = F.silu(projected[:, :lowrank] * projection_scale)
    gate = F.linear(down, up_weight)
    mixed = torch_hyperconnection_mix_epilogue(gate, normalized, hc_count, hidden_size)
    inject = (
        projected[:, lowrank:] * projection_scale
        if projection_weight.shape[0] != lowrank
        else None
    )
    return mixed, inject


@register_kernel(
    "hyperconnection",
    "mix_epilogue",
    name="torch_hyperconnection_mix_epilogue",
    solution="torch",
    signatures=_EPILOGUE_SIGNATURES,
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_hyperconnection_mix_epilogue(
    gate: torch.Tensor,
    normalized: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    """Reference sigmoid-weighted mean across residual branches."""
    weights = torch.sigmoid(gate).unflatten(-1, (hc_count, hidden_size))
    branches = normalized.unflatten(-1, (hc_count, hidden_size))
    return (weights * branches).mean(dim=-2).to(normalized.dtype)


@register_kernel(
    "hyperconnection",
    "combine",
    name="torch_hyperconnection_combine",
    solution="torch",
    signatures=_COMBINE_SIGNATURES,
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_hyperconnection_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_logits: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    """Reference gated residual-stream update."""
    inject = 2.0 * torch.sigmoid(inject_logits)
    combined = residual.unflatten(-1, (hc_count, hidden_size)) + (
        block_output.unsqueeze(-2) * inject.unsqueeze(-1)
    )
    return combined.flatten(-2).to(block_output.dtype)


__all__ = [
    "torch_hyperconnection_combine",
    "torch_hyperconnection_mix",
    "torch_hyperconnection_mix_epilogue",
]
