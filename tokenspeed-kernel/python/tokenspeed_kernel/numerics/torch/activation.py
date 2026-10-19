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

"""PyTorch references for ``tokenspeed_kernel.ops.activation``.

Gated activations split the last dimension into ``[gate | up]`` halves. Math
runs in FP32 (FP64 for FP64 inputs) and rounds once to the input dtype.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.quantization import (
    quantize_fp8_groups,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FLOAT_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


def _halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[gate | up]`` halves in FP32, or FP64 for FP64 inputs."""
    gate, up = x.chunk(2, dim=-1)
    if x.dtype == torch.float64:
        return gate, up
    return gate.float(), up.float()


@register_kernel(
    "activation",
    "sigmoid_mul",
    name="torch_sigmoid_mul",
    solution="torch",
    signatures=format_signatures(("x", "gate"), "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_sigmoid_mul(
    x: torch.Tensor, gate: torch.Tensor, head_bias: torch.Tensor | None
) -> torch.Tensor:
    """``x * sigmoid(gate + head_bias)``, with one bias per head of ``gate``.

    ``gate`` is ``[tokens, width]`` or ``[tokens, heads, head_dim]`` for
    ``x`` of ``[tokens, heads * head_dim]``; ``head_bias`` is ``[heads]``.
    """
    z = gate.float()
    if head_bias is not None:
        z = z.reshape(x.shape[0], head_bias.shape[0], -1) + head_bias.float()[:, None]
    return (x.float() * (1.0 / (1.0 + torch.exp(-z))).reshape(x.shape)).to(x.dtype)


@register_kernel(
    "activation",
    "silu_and_mul",
    name="torch_silu_and_mul",
    solution="torch",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_silu_and_mul(x: torch.Tensor, limit: float | None) -> torch.Tensor:
    """``silu(gate) * up``; a ``limit`` clamps gate above and up on both sides."""
    gate, up = _halves(x)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return (F.silu(gate) * up).to(x.dtype)


def eager_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` as eager ops in ``x``'s dtype, each rounding once.

    Unlike :func:`torch_silu_and_mul`, ``silu(gate)`` rounds to ``x``'s dtype
    before the multiply, as kernels that store it at that width do.
    """
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


@register_kernel(
    "activation",
    "swiglu_oai",
    name="torch_swiglu_oai",
    solution="torch",
    signatures=format_signatures("gate_up", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_swiglu_oai(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """GPT-OSS SwiGLU: ``g * sigmoid(alpha * g) * (u + 1)`` on clamped halves."""
    return swiglu(gate_up, alpha=alpha, limit=limit, beta=1.0).to(gate_up.dtype)


def swiglu(
    gate_up: torch.Tensor, *, alpha: float, limit: float | None, beta: float
) -> torch.Tensor:
    """``g * sigmoid(alpha * g) * (u + beta)`` in FP32 (or FP64 inputs' dtype).

    A ``limit`` clamps gate above and up on both sides first.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    if gate.dtype != torch.float64:
        gate, up = gate.float(), up.float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


def deinterleave_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    """``[g0, u0, g1, u1, ...]`` columns to the ``[gate | up]`` halves layout."""
    return torch.cat((gate_up[..., 0::2], gate_up[..., 1::2]), dim=-1)


def situ(x: torch.Tensor, beta: float) -> torch.Tensor:
    """SiTU, the soft-capped SiLU ``beta * tanh(x / beta) * sigmoid(x)``."""
    return beta * torch.tanh(x / beta) * torch.sigmoid(x)


@register_kernel(
    "activation",
    "situ_and_mul",
    name="torch_situ_and_mul",
    solution="torch",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_situ_and_mul(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    """``situ(gate) * up``; a ``linear_beta`` soft-caps up with ``tanh``."""
    gate, up = _halves(x)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ(gate, beta) * up).to(x.dtype)


@register_kernel(
    "activation",
    "fused_gate_sigmoid_mul_add",
    name="torch_fused_gate_sigmoid_mul_add",
    solution="torch",
    signatures=format_signatures("hidden_states", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """``final + sigmoid(hidden_states @ gate_weight) * shared_output``."""
    gate = (hidden_states.float() @ gate_weight.float().unsqueeze(1)).sigmoid()
    out = final_hidden_states.float() + gate * shared_output.float()
    return out.to(final_hidden_states.dtype)


@register_kernel(
    "activation",
    "fused_swiglu_fp8_ue8m0",
    name="torch_fused_swiglu_fp8_ue8m0",
    solution="torch",
    signatures=format_signatures("gate_up", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_fused_swiglu_fp8_ue8m0(
    gate_up: torch.Tensor,
    swiglu_limit: float,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``g * sigmoid(alpha * g) * (u + beta)`` quantized to FP8 per 128 columns.

    A positive ``swiglu_limit`` clamps gate above and up on both sides.

    Returns:
        ``(codes, scales)``: FP8 E4M3 codes shaped like one half of
        ``gate_up`` and FP32 power-of-two scales ``[rows, columns / 128]``
        (unpacked, unlike the kernel's packed UE8M0 bytes).
    """
    y = swiglu(
        gate_up,
        alpha=swiglu_alpha,
        limit=swiglu_limit if swiglu_limit > 0 else None,
        beta=swiglu_beta,
    )
    return quantize_fp8_groups(y, 128, power_of_two_scale=True)


@register_kernel(
    "activation",
    "relu2",
    name="torch_relu2",
    solution="torch",
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_relu2(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU, rounded once to ``x``'s dtype."""
    return torch.relu(x).square()
