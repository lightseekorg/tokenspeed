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

"""Sigmoid gating of attention outputs."""

from __future__ import annotations

import torch


def apply_attention_output_gate(
    output: torch.Tensor,
    gate: torch.Tensor,
    gate_bias: torch.Tensor,
    floor: float,
    temperature: float,
) -> torch.Tensor:
    """Apply a biased sigmoid gate with a floor and temperature.

    Args:
        output: Attention output shaped ``[..., heads, head_dim]``.
        gate: Projected gate logits with the same shape as output.
        gate_bias: One learned scalar per head, broadcast over head_dim.
        floor: Lower bound in ``floor + (1 - floor) * sigmoid(logits)``.
        temperature: Positive divisor of the gate logits after adding bias.

    Returns:
        A new tensor with the output's shape and dtype. Bias addition, sigmoid,
        and multiplication use FP32; the result is cast once to the output dtype.
    """
    if gate.ndim < 2 or output.shape != gate.shape:
        raise ValueError("attention output and gate must share head and width axes")
    if gate_bias.ndim != 1 or gate_bias.shape[0] != gate.shape[-2]:
        raise ValueError("gate_bias must contain one scalar per attention head")
    if temperature <= 0:
        raise ValueError(
            f"attention gate temperature must be positive, got {temperature}"
        )
    bias_shape = (1,) * (gate.ndim - 2) + (gate.shape[-2], 1)
    logits = (gate.float() + gate_bias.float().reshape(bias_shape)) / float(temperature)
    multiplier = float(floor) + (1.0 - float(floor)) * torch.sigmoid(logits)
    return (output.float() * multiplier).to(output.dtype)
