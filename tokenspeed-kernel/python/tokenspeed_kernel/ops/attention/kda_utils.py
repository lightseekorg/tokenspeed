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

"""Shared result types for registered Kimi Delta Attention kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KdaGatedNormRequest:
    """Attempt-and-verify handoff of the KDA output gated RMSNorm.

    The caller stashes the norm operands before invoking the fused decode;
    an implementation that can fold ``rmsnorm(o) * weight * sigmoid(gate)``
    into its epilogue does so and sets ``consumed``. The caller applies the
    host-side norm only when the stash comes back unconsumed, so norm
    eligibility stays entirely kernel-side (no env knobs, no capability
    negotiation).

    Attributes:
        weight: ``[head_dim]`` RMSNorm weight (bf16 or fp32).
        gate: ``[tokens, heads * head_dim]`` raw gate logits (sigmoid is
            applied by whoever performs the norm); may be a column slice of
            the merged projection output (strided rows, dense last dim).
        eps: RMSNorm epsilon.
        consumed: Set by the kernel wrapper when the norm was fused; the
            decision is shape/dtype-deterministic, so it is stable across
            CUDA-graph capture and replay.
    """

    weight: torch.Tensor
    gate: torch.Tensor
    eps: float
    consumed: bool = False


@dataclass(frozen=True)
class KdaPrefillResult:
    """Results from a packed KDA prefill.

    Attributes:
        out: Packed output ``[1, total_tokens, heads, value_dim]``.
        final_state: One final recurrent state per packed sequence.
    """

    out: torch.Tensor
    final_state: torch.Tensor
