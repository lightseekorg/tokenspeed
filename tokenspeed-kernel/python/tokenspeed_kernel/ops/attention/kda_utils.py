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


@dataclass(frozen=True)
class KdaPrefillResult:
    """Results from a packed KDA prefill.

    Attributes:
        out: Packed output ``[1, total_tokens, heads, value_dim]``.
        final_state: One final recurrent state per packed sequence.
    """

    out: torch.Tensor
    final_state: torch.Tensor


@dataclass(frozen=True)
class KdaFusedDecodeResult:
    """Result from an optional pre-convolution KDA decode fusion.

    Attributes:
        out: Packed decode output ``[1, batch, heads, value_dim]``.
        output_norm_applied: Whether the selected kernel applied the output
            gate and RMSNorm, so the caller must not apply them again.
    """

    out: torch.Tensor
    output_norm_applied: bool
