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

"""Ascend rotary embedding operator."""

from __future__ import annotations

import torch
import torch_npu


def apply_rope(
    *,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    q_rope_out: torch.Tensor | None = None,
    k_rope_out: torch.Tensor | None = None,
) -> None:
    """Apply RoPE to Q and K, writing into outputs or the inputs in place."""
    q_out, k_out = torch_npu.npu_mrope(
        positions,
        q,
        k,
        cos_sin_cache.to(q.dtype),
        head_size,
        mrope_section=[0, 0, 0],
        rotary_mode="half" if is_neox else "interleave",
    )
    (q if q_rope_out is None else q_rope_out).copy_(q_out)
    (k if k_rope_out is None else k_rope_out).copy_(k_out)


__all__ = ["apply_rope"]
