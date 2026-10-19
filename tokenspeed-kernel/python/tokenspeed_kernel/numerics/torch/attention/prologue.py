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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.prologue``: the
per-head norm and rotary embedding between projection and attention."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def mrope_pair_rows(
    section: tuple[int, ...], interleaved: bool, half: int
) -> list[int]:
    """The T, H or W position row of each rotary pair (Qwen2-VL and Qwen3-VL).

    Contiguous sections take ``section[r]`` consecutive pairs from row ``r``;
    interleaved ones alternate T, H, W per pair while H and W pairs remain.
    """
    if not interleaved:
        return [row for row, pairs in enumerate(section) for _ in range(pairs)]
    _, h, w = section
    return [
        1 if j % 3 == 1 and j < 3 * h else 2 if j % 3 == 2 and j < 3 * w else 0
        for j in range(half)
    ]


@register_kernel(
    "attention",
    "head_norm_rope",
    name="torch_head_norm_rope",
    solution="torch",
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_head_norm_rope(
    x: torch.Tensor,
    head_dim: int,
    *,
    norm_weight: torch.Tensor | None,
    norm_weight_offset: float,
    norm_eps: float,
    cos_sin_cache: torch.Tensor | None,
    positions: torch.Tensor | None,
    neox: bool,
    mrope_section: tuple[int, ...] | None,
    mrope_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 per-head RMSNorm, then rotary embedding, of
    ``[tokens, heads * head_dim]`` rows.

    Args:
        x: Rows to transform.
        head_dim: Width of each head.
        norm_weight: Stored ``[head_dim]`` norm weight, or None for no norm.
        norm_weight_offset: Added to the weight (1.0 for Gemma-style).
        norm_eps: Added to the mean square.
        cos_sin_cache: ``[max_position, rotary_dim]`` packed ``[cos | sin]``,
            or None for no rotation.
        positions: ``[tokens]``, or ``[3, tokens]`` T/H/W rows for M-RoPE.
        neox: Pair ``i`` with ``i + rotary_dim / 2``; else adjacent pairs.
        mrope_section: Pairs per T/H/W row for 2-D ``positions``.
        mrope_interleaved: Whether the M-RoPE rows alternate per pair.

    Returns:
        ``(values, magnitude)`` in FP64, flattened like ``x``: the result,
        and the magnitude of the terms each output sums (for error bounds).
    """
    x = x.double().reshape(x.shape[0], -1, head_dim)
    if norm_weight is not None:
        rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
        x = x * rms * (norm_weight.double() + norm_weight_offset)
    magnitude = x.abs()
    if cos_sin_cache is not None:
        half = cos_sin_cache.shape[-1] // 2
        table = cos_sin_cache.double()
        if positions.ndim == 2:
            pair_rows = mrope_pair_rows(mrope_section, mrope_interleaved, half)
            rows = torch.tensor(pair_rows * 2, device=x.device)
            cols = torch.arange(2 * half, device=x.device)
            cos_sin = table[positions[rows].T, cols]
        else:
            cos_sin = table[positions]
        cos, sin = cos_sin[:, None, :half], cos_sin[:, None, half:]
        if neox:
            i1, i2 = slice(0, half), slice(half, 2 * half)
        else:
            i1, i2 = slice(0, 2 * half, 2), slice(1, 2 * half, 2)
        x1, x2 = x[..., i1].clone(), x[..., i2].clone()
        x[..., i1] = x1 * cos - x2 * sin
        x[..., i2] = x2 * cos + x1 * sin
        magnitude[..., i1] = x1.abs() * cos.abs() + x2.abs() * sin.abs()
        magnitude[..., i2] = x2.abs() * cos.abs() + x1.abs() * sin.abs()
    return x.flatten(1), magnitude.flatten(1)
