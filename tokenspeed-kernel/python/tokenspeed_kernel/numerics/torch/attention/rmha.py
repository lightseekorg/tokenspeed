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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.rmha``:
attention with a learned relative-position logit bias."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def rel_attention_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    rel_logits: torch.Tensor | None,
    rel_extent: int,
    window_left: int,
    scale: float,
) -> torch.Tensor:
    """Masked FP32 logits ``[H, Sq, Sk]`` of the last ``Sq`` positions.

    Query row ``i`` sits at position ``Sk - Sq + i``; key ``j`` at distance
    ``d = pos - j`` is causal (``d >= 0``), inside a non-negative
    ``window_left`` (``d <= window_left``), and adds ``rel_logits[i, h, d]``
    when ``d < rel_extent``.

    Args:
        q: ``[Sq, H, D]``.
        k: ``[Sk, KV, D]``.
        rel_logits: ``[Sq, H, rel_extent]`` relative logits, or None.
        rel_extent: Distances with a relative logit.
        window_left: Sliding window, or ``-1`` for none.
        scale: Logit scale.
    """
    Sq, H, _ = q.shape
    Sk, KV, _ = k.shape
    k = k.repeat_interleave(H // KV, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    q_pos = torch.arange(Sq, device=q.device) + (Sk - Sq)
    kv_pos = torch.arange(Sk, device=q.device)
    dist = q_pos[:, None] - kv_pos[None, :]
    if rel_logits is not None:
        in_range = (dist >= 0) & (dist < rel_extent)
        idx = dist.clamp(0, rel_extent - 1)
        bias = rel_logits.float().gather(-1, idx.unsqueeze(1).expand(Sq, H, Sk))
        bias = torch.where(in_range.unsqueeze(1), bias, 0.0)
        logits = logits + bias.permute(1, 0, 2)
    mask = dist < 0
    if window_left >= 0:
        mask |= dist > window_left
    return logits.masked_fill_(mask.unsqueeze(0), float("-inf"))


@register_kernel(
    "attention",
    "rel_attention",
    name="torch_rel_attention",
    solution="torch",
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_rel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_logits: torch.Tensor | None,
    rel_extent: int,
    window_left: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative-bias attention of one sequence; see :func:`rel_attention_logits`.

    Returns:
        FP32 ``(out [Sq, H, D], lse [Sq, H])``, natural-log LSE.
    """
    logits = rel_attention_logits(q, k, rel_logits, rel_extent, window_left, scale)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    out = torch.einsum("hqk,khd->qhd", logits.softmax(-1), v.float())
    return out, logits.logsumexp(-1).transpose(0, 1)
