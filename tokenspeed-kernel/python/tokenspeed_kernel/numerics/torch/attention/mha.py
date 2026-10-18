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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.mha``, and the
dense attention core the other attention references build on."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def gather_pages(
    cache: torch.Tensor, page_table: torch.Tensor, length: int
) -> torch.Tensor:
    """The first ``length`` rows of one request's paged cache.

    Args:
        cache: ``[num_pages, page_size, ...]`` paged rows.
        page_table: ``[max_pages]`` page ids of the request.
        length: Rows to return.

    Returns:
        ``[length, ...]`` rows in cache order.
    """
    pages = page_table[: math.ceil(length / cache.shape[1])].long()
    return cache[pages].flatten(0, 1)[:length]


@register_kernel(
    "attention",
    "dense_attention",
    name="torch_dense_attention",
    solution="torch",
    signatures=format_signatures(
        ("q", "k", "v"),
        "dense",
        {torch.bfloat16, torch.float16, torch.float32, torch.float64},
    ),
    priority=Priority.REFERENCE,
)
def torch_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal_offset: int | None,
    window_left: int | None,
    sinks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped-query attention of one sequence over dense keys and values.

    Query row ``i`` sees key ``j`` when ``j <= causal_offset + i`` (all keys
    for ``causal_offset=None``) and, with ``window_left``, when
    ``causal_offset + i - j <= window_left``. A query row that sees no key
    returns zeros and ``-inf`` LSE. A per-head ``sinks`` logit joins each
    softmax denominator without a value.

    Args:
        q: ``[Lq, Hq, D]``.
        k: ``[Lk, Hkv, D]``; ``Hq`` must be a multiple of ``Hkv``.
        v: ``[Lk, Hkv, Dv]``.
        scale: Logit scale.
        causal_offset: Last key position row 0 sees, or None for no mask.
        window_left: Sliding-window size to the left, or None.
        sinks: Optional ``[Hq]`` attention-sink logits.

    Returns:
        ``(out [Lq, Hq, Dv], lse [Lq, Hq])`` in FP32 (FP64 for FP64 inputs);
        the LSE is natural-log and excludes the sinks.
    """
    compute = torch.float64 if q.dtype == torch.float64 else torch.float32
    group = q.shape[1] // k.shape[1]
    keys = k.to(compute).repeat_interleave(group, dim=1)
    values = v.to(compute).repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.to(compute), keys) * scale
    if causal_offset is not None:
        rows = torch.arange(q.shape[0], device=q.device)[:, None] + causal_offset
        cols = torch.arange(k.shape[0], device=q.device)[None, :]
        visible = cols <= rows
        if window_left is not None:
            visible &= rows - cols <= window_left
        scores = scores.masked_fill(~visible, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    if sinks is not None:
        sink = sinks.to(compute)[:, None, None].expand(-1, q.shape[0], 1)
        weights = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
    else:
        weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights)
    out = torch.einsum("hqk,khd->qhd", weights, values)
    return out, lse.transpose(0, 1)


def torch_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Causal attention of one sequence with ``1 / sqrt(D)`` scaling, in
    ``q``'s dtype (``[L, H, D]`` layouts); see :func:`torch_dense_attention`."""
    out, _ = torch_dense_attention(
        q,
        k,
        v,
        scale=q.shape[-1] ** -0.5,
        causal_offset=0,
        window_left=None,
        sinks=None,
    )
    return out.to(q.dtype)


@register_kernel(
    "attention",
    "merge_attention_states",
    name="torch_merge_attention_states",
    solution="torch",
    signatures=format_signatures(
        ("v_a", "v_b"), "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_merge_attention_states(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    lse_scale_log2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention partials over disjoint keys by their LSE.

    The math runs in base 2: ``lse_scale_log2`` converts the input LSE base
    to log2 (``log2(e)`` for natural-log LSEs, 1 for base-2) and back.

    Args:
        v_a, v_b: ``[..., heads, D]`` normalized partial outputs.
        s_a, s_b: ``[..., heads]`` their LSEs.

    Returns:
        ``(merged values in v_a's dtype, merged FP32 LSE)``.
    """
    s_a_log2 = s_a.float() * lse_scale_log2
    s_b_log2 = s_b.float() * lse_scale_log2
    s_max = torch.maximum(s_a_log2, s_b_log2)
    w_a = torch.exp2(s_a_log2 - s_max)
    w_b = torch.exp2(s_b_log2 - s_max)
    sum_w = w_a + w_b
    v_merged = (
        w_a.unsqueeze(-1) * v_a.float() + w_b.unsqueeze(-1) * v_b.float()
    ) / sum_w.unsqueeze(-1)
    s_merged = (torch.log2(sum_w) + s_max) * (1.0 / lse_scale_log2)
    return v_merged.to(v_a.dtype), s_merged
