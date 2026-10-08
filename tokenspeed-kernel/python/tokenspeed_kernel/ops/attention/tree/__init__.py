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

"""Draft-tree attention (docs/design/tree-speculation.md)."""

from __future__ import annotations

import torch
from tokenspeed_kernel.profiling import kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["tree_window_attention"]

# Tree visibility is one 64-bit mask word per query row.
MAX_TREE_SLOTS = 64


def tree_window_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    mask: torch.Tensor,
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    *,
    rows_per_req: int,
    window: int,
    page_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """Finish draft-tree attention from a causal prefix partial (cascade).

    Each request's last ``window`` keys are its tree window: query row ``r``
    sees window key ``j`` when bit ``j`` of its mask is set, and every earlier
    (prefix) key. The caller has run a causal ``q_len = rows_per_req`` decode
    over the ``P`` prefix keys, so row ``r``'s partial covers keys
    ``[0, P - rows_per_req + 1 + r)``; this attends the rest (the prefix tail
    and the window) and merges it into that partial. A row whose partial covers
    no key ignores it.

    Args:
        q: ``[bs * rows_per_req, num_q_heads, head_dim]`` queries.
        k_cache: ``[slots, num_kv_heads, head_dim]`` token rows of the cache, in
            ``q.dtype`` or unscaled FP8 E4M3.
        v_cache: laid out like ``k_cache``.
        page_table: ``[bs, max_pages]`` int32 page ids; slot = page * page_size + offset.
        seq_lens: ``[bs]`` int32 keys per request, the window included (>= window).
        mask: ``[bs * rows_per_req]`` int64 window visibility per query row.
        prefix_out: ``[bs * rows_per_req, num_q_heads, head_dim]`` causal prefix output.
        prefix_lse: ``[bs * rows_per_req, num_q_heads]`` float32 base-2
            log-sum-exp of the prefix scores (scaled by ``sm_scale``).
        rows_per_req: query rows per request, at most 64.
        window: tree window width in keys, at most 64.
        page_size: tokens per page.
        sm_scale: softmax scale applied to ``q . k``.

    Returns:
        The attention output in ``q.dtype``.
    """
    if rows_per_req > MAX_TREE_SLOTS or window > MAX_TREE_SLOTS:
        raise ValueError(
            f"{rows_per_req} rows over a {window}-key window; at most {MAX_TREE_SLOTS}"
        )
    signature = format_signature(
        q=dense_tensor_format(q.dtype),
        k_cache=dense_tensor_format(k_cache.dtype),
        v_cache=dense_tensor_format(v_cache.dtype),
    )
    kernel = select_kernel("attention", "tree_window", signature)
    with kernel_scope(
        "attention",
        "tree_window",
        q.dtype,
        kernel_name=kernel.name,
        batch_size=q.shape[0] // rows_per_req,
        num_q_heads=q.shape[1],
        head_dim=q.shape[2],
    ):
        return kernel(
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            mask,
            prefix_out,
            prefix_lse,
            rows_per_req=rows_per_req,
            window=window,
            page_size=page_size,
            sm_scale=sm_scale,
        )


import tokenspeed_kernel.ops.attention.tree.triton  # noqa: E402,F401
