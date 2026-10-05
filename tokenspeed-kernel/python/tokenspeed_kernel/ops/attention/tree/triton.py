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

"""Draft-tree attention as a cascade over a paged KV cache.

Each request owns ``R`` query rows over its committed prefix and a ``W``-key
tree window at the end of its keys; row ``r`` sees window key ``j`` when bit
``j`` of its 64-bit mask is set. A causal ``q_len = R`` decode over the prefix
(trtllm-gen, with its log-sum-exp) covers keys ``[0, P - R + 1 + r)`` of row
``r``; ``triton_tree_window_attention`` attends the rest (the prefix tail and
the window) and merges both in one kernel. Target verify (``R == W == N``) and
draft lanes (``R == K``, ``W == (S - 1) * K``) both use it.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = ["triton_tree_window_attention"]


@triton.jit
def _tree_window_merge_kernel(
    q_ptr,  # [bs * R, Hq, D]
    k_ptr,  # [slots, Hkv, D] token rows of the paged cache
    v_ptr,
    table_ptr,  # [bs, max_pages] int32
    seq_lens_ptr,  # [bs] int32, including the W window keys
    mask_ptr,  # [bs * R] int64
    prefix_out_ptr,  # [bs * R, Hq, D] causal prefix partial
    prefix_lse_ptr,  # [bs * R, Hq] float32, base 2
    out_ptr,  # [bs * R, Hq, D]
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_table,
    stride_pt,
    stride_ph,
    stride_pd,
    stride_lt,
    stride_lh,
    stride_ot,
    stride_oh,
    stride_od,
    sm_scale_log2,
    R: tl.constexpr,
    W: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    ROWS_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    kv_head = tl.program_id(1)
    rows = tl.program_id(2) * ROWS_BLOCK + tl.arange(0, ROWS_BLOCK)
    row_ok = rows < R * GROUP
    node = rows // GROUP
    head = kv_head * GROUP + rows % GROUP
    token = req * R + node
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + token[:, None] * stride_qt
        + head[:, None] * stride_qh
        + dims[None, :] * stride_qd,
        mask=row_ok[:, None],
        other=0.0,
    )
    bits = tl.load(mask_ptr + token, mask=row_ok, other=0)

    # Row r's partial covers keys [0, prefix - R + 1 + r); attend the prefix tail and the window.
    seq_len = tl.load(seq_lens_ptr + req)
    prefix = seq_len - W
    first = prefix - (R - 1)
    covered = first + node
    has_partial = row_ok & (covered > 0)
    run_max = tl.load(
        prefix_lse_ptr + token * stride_lt + head * stride_lh,
        mask=has_partial,
        other=float("-inf"),
    )
    run_sum = tl.where(has_partial, 1.0, 0.0)
    acc = tl.load(
        prefix_out_ptr
        + token[:, None] * stride_pt
        + head[:, None] * stride_ph
        + dims[None, :] * stride_pd,
        mask=has_partial[:, None],
        other=0.0,
    ).to(tl.float32)

    cols = tl.arange(0, BLOCK)
    for tile in range(0, R - 1 + W, BLOCK):
        col = tile + cols
        pos = first + col
        col_ok = (col < R - 1 + W) & (pos >= 0)
        page = tl.load(
            table_ptr + req * stride_table + pos // PAGE, mask=col_ok, other=0
        )
        slot = page.to(tl.int64) * PAGE + pos % PAGE
        k = tl.load(
            k_ptr
            + slot[:, None] * stride_kt
            + kv_head * stride_kh
            + dims[None, :] * stride_kd,
            mask=col_ok[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr
            + slot[:, None] * stride_vt
            + kv_head * stride_vh
            + dims[None, :] * stride_vd,
            mask=col_ok[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
        in_window = col >= R - 1
        bit = (
            (bits[:, None] >> tl.maximum(col - (R - 1), 0).to(tl.int64)[None, :]) & 1
        ) != 0
        visible = col_ok[None, :] & tl.where(
            in_window[None, :], bit, pos[None, :] >= covered[:, None]
        )
        scores = tl.where(visible, scores, float("-inf"))
        # A slot no row here sees may hold stale or padding K/V; keep it out of P.V (0 * NaN).
        seen = tl.max(visible.to(tl.int32), axis=0) > 0
        v = tl.where(seen[:, None], v, 0.0)
        new_max = tl.maximum(run_max, tl.max(scores, axis=1))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        alpha = tl.exp2(run_max - safe_max)
        probs = tl.exp2(scores - safe_max[:, None])
        run_sum = run_sum * alpha + tl.sum(probs, axis=1)
        acc = acc * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
        run_max = new_max

    out = acc / tl.where(run_sum > 0.0, run_sum, 1.0)[:, None]
    tl.store(
        out_ptr
        + token[:, None] * stride_ot
        + head[:, None] * stride_oh
        + dims[None, :] * stride_od,
        out.to(out_ptr.dtype.element_ty),
        mask=row_ok[:, None],
    )


@register_kernel(
    "attention",
    "tree_window",
    name="triton_tree_window_attention",
    solution="triton",
    # Its prefix partial comes from trtllm-gen, so trees run it on NVIDIA only.
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
)
def triton_tree_window_attention(
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
    num_rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group = num_q_heads // num_kv_heads
    bs = num_rows // rows_per_req
    out = torch.empty_like(q)
    if bs == 0:
        return out
    rows_block = 16
    _tree_window_merge_kernel[
        (bs, num_kv_heads, triton.cdiv(rows_per_req * group, rows_block))
    ](
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        mask,
        prefix_out,
        prefix_lse,
        out,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        page_table.stride(0),
        *prefix_out.stride(),
        *prefix_lse.stride(),
        *out.stride(),
        sm_scale * 1.4426950408889634,
        R=rows_per_req,
        W=window,
        GROUP=group,
        HEAD_DIM=head_dim,
        PAGE=page_size,
        ROWS_BLOCK=rows_block,
        BLOCK=32,
        num_warps=4,
    )
    return out
