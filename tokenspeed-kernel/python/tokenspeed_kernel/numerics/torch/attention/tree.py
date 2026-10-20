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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.tree``: attention
of draft-tree rows over a request's prefix and tree window."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "tree_window_attention",
    name="torch_tree_window_attention",
    solution="torch",
    signatures=format_signatures("q", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_tree_window_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    prefix_len: int,
    window_masks: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One request's tree rows over its full prefix and their window keys.

    Row ``i`` sees every prefix key and window key ``j`` (position
    ``prefix_len + j``) when bit ``j`` of ``window_masks[i]`` is set.

    Args:
        q: ``[rows, heads, D]``.
        keys: ``[prefix_len + window, kv_heads, D]``.
        values: ``[prefix_len + window, kv_heads, Dv]``.
        window_masks: ``[rows]`` int64 bitmasks (bit 63 is the sign bit).
        scale: Logit scale.

    Returns:
        FP32 ``[rows, heads, Dv]``.
    """
    group = q.shape[1] // keys.shape[1]
    kk = keys.float().repeat_interleave(group, 1)
    vv = values.float().repeat_interleave(group, 1)
    s = torch.einsum("nhd,xhd->hnx", q.float(), kk) * scale
    window = keys.shape[0] - prefix_len
    bits = torch.arange(window, device=q.device)
    visible = torch.ones(q.shape[0], keys.shape[0], dtype=torch.bool, device=q.device)
    visible[:, prefix_len:] = (window_masks.to(q.device)[:, None] >> bits) & 1 == 1
    s = s.masked_fill(~visible[None], float("-inf"))
    return torch.einsum("hnx,xhd->nhd", torch.softmax(s, -1), vv)
