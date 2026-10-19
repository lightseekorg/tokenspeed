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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.dsv4``: sparse
attention over selected latent rows with a per-head attention sink, and the
decode-context-parallel merge of per-shard partials."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def dsv4_attend_rows(
    q: torch.Tensor, keys: torch.Tensor, sink: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """One token's sink-softmax attention over latent rows that are also the
    values.

    Args:
        q: ``[heads, dim]``.
        keys: ``[rows, dim]`` selected latent rows.
        sink: ``[heads]`` sink logits; they join the denominator only.
        scale: Logit scale.

    Returns:
        FP32 ``(out [heads, dim], lse [heads])``; the LSE excludes the sink
        and is ``-inf`` with no rows.
    """
    keys = keys.float()
    logits = q.float() @ keys.T * scale
    lse = torch.logsumexp(logits, dim=1)
    logits = torch.cat((logits, sink.float()[:, None]), dim=1)
    return logits.softmax(dim=1)[:, :-1] @ keys, lse


@register_kernel(
    "attention",
    "dsv4_selected_attention",
    name="torch_dsv4_selected_attention",
    solution="torch",
    signatures=format_signatures("q", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_dsv4_selected_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`dsv4_attend_rows` of each token over its selected rows.

    Token ``t`` reads ``kv`` rows ``slots[t, :lens[t]]``, skipping slots
    outside ``kv``.

    Args:
        q: ``[tokens, heads, dim]``.
        kv: ``[rows, dim]`` latent rows (any leading layout flattening to it).
        slots: ``[tokens, topk]``.
        lens: ``[tokens]``.
        sink: ``[heads]``.
        scale: Logit scale.

    Returns:
        ``(out [tokens, heads, dim]`` in ``q``'s dtype, FP32 ``lse [tokens,
        heads])``.
    """
    rows = kv.reshape(-1, q.shape[-1])
    out = torch.zeros_like(q)
    lse = torch.empty(q.shape[:2], device=q.device)
    for token in range(q.shape[0]):
        selected = slots[token, : int(lens[token])].long()
        selected = selected[(selected >= 0) & (selected < rows.shape[0])]
        token_out, lse[token] = dsv4_attend_rows(q[token], rows[selected], sink, scale)
        out[token] = token_out.to(q.dtype)
    return out, lse


@register_kernel(
    "attention",
    "dsv4_merge_partials",
    name="torch_dsv4_merge_partials",
    solution="torch",
    signatures=format_signatures("lse", "dense", {torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_dsv4_merge_partials(
    partials: list[torch.Tensor], lses: list[torch.Tensor], sink: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine per-shard no-sink partials by their LSE, then apply the sink.

    Non-finite shard LSEs count as empty shards. The merged output scales by
    ``sigmoid(total_lse - sink)``, zero where every shard is empty.

    Args:
        partials: Per shard ``[tokens, heads, dim]`` normalized outputs.
        lses: Per shard ``[tokens, heads]`` no-sink LSEs.
        sink: ``[heads]`` sink logits.

    Returns:
        FP32 ``(out [tokens, heads, dim], total_lse [tokens, heads])``.
    """
    stacked = torch.stack(lses, dim=0)
    finite = torch.where(torch.isfinite(stacked), stacked, -torch.inf)
    total = torch.logsumexp(finite, dim=0)
    weights = torch.nan_to_num(torch.exp(finite - total), nan=0.0)
    merged = sum(
        w[..., None] * p.float() for w, p in zip(weights, partials, strict=True)
    )
    factor = torch.where(
        torch.isinf(total), torch.zeros_like(total), torch.sigmoid(total - sink)
    )
    return merged * factor[..., None], total
