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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.mla``: multi-head
latent attention, where every head reads one shared latent cache row whose
first ``kv_lora_rank`` features double as the value."""

from __future__ import annotations

import torch
from tokenspeed_kernel.numerics.torch.attention.mha import (
    gather_pages,
    torch_dense_attention,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "mla_paged_attention",
    name="torch_mla_paged_attention",
    solution="torch",
    signatures=format_signatures(
        "q", "dense", {torch.bfloat16, torch.float8_e4m3fn, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_mla_paged_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    kv_lora_rank: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal MLA of each request's last ``q_len`` positions over its cache.

    Query position ``i`` of a request with ``L`` cached rows sees rows
    ``[0, L - q_len + 1 + i)``; positions that see none return zeros.

    Args:
        q: ``[batch, q_len, heads, qk_dim]`` queries.
        kv_cache: ``[num_pages, page_size, qk_dim]`` latent rows (a trailing
            singleton head dim is accepted).
        page_table: ``[batch, max_pages]``.
        cache_seqlens: ``[batch]`` cached rows per request, including the
            queries' own.
        kv_lora_rank: Latent features that form the value.
        softmax_scale: Logit scale.

    Returns:
        FP32 ``(out [batch, q_len, heads, kv_lora_rank], lse [batch, q_len,
        heads])``, natural-log LSE.
    """
    cache = kv_cache.reshape(*kv_cache.shape[:2], kv_cache.shape[-1])
    out = torch.zeros(*q.shape[:-1], kv_lora_rank, device=q.device)
    lse = torch.full(q.shape[:-1], float("-inf"), device=q.device)
    q_len = q.shape[1]
    for request, length in enumerate(cache_seqlens.tolist()):
        kv = gather_pages(cache, page_table[request], length)[:, None].float()
        out[request], lse[request] = torch_dense_attention(
            q[request].float(),
            kv,
            kv[..., :kv_lora_rank],
            scale=softmax_scale,
            causal_offset=length - q_len,
            window_left=None,
            sinks=None,
        )
    return out, lse


@register_kernel(
    "attention",
    "mla_kv_pack",
    name="torch_mla_kv_pack",
    solution="torch",
    signatures=format_signatures("k_nope", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_mla_kv_pack(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    k_scale_inv: float | torch.Tensor,
    v_scale_inv: float | torch.Tensor,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast ``k_pe`` across heads, append it to ``k_nope``, then scale
    and cast K and V to ``fp8_dtype``.

    ``k_pe`` is ``[tokens, rope]`` or ``[tokens, 1, rope]``.
    """
    if k_pe.ndim == 2:
        k_pe = k_pe.unsqueeze(1)
    k_pe = k_pe.expand(-1, k_nope.shape[1], -1)
    return (
        (torch.cat((k_nope, k_pe), dim=-1).float() * k_scale_inv).to(fp8_dtype),
        (v.float() * v_scale_inv).to(fp8_dtype),
    )


@register_kernel(
    "attention",
    "mla_value_projection",
    name="torch_mla_value_projection",
    solution="torch",
    signatures=format_signatures("attention", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_mla_value_projection(
    attention: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor | None
) -> torch.Tensor:
    """Per-head projection of the latent attention output, optionally gated.

    Args:
        attention: ``[tokens, heads, kv_lora_rank]``.
        weight: ``[heads, kv_lora_rank, v_head_dim]``.
        gate: Optional ``[tokens, heads * v_head_dim]`` sigmoid gate logits.

    Returns:
        FP32 ``[tokens, heads * v_head_dim]``.
    """
    out = torch.bmm(attention.transpose(0, 1).float(), weight.float()).transpose(0, 1)
    out = out.reshape(attention.shape[0], -1)
    if gate is not None:
        out = out * torch.sigmoid(gate.float())
    return out
