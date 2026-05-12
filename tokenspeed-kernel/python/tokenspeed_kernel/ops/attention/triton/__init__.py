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

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.ops.attention.triton.mha_decode import decode_attention_fwd
from tokenspeed_kernel.ops.attention.triton.mha_prefill import prefill_attention_fwd
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel


@register_kernel(
    "attention",
    "mha_prefill",
    name="triton_mha_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    dtypes={torch.float16, torch.bfloat16},
    priority=Priority.PERFORMANT,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def triton_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    batch_size = cu_seqlens_q.shape[0] - 1
    out = torch.empty_like(q)
    cache_seqlens = torch.empty((0,), dtype=torch.int32, device=q.device)
    empty_k = torch.empty((0, k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device)
    empty_v = torch.empty((0, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device)
    sm_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    )
    prefill_attention_fwd(
        q,
        k,
        v,
        out,
        empty_k,
        empty_v,
        cu_seqlens_q,
        cache_seqlens,
        None,
        is_causal,
        max_seqlen_q,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sliding_window_size=window_left,
        sinks=sinks,
        has_kv_cache=False,
    )
    return out


@register_kernel(
    "attention",
    "mha_prefill_with_kvcache",
    name="triton_mha_prefill_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    dtypes={torch.float16, torch.bfloat16},
    priority=Priority.PERFORMANT,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def triton_mha_prefill_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    extend_from_cache = k is None or v is None
    if extend_from_cache:
        if k is not None or v is not None:
            raise ValueError("k and v must both be provided or both be None")
        k = torch.empty(
            (0, k_cache.shape[2], k_cache.shape[3]),
            dtype=k_cache.dtype,
            device=k_cache.device,
        )
        v = torch.empty(
            (0, v_cache.shape[2], v_cache.shape[3]),
            dtype=v_cache.dtype,
            device=v_cache.device,
        )

    out = torch.empty_like(q)
    sm_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    )
    prefill_attention_fwd(
        q,
        k,
        v,
        out,
        k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]),
        v_cache.view(-1, v_cache.shape[2], v_cache.shape[3]),
        cu_seqlens_q,
        cache_seqlens,
        None,
        is_causal,
        max_seqlen_q,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sliding_window_size=window_left,
        sinks=sinks,
        page_table=page_table,
        page_table_stride_b=page_table.stride(0),
        page_size=k_cache.shape[1],
        extend_from_cache=extend_from_cache,
        has_kv_cache=True,
    )
    return out


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="triton_mha_decode_with_kvcache_cached",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    dtypes={torch.float16, torch.bfloat16},
    priority=Priority.PERFORMANT,
    traits={
        "query_len": frozenset({1}),
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def triton_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    out = torch.empty_like(q)
    max_kv_splits = 4
    attn_logits = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    attn_lse = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        dtype=torch.float32,
        device=q.device,
    )
    num_kv_splits = torch.ones(
        (cache_seqlens.shape[0],), dtype=torch.int32, device=q.device
    )
    sm_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    )
    decode_attention_fwd(
        q,
        k_cache.view(-1, k_cache.shape[2], k_cache.shape[3]),
        v_cache.view(-1, v_cache.shape[2], v_cache.shape[3]),
        out,
        page_table,
        cache_seqlens,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        page_table.stride(0),
        k_cache.shape[1],
        window_left,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sinks=sinks,
    )
    return out
