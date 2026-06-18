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
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.mha_decode import decode_attention_fwd
from tokenspeed_kernel.ops.attention.triton.mha_prefill import prefill_attention_fwd
from tokenspeed_kernel.ops.attention.triton.mla_decode import (  # noqa: F401
    triton_mla_decode_with_kvcache,
)
from tokenspeed_kernel.ops.attention.triton.mla_prefill import (  # noqa: F401
    triton_mla_prefill,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def attn_merge_state_kernel(
    OutA,
    LseA,
    OutB,
    LseB,
    Out,
    Lse,
    head_dim: tl.constexpr,
    lse_scale_log2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    value_offsets = row * head_dim + offs_d

    lse_a = tl.load(LseA + row).to(tl.float32)
    lse_b = tl.load(LseB + row).to(tl.float32)
    lse_a_log2 = lse_a * lse_scale_log2
    lse_b_log2 = lse_b * lse_scale_log2
    lse_max_log2 = tl.maximum(lse_a_log2, lse_b_log2)

    weight_a = tl.exp2(lse_a_log2 - lse_max_log2)
    weight_b = tl.exp2(lse_b_log2 - lse_max_log2)
    denom = weight_a + weight_b

    out_a = tl.load(OutA + value_offsets, mask=mask_d, other=0.0).to(tl.float32)
    out_b = tl.load(OutB + value_offsets, mask=mask_d, other=0.0).to(tl.float32)
    out = (out_a * weight_a + out_b * weight_b) / denom
    merged_lse = (lse_max_log2 + tl.log2(denom)) / lse_scale_log2

    tl.store(Out + value_offsets, out, mask=mask_d)
    tl.store(Lse + row, merged_lse)


@register_kernel(
    "attention",
    "mha_prefill",
    name="triton_mha_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(q)
    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    cache_seqlens = torch.empty((0,), dtype=torch.int32, device=q.device)
    empty_k = torch.empty((0, k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device)
    empty_v = torch.empty((0, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device)
    prefill_attention_fwd(
        q,
        k,
        v,
        out,
        empty_k,
        empty_v,
        cu_seqlens,
        cache_seqlens,
        None,
        True,
        max_seqlen,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        logit_cap=logit_cap,
        sliding_window_size=window_left,
        sinks=sinks,
        has_kv_cache=False,
        lse_extend=lse,
    )
    if return_lse:
        return out, lse
    return out


@register_kernel(
    "attention",
    "mha_extend_with_kvcache",
    name="triton_mha_extend_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "is_causal": frozenset({False, True}),
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
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
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        logit_cap=logit_cap,
        sliding_window_size=window_left,
        sinks=sinks,
        page_table=page_table,
        page_table_stride_b=page_table.stride(0),
        page_size=k_cache.shape[1],
        has_kv_cache=True,
        lse_extend=lse,
    )
    if return_lse:
        return out, lse
    return out


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="triton_mha_decode_with_kvcache_cached",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
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
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        logit_cap=logit_cap,
        sinks=sinks,
    )
    return out


@register_kernel(
    "attention",
    "attn_merge_state",
    name="triton_attn_merge_state",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("out_a", "out_b"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={},
    tags={"portability"},
)
def triton_attn_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    lse_scale_log2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(out_a)
    lse = torch.empty_like(lse_a)
    total_rows = out_a.shape[0] * out_a.shape[1]
    head_dim = out_a.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    attn_merge_state_kernel[(total_rows,)](
        out_a,
        lse_a,
        out_b,
        lse_b,
        out,
        lse,
        head_dim,
        float(lse_scale_log2),
        BLOCK_D=block_d,
    )
    return out, lse
