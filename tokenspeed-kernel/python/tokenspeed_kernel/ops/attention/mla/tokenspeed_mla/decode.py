"""Registered TokenSpeed MLA decode implementation."""

import math

import torch
from tokenspeed_kernel.ops.attention.mla.tokenspeed_mla._bindings import (
    get_num_sm,
    tokenspeed_mla_decode,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_WORKSPACES: dict[tuple[torch.device, int, int, int], torch.Tensor] = {}


def _workspace(
    device: torch.device, num_heads: int, q_len: int, kv_lora_rank: int
) -> torch.Tensor:
    key = (device, num_heads, q_len, kv_lora_rank)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        size = get_num_sm(device) * num_heads * q_len * (kv_lora_rank + 1) * 4
        workspace = torch.empty(size, dtype=torch.int8, device=device)
        _WORKSPACES[key] = workspace
    return workspace


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="tokenspeed_mla_decode_with_kvcache",
    solution="tokenspeed_mla",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(
        ("q", "kv_cache"),
        "dense",
        {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "q_len": frozenset({1, 2, 3, 4}),
        "num_q_heads": frozenset(range(1, 129)),
        "page_size": frozenset({32, 64}),
        "kv_lora_rank": frozenset({512}),
        "qk_rope_head_dim": frozenset({64}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
    },
    tags={"latency", "throughput"},
)
def tokenspeed_mla_decode_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run TokenSpeed MLA decode through the unified attention API."""
    if logit_cap != 0.0:
        raise ValueError("TokenSpeed MLA decode does not support logit_cap")
    del qk_nope_head_dim

    if q.ndim != 4 or q.shape[1] not in (1, 2, 3, 4):
        raise ValueError("TokenSpeed MLA decode requires q_len in [1, 4]")
    if not 1 <= q.shape[2] <= 128:
        raise ValueError("TokenSpeed MLA decode requires 1 to 128 query heads")
    if kv_lora_rank != 512 or qk_rope_head_dim != 64:
        raise ValueError("TokenSpeed MLA decode requires rank 512 and RoPE dim 64")

    if kv_cache.ndim == 4:
        if kv_cache.shape[2] == 1:
            kv_cache = kv_cache.squeeze(2)
        elif kv_cache.shape[1] == 1:
            kv_cache = kv_cache.squeeze(1)
        else:
            raise ValueError(
                "TokenSpeed MLA expects a singleton KV-head dimension in the cache"
            )
    if kv_cache.ndim != 3 or kv_cache.shape[1] not in (32, 64):
        raise ValueError("TokenSpeed MLA decode requires page size 32 or 64")
    if q.dtype != kv_cache.dtype:
        raise TypeError("TokenSpeed MLA decode requires matching query and KV dtypes")

    result = tokenspeed_mla_decode(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=_workspace(q.device, q.shape[2], q.shape[1], kv_lora_rank),
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=page_table,
        seq_lens=cache_seqlens,
        max_seq_len=max_seqlen_k,
        softmax_scale=softmax_scale,
        output_scale=1.0,
        out=out,
        is_var_seq=True,
        causal_mask=True,
        enable_pdl=False,
        return_lse=return_lse,
        causal_seqs=None,
        cp_world=1,
        cp_rank=0,
    )
    if not return_lse:
        return result
    output, lse_log2 = result
    return output, lse_log2 * math.log(2.0)
