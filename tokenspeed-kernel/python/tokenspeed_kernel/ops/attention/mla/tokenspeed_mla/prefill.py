"""Registered TokenSpeed MLA prefill implementation."""

import math

import torch
from tokenspeed_kernel.ops.attention.mla.tokenspeed_mla._bindings import (
    tokenspeed_mla_prefill,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "mla_prefill",
    name="tokenspeed_mla_prefill",
    solution="tokenspeed_mla",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(("q", "k", "v"), "dense", {torch.float8_e4m3fn}),
    priority=Priority.SPECIALIZED,
    traits={
        "qk_head_dim": frozenset({192}),
        "v_head_dim": frozenset({128}),
        "is_causal": frozenset({False, True}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
    },
    tags={"throughput"},
)
def tokenspeed_mla_prefill_adapter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    seq_lens_kv: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run TokenSpeed MLA prefill through the unified attention API."""
    if logit_cap != 0.0:
        raise ValueError("TokenSpeed MLA prefill does not support logit_cap")
    if q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("TokenSpeed MLA prefill requires FP8 E4M3 Q, K, and V")
    if q.shape[-1] != 192 or k.shape[-1] != 192 or v.shape[-1] != 128:
        raise ValueError("TokenSpeed MLA prefill requires QK dim 192 and V dim 128")
    if out is not None and (out.dtype != torch.bfloat16 or not out.is_contiguous()):
        raise ValueError("TokenSpeed MLA prefill out must be contiguous BF16")
    if seq_lens_kv is None:
        seq_lens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]

    result = tokenspeed_mla_prefill(
        query=q.contiguous(),
        key=k.contiguous(),
        value=v.contiguous(),
        seq_lens=seq_lens_kv,
        cum_seq_lens=cu_seqlens_kv,
        max_seq_len=max_seqlen_kv,
        batch_size=cu_seqlens_q.numel() - 1,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        return_lse=return_lse,
        cum_seq_lens_q=cu_seqlens_q,
        max_seq_len_q=max_seqlen_q,
        enable_pdl=False,
        out=out,
    )
    if not return_lse:
        return result
    output, lse_log2 = result
    return output, lse_log2 * math.log(2.0)
