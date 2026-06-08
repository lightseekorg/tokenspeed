# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Architecture-selected FlashAttention kernels."""

import math

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel_nvidia.registration import Priority, error_fn, register_kernel

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "get_scheduler_metadata",
    "mha_decode_scheduler_metadata",
]

flash_attn_func = error_fn
flash_attn_varlen_func = error_fn
flash_attn_with_kvcache = error_fn
get_scheduler_metadata = error_fn

platform = current_platform()

# ------------------------------------------------------------------------------
# Kernel registration
# ------------------------------------------------------------------------------


# FA4 CUTE kernels currently support SM 10.0 (B100/B200/GB200) variants only.
# B300/GB300 (SM 10.3 / sm_103) is not yet supported by flash-attn — its CUTE
# arch enum value falls outside the [sm_100, sm_110f] range checked in
# FlashAttentionForwardSm100, causing an AssertionError at call time.
if (
    platform.is_nvidia
    and platform.is_blackwell
    and platform.arch_version == ArchVersion(10, 0)
):
    try:
        from flash_attn.cute import (
            flash_attn_func,
            flash_attn_varlen_func,
        )
    except ImportError:
        pass

    # FA4 on Blackwell supports prefill head_dim in [8, 256] divisible by 8
    # (and (192, 128) for DeepSeek MLA, not applicable here). Cached paths pass
    # seqused_k and remain limited to <=128 by upstream FA4.
    _FA4_BLACKWELL_PREFILL_HEAD_DIMS = frozenset(range(8, 257, 8))
    _FA4_BLACKWELL_DECODE_HEAD_DIMS = frozenset(range(8, 129, 8))

    @register_kernel(
        "attention",
        "mha_prefill",
        name="fa4_mha_prefill",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED + 3,
        traits={
            "head_dim": _FA4_BLACKWELL_PREFILL_HEAD_DIMS,
            "sliding_window": frozenset({False}),
            "support_sinks": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
        },
        tags={"throughput"},
    )
    def fa4_mha_prefill(
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
        out, lse = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=1.0 / math.sqrt(q.shape[-1]),
            causal=True,
            return_lse=return_lse,
        )
        if return_lse:
            return out, lse.transpose(0, 1).contiguous()
        return out

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="fa4_mha_extend_with_kvcache_cached",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED + 3,
        traits={
            "head_dim": _FA4_BLACKWELL_DECODE_HEAD_DIMS,
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False}),
            "support_sinks": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
        },
        tags={"throughput"},
    )
    def fa4_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
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
        out, lse = flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=cache_seqlens,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=1.0 / math.sqrt(q.shape[-1]),
            causal=is_causal,
            return_lse=return_lse,
        )
        if return_lse:
            return out, lse.transpose(0, 1).contiguous()
        return out

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="fa4_mha_decode_with_kvcache",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED + 3,
        traits={
            "head_dim": _FA4_BLACKWELL_DECODE_HEAD_DIMS,
            "sliding_window": frozenset({False}),
            "support_sinks": frozenset({False}),
            "return_lse": frozenset({False}),
            "support_logit_cap": frozenset({False}),
        },
        tags={"latency"},
    )
    def fa4_mha_decode_with_kvcache(
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
        batch_size = cache_seqlens.shape[0]
        q_reshaped = q.view(batch_size, 1, q.shape[1], q.shape[2])
        out, _ = flash_attn_varlen_func(
            q=q_reshaped,
            k=k_cache,
            v=v_cache,
            seqused_k=cache_seqlens,
            page_table=page_table,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=1.0 / math.sqrt(q.shape[-1]),
            causal=False,
        )
        return out.view_as(q)

elif platform.is_nvidia and platform.is_hopper:
    try:
        from flash_attn_interface import (
            flash_attn_func,
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
            get_scheduler_metadata,
        )
    except ImportError:
        pass

    @register_kernel(
        "attention",
        "mha_prefill",
        name="fa3_mha_prefill",
        solution="fa3",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED + 3,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
        tags={"throughput"},
    )
    def fa3_mha_prefill(
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
    ) -> torch.Tensor:
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=1.0 / math.sqrt(q.shape[-1]),
            causal=True,
            window_size=((window_left, 0) if window_left >= 0 else (-1, -1)),
            softcap=logit_cap,
            sinks=sinks,
        )

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="fa3_mha_extend_with_kvcache_cached",
        solution="fa3",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED + 3,
        traits={
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
        tags={"throughput"},
    )
    def fa3_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
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
    ) -> torch.Tensor:
        cu_seqlens_k_new = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=1.0 / math.sqrt(q.shape[-1]),
            causal=is_causal,
            window_size=((window_left, 0) if window_left >= 0 else (-1, -1)),
            softcap=logit_cap,
            sinks=sinks,
        )

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="fa3_mha_decode_with_kvcache_cached",
        solution="fa3",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED + 3,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
        tags={"latency"},
    )
    def fa3_mha_decode_with_kvcache(
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
        scheduler_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = flash_attn_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=1.0 / math.sqrt(q.shape[-1]),
            causal=False,
            window_size=((window_left, 0) if window_left >= 0 else (-1, -1)),
            softcap=logit_cap,
            sinks=sinks,
            scheduler_metadata=scheduler_metadata,
        )
        return out.view_as(q)


# ------------------------------------------------------------------------------
# Direct export
# ------------------------------------------------------------------------------


def mha_decode_scheduler_metadata(
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    cache_seqlens: torch.Tensor,
    qkv_dtype: torch.dtype,
    page_size: int,
    causal: bool = True,
) -> torch.Tensor | None:
    """Pre-compute decode scheduler metadata once per scheduler step.

    Only the FA3 decode kernel consumes pre-computed scheduler metadata; on
    every other backend the kernel computes it internally and this helper
    returns ``None`` so callers can pass through unconditionally.
    """
    if get_scheduler_metadata is error_fn:
        return None
    return get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        headdim=headdim,
        cache_seqlens=cache_seqlens,
        qkv_dtype=qkv_dtype,
        page_size=page_size,
        causal=causal,
    )
