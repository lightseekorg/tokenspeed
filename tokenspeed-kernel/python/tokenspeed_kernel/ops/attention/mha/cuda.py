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
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

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


if platform.is_blackwell_plus:
    from flash_attn.cute import (
        flash_attn_func,
        flash_attn_varlen_func,
    )

if platform.is_nvidia and platform.is_blackwell:
    # FA4 on Blackwell supports prefill head_dim in [8, 256] divisible by 8,
    # but the 256-wide MHA path mishandles non-contiguous V split views, so we
    # restrict it to <256 for now until that is resolved.
    # Keep the plain MHA registrations capped at SM100 so B300 retains its
    # FlashInfer route.
    _FA4_BLACKWELL_PREFILL_HEAD_DIMS = frozenset(range(8, 256, 8))
    _FA4_BLACKWELL_DECODE_HEAD_DIMS = frozenset(range(8, 129, 8))

    import inspect

    _FA4_HAS_BLOCKSCALED = "sfq" in inspect.signature(flash_attn_varlen_func).parameters

    @register_kernel(
        "attention",
        "mha_prefill",
        name="fa4_mha_prefill",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _FA4_BLACKWELL_PREFILL_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "support_skip_softmax": frozenset({False}),
        },
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
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        out, lse = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=((window_left, 0) if window_left >= 0 else (None, None)),
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
            max_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _FA4_BLACKWELL_DECODE_HEAD_DIMS,
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
        },
    )
    def fa4_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
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
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        window_size = (window_left, 0) if window_left >= 0 else (-1, -1)
        out, lse = flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=cache_seqlens,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=is_causal,
            window_size=window_size,
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
            max_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _FA4_BLACKWELL_DECODE_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False}),
            "return_lse": frozenset({False}),
            "support_logit_cap": frozenset({False}),
        },
    )
    def fa4_mha_decode_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        max_seqlen_q: int = 1,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        batch_size = cache_seqlens.shape[0]
        q_reshaped = q.view(batch_size, max_seqlen_q, q.shape[1], q.shape[2])
        window_size = (window_left, 0) if window_left >= 0 else (-1, -1)
        out, _ = flash_attn_varlen_func(
            q=q_reshaped,
            k=k_cache,
            v=v_cache,
            seqused_k=cache_seqlens,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=max_seqlen_q > 1,
            window_size=window_size,
        )
        return out.view_as(q)

    # --------------------------------------------------------------------------
    # Dense (unscaled) FP8-E4M3 MHA: fp8 q + paged fp8 K/V at implicit scale
    # 1.0, bf16 output. e5m2 is excluded, matching vLLM's FLASH_ATTN gate.
    # --------------------------------------------------------------------------

    _FA4_FP8_DENSE_SIGNATURES = format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float8_e4m3fn}
    )

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="fa4_mha_extend_with_kvcache_fp8",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_FA4_FP8_DENSE_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _FA4_BLACKWELL_DECODE_HEAD_DIMS,
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False}),
            "return_lse": frozenset({False}),
            "support_logit_cap": frozenset({False}),
        },
    )
    def fa4_mha_extend_with_kvcache_fp8(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
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
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        window_size = (window_left, 0) if window_left >= 0 else (-1, -1)
        out, _ = flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=cache_seqlens,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=is_causal,
            window_size=window_size,
        )
        return out

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="fa4_mha_decode_with_kvcache_fp8",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_FA4_FP8_DENSE_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _FA4_BLACKWELL_DECODE_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False}),
            "return_lse": frozenset({False}),
            "support_logit_cap": frozenset({False}),
        },
    )
    def fa4_mha_decode_with_kvcache_fp8(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        max_seqlen_q: int = 1,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        batch_size = cache_seqlens.shape[0]
        q_reshaped = q.view(batch_size, max_seqlen_q, q.shape[1], q.shape[2])
        window_size = (window_left, 0) if window_left >= 0 else (-1, -1)
        out, _ = flash_attn_varlen_func(
            q=q_reshaped,
            k=k_cache,
            v=v_cache,
            seqused_k=cache_seqlens,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=max_seqlen_q > 1,
            window_size=window_size,
        )
        return out.view_as(q)

    # --------------------------------------------------------------------------
    # MXFP8 block-scaled MHA: fp8-e4m3 q + paged KV with UE8M0 vector-32 scale
    # factors. The KV scale layout is the interleaved BlockScaledBasicChunk
    # atom ([num_pages, num_kv_heads, 32, 4, 4]) that the fork requires at
    # page_size 128 (written by store_sf_interleaved); q scales stay flat
    # K-major because pack_gqa is forced on and that path uses the cp.async
    # scale loader. Requires the blockscaled fork build (sfq in the varlen
    # interface).
    # --------------------------------------------------------------------------

    from tokenspeed_kernel.signature import MXFP8_BLOCK_SCALE as _MXFP8_KV_BLOCK_SCALE

    _MXFP8_ATTENTION_SIGNATURES = format_signatures(
        ("q", "k_cache", "v_cache"),
        "mxfp8",
        {torch.float8_e4m3fn},
        scale=_MXFP8_KV_BLOCK_SCALE,
    )

    if _FA4_HAS_BLOCKSCALED:

        @register_kernel(
            "attention",
            "mha_decode_with_kvcache",
            name="fa4_mha_decode_with_kvcache_mxfp8",
            solution="fa4",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(10, 0),
                max_arch_version=ArchVersion(10, 0),
                vendors=frozenset({"nvidia"}),
            ),
            signatures=_MXFP8_ATTENTION_SIGNATURES,
            priority=Priority.SPECIALIZED,
            traits={
                "head_dim": frozenset({128}),
                "page_size": frozenset({128}),
                "sliding_window": frozenset({False, True}),
                "support_sinks": frozenset({False}),
                "return_lse": frozenset({False}),
                "support_logit_cap": frozenset({False}),
            },
        )
        def fa4_mha_decode_with_kvcache_mxfp8(
            q: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            page_table: torch.Tensor,
            cache_seqlens: torch.Tensor,
            max_seqlen_k: int,
            max_seqlen_q: int = 1,
            window_left: int = -1,
            logit_cap: float = 0.0,
            sinks: torch.Tensor | None = None,
            return_lse: bool = False,
            softmax_scale: float | None = None,
            q_scale: torch.Tensor | None = None,
            k_scale: torch.Tensor | None = None,
            v_scale: torch.Tensor | None = None,
            enable_pdl: bool = False,
        ) -> torch.Tensor:
            if softmax_scale is None:
                softmax_scale = 1.0 / math.sqrt(q.shape[-1])
            window_size = (window_left, 0) if window_left >= 0 else (None, None)
            batch_size = cache_seqlens.shape[0]
            q_reshaped = q.view(batch_size, max_seqlen_q, q.shape[1], q.shape[2])
            sfq = q_scale.view(
                batch_size, max_seqlen_q, q_scale.shape[-2], q_scale.shape[-1]
            )
            out, _ = flash_attn_varlen_func(
                q=q_reshaped,
                k=k_cache,
                v=v_cache,
                seqused_k=cache_seqlens,
                page_table=page_table,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=max_seqlen_q > 1,
                window_size=window_size,
                pack_gqa=True,
                sfq=sfq,
                sfk=k_scale,
                sfv=v_scale,
            )
            return out.view(q.shape[0], q.shape[1], v_cache.shape[-1]).to(
                torch.bfloat16
            )

        @register_kernel(
            "attention",
            "mha_extend_with_kvcache",
            name="fa4_mha_extend_with_kvcache_mxfp8",
            solution="fa4",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(10, 0),
                max_arch_version=ArchVersion(10, 0),
                vendors=frozenset({"nvidia"}),
            ),
            signatures=_MXFP8_ATTENTION_SIGNATURES,
            priority=Priority.SPECIALIZED,
            traits={
                "head_dim": frozenset({128}),
                "page_size": frozenset({128}),
                "is_causal": frozenset({True}),
                "sliding_window": frozenset({False, True}),
                "support_sinks": frozenset({False}),
                "return_lse": frozenset({False}),
                "support_logit_cap": frozenset({False}),
            },
        )
        def fa4_mha_extend_with_kvcache_mxfp8(
            q: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_kv: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            page_table: torch.Tensor,
            cache_seqlens: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            is_causal: bool = True,
            window_left: int = -1,
            logit_cap: float = 0.0,
            sinks: torch.Tensor | None = None,
            return_lse: bool = False,
            softmax_scale: float | None = None,
            q_scale: torch.Tensor | None = None,
            k_scale: torch.Tensor | None = None,
            v_scale: torch.Tensor | None = None,
            enable_pdl: bool = False,
        ) -> torch.Tensor:
            if softmax_scale is None:
                softmax_scale = 1.0 / math.sqrt(q.shape[-1])
            window_size = (window_left, 0) if window_left >= 0 else (None, None)
            out, _ = flash_attn_varlen_func(
                q=q,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=cache_seqlens,
                page_table=page_table,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=is_causal,
                window_size=window_size,
                sfq=q_scale,
                sfk=k_scale,
                sfv=v_scale,
            )
            return out.to(torch.bfloat16)

elif platform.is_nvidia and platform.is_hopper:
    from flash_attn_interface import (
        flash_attn_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    )

    @register_kernel(
        "attention",
        "mha_prefill",
        name="fa3_mha_prefill",
        solution="fa3",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            max_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False}),
            "support_skip_softmax": frozenset({False}),
        },
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
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
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
            max_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def fa3_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
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
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
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
            max_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def fa3_mha_decode_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        max_seqlen_q: int = 1,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        batch_size = cache_seqlens.shape[0]
        out = flash_attn_with_kvcache(
            q=q.view(batch_size, max_seqlen_q, q.shape[1], q.shape[2]),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=max_seqlen_q > 1,
            window_size=((window_left, 0) if window_left >= 0 else (-1, -1)),
            softcap=logit_cap,
            sinks=sinks,
        )
        return out.view_as(q)
