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
    # Relative-attention kernels support both SM100 and SM103. Keep the plain
    # MHA registrations capped at SM100 so B300 retains its FlashInfer route.
    _FA4_BLACKWELL_PREFILL_HEAD_DIMS = frozenset(range(8, 256, 8))
    _FA4_BLACKWELL_DECODE_HEAD_DIMS = frozenset(range(8, 129, 8))

    import inspect

    _FA4_HAS_FUSED_REL_BIAS = (
        "rel_bias" in inspect.signature(flash_attn_varlen_func).parameters
    )
    _FA4_HAS_BLOCKSCALED = "sfq" in inspect.signature(flash_attn_varlen_func).parameters

    import os

    # Default on (bit-exact vs reference); TSMHA_REL_DECODE=0 falls back to fork varlen + fused rel_bias.
    _TSMHA_REL_DECODE = os.environ.get("TSMHA_REL_DECODE", "1") == "1"
    # Same for varlen prefill/extend; TSMHA_REL_EXTEND=0 falls back likewise.
    _TSMHA_REL_EXTEND = os.environ.get("TSMHA_REL_EXTEND", "1") == "1"
    # v2 single-launch gqa-decode; TSMHA_DECODE_V2=0 falls back to the fork.
    _TSMHA_DECODE_V2 = os.environ.get("TSMHA_DECODE_V2", "1") == "1"

    def _tsmha_varlen_eligible(q, num_kv_heads, rel_logits, window_left) -> bool:
        extent = rel_logits.shape[-1]
        return (
            _TSMHA_REL_EXTEND
            and q.dtype in (torch.bfloat16, torch.float16)
            and extent % 128 == 0
            and q.shape[-1] in (64, 96, 128)
            and (window_left < 0 or window_left + 1 == extent)
            and q.shape[1] % num_kv_heads == 0
        )

    def _rel_attention_kwargs(
        rel_logits: torch.Tensor,
        window_left: int,
    ) -> tuple[dict, tuple[int | None, int | None]]:
        """Build the flash-attn call arguments applying the relative bias.

        Prefers the fused rel_bias (ShearingBias) path when the installed
        flash-attn build ships it and the call satisfies its constraints:
        the rel extent must be a multiple of 128, and with a sliding window
        the window length must equal the extent (distances beyond the window
        are masked anyway, so the table is sliced to it). Otherwise the bias
        is applied via the generic ``score_mod`` gather. NOTE: the installed
        fa4 wheel does not ship fused rel_bias, so score_mod is the LIVE bias
        mechanism for this varlen family today — do not remove it until the
        wheel does (checked 2026-07-14).

        Args:
            rel_logits: Relative bias logits with shape
                [total_q, num_q_heads, rel_extent]. Rows are batch-flattened
                query positions.
            window_left: Exclusive left sliding-window size, -1 for full
                causal attention.

        Returns:
            (extra_kwargs, window_size) to splat into flash_attn_varlen_func.
        """
        extent = rel_logits.shape[-1]
        window_size: tuple[int | None, int | None] = (
            (window_left, 0) if window_left >= 0 else (None, None)
        )
        if _FA4_HAS_FUSED_REL_BIAS:
            if window_left >= 0:
                eff = window_left + 1
                if eff % 128 == 0 and eff <= extent:
                    if eff != extent:
                        rel_logits = rel_logits[..., :eff].contiguous()
                    return {"rel_bias": rel_logits}, window_size
            elif extent % 128 == 0:
                return {"rel_bias": rel_logits}, window_size
        from tokenspeed_kernel.ops.attention.score_mods import (
            get_relative_bias_score_mod,
        )

        return {
            "score_mod": get_relative_bias_score_mod(extent),
            "aux_tensors": [rel_logits],
        }, window_size

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

    # rel_mha kernels: the fused rel_bias strategy is a per-kernel detail.

    @register_kernel(
        "attention",
        "rel_mha_prefill",
        name="fa4_rel_mha_prefill",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _FA4_BLACKWELL_PREFILL_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def fa4_rel_mha_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rel_logits: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: list[int],
        max_seqlen: int,
        window_left: int = -1,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        enable_pdl: bool = False,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        if (
            not return_lse
            and cu_seqlens.shape[0] - 1 <= 1024
            and _tsmha_varlen_eligible(q, k.shape[1], rel_logits, window_left)
        ):
            from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_extend import (
                is_compiled_for,
                rel_mha_varlen_tsmha,
            )

            if (
                is_compiled_for(
                    q,
                    k.shape[1],
                    rel_logits.shape[-1],
                    window_left >= 0,
                    False,
                    enable_pdl,
                )
                or not torch.cuda.is_current_stream_capturing()
            ):
                return rel_mha_varlen_tsmha(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    rel_logits=rel_logits,
                    window_left=window_left,
                    max_seqlen_k=max_seqlen,
                    cu_seqlens_k=cu_seqlens,
                    softmax_scale=softmax_scale,
                    enable_pdl=enable_pdl,
                )
        extra, window_size = _rel_attention_kwargs(rel_logits, window_left)
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
            window_size=window_size,
            return_lse=return_lse,
            **extra,
        )
        if return_lse:
            return out, lse.transpose(0, 1).contiguous()
        return out

    @register_kernel(
        "attention",
        "rel_mha_extend_with_kvcache",
        name="fa4_rel_mha_extend_with_kvcache_cached",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _FA4_BLACKWELL_DECODE_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def fa4_rel_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        rel_logits: torch.Tensor,
        window_left: int = -1,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        enable_pdl: bool = False,
        # Call parity with the dispatcher's unconditional scale pass; the
        # dense path takes no scales (mxfp8 rides its own registration).
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        if (
            not return_lse
            # k*128-token pages ride the same kernel via paged_kv_blocks_per_page (hetero slots).
            and k_cache.shape[1] % 128 == 0
            and cu_seqlens_q.shape[0] - 1 <= 1024
            and _tsmha_varlen_eligible(q, k_cache.shape[2], rel_logits, window_left)
        ):
            from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_extend import (
                is_compiled_for,
                rel_mha_varlen_tsmha,
            )

            if (
                is_compiled_for(
                    q,
                    k_cache.shape[2],
                    rel_logits.shape[-1],
                    window_left >= 0,
                    True,
                    enable_pdl,
                )
                or not torch.cuda.is_current_stream_capturing()
            ):
                return rel_mha_varlen_tsmha(
                    q=q,
                    k=k_cache,
                    v=v_cache,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    rel_logits=rel_logits,
                    window_left=window_left,
                    max_seqlen_k=max_seqlen_k,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    softmax_scale=softmax_scale,
                    enable_pdl=enable_pdl,
                )
        extra, window_size = _rel_attention_kwargs(rel_logits, window_left)
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
            causal=True,
            window_size=window_size,
            return_lse=return_lse,
            **extra,
        )
        if return_lse:
            return out, lse.transpose(0, 1).contiguous()
        return out

    @register_kernel(
        "attention",
        "rel_mha_decode_with_kvcache",
        name="fa4_rel_mha_decode_with_kvcache",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _FA4_BLACKWELL_DECODE_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def fa4_rel_mha_decode_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        rel_logits: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int = 1,
        window_left: int = -1,
        softmax_scale: float | None = None,
        enable_pdl: bool = False,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # v2 tsmha: MTP verify (uniform msq > 1) rides the kernel's NATIVE
        # prediction dimension; non-uniform msq takes the fork varlen path.
        _native_multiq = (
            max_seqlen_q > 1
            and q.shape[0] == cache_seqlens.shape[0] * max_seqlen_q
            and page_table.shape[0] == cache_seqlens.shape[0]
        )
        if (
            _TSMHA_DECODE_V2
            and (max_seqlen_q == 1 or _native_multiq)
            and k_cache.shape[1] in (64, 128, 256)
            and q.dtype in (torch.bfloat16, torch.float16)
        ):
            from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode_v2 import (
                is_compiled_for_v2,
                rel_mha_decode_tsmha_v2,
                v2_buffers_ready,
            )

            # Under capture the compiled kernel AND every per-B buffer must
            # pre-exist (capture-mempool allocations get clobbered); missing
            # -> this bucket bakes the v1 path instead.
            _rl = rel_logits.reshape(q.shape[0], q.shape[1], -1)
            _pred = max_seqlen_q if _native_multiq else 1
            if not torch.cuda.is_current_stream_capturing() or (
                is_compiled_for_v2(
                    q, k_cache, _rl, window_left, enable_pdl, prediction=_pred
                )
                and v2_buffers_ready(q, window_left, page_table, prediction=_pred)
            ):
                # v2 returns (rows, H*D); every other route returns q.shape.
                # Zero-copy view keeps the public op's contract uniform.
                return rel_mha_decode_tsmha_v2(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    rel_logits=_rl,
                    window_left=window_left,
                    softmax_scale=softmax_scale,
                    enable_pdl=enable_pdl,
                    prediction=_pred,
                ).view(q.shape)
        if (
            _TSMHA_REL_DECODE
            and max_seqlen_q == 1
            # k*128 pages ride the same kernel via paged_kv_blocks_per_page; bitwise-verified for k=2.
            and k_cache.shape[1] % 128 == 0
            and rel_logits.shape[-1] % 128 == 0
            and q.dtype in (torch.bfloat16, torch.float16)
        ):
            from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode import (
                is_compiled_for,
                rel_mha_decode_tsmha,
            )

            # Never compile during graph capture; the engine's uncaptured warmup decode compiles first.
            if (
                is_compiled_for(q, k_cache, rel_logits, window_left, enable_pdl)
                or not torch.cuda.is_current_stream_capturing()
            ):
                return rel_mha_decode_tsmha(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    rel_logits=rel_logits.reshape(q.shape[0], q.shape[1], -1),
                    cu_seqlens_q=cu_seqlens_q,
                    window_left=window_left,
                    softmax_scale=softmax_scale,
                    enable_pdl=enable_pdl,
                )
        # Varlen required: batch mode reports offset_q 0 for every request (wrong rel_logits rows).
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        extra, window_size = _rel_attention_kwargs(rel_logits, window_left)
        # Use split-kv heuristic (num_splits=0) for full attention.
        num_splits = 0 if window_left < 0 else 1
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
            causal=True,
            window_size=window_size,
            num_splits=num_splits,
            **extra,
        )
        return out

    @register_kernel(
        "attention",
        "rel_mha_decode_with_kvcache",
        name="fa4_rel_mha_decode_with_kvcache_mxfp8",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_ATTENTION_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def fa4_rel_mha_decode_with_kvcache_mxfp8(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        rel_logits: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int = 1,
        window_left: int = -1,
        softmax_scale: float | None = None,
        enable_pdl: bool = False,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # No score_mod fallback for fp8: ineligible configs raise instead of silently degrading.
        if not (
            max_seqlen_q == 1
            and k_cache.shape[1] % 128 == 0
            and rel_logits.shape[-1] % 128 == 0
        ):
            raise RuntimeError(
                "MXFP8 rel decode requires max_seqlen_q == 1, page_size % 128 "
                f"== 0 and extent % 128 == 0 (got q {max_seqlen_q}, page "
                f"{k_cache.shape[1]}, extent {rel_logits.shape[-1]})"
            )
        from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_decode import (
            is_compiled_for,
            rel_mha_decode_tsmha,
        )

        if (
            not is_compiled_for(
                q, k_cache, rel_logits, window_left, enable_pdl, blockscaled=True
            )
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "MXFP8 rel decode kernel not compiled before CUDA-graph "
                "capture; call the tokenspeed_mha rel_decode warmup with "
                "blockscaled=True first"
            )
        return rel_mha_decode_tsmha(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            rel_logits=rel_logits.reshape(q.shape[0], q.shape[1], -1),
            cu_seqlens_q=cu_seqlens_q,
            window_left=window_left,
            softmax_scale=softmax_scale,
            enable_pdl=enable_pdl,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    @register_kernel(
        "attention",
        "rel_mha_extend_with_kvcache",
        name="fa4_rel_mha_extend_with_kvcache_mxfp8",
        solution="fa4",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_ATTENTION_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def fa4_rel_mha_extend_with_kvcache_mxfp8(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        rel_logits: torch.Tensor,
        window_left: int = -1,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        enable_pdl: bool = False,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        extent = rel_logits.shape[-1]
        if not (
            not return_lse
            and k_cache.shape[1] % 128 == 0
            and extent % 128 == 0
            and cu_seqlens_q.shape[0] - 1 <= 1024
            and (window_left < 0 or window_left + 1 == extent)
            and q.shape[1] % k_cache.shape[2] == 0
        ):
            raise RuntimeError(
                "MXFP8 rel extend constraint violated (page_size % 128, "
                "extent % 128, window_left + 1 == extent for SWA, B <= 1024, "
                "no LSE)"
            )
        from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha.rel_extend import (
            is_compiled_for,
            rel_mha_varlen_tsmha,
        )

        if (
            not is_compiled_for(
                q,
                k_cache.shape[2],
                extent,
                window_left >= 0,
                True,
                enable_pdl,
                k_cache.shape[1] // 128,
                blockscaled=True,
                out_dtype=rel_logits.dtype,
            )
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "MXFP8 rel extend kernel not compiled before CUDA-graph "
                "capture; warm up the tokenspeed_mha rel_extend config with "
                "blockscaled=True first"
            )
        return rel_mha_varlen_tsmha(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            rel_logits=rel_logits,
            window_left=window_left,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            enable_pdl=enable_pdl,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

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
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False}),
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
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
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
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
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
