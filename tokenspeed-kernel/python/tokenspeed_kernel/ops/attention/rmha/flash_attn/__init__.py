"""FlashAttention relative-MHA registrations."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import MXFP8_BLOCK_SCALE, format_signatures

platform = current_platform()

if platform.is_nvidia and platform.is_blackwell:
    from tokenspeed_kernel.ops.attention.mha.flash_attn import (
        fa4_rel_mha_decode_with_kvcache as _decode_impl,
    )
    from tokenspeed_kernel.ops.attention.mha.flash_attn import (
        fa4_rel_mha_decode_with_kvcache_mxfp8 as _decode_mxfp8_impl,
    )
    from tokenspeed_kernel.ops.attention.mha.flash_attn import (
        fa4_rel_mha_extend_with_kvcache as _extend_impl,
    )
    from tokenspeed_kernel.ops.attention.mha.flash_attn import (
        fa4_rel_mha_extend_with_kvcache_mxfp8 as _extend_mxfp8_impl,
    )
    from tokenspeed_kernel.ops.attention.mha.flash_attn import (
        fa4_rel_mha_prefill as _prefill_impl,
    )

    _PREFILL_HEAD_DIMS = frozenset(range(8, 256, 8))
    _DECODE_HEAD_DIMS = frozenset(range(8, 129, 8))
    _MXFP8_SIGNATURES = format_signatures(
        ("q", "k_cache", "v_cache"),
        "mxfp8",
        {torch.float8_e4m3fn},
        scale=MXFP8_BLOCK_SCALE,
    )
    _CAPABILITY = CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0), vendors=frozenset({"nvidia"})
    )

    @register_kernel(
        "attention",
        "rel_mha_prefill",
        name="fa4_rel_mha_prefill",
        solution="fa4",
        capability=_CAPABILITY,
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _PREFILL_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def fa4_rel_mha_prefill(*args, **kwargs):
        return _prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_extend_with_kvcache",
        name="fa4_rel_mha_extend_with_kvcache_cached",
        solution="fa4",
        capability=_CAPABILITY,
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _DECODE_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def fa4_rel_mha_extend_with_kvcache(*args, **kwargs):
        return _extend_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_decode_with_kvcache",
        name="fa4_rel_mha_decode_with_kvcache",
        solution="fa4",
        capability=_CAPABILITY,
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": _DECODE_HEAD_DIMS,
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def fa4_rel_mha_decode_with_kvcache(*args, **kwargs):
        return _decode_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_decode_with_kvcache",
        name="fa4_rel_mha_decode_with_kvcache_mxfp8",
        solution="fa4",
        capability=_CAPABILITY,
        signatures=_MXFP8_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def fa4_rel_mha_decode_with_kvcache_mxfp8(*args, **kwargs):
        return _decode_mxfp8_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_extend_with_kvcache",
        name="fa4_rel_mha_extend_with_kvcache_mxfp8",
        solution="fa4",
        capability=_CAPABILITY,
        signatures=_MXFP8_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def fa4_rel_mha_extend_with_kvcache_mxfp8(*args, **kwargs):
        return _extend_mxfp8_impl(*args, **kwargs)
