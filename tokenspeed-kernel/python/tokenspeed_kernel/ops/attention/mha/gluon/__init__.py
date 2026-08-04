"""AMD Gluon MHA registrations."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        gluon_mha_decode_gfx950 as _decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        gluon_mha_extend_gfx950 as _extend_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
        gluon_mha_prefill_gfx950 as _prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mha.decode import (
        gluon_mha_decode_gfx1250 as _decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mha.prefill import (
        gluon_mha_prefill_gfx1250 as _prefill_gfx1250_impl,
    )

    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    _GFX1250 = CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    )

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="gluon_mha_decode_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_mha_decode_gfx950(*args, **kwargs):
        return _decode_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="gluon_mha_decode_gfx1250",
        solution="gluon",
        capability=_GFX1250,
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "q_len": frozenset({1}),
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64, 128}),
            "sliding_window": frozenset({False}),
            "support_sinks": frozenset({False}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_mha_decode_gfx1250(*args, **kwargs):
        return _decode_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_prefill",
        name="gluon_mha_prefill_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_prefill_gfx950(*args, **kwargs):
        return _prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_prefill",
        name="gluon_mha_prefill_gfx1250",
        solution="gluon",
        capability=_GFX1250,
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_prefill_gfx1250(*args, **kwargs):
        return _prefill_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="gluon_mha_extend_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64}),
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_extend_gfx950(*args, **kwargs):
        return _extend_impl(*args, **kwargs)
